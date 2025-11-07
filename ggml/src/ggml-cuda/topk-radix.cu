#include "topk-radix.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../../include/ggml-cuda-radix.h"
#ifndef SEL_DEBUG
#define SEL_DEBUG 0
#endif
#ifndef SEL_DEBUG_COL
#define SEL_DEBUG_COL 0
#endif


// float -> key mapping ascending; to get descending selection we pick largest keys
static __device__ __forceinline__ uint32_t float_to_key_desc(float x) {
    uint32_t u = __float_as_uint(x);
    if ((int32_t)u < 0) {
        return ~u;
    } else {
        return u ^ 0x80000000u;
    }
}

// Compute K-th threshold bin for top byte of keys of a given column
// Here we implement a block-per-column approach where each block processes N elements.
// We use shared histogram of 256 bins.
static __global__ void k_histogram_topbyte(const float * __restrict__ scores,
                                           int N, int T, int ld,
                                           uint32_t * __restrict__ thr_bins,
                                           uint32_t * __restrict__ gt_counts) {
    int t = blockIdx.x;
    if (t >= T) return;
    extern __shared__ uint32_t hist[]; // 256 bins
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();

    const float * col = scores + (size_t)ld * t;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        atomicAdd(&hist[(key >> 24) & 0xFFu], 1u);
    }
    __syncthreads();

    // Compute thr bin: find largest b such that sum_{bb>b} hist[bb] < K <= sum_{bb>=b} hist[bb]
    // We cannot know K here; we just store histogram; instead, to keep kernels simple we output prefix sums of greater counts
    // But for simplicity in this first version, we store full hist to global and compute thr on host or another kernel.
    // To keep memory low, compute cumulatives here and write them to gt_counts buffer laid out [256, T].
    // gt_counts[bb + 256*t] = sum_{k>bb} hist[k]
    // One thread computes cumulative
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (int b = 255; b >= 0; --b) {
            gt_counts[b + 256*t] = sum;
            sum += hist[b];
        }
        // store raw hist threshold bin placeholder
        thr_bins[t] = 0; // not used in this version
    }
}

// select indices > threshold bin and collect equals for tail passes; simplified single-pass fallback uses argsort for small N
static __global__ void k_select_topk_bins(const float * __restrict__ scores,
                                          int N, int T, int ld, int k, int eq_capacity,
                                          const uint32_t * __restrict__ gt_counts, // [256, T]
                                          int * __restrict__ idx_out) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float * col = scores + (size_t)ld * t;
    // initialize output indices defensively to 0
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) idx_out[i + k*t] = 0;
    }
    __syncthreads();
    // find thr0 from gt_counts
    int thr0 = 0;
    for (int b = 255; b >= 0; --b) {
        uint32_t sgt = gt_counts[b + 256*t];
        uint32_t prev = (b == 0 ? (uint32_t)N : gt_counts[(b - 1) + 256*t]);
        uint32_t eq   = prev - gt_counts[b + 256*t];
        if (sgt < (uint32_t)k && sgt + eq >= (uint32_t)k) { thr0 = b; break; }
    }
    if (SEL_DEBUG && blockIdx.x == SEL_DEBUG_COL && threadIdx.x == 0) {
        uint32_t sgt = gt_counts[thr0 + 256*t];
        uint32_t prev = (thr0 == 0 ? (uint32_t)N : gt_counts[(thr0 - 1) + 256*t]);
        uint32_t eq   = prev - gt_counts[thr0 + 256*t];
        printf("[t=%d] thr0=%d sgt=%u eq=%u k=%d\n", t, thr0, sgt, eq, k);
    }
    // Now collect selected (> thr0) and equality list; parallelize by striding threads
        // Stage A: write indices for values with top-byte bin > thr0
    uint32_t sgt0 = gt_counts[thr0 + 256*t];
    int take_gt0 = min(k, (int)sgt0);
    __shared__ int sel_gt0;
    if (threadIdx.x == 0) sel_gt0 = 0;
    __syncthreads();
    if (take_gt0 > 0) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int bin = (key >> 24) & 0xFF;
            if (bin > thr0) {
                int pos = atomicAdd(&sel_gt0, 1);
                if (pos < take_gt0) idx_out[pos + k*t] = i;
            }
        }
    }
    __syncthreads();
    int remaining = k - take_gt0;
    if (remaining <= 0) return;

    // Shared buffers
    extern __shared__ int shared[];
    int * eq_buf = shared; // used only for final tie pool
    __shared__ int pool_count;
    __shared__ int sel_sofar;
    if (threadIdx.x == 0) { pool_count = 0; sel_sofar = take_gt0; }
    __syncthreads();


    // Build eq0 list (b0 == thr0) into eq_buf; use as candidate set for refinement
    __shared__ int eq0_count;
    if (threadIdx.x == 0) eq0_count = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int b0 = (key >> 24) & 0xFF;
        if (b0 == thr0) {
            int p = atomicAdd(&eq0_count, 1);
            if (p < eq_capacity) eq_buf[p] = i;
        }
    }
    __syncthreads();

    // Stage B: build b1 histogram from eq0 list; fallback to full scan if eq0 overflowed
    __shared__ unsigned int h2[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h2[i] = 0u;
    __syncthreads();
    bool fallback_b1 = (eq0_count > eq_capacity);
    if (!fallback_b1) {
        for (int j = threadIdx.x; j < eq0_count; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b1 = (key >> 16) & 0xFF;
            atomicAdd(&h2[b1], 1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int b0 = (key >> 24) & 0xFF;
            if (b0 == thr0) {
                int b1 = (key >> 16) & 0xFF;
                atomicAdd(&h2[b1], 1u);
            }
        }
    }
    __syncthreads();
    int thr1 = 0;
    if (threadIdx.x == 0) {
        unsigned int sum = 0; unsigned int need = remaining;
        thr1 = 255;
        for (int b = 255; b >= 0; --b) {
            unsigned int sgt = sum;
            unsigned int eqb = h2[b];
            if (sgt < need && sgt + eqb >= need) { thr1 = b; break; }
            sum += eqb;
        }
    }
    __syncthreads();
    __shared__ int s_thr1; if (threadIdx.x==0) s_thr1 = thr1; __syncthreads(); thr1 = s_thr1;
    // Add strictly greater than thr1 on second byte; use eq0 subset if available
    if (remaining > 0) {
        if (!fallback_b1) {
            for (int j = threadIdx.x; j < eq0_count; j += blockDim.x) {
                int idx = eq_buf[j];
                uint32_t key = float_to_key_desc(col[idx]);
                int b1 = (key >> 16) & 0xFF;
                if (b1 > thr1) {
                    int pos = atomicAdd(&sel_sofar, 1);
                    if (pos < k) idx_out[pos + k*t] = idx;
                }
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                uint32_t key = float_to_key_desc(col[i]);
                int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
                int b1 = (key >> 16) & 0xFF; if (b1 <= thr1) continue;
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = i;
            }
        }
    }
    __syncthreads();
    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Stage C: Third-byte histogram and selection over eq2 candidates (b1==thr1)
    __shared__ unsigned int h3[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h3[i] = 0u;
    __syncthreads();

    __shared__ int eq2_count;
    if (threadIdx.x == 0) eq2_count = 0;
    __syncthreads();

    bool fallback_b2 = false;
    if (!fallback_b1) {
        for (int j = threadIdx.x; j < eq0_count; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b1 = (key >> 16) & 0xFF;
            if (b1 == thr1) {
                int p = atomicAdd(&eq2_count, 1);
                if (p < eq_capacity) eq_buf[p] = idx; // pack into front
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) fallback_b2 = (eq2_count > eq_capacity);
    } else {
        fallback_b2 = true;
    }
    __syncthreads();

    if (!fallback_b2) {
        int lim = min(eq2_count, eq_capacity);
        for (int j = threadIdx.x; j < lim; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b2 = (key >> 8) & 0xFF;
            atomicAdd(&h3[b2], 1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
            int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue;
            int b2 = (key >> 8) & 0xFF;
            atomicAdd(&h3[b2], 1u);
        }
    }
    __syncthreads();

    int thr2 = 0;
    if (threadIdx.x == 0) {
        unsigned int sum = 0; unsigned int need = remaining;
        thr2 = 255;
        for (int b = 255; b >= 0; --b) {
            unsigned int sgt = sum;
            unsigned int eqb = h3[b];
            if (sgt < need && sgt + eqb >= need) { thr2 = b; break; }
            sum += eqb;
        }
    }
    __syncthreads();
    __shared__ int s_thr2; if (threadIdx.x==0) s_thr2 = thr2; __syncthreads(); thr2 = s_thr2;

    if (remaining > 0) {
        if (!fallback_b2) {
            int lim = min(eq2_count, eq_capacity);
            for (int j = threadIdx.x; j < lim; j += blockDim.x) {
                int idx = eq_buf[j];
                uint32_t key = float_to_key_desc(col[idx]);
                int b2 = (key >> 8) & 0xFF; if (b2 <= thr2) continue;
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = idx;
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                uint32_t key = float_to_key_desc(col[i]);
                int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
                int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue;
                int b2 = (key >> 8) & 0xFF; if (b2 <= thr2) continue;
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = i;
            }
        }
    }
    __syncthreads();

    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Prepare pool of b2 == thr2 candidates
    if (threadIdx.x == 0) pool_count = 0;
    __syncthreads();

    if (!fallback_b2) {
        int lim = min(eq2_count, eq_capacity);
        for (int j = threadIdx.x; j < lim; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b2 = (key >> 8) & 0xFF;
            if (b2 == thr2) {
                int p = atomicAdd(&pool_count, 1);
                if (p < eq_capacity) eq_buf[p] = idx;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && pool_count > eq_capacity) pool_count = eq_capacity;
        __syncthreads();
    } else {
        // Fallback: reduce equal third-byte candidates from full column
        const int LOCAL_TOP = 4;
        int loc_idx[LOCAL_TOP]; float loc_val[LOCAL_TOP];
        for (int l = 0; l < LOCAL_TOP; ++l) { loc_idx[l] = -1; loc_val[l] = -1.0e30f; }
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
            int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue;
            int b2 = (key >> 8) & 0xFF; if (b2 != thr2) continue;
            float v = col[i];
            int min_pos = 0; float min_val = loc_val[0];
            for (int l = 1; l < LOCAL_TOP; ++l) if (loc_val[l] < min_val) { min_val = loc_val[l]; min_pos = l; }
            if (v > min_val) { loc_val[min_pos] = v; loc_idx[min_pos] = i; }
        }
        if (threadIdx.x == 0) pool_count = 0;
        __syncthreads();
        for (int l = 0; l < LOCAL_TOP; ++l) {
            int idx = loc_idx[l];
            if (idx >= 0) {
                int p = atomicAdd(&pool_count, 1);
                if (p < eq_capacity) eq_buf[p] = idx;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && pool_count > eq_capacity) pool_count = eq_capacity;
        __syncthreads();
    }

    // Fill remaining from pool
    for (int r = 0; r < remaining; ++r) {
        __syncthreads();
        float best_val = -1.0e30f; int best_idx = -1;
        for (int i0 = threadIdx.x; i0 < pool_count; i0 += blockDim.x) {
            int idx = eq_buf[i0]; if (idx < 0) continue;
            float v = col[idx];
            if (v > best_val || (v == best_val && idx < best_idx)) { best_val = v; best_idx = idx; }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float v2 = __shfl_down_sync(0xffffffff, best_val, offset);
            int   i2 = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (v2 > best_val || (v2 == best_val && i2 < best_idx)) { best_val = v2; best_idx = i2; }
        }
        __shared__ float warp_best_val[32];
        __shared__ int   warp_best_idx[32];
        __shared__ int   s_selected;
        int wid = threadIdx.x >> 5; int lane = threadIdx.x & 31;
        if (lane == 0) { warp_best_val[wid] = best_val; warp_best_idx[wid] = best_idx; }
        __syncthreads();
        if (wid == 0) {
            float bv = (threadIdx.x < (blockDim.x+31)/32) ? warp_best_val[lane] : -1.0e30f;
            int   bi = (threadIdx.x < (blockDim.x+31)/32) ? warp_best_idx[lane] : -1;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float v2 = __shfl_down_sync(0xffffffff, bv, offset);
                int   i2 = __shfl_down_sync(0xffffffff, bi, offset);
                if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; }
            }
            if (lane == 0) {
                s_selected = bi;
                if (s_selected >= 0) {
                    int pos = atomicAdd(&sel_sofar, 1);
                    if (pos < k) idx_out[pos + k*t] = s_selected;
                }
            }
        }
        __syncthreads();
        if (s_selected >= 0) {
            for (int j = threadIdx.x; j < pool_count; j += blockDim.x) {
                if (eq_buf[j] == s_selected) { eq_buf[j] = -1; }
            }
        }
        __syncthreads();
    }



    
}



// One-pass block-level top-k selection (register-resident per-thread local top, then block reduction)
#define TOPK_THREADS 1024
#define TOPK_LOCAL   4

static __global__ void k_block_select_topk(const float * __restrict__ scores,
                                           int N, int T, int ld, int k,
                                           int * __restrict__ idx_out) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float * col = scores + (size_t)ld * t;
    if (N <= 0) { if (threadIdx.x == 0) { for (int r = 0; r < k; ++r) idx_out[r + k*t] = 0; } return; }

    float lval[TOPK_LOCAL]; int lidx[TOPK_LOCAL];
    for (int i = 0; i < TOPK_LOCAL; ++i) { lval[i] = -1.0e30f; lidx[i] = -1; }

    // Strided scan with fill-first-empty then replace-min (tie-break by smaller index)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = col[i];
        int empty = -1;
        #pragma unroll
        for (int j = 0; j < TOPK_LOCAL; ++j) { if (lidx[j] == -1) { empty = j; break; } }
        if (empty != -1) {
            lval[empty] = v; lidx[empty] = i;
        } else {
            int min_pos = 0; float min_val = lval[0];
            #pragma unroll
            for (int j = 1; j < TOPK_LOCAL; ++j) {
                if (lval[j] < min_val || (lval[j] == min_val && lidx[j] > lidx[min_pos])) { min_val = lval[j]; min_pos = j; }
            }
            if (v > min_val || (v == min_val && i < lidx[min_pos])) { lval[min_pos] = v; lidx[min_pos] = i; }
        }
    }
    // Post-scan: ensure no empty slots
    #pragma unroll
    for (int j = 0; j < TOPK_LOCAL; ++j) {
        if (lidx[j] < 0) {
            int idx = (threadIdx.x * TOPK_LOCAL + j) % N;
            lidx[j] = idx; lval[j] = col[idx];
        }
    }

    __shared__ float s_vals[TOPK_THREADS*TOPK_LOCAL];
    __shared__ int   s_ids[TOPK_THREADS*TOPK_LOCAL];

    int base = threadIdx.x * TOPK_LOCAL;
    #pragma unroll
    for (int j = 0; j < TOPK_LOCAL; ++j) { s_vals[base + j] = lval[j]; s_ids[base + j] = lidx[j]; }
    __syncthreads();

    const int M = TOPK_THREADS*TOPK_LOCAL;

    __shared__ float warp_best_val[32];
    __shared__ int   warp_best_pos[32];
    __shared__ int   s_block_best_pos;
    __shared__ int   s_sel_idx;

    int wid = threadIdx.x >> 5; int lane = threadIdx.x & 31;

    for (int r = 0; r < k; ++r) {
        float best_val = -1.0e30f; int best_pos = -1;
        for (int p = threadIdx.x; p < M; p += blockDim.x) {
            float v = s_vals[p];
            if (v > best_val) { best_val = v; best_pos = p; }
        }
        // warp reduce (max by value, tie-break by smaller index)
        for (int off = 16; off > 0; off >>= 1) {
            float v2 = __shfl_down_sync(0xffffffff, best_val, off);
            int   p2 = __shfl_down_sync(0xffffffff, best_pos, off);
            if (v2 > best_val || (v2 == best_val && p2 < best_pos)) { best_val = v2; best_pos = p2; }
        }
        if (lane == 0) { warp_best_val[wid] = best_val; warp_best_pos[wid] = best_pos; }
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = warp_best_val[lane]; int p = warp_best_pos[lane];
            for (int off = 16; off > 0; off >>= 1) {
                float v2 = __shfl_down_sync(0xffffffff, v, off);
                int   p2 = __shfl_down_sync(0xffffffff, p, off);
                if (v2 > v || (v2 == v && p2 < p)) { v = v2; p = p2; }
            }
            if (lane == 0) s_block_best_pos = p;
        }
        __syncthreads();
        int bp = s_block_best_pos;
        if (threadIdx.x == 0) {
            int out_idx = (bp >= 0 && bp < (int)(TOPK_THREADS*TOPK_LOCAL)) ? s_ids[bp] : 0;
            if (out_idx < 0 || out_idx >= N) out_idx = 0;
            s_sel_idx = out_idx;
            idx_out[r + k*t] = out_idx;
        }
        __syncthreads();
        // Dedup: mark all entries with selected idx as consumed
        for (int p = threadIdx.x; p < (int)(TOPK_THREADS*TOPK_LOCAL); p += blockDim.x) {
            if (s_ids[p] == s_sel_idx) s_vals[p] = -1.0e30f;
        }
        __syncthreads();
    }
}
#undef TOPK_THREADS
#undef TOPK_LOCAL
void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    const char * impl = getenv("LLAMA_SPARSE_TOPK_IMPL");
    if (impl && strcmp(impl, "block") == 0) {
        // one-pass block select
        const int threads = 1024;
        k_block_select_topk<<<T, threads, 0, stream>>>(scores_d, N, T, /*ld=*/N, k, idx_d);
        return;
    }
    // default radix path (existing)
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d  = nullptr;
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    const int hist_threads = 256;
    const size_t hist_shmem = 256 * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    const int sel_threads = 1024;
    int cap_env = 0;
    const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) { cap_env = atoi(env_cap); if (cap_env < 0) cap_env = 0; }
    int cap_default = 4096;
    const int eq_cap = max(k, min(N, cap_env ? cap_env : cap_default));
    size_t sel_shmem = (size_t) eq_cap * sizeof(int);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap, gt_counts_d, idx_d);

    cudaFree(gt_counts_d);
    cudaFree(thr_bins_d);
}


extern "C" void ggml_cuda_topk_radix_indices_host(const float * scores_h, int N, int T, int k, int * idx_h) {
    // Simple host wrapper: copy scores to device, allocate idx device, invoke kernels, copy back
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    int * idx_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&idx_d, sizeof(int) * (size_t)k * T);
    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);
    ggml_cuda_topk_radix_indices_device(ctx, scores_d, N, T, k, idx_d);
    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(scores_d);
    cudaFree(idx_d);
}

extern "C" void ggml_cuda_topk_histogram_host(const float * scores_h, int N, int T,
                                               unsigned int * gt_counts_h, unsigned int * thr_bins_h) {
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);

    const int hist_threads = 256;
    const size_t hist_shmem = 256 * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    cudaMemcpyAsync(gt_counts_h, gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thr_bins_h,  thr_bins_d,  sizeof(uint32_t) * (size_t)T,        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(thr_bins_d);
}

extern "C" void ggml_cuda_topk_select_host(const float * scores_h, int N, int T, int k,
                                            const unsigned int * gt_counts_h, int * idx_h) {
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    uint32_t * gt_counts_d = nullptr;
    int * idx_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&idx_d, sizeof(int) * (size_t)k * T);

    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gt_counts_d, gt_counts_h, sizeof(uint32_t) * 256 * (size_t)T, cudaMemcpyHostToDevice, stream);

    const int sel_threads = 1024;
    int cap_env = 0; const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) { cap_env = atoi(env_cap); if (cap_env < 0) cap_env = 0; }
    int cap_default = 4096;
    const int eq_cap_host = max(k, min(N, cap_env ? cap_env : cap_default));
    const size_t sel_shmem = (size_t) eq_cap_host * sizeof(int);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap_host, gt_counts_d, idx_d);

    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(idx_d);
}
