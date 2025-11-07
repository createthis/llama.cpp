#include "topk-radix.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../../include/ggml-cuda-radix.h"
#include <stdlib.h>

#ifndef SEL_DEBUG
#define SEL_DEBUG 0
#endif
#ifndef SEL_DEBUG_COL
#define SEL_DEBUG_COL 0
#endif
static inline __host__ __device__ int env_threads_or_default(const char * name, int deflt) {
    int v = deflt;
#ifndef __CUDA_ARCH__
    const char * e = getenv(name);
    if (e && *e) {
        int t = atoi(e);
        if (t > 0) v = t;
    }
#endif
    if (v < 128) v = 128;
    if (v > 1024) v = 1024;
    v = (v + 31) & ~31;
    return v;
}



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
    // dynamic shared memory: [warp_count*256] per-warp histograms + [256] final hist
    extern __shared__ uint32_t shmem[];
    const int warp_count = blockDim.x >> 5;
    uint32_t * hist_warp = shmem;
    uint32_t * hist      = shmem + warp_count * 256;
    for (int i = threadIdx.x; i < 256 * (warp_count + 1); i += blockDim.x) shmem[i] = 0u;
    __syncthreads();

    const float * col = scores + (size_t)ld * t;
    // accumulate per-warp histograms
    uint32_t * my_hist = hist_warp + ((threadIdx.x >> 5) * 256);
    int i4 = threadIdx.x * 4;
    for (; i4 + 3 < N; i4 += blockDim.x * 4) {
        float4 v = *((const float4 *)(col + i4));
        uint32_t k0 = float_to_key_desc(v.x);
        uint32_t k1 = float_to_key_desc(v.y);
        uint32_t k2 = float_to_key_desc(v.z);
        uint32_t k3 = float_to_key_desc(v.w);
        atomicAdd(&my_hist[(k0 >> 24) & 0xFFu], 1u);
        atomicAdd(&my_hist[(k1 >> 24) & 0xFFu], 1u);
        atomicAdd(&my_hist[(k2 >> 24) & 0xFFu], 1u);
        atomicAdd(&my_hist[(k3 >> 24) & 0xFFu], 1u);
    }
    int rem = N & 3;
    int tail_start = N - rem;
    int li = tail_start + threadIdx.x;
    if (threadIdx.x < rem && li < N) {
        uint32_t key = float_to_key_desc(col[li]);
        atomicAdd(&my_hist[(key >> 24) & 0xFFu], 1u);
    }

    __syncthreads();

    // reduce to final hist
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        uint32_t s = 0u;
        for (int w = 0; w < warp_count; ++w) s += hist_warp[w*256 + b];
        hist[b] = s;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (int b = 255; b >= 0; --b) {
            gt_counts[b + 256*t] = sum;
            sum += hist[b];
        }
        thr_bins[t] = 0;
    }
}


// select indices > threshold bin and collect equals for tail passes; simplified single-pass fallback uses argsort for small N

// select indices > threshold bin and collect equals for tail passes
static __global__ void k_select_topk_bins(const float * __restrict__ scores,
                                          int N, int T, int ld, int k, int eq_capacity,
                                          const uint32_t * __restrict__ gt_counts, // [256, T]
                                          int * __restrict__ idx_out) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float * col = scores + (size_t)ld * t;
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
#if SEL_DEBUG
    if (blockIdx.x == SEL_DEBUG_COL && threadIdx.x == 0) {
        uint32_t sgt = gt_counts[thr0 + 256*t];
        uint32_t prev = (thr0 == 0 ? (uint32_t)N : gt_counts[(thr0 - 1) + 256*t]);
        uint32_t eq   = prev - gt_counts[thr0 + 256*t];
        printf("[t=%d] thr0=%d sgt=%u eq=%u k=%d\n", t, thr0, sgt, eq, k);
    }
#endif

    uint32_t sgt0 = gt_counts[thr0 + 256*t];
    int take_gt0 = min(k, (int)sgt0);

    extern __shared__ int shared[];
    int * eq_buf = shared; // dynamic shared memory for candidate buffers

    __shared__ int sel_gt0;
    __shared__ int sel_sofar;
    __shared__ int eq0_count;
    if (threadIdx.x == 0) { sel_gt0 = 0; sel_sofar = 0; eq0_count = 0; }
    __syncthreads();

    __shared__ unsigned int h2[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h2[i] = 0u;
    __syncthreads();

    // Vectorized Stage A streaming pass
    int i4 = threadIdx.x * 4;
    for (; i4 + 3 < N; i4 += blockDim.x * 4) {
        float4 v = *((const float4 *)(col + i4));
        uint32_t k0 = float_to_key_desc(v.x);
        uint32_t k1 = float_to_key_desc(v.y);
        uint32_t k2 = float_to_key_desc(v.z);
        uint32_t k3 = float_to_key_desc(v.w);
        uint32_t arr[4] = {k0,k1,k2,k3};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int idx = i4 + j;
            int b0 = (arr[j] >> 24) & 0xFF;
            if (b0 > thr0) {
                int pos = atomicAdd(&sel_gt0, 1);
                if (pos < take_gt0) idx_out[pos + k*t] = idx;
            } else if (b0 == thr0) {
                int p = atomicAdd(&eq0_count, 1);
                if (p < eq_capacity) eq_buf[p] = idx;
                int b1 = (arr[j] >> 16) & 0xFF;
                atomicAdd(&h2[b1], 1u);
            }
        }
    }
    int rem = N & 3;
    int tail_start = N - rem;
    int ti = tail_start + threadIdx.x;
    if (threadIdx.x < rem && ti < N) {
        uint32_t key = float_to_key_desc(col[ti]);
        int b0 = (key >> 24) & 0xFF;
        if (b0 > thr0) {
            int pos = atomicAdd(&sel_gt0, 1);
            if (pos < take_gt0) idx_out[pos + k*t] = ti;
        } else if (b0 == thr0) {
            int p = atomicAdd(&eq0_count, 1);
            if (p < eq_capacity) eq_buf[p] = ti;
            int b1 = (key >> 16) & 0xFF;
            atomicAdd(&h2[b1], 1u);
        }
    }
    __syncthreads();
    sel_sofar = take_gt0;

    int remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Determine if eq0 overflowed shared buffer
    __shared__ int fallback_b1;
    if (threadIdx.x == 0) fallback_b1 = (eq0_count > eq_capacity);
    __syncthreads();

    // Compute thr1 from h2 histogram
    __shared__ int thr1;
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

    // Stage B: add items with b1>thr1 and build h3 over b1==thr1
    __shared__ unsigned int h3[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h3[i] = 0u;
    __shared__ int eq2_count;
    if (threadIdx.x == 0) eq2_count = 0;
    __syncthreads();

    if (!fallback_b1) {
        for (int j = threadIdx.x; j < eq0_count; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b1 = (key >> 16) & 0xFF;
            if (b1 > thr1) {
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = idx;
            } else if (b1 == thr1) {
                int b2 = (key >> 8) & 0xFF;
                atomicAdd(&h3[b2], 1u);
                int p = atomicAdd(&eq2_count, 1);
                if (p < eq_capacity) eq_buf[p] = idx; // reuse eq_buf to store eq2 set
            }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
            int b1 = (key >> 16) & 0xFF;
            if (b1 > thr1) {
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = i;
            } else if (b1 == thr1) {
                int b2 = (key >> 8) & 0xFF;
                atomicAdd(&h3[b2], 1u);
                int p = atomicAdd(&eq2_count, 1);
                if (p < eq_capacity) eq_buf[p] = i;
            }
        }
    }
    __syncthreads();

    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Compute thr2 on h3
    __shared__ int thr2;
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

    // Stage C: add b2>thr2 from eq2 set (or fallback) and build pool of ==thr2
    if (!fallback_b1) {
        int lim = min(eq2_count, eq_capacity);
        for (int j = threadIdx.x; j < lim; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = float_to_key_desc(col[idx]);
            int b2 = (key >> 8) & 0xFF;
            if (b2 > thr2) {
                int pos = atomicAdd(&sel_sofar, 1);
                if (pos < k) idx_out[pos + k*t] = idx;
            }
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
    __syncthreads();

    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Prepare pool of b2 == thr2 candidates
    __shared__ int pool_count;
    if (threadIdx.x == 0) pool_count = 0;
    __syncthreads();

    if (!fallback_b1) {
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

void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    // Radix-like path: histogram top byte + select with tie refinement
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d  = nullptr;
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    const int hist_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    const size_t hist_shmem = (size_t)(((hist_threads/32) + 1) * 256) * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    // Equal-bin selection kernel; bound dynamic shared memory to device limit
    const int sel_threads = hist_threads;
    // Conservative eq buffer capacity to avoid exceeding per-block shared mem
    int cap_env = 0;
    const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) { cap_env = atoi(env_cap); if (cap_env < 0) cap_env = 0; }
    int cap_default = 4096;
    const int eq_cap = max(k, min(N, cap_env ? cap_env : cap_default));
    size_t sel_shmem = (size_t) eq_cap * sizeof(int);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_select_topk_bins, (int)sel_shmem);
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

    const int hist_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    const size_t hist_shmem = (size_t)(((hist_threads/32) + 1) * 256) * sizeof(uint32_t);
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

    const int sel_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    int cap_env = 0; const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) { cap_env = atoi(env_cap); if (cap_env < 0) cap_env = 0; }
    int cap_default = 4096;
    const int eq_cap_host = max(k, min(N, cap_env ? cap_env : cap_default));
    const size_t sel_shmem = (size_t) eq_cap_host * sizeof(int);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_select_topk_bins, (int)sel_shmem);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap_host, gt_counts_d, idx_d);

    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(idx_d);
}
