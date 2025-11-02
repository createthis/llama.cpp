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


// simple bitonic top-k per column (descending)


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
    extern __shared__ int shared[];
    int * eq_buf = shared; // temporary storage of eq indices (bounded)
    __shared__ int eq_count;
    __shared__ int eq_raw;
    __shared__ int sel_count;
    if (threadIdx.x == 0) { eq_count = 0; sel_count = 0; }
    __syncthreads();
    if (SEL_DEBUG && blockIdx.x == SEL_DEBUG_COL && threadIdx.x == 0) {
        printf("[t=%d] start collect: eq_count=%d sel_count=%d\n", t, eq_count, sel_count);
    }

    if (threadIdx.x == 0) eq_raw = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int bin = (key >> 24) & 0xFF;
        if (bin > thr0) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) {
                int clamped = i < 0 ? 0 : (i >= N ? (N - 1) : i);
                idx_out[pos + k*t] = clamped;
            }
        } else if (bin == thr0) {
            int raw = atomicAdd(&eq_raw, 1);
            if (raw < eq_capacity) {
                int pos = atomicAdd(&eq_count, 1);
                if (pos < eq_capacity) eq_buf[pos] = i;
            }
        }
    }
    __syncthreads();

    // cap eq_count to allocated shared buffer capacity and adjust remaining based on raw count
    if (threadIdx.x == 0) {
        if (eq_count > eq_capacity) eq_count = eq_capacity;
        int selected_gt = sel_count;
        int needed = k - selected_gt;
        if (needed < 0) needed = 0;
        if (eq_raw < needed) {
            // not enough equal-bin candidates to fill K: will fall back to skipping remaining
        }
    }
    __syncthreads();

    int remaining = k - sel_count;
    if (SEL_DEBUG && blockIdx.x == SEL_DEBUG_COL && threadIdx.x == 0) {
        printf("[t=%d] after collect: sel_count=%d eq_count=%d remaining=%d\n", t, sel_count, eq_count, remaining);
    }
    if (remaining <= 0) return;

    // Reduce equal-bin candidates to a small pool per-thread (LOCAL_TOP)
    const int LOCAL_TOP = 4;
    int loc_idx[LOCAL_TOP];
    float loc_val[LOCAL_TOP];
    for (int l = 0; l < LOCAL_TOP; ++l) { loc_idx[l] = -1; loc_val[l] = -1.0e30f; }
    for (int i0 = threadIdx.x; i0 < eq_count; i0 += blockDim.x) {
        int idx = eq_buf[i0];
        if (idx < 0) continue;
        float v = col[idx];
        int min_pos = 0; float min_val = loc_val[0];
        for (int l = 1; l < LOCAL_TOP; ++l) if (loc_val[l] < min_val) { min_val = loc_val[l]; min_pos = l; }
        if (v > min_val) { loc_val[min_pos] = v; loc_idx[min_pos] = idx; }
    }
    // compact per-thread local top into eq_buf contiguously
    __shared__ int pool_count;
    if (threadIdx.x == 0) pool_count = 0;
    __syncthreads();
    for (int l = 0; l < LOCAL_TOP; ++l) {
        int idx = loc_idx[l];
        if (idx >= 0) {
            int pos = atomicAdd(&pool_count, 1);
            if (pos < eq_capacity) eq_buf[pos] = idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (pool_count > eq_capacity) pool_count = eq_capacity;
        eq_count = pool_count;
    }
    __syncthreads();

    // Fill remaining from reduced pool
    for (int r = 0; r < remaining; ++r) {
        __syncthreads();
        float best_val = -1.0e30f;
        int best_idx = -1;
        for (int i0 = threadIdx.x; i0 < eq_count; i0 += blockDim.x) {
            int idx = eq_buf[i0];
            if (idx < 0) continue; // skip removed entries
            float v = col[idx];
            if (v > best_val || (v == best_val && idx < best_idx)) {
                best_val = v; best_idx = idx;
            }
        }
        // reduce to find block-best
        // warp-level reductions
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
            // reduce across warps
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
                    int prev = sel_count;
                    int pos = atomicAdd(&sel_count, 1);
                    if (SEL_DEBUG && blockIdx.x == SEL_DEBUG_COL) {
                        printf("[t=%d] select iter: best_idx=%d prev_sel=%d new_sel=%d\n", t, s_selected, prev, sel_count);
                    }
                    if (pos < k) {
                        int clamped = s_selected < 0 ? 0 : (s_selected >= N ? (N - 1) : s_selected);
                        idx_out[pos + k*t] = clamped;
                    }
                }
            }
        }
        __syncthreads();
        // remove candidate from eq_buf only in non-streaming path
        if (s_selected >= 0) {
            for (int j = threadIdx.x; j < eq_count; j += blockDim.x) {
                if (eq_buf[j] == s_selected) { eq_buf[j] = -1; }
            }
        }
        __syncthreads();
        if (SEL_DEBUG && blockIdx.x == SEL_DEBUG_COL && threadIdx.x == 0) {
            printf("[t=%d] end iter: sel_count=%d\n", t, sel_count);
        }
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

    const int hist_threads = 256;
    const size_t hist_shmem = 256 * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    // Equal-bin selection kernel; bound dynamic shared memory to device limit
    const int sel_threads = 256;
    // Conservative eq buffer capacity to avoid exceeding per-block shared mem
    const int eq_cap = max(k, min(N, 12000));
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

    const int sel_threads = 256;
    const int eq_cap_host = max(k, min(N, 4096));
    const size_t sel_shmem = (size_t) eq_cap_host * sizeof(int);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap_host, gt_counts_d, idx_d);

    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(idx_d);
}
