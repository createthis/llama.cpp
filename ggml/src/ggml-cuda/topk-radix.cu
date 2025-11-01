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

// select indices using multi-pass streaming refinement, no large shared memory
static __global__ void k_select_topk_bins(const float * __restrict__ scores,
                                          int N, int T, int ld, int k,
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

    __shared__ int sel_count;
    if (threadIdx.x == 0) sel_count = 0;
    __syncthreads();

    // Phase A: collect strictly greater than thr0
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int bin = (key >> 24) & 0xFF;
        if (bin > thr0) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) {
                idx_out[pos + k*t] = i;
            }
        }
    }
    __syncthreads();

    int remaining = k - sel_count;
    if (remaining <= 0) return;

    // thresholds for sub-bytes
    int thr1 = 255, thr2 = 255, thr3 = 255;

    // Helper lambda: compute histogram for candidates equal to previous bytes
    auto hist_pass = [&](int shift, int thr_prev2, int thr_prev1) {
        __shared__ unsigned int c2[256];
        for (int i = threadIdx.x; i < 256; i += blockDim.x) c2[i] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t key = float_to_key_desc(col[i]);
            int b0 = (key >> 24) & 0xFF;
            if (b0 != thr0) continue;
            if (shift <= 16 && thr_prev2 >= 0) {
                int b1 = (key >> 16) & 0xFF;
                if (b1 != thr_prev2) continue;
            }
            if (shift <= 8 && thr_prev1 >= 0) {
                int b2 = (key >> 8) & 0xFF;
                if (b2 != thr_prev1) continue;
            }
            int bb = (key >> shift) & 0xFF;
            atomicAdd(&c2[bb], 1u);
        }
        __syncthreads();
        // compute threshold for this pass
        int thr = 255;
        if (threadIdx.x == 0) {
            unsigned int sum = 0;
            for (int b = 255; b >= 0; --b) {
                unsigned int sgt = sum;
                unsigned int prev = sum + c2[b];
                if ((unsigned int)remaining > sgt && (unsigned int)remaining <= prev) { thr = b; break; }
                sum = prev;
            }
        }
        __syncthreads();
        return thr;
    };

    // pass 1: bits 16..23
    thr1 = hist_pass(16, -1, -1);
    __syncthreads();
    // emit strictly greater than thr1 among candidates eq thr0
    for (int i = threadIdx.x; i < N && remaining > 0; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
        int b1 = (key >> 16) & 0xFF; if (b1 > thr1) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();
    remaining = k - sel_count;
    if (remaining <= 0) return;

    // pass 2: bits 8..15
    thr2 = hist_pass(8, thr1, -1);
    __syncthreads();
    for (int i = threadIdx.x; i < N && remaining > 0; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
        int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue;
        int b2 = (key >> 8) & 0xFF; if (b2 > thr2) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();
    remaining = k - sel_count;
    if (remaining <= 0) return;

    // pass 3: bits 0..7
    thr3 = hist_pass(0, thr1, thr2);
    __syncthreads();
    for (int i = threadIdx.x; i < N && remaining > 0; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue;
        int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue;
        int b2 = (key >> 8) & 0xFF;  if (b2 != thr2) continue;
        int b3 = (key >> 0) & 0xFF;  if (b3 > thr3) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();
    remaining = k - sel_count;
    if (remaining <= 0) return;

    // Final fill: keys exactly equal to threshold bytes
    for (int i = threadIdx.x; i < N && remaining > 0; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int b0 = (key >> 24) & 0xFF;
        int b1 = (key >> 16) & 0xFF;
        int b2 = (key >> 8)  & 0xFF;
        int b3 = (key >> 0)  & 0xFF;
        if (b0 == thr0 && b1 == thr1 && b2 == thr2 && b3 == thr3) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();

    // Robust fallback: if still not enough, iteratively pick best remaining values
    remaining = k - sel_count;
    if (remaining > 0) {
        __shared__ int sel_total;
        if (threadIdx.x == 0) sel_total = sel_count;
        __syncthreads();
        for (int r = 0; r < remaining; ++r) {
            float best_val = -1.0e30f;
            int best_idx = -1;
            // scan entire column to find best not yet selected
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                // membership check against already written selections
                bool already = false;
                int limit = sel_total + r;
                for (int j = 0; j < limit; ++j) {
                    if (idx_out[j + k*t] == i) { already = true; break; }
                }
                if (already) continue;
                float v = col[i];
                if (v > best_val || (v == best_val && (best_idx < 0 || i < best_idx))) {
                    best_val = v; best_idx = i;
                }
            }
            // warp-level reduce
            for (int offset = 16; offset > 0; offset >>= 1) {
                float v2 = __shfl_down_sync(0xffffffff, best_val, offset);
                int   i2 = __shfl_down_sync(0xffffffff, best_idx, offset);
                if (v2 > best_val || (v2 == best_val && (best_idx < 0 || i2 < best_idx))) { best_val = v2; best_idx = i2; }
            }
            __shared__ float warp_best_val2[32];
            __shared__ int   warp_best_idx2[32];
            int wid = threadIdx.x >> 5; int lane = threadIdx.x & 31;
            if (lane == 0) { warp_best_val2[wid] = best_val; warp_best_idx2[wid] = best_idx; }
            __syncthreads();
            int num_warps = (blockDim.x + 31) / 32;
            float bv = (lane < num_warps) ? warp_best_val2[lane] : -1.0e30f;
            int   bi = (lane < num_warps) ? warp_best_idx2[lane] : -1;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float v2 = __shfl_down_sync(0xffffffff, bv, offset);
                int   i2 = __shfl_down_sync(0xffffffff, bi, offset);
                if (v2 > bv || (v2 == bv && (bi < 0 || i2 < bi))) { bv = v2; bi = i2; }
            }
            if (lane == 0 && wid == 0) {
                int outp = sel_total + r;
                if (outp < k && bi >= 0) idx_out[outp + k*t] = bi;
            }
            __syncthreads();
        }
    }
}

void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int ld, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    // Radix-like path: histogram top byte + select with tie refinement
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d  = nullptr;
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    const int hist_threads = 256;
    const size_t hist_shmem = 256 * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, ld, thr_bins_d, gt_counts_d);

    // Equal-bin selection kernel; streaming refinement uses minimal shared memory
    const int sel_threads = 256;
    const size_t sel_shmem = 0;
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, ld, k, gt_counts_d, idx_d);

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
    ggml_cuda_topk_radix_indices_device(ctx, scores_d, N, T, N, k, idx_d);
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
    const size_t sel_shmem = 0;
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, gt_counts_d, idx_d);

    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(idx_d);
}
