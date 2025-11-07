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

    // Round 0: use provided greater-counts to find threshold on top byte
    int thr0 = 0;
    for (int b = 255; b >= 0; --b) {
        uint32_t sgt = gt_counts[b + 256*t];
        uint32_t prev = (b == 0 ? (uint32_t)N : gt_counts[(b - 1) + 256*t]);
        uint32_t eq   = prev - gt_counts[b + 256*t];
        if (sgt < (uint32_t)k && sgt + eq >= (uint32_t)k) { thr0 = b; break; }
    }
    __shared__ int sel_sofar;
    if (threadIdx.x == 0) sel_sofar = 0;
    __syncthreads();

    uint32_t sgt0 = gt_counts[thr0 + 256*t];
    int take_gt0 = min(k, (int)sgt0);

    extern __shared__ int eq_buf[]; // dynamic shared memory as candidate buffer
    __shared__ int eq_count0_store;
    if (threadIdx.x == 0) eq_count0_store = 0;
    __shared__ int eq_count0_total;
    if (threadIdx.x == 0) eq_count0_total = 0;
    __syncthreads();

    // Select MSB > thr0 and collect MSB == thr0 candidates
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = __float_as_uint(col[i]);
        key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
        int b0 = (key >> 24) & 0xFF;
        if (b0 > thr0) {
            int pos = atomicAdd(&sel_sofar, 1);
            if (pos < take_gt0) idx_out[pos + k*t] = i;
        } else if (b0 == thr0) {
            atomicAdd(&eq_count0_total, 1);
            int p = atomicAdd(&eq_count0_store, 1);
            if (p < eq_capacity) eq_buf[p] = i;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) sel_sofar = take_gt0;
    __syncthreads();

    int remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Helper lambdas (device inline) simulated via macros
    // Build histogram for next byte over a set (either eq buffer if fully stored, or full column with predicates)
    auto build_hist = [&](int round, int count_stored, int count_total, int prev_thr0, int prev_thr1, int prev_thr2, unsigned int *hist_out){
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist_out[i] = 0u;
        __syncthreads();
        if (count_stored == count_total && count_total <= eq_capacity) {
            // iterate stored candidates
            for (int j = threadIdx.x; j < count_stored; j += blockDim.x) {
                int idx = eq_buf[j];
                uint32_t key = __float_as_uint(col[idx]);
                key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
                int byte;
                if (round == 1) byte = (key >> 16) & 0xFF; // b1
                else if (round == 2) byte = (key >> 8) & 0xFF; // b2
                else byte = key & 0xFF; // b3
                atomicAdd(&hist_out[byte], 1u);
            }
        } else {
            // iterate full column with predicates on previous rounds' bytes
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                uint32_t raw = __float_as_uint(col[i]);
                uint32_t key = ((int32_t)raw < 0) ? ~raw : (raw ^ 0x80000000u);
                int b0 = (key >> 24) & 0xFF; if (b0 != prev_thr0) continue;
                if (round >= 2) { int b1 = (key >> 16) & 0xFF; if (b1 != prev_thr1) continue; }
                if (round >= 3) { int b2 = (key >> 8) & 0xFF; if (b2 != prev_thr2) continue; }
                int byte;
                if (round == 1) byte = (key >> 16) & 0xFF; // b1
                else if (round == 2) byte = (key >> 8) & 0xFF; // b2
                else byte = key & 0xFF; // b3
                atomicAdd(&hist_out[byte], 1u);
            }
        }
        __syncthreads();
    };

    // Round 1 (b1)
    __shared__ unsigned int h1[256];
    build_hist(1, eq_count0_store, eq_count0_total, thr0, 0, 0, h1);
    __shared__ int thr1;
    if (threadIdx.x == 0) {
        unsigned int sum = 0; unsigned int need = remaining;
        thr1 = 255;
        for (int b = 255; b >= 0; --b) { unsigned int sgt = sum; unsigned int eqb = h1[b]; if (sgt < need && sgt + eqb >= need) { thr1 = b; break; } sum += eqb; }
    }
    __syncthreads();

    // Select b1 > thr1; collect b1 == thr1 for next round
    __shared__ int eq_count1_store; if (threadIdx.x == 0) eq_count1_store = 0; __shared__ int eq_count1_total; if (threadIdx.x == 0) eq_count1_total = 0; __syncthreads();
    if (eq_count0_store == eq_count0_total && eq_count0_total <= eq_capacity) {
        for (int j = threadIdx.x; j < eq_count0_store; j += blockDim.x) {
            int idx = eq_buf[j];
            uint32_t key = __float_as_uint(col[idx]); key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
            int b1 = (key >> 16) & 0xFF;
            if (b1 > thr1) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = idx; }
            else if (b1 == thr1) { atomicAdd(&eq_count1_total, 1); int p = atomicAdd(&eq_count1_store, 1); if (p < eq_capacity) eq_buf[p] = idx; }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw = __float_as_uint(col[i]); uint32_t key = ((int32_t)raw < 0) ? ~raw : (raw ^ 0x80000000u);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue; int b1 = (key >> 16) & 0xFF;
            if (b1 > thr1) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = i; }
            else if (b1 == thr1) { atomicAdd(&eq_count1_total, 1); int p = atomicAdd(&eq_count1_store, 1); if (p < eq_capacity) eq_buf[p] = i; }
        }
    }
    __syncthreads();
    remaining = k - sel_sofar; if (remaining <= 0) return;

    // Round 2 (b2)
    __shared__ unsigned int h2[256];
    build_hist(2, eq_count1_store, eq_count1_total, thr0, thr1, 0, h2);
    __shared__ int thr2;
    if (threadIdx.x == 0) {
        unsigned int sum = 0; unsigned int need = remaining; thr2 = 255;
        for (int b = 255; b >= 0; --b) { unsigned int sgt = sum; unsigned int eqb = h2[b]; if (sgt < need && sgt + eqb >= need) { thr2 = b; break; } sum += eqb; }
    }
    __syncthreads();

    __shared__ int eq_count2_store; if (threadIdx.x == 0) eq_count2_store = 0; __shared__ int eq_count2_total; if (threadIdx.x == 0) eq_count2_total = 0; __syncthreads();
    if (eq_count1_store == eq_count1_total && eq_count1_total <= eq_capacity) {
        for (int j = threadIdx.x; j < eq_count1_store; j += blockDim.x) {
            int idx = eq_buf[j]; uint32_t key = __float_as_uint(col[idx]); key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
            int b2 = (key >> 8) & 0xFF; if (b2 > thr2) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = idx; }
            else if (b2 == thr2) { atomicAdd(&eq_count2_total, 1); int p = atomicAdd(&eq_count2_store, 1); if (p < eq_capacity) eq_buf[p] = idx; }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw = __float_as_uint(col[i]); uint32_t key = ((int32_t)raw < 0) ? ~raw : (raw ^ 0x80000000u);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue; int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue; int b2 = (key >> 8) & 0xFF;
            if (b2 > thr2) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = i; }
            else if (b2 == thr2) { atomicAdd(&eq_count2_total, 1); int p = atomicAdd(&eq_count2_store, 1); if (p < eq_capacity) eq_buf[p] = i; }
        }
    }
    __syncthreads(); remaining = k - sel_sofar; if (remaining <= 0) return;

    // Round 3 (b3)
    __shared__ unsigned int h3[256];
    build_hist(3, eq_count2_store, eq_count2_total, thr0, thr1, thr2, h3);
    __shared__ int thr3;
    if (threadIdx.x == 0) {
        unsigned int sum = 0; unsigned int need = remaining; thr3 = 255;
        for (int b = 255; b >= 0; --b) { unsigned int sgt = sum; unsigned int eqb = h3[b]; if (sgt < need && sgt + eqb >= need) { thr3 = b; break; } sum += eqb; }
    }
    __syncthreads();

    // Select b3>thr3
    if (eq_count2_store == eq_count2_total && eq_count2_total <= eq_capacity) {
        for (int j = threadIdx.x; j < eq_count2_store; j += blockDim.x) {
            int idx = eq_buf[j]; uint32_t key = __float_as_uint(col[idx]); key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
            int b3 = key & 0xFF; if (b3 > thr3) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = idx; }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw = __float_as_uint(col[i]); uint32_t key = ((int32_t)raw < 0) ? ~raw : (raw ^ 0x80000000u);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue; int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue; int b2 = (key >> 8) & 0xFF; if (b2 != thr2) continue; int b3 = key & 0xFF;
            if (b3 > thr3) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = i; }
        }
    }
    __syncthreads(); remaining = k - sel_sofar; if (remaining <= 0) return;

    // Final: fill equals of last byte up to remaining
    if (eq_count2_store == eq_count2_total && eq_count2_total <= eq_capacity) {
        for (int j = threadIdx.x; j < eq_count2_store; j += blockDim.x) {
            if (remaining <= 0) break; // not atomic-safe but guarded by sel_sofar below
            int idx = eq_buf[j]; uint32_t key = __float_as_uint(col[idx]); key = ((int32_t)key < 0) ? ~key : (key ^ 0x80000000u);
            int b3 = key & 0xFF; if (b3 == thr3) { int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = idx; }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (remaining <= 0) break;
            uint32_t raw = __float_as_uint(col[i]); uint32_t key = ((int32_t)raw < 0) ? ~raw : (raw ^ 0x80000000u);
            int b0 = (key >> 24) & 0xFF; if (b0 != thr0) continue; int b1 = (key >> 16) & 0xFF; if (b1 != thr1) continue; int b2 = (key >> 8) & 0xFF; if (b2 != thr2) continue; int b3 = key & 0xFF; if (b3 != thr3) continue;
            int pos = atomicAdd(&sel_sofar, 1); if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();

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
