#include "topk-radix.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include "../../include/ggml-cuda-radix.h"


// simple bitonic top-k per column (descending)
static __global__ void k_topk_desc_f32_i32(const float * x, int * dst, int ncols, int nrows, int k, int ncols_pad) {
    int col = threadIdx.x;
    int row = blockIdx.y;
    if (row >= nrows) return;
    if (col >= ncols_pad) return;
    extern __shared__ int idx[];
    const float * x_row = x + row * ncols;
    idx[col] = col;
    __syncthreads();
    // bitonic sort indices by x_row[idx]
    for (int K = 2; K <= ncols_pad; K <<= 1) {
        for (int J = K >> 1; J > 0; J >>= 1) {
            int ixj = col ^ J;
            if (ixj > col) {
                if ((col & K) == 0) {
                    // ascending within block, but we want overall descending -> flip comparisons
                    if (idx[col] >= ncols || (idx[ixj] < ncols && x_row[idx[col]] < x_row[idx[ixj]])) {
                        int tmp = idx[col]; idx[col] = idx[ixj]; idx[ixj] = tmp;
                    }
                } else {
                    if (idx[ixj] >= ncols || (idx[col] < ncols && x_row[idx[col]] > x_row[idx[ixj]])) {
                        int tmp = idx[col]; idx[col] = idx[ixj]; idx[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    // write top-k
    if (col < k && col < ncols) {
        dst[row * k + col] = idx[col];
    }
}

// float -> key mapping ascending; to get descending selection we pick largest keys
static __device__ __forceinline__ uint32_t float_to_key_desc(float x) {
    uint32_t u = __float_as_uint(x);
    if ((int32_t)u < 0) {
        return ~u;
    } else {
        return u | 0x80000000u;
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
                                          int N, int T, int ld, int k,
                                          const uint32_t * __restrict__ gt_counts, // [256, T]
                                          int * __restrict__ idx_out) {            // [k, T]
    int t = blockIdx.x;
    if (t >= T) return;

    const float * col = scores + (size_t)ld * t;
    // find thr0 from gt_counts
    int thr0 = 0;
    uint32_t gt = 0; (void)gt;
    for (int b = 255; b >= 0; --b) {
        uint32_t sgt = gt_counts[b + 256*t];
        // reconstruct eq as gt_counts[b] - gt_counts[b-1] with special case; approximate by counting on-the-fly
        // simpler: compute sgt and then scan eq by reading column - but that is expensive; instead, rely on two-phase approach:
        // Here we will compute eq by reading column, but only once.
        // Decide thr candidate by estimating eq via another pass below.
        // Pick the first b where sgt < k and break; we'll refine below.
        if (sgt < (uint32_t)k) { thr0 = b; gt = sgt; break; }
    }
    // Now collect selected (> thr0) and equality list; parallelize by striding threads
    extern __shared__ int shared[];
    int * eq_buf = shared; // temporary storage of eq indices, length N in worst case; we limit by blockDim.x cooperative compaction
    __shared__ int eq_count;
    __shared__ int sel_count;
    if (threadIdx.x == 0) { eq_count = 0; sel_count = 0; }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t key = float_to_key_desc(col[i]);
        int bin = (key >> 24) & 0xFF;
        if (bin > thr0) {
            int pos = atomicAdd(&sel_count, 1);
            if (pos < k) idx_out[pos + k*t] = i;
        } else if (bin == thr0) {
            int pos = atomicAdd(&eq_count, 1);
            eq_buf[pos] = i;
        }
    }
    __syncthreads();

    int remaining = k - sel_count;
    if (remaining <= 0) return;

    // If eq_count is large, do a simple partial selection by values among eq candidates
    // For simplicity, do a block-wide partial selection using naive nth_element-like iterations (small k typical, e.g., <= 128)
    // We will fill remaining slots with the largest values among eq_buf.

    // First: load values of eq candidates into shared memory in chunks
    // Use simple repeated passes selecting max remaining each time
    for (int r = 0; r < remaining; ++r) {
        float best_val = -1.0e30f;
        int best_idx = -1;
        for (int i0 = threadIdx.x; i0 < eq_count; i0 += blockDim.x) {
            int idx = eq_buf[i0];
            float v = col[idx];
            // compare via value
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
                // write selection and mark this candidate to -inf so it won't be selected again
                int pos = atomicAdd(&sel_count, 1);
                if (pos < k) idx_out[pos + k*t] = bi;
                // mark by setting its value in eq_buf to sentinel index that thread comparisons will skip
                // simplest is to set eq candidate value to -inf by writing to global col? cannot. So we mark by setting index to -1.
                // We need to remove candidate bi from eq_buf: do a pass to set matches to -1
                for (int j = lane; j < eq_count; j += 32) { if (eq_buf[j] == bi) { eq_buf[j] = -1; } }
            }
        }
        __syncthreads();
        // threads skip -1 entries next iteration implicitly since they yield best_idx=-1 which loses ties
    }
}

void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    // Simple bitonic per-column argsort then take top-k indices
    int ncols = N;
    int nrows = T;
    int ncols_pad = 1; while (ncols_pad < ncols) ncols_pad <<= 1;
    dim3 block_dims(ncols_pad, 1, 1);
    dim3 grid_dims(1, nrows, 1);
    size_t shmem = ncols_pad * sizeof(int);
    k_topk_desc_f32_i32<<<grid_dims, block_dims, shmem, stream>>>(scores_d, idx_d, ncols, nrows, k, ncols_pad);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA topk kernel launch error: %s\n", cudaGetErrorString(err));
    }
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
