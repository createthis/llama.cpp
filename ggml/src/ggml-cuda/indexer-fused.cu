#include "common.cuh"
#ifndef LLAMA_ENABLE_CP_ASYNC
#define LLAMA_ENABLE_CP_ASYNC 1
#endif

#include <mma.h>
using namespace nvcuda;

#include <cuda_runtime.h>
#include <cuda.h>

#include <cute/arch/copy_sm90_desc.hpp> // for cute::TmaDescriptor
#include <cuda_pipeline_primitives.h>

#if CUDART_VERSION >= 12000
extern "C" void ggml_deepgemm_paged_mqa_logits_sm100(
    uint32_t batch_size,
    uint64_t logits_stride,
    uint64_t block_table_stride,
    const uint32_t * context_lens,
    float * logits,
    const uint32_t * block_table,
    const uint32_t * schedule_meta,
    cute::TmaDescriptor tma_q,
    cute::TmaDescriptor tma_kv,
    cute::TmaDescriptor tma_kv_scales,
    cute::TmaDescriptor tma_w,
    uint32_t next_n,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_kv,
    bool     is_context_lens_2d,
    uint32_t num_q_stages,
    uint32_t num_kv_stages,
    uint32_t split_kv,
    uint32_t num_specialized_threads,
    uint32_t num_math_threads,
    cudaStream_t stream,
    size_t shmem_bytes);

static inline bool dg_fp8_encode_tma_2d(
    cute::TmaDescriptor &desc,
    CUtensorMapDataType type,
    void *base,
    uint32_t inner_dim, uint32_t outer_dim,
    uint32_t box_inner, uint32_t box_outer,
    uint32_t outer_stride_elems,
    size_t elem_size) {
    cuuint64_t dims[2]    = { (cuuint64_t) inner_dim, (cuuint64_t) outer_dim };
    cuuint64_t strides[1] = { (cuuint64_t) outer_stride_elems * (cuuint64_t) elem_size };
    cuuint32_t box[2]     = { (cuuint32_t) box_inner, (cuuint32_t) box_outer };
    cuuint32_t elem_strides[2] = { 1u, 1u };
    CUresult res = cuTensorMapEncodeTiled(
        reinterpret_cast<CUtensorMap*>(&desc), type,
        2u, base, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

static inline bool dg_fp8_encode_tma_3d(
    cute::TmaDescriptor &desc,
    CUtensorMapDataType type,
    void *base,
    uint32_t dim0, uint32_t dim1, uint32_t dim2,
    uint32_t box0, uint32_t box1, uint32_t box2,
    uint32_t stride0_elems, uint32_t stride1_elems,
    size_t elem_size) {
    cuuint64_t dims[3]    = { (cuuint64_t) dim0, (cuuint64_t) dim1, (cuuint64_t) dim2 };
    cuuint64_t strides[2] = {
        (cuuint64_t) stride0_elems * (cuuint64_t) elem_size,
        (cuuint64_t) stride1_elems * (cuuint64_t) elem_size
    };
    cuuint32_t box[3]     = { (cuuint32_t) box0, (cuuint32_t) box1, (cuuint32_t) box2 };
    cuuint32_t elem_strides[3] = { 1u, 1u, 1u };
    CUresult res = cuTensorMapEncodeTiled(
        reinterpret_cast<CUtensorMap*>(&desc), type,
        3u, base, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}
#endif // CUDART_VERSION >= 12000


#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "../../include/ggml-cuda-indexer.h"

struct fp8_e4_t;
extern "C" int call(fp8_e4_t* __restrict__ IndexQ, fp8_e4_t* __restrict__ IndexK, float* __restrict__ IndexKScale, float* __restrict__ Logits, float* __restrict__ Weights, int* __restrict__ CuSeqLenKS, int* __restrict__ CuSeqLenKE, int seq_len_kv, int seq_len, cudaStream_t stream);
extern "C" int init();
extern "C" const char* get_last_error();


#ifndef SEL_DEBUG
#endif


#if __CUDA_ARCH__ >= 800
// Forward declarations for cp.async helpers used by kernels defined earlier in this file
static __device__ __forceinline__ void cp_async_16B_all(void* __restrict__ dst, const void* __restrict__ src, size_t bytes);
static __device__ __forceinline__ void cp_async_16B_issue_all(void* __restrict__ dst, const void* __restrict__ src, size_t bytes);
#endif

#ifndef LAUNCH_PROFILE_KERNEL
#define LAUNCH_PROFILE_KERNEL(TAG_STR, TAGNAME, STREAM, LAUNCH_STMT, D_, H_, Tc_, KV_) do { \
    if (__prof_env && *__prof_env) { \
        cudaEvent_t __e0, __e1; cudaEventCreate(&__e0); cudaEventCreate(&__e1); \
        cudaEventRecord(__e0, STREAM); \
        LAUNCH_STMT; \
        cudaEventRecord(__e1, STREAM); \
        cudaEventSynchronize(__e1); \
        float __ms = 0.0f; cudaEventElapsedTime(&__ms, __e0, __e1); \
        cudaEventDestroy(__e0); cudaEventDestroy(__e1); \
        static int __cnt_##TAGNAME = 0; \
        static double __sum_##TAGNAME = 0.0; \
        __sum_##TAGNAME += __ms; \
        __cnt_##TAGNAME++; \
        if (__prof_each_env && *__prof_each_env) { \
            fprintf(stderr, "[" TAG_STR "] TILELANG_INDEXER D=%d H=%d Tc=%d kv=%d ms=%.3f\n", D_, H_, Tc_, KV_, __ms); \
        } else if (__cnt_##TAGNAME % 50 == 0) { \
            fprintf(stderr, "[" TAG_STR "] TILELANG_INDEXER D=%d H=%d Tc=%d kv=%d avg_ms=%.3f over 50 calls\n", D_, H_, Tc_, KV_, (float)(__sum_##TAGNAME/50.0)); \
            __sum_##TAGNAME = 0.0; \
        } \
    } else { \
        LAUNCH_STMT; \
    } \
} while(0)
#endif

#if __CUDA_ARCH__ >= 800 && defined(LLAMA_ENABLE_CP_ASYNC)
static __device__ inline void cp_async_16b(void * smem_ptr, const void * gmem_ptr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_ptr));
}
static __device__ inline void cp_async_commit() {
    asm volatile ("cp.async.commit_group;\n" ::);
}
static __device__ inline void cp_async_wait() {
    asm volatile ("cp.async.wait_group 0;\n" ::);
}
#endif

// helper kernels
static __device__ __forceinline__ uint8_t f32_to_fp8e4m3(float);
static __device__ __forceinline__ float fp8e4m3_to_f32(uint8_t);

__global__ void k_fill_int(int *arr, int n, int val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void k_colmajor_DN_to_rowmajor_ND(const float *src, int D, int N, float *dst) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && d < D) {
        dst[(size_t)n * (size_t)D + (size_t)d] = src[(size_t)d + (size_t)D * (size_t)n];
    }
}
__global__ void k_transpose_TcKv_to_KvTc(const float *in, int Tc, int kv, float *out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < Tc && k < kv) out[k + (size_t)kv * t] = in[(size_t)t * kv + k];
}

// Row-major float32 -> FP8 E4M3 packer (no per-row scaling)
__global__ void k_rowmajor_f32_to_fp8_e4m3(const float *src, int rows, int cols, unsigned char *dst) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)rows * (size_t)cols;
    if (idx < total) {
        float x = src[idx];
        __nv_fp8_storage_t v = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
        dst[idx] = (unsigned char) v;
    }
}

// Row-major float32 -> per-row absolute max
__global__ void k_rowmajor_f32_rowwise_absmax(const float *src, int rows, int cols, float *row_amax) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float maxv = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = fabsf(src[(size_t)row * (size_t)cols + (size_t)c]);
        if (v > maxv) maxv = v;
    }
    row_amax[row] = maxv;
}

// Compute per-row FP8 scales (amax/448) and their reciprocals
__global__ void k_fp8_compute_row_scales(const float *row_amax, int rows, float *sf, float *inv_sf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float a = row_amax[i];
    if (a < 1e-4f) a = 1e-4f;
    float s = a / 448.0f;
    sf[i] = s;
    inv_sf[i] = 1.0f / s;
}

// Row-major float32 -> FP8 E4M3 with per-row scaling
__global__ void k_rowmajor_f32_to_fp8_e4m3_rowwise_scaled(const float *src, int rows, int cols,
                                                          const float *inv_sf, unsigned char *dst) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)rows * (size_t)cols;
    if (idx >= total) return;
    int row = (int)(idx / (size_t)cols);
    float x = src[idx] * inv_sf[row];
    __nv_fp8_storage_t v = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    dst[idx] = (unsigned char) v;
}

// Elementwise product of two float vectors
__global__ void k_elemwise_mul(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

// helpers to read env
static inline int getenv_int_(const char * name, int def) {
    const char * s = getenv(name);
    if (!s || !*s) return def;
    int v = atoi(s);
    return v > 0 ? v : def;
}


// Simple baseline fused kernel: compute K^T * Q -> ReLU, then per-head weighted sum, multiply k_scale.
// This is a placeholder for a fully-optimized version. It assumes row-major contiguous inputs.
static inline bool sparse_debug_on(){ const char *d=getenv("LLAMA_SPARSE_DEBUG_INDEXER"); return d && *d && atoi(d)!=0; }


// Tiled, shared-memory fused kernel (float inputs, float accum)
// Q: [D, Tc*H], K: [D, kv], W: [H, Tc], k_scale: [kv]; Out: [kv, Tc]
__global__ void k_indexer_logits_tiled_f32(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ W,
    const float * __restrict__ k_scale,
    int D, int H, int Tc, int kv,
    const int * __restrict__ starts,
    const int * __restrict__ ends,
    int D_TILE, int BLOCK_Q, int BLOCK_N, int exact_flag,
    int HEAD_CHUNK_ARG,
    int PIPE_STAGES_ARG,
    float * __restrict__ Out) {
    __shared__ int s_min_blk;
    __shared__ int s_max_blk;
    __shared__ float s_K_sf[512];

    // Dynamic select exact vs optimized based on workload or env
    bool exact = (exact_flag != 0);
    /* env read on host */
    (void)Tc; (void)kv;
    if (exact) {
        // Exact global-load path (bit-exact with reference; slower)
        int t_local = threadIdx.x;
        int k_local = threadIdx.y;
        int t0 = blockIdx.x * BLOCK_Q;
        int k0 = blockIdx.y * BLOCK_N;
        int token = t0 + t_local;
        int kv_idx = k0 + k_local;
        // Compute union window for this block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int smin = 0; int smax = kv;
            if (starts != nullptr && ends != nullptr) {
                smin = kv; smax = 0;
                for (int q = 0; q < BLOCK_Q; ++q) {
                    int tok = t0 + q;
                    if (tok < Tc) {
                        int s0 = starts[tok]; int e0 = ends[tok];
                        if (s0 < 0) s0 = 0; if (s0 > kv) s0 = kv;
                        if (e0 < 0) e0 = 0; if (e0 > kv) e0 = kv;
                        if (s0 < smin) smin = s0;
                        if (e0 > smax) smax = e0;
                    }
                }
                if (smin > smax) smin = smax;
            }
            s_min_blk = smin; s_max_blk = smax;
        }
        __syncthreads();
        int smin_blk = s_min_blk; int smax_blk = s_max_blk;
        bool in_bounds = (t_local < BLOCK_Q) && (k_local < BLOCK_N) && (token < Tc) && (kv_idx < kv);
        bool in_union  = (kv_idx >= smin_blk && kv_idx < smax_blk);

        float acc = 0.0f;
        if (in_bounds && in_union) {
            for (int h = 0; h < H; ++h) {
                const float *qv = Q + (size_t)D * (token*H + h);
                const float *kvp= K + (size_t)D * kv_idx;
                float dot = 0.0f;
                #pragma unroll 1
                for (int d = 0; d < D; ++d) dot += qv[d] * kvp[d];
                if (dot < 0.0f) dot = 0.0f;
                acc += dot * W[h + (size_t)H * token];
            }
            acc *= k_scale[kv_idx];
            if (starts != nullptr && ends != nullptr) {
                int s0 = starts[token]; int e0 = ends[token];
                if (s0 < 0) s0 = 0; if (s0 > kv) s0 = kv;
                if (e0 < 0) e0 = 0; if (e0 > kv) e0 = kv;
                if (kv_idx < s0 || kv_idx >= e0) acc = 0.0f;
            }
        }
        if (in_bounds) {
            Out[kv_idx + (size_t)kv * token] = acc;
        }
        return;
    }
    // Optimized shared-memory tiled path with head-chunked reduction
    int t_local = threadIdx.x; // [0..BLOCK_Q)
    int k_local = threadIdx.y; // [0..BLOCK_N)
    int t0 = blockIdx.x * BLOCK_Q;
    int k0 = blockIdx.y * BLOCK_N;
    int token = t0 + t_local;
    int kv_idx = k0 + k_local;
    // Compute union window for this block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int smin = 0; int smax = kv;
        if (starts != nullptr && ends != nullptr) {
            smin = kv; smax = 0;
            for (int q = 0; q < BLOCK_Q; ++q) {
                int tok = t0 + q;
                if (tok < Tc) {
                    int s0 = starts[tok]; int e0 = ends[tok];
                    if (s0 < 0) s0 = 0; if (s0 > kv) s0 = kv;
                    if (e0 < 0) e0 = 0; if (e0 > kv) e0 = kv;
                    if (s0 < smin) smin = s0;
                    if (e0 > smax) smax = e0;
                }
            }
            if (smin > smax) smin = smax;
        }
        s_min_blk = smin; s_max_blk = smax;
    }
    __syncthreads();

    // Per-row K scale (amax/448) for BLOCK_N kv rows in this tile, matching CPU FP8 path
    if (threadIdx.x == 0) {
        int j = threadIdx.y;
        if (j < BLOCK_N) {
            int row = k0 + j;
            float maxv = 0.0f;
            if (row < kv) {
                for (int d0 = 0; d0 < D; ++d0) {
                    float v = K[(size_t)d0 + (size_t)D * (size_t)row];
                    float av = fabsf(v);
                    if (av > maxv) maxv = av;
                }
            }
            if (maxv < 1e-4f) maxv = 1e-4f;
            s_K_sf[j] = maxv / 448.0f;
        }
    }
    __syncthreads();

    // Head-chunk size
    int Hc = HEAD_CHUNK_ARG > 0 ? HEAD_CHUNK_ARG : 16;
    if (Hc < 1) Hc = 1;
    if (Hc > H) Hc = H;

    extern __shared__ float shmem[];
    // Double-buffered layout if PIPE_STAGES_ARG >= 2:
    // [K0][Q0][K1][Q1][W] else [K0][Q0][W]
    int STAGES = (PIPE_STAGES_ARG >= 2 ? 2 : 1);
    int sizeK = D_TILE * BLOCK_N;
    int sizeQ = D_TILE * BLOCK_Q * Hc;
    float * K0 = shmem;
    float * Q0 = K0 + sizeK;
    float * K1 = (STAGES == 2) ? (Q0 + sizeQ) : K0;
    float * Q1 = (STAGES == 2) ? (K1 + sizeK) : Q0;
    float * W_sh = (STAGES == 2) ? (Q1 + sizeQ) : (Q0 + sizeQ);

    // Accumulator per (kv,row) x (token,col)
    float acc = 0.0f;

    for (int h0 = 0; h0 < H; h0 += Hc) {
        int hc = min(Hc, H - h0);
#if SEL_DEBUG
      if(threadIdx.x==0 && blockIdx.x==0){
        printf("[IDX_DBG] tiled_f32 params: D=%d H=%d Tc=%d kv=%d BLOCK_Q=%d BLOCK_N=%d Hc=%d\n", D, H, Tc, kv, BLOCK_Q, BLOCK_N, hc);
      }
#endif

        // load weights W[h0:h0+hc, token-range]
        // cooperative load: map 2D [hc, BLOCK_Q]
        int stride2 = blockDim.x * blockDim.y;
        int tid2 = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid2; idx < hc*BLOCK_Q; idx += stride2) {
            int hi = idx / BLOCK_Q;
            int q  = idx % BLOCK_Q;
            int tok = t0 + q;
            float wv = 0.0f;
            if (tok < Tc) wv = W[(h0 + hi) + (size_t)H * tok];
            W_sh[hi * BLOCK_Q + q] = wv;
        }
        __syncthreads();

        // Compute S = K^T * Q over this head-chunk
        float sum_hc = 0.0f;
        // Accumulate per-head dot across D, then apply ReLU and weights once
        const int MAX_HC = 64;
        float dot_vec[MAX_HC];
        for (int i = 0; i < hc; ++i) dot_vec[i] = 0.0f;
        for (int d0 = 0, stage = 0; d0 < D; d0 += D_TILE, stage ^= 1) {
            int cur = min(D_TILE, D - d0);
            int stride = blockDim.x * blockDim.y;
            int tid = threadIdx.y * blockDim.x + threadIdx.x;

            // Select buffers
            float * Kbuf = (STAGES == 2 && (stage == 1)) ? K1 : K0;
            float * Qbuf = (STAGES == 2 && (stage == 1)) ? Q1 : Q0;

            // Cooperative (or cp.async) load Kbuf: [cur, BLOCK_N] and Qbuf: [cur, BLOCK_Q*hc]
#if __CUDA_ARCH__ >= 800 && defined(LLAMA_ENABLE_CP_ASYNC)
            if (PIPE_STAGES_ARG >= 2) {
                // K: cooperative load
                for (int idx = tid; idx < cur * BLOCK_N; idx += stride) {
                    int di = idx / BLOCK_N;
                    int j  = idx % BLOCK_N;
                    int gk = k0 + j;
                    float v = 0.0f;
                    if (gk < kv) v = K[(size_t)(d0 + di) + (size_t)D * gk];
                    Kbuf[di * BLOCK_N + j] = v;
                }
                // Q: cp.async 16B using transposed layout [BLOCK_Q*hc, D_TILE]
                int groups = (cur / 4);
                for (int idx = tid; idx < groups * BLOCK_Q * hc; idx += stride) {
                    int di4 = (idx % groups) * 4;
                    int rem = idx / groups;
                    int q   = rem % BLOCK_Q;
                    int hi  = rem / BLOCK_Q;
                    int gt  = t0 + q;
                    const float * gptr = (gt < Tc) ? &Q[(size_t)(d0 + di4) + (size_t)D * (gt*H + (h0 + hi))] : nullptr;
                    float * sptr = &Qbuf[(hi * BLOCK_Q + q) * D_TILE + di4];
                    if (gptr) {
                        cp_async_16b(sptr, gptr);
                    } else {
                        // zero-fill when out of range
                        reinterpret_cast<float4*>(sptr)[0] = make_float4(0.f,0.f,0.f,0.f);
                    }
                }
                cp_async_commit();
                cp_async_wait();
                __syncthreads();
                // Handle Q tail when cur % 4 != 0 via cooperative scalar loads into transposed storage
                int tail = cur & 3;
                if (tail) {
                    for (int idx = tid; idx < tail * BLOCK_Q * hc; idx += stride) {
                        int di = idx % tail;
                        int rem = idx / tail;
                        int q   = rem % BLOCK_Q;
                        int hi  = rem / BLOCK_Q;
                        int gt  = t0 + q;
                        float v = 0.0f;
                        if (gt < Tc) v = Q[(size_t)(d0 + (cur - tail) + di) + (size_t)D * (gt*H + (h0 + hi))];
                        Qbuf[(hi * BLOCK_Q + q) * D_TILE + (cur - tail) + di] = v;
                    }
                }
            } else
#endif
            {
                for (int idx = tid; idx < cur * BLOCK_N; idx += stride) {
                    int di = idx / BLOCK_N;
                    int j  = idx % BLOCK_N;
                    int gk = k0 + j;
                    float v = 0.0f;
                    if (gk < kv) v = K[(size_t)(d0 + di) + (size_t)D * gk];
                    Kbuf[di * BLOCK_N + j] = v;
                }
                for (int idx = tid; idx < cur * BLOCK_Q * hc; idx += stride) {
                    int di = idx / (BLOCK_Q * hc);
                    int rem = idx % (BLOCK_Q * hc);
                    int q   = rem % BLOCK_Q;
                    int hi  = rem / BLOCK_Q;
                    int gt  = t0 + q;
                    float v = 0.0f;
                    if (gt < Tc) v = Q[(size_t)(d0 + di) + (size_t)D * (gt*H + (h0 + hi))];
                    Qbuf[di * (BLOCK_Q * hc) + hi * BLOCK_Q + q] = v;
                }
                __syncthreads();
            }

            // Apply FP8 E4M3 quant/dequant in shared memory to match CPU Lightning Indexer
            // Kbuf: [cur, BLOCK_N] row-major, with per-row scale s_K_sf[j]
            for (int idx = tid; idx < cur * BLOCK_N; idx += stride) {
                int di = idx / BLOCK_N;
                int j  = idx % BLOCK_N;
                float x = Kbuf[di * BLOCK_N + j];
                float sf = s_K_sf[j];
                float scaled = x / sf;
                uint8_t code = f32_to_fp8e4m3(scaled);
                float dec = fp8e4m3_to_f32(code);
                Kbuf[di * BLOCK_N + j] = dec;
            }
            // Qbuf: contiguous buffer of size cur * BLOCK_Q * hc (layout depends on cp.async)
            for (int idx = tid; idx < cur * BLOCK_Q * hc; idx += stride) {
                float x = Qbuf[idx];
                uint8_t code = f32_to_fp8e4m3(x);
                float dec = fp8e4m3_to_f32(code);
                Qbuf[idx] = dec;
            }
            __syncthreads();

            // Compute partial dot across cur for this (kv_idx, token), accumulating per head
            float * Kcomp = Kbuf;
            for (int di = 0; di < cur; ++di) {
                float kval = Kcomp[di * BLOCK_N + k_local];
                for (int hi = 0; hi < hc; ++hi) {
#if __CUDA_ARCH__ >= 800 && defined(LLAMA_ENABLE_CP_ASYNC)
                    // When cp.async-enabled, Qbuf is transposed [BLOCK_Q*hc, D_TILE]
                    float qval = Qbuf[(hi * BLOCK_Q + t_local) * D_TILE + di];
#else
                    // Cooperative layout: Qbuf [cur, BLOCK_Q*hc]
                    float qval = Qbuf[di * (BLOCK_Q * hc) + hi * BLOCK_Q + t_local];
#endif
                    dot_vec[hi] += kval * qval;
                }
            }
            __syncthreads();
        }
        // Apply ReLU and weights, then sum into this tile accumulator
        for (int hi = 0; hi < hc; ++hi) {
            float tmp = dot_vec[hi];
            if (tmp < 0.0f) tmp = 0.0f;
            sum_hc += tmp * W_sh[hi * BLOCK_Q + t_local];
        }
        acc += sum_hc;
    }

    // Apply combined scale k_scale * K_sf (per-row amax scale) to match CPU FP8 path
    float sf_row = s_K_sf[k_local];
    acc *= k_scale[kv_idx] * sf_row;
    Out[kv_idx + (size_t)kv * token] = (starts && ends) ? ((kv_idx >= starts[token] && kv_idx < ends[token]) ? acc : 0.0f) : acc;
}

// mqa_attn_return_logits_kernel_port.cu
// Self-contained: cp.async double-buffer + WMMA (FP16) port of TileLang kernel.
// Grid: grid.x = ceil_div(seq_len, block_Q); block.x = threads (multiple of 32).
// Build: nvcc -std=c++17 -arch=sm_80 -lineinfo -Xptxas -v -c mqa_attn_return_logits_kernel_port.cu
// Note: Inputs (IndexQ/IndexK/Weights) are float32; converted to __half in shared.

static __device__ __forceinline__ size_t align16(size_t x){ return (x+15u)&~size_t(15u); }

#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__
void cp_async_16B_all(void* __restrict__ dst, const void* __restrict__ src, size_t bytes){
  const size_t n16 = bytes & ~size_t(15);
  const char *s = (const char*)src; char *d = (char*)dst;
  for(size_t i=(size_t)threadIdx.x*16;i<n16;i+= (size_t)blockDim.x*16){
    __pipeline_memcpy_async(d+i, s+i, 16);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
}

#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__
void cp_async_16B_issue_all(void* __restrict__ dst, const void* __restrict__ src, size_t bytes){
  const size_t n16 = bytes & ~size_t(15);
  const char *s = (const char*)src; char *d = (char*)dst;
  for(size_t i=(size_t)threadIdx.x*16;i<n16;i+= (size_t)blockDim.x*16){
    __pipeline_memcpy_async(d+i, s+i, 16);
  }
  __pipeline_commit();
}
#endif

#endif


__global__ __launch_bounds__(640, 1) void k_tl_mqa_attn_return_logits_port(
  const float * __restrict__ IndexQ,     // [seq_len*heads, index_dim]
  const float * __restrict__ IndexK,     // [seq_len_kv,    index_dim]
  const float * __restrict__ IndexKScale,// [seq_len_kv]
  float * __restrict__ Logits,           // [seq_len, seq_len_kv]
  const float * __restrict__ Weights,    // [seq_len, heads]
  const int   * __restrict__ CuSeqLenKS, // [seq_len]
  const int   * __restrict__ CuSeqLenKE, // [seq_len]
  int seq_len, int seq_len_kv, int heads, int index_dim,
  int block_N, int /*num_stages*/, int /*threads*/, int block_Q)
{
  const int bx = blockIdx.x;
  const int seq_len_i = bx * block_Q;

  // ---- shared memory layout (byte addressed, 16B aligned) ----
  extern __shared__ unsigned char smem[];
  size_t off = 0;

  const int WM=16, WN=16, WK=16;                  // WMMA tile
  const int warps = blockDim.x >> 5;

  const size_t Q_rows = (size_t)block_Q * (size_t)heads;
  const size_t Q_cols = (size_t)index_dim;
  const size_t K_rows_max = (size_t)block_N;
  const size_t K_cols = (size_t)index_dim;

  // f32 staging
  off = align16(off); float* Qs_f32 = (float*)(smem + off); off += Q_rows*Q_cols*sizeof(float);
  // two ping-pong K slabs (f32)
  off = align16(off); float* K0_f32 = (float*)(smem + off); off += K_rows_max*K_cols*sizeof(float);
  off = align16(off); float* K1_f32 = (float*)(smem + off); off += K_rows_max*K_cols*sizeof(float);
  // two ping-pong k_scale vectors
  off = align16(off); float* ks0    = (float*)(smem + off); off += K_rows_max*sizeof(float);
  off = align16(off); float* ks1    = (float*)(smem + off); off += K_rows_max*sizeof(float);
  // logits scratch for current K tile
  off = align16(off); float* logits_blk = (float*)(smem + off); off += K_rows_max*(size_t)block_Q*sizeof(float);

  // f16 WMMA slabs
  off = align16(off); __half* Qs_f16 = (__half*)(smem + off); off += Q_rows*Q_cols*sizeof(__half);
  off = align16(off); __half* K0_f16 = (__half*)(smem + off); off += K_rows_max*K_cols*sizeof(__half);
  off = align16(off); __half* K1_f16 = (__half*)(smem + off); off += K_rows_max*K_cols*sizeof(__half);

  // per-warp C tile scratch (float, WM*WN each)
  off = align16(off); float* Csh = (float*)(smem + off); off += (size_t)warps*(WM*WN)*sizeof(float);

  // ---- compute cu_k_s_min / cu_k_e_max for this block ----
  __shared__ int cu_k_s_min_s, cu_k_e_max_s;
  if(threadIdx.x==0){
    int smin= 2147483647, emax= -2147483648;
    for(int bq=0;bq<block_Q;++bq){
      int t = seq_len_i + bq;
      int v = (t<seq_len)? CuSeqLenKS[t] : 0;
      if(v>seq_len_kv) v = seq_len_kv;
      if(v < smin) smin = v;
    }
    for(int bq=0;bq<block_Q;++bq){
      int t = seq_len_i + bq;
      int v = (t<seq_len)? CuSeqLenKE[t] : 0;
      if(v>seq_len_kv) v = seq_len_kv;
      if(v > emax) emax = v;
    }
    cu_k_s_min_s = smin; cu_k_e_max_s = emax;
  }
  __syncthreads();
  const int cu_k_s_min = cu_k_s_min_s;
  const int cu_k_e_max = cu_k_e_max_s;

  // ---- stage Q block (seq range [seq_len_i, seq_len_i+block_Q), all heads) ----
  {
    const size_t bytesQ = Q_rows*Q_cols*sizeof(float);
#if __CUDA_ARCH__ >= 800
    bool ok = ((((uintptr_t)Qs_f32)&0xF)==0) &&
              ((((uintptr_t)(IndexQ + (size_t)seq_len_i*(size_t)heads*(size_t)index_dim))&0xF)==0) &&
              (bytesQ>=16);
    if(ok){
      const float* srcQ = IndexQ + (size_t)seq_len_i*(size_t)heads*(size_t)index_dim;
      cp_async_16B_all(Qs_f32, srcQ, bytesQ);
    } else
#endif
    {
      for(size_t t=threadIdx.x;t<Q_rows*Q_cols;t+=blockDim.x){
        size_t r=t/Q_cols, c=t%Q_cols;
        int bq=(int)(r/(size_t)heads), h=(int)(r%(size_t)heads);
        int tok=seq_len_i+bq;
        float v=0.f; if(tok<seq_len) v = IndexQ[(size_t)(tok*heads+h)*(size_t)index_dim + c];
        Qs_f32[t]=v;
      }
      __syncthreads();
    }
    // convert Q to half (pad Nq to multiple of WN for WMMA B)
    const int Nq=(int)Q_rows, Nq_pad=((Nq+WN-1)/WN)*WN;
    for(size_t t=threadIdx.x;t<(size_t)Nq_pad*(size_t)index_dim;t+=blockDim.x) Qs_f16[t]=__float2half(0.f);
    __syncthreads();
    for(size_t t=threadIdx.x;t<Q_rows*Q_cols;t+=blockDim.x) Qs_f16[t]=__float2half_rn(Qs_f32[t]);
    __syncthreads();
  }

  // zero logits scratch
  for(size_t t=threadIdx.x;t<(size_t)block_N*(size_t)block_Q;t+=blockDim.x) logits_blk[t]=0.f;
  __syncthreads();

  // ---- loop over K tiles with cp.async ping-pong ----
  int total_k = cu_k_e_max - cu_k_s_min; if(total_k<0) total_k=0;
  const int iters = (total_k + block_N - 1)/block_N;

  // Preload first tile into buffer 0
  int curN0 = min(block_N, max(0, cu_k_e_max - cu_k_s_min));
  if(iters>0){
    const size_t bytesK = (size_t)curN0*(size_t)index_dim*sizeof(float);
    const size_t bytesKs= (size_t)curN0*sizeof(float);
#if __CUDA_ARCH__ >= 800
    bool okK  = ((((uintptr_t)K0_f32)&0xF)==0) && ((((uintptr_t)(IndexK + (size_t)cu_k_s_min*(size_t)index_dim))&0xF)==0) && (bytesK>=16);
    bool okKs = ((((uintptr_t)ks0   )&0xF)==0) && ((((uintptr_t)(IndexKScale + (size_t)cu_k_s_min))&0xF)==0) && (bytesKs>=16);
    if(okK)  cp_async_16B_all(K0_f32, IndexK     + (size_t)cu_k_s_min*(size_t)index_dim, bytesK);  else {
#endif
      for(size_t t=threadIdx.x;t<(size_t)curN0*(size_t)index_dim;t+=blockDim.x){
        size_t r=t/(size_t)index_dim, c=t%(size_t)index_dim;
        K0_f32[t]=IndexK[(size_t)(cu_k_s_min+(int)r)*(size_t)index_dim + c];
      }
      __syncthreads();
#if __CUDA_ARCH__ >= 800
    }
    if(okKs) cp_async_16B_all(ks0, IndexKScale + (size_t)cu_k_s_min, bytesKs); else {
#endif
      for(size_t t=threadIdx.x;t<(size_t)curN0;t+=blockDim.x) ks0[t]=IndexKScale[cu_k_s_min+(int)t];
      __syncthreads();
#if __CUDA_ARCH__ >= 800
    }
#endif
    // convert first K to half
    for(size_t t=threadIdx.x;t<(size_t)curN0*(size_t)index_dim;t+=blockDim.x) K0_f16[t]=__float2half_rn(K0_f32[t]);
    __syncthreads();
  }

  const int warp_id = threadIdx.x>>5;
  const int lane    = threadIdx.x&31;
  const int warps_pb= warps;
  const int Nq_all  = (int)Q_rows;
  const int tiles_n = (Nq_all + WN - 1)/WN;

  for(int it=0; it<iters; ++it){
    // Preload next tile into the other buffer
    int k_start_next = cu_k_s_min + (it+1)*block_N;
    int curN1 = 0;
    if(it+1 < iters){
      curN1 = min(block_N, cu_k_e_max - k_start_next);
      const size_t bytesK = (size_t)curN1*(size_t)index_dim*sizeof(float);
      const size_t bytesKs= (size_t)curN1*sizeof(float);
#if __CUDA_ARCH__ >= 800
      bool okK  = ((((uintptr_t)K1_f32)&0xF)==0) && ((((uintptr_t)(IndexK + (size_t)k_start_next*(size_t)index_dim))&0xF)==0) && (bytesK>=16);
      bool okKs = ((((uintptr_t)ks1   )&0xF)==0) && ((((uintptr_t)(IndexKScale + (size_t)k_start_next))&0xF)==0) && (bytesKs>=16);
      if(okK)  cp_async_16B_all(K1_f32, IndexK     + (size_t)k_start_next*(size_t)index_dim, bytesK); else {
#endif
        for(size_t t=threadIdx.x;t<(size_t)curN1*(size_t)index_dim;t+=blockDim.x){
          size_t r=t/(size_t)index_dim, c=t%(size_t)index_dim;
          K1_f32[t]=IndexK[(size_t)(k_start_next+(int)r)*(size_t)index_dim + c];
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
      }
      if(okKs) cp_async_16B_all(ks1, IndexKScale + (size_t)k_start_next, bytesKs); else {
#endif
        for(size_t t=threadIdx.x;t<(size_t)curN1;t+=blockDim.x) ks1[t]=IndexKScale[k_start_next+(int)t];
        __syncthreads();
#if __CUDA_ARCH__ >= 800
      }
#endif
    }

    // Compute on current tile buffer (0)
    const int k_start_cur = cu_k_s_min + it*block_N;
    const int curN0_it = min(block_N, cu_k_e_max - k_start_cur);

    // If we just prefetched next, convert it while we compute? (keep simple: convert after prefetch)
    if(it==0 || it>0){ /* K0_f16 already converted for first; for subsequent we swap below */ }

    // WMMA over current tile (K0_f16 vs Qs_f16)
    const int tiles_m = (curN0_it + WM - 1)/WM;

    for(int tile_lin = warp_id; tile_lin < tiles_m*tiles_n; tile_lin += warps_pb){
      const int tile_m = tile_lin / tiles_n;     // along bn
      const int tile_n = tile_lin % tiles_n;     // along (bq*heads)

      wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
      wmma::fill_fragment(c, 0.0f);

      for(int kk=0; kk<index_dim; kk+=WK){
        const __half* Ap = K0_f16 + (size_t)tile_m*WM*(size_t)index_dim + kk;
        const __half* Bp = Qs_f16 + (size_t)tile_n*WN*(size_t)index_dim + kk;
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, Ap, index_dim);
        wmma::load_matrix_sync(b, Bp, index_dim);
        wmma::mma_sync(c, a, b, c);
      }

      float* cptr = Csh + (size_t)warp_id*(WM*WN);
      wmma::store_matrix_sync(cptr, c, WN, wmma::mem_row_major);
      __syncwarp();

      // Post-process (ReLU * weight * k_scale), reduce heads within warp, no atomics
      const int base_bq = (tile_n*WN) / heads;
      const int max_cols = min(WN, (int)Q_rows - tile_n*WN);
      const int groups = max(0, (max_cols + heads - 1) / heads);
      for (int mi = lane; mi < WM; mi += 32) {
        int bn = tile_m*WM + mi;
        if (bn >= curN0_it) continue;
        float ks = ks0[bn];
        float acc_g[16];
        #pragma unroll
        for (int u = 0; u < 16; ++u) acc_g[u] = 0.0f;
        // accumulate contributions for all columns this tile covers
        #pragma unroll
        for (int cj = 0; cj < WN; ++cj) {
          int ncol = tile_n*WN + cj; // = bq*heads + h
          if (cj >= max_cols || ncol >= (int)Q_rows) break;
          float val = cptr[mi*WN + cj];
          if (val < 0.f) val = 0.f;
          int bq_abs = ncol / heads;
          int h = ncol % heads;
          int tok = seq_len_i + bq_abs;
          float w = 0.0f;
          if (tok < seq_len) w = Weights[(size_t)tok*(size_t)heads + h];
          int u = bq_abs - base_bq;
          if (u >= 0 && u < 16) acc_g[u] += val * w;
        }
        // write partial sums to logits scratch (atomic to avoid inter-warp races across tile_n)
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
          if (u >= groups) break;
          int bq_abs = base_bq + u;
          int tok = seq_len_i + bq_abs;
          if (bq_abs < block_Q && tok < seq_len) {
            atomicAdd(&logits_blk[(size_t)bn*(size_t)block_Q + (size_t)bq_abs], acc_g[u] * ks);
          }
        }
      }
      __syncwarp();
    }
    __syncthreads();

    // Write this tile’s logits to global
    for(size_t t=threadIdx.x;t<(size_t)curN0_it*(size_t)block_Q;t+=blockDim.x){
      int bn = (int)(t/(size_t)block_Q);
      int bq = (int)(t%(size_t)block_Q);
      int tok = seq_len_i + bq;
      if(tok < seq_len){
        int kv_col = k_start_cur + bn;
        if(kv_col < seq_len_kv)
          Logits[(size_t)tok*(size_t)seq_len_kv + (size_t)kv_col] =
            logits_blk[(size_t)bn*(size_t)block_Q + (size_t)bq];
      }
    }
    __syncthreads();

    // Clear scratch for next tile
    for(size_t t=threadIdx.x;t<(size_t)curN0_it*(size_t)block_Q;t+=blockDim.x)
      logits_blk[t]=0.f;
    __syncthreads();

    // Swap ping-pong: convert next tile, then swap buffers
    if(it+1 < iters){
      // convert next K (in K1_f32 -> K1_f16)
      for(size_t t=threadIdx.x;t<(size_t)curN1*(size_t)index_dim;t+=blockDim.x)
        K1_f16[t]=__float2half_rn(K1_f32[t]);
      __syncthreads();
      // swap pointers
      float* tmpf; __half* tmph; float* tmps;
      tmpf=K0_f32; K0_f32=K1_f32; K1_f32=tmpf;
      tmph=K0_f16; K0_f16=K1_f16; K1_f16=tmph;
      tmps=ks0;    ks0   =ks1;    ks1   =tmps;
    }
  }
}

// Device-side FP8 E4M3 helpers using native PTX conversions
static __device__ __forceinline__ uint8_t f32_to_fp8e4m3(float x) {
    uint16_t tmp;
    float zero = 0.0f;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(zero), "f"(x));
    return static_cast<uint8_t>(tmp & 0xFFu);
}

static __device__ __forceinline__ float fp8e4m3_to_f32(uint8_t code) {
    uint16_t bits = code;
    uint32_t packed;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    return __half2float(reinterpret_cast<half2 const &>(packed).x);
}

// WMMA 16x16x16 BF16 (float input cast to bf16), one warp per block
__global__ void k_indexer_logits_wmma16_bf16(
    const float * __restrict__ Q, // [D, Tc*H]
    const float * __restrict__ K, // [D, kv]
    const float * __restrict__ W, // [H, Tc]
    const float * __restrict__ k_scale, // [kv]
    int D, int H, int Tc, int kv,
    const int * __restrict__ starts,
    const int * __restrict__ ends,
    float * __restrict__ Out) {
#if __CUDA_ARCH__ >= 800
    const int tokens_per_tile = max(1, 16 / H);
    const int t0 = blockIdx.x * tokens_per_tile;
    const int k0 = blockIdx.y * 16;
    if (t0 >= Tc || k0 >= kv) return;

    int smin_blk = 0, smax_blk = kv;
    if (starts != nullptr && ends != nullptr) {
        smin_blk = kv; smax_blk = 0;
        for (int tl = 0; tl < tokens_per_tile; ++tl) {
            int tok = t0 + tl;
            if (tok < Tc) {
                int s0 = starts[tok]; int e0 = ends[tok];
                if (s0 < 0) s0 = 0; if (s0 > kv) s0 = kv;
                if (e0 < 0) e0 = 0; if (e0 > kv) e0 = kv;
                if (s0 < smin_blk) smin_blk = s0;
                if (e0 > smax_blk) smax_blk = e0;
            }
        }
        if (smin_blk > smax_blk) smin_blk = smax_blk;
    }
    int curN_all = kv - k0; if (curN_all < 0) curN_all = 0; int curN = curN_all < 16 ? curN_all : 16;
    if (starts != nullptr && ends != nullptr) {
        if (k0 >= smax_blk || (k0 + 16) <= smin_blk) {
            int lane = threadIdx.x & 31;
            for (int idx = lane; idx < curN * tokens_per_tile; idx += 32) {
                int mi = idx / tokens_per_tile; int tl = idx % tokens_per_tile; int kv_idx = k0 + mi; int tok = t0 + tl;
                if (tok < Tc && kv_idx < kv) Out[(size_t)kv_idx + (size_t)kv * tok] = 0.0f;
            }
            return;
        }
    }

    // Per-row K scale (amax/448) for 16-row tile
    __shared__ float K_sf[16];
    int lane = threadIdx.x & 31;
    if (lane < 16) K_sf[lane] = 1.0f;
    __syncthreads();
    for (int mi = 0; mi < 16; ++mi) {
        int kv_idx = k0 + mi;
        float local_max = 0.0f;
        if (kv_idx < kv) {
            for (int d0 = lane; d0 < D; d0 += 32) {
                float v = K[(size_t)d0 + (size_t)D * (size_t)kv_idx];
                float av = fabsf(v);
                if (av > local_max) local_max = av;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, local_max, off);
            if (other > local_max) local_max = other;
        }
        if (lane == 0 && kv_idx < kv) {
            float maxv = local_max;
            if (maxv < 1e-4f) maxv = 1e-4f;
            K_sf[mi] = maxv / 448.0f;
        }
        __syncwarp();
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ __nv_bfloat16 A_sh[16*16]; // row-major
    __shared__ __nv_bfloat16 B_sh[16*16]; // col-major
    __shared__ float         C_sh[16*16];

    // Iterate K dimension in 16-slices
    for (int d0 = 0; d0 < D; d0 += 16) {
        int lane2 = threadIdx.x & 31;
        // Load A_sh with FP8 quant/dequant and per-row scaling
        for (int idx = lane2; idx < 16*16; idx += 32) {
            int mi = idx / 16;
            int di = idx % 16;
            int kv_idx = k0 + mi;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (kv_idx < kv && d0 + di < D) {
                float f = K[(size_t)(d0 + di) + (size_t)D * (size_t)kv_idx];
                float sf = K_sf[mi];
                float scaled = f / sf;
                uint8_t code = f32_to_fp8e4m3(scaled);
                float dec = fp8e4m3_to_f32(code);
                v = __float2bfloat16(dec);
            }
            A_sh[mi * 16 + di] = v;
        }
        // Load B_sh (col-major) with FP8 quant/dequant
        for (int idx = lane2; idx < 16*16; idx += 32) {
            int di = idx / 16; // k index
            int cj = idx % 16; // column index 0..15 => (tok_local,h)
            int tok_local = cj / H;
            int h = cj % H;
            int tok = t0 + tok_local;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (tok < Tc && h < H && d0 + di < D) {
                float f = Q[(size_t)(d0 + di) + (size_t)D * (size_t)(tok*H + h)];
                uint8_t code = f32_to_fp8e4m3(f);
                float dec = fp8e4m3_to_f32(code);
                v = __float2bfloat16(dec);
            }
            B_sh[cj * 16 + di] = v;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_sh, 16);
        wmma::load_matrix_sync(b_frag, B_sh, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(C_sh, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    int lane3 = threadIdx.x & 31;
    for (int idx = lane3; idx < 16 * tokens_per_tile; idx += 32) {
        int mi = idx / tokens_per_tile;
        int tl = idx % tokens_per_tile;
        int kv_idx = k0 + mi;
        int tok = t0 + tl;
        if (kv_idx < kv && tok < Tc) {
            float s = 0.0f;
            int col_base = tl * H;
            for (int h = 0; h < H; ++h) {
                float v = C_sh[mi * 16 + (col_base + h)];
                if (v < 0.0f) v = 0.0f;
                float w = W[h + (size_t)H * tok];
                s += v * w;
            }
            // Apply combined scale k_scale * K_sf, matching CPU FP8 path
            s *= k_scale[kv_idx] * K_sf[mi];
            if (starts != nullptr && ends != nullptr) {
                int s0 = starts[tok];
                int e0 = ends[tok];
                if (s0 < 0) s0 = 0;
                if (s0 > kv) s0 = kv;
                if (e0 < 0) e0 = 0;
                if (e0 > kv) e0 = kv;
                Out[kv_idx + (size_t)kv * tok] = (kv_idx >= s0 && kv_idx < e0) ? s : 0.0f;
            } else {
                Out[kv_idx + (size_t)kv * tok] = s;
            }
        }
    }
#endif
}

// WMMA 16x16 with head grouping: supports H multiple of 16
// Updated to consume FP8-encoded K with per-row scales provided by the host.
__global__ void k_indexer_logits_wmma16_f32_hgrp(
    const float  * __restrict__ Q,      // [D, Tc*H] (F32)
    const uint8_t* __restrict__ K_fp8,  // [kv, D] row-major FP8 codes
    const float  * __restrict__ K_sf_g, // [kv] per-row scale (amax/448)
    const float  * __restrict__ W,      // [H, Tc]
    const float  * __restrict__ k_scale,// [kv]
    int D, int H, int Tc, int kv,
    const int * __restrict__ starts,
    const int * __restrict__ ends,
    float * __restrict__ Out) {
#if __CUDA_ARCH__ >= 800
    const int tokens_per_tile = 1;
    const int t0 = blockIdx.x * tokens_per_tile;
    const int k0 = blockIdx.y * 16;
    if (t0 >= Tc || k0 >= kv) return;

    // Per-row K scale (amax/448) for 16-row tile, loaded from global K_sf_g
    __shared__ float K_sf[16];
    int lane = threadIdx.x & 31;
    if (lane < 16) {
        int kv_idx = k0 + lane;
        float sf = 1.0f;
        if (kv_idx < kv && K_sf_g) {
            sf = K_sf_g[kv_idx];
        }
        if (sf < 1e-4f) sf = 1e-4f;
        K_sf[lane] = sf;
    }
    __syncthreads();

    __shared__ __half A_sh[16*16]; // row-major K tile (FP8-decoded, scaled later)
    __shared__ __half B_sh[16*16]; // col-major Q tile (FP8-quantized then decoded)
    __shared__ float  C_sh[16*16]; // accumulator dump
    __shared__ float  S_acc[16];   // accumulate per kv row

    if (threadIdx.x < 16) S_acc[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int h0 = 0; h0 < H; h0 += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Iterate K dimension in 16-slices
        for (int d0 = 0; d0 < D; d0 += 16) {
            int lane2 = threadIdx.x & 31;
            // Load A_sh from FP8 K codes: rows are kv rows, cols are k-slice
            for (int idx = lane2; idx < 16*16; idx += 32) {
                int mi = idx / 16; // row within tile
                int di = idx % 16; // col within tile
                int kv_idx = k0 + mi;
                __half v = __float2half_rn(0.0f);
                if (kv_idx < kv && d0 + di < D && K_fp8) {
                    size_t off = (size_t)kv_idx * (size_t)D + (size_t)(d0 + di);
                    uint8_t code = K_fp8[off];
                    float dec = fp8e4m3_to_f32(code);
                    v = __float2half_rn(dec);
                }
                A_sh[mi * 16 + di] = v;
            }
            // Load B_sh: columns=16 heads in group, rows=16 k-slice; col-major, FP8 quant/dequant
            for (int idx = lane2; idx < 16*16; idx += 32) {
                int di = idx / 16; // k index
                int cj = idx % 16; // head col 0..15
                int h = h0 + cj;
                int tok = t0; // one token per tile
                __half v = __float2half_rn(0.0f);
                if (tok < Tc && h < H && d0 + di < D) {
                    float f = Q[(size_t)(d0 + di) + (size_t)D * (size_t)(tok*H + h)];
                    uint8_t code = f32_to_fp8e4m3(f);
                    float dec = fp8e4m3_to_f32(code);
                    v = __float2half_rn(dec);
                }
                B_sh[cj * 16 + di] = v;
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, A_sh, 16);
            wmma::load_matrix_sync(b_frag, B_sh, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }

        // Dump accumulators to shared
        wmma::store_matrix_sync(C_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        // Accumulate this head-group contribution into S_acc per row
        int lane3 = threadIdx.x & 31;
        for (int mi = lane3; mi < 16; mi += 32) {
            float srow = 0.0f;
            for (int cj = 0; cj < 16; ++cj) {
                float v = C_sh[mi * 16 + cj];
                if (v < 0.0f) v = 0.0f;
                int h = h0 + cj;
                if (h < H) {
                    float w = W[h + (size_t)H * (size_t)t0];
                    srow += v * w;
                }
            }
            atomicAdd(&S_acc[mi], srow);
        }
        __syncthreads();
    }

    // Write out with combined scale KS * K_sf and windowing
    for (int mi = lane; mi < 16; mi += 32) {
        int kv_idx = k0 + mi;
        if (kv_idx < kv && t0 < Tc) {
            float srow = S_acc[mi] * k_scale[kv_idx] * K_sf[mi];
            if (starts != nullptr && ends != nullptr) {
                int s0 = starts[t0];
                int e0 = ends[t0];
                if (s0 < 0) s0 = 0;
                if (s0 > kv) s0 = kv;
                if (e0 < 0) e0 = 0;
                if (e0 > kv) e0 = kv;
                Out[kv_idx + (size_t)kv * (size_t)t0] = (kv_idx >= s0 && kv_idx < e0) ? srow : 0.0f;
            } else {
                Out[kv_idx + (size_t)kv * (size_t)t0] = srow;
            }
        }
    }
#endif
}

extern "C" void ggml_cuda_indexer_logits_fused_device(ggml_backend_cuda_context & ctx,
                                                       const float * dQ,
                                                       const float * dK,
                                                       const float * dW,
                                                       const float * dKS,
                                                       const int * dStarts, const int * dEnds,
                                                       int D, int H, int Tc, int kv_end,
                                                       float * dOut) {
    cudaStream_t stream = ctx.stream();
    // Ensure starts/ends are device-resident copies (handles host or device sources)
    const int * dStarts_dev = dStarts;
    const int * dEnds_dev   = dEnds;
    int * dStarts_tmp = nullptr;
    int * dEnds_tmp   = nullptr;
    if (dStarts) {
        dStarts_dev = dStarts;
    }
    if (dEnds)   {
        dEnds_dev   = dEnds;
    }

    // env knobs for tile sizes with heuristics when unset
    const char *env_bq = getenv("LLAMA_INDEXER_BLOCK_Q");
    int BLOCK_Q = env_bq ? max(1, atoi(env_bq)) : 2; // safe default; larger can explode memory
    const char *env_bn = getenv("LLAMA_INDEXER_BLOCK_N");
    int BLOCK_N = env_bn ? max(1, atoi(env_bn)) : (kv_end >= 512 ? 256 : 128);
    const char *env_dt = getenv("LLAMA_INDEXER_D_TILE");
    int D_TILE = env_dt ? max(16, atoi(env_dt)) : 32;
    size_t work_elems = (size_t)Tc * (size_t)kv_end;
    int exact_flag = (work_elems <= 4096) ? 1 : 0;
    {
        const char *e = getenv("LLAMA_INDEXER_EXACT");
        if (e && *e && atoi(e)!=0) exact_flag = 1;
    }
    // Select kernel based on env; default to tiled
    bool use_wmma = false;
    bool do_not_use_wmma = false;
    {
        const char *s = getenv("LLAMA_INDEXER_USE_WMMA");
        if (s && atoi(s) != 0) use_wmma = true;
        if (s && atoi(s) == 0) do_not_use_wmma = true;
    }
        // Heuristics:
    if (!use_wmma) {
        size_t work = (size_t)Tc * (size_t)kv_end;
        // prefer WMMA when legal: standard (H<=16) or head-grouped (H%16==0)
        if (D % 16 == 0 && ((((H <= 16) && ((16 % H) == 0)) || ((H % 16) == 0))) && work >= 16384 && !do_not_use_wmma) {
            use_wmma = 1;
        }
    }

    const char *dg_env = getenv("LLAMA_DG_FP8");
    const bool use_dg_fp8 = (dg_env && *dg_env && atoi(dg_env) != 0);
    if (sparse_debug_on()) printf("[INDEXER_DISPATCH] use_wmma=%d D=%d H=%d Tc=%d kv=%d BLOCK_Q=%d BLOCK_N=%d D_TILE=%d\n", (int)use_wmma, D, H, Tc, kv_end, BLOCK_Q, BLOCK_N, D_TILE);
    // Optional: TL port path in device wrapper
    const char * __prof_env = getenv("LLAMA_SPARSE_PROF");
    auto * __prof_each_env = getenv("LLAMA_SPARSE_PROF_EACH");
    if (use_dg_fp8) {
        if (sparse_debug_on()) fprintf(stderr, "[INDEXER_DISPATCH] using DeepGEMM FP8 paged MQA logits path\n");

#if CUDART_VERSION >= 12000
        // DeepSeek V3.2-Exp: use DeepGEMM FP8 paged MQA logits kernel when shapes match
        const int block_kv = 64;
        const int num_math_warp_groups = 1;
        const int num_specialized_threads = 128;
        const int num_math_threads = num_math_warp_groups * 128;
        if (H == 64 && (D == 64 || D == 128) && (Tc == 1 || Tc == 2) && kv_end % block_kv == 0) {
            int batch_size = 1;
            int next_n = Tc;
            int num_heads = H;
            int head_dim = D;
            int num_kv_blocks = kv_end / block_kv;
            int max_context_len = kv_end;

            ggml_cuda_pool & pool = ctx.pool(ggml_cuda_get_device());

            // Build DeepGEMM-style context_lens (1D) and block_table
            ggml_cuda_pool_alloc<unsigned int> __ctx_lens(pool, batch_size);
            ggml_cuda_pool_alloc<unsigned int> __block_tbl(pool, (size_t)batch_size * (size_t)num_kv_blocks);
            unsigned int *d_ctx_lens = __ctx_lens.get();
            unsigned int *d_block_tbl = __block_tbl.get();
            CUDA_CHECK(cudaMemsetAsync(d_block_tbl, 0, sizeof(unsigned int)*(size_t)batch_size*(size_t)num_kv_blocks, stream));
            unsigned int h_ctx[1] = { (unsigned int) kv_end };
            CUDA_CHECK(cudaMemcpyAsync(d_ctx_lens, h_ctx, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
            // block_table[0][i] = i
            CUDA_CHECK(cudaMemsetAsync(d_block_tbl, 0, sizeof(unsigned int)*(size_t)num_kv_blocks, stream));
            {
                // simple host init for block_table
                std::vector<unsigned int> h_bt(num_kv_blocks);
                for (int i = 0; i < num_kv_blocks; ++i) h_bt[i] = (unsigned int) i;
                CUDA_CHECK(cudaMemcpyAsync(d_block_tbl, h_bt.data(), sizeof(unsigned int)*h_bt.size(), cudaMemcpyHostToDevice, stream));
            }

            // Build schedule_meta for single-CTA execution (grid.x = 1)
            // Equivalent to DeepGEMM's smxx_paged_mqa_logits_metadata for batch_size=1
            const int split_kv = block_kv * num_math_warp_groups;
            (void) split_kv; // silence unused warning for now
            const int num_sms = 1;
            ggml_cuda_pool_alloc<unsigned int> __sched(pool, (size_t)(num_sms + 1) * 2);
            unsigned int *d_sched = __sched.get();
            unsigned int h_sched[4] = {0u, 0u, (unsigned int) batch_size, 0u};
            CUDA_CHECK(cudaMemcpyAsync(d_sched, h_sched, sizeof(h_sched), cudaMemcpyHostToDevice, stream));

            // Allocate logits [batch*next_n, aligned_max_context_len]
            const int aligned_max_context_len = ((max_context_len + num_math_warp_groups*block_kv - 1) / (num_math_warp_groups*block_kv))*(num_math_warp_groups*block_kv);
            ggml_cuda_pool_alloc<float> __dg_logits(pool, (size_t)batch_size * (size_t)next_n * (size_t)aligned_max_context_len);
            float *d_dg_logits = __dg_logits.get();
            CUDA_CHECK(cudaMemsetAsync(d_dg_logits, 0, sizeof(float)*(size_t)batch_size*(size_t)next_n*(size_t)aligned_max_context_len, stream));

            // Build FP8 Q, K and scales in row-major and flatten
            ggml_cuda_pool_alloc<float> __Qrm(pool, (size_t)(Tc*H) * (size_t)D);
            ggml_cuda_pool_alloc<float> __Krm(pool, (size_t)kv_end * (size_t)D);
            ggml_cuda_pool_alloc<float> __Wrm(pool, (size_t)Tc * (size_t)H);
            float *dQrm = __Qrm.get();
            float *dKrm = __Krm.get();
            float *dWrm = __Wrm.get();
            dim3 tbT(32, 8);
            dim3 gdQ((Tc*H + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
            dim3 gdK((kv_end + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
            dim3 gdW((Tc + tbT.x - 1)/tbT.x, (H + tbT.y - 1)/tbT.y);
            k_colmajor_DN_to_rowmajor_ND<<<gdQ, tbT, 0, stream>>>(dQ, D, Tc*H, dQrm);
            k_colmajor_DN_to_rowmajor_ND<<<gdK, tbT, 0, stream>>>(dK, D, kv_end, dKrm);
            k_colmajor_DN_to_rowmajor_ND<<<gdW, tbT, 0, stream>>>(dW, H, Tc, dWrm);

            // FP8 Q (no per-row scaling)
            ggml_cuda_pool_alloc<unsigned char> __Qfp8(pool, (size_t)(Tc*H) * (size_t)D);
            unsigned char *dQfp8 = __Qfp8.get();
            {
                size_t total = (size_t)(Tc*H) * (size_t)D;
                dim3 tb(256);
                dim3 gd((unsigned)((total + tb.x - 1)/tb.x));
                k_rowmajor_f32_to_fp8_e4m3<<<gd, tb, 0, stream>>>(dQrm, (int)(Tc*H), D, dQfp8);
            }

            // FP8 K with per-row scaling and combined k_scale
            ggml_cuda_pool_alloc<unsigned char> __Kfp8(pool, (size_t)kv_end * (size_t)D);
            ggml_cuda_pool_alloc<float> __Kamax(pool, (size_t)kv_end);
            ggml_cuda_pool_alloc<float> __Ksf(pool, (size_t)kv_end);
            ggml_cuda_pool_alloc<float> __KsfInv(pool, (size_t)kv_end);
            ggml_cuda_pool_alloc<float> __IdxKScale(pool, (size_t)kv_end);
            unsigned char *dKfp8 = __Kfp8.get();
            float *dKamax = __Kamax.get();
            float *dKsf   = __Ksf.get();
            float *dKsfInv= __KsfInv.get();
            float *dIdxKScale = __IdxKScale.get();
            {
                int rowsK = kv_end;
                int colsK = D;
                int threadsA = 256;
                int blocksA = (rowsK + threadsA - 1) / threadsA;
                k_rowmajor_f32_rowwise_absmax<<<blocksA, threadsA, 0, stream>>>(dKrm, rowsK, colsK, dKamax);
                k_fp8_compute_row_scales<<<blocksA, threadsA, 0, stream>>>(dKamax, rowsK, dKsf, dKsfInv);
                size_t total = (size_t)rowsK * (size_t)colsK;
                dim3 tb(256);
                dim3 gd((unsigned)((total + tb.x - 1)/tb.x));
                k_rowmajor_f32_to_fp8_e4m3_rowwise_scaled<<<gd, tb, 0, stream>>>(dKrm, rowsK, colsK, dKsfInv, dKfp8);
                k_elemwise_mul<<<blocksA, threadsA, 0, stream>>>(dKS, dKsf, dIdxKScale, rowsK);
            }

            // Build fused KV cache buffer [num_kv_blocks, block_kv, 1, head_dim+4]
            const int head_dim_with_sf = head_dim + 4;
            ggml_cuda_pool_alloc<unsigned char> __FusedKV(pool, (size_t)num_kv_blocks * (size_t)block_kv * (size_t)head_dim_with_sf);
            unsigned char *dFusedKV = __FusedKV.get();
            CUDA_CHECK(cudaMemsetAsync(dFusedKV, 0, (size_t)num_kv_blocks * (size_t)block_kv * (size_t)head_dim_with_sf, stream));
            {
                // Layout: [num_blocks, block_kv, 1, head_dim+4]
                dim3 tb(256);
                dim3 gd((unsigned)((kv_end * head_dim + tb.x - 1)/tb.x));
                // pack Kfp8 into leading head_dim slots; then we will separately feed scales via TMA
                // here we only lay out FP8 values row-major
                // index: row r in [0,kv_end), col c in [0,head_dim)
                auto pack = [=] __device__ (int idx) {};
            }
            // For DeepGEMM kernel we actually pass Kfp8 and scales separately via tensor_map_kv and tensor_map_kv_scales,
            // so we do not need a true fused buffer here; instead we treat dKfp8 as [num_kv_blocks, block_kv, head_dim]

            // Create TMA descriptors using driver API wrappers
            cute::TmaDescriptor tma_q{}, tma_kv{}, tma_kv_scales{}, tma_w{};
            // Q: [head_dim, batch*next_n*num_heads]
            dg_fp8_encode_tma_2d(tma_q, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                 dQfp8,
                                 (uint32_t) head_dim,
                                 (uint32_t)(batch_size * next_n * num_heads),
                                 (uint32_t) head_dim,
                                 (uint32_t)(next_n * num_heads),
                                 (uint32_t) head_dim,
                                 sizeof(unsigned char));
            // KV: [head_dim, block_kv, num_kv_blocks]
            dg_fp8_encode_tma_3d(tma_kv, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                 dKfp8,
                                 (uint32_t) head_dim,
                                 (uint32_t) block_kv,
                                 (uint32_t) num_kv_blocks,
                                 (uint32_t) head_dim,
                                 (uint32_t) block_kv,
                                 1u,
                                 (uint32_t) head_dim,
                                 (uint32_t)(head_dim * block_kv),
                                 sizeof(unsigned char));
            // KV scales: [block_kv, num_kv_blocks]
            dg_fp8_encode_tma_2d(tma_kv_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 dIdxKScale,
                                 (uint32_t) block_kv,
                                 (uint32_t) num_kv_blocks,
                                 (uint32_t) block_kv,
                                 1u,
                                 (uint32_t) block_kv,
                                 sizeof(float));
            // Weights: [next_n*num_heads, batch_size]
            dg_fp8_encode_tma_2d(tma_w, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 dWrm,
                                 (uint32_t)(next_n * num_heads),
                                 (uint32_t) batch_size,
                                 (uint32_t)(next_n * num_heads),
                                 1u,
                                 (uint32_t)(next_n * num_heads),
                                 sizeof(float));

            // Launch DeepGEMM kernel via external wrapper TU
            const uint32_t kNextN = 2;
            const uint32_t kNumHeads = 64;
            const uint32_t kNumQStages = 3;
            const uint32_t kNumKVStages = 3;
            const uint32_t SPLIT_KV = block_kv * num_math_warp_groups;
            const uint32_t kNumSpecializedThreads = 128;
            const uint32_t kNumMathThreads = num_math_warp_groups * 128;
            // Dynamic shared memory size, mirroring DeepGEMM host runtime
            const int swizzle_alignment = head_dim * 8;
            auto align_int = [](int x, int a) { return (x + a - 1) / a * a; };
            const int smem_q_size_per_stage = next_n * num_heads * head_dim * 1; // __nv_fp8_e4m3
            const int smem_weight_size_per_stage = next_n * num_heads * (int)sizeof(float);
            const int aligned_smem_weight_size_per_stage = align_int(smem_weight_size_per_stage, swizzle_alignment);
            const int smem_q_pipe_size = kNumQStages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) +
                                         align_int(kNumQStages * 8 * 2, swizzle_alignment);
            const int smem_kv_size_per_stage = block_kv * head_dim * 1; // __nv_fp8_e4m3
            const int smem_kv_scale_size_per_stage = block_kv * (int)sizeof(float);
            const int aligned_smem_kv_scale_size_per_stage = align_int(smem_kv_scale_size_per_stage, swizzle_alignment);
            const int smem_kv_pipe_size = kNumKVStages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) +
                                          align_int(kNumKVStages * 8 * 2, swizzle_alignment);
            const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
            const int smem_tmem_ptr = 4;
            const int smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
            size_t shmem_bytes = (size_t) smem_size;

            LAUNCH_PROFILE_KERNEL("PROFILE_DG_FP8", DG_FP8, stream, ([&](){
                ggml_deepgemm_paged_mqa_logits_sm100(
                    (uint32_t) batch_size,
                    (uint64_t) aligned_max_context_len,
                    (uint64_t) num_kv_blocks,
                    d_ctx_lens,
                    d_dg_logits,
                    d_block_tbl,
                    d_sched,
                    tma_q,
                    tma_kv,
                    tma_kv_scales,
                    tma_w,
                    kNextN,
                    kNumHeads,
                    (uint32_t) head_dim,
                    (uint32_t) block_kv,
                    false,
                    kNumQStages,
                    kNumKVStages,
                    SPLIT_KV,
                    kNumSpecializedThreads,
                    kNumMathThreads,
                    stream,
                    shmem_bytes);
            })(), D, H, Tc, kv_end);

            // Map logits [batch*next_n, max_context_len] back to Out [kv, Tc]
            // For our simple case (batch=1), token t in [0,Tc), kv index k in [0,kv_end)
            dim3 tb(32, 4);
            dim3 gd((kv_end + tb.x - 1)/tb.x, (Tc + tb.y - 1)/tb.y);
            k_transpose_TcKv_to_KvTc<<<gd, tb, 0, stream>>>(d_dg_logits, Tc, kv_end, dOut);
            CUDA_CHECK(cudaGetLastError());
            cudaStreamSynchronize(stream);
            if (dStarts_tmp) cudaFree(dStarts_tmp);
            if (dEnds_tmp) cudaFree(dEnds_tmp);
            return;
        }
#endif // CUDART_VERSION >= 12000
    } else if (const char *s = getenv("LLAMA_INDEXER_TL_PORT"); s && atoi(s) != 0) {

        ggml_cuda_pool & __pool = ctx.pool(ggml_cuda_get_device());
          bool use_tma_fp8 = false;
          if (const char *e = getenv("LLAMA_TL_FP8"); e && atoi(e) != 0) use_tma_fp8 = true;
          // Prepare starts/ends (CuSeqLenKS/KE). If provided by caller via GGML op src[4]/src[5],
          // use them; otherwise synthesize [0, kv_end) per token.
          ggml_cuda_pool_alloc<int> __KS_i(__pool, (size_t)Tc);
          ggml_cuda_pool_alloc<int> __KE_i(__pool, (size_t)Tc);
          int *dKS_i = __KS_i.get(), *dKE_i = __KE_i.get();
          // Default: fill 0..kv_end
          int tblocks = (Tc + 255) / 256;
          k_fill_int<<<tblocks, 256, 0, stream>>>(dKS_i, Tc, 0);
          k_fill_int<<<tblocks, 256, 0, stream>>>(dKE_i, Tc, kv_end);
          ggml_cuda_pool_alloc<float> __Logits(__pool, (size_t)Tc * (size_t)kv_end);
          float *dLogits = __Logits.get();
          cudaMemsetAsync(dLogits, 0, sizeof(float) * (size_t)Tc * (size_t)kv_end, stream);
          int block_N = getenv_int_("LLAMA_TL_BLOCK_N", 256);
          int threads = getenv_int_("LLAMA_TL_THREADS", 640);
          int block_Q = getenv_int_("LLAMA_TL_BLOCK_Q", max(1, 128 / max(1, H)));
          int num_stages = getenv_int_("LLAMA_TL_NUM_STAGES", 3);
          auto align16 = [](size_t x) { return (x + 15u) & ~size_t(15u); };
          if (sparse_debug_on()) {
              int WM = 16, WN = 16;
              int Q_rows = block_Q * H;
              int Nq_pad = ((Q_rows + WN - 1) / WN) * WN;
              size_t Qs_f16_alloc_bytes = (size_t)block_Q * (size_t)H * (size_t)D * sizeof(__half);
              size_t Qs_f16_needed_bytes = (size_t)Nq_pad * (size_t)D * sizeof(__half);
              int K_rows_max = block_N;
              int K_rows_pad = ((K_rows_max + WM - 1) / WM) * WM;
              size_t K_f16_alloc_bytes = (size_t)K_rows_max * (size_t)D * sizeof(__half);
              size_t K_f16_needed_bytes = (size_t)K_rows_pad * (size_t)D * sizeof(__half);
              fprintf(stderr,
                      "[TL_PORT_DEBUG] Q_rows=%d Nq_pad=%d Qs_f16_alloc=%zu Qs_f16_needed=%zu (diff=%zd) | "
                      "K_rows_max=%d K_rows_pad=%d K_f16_alloc=%zu K_f16_needed=%zu (diff=%zd)\n",
                      Q_rows, Nq_pad, (size_t)Qs_f16_alloc_bytes, (size_t)Qs_f16_needed_bytes, (ssize_t)Qs_f16_needed_bytes - (ssize_t)Qs_f16_alloc_bytes,
                      K_rows_max, K_rows_pad, (size_t)K_f16_alloc_bytes, (size_t)K_f16_needed_bytes, (ssize_t)K_f16_needed_bytes - (ssize_t)K_f16_alloc_bytes);
          }

          int maxOpt = 0;
          cudaDeviceGetAttribute(&maxOpt, cudaDevAttrMaxSharedMemoryPerBlockOptin, ggml_cuda_get_device());
          size_t max_shmem = (size_t)(maxOpt > 0 ? maxOpt : 98304);
          dim3 gridTL((Tc + block_Q - 1) / block_Q);
          // Convert Q [D, Tc*H] to row-major [Tc*H, D]; K [D, kv] to [kv, D]; W [H, Tc] to [Tc, H]
          ggml_cuda_pool_alloc<float> __Qrm(__pool, (size_t)(Tc*H) * (size_t)D);
          ggml_cuda_pool_alloc<float> __Krm(__pool, (size_t)kv_end * (size_t)D);
          ggml_cuda_pool_alloc<float> __Wrm(__pool, (size_t)Tc * (size_t)H);
          float *dQrm = __Qrm.get();
          float *dKrm = __Krm.get();
          float *dWrm = __Wrm.get();
          if (dStarts_dev != nullptr && dEnds_dev != nullptr) {
              CUDA_CHECK(cudaMemcpyAsync(dKS_i, dStarts, sizeof(int)*(size_t)Tc, cudaMemcpyDeviceToDevice, stream));
              CUDA_CHECK(cudaMemcpyAsync(dKE_i, dEnds,   sizeof(int)*(size_t)Tc, cudaMemcpyDeviceToDevice, stream));
          }
          if (use_tma_fp8) {
              if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                  fprintf(stderr, "[TL_FP8_DBG] entering TL FP8 path: D=%d H=%d Tc=%d kv=%d\n", D, H, Tc, kv_end);
                  fflush(stderr);
              }
              static bool __tl_fp8_init_done = false;
              static int  __tl_fp8_init_status = 0;
              if (!__tl_fp8_init_done) {
                  __tl_fp8_init_status = init();
                  __tl_fp8_init_done = true;
                  if (__tl_fp8_init_status != 0) {
                      const char * __err = get_last_error();
                      fprintf(stderr, "[TL_FP8] init() failed status=%d err=%s\n", __tl_fp8_init_status, __err ? __err : "<none>");
                  }
              }
              ggml_cuda_pool_alloc<__half> __Qh(__pool, (size_t)(Tc*H) * (size_t)D);
              ggml_cuda_pool_alloc<__half> __Kh(__pool, (size_t)kv_end * (size_t)D);
              ggml_cuda_pool_alloc<unsigned char> __Qfp8(__pool, (size_t)(Tc*H) * (size_t)D);
              ggml_cuda_pool_alloc<unsigned char> __Kfp8(__pool, (size_t)kv_end * (size_t)D);
              ggml_cuda_pool_alloc<float> __Kamax(__pool, (size_t)kv_end);
              ggml_cuda_pool_alloc<float> __Ksf(__pool, (size_t)kv_end);
              ggml_cuda_pool_alloc<float> __KsfInv(__pool, (size_t)kv_end);
              ggml_cuda_pool_alloc<float> __IdxKScale(__pool, (size_t)kv_end);
              unsigned char *dQfp8 = __Qfp8.get(), *dKfp8 = __Kfp8.get();
              float *dKamax = __Kamax.get();
              float *dKsf   = __Ksf.get();
              float *dKsfInv= __KsfInv.get();
              float *dIdxKScale = __IdxKScale.get();
              dim3 tbH(256);
              dim3 tbT(32, 8);
              dim3 gdQ((Tc*H + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
              dim3 gdK((kv_end + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
              dim3 gdW((Tc + tbT.x - 1)/tbT.x, (H + tbT.y - 1)/tbT.y);
              // Convert GGML column-major to row-major layouts matching TileLang kernel
              k_colmajor_DN_to_rowmajor_ND<<<gdQ, tbT, 0, stream>>>(dQ, D, Tc*H, dQrm);
              if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                  std::vector<float> hQrm((size_t)(Tc*H)*(size_t)D);
                  cudaMemcpy(hQrm.data(), dQrm, hQrm.size()*sizeof(float), cudaMemcpyDeviceToHost);
                  int maxd = D < 8 ? D : 8;
                  fprintf(stderr, "[TL_FP8_DBG] Qrm[0,0..%d]:", maxd-1);
                  for (int d0 = 0; d0 < maxd; ++d0) fprintf(stderr, " % .6f", hQrm[d0]);
                  fprintf(stderr, "\n");
              }
              k_colmajor_DN_to_rowmajor_ND<<<gdK, tbT, 0, stream>>>(dK, D, kv_end, dKrm);
              k_colmajor_DN_to_rowmajor_ND<<<gdW, tbT, 0, stream>>>(dW, H, Tc, dWrm);
              if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                  std::vector<float> hKrm((size_t)kv_end*(size_t)D);
                  cudaMemcpy(hKrm.data(), dKrm, hKrm.size()*sizeof(float), cudaMemcpyDeviceToHost);
                  int maxd = D < 8 ? D : 8;
                  fprintf(stderr, "[TL_FP8_DBG] Krm[0,0..%d]:", maxd-1);
                  for (int d0 = 0; d0 < maxd; ++d0) fprintf(stderr, " % .6f", hKrm[d0]);
                  fprintf(stderr, "\n");
                  std::vector<float> hWrm((size_t)Tc*(size_t)H);
                  cudaMemcpy(hWrm.data(), dWrm, hWrm.size()*sizeof(float), cudaMemcpyDeviceToHost);
                  int maxh = H < 8 ? H : 8;
                  fprintf(stderr, "[TL_FP8_DBG] Wrm[0,0..%d] for token0:", maxh-1);
                  for (int h0 = 0; h0 < maxh; ++h0) fprintf(stderr, " % .6f", hWrm[h0]);
                  fprintf(stderr, "\n");
              }
              // Q: direct FP8 cast (no per-row scaling, as in TileLang test)
              {
                  size_t Qtotal = (size_t)(Tc*H) * (size_t)D;
                  dim3 gdQfp8((unsigned)((Qtotal + tbH.x - 1)/tbH.x));
                  k_rowmajor_f32_to_fp8_e4m3<<<gdQfp8, tbH, 0, stream>>>(dQrm, (int)(Tc*H), D, dQfp8);
                  CUDA_CHECK(cudaGetLastError());
              }
              // K: compute per-row amax, scales, and quantize with per-row scaling
              {
                  int rowsK = kv_end;
                  int colsK = D;
                  int threadsAmax = 256;
                  int blocksAmax = (rowsK + threadsAmax - 1) / threadsAmax;
                  k_rowmajor_f32_rowwise_absmax<<<blocksAmax, threadsAmax, 0, stream>>>(dKrm, rowsK, colsK, dKamax);
                  CUDA_CHECK(cudaGetLastError());
                  k_fp8_compute_row_scales<<<blocksAmax, threadsAmax, 0, stream>>>(dKamax, rowsK, dKsf, dKsfInv);
                  CUDA_CHECK(cudaGetLastError());
                  size_t Ktotal = (size_t)rowsK * (size_t)colsK;
                  dim3 gdKfp8((unsigned)((Ktotal + tbH.x - 1)/tbH.x));
                  k_rowmajor_f32_to_fp8_e4m3_rowwise_scaled<<<gdKfp8, tbH, 0, stream>>>(dKrm, rowsK, colsK, dKsfInv, dKfp8);
                  CUDA_CHECK(cudaGetLastError());
                  // Combine GGML k_scale (dKS) with FP8 per-row scale into effective IndexKScale
                  k_elemwise_mul<<<blocksAmax, threadsAmax, 0, stream>>>(dKS, dKsf, dIdxKScale, rowsK);
                  CUDA_CHECK(cudaGetLastError());
                  if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                      std::vector<float> hKS(rowsK);
                      std::vector<float> hKsf(rowsK);
                      std::vector<float> hIdxKScale(rowsK);
                      cudaMemcpy(hKS.data(), dKS, rowsK*sizeof(float), cudaMemcpyDeviceToHost);
                      cudaMemcpy(hKsf.data(), dKsf, rowsK*sizeof(float), cudaMemcpyDeviceToHost);
                      cudaMemcpy(hIdxKScale.data(), dIdxKScale, rowsK*sizeof(float), cudaMemcpyDeviceToHost);
                      int maxr = rowsK < 8 ? rowsK : 8;
                      fprintf(stderr, "[TL_FP8_DBG] KS[0..%d]:", maxr-1);
                      for (int i = 0; i < maxr; ++i) fprintf(stderr, " % .6f", hKS[i]);
                      fprintf(stderr, "\n");
                      fprintf(stderr, "[TL_FP8_DBG] Ksf[0..%d]:", maxr-1);
                      for (int i = 0; i < maxr; ++i) fprintf(stderr, " % .6f", hKsf[i]);
                      fprintf(stderr, "\n");
                      fprintf(stderr, "[TL_FP8_DBG] IdxKScale[0..%d]:", maxr-1);
                      for (int i = 0; i < maxr; ++i) fprintf(stderr, " % .6f", hIdxKScale[i]);
                      fprintf(stderr, "\n");
                  }
              }

              int __tl_call_status = 0;
              LAUNCH_PROFILE_KERNEL("PROFILE_TL_FP8_KONLY", TL_ONLY, stream, ([&](){
                          __tl_call_status = call(reinterpret_cast<fp8_e4_t*>(dQfp8), reinterpret_cast<fp8_e4_t*>(dKfp8),
                                                   dIdxKScale,
                                                   dLogits, dWrm,
                                                   dKS_i, dKE_i,
                                                   kv_end, Tc, stream);
                          })(), D, H, Tc, kv_end);
              if (__tl_call_status != 0) {
                  const char * __err = get_last_error();
                  fprintf(stderr, "[TL_FP8] call() failed status=%d err=%s\n", __tl_call_status, __err ? __err : "<none>");
              }
              CUDA_CHECK(cudaGetLastError());
              dim3 tblock(32, 8);
              dim3 tgrid((Tc + tblock.x - 1)/tblock.x, (kv_end + tblock.y - 1)/tblock.y);
              k_transpose_TcKv_to_KvTc<<<tgrid, tblock, 0, stream>>>(dLogits, Tc, kv_end, dOut);
              CUDA_CHECK(cudaGetLastError());
              if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                  std::vector<float> hLogits((size_t)Tc*(size_t)kv_end);
                  cudaMemcpy(hLogits.data(), dOut, hLogits.size()*sizeof(float), cudaMemcpyDeviceToHost);
                  int maxk = kv_end < 8 ? kv_end : 8;
                  int maxt = Tc < 2 ? Tc : 2;
                  fprintf(stderr, "[TL_FP8_DBG] Logits (kv x T) sample:\n");
                  for (int k = 0; k < maxk; ++k) {
                      for (int t = 0; t < maxt; ++t) {
                          float v = hLogits[k + (size_t)kv_end * t];
                          fprintf(stderr, "  L[%d,%d]=% .6f", k, t, v);
                      }
                      fprintf(stderr, "\n");
                  }
              }

          } else {
              dim3 tbT(32, 8);
              dim3 gdQ((Tc*H + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
              k_colmajor_DN_to_rowmajor_ND<<<gdQ, tbT, 0, stream>>>(dQ, D, Tc*H, dQrm);
              if (getenv("LLAMA_INDEXER_TL_FP8_DEBUG")) {
                  std::vector<float> hQrm((size_t)(Tc*H)*(size_t)D);
                  cudaMemcpy(hQrm.data(), dQrm, hQrm.size()*sizeof(float), cudaMemcpyDeviceToHost);
                  int maxd = D < 8 ? D : 8;
                  fprintf(stderr, "[TL_FP8_DBG] Qrm[0,0..%d]:", maxd-1);
                  for (int d0 = 0; d0 < maxd; ++d0) fprintf(stderr, " % .6f", hQrm[d0]);
                  fprintf(stderr, "\n");
              }
              dim3 gdK((kv_end + tbT.x - 1)/tbT.x, (D + tbT.y - 1)/tbT.y);
              k_colmajor_DN_to_rowmajor_ND<<<gdK, tbT, 0, stream>>>(dK, D, kv_end, dKrm);
              dim3 gdW((Tc + tbT.x - 1)/tbT.x, (H + tbT.y - 1)/tbT.y);
              k_colmajor_DN_to_rowmajor_ND<<<gdW, tbT, 0, stream>>>(dW, H, Tc, dWrm);
              auto compute_smem = [&](int bq, int bn) -> size_t {
                  // Mirror k_tl_mqa_attn_return_logits_port shared layout (float staging + f16)
                  const int WM = 16, WN = 16;
                  const int warps = max(1, threads/32);
                  size_t off = 0;
                  // Qs_f32
                  off = align16(off); off += (size_t)bq * (size_t)H * (size_t)D * sizeof(float);
                  // K0_f32, K1_f32
                  off = align16(off); off += (size_t)bn * (size_t)D * sizeof(float);
                  off = align16(off); off += (size_t)bn * (size_t)D * sizeof(float);
                  // ks0, ks1
                  off = align16(off); off += (size_t)bn * sizeof(float);
                  off = align16(off); off += (size_t)bn * sizeof(float);
                  // logits_blk
                  off = align16(off); off += (size_t)bn * (size_t)bq * sizeof(float);
                  // Qs_f16, K0_f16, K1_f16
                  off = align16(off); off += (size_t)bq * (size_t)H * (size_t)D * sizeof(__half);
                  off = align16(off); off += (size_t)bn * (size_t)D * sizeof(__half);
                  off = align16(off); off += (size_t)bn * (size_t)D * sizeof(__half);
                  // Csh per warp
                  off = align16(off); off += (size_t)warps * (WM*WN) * sizeof(float);
                  return off;
              };
              size_t shmem_bytes = compute_smem(block_Q, block_N);
              while (shmem_bytes > max_shmem && (block_Q > 1 || block_N > 1)) {
                  if (block_N >= block_Q && block_N > 1) block_N = (block_N + 1) / 2; else if (block_Q > 1) block_Q = (block_Q + 1) / 2;
                  shmem_bytes = compute_smem(block_Q, block_N);
              }
              CUDA_SET_SHARED_MEMORY_LIMIT(k_tl_mqa_attn_return_logits_port, (int)shmem_bytes);
              LAUNCH_PROFILE_KERNEL("PROFILE_TL_ONLY", TL_ONLY, stream, ([&](){
                          k_tl_mqa_attn_return_logits_port<<<gridTL, threads, shmem_bytes, stream>>>(
                                  dQrm, dKrm, dKS, dLogits, dWrm, dKS_i, dKE_i,
                                  Tc, kv_end, H, D, block_N, num_stages, threads, block_Q);
                          })(), D, H, Tc, kv_end);
              CUDA_CHECK(cudaGetLastError());
              dim3 tblock(32, 8);
              dim3 tgrid((Tc + tblock.x - 1)/tblock.x, (kv_end + tblock.y - 1)/tblock.y);
              k_transpose_TcKv_to_KvTc<<<tgrid, tblock, 0, stream>>>(dLogits, Tc, kv_end, dOut);
              CUDA_CHECK(cudaGetLastError());
          }
          cudaStreamSynchronize(stream);
          if (dStarts_tmp) cudaFree(dStarts_tmp);
          if (dEnds_tmp) cudaFree(dEnds_tmp);
          return;

    }
    if (use_wmma && D % 16 == 0 && (size_t)Tc * kv_end > 4096) {
        dim3 block(32,1,1);
        const int tokens_per_tile = max(1, 16 / min(H,16));
        dim3 grid((Tc + tokens_per_tile - 1) / tokens_per_tile, (kv_end + 15) / 16, 1);
        if (H % 16 == 0) {
            if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=wmma_hgrp grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);
            // Optional: build FP8 K+scale scratch for WMMA HGRP when no DeepGEMM path is active
            unsigned char *dKfp8 = nullptr;
            float *dKsf = nullptr;
            bool have_fp8_k = false;
            if (!use_dg_fp8) {
                ggml_cuda_pool & pool = ctx.pool(ggml_cuda_get_device());
                ggml_cuda_pool_alloc<unsigned char> __Kfp8(pool, (size_t)kv_end * (size_t)D);
                ggml_cuda_pool_alloc<float>         __Ksf (pool, (size_t)kv_end);
                dKfp8 = __Kfp8.get();
                dKsf  = __Ksf.get();

                // Row-major staging for K: [kv_end, D]
                ggml_cuda_pool_alloc<float> __Krm(pool, (size_t)kv_end * (size_t)D);
                float *dKrm = __Krm.get();
                dim3 tbT(32, 8);
                dim3 gdK((kv_end + tbT.x - 1)/tbT.x,
                         (D      + tbT.y - 1)/tbT.y);
                k_colmajor_DN_to_rowmajor_ND<<<gdK, tbT, 0, stream>>>(dK, D, kv_end, dKrm);

                int rowsK = kv_end;
                int colsK = D;
                int threadsA = 256;
                int blocksA  = (rowsK + threadsA - 1) / threadsA;
                ggml_cuda_pool_alloc<float> __Kamax (pool, (size_t)rowsK);
                ggml_cuda_pool_alloc<float> __KsfInv(pool, (size_t)rowsK);
                float *dKamax  = __Kamax.get();
                float *dKsfInv = __KsfInv.get();
                k_rowmajor_f32_rowwise_absmax<<<blocksA, threadsA, 0, stream>>>(
                    dKrm, rowsK, colsK, dKamax);
                k_fp8_compute_row_scales<<<blocksA, threadsA, 0, stream>>>(
                    dKamax, rowsK, dKsf, dKsfInv);

                size_t total = (size_t)rowsK * (size_t)colsK;
                dim3 tb(256);
                dim3 gd((unsigned)((total + tb.x - 1)/tb.x));
                k_rowmajor_f32_to_fp8_e4m3_rowwise_scaled<<<gd, tb, 0, stream>>>(
                    dKrm, rowsK, colsK, dKsfInv, dKfp8);

                have_fp8_k = true;
            }
            const float        *K_sf_g = have_fp8_k ? dKsf  : nullptr;
            const unsigned char*K_fp8  = have_fp8_k ? dKfp8 : nullptr;
            LAUNCH_PROFILE_KERNEL("PROFILE_WMMA_HGRP_ONLY", WMMA_HGRP_ONLY, stream, ([&](){
                        k_indexer_logits_wmma16_f32_hgrp<<<grid, block, 0, stream>>>(
                            dQ, K_fp8, K_sf_g, dW, dKS, D, H, Tc, kv_end,
                            dStarts_dev, dEnds_dev, dOut);
                        })(), D, H, Tc, kv_end);

        } else if (H <= 16 && (16 % H) == 0) {

            if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=wmma grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);
            LAUNCH_PROFILE_KERNEL("PROFILE_WMMA_ONLY", WMMA_ONLY, stream, ([&]{
                        k_indexer_logits_wmma16_bf16<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dStarts_dev, dEnds_dev, dOut);
                        })(), D, H, Tc, kv_end);

        } else {
            // not WMMA-friendly; fallback to tiled below
            int HEAD_CHUNK = getenv_int_("LLAMA_INDEXER_HEAD_CHUNK", 32);
            if (HEAD_CHUNK > 64) HEAD_CHUNK = 64;
            int PIPE_STAGES = getenv_int_("LLAMA_INDEXER_PIPE_STAGES", 2);
            if (PIPE_STAGES < 1) PIPE_STAGES = 1;
            if (PIPE_STAGES > 2) PIPE_STAGES = 2;
            int maxThreadsPerBlock = 1024;
            int threadsPerBlock = BLOCK_Q * BLOCK_N;
            if (threadsPerBlock > maxThreadsPerBlock) {
                int new_BLOCK_N = maxThreadsPerBlock / max(1, BLOCK_Q);
                if (new_BLOCK_N < 1) new_BLOCK_N = 1;
                if (sparse_debug_on()) printf("[INDEXER_DISPATCH] clamp BLOCK_N %d->%d due to threadsPerBlock=%d>=%d\n", BLOCK_N, new_BLOCK_N, threadsPerBlock, maxThreadsPerBlock);
                BLOCK_N = new_BLOCK_N;
            }
            dim3 blockT(BLOCK_Q, BLOCK_N);
            dim3 gridT((Tc + BLOCK_Q - 1)/BLOCK_Q, (kv_end + BLOCK_N - 1)/BLOCK_N);
            size_t shmem = (size_t)D_TILE * BLOCK_N * sizeof(float)
                         + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)
                         + (PIPE_STAGES >= 2 ? ((size_t)D_TILE * BLOCK_N * sizeof(float) + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)) : 0)
                         + (size_t)HEAD_CHUNK * BLOCK_Q * sizeof(float);
            CUDA_SET_SHARED_MEMORY_LIMIT(k_indexer_logits_tiled_f32, (int)shmem);
            if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=tiled grid=(%d,%d) block=(%d,%d) shmem=%zu Hc=%d stages=%d\n", gridT.x, gridT.y, blockT.x, blockT.y, (size_t)shmem, HEAD_CHUNK, PIPE_STAGES);
            LAUNCH_PROFILE_KERNEL("PROFILE_TILED_ONLY_2", TILED_ONLY_2, stream, ([&]{
                        k_indexer_logits_tiled_f32<<<gridT, blockT, shmem, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dStarts_dev, dEnds_dev, D_TILE, BLOCK_Q, BLOCK_N, exact_flag, HEAD_CHUNK, PIPE_STAGES, dOut);
                        })(), D, H, Tc, kv_end);


            cudaStreamSynchronize(stream);
            if (dStarts_tmp) cudaFree(dStarts_tmp);
            if (dEnds_tmp) cudaFree(dEnds_tmp);
            return;
        }
        {
            cudaError_t __err = cudaGetLastError();
            if (__err != cudaSuccess) {
                if (sparse_debug_on()) printf("[INDEXER_DISPATCH] WMMA launch failed: %s, falling back to tiled.\n", cudaGetErrorString(__err));
                int HEAD_CHUNK = getenv_int_("LLAMA_INDEXER_HEAD_CHUNK", 32);
                if (HEAD_CHUNK > 64) HEAD_CHUNK = 64;
                int PIPE_STAGES = getenv_int_("LLAMA_INDEXER_PIPE_STAGES", 2);
                if (PIPE_STAGES < 1) PIPE_STAGES = 1;
                if (PIPE_STAGES > 2) PIPE_STAGES = 2;
                int maxThreadsPerBlock = 1024;
                int threadsPerBlock = BLOCK_Q * BLOCK_N;
                if (threadsPerBlock > maxThreadsPerBlock) {
                    int new_BLOCK_N = maxThreadsPerBlock / max(1, BLOCK_Q);
                    if (new_BLOCK_N < 1) new_BLOCK_N = 1;
                    BLOCK_N = new_BLOCK_N;
                }
                dim3 blockT(BLOCK_Q, BLOCK_N);
                dim3 gridT((Tc + BLOCK_Q - 1)/BLOCK_Q, (kv_end + BLOCK_N - 1)/BLOCK_N);
                size_t shmem = (size_t)D_TILE * BLOCK_N * sizeof(float)
                             + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)
                             + (PIPE_STAGES >= 2 ? ((size_t)D_TILE * BLOCK_N * sizeof(float) + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)) : 0)
                             + (size_t)HEAD_CHUNK * BLOCK_Q * sizeof(float);
                CUDA_SET_SHARED_MEMORY_LIMIT(k_indexer_logits_tiled_f32, (int)shmem);
                if (sparse_debug_on()) printf("[INDEXER_DISPATCH] fallback tiled grid=(%d,%d) block=(%d,%d) shmem=%zu Hc=%d stages=%d\n", gridT.x, gridT.y, blockT.x, blockT.y, (size_t)shmem, HEAD_CHUNK, PIPE_STAGES);
                k_indexer_logits_tiled_f32<<<gridT, blockT, shmem, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dStarts_dev, dEnds_dev, D_TILE, BLOCK_Q, BLOCK_N, exact_flag, HEAD_CHUNK, PIPE_STAGES, dOut);
            }
        }
    } else {
        int HEAD_CHUNK = getenv_int_("LLAMA_INDEXER_HEAD_CHUNK", 32);
        if (HEAD_CHUNK > 64) HEAD_CHUNK = 64;
        int PIPE_STAGES = getenv_int_("LLAMA_INDEXER_PIPE_STAGES", 2);
        if (PIPE_STAGES < 1) PIPE_STAGES = 1;
        if (PIPE_STAGES > 2) PIPE_STAGES = 2;
        // Sanity clamp block dims to device maximum threads per block
        int maxThreadsPerBlock = 1024;
        int threadsPerBlock = BLOCK_Q * BLOCK_N;
        if (threadsPerBlock > maxThreadsPerBlock) {
            int new_BLOCK_N = maxThreadsPerBlock / max(1, BLOCK_Q);
            if (new_BLOCK_N < 1) new_BLOCK_N = 1;
            if (sparse_debug_on()) printf("[INDEXER_DISPATCH] clamp BLOCK_N %d->%d due to threadsPerBlock=%d>=%d\n", BLOCK_N, new_BLOCK_N, threadsPerBlock, maxThreadsPerBlock);
            BLOCK_N = new_BLOCK_N;
        }
        dim3 blockT(BLOCK_Q, BLOCK_N);
        dim3 gridT((Tc + BLOCK_Q - 1)/BLOCK_Q, (kv_end + BLOCK_N - 1)/BLOCK_N);
        size_t shmem = (size_t)D_TILE * BLOCK_N * sizeof(float)
                     + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)
                     + (PIPE_STAGES >= 2 ? ((size_t)D_TILE * BLOCK_N * sizeof(float) + (size_t)D_TILE * BLOCK_Q * HEAD_CHUNK * sizeof(float)) : 0)
                     + (size_t)HEAD_CHUNK * BLOCK_Q * sizeof(float);
        // Raise per-kernel dynamic shared memory limit to our requirement (best-effort)
        CUDA_SET_SHARED_MEMORY_LIMIT(k_indexer_logits_tiled_f32, (int)shmem);
        if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=tiled grid=(%d,%d) block=(%d,%d) shmem=%zu Hc=%d stages=%d\n", gridT.x, gridT.y, blockT.x, blockT.y, (size_t)shmem, HEAD_CHUNK, PIPE_STAGES);
        LAUNCH_PROFILE_KERNEL("PROFILE_TILED_ONLY_1", TILED_ONLY_1, stream, ([&]{
                    k_indexer_logits_tiled_f32<<<gridT, blockT, shmem, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dStarts_dev, dEnds_dev, D_TILE, BLOCK_Q, BLOCK_N, exact_flag, HEAD_CHUNK, PIPE_STAGES, dOut);
                    })(), D, H, Tc, kv_end);
    }

    if (dStarts_tmp) cudaFree(dStarts_tmp);
    if (dEnds_tmp)   cudaFree(dEnds_tmp);
}
