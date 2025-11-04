#include "common.cuh"
#include <mma.h>
using namespace nvcuda;

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../../include/ggml-cuda-indexer.h"
#ifndef SEL_DEBUG
#endif

// Simple baseline fused kernel: compute K^T * Q -> ReLU, then per-head weighted sum, multiply k_scale.
// This is a placeholder for a fully-optimized version. It assumes row-major contiguous inputs.

// helpers to read env
static inline int getenv_int_(const char * name, int def) {
    const char * s = getenv(name);
    if (!s || !*s) return def;
    int v = atoi(s);
    return v > 0 ? v : def;
}
static inline bool sparse_debug_on(){ const char *d=getenv("LLAMA_SPARSE_DEBUG"); return d && *d && atoi(d)!=0; }


// Tiled, shared-memory fused kernel (float inputs, float accum)
// Q: [D, Tc*H], K: [D, kv], W: [H, Tc], k_scale: [kv]; Out: [kv, Tc]
__global__ void k_indexer_logits_tiled_f32(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ W,
    const float * __restrict__ k_scale,
    int D, int H, int Tc, int kv,
    int D_TILE, int BLOCK_Q, int BLOCK_N,
    float * __restrict__ Out) {
    int t_local = threadIdx.x; // [0..BLOCK_Q)
    int k_local = threadIdx.y; // [0..BLOCK_N)
    int t0 = blockIdx.x * BLOCK_Q;
    int k0 = blockIdx.y * BLOCK_N;
    int token = t0 + t_local;
    int kv_idx = k0 + k_local;
    if (t_local >= BLOCK_Q || k_local >= BLOCK_N) return;
    if (token >= Tc || kv_idx >= kv) return;

    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
        float dot = 0.0f;
        const float * qv = Q + (size_t)D * (token*H + h);
        const float * kvp = K + (size_t)D * kv_idx;
        #pragma unroll 1
        for (int d = 0; d < D; ++d) {
            dot += qv[d] * kvp[d];
        }
        if (dot < 0.0f) dot = 0.0f;
        float w = W[h + (size_t)H * token];
        acc += dot * w;
    }
    acc *= k_scale[kv_idx];
    Out[kv_idx + (size_t)kv * token] = acc;
}

// WMMA kernel (half input, float accum) for D multiple of 16
__global__ void k_indexer_logits_wmma_hf(
    const half * __restrict__ Q, // [D, Tc*H]
    const half * __restrict__ K, // [D, kv]
    const float* __restrict__ W, // [H, Tc]
    const float* __restrict__ k_scale, // [kv]
    int D, int H, int Tc, int kv,
    int BLOCK_Q, int BLOCK_N,
    float * __restrict__ Out) {
#if __CUDA_ARCH__ >= 800
    int t0 = blockIdx.x * BLOCK_Q;
    int k0 = blockIdx.y * BLOCK_N;
    int token = t0 + threadIdx.x; // 1 thread per token column in this simple prototype
    if (token >= Tc) return;
    // One warp handles BLOCK_N rows for current token; keep prototype minimal
    int lane = threadIdx.x & 31;
    for (int rowBase = 0; rowBase < BLOCK_N; rowBase += 16) {
        int kv_idx = k0 + rowBase;
        if (kv_idx >= kv) break;
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
            wmma::fill_fragment(cFrag, 0.0f);
            for (int d0 = 0; d0 < D; d0 += 16) {
                const half * a = &K[d0 + (size_t)D * kv_idx]; // [16 x 1] columns from K for 16 rows
                const half * b = &Q[d0 + (size_t)D * (token*H + h)]; // [16 x 1]
                // Fake a 16x16x16 by broadcasting to fill fragments (prototype, not fully efficient)
                __shared__ half As[16*16], Bs[16*16];
                if (lane < 16) {
                    As[lane] = a[lane];
                    Bs[lane] = b[lane];
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> aFrag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;
                wmma::load_matrix_sync(aFrag, As, 16);
                wmma::load_matrix_sync(bFrag, Bs, 16);
                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                __syncthreads();
            }
            float dot = 0.0f;
            // Reduce cFrag (sum all elements) as prototype
            for (int i = 0; i < cFrag.num_elements; ++i) dot += cFrag.x[i];
            if (dot < 0.0f) dot = 0.0f;
            float w = W[h + (size_t)H * token];
            acc += dot * w;
        }
        acc *= k_scale[kv_idx];
        if (lane == 0) Out[kv_idx + (size_t)kv * token] = acc;
    }
#else
    (void)Q; (void)K; (void)W; (void)k_scale; (void)D; (void)H; (void)Tc; (void)kv; (void)BLOCK_Q; (void)BLOCK_N; (void)Out;
#endif
}


// WMMA 16x16x16 (float input cast to half), one warp per block
__global__ void k_indexer_logits_wmma16_f32(
    const float * __restrict__ Q, // [D, Tc*H]
    const float * __restrict__ K, // [D, kv]
    const float * __restrict__ W, // [H, Tc]
    const float * __restrict__ k_scale, // [kv]
    int D, int H, int Tc, int kv,
    float * __restrict__ Out) {
#if __CUDA_ARCH__ >= 700
    const int tokens_per_tile = max(1, 16 / H);
    const int t0 = blockIdx.x * tokens_per_tile;
    const int k0 = blockIdx.y * 16;
    if (t0 >= Tc || k0 >= kv) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ __half A_sh[16*16]; // row-major
    __shared__ __half B_sh[16*16]; // col-major

    // Iterate K dimension in 16-slices
    for (int d0 = 0; d0 < D; d0 += 16) {
        // Load A_sh: rows are kv rows, cols are k-slice
        int lane = threadIdx.x & 31;
        for (int idx = lane; idx < 16*16; idx += 32) {
            int mi = idx / 16;
            int di = idx % 16;
            int kv_idx = k0 + mi;
            __half v = __float2half(0.0f);
            if (kv_idx < kv && d0 + di < D) {
                float f = K[(size_t)(d0 + di) + (size_t)D * kv_idx];
                v = __float2half(f);
            }
            A_sh[mi * 16 + di] = v;
        }
        // Load B_sh: columns are token*H + h, col-major
        for (int idx = lane; idx < 16*16; idx += 32) {
            int di = idx / 16; // k index
            int cj = idx % 16; // column index 0..15 => (tok_local,h)
            int tok_local = cj / H;
            int h = cj % H;
            int tok = t0 + tok_local;
            __half v = __float2half(0.0f);
            if (tok < Tc && d0 + di < D) {
                float f = Q[(size_t)(d0 + di) + (size_t)D * (tok*H + h)];
                v = __float2half(f);
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

    __shared__ float C_sh[16*16];
    wmma::store_matrix_sync(C_sh, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    // For each kv row and token in tile, reduce across heads and write
    int lane = threadIdx.x & 31;
    for (int idx = lane; idx < 16 * tokens_per_tile; idx += 32) {
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
            s *= k_scale[kv_idx];
            Out[kv_idx + (size_t)kv * tok] = s;
        }
    }
#endif
}


// WMMA 16x16x16 BF16 (float input cast to bf16), one warp per block
__global__ void k_indexer_logits_wmma16_bf16(
    const float * __restrict__ Q, // [D, Tc*H]
    const float * __restrict__ K, // [D, kv]
    const float * __restrict__ W, // [H, Tc]
    const float * __restrict__ k_scale, // [kv]
    int D, int H, int Tc, int kv,
    float * __restrict__ Out) {
#if __CUDA_ARCH__ >= 800
    const int tokens_per_tile = max(1, 16 / H);
    const int t0 = blockIdx.x * tokens_per_tile;
    const int k0 = blockIdx.y * 16;
    if (t0 >= Tc || k0 >= kv) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ __nv_bfloat16 A_sh[16*16]; // row-major
    __shared__ __nv_bfloat16 B_sh[16*16]; // col-major

    // Iterate K dimension in 16-slices
    for (int d0 = 0; d0 < D; d0 += 16) {
        int lane = threadIdx.x & 31;
        // Load A_sh
        for (int idx = lane; idx < 16*16; idx += 32) {
            int mi = idx / 16;
            int di = idx % 16;
            int kv_idx = k0 + mi;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (kv_idx < kv && d0 + di < D) {
                float f = K[(size_t)(d0 + di) + (size_t)D * kv_idx];
                v = __float2bfloat16(f);
            }
            A_sh[mi * 16 + di] = v;
        }
        // Load B_sh (col-major)
        for (int idx = lane; idx < 16*16; idx += 32) {
            int di = idx / 16; // k index
            int cj = idx % 16; // column index 0..15 => (tok_local,h)
            int tok_local = cj / H;
            int h = cj % H;
            int tok = t0 + tok_local;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (tok < Tc && d0 + di < D) {
                float f = Q[(size_t)(d0 + di) + (size_t)D * (tok*H + h)];
                v = __float2bfloat16(f);
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

    __shared__ float C_sh[16*16];
    wmma::store_matrix_sync(C_sh, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    int lane = threadIdx.x & 31;
    for (int idx = lane; idx < 16 * tokens_per_tile; idx += 32) {
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
            s *= k_scale[kv_idx];
            Out[kv_idx + (size_t)kv * tok] = s;
        }
    }
#endif
}

__global__ void k_indexer_logits_fused(const float * __restrict__ Q, // [D, Tc*H]
                                       const float * __restrict__ K, // [D, kv]
                                       const float * __restrict__ W, // [H, Tc]
                                       const float * __restrict__ k_scale, // [kv]
                                       int D, int H, int Tc, int kv,
                                       float * __restrict__ out) {   // [kv, Tc]
    int tc = blockIdx.x * blockDim.x + threadIdx.x; // token col [0..Tc)
    int kv_idx = blockIdx.y * blockDim.y + threadIdx.y; // kv row [0..kv)
    if (tc >= Tc || kv_idx >= kv) return;

#ifdef SEL_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[fused] D=%d H=%d Tc=%d kv=%d\n", D, H, Tc, kv);
    }
#endif

    // For each head, compute dot(K[:,kv_idx], Q[:,tc*H + h])
    float acc_logits = 0.0f;
    for (int h = 0; h < H; ++h) {
        const float * qv = Q + (size_t)D * (tc*H + h);
        const float * kvp = K + (size_t)D * kv_idx;
        float dot = 0.0f;
        // naive dot
        for (int d = 0; d < D; ++d) dot += qv[d] * kvp[d];
        if (dot < 0.0f) dot = 0.0f; // ReLU
        float w = W[h + (size_t)H * tc];

#ifdef SEL_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tc < 2 && kv_idx < 2 && threadIdx.x == 0 && threadIdx.y == 0) {
            // full-precision raw over D
            float raw2 = 0.0f;
            for (int dd = 0; dd < D; ++dd) raw2 += qv[dd] * kvp[dd];
            float rel2 = raw2 < 0.0f ? 0.0f : raw2;
            printf("[fused] step tc=%d kv=%d h=%d raw=%.6e rel=%.6e w=%.6e acc_pre=%.6e\n",
                   tc, kv_idx, h, raw2, rel2, w, acc_logits);
        }
#endif
        acc_logits += dot * w;
#ifdef SEL_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tc < 2 && kv_idx < 2 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[fused] step tc=%d kv=%d h=%d acc_post=%.6e\n",
                   tc, kv_idx, h, acc_logits);
        }
#endif

    }
    float ks = k_scale[kv_idx];

#ifdef SEL_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tc < 2 && kv_idx < 2 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[fused] final tc=%d kv=%d acc=%.6e ks=%.6e out=%.6e\n",
               tc, kv_idx, acc_logits, ks, acc_logits*ks);
    }
#endif
    out[kv_idx + (size_t)kv * tc] = acc_logits * ks;


#ifdef SEL_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tc < 2 && kv_idx < 2 && threadIdx.x == 0 && threadIdx.y == 0) {
        // dump for h=0..min(H,2)
        for (int hh = 0; hh < (H < 2 ? H : 2); ++hh) {
            int col = tc*H + hh; // current mapping used in kernel
            const float * qv0 = Q + (size_t)D * col;
            const float * kvp0 = K + (size_t)D * kv_idx;
            printf("[fused] tc=%d kv=%d h=%d col=%d W=%.6e k_scale=%.6e\n", 
                tc, kv_idx, hh, col, W[hh + (size_t)H * tc], k_scale[kv_idx]);
            for (int dd = 0; dd < (D < 8 ? D : 8); ++dd) {
                printf("  q[%d]=%.6e k[%d]=%.6e\n", 
                dd, qv0[dd], dd, kvp0[dd]);
            }
        }
        if (!isfinite(acc_logits)) printf("[fused] acc_logits non-finite at tc=%d kv=%d\n",
            tc, kv_idx);
        float vtest = acc_logits * k_scale[kv_idx];
        if (!isfinite(vtest)) printf("[fused] scaled out non-finite at tc=%d kv=%d\n",
            tc, kv_idx);
    }
#endif

}

extern "C" void ggml_cuda_indexer_logits_fused_host(const float * Q,
                                                     const float * K,
                                                     const float * W,
                                                     const float * k_scale,
                                                     int D, int H, int Tc, int kv_end,
                                                     float * out) {
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    size_t qsz = (size_t)D * Tc * H;
    size_t ksz = (size_t)D * kv_end;
    size_t wsz = (size_t)H * Tc;
    size_t osz = (size_t)kv_end * Tc;
    float *dQ=nullptr, *dK=nullptr, *dW=nullptr, *dKS=nullptr, *dO=nullptr;
    cudaMalloc(&dQ, qsz*sizeof(float));
    cudaMalloc(&dK, ksz*sizeof(float));
    cudaMalloc(&dW, wsz*sizeof(float));
    cudaMalloc(&dKS, kv_end*sizeof(float));
    cudaMalloc(&dO, osz*sizeof(float));
    cudaMemcpyAsync(dQ, Q, qsz*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dK, K, ksz*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dW, W, wsz*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dKS, k_scale, kv_end*sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(32, 4);
    dim3 grid((Tc + block.x - 1)/block.x, (kv_end + block.y - 1)/block.y);
    if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=naive grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);
        k_indexer_logits_fused<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dO);

    cudaMemcpyAsync(out, dO, osz*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dQ); cudaFree(dK); cudaFree(dW); cudaFree(dKS); cudaFree(dO);
}

extern "C" void ggml_cuda_indexer_logits_fused_device(ggml_backend_cuda_context & ctx,
                                                       const float * dQ,
                                                       const float * dK,
                                                       const float * dW,
                                                       const float * dKS,
                                                       int D, int H, int Tc, int kv_end,
                                                       float * dOut) {
    cudaStream_t stream = ctx.stream();
    // env knobs for tile sizes
    int BLOCK_Q = getenv_int_("LLAMA_INDEXER_BLOCK_Q", 1);
    int BLOCK_N = getenv_int_("LLAMA_INDEXER_BLOCK_N", 64);
    int D_TILE  = getenv_int_("LLAMA_INDEXER_D_TILE", 32);
    // Select kernel based on env; default to tiled
    bool use_naive = false;
    if (const char *s = getenv("LLAMA_INDEXER_USE_NAIVE"); s && atoi(s) != 0) use_naive = true;
    bool use_wmma = false;
    if (const char *s = getenv("LLAMA_INDEXER_USE_WMMA"); s && atoi(s) != 0) use_wmma = true;

    if (sparse_debug_on()) printf("[INDEXER_DISPATCH] use_naive=%d use_wmma=%d D=%d H=%d Tc=%d kv=%d BLOCK_Q=%d BLOCK_N=%d D_TILE=%d\n", (int)use_naive, (int)use_wmma, D, H, Tc, kv_end, BLOCK_Q, BLOCK_N, D_TILE);
    if (use_naive) {
        dim3 block(32, 4);
        dim3 grid((Tc + block.x - 1)/block.x, (kv_end + block.y - 1)/block.y);
        if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=naive grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);
        k_indexer_logits_fused<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dOut);
    } else if (use_wmma && D % 16 == 0 && (size_t)Tc * kv_end > 4096 && H <= 16 && (16 % H) == 0) {
        // launch WMMA16 path (skip for tiny problems)
        dim3 block(32,1,1);
        const int tokens_per_tile = max(1, 16 / H);
        dim3 grid((Tc + tokens_per_tile - 1) / tokens_per_tile, (kv_end + 15) / 16, 1);
        if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=wmma grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);
        k_indexer_logits_wmma16_f32<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dOut);
    } else {
        dim3 blockT(BLOCK_Q, BLOCK_N);
        dim3 gridT((Tc + BLOCK_Q - 1)/BLOCK_Q, (kv_end + BLOCK_N - 1)/BLOCK_N);
        size_t shmem = (size_t)D_TILE * BLOCK_N * sizeof(float) + (size_t)D_TILE * (BLOCK_Q * H) * sizeof(float);
        if (sparse_debug_on()) printf("[INDEXER_DISPATCH] launch=tiled grid=(%d,%d) block=(%d,%d) shmem=%zu\n", gridT.x, gridT.y, blockT.x, blockT.y, (size_t)shmem);
        k_indexer_logits_tiled_f32<<<gridT, blockT, shmem, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, D_TILE, BLOCK_Q, BLOCK_N, dOut);
    }
}

