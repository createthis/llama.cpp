#include "common.cuh"
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

    extern __shared__ float shmem[];
    float * K_sh = shmem; // [D_TILE, BLOCK_N]
    float * Q_sh = K_sh + (size_t)D_TILE * BLOCK_N; // [D_TILE, BLOCK_Q*H]

    float acc = 0.0f;

    // iterate heads; accumulate head contributions into acc
    for (int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (int d0 = 0; d0 < D; d0 += D_TILE) {
            int cur = min(D_TILE, D - d0);
            // cooperative load K_sh[d, j]
            int totalK = cur * BLOCK_N;
            int stride = blockDim.x * blockDim.y;
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            for (int idx = tid; idx < totalK; idx += stride) {
                int di = idx / BLOCK_N;
                int j  = idx % BLOCK_N;
                int gk = k0 + j;
                float v = 0.0f;
                if (gk < kv) v = K[(size_t)(d0 + di) + (size_t)D * gk];
                K_sh[di * BLOCK_N + j] = v;
            }
            // cooperative load Q_sh[d, q*H + h'] for all q in tile and all heads
            int totalQ = cur * (BLOCK_Q * H);
            for (int idx = tid; idx < totalQ; idx += stride) {
                int di = idx / (BLOCK_Q * H);
                int rem = idx % (BLOCK_Q * H);
                int q   = rem / H;
                int hh  = rem % H;
                int gt  = t0 + q;
                float v = 0.0f;
                if (gt < Tc) v = Q[(size_t)(d0 + di) + (size_t)D * (gt*H + hh)];
                Q_sh[di * (BLOCK_Q * H) + rem] = v;
            }
            __syncthreads();
            // accumulate this tile for current (token,h,kv_idx)
            // dot += sum_{di=0..cur-1} K_sh[di, k_local] * Q_sh[di, t_local*H + h]
            int qoff = t_local * H + h;
            for (int di = 0; di < cur; ++di) {
                dot += K_sh[di * BLOCK_N + k_local] * Q_sh[di * (BLOCK_Q * H) + qoff];
            }
            __syncthreads();
        }
        if (dot < 0.0f) dot = 0.0f;
        float w = W[h + (size_t)H * token];
        acc += dot * w;
    }
    acc *= k_scale[kv_idx];
    Out[kv_idx + (size_t)kv * token] = acc;
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

    if (use_naive) {
        dim3 block(32, 4);
        dim3 grid((Tc + block.x - 1)/block.x, (kv_end + block.y - 1)/block.y);
        k_indexer_logits_fused<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dOut);
    } else {
        dim3 block(BLOCK_Q, BLOCK_N);
        dim3 grid((Tc + BLOCK_Q - 1)/BLOCK_Q, (kv_end + BLOCK_N - 1)/BLOCK_N);
        size_t shmem = (size_t)D_TILE * BLOCK_N * sizeof(float) + (size_t)D_TILE * (BLOCK_Q * H) * sizeof(float);
        k_indexer_logits_tiled_f32<<<grid, block, shmem, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, D_TILE, BLOCK_Q, BLOCK_N, dOut);
    }
}

