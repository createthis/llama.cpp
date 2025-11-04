#include "common.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../../include/ggml-cuda-indexer.h"
#ifndef SEL_DEBUG
#define SEL_DEBUG 2
#endif

// Simple baseline fused kernel: compute K^T * Q -> ReLU, then per-head weighted sum, multiply k_scale.
// This is a placeholder for a fully-optimized version. It assumes row-major contiguous inputs.

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
        const float * qv = Q + (size_t)D * (tc + h*Tc);
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
    dim3 block(32, 4);
    dim3 grid((Tc + block.x - 1)/block.x, (kv_end + block.y - 1)/block.y);
    k_indexer_logits_fused<<<grid, block, 0, stream>>>(dQ, dK, dW, dKS, D, H, Tc, kv_end, dOut);
}

