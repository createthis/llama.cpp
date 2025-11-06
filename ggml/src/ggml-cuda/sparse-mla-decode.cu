#include "common.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Fused sparse MLA decode kernel supporting MQA/GQA (Hq may differ from Hkv)
// Layouts:
//  Q: [D, Hq]
//  K: [D, Hkv, N]
//  V: [Dv, Hkv, N]
//  topk: [Ksel]
// Output:
//  Out: [Dv, Hq]

__global__ void k_sparse_mla_decode(const float * __restrict__ Q,
                                    const float * __restrict__ K,
                                    const float * __restrict__ V,
                                    const int32_t * __restrict__ topk,
                                    int D, int Hq, int Hkv, int Dv, int N, int Ksel,
                                    float kq_scale, float softcap,
                                    float * __restrict__ Out) {
    int h = blockIdx.x;
    if (h >= Hq) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wcount = (blockDim.x + 31) >> 5;

    extern __shared__ float smem[];
    float * scores = smem;                     // Ksel floats
    float * s_wmax = scores + Ksel;            // wcount floats
    float * s_wsum = s_wmax + wcount;          // wcount floats

    // compute logits and local max
    float m_local = -1e30f;
    const int hk = (Hkv == 1 ? 0 : (h % Hkv));
    const float * __restrict__ qh = Q + (size_t)D * h;

    for (int i = threadIdx.x; i < Ksel; i += blockDim.x) {
        int idx = topk[i];
        float dot = -1e30f;
        if (idx >= 0 && idx < N) {
            const float * __restrict__ kh = K + (size_t)D * (hk + (size_t)Hkv * idx);
            dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += qh[d] * kh[d];
            dot *= kq_scale;
            if (softcap > 0.0f) {
                dot = tanhf(dot / softcap) * softcap;
            }
        }
        scores[i] = dot;
        m_local = fmaxf(m_local, dot);
    }

    // warp reduce max
    for (int off = 16; off > 0; off >>= 1) {
        m_local = fmaxf(m_local, __shfl_down_sync(0xffffffff, m_local, off));
    }
    if (lane == 0) {
        s_wmax[warp] = m_local;
    }
    __syncthreads();

    // block reduce max using first warp
    float maxv = -1e30f;
    if (warp == 0) {
        float v = (lane < wcount) ? s_wmax[lane] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
        }
        if (lane == 0) s_wmax[0] = v;
    }
    __syncthreads();
    maxv = s_wmax[0];

    // compute exp and local sum
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < Ksel; i += blockDim.x) {
        float e = __expf(scores[i] - maxv);
        scores[i] = e;
        lsum += e;
    }
    for (int off = 16; off > 0; off >>= 1) {
        lsum += __shfl_down_sync(0xffffffff, lsum, off);
    }
    if (lane == 0) {
        s_wsum[warp] = lsum;
    }
    __syncthreads();

    // block reduce sum using first warp
    float snorm = 0.0f;
    if (warp == 0) {
        float v = (lane < wcount) ? s_wsum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) s_wsum[0] = v;
    }
    __syncthreads();
    snorm = s_wsum[0];
    float inv = snorm > 0.0f ? 1.0f / snorm : 0.0f;

    // accumulate output for this head
    for (int dv = threadIdx.x; dv < Dv; dv += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < Ksel; ++i) {
            int idx = topk[i]; if (idx < 0 || idx >= N) continue;
            float p = scores[i] * inv;
            const float * __restrict__ vh = V + (size_t)Dv * (hk + (size_t)Hkv * idx);
            acc += p * vh[dv];
        }
        Out[dv + (size_t)Dv * h] = acc;
    }
}

extern "C" void ggml_cuda_sparse_mla_decode_device(ggml_backend_cuda_context & ctx,
                                                    const float * q,
                                                    const float * k,
                                                    const float * v,
                                                    const int32_t * topk,
                                                    int D, int Hq, int Hkv, int Dv,
                                                    int N, int Ksel,
                                                    float kq_scale, float softcap,
                                                    float * out);

extern "C" void ggml_cuda_sparse_mla_decode_device(ggml_backend_cuda_context & ctx,
                                                    const float * q,
                                                    const float * k,
                                                    const float * v,
                                                    const int32_t * topk,
                                                    int D, int Hq, int Hkv, int Dv,
                                                    int N, int Ksel,
                                                    float kq_scale, float softcap,
                                                    float * out) {
    dim3 grid(Hq);
    dim3 block(128);
    int warps = (block.x + 31) >> 5;
    size_t shmem = (size_t)Ksel * sizeof(float) + 2 * (size_t)warps * sizeof(float);
    k_sparse_mla_decode<<<grid, block, shmem, ctx.stream()>>>(q, k, v, topk, D, Hq, Hkv, Dv, N, Ksel, kq_scale, softcap, out);
}
