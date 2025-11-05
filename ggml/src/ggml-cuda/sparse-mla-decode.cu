#include "common.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void k_sparse_mla_decode(const float * __restrict__ Q,
                                    const float * __restrict__ K,
                                    const float * __restrict__ V,
                                    const int32_t * __restrict__ topk,
                                    int D, int H, int Dv, int N, int Ksel,
                                    float kq_scale, float softcap,
                                    float * __restrict__ Out) {
    int h = blockIdx.x;
    if (h >= H) return;
    extern __shared__ float smem[];
    float * scores = smem; // Ksel
    float m = -1e30f;
    for (int i = threadIdx.x; i < Ksel; i += blockDim.x) {
        int idx = topk[i]; if (idx < 0 || idx >= N) { scores[i] = -1e30f; continue; }
        const float * qh = Q + (size_t)D*h;
        const float * kh = K + (size_t)D*(h + (size_t)H*idx);
        float dot = 0.0f;
        for (int d=0; d<D; ++d) dot += qh[d] * kh[d];
        dot *= kq_scale;
        if (softcap > 0.f) {
            dot = tanhf(dot / softcap) * softcap;
        }
        if (dot < -1e30f) dot = -1e30f;
        scores[i] = dot;
        m = fmaxf(m, dot);
    }
    __shared__ float smax;
    __shared__ float snorm;
    float tmax = m;
    for (int off=16; off>0; off>>=1) tmax = fmaxf(tmax, __shfl_down_sync(0xffffffff, tmax, off));
    if ((threadIdx.x & 31) == 0) atomicMax((int*)&smax, __float_as_int(tmax));
    __syncthreads();
    float maxv = __int_as_float((int)smax);
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < Ksel; i += blockDim.x) {
        float e = __expf(scores[i] - maxv);
        scores[i] = e;
        lsum += e;
    }
    for (int off=16; off>0; off>>=1) lsum += __shfl_down_sync(0xffffffff, lsum, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&snorm, lsum);
    __syncthreads();
    float inv = 1.0f / snorm;
    for (int dv = threadIdx.x; dv < Dv; dv += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < Ksel; ++i) {
            int idx = topk[i]; if (idx < 0 || idx >= N) continue;
            float p = scores[i] * inv;
            const float * vh = V + (size_t)Dv*(h + (size_t)H*idx);
            acc += p * vh[dv];
        }
        Out[dv + (size_t)Dv*h] = acc;
    }
}

extern "C" void ggml_cuda_sparse_mla_decode_device(ggml_backend_cuda_context & ctx,
                                                    const float * q,
                                                    const float * k,
                                                    const float * v,
                                                    const int32_t * topk,
                                                    int D, int H, int Dv,
                                                    int N, int Ksel,
                                                    float kq_scale, float softcap,
                                                    float * out) {
    dim3 grid(H);
    dim3 block(128);
    size_t shmem = (size_t)Ksel * sizeof(float);
    k_sparse_mla_decode<<<grid, block, shmem, ctx.stream()>>>(q,k,v,topk,D,H,Dv,N,Ksel,kq_scale,softcap,out);
}
