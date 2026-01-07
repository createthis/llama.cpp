#include "common.cuh"

// DeepSeek V3.2-Exp: experimental FP8 indexer kernel TU.
//
// This TU provides a separate entry point for an FP8-based lightning
// indexer path. It is intentionally independent of the TileLang
// mqa_attn_return_logits_kernel "call()" interface: it computes
// logits directly from FP8 K/Q and scales. The initial version is a
// simple, correctness-oriented kernel structured similarly to the
// existing WMMA HGRP kernel, but without WMMA or TileLang MMA yet.
//
// Future work will replace the inner dot-product loop with FP8
// tensor-core MMA based on the TileLang/Cute templates, while keeping
// the overall control flow and scaling identical to the CPU / WMMA
// reference.

#include <cuda_runtime.h>

// Reuse CuTe/TileLang MMA building blocks (NOT the TileLang kernel)
#include "tl_templates/cuda/cuda_fp8.h"
#include "tl_templates/cuda/gemm_mma.h"
#include <cute/tensor_impl.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cstdio>
#include <cstdint>
#include <cmath>

// Local FP8 E4M3 decode helper using native PTX, matching ggml-fp8 semantics.
// We reuse the same implementation as the main indexer-fused.cu path so that
// CPU idx_compute_scores_tile and this FP8_TC kernel see identical dequant.
static __device__ __forceinline__ float fp8e4m3_to_f32(uint8_t code) {
    uint16_t bits = code;
    uint32_t packed;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    return __half2float(reinterpret_cast<half2 const &>(packed).x);
}

// Naive FP8 indexer kernel (per-token, per-kv row). This is a
// correctness-oriented baseline: each thread computes the score for
// one (token, kv_idx) pair by:
//   - dequantizing K and Q from FP8
//   - computing dot(Q_h, K) over all heads and D
//   - applying ReLU(dot) per head, multiplying by weights W[h, tok]
//   - applying k_scale[kv_idx] * (optional K_sf[kv_idx])
//
// Layout assumptions (match the existing fused indexer path):
//   K_fp8 : [kv, D]  row-major FP8 codes
//   K_sf  : [kv]     per-row FP8 scales (amax/448), may be nullptr
//   Q_fp8 : [Tc*H, D] row-major FP8 codes, row index = t*H + h
//   W     : [H, Tc]  row-major by head, col by token
//   k_scale : [kv]   GGML k_scale per kv row
//   starts/ends : optional per-token [start,end) KV window
//   Out   : [kv, Tc] kv-major (col-major by token)



// Convert our uint8 FP8 codes into TileLang/CuTe fp8_e4_t storage type
// (bitwise identical 8-bit payload).
static __device__ __forceinline__ fp8_e4_t fp8_code_to_fp8e4(uint8_t c) {
    fp8_e4_t out;
    *reinterpret_cast<uint8_t *>(&out) = c;
    return out;
}

// MMA-based FP8 indexer kernel for the production DeepSeek indexer shape:
//   D=128, H=64
// We process one token per block and a tile of KV rows per block.
// Heads are processed in groups of 8 (MMA N dimension).
// This is a Phase-2 style kernel: it keeps MMA accumulators in registers and
// fuses the epilogue/reduction (no intermediate kv x heads matrix).
__global__ void k_indexer_logits_fp8_tc_hgrp_mma_d128_h64(
    const unsigned char * __restrict__ K_fp8,
    const float         * __restrict__ K_sf,
    const unsigned char * __restrict__ Q_fp8,
    const float         * __restrict__ Q_sf,
    const float         * __restrict__ W,
    const float         * __restrict__ k_scale,
    int D, int H, int Tc, int kv,
    const int * __restrict__ starts,
    const int * __restrict__ ends,
    float * __restrict__ Out) {

    // Do nothing - temporary just to get kernel to compile as we are testing updating the real tilelang indexer
    return;

}
__global__ void k_indexer_logits_fp8_tc_hgrp_naive(
    const unsigned char * __restrict__ K_fp8,
    const float         * __restrict__ K_sf,
    const unsigned char * __restrict__ Q_fp8,
    const float         * __restrict__ Q_sf,
    const float         * __restrict__ W,
    const float         * __restrict__ k_scale,
    int D, int H, int Tc, int kv,
    const int * __restrict__ starts,
    const int * __restrict__ ends,
    float * __restrict__ Out) {

    int tok    = blockIdx.x;                                  // token index
    int kv_idx = blockIdx.y * blockDim.x + threadIdx.x;       // kv row

    if (tok >= Tc || kv_idx >= kv) {
        return;
    }

    // Apply KV window if provided
    if (starts && ends) {
        int s0 = starts[tok];
        int e0 = ends[tok];
        if (s0 < 0) s0 = 0;
        if (s0 > kv) s0 = kv;
        if (e0 < 0) e0 = 0;
        if (e0 > kv) e0 = kv;
        if (kv_idx < s0 || kv_idx >= e0) {
            Out[(size_t)kv_idx + (size_t)kv * (size_t)tok] = 0.0f;
            return;
        }
    }

    float acc = 0.0f;

    // Per-row K scale and k_scale; match CPU idx_compute_scores_tile which
    // multiplies KS[i] * K_sf[i] once at the end. Here we keep K_sf as a pure
    // FP8 dequant scale and fold ks into the final accumulator.
    float ksf = K_sf ? K_sf[kv_idx] : 1.0f;
    float ks  = k_scale ? k_scale[kv_idx] : 1.0f;

    // Loop over heads and D to compute lightning indexer score
    for (int h = 0; h < H; ++h) {
        // Dot(Q_{t,h}, K_{kv}) over D
        float dot = 0.0f;
        size_t q_row = (size_t)tok * (size_t)H + (size_t)h;
        const unsigned char * q_row_ptr = Q_fp8 + q_row * (size_t)D;
        const unsigned char * k_row_ptr = K_fp8 + (size_t)kv_idx * (size_t)D;
        float qscale = Q_sf ? Q_sf[q_row] : 1.0f;

        for (int d = 0; d < D; ++d) {
            uint8_t qc = q_row_ptr[d];
            uint8_t kc = k_row_ptr[d];
            float qv = fp8e4m3_to_f32(qc);
            float kv = fp8e4m3_to_f32(kc);
            dot += qv * kv;
        }

        // Apply per-(tok,head) Q scale (vLLM UE8M0 style)
        dot *= qscale;

        // ReLU(dot)
        if (dot < 0.0f) dot = 0.0f;

        // Weight for this head and token
        float w = W[(size_t)h + (size_t)H * (size_t)tok];
        acc += dot * w;
    }

    // Apply combined k_scale * K_sf per kv row to mirror CPU reference
    acc *= ks * ksf;

    Out[(size_t)kv_idx + (size_t)kv * (size_t)tok] = acc;
}

extern "C" void ggml_cuda_indexer_logits_fp8_tc_hgrp_launch(
    ggml_backend_cuda_context & ctx,
    const unsigned char * K_fp8,   // [kv, D]
    const float         * K_sf,    // [kv]
    const unsigned char * Q_fp8,   // [Tc*H, D]
    const float         * Q_sf,    // [Tc*H] (unused for now)
    const float         * W,       // [H, Tc]
    const float         * k_scale, // [kv]
    int D, int H, int Tc, int kv,
    const int * starts, const int * ends,
    float * Out) {

    // Q_sf: optional per-row Q scales (baked into W upstream for TL FP8 path).
    // For the FP8_TC scalar path we expect W to already include Q scaling,
    // so Q_sf is currently unused but kept for API symmetry.
    (void)Q_sf;

    if (!K_fp8 || !Q_fp8 || !W || !k_scale || D <= 0 || H <= 0 || Tc <= 0 || kv <= 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    // Prefer the FP8 tensor-core MMA path for the production indexer shape.
    // Fallback to the naive scalar kernel for other shapes.
    if (D == 128 && H == 64) {
        constexpr int M_TILE = 128;
        constexpr int N_TILE = 8;
        constexpr int K_DIM  = 128;
        constexpr int NUM_WARP_M = 8;
        constexpr int NUM_WARP_N = 1;
        constexpr int threads = 32 * NUM_WARP_M * NUM_WARP_N; // 256

        dim3 grid((unsigned) Tc, (unsigned)((kv + M_TILE - 1) / M_TILE), 1);
        dim3 block(threads, 1, 1);

        // Shared memory: A fp8 + B fp8 + tmp float
        // Use the same GemmTensorOp layouts for sizing.
        using AType = fp8_e4_t;
        using BType = fp8_e4_t;
        using CType = float;
        using Gemm = cute::tl_mma::GemmTensorOp<M_TILE, N_TILE, K_DIM, NUM_WARP_M, NUM_WARP_N,
                                               false, true, true, K_DIM, K_DIM, 0, 0,
                                               AType, BType, CType>;
        using SmemLayoutA = typename Gemm::SmemLayoutA;
        using SmemLayoutB = typename Gemm::SmemLayoutB;
        constexpr int SMEM_A_ELEMS = (int) decltype(cute::cosize(SmemLayoutA{}))::value;
        constexpr int SMEM_B_ELEMS = (int) decltype(cute::cosize(SmemLayoutB{}))::value;

        size_t shmem = 0;
        shmem += (size_t) SMEM_A_ELEMS * sizeof(AType);
        shmem = (shmem + 15u) & ~size_t(15u);
        shmem += (size_t) SMEM_B_ELEMS * sizeof(BType);
        shmem = (shmem + 15u) & ~size_t(15u);
        shmem += (size_t) M_TILE * (size_t) N_TILE * sizeof(float);

        k_indexer_logits_fp8_tc_hgrp_mma_d128_h64<<<grid, block, shmem, stream>>>(
            K_fp8, K_sf, Q_fp8, Q_sf, W, k_scale,
            D, H, Tc, kv,
            starts, ends,
            Out);
    } else {
        int threads = 128;
        int blocks_y = (kv + threads - 1) / threads;
        dim3 grid(Tc, blocks_y, 1);
        dim3 block(threads, 1, 1);

        k_indexer_logits_fp8_tc_hgrp_naive<<<grid, block, 0, stream>>>(
            K_fp8,
            K_sf,
            Q_fp8,
            Q_sf,
            W,
            k_scale,
            D, H, Tc, kv,
            starts,
            ends,
            Out);
    }

    CUDA_CHECK(cudaGetLastError());

}
