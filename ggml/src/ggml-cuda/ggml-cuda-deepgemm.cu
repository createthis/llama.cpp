#include "common.cuh"

// This TU provides a single entry point that wraps the DeepGEMM
// FP8 paged MQA logits kernel and exposes it via a C-style API
// consumable from indexer-fused.cu without pulling in all the
// heavy CuTe / DeepGEMM headers there.

#include <cuda_runtime.h>
#include <cuda.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

extern "C" void ggml_deepgemm_paged_mqa_logits_sm90(
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
    size_t shmem_bytes) {

    dim3 grid(1, 1, 1);
    dim3 block(num_specialized_threads + num_math_threads, 1, 1);

    // We currently only support the specific configuration used by
    // the DeepSeek V3.2-Exp indexer path (kNextN=2, H=64, D=64/128,
    // block_kv=64, num_q_stages=num_kv_stages=3, split_kv=block_kv).
    // Assert these invariants here so the wrapper fails loudly if
    // called with unsupported shapes.
    if (next_n != 2 || num_heads != 64 || block_kv != 64 ||
        num_q_stages != 3 || num_kv_stages != 3) {
        // fall back to cudaErrorNotSupported via an early return
        // (the caller should gate use of this path based on shapes).
        return;
    }

    constexpr uint32_t kNextN     = 2;
    constexpr uint32_t kNumHeads  = 64;
    constexpr uint32_t kNumQStages  = 3;
    constexpr uint32_t kNumKVStages = 3;
    constexpr uint32_t kNumSpecializedThreads = 128;
    constexpr uint32_t kNumMathThreadsConst   = 128;

    // Guard against mismatched runtime parameters
    if (num_specialized_threads != kNumSpecializedThreads ||
        num_math_threads       != kNumMathThreadsConst) {
        return;
    }

    // Choose the template instantiation based on head_dim to avoid
    // forcing indexer-fused.cu to do it.
    if (head_dim == 64 || head_dim == 128) {
        using Kernel = void (*)(
            const uint32_t, const uint64_t, const uint64_t,
            const uint32_t*, float*, const uint32_t*, const uint32_t*,
            cute::TmaDescriptor, cute::TmaDescriptor, cute::TmaDescriptor, cute::TmaDescriptor);

        Kernel kernel = nullptr;

        if (head_dim == 64) {
            kernel = sm90_fp8_paged_mqa_logits<
                kNextN, kNumHeads,
                64u, 64u,
                false,
                kNumQStages, kNumKVStages,
                64u,
                kNumSpecializedThreads, kNumMathThreadsConst>;
        } else { // head_dim == 128
            kernel = sm90_fp8_paged_mqa_logits<
                kNextN, kNumHeads,
                128u, 64u,
                false,
                kNumQStages, kNumKVStages,
                64u,
                kNumSpecializedThreads, kNumMathThreadsConst>;
        }

        // Configure dynamic shared memory
        cudaFuncSetAttribute(
            (const void*) kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int) shmem_bytes);

        kernel<<<grid, block, shmem_bytes, stream>>>(
            batch_size,
            logits_stride,
            block_table_stride,
            context_lens,
            logits,
            block_table,
            schedule_meta,
            tma_q,
            tma_kv,
            tma_kv_scales,
            tma_w);
    }
}
