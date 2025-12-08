#include "common.cuh"
#include <cuda_fp16.h>
// FP8 indexer K cache quantization for DeepSeek V3.2.
// Layout matches vLLM's DeepseekV32IndexerCache indexer_k_quant_and_cache:
//   kv_cache: [num_blocks, cache_block_size, cache_stride]
//   cache_stride = head_dim + (head_dim / quant_block_size) * 4 bytes (scales)
// Per block, FP8 values for all tokens (cache_block_size * head_dim bytes)
// are stored first, followed by per-quant-block FP32 scales.
namespace ggml_cuda_fp8_indexer {
static __device__ __forceinline__ uint8_t f32_to_fp8e4m3(float x) {
#if __CUDA_ARCH__ >= 900
    uint16_t tmp;
    float zero = 0.0f;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(zero), "f"(x));
    return static_cast<uint8_t>(tmp & 0xFFu);
#else
    // Fallback: clamp and scale into [-448,448] and use a simple linear quant.
    x = fmaxf(fminf(x, 448.0f), -448.0f);
    float s = x / 448.0f; // [-1,1]
    int q = __float2int_rn(s * 127.0f);
    q = max(-127, min(127, q));
    return static_cast<uint8_t>(q & 0xFF);
#endif
}
// Quantize K rows into FP8 cache with per-quant-block FP32 scales.
//   k:        [num_tokens, head_dim]
//   kv_cache: [num_blocks, cache_block_size, cache_stride]
//   slot_map: [num_tokens] -> global slot index in kv_cache
__global__ void k_indexer_fp8_quant_and_cache_kernel(
    const float * __restrict__ k,      // [num_tokens, head_dim]
    uint8_t * __restrict__ kv_cache,   // [num_blocks, cache_block_size, cache_stride]
    const int64_t * __restrict__ slot_map, // [num_tokens]
    int head_dim,
    int quant_bs,
    int cache_block_size,
    int cache_stride) {
    constexpr int VEC_SIZE = 4; // we process 4 floats (16B) at a time
    const int64_t token_idx = blockIdx.x;
    const int64_t head_dim_idx =
        (blockIdx.y * blockDim.y * blockDim.x +
         threadIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
    if (head_dim_idx >= head_dim) return;
    const int64_t slot = slot_map[token_idx];
    if (slot < 0) return; // padded token
    const int64_t block_idx    = slot / cache_block_size;
    const int64_t block_offset = slot % cache_block_size;
    // Load a vector of VEC_SIZE values from K
    const int64_t k_offset_vec = (token_idx * (int64_t) head_dim + head_dim_idx) / VEC_SIZE;
    float2 packed = reinterpret_cast<const float2*>(k)[k_offset_vec];
    float *vals = reinterpret_cast<float*>(&packed);
    // Compute local amax over this vector
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        amax = fmaxf(amax, fabsf(vals[i]));
    }
#if __CUDA_ARCH__ >= 700
    // Warp-wide reduction of amax within this quant block.
    for (int mask = 16; mask > 0; mask >>= 1) {
    #ifdef USE_ROCM
        amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
    #else
        amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
    #endif
    }
#endif
    float scale = fmaxf(amax, 1e-4f) / 448.0f;
    // Base offset of this block in kv_cache
    const int64_t block_base =
        block_idx * (int64_t) cache_block_size * cache_stride;
    // FP8 values region: [cache_block_size * head_dim] bytes per block
    const int64_t vals_base = block_base + block_offset * (int64_t) head_dim;
    const int64_t dst_offset = vals_base + head_dim_idx;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (head_dim_idx + i < head_dim) {
            float v = vals[i];
            float scaled = v / scale;
            uint8_t code = f32_to_fp8e4m3(scaled);
            kv_cache[dst_offset + i] = code;
        }
    }
    // Write FP32 scale for this quant block. The block index along head_dim is
    // (block_offset * head_dim + head_dim_idx) / quant_bs.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const int64_t block_linear = block_offset * (int64_t) head_dim + head_dim_idx;
        const int64_t scale_block_idx = block_linear / quant_bs;
        const int64_t scales_base = block_base + (int64_t) cache_block_size * head_dim;
        const int64_t scale_byte_offset = scales_base + scale_block_idx * 4; // 4 bytes per FP32 scale
        *reinterpret_cast<float*>(&kv_cache[scale_byte_offset]) = scale;
    }
}
} // namespace ggml_cuda_fp8_indexer
extern "C" void ggml_cuda_indexer_k_cache_fp8_quantize(
    ggml_backend_cuda_context & ctx,
    const float * dK,           // [num_tokens, head_dim]
    uint8_t * dKvCache,         // [num_blocks, cache_block_size, cache_stride]
    const int64_t * dSlotMap,   // [num_tokens]
    int num_tokens,
    int head_dim,
    int quant_bs,
    int cache_block_size,
    int cache_stride) {
    cudaStream_t stream = ctx.stream();
    constexpr int VEC_SIZE = 4;
    dim3 grid(num_tokens,
              (head_dim + quant_bs * VEC_SIZE - 1) / (quant_bs * VEC_SIZE));
    dim3 block(32, VEC_SIZE);
    ggml_cuda_fp8_indexer::k_indexer_fp8_quant_and_cache_kernel<<<grid, block, 0, stream>>>(
        dK, dKvCache, dSlotMap, head_dim, quant_bs, cache_block_size, cache_stride);
}
