#pragma once
#include "ggml-cuda.h"
#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare the CUDA context type; definition is in common.cuh
struct ggml_backend_cuda_context;


// DeepSeek V3.2: FP8 indexer K cache quantization helper (CUDA backend)
void ggml_cuda_indexer_k_cache_fp8_quantize(
    struct ggml_backend_cuda_context & ctx,
    const float * dK,           // [num_tokens, head_dim]
    unsigned char * dKvCache,   // [num_blocks, cache_block_size, cache_stride]
    const long long * dSlotMap, // [num_tokens]
    int num_tokens,
    int head_dim,
    int quant_bs,
    int cache_block_size,
    int cache_stride);

// Gather a contiguous KV range for one stream from the FP8 indexer K cache
// into a row-major FP8 K matrix plus per-quant-block FP32 scales.
//
// Inputs:
//   dKvCache         : [num_blocks, cache_block_size, cache_stride] (I8)
//   head_dim         : indexer head dimension D
//   quant_bs         : quantization block size used when writing the cache
//   cache_block_size : tokens per block (kv_size)
//   cache_stride     : bytes per token (D + scales_per_token*4)
//   stream_id        : which block/stream to gather from
//   kv_start, kv_len : slice [kv_start, kv_start+kv_len) within that stream
//
// Outputs:
//   dK_fp8_out  : [kv_len, head_dim] row-major FP8 codes
//   dScale_out  : [kv_len, head_dim/quant_bs] FP32 scales (one per quant block)
void ggml_cuda_indexer_k_cache_fp8_gather_wmma_hgrp(
    struct ggml_backend_cuda_context & ctx,
    const unsigned char * dKvCache,
    int head_dim,
    int quant_bs,
    int cache_block_size,
    int cache_stride,
    int stream_id,
    int kv_start,
    int kv_len,
    unsigned char * dK_fp8_out,
    float * dScale_out);

// Derive per-token KV window ends from device-resident mask [N_kv, T]
// mask values <= -1e29 are treated as masked; ends[t] = last i where mask[i,t] > -1e29, or 0 if none
void ggml_cuda_mask_window_ends_device(struct ggml_backend_cuda_context & ctx,
                                       const float * dMask, int N_kv, int T,
                                       int * dEnds);

// Device-resident entry: takes device pointers and current CUDA context
void ggml_cuda_indexer_logits_fused_device(struct ggml_backend_cuda_context & ctx,
                                           const float * dQ,
                                           const float * dK,
                                           const float * dW,
                                           const float * dKS,
                                           const int * dStarts, const int * dEnds,
                                           int D, int H, int Tc, int kv_end,
                                           float * dOut);

// Derive per-token KV window ends from device-resident mask and copy to host buffer
void ggml_cuda_mask_window_ends_device_to_host(struct ggml_backend_cuda_context & ctx,
                                               const float * dMask, int N_kv, int T, int * hEnds);

// Simple convenience wrappers using current device and default stream
void ggml_cuda_mask_window_ends_device_to_host_simple(const float * dMask, int N_kv, int T, int * hEnds);
void ggml_cuda_mask_window_starts_device_to_host_simple(const float * dMask, int N_kv, int T, int * hStarts);

#ifdef __cplusplus
}
#endif
