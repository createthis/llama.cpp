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
