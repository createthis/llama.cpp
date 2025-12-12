#ifndef LLAMA_SPARSE_TOPK_H
#define LLAMA_SPARSE_TOPK_H

#include <functional>
#include "ggml.h"
#include "ggml-backend.h"

namespace llama {

using std::function;

using indexer_fused_hook_t = ggml_tensor * (*)(
    ggml_context * ctx,
    ggml_tensor * q_tile2d,
    ggml_tensor * k_slice,
    ggml_tensor * w_slice,
    ggml_tensor * k_scale_head,
    ggml_tensor * k_indexer_fp8_sidecar,
    int64_t       t0,
    int64_t       Tc,
    int64_t       kv_start,
    int64_t       kv_end,
    int32_t       quant_bs,
    int32_t       cache_block_size,
    int32_t       cache_stride);

void set_indexer_fused_hook(indexer_fused_hook_t hook);

// Top-k selector implementation for DeepSeek V3.2
struct sparse_attn_topk {
    // Lightning Indexer KV-aware selection
    static ggml_tensor * select_topk_tokens_indexer_kvaware(
        ggml_context * ctx,
        ggml_tensor * q_indexer,   // [D_index, H_index, T]
        ggml_tensor * k_indexer,   // [D_index, N_kv]
        ggml_tensor * weights,     // [H_index, T]
        ggml_tensor * kq_mask,     // [N_kv, PAD(T)] or [N_kv, T]
        int64_t top_k,
        const function<void(ggml_tensor *, const char *, int)> & cb,
        ggml_cgraph * gf,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu,
        ggml_tensor * k_indexer_fp8_sidecar,
        int32_t quant_bs,
        int32_t cache_block_size,
        int32_t cache_stride);

    // new: compute top-k indices per column for a scores matrix [N, T]
    static ggml_tensor * topk_radix_indices(
        ggml_context * ctx,
        ggml_tensor * scores, // [N, T]
        int64_t k);
    // Windows helper: derive per-token [start,end) from mask or used_kv
    static ggml_tensor * derive_kv_windows(ggml_context * ctx, ggml_tensor * kq_mask, int64_t T, int64_t N_kv, ggml_tensor ** out_starts, ggml_tensor ** out_ends);
};

} // namespace llama

#endif // LLAMA_SPARSE_TOPK_H

