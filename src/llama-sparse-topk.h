#ifndef LLAMA_SPARSE_TOPK_H
#define LLAMA_SPARSE_TOPK_H

#include <functional>
#include "ggml.h"
#include "ggml-backend.h"

namespace llama {

using std::function;

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
        ggml_backend_t backend_cpu);

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

