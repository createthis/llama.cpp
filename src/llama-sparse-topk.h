#ifndef LLAMA_SPARSE_TOPK_H
#define LLAMA_SPARSE_TOPK_H

#include <functional>
#include "ggml.h"

namespace llama {

using std::function;

// Top-k selector implementation for DeepSeek V3.2
struct sparse_attn_topk {
    // Identify top-k important tokens for sparse attention
    static ggml_tensor * select_topk_tokens(
        ggml_context * ctx,
        ggml_tensor * token_importance,
        int64_t n_tokens,
        const function<void(ggml_tensor *, const char *, int)> & cb);

    // KV-aware top-k selection using attention Q and cached K, with mask.
    // Returns absolute KV indices of shape [top_k, n_tokens].
    static ggml_tensor * select_topk_tokens_kvaware(
        ggml_context * ctx,
        ggml_tensor * q_cur,     // [Dq, Hq, T]
        ggml_tensor * k_cache,   // [Dk, Hkv, N_kv]
        ggml_tensor * kq_mask,   // [N_kv, PAD(T)] or broadcastable
        int64_t top_k,
        const function<void(ggml_tensor *, const char *, int)> & cb);
};

} // namespace llama

#endif // LLAMA_SPARSE_TOPK_H