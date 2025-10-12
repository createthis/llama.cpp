#ifndef LLAMA_SPARSE_ATTN_H
#define LLAMA_SPARSE_ATTN_H

#include "ggml-cpp.h"

// Forward declarations
struct llama_model;
struct llm_graph_params;

namespace llama {

// Sparse attention indexer implementation for DeepSeek V3.2
struct sparse_attn_indexer {
    // Compute token importance scores using the lightning indexer
    static ggml_tensor * compute_token_importance(
        ggml_context * ctx,
        const llama_model & model,
        int layer_idx,
        ggml_tensor * cur,
        bool is_lite,
        const std::function<void(ggml_tensor *, const char *, int)> & cb);

    // Identify top-k important tokens for sparse attention
    static ggml_tensor * select_topk_tokens(
        ggml_context * ctx,
        ggml_tensor * token_importance,
        int64_t n_tokens,
        const std::function<void(ggml_tensor *, const char *, int)> & cb);

    // Apply sparse attention using top-k indices
    static ggml_tensor * apply_sparse_attention(
        ggml_context * ctx,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * topk_indices,
        int64_t n_tokens,
        int64_t top_k,
        const std::function<void(ggml_tensor *, const char *, int)> & cb);


};

} // namespace llama

#endif // LLAMA_SPARSE_ATTN_H