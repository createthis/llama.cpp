#ifndef LLAMA_SPARSE_MLA_FWD_H
#define LLAMA_SPARSE_MLA_FWD_H

#include <functional>
#include "ggml-cpp.h"

// Forward declarations
struct llm_graph_params;

namespace llama {

using std::function;

// Sparse Multi-Query Attention Forward implementation for DeepSeek V3.2
// Corresponds to tilelang's sparse_mla_fwd.py
struct sparse_mla_fwd {
    // Apply sparse attention using top-k indices
    static ggml_tensor * apply_sparse_attention(
        ggml_context * ctx,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * topk_indices,
        int64_t n_tokens,
        int64_t top_k,
        const function<void(ggml_tensor *, const char *, int)> & cb);

    // KV-aware variant: gather from full KV cache tensors instead of current block
    static ggml_tensor * apply_sparse_attention_kvaware(
        ggml_context * ctx,
        ggml_tensor * q_cur,     // [Dq, Hq, T]
        ggml_tensor * k_cache,   // [Dk, Hkv, N_kv]
        ggml_tensor * v_cache,   // [Dv, Hkv, N_kv]
        ggml_tensor * topk_indices, // [top_k, T]
        int64_t n_tokens,
        int64_t top_k,
        const function<void(ggml_tensor *, const char *, int)> & cb);
};


} // namespace llama

#endif // LLAMA_SPARSE_MLA_FWD_H
