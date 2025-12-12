#ifndef LLAMA_SPARSE_MLA_FWD_H
#define LLAMA_SPARSE_MLA_FWD_H

#include <functional>
#include "ggml-cpp.h"

// Forward declarations
struct llm_graph_params;

namespace llama {

using std::function;

using sparse_mla_fused_hook_t = ggml_tensor * (*) (
    ggml_context * ctx,
    ggml_tensor * out2d,
    ggml_tensor * q2d,
    ggml_tensor * k_cache,
    ggml_tensor * v_cache,
    ggml_tensor * v_gather_src,
    ggml_tensor * idx1d,
    float kq_scale,
    float softcap,
    ggml_tensor * kv_dsmla_blob);

void set_sparse_mla_fused_hook(sparse_mla_fused_hook_t hook);

// Sparse Multi-Query Attention Forward implementation for DeepSeek V3.2
// Corresponds to tilelang's sparse_mla_fwd.py
struct sparse_mla_fwd {
    // KV-aware variant: gather from full KV cache tensors instead of current block
    static ggml_tensor * apply_sparse_attention_kvaware(
        ggml_context * ctx,
        ggml_tensor * q_cur,     // [Dq, Hq, T]
        ggml_tensor * k_cache,   // [Dk, Hkv, N_kv]
        ggml_tensor * v_cache,   // [Dv, Hkv, N_kv]
        ggml_tensor * topk_indices, // [top_k, T]
        int64_t n_tokens,
        int64_t top_k,
        float kq_scale,
        ggml_tensor * kq_mask,
        float attn_softcap,
        ggml_tensor * kv_dsmla_blob,
        const function<void(ggml_tensor *, const char *, int)> & cb);
};


} // namespace llama

#endif // LLAMA_SPARSE_MLA_FWD_H
