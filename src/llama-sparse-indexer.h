#ifndef LLAMA_SPARSE_INDEXER_H
#define LLAMA_SPARSE_INDEXER_H

#include <functional>
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "llama-kv-cache.h"

// Forward declarations
struct llama_model;

namespace llama {

using std::function;

// Triplet outputs for KV-aware Lightning Indexer
struct IndexerKVTriplet {
    ggml_tensor * q_indexer;
    ggml_tensor * k_indexer_cache;
    ggml_tensor * idx_weights;
};


// Lightning indexer helpers for DeepSeek V3.2
struct sparse_attn_indexer {
    static ggml_tensor * idx_compute_scores_tile(
        ggml_context * ctx,
        ggml_tensor * q3d,
        ggml_tensor * a_k,
        ggml_tensor * weights,
        ggml_tensor * k_scale_2d,
        int64_t D, int64_t H,
        int64_t Tc, int64_t kv_end,
        int64_t t0);

    // Build KV-aware top-k token indices using the Lightning Indexer tensors.
    // If mctx is nullptr, uses freshly computed K_indexer directly without cache writes.
    static IndexerKVTriplet compute_indexer_triplet(
        ggml_context * ctx,
        const llama_model & model,
        int layer_idx,
        ggml_tensor * cur,
        int64_t n_tokens,
        const llama_kv_cache_context * mctx,
        ggml_tensor * k_idxs,
        ggml_tensor * inp_pos,
        int64_t n_rot,
        int rope_type,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        const function<void(ggml_tensor *, const char *, int)> & cb,
        ggml_cgraph * gf);

    static ggml_tensor * build_kvaware_topk_indices(
        ggml_context * ctx,
        const llama_model & model,
        int layer_idx,
        ggml_tensor * cur,                  // [n_embd, T]
        int64_t n_tokens,
        const llama_kv_cache_context * mctx,
        ggml_tensor * k_idxs,
        ggml_tensor * kq_mask,
        int64_t top_k,
        ggml_tensor * inp_pos,
        int64_t n_rot,
        int rope_type,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        const function<void(ggml_tensor *, const char *, int)> & cb,
        ggml_cgraph * gf,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);
};

} // namespace llama

#endif // LLAMA_SPARSE_INDEXER_H
