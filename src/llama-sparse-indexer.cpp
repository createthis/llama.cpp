#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "llama-sparse-topk.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <string>

// Helper function to get memory usage in human-readable format
static std::string format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    size_t unit_idx = 0;
    double size = bytes;

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_idx]);
    return std::string(buffer);
}

namespace llama {

using std::function;


IndexerKVTriplet sparse_attn_indexer::compute_indexer_triplet(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    int64_t n_tokens,
    const llama_kv_cache_context * mctx,
    ggml_tensor * k_idxs,
    const function<void(ggml_tensor *, const char *, int)> & cb,
    ggml_cgraph * gf) {
    // Compute Indexer K for current tokens and (optionally) write to cache
    ggml_tensor * Kindexer_cur = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    Kindexer_cur = ggml_norm(ctx, Kindexer_cur, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm != nullptr) {
        ggml_tensor * gamma = model.layers[layer_idx].attn_indexer_k_norm;
        ggml_tensor * gamma_r = ggml_repeat(ctx, gamma, Kindexer_cur);
        Kindexer_cur = ggml_mul(ctx, Kindexer_cur, gamma_r);
    }
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        ggml_tensor * beta = model.layers[layer_idx].attn_indexer_k_norm_bias;
        ggml_tensor * beta_r = ggml_repeat(ctx, beta, Kindexer_cur);
        Kindexer_cur = ggml_add(ctx, Kindexer_cur, beta_r);
    }
    cb(Kindexer_cur, "indexer_k_norm", layer_idx);
    if (mctx && gf) {
        ggml_tensor * Kindexer_cur_3d = ggml_reshape_3d(ctx, Kindexer_cur, Kindexer_cur->ne[0], 1, n_tokens);
        ggml_build_forward_expand(gf, mctx->cpy_k_indexer(ctx, Kindexer_cur_3d, k_idxs, layer_idx));
    }
    // Build q_indexer and weights
    ggml_tensor * qsrc = nullptr;
    if (model.layers[layer_idx].wq_a != nullptr) {
        qsrc = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        qsrc = ggml_norm(ctx, qsrc, 1e-5f);
    } else {
        qsrc = ggml_norm(ctx, cur, 1e-5f);
    }
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qsrc);
    // index head dim (head_dim in Tilelang)
    const int64_t D_index = model.layers[layer_idx].attn_indexer_wk->ne[1];
    // indexer head count (n_heads in Tilelang)
    const int64_t H_index = model.layers[layer_idx].attn_indexer_wq_b->ne[1] / D_index;
    q_indexer = ggml_reshape_3d(ctx, q_indexer, D_index, H_index, n_tokens);
    cb(q_indexer, "indexer_q", layer_idx);
    ggml_tensor * idx_weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur);
    // Scale weights by 1/sqrt(H_index) to match TileLang indexer behavior
    // see https://github.com/tile-ai/tilelang/blob/5cb5c068bc9a1a0b38c46bac915a8c2743eb1442/examples/deepseek_v32/inference/model.py#L500
    idx_weights = ggml_scale(ctx, idx_weights, 1.0f / sqrtf((float) H_index));
    cb(idx_weights, "indexer_weights", layer_idx);
    ggml_tensor * Kindexer_cache = mctx ? mctx->get_k_indexer(ctx, layer_idx)
                                        : ggml_reshape_2d(ctx, Kindexer_cur, D_index, n_tokens);
    IndexerKVTriplet out{ q_indexer, Kindexer_cache, idx_weights };
    return out;
}

ggml_tensor * sparse_attn_indexer::build_kvaware_topk_indices(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    int64_t n_tokens,
    const llama_kv_cache_context * mctx,
    ggml_tensor * k_idxs,
    ggml_tensor * kq_mask,
    int64_t top_k,
    const function<void(ggml_tensor *, const char *, int)> & cb,
    ggml_cgraph * gf)
{
    printf("=== SPARSE INDEXER: build_kvaware_topk_indices L%d ===\n", layer_idx);
    size_t initial_mem = ggml_used_mem(ctx);
    printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
    fflush(stdout);
    IndexerKVTriplet trip = compute_indexer_triplet(ctx, model, layer_idx, cur, n_tokens, mctx, k_idxs, cb, gf);
    ggml_tensor * Kindexer_cache = trip.k_indexer_cache;
    if (top_k <= 0) {
        top_k = std::max<int64_t>(64, std::min<int64_t>(1024, Kindexer_cache->ne[1]));
    }
    ggml_tensor * kvaware_indices = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx, trip.q_indexer, Kindexer_cache, trip.idx_weights, kq_mask, top_k, cb);
    printf("SPARSE INDEXER: Final topk_indices [k,T]=[%" PRId64 ", %" PRId64 "]\n",
           kvaware_indices->ne[0], kvaware_indices->ne[1]);
    printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(),
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);
    return kvaware_indices;
}


} // namespace llama

