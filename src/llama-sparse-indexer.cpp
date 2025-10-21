#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"

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

// Per-query lightning indexer: returns [n_tokens, n_tokens] logits matrix
// I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)

ggml_tensor * sparse_attn_indexer::compute_token_importance(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    bool is_lite,
    const function<void(ggml_tensor *, const char *, int)> & cb) {

    printf("=== SPARSE INDEXER: Starting compute_token_importance for layer %d ===\n", layer_idx);
    size_t initial_mem = ggml_used_mem(ctx);
    printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
    printf("Input tensor cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    printf("Input tensor cur total elements: %" PRId64 "\n", ggml_nelements(cur));
    fflush(stdout);

    const int64_t n_tokens = cur->ne[1];

    // 1) Build query representation for all tokens and normalize
    ggml_tensor * qr = nullptr;
    if (!is_lite) {
        // [q_lora_rank, n_tokens]
        qr = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        cb(qr, "indexer_qr_all", layer_idx);
        qr = ggml_norm(ctx, qr, 1e-5f);
        cb(qr, "indexer_qr_all_norm", layer_idx);
    } else {
        qr = ggml_norm(ctx, cur, 1e-5f);
        cb(qr, "indexer_qr_all_norm_lite", layer_idx);
    }

    // 2) Key projection and RMSNorm (with weight and optional bias)
    ggml_tensor * k_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur); // [index_head_dim, n_tokens]
    cb(k_indexer, "indexer_k", layer_idx);
    k_indexer = ggml_norm(ctx, k_indexer, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm != nullptr) {
        ggml_tensor * w = model.layers[layer_idx].attn_indexer_k_norm; // [index_head_dim]
        ggml_tensor * w_b = ggml_repeat(ctx, w, k_indexer);
        k_indexer = ggml_mul(ctx, k_indexer, w_b);
    }
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        ggml_tensor * b = model.layers[layer_idx].attn_indexer_k_norm_bias; // [index_head_dim]
        ggml_tensor * b_b = ggml_repeat(ctx, b, k_indexer);
        k_indexer = ggml_add(ctx, k_indexer, b_b);
    }
    cb(k_indexer, "indexer_k_norm", layer_idx);

    // 3) Per-token head weights: [H, T]
    ggml_tensor * weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur);
    cb(weights, "indexer_weights", layer_idx);

    // 4) Project queries into indexer space for all heads: [D*H, T]
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qr);
    cb(q_indexer, "indexer_q", layer_idx);

    // DeepSeek V3.2 indexer dims: derive from tensors to avoid config drift
    const int64_t index_head_dim = model.layers[layer_idx].attn_indexer_wk->ne[1];
    const int64_t index_n_heads  = model.layers[layer_idx].attn_indexer_weights_proj->ne[1];
    const int64_t wq_b_cols      = model.layers[layer_idx].attn_indexer_wq_b->ne[1];
    // Basic consistency check: attn_indexer_wq_b should have D_index * H_index columns
    GGML_ASSERT(wq_b_cols == index_head_dim * index_n_heads);
    printf("[DSA][L%02d] Indexer dims: D_index=%lld H_index=%lld wq_b_cols=%lld\n",
           layer_idx,
           (long long) index_head_dim,
           (long long) index_n_heads,
           (long long) wq_b_cols);
    fflush(stdout);

    // Reshape K to [D, T]
    k_indexer = ggml_reshape_2d(ctx, k_indexer, index_head_dim, n_tokens);
    cb(k_indexer, "indexer_k_reshape", layer_idx);

    // Reshape Q to [D, H, T] then permute to [D, T, H] and tile heads into columns: [D, T*H]
    ggml_tensor * q3 = ggml_reshape_3d(ctx, q_indexer, index_head_dim, index_n_heads, n_tokens);
    cb(q3, "indexer_q_3d", layer_idx);
    ggml_tensor * q_perm = ggml_permute(ctx, q3, 0, 2, 1, 3);
    cb(q_perm, "indexer_q_perm_D_T_H", layer_idx);
    ggml_tensor * q_perm_cont = ggml_cont(ctx, q_perm);
    ggml_tensor * q_tiled = ggml_reshape_2d(ctx, q_perm_cont, index_head_dim, n_tokens * index_n_heads);
    cb(q_tiled, "indexer_q_tiled", layer_idx);

    // 5) Compute logits for all heads in one matmul: K^T @ Q_tiled -> [T, T*H]
    printf("[DSA][L%02d] Matmul shapes: K=[%lld,%lld] Q_tiled=[%lld,%lld]\n",
           layer_idx,
           (long long) k_indexer->ne[0], (long long) k_indexer->ne[1],
           (long long) q_tiled->ne[0],   (long long) q_tiled->ne[1]);
    fflush(stdout);
    ggml_tensor * logits_concat = ggml_mul_mat(ctx, k_indexer, q_tiled);
    cb(logits_concat, "indexer_logits_concat_T_TH", layer_idx);
    logits_concat = ggml_relu(ctx, logits_concat);
    cb(logits_concat, "indexer_logits_relu", layer_idx);

    // Ensure contiguity before reshape
    logits_concat = ggml_cont(ctx, logits_concat);

    // 6) Apply per-token head weights: reshape to [T, H, T]
    ggml_tensor * logits_3d = ggml_reshape_3d(ctx, logits_concat, n_tokens, index_n_heads, n_tokens);
    ggml_tensor * w_T_H = ggml_transpose(ctx, weights);                 // [T, H]
    ggml_tensor * w_T_H_cont = ggml_cont(ctx, w_T_H);
    ggml_tensor * w_T_H_1 = ggml_reshape_3d(ctx, w_T_H_cont, n_tokens, index_n_heads, 1);
    ggml_tensor * w_bcast = ggml_repeat(ctx, w_T_H_1, logits_3d);       // [T, H, T]
    ggml_tensor * weighted = ggml_mul(ctx, logits_3d, w_bcast);
    cb(weighted, "indexer_logits_weighted_T_H_T", layer_idx);
    printf("[DSA][L%02d] logits_3d=[%lld,%lld,%lld] weights=[%lld,%lld]\n",
           layer_idx,
           (long long) logits_3d->ne[0], (long long) logits_3d->ne[1], (long long) logits_3d->ne[2],
           (long long) weights->ne[0], (long long) weights->ne[1]);
    fflush(stdout);

    // 7) Sum across heads -> [1, T, T] -> reshape to [T, T]
    ggml_tensor * weighted_perm = ggml_permute(ctx, weighted, 1, 0, 2, 3);
    // Ensure contiguous layout before sum_rows to satisfy nb[0] == sizeof(float)
    weighted_perm = ggml_cont(ctx, weighted_perm);
    ggml_tensor * summed = ggml_sum_rows(ctx, weighted_perm);
    cb(summed, "indexer_logits_summed_1_T_T", layer_idx);
    ggml_tensor * token_importance = ggml_reshape_2d(ctx, summed, n_tokens, n_tokens);
    cb(token_importance, "token_importance_final_T_T", layer_idx);

    printf("SPARSE INDEXER: Final token_importance [T,T]=[%" PRId64 ", %" PRId64 "]\n",
           token_importance->ne[0], token_importance->ne[1]);
    printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(),
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);

    return token_importance;
}

} // namespace llama

