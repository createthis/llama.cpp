#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>

namespace llama {

using std::function;

ggml_tensor * sparse_attn_indexer::compute_token_importance(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    bool is_lite,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // DeepSeek V3.2 Lightning Indexer implementation
    // Mathematical formula: I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    
    // Indexer query projection (wq_a) - for non-lite version
    ggml_tensor * qr = nullptr;
    if (!is_lite) {
        // First projection: cur -> wq_a -> qr
        qr = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        cb(qr, "indexer_qr", layer_idx);

        // Normalize the query representation
        qr = ggml_norm(ctx, qr, 1e-5f);
        cb(qr, "indexer_qr_norm", layer_idx);
    } else {
        // For lite version, use the current hidden state directly
        qr = cur;
    }

    // Indexer key projection (wk) - k^I_s from the formula
    // This projects the current hidden state to the indexer key space
    ggml_tensor * k_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    cb(k_indexer, "indexer_k", layer_idx);

    // Indexer key normalization (k_norm)
    k_indexer = ggml_norm(ctx, k_indexer, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        k_indexer = ggml_add(ctx, k_indexer, model.layers[layer_idx].attn_indexer_k_norm_bias);
    }
    cb(k_indexer, "indexer_k_norm", layer_idx);

    // Indexer weights projection - w^I_{t,j} from the formula
    // These are the per-head weights for the indexer
    ggml_tensor * weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur);
    cb(weights, "indexer_weights", layer_idx);

    // Indexer query projection (wq_b) - q^I_{t,j} from the formula
    // This projects the normalized query representation to the indexer query space
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qr);
    cb(q_indexer, "indexer_q", layer_idx);

    // From DeepSeek V3.2 config: index_n_heads = 64, index_head_dim = 128
    const int64_t index_n_heads = 64;
    const int64_t index_head_dim = 128;
    
    // Reshape q_indexer to [n_tokens, index_n_heads, index_head_dim]
    // q_indexer should have shape [n_tokens, index_n_heads * index_head_dim]
    q_indexer = ggml_reshape_3d(ctx, q_indexer, index_head_dim, index_n_heads, ggml_nelements(q_indexer) / (index_head_dim * index_n_heads));
    cb(q_indexer, "indexer_q_reshape", layer_idx);

    // Reshape k_indexer to [n_tokens, index_head_dim] for compatibility
    // k_indexer should have shape [n_tokens, index_head_dim * index_n_heads]
    k_indexer = ggml_reshape_2d(ctx, k_indexer, index_head_dim * index_n_heads, ggml_nelements(k_indexer) / (index_head_dim * index_n_heads));
    cb(k_indexer, "indexer_k_reshape", layer_idx);

    // Compute the dot product between q_indexer and k_indexer
    // q_indexer: [n_tokens, index_n_heads, index_head_dim]
    // k_indexer: [n_tokens, index_head_dim * index_n_heads]
    
    // First, reshape q_indexer to [n_tokens * index_n_heads, index_head_dim]
    ggml_tensor * q_flat = ggml_reshape_2d(ctx, q_indexer, index_head_dim, ggml_nelements(q_indexer) / index_head_dim);
    cb(q_flat, "indexer_q_flat", layer_idx);

    // Compute dot product: q_flat @ k_indexer^T
    // q_flat: [n_tokens * index_n_heads, index_head_dim]
    // k_indexer: [n_tokens, index_head_dim * index_n_heads]
    
    // Transpose k_indexer to [index_head_dim * index_n_heads, n_tokens]
    ggml_tensor * k_transposed = ggml_cont(ctx, ggml_transpose(ctx, k_indexer));
    cb(k_transposed, "indexer_k_transposed", layer_idx);

    // Now compute the dot product
    ggml_tensor * dot_product = ggml_mul_mat(ctx, q_flat, k_transposed);
    cb(dot_product, "indexer_dot_product", layer_idx);

    // Apply ReLU activation as per the formula
    ggml_tensor * relu_scores = ggml_relu(ctx, dot_product);
    cb(relu_scores, "indexer_relu_scores", layer_idx);

    // Reshape relu_scores to [n_tokens, index_n_heads, n_tokens]
    const int64_t n_tokens = ggml_nelements(weights) / index_n_heads;
    ggml_tensor * relu_3d = ggml_reshape_3d(ctx, relu_scores, n_tokens, index_n_heads, n_tokens);
    cb(relu_3d, "indexer_relu_3d", layer_idx);

    // Permute to [index_n_heads, n_tokens, n_tokens]
    ggml_tensor * relu_permuted = ggml_cont(ctx, ggml_permute(ctx, relu_3d, 1, 0, 2, 3));
    cb(relu_permuted, "indexer_relu_permuted", layer_idx);

    // Reshape weights to [index_n_heads, n_tokens, 1] for broadcasting
    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, index_n_heads, n_tokens);
    weights_3d = ggml_cont(ctx, ggml_permute(ctx, weights_3d, 1, 2, 0, 3)); // [index_n_heads, n_tokens, 1]
    cb(weights_3d, "indexer_weights_3d", layer_idx);

    // Multiply ReLU scores by weights: w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    // Broadcast weights across the n_tokens dimension
    ggml_tensor * weighted_scores = ggml_mul(ctx, relu_permuted, weights_3d);
    cb(weighted_scores, "indexer_weighted_scores", layer_idx);

    // Sum across heads to get final importance scores: sum_{j=1}^{H^I}
    ggml_tensor * token_importance = ggml_sum(ctx, weighted_scores);
    cb(token_importance, "token_importance", layer_idx);

    // Reshape to [n_tokens, n_tokens] for compatibility with top-k selection
    token_importance = ggml_reshape_2d(ctx, token_importance, n_tokens, n_tokens);
    cb(token_importance, "token_importance_final", layer_idx);

    return token_importance;
}

} // namespace llama