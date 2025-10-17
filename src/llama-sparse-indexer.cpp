#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>

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
    
    // Get n_tokens from the weights tensor (moved from later in the function)
    const int64_t n_tokens = ggml_nelements(weights) / index_n_heads;
    
    // Reshape q_indexer to [index_n_heads, index_head_dim, n_tokens]
    // q_indexer should have shape [index_n_heads * index_head_dim, n_tokens] from matrix multiplication
    q_indexer = ggml_reshape_3d(ctx, q_indexer, index_n_heads, index_head_dim, n_tokens);
    cb(q_indexer, "indexer_q_reshape", layer_idx);

    // Reshape k_indexer to [index_head_dim * index_n_heads, n_tokens]
    // k_indexer should have shape [index_head_dim * index_n_heads, n_tokens] from matrix multiplication
    k_indexer = ggml_reshape_2d(ctx, k_indexer, index_head_dim * index_n_heads, n_tokens);
    cb(k_indexer, "indexer_k_reshape", layer_idx);

    // Compute the dot product: q_indexer [index_n_heads, index_head_dim, n_tokens] · k_indexer [index_head_dim * index_n_heads, n_tokens]
    // We need to reshape k_indexer to match the structure of q_indexer
    
    // Reshape k_indexer to [index_head_dim, index_n_heads, n_tokens]
    k_indexer = ggml_reshape_3d(ctx, k_indexer, index_head_dim, index_n_heads, n_tokens);
    cb(k_indexer, "indexer_k_3d", layer_idx);
    
    // Permute k_indexer to [index_n_heads, index_head_dim, n_tokens] to match q_indexer structure
    ggml_tensor * k_permuted = ggml_cont(ctx, ggml_permute(ctx, k_indexer, 1, 0, 2, 3));
    cb(k_permuted, "indexer_k_permuted", layer_idx);
    
    // Now both q_indexer and k_permuted have shape [index_n_heads, index_head_dim, n_tokens]
    // We need to compute the dot product for each head and token pair
    
    // Reshape q_indexer to [index_head_dim, index_n_heads * n_tokens]
    ggml_tensor * q_reshaped = ggml_reshape_2d(ctx, q_indexer, index_head_dim, index_n_heads * n_tokens);
    cb(q_reshaped, "indexer_q_reshaped", layer_idx);
    
    // Reshape k_permuted to [index_head_dim, index_n_heads * n_tokens]
    ggml_tensor * k_reshaped = ggml_reshape_2d(ctx, k_permuted, index_head_dim, index_n_heads * n_tokens);
    cb(k_reshaped, "indexer_k_reshaped", layer_idx);
    
    // Compute dot product: result will be [index_n_heads * n_tokens, index_n_heads * n_tokens]
    ggml_tensor * dot_product = ggml_mul_mat(ctx, q_reshaped, k_reshaped);
    cb(dot_product, "indexer_dot_product", layer_idx);
    
    // Apply ReLU activation
    ggml_tensor * relu_scores = ggml_relu(ctx, dot_product);
    cb(relu_scores, "indexer_relu_scores", layer_idx);
    
    // Reshape relu_scores to [index_n_heads, n_tokens, index_n_heads, n_tokens]
    ggml_tensor * relu_4d = ggml_reshape_4d(ctx, relu_scores, n_tokens, index_n_heads, n_tokens, index_n_heads);
    cb(relu_4d, "indexer_relu_4d", layer_idx);
    
    // Permute to [index_n_heads, n_tokens, index_n_heads, n_tokens] for weight multiplication
    ggml_tensor * relu_permuted = ggml_cont(ctx, ggml_permute(ctx, relu_4d, 1, 0, 3, 2));
    cb(relu_permuted, "indexer_relu_permuted", layer_idx);
    
    // Reshape weights to [index_n_heads, n_tokens, 1] for broadcasting
    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, n_tokens, index_n_heads);
    weights_3d = ggml_cont(ctx, ggml_permute(ctx, weights_3d, 2, 1, 0, 3)); // [index_n_heads, n_tokens, 1]
    cb(weights_3d, "indexer_weights_3d", layer_idx);
    
    // Multiply ReLU scores by weights: w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    // We need to broadcast weights across the appropriate dimensions
    ggml_tensor * weighted_scores = ggml_mul(ctx, relu_permuted, weights_3d);
    cb(weighted_scores, "indexer_weighted_scores", layer_idx);
    
    // Sum across heads to get final importance scores: sum_{j=1}^{H^I} and sum over the other head dimension
    // weighted_scores has shape [index_n_heads, n_tokens, index_n_heads, n_tokens]
    // We need to sum over both head dimensions
    
    // Reshape to [index_n_heads * index_n_heads, n_tokens * n_tokens] for reduction
    ggml_tensor * weighted_2d = ggml_reshape_2d(ctx, weighted_scores, index_n_heads * index_n_heads, n_tokens * n_tokens);
    cb(weighted_2d, "weighted_2d", layer_idx);
    
    // Sum along rows (head dimensions) to get [1, n_tokens * n_tokens]
    ggml_tensor * token_importance_sum = ggml_sum_rows(ctx, weighted_2d);
    cb(token_importance_sum, "token_importance_sum", layer_idx);
    
    // Reshape to [n_tokens, n_tokens] for compatibility with top-k selection
    if (ggml_nelements(token_importance_sum) != n_tokens * n_tokens) {
        printf("Error: Cannot reshape token_importance_sum to [n_tokens, n_tokens]\n");
        printf("Expected %" PRId64 " elements, but have %" PRId64 " elements\n",
               n_tokens * n_tokens, ggml_nelements(token_importance_sum));
        return nullptr;
    }
    
    ggml_tensor * token_importance = ggml_reshape_2d(ctx, token_importance_sum, n_tokens, n_tokens);
    if (token_importance) {
        cb(token_importance, "token_importance_final", layer_idx);
    } else {
        printf("Error: token_importance is null after reshape\n");
        return nullptr;
    }

    return token_importance;
}

} // namespace llama