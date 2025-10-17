#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>

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
    
    printf("=== SPARSE INDEXER: Starting compute_token_importance for layer %d ===\n", layer_idx);
    printf("Input tensor cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    printf("Input tensor cur total elements: %" PRId64 "\n", ggml_nelements(cur));
    fflush(stdout);
    
    // Indexer query projection (wq_a) - for non-lite version
    ggml_tensor * qr = nullptr;
    if (!is_lite) {
        printf("SPARSE INDEXER: Using non-lite version\n");
        fflush(stdout);
        
        // First projection: cur -> wq_a -> qr
        qr = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        cb(qr, "indexer_qr", layer_idx);
        printf("SPARSE INDEXER: After wq_a projection, qr shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               qr->ne[0], qr->ne[1], qr->ne[2], qr->ne[3]);
        fflush(stdout);

        // Normalize the query representation
        qr = ggml_norm(ctx, qr, 1e-5f);
        cb(qr, "indexer_qr_norm", layer_idx);
        printf("SPARSE INDEXER: After normalization, qr shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               qr->ne[0], qr->ne[1], qr->ne[2], qr->ne[3]);
        fflush(stdout);
    } else {
        printf("SPARSE INDEXER: Using lite version\n");
        fflush(stdout);
        // For lite version, use the current hidden state directly
        qr = cur;
    }

    // Indexer key projection (wk) - k^I_s from the formula
    // This projects the current hidden state to the indexer key space
    ggml_tensor * k_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    cb(k_indexer, "indexer_k", layer_idx);
    printf("SPARSE INDEXER: After wk projection, k_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
    fflush(stdout);

    // Indexer key normalization (k_norm)
    k_indexer = ggml_norm(ctx, k_indexer, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        k_indexer = ggml_add(ctx, k_indexer, model.layers[layer_idx].attn_indexer_k_norm_bias);
    }
    cb(k_indexer, "indexer_k_norm", layer_idx);
    printf("SPARSE INDEXER: After k_norm, k_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
    fflush(stdout);

    // Indexer weights projection - w^I_{t,j} from the formula
    // These are the per-head weights for the indexer
    ggml_tensor * weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur);
    cb(weights, "indexer_weights", layer_idx);
    printf("SPARSE INDEXER: After weights_proj, weights shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3]);
    printf("SPARSE INDEXER: weights total elements: %" PRId64 "\n", ggml_nelements(weights));
    fflush(stdout);

    // Indexer query projection (wq_b) - q^I_{t,j} from the formula
    // This projects the normalized query representation to the indexer query space
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qr);
    cb(q_indexer, "indexer_q", layer_idx);
    printf("SPARSE INDEXER: After wq_b projection, q_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           q_indexer->ne[0], q_indexer->ne[1], q_indexer->ne[2], q_indexer->ne[3]);
    printf("SPARSE INDEXER: q_indexer total elements: %" PRId64 "\n", ggml_nelements(q_indexer));
    fflush(stdout);

    // From DeepSeek V3.2 config: index_n_heads = 64, index_head_dim = 128
    const int64_t index_n_heads = 64;
    const int64_t index_head_dim = 128;
    
    // Get n_tokens from the weights tensor (moved from later in the function)
    const int64_t n_tokens = ggml_nelements(weights) / index_n_heads;
    printf("SPARSE INDEXER: Calculated n_tokens = %" PRId64 " / %" PRId64 " = %" PRId64 "\n", 
           ggml_nelements(weights), index_n_heads, n_tokens);
    fflush(stdout);
    
    // Reshape q_indexer to [index_n_heads, index_head_dim, n_tokens]
    // q_indexer should have shape [index_n_heads * index_head_dim, n_tokens] from matrix multiplication
    q_indexer = ggml_reshape_3d(ctx, q_indexer, index_n_heads, index_head_dim, n_tokens);
    cb(q_indexer, "indexer_q_reshape", layer_idx);

    // Reshape k_indexer to [index_head_dim, n_tokens]
    k_indexer = ggml_reshape_2d(ctx, k_indexer, index_head_dim, n_tokens);
    cb(k_indexer, "indexer_k_reshape", layer_idx);
    
    // Compute the dot product: q_indexer [index_n_heads, index_head_dim, n_tokens] · k_indexer [index_head_dim, n_tokens]
    // We need to reshape and permute to align dimensions properly
    
    // Permute q_indexer to [index_head_dim, index_n_heads, n_tokens]
    ggml_tensor * q_permuted = ggml_cont(ctx, ggml_permute(ctx, q_indexer, 1, 0, 2, 3));
    cb(q_permuted, "indexer_q_permuted", layer_idx);
    
    // Reshape q_permuted to [index_head_dim, index_n_heads * n_tokens]
    ggml_tensor * q_reshaped = ggml_reshape_2d(ctx, q_permuted, index_head_dim, index_n_heads * n_tokens);
    cb(q_reshaped, "indexer_q_reshaped", layer_idx);
    
    // Reshape k_indexer to [index_head_dim, n_tokens] (already done)
    ggml_tensor * k_reshaped = k_indexer;
    cb(k_reshaped, "indexer_k_reshaped", layer_idx);
    
    // Compute dot product: result will be [index_n_heads * n_tokens, n_tokens]
    // This gives us the dot product for each (head, query_token) pair with each key_token
    ggml_tensor * dot_product = ggml_mul_mat(ctx, q_reshaped, k_reshaped);
    cb(dot_product, "indexer_dot_product", layer_idx);
    
    // Apply ReLU activation
    ggml_tensor * relu_scores = ggml_relu(ctx, dot_product);
    cb(relu_scores, "indexer_relu_scores", layer_idx);
    
    // Reshape relu_scores to [index_n_heads, n_tokens, n_tokens]
    ggml_tensor * relu_3d = ggml_reshape_3d(ctx, relu_scores, n_tokens, index_n_heads, n_tokens);
    cb(relu_3d, "indexer_relu_3d", layer_idx);
    
    // Permute to [index_n_heads, n_tokens, n_tokens] for weight multiplication
    ggml_tensor * relu_permuted = ggml_cont(ctx, ggml_permute(ctx, relu_3d, 1, 0, 2, 3));
    cb(relu_permuted, "indexer_relu_permuted", layer_idx);
    
    // Reshape weights to [index_n_heads, n_tokens, 1] for broadcasting
    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, n_tokens, index_n_heads);
    weights_3d = ggml_cont(ctx, ggml_permute(ctx, weights_3d, 2, 1, 0, 3)); // [index_n_heads, n_tokens, 1]
    cb(weights_3d, "indexer_weights_3d", layer_idx);
    
    // Multiply ReLU scores by weights: w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    // We need to broadcast weights across the appropriate dimensions
    ggml_tensor * weighted_scores = ggml_mul(ctx, relu_permuted, weights_3d);
    cb(weighted_scores, "indexer_weighted_scores", layer_idx);
    
    // Sum across heads to get final importance scores: sum_{j=1}^{H^I}
    // weighted_scores has shape [index_n_heads, n_tokens, n_tokens]
    // We need to sum over the head dimension
    
    // Sum along the head dimension to get [n_tokens, n_tokens]
    ggml_tensor * token_importance = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, weighted_scores, index_n_heads, n_tokens * n_tokens));
    cb(token_importance, "token_importance", layer_idx);
    
    // Reshape to [n_tokens, n_tokens] if needed
    token_importance = ggml_reshape_2d(ctx, token_importance, n_tokens, n_tokens);
    if (token_importance) {
        cb(token_importance, "token_importance_final", layer_idx);
    } else {
        printf("Error: token_importance is null after reshape\n");
        return nullptr;
    }

    return token_importance;
}

} // namespace llama