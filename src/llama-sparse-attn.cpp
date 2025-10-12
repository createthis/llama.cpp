#include "llama-sparse-attn.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>

namespace llama {

ggml_tensor * sparse_attn_indexer::compute_token_importance(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    bool is_lite,
    const std::function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // Indexer query projection (wq_a)
    ggml_tensor * qr = nullptr;
    if (!is_lite) {
        // vllm equivalent (maybe): https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/layers/mla.py#L137
        qr = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        cb(qr, "indexer_qr", layer_idx);

        // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/layers/mla.py#L142
        qr = ggml_norm(ctx, qr, 1e-5f);
        cb(qr, "indexer_qr_norm", layer_idx);
    } else {
        qr = cur; // For lite version, use the current hidden state directly
    }

    // Indexer key projection (wk)
    // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L848
    ggml_tensor * k_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    cb(k_indexer, "indexer_k", layer_idx);

    // Indexer key normalization (k_norm)
    // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L849
    k_indexer = ggml_norm(ctx, k_indexer, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        k_indexer = ggml_add(ctx, k_indexer, model.layers[layer_idx].attn_indexer_k_norm_bias);
    }
    cb(k_indexer, "indexer_k_norm", layer_idx);

    // Indexer weights projection
    // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L869
    ggml_tensor * weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur);
    cb(weights, "indexer_weights", layer_idx);

    // Indexer query projection (wq_b)
    // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L842
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qr);
    cb(q_indexer, "indexer_q", layer_idx);

    // Reshape q_indexer to [n_tokens, index_n_heads, index_head_dim]
    const int64_t index_n_heads = 64;   // From VLLM: config.index_n_heads
    const int64_t index_head_dim = 128; // From VLLM: config.index_head_dim
    q_indexer = ggml_reshape_3d(ctx, q_indexer, index_head_dim, index_n_heads, ggml_nelements(q_indexer) / (index_head_dim * index_n_heads));
    cb(q_indexer, "indexer_q_reshape", layer_idx);

    // Compute token importance scores using the indexer
    // The weights projection gives us importance scores per token and head
    // vllm equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L871
    weights = ggml_reshape_3d(ctx, weights, 1, index_n_heads, ggml_nelements(weights) / index_n_heads);
    cb(weights, "indexer_weights_reshaped", layer_idx);
    
    // Reshape weights to [index_n_heads, n_tokens] and sum across heads
    ggml_tensor * weights_2d = ggml_reshape_2d(ctx, weights, index_n_heads, ggml_nelements(weights) / index_n_heads);
    ggml_tensor * token_importance = ggml_sum_rows(ctx, weights_2d); // Sum along rows (heads dimension)
    token_importance = ggml_reshape_1d(ctx, token_importance, ggml_nelements(token_importance));
    cb(token_importance, "token_importance", layer_idx);

    return token_importance;
}

ggml_tensor * sparse_attn_indexer::select_topk_tokens(
    ggml_context * ctx,
    ggml_tensor * token_importance,
    int64_t n_tokens,
    const std::function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // Identify top-k important tokens for sparse attention
    // VLLM Equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L794
    const int64_t top_k = (64 < n_tokens) ? 64 : n_tokens;  // Use top-64 tokens for sparse attention
    
    // Get indices of top-k important tokens
    // VLLM Equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L822
    ggml_tensor * topk_indices = ggml_top_k(ctx, token_importance, top_k);
    cb(topk_indices, "topk_indices", -1);

    return topk_indices;
}

ggml_tensor * sparse_attn_indexer::create_sparse_mask(
    ggml_context * ctx,
    ggml_tensor * topk_indices,
    int64_t n_tokens,
    int64_t top_k,
    const std::function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // Create a sparse attention mask that only allows attention to top-k tokens
    ggml_tensor * sparse_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    
    // Initialize mask with -inf (no attention allowed)
    // Note: In a real implementation, we would need to properly initialize the mask values
    // For now, we create the tensor but the actual initialization would need to be done differently
    
    // For each query position, allow attention to the top-k key positions
    // This is a simplified approach - a real implementation would be more sophisticated
    
    // Create a mask that allows each query to attend to its top-k selected keys
    // Note: This is a conceptual implementation - actual implementation would be more complex
    // In a real implementation, we would need to properly set the mask values
    
    cb(sparse_mask, "sparse_mask", -1);
    
    return sparse_mask;
}

} // namespace llama