#include "llama-sparse-attn.h"
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
    
    // Reshape to [n_tokens, 1] to ensure proper dimensions for ggml_top_k
    const int64_t n_tokens = ggml_nelements(token_importance);
    token_importance = ggml_reshape_2d(ctx, token_importance, n_tokens, 1);
    cb(token_importance, "token_importance", layer_idx);

    return token_importance;
}

ggml_tensor * sparse_attn_indexer::select_topk_tokens(
    ggml_context * ctx,
    ggml_tensor * token_importance,
    int64_t n_tokens,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // Identify top-k important tokens for sparse attention
    // VLLM Equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L794
    const int64_t actual_n_tokens = ggml_nelements(token_importance); // Extract actual number of tokens from tensor
    const int64_t top_k = (64 < actual_n_tokens) ? 64 : actual_n_tokens;  // Use top-64 tokens for sparse attention
    
    // Ensure token_importance has the correct dimensions for ggml_top_k
    // It should be a 2D tensor with dimensions [actual_n_tokens, 1]
    if (token_importance->ne[0] != actual_n_tokens || token_importance->ne[1] != 1) {
        token_importance = ggml_reshape_2d(ctx, token_importance, actual_n_tokens, 1);
        cb(token_importance, "token_importance_reshaped", -1);
    }
    
    // Get indices of top-k important tokens
    // VLLM Equivalent: https://github.com/vllm-project/vllm/blob/067da2d1df141363f0ad65939049709b2dbd5080/vllm/model_executor/models/deepseek_v2.py#L822
    ggml_tensor * topk_indices = ggml_top_k(ctx, token_importance, top_k);
    cb(topk_indices, "topk_indices", -1);

    // topk_indices now has dimensions [top_k, 1, 1, 1]
    return topk_indices;
}

ggml_tensor * sparse_attn_indexer::apply_sparse_attention(
    ggml_context * ctx,
    ggml_tensor * q_cur,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    ggml_tensor * topk_indices,
    int64_t n_tokens,
    int64_t top_k,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // Select only the top-k key-value pairs for sparse attention
    // ggml_get_rows expects: a->ne[2] == b->ne[1] and a->ne[3] == b->ne[2]
    // For k_cur: dimensions are [n_embd_head, n_head_kv, n_tokens]
    // For topk_indices: dimensions are [top_k, 1, 1, 1]
    // We need to properly prepare the tensors for ggml_get_rows
    // k_cur has dimensions [n_embd_head, n_head_kv, n_tokens]
    // We want to select specific tokens (along the n_tokens dimension)
    
    // Extract dimensions from the tensors
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head_kv = k_cur->ne[1];
    const int64_t actual_n_tokens = k_cur->ne[2]; // Extract actual n_tokens from tensor
    
    // We need to select specific tokens (along the n_tokens dimension)
    // k_cur has dimensions [n_embd_head, n_head_kv, actual_n_tokens]
    // We want to reshape it to [n_embd_head, n_head_kv * actual_n_tokens] to use ggml_get_rows
    
    // First, reshape k_cur to [n_embd_head, n_head_kv * actual_n_tokens] to treat tokens as "rows"
    ggml_tensor * k_cur_2d = ggml_reshape_2d(ctx, k_cur, n_embd_head, n_head_kv * actual_n_tokens);
    cb(k_cur_2d, "k_cur_2d", -1);
    
    // Similarly for v_cur
    ggml_tensor * v_cur_2d = ggml_reshape_2d(ctx, v_cur, n_embd_head, n_head_kv * actual_n_tokens);
    cb(v_cur_2d, "v_cur_2d", -1);
    
    // Now we need to create indices that select the sparse tokens
    // The indices should point to specific token positions within the flattened [n_head_kv * n_tokens] dimension
    // We need to convert the token indices to account for the head dimension
    
    // Create indices that account for the head dimension: index = token_index * n_head_kv + head_offset
    // But since we want to select entire tokens (across all heads), we need a different approach
    
    // Instead, let's use a simpler approach: reshape to [n_embd_head * n_head_kv, actual_n_tokens] and transpose
    ggml_tensor * k_cur_flat = ggml_reshape_2d(ctx, k_cur, n_embd_head * n_head_kv, actual_n_tokens);
    cb(k_cur_flat, "k_cur_flat", -1);
    
    ggml_tensor * v_cur_flat = ggml_reshape_2d(ctx, v_cur, n_embd_head * n_head_kv, actual_n_tokens);
    cb(v_cur_flat, "v_cur_flat", -1);
    
    // Transpose to [actual_n_tokens, n_embd_head * n_head_kv] so tokens become rows
    ggml_tensor * k_cur_transposed = ggml_cont(ctx, ggml_transpose(ctx, k_cur_flat));
    cb(k_cur_transposed, "k_cur_transposed", -1);
    
    ggml_tensor * v_cur_transposed = ggml_cont(ctx, ggml_transpose(ctx, v_cur_flat));
    cb(v_cur_transposed, "v_cur_transposed", -1);
    
    // Reshape indices to [top_k, 1]
    ggml_tensor * indices_2d = ggml_reshape_2d(ctx, topk_indices, top_k, 1);
    cb(indices_2d, "indices_2d", -1);
    
    // Select the sparse tokens as rows
    ggml_tensor * k_sparse_transposed = ggml_get_rows(ctx, k_cur_transposed, indices_2d);
    ggml_tensor * v_sparse_transposed = ggml_get_rows(ctx, v_cur_transposed, indices_2d);
    
    // Transpose back to [n_embd_head * n_head_kv, top_k]
    ggml_tensor * k_sparse_flat = ggml_cont(ctx, ggml_transpose(ctx, k_sparse_transposed));
    ggml_tensor * v_sparse_flat = ggml_cont(ctx, ggml_transpose(ctx, v_sparse_transposed));
    
    // Reshape back to the original head dimensions [n_embd_head, n_head_kv, top_k]
    ggml_tensor * k_sparse = ggml_reshape_3d(ctx, k_sparse_flat, n_embd_head, n_head_kv, top_k);
    ggml_tensor * v_sparse = ggml_reshape_3d(ctx, v_sparse_flat, n_embd_head, n_head_kv, top_k);
    
    cb(k_sparse, "k_sparse", -1);
    cb(v_sparse, "v_sparse", -1);
    
    // Compute attention scores with sparse keys
    // We need to reshape the tensors for compatible matrix multiplication
    
    // Extract dimensions from q_cur
    const int64_t n_embd_head_q = q_cur->ne[0];
    const int64_t n_head_q = q_cur->ne[1];
    const int64_t actual_n_tokens_q = q_cur->ne[2]; // Extract actual n_tokens from query tensor
    
    // Reshape q_cur to [n_head_q * actual_n_tokens_q, n_embd_head_q]
    ggml_tensor * q_2d = ggml_reshape_2d(ctx, q_cur, n_embd_head_q, n_head_q * actual_n_tokens_q);
    q_2d = ggml_cont(ctx, ggml_transpose(ctx, q_2d)); // Transpose to [n_head_q * actual_n_tokens_q, n_embd_head_q]
    cb(q_2d, "q_2d", -1);
    
    // Reshape k_sparse to [n_head_kv * top_k, n_embd_head]
    ggml_tensor * k_sparse_2d = ggml_reshape_2d(ctx, k_sparse, n_embd_head, n_head_kv * top_k);
    k_sparse_2d = ggml_cont(ctx, ggml_transpose(ctx, k_sparse_2d)); // Transpose to [n_head_kv * top_k, n_embd_head]
    cb(k_sparse_2d, "k_sparse_2d", -1);
    
    // Compute Q @ K^T to get [n_head_q * actual_n_tokens_q, n_head_kv * top_k]
    ggml_tensor * attn_scores = ggml_mul_mat(ctx, q_2d, k_sparse_2d);
    cb(attn_scores, "attn_scores_sparse", -1);
    
    // Apply attention scaling
    const float kq_scale = 1.0f / sqrtf((float)n_embd_head); // Use same scaling as dense attention
    attn_scores = ggml_scale(ctx, attn_scores, kq_scale);
    cb(attn_scores, "attn_scores_scaled", -1);
    
    // Apply softmax to sparse attention scores
    ggml_tensor * attn_weights = ggml_soft_max(ctx, attn_scores);
    cb(attn_weights, "attn_weights_sparse", -1);
    
    // Reshape v_sparse to [n_head_kv * top_k, n_embd_head]
    ggml_tensor * v_sparse_2d = ggml_reshape_2d(ctx, v_sparse, n_embd_head, n_head_kv * top_k);
    v_sparse_2d = ggml_cont(ctx, ggml_transpose(ctx, v_sparse_2d)); // Transpose to [n_head_kv * top_k, n_embd_head]
    cb(v_sparse_2d, "v_sparse_2d", -1);
    
    // Compute attention weights @ V to get [n_head_q * actual_n_tokens_q, n_embd_head]
    ggml_tensor * output_2d = ggml_mul_mat(ctx, attn_weights, v_sparse_2d);
    cb(output_2d, "output_2d", -1);
    
    // Transpose back to [n_embd_head, n_head_q * actual_n_tokens_q]
    ggml_tensor * output_transposed = ggml_cont(ctx, ggml_transpose(ctx, output_2d));
    cb(output_transposed, "output_transposed", -1);
    
    // Reshape back to original dimensions [n_embd_head, n_head_q, actual_n_tokens_q]
    ggml_tensor * output = ggml_reshape_3d(ctx, output_transposed, n_embd_head_q, n_head_q, actual_n_tokens_q);
    cb(output, "sparse_attn_out", -1);
    
    return output;
}

} // namespace llama