#include "llama-sparse-attn.h"
#include "llama-impl.h"

#include <cmath>

namespace llama {

using std::function;

ggml_tensor * sparse_mla_fwd::apply_sparse_attention(
    ggml_context * ctx,
    ggml_tensor * q_cur,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    ggml_tensor * topk_indices,
    int64_t n_tokens,
    int64_t top_k,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // DeepSeek V3.2 Sparse Attention Application
    // Applies attention only to the top-k selected tokens
    
    // Extract dimensions from the tensors
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head_kv = k_cur->ne[1];
    const int64_t actual_n_tokens = k_cur->ne[2]; // Extract actual n_tokens from tensor
    
    // Extract dimensions from q_cur
    const int64_t n_embd_head_q = q_cur->ne[0];
    const int64_t n_head_q = q_cur->ne[1];
    const int64_t actual_n_tokens_q = q_cur->ne[2]; // Extract actual n_tokens from query tensor
    
    // Reshape key and value tensors to prepare for sparse selection
    // We need to flatten the tensors so tokens become rows for ggml_get_rows
    
    // Reshape k_cur to [n_embd_head * n_head_kv, actual_n_tokens]
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
    
    // Select the sparse tokens as rows using ggml_get_rows
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