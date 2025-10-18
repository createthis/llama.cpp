#include "llama-sparse-mla-fwd.h"
#include "llama-impl.h"

#include <cmath>
#include <cstdio>
#include <cinttypes>

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
    (void)n_tokens; // Unused parameter
    
    // DeepSeek V3.2 Sparse Attention Application
    // Applies attention only to the top-k selected tokens
    
    // Extract dimensions from the tensors (treat K and V independently)
    const int64_t n_embd_head_k = k_cur->ne[0];
    const int64_t n_head_kv     = k_cur->ne[1];
    const int64_t actual_n_tokens_k = k_cur->ne[2];

    const int64_t n_embd_head_v = v_cur->ne[0];
    const int64_t n_head_kv_v   = v_cur->ne[1];
    const int64_t actual_n_tokens_v = v_cur->ne[2];

    // Extract dimensions from q_cur
    const int64_t n_embd_head_q = q_cur->ne[0];
    const int64_t n_head_q      = q_cur->ne[1];
    const int64_t actual_n_tokens_q = q_cur->ne[2];
    
    printf("SPARSE MLA: Starting apply_sparse_attention\n");
    printf("SPARSE MLA: q_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_q, n_head_q, actual_n_tokens_q);
    printf("SPARSE MLA: k_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_k, n_head_kv, actual_n_tokens_k);
    printf("SPARSE MLA: v_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_v, n_head_kv_v, actual_n_tokens_v);
    printf("SPARSE MLA: topk_indices shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    fflush(stdout);
    
    // Reshape key and value tensors to prepare for sparse selection
    // We need to reshape the tensors so tokens become rows for ggml_get_rows
    
    // Reshape K to 4D: [n_embd_head_k * n_head_kv, actual_n_tokens_k, 1, 1]
    // This format is required for ggml_get_rows
    ggml_tensor * k_cur_4d = ggml_reshape_4d(ctx, k_cur, n_embd_head_k * n_head_kv, actual_n_tokens_k, 1, 1);
    cb(k_cur_4d, "k_cur_4d", -1);
    
    // Reshape V using its own head dims
    ggml_tensor * v_cur_4d = ggml_reshape_4d(ctx, v_cur, n_embd_head_v * n_head_kv_v, actual_n_tokens_v, 1, 1);
    cb(v_cur_4d, "v_cur_4d", -1);
    
    printf("SPARSE MLA: After reshape to 4D - k_cur_4d shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_cur_4d->ne[0], k_cur_4d->ne[1], k_cur_4d->ne[2], k_cur_4d->ne[3]);
    printf("SPARSE MLA: After reshape to 4D - v_cur_4d shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           v_cur_4d->ne[0], v_cur_4d->ne[1], v_cur_4d->ne[2], v_cur_4d->ne[3]);
    
    // Prepare the indices tensor for ggml_get_rows
    // ggml_get_rows expects indices to have shape [n_rows, ne2, ne3, 1]
    // where ne2 == a->ne[2] and ne3 == a->ne[3]
    // Our topk_indices has shape [top_k, 1, 1, 1], so we need to reshape it to [top_k, 1, 1, 1]
    // but first ensure it's the right type
    
    // Convert indices to I32 if needed
    ggml_tensor * indices_i32 = topk_indices;
    if (topk_indices->type != GGML_TYPE_I32) {
        indices_i32 = ggml_cast(ctx, topk_indices, GGML_TYPE_I32);
        cb(indices_i32, "indices_i32", -1);
    }
    
    // Reshape indices to [top_k, 1, 1, 1] for ggml_get_rows
    ggml_tensor * indices_4d = ggml_reshape_4d(ctx, indices_i32, top_k, 1, 1, 1);
    cb(indices_4d, "indices_4d", -1);
    
    printf("SPARSE MLA: After indices reshape - indices_4d shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           indices_4d->ne[0], indices_4d->ne[1], indices_4d->ne[2], indices_4d->ne[3]);
    
    // Use ggml_get_rows to select the sparse tokens
    // This will select rows from k_cur_4d and v_cur_4d based on the indices
    ggml_tensor * k_sparse_4d = ggml_get_rows(ctx, k_cur_4d, indices_4d);
    cb(k_sparse_4d, "k_sparse_4d", -1);
    
    ggml_tensor * v_sparse_4d = ggml_get_rows(ctx, v_cur_4d, indices_4d);
    cb(v_sparse_4d, "v_sparse_4d", -1);
    
    // The result of ggml_get_rows has shape [n_embd_head_* * n_head_kv_*, top_k, 1, 1]
    // Reshape to 3D with explicit head dims
    ggml_tensor * k_sparse = ggml_reshape_3d(ctx, k_sparse_4d, n_embd_head_k, n_head_kv, top_k);
    cb(k_sparse, "k_sparse", -1);
    printf("SPARSE MLA: After get_rows and reshape - k_sparse shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_sparse->ne[0], k_sparse->ne[1], k_sparse->ne[2]);
    fflush(stdout);
    
    ggml_tensor * v_sparse = ggml_reshape_3d(ctx, v_sparse_4d, n_embd_head_v, n_head_kv_v, top_k);
    cb(v_sparse, "v_sparse", -1);
    
    // Make sure the tensors are contiguous
    k_sparse = ggml_cont(ctx, k_sparse);
    v_sparse = ggml_cont(ctx, v_sparse);
    
    // Compute attention scores with sparse keys
    // Reshape q_cur to [n_embd_head_q, n_head_q * actual_n_tokens_q]
    ggml_tensor * q_2d = ggml_reshape_2d(ctx, q_cur, n_embd_head_q, n_head_q * actual_n_tokens_q);
    cb(q_2d, "q_2d", -1);
    
    printf("SPARSE MLA: After q_cur reshape - q_2d shape: [%" PRId64 ", %" PRId64 "]\n", 
           q_2d->ne[0], q_2d->ne[1]);
    fflush(stdout);
    
    // Reshape k_sparse to [n_embd_head_k, n_head_kv * top_k]
    ggml_tensor * k_sparse_2d = ggml_reshape_2d(ctx, k_sparse, n_embd_head_k, n_head_kv * top_k);
    cb(k_sparse_2d, "k_sparse_2d", -1);
    
    // Compute Q^T @ K to get [n_head_q * actual_n_tokens_q, n_head_kv * top_k]
    ggml_tensor * attn_scores = ggml_mul_mat(ctx, k_sparse_2d, q_2d);
    cb(attn_scores, "attn_scores_sparse", -1);
    
    // Apply attention scaling
    const float kq_scale = 1.0f / sqrtf((float)n_embd_head_k);
    attn_scores = ggml_scale(ctx, attn_scores, kq_scale);
    cb(attn_scores, "attn_scores_scaled", -1);
    
    // Apply softmax to sparse attention scores
    ggml_tensor * attn_weights = ggml_soft_max(ctx, attn_scores);
    cb(attn_weights, "attn_weights_sparse", -1);
    
    // Reshape v_sparse to [n_head_kv_v * top_k, n_embd_head_v] for proper matrix multiplication
    ggml_tensor * v_sparse_2d = ggml_reshape_2d(ctx, v_sparse, n_embd_head_v, n_head_kv_v * top_k);
    v_sparse_2d = ggml_cont(ctx, ggml_transpose(ctx, v_sparse_2d)); // [n_head_kv_v * top_k, n_embd_head_v]
    cb(v_sparse_2d, "v_sparse_2d", -1);
    
    printf("SPARSE MLA: After v_sparse reshape and transpose - v_sparse_2d shape: [%" PRId64 ", %" PRId64 "]\n", 
           v_sparse_2d->ne[0], v_sparse_2d->ne[1]);
    fflush(stdout);
    
    // Compute attn_weights @ V to get [n_head_q * actual_n_tokens_q, n_embd_head_v]
    ggml_tensor * output_2d = ggml_mul_mat(ctx, attn_weights, v_sparse_2d);
    cb(output_2d, "output_2d", -1);
    
    // Transpose back to [n_embd_head_v, n_head_q * actual_n_tokens_q]
    ggml_tensor * output_transposed = ggml_cont(ctx, ggml_transpose(ctx, output_2d));
    cb(output_transposed, "output_transposed", -1);
    
    // Reshape back to dimensions [n_embd_head_v, n_head_q, actual_n_tokens_q]
    ggml_tensor * output = ggml_reshape_3d(ctx, output_transposed, n_embd_head_v, n_head_q, actual_n_tokens_q);
    cb(output, "sparse_attn_out", -1);
    
    printf("SPARSE MLA: Final output shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           output->ne[0], output->ne[1], output->ne[2]);
    printf("SPARSE MLA: apply_sparse_attention completed successfully\n");
    fflush(stdout);
    
    return output;
}

} // namespace llama