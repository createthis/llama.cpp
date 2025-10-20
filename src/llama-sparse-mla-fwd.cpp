#include "llama-sparse-mla-fwd.h"
#include "llama-impl.h"

#include <cmath>
#include <cstdio>
#include <cinttypes>

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

ggml_tensor * sparse_mla_fwd::apply_sparse_attention(
    ggml_context * ctx,
    ggml_tensor * q_cur,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    ggml_tensor * topk_indices,
    int64_t n_tokens,
    int64_t top_k,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    (void)n_tokens;

    const int64_t n_embd_head_k = k_cur->ne[0];
    const int64_t n_head_kv     = k_cur->ne[1];
    const int64_t actual_n_tokens_k = k_cur->ne[2];

    const int64_t n_embd_head_v = v_cur->ne[0];
    const int64_t n_head_kv_v   = v_cur->ne[1];
    const int64_t actual_n_tokens_v = v_cur->ne[2];

    const int64_t n_embd_head_q = q_cur->ne[0];
    const int64_t n_head_q      = q_cur->ne[1];
    const int64_t actual_n_tokens_q = q_cur->ne[2];

    printf("SPARSE MLA: Starting apply_sparse_attention\n");
    size_t initial_mem = ggml_used_mem(ctx);
    printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
    printf("SPARSE MLA: q_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_q, n_head_q, actual_n_tokens_q);
    printf("SPARSE MLA: k_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_k, n_head_kv, actual_n_tokens_k);
    printf("SPARSE MLA: v_cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_v, n_head_kv_v, actual_n_tokens_v);
    printf("SPARSE MLA: topk_indices shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    printf("SPARSE MLA DBG: q_cur  ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "] nb=[%zu,%zu,%zu,%zu] type=%d\n",
           q_cur->ne[0], q_cur->ne[1], q_cur->ne[2], q_cur->nb[0], q_cur->nb[1], q_cur->nb[2], q_cur->nb[3], (int)q_cur->type);
    printf("SPARSE MLA DBG: k_cur  ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "] nb=[%zu,%zu,%zu,%zu] type=%d\n",
           k_cur->ne[0], k_cur->ne[1], k_cur->ne[2], k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3], (int)k_cur->type);
    printf("SPARSE MLA DBG: v_cur  ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "] nb=[%zu,%zu,%zu,%zu] type=%d\n",
           v_cur->ne[0], v_cur->ne[1], v_cur->ne[2], v_cur->nb[0], v_cur->nb[1], v_cur->nb[2], v_cur->nb[3], (int)v_cur->type);
    fflush(stdout);

    // Reshape key and value tensors for row selection
    ggml_tensor * k_cur_4d = ggml_reshape_4d(ctx, k_cur, n_embd_head_k * n_head_kv, actual_n_tokens_k, 1, 1);
    ggml_tensor * v_cur_4d = ggml_reshape_4d(ctx, v_cur, n_embd_head_v * n_head_kv_v, actual_n_tokens_v, 1, 1);
    cb(k_cur_4d, "k_cur_4d", -1);
    cb(v_cur_4d, "v_cur_4d", -1);

    // Prepare indices as [top_k, T, 1, 1] without creating copies (avoid CUDA i32->i32 cpy)
    ggml_tensor * indices_full = nullptr;
    // Use the topk_indices tensor directly; avoid reshape which requires contiguity
    indices_full = topk_indices;
    cb(indices_full, "indices_full", -1);
    // Basic assertions on index tensor shape/type
    GGML_ASSERT(indices_full->type == GGML_TYPE_I32);
    GGML_ASSERT(indices_full->ne[0] == top_k);

    // 2D view of Q: [Dq, Hq*T]
    ggml_tensor * q_all_2d = ggml_reshape_2d(ctx, q_cur, n_embd_head_q, n_head_q * actual_n_tokens_q);

    ggml_tensor * output_acc = nullptr;
    for (int64_t t = 0; t < actual_n_tokens_q; ++t) {
        // Slice indices for query t: [top_k, 1, 1, 1]
        ggml_tensor * idx_t_4d = nullptr;
        if (indices_full->ne[1] == 1) {
            // Use the same column for all t
            idx_t_4d = ggml_reshape_4d(ctx, indices_full, top_k, 1, 1, 1);
        } else {
            size_t idx_off = t * indices_full->nb[1];
            ggml_tensor * idx_t_2d = ggml_view_2d(ctx, indices_full, top_k, 1, indices_full->nb[1], idx_off);
            idx_t_4d = ggml_reshape_4d(ctx, idx_t_2d, top_k, 1, 1, 1);
        }

        // Select K and V rows
        ggml_tensor * k_sel_4d = ggml_get_rows(ctx, k_cur_4d, idx_t_4d); // [Dk*Hkv, top_k, 1, 1]
        ggml_tensor * v_sel_4d = ggml_get_rows(ctx, v_cur_4d, idx_t_4d); // [Dv*Hkv, top_k, 1, 1]

        // Reshape for matmul
        ggml_tensor * k_sel_2d = ggml_reshape_2d(ctx, k_sel_4d, n_embd_head_k, n_head_kv * top_k);
        ggml_tensor * v_sel_2d = ggml_reshape_2d(ctx, v_sel_4d, n_embd_head_v, n_head_kv_v * top_k);
        v_sel_2d = ggml_cont(ctx, ggml_transpose(ctx, v_sel_2d)); // [n_head_kv_v * top_k, n_embd_head_v]

        // Slice Q for t: columns [t*Hq .. t*Hq + Hq - 1]
        size_t q_off = t * n_head_q * q_all_2d->nb[1];
        ggml_tensor * q_t_2d = ggml_view_2d(ctx, q_all_2d, n_embd_head_q, n_head_q, q_all_2d->nb[1], q_off);

        // Attention: scores and weights
        ggml_tensor * scores_t = ggml_mul_mat(ctx, k_sel_2d, q_t_2d); // [n_head_kv*top_k, n_head_q]
        // use base scale for block-local attention path
        scores_t = ggml_scale(ctx, scores_t, 1.0f / sqrtf((float)n_embd_head_k));
        ggml_tensor * weights_t = ggml_soft_max(ctx, scores_t);

        // Weighted V and output for t
        ggml_tensor * out2d_t = ggml_mul_mat(ctx, weights_t, v_sel_2d); // [n_head_q, n_embd_head_v]
        ggml_tensor * out2d_t_T = ggml_cont(ctx, ggml_transpose(ctx, out2d_t)); // [n_embd_head_v, n_head_q]
        ggml_tensor * out3d_t = ggml_reshape_3d(ctx, out2d_t_T, n_embd_head_v, n_head_q, 1);

        if (output_acc == nullptr) {
            output_acc = out3d_t;
        } else {
            output_acc = ggml_concat(ctx, output_acc, out3d_t, 2);
        }
    }

    cb(output_acc, "sparse_attn_out", -1);
    printf("SPARSE MLA: Final output shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           output_acc->ne[0], output_acc->ne[1], output_acc->ne[2]);
    printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(),
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    printf("SPARSE MLA: apply_sparse_attention completed successfully\n");
    fflush(stdout);

    return output_acc;
}

  ggml_tensor * sparse_mla_fwd::apply_sparse_attention_kvaware(
      ggml_context * ctx,
      ggml_tensor * q_cur,
      ggml_tensor * k_cache,
      ggml_tensor * v_cache,
      ggml_tensor * topk_indices,
      int64_t n_tokens,
      int64_t top_k,
      float kq_scale,
      ggml_tensor * kq_mask,
      float attn_softcap,
      const function<void(ggml_tensor *, const char *, int)> & cb) {
      (void)n_tokens;

      const int64_t Dk   = k_cache->ne[0];
      const int64_t Hkv  = k_cache->ne[1];
      const int64_t N_kv = k_cache->ne[2];
      const int64_t Dv   = v_cache->ne[0];
      const int64_t Hkv_v= v_cache->ne[1];
      const int64_t N_kv_v=v_cache->ne[2];

      const int64_t Dq   = q_cur->ne[0];
      const int64_t Hq   = q_cur->ne[1];
      const int64_t T    = q_cur->ne[2];

      cb(k_cache, "kvaware_k_cache", -1);
      cb(v_cache, "kvaware_v_cache", -1);
      cb(q_cur,   "kvaware_q_cur",   -1);
      cb(topk_indices, "kvaware_topk_indices", -1);
      printf("SPARSE MLA KV-AWARE DBG: Q=[%" PRId64 ",%" PRId64 ",%" PRId64 "] K=[%" PRId64 ",%" PRId64 ",%" PRId64 "] V=[%" PRId64 ",%" PRId64 ",%" PRId64 "] topk=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
             Dq, Hq, T, Dk, Hkv, N_kv, Dv, Hkv_v, N_kv_v,
             topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
      fflush(stdout);

      ggml_tensor * K4d = ggml_reshape_4d(ctx, k_cache, Dk*Hkv, N_kv, 1, 1);
      ggml_tensor * V4d = ggml_reshape_4d(ctx, v_cache, Dv*Hkv_v, N_kv_v, 1, 1);

      ggml_tensor * q_all_2d = ggml_reshape_2d(ctx, q_cur, Dq, Hq*T);

      ggml_tensor * output_acc = nullptr;
      for (int64_t t = 0; t < T; ++t) {
          GGML_ASSERT(topk_indices->ne[0] == top_k);
          ggml_tensor * idx_t_2d = ggml_view_2d(ctx, topk_indices, top_k, 1, topk_indices->nb[1], t * topk_indices->nb[1]);
          ggml_tensor * idx_t_4d = ggml_reshape_4d(ctx, idx_t_2d, top_k, 1, 1, 1);

          ggml_tensor * k_sel_4d = ggml_get_rows(ctx, K4d, idx_t_4d); // [Dk*Hkv, top_k]
          ggml_tensor * v_sel_4d = ggml_get_rows(ctx, V4d, idx_t_4d); // [Dv*Hkv, top_k]

          ggml_tensor * k_sel_2d = ggml_reshape_2d(ctx, k_sel_4d, Dk, Hkv*top_k);
          ggml_tensor * v_sel_2d = ggml_reshape_2d(ctx, v_sel_4d, Dv, Hkv_v*top_k);
          // ensure contiguous [Hkv_v*top_k, Dv] without extra transpose memory when possible
          if (ggml_is_contiguous(k_sel_2d) && ggml_is_contiguous(v_sel_2d)) {
              v_sel_2d = ggml_view_2d(ctx, v_sel_2d, Hkv_v*top_k, Dv, v_sel_2d->nb[1], 0);
          } else {
              v_sel_2d = ggml_cont(ctx, ggml_transpose(ctx, v_sel_2d));
          }

          size_t q_off = t * Hq * q_all_2d->nb[1];
          ggml_tensor * q_t_2d = ggml_view_2d(ctx, q_all_2d, Dq, Hq, q_all_2d->nb[1], q_off);

          ggml_tensor * scores_t = ggml_mul_mat(ctx, k_sel_2d, q_t_2d); // [Hkv*top_k, Hq]
          scores_t = ggml_scale(ctx, scores_t, kq_scale);
          // add mask/alibi bias if provided: gather kq_mask rows by indices and add to scores
          if (kq_mask) {
              // kq_mask is [N_kv, PAD(T)]; take column t -> [N_kv,1]
              ggml_tensor * mask_col = ggml_view_2d(ctx, kq_mask, kq_mask->ne[0], 1, kq_mask->nb[1], t * kq_mask->nb[1]);
              // convert to [1, N_kv] so get_rows yields [1, top_k, 1, 1]
              ggml_tensor * mask_vec = ggml_transpose(ctx, mask_col); // [1, N_kv]
              if (mask_vec->type != scores_t->type) {
                  mask_vec = ggml_cast(ctx, mask_vec, scores_t->type);
              }
              ggml_tensor * mask_rows_4d = ggml_get_rows(ctx, mask_vec, idx_t_4d); // [1, top_k, 1, 1]
              // reshape to [top_k, 1]
              ggml_tensor * mask_rows_2d = ggml_reshape_2d(ctx, mask_rows_4d, top_k, 1); // [top_k,1]

              // build a view of scores' column shape [Hkv*top_k, 1]
              ggml_tensor * scores_col_view = ggml_view_2d(ctx,
                  scores_t,
                  /*ne0*/ Hkv*top_k, /*ne1*/ 1,
                  /*nb1*/ scores_t->nb[0],
                  /*offset*/ 0);

              // repeat mask across heads only -> [Hkv*top_k, 1]
              ggml_tensor * mask_bias = ggml_repeat(ctx, mask_rows_2d, scores_col_view); // [Hkv*top_k, 1]

              // add with column broadcast: [Hkv*top_k, Hq] + [Hkv*top_k, 1]
              scores_t = ggml_add(ctx, scores_t, mask_bias);
          }

          // apply tanh softcap if enabled
          if (attn_softcap > 0.0f) {
              scores_t = ggml_scale(ctx, scores_t, 1.0f / attn_softcap);
              scores_t = ggml_tanh(ctx, scores_t);
              scores_t = ggml_scale(ctx, scores_t, attn_softcap);
          }
          ggml_tensor * weights_t = ggml_soft_max(ctx, scores_t);

          ggml_tensor * out2d_t = ggml_mul_mat(ctx, weights_t, v_sel_2d); // [Hq, Dv]
          ggml_tensor * out2d_t_T = ggml_cont(ctx, ggml_transpose(ctx, out2d_t)); // [Dv, Hq]
          ggml_tensor * out3d_t = ggml_reshape_3d(ctx, out2d_t_T, Dv, Hq, 1);

          if (!output_acc) output_acc = out3d_t; else output_acc = ggml_concat(ctx, output_acc, out3d_t, 2);
      }

      cb(output_acc, "kvaware_sparse_attn_out", -1);
      return output_acc;
  }


} // namespace llama
