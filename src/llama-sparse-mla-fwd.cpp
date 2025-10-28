#include "llama-sparse-mla-fwd.h"
#include "llama-impl.h"

#include <cmath>
#include <cstdio>
#include <cinttypes>

#include <climits>

namespace llama {

using std::function;


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

      int64_t Dk   = k_cache->ne[0];
      int64_t Hkv  = k_cache->ne[1];
      int64_t N_kv = k_cache->ne[2];
      int64_t Dv   = v_cache->ne[0];
      int64_t Hkv_v= v_cache->ne[1];
      int64_t N_kv_v= v_cache->ne[2];

      const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
      const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);

      // Normalize V layout: expected effective layout is [Dv, Hkv_v, N_kv]
      if (dbg) {
          size_t bytes_topk = (size_t)topk_indices->ne[0] * (size_t)topk_indices->ne[1] * ggml_type_size(topk_indices->type);
          printf("[MLA-DBG] topk_indices ne=[%lld,%lld] bytes=%zu (INT_MAX=%d)\n",
                 (long long)topk_indices->ne[0], (long long)topk_indices->ne[1], bytes_topk, INT_MAX);
          fflush(stdout);
      }

      // Some builds return V cache with transposed layout [N_kv, Hkv_v, Dv, ns].
      ggml_tensor * V_gather_src = nullptr;
      if (N_kv_v == N_kv) {
          // Normal layout: [Dv, Hkv_v, N_kv, ns]
          V_gather_src = v_cache;
      } else if (Dv == N_kv) {
          // Transposed layout: [N_kv, Hkv_v, Dv, ns] -> permute to [Dv, Hkv_v, N_kv, ns]
          ggml_tensor * v_perm = ggml_permute(ctx, v_cache, 2, 1, 0, 3);
          v_perm = ggml_cont(ctx, v_perm);
          Dv     = v_perm->ne[0];
          Hkv_v  = v_perm->ne[1];
          N_kv_v = v_perm->ne[2];
          V_gather_src = v_perm;
      } else {
          // Unexpected; proceed without permute but warn
          printf("[SPARSE-MLA][WARN] V cache unexpected layout: v_cache=[%lld,%lld,%lld,%lld], K N_kv=%lld\n",
                 (long long) v_cache->ne[0], (long long) v_cache->ne[1], (long long) v_cache->ne[2], (long long) v_cache->ne[3], (long long) N_kv);
          fflush(stdout);
          V_gather_src = v_cache;
          // best effort: if v_cache->ne[0] == N_kv, treat as transposed
          if (v_cache->ne[0] == N_kv) {
              ggml_tensor * v_perm = ggml_permute(ctx, v_cache, 2, 1, 0, 3);
              v_perm = ggml_cont(ctx, v_perm);
              Dv     = v_perm->ne[0];
              Hkv_v  = v_perm->ne[1];
              N_kv_v = v_perm->ne[2];
              V_gather_src = v_perm;
          }
      }

      const int64_t Dq   = q_cur->ne[0];
      const int64_t Hq   = q_cur->ne[1];
      const int64_t T    = q_cur->ne[2];

      if (dbg) {
          cb(k_cache, "kvaware_k_cache", -1);
          cb(v_cache, "kvaware_v_cache", -1);
          cb(q_cur,   "kvaware_q_cur",   -1);
          cb(topk_indices, "kvaware_topk_indices", -1);
          printf("[SPARSE-MLA] Dq=%lld Hq=%lld T=%lld Dk=%lld Hkv=%lld N_kv=%lld Dv=%lld Hkv_v=%lld\n",
                 (long long) Dq, (long long) Hq, (long long) T,
                 (long long) Dk, (long long) Hkv, (long long) N_kv,
                 (long long) Dv, (long long) Hkv_v);
          fflush(stdout);

          printf("SPARSE MLA KV-AWARE DBG: Q=[%" PRId64 ",%" PRId64 ",%" PRId64 "] K=[%" PRId64 ",%" PRId64 ",%" PRId64 "] V=[%" PRId64 ",%" PRId64 ",%" PRId64 "] topk=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 Dq, Hq, T, Dk, Hkv, N_kv, Dv, Hkv_v, N_kv_v,
                 topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
          fflush(stdout);
      }

      ggml_tensor * K4d = ggml_reshape_4d(ctx, k_cache, Dk*Hkv, N_kv, 1, 1);
      ggml_tensor * V4d = ggml_reshape_4d(ctx, V_gather_src, Dv*Hkv_v, N_kv_v, 1, 1);

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
          // CUDA matmul requires row-contiguous inputs
          k_sel_2d = ggml_cont(ctx, k_sel_2d);

          // ensure contiguous [Hkv_v*top_k, Dv] with a real transpose to avoid stride/view aliasing
          if (dbg && t < 2) {
              size_t bytes_k_sel = (size_t)k_sel_2d->ne[0]*(size_t)k_sel_2d->ne[1]*ggml_type_size(k_sel_2d->type);
              size_t bytes_v_sel = (size_t)v_sel_2d->ne[0]*(size_t)v_sel_2d->ne[1]*ggml_type_size(v_sel_2d->type);
              size_t bytes_scores = (size_t)(Hkv*top_k)*(size_t)Hq*ggml_type_size(GGML_TYPE_F32);
              printf("[MLA-DBG] t=%lld k_sel_2d ne=[%lld,%lld] bytes=%zu; v_sel_2d ne=[%lld,%lld] bytes=%zu; scores bytes=%zu\n",
                     (long long)t, (long long)k_sel_2d->ne[0], (long long)k_sel_2d->ne[1], bytes_k_sel,
                     (long long)v_sel_2d->ne[0], (long long)v_sel_2d->ne[1], bytes_v_sel, bytes_scores);
              fflush(stdout);
          }

          v_sel_2d = ggml_cont(ctx, ggml_transpose(ctx, v_sel_2d));
          // Note: cannot read tensor->data during graph build; only log shapes here to avoid invalid dereference
          if (dbg && t < 2) {
              printf("[SPARSE-DBG-INDICES] t=%lld: top_k=%lld N_kv=%lld\n",
                     (long long) t, (long long) top_k, (long long) N_kv);
          }

          size_t q_off = t * Hq * q_all_2d->nb[1];
          ggml_tensor * q_t_2d = ggml_view_2d(ctx, q_all_2d, Dq, Hq, q_all_2d->nb[1], q_off);

          // Ensure RHS is contiguous for CUDA matmul
          q_t_2d = ggml_cont(ctx, q_t_2d);
          ggml_tensor * scores_t = ggml_mul_mat(ctx, k_sel_2d, q_t_2d); // [Hkv*top_k, Hq]
          // debug marker: scores computed pre-scale
          if (dbg) printf("[SPARSE-DBG-MLA] t=%lld scores pre-scale\n", (long long) t);
          scores_t = ggml_scale(ctx, scores_t, kq_scale);
          // add mask/alibi bias if provided: gather kq_mask rows by indices and add to scores
          if (dbg && (t == 0 || t == T - 1)) {
              cb(scores_t, "mla_scores_pre_mask", -1);
          }

          if (kq_mask && kq_mask->ne[0] == N_kv && kq_mask->ne[1] >= T) {
              // kq_mask is [N_kv, PAD(T)]; take column t -> [N_kv,1]
              ggml_tensor * mask_col = ggml_view_2d(ctx, kq_mask, kq_mask->ne[0], 1, kq_mask->nb[1], t * kq_mask->nb[1]);
              // convert to [1, N_kv] so get_rows yields [1, top_k, 1, 1]
              ggml_tensor * mask_vec = ggml_transpose(ctx, mask_col); // [1, N_kv]
              if (mask_vec->type != scores_t->type) {
                  mask_vec = ggml_cast(ctx, mask_vec, scores_t->type);
              }
              // CUDA get_rows requires row-contiguous source
              mask_vec = ggml_cont(ctx, mask_vec);
              ggml_tensor * mask_rows_4d = ggml_get_rows(ctx, mask_vec, idx_t_4d); // [1, top_k, 1, 1]
              // reshape to [top_k, 1]
              ggml_tensor * mask_rows_2d = ggml_reshape_2d(ctx, mask_rows_4d, top_k, 1); // [top_k,1]

              // build a view of scores' column shape [Hkv*top_k, 1]
              ggml_tensor * scores_col_view = ggml_view_2d(ctx,
                  scores_t,
                  /*ne0*/ Hkv*top_k, /*ne1*/ 1,
                  /*nb1*/ scores_t->nb[1],
                  /*offset*/ 0);

              // repeat mask across heads only -> [Hkv*top_k, 1]
              if (t == 0 || t == T - 1) {
                  cb(scores_t, "mla_scores_post_mask", -1);
              }

              ggml_tensor * mask_bias = ggml_repeat(ctx, mask_rows_2d, scores_col_view); // [Hkv*top_k, 1]

              // add with column broadcast: [Hkv*top_k, Hq] + [Hkv*top_k, 1]
              scores_t = ggml_add(ctx, scores_t, mask_bias);
          } else if (kq_mask) {
              printf("[SPARSE-MLA] Skipping kq_mask: dims mismatched (mask=[%lld,%lld], K N_kv=%lld, T=%lld)\n",
                     (long long) kq_mask->ne[0], (long long) kq_mask->ne[1], (long long) N_kv, (long long) T);
              fflush(stdout);
          }


          // diagnostic: sample masked scores at first/last token
          if (t == 0 || t == T - 1) {
            cb(scores_t, "mla_scores_post_mask_bias", -1);
          }


          // apply tanh softcap if enabled
          if (attn_softcap > 0.0f) {
              scores_t = ggml_scale(ctx, scores_t, 1.0f / attn_softcap);
              scores_t = ggml_tanh(ctx, scores_t);
              scores_t = ggml_scale(ctx, scores_t, attn_softcap);
          }
          // debug marker: scores post-mask/softcap
          if (dbg) printf("[SPARSE-DBG-MLA] t=%lld scores post-mask/softcap\n", (long long) t);
          // Clamp infinities to large finite values to avoid NaNs in softmax when all entries are masked
          scores_t = ggml_clamp(ctx, scores_t, -1e30f, 1e30f);
          ggml_tensor * weights_t = ggml_soft_max(ctx, scores_t);
          // Be conservative: ensure operands of second matmul are contiguous
          weights_t = ggml_cont(ctx, weights_t);
          v_sel_2d  = ggml_cont(ctx, v_sel_2d);

          if (dbg && (t == 0 || t == T - 1)) {
              cb(weights_t, "mla_weights_sample", -1);
          }

          ggml_tensor * out2d_t = ggml_mul_mat(ctx, weights_t, v_sel_2d); // [Hq, Dv]
          ggml_tensor * out2d_t_T = ggml_cont(ctx, ggml_transpose(ctx, out2d_t)); // [Dv, Hq]
          ggml_tensor * out3d_t = ggml_reshape_3d(ctx, out2d_t_T, Dv, Hq, 1);
          // Sanity guard: Dv should be kv_lora_rank for MLA path; for MHA path Dv should be n_embd_head_v
          // We don't have kv_lora_rank here; the caller asserts later. No-op here.

          if (!output_acc) output_acc = out3d_t; else output_acc = ggml_concat(ctx, output_acc, out3d_t, 2);
      }

      cb(output_acc, "kvaware_sparse_attn_out", -1);
      return output_acc;
  }


} // namespace llama
