#include "llama-sparse-topk.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace llama {

using std::function;


  ggml_tensor * sparse_attn_topk::select_topk_tokens_indexer_kvaware(
      ggml_context * ctx,
      ggml_tensor * q_indexer,   // [D, H, T]
      ggml_tensor * k_indexer,   // [D, N_kv]
      ggml_tensor * weights,     // [H, T]
      ggml_tensor * kq_mask,     // [N_kv, T] or [N_kv, PAD(T)]
      int64_t top_k,
      const std::function<void(ggml_tensor *, const char *, int)> & cb,
      ggml_cgraph * gf,
      ggml_backend_sched_t sched,
      ggml_backend_t backend_cpu) {
      const int64_t D    = q_indexer->ne[0];
      const int64_t H    = q_indexer->ne[1];
      const int64_t T    = q_indexer->ne[2];
      const int64_t N_kv = k_indexer->ne[1];

      const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
      const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);

      if (dbg) {
          printf("SPARSE TOPK KV-AWARE (INDEXER): q_indexer [D,H,T]=[%" PRId64 ",%" PRId64 ",%" PRId64 "]\n", D, H, T);
          printf("SPARSE TOPK KV-AWARE (INDEXER): k_indexer dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                 k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
          printf("SPARSE TOPK KV-AWARE (INDEXER): weights [H,T]=[%" PRId64 ",%" PRId64 "]\n",
                 weights ? weights->ne[0] : -1, weights ? weights->ne[1] : -1);
          if (kq_mask) {
              printf("SPARSE TOPK KV-AWARE (INDEXER): kq_mask dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                     kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3], (int)kq_mask->type);
          }
          fflush(stdout);
      }

      // Shape/contiguity assertions for weights [H, T]
      GGML_ASSERT(D > 0 && H > 0 && T > 0 && N_kv > 0);
      GGML_ASSERT(weights != nullptr);
      GGML_ASSERT(weights->ne[0] == H);
      GGML_ASSERT(weights->ne[1] >= T);
      GGML_ASSERT(weights->nb[0] == (size_t) ggml_type_size(weights->type));
      GGML_ASSERT(weights->nb[1] == (size_t) ggml_row_size(weights->type, weights->ne[0]));
      // Ensure indexer K depth matches indexer Q depth
      GGML_ASSERT(k_indexer->ne[0] == D);
      // KV indexer currently expected as 2D [D, N_kv] or 3D with singleton stream
      GGML_ASSERT(k_indexer->ne[2] <= 1);


      // Q as [D, H*T]
      ggml_tensor * q_perm   = ggml_permute(ctx, q_indexer, 0, 2, 1, 3);   // [D, T, H]
      ggml_tensor * q_cont   = ggml_cont(ctx, q_perm);
      ggml_tensor * Q2d_full = ggml_reshape_2d(ctx, q_cont, D, T*H);
      cb(Q2d_full, "idxkv_Q2d_full", -1);


      // Diagnostics: sample K indexer head/tail once per call
      if (dbg) {
          const int64_t d0 = std::min<int64_t>(k_indexer->ne[0], (int64_t)8);
          const int64_t c0 = std::min<int64_t>(k_indexer->ne[1], (int64_t)8);
          // head columns
          ggml_tensor * kidx_head = ggml_view_2d(ctx, k_indexer, d0, c0, k_indexer->nb[1], 0);
          cb(kidx_head, "idxkv_k_indexer_head", -1);
          if (gf) { ggml_set_output(kidx_head); ggml_build_forward_expand(gf, kidx_head); }
          // tail columns
          if (k_indexer->ne[1] > c0) {
              size_t off_tail = (k_indexer->ne[1] - c0) * k_indexer->nb[1];
              ggml_tensor * kidx_tail = ggml_view_2d(ctx, k_indexer, d0, c0, k_indexer->nb[1], off_tail);
              cb(kidx_tail, "idxkv_k_indexer_tail", -1);
              if (gf) { ggml_set_output(kidx_tail); ggml_build_forward_expand(gf, kidx_tail); }
          }
      }

      ggml_tensor * mask_full = nullptr;
      if (kq_mask) {
          cb(kq_mask, "idxkv_kq_mask", -1);
          ggml_tensor * tmp_mask = ggml_cont(ctx, kq_mask);
          cb(tmp_mask, "idxkv_mask2d", -1);
          if (tmp_mask->ne[0] == N_kv && tmp_mask->ne[1] >= T) {
              mask_full = tmp_mask; // full-width [N_kv, PAD(T)] for slicing per tile
          } else {
              printf("[TOPK-INDEXER] kq_mask dims [%lld,%lld] mismatch N_kv=%lld,T=%lld; ignoring mask for indexer selection\n",
                     (long long) tmp_mask->ne[0], (long long) tmp_mask->ne[1], (long long) N_kv, (long long) T);
              fflush(stdout);
              mask_full = nullptr;
          }
      }

      const int64_t k = std::min<int64_t>(top_k, N_kv);
      int64_t TILE_T = 128; // larger default tile improves GEMM utilization; overridable via env
      if (const char *env = getenv("LLAMA_SPARSE_TOPK_TILE_T")) {
          long v = strtol(env, nullptr, 10);
          if (v > 0 && v <= 4096) TILE_T = v;
      }
      if (dbg) {
          printf("[TOPK-INDEXER] N_kv=%lld T=%lld k=%lld TILE_T=%lld H=%lld D=%lld\n",
                 (long long) N_kv, (long long) T, (long long) k, (long long) TILE_T, (long long) H, (long long) D);
          fflush(stdout);
      }

      // K-scale proxy: RMS over D for each KV column
      ggml_tensor * k_sqr = ggml_sqr(ctx, k_indexer);                  // [D, N_kv]
      ggml_tensor * k_sum = ggml_sum_rows(ctx, k_sqr);                  // [1, N_kv]
      ggml_tensor * k_mean = ggml_scale(ctx, k_sum, 1.0f / (float) D);  // [1, N_kv]
      ggml_tensor * k_scale_vec = ggml_sqrt(ctx, k_mean);               // [1, N_kv]
      ggml_tensor * k_scale_2d = ggml_transpose(ctx, k_scale_vec);      // [N_kv, 1]
      k_scale_2d = ggml_cont(ctx, k_scale_2d);
      cb(k_scale_2d, "idxkv_k_scale_proxy", -1);

      ggml_tensor * result = nullptr; // [k, T]
      for (int64_t t0 = 0; t0 < T; t0 += TILE_T) {
          const int64_t Tc = std::min<int64_t>(TILE_T, T - t0);

          // Q_tile all heads: [D, H*Tc]
          size_t q_off = t0 * H * Q2d_full->nb[1];
          ggml_tensor * Q_tile_all = ggml_view_2d(ctx, Q2d_full, D, H * Tc, Q2d_full->nb[1], q_off);

          // Accumulate per-head contributions into [N_kv, Tc]
          ggml_tensor * scores_tc = nullptr; // unused placeholder
          // Compute logits for all heads in one GEMM: [N_kv, H*Tc]
          ggml_tensor * logits_all = ggml_mul_mat(ctx, k_indexer, Q_tile_all);
          if (dbg && t0 == 0) {
              cb(logits_all, "idxkv_logits_all", -1);
          }
          // Reshape and apply ReLU: [N_kv, H, Tc]
          ggml_tensor * logits_resh = ggml_reshape_3d(ctx, logits_all, N_kv, H, Tc);
          ggml_tensor * logits_act  = ggml_relu(ctx, logits_resh);
          // Weights slice [H, Tc] and broadcast-mul, then sum over H → [N_kv, Tc]
          ggml_tensor * w_slice = ggml_view_2d(ctx, weights, H, Tc, weights->nb[1], t0*weights->nb[1]);

          // reshape to [1, H, Tc] so it can broadcast across N_kv
          ggml_tensor * w3 = ggml_reshape_3d(ctx, w_slice, 1, H, Tc);
          ggml_tensor * w_bcast = ggml_repeat(ctx, w3, logits_act);
          ggml_tensor * contrib  = ggml_mul(ctx, logits_act, w_bcast);   // [N_kv, H, Tc]
          // Sum over head dimension (ne1): permute to [H, N_kv, Tc] and sum rows
          ggml_tensor * contrib_perm = ggml_permute(ctx, contrib, 1, 0, 2, 3);
          contrib_perm = ggml_cont(ctx, contrib_perm);
          ggml_tensor * sum_h = ggml_sum_rows(ctx, contrib_perm);        // [1, N_kv, Tc]
          scores_tc = ggml_reshape_2d(ctx, sum_h, N_kv, Tc);             // [N_kv, Tc]


          // Safe K-scale proxy application after head reduction (always apply)
          {
              ggml_tensor * k_scale_bcast = ggml_repeat(ctx, k_scale_2d, scores_tc); // [N_kv, Tc]
              scores_tc = ggml_mul(ctx, scores_tc, k_scale_bcast);
          }

          // Debug (cb): per-tile scalar sums (no deref)
          if (dbg) {
              ggml_tensor * idxkv_scores_sum = ggml_sum(ctx, scores_tc);
              ggml_tensor * idxkv_scores_ssq = ggml_sum(ctx, ggml_sqr(ctx, scores_tc));
              ggml_tensor * idxkv_scores_post_abs_sum = nullptr;
              if (t0 == 0) {
                  idxkv_scores_post_abs_sum = ggml_sum(ctx, ggml_abs(ctx, scores_tc));
                  cb(idxkv_scores_post_abs_sum, "idxkv_scores_post_abs_sum", -1);
                  if (gf) {
                      ggml_set_output(idxkv_scores_post_abs_sum);
                      ggml_build_forward_expand(gf, idxkv_scores_post_abs_sum);
                  }
              }
              cb(idxkv_scores_sum,  "idxkv_scores_sum",  -1);
              cb(idxkv_scores_ssq,  "idxkv_scores_ssq",  -1);
          }

          scores_tc = ggml_cont(ctx, scores_tc);

          // mask tile if available
          if (mask_full) {
              ggml_tensor * mask_tc = ggml_view_2d(ctx, mask_full, N_kv, Tc, mask_full->nb[1], t0 * mask_full->nb[1]);
              mask_tc = ggml_cont(ctx, mask_tc);
              if (mask_tc->type != scores_tc->type) {
                  mask_tc = ggml_cast(ctx, mask_tc, scores_tc->type);
                  mask_tc = ggml_cont(ctx, mask_tc);
              }
              if (dbg) cb(mask_tc,  "idxkv_mask_tc",  -1);

              // Ensure both operands have row-contiguous layout for safe broadcast add
              GGML_ASSERT(scores_tc->nb[0] == (size_t) ggml_type_size(scores_tc->type));
              GGML_ASSERT(mask_tc->nb[0]   == (size_t) ggml_type_size(mask_tc->type));
              if (dbg && t0 == 0) {
                  printf("[TOPK-INDEXER-DBG] (INDEXER) add mask: scores_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d | mask_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d\n",
                         scores_tc->ne[0], scores_tc->ne[1], scores_tc->nb[0], scores_tc->nb[1], (int) scores_tc->type,
                         mask_tc->ne[0], mask_tc->ne[1], mask_tc->nb[0], mask_tc->nb[1], (int) mask_tc->type);
                  fflush(stdout);
              }
              scores_tc = ggml_add(ctx, scores_tc, mask_tc);
              // Clamp after mask to avoid inf in diagnostics and stabilize top-k
              scores_tc = ggml_clamp(ctx, scores_tc, -1e30f, 1e30f);

              if (t0 == 0) {
                  ggml_tensor * idxkv_scores_post_mask_abs_sum = ggml_sum(ctx, ggml_abs(ctx, scores_tc));
                  cb(idxkv_scores_post_mask_abs_sum, "idxkv_scores_post_mask_abs_sum", -1);
                  if (gf) {
                      ggml_set_output(idxkv_scores_post_mask_abs_sum);
                      ggml_build_forward_expand(gf, idxkv_scores_post_mask_abs_sum);
                  }
              }
          }

          // top-k for this tile -> [k, Tc]
          // Ensure top-k runs on CPU to avoid CUDA backend returning invalid indices for generic shapes
          ggml_tensor * scores_for_topk = scores_tc;
          // Keep on device by default; only copy to host in debug mode
          if (dbg && scores_tc->buffer && !ggml_backend_buffer_is_host(scores_tc->buffer)) {
              ggml_tensor * host_scores = ggml_dup_tensor(ctx, scores_tc);
              ggml_set_name(host_scores, "idxkv_scores_tc_host");
              host_scores = ggml_cpy(ctx, scores_tc, host_scores);
              scores_for_topk = host_scores;
          }
          if (dbg && sched) {
              ggml_backend_t chosen_in = ggml_backend_sched_get_tensor_backend(sched, scores_for_topk);
              const char * in_name = chosen_in ? ggml_backend_name(chosen_in) : NULL;
              printf("[TOPK] chosen backend for scores_for_topk: %s (non-null=%d)\n", in_name ? in_name : "null", chosen_in ? 1 : 0);
              fflush(stdout);
          }
          // Log scores_for_topk (tile 0 only) and reductions to materialize in eval-callback
          if (dbg && t0 == 0) {
              cb(scores_for_topk, "idxkv_scores_for_topk", -1);
              ggml_tensor * sft_sum   = ggml_sum(ctx, scores_for_topk);
              ggml_tensor * sft_sumsq = ggml_sum(ctx, ggml_sqr(ctx, scores_for_topk));
              cb(sft_sum,   "idxkv_scores_for_topk_sum",   -1);
              cb(sft_sumsq, "idxkv_scores_for_topk_sumsq", -1);
              if (gf) {
                  ggml_set_output(scores_for_topk);
                  ggml_set_output(sft_sum);
                  ggml_set_output(sft_sumsq);
                  ggml_build_forward_expand(gf, scores_for_topk);
                  ggml_build_forward_expand(gf, sft_sum);
                  ggml_build_forward_expand(gf, sft_sumsq);
              }
          }
          // Clamp infinities from mask to large finite values to stabilize argsort/top-k
          ggml_tensor * scores_clamped = ggml_clamp(ctx, scores_for_topk, -1e30f, 1e30f);
          ggml_tensor * topk_tc = ggml_top_k(ctx, scores_clamped, k);
          if (dbg) {
              cb(topk_tc->src[0], "idxkv_argsort", -1);
              cb(topk_tc, "idxkv_topk", -1);
              if (t0 == 0) {
                  ggml_tensor * topk_f32 = ggml_cast(ctx, topk_tc, GGML_TYPE_F32);
                  ggml_tensor * idxkv_topk_idx_sum    = ggml_sum(ctx, topk_f32);
                  ggml_tensor * idxkv_topk_idx_sumsq  = ggml_sum(ctx, ggml_sqr(ctx, topk_f32));
                  cb(idxkv_topk_idx_sum,   "idxkv_topk_idx_sum",   -1);
                  cb(idxkv_topk_idx_sumsq, "idxkv_topk_idx_sumsq", -1);
                  if (gf) {
                      ggml_set_output(idxkv_topk_idx_sum);
                      ggml_set_output(idxkv_topk_idx_sumsq);
                      ggml_build_forward_expand(gf, idxkv_topk_idx_sum);
                      ggml_build_forward_expand(gf, idxkv_topk_idx_sumsq);
                  }
              }
          }
          result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
      }

      cb(result, "idxkv_topk_indices_k_T", -1);
      if (gf && result) {
          ggml_set_output(result);
          ggml_build_forward_expand(gf, result);
      }
      // Also provide a float32 view for eval-callback visibility on platforms that skip integer dumps
      if (dbg && result) {
          ggml_tensor * result_f32 = ggml_cast(ctx, result, GGML_TYPE_F32);
          cb(result_f32, "idxkv_topk_indices_k_T_f32", -1);
          if (gf) {
              ggml_set_output(result_f32);
              ggml_build_forward_expand(gf, result_f32);
          }
      }
      if (dbg && result) {
          printf("SPARSE TOPK KV-AWARE (INDEXER): result topk_indices dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                 result->ne[0], result->ne[1], result->ne[2], result->ne[3], (int)result->type);
          fflush(stdout);
      }
      // Keep indices on device by default to avoid host syncs during get_rows
      return result;
  }

} // namespace llama
