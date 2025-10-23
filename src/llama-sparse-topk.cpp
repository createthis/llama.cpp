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


      ggml_tensor * mask2d = nullptr;
      if (kq_mask) {
          mask2d = ggml_cont(ctx, kq_mask);
          if (mask2d->ne[0] == N_kv && mask2d->ne[1] >= T) {
              mask2d = ggml_view_2d(ctx, mask2d, N_kv, T, mask2d->nb[1], 0);
          }
          // Only 2D [N_kv, >=T] masks are supported here; fail fast if not satisfied
          GGML_ASSERT(mask2d->ne[0] == N_kv && mask2d->ne[1] >= T);
          GGML_ASSERT(mask2d->nb[0] == (size_t) ggml_type_size(mask2d->type));
      }

      const int64_t k = std::min<int64_t>(top_k, N_kv);
      int64_t TILE_T = 32;
      if (const char *env = getenv("LLAMA_SPARSE_TOPK_TILE_T")) {
          long v = strtol(env, nullptr, 10);
          if (v > 0 && v <= 4096) TILE_T = v;
      }
      printf("[TOPK-INDEXER] N_kv=%lld T=%lld k=%lld TILE_T=%lld H=%lld D=%lld\n",
             (long long) N_kv, (long long) T, (long long) k, (long long) TILE_T, (long long) H, (long long) D);
      fflush(stdout);

      ggml_tensor * result = nullptr; // [k, T]
      for (int64_t t0 = 0; t0 < T; t0 += TILE_T) {
          const int64_t Tc = std::min<int64_t>(TILE_T, T - t0);

          // Q_tile all heads: [D, H*Tc]
          size_t q_off = t0 * H * Q2d_full->nb[1];
          ggml_tensor * Q_tile_all = ggml_view_2d(ctx, Q2d_full, D, H * Tc, Q2d_full->nb[1], q_off);

          // Accumulate per-head contributions into [N_kv, Tc]
          ggml_tensor * scores_tc = nullptr;
          for (int64_t h = 0; h < H; ++h) {
              size_t h_off = h * Tc * Q2d_full->nb[1];
              ggml_tensor * Q_tile_h = ggml_view_2d(ctx, Q_tile_all, D, Tc, Q2d_full->nb[1], h_off);
              ggml_tensor * logits_h = ggml_mul_mat(ctx, k_indexer, Q_tile_h); // [N_kv, Tc]

              // Diagnostics (tile t0==0 only): pre-ReLU magnitude vs positive mass for head h=0
              if (t0 == 0 && h == 0) {
                  ggml_tensor * logits_abs_sum = ggml_sum(ctx, ggml_abs(ctx, logits_h));   // scalar
                  ggml_tensor * logits_relu    = ggml_relu(ctx, logits_h);
                  ggml_tensor * logits_relu_sum= ggml_sum(ctx, logits_relu);               // scalar
                  cb(logits_abs_sum,  "idxkv_logits_pre_abs_sum",  -1);
                  cb(logits_relu_sum, "idxkv_logits_pre_relu_sum", -1);
                  // Also log a small sample of logits_h before ReLU for inspection
                  const int64_t sample_rows = std::min<int64_t>(N_kv, 8);
                  const int64_t sample_cols = std::min<int64_t>(Tc, 4);
                  ggml_tensor * logits_sample = ggml_view_2d(ctx, logits_h, sample_cols, sample_rows, logits_h->nb[1], 0);
                  cb(logits_sample, "idxkv_logits_sample_pre_relu", -1);
                  // Ensure contiguous host buffer for sample reductions to avoid CUDA assert
                  ggml_tensor * logits_sample_host = logits_sample;
                  if (logits_sample->buffer && !ggml_backend_buffer_is_host(logits_sample->buffer)) {
                      ggml_tensor * tmp = ggml_dup_tensor(ctx, logits_sample);
                      ggml_set_name(tmp, "idxkv_logits_sample_pre_relu_host");
                      tmp = ggml_cpy(ctx, logits_sample, tmp);
                      logits_sample_host = tmp;
                  }
                  logits_sample_host = ggml_cont(ctx, logits_sample_host);
                  // Add reductions so the sample is materialized in eval-callback
                  ggml_tensor * logits_sample_sum    = ggml_sum(ctx, logits_sample_host);
                  ggml_tensor * logits_sample_sumsq  = ggml_sum(ctx, ggml_sqr(ctx, logits_sample_host));
                  cb(logits_sample_sum,   "idxkv_logits_sample_pre_relu_sum",   -1);
                  cb(logits_sample_sumsq, "idxkv_logits_sample_pre_relu_sumsq", -1);
                  if (gf) {
                      ggml_set_output(logits_abs_sum);
                      ggml_set_output(logits_relu_sum);
                      ggml_set_output(logits_sample);
                      ggml_set_output(logits_sample_sum);
                      ggml_set_output(logits_sample_sumsq);
                      ggml_build_forward_expand(gf, logits_abs_sum);
                      ggml_build_forward_expand(gf, logits_relu_sum);
                      ggml_build_forward_expand(gf, logits_sample);
                      ggml_build_forward_expand(gf, logits_sample_sum);
                      ggml_build_forward_expand(gf, logits_sample_sumsq);
                  }
                  // restore logits_h to pre-ReLU for normal flow
              }

              logits_h = ggml_relu(ctx, logits_h);

              // weights[h, t0:t0+Tc] -> [1, Tc] -> broadcast to [N_kv, Tc]
              // Bounds checks for per-head slice
              GGML_ASSERT(h >= 0 && h < weights->ne[0]);
              GGML_ASSERT(t0 >= 0 && t0 + Tc <= weights->ne[1]);
              ggml_tensor * w_tile = ggml_view_2d(ctx, weights, 1, Tc, weights->nb[1], h*weights->nb[0] + t0*weights->nb[1]);
              if (t0 == 0 && h == 0) {
                  ggml_tensor * w_tile_abs_sum = ggml_sum(ctx, ggml_abs(ctx, w_tile)); // scalar
                  cb(w_tile_abs_sum, "idxkv_w_tile_abs_sum", -1);
                  if (gf) {
                      ggml_set_output(w_tile_abs_sum);
                      ggml_build_forward_expand(gf, w_tile_abs_sum);
                  }
              }
              ggml_tensor * w_b    = ggml_repeat(ctx, w_tile, logits_h);
              ggml_tensor * contrib = ggml_mul(ctx, logits_h, w_b);

              scores_tc = scores_tc ? ggml_add(ctx, scores_tc, contrib) : contrib;
          }


          // Debug (cb): per-tile scalar sums (no deref)
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

          scores_tc = ggml_cont(ctx, scores_tc);

          // mask tile if available
          if (mask2d) {
              ggml_tensor * mask_tc = ggml_view_2d(ctx, mask2d, N_kv, Tc, mask2d->nb[1], t0 * mask2d->nb[1]);
              mask_tc = ggml_cont(ctx, mask_tc);
              if (mask_tc->type != scores_tc->type) {
                  mask_tc = ggml_cast(ctx, mask_tc, scores_tc->type);
                  mask_tc = ggml_cont(ctx, mask_tc);
              }
              // Ensure both operands have row-contiguous layout for safe broadcast add
              GGML_ASSERT(scores_tc->nb[0] == (size_t) ggml_type_size(scores_tc->type));
              GGML_ASSERT(mask_tc->nb[0]   == (size_t) ggml_type_size(mask_tc->type));
              if (t0 == 0) {
                  printf("[TOPK-INDEXER-DBG] (INDEXER) add mask: scores_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d | mask_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d\n",
                         scores_tc->ne[0], scores_tc->ne[1], scores_tc->nb[0], scores_tc->nb[1], (int) scores_tc->type,
                         mask_tc->ne[0], mask_tc->ne[1], mask_tc->nb[0], mask_tc->nb[1], (int) mask_tc->type);
                  fflush(stdout);
              }
              scores_tc = ggml_add(ctx, scores_tc, mask_tc);
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
          if (scores_tc->buffer && !ggml_backend_buffer_is_host(scores_tc->buffer)) {
              ggml_tensor * host_scores = ggml_dup_tensor(ctx, scores_tc);
              ggml_set_name(host_scores, "idxkv_scores_tc_host");
              host_scores = ggml_cpy(ctx, scores_tc, host_scores);
              scores_for_topk = host_scores;
          }
          if (sched) {
              ggml_backend_t chosen_in = ggml_backend_sched_get_tensor_backend(sched, scores_for_topk);
              const char * in_name = chosen_in ? ggml_backend_name(chosen_in) : NULL;
              printf("[TOPK] chosen backend for scores_for_topk: %s (non-null=%d)\n", in_name ? in_name : "null", chosen_in ? 1 : 0);
              fflush(stdout);
          }
          // Log scores_for_topk (tile 0 only) and reductions to materialize in eval-callback
          if (t0 == 0) {
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
          if (sched && backend_cpu) {
              ggml_backend_sched_set_tensor_backend(sched, topk_tc, backend_cpu);
              const char * bname = ggml_backend_name(backend_cpu);
              printf("[TOPK] assigned backend for topk_tc: %s (non-null=%d)\n", bname ? bname : "null", backend_cpu ? 1 : 0);
              fflush(stdout);
          }
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
          result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
      }

      cb(result, "idxkv_topk_indices_k_T", -1);
      if (gf && result) {
          ggml_set_output(result);
          ggml_build_forward_expand(gf, result);
      }
      // Also provide a float32 view for eval-callback visibility on platforms that skip integer dumps
      if (result) {
          ggml_tensor * result_f32 = ggml_cast(ctx, result, GGML_TYPE_F32);
          cb(result_f32, "idxkv_topk_indices_k_T_f32", -1);
          if (gf) {
              ggml_set_output(result_f32);
              ggml_build_forward_expand(gf, result_f32);
          }
      }
      if (result) {
          printf("SPARSE TOPK KV-AWARE (INDEXER): result topk_indices dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                 result->ne[0], result->ne[1], result->ne[2], result->ne[3], (int)result->type);
          fflush(stdout);
      }
      // ensure final indices tensor is on host: duplicate and copy to host-backed tensor, then prefer CPU via scheduler
      ggml_tensor * result_host = ggml_dup_tensor(ctx, result);
      ggml_set_name(result_host, "kvaware_topk_indices");
      result_host = ggml_cpy(ctx, result, result_host);
      if (sched && backend_cpu) {
          ggml_backend_sched_set_tensor_backend(sched, result_host, backend_cpu);
          const char * bname2 = ggml_backend_name(backend_cpu);
          printf("[TOPK] assigned backend for kvaware_topk_indices: %s (non-null=%d)\n", bname2 ? bname2 : "null", backend_cpu ? 1 : 0);
          fflush(stdout);
      }
      return result_host;
  }

} // namespace llama
