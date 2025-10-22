#include "llama-sparse-topk.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace llama {

using std::function;

// Select per-query top-k indices from a [n_kv, n_q] logits matrix.
// Applies a causal mask (no attending to future positions) before top-k.

ggml_tensor * sparse_attn_topk::select_topk_tokens(
    ggml_context * ctx,
    ggml_tensor * token_importance,
    int64_t n_tokens,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    (void)n_tokens; // Unused parameter

    printf("SPARSE TOPK: Starting select_topk_tokens\n");
    printf("SPARSE TOPK: token_importance shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           token_importance->ne[0], token_importance->ne[1], token_importance->ne[2], token_importance->ne[3]);
    fflush(stdout);

    // Apply causal mask so queries cannot attend to positions s > t within the block
    ggml_tensor * logits_masked = ggml_diag_mask_inf(ctx, token_importance, /*n_past=*/0);
    cb(logits_masked, "topk_logits_masked", -1);

    const int64_t n_kv = logits_masked->ne[0];
    const int64_t n_q  = logits_masked->ne[1];
    const int64_t top_k = (64 < n_kv) ? 64 : n_kv; // default k=64 capped by available keys

    printf("SPARSE TOPK: n_kv=%" PRId64 ", n_q=%" PRId64 ", top_k=%" PRId64 "\n", n_kv, n_q, top_k);
    fflush(stdout);

    // ggml_top_k over a 2D tensor returns [top_k, n_q, 1, 1] (per-column top-k)
    ggml_tensor * topk_indices = ggml_top_k(ctx, logits_masked, top_k);
    cb(topk_indices, "topk_indices", -1);

    printf("SPARSE TOPK: Final topk_indices shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    fflush(stdout);

    return topk_indices;
}

  // KV-aware: compute logits over full KV cache and select top-k absolute indices per query
  ggml_tensor * sparse_attn_topk::select_topk_tokens_kvaware(
      ggml_context * ctx,
      ggml_tensor * q_cur,
      ggml_tensor * k_cache,
      ggml_tensor * kq_mask,
      int64_t top_k,
      const function<void(ggml_tensor *, const char *, int)> & cb) {
      printf("SPARSE TOPK KV-AWARE: q_cur [Dq,Hq,T]=[%" PRId64 ",%" PRId64 ",%" PRId64 "]\n", q_cur->ne[0], q_cur->ne[1], q_cur->ne[2]);
      printf("SPARSE TOPK KV-AWARE: k_cache [Dk,Hkv,N_kv]=[%" PRId64 ",%" PRId64 ",%" PRId64 "]\n", k_cache->ne[0], k_cache->ne[1], k_cache->ne[2]);
      fflush(stdout);

      const int64_t Dq   = q_cur->ne[0];
      const int64_t Hq   = q_cur->ne[1];
      const int64_t T    = q_cur->ne[2];
      const int64_t Dk   = k_cache->ne[0];
      const int64_t Hkv  = k_cache->ne[1];
      const int64_t N_kv = k_cache->ne[2];

      ggml_tensor * K2d = ggml_reshape_2d(ctx, k_cache, Dk, Hkv * N_kv);
      cb(K2d, "kvaware_K2d", -1);
      // For Hkv == 1, view K as [Dk, N_kv]
      ggml_tensor * Ksum2d = nullptr;
      if (Hkv == 1) {
          Ksum2d = ggml_view_2d(ctx, K2d, Dk, N_kv, K2d->nb[1], 0);
      }

      // Prepare mask as 2D [N_kv, T] if provided
      ggml_tensor * mask2d = nullptr;
      if (kq_mask) {
          mask2d = ggml_cont(ctx, kq_mask);
          if (mask2d->ne[0] == N_kv && mask2d->ne[1] >= T) {
              mask2d = ggml_view_2d(ctx, mask2d, N_kv, T, mask2d->nb[1], 0);
          }
          // Only support 2D [N_kv, >=T] here
          GGML_ASSERT(mask2d->ne[0] == N_kv && mask2d->ne[1] >= T);
          GGML_ASSERT(mask2d->nb[0] == (size_t) ggml_type_size(mask2d->type));
      }

      // Vectorized implementation across all T to avoid O(T) node explosion
      const int64_t k = std::min<int64_t>(top_k, N_kv);

      // 2D view of Q: [Dq, Hq*T]
      ggml_tensor * Q2d_full = ggml_reshape_2d(ctx, q_cur, Dq, Hq * T);
      cb(Q2d_full, "kvaware_Q2d", -1);

      // Adapt Q rows to match Dk if necessary across the entire view
      if (Dq != Dk) {
          if (Dq < Dk) {
              Q2d_full = ggml_pad(ctx, Q2d_full, Dk - Dq, 0, 0, 0);
          } else {
              Q2d_full = ggml_view_2d(ctx, Q2d_full, Dk, Hq * T, Q2d_full->nb[1], 0);
          }
      }

      // Chunked computation across tokens to avoid materializing [N_kv, Hq*T]
      int64_t TILE_T = 32; // tuneable; can override via LLAMA_SPARSE_TOPK_TILE_T
      if (const char *env = getenv("LLAMA_SPARSE_TOPK_TILE_T")) {
          long v = strtol(env, nullptr, 10);
          if (v > 0 && v <= 4096) TILE_T = v;
      }
      ggml_tensor * result = nullptr; // [k, T]
      printf("[TOPK-QK] N_kv=%lld T=%lld k=%lld TILE_T=%lld Hq=%lld Dq=%lld\n",
             (long long) N_kv, (long long) T, (long long) k, (long long) TILE_T, (long long) Hq, (long long) Dq);
      fflush(stdout);

      for (int64_t t0 = 0; t0 < T; t0 += TILE_T) {
          const int64_t Tc = std::min<int64_t>(TILE_T, T - t0);
          // Q_tile: [Dk, Hq*Tc]
          size_t q_off = t0 * Hq * Q2d_full->nb[1];
          ggml_tensor * Q_tile = ggml_view_2d(ctx, Q2d_full, Dk, Hq * Tc, Q2d_full->nb[1], q_off);

          // logits_tile: [*, Hq*Tc]
          ggml_tensor * logits_tile = nullptr;
          if (Hkv == 1) {
              logits_tile = ggml_mul_mat(ctx, Ksum2d, Q_tile);                // [N_kv, Hq*Tc]
          } else {
              ggml_tensor * tmp = ggml_mul_mat(ctx, K2d, Q_tile);             // [Hkv*N_kv, Hq*Tc]
              ggml_tensor * tmp3 = ggml_reshape_3d(ctx, tmp, Hkv, N_kv, Hq*Tc); // [Hkv, N_kv, Hq*Tc]
              ggml_tensor * sum_hkv = ggml_sum_rows(ctx, tmp3);               // [1, N_kv, Hq*Tc]
              logits_tile = ggml_reshape_2d(ctx, sum_hkv, N_kv, Hq*Tc);       // [N_kv, Hq*Tc]
          }

          // Sum over Hq -> [N_kv, Tc]
          ggml_tensor * logits3_q = ggml_reshape_3d(ctx, logits_tile, N_kv, Hq, Tc); // [N_kv, Hq, Tc]
          ggml_tensor * perm_q    = ggml_permute(ctx, logits3_q, 1, 0, 2, 3);        // [Hq, N_kv, Tc]
          perm_q = ggml_cont(ctx, perm_q);
          ggml_tensor * sum_hq    = ggml_sum_rows(ctx, perm_q);                      // [1, N_kv, Tc]
          ggml_tensor * scores_tc = ggml_reshape_2d(ctx, sum_hq, N_kv, Tc);          // [N_kv, Tc]
          // Ensure CUDA-friendly layout for subsequent broadcast add with mask
          scores_tc = ggml_cont(ctx, scores_tc);

          // apply mask tile if available: [N_kv, Tc]
          if (mask2d) {
              ggml_tensor * mask_tc = ggml_view_2d(ctx, mask2d, N_kv, Tc, mask2d->nb[1], t0 * mask2d->nb[1]);
              // ensure CUDA-friendly contiguous strides for broadcast add (src1 must have nb0 == sizeof(type))
              mask_tc   = ggml_cont(ctx, mask_tc);
              // scores as src0 should also be contiguous rows to avoid scheduler selecting awkward views
              scores_tc = ggml_cont(ctx, scores_tc);
              // ensure both operands have the same data type to avoid mixed-type CUDA kernels
              if (mask_tc->type != scores_tc->type) {
                  mask_tc = ggml_cast(ctx, mask_tc, scores_tc->type);
                  mask_tc = ggml_cont(ctx, mask_tc);
              }
              // debug: log operand layouts for the first tile only
              if (t0 == 0) {
                  printf("[TOPK-INDEXER-DBG] add mask: scores_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d | mask_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d\n",
                         scores_tc->ne[0], scores_tc->ne[1], scores_tc->nb[0], scores_tc->nb[1], (int)scores_tc->type,
                         mask_tc->ne[0], mask_tc->ne[1], mask_tc->nb[0], mask_tc->nb[1], (int)mask_tc->type);
                  fflush(stdout);
              }
              scores_tc = ggml_add(ctx, scores_tc, mask_tc);
          }

          // top-k for this tile -> [k, Tc]
          ggml_tensor * topk_tc = ggml_top_k(ctx, scores_tc, k);

          // accumulate across tiles along dim 1
          result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
      }

      cb(result, "kvaware_topk_indices_k_T", -1);
      if (result) {
          printf("SPARSE TOPK KV-AWARE (QK): result topk_indices dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                 result->ne[0], result->ne[1], result->ne[2], result->ne[3], (int)result->type);
          fflush(stdout);
      }
      return result;
  }




  ggml_tensor * sparse_attn_topk::select_topk_tokens_indexer_kvaware(
      ggml_context * ctx,
      ggml_tensor * q_indexer,   // [D, H, T]
      ggml_tensor * k_indexer,   // [D, N_kv]
      ggml_tensor * weights,     // [H, T]
      ggml_tensor * kq_mask,     // [N_kv, T] or [N_kv, PAD(T)]
      int64_t top_k,
      const std::function<void(ggml_tensor *, const char *, int)> & cb) {
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
              logits_h = ggml_relu(ctx, logits_h);

              // weights[h, t0:t0+Tc] -> [1, Tc] -> broadcast to [N_kv, Tc]
              // Bounds checks for per-head slice
              GGML_ASSERT(h >= 0 && h < weights->ne[0]);
              GGML_ASSERT(t0 >= 0 && t0 + Tc <= weights->ne[1]);
              ggml_tensor * w_tile = ggml_view_2d(ctx, weights, 1, Tc, weights->nb[1], h*weights->nb[0] + t0*weights->nb[1]);
              ggml_tensor * w_b    = ggml_repeat(ctx, w_tile, logits_h);
              ggml_tensor * contrib = ggml_mul(ctx, logits_h, w_b);

              scores_tc = scores_tc ? ggml_add(ctx, scores_tc, contrib) : contrib;
          }

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
          }

          // top-k for this tile -> [k, Tc]
          ggml_tensor * topk_tc = ggml_top_k(ctx, scores_tc, k);
          result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
      }

      cb(result, "idxkv_topk_indices_k_T", -1);
      if (result) {
          printf("SPARSE TOPK KV-AWARE (INDEXER): result topk_indices dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                 result->ne[0], result->ne[1], result->ne[2], result->ne[3], (int)result->type);
          fflush(stdout);
      }
      return result;
  }

} // namespace llama
