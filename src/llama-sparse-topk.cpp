#include "llama-sparse-topk.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>

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
      const int64_t TILE_T = 128; // tuneable
      ggml_tensor * result = nullptr; // [k, T]
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
          ggml_tensor * sum_hq    = ggml_sum_rows(ctx, perm_q);                        // [1, N_kv, Tc]
          ggml_tensor * scores_tc = ggml_reshape_2d(ctx, sum_hq, N_kv, Tc);            // [N_kv, Tc]

          // apply mask tile if available: [N_kv, Tc]
          if (mask2d) {
              ggml_tensor * mask_tc = ggml_view_2d(ctx, mask2d, N_kv, Tc, mask2d->nb[1], t0 * mask2d->nb[1]);
              scores_tc = ggml_add(ctx, scores_tc, mask_tc);
          }

          // top-k for this tile -> [k, Tc]
          ggml_tensor * topk_tc = ggml_top_k(ctx, scores_tc, k);

          // accumulate across tiles along dim 1
          result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
      }

      cb(result, "kvaware_topk_indices_k_T", -1);
      return result;
  }


} // namespace llama
