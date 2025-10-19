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
      ggml_tensor * Q2d = ggml_reshape_2d(ctx, q_cur,   Dq, Hq * T);
      cb(K2d, "kvaware_K2d", -1);
      cb(Q2d, "kvaware_Q2d", -1);

      // Adapt Q rows to match Dk if necessary
      if (Dq != Dk) {
          if (Dq < Dk) {
              Q2d = ggml_pad(ctx, Q2d, Dk - Dq, 0, 0, 0);
          } else {
              Q2d = ggml_view_2d(ctx, Q2d, Dk, Hq*T, Q2d->nb[1], 0);
          }
          cb(Q2d, "kvaware_Q2d_adapted", -1);
      }

      ggml_tensor * logits = ggml_mul_mat(ctx, K2d, Q2d); // [Hkv*N_kv, Hq*T]
      cb(logits, "kvaware_logits_KNq_HqT", -1);

      ggml_tensor * logits4 = ggml_reshape_4d(ctx, logits, N_kv, Hkv, Hq, T);
      cb(logits4, "kvaware_logits4_Nkv_Hkv_Hq_T", -1);

      // Sum across Hkv, Hq -> [N_kv, T]
      ggml_tensor * tmp1 = ggml_permute(ctx, logits4, 1, 0, 2, 3); // [Hkv, N_kv, Hq, T]
      tmp1 = ggml_cont(ctx, tmp1);
      ggml_tensor * sum1 = ggml_sum_rows(ctx, tmp1);               // [1, N_kv, Hq, T]
      ggml_tensor * logits3 = ggml_reshape_3d(ctx, sum1, N_kv, Hq, T);
      ggml_tensor * tmp2 = ggml_permute(ctx, logits3, 1, 0, 2, 3); // [Hq, N_kv, T, 1]
      tmp2 = ggml_cont(ctx, tmp2);
      ggml_tensor * sum2 = ggml_sum_rows(ctx, tmp2);               // [1, N_kv, T, 1]
      ggml_tensor * logits2 = ggml_reshape_2d(ctx, sum2, N_kv, T); // [N_kv, T]
      cb(logits2, "kvaware_logits2_Nkv_T", -1);

      ggml_tensor * logits_masked = logits2;
      if (kq_mask) {
          ggml_tensor * mask = ggml_cont(ctx, kq_mask);
          if (mask->ne[0] == N_kv && mask->ne[1] >= T) {
              mask = ggml_view_2d(ctx, mask, N_kv, T, mask->nb[1], 0);
          }
          logits_masked = ggml_add(ctx, logits2, mask);
      }
      cb(logits_masked, "kvaware_logits_masked_Nkv_T", -1);

      const int64_t k = std::min<int64_t>(top_k, N_kv);
      ggml_tensor * topk_indices = ggml_top_k(ctx, logits_masked, k); // [k, T]
      cb(topk_indices, "kvaware_topk_indices_k_T", -1);
      return topk_indices;
  }


} // namespace llama
