#include "llama-sparse-topk.h"
#include <algorithm>
#include <vector>
#include <cstdint>

#include "llama-impl.h"
#include <cstring>


#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <climits>
namespace {

static inline uint32_t float_to_key_desc(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    // Map float bits to monotonically increasing unsigned keys (ascending order):
    // TileLang-compatible mapping: negative -> bitwise NOT, non-negative -> set sign bit
    if ((int32_t)u < 0) {
        u = ~u;
    } else {
        u |= 0x80000000u;
    }
    return u;
}

struct radix_topk_userdata {
    // currently unused; k is taken from dst->ne[0]
};

static void radix_topk_custom(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void)userdata;
    const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
    const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);
    ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    const int64_t N = src0->ne[0];
    const int64_t k = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);
    const size_t src_nb1 = src0->nb[1];
    const size_t src_nb0 = src0->nb[0];
    const size_t dst_nb1 = dst->nb[1];

    for (int64_t r = ith; r < nr; r += nth) {
        const char * row0 = (const char *)src0->data + r * src_nb1;
        int32_t * out_idx = (int32_t *)((char *)dst->data + r * dst_nb1);
        const int64_t KK = k < N ? k : N;

        // Precompute keys for this row
        std::vector<uint32_t> keys(N);
        for (int64_t i = 0; i < N; ++i) {
            float v = *(const float *)(row0 + (size_t)i*src_nb0);
            keys[i] = float_to_key_desc(v);
        }

        // Stage 1: histogram of high 8 bits (bits 31..24)
        uint32_t counts[256] = {0};
        for (int64_t i = 0; i < N; ++i) {
            uint32_t bin = (keys[i] >> 24) & 0xFFu;
            counts[bin]++;
        }
        // Find threshold bin: number of items with bin > thr0 is sum of counts above thr0
        auto sum_greater = [&](int b){ uint32_t s=0; for (int bb=b+1; bb<256; ++bb) s += counts[bb]; return s; };
        int thr0 = 0;
        uint32_t gt = 0;
        for (int b = 255; b >= 0; --b) {
            uint32_t sgt = sum_greater(b);
            uint32_t eq  = counts[b];
            if (sgt < (uint32_t)KK && sgt + eq >= (uint32_t)KK) { thr0 = b; gt = sgt; break; }
        }
        uint32_t eq0 = counts[thr0];
        int64_t remaining = (int64_t)KK - (int64_t)gt;
        if (remaining < 0) remaining = 0;

        // Collect selected (> thr0) and eq candidates
        std::vector<int32_t> selected; selected.reserve(KK);
        std::vector<int32_t> eq_list; eq_list.reserve(eq0);
        for (int64_t i = 0; i < N; ++i) {
            uint32_t bin = (keys[i] >> 24) & 0xFFu;
            if ((int)bin > thr0) {
                if ((int64_t)selected.size() < KK) selected.push_back((int32_t)i);
            } else if ((int)bin == thr0) {
                eq_list.push_back((int32_t)i);
            }
        }
        remaining = (int64_t)KK - (int64_t)selected.size();

        // Safety check: ensure we have enough candidates to fill K
        if ((int64_t)selected.size() + (int64_t)eq_list.size() < KK) {
            // Fallback: use partial_sort to guarantee correctness
            std::vector<int32_t> idx(N);
            for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(idx.begin(), idx.begin() + KK, idx.end(), cmp);
            for (int64_t i = 0; i < KK; ++i) out_idx[i] = idx[i];
            continue;
        }

        // Tail passes for equal bin
        int shifts[3] = {16, 8, 0};
        for (int pass = 0; pass < 3 && remaining > 0 && !eq_list.empty(); ++pass) {
            uint32_t c2[256] = {0};
            for (int idx : eq_list) {
                uint32_t bin = (keys[idx] >> shifts[pass]) & 0xFFu;
                c2[bin]++;
            }
            auto sum_greater2 = [&](int b){ uint32_t s=0; for (int bb=b+1; bb<256; ++bb) s += c2[bb]; return s; };
            int thr = 255;
            for (int b = 255; b >= 0; --b) {
                uint32_t sgt = sum_greater2(b);
                uint32_t eq  = c2[b];
                if (sgt < (uint32_t)remaining && sgt + eq >= (uint32_t)remaining) { thr = b; break; }
            }
            std::vector<int32_t> next_eq; next_eq.reserve(c2[thr]);
            // Add strictly greater than thr
            for (int idx : eq_list) {
                uint32_t bin = (keys[idx] >> shifts[pass]) & 0xFFu;
                if ((int)bin > thr) {
                    if ((int64_t)selected.size() < KK) { selected.push_back(idx); }
                } else if ((int)bin == thr) {
                    next_eq.push_back(idx);
                }
            }
            eq_list.swap(next_eq);
            remaining = (int64_t)KK - (int64_t)selected.size();
            if ((int64_t)selected.size() + (int64_t)eq_list.size() < remaining) {
                // Fallback safety
                break;
            }
        }
        // Final fill from eq_list if still remaining
        for (int64_t i = 0; i < (int64_t)eq_list.size() && (int64_t)selected.size() < KK; ++i) {
            selected.push_back(eq_list[i]);
        }

        // As a final fallback, if still not enough, use partial_sort
        if ((int64_t)selected.size() < KK) {
            std::vector<int32_t> idx(N);
            for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(idx.begin(), idx.begin() + KK, idx.end(), cmp);
            for (int64_t i = 0; i < KK; ++i) out_idx[i] = idx[i];
            continue;
        }

        // Output first KK indices (order arbitrary)
        for (int64_t i = 0; i < KK; ++i) out_idx[i] = selected[i];

        // Debug: compare with partial_sort for a few rows
        if (r < 8) {
            std::vector<int32_t> ref(N);
            for (int64_t i = 0; i < N; ++i) ref[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(ref.begin(), ref.begin() + KK, ref.end(), cmp);
            if (dbg) {
                printf("[radix debug] row=%lld top: ", (long long)r);
                for (int ii = 0; ii < (int)std::min<int64_t>(8, KK); ++ii) printf("%d ", out_idx[ii]);
                printf("| ref: ");
                for (int ii = 0; ii < (int)std::min<int64_t>(8, KK); ++ii) printf("%d ", ref[ii]);
                printf("\n");
                fflush(stdout);
            }
        }
    }
}

} // anonymous namespace


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
          ggml_tensor * tmp_mask = kq_mask;
          cb(tmp_mask, "idxkv_mask2d", -1);
          if (tmp_mask->ne[0] == N_kv && tmp_mask->ne[1] >= T) {
              mask_full = tmp_mask; // use original mask; slice per tile and only contiguize the tile
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

          // Per-head weighted sum path to avoid giant broadcast tensors
          {
              ggml_tensor * scores_acc = nullptr;
              for (int64_t h_idx = 0; h_idx < H; ++h_idx) {
                  size_t off_h = (size_t)h_idx * logits_act->nb[1];
                  ggml_tensor * logits_h = ggml_view_2d(ctx, logits_act, N_kv, Tc, logits_act->nb[2], off_h);

                  size_t w_off_h = (size_t)h_idx * w_slice->nb[0];
                  ggml_tensor * w_row = ggml_view_2d(ctx, w_slice, 1, Tc, w_slice->nb[1], w_off_h);
                  ggml_tensor * w_row_b = ggml_repeat(ctx, w_row, logits_h);
                  ggml_tensor * contrib_h = ggml_mul(ctx, logits_h, w_row_b);
                  scores_acc = scores_acc ? ggml_add(ctx, scores_acc, contrib_h) : contrib_h;
              }
              scores_tc = scores_acc;
          }



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
          // Ensure contiguous for CUDA op only if needed, then clamp infinities
          ggml_tensor * scores_pre = ggml_is_contiguous(scores_for_topk) ? scores_for_topk : ggml_cont(ctx, scores_for_topk);
          ggml_tensor * scores_clamped = ggml_clamp(ctx, scores_pre, -1e30f, 1e30f);
          // Compute top-k indices via CUDA radix selection
          ggml_tensor * topk_tc = ggml_sparse_topk_radix(ctx, scores_clamped, k);
          if (dbg && t0 == 0) {
              cb(topk_tc, "idxkv_topk_radix", -1);
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



ggml_tensor * llama::sparse_attn_topk::topk_radix_indices(
    ggml_context * ctx,
    ggml_tensor * scores, // [N, T]
    int64_t k) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32);
    ggml_tensor * args[1] = { scores };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_I32,
        /*ne0*/ k,
        /*ne1*/ scores->ne[1],
        /*ne2*/ 1,
        /*ne3*/ 1,
        args,
        /*n_args*/ 1,
        /*fun*/ radix_topk_custom,
        /*n_tasks*/ GGML_N_TASKS_MAX,
        /*userdata*/ nullptr);
}

} // namespace llama
