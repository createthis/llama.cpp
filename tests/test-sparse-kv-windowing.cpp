#include "../src/llama-sparse-topk.h"

#include <ggml.h>
#include <ggml-cpu.h>

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace llama;

// Synthetic single-layer, long-KV style test that verifies:
// - derive_kv_windows produces per-token [start,end) consistent with a
//   hand-constructed kq_mask.
// - select_topk_tokens_indexer_kvaware never returns indices outside the
//   allowed window implied by the mask.
//
// This test is CPU-focused and does not depend on a full llama_kv_cache
// or model; it operates directly on the sparse-attn indexer/top-k helpers.

int main() {
    std::printf("[sparse-kv-windowing] starting test...\\n");
    std::fflush(stdout);

    const int64_t N_kv    = 128;   // number of KV positions
    const int64_t T       = 8;     // number of tokens
    const int64_t D_index = 32;    // indexer head dim
    const int64_t H_index = 4;     // indexer head count
    const int64_t top_k   = 8;     // top-k per token

    ggml_init_params p{};
    p.mem_size   = 64ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q_indexer = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_index, H_index, T);
    ggml_tensor * k_indexer = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_index, N_kv);
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_index, T);
    ggml_tensor * kq_mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_kv, T);

    // Fill q_indexer, k_indexer, weights with simple deterministic patterns.
    // Exact values are unimportant; we only care about mask-respecting behaviour.
    float * q_data = (float *) q_indexer->data;
    float * k_data = (float *) k_indexer->data;
    float * w_data = (float *) weights->data;
    float * m_data = (float *) kq_mask->data;

    for (int64_t i = 0; i < ggml_nelements(q_indexer); ++i) {
        q_data[i] = 0.01f * float(i % 113);
    }
    for (int64_t i = 0; i < ggml_nelements(k_indexer); ++i) {
        k_data[i] = 0.02f * float((i * 7) % 97);
    }
    for (int64_t i = 0; i < ggml_nelements(weights); ++i) {
        w_data[i] = 0.03f * float((i * 13) % 89);
    }

    // Build per-token KV windows: for token t, only indices [0, W_t) are visible.
    // W_t grows with t to exercise varying window sizes.
    std::vector<int32_t> window_len(T);
    for (int64_t t = 0; t < T; ++t) {
        int64_t W = 16 + 4 * t; // 16, 20, 24, ...
        if (W > N_kv) W = N_kv;
        window_len[t] = (int32_t) W;
    }

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < N_kv; ++s) {
            float v = (s < window_len[t]) ? 0.0f : -INFINITY;
            // kq_mask is laid out as [N_kv, T] with row-major stride nb[1] = N_kv*sizeof(float).
            m_data[t * N_kv + s] = v;
        }
    }

    auto cb = [](ggml_tensor *, const char *, int) {};

    // 1) Validate derive_kv_windows against the synthetic mask.
    ggml_tensor * starts_t = nullptr;
    ggml_tensor * ends_t   = nullptr;

    ggml_tensor * starts = sparse_attn_topk::derive_kv_windows(
        ctx, kq_mask, T, N_kv, &starts_t, &ends_t);

    if (!starts || !starts_t || !ends_t) {
        std::printf("[sparse-kv-windowing] derive_kv_windows returned null tensor(s)\\n");
        ggml_free(ctx);
        return 1;
    }

    GGML_ASSERT(starts == starts_t);

    int32_t * starts_data = (int32_t *) starts_t->data;
    int32_t * ends_data   = (int32_t *) ends_t->data;

    for (int64_t t = 0; t < T; ++t) {
        if (starts_data[t] != 0) {
            std::printf("[sparse-kv-windowing] token %lld: expected start 0, got %d\\n",
                        (long long) t, starts_data[t]);
            ggml_free(ctx);
            return 1;
        }
        if (ends_data[t] != window_len[t]) {
            std::printf("[sparse-kv-windowing] token %lld: expected end %d, got %d\\n",
                        (long long) t, window_len[t], ends_data[t]);
            ggml_free(ctx);
            return 1;
        }
    }

    std::printf("[sparse-kv-windowing] derive_kv_windows OK\\n");
    std::fflush(stdout);

    // 2) Run full sparse indexer top-k selection and verify all returned
    // indices lie within the allowed KV window [0, W_t) per token.
    ggml_tensor * topk_indices = sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx,
        q_indexer,
        k_indexer,
        weights,
        kq_mask,
        top_k,
        cb,
        /*gf=*/ nullptr,
        /*sched=*/ nullptr,
        /*backend_cpu=*/ nullptr,
        /*k_indexer_fp8_sidecar=*/ nullptr,
        /*quant_bs=*/ 0,
        /*cache_block_size=*/ 0,
        /*cache_stride=*/ 0);

    if (!topk_indices) {
    // Execute a minimal GGML graph so that topk_indices is materialized.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, topk_indices);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

        std::printf("[sparse-kv-windowing] select_topk_tokens_indexer_kvaware returned null\\n");
        ggml_free(ctx);
        return 1;
    }

    if (topk_indices->ne[0] != top_k || topk_indices->ne[1] != T) {
        std::printf("[sparse-kv-windowing] unexpected topk_indices shape [%lld,%lld,...], expected [%lld,%lld]\\n",
                    (long long) topk_indices->ne[0], (long long) topk_indices->ne[1],
                    (long long) top_k, (long long) T);
        ggml_free(ctx);
        return 1;
    }

    // Access indices via byte offsets using nb[0]/nb[1]. Layout is [k, T].
    for (int64_t t = 0; t < T; ++t) {
        int32_t W = window_len[t];
        for (int64_t i = 0; i < top_k; ++i) {
            size_t off = (size_t) i * topk_indices->nb[0] +
                         (size_t) t * topk_indices->nb[1];
            int32_t idx = *(int32_t *) ((char *) topk_indices->data + off);
            if (idx < 0 || idx >= W) {
                std::printf(
                    "[sparse-kv-windowing] token %lld: topk index %d at position %lld "
                    "violates window [0,%d) (N_kv=%lld)\\n",
                    (long long) t, idx, (long long) i, (int) W, (long long) N_kv);
                ggml_free(ctx);
                return 1;
            }
        }
    }

    std::printf("[sparse-kv-windowing] mask-respecting top-k indices OK\\n");
    std::printf("TEST PASS\\n");
    ggml_free(ctx);
    return 0;
}
