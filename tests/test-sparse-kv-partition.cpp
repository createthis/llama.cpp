#include "../src/llama-sparse-topk.h"

#include <ggml.h>
#include <ggml-cpu.h>

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace llama;

// CPU-only partitioned KV-top-k semantics test.
//
// We construct a scores matrix [N_kv, T] with a synthetic A/B partition:
// - Conceptually two requests A and B laid out back-to-back in KV.
// - For A tokens (t < T/2): only [0, N_A) may be attended.
// - For B tokens (t >= T/2): only [N_A, N_kv) may be attended.
//
// We apply the same mask semantics used by the sparse indexer path
// (adding 0 for visible positions and -INF for masked positions, then
// clamping). We then run the CPU radix top-k helper
// sparse_attn_topk::topk_radix_indices and verify that no selected
// indices cross the partition boundary.

int main() {
    std::printf("[sparse-kv-partition] starting test...\n");
    std::fflush(stdout);

    const int64_t N_A   = 32;
    const int64_t N_B   = 48;
    const int64_t N_kv  = N_A + N_B; // 80
    const int64_t T     = 8;         // tokens
    const int64_t top_k = 8;

    ggml_init_params p{};
    p.mem_size   = 16ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    GGML_ASSERT(ctx != nullptr);

    // scores [N_kv, T], mask [N_kv, T]
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_kv, T);
    ggml_tensor * mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_kv, T);

    float * s_data = (float *) scores->data;
    float * m_data = (float *) mask->data;

    // Construct partitioned scores + mask:
    //  - For A tokens (t < T/2): [0,N_A) visible, [N_A,N_kv) masked
    //  - For B tokens (t >= T/2): [0,N_A) masked, [N_A,N_kv) visible
    for (int64_t t = 0; t < T; ++t) {
        bool is_A = (t < T/2);
        for (int64_t s = 0; s < N_kv; ++s) {
            bool visible = is_A ? (s < N_A) : (s >= N_A);
            // base score: simple deterministic pattern so top-k is well-defined
            float base = 0.01f * float((s + 17 * t) % 101);
            s_data[t * N_kv + s] = base;
            m_data[t * N_kv + s] = visible ? 0.0f : -INFINITY;
        }
    }

    // Apply mask in the same style as the sparse indexer path:
    // scores + mask, then clamp to avoid +/-inf in radix key conversion.
    ggml_tensor * scores_masked = ggml_add(ctx, scores, mask);
    scores_masked = ggml_clamp(ctx, scores_masked, -1e30f, 1e30f);

    // CPU-only radix top-k (no CUDA, no scheduler), identical pattern to
    // test-sparse-topk-radix.cpp.
    ggml_tensor * topk_indices = sparse_attn_topk::topk_radix_indices(ctx, scores_masked, top_k);
    if (!topk_indices) {
        std::printf("[sparse-kv-partition] topk_indices is null\n");
        ggml_free(ctx);
        return 1;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, topk_indices);
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

    if (topk_indices->ne[0] != top_k || topk_indices->ne[1] != T) {
        std::printf("[sparse-kv-partition] unexpected topk_indices shape [%lld,%lld,...], expected [%lld,%lld]\n",
                    (long long) topk_indices->ne[0], (long long) topk_indices->ne[1],
                    (long long) top_k, (long long) T);
        ggml_free(ctx);
        return 1;
    }

    int fails = 0;
    for (int64_t t = 0; t < T; ++t) {
        bool is_A = (t < T/2);
        for (int64_t i = 0; i < top_k; ++i) {
            size_t off = (size_t) i * topk_indices->nb[0] +
                         (size_t) t * topk_indices->nb[1];
            int32_t idx = *(int32_t *) ((char *) topk_indices->data + off);
            if (is_A) {
                if (idx < 0 || idx >= N_A) {
                    std::printf("[sparse-kv-partition] token %lld (A): idx=%d out of [0,%lld)\n",
                                (long long) t, idx, (long long) N_A);
                    ++fails;
                }
            } else {
                if (idx < N_A || idx >= N_kv) {
                    std::printf("[sparse-kv-partition] token %lld (B): idx=%d out of [%lld,%lld)\n",
                                (long long) t, idx, (long long) N_A, (long long) N_kv);
                    ++fails;
                }
            }
        }
    }

    if (fails != 0) {
        std::printf("[sparse-kv-partition] detected %d invalid indices\n", fails);
        ggml_free(ctx);
        return 1;
    }

    std::printf("[sparse-kv-partition] partition-respecting top-k indices OK\n");
    std::printf("TEST PASS\n");
    ggml_free(ctx);
    return 0;
}
