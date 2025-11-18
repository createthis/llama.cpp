#include <ggml.h>

#include "llama-sparse-indexer.h"

#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstring>

using namespace llama;

static void cpu_reference_scores(
    const std::vector<float> & Q,   // [D, Tc, H]
    const std::vector<float> & K,   // [D, kv]
    const std::vector<float> & W,   // [H, Tc]
    const std::vector<float> & KS,  // [kv]
    int64_t D, int64_t H, int64_t Tc, int64_t kv,
    std::vector<float> & out) {
    out.assign((size_t)kv * (size_t)Tc, 0.0f);
    for (int64_t tc = 0; tc < Tc; ++tc) {
        for (int64_t kv_i = 0; kv_i < kv; ++kv_i) {
            float acc = 0.0f;
            for (int64_t h = 0; h < H; ++h) {
                const float *qv  = &Q[(size_t)D*((size_t)tc + (size_t)Tc*h)];
                const float *kvp = &K[(size_t)kv_i*D];
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    dot += qv[d]*kvp[d];
                }
                if (dot < 0.0f) dot = 0.0f;
                // W stored column-major [H, Tc] in ggml (H is fastest)
                acc += dot * W[(size_t)h + (size_t)H * (size_t)tc];
            }
            // out is [kv, Tc] laid out as row-major
            out[(size_t)kv_i + (size_t)kv * (size_t)tc] = acc * KS[(size_t)kv_i];
        }
    }
}

int main() {
    const int64_t D  = 64;
    const int64_t H  = 8;
    const int64_t Tc = 64;
    const int64_t kv = 4096;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t)D*Tc*H);
    std::vector<float> K((size_t)D*kv);
    std::vector<float> W((size_t)H*Tc);
    std::vector<float> KS((size_t)kv);

    for (auto & v : Q)  v = dist(rng);
    for (auto & v : K)  v = dist(rng);
    for (auto & v : W)  v = dist(rng);
    for (auto & v : KS) v = std::max(0.1f, std::fabs(dist(rng)));

    // Build CPU reference
    std::vector<float> ref;
    cpu_reference_scores(Q, K, W, KS, D, H, Tc, kv, ref);

    ggml_init_params ip{};
    ip.mem_size = 256ull*1024*1024;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("ctx init failed\n");
        return 1;
    }

    // q3d: [D, Tc, H] to match sparse_attn_indexer::idx_compute_scores_tile
    ggml_tensor * q3d  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Tc, H);
    ggml_tensor * a_k  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, kv);
    ggml_tensor * w2d  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, Tc);
    ggml_tensor * ks1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv);
    ggml_tensor * ks2d = ggml_reshape_2d(ctx, ks1d, kv, 1);

    std::memcpy(q3d->data,  Q.data(),  ggml_nbytes(q3d));
    std::memcpy(a_k->data,  K.data(),  ggml_nbytes(a_k));
    std::memcpy(w2d->data,  W.data(),  ggml_nbytes(w2d));
    std::memcpy(ks1d->data, KS.data(), ggml_nbytes(ks1d));

    ggml_tensor * scores = sparse_attn_indexer::idx_compute_scores_tile(
        ctx, q3d, a_k, w2d, ks2d, D, H, Tc, kv, 0, /*use_fp16=*/false);

    const int iters = 10;

    auto run_once = [&]() {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, scores);
        ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);
    };

    // warmup
    run_once();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_once();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double avg_ms = 1000.0 * elapsed_s / iters;

    printf("[IDX_TILE_REF] idx_compute_scores_tile (CPU): D=%lld H=%lld Tc=%lld kv=%lld iters=%d avg_ms=%.3f\n",
           (long long)D, (long long)H, (long long)Tc, (long long)kv, iters, avg_ms);

    // scores is [kv, Tc] with row-major layout
    std::vector<float> out((size_t)kv * (size_t)Tc);
    std::memcpy(out.data(), scores->data, ggml_nbytes(scores));

    int mism = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < out.size(); ++i) {
        float da = std::fabs(out[i] - ref[i]);
        if (da > 1e-3f) mism++;
        if (da > max_abs) max_abs = da;
    }

    printf("idx_compute_scores_tile test: mism=%d max_abs=%.6f\n", mism, max_abs);
    printf("TEST %s\n", mism == 0 ? "PASS" : "FAIL");

    ggml_free(ctx);
    return mism == 0 ? 0 : 1;
}
