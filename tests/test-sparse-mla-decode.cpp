#include "../src/llama-sparse-mla-fwd.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>


#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>

using namespace llama;

static void ref_sparse_decode_mqa(
    const float * Q,       // [D * Hq]
    const float * K,       // [D * Hkv * N]
    const float * V,       // [Dv * Hkv * N]
    const int32_t * topk,  // [Ksel]
    int D, int Hq, int Hkv, int Dv, int N, int Ksel,
    float kq_scale, float softcap,
    std::vector<float> & Out) { // [Dv * Hq]

    Out.assign((size_t) Dv * Hq, 0.0f);
    for (int h = 0; h < Hq; ++h) {
        const int hk = (Hkv == 1 ? 0 : (h % Hkv));
        std::vector<float> scores(Ksel);
        float m = -1e30f;
        for (int i = 0; i < Ksel; ++i) {
            int idx = topk[i];
            float dot = -1e30f;
            if (idx >= 0 && idx < N) {
                dot = 0.0f;
                const float * qh = Q + (size_t) D * h;
                const float * kh = K + (size_t) D * (hk + (size_t) Hkv * idx);
                for (int d = 0; d < D; ++d) dot += qh[d] * kh[d];
                dot *= kq_scale;
                if (softcap > 0.f) dot = std::tanh(dot / softcap) * softcap;
            }
            scores[i] = dot;
            m = std::max(m, dot);
        }
        float ssum = 0.0f;
        for (int i = 0; i < Ksel; ++i) {
            scores[i] = std::exp(scores[i] - m);
            ssum += scores[i];
        }
        float inv = 1.0f / ssum;
        for (int dv = 0; dv < Dv; ++dv) {
            float acc = 0.0f;
            for (int i = 0; i < Ksel; ++i) {
                int idx = topk[i];
                if (idx < 0 || idx >= N) continue;
                float p = scores[i] * inv;
                const float * vh = V + (size_t) Dv * (hk + (size_t) Hkv * idx);
                acc += p * vh[dv];
            }
            Out[dv + (size_t) Dv * h] = acc;
        }
    }
}

int main() {
    // Small synthetic MQA-like configuration: D != Dv, Hq may differ from Hkv
    const int D   = 32;
    const int Hq  = 3;
    const int Hkv = 1;
    const int Dv  = 24;
    const int N   = 17;
    const int Ksel = 5;

    const float kq_scale = 0.8f;
    const float softcap  = 0.5f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t) D * Hq);
    std::vector<float> K((size_t) D * Hkv * N);
    std::vector<float> V((size_t) Dv * Hkv * N);
    std::vector<int32_t> TOPK(Ksel);

    for (auto & v : Q) v = dist(rng);
    for (auto & v : K) v = dist(rng);
    for (auto & v : V) v = dist(rng);

    // Spread indices; include some small and some near the end
    for (int i = 0; i < Ksel; ++i) TOPK[i] = (i * 7) % N;

    ggml_init_params ip{};
    ip.mem_size   = 64ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("Failed to init ggml context\n");
        return 1;
    }

    // Shapes expected by apply_sparse_attention_kvaware:
    // q_cur   : [Dq, Hq, T]
    // k_cache : [Dk, Hkv, N_kv]
    // v_cache : [Dv, Hkv_v, N_kv]
    const int64_t T = 1; // single token for this test

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dv, Hkv, N);
    ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, Ksel, T); // [Ksel, T]

    // Populate tensors directly (CPU backend)
    memcpy(q_cur->data, Q.data(), ggml_nbytes(q_cur));
    memcpy(k_cache->data, K.data(), ggml_nbytes(k_cache));
    memcpy(v_cache->data, V.data(), ggml_nbytes(v_cache));
    memcpy(idx->data, TOPK.data(), ggml_nbytes(idx));

    auto cb = [](ggml_tensor * t, const char * name, int il) {
        (void) t; (void) name; (void) il;
    };

    ggml_tensor * out = sparse_mla_fwd::apply_sparse_attention_kvaware(
        ctx, q_cur, k_cache, v_cache, idx,
        /*n_tokens=*/T,
        /*top_k=*/Ksel,
        kq_scale,
        /*kq_mask=*/nullptr,
        /*attn_softcap=*/softcap,
        cb);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

    std::vector<float> O_gpu((size_t) Dv * Hq);
    memcpy(O_gpu.data(), out->data, ggml_nbytes(out));

    std::vector<float> O_ref;
    ref_sparse_decode_mqa(
        Q.data(), K.data(), V.data(), TOPK.data(),
        D, Hq, Hkv, Dv, N, Ksel,
        kq_scale, softcap,
        O_ref);

    float max_abs = 0.0f;
    int mism = 0;
    for (size_t i = 0; i < O_ref.size(); ++i) {
        float d = std::fabs(O_ref[i] - O_gpu[i]);
        if (d > max_abs) max_abs = d;
        if (d > 5e-3f) mism++;
    }

    printf("sparse mla decode (CPU) mism=%d max_abs=%.6f\n", mism, max_abs);

    ggml_free(ctx);

    printf("TEST %s\n", mism == 0 ? "PASS" : "FAIL");
    return mism == 0 ? 0 : 1;
}
