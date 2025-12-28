#include <ggml-cpp.h>
#include <cstring>

#include <ggml.h>
#include <ggml-cpu.h>

#include "../src/llama-sparse-mla-fwd.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <random>
#include <algorithm>

using namespace llama;

int main() {
    printf("[sparse-mla-glue-ds32] starting test...\n");

    // Small but nontrivial MLA-like configuration
    const int64_t Dq   = 16;
    const int64_t Dk   = 16;
    const int64_t Dv   = 8;
    const int64_t Hq   = 2;
    const int64_t Hkv  = 2;
    const int64_t N_kv = 10;
    const int64_t T    = 3;
    const int64_t Ksel = 4;

    // Host buffers for inputs
    std::vector<float> hQ ((size_t) Dq * Hq * T);
    std::vector<float> hK ((size_t) Dk * Hkv * N_kv);
    std::vector<float> hV ((size_t) Dv * Hkv * N_kv);
    std::vector<int32_t> hIdx((size_t) Ksel * T);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto & v : hQ)  v = dist(rng);
    for (auto & v : hK)  v = dist(rng);
    for (auto & v : hV)  v = dist(rng);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < Ksel; ++k) {
            hIdx[(size_t) k + (size_t) Ksel * (size_t) t] = (int32_t) ((k + t) % N_kv);
        }
    }

    ggml_init_params p{};
    p.mem_size   = 64ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;  // use context allocator and ggml_graph_compute_with_ctx

    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        fprintf(stderr, "ctx init failed\n");
        return 1;
    }

    ggml_tensor * q_cur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dq,  Hq,  T);
    ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk,  Hkv, N_kv);
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dv,  Hkv, N_kv);
    ggml_tensor * topk    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, Ksel, T);

    // Copy host buffers into tensors
    std::memcpy(q_cur->data,   hQ.data(),   ggml_nbytes(q_cur));
    std::memcpy(k_cache->data, hK.data(),   ggml_nbytes(k_cache));
    std::memcpy(v_cache->data, hV.data(),   ggml_nbytes(v_cache));
    std::memcpy(topk->data,    hIdx.data(), ggml_nbytes(topk));

    auto cb_noop = [](ggml_tensor *, const char *, int) {};

    ggml_tensor * out = sparse_mla_fwd::apply_sparse_attention_kvaware(
        ctx, q_cur, k_cache, v_cache, topk,
        /*n_tokens=*/T,
        /*top_k=*/Ksel,
        /*kq_scale=*/1.0f,
        /*kq_mask=*/nullptr,
        /*attn_softcap=*/0.0f,
        /*kv_dsmla_blob=*/nullptr,
        cb_noop);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);

    // Read GGML output
    std::vector<float> out_ggml(ggml_nelements(out));
    std::memcpy(out_ggml.data(), out->data, out_ggml.size() * sizeof(float));

    // CPU reference implementation using the same layouts as the host buffers.
    std::vector<float> out_ref(out_ggml.size(), 0.0f);

    auto idx_q = [=](int64_t d, int64_t hq, int64_t t) -> size_t {
        return (size_t) d + (size_t) Dq * ((size_t) hq + (size_t) Hq * (size_t) t);
    };
    auto idx_k = [=](int64_t d, int64_t hk, int64_t s) -> size_t {
        return (size_t) d + (size_t) Dk * ((size_t) hk + (size_t) Hkv * (size_t) s);
    };
    auto idx_v = [=](int64_t dv, int64_t hk, int64_t s) -> size_t {
        return (size_t) dv + (size_t) Dv * ((size_t) hk + (size_t) Hkv * (size_t) s);
    };

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t hq = 0; hq < Hq; ++hq) {
            const int R = (int) (Hkv * Ksel);
            std::vector<float> logits(R);
            std::vector<int>   idx_flat(R);

            // Build logits over (hk, i)
            int r = 0;
            for (int64_t hk = 0; hk < Hkv; ++hk) {
                for (int64_t i = 0; i < Ksel; ++i, ++r) {
                    int32_t s = hIdx[(size_t) i + (size_t) Ksel * (size_t) t];

                    float dot = 0.0f;
                    for (int64_t d = 0; d < Dq; ++d) {
                        float qv = hQ[idx_q(d, hq, t)];
                        float kv = hK[idx_k(d, hk, s)];
                        dot += qv * kv;
                    }
                    logits[r]   = dot;
                    idx_flat[r] = s;
                }
            }

            // Softmax over r
            float maxv = *std::max_element(logits.begin(), logits.end());
            float sum  = 0.0f;
            for (int i = 0; i < R; ++i) {
                logits[i] = std::exp(logits[i] - maxv);
                sum += logits[i];
            }
            if (sum <= 0.0f) sum = 1.0f;
            for (int i = 0; i < R; ++i) logits[i] /= sum;

            // Accumulate output over V
            for (int64_t dv = 0; dv < Dv; ++dv) {
                float acc = 0.0f;
                r = 0;
                for (int64_t hk = 0; hk < Hkv; ++hk) {
                    for (int64_t i = 0; i < Ksel; ++i, ++r) {
                        int32_t s = idx_flat[r];
                        float   p = logits[r];
                        float   vv = hV[idx_v(dv, hk, s)];
                        acc += p * vv;
                    }
                }

                size_t out_off = (size_t) dv
                               + (size_t) Dv * ((size_t) hq + (size_t) Hq * (size_t) t);
                out_ref[out_off] = acc;
            }
        }
    }

    int mism = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_ggml.size(); ++i) {
        float da = std::fabs(out_ggml[i] - out_ref[i]);
        if (da > 1e-3f) ++mism;
        if (da > max_abs) max_abs = da;
    }

    printf("[sparse-mla-glue-ds32] mismatches=%d max_abs=%.6g\n", mism, (double) max_abs);
    printf("TEST %s\n", mism == 0 ? "PASS" : "FAIL");

    ggml_free(ctx);
    return mism == 0 ? 0 : 1;
}
