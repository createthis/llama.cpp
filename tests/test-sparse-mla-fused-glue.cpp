#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include "../src/llama-sparse-mla-fwd.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

using namespace llama;

// Glue-focused test for ggml_sparse_mla_decode_fused wiring.
// We exercise the T==1 fused path in sparse_mla_fwd::apply_sparse_attention_kvaware
// and use a hook to inspect the tensors passed to the fused OP. We only check
// structural invariants here (shapes, layout), not kernel math (covered by
// test-sparse-mla-decode-mqa-cuda).

static ggml_tensor * g_topk = nullptr;

static ggml_tensor * sparse_mla_fused_test_hook(
    ggml_context * ctx,
    ggml_tensor  * out2d,
    ggml_tensor  * q2d,
    ggml_tensor  * k_cache,
    ggml_tensor  * v_cache,
    ggml_tensor  * v_gather_src,
    ggml_tensor  * idx1d,
    float          kq_scale,
    float          softcap,
    ggml_tensor  * kv_dsmla_blob) {

    (void)ctx; (void)kq_scale; (void)softcap; (void)kv_dsmla_blob;

    const int64_t Dv  = out2d->ne[0];
    const int64_t Hq  = out2d->ne[1];
    const int64_t Dq  = q2d->ne[0];
    const int64_t Hq_q= q2d->ne[1];
    const int64_t Dk  = k_cache->ne[0];
    const int64_t Hkv = k_cache->ne[1];
    const int64_t N   = k_cache->ne[2];

    printf("[MLA-FUSED-GLUE] out2d ne=[%lld,%lld]\n", (long long) Dv, (long long) Hq);
    printf("[MLA-FUSED-GLUE] q2d   ne=[%lld,%lld]\n", (long long) Dq, (long long) Hq_q);
    printf("[MLA-FUSED-GLUE] k_cache ne=[%lld,%lld,%lld]\n",
           (long long) Dk, (long long) Hkv, (long long) N);

    GGML_ASSERT(Dq == Dk);
    GGML_ASSERT(Hq == Hq_q);
    GGML_ASSERT(Hkv > 0);
    GGML_ASSERT(N   > 0);

    if (g_topk) {
        const int64_t Ksel = idx1d->ne[0];
        GGML_ASSERT(g_topk->ne[1] == 1); // [K,1]
        GGML_ASSERT(g_topk->ne[0] == Ksel);
        printf("[MLA-FUSED-GLUE] Ksel=%lld\n", (long long) Ksel);
    }

    // V layout: for this test we use the normal layout [Dv,Hkv,N]. Ensure that
    // the normalized V_gather_src tensor matches v_cache in shape and shares
    // the same data pointer (no unexpected permute).
    GGML_ASSERT(v_cache->ne[0] == Dv);
    GGML_ASSERT(v_cache->ne[1] == Hkv);
    GGML_ASSERT(v_cache->ne[2] == N);

    GGML_ASSERT(v_gather_src->ne[0] == v_cache->ne[0]);
    GGML_ASSERT(v_gather_src->ne[1] == v_cache->ne[1]);
    GGML_ASSERT(v_gather_src->ne[2] == v_cache->ne[2]);

    printf("[MLA-FUSED-GLUE] v_cache     ne=[%lld,%lld,%lld] ptr=%p\n",
           (long long) v_cache->ne[0], (long long) v_cache->ne[1], (long long) v_cache->ne[2], v_cache->data);
    printf("[MLA-FUSED-GLUE] v_gather_src ne=[%lld,%lld,%lld] ptr=%p\n",
           (long long) v_gather_src->ne[0], (long long) v_gather_src->ne[1], (long long) v_gather_src->ne[2], v_gather_src->data);

    GGML_ASSERT(v_gather_src->data == v_cache->data);

    printf("[MLA-FUSED-GLUE] hook OK\n");
    return out2d;
}

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping sparse mla fused glue test\n");
    return 0;
#else
    // DeepSeek-like shapes, small N/Ksel for clarity
    const int Dq   = 576;
    const int Hq   = 8;
    const int Hkv  = 1;
    const int Dv   = 512;
    const int N    = 64;    // N_kv
    const int Ksel = 8;     // top_k

    float kq_scale = 1.0f;
    float softcap  = 0.0f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q3((size_t)Dq * Hq * 1);
    std::vector<float> K((size_t)Dq * Hkv * N);
    std::vector<float> V((size_t)Dv * Hkv * N);
    std::vector<int32_t> TOPK((size_t)Ksel);

    for (auto &v : Q3) v = dist(rng);
    for (auto &v : K)  v = dist(rng);
    for (auto &v : V)  v = dist(rng);
    for (int i = 0; i < Ksel; ++i) TOPK[i] = (i * 3) % N;

    ggml_init_params ip{};
    ip.mem_size   = 64ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;

    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dq, Hq, 1);
    memcpy(q_cur->data, Q3.data(), Q3.size() * sizeof(float));

    ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dq, Hkv, N);
    memcpy(k_cache->data, K.data(), K.size() * sizeof(float));

    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dv, Hkv, N);
    memcpy(v_cache->data, V.data(), V.size() * sizeof(float));

    g_topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, Ksel, 1);
    memcpy(g_topk->data, TOPK.data(), TOPK.size() * sizeof(int32_t));

    ggml_tensor * kq_mask       = nullptr;
    ggml_tensor * kv_dsmla_blob = nullptr;

    set_sparse_mla_fused_hook(&sparse_mla_fused_test_hook);

    auto cb = [](ggml_tensor *, const char *, int) {};
    ggml_tensor * out3d = sparse_mla_fwd::apply_sparse_attention_kvaware(
        ctx,
        q_cur,
        k_cache,
        v_cache,
        g_topk,
        /*n_tokens=*/1,
        /*top_k=*/Ksel,
        kq_scale,
        kq_mask,
        softcap,
        kv_dsmla_blob,
        cb);

    (void) out3d; // we don't need to execute the graph; hook runs during graph build

    // Clear hook and free context
    set_sparse_mla_fused_hook(nullptr);
    ggml_free(ctx);

    printf("TEST PASS\n");
    return 0;
#endif
}
