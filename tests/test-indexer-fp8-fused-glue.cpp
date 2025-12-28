#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include "../src/llama-sparse-topk.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

using namespace llama;

// Glue-focused test for fused indexer path. We inject a hook via
// set_indexer_fused_hook and verify the shapes and basic invariants
// of the tensors passed to the fused GGML_OP_INDEXER_FUSED op. This
// does not exercise the WMMA/FP8 math itself (covered elsewhere),
// but ensures that q_tile2d, k_slice, w_slice, k_scale, and optional
// FP8 sidecar are wired correctly from sparse_attn_topk.

static bool g_hook_called = false;

static ggml_tensor * indexer_fused_test_hook(
    ggml_context * ctx,
    ggml_tensor * q_tile2d,
    ggml_tensor * k_slice,
    ggml_tensor * w_slice,
    ggml_tensor * k_scale_head,
    ggml_tensor * k_indexer_fp8_sidecar,
    int64_t       t0,
    int64_t       Tc,
    int64_t       kv_start,
    int64_t       kv_end,
    int32_t       quant_bs,
    int32_t       cache_block_size,
    int32_t       cache_stride) {

    (void) ctx;
    (void) k_indexer_fp8_sidecar;
    (void) kv_start;
    (void) quant_bs;
    (void) cache_block_size;
    (void) cache_stride;

    g_hook_called = true;

    GGML_ASSERT(q_tile2d != nullptr);
    GGML_ASSERT(k_slice  != nullptr);
    GGML_ASSERT(w_slice  != nullptr);
    GGML_ASSERT(k_scale_head != nullptr);

    const int64_t D   = q_tile2d->ne[0];
    const int64_t TcH = q_tile2d->ne[1];
    const int64_t Dk  = k_slice->ne[0];
    const int64_t kv  = k_slice->ne[1];
    const int64_t H   = w_slice->ne[0];
    const int64_t Tc_local = w_slice->ne[1];

    GGML_ASSERT(D == Dk);
    GGML_ASSERT(Tc_local == Tc);
    GGML_ASSERT(TcH == Tc_local * H);
    GGML_ASSERT(k_scale_head->ne[0] == kv);

    GGML_ASSERT(t0 >= 0);
    GGML_ASSERT(Tc > 0);
    GGML_ASSERT(kv_end >= kv);

    printf("[INDEXER-FP8-GLUE] D=%lld H=%lld Tc=%lld t0=%lld kv=%lld kv_end=%lld\n",
           (long long) D, (long long) H, (long long) Tc_local,
           (long long) t0, (long long) kv, (long long) kv_end);

    return nullptr;
}

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping indexer FP8 fused glue test\n");
    return 0;
#else
    // Simple synthetic shapes; we only need to trigger the fused
    // GGML_OP_INDEXER_FUSED path once so the hook can inspect
    // the inputs.
    const int64_t D_index = 128;
    const int64_t H_index = 64;
    const int64_t T       = 2;
    const int64_t N_kv    = 4096;
    const int64_t top_k   = 64;

    ggml_init_params ip{};
    ip.mem_size   = 64ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;

    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q_indexer = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_index, H_index, T);
    ggml_tensor * k_indexer = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_index, N_kv);
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_index, T);
    ggml_tensor * kq_mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_kv, T);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q_data(ggml_nelements(q_indexer));
    std::vector<float> k_data(ggml_nelements(k_indexer));
    std::vector<float> w_data(ggml_nelements(weights));
    std::vector<float> m_data(ggml_nelements(kq_mask));

    for (auto &v : q_data) v = dist(rng);
    for (auto &v : k_data) v = dist(rng);
    for (auto &v : w_data) v = dist(rng);
    for (auto &v : m_data) v = 0.0f;

    std::memcpy(q_indexer->data, q_data.data(), q_data.size() * sizeof(float));
    std::memcpy(k_indexer->data, k_data.data(), k_data.size() * sizeof(float));
    std::memcpy(weights->data,   w_data.data(), w_data.size() * sizeof(float));
    std::memcpy(kq_mask->data,   m_data.data(), m_data.size() * sizeof(float));

    // Register the hook and force fused+device path.
    set_indexer_fused_hook(&indexer_fused_test_hook);
    setenv("LLAMA_SPARSE_INDEXER_FUSED", "1", 1);
    setenv("LLAMA_SPARSE_INDEXER_FUSED_DEVICE", "1", 1);

    auto cb = [](ggml_tensor *, const char *, int) {};

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t cpu_dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);

    ggml_backend_t backs[2] = { cuda, cpu };
    int nb = cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, nb, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_tensor * topk_indices = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx,
        q_indexer,
        k_indexer,
        weights,
        kq_mask,
        top_k,
        cb,
        gf,
        sched,
        /*backend_cpu=*/cpu,
        /*k_indexer_fp8_sidecar=*/nullptr,
        /*quant_bs=*/0,
        /*cache_block_size=*/0,
        /*cache_stride=*/0);

    (void) topk_indices;

    // We don't need to execute the graph; the hook should have been
    // invoked during graph construction.

    set_indexer_fused_hook(nullptr);
    ggml_backend_sched_free(sched);
    if (cuda) ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    if (!g_hook_called) {
        printf("indexer fused glue hook was not called. TEST FAIL\n");
        return 1;
    }

    printf("TEST PASS\n");
    return 0;
#endif
}
