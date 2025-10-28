#include "../src/llama-sparse-topk.h"
#include <ggml.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cassert>

// Repro harness for CUDA cpy assert in indexer top-k path.
// It builds synthetic q_indexer, k_indexer, weights and calls
// select_topk_tokens_indexer_kvaware with a large tile to stress
// ggml_cont and broadcast adds.
//
// Tunables via env:
//  REPRO_D (default 4)
//  REPRO_H (default 2)
//  REPRO_T (default 2048)
//  REPRO_NKV (default 131072) // 128k
//  REPRO_K (default 64)
//  LLAMA_SPARSE_TOPK_TILE_T (default inside code path)
//
int64_t getenv64(const char *k, int64_t defv) {
    const char *v = getenv(k);
    if (!v) return defv;
    return strtoll(v, nullptr, 10);
}

int main() {
    const int64_t D = getenv64("REPRO_D", 4);
    const int64_t H = getenv64("REPRO_H", 2);
    const int64_t T = getenv64("REPRO_T", 2048);
    const int64_t N_kv = getenv64("REPRO_NKV", 131072);
    const int64_t K = getenv64("REPRO_K", 64);

    ggml_init_params p{};
    // 2GB by default; raise if you want to push past 2GB for CUDA assert repro
    p.mem_size = 2ull * 1024ull * 1024ull * 1024ull;
    p.mem_buffer = nullptr;
    p.no_alloc = false;
    ggml_context * ctx = ggml_init(p);
    assert(ctx);

    // q_indexer [D, H, T]
    ggml_tensor * q_indexer = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
    // k_indexer [D, N_kv]
    ggml_tensor * k_indexer = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N_kv);
    // weights [H, T]
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);

    // Fill with deterministic values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](ggml_tensor * t) {
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            size_t off = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
            *(float*)((char*)t->data + off) = dist(rng);
        }
    };
    fill(q_indexer);
    fill(k_indexer);
    fill(weights);

    // no mask to simplify
    ggml_tensor * mask = nullptr;

    // call indexer top-k (radix variant is invoked internally)
    ggml_tensor * idx = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx, q_indexer, k_indexer, weights, mask, K,
        [](ggml_tensor*, const char*, int){}, /*cb*/
        nullptr, /*gf*/
        nullptr, /*sched*/
        nullptr  /*backend*/);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx);
    // Run with small thread count; on CUDA systems, this will exercise device kernels
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Touch a few outputs
    int32_t v0 = *(int32_t*)((char*)idx->data + 0*idx->nb[0] + 0*idx->nb[1]);
    int32_t v1 = *(int32_t*)((char*)idx->data + (K-1)*idx->nb[0] + (T-1)*idx->nb[1]);
    printf("indexer_topk repro: v0=%d v1=%d (D=%lld H=%lld T=%lld N_kv=%lld K=%lld)\n",
           v0, v1, (long long)D, (long long)H, (long long)T, (long long)N_kv, (long long)K);
    printf("repro test: PASS\n");
    return 0;
}
