#include "../src/llama-sparse-topk.h"
#include "../src/llama-impl.h"

#include <ggml.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include <cassert>
#include <cstdio>
#include <vector>
#include <random>

// This test tries to reproduce a large CUDA memcpy (cpy.cu assert) by
// constructing a very large [N,T] scores tensor and running radix top-k on it.
// We keep sizes realistic but bounded to fit CI memory.

int main() {
    // Use moderate sizes to avoid OOM in CI; aim to create large nb1 strides
    const int64_t N = 32768;   // rows (N_kv)
    const int64_t T = 64;      // columns (tokens per tile)
    const int64_t K = 64;      // top-k

    ggml_init_params p{};
    p.mem_size = 1024ull * 1024ull * 1024ull; // 1 GB
    p.mem_buffer = nullptr;
    p.no_alloc = false;
    ggml_context * ctx = ggml_init(p);
    assert(ctx);

    // Create scores [N, T]
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);

    // Fill with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    for (int64_t j = 0; j < T; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            size_t off = j*scores->nb[1] + i*scores->nb[0];
            *(float*)((char*)scores->data + off) = dist(rng);
        }
    }

    ggml_tensor * idx = llama::sparse_attn_topk::topk_radix_indices(ctx, scores, K);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx);
    // Compute on 1 thread to keep memory pressure predictable
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Touch a few results
    int32_t v0 = *(int32_t*)((char*)idx->data + 0*idx->nb[0] + 0*idx->nb[1]);
    int32_t v1 = *(int32_t*)((char*)idx->data + (K-1)*idx->nb[0] + (T-1)*idx->nb[1]);
    printf("topk_radix_indices large test: v0=%d v1=%d\n", v0, v1);

    printf("large-cuda-like topk test: PASS\n");
    return 0;
}
