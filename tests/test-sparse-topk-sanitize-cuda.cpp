#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <random>

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping\n");
    return 0;
#else
    // Keep it deterministic and small
    const int N = 4096;  // rows (kv)
    const int T = 3;     // columns (tokens)
    const int K = 64;    // top-k

    // Encourage TL path for top-k op (if available)
    setenv("LLAMA_SPARSE_TOPK_TL", "1", 1);

    // Build ggml context
    ggml_init_params ip{};
    ip.mem_size = 64ull * 1024 * 1024;
    ip.no_alloc = true; // use backend alloc
    ggml_context * ctx = ggml_init(ip);

    // Tensors: scores [N, T], starts[T], ends[T]
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
    ggml_tensor * starts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * ends   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

    // Random scores
    std::vector<float> scores_h((size_t)N * T);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto & v : scores_h) v = dist(rng);

    // Windows with end-start < K so under-fill happens if not handled
    std::vector<int32_t> starts_h((size_t)T, 0);
    std::vector<int32_t> ends_h  ((size_t)T, 1); // extreme small window

    ggml_tensor * topk = ggml_sparse_topk_radix_ex(ctx, scores, K, starts, ends);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, topk);

    ggml_backend_dev_t dev_cuda = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t dev_cpu  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be_cuda = dev_cuda ? ggml_backend_dev_init(dev_cuda, nullptr) : nullptr;
    ggml_backend_t be_cpu  = ggml_backend_dev_init(dev_cpu,  nullptr);

    ggml_backend_t backends[2] = { be_cuda, be_cpu };
    int n_backends = be_cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(scores, scores_h.data(), 0, ggml_nbytes(scores));
    ggml_backend_tensor_set(starts, starts_h.data(), 0, ggml_nbytes(starts));
    ggml_backend_tensor_set(ends,   ends_h.data(),   0, ggml_nbytes(ends));

    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<int32_t> idx_h((size_t)K * T, -1);
    ggml_backend_tensor_get(topk, idx_h.data(), 0, ggml_nbytes(topk));

    bool ok = true;
    for (int t = 0; t < T && ok; ++t) {
        for (int i = 0; i < K; ++i) {
            int v = idx_h[i + K*t];
            if (v < 0 || v >= N) {
                fprintf(stderr, "OOB index at t=%d i=%d: %d (N=%d)\n", t, i, v, N);
                ok = false; break;
            }
        }
    }

    printf("sanitize-topk-window-underfill: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#endif
}
