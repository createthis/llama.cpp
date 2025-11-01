#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

static void build_and_run(int N, int T, int K) {
    // host scores
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> scores_h((size_t)N*T);
    for (auto & v : scores_h) v = dist(rng);

    ggml_init_params ip{}; ip.mem_size = 64ull*1024*1024; ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);

    // run on default backend (will pick CUDA if available)
    ggml_backend_t be = ggml_backend_init_best();

    ggml_tensor* idx = ggml_sparse_topk_radix(ctx, scores, K);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx);

    // allocate backend memory for tensors in this context (after building the graph)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == NULL) {
        printf("Failed to allocate backend buffer for tensors\n");
        return;
    }

    // copy host scores into device tensor
    ggml_backend_tensor_set(scores, scores_h.data(), 0, ggml_nbytes(scores));

    ggml_status st = ggml_backend_graph_compute(be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("backend compute failed: %d\n", (int)st);
        return;
    }

    // read back indices from device
    std::vector<int> idx_h((size_t)K*T, -1);
    ggml_backend_tensor_get(idx, idx_h.data(), 0, ggml_nbytes(idx));

    // validate threshold criterion per column
    int ok = 1;
    for (int t = 0; t < T; ++t) {
        std::vector<float> col(N);
        for (int i = 0; i < N; ++i) col[i] = scores_h[i + (size_t)N*t];
        std::vector<float> sorted = col;
        std::nth_element(sorted.begin(), sorted.begin() + (K-1), sorted.end(), std::greater<float>());
        float thresh = sorted[K-1];
        std::vector<char> seen(N, 0);
        for (int i = 0; i < K; ++i) {
            int ix = idx_h[i + (size_t)K*t];
            if (ix < 0 || ix >= N) { printf("op-cuda: invalid idx %d\n", ix); ok = 0; break; }
            if (seen[ix]) { printf("op-cuda: duplicate idx %d\n", ix); ok = 0; break; }
            seen[ix] = 1;
            if (!(col[ix] + 0.0f >= thresh)) { printf("op-cuda: below threshold\n"); ok = 0; break; }
        }
        if (!ok) break;
    }

    printf("sparse_topk_radix op test (%d,%d,%d): %s\n", N, T, K, ok?"PASS":"FAIL");
    ggml_free(ctx);
}

int main() {
    build_and_run(1024, 3, 32);
    build_and_run(32768, 2, 64); // exercises streaming fallback
    return 0;
}
