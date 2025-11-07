#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>

static void build_and_run(int N, int T, int K) {
    // host scores
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> scores_h((size_t)N*T);
    for (auto & v : scores_h) v = dist(rng);

    ggml_init_params ip{}; ip.mem_size = 64ull*1024*1024; ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
    ggml_tensor* idx = ggml_sparse_topk_radix(ctx, scores, K);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        // fallback: pick any CUDA-like device if name lookup fails
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { cuda_dev = d; break; }
        }
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) { printf("no CPU device found\n"); ggml_free(ctx); return; }

    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu || (!cuda && cuda_dev)) { printf("backend init failed\n"); if (cuda) ggml_backend_free(cuda); if (cpu) ggml_backend_free(cpu); ggml_free(ctx); return; }

    ggml_backend_t backs_arr[2] = { cuda, cpu };
    int n_backs = cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs_arr, nullptr, n_backs, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { printf("sched init failed\n"); if (cuda) ggml_backend_free(cuda); ggml_backend_free(cpu); ggml_free(ctx); return; }

    ggml_backend_sched_reset(sched);
    // Reserve exact buffer sizes to avoid reallocation warnings during alloc_graph
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    // copy host scores into device tensor
    ggml_backend_tensor_set(scores, scores_h.data(), 0, ggml_nbytes(scores));

    printf("starting compute\n");
    fflush(stdout);
    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("backend compute failed: %d\n", (int)st);
        fflush(stdout);
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
    if (!ok) {
        // force non-zero exit to make ctest fail
        std::fprintf(stderr, "sparse_topk_radix op test failed (N=%d T=%d K=%d)\n", N, T, K);
        std::fflush(stderr);
        std::exit(1);
    }


    printf("sparse_topk_radix op test (%d,%d,%d): %s\n", N, T, K, ok?"PASS":"FAIL");
    fflush(stdout);
    ggml_graph_clear(gf);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    //build_and_run(1024, 3, 32);
    build_and_run(163840, 1, 256); // stress target config similar to 163840x1x256
    return 0;
}
