#include "../src/llama-sparse-topk.h"
#include "../src/llama-impl.h"

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include <ggml-alloc.h>
#include <ggml-cpp.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>

#ifdef GGML_USE_CUDA
static int test_cuda_topk_radix_host();
#endif

// Reference top-k (descending) using std::partial_sort on indices
static void ref_topk(const float * row, int64_t N, int64_t k, std::vector<int32_t> & out) {
    out.resize(k);
    std::vector<int32_t> idx(N);
    for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
    auto cmp = [&](int32_t a, int32_t b){ return row[a] > row[b]; };
    if (k < N) {
        std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), cmp);
        std::copy(idx.begin(), idx.begin()+k, out.begin());
    } else {
        std::sort(idx.begin(), idx.end(), cmp);
        std::copy(idx.begin(), idx.begin()+k, out.begin());
    }
}

int main() {
    // Build a small random problem: N_kv=4096, T=8
    const int64_t N_kv = 4096;
    const int64_t T = 8;
    const int64_t H = 4;
    const int64_t D = 64;
    // removed unused variable k

    // Initialize GGML context
    ggml_init_params p{};
    p.mem_size = 256ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc = false;
    ggml_context * ctx = ggml_init(p);
    assert(ctx);
    // Create tensors: q_indexer [D,H,T], k_indexer [D,N_kv], weights [H,T]
    ggml_tensor * q_indexer = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
    ggml_tensor * k_indexer = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N_kv);
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);


    // Fill with random values and write into tensors directly (no backend set/get)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill_tensor = [&](ggml_tensor * t, std::vector<float> & host) {
        host.resize(ggml_nelements(t));
        for (auto & v : host) v = dist(rng);
        // write respecting strides
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            size_t off = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
            size_t lin = i0 + t->ne[0]*(i1 + t->ne[1]*(i2 + t->ne[2]*i3));
            *(float*)((char*)t->data + off) = host[lin];
        }
    };
    std::vector<float> Q, Khost, Whost;
    fill_tensor(q_indexer, Q);
    fill_tensor(k_indexer, Khost);
    fill_tensor(weights,   Whost);

    int status = 0;
#ifdef GGML_USE_CUDA
    // also run CUDA host-wrapper validation
    status |= test_cuda_topk_radix_host();
#endif
    return status;
}

#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#include <ggml-cuda-radix.h>
#endif

static int test_cuda_topk_radix_host() {
#ifdef GGML_USE_CUDA
    printf("Running CUDA radix top-k host wrapper test...\n");
    const int N = 1024;
    const int T = 8;
    const int k = 32;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> scores((size_t)N*T);
    for (auto & v : scores) v = dist(rng);
    std::vector<int> idx_cuda((size_t)k*T);
    // call host wrapper
    ggml_cuda_topk_radix_indices_host(scores.data(), N, T, k, idx_cuda.data());
    // compute reference per column and check threshold criterion
    for (int t = 0; t < T; ++t) {
        std::vector<float> col(N);
        for (int i = 0; i < N; ++i) col[i] = scores[i + N*t];
        std::vector<float> sorted = col;
        std::nth_element(sorted.begin(), sorted.begin() + (k-1), sorted.end(), std::greater<float>());
        float thresh = sorted[k-1];
        std::vector<char> seen(N, 0);
        for (int i = 0; i < k; ++i) {
            int idx = idx_cuda[i + k*t];
            if (idx < 0 || idx >= N) {
                printf("CUDA: column %d out-of-bounds index %d N=%d\n", t, idx, N);
                return 1;
            }
            if (seen[idx]) {
                printf("CUDA: column %d duplicate index %d\n", t, idx);
                return 1;
            }
            seen[idx] = 1;
            if (!(col[idx] + 0.0f >= thresh)) {
                printf("CUDA: column %d value below threshold v=%.6f thresh=%.6f\n", t, col[idx], thresh);
                return 1;
            }
        }
    }
    printf("CUDA radix top-k host wrapper: PASS\n");
    return 0;
#else
    (void)0; // unused
    return 0;
#endif
}

