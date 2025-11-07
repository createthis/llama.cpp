#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>
#include <cassert>
#include <cstring>

#ifdef GGML_USE_CUDA
#include <ggml-cuda-radix.h>
#endif

static inline uint32_t float_to_key_desc(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    if ((int32_t)u < 0) { return ~u; } else { return u | 0x80000000u; }
}

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping histogram test\n");
    return 0;
#else
    printf("Testing CUDA histogram kernel (top byte) ...\n");
    const int N = 512; // ensure 16-byte alignment for float4 loads (multiple of 4)
    const int T = 7;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.0f, 3.0f);
    std::vector<float> scores((size_t)N*T);
    for (auto & v : scores) v = dist(rng);

    std::vector<unsigned int> gt_counts_gpu(256*(size_t)T, 0);
    std::vector<unsigned int> thr_bins_gpu((size_t)T, 0);

    // call CUDA host wrapper
    ggml_cuda_topk_histogram_host(scores.data(), N, T,
                                  gt_counts_gpu.data(), thr_bins_gpu.data());

    // CPU reference
    std::vector<unsigned int> gt_counts_ref(256*(size_t)T, 0);
    for (int t = 0; t < T; ++t) {
        unsigned int hist[256] = {0};
        for (int i = 0; i < N; ++i) {
            uint32_t key = float_to_key_desc(scores[i + (size_t)N*t]);
            hist[(key >> 24) & 0xFFu]++;
        }
        unsigned int sum = 0;
        for (int b = 255; b >= 0; --b) {
            gt_counts_ref[b + 256*(size_t)t] = sum;
            sum += hist[b];
        }
    }

    // Compare
    bool ok = true;
    for (size_t i = 0; i < gt_counts_ref.size(); ++i) {
        if (gt_counts_ref[i] != gt_counts_gpu[i]) {
            printf("Mismatch at %zu: ref=%u gpu=%u\n", i, gt_counts_ref[i], gt_counts_gpu[i]);
            ok = false; break;
        }
    }

    if (ok) {
        printf("Histogram CUDA test: PASS\n");
        return 0;
    } else {
        printf("Histogram CUDA test: FAIL\n");
        return 1;
    }
#endif
}
