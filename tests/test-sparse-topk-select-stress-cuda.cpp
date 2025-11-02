#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <algorithm>
#ifdef GGML_USE_CUDA
#include <ggml-cuda-radix.h>
#endif
static inline uint32_t float_to_key_desc(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    if ((int32_t)u < 0) { return ~u; } else { return u | 0x80000000u; }
}
int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping select stress test\n");
    return 0;
#else
    printf("Select stress CUDA ...\n");
    const int N = 16384;
    const int T = 3;
    const int K = 32;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> scores((size_t)N*T);
    for (auto & v : scores) v = dist(rng);
    // Build gt_counts on CPU
    std::vector<unsigned int> gt_counts(256*(size_t)T, 0);
    for (int t = 0; t < T; ++t) {
        unsigned int hist[256] = {0};
        for (int i = 0; i < N; ++i) {
            uint32_t key = float_to_key_desc(scores[i + (size_t)N*t]);
            hist[(key >> 24) & 0xFFu]++;
        }
        unsigned int sum = 0;
        for (int b = 255; b >= 0; --b) {
            gt_counts[b + 256*(size_t)t] = sum;
            sum += hist[b];
        }
    }
    std::vector<int> idx_gpu((size_t)K*T, -1);
    ggml_cuda_topk_select_host(scores.data(), N, T, K, gt_counts.data(), idx_gpu.data());
    // Validate
    bool ok = true;
    for (int t = 0; t < T; ++t) {
        std::vector<float> col(N);
        for (int i = 0; i < N; ++i) col[i] = scores[i + (size_t)N*t];
        std::vector<float> sorted = col;
        std::nth_element(sorted.begin(), sorted.begin() + (K-1), sorted.end(), std::greater<float>());
        float thresh = sorted[K-1];
        std::vector<char> seen(N, 0);
        for (int i = 0; i < K; ++i) {
            int idx = idx_gpu[i + K*t];
            if (idx < 0 || idx >= N) { printf("Stress-select: invalid idx %d\n", idx); ok = false; break; }
            if (seen[idx]) { printf("Stress-select: duplicate idx %d\n", idx); ok = false; break; }
            seen[idx] = 1;
            if (!(col[idx] + 0.0f >= thresh)) { printf("Stress-select: below threshold\n"); ok = false; break; }
        }
        if (!ok) break;
    }
    printf("Select stress CUDA: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#endif
}
