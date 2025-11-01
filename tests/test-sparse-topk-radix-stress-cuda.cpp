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
    printf("CUDA not enabled; skipping radix stress test\n");
    return 0;
#else
    printf("Radix end-to-end stress CUDA (streaming default) ...\n");
    const int N = 32768;
    const int T = 2;
    const int K = 64;
    std::mt19937 rng(2027);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
    std::vector<float> scores((size_t)N*T);
    for (auto & v : scores) v = dist(rng);
    std::vector<int> idx_gpu((size_t)K*T, -1);
    // Exercise full path via histogram+select inside device wrapper
    ggml_cuda_topk_radix_indices_host(scores.data(), N, T, K, idx_gpu.data());
    // Validate per column
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
            if (idx < 0 || idx >= N) { printf("Stress-radix: invalid idx %d\n", idx); ok = false; break; }
            if (seen[idx]) { printf("Stress-radix: duplicate idx %d\n", idx); ok = false; break; }
            seen[idx] = 1;
            if (!(col[idx] + 0.0f >= thresh)) { printf("Stress-radix: below threshold\n"); ok = false; break; }
        }
        if (!ok) break;
    }
    printf("Radix stress CUDA: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#endif
}
