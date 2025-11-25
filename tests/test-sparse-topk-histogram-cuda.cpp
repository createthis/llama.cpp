#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>
#include <cassert>
#include <cstring>

#ifdef GGML_USE_CUDA
#include <ggml-cuda-radix.h>
#endif

static inline uint16_t float_to_half_bits_rtne(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t sub = mant >> (1 - exp);
        // round to nearest even
        if (sub & 0x00001000u) sub += 0x00002000u;
        return (uint16_t)(sign | (sub >> 13));
    } else if (exp >= 31) {
        // Inf/NaN
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        mant >>= 13;
        return (uint16_t)(sign | 0x7C00u | mant | (mant == 0));
    } else {
        // round to nearest even
        if (mant & 0x00001000u) {
            mant += 0x00002000u;
            if (mant & 0x00800000u) { mant = 0; exp += 1; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
        }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }
}

static inline uint32_t float_to_key_desc(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    if ((int32_t)u < 0) { return ~u; } else { return u ^ 0x80000000u; }
}
static inline uint8_t key32_msb_bin_desc_host(float x) {
    return (uint8_t)(float_to_key_desc(x) >> 24);
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

    // CPU reference using fp16 coarse bin mapping (to match kernel)
    std::vector<unsigned int> gt_counts_ref(256*(size_t)T, 0);
    for (int t = 0; t < T; ++t) {
        unsigned int hist[256] = {0};
        for (int i = 0; i < N; ++i) {
            uint8_t b0 = key32_msb_bin_desc_host(scores[i + (size_t)N*t]);
            hist[b0]++;
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
