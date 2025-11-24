#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

// CUTLASS FP8 E4M3 reference implementation
#include "cutlass/float8.h"

using cutlass::float_e4m3_t;

extern "C" {
void ggml_e4m3_to_fp32_row(const uint8_t * x, float * y, int64_t k);
void ggml_fp32_to_e4m3_row_ref(const float * x, uint8_t * y, int64_t k);
}


static inline uint32_t f32_bits(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}

int main() {
#ifndef GGML_USE_CUDA
    std::printf("CUDA not enabled; skipping fp8 e4m3 test\n");
    return 0;
#else
    // --- Decode test: 256 FP8 codes ---
    int mism_decode = 0;
    float max_abs_decode = 0.0f;
    for (int b = 0; b < 256; ++b) {
        uint8_t code = (uint8_t) b;
        float cpu; ggml_e4m3_to_fp32_row(&code, &cpu, 1);
        float ref = float_e4m3_t::to_float(float_e4m3_t::bitcast(code));
        if (std::isnan(cpu) && std::isnan(ref)) continue;
        if (cpu == 0.0f && ref == 0.0f) continue; // treat +0 and -0 as equal
        uint32_t bc = f32_bits(cpu);
        uint32_t br = f32_bits(ref);
        if (bc != br) {
            ++mism_decode;
            float da = std::fabs(cpu - ref);
            if (da > max_abs_decode) max_abs_decode = da;
        }
    }
    std::printf("FP8 E4M3 decode mismatches=%d max_abs_diff=%.6f\n", mism_decode, max_abs_decode);
    if (mism_decode != 0) {
        std::printf("TEST FAIL (decode)\n");
        return 1;
    }

    // --- Encode test: random + edge cases ---
    const int N = 131072;
    std::vector<float> vals(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (int i = 0; i < N; ++i) vals[i] = dist(rng);
    if (N >= 16) {
        vals[0] = 0.0f;
        vals[1] = -0.0f;
        vals[2] = std::numeric_limits<float>::infinity();
        vals[3] = -std::numeric_limits<float>::infinity();
        vals[4] = std::numeric_limits<float>::quiet_NaN();
        vals[5] = std::numeric_limits<float>::denorm_min();
        vals[6] = 1e-10f;
        vals[7] = -1e-10f;
    }

    int mism_encode = 0;
    for (int i = 0; i < N; ++i) {
        float x = vals[i];
        uint8_t c; ggml_fp32_to_e4m3_row_ref(&x, &c, 1);
        uint8_t r = float_e4m3_t::from_float(x).storage;
        if (c != r) {
            ++mism_encode;
        }
    }
    std::printf("FP8 E4M3 encode mismatches=%d over %d samples\n", mism_encode, N);
    if (mism_encode != 0) {
        std::printf("TEST FAIL (encode)\n");
        return 1;
    }

    std::printf("FP8 E4M3 CPU helper matches CUTLASS float_e4m3_t: TEST PASS\n");
    return 0;
#endif
}
