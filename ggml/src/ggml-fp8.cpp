#include <cassert>
#include <algorithm>
#include <cstring>
#include <cstdio>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-fp8.h"

union fp32_int32 {
    float f;
    uint32_t bits;
};


static inline uint8_t float_to_e4m3_bits(float flt) {
    constexpr int FP32_NUM_BITS           = 32;
    constexpr int FP32_NUM_MANTISSA_BITS  = 23;
    constexpr int FP32_EXPONENT_BIAS      = 127;

    constexpr int FP8_NUM_EXPONENT_BITS   = 4;
    constexpr int FP8_NUM_MANTISSA_BITS   = 3;
    constexpr uint8_t FP8_NAN             = 0x7fu;
    constexpr int FP8_MAX_EXPONENT        = 7;
    constexpr int FP8_MIN_EXPONENT        = -6;
    constexpr int FP8_EXPONENT_BIAS       = 7;
    constexpr uint8_t FP8_EXPONENT_MASK   = (1u << FP8_NUM_EXPONENT_BITS) - 1u;
    constexpr uint8_t FP8_MANTISSA_MASK   = (1u << FP8_NUM_MANTISSA_BITS) - 1u;
    constexpr uint8_t FP8_MAX_FLT         = 0x7eu;

    auto is_nan_f = [](float v) {
        uint32_t s; std::memcpy(&s, &v, sizeof(s));
        return (s & 0x7fffffffu) > 0x7f800000u;
    };
    auto is_inf_f = [](float v) {
        uint32_t s; std::memcpy(&s, &v, sizeof(s));
        return (s == 0x7f800000u) || (s == 0xff800000u);
    };

    uint32_t s; std::memcpy(&s, &flt, sizeof(s));
    uint8_t sign = uint8_t((s >> 24) & 0x80u);
    int32_t exp = int32_t((s >> FP32_NUM_MANTISSA_BITS) & 0xffu) - FP32_EXPONENT_BIAS;
    int mantissa = int(s & 0x7fffffu);
    uint8_t u = 0;
    uint8_t const kF8_NaN = FP8_NAN;

    if (is_nan_f(flt)) {
        return kF8_NaN;
    }
    if (is_inf_f(flt)) {
        return uint8_t(sign | FP8_MAX_FLT);
    }
    if (exp == -128) {
        return uint8_t(sign | FP8_MAX_FLT);
    }

    int sticky_bit = 0;
    bool skip_sign = false;
    bool may_be_nan = false;

    if (exp >= FP8_MIN_EXPONENT && exp <= FP8_MAX_EXPONENT) {
        exp = exp + FP8_EXPONENT_BIAS;
        u = uint8_t((uint32_t(exp) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
        u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
    } else if (exp < FP8_MIN_EXPONENT) {
        int rshift = (FP8_MIN_EXPONENT - exp);
        if (rshift < FP32_NUM_BITS) {
            mantissa |= (1 << FP32_NUM_MANTISSA_BITS);
            sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);
            mantissa = (mantissa >> rshift);
            u = uint8_t((mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
        } else {
            mantissa = 0;
            u = 0;
        }
    } else {
        if (exp == (FP8_MAX_EXPONENT + 1)) {
            uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            if (mantissa_tmp < FP8_MANTISSA_MASK) {
                exp = exp + FP8_EXPONENT_BIAS;
                u = uint8_t(uint32_t(exp) << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
                may_be_nan = (mantissa_tmp == (FP8_MANTISSA_MASK - 1));
            } else {
                return uint8_t(sign | FP8_MAX_FLT);
            }
        } else {
            return uint8_t(sign | FP8_MAX_FLT);
        }
    }

    int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
    int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
    sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
        u = uint8_t(u + 1);
        if (may_be_nan) {
            skip_sign = true;
        }
    }

    if (u > FP8_MAX_FLT) {
        u = uint8_t(sign | FP8_MAX_FLT);
    }
    if (!skip_sign) {
        u = uint8_t(u | sign);
    }
    return u;
}

template<int E>
inline FP8<E> float_to_fp8(float value) {
    if constexpr (E == 4) {
        FP8<E> out;
        out.bits = float_to_e4m3_bits(value);
        return out;
    }

    FP8<E> out;
    fp32_int32 in = {value};
    // the sign
    out.bits = (in.bits >> 24) & 0x80;
    // value without sign
    in.bits &= 0x7fffffff;
    //GGML_ASSERT(in.bits < 0x7f800000); // +/- infinity or NAN
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
        // OK: S.0000000
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS-127>();
        // - trunc
        //uint32_t eps = 0;
        // - rounding half away from zero
        //uint32_t eps = 0x400000>>FP8<E>::M;
        // - rounding half toward zero
        //uint32_t eps = 0x3fffff>>FP8<E>::M;
        // - rounding to nearest even
        uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
        // shift mantissa.
        in.bits += eps;
        out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
    }
    return out;
}

template<int E>
inline float fp8_to_float(const FP8<E>& in) {
    if constexpr (E == 4) {
        const uint8_t v = in.bits;
        const uint32_t exp = (v >> 3) & 0x0F;
        const uint32_t mant = v & 0x07;
        if (exp == 0x0F && mant == 0x07) {
            fp32_int32 out_nan;
            out_nan.bits = 0x7fffffff; // FP32_NAN pattern used in tests
            return out_nan.f;
        }
    }

    fp32_int32 out = {0};
    out.bits = in.bits & 0x80;
    out.bits <<= 24;
    uint32_t _bits = in.bits & 0x7F;
    _bits <<= (23-FP8<E>::M);
    out.bits |= _bits;
    out.f *= exp_f2<127-FP8<E>::E_BIAS>();
    return out.f;
}

template<int E>
static inline void conv(const FP8<E>* x, float* y, int64_t size) {
    for (int64_t i=0; i<size; i++) {
        y[i] = fp8_to_float(x[i]);
    }
}

template<int E>
static inline void conv(const float* x, FP8<E>* y, int64_t size) {
    for (int64_t i=0; i<size; i++) {
        y[i] = float_to_fp8<E>(x[i]);
    }
}

template <int E, int QK>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E, int QK>
static inline void conv(const bloc_fp8<E, QK>* x, float* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        for (int64_t i=0; i<QK; i++) {
            y[q*QK+i] = fp8_to_float(x[q].qs[i])*(x[q].d);
        }
    }
}

template <int E, int QK>
static inline void conv(const float* x, bloc_fp8<E, QK>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX/m;
        y[q].d = m/FP8<E>::MAX;
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i] = float_to_fp8<E>(x[q*QK+i]*D);
        }
    }
}

// the C API.
void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<5>*>(x), y, k);
}
void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<5>*>(y), k);
}

void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<4>*>(x), y, k);
}
void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<4>*>(y), k);
}

void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<4, QK_K>*>(x), y, k);
}
void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}

void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<3, QK_K>*>(x), y, k);
}
void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}
