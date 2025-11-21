#pragma once

#include <cstdint>
#include <cstring>

// Minimal E4M3 (FP8) emulation matching CUDA convert_float_to_fp8/convert_fp8_to_float.
// This helper is used by multiple tests to ensure they exercise the same host-side
// FP8 behavior.
struct fp8e4m3_cpu {
    static constexpr bool   IS_E4M3                 = true;
    static constexpr int    FP32_NUM_BITS          = 32;
    static constexpr int    FP32_NUM_EXPONENT_BITS = 8;
    static constexpr int    FP32_NUM_MANTISSA_BITS = 23;
    static constexpr uint32_t FP32_NAN             = 0x7fffffffu;
    static constexpr uint32_t FP32_INFINITY_MASK   = 0x7f800000u;
    static constexpr int    FP32_MAX_EXPONENT      = 127;
    static constexpr int    FP32_MIN_EXPONENT      = -126;
    static constexpr int    FP32_EXPONENT_BIAS     = 127;

    static constexpr int    FP8_NUM_BITS           = 8;
    static constexpr int    FP8_NUM_EXPONENT_BITS  = 4;
    static constexpr int    FP8_NUM_MANTISSA_BITS  = 3;
    static constexpr uint8_t FP8_NAN               = 0x7fu;
    static constexpr uint8_t FP8_INFINITY_MASK     = 0x78u;
    static constexpr int    FP8_MAX_EXPONENT       = 7;
    static constexpr int    FP8_MIN_EXPONENT       = -6;
    static constexpr int    FP8_EXPONENT_BIAS      = 7;

    static constexpr uint8_t FP8_EXPONENT_MASK     = (1u << FP8_NUM_EXPONENT_BITS) - 1u;
    static constexpr uint8_t FP8_MANTISSA_MASK     = (1u << FP8_NUM_MANTISSA_BITS) - 1u;
    static constexpr uint8_t FP8_MAX_FLT           = 0x7eu;

    static inline bool isfinite(float flt) {
        uint32_t s; std::memcpy(&s, &flt, sizeof(s));
        return (s & 0x7f800000u) < 0x7f800000u;
    }

    static inline bool isnan(float flt) {
        uint32_t s; std::memcpy(&s, &flt, sizeof(s));
        return (s & 0x7fffffffu) > 0x7f800000u;
    }

    static inline bool isinf(float flt) {
        uint32_t s; std::memcpy(&s, &flt, sizeof(s));
        return (s == 0x7f800000u) || (s == 0xff800000u);
    }

    static inline uint8_t convert_float_to_fp8(float const &flt) {
        uint32_t s; std::memcpy(&s, &flt, sizeof(s));
        uint8_t sign = uint8_t((s >> 24) & 0x80u);
        int32_t exp = int32_t((s >> FP32_NUM_MANTISSA_BITS) & 0xffu) - FP32_EXPONENT_BIAS;
        int mantissa = int(s & 0x7fffffu);
        uint8_t u = 0;
        uint8_t const kF8_NaN = FP8_NAN;

        if (isnan(flt)) {
            return kF8_NaN;
        }
        if (isinf(flt)) {
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

    static inline float convert_fp8_to_float(uint8_t const &x) {
        uint32_t constexpr kF32_NAN = FP32_NAN;
        uint8_t const &f8 = x;
        uint32_t sign = (f8 >> (FP8_NUM_BITS - 1)) & 1u;
        uint32_t exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        uint32_t mantissa = f8 & FP8_MANTISSA_MASK;
        unsigned f = (sign << (FP32_NUM_BITS - 1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
            f = kF32_NAN;
        } else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);
            f = f |
                (exp << FP32_NUM_MANTISSA_BITS) |
                (mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
        } else if (exp == 0) {
            if (mantissa) {
                exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
                while ((mantissa & (1u << FP8_NUM_MANTISSA_BITS)) == 0u) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= FP8_MANTISSA_MASK;
                f = f |
                    (exp << FP32_NUM_MANTISSA_BITS) |
                    (mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            } else {
                // zero
            }
        } else {
            if (mantissa == 0) {
                f = (f | 0x7f800000u);
            } else {
                f = kF32_NAN;
            }
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }
};

inline float f32_to_fp8e4m3_to_f32(float x) {
    uint8_t b = fp8e4m3_cpu::convert_float_to_fp8(x);
    return fp8e4m3_cpu::convert_fp8_to_float(b);
}
