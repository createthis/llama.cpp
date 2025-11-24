#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include "../src/llama-sparse-indexer.h"

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <chrono>
using namespace llama;
// Helpers to simulate CUDA half rounding (round-to-nearest-even) on CPU
static inline uint16_t float_to_half_bits_rtne(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t sub = mant >> (1 - exp);
        if (sub & 0x00001000u) sub += 0x00002000u; // round to nearest even
        return (uint16_t)(sign | (sub >> 13));
    } else if (exp >= 31) {
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        mant >>= 13;
        return (uint16_t)(sign | 0x7C00u | mant | (mant == 0));
    } else {
        if (mant & 0x00001000u) {
            mant += 0x00002000u;
            if (mant & 0x00800000u) {
                mant = 0;
                exp += 1;
                if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
            }
        }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }
}
static inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t exp_f = exp - 15 + 127;
        out = sign | (exp_f << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &out, sizeof(f));
    return f;
}
static inline float f32_to_f16_to_f32(float x) {
    return half_bits_to_float(float_to_half_bits_rtne(x));
}


static inline uint16_t float_to_bf16_bits_rtne(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t l = x & 0x0000FFFFu;
    uint32_t u = x >> 16;
    uint32_t round = (l > 0x8000u) || (l == 0x8000u && (u & 1));
    return (uint16_t)(u + round);
}
static inline float bf16_bits_to_float(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; std::memcpy(&f, &u, sizeof(f));
    return f;
}
static inline float f32_to_bf16_to_f32(float x) {
    return bf16_bits_to_float(float_to_bf16_bits_rtne(x));
}


#include "fp8-e4m3-cpu.h"

static void cpu_indexer_logits_bf16like(const float *Q, const float *K, const float *W, const float *k_scale,
                                       int D, int H, int Tc, int kv, std::vector<float> &out) {
    out.assign((size_t)kv*Tc, 0.0f);
    for (int tc=0; tc<Tc; ++tc) {
        for (int i=0; i<kv; ++i) {
            float acc = 0.0f;
            for (int h=0; h<H; ++h) {
                const float *qv = Q + (size_t)D*(tc*H + h);
                const float *kvp= K + (size_t)D*i;
                float dot=0.0f;
                for (int d=0; d<D; ++d) {
                    float qh = f32_to_bf16_to_f32(qv[d]);
                    float kh = f32_to_bf16_to_f32(kvp[d]);
                    dot += qh*kh;
                }
                if (dot < 0.0f) dot = 0.0f; // ReLU
                acc += dot * W[h + (size_t)H*tc];
            }
            out[i + (size_t)kv*tc] = acc * k_scale[i];
        }
    }
}

static void cpu_indexer_logits_f16like(const float *Q, const float *K, const float *W, const float *k_scale,
                                       int D, int H, int Tc, int kv, std::vector<float> &out) {
    out.assign((size_t)kv*Tc, 0.0f);
    for (int tc=0; tc<Tc; ++tc) {
        for (int i=0; i<kv; ++i) {
            float acc = 0.0f;
            for (int h=0; h<H; ++h) {
                const float *qv = Q + (size_t)D*(tc*H + h);
                const float *kvp= K + (size_t)D*i;
                float dot=0.0f; 
                for (int d=0; d<D; ++d) {
                    float qh = f32_to_f16_to_f32(qv[d]);
                    float kh = f32_to_f16_to_f32(kvp[d]);
                    dot += qh*kh;
                }
                if (dot < 0.0f) dot = 0.0f; // ReLU
                acc += dot * W[h + (size_t)H*tc];
            }
            out[i + (size_t)kv*tc] = acc * k_scale[i];
        }
    }
}


static void cpu_indexer_logits(const float *Q, const float *K, const float *W, const float *k_scale,
                               int D, int H, int Tc, int kv, std::vector<float> &out) {
    out.assign((size_t)kv*Tc, 0.0f);
    for (int tc=0; tc<Tc; ++tc) {
        for (int i=0; i<kv; ++i) {
            float acc = 0.0f;
            for (int h=0; h<H; ++h) {
                const float *qv = Q + (size_t)D*(tc*H + h);
                const float *kvp= K + (size_t)D*i;
                float dot=0.0f;
                for (int d=0; d<D; ++d) dot += qv[d]*kvp[d];
                if (dot < 0.0f) dot = 0.0f; // ReLU
                acc += dot * W[h + (size_t)H*tc];
            }
            out[i + (size_t)kv*tc] = acc * k_scale[i];
        }
    }
}

static void cpu_indexer_logits_fp8like(const float *Q, const float *K, const float *W, const float *k_scale,
                                          int D, int H, int Tc, int kv, std::vector<float> &out) {
    out.assign((size_t)kv*Tc, 0.0f);
    // Host-side FP8-like reference for TL_FP8 path.
    // We use the same E4M3 quantization math as the CUDA helper in llama.pp.cu
    // (see float_e4m3_t::from_float / to_float), approximated here by
    // fp8e4m3_cpu::convert_float_to_fp8 / convert_fp8_to_float acting on float.

    // Per-row amax and scale for K: match device kernels k_rowmajor_f32_rowwise_absmax
    // and k_fp8_compute_row_scales up to the FP8 dynamic range choice.
    std::vector<float> K_amax(kv, 0.0f);
    std::vector<float> K_sf(kv, 0.0f);
    for (int i = 0; i < kv; ++i) {
        float maxv = 0.0f;
        const float *kvp = K + (size_t)D * i;
        for (int d = 0; d < D; ++d) {
            float v = std::fabs(kvp[d]);
            if (v > maxv) maxv = v;
        }
        if (maxv < 1e-4f) maxv = 1e-4f;
        K_amax[i] = maxv;
        // DEVICE: s = amax/448; here we reuse the same 448 constant
        K_sf[i]   = maxv / 448.0f;
    }

    for (int tc = 0; tc < Tc; ++tc) {
        for (int i = 0; i < kv; ++i) {
            float acc = 0.0f;
            const float *kvp = K + (size_t)D * i;
            float sf_k = K_sf[i];
            for (int h = 0; h < H; ++h) {
                const float *qv = Q + (size_t)D * (tc*H + h);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    // Q: direct E4M3 quant/dequant
                    float qh = f32_to_fp8e4m3_to_f32(qv[d]);
                    // K: per-row scaled by sf_k before quantization, as in device path
                    float kh = f32_to_fp8e4m3_to_f32(kvp[d] / sf_k);
                    dot += qh * kh;
                }
                if (dot < 0.0f) dot = 0.0f; // ReLU
                acc += dot * W[h + (size_t)H * tc];
            }
            // Effective scale absorbs both device-side fp8 K scale and IndexKScale
            out[i + (size_t)kv * tc] = acc * k_scale[i] * sf_k;
        }
    }
}


int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping fused indexer op test\n");
    return 0;
#else
    const char *tl_fp8_env = std::getenv("LLAMA_TL_FP8");
    const char *wmma_env = std::getenv("LLAMA_INDEXER_USE_WMMA");
    bool use_fp8_ref = ( (tl_fp8_env && std::atoi(tl_fp8_env) != 0) ||
                         (wmma_env && std::atoi(wmma_env) != 0) );

    int D=128, H=16, Tc=2, kv=4096, end=kv/4;
    if (tl_fp8_env && std::atoi(tl_fp8_env) != 0) {
      D=64;
      H=4;
    }

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    std::vector<float> Q((size_t)D*Tc*H), K((size_t)D*kv), W((size_t)H*Tc), KS(kv);
    for (auto &v:Q)  v=dist(rng);
    for (auto &v:K)  v=dist(rng);
    for (auto &v:W)  v=dist(rng);
    for (auto &v:KS) v=std::max(0.1f, std::abs(dist(rng)));

    ggml_init_params ip{};
    ip.mem_size = 64ull*1024*1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("ctx init failed\n");
        return 1;
    }

    ggml_tensor * q2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Tc*H);
    // Build CPU reference that matches the math path (FP16 in TL kernel)

    ggml_tensor * k2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, kv);
    const char *tl = std::getenv("LLAMA_INDEXER_TL_PORT");
    bool use_bf16_ref = (wmma_env && std::atoi(wmma_env) != 0 && H < 16);
    bool use_fp16_ref = (!use_bf16_ref) && ((tl && std::atoi(tl) != 0) || (wmma_env && std::atoi(wmma_env) != 0));

    ggml_tensor * w2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, Tc);
    ggml_tensor * ks  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv);

    // Build simple per-token KV windows for performance test
    std::vector<int32_t> starts_h(Tc, 0);
    std::vector<int32_t> ends_h(Tc, end);
    // Create GGML tensors for starts/ends
    ggml_tensor * starts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tc);
    ggml_tensor * ends   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tc);
    // Use ex variant to pass windows
    ggml_tensor * out = ggml_indexer_logits_fused_ex(ctx, q2d, k2d, w2d, ks, starts, ends);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        // fallback: pick any CUDA-like device if name lookup fails
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                cuda_dev = d;
                break;
            }
        }
    }

    // Continue test setup

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        printf("no CPU device found\n");
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> O_cpu;

    if (use_fp8_ref) {
        // When TL FP8 path is enabled, build scores via the proven CPU Lightning Indexer
        // implementation (idx_compute_scores_tile), then apply FP8 emulation if desired
        ggml_init_params ip_ref{};
        ip_ref.mem_size   = 256ull * 1024 * 1024;
        ip_ref.mem_buffer = nullptr;
        ip_ref.no_alloc   = false;
        ggml_context * ctx_ref = ggml_init(ip_ref);
        if (!ctx_ref) {
            printf("ctx_ref init failed\n");
            return 1;
        }

        const int64_t D_ref  = D;
        const int64_t H_ref  = H;
        const int64_t Tc_ref = Tc;
        const int64_t kv_ref = kv;

        ggml_tensor * q3d  = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, D_ref, Tc_ref, H_ref);
        ggml_tensor * a_k  = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, D_ref, kv_ref);
        ggml_tensor * w2d  = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, H_ref, Tc_ref);
        ggml_tensor * ks1d = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, kv_ref);
        ggml_tensor * ks2d = ggml_reshape_2d(ctx_ref, ks1d, kv_ref, 1);

        // Remap Q from [D, Tc*H] layout (column-major, with column index = tc*H + h)
        // into q3d [D, Tc, H] so that idx_compute_scores_tile sees the same
        // logical q_{t,h} vectors as the fused CUDA kernel.
        {
            std::vector<float> Q3d((size_t)D_ref * (size_t)Tc_ref * (size_t)H_ref);
            for (int64_t tc = 0; tc < Tc_ref; ++tc) {
                for (int64_t h = 0; h < H_ref; ++h) {
                    for (int64_t d = 0; d < D_ref; ++d) {
                        size_t src = (size_t)d + (size_t)D_ref * ((size_t)tc * (size_t)H_ref + (size_t)h);
                        size_t dst = (size_t)d + (size_t)D_ref * ((size_t)tc + (size_t)Tc_ref * (size_t)h);
                        Q3d[dst] = Q[src];
                    }
                }
            }
            std::memcpy(q3d->data, Q3d.data(), Q3d.size() * sizeof(float));
        }
        std::memcpy(a_k->data,  K.data(),  ggml_nbytes(a_k));
        std::memcpy(w2d->data,  W.data(),  ggml_nbytes(w2d));
        std::memcpy(ks1d->data, KS.data(), ggml_nbytes(ks1d));

        ggml_tensor * scores_ref = llama::sparse_attn_indexer::idx_compute_scores_tile(
            ctx_ref,
            q3d,
            a_k,
            w2d,
            ks2d,
            D_ref,
            H_ref,
            Tc_ref,
            kv_ref,
            /*t0=*/0,
            /*use_fp16=*/false);

        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
        ggml_build_forward_expand(gf_ref, scores_ref);
        ggml_graph_compute_with_ctx(ctx_ref, gf_ref, /*n_threads=*/1);

        O_cpu.resize((size_t)kv_ref * (size_t)Tc_ref);
        std::memcpy(O_cpu.data(), scores_ref->data, O_cpu.size() * sizeof(float));

        // NOTE: idx_compute_scores_tile already mirrors the TL FP8 math when
        // LLAMA_TL_FP8 is enabled, so we no longer apply an extra f32->fp8->f32
        // round-trip on the final scores here. This keeps the CPU reference
        // numerically aligned with the TileLang CUDA kernel output.

        ggml_free(ctx_ref);
    } else if (use_bf16_ref) {
        cpu_indexer_logits_bf16like(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_cpu);
    } else if (use_fp16_ref) {
        cpu_indexer_logits_f16like(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_cpu);
    } else {
        cpu_indexer_logits(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_cpu);
    }
    // Zero CPU reference outside window to align with GPU windowing for this test
    for (int tc=0; tc<Tc; ++tc) {
        int end_tc = ends_h[tc];
        for (int i=end_tc; i<kv; ++i) O_cpu[i + (size_t)kv*tc] = 0.0f;
    }



    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu || (!cuda && cuda_dev)) {
        printf("backend init failed\n");
        if (cuda) ggml_backend_free(cuda);
        if (cpu) ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_t backs_arr[2] = { cuda, cpu };
    int n_backs = cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs_arr, nullptr, n_backs, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        printf("sched init failed\n");
        if (cuda) ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_sched_reset(sched);
    // Place all inputs and output on CUDA so fused op runs on GPU
    if (cuda) {
        ggml_backend_sched_set_tensor_backend(sched, q2d, cuda);
        ggml_backend_sched_set_tensor_backend(sched, k2d, cuda);
        ggml_backend_sched_set_tensor_backend(sched, w2d, cuda);
        ggml_backend_sched_set_tensor_backend(sched, ks,  cuda);
        ggml_backend_sched_set_tensor_backend(sched, starts, cuda);
        ggml_backend_sched_set_tensor_backend(sched, ends,   cuda);
        ggml_backend_sched_set_tensor_backend(sched, out,    cuda);
    }
    
    // Place fused op on CUDA backend explicitly so inputs get device buffers
    if (cuda) ggml_backend_sched_set_tensor_backend(sched, out, cuda);
    // Reserve exact buffer sizes to avoid reallocation warnings during alloc_graph
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    // Print backend placement for tensors
    ggml_backend_t bq = ggml_backend_sched_get_tensor_backend(sched, q2d);
    ggml_backend_t bk = ggml_backend_sched_get_tensor_backend(sched, k2d);
    ggml_backend_t bw = ggml_backend_sched_get_tensor_backend(sched, w2d);
    ggml_backend_t bs = ggml_backend_sched_get_tensor_backend(sched, ks);
    printf("placements: q=%s k=%s w=%s s=%s\n",
           bq ? ggml_backend_name(bq) : "null",
           bk ? ggml_backend_name(bk) : "null",
           bw ? ggml_backend_name(bw) : "null",
           bs ? ggml_backend_name(bs) : "null");
    printf("q2d ne=[%lld,%lld] nb=[%zu,%zu] type=%d\n", (long long)q2d->ne[0], (long long)q2d->ne[1], (size_t)q2d->nb[0], (size_t)q2d->nb[1], (int)q2d->type);
    printf("k2d ne=[%lld,%lld] nb=[%zu,%zu] type=%d\n", (long long)k2d->ne[0], (long long)k2d->ne[1], (size_t)k2d->nb[0], (size_t)k2d->nb[1], (int)k2d->type);
    printf("w2d ne=[%lld,%lld] nb=[%zu,%zu] type=%d\n", (long long)w2d->ne[0], (long long)w2d->ne[1], (size_t)w2d->nb[0], (size_t)w2d->nb[1], (int)w2d->type);
    printf("ks  ne=[%lld] nb0=%zu type=%d\n", (long long)ks->ne[0], (size_t)ks->nb[0], (int)ks->type);
    ggml_backend_t bout = ggml_backend_sched_get_tensor_backend(sched, out);
    printf("out ne=[%lld,%lld] nb=[%zu,%zu] type=%d backend=%s\n", (long long)out->ne[0], (long long)out->ne[1], (size_t)out->nb[0], (size_t)out->nb[1], (int)out->type, bout?ggml_backend_name(bout):"null");

    ggml_backend_tensor_set(q2d, Q.data(), 0, ggml_nbytes(q2d));
    ggml_backend_tensor_set(k2d, K.data(), 0, ggml_nbytes(k2d));
    ggml_backend_tensor_set(w2d, W.data(), 0, ggml_nbytes(w2d));
    ggml_backend_tensor_set(ks,  KS.data(),0, ggml_nbytes(ks));
    // ensure starts/ends are uploaded to device
    ggml_backend_tensor_set(starts, starts_h.data(), 0, sizeof(int32_t)*(size_t)Tc);
    ggml_backend_tensor_set(ends,   ends_h.data(),   0, sizeof(int32_t)*(size_t)Tc);

    printf("starting compute\n");
    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    printf("here5.4\n");
    fflush(stdout);
    if (st != GGML_STATUS_SUCCESS) {
        printf("backend compute failed: %d\n", (int)st);
        fflush(stdout);
        ggml_backend_sched_free(sched);
        if (cuda) ggml_backend_free(cuda);
          ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> O_gpu((size_t)kv*Tc, 0.0f);
    ggml_backend_tensor_get(out, O_gpu.data(), 0, ggml_nbytes(out));
    // already computed O_cpu above matching math path
    printf("D=%d H=%d Tc=%d kv=%d\n", D,H,Tc,kv);
    int tt = Tc < 2 ? Tc : 2;
    int kk = kv < 8 ? kv : 8;
    for (int tc=0; tc<tt; ++tc) {
        for (int i=0; i<kk; ++i) {
            float a = O_cpu[i + (size_t)kv*tc];
            float b = O_gpu[i + (size_t)kv*tc];
            printf("out[%d,%d] cpu=%.6f gpu=%.6f diff=%.6f\n", i, tc, a, b, fabsf(a-b));
        }
    }
    printf("W[H,Tc] first 4x3:\n");
    for (int h=0; h<(H<4?H:4); ++h) {
        for (int tc=0; tc<Tc; ++tc) {
            printf(" % .6f", W[h + (size_t)H*tc]);
        }
        printf("\n");
    }
    printf("KS first 8: ");
    for (int i=0; i<(kv<8?kv:8); ++i) printf(" % .6f", KS[i]);
    printf("\n");

    int mism=0;
    float max_abs=0.0f;
    for (size_t i=0;i<O_cpu.size();++i){
        float da = std::abs(O_cpu[i]-O_gpu[i]);
        if (da>max_abs) max_abs=da;
        if (da>1e-3f) mism++;
    }
    printf("fused op ggml test: mism=%d max_abs=%.6f\n", mism, max_abs);
    printf("TEST %s\n", mism==0?"PASS":"FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);


    // Optional prefill-scale bench (env-controlled)
    const char *BENCH = std::getenv("LLAMA_INDEXER_BENCH");
    if (BENCH && std::atoi(BENCH) != 0) {
        int BD = std::getenv("LLAMA_INDEXER_BENCH_D") ? std::atoi(std::getenv("LLAMA_INDEXER_BENCH_D")) : 128;
        int BH = std::getenv("LLAMA_INDEXER_BENCH_H") ? std::atoi(std::getenv("LLAMA_INDEXER_BENCH_H")) : 64;
        int BT = std::getenv("LLAMA_INDEXER_BENCH_T") ? std::atoi(std::getenv("LLAMA_INDEXER_BENCH_T")) : 512;
        int BK = std::getenv("LLAMA_INDEXER_BENCH_KV") ? std::atoi(std::getenv("LLAMA_INDEXER_BENCH_KV")) : 32768;
        int iters = std::getenv("LLAMA_INDEXER_BENCH_ITERS") ? std::atoi(std::getenv("LLAMA_INDEXER_BENCH_ITERS")) : 10;
        printf("[BENCH] running prefill-scale bench D=%d H=%d Tc=%d kv=%d iters=%d\n", BD, BH, BT, BK, iters);
        ggml_init_params ip2{};
        ip2.mem_size = 256ull*1024*1024;
        ip2.no_alloc = true;
        ggml_context * ctx2 = ggml_init(ip2);
        if (!ctx2) printf("ctx2 init failed\n");
        else {
            std::mt19937 rng2(2025);
            std::uniform_real_distribution<float> dist2(-1.0f,1.0f);
            std::vector<float> QB((size_t)BD*BT*BH), KB((size_t)BD*BK), WB((size_t)BH*BT), KSB(BK);
            for (auto &v:QB)  v=dist2(rng2);
            for (auto &v:KB)  v=dist2(rng2);
            for (auto &v:WB)  v=dist2(rng2);
            for (auto &v:KSB) v=std::max(0.1f, std::abs(dist2(rng2)));
            ggml_tensor * q2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, BD, BT*BH);
            ggml_tensor * k2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, BD, BK);
            ggml_tensor * w2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, BH, BT);
            ggml_tensor * ks2= ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, BK);
            ggml_backend_dev_t cuda_dev2 = ggml_backend_dev_by_name("CUDA0");
            if (!cuda_dev2) {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        cuda_dev2 = d;
                        break;
                    }
                }
            }
            ggml_backend_dev_t cpu_dev2 = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_t cuda2 = cuda_dev2 ? ggml_backend_dev_init(cuda_dev2, nullptr) : nullptr;
            ggml_backend_t cpu2  = ggml_backend_dev_init(cpu_dev2, nullptr);
            ggml_backend_t backs2[2] = { cuda2, cpu2 };
            int n_backs2 = cuda2 ? 2 : 1;
            ggml_backend_sched_t sched2 = ggml_backend_sched_new(backs2, nullptr, n_backs2, GGML_DEFAULT_GRAPH_SIZE, false, true);
            ggml_tensor * out2 = ggml_indexer_logits_fused(ctx2, q2, k2, w2, ks2);
            ggml_cgraph * gf2 = ggml_new_graph(ctx2);
            ggml_build_forward_expand(gf2, out2);
            ggml_backend_sched_reset(sched2);
            ggml_backend_sched_alloc_graph(sched2, gf2);
            // set data after allocation
            ggml_backend_tensor_set(q2, QB.data(), 0, ggml_nbytes(q2));
            ggml_backend_tensor_set(k2, KB.data(), 0, ggml_nbytes(k2));
            ggml_backend_tensor_set(w2, WB.data(), 0, ggml_nbytes(w2));
            ggml_backend_tensor_set(ks2,KSB.data(), 0, ggml_nbytes(ks2));
            // warmup
            ggml_backend_sched_graph_compute(sched2, gf2);
            double sum_ms=0.0;
            for (int it=0; it<iters; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                ggml_backend_sched_graph_compute(sched2, gf2);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = 1e3 * std::chrono::duration<double>(t1 - t0).count();
                sum_ms += ms;
            }
            double avg = sum_ms / iters;
            printf("[BENCH] FUSED_INDEXER avg_ms=%.3f (D=%d H=%d Tc=%d kv=%d iters=%d)\n", avg, BD, BH, BT, BK, iters);
            ggml_backend_sched_free(sched2);
            if (cuda2) ggml_backend_free(cuda2);
            ggml_backend_free(cpu2);
            ggml_free(ctx2);
        }
    }
    return mism==0?0:1;
#endif
}
