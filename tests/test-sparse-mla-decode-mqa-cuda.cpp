#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

static void ref_sparse_decode_mqa(
    const float *Q,        // [D * Hq]
    const float *K,        // [D * Hkv * N]
    const float *V,        // [Dv * Hkv * N]
    const int32_t *topk,   // [Ksel]
    int D, int Hq, int Hkv, int Dv, int N, int Ksel,
    float kq_scale, float softcap,
    std::vector<float> &Out) { // [Dv * Hq]

    Out.assign((size_t)Dv*Hq, 0.0f);
    for (int h = 0; h < Hq; ++h) {
        const int hk = (Hkv == 1 ? 0 : (h % Hkv));
        std::vector<float> scores(Ksel);
        float m = -1e30f;
        for (int i = 0; i < Ksel; ++i) {
            int idx = topk[i];
            float dot = -1e30f;
            if (idx >= 0 && idx < N) {
                dot = 0.0f;
                const float * qh = Q + (size_t)D*h;
                const float * kh = K + (size_t)D*(hk + (size_t)Hkv*idx);
                for (int d = 0; d < D; ++d) dot += qh[d]*kh[d];
                dot *= kq_scale;
                if (softcap > 0.f) dot = std::tanh(dot/softcap)*softcap;
            }
            scores[i] = dot; m = std::max(m, dot);
        }
        float ssum = 0.0f;
        for (int i = 0; i < Ksel; ++i) { scores[i] = std::exp(scores[i]-m); ssum += scores[i]; }
        float inv = 1.0f/ssum;
        for (int dv = 0; dv < Dv; ++dv) {
            float acc = 0.0f;
            for (int i = 0; i < Ksel; ++i) {
                int idx = topk[i]; if (idx < 0 || idx >= N) continue;
                float p = scores[i] * inv;
                const float * vh = V + (size_t)Dv*(hk + (size_t)Hkv*idx);
                acc += p * vh[dv];
            }
            Out[dv + (size_t)Dv*h] = acc;
        }
    }
}

int main(){
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping sparse mla decode MQA test\n");
    return 0;
#else
    // MQA/GQA repro: Hq != Hkv
    const int D=128, Hq=8, Hkv=1, Dv=128, N=1024, Ksel=64;
    float kq_scale = 1.0f; float softcap = 0.0f;
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    std::vector<float> Q((size_t)D*Hq), K((size_t)D*Hkv*N), V((size_t)Dv*Hkv*N);
    std::vector<int32_t> TOPK(Ksel);
    for (auto &v:Q) v=dist(rng);
    for (auto &v:K) v=dist(rng);
    for (auto &v:V) v=dist(rng);
    for (int i=0;i<Ksel;++i) TOPK[i] = (i*37) % N; // spread indices

    ggml_init_params ip{}; ip.mem_size = 64ull*1024*1024; ip.no_alloc=true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Hq);
    ggml_tensor * kc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
    ggml_tensor * vc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dv, Hkv, N);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Ksel);

    ggml_tensor * out = ggml_sparse_mla_decode_fused(ctx, q2d, kc, vc, idx, kq_scale, softcap);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t cpu_dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t backs[2] = {cuda, cpu};
    int nb = cuda?2:1;

    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, nb, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(q2d, Q.data(), 0, ggml_nbytes(q2d));
    ggml_backend_tensor_set(kc,  K.data(), 0, ggml_nbytes(kc));
    ggml_backend_tensor_set(vc,  V.data(), 0, ggml_nbytes(vc));
    ggml_backend_tensor_set(idx, TOPK.data(), 0, ggml_nbytes(idx));

    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (st != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }

    std::vector<float> O_gpu((size_t)Dv*Hq);
    ggml_backend_tensor_get(out, O_gpu.data(), 0, ggml_nbytes(out));

    std::vector<float> O_ref;
    ref_sparse_decode_mqa(Q.data(), K.data(), V.data(), TOPK.data(), D,Hq,Hkv,Dv,N,Ksel,kq_scale,softcap,O_ref);
    float max_abs=0.0f; int mism=0; for (size_t i=0;i<O_ref.size();++i){ float d=fabsf(O_ref[i]-O_gpu[i]); if (d>max_abs) max_abs=d; if (d>5e-3f) mism++; }
    printf("sparse mla decode fused (MQA): mism=%d max_abs=%.6f\n", mism, max_abs);

    ggml_backend_sched_free(sched);
    if (cuda) ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    printf("TEST %s\n", mism==0?"PASS":"FAIL");
    return mism==0?0:1;
#endif
}
