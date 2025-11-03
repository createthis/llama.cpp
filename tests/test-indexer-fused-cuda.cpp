#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#ifdef GGML_USE_CUDA
#include <ggml-cuda-indexer.h>
#endif

static void cpu_indexer_logits(const float *Q, const float *K, const float *W, const float *k_scale,
                               int D, int H, int Tc, int kv, std::vector<float> &out) {
    out.assign((size_t)kv*Tc, 0.0f);
    for (int tc=0; tc<Tc; ++tc) {
        for (int i=0; i<kv; ++i) {
            float acc = 0.0f;
            for (int h=0; h<H; ++h) {
                const float *qv = Q + (size_t)D*(tc*H + h);
                const float *kvp= K + (size_t)D*i;
                float dot=0.0f; for (int d=0; d<D; ++d) dot += qv[d]*kvp[d];
                if (dot < 0.0f) dot = 0.0f; // ReLU
                acc += dot * W[h + (size_t)H*tc];
            }
            out[i + (size_t)kv*tc] = acc * k_scale[i];
        }
    }
}

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping fused indexer test\n");
    return 0;
#else
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    int D=64, H=4, Tc=3, kv=128;
    std::vector<float> Q((size_t)D*Tc*H), K((size_t)D*kv), W((size_t)H*Tc), KS(kv);
    for (auto &v:Q) v=dist(rng);
    for (auto &v:K) v=dist(rng);
    for (auto &v:W) v=dist(rng);
    for (auto &v:KS) v=std::max(0.1f, std::abs(dist(rng)));
    std::vector<float> O_cpu, O_gpu((size_t)kv*Tc);
    cpu_indexer_logits(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_cpu);
    ggml_cuda_indexer_logits_fused_host(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_gpu.data());
    // compare
    int mism=0; float max_abs=0.0f;
    for (size_t i=0;i<O_cpu.size();++i){
        float a=O_cpu[i], b=O_gpu[i];
        float da=std::abs(a-b); if (da>max_abs) max_abs=da;
        if (da>1e-3f) mism++;
    }
    printf("fused indexer logits test: mism=%d max_abs=%.6f\n", mism, max_abs);
    if (mism==0) { printf("TEST PASS\n"); return 0; } else { printf("TEST FAIL\n"); return 1; }
#endif
}
