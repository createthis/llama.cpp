#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>

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
    printf("CUDA not enabled; skipping fused indexer op test\n");
    return 0;
#else
    const int D=64, H=4, Tc=3, kv=256;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    std::vector<float> Q((size_t)D*Tc*H), K((size_t)D*kv), W((size_t)H*Tc), KS(kv);
    for (auto &v:Q)  v=dist(rng);
    for (auto &v:K)  v=dist(rng);
    for (auto &v:W)  v=dist(rng);
    for (auto &v:KS) v=std::max(0.1f, std::abs(dist(rng)));

    ggml_init_params ip{}; ip.mem_size = 64ull*1024*1024; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("ctx init failed\n"); return 1; }

    ggml_tensor * q2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Tc*H);
    ggml_tensor * k2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, kv);
    ggml_tensor * w2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, Tc);
    ggml_tensor * ks  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv);

    ggml_tensor * out = ggml_indexer_logits_fused(ctx, q2d, k2d, w2d, ks);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        // fallback: pick any CUDA-like device if name lookup fails
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { cuda_dev = d; break; }
        }
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) { printf("no CPU device found\n"); ggml_free(ctx); return 1; }

    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu || (!cuda && cuda_dev)) { printf("backend init failed\n"); if (cuda) ggml_backend_free(cuda); if (cpu) ggml_backend_free(cpu); ggml_free(ctx); return 1; }

    ggml_backend_t backs_arr[2] = { cuda, cpu };
    int n_backs = cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs_arr, nullptr, n_backs, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { printf("sched init failed\n"); if (cuda) ggml_backend_free(cuda); ggml_backend_free(cpu); ggml_free(ctx); return 1; }

    ggml_backend_sched_reset(sched);
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

    std::vector<float> O_cpu;
    cpu_indexer_logits(Q.data(), K.data(), W.data(), KS.data(), D,H,Tc,kv, O_cpu);
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

    int mism=0; float max_abs=0.0f;
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
        ggml_init_params ip2{}; ip2.mem_size = 256ull*1024*1024; ip2.no_alloc = true;
        ggml_context * ctx2 = ggml_init(ip2);
        if (!ctx2) { printf("ctx2 init failed\n"); }
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
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { cuda_dev2 = d; break; }
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
