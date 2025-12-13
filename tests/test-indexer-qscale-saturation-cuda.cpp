#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static std::vector<int> topk_indices_host(const std::vector<float> & logits, int k) {
    const int n = (int) logits.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    auto cmp = [&](int a, int b) {
        const float va = logits[a];
        const float vb = logits[b];
        if (va != vb) return va > vb;
        return a < b;
    };

    if (k > n) k = n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
    idx.resize(k);
    return idx;
}

static int count_mod(const std::vector<int> & idx, int mod, int target) {
    int c = 0;
    for (int v : idx) {
        if (v % mod == target) ++c;
    }
    return c;
}

int main() {
#ifndef GGML_USE_CUDA
    std::printf("CUDA not enabled; skipping indexer qscale saturation test\n");
    return 0;
#else
    // This test demonstrates a known failure mode of the current WMMA-HGRP fused
    // indexer kernel: Q is quantized to FP8 via direct cast (scale=1), so large
    // values saturate to FP8 max (~448) and many logits tie, causing unstable top-k.
    //
    // Expected behavior (vLLM): compute per-(token,head) q_scale and apply it
    // (typically folded into weights), preventing saturation.
    //
    // The test constructs K as repeated basis vectors so each KV row score depends
    // on a single Q dimension. We create 8 large Q dims with distinct magnitudes.
    // With correct q_scale, the highest dim (127) should dominate all top-k.
    // Without q_scale, these 8 dims saturate and tie, so top-k includes all 8.

    const int D  = 128;
    const int H  = 64;
    const int Tc = 1;
    const int kv = 8192;          // must satisfy Tc*kv > 4096 for WMMA path
    const int top_k = 64;

    // Force WMMA path and disable other experimental paths.
    setenv("LLAMA_INDEXER_USE_WMMA", "1", 1);
    setenv("LLAMA_DG_FP8", "0", 1);
    setenv("LLAMA_INDEXER_TL_PORT", "0", 1);

    // Build Q2d [D, Tc*H], K2d [D, kv], W2d [H, Tc], KS [kv]
    std::vector<float> Q((size_t)D * (size_t)Tc * (size_t)H, 0.0f);
    std::vector<float> K((size_t)D * (size_t)kv, 0.0f);
    std::vector<float> W((size_t)H * (size_t)Tc, 0.0f);
    std::vector<float> KS((size_t)kv, 1.0f);

    // One-hot weights: only head 0 contributes.
    W[0] = 1.0f;

    // K as repeated basis vectors: row i attends to dim d = i % D.
    for (int i = 0; i < kv; ++i) {
        const int d = i % D;
        K[(size_t)d + (size_t)D * (size_t)i] = 1.0f;
    }

    // Construct Q for head 0: dims 120..127 have descending magnitudes.
    // Choose max so UE8M0 q_scale would be exactly 8 (max = 448*8 = 3584).
    // Values after scaling would be [448,416,...,224], which are representable.
    for (int d = 120; d <= 127; ++d) {
        const int j = 127 - d;
        const float q_div = 448.0f - 32.0f * (float)j; // 448,416,...
        const float q = 8.0f * q_div;                  // 3584,3328,...
        Q[(size_t)d + (size_t)D * 0] = q; // token 0, head 0 => column 0
    }

    ggml_init_params ip{};
    ip.mem_size   = 128ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::printf("ctx init failed\n");
        return 1;
    }

    ggml_tensor * q2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Tc * H);
    ggml_tensor * k2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, kv);
    ggml_tensor * w2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, Tc);
    ggml_tensor * ks  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv);

    ggml_tensor * starts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tc);
    ggml_tensor * ends   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tc);

    // Full-window.
    int32_t starts_h[1] = { 0 };
    int32_t ends_h[1]   = { kv };

    ggml_tensor * out = ggml_indexer_logits_fused_ex(ctx, q2d, k2d, w2d, ks,
                                                     starts, ends,
                                                     /*k_indexer_fp8_sidecar*/ nullptr,
                                                     /*quant_bs*/ 0,
                                                     /*cache_block_size*/ 0,
                                                     /*cache_stride*/ 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                cuda_dev = d;
                break;
            }
        }
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::printf("no CPU backend\n");
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cuda || !cpu) {
        std::printf("backend init failed\n");
        if (cuda) ggml_backend_free(cuda);
        if (cpu) ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_t backs[2] = { cuda, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        std::printf("sched init failed\n");
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_set_tensor_backend(sched, q2d, cuda);
    ggml_backend_sched_set_tensor_backend(sched, k2d, cuda);
    ggml_backend_sched_set_tensor_backend(sched, w2d, cuda);
    ggml_backend_sched_set_tensor_backend(sched, ks,  cuda);
    ggml_backend_sched_set_tensor_backend(sched, starts, cuda);
    ggml_backend_sched_set_tensor_backend(sched, ends,   cuda);
    ggml_backend_sched_set_tensor_backend(sched, out,    cuda);

    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(q2d, Q.data(), 0, ggml_nbytes(q2d));
    ggml_backend_tensor_set(k2d, K.data(), 0, ggml_nbytes(k2d));
    ggml_backend_tensor_set(w2d, W.data(), 0, ggml_nbytes(w2d));
    ggml_backend_tensor_set(ks,  KS.data(), 0, ggml_nbytes(ks));
    ggml_backend_tensor_set(starts, starts_h, 0, sizeof(starts_h));
    ggml_backend_tensor_set(ends,   ends_h,   0, sizeof(ends_h));

    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("graph compute failed: %d\n", (int)st);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> logits((size_t)kv);
    ggml_backend_tensor_get(out, logits.data(), 0, ggml_nbytes(out));

    // Compute top-k from GPU logits.
    std::vector<int> topk = topk_indices_host(logits, top_k);

    // If q_scale is applied correctly, the highest-magnitude dim (127) should
    // dominate, and top-k should be exclusively indices where (idx % D == 127).
    const int want_dim = 127;
    const int got_dim127 = count_mod(topk, D, want_dim);

    std::printf("[qscale-sat] top_k=%d count(idx%%D==%d)=%d\n", top_k, want_dim, got_dim127);

    // Print a small sample for debugging.
    std::printf("[qscale-sat] first 16 topk indices:");
    for (int i = 0; i < 16 && i < (int)topk.size(); ++i) std::printf(" %d", topk[i]);
    std::printf("\n");

    // Threshold: after a q_scale fix, expect almost all to be dim 127.
    // Current buggy behavior (direct-cast FP8 Q) yields ties and includes dims 120..127,
    // so dim127 count is typically ~8.
    const int threshold = 56;
    const bool ok = got_dim127 >= threshold;
    std::printf("TEST %s\n", ok ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    return ok ? 0 : 1;
#endif
}
