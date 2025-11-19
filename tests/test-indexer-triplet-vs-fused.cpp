#include "../src/llama-sparse-indexer.h"
#include "../src/llama-sparse-topk.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml.h>
#include <ggml-cpp.h>
#include <ggml-cuda.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace llama;

struct CpuContext {
    ggml_context * ctx;
    CpuContext() {
        ggml_init_params p{};
        p.mem_size   = 256ull * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = false;
        ctx = ggml_init(p);
    }
    ~CpuContext() {
        if (ctx) ggml_free(ctx);
    }
};

static llama_model * create_test_model(ggml_context * ctx, int num_layers = 1) {
    llama_model_params params = llama_model_default_params();
    llama_model * model = new llama_model(params);
    model->arch = LLM_ARCH_DEEPSEEK3_2;
    model->layers.resize(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        llama_layer & layer = model->layers[i];

        const int64_t hidden_dim     = 512;
        const int64_t index_head_dim = 128;
        const int64_t index_n_heads  = 64;

        layer.attn_indexer_wk          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_head_dim);
        layer.attn_indexer_wq_b        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        layer.attn_indexer_weights_proj= ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_n_heads);
        layer.attn_indexer_k_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, index_head_dim);
        layer.wq_a                     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);

        auto fill_tensor = [](ggml_tensor * t) {
            size_t n = ggml_nelements(t);
            std::vector<float> tmp(n);
            for (size_t k = 0; k < n; ++k) tmp[k] = (float) rand() / RAND_MAX;
            memcpy(t->data, tmp.data(), n * sizeof(float));
        };

        fill_tensor(layer.attn_indexer_wk);
        fill_tensor(layer.attn_indexer_wq_b);
        fill_tensor(layer.attn_indexer_weights_proj);
        fill_tensor(layer.attn_indexer_k_norm_bias);
        fill_tensor(layer.wq_a);
    }

    return model;
}

static void cleanup_test_model(llama_model * model) {
    delete model;
}

int main() {
#ifndef GGML_USE_CUDA
    printf("CUDA not enabled; skipping indexer triplet vs fused test\n");
    return 0;
#else
    srand(42);

    CpuContext cpu_ctx;
    if (!cpu_ctx.ctx) {
        printf("cpu_ctx init failed\n");
        return 1;
    }

    llama_model * model = create_test_model(cpu_ctx.ctx, 1);

    const int64_t n_tokens = 2;
    const int64_t hidden_dim = 512;

    ggml_tensor * cur = ggml_new_tensor_2d(cpu_ctx.ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    std::vector<float> cur_data((size_t) hidden_dim * (size_t) n_tokens);
    for (size_t i = 0; i < cur_data.size(); ++i) cur_data[i] = (float) rand() / RAND_MAX;
    memcpy(cur->data, cur_data.data(), cur_data.size() * sizeof(float));

    auto cb_noop = [](ggml_tensor *, const char *, int) {};

    IndexerKVTriplet trip = sparse_attn_indexer::compute_indexer_triplet(
        cpu_ctx.ctx, *model, 0, cur, n_tokens,
        /*mctx*/ nullptr, /*k_idxs*/ nullptr,
        /*inp_pos*/ nullptr, /*n_rot*/ 0, /*rope_type*/ 0, /*n_ctx_orig*/ n_tokens,
        /*freq_base*/ 10000.0f, /*freq_scale*/ 1.0f,
        /*ext_factor*/ 1.0f, /*attn_factor*/ 1.0f,
        /*beta_fast*/ 1.0f, /*beta_slow*/ 1.0f,
        cb_noop, /*gf*/ nullptr);

    ggml_tensor * q_indexer = trip.q_indexer;      // [D_index, H_index, T]
    ggml_tensor * k_indexer = trip.k_indexer_cache;// [D_index, N_kv]
    ggml_tensor * idx_weights = trip.idx_weights;  // [H_index, T]

    if (!q_indexer || !k_indexer || !idx_weights) {
        printf("triplet tensors are null; aborting\n");
        cleanup_test_model(model);
        return 1;
    }

    const int64_t D_index = q_indexer->ne[0];
    const int64_t H_index = q_indexer->ne[1];
    const int64_t T       = q_indexer->ne[2];
    const int64_t N_kv    = k_indexer->ne[1];

    printf("[IDX_TRIPLET] D=%" PRId64 " H=%" PRId64 " T=%" PRId64 " N_kv=%" PRId64 "\n",
           D_index, H_index, T, N_kv);

    // Build k_scale_2d as ones so it does not alter comparison
    ggml_tensor * ks_vec = ggml_new_tensor_1d(cpu_ctx.ctx, GGML_TYPE_F32, N_kv);
    std::vector<float> ks_host((size_t) N_kv, 1.0f);
    memcpy(ks_vec->data, ks_host.data(), ks_host.size() * sizeof(float));
    ggml_tensor * k_scale_2d = ggml_reshape_2d(cpu_ctx.ctx, ks_vec, N_kv, 1);

    // Build inputs for idx_compute_scores_tile: q3d [D,T,H], a_k [D,N_kv], w2d [H,T]
    // q_indexer is [D,H,T]; match sparse_topk pipeline: permute to [D,T,H] then contiguize.
    ggml_tensor * q_perm = ggml_permute(cpu_ctx.ctx, q_indexer, 0, 2, 1, 3);
    ggml_tensor * q3d   = ggml_cont(cpu_ctx.ctx, q_perm);
    ggml_tensor * a_k   = ggml_reshape_2d(cpu_ctx.ctx, k_indexer, D_index, N_kv);
    ggml_tensor * w2d   = idx_weights; // already [H_index, T]

    ggml_tensor * scores_tc = sparse_attn_indexer::idx_compute_scores_tile(
        cpu_ctx.ctx, q3d, a_k, w2d, k_scale_2d,
        D_index, H_index, T, N_kv,
        /*t0=*/0, /*use_fp16=*/false);

    ggml_cgraph * gf_cpu = ggml_new_graph(cpu_ctx.ctx);
    ggml_build_forward_expand(gf_cpu, scores_tc);
    ggml_graph_compute_with_ctx(cpu_ctx.ctx, gf_cpu, /*n_threads=*/1);

    std::vector<float> scores_trip((size_t) N_kv * (size_t) T);
    memcpy(scores_trip.data(), scores_tc->data, scores_trip.size() * sizeof(float));

    // Host buffers laid out as fused kernel expects: Q2d [D, T*H], K2d [D,N_kv], W2d [H,T], KS [N_kv]
    std::vector<float> Q2d((size_t) D_index * (size_t) T * (size_t) H_index);
    std::vector<float> K2d((size_t) D_index * (size_t) N_kv);
    std::vector<float> W2d((size_t) H_index * (size_t) T);

    auto load3 = [](ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
        char * base = (char *) t->data;
        size_t off = (size_t) i0 * t->nb[0]
                   + (size_t) i1 * t->nb[1]
                   + (size_t) i2 * t->nb[2];
        return *(float *) (base + off);
    };

    auto load2 = [](ggml_tensor * t, int64_t i0, int64_t i1) {
        char * base = (char *) t->data;
        size_t off = (size_t) i0 * t->nb[0]
                   + (size_t) i1 * t->nb[1];
        return *(float *) (base + off);
    };

    // Q2d: [D_index, T*H_index], column index = t*H_index + h
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H_index; ++h) {
            size_t col = (size_t) t * (size_t) H_index + (size_t) h;
            for (int64_t d = 0; d < D_index; ++d) {
                float v = load3(q3d, d, t, h);
                Q2d[(size_t) d + (size_t) D_index * col] = v;
            }
        }
    }

    // K2d: [D_index, N_kv]
    for (int64_t kv_i = 0; kv_i < N_kv; ++kv_i) {
        for (int64_t d = 0; d < D_index; ++d) {
            float v = load2(a_k, d, kv_i);
            K2d[(size_t) d + (size_t) D_index * (size_t) kv_i] = v;
        }
    }

    // W2d: [H_index, T]
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H_index; ++h) {
            float v = load2(idx_weights, h, t);
            W2d[(size_t) h + (size_t) H_index * (size_t) t] = v;
        }
    }

    // --- Fused CUDA path ---
    ggml_init_params ip_fused{};
    ip_fused.mem_size = 64ull * 1024 * 1024;
    ip_fused.no_alloc = true;
    ggml_context * ctx_fused = ggml_init(ip_fused);
    if (!ctx_fused) {
        printf("ctx_fused init failed\n");
        cleanup_test_model(model);
        return 1;
    }

    ggml_tensor * q2d = ggml_new_tensor_2d(ctx_fused, GGML_TYPE_F32, D_index, T * H_index);
    ggml_tensor * k2d = ggml_new_tensor_2d(ctx_fused, GGML_TYPE_F32, D_index, N_kv);
    ggml_tensor * w2d_fused = ggml_new_tensor_2d(ctx_fused, GGML_TYPE_F32, H_index, T);
    ggml_tensor * ks  = ggml_new_tensor_1d(ctx_fused, GGML_TYPE_F32, N_kv);

    ggml_tensor * out = ggml_indexer_logits_fused_ex(ctx_fused, q2d, k2d, w2d_fused, ks, nullptr, nullptr);
    ggml_cgraph * gf_fused = ggml_new_graph(ctx_fused);
    ggml_build_forward_expand(gf_fused, out);

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

    if (!cuda_dev) {
        printf("no CUDA device found; skipping fused comparison\n");
        ggml_free(ctx_fused);
        cleanup_test_model(model);
        return 0;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cuda = ggml_backend_dev_init(cuda_dev, nullptr);
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);

    ggml_backend_t backs_arr[2] = { cuda, cpu };
    int n_backs = cuda ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs_arr, nullptr, n_backs, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        printf("sched init failed\n");
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_free(ctx_fused);
        cleanup_test_model(model);
        return 1;
    }

    ggml_backend_sched_reset(sched);
    if (cuda) {
        ggml_backend_sched_set_tensor_backend(sched, q2d, cuda);
        ggml_backend_sched_set_tensor_backend(sched, k2d, cuda);
        ggml_backend_sched_set_tensor_backend(sched, w2d_fused, cuda);
        ggml_backend_sched_set_tensor_backend(sched, ks,  cuda);
        ggml_backend_sched_set_tensor_backend(sched, out, cuda);
    }
    ggml_backend_sched_reserve(sched, gf_fused);
    ggml_backend_sched_alloc_graph(sched, gf_fused);

    // Upload host data
    ggml_backend_tensor_set(q2d, Q2d.data(), 0, ggml_nbytes(q2d));
    ggml_backend_tensor_set(k2d, K2d.data(), 0, ggml_nbytes(k2d));
    ggml_backend_tensor_set(w2d_fused, W2d.data(), 0, ggml_nbytes(w2d_fused));
    ggml_backend_tensor_set(ks,  ks_host.data(), 0, ggml_nbytes(ks));

    // Prefer tiled CUDA indexer; exactness already validated by test-indexer-fused-op-cuda
    //setenv("LLAMA_INDEXER_TL_PORT", "0", 1);

    ggml_backend_sched_graph_compute(sched, gf_fused);

    std::vector<float> scores_fused((size_t) N_kv * (size_t) T);
    ggml_backend_tensor_get(out, scores_fused.data(), 0, ggml_nbytes(out));

    // Compare CPU triplet+tile vs fused CUDA
    int mism = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < scores_trip.size(); ++i) {
        float da = fabs(scores_trip[i] - scores_fused[i]);
        if (da > 1e-2f) ++mism;
        if (da > max_abs) max_abs = da;
    }


    // Debug: print a small grid of cpu vs fused scores for inspection
    {
        int max_kv_print = (int) (N_kv < 8 ? N_kv : 8);
        int max_t_print  = (int) (T    < 8 ? T    : 8);
        printf("[IDX_TRIPLET_VS_FUSED] sample grid (kv x T):\n");
        for (int kv_i = 0; kv_i < max_kv_print; ++kv_i) {
            printf("kv=%d:", kv_i);
            for (int t = 0; t < max_t_print; ++t) {
                size_t idx = (size_t) kv_i + (size_t) N_kv * (size_t) t;
                float cpu_v   = scores_trip[idx];
                float fused_v = scores_fused[idx];
                float diff    = fused_v - cpu_v;
                printf(" [%d,%d]=(%8.3f,%8.3f, diff=%8.3f)\n", kv_i, t, cpu_v, fused_v, diff);
            }
            printf("\n");
        }
    }

    // Direct CPU Eq. (1) from q_indexer / k_indexer / idx_weights for small kv,t
    {
        auto load3 = [](ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
            char * base = (char *) t->data;
            size_t off = (size_t) i0 * t->nb[0]
                       + (size_t) i1 * t->nb[1]
                       + (size_t) i2 * t->nb[2];
            return *(float *) (base + off);
        };
        auto load2 = [](ggml_tensor * t, int64_t i0, int64_t i1) {
            char * base = (char *) t->data;
            size_t off = (size_t) i0 * t->nb[0]
                       + (size_t) i1 * t->nb[1];
            return *(float *) (base + off);
        };
        int max_kv_print = (int) (N_kv < 4 ? N_kv : 4);
        int max_t_print  = (int) (T    < 4 ? T    : 4);
        printf("[IDX_TRIPLET_EQ1] sample grid (kv x T) from direct formula:\n");
        for (int kv_i = 0; kv_i < max_kv_print; ++kv_i) {
            printf("kv=%d:", kv_i);
            for (int t = 0; t < max_t_print; ++t) {
                float acc_direct = 0.0f;
                for (int h = 0; h < H_index; ++h) {
                    // dot(q_{t,h}, k_s)
                    float dot = 0.0f;
                    for (int d = 0; d < D_index; ++d) {
                        float qv = load3(q_indexer, d, h, t);
                        float kv = load2(k_indexer, d, kv_i);
                        dot += qv * kv;
                    }
                    if (dot < 0.0f) dot = 0.0f;
                    float w = load2(idx_weights, h, t);
                    acc_direct += dot * w;
                }
                printf(" [%d,%d]=%8.3f", kv_i, t, acc_direct);
            }
            printf("\n");
        }
    }
    printf("[IDX_TRIPLET_VS_FUSED] mism=%d max_abs=%.6f\n", mism, max_abs);
    printf("TEST %s\n", mism == 0 ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx_fused);
    cleanup_test_model(model);

    return mism == 0 ? 0 : 1;
#endif
}
