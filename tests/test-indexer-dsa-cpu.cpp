#include <cstring>
#include "../src/llama-sparse-indexer.h"
#include "../src/llama-sparse-topk.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <random>
#include <algorithm>

using namespace llama;

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

        layer.attn_indexer_wk           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_head_dim);
        layer.attn_indexer_wq_b         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        layer.attn_indexer_weights_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, index_n_heads);
        layer.attn_indexer_k_norm_bias  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, index_head_dim);
        layer.wq_a                      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);

        std::mt19937 rng(42 + i);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto fill = [&](ggml_tensor * t) {
            size_t n = ggml_nelements(t);
            std::vector<float> tmp(n);
            for (size_t k = 0; k < n; ++k) tmp[k] = dist(rng);
            std::memcpy(t->data, tmp.data(), n * sizeof(float));
        };

        fill(layer.attn_indexer_wk);
        fill(layer.attn_indexer_wq_b);
        fill(layer.attn_indexer_weights_proj);
        fill(layer.attn_indexer_k_norm_bias);
        fill(layer.wq_a);
    }

    return model;
}

static void cleanup_test_model(llama_model * model) {
    delete model;
}

int main() {
    printf("[indexer-dsa-cpu] starting test...\n");

    ggml_init_params p{};
    p.mem_size   = 256ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;

    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        fprintf(stderr, "ctx init failed\n");
        return 1;
    }

    llama_model * model = create_test_model(ctx, 1);

    const int64_t hidden_dim = 512;
    const int64_t T          = 6;

    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, T);
    {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> tmp((size_t) hidden_dim * (size_t) T);
        for (auto & v : tmp) v = dist(rng);
        std::memcpy(cur->data, tmp.data(), tmp.size() * sizeof(float));
    }

    auto cb_noop = [](ggml_tensor *, const char *, int) {};

    IndexerKVTriplet trip = sparse_attn_indexer::compute_indexer_triplet(
        ctx, *model, 0, cur, T,
        /*mctx*/ nullptr, /*k_idxs*/ nullptr,
        /*inp_pos*/ nullptr, /*n_rot*/ 0, /*rope_type*/ 0, /*n_ctx_orig*/ (int) T,
        /*freq_base*/ 10000.0f, /*freq_scale*/ 1.0f,
        /*ext_factor*/ 1.0f, /*attn_factor*/ 1.0f,
        /*beta_fast*/ 1.0f, /*beta_slow*/ 1.0f,
        cb_noop, /*gf*/ nullptr);

    ggml_tensor * q_indexer = trip.q_indexer;      // [D_index, H_index, T]
    ggml_tensor * k_indexer = trip.k_indexer_cache;// [D_index, N_kv]
    ggml_tensor * w_indexer = trip.idx_weights;    // [H_index, T]

    if (!q_indexer || !k_indexer || !w_indexer) {
        fprintf(stderr, "triplet tensors are null; aborting\n");
        cleanup_test_model(model);
        ggml_free(ctx);
        return 1;
    }

    const int64_t D_index = q_indexer->ne[0];
    const int64_t H_index = q_indexer->ne[1];
    const int64_t T_eff   = q_indexer->ne[2];
    const int64_t N_kv    = k_indexer->ne[1];

    printf("[indexer-dsa-cpu] D_index=%lld H_index=%lld T=%lld N_kv=%lld\n",
           (long long) D_index, (long long) H_index, (long long) T_eff, (long long) N_kv);

    GGML_ASSERT(T_eff == T);

    // Build scores tensor [N_kv, T] according to DSA equation
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_kv, T_eff);
    float * scores_data = (float *) scores->data;

    const size_t q_nb0 = q_indexer->nb[0];
    const size_t q_nb1 = q_indexer->nb[1];
    const size_t q_nb2 = q_indexer->nb[2];
    const size_t k_nb0 = k_indexer->nb[0];
    const size_t k_nb1 = k_indexer->nb[1];
    const size_t w_nb0 = w_indexer->nb[0];
    const size_t w_nb1 = w_indexer->nb[1];

    for (int64_t t = 0; t < T_eff; ++t) {
        for (int64_t s = 0; s < N_kv; ++s) {
            float acc = 0.0f;
            for (int64_t j = 0; j < H_index; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D_index; ++d) {
                    const char * q_base = (const char *) q_indexer->data;
                    const char * k_base = (const char *) k_indexer->data;
                    size_t q_off = (size_t) d * q_nb0 + (size_t) j * q_nb1 + (size_t) t * q_nb2;
                    size_t k_off = (size_t) d * k_nb0 + (size_t) s * k_nb1;
                    float qv = *(const float *)(q_base + q_off);
                    float kv = *(const float *)(k_base + k_off);
                    dot += qv * kv;
                }
                if (dot < 0.0f) dot = 0.0f;

                const char * w_base = (const char *) w_indexer->data;
                size_t w_off = (size_t) j * w_nb0 + (size_t) t * w_nb1;
                float wv = *(const float *)(w_base + w_off);
                acc += wv * dot;
            }
            scores_data[(size_t) s + (size_t) N_kv * (size_t) t] = acc;
        }
    }

    // Build reference top-k (k_ref = 4)
    const int k_ref = 4;
    std::vector<int> ref_idx((size_t) T_eff * (size_t) k_ref);
    for (int64_t t = 0; t < T_eff; ++t) {
        std::vector<int> idx((size_t) N_kv);
        for (int64_t s = 0; s < N_kv; ++s) idx[(size_t) s] = (int) s;
        auto cmp = [&](int a, int b) {
            float va = scores_data[(size_t) a + (size_t) N_kv * (size_t) t];
            float vb = scores_data[(size_t) b + (size_t) N_kv * (size_t) t];
            if (va != vb) return va > vb;
            return a < b;
        };
        std::partial_sort(idx.begin(), idx.begin() + k_ref, idx.end(), cmp);
        for (int i = 0; i < k_ref; ++i) {
            ref_idx[(size_t) t * (size_t) k_ref + (size_t) i] = idx[(size_t) i];
        }
    }

    // Use CPU radix top-k helper on the same scores
    ggml_tensor * topk = llama::sparse_attn_topk::topk_radix_indices(ctx, scores, k_ref);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, topk);
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);

    int32_t * topk_data = (int32_t *) topk->data;
    GGML_ASSERT(topk->ne[0] == k_ref);
    GGML_ASSERT(topk->ne[1] == T_eff);

    int mism = 0;
    for (int64_t t = 0; t < T_eff; ++t) {
        std::vector<int> got(k_ref);
        std::vector<int> exp(k_ref);
        for (int i = 0; i < k_ref; ++i) {
            got[i] = topk_data[(size_t) i + (size_t) k_ref * (size_t) t];
            exp[i] = ref_idx[(size_t) t * (size_t) k_ref + (size_t) i];
        }
        std::sort(got.begin(), got.end());
        std::sort(exp.begin(), exp.end());
        if (got != exp) {
            ++mism;
            printf("[indexer-dsa-cpu] token %lld mismatch: got=", (long long) t);
            for (int v : got) printf(" %d", v);
            printf(" exp=");
            for (int v : exp) printf(" %d", v);
            printf("\n");
        }
    }

    printf("[indexer-dsa-cpu] mismatches=%d\n", mism);
    printf("TEST %s\n", mism == 0 ? "PASS" : "FAIL");

    cleanup_test_model(model);
    ggml_free(ctx);

    return mism == 0 ? 0 : 1;
}
