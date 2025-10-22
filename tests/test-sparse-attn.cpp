#include "../src/llama-sparse-indexer.h"
#include "../src/llama-sparse-mla-fwd.h"
#include "../src/llama-sparse-topk.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml-alloc.h>
#include <ggml-backend-impl.h>
#include <ggml-cpp.h>
#include <ggml-impl.h>
#include <ggml.h>

#include <cassert>

// Include llama.h for model parameter functions
#include "../include/llama.h"

#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <memory>
#include <stdexcept>

// Simple test to verify sparse attention tensor operations
// This test focuses on the core operations without loading a full model

// Function declarations
void test_compute_indexer_triplet();
void test_select_topk_tokens_indexer_kvaware();

struct TestContext {
    ggml_context * ctx;
    ggml_backend_t backend;

    TestContext() {
        // Create a simple CPU backend for testing
        backend = ggml_backend_cpu_init();

        // Create a context with reasonable size and let GGML handle allocation
        ggml_init_params p{};
        p.mem_size   = 100 * 1024 * 1024; // 100MB
        p.mem_buffer = nullptr;
        p.no_alloc   = false; // Let GGML handle allocation
        ctx = ggml_init(p);
    }

    ~TestContext() {
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// Helper function to create a minimal llama_model instance for testing
// This creates a real llama_model instance and populates the sparse attention tensors
static llama_model* create_test_model(TestContext & test_ctx, int num_layers = 1) {
    // Create model parameters with default values
    llama_model_params params = llama_model_default_params();

    // Create a real llama_model instance
    llama_model* model = new llama_model(params);

    // Set the architecture to DeepSeek3_2
    model->arch = LLM_ARCH_DEEPSEEK3_2;

    // Initialize the layers vector
    model->layers.resize(num_layers);

    // Create and populate sparse attention tensors for each layer
    for (int i = 0; i < num_layers; i++) {
        llama_layer& layer = model->layers[i];

        // Based on DeepSeek V3.2-Exp architecture
        const int64_t hidden_dim = 512;
        const int64_t index_n_heads = 64;
        const int64_t index_head_dim = 128;

        // Indexer key projection
        layer.attn_indexer_wk = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim);

        // Indexer query projection (wq_b)
        layer.attn_indexer_wq_b = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);

        // Indexer weights projection
        layer.attn_indexer_weights_proj = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_n_heads);

        // Indexer normalization bias
        layer.attn_indexer_k_norm_bias = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, index_head_dim);

        // Query projection (wq_a) for non-lite version
        layer.wq_a = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);

        // Initialize tensors with random data
        std::vector<float> wk_data(ggml_nelements(layer.attn_indexer_wk));
        for (size_t j = 0; j < wk_data.size(); j++) {
            wk_data[j] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_wk->data, wk_data.data(), wk_data.size() * sizeof(float));

        std::vector<float> wq_b_data(ggml_nelements(layer.attn_indexer_wq_b));
        for (size_t j = 0; j < wq_b_data.size(); j++) {
            wq_b_data[j] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_wq_b->data, wq_b_data.data(), wq_b_data.size() * sizeof(float));

        std::vector<float> weights_proj_data(ggml_nelements(layer.attn_indexer_weights_proj));
        for (size_t j = 0; j < weights_proj_data.size(); j++) {
            weights_proj_data[j] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_weights_proj->data, weights_proj_data.data(), weights_proj_data.size() * sizeof(float));

        if (layer.attn_indexer_k_norm_bias != nullptr) {
            std::vector<float> bias_data(ggml_nelements(layer.attn_indexer_k_norm_bias));
            for (size_t j = 0; j < bias_data.size(); j++) {
                bias_data[j] = (float)rand() / RAND_MAX;
            }
            memcpy(layer.attn_indexer_k_norm_bias->data, bias_data.data(), bias_data.size() * sizeof(float));
        }

        if (layer.wq_a != nullptr) {
            std::vector<float> wq_a_data(ggml_nelements(layer.wq_a));
            for (size_t j = 0; j < wq_a_data.size(); j++) {
                wq_a_data[j] = (float)rand() / RAND_MAX;
            }
            memcpy(layer.wq_a->data, wq_a_data.data(), wq_a_data.size() * sizeof(float));
        }
    }

    return model;
}

// Helper function to cleanup the test model
static void cleanup_test_model(llama_model* model) {
    if (model) {
        delete model;
    }
}

// Test the Lightning Indexer kv-aware topk builder
void test_compute_indexer_triplet() {
    printf("Testing compute_indexer_triplet...\n");
    fflush(stdout);


    TestContext test_ctx;

    // Create a real llama_model instance instead of mocking
    llama_model* model = create_test_model(test_ctx, 1);

    // Create a mock current hidden state tensor
    const int64_t n_tokens = 16;
    const int64_t hidden_dim = 512;

    ggml_tensor * cur = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, n_tokens);

    // Initialize with random values
    std::vector<float> cur_data(hidden_dim * n_tokens);
    for (size_t i = 0; i < cur_data.size(); i++) {
        cur_data[i] = (float)rand() / RAND_MAX;
    }
    // Copy data directly to tensor memory
    memcpy(cur->data, cur_data.data(), cur_data.size() * sizeof(float));

    // Create a simple callback function
    auto cb = [](ggml_tensor * tensor, const char * name, int layer_idx) {
        (void)layer_idx; // Unused parameter
        printf("Tensor '%s' (layer %d): shape [", name, layer_idx);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->ne[i] > 1) {
                printf("%" PRId64, tensor->ne[i]);
                if (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) {
                    printf(", ");
                }
            }
        }
        printf("]\n");
        fflush(stdout);
    };

    // Test both lite and non-lite versions
    for (bool is_lite : {false, true}) {
        printf("Testing %s version...\n", is_lite ? "lite" : "non-lite");
        fflush(stdout);

        try {
            auto trip = llama::sparse_attn_indexer::compute_indexer_triplet(
                test_ctx.ctx, *model, 0, cur, n_tokens, /*mctx*/ nullptr, /*k_idxs*/ nullptr,
                /*inp_pos*/ nullptr, /*n_rot*/ 0, /*rope_type*/ 0, /*n_ctx_orig*/ 0,
                /*freq_base*/ 0.0f, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f,
                /*beta_fast*/ 1.0f, /*beta_slow*/ 1.0f,
                cb, /*gf*/ nullptr);

            if (trip.q_indexer && trip.k_indexer_cache && trip.idx_weights) {
                printf("Success: triplet shapes q_indexer=[%" PRId64 ", %" PRId64 ", %" PRId64 "] k_indexer_cache=[%" PRId64 ", %" PRId64 "] idx_weights=[%" PRId64 ", %" PRId64 "]\n",
                       trip.q_indexer->ne[0], trip.q_indexer->ne[1], trip.q_indexer->ne[2],
                       trip.k_indexer_cache->ne[0], trip.k_indexer_cache->ne[1],
                       trip.idx_weights->ne[0], trip.idx_weights->ne[1]);
                fflush(stdout);
            } else {
                printf("Error: triplet contains null tensor(s)\n");
                fflush(stdout);
            }
        } catch (const std::exception& e) {
            printf("Exception: %s\n", e.what());
            fflush(stdout);
        }
    }

    // Cleanup the model
    cleanup_test_model(model);

    printf("compute_indexer_triplet test completed\n\n");
    fflush(stdout);
}


// Test the select_topk_tokens_indexer_kvaware function
void test_select_topk_tokens_indexer_kvaware() {
    printf("Testing select_topk_tokens_indexer_kvaware...\n");
    fflush(stdout);

    TestContext test_ctx;

    // Simple synthetic shapes
    const int64_t D_index = 128;
    const int64_t H_index = 8;
    const int64_t T       = 32;
    const int64_t N_kv    = 64;
    const int64_t top_k   = 8;

    // Allocate tensors
    ggml_tensor * q_indexer = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, D_index, H_index, T);
    ggml_tensor * k_indexer = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, D_index, N_kv);
    ggml_tensor * weights   = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, H_index, T);
    ggml_tensor * kq_mask   = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, N_kv, T);

    // Initialize data
    std::vector<float> qidx_data(ggml_nelements(q_indexer));
    std::vector<float> kidx_data(ggml_nelements(k_indexer));
    std::vector<float> w_data(ggml_nelements(weights));
    std::vector<float> mask_data(ggml_nelements(kq_mask), 0.0f);

    for (size_t i = 0; i < qidx_data.size(); ++i) qidx_data[i] = (float) rand() / RAND_MAX;
    for (size_t i = 0; i < kidx_data.size(); ++i) kidx_data[i] = (float) rand() / RAND_MAX;
    for (size_t i = 0; i < w_data.size();   ++i) w_data[i]    = (float) rand() / RAND_MAX;

    // Mask out some KV rows for the first few tokens
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t j = 0; j < N_kv/16; ++j) {
            mask_data[t * N_kv + j] = -INFINITY;
        }
    }

    memcpy(q_indexer->data, qidx_data.data(), qidx_data.size() * sizeof(float));
    memcpy(k_indexer->data, kidx_data.data(), kidx_data.size() * sizeof(float));
    memcpy(weights->data,   w_data.data(),   w_data.size()   * sizeof(float));
    memcpy(kq_mask->data,   mask_data.data(),mask_data.size()* sizeof(float));

    auto cb = [](ggml_tensor * tensor, const char * name, int layer_idx) {
        (void)layer_idx;
        printf("Tensor '%s': shape [", name);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->ne[i] > 1) {
                printf("%" PRId64, tensor->ne[i]);
                if (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) {
                    printf(", ");
                }
            }
        }
        printf("]\n");
        fflush(stdout);
    };

    try {
        ggml_tensor * topk_indices = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
            test_ctx.ctx, q_indexer, k_indexer, weights, kq_mask, top_k, cb);

        if (topk_indices) {
            assert(topk_indices->ne[0] == top_k);
            assert(topk_indices->ne[1] == T);
            printf("Success: idxkv topk_indices tensor created with shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                   topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
            fflush(stdout);
        } else {
            printf("Error: idxkv topk_indices is null\n");
            fflush(stdout);
        }
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
        fflush(stdout);
    }

    printf("select_topk_tokens_indexer_kvaware test completed\n\n");
    fflush(stdout);
}


// Main test function
int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention Unit Tests ===\n\n");
    fflush(stdout);

    // Initialize random seed for reproducible tests
    srand(42);

    // Run individual tests
    test_compute_indexer_triplet();
    test_select_topk_tokens_indexer_kvaware();

    printf("=== All tests completed ===\n");
    fflush(stdout);

    return 0;
}
