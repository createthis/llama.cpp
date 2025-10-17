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

// Test to reproduce the specific issue with 4096 tokens

struct TestContext {
    ggml_context * ctx;
    ggml_backend_t backend;
    
    TestContext() {
        // Create a simple CPU backend for testing
        backend = ggml_backend_cpu_init();
        
        // Create a context with reasonable size and let GGML handle allocation
        ggml_init_params p{};
        //p.mem_size   = 32212254720; // 30GB
        p.mem_size   = 21474836480; // 20GB
        //p.mem_size   = 16106127360; // 16GB
        //p.mem_size   = (20 * 1024 * 1024 * 1024); // 20GB
        p.mem_buffer = nullptr;
        p.no_alloc   = false; // Let GGML handle allocation
        ctx = ggml_init(p);
    }
    
    ~TestContext() {
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// Function declarations
static llama_model* create_test_model(TestContext & test_ctx, int num_layers);
static void cleanup_test_model(llama_model* model);
void test_4096_tokens();

// Helper function to create a minimal llama_model instance for testing
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
        const int64_t hidden_dim = 7168;  // From the error message
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

// Test that reproduces the exact issue with 4096 tokens
void test_4096_tokens() {
    printf("Testing with 4096 tokens (the problematic case from the error)...\n");
    fflush(stdout);
    
    // Create a mock current hidden state tensor with 4096 tokens
    const int64_t n_tokens = 4096; // The problematic case
    const int64_t hidden_dim = 7168; // From the error message
    
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
    
    printf("Testing with n_tokens = %" PRId64 " (the problematic case)\n", n_tokens);
    fflush(stdout);
    
    // Test both lite and non-lite versions
    for (bool is_lite : {false, true}) {
        printf("Testing %s version...\n", is_lite ? "lite" : "non-lite");
        fflush(stdout);

        TestContext test_ctx;
        
        // Create a real llama_model instance instead of mocking
        llama_model* model = create_test_model(test_ctx, 1);
        
        
        ggml_tensor * cur = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
        
        // Initialize with random values
        std::vector<float> cur_data(hidden_dim * n_tokens);
        for (size_t i = 0; i < cur_data.size(); i++) {
            cur_data[i] = (float)rand() / RAND_MAX;
        }

        // Copy data directly to tensor memory
        memcpy(cur->data, cur_data.data(), cur_data.size() * sizeof(float));
        
        try {
            ggml_tensor * token_importance = llama::sparse_attn_indexer::compute_token_importance(
                test_ctx.ctx, *model, 0, cur, is_lite, cb);
            
            if (token_importance) {
                printf("Success: token_importance tensor created with shape [%" PRId64 ", %" PRId64 "]\n", 
                       token_importance->ne[0], token_importance->ne[1]);
                fflush(stdout);
            } else {
                printf("Error: token_importance is null\n");
                fflush(stdout);
            }
        } catch (const std::exception& e) {
            printf("Exception: %s\n", e.what());
            fflush(stdout);
        }
        
        // Cleanup the model
        cleanup_test_model(model);
    }
    
    printf("4096 tokens test completed\n\n");
    fflush(stdout);
}

// Main test function
int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention 4096 Tokens Test ===\n\n");
    fflush(stdout);
    
    // Initialize random seed for reproducible tests
    srand(42);
    
    // Run the test
    test_4096_tokens();
    
    printf("=== 4096 tokens test completed ===\n");
    fflush(stdout);
    
    return 0;
}
