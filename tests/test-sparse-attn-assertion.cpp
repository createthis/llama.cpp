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

// Test to reproduce the specific GGML assertion error from compute_token_importance

struct TestContext {
    ggml_context * ctx;
    ggml_backend_t backend;
    
    TestContext() {
        // Create a simple CPU backend for testing
        backend = ggml_backend_cpu_init();
        
        // Create a context with reasonable size and let GGML handle allocation
        ggml_init_params p{};
        p.mem_size   = 1024 * 1024 * 1024; // 1GB
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
llama_model* create_test_model(TestContext & test_ctx, int num_layers);
void cleanup_test_model(llama_model* model);
void test_assertion_error_reproduction();
void test_problematic_reshape_operation();
void test_various_token_counts();

// Helper function to create a minimal llama_model instance for testing
llama_model* create_test_model(TestContext & test_ctx, int num_layers = 1) {
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
        layer.attn_indexer_wk = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        
        // Indexer query projection (wq_b)
        layer.attn_indexer_wq_b = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        
        // Indexer weights projection
        layer.attn_indexer_weights_proj = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_n_heads);
        
        // Indexer normalization bias
        layer.attn_indexer_k_norm_bias = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, index_head_dim * index_n_heads);
        
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
void cleanup_test_model(llama_model* model) {
    if (model) {
        delete model;
    }
}

// Test that reproduces the exact assertion error from compute_token_importance
void test_assertion_error_reproduction() {
    printf("Testing assertion error reproduction...\n");
    fflush(stdout);
    
    TestContext test_ctx;
    
    // Create a real llama_model instance instead of mocking
    llama_model* model = create_test_model(test_ctx, 1);
    
    // Create a mock current hidden state tensor with specific dimensions
    // that might trigger the assertion error
    const int64_t n_tokens = 17; // Use a prime number to trigger potential division issues
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
    
    printf("Testing with n_tokens = %" PRId64 " (prime number)\n", n_tokens);
    fflush(stdout);
    
    // Test both lite and non-lite versions
    for (bool is_lite : {false, true}) {
        printf("Testing %s version...\n", is_lite ? "lite" : "non-lite");
        fflush(stdout);
        
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
    }
    
    // Cleanup the model
    cleanup_test_model(model);
    
    printf("assertion error reproduction test completed\n\n");
    fflush(stdout);
}

// Test the specific reshape operation that's causing the assertion
void test_problematic_reshape_operation() {
    printf("Testing the problematic reshape operation...\n");
    fflush(stdout);
    
    TestContext test_ctx;
    
    // Create a tensor with dimensions that might cause the assertion
    const int64_t index_head_dim = 128;
    const int64_t index_n_heads = 64;
    const int64_t n_tokens = 17; // Prime number to trigger potential issues
    
    // Create a tensor that simulates q_indexer after matrix multiplication
    // This should have shape [n_tokens, index_n_heads * index_head_dim]
    const int64_t q_indexer_ne0 = index_n_heads * index_head_dim;
    const int64_t q_indexer_ne1 = n_tokens;
    
    ggml_tensor * q_indexer = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, q_indexer_ne0, q_indexer_ne1);
    
    printf("Created q_indexer tensor with shape [%" PRId64 ", %" PRId64 "]\n", q_indexer->ne[0], q_indexer->ne[1]);
    printf("Total elements: %" PRId64 "\n", ggml_nelements(q_indexer));
    fflush(stdout);
    
    // This is the exact operation from line 68 of llama-sparse-indexer.cpp
    const int64_t target_ne0 = index_head_dim;
    const int64_t target_ne1 = index_n_heads;
    const int64_t target_ne2 = ggml_nelements(q_indexer) / (index_head_dim * index_n_heads);
    
    printf("Attempting to reshape to [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", target_ne0, target_ne1, target_ne2);
    printf("Expected elements: %" PRId64 "\n", target_ne0 * target_ne1 * target_ne2);
    printf("Actual elements: %" PRId64 "\n", ggml_nelements(q_indexer));
    
    // Check if the reshape will work
    if (ggml_nelements(q_indexer) == target_ne0 * target_ne1 * target_ne2) {
        printf("SUCCESS: Reshape dimensions match!\n");
        fflush(stdout);
        
        // This should work
        ggml_tensor * reshaped = ggml_reshape_3d(test_ctx.ctx, q_indexer, target_ne0, target_ne1, target_ne2);
        printf("Reshape successful: new shape [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               reshaped->ne[0], reshaped->ne[1], reshaped->ne[2]);
        fflush(stdout);
    } else {
        printf("ERROR: Dimension mismatch detected! This will cause the assertion.\n");
        fflush(stdout);
        printf("Tensor has %" PRId64 " elements, but reshape expects %" PRId64 " elements\n",
               ggml_nelements(q_indexer), target_ne0 * target_ne1 * target_ne2);
        fflush(stdout);
        
        // Debug the calculation
        printf("Debug calculation:\n");
        printf("  ggml_nelements(q_indexer) = %" PRId64 "\n", ggml_nelements(q_indexer));
        printf("  index_head_dim * index_n_heads = %" PRId64 " * %" PRId64 " = %" PRId64 "\n", 
               index_head_dim, index_n_heads, index_head_dim * index_n_heads);
        printf("  Division result = %" PRId64 " / %" PRId64 " = %" PRId64 "\n", 
               ggml_nelements(q_indexer), index_head_dim * index_n_heads, 
               ggml_nelements(q_indexer) / (index_head_dim * index_n_heads));
        printf("  Remainder = %" PRId64 "\n", 
               ggml_nelements(q_indexer) % (index_head_dim * index_n_heads));
        fflush(stdout);
    }
    
    printf("problematic reshape operation test completed\n\n");
    fflush(stdout);
}

// Test with different token counts to find the problematic case
void test_various_token_counts() {
    printf("Testing various token counts to find the problematic case...\n");
    fflush(stdout);
    
    TestContext test_ctx;
    
    const int64_t index_head_dim = 128;
    const int64_t index_n_heads = 64;
    const int64_t divisor = index_head_dim * index_n_heads; // 8192
    
    printf("Testing with divisor = %" PRId64 " (128 * 64)\n", divisor);
    fflush(stdout);
    
    // Test various token counts
    int64_t token_counts[] = {1, 2, 4, 8, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    
    for (int64_t n_tokens : token_counts) {
        printf("Testing n_tokens = %" PRId64 "...\n", n_tokens);
        fflush(stdout);
        
        // Create a tensor that simulates q_indexer
        const int64_t q_indexer_ne0 = index_n_heads * index_head_dim; // 8192
        const int64_t q_indexer_ne1 = n_tokens;
        
        ggml_tensor * q_indexer = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, q_indexer_ne0, q_indexer_ne1);
        
        const int64_t target_ne0 = index_head_dim;
        const int64_t target_ne1 = index_n_heads;
        const int64_t target_ne2 = ggml_nelements(q_indexer) / divisor;
        
        if (ggml_nelements(q_indexer) == target_ne0 * target_ne1 * target_ne2) {
            printf("  SUCCESS: Reshape will work for n_tokens = %" PRId64 "\n", n_tokens);
        } else {
            printf("  ERROR: Reshape will FAIL for n_tokens = %" PRId64 "\n", n_tokens);
            printf("    Expected %" PRId64 " elements, but tensor has %" PRId64 " elements\n",
                   target_ne0 * target_ne1 * target_ne2, ggml_nelements(q_indexer));
            printf("    Remainder = %" PRId64 "\n", ggml_nelements(q_indexer) % divisor);
        }
        fflush(stdout);
    }
    
    printf("various token counts test completed\n\n");
    fflush(stdout);
}

// Main test function
int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention Assertion Reproduction Tests ===\n\n");
    fflush(stdout);
    
    // Initialize random seed for reproducible tests
    srand(42);
    
    // Run individual tests
    test_assertion_error_reproduction();
    test_problematic_reshape_operation();
    test_various_token_counts();
    
    printf("=== All assertion reproduction tests completed ===\n");
    fflush(stdout);
    
    return 0;
}
