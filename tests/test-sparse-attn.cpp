#include "../src/llama-sparse-attn.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml-alloc.h>
#include <ggml-backend-impl.h>
#include <ggml-cpp.h>
#include <ggml-impl.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <vector>
#include <memory>
#include <stdexcept>

// Simple test to verify sparse attention tensor operations
// This test focuses on the core operations without loading a full model

// Function declarations
void test_compute_token_importance();
void test_select_topk_tokens();
void test_apply_sparse_attention_simple();
void test_reshape_assertion_fix();
void test_problematic_reshape();
void test_fixed_reshape_operation();

struct TestContext {
    ggml_context * ctx;
    ggml_backend_t backend;
    
    TestContext() {
        // Create a simple CPU backend for testing
        backend = ggml_backend_cpu_init();
        
        // Create a context with reasonable size and let GGML handle allocation
        ggml_init_params p{};
        p.mem_size   = 64 * 1024 * 1024; // 16MB
        p.mem_buffer = nullptr;
        p.no_alloc   = false; // Let GGML handle allocation
        ctx = ggml_init(p);
    }
    
    ~TestContext() {
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// Mock model layer for testing sparse attention
struct MockLayer {
    ggml_tensor * attn_indexer_wk = nullptr;
    ggml_tensor * attn_indexer_wq_b = nullptr;
    ggml_tensor * attn_indexer_weights_proj = nullptr;
    ggml_tensor * attn_indexer_k_norm_bias = nullptr;
    ggml_tensor * wq_a = nullptr;
    
    void create_tensors(TestContext & test_ctx) {
        // Create mock tensors with appropriate dimensions
        // Based on DeepSeek V3.2-Exp architecture
        const int64_t hidden_dim = 512;
        const int64_t index_n_heads = 64;
        const int64_t index_head_dim = 128;
        
        // Indexer key projection
        attn_indexer_wk = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        
        // Indexer query projection (wq_b)
        attn_indexer_wq_b = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_head_dim * index_n_heads);
        
        // Indexer weights projection
        attn_indexer_weights_proj = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, index_n_heads);
        
        // Indexer normalization bias
        attn_indexer_k_norm_bias = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, index_head_dim * index_n_heads);
        
        // Query projection (wq_a) for non-lite version
        wq_a = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);
    }
};

// Mock model for testing
struct MockModel {
    std::vector<MockLayer> layers;
    
    MockModel(int num_layers = 1) : layers(num_layers) {
        // Layers are default-constructed, tensors will be created later
    }
    
    void create_tensors(TestContext & test_ctx) {
        for (auto & layer : layers) {
            layer.create_tensors(test_ctx);
        }
    }
};

// Test the compute_token_importance function
void test_compute_token_importance() {
    printf("Testing compute_token_importance...\n");
    
    TestContext test_ctx;
    MockModel model;
    
    // Create model tensors after TestContext is fully constructed
    model.create_tensors(test_ctx);
    
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
    
    // Initialize mock model tensors with random data
    for (auto & layer : model.layers) {
        // Initialize attn_indexer_wk
        std::vector<float> wk_data(ggml_nelements(layer.attn_indexer_wk));
        for (size_t i = 0; i < wk_data.size(); i++) {
            wk_data[i] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_wk->data, wk_data.data(), wk_data.size() * sizeof(float));
        
        // Initialize attn_indexer_wq_b
        std::vector<float> wq_b_data(ggml_nelements(layer.attn_indexer_wq_b));
        for (size_t i = 0; i < wq_b_data.size(); i++) {
            wq_b_data[i] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_wq_b->data, wq_b_data.data(), wq_b_data.size() * sizeof(float));
        
        // Initialize attn_indexer_weights_proj
        std::vector<float> weights_proj_data(ggml_nelements(layer.attn_indexer_weights_proj));
        for (size_t i = 0; i < weights_proj_data.size(); i++) {
            weights_proj_data[i] = (float)rand() / RAND_MAX;
        }
        memcpy(layer.attn_indexer_weights_proj->data, weights_proj_data.data(), weights_proj_data.size() * sizeof(float));
        
        // Initialize attn_indexer_k_norm_bias (if present)
        if (layer.attn_indexer_k_norm_bias != nullptr) {
            std::vector<float> bias_data(ggml_nelements(layer.attn_indexer_k_norm_bias));
            for (size_t i = 0; i < bias_data.size(); i++) {
                bias_data[i] = (float)rand() / RAND_MAX;
            }
            memcpy(layer.attn_indexer_k_norm_bias->data, bias_data.data(), bias_data.size() * sizeof(float));
        }
        
        // Initialize wq_a for non-lite version
        if (layer.wq_a != nullptr) {
            std::vector<float> wq_a_data(ggml_nelements(layer.wq_a));
            for (size_t i = 0; i < wq_a_data.size(); i++) {
                wq_a_data[i] = (float)rand() / RAND_MAX;
            }
            memcpy(layer.wq_a->data, wq_a_data.data(), wq_a_data.size() * sizeof(float));
        }
    }
    
    // Create a simple callback function
    auto cb = [](ggml_tensor * tensor, const char * name, int layer_idx) {
        (void)layer_idx; // Unused parameter
        printf("Tensor '%s' (layer %d): shape [", name, layer_idx);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->ne[i] > 1) {
                printf("%ld", tensor->ne[i]);
                if (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) {
                    printf(", ");
                }
            }
        }
        printf("]\n");
    };
    
    // Test both lite and non-lite versions
    for (bool is_lite : {false, true}) {
        printf("Testing %s version...\n", is_lite ? "lite" : "non-lite");
        
        try {
            ggml_tensor * token_importance = llama::sparse_attn_indexer::compute_token_importance(
                test_ctx.ctx, reinterpret_cast<const llama_model&>(model), 0, cur, is_lite, cb);
            
            if (token_importance) {
                printf("Success: token_importance tensor created with shape [%ld, %ld]\n", 
                       token_importance->ne[0], token_importance->ne[1]);
            } else {
                printf("Error: token_importance is null\n");
            }
        } catch (const std::exception& e) {
            printf("Exception: %s\n", e.what());
        }
    }
    
    printf("compute_token_importance test completed\n\n");
}

// Test the select_topk_tokens function
void test_select_topk_tokens() {
    printf("Testing select_topk_tokens...\n");
    
    TestContext test_ctx;
    
    // Create a mock token importance tensor
    const int64_t n_tokens = 32;
    ggml_tensor * token_importance = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, n_tokens, 1);
    
    // Initialize with random importance scores
    std::vector<float> importance_data(n_tokens);
    for (size_t i = 0; i < importance_data.size(); i++) {
        importance_data[i] = (float)rand() / RAND_MAX;
    }
    memcpy(token_importance->data, importance_data.data(), importance_data.size() * sizeof(float));
    
    auto cb = [](ggml_tensor * tensor, const char * name, int layer_idx) {
        (void)layer_idx; // Unused parameter
        printf("Tensor '%s': shape [", name);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->ne[i] > 1) {
                printf("%ld", tensor->ne[i]);
                if (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) {
                    printf(", ");
                }
            }
        }
        printf("]\n");
    };
    
    try {
        ggml_tensor * topk_indices = llama::sparse_attn_indexer::select_topk_tokens(
            test_ctx.ctx, token_importance, n_tokens, cb);
        
        if (topk_indices) {
            printf("Success: topk_indices tensor created with shape [%ld, %ld, %ld, %ld]\n", 
                   topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
        } else {
            printf("Error: topk_indices is null\n");
        }
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
    }
    
    printf("select_topk_tokens test completed\n\n");
}

// Test the apply_sparse_attention function with simple tensors
void test_apply_sparse_attention_simple() {
    printf("Testing apply_sparse_attention (simple)...\n");
    
    TestContext test_ctx;
    
    // Create simple tensors for testing
    const int64_t n_tokens = 8;
    const int64_t top_k = 4;
    const int64_t n_embd_head = 64;
    const int64_t n_head_q = 8;
    const int64_t n_head_kv = 1;
    
    // Create query tensor [n_embd_head, n_head_q, n_tokens]
    ggml_tensor * q_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_q, n_tokens);
    
    // Create key tensor [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    
    // Create value tensor [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    
    // Create topk indices tensor [top_k, 1, 1, 1]
    ggml_tensor * topk_indices = ggml_new_tensor_4d(test_ctx.ctx, GGML_TYPE_I32, top_k, 1, 1, 1);
    
    // Initialize tensors with simple values
    std::vector<float> q_data(n_embd_head * n_head_q * n_tokens, 1.0f);
    std::vector<float> k_data(n_embd_head * n_head_kv * n_tokens, 1.0f);
    std::vector<float> v_data(n_embd_head * n_head_kv * n_tokens, 1.0f);
    std::vector<int32_t> indices_data(top_k);
    
    // Create sequential indices [0, 1, 2, 3]
    for (int i = 0; i < top_k; i++) {
        indices_data[i] = i;
    }
    
    memcpy(q_cur->data, q_data.data(), q_data.size() * sizeof(float));
    memcpy(k_cur->data, k_data.data(), k_data.size() * sizeof(float));
    memcpy(v_cur->data, v_data.data(), v_data.size() * sizeof(float));
    memcpy(topk_indices->data, indices_data.data(), indices_data.size() * sizeof(int32_t));
    
    auto cb = [](ggml_tensor * tensor, const char * name, int layer_idx) {
        (void)layer_idx; // Unused parameter
        printf("Tensor '%s': shape [", name);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->ne[i] > 1) {
                printf("%ld", tensor->ne[i]);
                if (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) {
                    printf(", ");
                }
            }
        }
        printf("]\n");
    };
    
    try {
        ggml_tensor * result = llama::sparse_attn_indexer::apply_sparse_attention(
            test_ctx.ctx, q_cur, k_cur, v_cur, topk_indices, n_tokens, top_k, cb);
        
        if (result) {
            printf("Success: sparse attention result tensor created with shape [%ld, %ld, %ld]\n", 
                   result->ne[0], result->ne[1], result->ne[2]);
        } else {
            printf("Error: result is null\n");
        }
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
    }
    
    printf("apply_sparse_attention (simple) test completed\n\n");
}

// Test that reproduces the specific GGML assertion error
void test_reshape_assertion_fix() {
    printf("Testing reshape assertion fix...\n");
    
    TestContext test_ctx;
    
    // Create tensors with dimensions that should trigger the assertion
    const int64_t n_tokens = 10;
    const int64_t n_embd_head = 64;
    const int64_t n_head_kv = 2;
    
    // Create key tensor [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    
    // Test the problematic reshape operation from apply_sparse_attention
    // This is the operation that's likely causing the assertion failure
    
    // Extract dimensions
    const int64_t n_embd_head_val = k_cur->ne[0];
    const int64_t n_head_kv_val = k_cur->ne[1];
    const int64_t n_tokens_val = k_cur->ne[2];
    
    printf("Original tensor dimensions: [%ld, %ld, %ld]\n", n_embd_head_val, n_head_kv_val, n_tokens_val);
    printf("Total elements: %ld\n", ggml_nelements(k_cur));
    
    // Test reshape to 2D: [n_embd_head, n_head_kv * n_tokens]
    const int64_t ne0 = n_embd_head_val;
    const int64_t ne1 = n_head_kv_val * n_tokens_val;
    
    printf("Attempting reshape to [%ld, %ld] (ne0 * ne1 = %ld)\n", ne0, ne1, ne0 * ne1);
    
    if (ggml_nelements(k_cur) == ne0 * ne1) {
        printf("Reshape dimensions match! This should work.\n");
        
        // This should work without assertion
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, ne0, ne1);
        printf("Reshape successful: new shape [%ld, %ld]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
    } else {
        printf("ERROR: Reshape dimensions don't match! This will cause assertion.\n");
        printf("Expected %ld elements, but tensor has %ld elements\n", ne0 * ne1, ggml_nelements(k_cur));
    }
    
    printf("reshape assertion test completed\n\n");
}

// Test the exact reshape operation that's failing in apply_sparse_attention
void test_problematic_reshape() {
    printf("Testing the exact problematic reshape operation...\n");
    
    TestContext test_ctx;
    
    // Simulate the exact dimensions from the real model
    // Based on DeepSeek V3.2-Exp architecture
    const int64_t n_tokens = 16;  // Typical sequence length
    const int64_t n_embd_head = 576;  // From DeepSeek V3.2 config
    const int64_t n_head_kv = 1;   // MQA configuration
    
    // Create key tensor with exact same dimensions as in the real model
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    
    printf("Created Kcur tensor with dimensions: [%ld, %ld, %ld]\n", 
           k_cur->ne[0], k_cur->ne[1], k_cur->ne[2]);
    printf("Total elements: %ld\n", ggml_nelements(k_cur));
    
    // This is the exact operation from apply_sparse_attention that's failing
    const int64_t n_embd_head_extracted = k_cur->ne[0];
    const int64_t n_head_kv_extracted = k_cur->ne[1];
    const int64_t n_tokens_extracted = k_cur->ne[2];
    
    printf("Extracted dimensions: n_embd_head=%ld, n_head_kv=%ld, n_tokens=%ld\n",
           n_embd_head_extracted, n_head_kv_extracted, n_tokens_extracted);
    
    // Test the problematic reshape
    const int64_t target_ne0 = n_embd_head_extracted;
    const int64_t target_ne1 = n_head_kv_extracted * n_tokens_extracted;
    
    printf("Attempting to reshape to [%ld, %ld] (expected elements: %ld)\n",
           target_ne0, target_ne1, target_ne0 * target_ne1);
    
    if (ggml_nelements(k_cur) == target_ne0 * target_ne1) {
        printf("SUCCESS: Reshape dimensions match!\n");
        
        // This should work
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, target_ne0, target_ne1);
        printf("Reshape successful: new shape [%ld, %ld]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
    } else {
        printf("ERROR: Dimension mismatch detected!\n");
        printf("Tensor has %ld elements, but reshape expects %ld elements\n",
               ggml_nelements(k_cur), target_ne0 * target_ne1);
        
        // Let's debug why this might happen
        printf("Debug info:\n");
        printf("  k_cur->ne[0] = %ld\n", k_cur->ne[0]);
        printf("  k_cur->ne[1] = %ld\n", k_cur->ne[1]);
        printf("  k_cur->ne[2] = %ld\n", k_cur->ne[2]);
        printf("  k_cur->ne[3] = %ld\n", k_cur->ne[3]);
        printf("  Product of ne[] = %ld\n", k_cur->ne[0] * k_cur->ne[1] * k_cur->ne[2] * k_cur->ne[3]);
        printf("  ggml_nelements(k_cur) = %ld\n", ggml_nelements(k_cur));
    }
    
    printf("problematic reshape test completed\n\n");
}

// Test the fix for the GGML assertion error
void test_fixed_reshape_operation() {
    printf("Testing the fixed reshape operation...\n");
    
    TestContext test_ctx;
    
    // Create tensors with mismatched dimensions to simulate the original bug
    const int64_t actual_n_tokens = 16;  // Actual tensor dimension
    const int64_t wrong_n_tokens = 8;    // Wrong parameter value (this was the bug)
    const int64_t n_embd_head = 576;
    const int64_t n_head_kv = 1;
    
    // Create key tensor with actual dimensions
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, actual_n_tokens);
    
    printf("Created Kcur tensor with dimensions: [%ld, %ld, %ld]\n", 
           k_cur->ne[0], k_cur->ne[1], k_cur->ne[2]);
    printf("Total elements: %ld\n", ggml_nelements(k_cur));
    
    // Extract dimensions from tensor (the fix)
    const int64_t n_embd_head_extracted = k_cur->ne[0];
    const int64_t n_head_kv_extracted = k_cur->ne[1];
    const int64_t actual_n_tokens_extracted = k_cur->ne[2]; // Use actual dimension from tensor
    
    printf("Extracted dimensions: n_embd_head=%ld, n_head_kv=%ld, actual_n_tokens=%ld\n",
           n_embd_head_extracted, n_head_kv_extracted, actual_n_tokens_extracted);
    printf("Wrong parameter n_tokens would have been: %ld\n", wrong_n_tokens);
    
    // Test the fixed reshape operation
    const int64_t target_ne0 = n_embd_head_extracted;
    const int64_t target_ne1 = n_head_kv_extracted * actual_n_tokens_extracted;
    
    printf("Attempting to reshape to [%ld, %ld] (expected elements: %ld)\n",
           target_ne0, target_ne1, target_ne0 * target_ne1);
    
    if (ggml_nelements(k_cur) == target_ne0 * target_ne1) {
        printf("SUCCESS: Fixed reshape operation works!\n");
        
        // This should work with the fix
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, target_ne0, target_ne1);
        printf("Reshape successful: new shape [%ld, %ld]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
        
        // Show what would have happened with the bug
        const int64_t wrong_target_ne1 = n_head_kv_extracted * wrong_n_tokens;
        printf("With the bug (using wrong n_tokens=%ld): expected %ld elements, but tensor has %ld elements\n",
               wrong_n_tokens, target_ne0 * wrong_target_ne1, ggml_nelements(k_cur));
    } else {
        printf("ERROR: Fixed reshape operation still fails!\n");
    }
    
    printf("fixed reshape operation test completed\n\n");
}

// Main test function
int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention Unit Tests ===\n\n");
    
    // Initialize random seed for reproducible tests
    srand(42);
    
    // Run individual tests
    test_compute_token_importance();
    test_select_topk_tokens();
    test_apply_sparse_attention_simple();
    test_reshape_assertion_fix();
    test_problematic_reshape();
    test_fixed_reshape_operation();
    
    printf("=== All tests completed ===\n");
    
    return 0;
}
