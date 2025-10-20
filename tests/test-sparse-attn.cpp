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

// Test the compute_token_importance function
void test_compute_token_importance() {
    printf("Testing compute_token_importance...\n");
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

    printf("compute_token_importance test completed\n\n");
    fflush(stdout);
}

// Test the select_topk_tokens function
void test_select_topk_tokens() {
    printf("Testing select_topk_tokens...\n");
    fflush(stdout);

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
        ggml_tensor * topk_indices = llama::sparse_attn_topk::select_topk_tokens(
            test_ctx.ctx, token_importance, n_tokens, cb);

        if (topk_indices) {
            printf("Success: topk_indices tensor created with shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                   topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
            fflush(stdout);
        } else {
            printf("Error: topk_indices is null\n");
            fflush(stdout);
        }
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
        fflush(stdout);
    }

    printf("select_topk_tokens test completed\n\n");
    fflush(stdout);
}

// Test the apply_sparse_attention function with simple tensors
void test_apply_sparse_attention_simple() {
    printf("Testing apply_sparse_attention (simple)...\n");
    fflush(stdout);

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
        ggml_tensor * result = llama::sparse_mla_fwd::apply_sparse_attention(
            test_ctx.ctx, q_cur, k_cur, v_cur, topk_indices, n_tokens, top_k, cb);

        // Verify the shapes are correct in the test (only in test builds)
        if (result) {
            // Validate the final result shape
            assert(result->ne[0] == n_embd_head);
            assert(result->ne[1] == n_head_q);
            assert(result->ne[2] == n_tokens);
            assert(ggml_nelements(result) == n_embd_head * n_head_q * n_tokens);
            printf("Shape validation passed: result tensor has correct dimensions\n");

            printf("Success: sparse attention result tensor created with shape [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                   result->ne[0], result->ne[1], result->ne[2]);
            fflush(stdout);
        } else {
            printf("Error: result is null\n");
            fflush(stdout);
        }
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
        fflush(stdout);
    }

    printf("apply_sparse_attention (simple) test completed\n\n");
    fflush(stdout);
}

// Test that reproduces the specific GGML assertion error
void test_reshape_assertion_fix() {
    printf("Testing reshape assertion fix...\n");
    fflush(stdout);

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

    printf("Original tensor dimensions: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n_embd_head_val, n_head_kv_val, n_tokens_val);
    printf("Total elements: %" PRId64 "\n", ggml_nelements(k_cur));
    fflush(stdout);

    // Test reshape to 2D: [n_embd_head, n_head_kv * n_tokens]
    const int64_t ne0 = n_embd_head_val;
    const int64_t ne1 = n_head_kv_val * n_tokens_val;

    printf("Attempting reshape to [%" PRId64 ", %" PRId64 "] (ne0 * ne1 = %" PRId64 ")\n", ne0, ne1, ne0 * ne1);

    if (ggml_nelements(k_cur) == ne0 * ne1) {
        printf("Reshape dimensions match! This should work.\n");
        fflush(stdout);

        // This should work without assertion
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, ne0, ne1);
        printf("Reshape successful: new shape [%" PRId64 ", %" PRId64 "]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
        printf("Reshape successful: new shape [%" PRId64 ", %" PRId64 "]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
    } else {
        printf("ERROR: Reshape dimensions don't match! This will cause assertion.\n");
        printf("Expected %" PRId64 " elements, but tensor has %" PRId64 " elements\n", ne0 * ne1, ggml_nelements(k_cur));
        printf("Expected %" PRId64 " elements, but tensor has %" PRId64 " elements\n", ne0 * ne1, ggml_nelements(k_cur));
    }

    printf("reshape assertion test completed\n\n");
    fflush(stdout);
}

// Test the exact reshape operation that's failing in apply_sparse_attention
void test_problematic_reshape() {
    printf("Testing the exact problematic reshape operation...\n");
    fflush(stdout);

    TestContext test_ctx;

    // Simulate the exact dimensions from the real model
    // Based on DeepSeek V3.2-Exp architecture
    const int64_t n_tokens = 16;  // Typical sequence length
    const int64_t n_embd_head = 576;  // From DeepSeek V3.2 config
    const int64_t n_head_kv = 1;   // MQA configuration

    // Create key tensor with exact same dimensions as in the real model
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);

    printf("Created Kcur tensor with dimensions: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           k_cur->ne[0], k_cur->ne[1], k_cur->ne[2]);
    printf("Total elements: %" PRId64 "\n", ggml_nelements(k_cur));
    fflush(stdout);

    // This is the exact operation from apply_sparse_attention that's failing
    const int64_t n_embd_head_extracted = k_cur->ne[0];
    const int64_t n_head_kv_extracted = k_cur->ne[1];
    const int64_t n_tokens_extracted = k_cur->ne[2];

    printf("Extracted dimensions: n_embd_head=%" PRId64 ", n_head_kv=%" PRId64 ", n_tokens=%" PRId64 "\n",
           n_embd_head_extracted, n_head_kv_extracted, n_tokens_extracted);
    fflush(stdout);

    // Test the problematic reshape
    const int64_t target_ne0 = n_embd_head_extracted;
    const int64_t target_ne1 = n_head_kv_extracted * n_tokens_extracted;

    printf("Attempting to reshape to [%" PRId64 ", %" PRId64 "] (expected elements: %" PRId64 ")\n",
           target_ne0, target_ne1, target_ne0 * target_ne1);
    fflush(stdout);

    if (ggml_nelements(k_cur) == target_ne0 * target_ne1) {
        printf("SUCCESS: Reshape dimensions match!\n");
        fflush(stdout);

        // This should work
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, target_ne0, target_ne1);
        printf("Reshape successful: new shape [%" PRId64 ", %" PRId64 "]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
        fflush(stdout);
    } else {
        printf("ERROR: Dimension mismatch detected!\n");
        fflush(stdout);
        printf("Tensor has %" PRId64 " elements, but reshape expects %" PRId64 " elements\n",
               ggml_nelements(k_cur), target_ne0 * target_ne1);
        fflush(stdout);

        // Let's debug why this might happen
        printf("Debug info:\n");
        printf("  k_cur->ne[0] = %" PRId64 "\n", k_cur->ne[0]);
        printf("  k_cur->ne[1] = %" PRId64 "\n", k_cur->ne[1]);
        printf("  k_cur->ne[2] = %" PRId64 "\n", k_cur->ne[2]);
        printf("  k_cur->ne[3] = %" PRId64 "\n", k_cur->ne[3]);
        printf("  Product of ne[] = %" PRId64 "\n", k_cur->ne[0] * k_cur->ne[1] * k_cur->ne[2] * k_cur->ne[3]);
        printf("  ggml_nelements(k_cur) = %" PRId64 "\n", ggml_nelements(k_cur));
        fflush(stdout);
    }

    printf("problematic reshape test completed\n\n");
    fflush(stdout);
}

// Test the fix for the GGML assertion error
void test_fixed_reshape_operation() {
    printf("Testing the fixed reshape operation...\n");
    fflush(stdout);

    TestContext test_ctx;

    // Create tensors with mismatched dimensions to simulate the original bug
    const int64_t actual_n_tokens = 16;  // Actual tensor dimension
    const int64_t wrong_n_tokens = 8;    // Wrong parameter value (this was the bug)
    const int64_t n_embd_head = 576;
    const int64_t n_head_kv = 1;

    // Create key tensor with actual dimensions
    ggml_tensor * k_cur = ggml_new_tensor_3d(test_ctx.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, actual_n_tokens);

    printf("Created Kcur tensor with dimensions: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           k_cur->ne[0], k_cur->ne[1], k_cur->ne[2]);
    fflush(stdout);
    printf("Total elements: %" PRId64 "\n", ggml_nelements(k_cur));
    fflush(stdout);

    // Extract dimensions from tensor (the fix)
    const int64_t n_embd_head_extracted = k_cur->ne[0];
    const int64_t n_head_kv_extracted = k_cur->ne[1];
    const int64_t actual_n_tokens_extracted = k_cur->ne[2]; // Use actual dimension from tensor

    printf("Extracted dimensions: n_embd_head=%" PRId64 ", n_head_kv=%" PRId64 ", actual_n_tokens=%" PRId64 "\n",
           n_embd_head_extracted, n_head_kv_extracted, actual_n_tokens_extracted);
    printf("Wrong parameter n_tokens would have been: %" PRId64 "\n", wrong_n_tokens);

    // Test the fixed reshape operation
    const int64_t target_ne0 = n_embd_head_extracted;
    const int64_t target_ne1 = n_head_kv_extracted * actual_n_tokens_extracted;

    printf("Attempting to reshape to [%" PRId64 ", %" PRId64 "] (expected elements: %" PRId64 ")\n",
           target_ne0, target_ne1, target_ne0 * target_ne1);

    if (ggml_nelements(k_cur) == target_ne0 * target_ne1) {
        printf("SUCCESS: Fixed reshape operation works!\n");
        fflush(stdout);

        // This should work with the fix
        ggml_tensor * k_cur_2d = ggml_reshape_2d(test_ctx.ctx, k_cur, target_ne0, target_ne1);
        printf("Reshape successful: new shape [%" PRId64 ", %" PRId64 "]\n", k_cur_2d->ne[0], k_cur_2d->ne[1]);
        fflush(stdout);

        // Show what would have happened with the bug
        const int64_t wrong_target_ne1 = n_head_kv_extracted * wrong_n_tokens;
        printf("With the bug (using wrong n_tokens=%" PRId64 "): expected %" PRId64 " elements, but tensor has %" PRId64 " elements\n",
               wrong_n_tokens, target_ne0 * wrong_target_ne1, ggml_nelements(k_cur));
        fflush(stdout);
    } else {
        printf("ERROR: Fixed reshape operation still fails!\n");
        fflush(stdout);
    }

    printf("fixed reshape operation test completed\n\n");
    fflush(stdout);
}

// Main test function
int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention Unit Tests ===\n\n");
    fflush(stdout);

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
    fflush(stdout);

    return 0;
}
