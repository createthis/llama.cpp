#include "../src/llama-sparse-indexer.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml.h>
#include <cstdio>
#include <cinttypes>
#include <vector>

using namespace llama;

static llama_model* create_test_model_noalloc(ggml_context * ctx, int num_layers,
                                              int64_t n_embd,
                                              int64_t index_head_dim,
                                              int64_t index_n_heads) {
    llama_model_params params = llama_model_default_params();
    llama_model* model = new llama_model(params);
    model->arch = LLM_ARCH_DEEPSEEK3_2;
    model->layers.resize(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        llama_layer & layer = model->layers[i];
        // Shapes chosen to satisfy ggml_mul_mat invariants used in compute_token_importance
        // cur: [n_embd, n_tokens]
        // wk:  [n_embd, index_head_dim]  -> wk * cur => [index_head_dim, n_tokens]
        layer.attn_indexer_wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, index_head_dim);
        // wq_a: [n_embd, n_embd] -> wq_a * x_t => [n_embd, 1]
        layer.wq_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        // wq_b: [n_embd, index_head_dim * index_n_heads] -> wq_b * qr => [index_head_dim*index_n_heads, 1]
        layer.attn_indexer_wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, index_head_dim * index_n_heads);
        // weights proj: [n_embd, index_n_heads] -> weights * x_t => [index_n_heads, 1]
        layer.attn_indexer_weights_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, index_n_heads);
        // k_norm bias
        layer.attn_indexer_k_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, index_head_dim);
    }

    return model;
}

int main() {
    printf("=== DeepSeek V3.2-Exp Sparse Attention no_alloc Unit Test ===\n");
    // Create ggml context with no_alloc=true to simulate graph build context
    ggml_init_params p{};
    p.mem_size   = 64ull * 1024 * 1024; // 64 MB for meta objects
    p.mem_buffer = nullptr;             // let ggml allocate
    p.no_alloc   = true;                // IMPORTANT: no_alloc

    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return 1;
    }

    const int64_t n_embd = 7168;
    const int64_t n_tokens = 4096;
    const int64_t index_head_dim = 128;
    const int64_t index_n_heads = 64;

    llama_model * model = create_test_model_noalloc(ctx, 1, n_embd, index_head_dim, index_n_heads);

    // Create a placeholder current hidden state tensor: [n_embd, n_tokens]
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    auto cb = [](ggml_tensor * t, const char * name, int il) {
        (void)il;
        if (!t) {
            printf("CB: %s is null\n", name);
            return;
        }
        printf("CB: %s: shape=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
               name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    };

    printf("About to call compute_token_importance...\n");
    ggml_tensor * token_importance = sparse_attn_indexer::compute_token_importance(
        ctx, *model, 0, cur, /*is_lite=*/false, cb);

    if (token_importance) {
        printf("OK: token_importance shape = [%" PRId64 ", %" PRId64 "]\n",
               token_importance->ne[0], token_importance->ne[1]);
    } else {
        printf("token_importance null\n");
    }

    delete model;
    ggml_free(ctx);

    printf("=== Test finished ===\n");
    return 0;
}
