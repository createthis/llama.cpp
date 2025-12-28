#include "../src/llama-sparse-indexer.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"
#include "../src/llama-sparse-topk.h"
#include "../src/llama-sparse-mla-fwd.h"

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

    printf("About to call build_kvaware_topk_indices...\n");
    ggml_tensor * topk_indices = llama::sparse_attn_indexer::build_kvaware_topk_indices(
        ctx, *model, 0, cur, n_tokens, /*mctx*/ nullptr, /*k_idxs*/ nullptr, /*kq_mask*/ nullptr, /*top_k*/ 64,
        /*inp_pos*/ nullptr, /*n_rot*/ 0, /*rope_type*/ 0, /*n_ctx_orig*/ 0,
        /*freq_base*/ 0.0f, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f,
        /*beta_fast*/ 1.0f, /*beta_slow*/ 1.0f,
        cb, /*gf*/ nullptr, /*sched*/ nullptr, /*backend_cpu*/ nullptr,
        /*k_indexer_fp8_sidecar=*/nullptr,
        /*quant_bs=*/0, /*cache_block_size=*/0, /*cache_stride=*/0);

    if (topk_indices) {
        printf("OK: topk_indices shape = [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
               topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    } else {
        printf("topk_indices null\n");
    }

    // Reproduce the exact shape configuration observed during runtime startup
    // q_cur: [576, 128, 4096], k_cur: [576, 1, 4096], v_cur: [512, 1, 4096]
    // This mismatch between K/V head dims triggers the reshape assert in apply_sparse_attention
    const int64_t q_embd_head = 576;
    const int64_t q_n_head    = 128;
    const int64_t kv_n_head   = 1;
    const int64_t k_embd_head = 576;
    const int64_t v_embd_head = 512; // intentionally different than K to reproduce the issue

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_embd_head, q_n_head, n_tokens);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_embd_head, kv_n_head, n_tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, v_embd_head, kv_n_head, n_tokens);

    printf("About to call apply_sparse_attention_kvaware (expected to reproduce runtime assertion)...\n");
    fflush(stdout);
    ggml_tensor * sparse_out = llama::sparse_mla_fwd::apply_sparse_attention_kvaware(
        ctx, q_cur, k_cur, v_cur, topk_indices, n_tokens, /*top_k=*/64, /*kq_scale=*/1.0f, /*kq_mask=*/nullptr, /*attn_softcap=*/0.0f, /*kv_dsmla_blob=*/nullptr, cb);
    (void)sparse_out;

    delete model;
    ggml_free(ctx);

    printf("=== Test finished ===\n");
    return 0;
}
