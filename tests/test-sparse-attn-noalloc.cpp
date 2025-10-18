#include "../src/llama-sparse-indexer.h"
#include "../src/llama-kv-cache.h"

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

    printf("About to call compute_token_importance...\n");
    ggml_tensor * token_importance = sparse_attn_indexer::compute_token_importance(
        ctx, *model, 0, cur, /*is_lite=*/false, cb);

    if (token_importance) {
        printf("OK: token_importance shape = [%" PRId64 ", %" PRId64 "]\n",
               token_importance->ne[0], token_importance->ne[1]);
    } else {
        printf("token_importance null\n");
    }

    // Build top-k indices from token_importance using the same path as runtime
    printf("Building top-k indices from token_importance...\n");
    ggml_tensor * topk_indices = llama::sparse_attn_topk::select_topk_tokens(ctx, token_importance, n_tokens, cb);
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

    printf("About to call apply_sparse_attention (expected to reproduce runtime assertion)...\n");
    fflush(stdout);
    ggml_tensor * sparse_out = llama::sparse_mla_fwd::apply_sparse_attention(
        ctx, q_cur, k_cur, v_cur, topk_indices, n_tokens, /*top_k=*/64, cb);
    (void)sparse_out;


    // Now simulate the layer output projection WO as in runtime and reproduce the ggml_can_mul_mat assert
    // In the real model, WO expects input feature size equal to n_embd (e.g., 7168), but our sparse output
    // currently has ne0 = v_embd_head (=512) without flattening across heads. This mismatch should trigger
    // GGML_ASSERT(ggml_can_mul_mat(a, b)).
    // Skip projecting with WO here to allow reaching the KV-cache assertion reproduction below.

    // Now reproduce the runtime assertion in llama_kv_cache::set_input_k_idxs by
    // intentionally calling it with an unallocated input tensor. This mirrors the
    // backtrace seen during runtime where ggml_backend_buffer_is_host(dst->buffer)
    // asserts because dst->buffer is null at set_inputs time.
    printf("Attempting to reproduce GGML_ASSERT(buffer) in set_input_k_idxs...\n");
    fflush(stdout);

    // Minimal hparams to construct a KV cache without offload
    model->hparams.n_layer = 1;
    model->hparams.n_head_arr[0] = 1;
    model->hparams.n_head_kv_arr[0] = 1;
    model->hparams.n_embd_head_k = 64;
    model->hparams.n_embd_head_v = 64;
    model->hparams.n_swa = 0;
    model->hparams.swa_type = LLAMA_SWA_TYPE_NONE;

    // Construct a small KV cache (CPU only, unified, no SWA), with kv_size=16
    llama_kv_cache kv(*model,
                      GGML_TYPE_F32,
                      GGML_TYPE_F32,
                      /*v_trans*/ true,
                      /*offload*/ false,
                      /*unified*/ true,
                      /*kv_size*/ 16,
                      /*n_seq_max*/ 1,
                      /*n_pad*/ 1,
                      /*n_swa*/ 0,
                      /*swa_type*/ LLAMA_SWA_TYPE_NONE,
                      /*filter*/ nullptr,
                      /*reuse*/ nullptr);

    // Wrap in a context that creates a dummy slot_info so we can call set_input_* APIs
    llama_kv_cache_context kv_ctx(&kv);

    // Build an input index tensor in the same ggml context as the rest of this test.
    // Since we never allocate the graph through the scheduler, this tensor will have
    // dst->buffer == nullptr, which will trigger the same assertion as in runtime.
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1; // must match kv_ctx's dummy slot size (1) to pass size assertions

    ggml_tensor * k_idxs = kv_ctx.build_input_k_idxs(ctx, ubatch);
    // Note: we purposefully do NOT allocate this tensor via backend scheduling.

    // This call is expected to hit: GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))
    // because k_idxs->buffer is null in this no_alloc-style unit test setup.
    kv_ctx.set_input_k_idxs(k_idxs, &ubatch);

    delete model;
    ggml_free(ctx);

    printf("=== Test finished ===\n");
    return 0;
}
