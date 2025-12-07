#include "../src/llama-kv-cache-fp8.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"

#include <ggml-alloc.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Simple unit test that exercises the DeepSeek V3.2 FP8 KV K blob
// layout (fp8_ds_mla-style 656-byte entries) by round-tripping
// synthetic latent + RoPE data through llama_kv_cache_fp8::cpy_k
// and llama_kv_cache_fp8::get_k.

static void test_fp8_kv_dsmla_roundtrip() {
    printf("[fp8-kv-dsmla] starting roundtrip test...\n");
    fflush(stdout);

    // Minimal hparams: 1 layer with KV, DeepSeek3.2 arch, kv_lora_rank=512, rope_dim=64
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = new llama_model(mparams);
    model->arch = LLM_ARCH_DEEPSEEK3_2;

    llama_hparams & hp = model->hparams;
    hp.n_layer      = 1;
    hp.n_layer_kv_from_start = 1; // has_kv(0) == true
    hp.n_lora_kv    = 512;        // kv_lora_rank
    hp.n_rot        = 64;         // rope_dim
    hp.n_embd       = 576;        // not used here directly

    // Ensure layers vector has at least 1 entry
    model->layers.resize(1);

    const uint32_t kv_size   = 4;  // a few KV cells per stream
    const uint32_t n_seq_max = 1;  // single stream
    const uint32_t n_pad     = 1;
    const uint32_t n_swa     = 0;

    // Construct an FP8 KV cache instance
    llama_kv_cache_fp8 kv_fp8(
        *model,
        GGML_TYPE_F16,  // ignored for DeepSeek V3.2 path
        GGML_TYPE_F16,  // ignored for DeepSeek V3.2 path
        /*v_trans*/ true,
        /*offload*/ false,
        /*unified*/ true,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        LLAMA_SWA_TYPE_NONE,
        /*filter*/ nullptr,
        /*reuse*/  nullptr);

    // Synthetic latent+RoPE per token
    const int64_t D_latent = 512;
    const int64_t D_rope   = 64;
    const int64_t D_total  = D_latent + D_rope;

    const int64_t n_tokens = 3; // write 3 tokens into first 3 KV cells

    // Build a ggml context for tensors
    ggml_init_params params = {};
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    // k_cur: [D_total, 1, n_tokens]
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_total, 1, n_tokens);

    // Fill with a simple deterministic pattern
    float * k_data = (float *) k_cur->data;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t d = 0; d < D_total; ++d) {
            float base = 0.01f * float(t + 1);
            // keep magnitudes reasonable for fp8 quantization
            k_data[d + D_total * t] = base * (1.0f + 0.001f * float(d));
        }
    }

    // k_idxs: global KV indices for each token, here 0,1,2
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    int64_t * idx_data = (int64_t *) k_idxs->data;
    for (int64_t t = 0; t < n_tokens; ++t) {
        idx_data[t] = t; // stream=0, cell=t
    }

    // Build a minimal slot_info that maps a single stream 0
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.strm = { 0 };
    sinfo.idxs = { std::vector<uint32_t>(kv_size) };
    for (uint32_t i = 0; i < kv_size; ++i) {
        sinfo.idxs[0][i] = i;
    }

    // Write into the FP8 K blob using the new DS-MLA cpy_k
    kv_fp8.cpy_k(ctx, k_cur, k_idxs, /*il=*/0, sinfo);

    // Read back using get_k: expect [576, 1, kv_size, ns=1]
    ggml_tensor * k_out = kv_fp8.get_k(ctx, /*il=*/0, kv_size, sinfo);
    GGML_ASSERT(k_out != nullptr);
    GGML_ASSERT(k_out->type == GGML_TYPE_F32);
    GGML_ASSERT(k_out->ne[0] == D_total);
    GGML_ASSERT(k_out->ne[1] == 1);
    GGML_ASSERT(k_out->ne[2] == kv_size);
    GGML_ASSERT(k_out->ne[3] == 1);

    const float * out_data = (const float *) k_out->data;

    // Compare only the first n_tokens cells; the rest are unspecified
    float max_abs_err = 0.0f;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t d = 0; d < D_total; ++d) {
            float orig = k_data[d + D_total * t];
            float got  = out_data[d + D_total * (t + kv_size * 0)];
            float err  = fabsf(orig - got);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    printf("[fp8-kv-dsmla] max_abs_err = %g\n", (double) max_abs_err);
    fflush(stdout);

    // FP8 + BF16 round-trip is lossy; allow a modest tolerance
    GGML_ASSERT(max_abs_err < 0.1f);

    ggml_free(ctx);
    delete model;

    printf("[fp8-kv-dsmla] roundtrip test PASSED\n");
    fflush(stdout);
}

int main() {
    test_fp8_kv_dsmla_roundtrip();
    return 0;
}
