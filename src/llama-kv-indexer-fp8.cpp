#include "llama-kv-cache.h"
#include "llama-model.h"

// DeepSeek V3.2 FP8 indexer KV cache initialization.
// This is factored out of llama-kv-cache.cpp to keep DeepSeek-specific
// logic separate from the generic KV cache implementation.

void llama_init_indexer_fp8_sidecar(const llama_model & model, llama_kv_cache & kv) {
    const char * env_fp8 = getenv("LLAMA_FP8_INDEXER_CACHE");
    const bool enable_fp8 = (env_fp8 && atoi(env_fp8) != 0);
    if (!enable_fp8) {
        return;
    }

    if (!kv.is_arch_deepseek_v3_2()) {
        return;
    }


    for (uint32_t il = 0; il < model.hparams.n_layer; ++il) {
        if (model.layers[il].attn_indexer_wk == nullptr) {
            continue;
        }

        const int64_t index_head_dim = model.layers[il].attn_indexer_wk->ne[1];

        // The kv_layer for this il may be re-used; map_layer_ids is internal
        // to llama_kv_cache. We expose allocation by going through
        // get_k_indexer_fp8_raw() as a signal, but actual tensor creation
        // still needs to be done inside llama_kv_cache to keep buffer
        // ownership consistent. For now, we only rely on kv_layer fields
        // being present; allocation remains in llama_kv_cache ctor.
        //
        // NOTE: This helper is currently a placeholder hook for future
        // DeepSeek-specific behaviors (e.g. validating layout or logging).
        (void)index_head_dim;
    }
}

