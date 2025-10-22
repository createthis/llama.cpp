#ifndef LLAMA_SPARSE_TOPK_H
#define LLAMA_SPARSE_TOPK_H

#include <functional>
#include "ggml.h"

namespace llama {

using std::function;

// Top-k selector implementation for DeepSeek V3.2
struct sparse_attn_topk {
    // Lightning Indexer KV-aware selection
    static ggml_tensor * select_topk_tokens_indexer_kvaware(
        ggml_context * ctx,
        ggml_tensor * q_indexer,   // [D_index, H_index, T]
        ggml_tensor * k_indexer,   // [D_index, N_kv]
        ggml_tensor * weights,     // [H_index, T]
        ggml_tensor * kq_mask,     // [N_kv, PAD(T)] or [N_kv, T]
        int64_t top_k,
        const function<void(ggml_tensor *, const char *, int)> & cb);
};

} // namespace llama

#endif // LLAMA_SPARSE_TOPK_H

