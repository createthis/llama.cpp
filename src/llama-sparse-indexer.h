#ifndef LLAMA_SPARSE_INDEXER_H
#define LLAMA_SPARSE_INDEXER_H

#include <functional>
#include "ggml.h"

// Forward declarations
struct llama_model;

namespace llama {

using std::function;

// Lightning indexer implementation for DeepSeek V3.2
// Based on the mathematical formula:
// I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
struct sparse_attn_indexer {
    // Compute token importance scores using the lightning indexer
    static ggml_tensor * compute_token_importance(
        ggml_context * ctx,
        const llama_model & model,
        int layer_idx,
        ggml_tensor * cur,
        bool is_lite,
        const function<void(ggml_tensor *, const char *, int)> & cb);
};

} // namespace llama

#endif // LLAMA_SPARSE_INDEXER_H