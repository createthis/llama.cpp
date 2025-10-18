#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <string>

// Helper function to get memory usage in human-readable format
static std::string format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    size_t unit_idx = 0;
    double size = bytes;
    
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_idx]);
    return std::string(buffer);
}

namespace llama {

using std::function;

// Modified to compute only the last row (I_{t,:}) for the last token t = T-1
ggml_tensor * sparse_attn_indexer::compute_token_importance(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    bool is_lite,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    
    // DeepSeek V3.2 Lightning Indexer implementation - Single row version
    // Mathematical formula: I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    
    printf("=== SPARSE INDEXER: Starting compute_token_importance for layer %d ===\n", layer_idx);
    size_t initial_mem = ggml_used_mem(ctx);
    printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
    printf("Input tensor cur shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    printf("Input tensor cur total elements: %" PRId64 "\n", ggml_nelements(cur));
    fflush(stdout);
    
    // Extract only the last token (t = T-1) to compute I_{t,:} for the last token
    const int64_t n_tokens = cur->ne[1];
    
    // Create index tensor for the last token
    ggml_tensor * t_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    int32_t * t_idx_data = (int32_t *)t_idx->data;
    t_idx_data[0] = n_tokens - 1;
    
    // Extract the last token: x_t = cur[:, T-1]
    ggml_tensor * x_t = ggml_get_rows(ctx, cur, t_idx);
    cb(x_t, "x_t_last_token", layer_idx);
    printf("SPARSE INDEXER: Extracted last token x_t shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           x_t->ne[0], x_t->ne[1], x_t->ne[2], x_t->ne[3]);
    fflush(stdout);
    
    // Indexer query projection (wq_a) - for non-lite version
    ggml_tensor * qr = nullptr;
    if (!is_lite) {
        printf("SPARSE INDEXER: Using non-lite version\n");
        fflush(stdout);
        
        // First projection: cur -> wq_a -> qr
        qr = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, x_t);
        cb(qr, "indexer_qr", layer_idx);
        printf("SPARSE INDEXER: After wq_a projection, qr shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               qr->ne[0], qr->ne[1], qr->ne[2], qr->ne[3]);
        printf("Memory usage after wq_a: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
               format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
        fflush(stdout);

        // Normalize the query representation
        qr = ggml_norm(ctx, qr, 1e-5f);
        cb(qr, "indexer_qr_norm", layer_idx);
        printf("SPARSE INDEXER: After normalization, qr shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               qr->ne[0], qr->ne[1], qr->ne[2], qr->ne[3]);
        printf("Memory usage after qr norm: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
               format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
        fflush(stdout);
    } else {
        printf("SPARSE INDEXER: Using lite version\n");
        fflush(stdout);
        // For lite version, use the current hidden state directly
        qr = x_t;
    }

    // Indexer key projection (wk) - k^I_s from the formula
    // This projects the current hidden state to the indexer key space
    ggml_tensor * k_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    cb(k_indexer, "indexer_k", layer_idx);
    printf("SPARSE INDEXER: After wk projection, k_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
    printf("Memory usage after wk: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);

    // Indexer key normalization (k_norm)
    k_indexer = ggml_norm(ctx, k_indexer, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        k_indexer = ggml_add(ctx, k_indexer, model.layers[layer_idx].attn_indexer_k_norm_bias);
    }
    cb(k_indexer, "indexer_k_norm", layer_idx);
    printf("SPARSE INDEXER: After k_norm, k_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
    printf("Memory usage after k_norm: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);

    // Indexer weights projection - w^I_{t,j} from the formula
    // These are the per-head weights for the indexer
    ggml_tensor * weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, x_t);
    cb(weights, "indexer_weights", layer_idx);
    printf("SPARSE INDEXER: After weights_proj, weights shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3]);
    printf("SPARSE INDEXER: weights total elements: %" PRId64 "\n", ggml_nelements(weights));
    printf("Memory usage after weights_proj: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);

    // Indexer query projection (wq_b) - q^I_{t,j} from the formula
    // This projects the normalized query representation to the indexer query space
    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qr);
    cb(q_indexer, "indexer_q", layer_idx);
    printf("SPARSE INDEXER: After wq_b projection, q_indexer shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           q_indexer->ne[0], q_indexer->ne[1], q_indexer->ne[2], q_indexer->ne[3]);
    printf("SPARSE INDEXER: q_indexer total elements: %" PRId64 "\n", ggml_nelements(q_indexer));
    printf("Memory usage after wq_b: %s (delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);

    // From DeepSeek V3.2 config: index_n_heads = 64, index_head_dim = 128
    const int64_t index_n_heads = 64;
    const int64_t index_head_dim = 128;
    
    // Reshape k_indexer to [index_head_dim, n_tokens]
    k_indexer = ggml_reshape_2d(ctx, k_indexer, index_head_dim, n_tokens);
    cb(k_indexer, "indexer_k_reshape", layer_idx);
    printf("SPARSE INDEXER: k_indexer reshaped to [%" PRId64 ", %" PRId64 "]\n", k_indexer->ne[0], k_indexer->ne[1]);
    fflush(stdout);
    
    // Reshape q_indexer to [index_n_heads, index_head_dim] (for single token)
    q_indexer = ggml_reshape_2d(ctx, q_indexer, index_head_dim, index_n_heads);
    cb(q_indexer, "indexer_q_reshape", layer_idx);
    printf("SPARSE INDEXER: q_indexer reshaped to [%" PRId64 ", %" PRId64 "]\n", q_indexer->ne[0], q_indexer->ne[1]);
    fflush(stdout);
    
    // Compute scores: k_indexer^T @ q_indexer -> [n_tokens, index_n_heads]
    // k_indexer is [index_head_dim, n_tokens], q_indexer is [index_head_dim, index_n_heads]
    ggml_tensor * scores = ggml_mul_mat(ctx, q_indexer, k_indexer);
    cb(scores, "indexer_scores", layer_idx);
    printf("SPARSE INDEXER: After dot product - scores shape: [%" PRId64 ", %" PRId64 "]\n", scores->ne[0], scores->ne[1]);
    fflush(stdout);
    
    // Apply ReLU activation
    ggml_tensor * relu_scores = ggml_relu(ctx, scores);
    cb(relu_scores, "indexer_relu_scores", layer_idx);
    printf("SPARSE INDEXER: After ReLU - relu_scores shape: [%" PRId64 ", %" PRId64 "]\n", relu_scores->ne[0], relu_scores->ne[1]);
    fflush(stdout);
    
    // Reshape weights to [index_n_heads, 1] for the single token
    weights = ggml_reshape_2d(ctx, weights, index_n_heads, 1);
    cb(weights, "indexer_weights_reshaped", layer_idx);
    printf("SPARSE INDEXER: weights reshaped to [%" PRId64 ", %" PRId64 "]\n", weights->ne[0], weights->ne[1]);
    fflush(stdout);
    
    // Contract heads with weights: relu_scores @ weights -> [n_tokens, 1]
    ggml_tensor * token_importance_col = ggml_mul_mat(ctx, weights, relu_scores);
    cb(token_importance_col, "token_importance_col", layer_idx);
    printf("SPARSE INDEXER: After weighted sum - token_importance_col shape: [%" PRId64 ", %" PRId64 "]\n", 
           token_importance_col->ne[0], token_importance_col->ne[1]);
    fflush(stdout);
    
    // Transpose to get row vector [1, n_tokens]
    ggml_tensor * token_importance = ggml_cont(ctx, ggml_transpose(ctx, token_importance_col));
    cb(token_importance, "token_importance_final", layer_idx);
    
    if (token_importance) {
        printf("SPARSE INDEXER: Final token_importance shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
               token_importance->ne[0], token_importance->ne[1], token_importance->ne[2], token_importance->ne[3]);
        printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(), 
               format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
        fflush(stdout);
    } else {
        printf("Error: token_importance is null after reshape\n");
        return nullptr;
    }

    return token_importance;
}

} // namespace llama
