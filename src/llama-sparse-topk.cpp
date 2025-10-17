#include "llama-sparse-topk.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>

namespace llama {

using std::function;

ggml_tensor * sparse_attn_topk::select_topk_tokens(
    ggml_context * ctx,
    ggml_tensor * token_importance,
    int64_t n_tokens,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    (void)n_tokens; // Unused parameter
    
    // DeepSeek V3.2 Top-k Selector implementation
    // Selects the top-k most important tokens for sparse attention
    
    printf("SPARSE TOPK: Starting select_topk_tokens\n");
    printf("SPARSE TOPK: token_importance shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           token_importance->ne[0], token_importance->ne[1], token_importance->ne[2], token_importance->ne[3]);
    printf("SPARSE TOPK: token_importance total elements: %" PRId64 "\n", ggml_nelements(token_importance));
    fflush(stdout);
    
    // The token_importance tensor should have shape [n_tokens, n_tokens]
    // We need to extract the importance scores for each token
    
    // Extract actual number of tokens from tensor
    const int64_t actual_n_tokens = token_importance->ne[0]; // Should be n_tokens
    
    // Use top-64 tokens for sparse attention (DeepSeek V3.2 default)
    const int64_t top_k = (64 < actual_n_tokens) ? 64 : actual_n_tokens;
    
    // For each token, we need to find the top-k important previous tokens
    // The token_importance matrix has shape [n_tokens, n_tokens] where
    // token_importance[i, j] represents the importance of token j for token i
    
    printf("SPARSE TOPK: actual_n_tokens = %" PRId64 ", top_k = %" PRId64 "\n", actual_n_tokens, top_k);
    fflush(stdout);
    
    // We need to extract the diagonal or specific rows depending on the current token
    // For simplicity, let's use the last row (current token's view of all previous tokens)
    
    // Create a tensor for the row index (last row)
    ggml_tensor * row_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    // Set the row index value to the last row
    int32_t * row_data = (int32_t *)row_index->data;
    row_data[0] = actual_n_tokens - 1;
    
    // Extract the importance scores for the current token (last row)
    ggml_tensor * current_importance = ggml_get_rows(ctx, token_importance, row_index);
    cb(current_importance, "current_importance", -1);
    
    printf("SPARSE TOPK: After get_rows - current_importance shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           current_importance->ne[0], current_importance->ne[1], current_importance->ne[2], current_importance->ne[3]);
    fflush(stdout);
    
    // Reshape to [actual_n_tokens, 1] for ggml_top_k
    current_importance = ggml_reshape_2d(ctx, current_importance, actual_n_tokens, 1);
    cb(current_importance, "current_importance_reshaped", -1);
    
    // Get indices of top-k important tokens
    // ggml_top_k returns a tensor with dimensions [top_k, 1, 1, 1]
    ggml_tensor * topk_indices = ggml_top_k(ctx, current_importance, top_k);
    cb(topk_indices, "topk_indices", -1);
    
    // topk_indices now has dimensions [top_k, 1, 1, 1]
    printf("SPARSE TOPK: Final topk_indices shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", 
           topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    printf("SPARSE TOPK: select_topk_tokens completed successfully\n");
    fflush(stdout);
    
    return topk_indices;
}

} // namespace llama