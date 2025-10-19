#include "llama-sparse-topk.h"
#include "llama-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>

namespace llama {

using std::function;

// Select per-query top-k indices from a [n_kv, n_q] logits matrix.
// Applies a causal mask (no attending to future positions) before top-k.

ggml_tensor * sparse_attn_topk::select_topk_tokens(
    ggml_context * ctx,
    ggml_tensor * token_importance,
    int64_t n_tokens,
    const function<void(ggml_tensor *, const char *, int)> & cb) {
    (void)n_tokens; // Unused parameter

    printf("SPARSE TOPK: Starting select_topk_tokens\n");
    printf("SPARSE TOPK: token_importance shape: [%%" PRId64 ", %%" PRId64 ", %%" PRId64 ", %%" PRId64 "]\n",
           token_importance->ne[0], token_importance->ne[1], token_importance->ne[2], token_importance->ne[3]);
    fflush(stdout);

    // Apply causal mask so queries cannot attend to positions s > t within the block
    ggml_tensor * logits_masked = ggml_diag_mask_inf(ctx, token_importance, /*n_past=*/0);
    cb(logits_masked, "topk_logits_masked", -1);

    const int64_t n_kv = logits_masked->ne[0];
    const int64_t n_q  = logits_masked->ne[1];
    const int64_t top_k = (64 < n_kv) ? 64 : n_kv; // default k=64 capped by available keys

    printf("SPARSE TOPK: n_kv=%" PRId64 ", n_q=%" PRId64 ", top_k=%" PRId64 "\n", n_kv, n_q, top_k);
    fflush(stdout);

    // ggml_top_k over a 2D tensor returns [top_k, n_q, 1, 1] (per-column top-k)
    ggml_tensor * topk_indices = ggml_top_k(ctx, logits_masked, top_k);
    cb(topk_indices, "topk_indices", -1);

    printf("SPARSE TOPK: Final topk_indices shape: [%%" PRId64 ", %%" PRId64 ", %%" PRId64 ", %%" PRId64 "]\n",
           topk_indices->ne[0], topk_indices->ne[1], topk_indices->ne[2], topk_indices->ne[3]);
    fflush(stdout);

    return topk_indices;
}

} // namespace llama
