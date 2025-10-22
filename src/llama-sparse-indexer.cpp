#include "llama-sparse-indexer.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "llama-sparse-topk.h"

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


IndexerKVTriplet sparse_attn_indexer::compute_indexer_triplet(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    int64_t n_tokens,
    const llama_kv_cache_context * mctx,
    ggml_tensor * k_idxs,
    ggml_tensor * inp_pos,
    int64_t n_rot,
    int rope_type,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    const function<void(ggml_tensor *, const char *, int)> & cb,
    ggml_cgraph * gf) {
    // Compute Indexer K for current tokens and (optionally) write to cache
    ggml_tensor * Kindexer_cur = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);
    Kindexer_cur = ggml_norm(ctx, Kindexer_cur, 1e-5f);
    if (model.layers[layer_idx].attn_indexer_k_norm != nullptr) {
        ggml_tensor * gamma = model.layers[layer_idx].attn_indexer_k_norm;
        ggml_tensor * gamma_r = ggml_repeat(ctx, gamma, Kindexer_cur);
        Kindexer_cur = ggml_mul(ctx, Kindexer_cur, gamma_r);
    }
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        ggml_tensor * beta = model.layers[layer_idx].attn_indexer_k_norm_bias;
        ggml_tensor * beta_r = ggml_repeat(ctx, beta, Kindexer_cur);
        Kindexer_cur = ggml_add(ctx, Kindexer_cur, beta_r);
    }
    cb(Kindexer_cur, "indexer_k_norm", layer_idx);

    // Apply RoPE to the first n_rot dims of K-indexer: view as [n_rot, 1, T]
    if (n_rot > 0) {
        ggml_tensor * Kidx_pe = ggml_view_3d(ctx, Kindexer_cur,
            n_rot, 1, n_tokens,
            ggml_row_size(Kindexer_cur->type, Kindexer_cur->ne[0]),
            ggml_row_size(Kindexer_cur->type, Kindexer_cur->ne[0]),
            0);
        Kidx_pe = ggml_rope_ext(ctx, Kidx_pe, inp_pos, nullptr,
            n_rot, (enum llama_rope_type) rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        // Reuse Kindexer_cur as concatenation of [pe, nope]
        ggml_tensor * Kidx_nope = ggml_view_3d(ctx, Kindexer_cur,
            Kindexer_cur->ne[0] - n_rot, 1, n_tokens,
            ggml_row_size(Kindexer_cur->type, Kindexer_cur->ne[0]),
            ggml_row_size(Kindexer_cur->type, Kindexer_cur->ne[0]),
            ggml_row_size(Kindexer_cur->type, n_rot));
        Kindexer_cur = ggml_concat(ctx, Kidx_pe, Kidx_nope, 0);
        cb(Kindexer_cur, "indexer_k_rope", layer_idx);
    }

    // Minimal rotate_activation approximation: circularly shift Dim0 by D/2
    {
        const int64_t Dtot = Kindexer_cur->ne[0];
        if (Dtot > 1) {
            const int64_t D2 = Dtot/2;
            if (D2 > 0) {
                ggml_tensor * k_hi = ggml_view_3d(ctx, Kindexer_cur,
                    Dtot - D2, Kindexer_cur->ne[1], Kindexer_cur->ne[2],
                    Kindexer_cur->nb[1], Kindexer_cur->nb[2], D2 * Kindexer_cur->nb[0]);
                ggml_tensor * k_lo = ggml_view_3d(ctx, Kindexer_cur,
                    D2, Kindexer_cur->ne[1], Kindexer_cur->ne[2],
                    Kindexer_cur->nb[1], Kindexer_cur->nb[2], 0);
                Kindexer_cur = ggml_concat(ctx, k_hi, k_lo, 0);
                cb(Kindexer_cur, "indexer_k_rot", layer_idx);
            }
        }
    }

    if (mctx && gf) {
        ggml_tensor * Kindexer_cur_3d = ggml_reshape_3d(ctx, Kindexer_cur, Kindexer_cur->ne[0], 1, n_tokens);
        ggml_build_forward_expand(gf, mctx->cpy_k_indexer(ctx, Kindexer_cur_3d, k_idxs, layer_idx));
    }
    // Build q_indexer and weights
    ggml_tensor * qsrc = nullptr;
    const bool has_wq_a = (model.layers[layer_idx].wq_a != nullptr);
    if (has_wq_a) {
        qsrc = ggml_mul_mat(ctx, model.layers[layer_idx].wq_a, cur);
        // Apply learned RMSNorm (attn_q_a_norm) like main attention path
        // This aligns with TileLang qr = q_norm(wq_a(x))
        qsrc = ggml_norm(ctx, qsrc, 1e-5f);
        if (model.layers[layer_idx].attn_q_a_norm) {
            ggml_tensor * gamma_q = model.layers[layer_idx].attn_q_a_norm;
            ggml_tensor * gamma_q_r = ggml_repeat(ctx, gamma_q, qsrc);
            qsrc = ggml_mul(ctx, qsrc, gamma_q_r);
            printf("[SPARSE-IDX-Q] L%d: applied attn_q_a_norm to indexer qsrc\n", layer_idx);
            fflush(stdout);
        } else {
            printf("[SPARSE-IDX-Q][WARN] L%d: attn_q_a_norm not found; using plain RMSNorm for indexer qsrc\n", layer_idx);
            fflush(stdout);
        }
    } else {
        qsrc = ggml_norm(ctx, cur, 1e-5f);
    }

    // Logging and sanity checks for potential lite-config mismatch
    const int64_t qsrc_in_dim = qsrc ? qsrc->ne[0] : -1;
    const int64_t wq_b_in_dim = model.layers[layer_idx].attn_indexer_wq_b ? model.layers[layer_idx].attn_indexer_wq_b->ne[0] : -1;
    const int64_t wq_b_out_dim = model.layers[layer_idx].attn_indexer_wq_b ? model.layers[layer_idx].attn_indexer_wq_b->ne[1] : -1;
    printf("[SPARSE-IDX-Q] L%d: has_wq_a=%d qsrc_in=%lld wq_b_in=%lld wq_b_out=%lld\n",
           layer_idx, (int)has_wq_a, (long long)qsrc_in_dim, (long long)wq_b_in_dim, (long long)wq_b_out_dim);
    fflush(stdout);

    if (model.layers[layer_idx].attn_indexer_wq_b && qsrc) {
        if (wq_b_in_dim != qsrc_in_dim) {
            printf("[SPARSE-IDX-Q][WARN] L%d: attn_indexer_wq_b input dim (%lld) != qsrc dim (%lld). Lite config?\n",
                   layer_idx, (long long) wq_b_in_dim, (long long) qsrc_in_dim);
            fflush(stdout);
        }
    }

    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qsrc);

    // index head dim (head_dim in Tilelang)
    const int64_t D_index = model.layers[layer_idx].attn_indexer_wk->ne[1];
    // indexer head count (n_heads in Tilelang)
    const int64_t H_index = model.layers[layer_idx].attn_indexer_wq_b->ne[1] / D_index;
    if ((model.layers[layer_idx].attn_indexer_wq_b->ne[1] % D_index) != 0) {
        printf("[SPARSE-IDX-Q][WARN] L%d: wq_b_out_dim (%lld) is not divisible by D_index (%lld)\n",
               layer_idx, (long long) model.layers[layer_idx].attn_indexer_wq_b->ne[1], (long long) D_index);
        fflush(stdout);
    }
    q_indexer = ggml_reshape_3d(ctx, q_indexer, D_index, H_index, n_tokens);

    // Apply RoPE to the first n_rot dims of q_indexer: view as [n_rot, H, T]
    if (n_rot > 0) {
        ggml_tensor * qidx_pe = ggml_view_3d(ctx, q_indexer,
            n_rot, H_index, n_tokens,
            ggml_row_size(q_indexer->type, D_index),
            ggml_row_size(q_indexer->type, D_index) * H_index,
            0);
        qidx_pe = ggml_rope_ext(ctx, qidx_pe, inp_pos, nullptr,
            n_rot, (enum llama_rope_type) rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * qidx_nope = ggml_view_3d(ctx, q_indexer,
            D_index - n_rot, H_index, n_tokens,
            ggml_row_size(q_indexer->type, D_index),
            ggml_row_size(q_indexer->type, D_index) * H_index,
            ggml_row_size(q_indexer->type, n_rot));
        q_indexer = ggml_concat(ctx, qidx_pe, qidx_nope, 0);
        cb(q_indexer, "indexer_q_rope", layer_idx);
    }

    // Minimal rotate_activation approximation on q_indexer: circular shift Dim0 by D/2
    {
        const int64_t Dtot = q_indexer->ne[0];
        if (Dtot > 1) {
            const int64_t D2 = Dtot/2;
            if (D2 > 0) {
                ggml_tensor * q_hi = ggml_view_3d(ctx, q_indexer,
                    Dtot - D2, q_indexer->ne[1], q_indexer->ne[2],
                    q_indexer->nb[1], q_indexer->nb[2], D2 * q_indexer->nb[0]);
                ggml_tensor * q_lo = ggml_view_3d(ctx, q_indexer,
                    D2, q_indexer->ne[1], q_indexer->ne[2],
                    q_indexer->nb[1], q_indexer->nb[2], 0);
                q_indexer = ggml_concat(ctx, q_hi, q_lo, 0);
                cb(q_indexer, "indexer_q_rot", layer_idx);
            }
        }
    }

    cb(q_indexer, "indexer_q", layer_idx);

    // Approximate q_scale via per-(head, token) RMS of q_indexer across D_index
    // q_indexer: [D_index, H_index, T]
    ggml_tensor * q_sqr = ggml_sqr(ctx, q_indexer);                                  // [D_index, H, T]
    ggml_tensor * q_sum = ggml_sum_rows(ctx, q_sqr);                                  // [1, H, T]
    ggml_tensor * q_mean= ggml_scale(ctx, q_sum, 1.0f / (float) D_index);             // [1, H, T]
    ggml_tensor * q_rms = ggml_sqrt(ctx, q_mean);                                     // [1, H, T]
    printf("[SPARSE-IDX-QRMS] L%d: computed q_rms over D_index; D_index=%" PRId64 " H=%" PRId64 " T=%" PRId64 "\n",
           layer_idx, D_index, H_index, n_tokens);
    fflush(stdout);

    // Build base weights from projection on cur
    ggml_tensor * idx_weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur); // [H, T]
    // Scale weights by 1/sqrt(H_index) and 1/sqrt(D_index), then multiply by q_rms
    idx_weights = ggml_scale(ctx, idx_weights, 1.0f / sqrtf((float) H_index));
    idx_weights = ggml_scale(ctx, idx_weights, 1.0f / sqrtf((float) D_index));

    // Broadcast q_rms [1,H,T] to [H,T] and multiply
    ggml_tensor * q_rms_2d = ggml_reshape_2d(ctx, q_rms, H_index, n_tokens);          // [H, T]
    idx_weights = ggml_mul(ctx, idx_weights, q_rms_2d);                               // [H, T]

    cb(idx_weights, "indexer_weights", layer_idx);
    ggml_tensor * Kindexer_cache = mctx ? mctx->get_k_indexer(ctx, layer_idx)
                                        : ggml_reshape_2d(ctx, Kindexer_cur, D_index, n_tokens);
    IndexerKVTriplet out{ q_indexer, Kindexer_cache, idx_weights };
    return out;
}

ggml_tensor * sparse_attn_indexer::build_kvaware_topk_indices(
    ggml_context * ctx,
    const llama_model & model,
    int layer_idx,
    ggml_tensor * cur,
    int64_t n_tokens,
    const llama_kv_cache_context * mctx,
    ggml_tensor * k_idxs,
    ggml_tensor * kq_mask,
    int64_t top_k,
    ggml_tensor * inp_pos,
    int64_t n_rot,
    int rope_type,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    const function<void(ggml_tensor *, const char *, int)> & cb,
    ggml_cgraph * gf)
{
    printf("=== SPARSE INDEXER: build_kvaware_topk_indices L%d ===\n", layer_idx);
    size_t initial_mem = ggml_used_mem(ctx);
    printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
    fflush(stdout);
    // Dump indexer dims and sanity-check shapes
    const int64_t D_index_dbg = model.layers[layer_idx].attn_indexer_wk->ne[1];
    const int64_t H_index_dbg = model.layers[layer_idx].attn_indexer_wq_b->ne[1] / D_index_dbg;
    printf("[SPARSE-DBG-IDX] L%d: D_index=%" PRId64 " H_index=%" PRId64 " n_tokens=%" PRId64 "\n", layer_idx, (int64_t) D_index_dbg, (int64_t) H_index_dbg, (int64_t) n_tokens);
    fflush(stdout);
    IndexerKVTriplet trip = compute_indexer_triplet(ctx, model, layer_idx, cur, n_tokens, mctx, k_idxs,
        inp_pos, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
        cb, gf);
    ggml_tensor * Kindexer_cache = trip.k_indexer_cache;
    if (top_k <= 0) {
        top_k = std::max<int64_t>(64, std::min<int64_t>(1024, Kindexer_cache->ne[1]));
    }
    ggml_tensor * kvaware_indices = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx, trip.q_indexer, Kindexer_cache, trip.idx_weights, kq_mask, top_k, cb);
    printf("SPARSE INDEXER: Final topk_indices [k,T]=[%" PRId64 ", %" PRId64 "]\n",
           kvaware_indices->ne[0], kvaware_indices->ne[1]);
    printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(),
           format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
    fflush(stdout);
    return kvaware_indices;
}


} // namespace llama

