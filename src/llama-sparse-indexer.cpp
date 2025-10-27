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

    const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
    const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);

    // Compute Indexer K for current tokens and (optionally) write to cache
    ggml_tensor * Kindexer_cur = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wk, cur);

    // Apply LayerNorm over D_index (per token), then gamma/beta
    const int64_t D_index = model.layers[layer_idx].attn_indexer_wk->ne[1];
    ggml_tensor * K3d = ggml_reshape_3d(ctx, Kindexer_cur, D_index, 1, n_tokens); // [D,1,T]
    ggml_tensor * K_mean = ggml_sum_rows(ctx, K3d);                                // [1,1,T]
    K_mean = ggml_scale(ctx, K_mean, 1.0f / (float) D_index);                      // [1,1,T]
    ggml_tensor * K_mean_rep = ggml_repeat(ctx, K_mean, K3d);                      // [D,1,T]
    ggml_tensor * K_centered = ggml_sub(ctx, K3d, K_mean_rep);                     // [D,1,T]
    ggml_tensor * K_var = ggml_sum_rows(ctx, ggml_sqr(ctx, K_centered));           // [1,1,T]
    K_var = ggml_scale(ctx, K_var, 1.0f / (float) D_index);                        // [1,1,T]
    ggml_tensor * K_var_eps = ggml_clamp(ctx, K_var, 1e-6f, 1e9f);                 // [1,1,T]
    ggml_tensor * K_std = ggml_sqrt(ctx, K_var_eps);                                // [1,1,T]
    ggml_tensor * K_std_rep = ggml_repeat(ctx, K_std, K_centered);                  // [D,1,T]
    ggml_tensor * K_normed = ggml_div(ctx, K_centered, K_std_rep);                  // [D,1,T]
    if (model.layers[layer_idx].attn_indexer_k_norm != nullptr) {
        ggml_tensor * gamma = model.layers[layer_idx].attn_indexer_k_norm;         // [D]
        ggml_tensor * gamma_r = ggml_repeat(ctx, gamma, K_normed);                 // [D,1,T]
        K_normed = ggml_mul(ctx, K_normed, gamma_r);
    }
    if (model.layers[layer_idx].attn_indexer_k_norm_bias != nullptr) {
        ggml_tensor * beta = model.layers[layer_idx].attn_indexer_k_norm_bias;     // [D]
        ggml_tensor * beta_r = ggml_repeat(ctx, beta, K_normed);                   // [D,1,T]
        K_normed = ggml_add(ctx, K_normed, beta_r);
    }
    // reshape back to [D, T]
    Kindexer_cur = ggml_reshape_2d(ctx, K_normed, D_index, n_tokens);
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

    // Removed rotate_activation on Kindexer_cur to match reference implementations

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
            if (dbg) {
                printf("[SPARSE-IDX-Q] L%d: applied attn_q_a_norm to indexer qsrc\n", layer_idx);
                fflush(stdout);
            }
        } else {
            printf("[SPARSE-IDX-Q][WARN] L%d: attn_q_a_norm not found; using plain RMSNorm for indexer qsrc\n", layer_idx);
            fflush(stdout);
        }
    } else {
        qsrc = ggml_norm(ctx, cur, 1e-5f);
    }

    if (dbg) {
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
    }

    ggml_tensor * q_indexer = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_wq_b, qsrc);

    // index head dim (head_dim in Tilelang) - already defined earlier as D_index
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

    // Removed rotate_activation on q_indexer to match reference implementations

    cb(q_indexer, "indexer_q", layer_idx);

    // Diagnostic: sample small window of q_indexer [D_index, H_index, T]
    {
        const int64_t sd0 = std::min<int64_t>(q_indexer->ne[0], (int64_t)8);
        const int64_t sd1 = std::min<int64_t>(q_indexer->ne[1], (int64_t)8);
        const int64_t sd2 = std::min<int64_t>(q_indexer->ne[2], (int64_t)8);
        ggml_tensor * q_sample = ggml_view_3d(ctx, q_indexer,
                sd0, sd1, sd2,
                q_indexer->nb[1], q_indexer->nb[2], 0);
        cb(q_sample, "indexer_q_sample", layer_idx);
    }


    // Approximate q_scale via per-(head, token) RMS of q_indexer across D_index
    // q_indexer: [D_index, H_index, T]
    ggml_tensor * q_sqr = ggml_sqr(ctx, q_indexer);                                  // [D_index, H, T]
    ggml_tensor * q_sum = ggml_sum_rows(ctx, q_sqr);                                  // [1, H, T]
    ggml_tensor * q_mean= ggml_scale(ctx, q_sum, 1.0f / (float) D_index);             // [1, H, T]
    ggml_tensor * q_rms = ggml_sqrt(ctx, q_mean);                                     // [1, H, T]
    if (dbg) printf("[SPARSE-IDX-QRMS] L%d: computed q_rms over D_index; D_index=%" PRId64 " H=%" PRId64 " T=%" PRId64 "\n",
           layer_idx, D_index, H_index, n_tokens);

    // Build base weights from projection on cur
    ggml_tensor * idx_weights = ggml_mul_mat(ctx, model.layers[layer_idx].attn_indexer_weights_proj, cur); // [H, T]


    // Diagnostic: sample small windows of K indexer cache [D_index, N_kv]
    if (mctx) {
        ggml_tensor * kidx_cache = mctx->get_k_indexer_full(const_cast<ggml_context*>(ctx), layer_idx);
        if (kidx_cache) {
            const int64_t d0 = std::min<int64_t>(kidx_cache->ne[0], (int64_t)8);
            const int64_t c0 = std::min<int64_t>(kidx_cache->ne[1], (int64_t)8);
            ggml_tensor * kcache_head = ggml_view_2d(ctx, kidx_cache, d0, c0, kidx_cache->nb[1], 0);
            cb(kcache_head, "indexer_k_cache_head", layer_idx);
            if (kidx_cache->ne[1] > c0) {
                size_t off_tail = (kidx_cache->ne[1] - c0) * kidx_cache->nb[1];
                ggml_tensor * kcache_tail = ggml_view_2d(ctx, kidx_cache, d0, c0, kidx_cache->nb[1], off_tail);
                cb(kcache_tail, "indexer_k_cache_tail", layer_idx);
            }
        }
    }

    // Diagnostic: sample small window of idx_weights [H_index, T]
    {
        ggml_tensor * idxw = idx_weights;
        const int64_t sw0 = std::min<int64_t>(idxw->ne[0], (int64_t)8);
        const int64_t sw1 = std::min<int64_t>(idxw->ne[1], (int64_t)8);
        ggml_tensor * idxw_sample = ggml_view_2d(ctx, idxw, sw0, sw1, idxw->nb[1], 0);
        cb(idxw_sample, "indexer_weights_sample", layer_idx);
    }

    fflush(stdout);
    // Scale weights by 1/sqrt(H_index) and 1/sqrt(D_index), then multiply by q_rms
    idx_weights = ggml_scale(ctx, idx_weights, 1.0f / sqrtf((float) H_index));
    idx_weights = ggml_scale(ctx, idx_weights, 1.0f / sqrtf((float) D_index));

    // Broadcast q_scale proxy [1,H,T] to [H,T] and multiply
    ggml_tensor * q_scale_proxy = ggml_reshape_2d(ctx, q_rms, H_index, n_tokens);     // [H, T]
    idx_weights = ggml_mul(ctx, idx_weights, q_scale_proxy);                          // [H, T]

    cb(idx_weights, "indexer_weights", layer_idx);
    ggml_tensor * Kindexer_cache = mctx ? mctx->get_k_indexer_full(ctx, layer_idx)
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
    ggml_cgraph * gf,
    ggml_backend_sched_t sched,
    ggml_backend_t backend_cpu)
{
    const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
    const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);

    size_t initial_mem = 0;
    if (dbg) {
        printf("=== SPARSE INDEXER: build_kvaware_topk_indices L%d ===\n", layer_idx);
        initial_mem = ggml_used_mem(ctx);
        printf("Initial memory usage: %s\n", format_memory_size(initial_mem).c_str());
        fflush(stdout);

        // Dump indexer dims and sanity-check shapes
        const int64_t D_index_dbg = model.layers[layer_idx].attn_indexer_wk->ne[1];
        const int64_t H_index_dbg = model.layers[layer_idx].attn_indexer_wq_b->ne[1] / D_index_dbg;
        printf("[SPARSE-DBG-IDX] L%d: D_index=%" PRId64 " H_index=%" PRId64 " n_tokens=%" PRId64 "\n", layer_idx, (int64_t) D_index_dbg, (int64_t) H_index_dbg, (int64_t) n_tokens);
        fflush(stdout);
    }
    IndexerKVTriplet trip = compute_indexer_triplet(ctx, model, layer_idx, cur, n_tokens, mctx, k_idxs,
        inp_pos, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
        cb, gf);
    // Use full-width indexer K view only for DeepSeek V3.2
    if (model.arch == LLM_ARCH_DEEPSEEK3_2 && mctx) {
        ggml_tensor * kidx_full = mctx->get_k_indexer_full(const_cast<ggml_context*>(ctx), layer_idx);
        if (kidx_full && kidx_full->ne[1] >= trip.k_indexer_cache->ne[1]) {
            trip.k_indexer_cache = kidx_full;
        }
    }
    ggml_tensor * Kindexer_cache = trip.k_indexer_cache;

    if (top_k <= 0) {
        top_k = std::max<int64_t>(64, std::min<int64_t>(1024, Kindexer_cache->ne[1]));
    }
    ggml_tensor * kvaware_indices = llama::sparse_attn_topk::select_topk_tokens_indexer_kvaware(
        ctx, trip.q_indexer, Kindexer_cache, trip.idx_weights, kq_mask, top_k, cb, gf, sched, backend_cpu);
    if (dbg) {
        printf("SPARSE INDEXER: Final topk_indices [k,T]=[%" PRId64 ", %" PRId64 "]\n",
               kvaware_indices->ne[0], kvaware_indices->ne[1]);
        printf("Final memory usage: %s (total delta: %s)\n", format_memory_size(ggml_used_mem(ctx)).c_str(),
               format_memory_size(ggml_used_mem(ctx) - initial_mem).c_str());
        fflush(stdout);
    }
    return kvaware_indices;
}


} // namespace llama

