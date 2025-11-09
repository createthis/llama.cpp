#include "llama-sparse-topk.h"
#include <algorithm>
#include <vector>
#include <cstdint>

#include "llama-impl.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda-indexer.h"
#endif


#include <cstring>

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <chrono>

#include <cstdlib>
#include <climits>
namespace {

static inline uint32_t float_to_key_desc(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    // Map float bits to monotonically increasing unsigned keys (ascending order):
    // TileLang-compatible mapping: negative -> bitwise NOT, non-negative -> set sign bit
    if ((int32_t)u < 0) {
        u = ~u;
    } else {
        u |= 0x80000000u;
    }
    return u;
}

struct radix_topk_userdata {
    // currently unused; k is taken from dst->ne[0]
};

static void radix_topk_custom(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void)userdata;
    const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
    const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);
    ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    const int64_t N = src0->ne[0];
    const int64_t k = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);
    const size_t src_nb1 = src0->nb[1];
    const size_t src_nb0 = src0->nb[0];
    const size_t dst_nb1 = dst->nb[1];

    for (int64_t r = ith; r < nr; r += nth) {
        const char * row0 = (const char *)src0->data + r * src_nb1;
        int32_t * out_idx = (int32_t *)((char *)dst->data + r * dst_nb1);
        const int64_t KK = k < N ? k : N;

        // Precompute keys for this row
        std::vector<uint32_t> keys(N);
        for (int64_t i = 0; i < N; ++i) {
            float v = *(const float *)(row0 + (size_t)i*src_nb0);
            keys[i] = float_to_key_desc(v);
        }

        // Stage 1: histogram of high 8 bits (bits 31..24)
        uint32_t counts[256] = {0};
        for (int64_t i = 0; i < N; ++i) {
            uint32_t bin = (keys[i] >> 24) & 0xFFu;
            counts[bin]++;
        }
        // Find threshold bin: number of items with bin > thr0 is sum of counts above thr0
        auto sum_greater = [&](int b){ uint32_t s=0; for (int bb=b+1; bb<256; ++bb) s += counts[bb]; return s; };
        int thr0 = 0;
        uint32_t gt = 0;
        for (int b = 255; b >= 0; --b) {
            uint32_t sgt = sum_greater(b);
            uint32_t eq  = counts[b];
            if (sgt < (uint32_t)KK && sgt + eq >= (uint32_t)KK) { thr0 = b; gt = sgt; break; }
        }
        uint32_t eq0 = counts[thr0];
        int64_t remaining = (int64_t)KK - (int64_t)gt;
        if (remaining < 0) remaining = 0;

        // Collect selected (> thr0) and eq candidates
        std::vector<int32_t> selected; selected.reserve(KK);
        std::vector<int32_t> eq_list; eq_list.reserve(eq0);
        for (int64_t i = 0; i < N; ++i) {
            uint32_t bin = (keys[i] >> 24) & 0xFFu;
            if ((int)bin > thr0) {
                if ((int64_t)selected.size() < KK) selected.push_back((int32_t)i);
            } else if ((int)bin == thr0) {
                eq_list.push_back((int32_t)i);
            }
        }
        remaining = (int64_t)KK - (int64_t)selected.size();

        // Safety check: ensure we have enough candidates to fill K
        if ((int64_t)selected.size() + (int64_t)eq_list.size() < KK) {
            // Fallback: use partial_sort to guarantee correctness
            std::vector<int32_t> idx(N);
            for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(idx.begin(), idx.begin() + KK, idx.end(), cmp);
            for (int64_t i = 0; i < KK; ++i) out_idx[i] = idx[i];
            continue;
        }

        // Tail passes for equal bin
        int shifts[3] = {16, 8, 0};
        for (int pass = 0; pass < 3 && remaining > 0 && !eq_list.empty(); ++pass) {
            uint32_t c2[256] = {0};
            for (int idx : eq_list) {
                uint32_t bin = (keys[idx] >> shifts[pass]) & 0xFFu;
                c2[bin]++;
            }
            auto sum_greater2 = [&](int b){ uint32_t s=0; for (int bb=b+1; bb<256; ++bb) s += c2[bb]; return s; };
            int thr = 255;
            for (int b = 255; b >= 0; --b) {
                uint32_t sgt = sum_greater2(b);
                uint32_t eq  = c2[b];
                if (sgt < (uint32_t)remaining && sgt + eq >= (uint32_t)remaining) { thr = b; break; }
            }
            std::vector<int32_t> next_eq; next_eq.reserve(c2[thr]);
            // Add strictly greater than thr
            for (int idx : eq_list) {
                uint32_t bin = (keys[idx] >> shifts[pass]) & 0xFFu;
                if ((int)bin > thr) {
                    if ((int64_t)selected.size() < KK) { selected.push_back(idx); }
                } else if ((int)bin == thr) {
                    next_eq.push_back(idx);
                }
            }
            eq_list.swap(next_eq);
            remaining = (int64_t)KK - (int64_t)selected.size();
            if ((int64_t)selected.size() + (int64_t)eq_list.size() < remaining) {
                // Fallback safety
                break;
            }
        }
        // Final fill from eq_list if still remaining
        for (int64_t i = 0; i < (int64_t)eq_list.size() && (int64_t)selected.size() < KK; ++i) {
            selected.push_back(eq_list[i]);
        }

        // As a final fallback, if still not enough, use partial_sort
        if ((int64_t)selected.size() < KK) {
            std::vector<int32_t> idx(N);
            for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(idx.begin(), idx.begin() + KK, idx.end(), cmp);
            for (int64_t i = 0; i < KK; ++i) out_idx[i] = idx[i];
            continue;
        }

        // Output first KK indices (order arbitrary)
        for (int64_t i = 0; i < KK; ++i) out_idx[i] = selected[i];

        // Debug: compare with partial_sort for a few rows
        if (r < 8) {
            std::vector<int32_t> ref(N);
            for (int64_t i = 0; i < N; ++i) ref[i] = (int32_t)i;
            auto cmp = [&](int32_t a, int32_t b){
                float va = *(const float *)(row0 + (size_t)a*src_nb0);
                float vb = *(const float *)(row0 + (size_t)b*src_nb0);
                if (va != vb) return va > vb;
                return a < b;
            };
            std::partial_sort(ref.begin(), ref.begin() + KK, ref.end(), cmp);
            if (dbg) {
                printf("[radix debug] row=%lld top: ", (long long)r);
                for (int ii = 0; ii < (int)std::min<int64_t>(8, KK); ++ii) printf("%d ", out_idx[ii]);
                printf("| ref: ");
                for (int ii = 0; ii < (int)std::min<int64_t>(8, KK); ++ii) printf("%d ", ref[ii]);
                printf("\n");
                fflush(stdout);
            }
        }
    }
}

} // anonymous namespace

namespace llama {

static inline int find_last_unmasked(const float * col, int N, size_t nb0) {
    // col is [N] as a column with row stride nb0; return last index+1 where value > -1e29
    for (int i = N-1; i >= 0; --i) {
        float v = *(const float *)((const char*)col + (size_t)i*nb0);
        if (v > -1.0e29f) return i+1;
    }
    return 0;
}

using std::function;

struct fused_indexer_userdata { };

static void fused_indexer_custom(ggml_tensor * /*dst*/, int /*ith*/, int /*nth*/, void * /*userdata*/) {
    // no-op custom CPU fallback; all real work happens in CUDA backend path
}

static ggml_tensor * build_indexer_fused_logits(
    ggml_context * ctx,
    ggml_tensor * q2d, // [D, Tc*H]
    ggml_tensor * k2d, // [D, kv]
    ggml_tensor * w2d, // [H, Tc]
    ggml_tensor * k_scale // [kv]
) {
    int64_t kv = k2d->ne[1];
    int64_t Tc = w2d->ne[1];
    ggml_tensor * args[4] = { q2d, k2d, w2d, k_scale };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, kv, Tc, 1, 1, args, 4, fused_indexer_custom, 1, nullptr);
}


static ggml_tensor * idx_compute_scores_tile(
    ggml_context * ctx,
    ggml_tensor * q3d,
    ggml_tensor * a_k,
    ggml_tensor * weights,
    ggml_tensor * k_scale_2d,
    int64_t D, int64_t H,
    int64_t Tc, int64_t kv_end,
    int64_t t0,
    bool use_fp16) {
    ggml_tensor * scores_acc = nullptr;
    long HEAD_CHUNK = H;
    if (const char *env = getenv("LLAMA_SPARSE_TOPK_HEAD_CHUNK")) {
        long v = strtol(env, nullptr, 10);
        if (v > 0) HEAD_CHUNK = v;
    }
    if (HEAD_CHUNK > (long)H) HEAD_CHUNK = (long)H;
    if (HEAD_CHUNK < 1) HEAD_CHUNK = 1;

    ggml_tensor * w_slice = ggml_view_2d(ctx, weights, H, Tc, weights->nb[1], t0*weights->nb[1]);

    for (int64_t h0 = 0; h0 < H; h0 += HEAD_CHUNK) {
        int64_t ch = std::min<int64_t>(HEAD_CHUNK, H - h0);
        size_t q_off_head = (size_t)t0 * q3d->nb[1] + (size_t)h0 * q3d->nb[2];
        ggml_tensor * q_chunk_3d = ggml_view_3d(ctx, q3d, D, Tc, ch, q3d->nb[1], q3d->nb[2], q_off_head);
        q_chunk_3d = ggml_cont(ctx, q_chunk_3d);
        ggml_tensor * q_chunk_2d = ggml_reshape_2d(ctx, q_chunk_3d, D, Tc*ch);
        ggml_tensor * b_q = q_chunk_2d;
        if (use_fp16 && q_chunk_2d->type != GGML_TYPE_F16) {
            b_q = ggml_cast(ctx, q_chunk_2d, GGML_TYPE_F16);
            b_q = ggml_cont(ctx, b_q);
        }
        ggml_tensor * k_slice = ggml_view_2d(ctx, a_k, D, kv_end, a_k->nb[1], 0);
        ggml_tensor * logits_chunk = ggml_mul_mat(ctx, k_slice, b_q); // [kv_end, Tc*ch]
        logits_chunk = ggml_cont(ctx, logits_chunk);
        ggml_tensor * logits_chunk_3d = ggml_reshape_3d(ctx, logits_chunk, kv_end, ch, Tc); // [kv_end, ch, Tc]
        logits_chunk_3d = ggml_relu(ctx, logits_chunk_3d);
        size_t w_off_chunk = (size_t)h0 * w_slice->nb[0];
        ggml_tensor * w_sub_2d = ggml_view_2d(ctx, w_slice, ch, Tc, w_slice->nb[1], w_off_chunk); // [ch, Tc]
        w_sub_2d = ggml_cont(ctx, w_sub_2d);
        ggml_tensor * w_sub_3d = ggml_reshape_3d(ctx, w_sub_2d, ch, 1, Tc); // [ch,1,Tc]
        ggml_tensor * log_p = ggml_permute(ctx, logits_chunk_3d, 1, 0, 2, 3); // [ch, N_kv, Tc]
        log_p = ggml_cont(ctx, log_p);
        ggml_tensor * w_bc  = ggml_repeat(ctx, w_sub_3d, log_p);             // [ch, N_kv, Tc]
        w_bc = ggml_cont(ctx, w_bc);
        ggml_tensor * prod  = ggml_mul(ctx, log_p, w_bc);                     // [ch, N_kv, Tc]
        ggml_tensor * sum_ch= ggml_sum_rows(ctx, prod);                       // [1, N_kv, Tc]
        ggml_tensor * sum_p = ggml_permute(ctx, sum_ch, 1, 2, 0, 3);          // [kv_end, Tc, 1]
        sum_p = ggml_cont(ctx, sum_p);
        ggml_tensor * scores_chunk = ggml_reshape_2d(ctx, sum_p, kv_end, Tc);   // [kv_end, Tc]
        scores_acc = scores_acc ? ggml_add(ctx, scores_acc, scores_chunk) : scores_chunk;
    }
    ggml_tensor * scores_tc = scores_acc;
    // Apply k_scale after head reduction
    ggml_tensor * k_scale_head = ggml_view_2d(ctx, k_scale_2d, kv_end, 1, k_scale_2d->nb[1], 0);
    ggml_tensor * k_scale_bcast = ggml_repeat(ctx, k_scale_head, scores_tc); // [kv_end, Tc]
    scores_tc = ggml_mul(ctx, scores_tc, k_scale_bcast);
    return scores_tc;
}


ggml_tensor * sparse_attn_topk::select_topk_tokens_indexer_kvaware(
    ggml_context * ctx,
    ggml_tensor * q_indexer,   // [D, H, T]
    ggml_tensor * k_indexer,   // [D, N_kv]
    ggml_tensor * weights,     // [H, T]
    ggml_tensor * kq_mask,     // [N_kv, T] or [N_kv, PAD(T)]
    int64_t top_k,
    const std::function<void(ggml_tensor *, const char *, int)> & cb,
    ggml_cgraph * gf,
    ggml_backend_sched_t sched,
    ggml_backend_t /*backend_cpu*/) {
    const int64_t D    = q_indexer->ne[0];
    const int64_t H    = q_indexer->ne[1];
    const int64_t T    = q_indexer->ne[2];
    const int64_t N_kv = k_indexer->ne[1];

    const char * ENV_SPARSE_DEBUG = getenv("LLAMA_SPARSE_DEBUG");
    const bool dbg = (ENV_SPARSE_DEBUG && atoi(ENV_SPARSE_DEBUG) != 0);

    if (dbg) {
        printf("SPARSE TOPK KV-AWARE (INDEXER): q_indexer [D,H,T]=[%" PRId64 ",%" PRId64 ",%" PRId64 "]\n", D, H, T);
        printf("SPARSE TOPK KV-AWARE (INDEXER): k_indexer dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
               k_indexer->ne[0], k_indexer->ne[1], k_indexer->ne[2], k_indexer->ne[3]);
        printf("SPARSE TOPK KV-AWARE (INDEXER): weights [H,T]=[%" PRId64 ",%" PRId64 "]\n",
               weights ? weights->ne[0] : -1, weights ? weights->ne[1] : -1);
        if (kq_mask) {

            printf("SPARSE TOPK KV-AWARE (INDEXER): kq_mask dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
                   kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3], (int)kq_mask->type);
        }
        fflush(stdout);
    }

    // Shape/contiguity assertions for weights [H, T]
    GGML_ASSERT(D > 0 && H > 0 && T > 0 && N_kv > 0);
    GGML_ASSERT(weights != nullptr);
    GGML_ASSERT(weights->ne[0] == H);
    GGML_ASSERT(weights->ne[1] >= T);
    GGML_ASSERT(weights->nb[0] == (size_t) ggml_type_size(weights->type));
    GGML_ASSERT(weights->nb[1] == (size_t) ggml_row_size(weights->type, weights->ne[0]));
    // Ensure indexer K depth matches indexer Q depth
    GGML_ASSERT(k_indexer->ne[0] == D);
    // KV indexer currently expected as 2D [D, N_kv] or 3D with singleton stream
    GGML_ASSERT(k_indexer->ne[2] <= 1);

    // Q as [D, H*T]
    ggml_tensor * q_perm   = ggml_permute(ctx, q_indexer, 0, 2, 1, 3);   // [D, T, H]
    ggml_tensor * q_cont   = ggml_cont(ctx, q_perm);
    ggml_tensor * Q2d_full = ggml_reshape_2d(ctx, q_cont, D, T*H);
    cb(Q2d_full, "idxkv_Q2d_full", -1);

    // Optional FP16 path for indexer GEMMs
    ggml_tensor * k_indexer_f16 = k_indexer;
    const char *env_fp16 = getenv("LLAMA_SPARSE_TOPK_FP16");
    const bool use_fp16 = (env_fp16 == nullptr || atoi(env_fp16) != 0);
    if (use_fp16 && k_indexer->type != GGML_TYPE_F16) {
        k_indexer_f16 = ggml_cast(ctx, k_indexer, GGML_TYPE_F16);
        k_indexer_f16 = ggml_cont(ctx, k_indexer_f16);
    }

    // Diagnostics: sample K indexer head/tail once per call
    if (dbg) {
        const int64_t d0 = std::min<int64_t>(k_indexer->ne[0], (int64_t)8);
        const int64_t c0 = std::min<int64_t>(k_indexer->ne[1], (int64_t)8);
        // head columns
        ggml_tensor * kidx_head = ggml_view_2d(ctx, k_indexer, d0, c0, k_indexer->nb[1], 0);
        cb(kidx_head, "idxkv_k_indexer_head", -1);
        if (gf) { ggml_set_output(kidx_head); ggml_build_forward_expand(gf, kidx_head); }
        // tail columns
        if (k_indexer->ne[1] > c0) {
            size_t off_tail = (k_indexer->ne[1] - c0) * k_indexer->nb[1];
            ggml_tensor * kidx_tail = ggml_view_2d(ctx, k_indexer, d0, c0, k_indexer->nb[1], off_tail);
            cb(kidx_tail, "idxkv_k_indexer_tail", -1);
            if (gf) { ggml_set_output(kidx_tail); ggml_build_forward_expand(gf, kidx_tail); }
        }
    }

    ggml_tensor * mask_full = nullptr;
    ggml_tensor * win_starts = nullptr;
    (void)win_starts;
    ggml_tensor * win_ends   = nullptr;
    bool have_windows = false;
#ifdef GGML_USE_CUDA
        if (kq_mask && kq_mask->buffer && !ggml_backend_buffer_is_host(kq_mask->buffer)) {
            std::vector<int32_t> starts_h((size_t)T, 0);
            ggml_cuda_mask_window_starts_device_to_host_simple((const float *)kq_mask->data, (int)N_kv, (int)T, starts_h.data());
            ggml_tensor * starts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            memcpy(starts->data, starts_h.data(), sizeof(int32_t) * (size_t)T);
            ggml_set_input(starts);
            win_starts = starts; have_windows = true;
        }
#endif
#ifdef GGML_USE_CUDA
        if (kq_mask && kq_mask->buffer && !ggml_backend_buffer_is_host(kq_mask->buffer)) {
            std::vector<int32_t> ends_h((size_t)T, 0);
            ggml_cuda_mask_window_ends_device_to_host_simple((const float *)kq_mask->data, (int)N_kv, (int)T, ends_h.data());
            ggml_tensor * ends = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            memcpy(ends->data, ends_h.data(), sizeof(int32_t) * (size_t)T);
            ggml_set_input(ends);
            win_ends = ends; have_windows = true;
        }
#endif

#ifdef GGML_USE_CUDA

#endif
    if (kq_mask) {
        cb(kq_mask, "idxkv_kq_mask", -1);
        ggml_tensor * tmp_mask = kq_mask;
        cb(tmp_mask, "idxkv_mask2d", -1);
        if (tmp_mask->ne[0] == N_kv && tmp_mask->ne[1] >= T) {
            mask_full = tmp_mask; // use original mask; slice per tile and only contiguize the tile
        } else {
            printf("[TOPK-INDEXER] kq_mask dims [%lld,%lld] mismatch N_kv=%lld,T=%lld; ignoring mask for indexer selection\n",
                   (long long) tmp_mask->ne[0], (long long) tmp_mask->ne[1], (long long) N_kv, (long long) T);
            fflush(stdout);
            mask_full = nullptr;
        }
    }

    const int64_t k = std::min<int64_t>(top_k, N_kv);
    int64_t TILE_T = use_fp16 ? 512 : 256; // heuristic default; overridable via env
    if (const char *env = getenv("LLAMA_SPARSE_TOPK_TILE_T")) {
        long v = strtol(env, nullptr, 10);
        if (v > 0 && v <= 8192) TILE_T = v;
    }
    if (dbg) {
        printf("[TOPK-INDEXER] N_kv=%lld T=%lld k=%lld TILE_T=%lld H=%lld D=%lld\n",
               (long long) N_kv, (long long) T, (long long) k, (long long) TILE_T, (long long) H, (long long) D);
        fflush(stdout);
    }

    // K-scale proxy: RMS over D for each KV column
    ggml_tensor * k_sqr = ggml_sqr(ctx, k_indexer);                  // [D, N_kv]
    ggml_tensor * k_sum = ggml_sum_rows(ctx, k_sqr);                  // [1, N_kv]
    ggml_tensor * k_mean = ggml_scale(ctx, k_sum, 1.0f / (float) D);  // [1, N_kv]
    ggml_tensor * k_scale_vec = ggml_sqrt(ctx, k_mean);               // [1, N_kv]
    ggml_tensor * k_scale_2d = ggml_transpose(ctx, k_scale_vec);      // [N_kv, 1]
    k_scale_2d = ggml_cont(ctx, k_scale_2d);
    cb(k_scale_2d, "idxkv_k_scale_proxy", -1);

    ggml_tensor * result = nullptr; // [k, T]
    for (int64_t t0 = 0; t0 < T; t0 += TILE_T) {
        const int64_t Tc = std::min<int64_t>(TILE_T, T - t0);
        // If we have per-token windows, compute aggregate kv_end for this tile
        int64_t kv_end = N_kv;
        if (have_windows && win_ends) {
            int32_t * e = (int32_t*)win_ends->data;
            int64_t max_e = 0;
            for (int64_t t = 0; t < Tc; ++t) max_e = std::max<int64_t>(max_e, (int64_t)e[t0 + t]);
            kv_end = std::min<int64_t>(N_kv, std::max<int64_t>(k, max_e));
        } else {
            const char *env_full_kv = getenv("LLAMA_SPARSE_TOPK_FULL_KV");
            const bool use_full_kv = (env_full_kv ? atoi(env_full_kv) != 0 : true);
            kv_end = use_full_kv ? N_kv : std::min<int64_t>(N_kv, std::max<int64_t>(k, t0 + Tc));
        }

        // Use contiguized [D, T, H] directly for head-wise tiles
        ggml_tensor * q3d = q_cont;

        ggml_tensor * scores_tc = nullptr;
        {
                // Host wall-clock timing for the tile compute path (portable)
                auto __t0_wall = std::chrono::high_resolution_clock::now();


            const char *env_fused = getenv("LLAMA_SPARSE_INDEXER_FUSED");
            bool use_fused = (env_fused ? atoi(env_fused) != 0 : true);
            if (use_fused) {
                // prepare q2d tile [D, Tc*H]
                size_t q_off = (size_t)t0 * q3d->nb[1];
                ggml_tensor * q_tile3d = ggml_view_3d(ctx, q3d, D, Tc, H, q3d->nb[1], q3d->nb[2], q_off);
                q_tile3d = ggml_cont(ctx, q_tile3d);
                ggml_tensor * q_tile2d = ggml_reshape_2d(ctx, q_tile3d, D, Tc*H);
                q_tile2d = ggml_cont(ctx, q_tile2d);
                // Determine kv window for this tile
                int64_t kv_s = 0;
                int64_t kv_e = kv_end;
                if (have_windows && win_ends) {
                    int32_t * e = (int32_t*)win_ends->data;
                    int64_t max_e = 0;
                    for (int64_t t = 0; t < Tc; ++t) max_e = std::max<int64_t>(max_e, (int64_t)e[t0 + t]);
                    kv_e = std::min<int64_t>(N_kv, std::max<int64_t>(k, max_e));
                }
                int64_t kv_len = std::max<int64_t>(0, kv_e - kv_s);
                // k slice [D, kv_len]
                ggml_tensor * k_slice = ggml_view_2d(ctx, k_indexer_f16, D, kv_len, k_indexer_f16->nb[1], kv_s * k_indexer_f16->nb[1]);
                k_slice = ggml_cont(ctx, k_slice);
                // w slice [H, Tc]
                ggml_tensor * w_slice = ggml_view_2d(ctx, weights, H, Tc, weights->nb[1], t0*weights->nb[1]);
                w_slice = ggml_cont(ctx, w_slice);
                // k_scale head [kv_len]
                ggml_tensor * ks_head = ggml_view_2d(ctx, k_scale_2d, kv_len, 1, k_scale_2d->nb[1], kv_s * k_scale_2d->nb[1]);
                ks_head = ggml_reshape_1d(ctx, ks_head, kv_len);
                ks_head = ggml_cont(ctx, ks_head);

                if (dbg && sched && t0 == 0) {
                    ggml_backend_t bq = ggml_backend_sched_get_tensor_backend(sched, q_tile2d);
                    ggml_backend_t bk = ggml_backend_sched_get_tensor_backend(sched, k_slice);
                    ggml_backend_t bw = ggml_backend_sched_get_tensor_backend(sched, w_slice);
                    ggml_backend_t bs = ggml_backend_sched_get_tensor_backend(sched, ks_head);
                    printf("[idxkv fused inputs strides] q nb=[%zu,%zu] k nb=[%zu,%zu] w nb=[%zu,%zu] ks nb0=%zu\n",
                           (size_t)q_tile2d->nb[0], (size_t)q_tile2d->nb[1],
                           (size_t)k_slice->nb[0], (size_t)k_slice->nb[1],
                           (size_t)w_slice->nb[0], (size_t)w_slice->nb[1],
                           (size_t)ks_head->nb[0]);
                    printf("[idxkv fused inputs] backends: q=%s k=%s w=%s ks=%s\n",
                           bq ? ggml_backend_name(bq) : "null",
                           bk ? ggml_backend_name(bk) : "null",
                           bw ? ggml_backend_name(bw) : "null",
                           bs ? ggml_backend_name(bs) : "null");
                    fflush(stdout);
                }

                const char *env_fused_dev = getenv("LLAMA_SPARSE_INDEXER_FUSED_DEVICE");
                bool use_fused_device = (env_fused_dev ? atoi(env_fused_dev) != 0 : true);
                if (dbg && t0 == 0) {
                    printf("[idxkv] fused_device=%d\n", (int)use_fused_device);
                    fflush(stdout);
                }
                scores_tc = use_fused_device ?
                    ggml_indexer_logits_fused(ctx, q_tile2d, k_slice, w_slice, ks_head) :
                    build_indexer_fused_logits(ctx, q_tile2d, k_slice, w_slice, ks_head);

                if (dbg && t0 == 0) {
                    ggml_tensor * ref_scores = idx_compute_scores_tile(ctx, q3d, k_indexer_f16, weights, k_scale_2d, D, H, Tc, kv_end, t0, use_fp16);
                    ggml_tensor * ref_head = ggml_view_2d(ctx, ref_scores, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), ref_scores->nb[1], 0);
                    cb(ref_head, "idxkv_scores_ref_head", -1);
                    if (gf) { ggml_set_output(ref_head); ggml_build_forward_expand(gf, ref_head); }
                    ggml_tensor * fused_head = ggml_view_2d(ctx, scores_tc, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), scores_tc->nb[1], 0);
                    cb(fused_head, "idxkv_scores_fused_head", -1);
                    if (gf) { ggml_set_output(fused_head); ggml_build_forward_expand(gf, fused_head); }
                    ggml_tensor * diff = ggml_sub(ctx, scores_tc, ref_scores);
                    ggml_tensor * L1 = ggml_sum(ctx, ggml_abs(ctx, diff));
                    cb(L1, "idxkv_scores_diff_L1", -1);
                    if (gf) { ggml_set_output(L1); ggml_build_forward_expand(gf, L1); }
                    ggml_tensor * q_samp = ggml_view_3d(ctx, q3d, std::min<int64_t>(D, (int64_t)8), std::min<int64_t>(Tc, (int64_t)2), std::min<int64_t>(H, (int64_t)2), q3d->nb[1], q3d->nb[2], 0);
                    cb(q_samp, "idxkv_q3d_sample", -1);
                    if (gf) { ggml_set_output(q_samp); ggml_build_forward_expand(gf, q_samp); }
                    ggml_tensor * w_samp = ggml_view_2d(ctx, w_slice, std::min<int64_t>(H, (int64_t)4), std::min<int64_t>(Tc, (int64_t)4), w_slice->nb[1], 0);
                    cb(w_samp, "idxkv_w_slice_sample", -1);
                    if (gf) { ggml_set_output(w_samp); ggml_build_forward_expand(gf, w_samp); }

                    size_t ks_off = 0; ggml_tensor * ks_samp = ggml_view_1d(ctx, ks_head, std::min<int64_t>(kv_end, (int64_t)8), ks_off);
                    cb(ks_samp, "idxkv_ks_head_sample", -1);

                    if (gf) { ggml_set_output(ks_samp); ggml_build_forward_expand(gf, ks_samp); }
                }

                // End wall timer and print (if LLAMA_SPARSE_PROF set)
                auto __t1_wall = std::chrono::high_resolution_clock::now();
                const char * __prof = getenv("LLAMA_SPARSE_PROF");
                if (__prof && *__prof) {
                    double __ms = 1e3 * std::chrono::duration<double>(__t1_wall - __t0_wall).count();
                    static int __cnt_idx_comp = 0; static double __sum_idx_comp = 0.0; __sum_idx_comp += __ms; __cnt_idx_comp++; if (__cnt_idx_comp % 50 == 0) { fprintf(stderr, "[PROFILE] IDX_TILE compute avg_ms=%.3f over 50 tiles (last t0=%lld Tc=%lld kv_end=%lld fused=%d)\n", (float)(__sum_idx_comp/50.0), (long long)t0, (long long)Tc, (long long)kv_end, (int)use_fused); __sum_idx_comp = 0.0; }
                }

            } else {
                scores_tc = idx_compute_scores_tile(ctx, q3d, k_indexer_f16, weights, k_scale_2d, D, H, Tc, kv_end, t0, use_fp16);
            }
        }

        // Safe K-scale proxy application after head reduction: skip if fused kernel already applied it
        {
            const char *env_fused2 = getenv("LLAMA_SPARSE_INDEXER_FUSED");
            bool use_fused2 = (env_fused2 && atoi(env_fused2) != 0);
            if (dbg && t0 == 0) {
                ggml_tensor * pre_head = ggml_view_2d(ctx, scores_tc, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), scores_tc->nb[1], 0);
                cb(pre_head, "idxkv_scores_pre_kScale_head", -1);
                printf("[idxkv] t0=%lld kv_end=%lld fused=%d\n",
                        (long long)t0, (long long)kv_end, (int)use_fused2);
                fflush(stdout);
            }
            if (!use_fused2) {
                ggml_tensor * k_scale_head = ggml_view_2d(ctx, k_scale_2d, kv_end, 1, k_scale_2d->nb[1], 0);
                ggml_tensor * k_scale_bcast = ggml_repeat(ctx, k_scale_head, scores_tc); // [kv_end, Tc]
                scores_tc = ggml_mul(ctx, scores_tc, k_scale_bcast);
                if (dbg && t0 == 0) {
                    ggml_tensor * post_head = ggml_view_2d(ctx, scores_tc, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), scores_tc->nb[1], 0);
                    cb(post_head, "idxkv_scores_post_kScale_head", -1);
                }
            }
        }

        // Debug-only summaries
        if (dbg) {
            ggml_tensor * idxkv_scores_sum = ggml_sum(ctx, scores_tc);
            ggml_tensor * idxkv_scores_ssq = ggml_sum(ctx, ggml_sqr(ctx, scores_tc));
            if (t0 == 0) {
                ggml_tensor * idxkv_scores_post_abs_sum = ggml_sum(ctx, ggml_abs(ctx, scores_tc));
                cb(idxkv_scores_post_abs_sum, "idxkv_scores_post_abs_sum", -1);
                if (gf) {
                    ggml_set_output(idxkv_scores_post_abs_sum);
                    ggml_build_forward_expand(gf, idxkv_scores_post_abs_sum);
                }
            }
            cb(idxkv_scores_sum,  "idxkv_scores_sum",  -1);
            cb(idxkv_scores_ssq,  "idxkv_scores_ssq",  -1);
        }

        scores_tc = ggml_cont(ctx, scores_tc);

        // mask tile if available
        if (mask_full) {
            ggml_tensor * mask_tc = ggml_view_2d(ctx, mask_full, kv_end, Tc, mask_full->nb[1], t0 * mask_full->nb[1]);
            mask_tc = ggml_cont(ctx, mask_tc);
            if (mask_tc->type != scores_tc->type) {
                mask_tc = ggml_cast(ctx, mask_tc, scores_tc->type);
                mask_tc = ggml_cont(ctx, mask_tc);
            }
            if (dbg) cb(mask_tc,  "idxkv_mask_tc",  -1);

            // Ensure both operands have row-contiguous layout for safe broadcast add
            GGML_ASSERT(scores_tc->nb[0] == (size_t) ggml_type_size(scores_tc->type));
            GGML_ASSERT(mask_tc->nb[0]   == (size_t) ggml_type_size(mask_tc->type));
            if (dbg && t0 == 0) {
                printf("[TOPK-INDEXER-DBG] (INDEXER) add mask: scores_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d | mask_tc ne=[%" PRId64 ",%" PRId64 "] nb=[%zu,%zu] type=%d\n",
                       scores_tc->ne[0], scores_tc->ne[1], scores_tc->nb[0], scores_tc->nb[1], (int) scores_tc->type,
                       mask_tc->ne[0], mask_tc->ne[1], mask_tc->nb[0], mask_tc->nb[1], (int) mask_tc->type);
                fflush(stdout);
            }

            if (dbg && t0 == 0) {
                ggml_tensor * mask_head = ggml_view_2d(ctx, mask_tc, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), mask_tc->nb[1], 0);
                cb(mask_head, "idxkv_mask_tc_head", -1);
            }
            scores_tc = ggml_add(ctx, scores_tc, mask_tc);
            // Clamp after mask to avoid inf in diagnostics and stabilize top-k
            scores_tc = ggml_clamp(ctx, scores_tc, -1e30f, 1e30f);

            if (dbg && t0 == 0) {
                ggml_tensor * post_mask_head = ggml_view_2d(ctx, scores_tc, std::min<int64_t>(kv_end, (int64_t)8), std::min<int64_t>(Tc, (int64_t)4), scores_tc->nb[1], 0);
                cb(post_mask_head, "idxkv_scores_post_mask_head", -1);
            }

            if (t0 == 0) {
                ggml_tensor * idxkv_scores_post_mask_abs_sum = ggml_sum(ctx, ggml_abs(ctx, scores_tc));
                cb(idxkv_scores_post_mask_abs_sum, "idxkv_scores_post_mask_abs_sum", -1);
                if (gf) {
                    ggml_set_output(idxkv_scores_post_mask_abs_sum);
                    ggml_build_forward_expand(gf, idxkv_scores_post_mask_abs_sum);
                }
            }
        }

        // top-k for this tile -> [k, Tc]
        // Ensure top-k runs on CPU to avoid CUDA backend returning invalid indices for generic shapes
        ggml_tensor * scores_for_topk = scores_tc;
        // Keep on device by default; only copy to host in debug mode
        if (dbg && scores_tc->buffer && !ggml_backend_buffer_is_host(scores_tc->buffer)) {
            ggml_tensor * host_scores = ggml_dup_tensor(ctx, scores_tc);
            ggml_set_name(host_scores, "idxkv_scores_tc_host");
            host_scores = ggml_cpy(ctx, scores_tc, host_scores);
            scores_for_topk = host_scores;
        }
        if (dbg && sched) {
            ggml_backend_t chosen_in = ggml_backend_sched_get_tensor_backend(sched, scores_for_topk);
            const char * in_name = chosen_in ? ggml_backend_name(chosen_in) : NULL;
            printf("[TOPK] chosen backend for scores_for_topk: %s (non-null=%d)\n", in_name ? in_name : "null", chosen_in ? 1 : 0);
            fflush(stdout);
        }
        // Log scores_for_topk (tile 0 only) and reductions to materialize in eval-callback
        if (dbg && t0 == 0) {
            cb(scores_for_topk, "idxkv_scores_for_topk", -1);
            ggml_tensor * sft_sum   = ggml_sum(ctx, scores_for_topk);
            ggml_tensor * sft_sumsq = ggml_sum(ctx, ggml_sqr(ctx, scores_for_topk));
            cb(sft_sum,   "idxkv_scores_for_topk_sum",   -1);
            cb(sft_sumsq, "idxkv_scores_for_topk_sumsq", -1);
            if (gf) {
                ggml_set_output(scores_for_topk);
        // profile top-k selection time per tile

                ggml_set_output(sft_sum);
                ggml_set_output(sft_sumsq);
                ggml_build_forward_expand(gf, scores_for_topk);
                ggml_build_forward_expand(gf, sft_sum);
                ggml_build_forward_expand(gf, sft_sumsq);
            }
        }
        // Ensure contiguous for CUDA op only if needed, then clamp infinities
        ggml_tensor * scores_pre = ggml_is_contiguous(scores_for_topk) ? scores_for_topk : ggml_cont(ctx, scores_for_topk);
        ggml_tensor * scores_clamped = ggml_clamp(ctx, scores_pre, -1e30f, 1e30f);
        // Compute top-k indices via CUDA radix selection
        const int64_t k_tile = std::min<int64_t>(k, scores_clamped->ne[0]);
        ggml_tensor * topk_tc = nullptr;
        if (have_windows && win_ends) {
            // slice starts/ends for this tile [t0, t0+Tc)
            ggml_tensor * starts_tile = nullptr;
            ggml_tensor * ends_tile   = nullptr;
            if (win_starts) {
                size_t off_s = (size_t)t0 * win_starts->nb[0];
                starts_tile = ggml_view_1d(ctx, win_starts, Tc, off_s);
                starts_tile = ggml_cont(ctx, starts_tile);
            }
            if (win_ends) {
                size_t off_e = (size_t)t0 * win_ends->nb[0];
                ends_tile   = ggml_view_1d(ctx, win_ends,   Tc, off_e);
                ends_tile   = ggml_cont(ctx, ends_tile);
            }
            topk_tc = ggml_sparse_topk_radix_ex(ctx, scores_clamped, (int)k_tile, starts_tile, ends_tile);
        } else {
            topk_tc = ggml_sparse_topk_radix(ctx, scores_clamped, (int)k_tile);
        }
        if (dbg && t0 == 0) {
            cb(topk_tc, "idxkv_topk_radix", -1);
            int64_t kk = std::min<int64_t>(k_tile, (int64_t)16);
            int64_t tt = std::min<int64_t>(Tc, (int64_t)4);
            ggml_tensor * topk_head = ggml_view_2d(ctx, topk_tc, kk, tt, topk_tc->nb[1], 0);
            cb(topk_head, "idxkv_topk_indices_head", -1);
            if (gf) { ggml_set_output(topk_head); ggml_build_forward_expand(gf, topk_head); }
        }
        // If we applied a KV window, adjust indices by the start offset
        if (have_windows && win_ends) {
            // current window start kv_s was 0 in this version; if non-zero in future, add kv_s here
            // For now, no offset is needed since kv_s==0 above. Placeholder for future starts support.
        }
        result = result ? ggml_concat(ctx, result, topk_tc, 1) : topk_tc;
    }

    cb(result, "idxkv_topk_indices_k_T", -1);
    if (gf && result) {
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);
    }
    // Also provide a float32 view for eval-callback visibility on platforms that skip integer dumps
    if (dbg && result) {
        ggml_tensor * result_f32 = ggml_cast(ctx, result, GGML_TYPE_F32);
        cb(result_f32, "idxkv_topk_indices_k_T_f32", -1);
        if (gf) {
            ggml_set_output(result_f32);
            ggml_build_forward_expand(gf, result_f32);
        }
    }
    if (dbg && result) {
        printf("SPARSE TOPK KV-AWARE (INDEXER): result topk_indices dims=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] type=%d\n",
               result->ne[0], result->ne[1], result->ne[2], result->ne[3], (int)result->type);
        fflush(stdout);
    }
    // Keep indices on device by default to avoid host syncs during get_rows
    return result;
}

ggml_tensor * llama::sparse_attn_topk::topk_radix_indices(
    ggml_context * ctx,
    ggml_tensor * scores, // [N, T]
    int64_t k) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32);
    ggml_tensor * args[1] = { scores };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_I32,
        /*ne0*/ k,
        /*ne1*/ scores->ne[1],
        /*ne2*/ 1,
        /*ne3*/ 1,
        args,
        /*n_args*/ 1,
        /*fun*/ radix_topk_custom,
        /*n_tasks*/ GGML_N_TASKS_MAX,
        /*userdata*/ nullptr);
}

ggml_tensor * llama::sparse_attn_topk::derive_kv_windows(ggml_context * ctx, ggml_tensor * kq_mask, int64_t T, int64_t N_kv, ggml_tensor ** out_starts, ggml_tensor ** out_ends) {
    *out_starts = nullptr; *out_ends = nullptr;
    if (!kq_mask) return nullptr;
    // Expect kq_mask: [N_kv, PAD(T)] row-contiguous on rows
    if (kq_mask->ne[0] != N_kv || kq_mask->ne[1] < T) return nullptr;
    // Compute starts=0 and ends per token as last unmasked+1
    ggml_tensor * starts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * ends   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    // Copy mask to host buffer and compute ends
    std::vector<float> mask_host((size_t)N_kv * T);
    ggml_backend_tensor_get(kq_mask, mask_host.data(), 0, ggml_nbytes(kq_mask));
    // Fill starts with zeros
    for (int64_t t = 0; t < T; ++t) { ((int32_t*)starts->data)[t] = 0; }
    // ends per column
    for (int64_t t = 0; t < T; ++t) {
        const float * col = (const float *)((const char*)kq_mask->data + (size_t)t*kq_mask->nb[1]);
        int e = find_last_unmasked(col, (int)N_kv, kq_mask->nb[0]);
        ((int32_t*)ends->data)[t] = e;
    }
    *out_starts = starts; *out_ends = ends;
    return starts;
}
} // namespace llama
