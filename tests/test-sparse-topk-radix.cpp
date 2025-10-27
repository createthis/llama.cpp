#include "../src/llama-sparse-topk.h"
#include "../src/llama-impl.h"

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include <ggml-alloc.h>
#include <ggml-cpp.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>

// Reference top-k (descending) using std::partial_sort on indices
static void ref_topk(const float * row, int64_t N, int64_t k, std::vector<int32_t> & out) {
    out.resize(k);
    std::vector<int32_t> idx(N);
    for (int64_t i = 0; i < N; ++i) idx[i] = (int32_t)i;
    auto cmp = [&](int32_t a, int32_t b){ return row[a] > row[b]; };
    if (k < N) {
        std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), cmp);
        std::copy(idx.begin(), idx.begin()+k, out.begin());
    } else {
        std::sort(idx.begin(), idx.end(), cmp);
        std::copy(idx.begin(), idx.begin()+k, out.begin());
    }
}

int main() {
    // Build a small random problem: N_kv=4096, T=8
    const int64_t N_kv = 4096;
    const int64_t T = 8;
    const int64_t H = 4;
    const int64_t D = 64;
    const int64_t k = 64;

    // Initialize GGML context
    ggml_init_params p{};
    p.mem_size = 256ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc = false;
    ggml_context * ctx = ggml_init(p);
    assert(ctx);
    // Create tensors: q_indexer [D,H,T], k_indexer [D,N_kv], weights [H,T]
    ggml_tensor * q_indexer = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
    ggml_tensor * k_indexer = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N_kv);
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);


    // Fill with random values and write into tensors directly (no backend set/get)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill_tensor = [&](ggml_tensor * t, std::vector<float> & host) {
        host.resize(ggml_nelements(t));
        for (auto & v : host) v = dist(rng);
        // write respecting strides
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            size_t off = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
            size_t lin = i0 + t->ne[0]*(i1 + t->ne[1]*(i2 + t->ne[2]*i3));
            *(float*)((char*)t->data + off) = host[lin];
        }
    };
    std::vector<float> Q, Khost, Whost;
    fill_tensor(q_indexer, Q);
    fill_tensor(k_indexer, Khost);
    fill_tensor(weights,   Whost);

    // Build top-k indices via our new radix top-k on precomputed scores
    // First, form the scores matrix for all tokens T: [N_kv, T]
    ggml_tensor * q_perm = ggml_permute(ctx, q_indexer, 0, 2, 1, 3); // [D, T, H]
    ggml_tensor * q_cont = ggml_cont(ctx, q_perm);
    ggml_tensor * Q2d    = ggml_reshape_2d(ctx, q_cont, D, T*H);     // [D, T*H]
    ggml_tensor * logits_all = ggml_mul_mat(ctx, k_indexer, Q2d);    // [N_kv, T*H]
    ggml_tensor * logits_resh= ggml_reshape_3d(ctx, logits_all, N_kv, H, T);
    ggml_tensor * logits_act = ggml_relu(ctx, logits_resh);
    ggml_tensor * w = ggml_reshape_3d(ctx, weights, 1, H, T);
    ggml_tensor * w_bc = ggml_repeat(ctx, w, logits_act);
    ggml_tensor * contrib = ggml_mul(ctx, logits_act, w_bc);
    ggml_tensor * contrib_perm = ggml_permute(ctx, contrib, 1, 0, 2, 3);
    contrib_perm = ggml_cont(ctx, contrib_perm);
    ggml_tensor * sum_h = ggml_sum_rows(ctx, contrib_perm);          // [1, N_kv, T]
    ggml_tensor * scores_2d = ggml_reshape_2d(ctx, sum_h, N_kv, T);  // [N_kv, T]
    // Apply k_scale proxy
    ggml_tensor * k_sqr = ggml_sqr(ctx, k_indexer);
    ggml_tensor * k_sum = ggml_sum_rows(ctx, k_sqr);
    ggml_tensor * k_mean= ggml_scale(ctx, k_sum, 1.0f/float(D));
    ggml_tensor * k_scale_vec = ggml_sqrt(ctx, k_mean);              // [1, N_kv]
    ggml_tensor * k_scale_2d = ggml_transpose(ctx, k_scale_vec);     // [N_kv, 1]
    k_scale_2d = ggml_cont(ctx, k_scale_2d);
    ggml_tensor * k_scale_b = ggml_repeat(ctx, k_scale_2d, scores_2d);
    ggml_tensor * scores_scaled = ggml_mul(ctx, scores_2d, k_scale_b);

    ggml_tensor * scores_cont = ggml_cont(ctx, scores_scaled);
    ggml_tensor * idx_radix = llama::sparse_attn_topk::topk_radix_indices(ctx, scores_cont, k);
    ggml_tensor * idx_ref   = ggml_top_k(ctx, scores_cont, k);
    // Build and execute graph to materialize both
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx_radix);
    ggml_build_forward_expand(gf, idx_ref);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // Copy both tensors to host
    std::vector<int32_t> RAD(ggml_nelements(idx_radix));
    std::vector<int32_t> REF(ggml_nelements(idx_ref));
    for (int64_t i3 = 0; i3 < idx_radix->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < idx_radix->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < idx_radix->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < idx_radix->ne[0]; ++i0) {
        size_t off = i3*idx_radix->nb[3] + i2*idx_radix->nb[2] + i1*idx_radix->nb[1] + i0*idx_radix->nb[0];
        size_t lin = i0 + idx_radix->ne[0]*(i1 + idx_radix->ne[1]*(i2 + idx_radix->ne[2]*i3));
        RAD[lin] = *(int32_t*)((char*)idx_radix->data + off);
    }
    for (int64_t i3 = 0; i3 < idx_ref->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < idx_ref->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < idx_ref->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < idx_ref->ne[0]; ++i0) {
        size_t off = i3*idx_ref->nb[3] + i2*idx_ref->nb[2] + i1*idx_ref->nb[1] + i0*idx_ref->nb[0];
        size_t lin = i0 + idx_ref->ne[0]*(i1 + idx_ref->ne[1]*(i2 + idx_ref->ne[2]*i3));
        REF[lin] = *(int32_t*)((char*)idx_ref->data + off);
    }
    if (RAD.size() != REF.size()) {
        printf("Size mismatch: radix=%zu ref=%zu\n", RAD.size(), REF.size());
        return 1;
    }

    // Validate correctness by threshold criterion per column (order and specific indices may differ under ties)
    const int64_t KK = idx_radix->ne[0];
    const int64_t Tcol = idx_radix->ne[1];
    const int64_t N = scores_cont->ne[0];
    const size_t nb0 = scores_cont->nb[0];
    const size_t nb1 = scores_cont->nb[1];
    for (int64_t t = 0; t < Tcol; ++t) {
        // collect column values
        std::vector<float> col(N);
        for (int64_t i = 0; i < N; ++i) {
            col[i] = *(float*)((char*)scores_cont->data + i*nb0 + t*nb1);
        }
        // compute threshold = K-th largest value
        std::vector<float> sorted = col;
        std::nth_element(sorted.begin(), sorted.begin() + (KK-1), sorted.end(), std::greater<float>());
        float thresh = sorted[KK-1];
        // gather radix indices for this column and verify
        std::vector<int32_t> rad_k(KK);
        for (int64_t i = 0; i < KK; ++i) {
            size_t lin = i + KK * t;
            rad_k[i] = RAD[lin];
        }
        // check bounds and uniqueness
        std::vector<char> seen(N, 0);
        for (int64_t i = 0; i < KK; ++i) {
            int idx = rad_k[i];
            if (idx < 0 || idx >= N) {
                printf("Column %lld: index out of bounds: %d (N=%lld)\n", (long long)t, idx, (long long)N);
                return 1;
            }
            if (seen[idx]) {
                printf("Column %lld: duplicate index in top-k: %d\n", (long long)t, idx);
                return 1;
            }
            seen[idx] = 1;
            float v = col[idx];
            if (!(v + 0.0f >= thresh)) { // allow equality under ties
                printf("Column %lld: value below threshold: idx=%d v=%.6f thresh=%.6f\n", (long long)t, idx, v, thresh);
                return 1;
            }
        }
    }

    printf("radix top-k unit test: PASS\n");
    return 0;
}
