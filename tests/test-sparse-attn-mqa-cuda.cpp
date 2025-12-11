#include "../src/llama-sparse-indexer.h"
#include "../src/llama-model.h"
#include "../src/llama-impl.h"
#include "../src/llama-sparse-topk.h"
#include "../src/llama-sparse-mla-fwd.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <cstdlib>

using namespace llama;

int main() {
    printf("=== Sparse MLA kv-aware MQA fused/non-fused repro ===\n");
    // Build graph (no_alloc=true), then schedule and compute with CUDA+CPU
    ggml_init_params p{};
    p.mem_size   = 64ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = true;

    ggml_context * ctx = ggml_init(p);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // Shapes to trigger fused decode (T==1) and Hq != Hkv
    const int64_t Dq = 576;     // per-head embed for Q/K
    const int64_t Hq = 128;     // query heads
    const int64_t Hkv = 1;      // KV heads (MQA)
    const int64_t Dv = 576;     // per-head embed for V
    const int64_t N_kv = 1024;  // KV cache length
    const int64_t T = 1;        // decode step to enable fused path
    const int64_t top_k = 64;

    // q_cur [Dq, Hq, T], k_cache [Dq, Hkv, N_kv], v_cache [Dv, Hkv, N_kv]
    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dq, Hq, T);
    ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dq, Hkv, N_kv);
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dv, Hkv, N_kv);

    // topk indices [top_k, T]
    ggml_tensor * topk_idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, T);

    // Build graph for kv-aware sparse attention
    auto cb = [](ggml_tensor * t, const char * name, int il) {
        (void)il; if (!t) { printf("CB: %s=null\n", name); return; }
        printf("CB: %s: shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n", name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    };

    ggml_tensor * out = llama::sparse_mla_fwd::apply_sparse_attention_kvaware(
        ctx, q_cur, k_cache, v_cache, topk_idx,
        /*n_tokens=*/T, /*top_k=*/top_k, /*kq_scale=*/1.0f,
        /*kq_mask=*/nullptr, /*attn_softcap=*/0.0f, /*kv_dsmla_blob=*/nullptr, cb);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t cpu_dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cuda = cuda_dev ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cuda || !cpu) { printf("Missing backend(s)\n"); return 0; }

    ggml_backend_t backs[2] = { cuda, cpu };
    int nb = 2;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, nb, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // Assign inputs to CUDA backend to avoid mixed-backend allocation issues
    ggml_backend_sched_set_tensor_backend(sched, q_cur,  cuda);
    ggml_backend_sched_set_tensor_backend(sched, k_cache, cuda);
    ggml_backend_sched_set_tensor_backend(sched, v_cache, cuda);
    ggml_backend_sched_set_tensor_backend(sched, topk_idx, cuda);
    // Allocate graph (no explicit reserve)
    ggml_backend_sched_alloc_graph(sched, gf);

    // Initialize host buffers
    std::vector<float> hQ((size_t)Dq*Hq*T, 0.0f);
    std::vector<float> hK((size_t)Dq*Hkv*N_kv, 0.0f);
    std::vector<float> hV((size_t)Dv*Hkv*N_kv, 0.0f);
    std::vector<int32_t> hTopK((size_t)top_k*T);
    for (int i = 0; i < top_k; ++i) hTopK[i] = i % (int)N_kv; // simple ascending indices

    // Upload to backend
    ggml_backend_tensor_set(q_cur,  hQ.data(), 0, ggml_nbytes(q_cur));
    ggml_backend_tensor_set(k_cache,hK.data(), 0, ggml_nbytes(k_cache));
    ggml_backend_tensor_set(v_cache,hV.data(), 0, ggml_nbytes(v_cache));
    ggml_backend_tensor_set(topk_idx, hTopK.data(), 0, ggml_nbytes(topk_idx));

    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (st != GGML_STATUS_SUCCESS) { printf("compute failed (%d)\n", (int)st); return 1; }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    printf("=== Repro finished ===\n");
    return 0;
}
