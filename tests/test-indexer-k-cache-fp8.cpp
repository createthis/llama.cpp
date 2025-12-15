#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml-cuda-indexer.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <limits>

// This test exercises the low-level FP8 indexer K cache quantization helper
// ggml_cuda_indexer_k_cache_fp8_quantize. It mirrors vLLM's
// indexer_k_quant_and_cache/cp_gather_indexer_k_quant_cache layout for a
// single-head DeepSeek V3.2 indexer cache.

int main() {
#ifndef GGML_USE_CUDA
    std::printf("CUDA not enabled; skipping FP8 indexer K cache test\n");
    return 0;
#else
    // We construct a tiny GGML graph that:
    //   1) Allocates a device tensor k_in [num_tokens, head_dim]
    //   2) Allocates a device slot_map [num_tokens]
    //   3) Allocates a device kv_cache8 [cache_stride, kv_size, n_stream]
    //   4) Invokes ggml_indexer_k_cache_fp8 as a custom op
    //   5) Copies kv_cache8 back to host and checks basic invariants.

    constexpr int num_tokens = 4;
    constexpr int head_dim   = 16;   // small but > quant block size / 2 for coverage
    constexpr int quant_bs   = 8;    // quantization block size in bytes
    constexpr int cache_block_size = 8; // kv_size per stream
    constexpr int n_stream   = 1;

    // Layout: [cache_stride, kv_size, n_stream]
    const int scales_per_row = (head_dim + quant_bs - 1) / quant_bs;
    const int cache_stride   = head_dim + scales_per_row * 4; // bytes per token
    const int kv_size        = cache_block_size;

    std::printf("[fp8-k-cache] num_tokens=%d head_dim=%d quant_bs=%d cache_block_size=%d cache_stride=%d\n",
                num_tokens, head_dim, quant_bs, cache_block_size, cache_stride);

    // Create GGML context with no_alloc=true so all tensor data lives in backend buffers
    ggml_init_params ip{};
    ip.mem_size   = 16ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::printf("ggml_init failed; skipping\n");
        return 1;
    }

    ggml_tensor * k_in     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_tokens, head_dim);
    ggml_tensor * slot_map = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, num_tokens);
    ggml_tensor * kv_cache8= ggml_new_tensor_3d(ctx, GGML_TYPE_I8, cache_stride, kv_size, n_stream);

    // Host-side initialization
    std::vector<float>      hK(num_tokens * head_dim);
    std::vector<long long>  hSlot(num_tokens);
    for (int t = 0; t < num_tokens; ++t) {
        hSlot[t] = t; // identity mapping into first num_tokens slots of stream 0
        for (int d = 0; d < head_dim; ++d) {
            // Deterministic pattern with both positive and negative values
            hK[t*head_dim + d] = (float) ((t + 1) * (d - head_dim/2));
        }
    }

    // Build graph with single ggml_indexer_k_cache_fp8 op
    ggml_tensor * op = ggml_indexer_k_cache_fp8(ctx, k_in, slot_map, kv_cache8,
                                                quant_bs, cache_block_size, cache_stride);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, op);

    // Initialize a CUDA backend and allocate a single device buffer for all tensors
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("CUDA backend init failed; skipping FP8 indexer K cache test\n");
        ggml_free(ctx);
        return 0;
    }

    const size_t align = ggml_backend_get_alignment(backend);
    const size_t size_k   = GGML_PAD(ggml_nbytes(k_in),      align);
    const size_t size_map = GGML_PAD(ggml_nbytes(slot_map),  align);
    const size_t size_dst = GGML_PAD(ggml_nbytes(kv_cache8), align);
    const size_t size_op  = GGML_PAD(ggml_nbytes(op),        align);

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_buffer(backend, size_k + size_map + size_dst + size_op);
    void * base_void = ggml_backend_buffer_get_base(buf);
    if (!buf || !base_void) {
        std::printf("backend buffer alloc failed\n");
        if (buf) ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }
    char * base = (char *) base_void;

    ggml_backend_tensor_alloc(buf, k_in,      base);
    ggml_backend_tensor_alloc(buf, slot_map,  base + size_k);
    ggml_backend_tensor_alloc(buf, kv_cache8, base + size_k + size_map);
    ggml_backend_tensor_alloc(buf, op,        base + size_k + size_map + size_dst);

    // Upload host data into device-resident tensors
    ggml_backend_tensor_set(k_in,     hK.data(),    0, hK.size()   * sizeof(float));
    ggml_backend_tensor_set(slot_map, hSlot.data(), 0, hSlot.size()* sizeof(long long));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("FP8 indexer K cache quantize test: BACKEND COMPUTE FAIL (%d)\n", (int) st);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    // Copy kv_cache back to host and inspect
    std::vector<unsigned char> hCache(ggml_nbytes(kv_cache8));
    ggml_backend_tensor_get(kv_cache8, hCache.data(), 0, hCache.size());

    // Simple checks:
    //  - For each written token slot, scales for its quant blocks are > 0
    //  - FP8 codes are non-zero when the corresponding K values are non-zero
    int n_bad_scale = 0;
    int n_bad_code  = 0;
    for (int t = 0; t < num_tokens; ++t) {
        long long slot = hSlot[t];
        if (slot < 0 || slot >= cache_block_size) continue;
        const int64_t block_idx    = 0;
        const int64_t block_offset = slot;
        const int64_t block_base   = block_idx * (int64_t) cache_block_size * cache_stride;
        const int64_t vals_base    = block_base + block_offset * (int64_t) head_dim;
        const int64_t scales_base  = block_base + (int64_t) cache_block_size * head_dim;
        // Check quant blocks along head_dim
        for (int b = 0; b < scales_per_row; ++b) {
            int64_t scale_off = scales_base + (block_offset * head_dim + b * quant_bs) * 4 / quant_bs;
            float sf;
            std::memcpy(&sf, &hCache[scale_off], sizeof(float));
            if (!(sf > 0.0f) || !std::isfinite(sf)) {
                ++n_bad_scale;
            }
        }
        // Check a few FP8 codes vs source magnitude
        for (int d = 0; d < head_dim; ++d) {
            float v = hK[t*head_dim + d];
            unsigned char code = hCache[vals_base + d];
            if (v == 0.0f && code != 0) {
                ++n_bad_code; // zero should quantize to zero
            }
            if (std::fabs(v) > 1e-3f && code == 0) {
                // non-trivial value should not vanish entirely
                ++n_bad_code;
            }
        }
    }

    std::printf("[fp8-k-cache] n_bad_scale=%d n_bad_code=%d\n", n_bad_scale, n_bad_code);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (n_bad_scale != 0 || n_bad_code != 0) {
        std::printf("FP8 indexer K cache quantize test: FAIL\n");
        return 1;
    }
    std::printf("FP8 indexer K cache quantize test: PASS\n");
    return 0;
#endif
}
