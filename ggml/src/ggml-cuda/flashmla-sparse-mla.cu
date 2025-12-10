#include "common.cuh"
#include "../../include/ggml-cuda-indexer.h"

// Thin wrapper that maps ggml's SPARSE_MLA_DECODE single-token decode
// into FlashMLA's sparse FP8 MLA decode kernel. This is intentionally
// narrow: it only handles the DeepSeek V3.2 MLA decode pattern
// (Hkv == 1, T == 1) and is gated by LLAMA_SPARSE_MLA_FLASHMLA.
//
// Layout assumptions:
//   q:   [Dq, Hq]        (column-major, leading dimension Dq)
//   k:   [Dk, Hkv, Nkv]  (column-major in ggml, we treat as row-major ND)
//   v:   [Dv, Hkv, Nkv]
//   topk:[K] indices into [0, Nkv)
//
// For now we implement a simple dense slice: we treat the entire KV
// range [0, Nkv) as a single sequence of length Nkv, and we pass
// indices as-is to FlashMLA (it will ignore -1 / out-of-range).

extern "C" void ggml_cuda_sparse_mla_decode_flashmla_sm100(
    ggml_backend_cuda_context & ctx,
    const float * q,
    const float * k,
    const float * v,
    const int32_t * topk,
    int Dq,
    int Hq,
    int Hkv,
    int Dv,
    int Nkv,
    int K,
    float kq_scale,
    float softcap,
    float * out) {

#if CUDART_VERSION < 12000
    // FlashMLA requires CUDA 12.0+; fall back to ggml path.
    (void)ctx; (void)q; (void)k; (void)v; (void)topk;
    (void)Dq; (void)Hq; (void)Hkv; (void)Dv; (void)Nkv; (void)K;
    (void)kq_scale; (void)softcap; (void)out;
    return;
#else
    cudaStream_t stream = ctx.stream();

    // Very narrow guard: only Hkv == 1 and Dq == 576, Dv == 512 are
    // expected for DeepSeek V3.2 MLA.
    if (Hkv != 1 || Dq <= 0 || Dv <= 0 || Nkv <= 0 || Hq <= 0 || K <= 0) {
        return;
    }

    // FlashMLA expects BF16 inputs and FP8 KV cache. Our current GGML
    // implementation materializes K/V in FP32; for a first wiring pass
    // we simply bail out and let the existing kernel run.
    //
    // Future work: plumb the DeepSeek V3.2 FP8 MLA KV cache into this
    // wrapper and call run_flash_splitkv_mla_fp8_sparse_kernel.
    (void)stream;
    (void)topk;
    (void)kq_scale;
    (void)softcap;
    (void)out;
    (void)k;
    (void)v;
    (void)q;
    (void)Dq; (void)Hq; (void)Dv; (void)Nkv; (void)K;
#endif
}
