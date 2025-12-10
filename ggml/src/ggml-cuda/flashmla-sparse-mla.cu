#include "common.cuh"
#include "../../include/ggml-cuda-indexer.h"

// Experimental FlashMLA-backed sparse MLA decode entry point.
// For now, this path reuses ggml's ggml_cuda_sparse_mla_decode_device
// for DeepSeek V3.2-shaped decode (Dq=576, Dv=512, Hkv=1) so that the
// FlashMLA-gated path is non-stub and numerically correct. Later we
// can replace this body with a true FlashMLA mapping.

extern "C" void ggml_cuda_sparse_mla_decode_device(
    ggml_backend_cuda_context & ctx,
    const float * q,
    const float * k,
    const float * v,
    const int32_t * topk,
    int D, int Hq, int Hkv, int Dv,
    int N, int Ksel,
    float kq_scale, float softcap,
    float * out);

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
    (void)ctx; (void)q; (void)k; (void)v; (void)topk;
    (void)Dq; (void)Hq; (void)Hkv; (void)Dv; (void)Nkv; (void)K;
    (void)kq_scale; (void)softcap; (void)out;
    return;
#else
    // For now: delegate to the existing sparse MLA decode kernel so the
    // FlashMLA path is non-stub and numerically matches the GGML path.
    ggml_cuda_sparse_mla_decode_device(
        ctx,
        q,
        k,
        v,
        topk,
        Dq,
        Hq,
        Hkv,
        Dv,
        Nkv,
        K,
        kq_scale,
        softcap,
        out);
#endif
}
