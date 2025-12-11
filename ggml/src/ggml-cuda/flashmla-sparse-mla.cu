#include "common.cuh"
#include "../../include/ggml-cuda-indexer.h"

#include "vendors/flashmla/params.h"
#include "vendors/flashmla/smxx/get_mla_metadata.h"
#include "vendors/flashmla/smxx/mla_combine.h"
#include "vendors/flashmla/sm100/decode/sparse_fp8/splitkv_mla.h"

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
    const unsigned char * kv_blob,
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
    // For now, only handle DeepSeek V3.2-shaped decode (Dq=576, Dv=512, Hkv=1)
    // and fall back to the GGML kernel otherwise.
    if (Dq != 576 || Dv != 512 || Hkv != 1) {
        ggml_cuda_sparse_mla_decode_device(ctx, q, k, v, topk,
                                           Dq, Hq, Hkv, Dv, Nkv, K,
                                           kq_scale, softcap, out);
        return;
    }

    // TODO: Implement full FlashMLA SM100 sparse FP8 decode wiring here by
    // constructing BF16 Q, DS-MLA FP8 kcache view, sparse indices, metadata
    // (GetDecodingMetadataParams), and DecodingParams, then calling:
    //   sm100::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    //   run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
    // For now, still delegate to the GGML F32 kernel until the mapping is
    // completed.
    ggml_cuda_sparse_mla_decode_device(ctx, q, k, v, topk,
                                       Dq, Hq, Hkv, Dv, Nkv, K,
                                       kq_scale, softcap, out);
#endif
}
