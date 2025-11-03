#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// Fused lightning-indexer logits kernel (scaffold): host wrapper copies inputs to device and back
// Inputs are row-major contiguous buffers:
//  - Q: [D, Tc*H] as row-major (leading dim D)
//  - K: [D, kv_end] row-major
//  - W: [H, Tc] row-major
//  - k_scale: [kv_end]
// Output:
//  - out: [kv_end, Tc] row-major
void ggml_cuda_indexer_logits_fused_host(const float * Q,
                                         const float * K,
                                         const float * W,
                                         const float * k_scale,
                                         int D, int H, int Tc, int kv_end,
                                         float * out);
#ifdef __cplusplus
}
#endif
