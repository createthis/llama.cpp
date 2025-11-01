#pragma once
#include "common.cuh"

// Launch device kernel(s) to compute per-column top-k indices using a radix selection approach.
// scores_d: device pointer to [N, T] with leading dimension ld (scores[i + ld*t])
// idx_d: device pointer to [k, T] row-major (idx[i + k*t])
void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int ld, int k,
                                         int * idx_d);
