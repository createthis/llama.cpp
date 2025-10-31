#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Compute top-k indices per column using a CUDA radix-style selection.
// scores is a row-major 2D array with shape [N, T]: element(i,t) at scores[i + N*t].
// Writes indices into idx (shape [k, T], same storage rule: idx[i + k*t]).
void ggml_cuda_topk_radix_indices_host(const float * scores, int N, int T, int k, int * idx);

#ifdef __cplusplus
}
#endif
