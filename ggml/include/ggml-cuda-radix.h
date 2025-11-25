#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Compute top-k indices per column using a CUDA radix-style selection.
// scores is a row-major 2D array with shape [N, T]: element(i,t) at scores[i + N*t].
// Writes indices into idx (shape [k, T], same storage rule: idx[i + k*t]).
void ggml_cuda_topk_radix_indices_host(const float * scores, int N, int T, int k, int * idx);

// Build per-column histogram on the top byte of float->key mapping.
// scores: [N, T] row-major. Outputs:
//  - gt_counts: size 256*T, gt_counts[b + 256*t] = sum_{bb>b} counts[bb]
//  - thr_bins:  size T (currently placeholder; can be 0)
void ggml_cuda_topk_histogram_host(const float * scores, int N, int T,
                                   unsigned int * gt_counts, unsigned int * thr_bins);

// Launch equal-bin selection kernel only, given precomputed histogram greater-counts per column
// scores: [N, T] row-major
// gt_counts: [256, T] greater-counts per bin
// idx: [k, T] output indices (row-major leading dimension k)
void ggml_cuda_topk_select_host(const float * scores, int N, int T, int k,
                                const unsigned int * gt_counts, int * idx);

#ifdef __cplusplus
}
#endif
