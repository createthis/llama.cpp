#include "topk-radix.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include "../../include/ggml-cuda-radix.h"
#include <stdlib.h>

#include <vector>
#include <algorithm>

static inline uint16_t host_float_to_half_bits_rtne(float f) {
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t sub = mant >> (1 - exp);
        if (sub & 0x00001000u) sub += 0x00002000u;
        return (uint16_t)(sign | (sub >> 13));
    } else if (exp >= 31) {
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        mant >>= 13; return (uint16_t)(sign | 0x7C00u | mant | (mant == 0));
    } else {
        if (mant & 0x00001000u) { mant += 0x00002000u; if (mant & 0x00800000u) { mant = 0; exp += 1; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); } }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }
}
static inline uint8_t host_convert_to_uint16_bin(float x) {
    uint16_t h = host_float_to_half_bits_rtne(x);
    uint16_t bits = (x < 0.0f) ? (uint16_t)(~h & 0xFFFFu) : (uint16_t)(h | 0x8000u);
    return (uint8_t)(bits >> 8);
}

template<int ID>
__device__ __forceinline__ void named_sync(int count) {
  asm volatile("bar.sync %0, %1;" :: "n"(ID), "r"(count) : "memory");
}

#ifndef SEL_DEBUG
#define SEL_DEBUG 0
#endif
#ifndef SEL_DEBUG_COL
#define SEL_DEBUG_COL 0
#endif

static inline __host__ __device__ int env_threads_or_default(const char * name, int deflt) {
    int v = deflt;
#ifndef __CUDA_ARCH__
    const char * e = getenv(name);
    if (e && *e) {
        int t = atoi(e);
        if (t > 0) v = t;
    }
#endif
    if (v < 128) v = 128;
    if (v > 1024) v = 1024;
    v = (v + 31) & ~31;
    return v;
}

// Key32-based MSB bin for descending order: transform float to lexicographic-descending key and take high byte
static __device__ __forceinline__ uint32_t key32_desc(float x) {
    uint32_t u = __float_as_uint(x);
    return ((int32_t)u < 0) ? ~u : (u ^ 0x80000000u);
}
static __device__ __forceinline__ uint8_t key32_msb_bin_desc(float x) {
    return (uint8_t)(key32_desc(x) >> 24);
}

// Compute K-th threshold bin for top byte of keys of a given column
// Here we implement a block-per-column approach where each block processes N elements.
// We use shared histogram of 256 bins.
static __global__ void k_histogram_topbyte(const float * __restrict__ scores,
                                           int N, int T, int ld,
                                           uint32_t * __restrict__ thr_bins,
                                           uint32_t * __restrict__ gt_counts) {
    int t = blockIdx.x;
    if (t >= T) return;
    // dynamic shared memory: [warp_count*256] per-warp histograms + [256] final hist
    extern __shared__ uint32_t shmem[];
    const int warp_count = blockDim.x >> 5;
    uint32_t * hist_warp = shmem;
    uint32_t * hist      = shmem + warp_count * 256;
    for (int i = threadIdx.x; i < 256 * (warp_count + 1); i += blockDim.x) shmem[i] = 0u;
    __syncthreads();

    const float * col = scores + (size_t)ld * t;
    // accumulate per-warp histograms
    uint32_t * my_hist = hist_warp + ((threadIdx.x >> 5) * 256);
    size_t __addr = (size_t)col;
    if ( ( (__addr & 0xFu) == 0u) ) {
        int i4 = threadIdx.x * 4;
        for (; i4 + 3 < N; i4 += blockDim.x * 4) {
            float4 v = *((const float4 *)(col + i4));
            uint8_t b0_0 = key32_msb_bin_desc(v.x);
            uint8_t b0_1 = key32_msb_bin_desc(v.y);
            uint8_t b0_2 = key32_msb_bin_desc(v.z);
            uint8_t b0_3 = key32_msb_bin_desc(v.w);
            if (b0_0==b0_1 && b0_1==b0_2 && b0_2==b0_3) {
                atomicAdd(&my_hist[b0_0], 4u);
            } else if (b0_0==b0_1 && b0_2==b0_3) {
                atomicAdd(&my_hist[b0_0], 2u);
                atomicAdd(&my_hist[b0_2], 2u);
            } else {
                atomicAdd(&my_hist[b0_0], 1u);
                atomicAdd(&my_hist[b0_1], 1u);
                atomicAdd(&my_hist[b0_2], 1u);
                atomicAdd(&my_hist[b0_3], 1u);
            }
        }
        int rem = N & 3;
        int tail_start = N - rem;
        int li = tail_start + threadIdx.x;
        if (threadIdx.x < rem && li < N) {
            uint8_t b0 = key32_msb_bin_desc(col[li]);
            atomicAdd(&my_hist[b0], 1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint8_t b0 = key32_msb_bin_desc(col[i]);
            atomicAdd(&my_hist[b0], 1u);
        }
    }
    __syncthreads();

    // reduce to final hist
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        uint32_t s = 0u;
        for (int w = 0; w < warp_count; ++w) s += hist_warp[w*256 + b];
        hist[b] = s;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (int b = 255; b >= 0; --b) {
            gt_counts[b + 256*t] = sum;
            sum += hist[b];
        }
        thr_bins[t] = 0;
    }
}

// select indices > threshold bin and collect equals for tail passes
static __global__ void k_select_topk_bins(const float * __restrict__ scores,
                                          int N, int T, int ld, int k, int eq_capacity,
                                          const uint32_t * __restrict__ gt_counts, // [256, T]
                                          int * __restrict__ idx_out) 
{
    int t = blockIdx.x;
    if (t >= T) return;
    const float * col = scores + (size_t)ld * t;
    // initialize output indices to -1 to avoid mistaking zeros as valid index 0
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) idx_out[i + k*t] = -1;
    }
    __syncthreads();

    // Round 0: determine thr0 (MSB) from gt_counts
    int thr0 = 0;
    for (int b = 255; b >= 0; --b) {
        uint32_t sgt = gt_counts[b + 256*t];
        uint32_t prev = (b == 0 ? (uint32_t)N : gt_counts[(b - 1) + 256*t]);
        uint32_t eq   = prev - gt_counts[b + 256*t];
        if (sgt < (uint32_t)k && sgt + eq >= (uint32_t)k) {
            thr0 = b;
            break;
        }
    }

    
    // per-warp histogram scratch (max 32 warps)
    __shared__ unsigned int pw_hist[32*256];
    __shared__ unsigned int pw_final[256];
    int warp_count = (blockDim.x + 31) >> 5;
    __shared__ int sel_sofar;
#if SEL_DEBUG
    __shared__ int sel_before_R1;
    __shared__ int sel_before_R2;
#endif
    if (threadIdx.x == 0) sel_sofar = 0;
    __syncthreads();
    uint32_t sgt0 = gt_counts[thr0 + 256*t];
    int take_gt0 = min(k, (int)sgt0);
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[SEL] t=%d thr0=%d sgt0=%u k=%d\n", t, thr0, sgt0, k);
    }
#endif

    extern __shared__ int s_eq[];
    int *eq0 = s_eq;
    int *eq1 = s_eq + eq_capacity;
    __shared__ int eq0_store;
    __shared__ int eq0_total;
    if (threadIdx.x == 0) {
        eq0_store = 0;
        eq0_total = 0;
    }
    __syncthreads();

    // Select b0>thr0 using per-bin prefix allocation; collect b0==thr0 into eq0 (store up to capacity)
    __shared__ int s_written[256];
    // initialize per-bin write cursors to the prefix count from gt_counts
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        s_written[b] = (int)gt_counts[b + 256*t];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t raw = __float_as_uint(col[i]);
        int b0 = (int)key32_msb_bin_desc(__uint_as_float(raw));
        if (b0 > thr0) {
            int pos = atomicAdd(&s_written[b0], 1);
            if (pos < take_gt0) idx_out[pos + k*t] = i;
        } else if (b0 == thr0) {
            atomicAdd(&eq0_total, 1);
            int p = atomicAdd(&eq0_store, 1);
            if (p < eq_capacity) eq0[p] = i;
        }
    }
    __syncthreads();
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[SEL] after R0: eq0_total=%d eq0_store=%d sel_sofar=%d take_gt0=%d\n", eq0_total, eq0_store, sel_sofar, take_gt0);
    }
#endif
    if (threadIdx.x == 0) sel_sofar = take_gt0;
    __syncthreads();
    int remaining = k - sel_sofar;
#if SEL_DEBUG
    if (threadIdx.x == 0) sel_before_R1 = sel_sofar;
#endif
    __syncthreads();

    // Round 1: build h1 over b1 on eq0 if fully stored, else scan full column with b0==thr0
    __shared__ unsigned int h1[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h1[i] = 0u;
    __syncthreads();
    if (eq0_total <= eq_capacity) {
        int lim0 = min(eq0_store, eq_capacity);
        for (int j = threadIdx.x; j < lim0; j += blockDim.x) {
            int idx = eq0[j];
            uint32_t raw = __float_as_uint(col[idx]);
            uint32_t key = ((int32_t)raw < 0)?~raw:(raw^0x80000000u);
            int b1 = (key>>16)&0xFF;
            atomicAdd(&h1[b1],1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw = __float_as_uint(col[i]);
            uint32_t key = ((int32_t)raw < 0)?~raw:(raw^0x80000000u);
            int b0c = (int)key32_msb_bin_desc(__uint_as_float(raw));
            if (b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            atomicAdd(&h1[b1],1u);
        }
    }
    __syncthreads();
    __shared__ int thr1;
    if (threadIdx.x==0){
        unsigned int sum=0,need=remaining;
        thr1=255;
        for(int b=255;b>=0;--b){
            unsigned int sgt=sum;
            unsigned int eqb=h1[b];
            if(sgt<need&&sgt+eqb>=need){
                thr1=b;
                break;
            }
            sum+=eqb;
        }
    }
    __syncthreads();
#if SEL_DEBUG
      if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
          unsigned int sum_h1 = 0;
          for (int b = 0; b < 256; ++b) sum_h1 += h1[b];
          printf("[SEL] R1: thr1=%d remaining=%d path=%s sum_h1=%u eq0_total=%d eq0_store=%d\n",
                 thr1, remaining, (eq0_total <= eq_capacity) ? "buf" : "fallback", sum_h1, eq0_total, eq0_store);
      }
#endif
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[TL_KERNEL] R1 thr1=%d\n", thr1);
    }
#endif


    // Select b1>thr1; collect b1==thr1 into eq1
    __shared__ int eq1_store;
    __shared__ int eq1_total;
    if (threadIdx.x == 0) {
        eq1_store = 0;
        eq1_total = 0;
    }
    __syncthreads();
    if (eq0_total <= eq_capacity) {
        int lim0 = min(eq0_store, eq_capacity);
        for (int j = threadIdx.x; j < lim0; j += blockDim.x) {
            int idx = eq0[j];
            uint32_t raw = __float_as_uint(col[idx]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b1=(key>>16)&0xFF;
            if(b1>thr1){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=idx;
            } else if(b1==thr1){
                atomicAdd(&eq1_total,1);
                int p=atomicAdd(&eq1_store,1);
#if SEL_DEBUG
                if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
                    printf("[SEL] R1 buf select end: sel_sofar=%d emitted=%d eq1_total=%d eq1_store=%d\n",
                            sel_sofar, sel_sofar - sel_before_R1, eq1_total, eq1_store);
                }
#endif

                if(p<eq_capacity) eq1[p]=idx;
            }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1>thr1){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=i;
            } else if(b1==thr1){
#if SEL_DEBUG
                if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
                    printf("[SEL] after R1 select: sel_sofar=%d eq1_total=%d eq1_store=%d\n", sel_sofar, eq1_total, eq1_store);
                }
#endif

                atomicAdd(&eq1_total,1);
                int p=atomicAdd(&eq1_store,1);
                if(p<eq_capacity) eq1[p]=i;
            }
        }
    }
    __syncthreads();
    remaining = k - sel_sofar;
    if (remaining <= 0) return;
#if SEL_DEBUG
    if (threadIdx.x == 0) sel_before_R2 = sel_sofar;
#endif
    __syncthreads();


    // Round 2: histogram on b2 using per-warp hist
    for (int i = threadIdx.x; i < 32*256; i += blockDim.x) pw_hist[i] = 0u;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) pw_final[i] = 0u;
    __syncthreads();
    if (eq1_total <= eq_capacity) {
        int lim1 = min(eq1_store, eq_capacity);
        for (int j = threadIdx.x; j < lim1; j += blockDim.x) {
            int idx=eq1[j];
            uint32_t raw=__float_as_uint(col[idx]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b2=(key>>8)&0xFF;
            atomicAdd(&pw_hist[((threadIdx.x>>5)*256) + b2], 1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1!=thr1) continue;
            int b2=(key>>8)&0xFF;
            atomicAdd(&pw_hist[((threadIdx.x>>5)*256) + b2], 1u);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        unsigned int s=0;
        for (int w=0; w<warp_count; ++w) s += pw_hist[w*256 + b];
        pw_final[b] = s;
    }
    __syncthreads();

    __shared__ int thr2;
    if (threadIdx.x==0){
        unsigned int sum=0,need=remaining;
        thr2=255;
        for(int b=255;b>=0;--b){
            unsigned int sgt=sum;
            unsigned int eqb=pw_final[b];
            if(sgt < need && sgt + eqb >= need){
                thr2=b;
                break;
            }
            sum+=eqb;
        }
    }
    __syncthreads();
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[SEL] thr2=%d remaining=%d (R2 path=%s)\n", thr2, remaining, (eq1_store==eq1_total && eq1_total<=eq_capacity)?"buf":"fallback");
    }
#endif

    // Select b2>thr2; collect b2==thr2 back into eq0 (ping-pong)
    __shared__ int eq2_total;
    if (threadIdx.x == 0) {
        eq0_store = 0;
        eq2_total = 0;
    }
    __syncthreads();
    if (eq1_total <= eq_capacity) {
        int lim1 = min(eq1_store, eq_capacity);
        for (int j = threadIdx.x; j < lim1; j += blockDim.x) {
            int idx=eq1[j];
            uint32_t raw=__float_as_uint(col[idx]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b2=(key>>8)&0xFF;
            if(b2>thr2){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=idx;
            } else if(b2==thr2){
                atomicAdd(&eq2_total,1);
                int p=atomicAdd(&eq0_store,1);
                if(p<eq_capacity) eq0[p]=idx;
            }
        }
        __syncthreads();
#if SEL_DEBUG
        if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
            printf("[SEL] R2 buf end: sel_sofar=%d emitted=%d eq2_total=%d eq0_store=%d lim1=%d eq1_store=%d eq1_total=%d\n",
                    sel_sofar, sel_sofar - sel_before_R2, eq2_total, eq0_store, lim1, eq1_store, eq1_total);
        }
#endif
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1!=thr1) continue;
            int b2=(key>>8)&0xFF;
            if(b2>thr2){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=i;
            } else if(b2==thr2){
                atomicAdd(&eq2_total,1);
                int p=atomicAdd(&eq0_store,1);
                if(p<eq_capacity) eq0[p]=i;
            }
        }
    }
    __syncthreads();
    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Round 3: histogram on b2 (third byte) using per-warp hist
    for (int i = threadIdx.x; i < 32*256; i += blockDim.x) pw_hist[i] = 0u;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) pw_final[i] = 0u;
    __syncthreads();
    if (eq2_total <= eq_capacity) {
        int lim0 = min(eq0_store, eq_capacity);
        for (int j = threadIdx.x; j < lim0; j += blockDim.x) {
            int idx=eq0[j];
            uint32_t raw=__float_as_uint(col[idx]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b3= key & 0xFF;
            atomicAdd(&pw_hist[((threadIdx.x>>5)*256) + b3], 1u);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1!=thr1) continue;
            int b2=(key>>8)&0xFF;
            if(b2!=thr2) continue;
            int b3= key & 0xFF;
            atomicAdd(&pw_hist[((threadIdx.x>>5)*256) + b3], 1u);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        unsigned int s=0;
        for (int w=0; w<warp_count; ++w) s += pw_hist[w*256 + b];
        pw_final[b] = s;
    }
    __syncthreads();

    __shared__ int thr3;
    if (threadIdx.x==0){
        unsigned int sum=0,need=remaining;
        thr3=255;
        for(int b=255; b>=0; --b){
            unsigned int sgt=sum;
            unsigned int eqb=pw_final[b];
            if(sgt < need && sgt + eqb >= need){
                thr3=b;
                break;
            }
            sum+=eqb;
        }
    }
    __syncthreads();
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[SEL] thr3=%d remaining=%d (R3 path=%s)\n", thr3, remaining, (eq0_store==eq2_total && eq2_total<=eq_capacity)?"buf":"fallback");
    }
#endif

    // Select b3>thr3; collect b3==thr3 into eq1
    __shared__ int eq3_total;
    if (threadIdx.x == 0) {
        eq1_store = 0;
        eq3_total = 0;
    }
    __syncthreads();
    if (eq2_total <= eq_capacity) {
        int lim0 = min(eq0_store, eq_capacity);
        for (int j = threadIdx.x; j < lim0; j += blockDim.x) {
            int idx=eq0[j];
            uint32_t raw=__float_as_uint(col[idx]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b3= key & 0xFF;
            if(b3>thr3){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=idx;
            } else if(b3==thr3){
                atomicAdd(&eq3_total,1);
                int p=atomicAdd(&eq1_store,1);
                if(p<eq_capacity) eq1[p]=idx;
#if SEL_DEBUG
                if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
                    printf("[SEL] indices for t=%d:\n", t);
                    for (int i = 0; i < k; ++i) {
                        int idx = idx_out[i + k*t];
                        if (idx >= 0 && idx < N) {
                            float v = col[idx];
                            unsigned int raw = __float_as_uint(v);
                            unsigned int key = ((int)raw < 0) ? ~raw : (raw ^ 0x80000000u);
                            int b0 = (int)key32_msb_bin_desc(v);
                            int b1 = (key >> 16) & 0xFF;
                            int b2 = (key >> 8) & 0xFF;
                            int b3 = key & 0xFF;
                            printf("(%d:%.5f b0=%d b1=%d b2=%d b3=%d)\n", idx, v, b0, b1, b2, b3);
                        } else {
                            printf("(%d:invalid)\n", idx);
                        }
                    }
                    printf("\n");
                }
#endif
            }
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1!=thr1) continue;
            int b2=(key>>8)&0xFF;
            if(b2!=thr2) continue;
            int b3= key & 0xFF;
            if(b3>thr3){
                int pos=atomicAdd(&sel_sofar,1);
                if(pos<k) idx_out[pos+k*t]=i;
            } else if(b3==thr3){
                atomicAdd(&eq3_total,1);
                int p=atomicAdd(&eq1_store,1);
                if(p<eq_capacity) eq1[p]=i;
            }
        }
    }
    __syncthreads();
    remaining = k - sel_sofar;
    if (remaining <= 0) return;

    // Final: fill remaining from equals (eq1) or scan predicated if overflow
    if (eq1_store == eq3_total && eq3_total <= eq_capacity) {
        int lim1 = min(eq1_store, eq_capacity);
        for (int j = threadIdx.x; j < lim1; j += blockDim.x) {
            int idx=eq1[j];
            int pos=atomicAdd(&sel_sofar,1);
            if (pos < k) idx_out[pos + k*t] = idx;
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            uint32_t raw=__float_as_uint(col[i]);
            uint32_t key=((int32_t)raw<0)?~raw:(raw^0x80000000u);
            int b0c=(int)key32_msb_bin_desc(__uint_as_float(raw));
            if(b0c!=thr0) continue;
            int b1=(key>>16)&0xFF;
            if(b1!=thr1) continue;
            int b2=(key>>8)&0xFF;
            if(b2!=thr2) continue;
            int b3= key & 0xFF;
#if SEL_DEBUG
            if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
                printf("[SEL] final indices t=%d: ", t);
                for (int i = 0; i < k; ++i) {
                    int idx = idx_out[i + k*t];
                    if (idx >= 0 && idx < N) {
                        float v = col[idx];
                        unsigned int raw = __float_as_uint(v);
                        unsigned int key = ((int)raw < 0) ? ~raw : (raw ^ 0x80000000u);
                        int b0 = (int)key32_msb_bin_desc(v);
                        int b1 = (key >> 16) & 0xFF;
                        int b2 = (key >> 8) & 0xFF;
                        int b3 = key & 0xFF;
                        printf("(%d:%.5f b0=%d b1=%d b2=%d b3=%d) ", idx, v, b0, b1, b2, b3);
                    } else {
                        printf("(%d:invalid) ", idx);
                    }
                }
                printf("\n");
            }
#endif

            if(b3!=thr3) continue;
            int pos=atomicAdd(&sel_sofar,1);
            if (pos < k) idx_out[pos + k*t] = i;
        }
    }
    __syncthreads();
#if SEL_DEBUG
    if (threadIdx.x == 0 && (SEL_DEBUG_COL==0 || blockIdx.x == SEL_DEBUG_COL)) {
        printf("[SEL] final sel_sofar=%d\n", sel_sofar);
    }
#endif
}

void ggml_cuda_topk_radix_indices_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    // Radix-like path: histogram top byte + select with tie refinement
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d  = nullptr;
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    const int hist_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    const size_t hist_shmem = (size_t)(((hist_threads/32) + 1) * 256) * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    // Equal-bin selection kernel; bound dynamic shared memory to device limit
    const int sel_threads = hist_threads;
    // Conservative eq buffer capacity to avoid exceeding per-block shared mem
    int cap_env = 0;
    const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) {
        cap_env = atoi(env_cap);
        if (cap_env < 0) cap_env = 0;
    }
    int cap_default = 4096;
    const int eq_cap = max(k, min(N, cap_env ? cap_env : cap_default));
    size_t sel_shmem = (size_t) (2*eq_cap) * sizeof(int);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_select_topk_bins, (int)sel_shmem);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap, gt_counts_d, idx_d);

    cudaFree(gt_counts_d);
    cudaFree(thr_bins_d);
}

extern "C" void ggml_cuda_topk_radix_indices_host(const float * scores_h, int N, int T, int k, int * idx_h) {
    // Simple host wrapper: copy scores to device, allocate idx device, invoke kernels, copy back
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    int * idx_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&idx_d, sizeof(int) * (size_t)k * T);
    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);
    ggml_cuda_topk_radix_indices_device(ctx, scores_d, N, T, k, idx_d);
    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(scores_d);
    cudaFree(idx_d);
}

extern "C" void ggml_cuda_topk_histogram_host(const float * scores_h, int N, int T,
                                               unsigned int * gt_counts_h, unsigned int * thr_bins_h) {
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    uint32_t * gt_counts_d = nullptr;
    uint32_t * thr_bins_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&thr_bins_d,  sizeof(uint32_t) * (size_t)T);

    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);

    const int hist_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    const size_t hist_shmem = (size_t)(((hist_threads/32) + 1) * 256) * sizeof(uint32_t);
    k_histogram_topbyte<<<T, hist_threads, hist_shmem, stream>>>(scores_d, N, T, /*ld=*/N, thr_bins_d, gt_counts_d);

    cudaMemcpyAsync(gt_counts_h, gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thr_bins_h,  thr_bins_d,  sizeof(uint32_t) * (size_t)T,        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(thr_bins_d);
}

extern "C" void ggml_cuda_topk_select_host(const float * scores_h, int N, int T, int k,
                                            const unsigned int * gt_counts_h, int * idx_h) {
    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = ctx.stream();
    float * scores_d = nullptr;
    uint32_t * gt_counts_d = nullptr;
    int * idx_d = nullptr;
    cudaMalloc(&scores_d, sizeof(float) * (size_t)N * T);
    cudaMalloc(&gt_counts_d, sizeof(uint32_t) * 256 * (size_t)T);
    cudaMalloc(&idx_d, sizeof(int) * (size_t)k * T);

    cudaMemcpyAsync(scores_d, scores_h, sizeof(float) * (size_t)N * T, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gt_counts_d, gt_counts_h, sizeof(uint32_t) * 256 * (size_t)T, cudaMemcpyHostToDevice, stream);

    const int sel_threads = env_threads_or_default("LLAMA_SPARSE_TOPK_THREADS", 1024);
    int cap_env = 0; const char *env_cap = getenv("LLAMA_SPARSE_TOPK_EQ_CAP");
    if (env_cap) {
        cap_env = atoi(env_cap);
        if (cap_env < 0) cap_env = 0;
    }
    int cap_default = 4096;
    const int eq_cap_host = max(k, min(N, cap_env ? cap_env : cap_default));
    const size_t sel_shmem = (size_t) (2*eq_cap_host) * sizeof(int);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_select_topk_bins, (int)sel_shmem);
    k_select_topk_bins<<<T, sel_threads, sel_shmem, stream>>>(scores_d, N, T, /*ld=*/N, k, eq_cap_host, gt_counts_d, idx_d);

    cudaMemcpyAsync(idx_h, idx_d, sizeof(int) * (size_t)k * T, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(scores_d);
    cudaFree(gt_counts_d);
    cudaFree(idx_d);
}

// -----------------------------------------------------------------------------
// TileLang DeepSeek V3.2 top-k selector (ported line-by-line to CUDA)
// This kernel mirrors the control flow and comments of
// /workspace/tilelang/examples/deepseek_v32/topk_selector.py
// Inputs:
//   input  : [batch, seq_len] float32 scores
//   index  : [batch, topk] int32 output indices
//   starts : [batch] int32 per-batch start index (inclusive)
//   ends   : [batch] int32 per-batch end index (exclusive)
// Notes:
// - BLOCK_SIZE is 1024 threads per block; one block per batch element
// - RADIX = 256; SMEM_INPUT_SIZE = 4096 (tie buffer per round)
// - convert_to_uint16 / convert_to_uint32 match the TileLang mapping
// Simple glue kernels for wiring the TileLang-ported selector
// -----------------------------------------------------------------------------

// Cast to float16, reinterpret bits, then map sign for descending order
static __device__ __forceinline__ uint16_t tl_convert_to_uint16(float x) {
    __half h = __float2half(x);
    unsigned short bits_uint = __half_as_ushort(h);
    bits_uint = (x < 0.0f) ? (unsigned short)(~bits_uint & 0xFFFFu)
                           : (unsigned short)(bits_uint | 0x8000u);
    return (uint16_t)(bits_uint >> 8);
}

static __device__ __forceinline__ uint32_t tl_convert_to_uint32(float x) {
    uint32_t bits_uint = __float_as_uint(x);
    return ((int32_t)bits_uint < 0) ? ~bits_uint : (bits_uint | 0x80000000u);
}

// Derive per-column end (exclusive) from scores by scanning for last value > threshold
static __global__ void k_derive_ends_from_scores(const float * __restrict__ scores,
                                                 int N, int T, int ld, float masked_thresh,
                                                 int * __restrict__ ends) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float * col = scores + (size_t)ld * t;
    int e = 0;
    for (int i = N - 1; i >= 0; --i) {
        float v = col[i];
        if (v > masked_thresh) { e = i + 1; break; }
    }
    ends[t] = e;
}

// Fixed configuration to match TileLang example
#ifndef TL_TOPK_RADIX
#define TL_TOPK_RADIX 256
#endif
#ifndef TL_TOPK_BLOCK_SIZE
#define TL_TOPK_BLOCK_SIZE 1024
#endif
#ifndef TL_TOPK_SMEM_INPUT_SIZE
#define TL_TOPK_SMEM_INPUT_SIZE 4096
#endif

// Port of tl_topk_impl kernel
static __global__ void k_tl_topk_port(
        const float * __restrict__ input, // [batch, seq_len]
        int batch,
        int seq_len,
        int topk,
        int * __restrict__ index,         // [batch, topk]
        const int * __restrict__ starts,  // [batch]
        const int * __restrict__ ends) {  // [batch]
    // with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
    int bx = blockIdx.x;
    if (bx >= batch) return;
    int tx = threadIdx.x; // T.get_thread_binding()

    // Shared allocations (names match TileLang code)
    __shared__ int s_threshold_bin_id[1];
    __shared__ int s_histogram[TL_TOPK_RADIX + 1];
    __shared__ int s_num_input[2];
    __shared__ int s_input_idx[2][TL_TOPK_SMEM_INPUT_SIZE];

    // Local vars (l_* prefix to mirror TileLang code)
    int l_threshold_bin_id = 0;
    int l_new_topk = topk;
    int l_num_input = 0;
    int l_bin_id32 = 0;
    int l_val = 0;
    int l_start_pos = 0;
    int l_start_idx = starts[bx];
    int l_end_idx   = ends[bx];

    // stage 1: use 8bit to do quick topk
    // T.fill(s_histogram, 0)
    for (int i = tx; i < TL_TOPK_RADIX + 1; i += blockDim.x) s_histogram[i] = 0;
    if (tx == 0) s_num_input[0] = 0; // T.fill(s_num_input[0], 0)

    __syncthreads();
    // for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    int iters = (seq_len + TL_TOPK_BLOCK_SIZE - 1) / TL_TOPK_BLOCK_SIZE;
    for (int s = 0; s < iters; ++s) {
        int input_idx = s * TL_TOPK_BLOCK_SIZE + tx;
        if (input_idx < l_end_idx && input_idx >= l_start_idx && input_idx < seq_len) {
            float v = input[(size_t)bx * seq_len + input_idx];
            uint16_t inval_int16 = tl_convert_to_uint16(v);
            atomicAdd(&s_histogram[inval_int16], 1);
        }
    }
    __syncthreads();

    // cumsum over RADIX bins (suffix-style), TileLang parity
    if (tx < TL_TOPK_RADIX) {
        for (int i = 0; i < 8; ++i) {
            int offset = 1 << i;
            named_sync<3>(TL_TOPK_RADIX);
            if (tx < TL_TOPK_RADIX - offset) {
                l_val = s_histogram[tx] + s_histogram[tx + offset];
            }
            named_sync<3>(TL_TOPK_RADIX);
            if (tx < TL_TOPK_RADIX - offset) {
                s_histogram[tx] = l_val;
            }
        }
        // find threshold bin id
        named_sync<3>(TL_TOPK_RADIX);
        if (s_histogram[tx] > l_new_topk && s_histogram[tx + 1] <= l_new_topk) {
            s_threshold_bin_id[0] = tx;
        }
    }
    __syncthreads();

    l_threshold_bin_id = s_threshold_bin_id[0];
    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1];
    __syncthreads();
    if (SEL_DEBUG && (bx == 0 && tx == 0)) {
        int sgt0_dbg = s_histogram[l_threshold_bin_id + 1];
        printf("[TL_KERNEL] thr0=%d sgt0=%d new_topk=%d\n", l_threshold_bin_id, sgt0_dbg, l_new_topk);
    }

    // collect all elements with exponent  threshold
    for (int s = 0; s < iters; ++s) {
        __syncthreads();
        int input_idx = s * TL_TOPK_BLOCK_SIZE + tx;
        if (input_idx < l_end_idx && input_idx >= l_start_idx && input_idx < seq_len) {
            float v = input[(size_t)bx * seq_len + input_idx];
            int bin_id = (int)tl_convert_to_uint16(v);
            l_bin_id32 = bin_id;
            if (l_bin_id32 > l_threshold_bin_id) {
                // pos = atomic_add(s_histogram[l_bin_id32 + 1], 1)
                int pos = atomicAdd(&s_histogram[l_bin_id32 + 1], 1);
                // index[bx, pos] = input_idx
                if (pos < topk) index[bx * topk + pos] = input_idx;
            } else if (l_bin_id32 == l_threshold_bin_id && l_new_topk > 0) {
                int pos = atomicAdd(&s_num_input[0], 1);
                if (pos < TL_TOPK_SMEM_INPUT_SIZE) {
                    s_input_idx[0][pos] = input_idx;
                }
            }
        }
    }

    // stage 2: tail pass
    for (int round = 0; round < 4; ++round) {
        if (l_new_topk <= 0) break; // T.loop_break()

        int r_idx = round & 1;
        l_start_pos = topk - l_new_topk;

        __syncthreads();
        // T.fill(s_histogram, 0)
        for (int i = tx; i < TL_TOPK_RADIX + 1; i += blockDim.x) s_histogram[i] = 0;
        if (tx == 0) s_num_input[r_idx ^ 1] = 0;
        __syncthreads();

        l_num_input = s_num_input[r_idx];
        if (SEL_DEBUG && bx == 0 && tx == 0) {
            printf("[TL_KERNEL] R%d start: l_new_topk=%d l_num_input=%d l_start_pos=%d\n", round, l_new_topk, l_num_input, l_start_pos);
        }
        int it2 = (l_num_input + TL_TOPK_BLOCK_SIZE - 1) / TL_TOPK_BLOCK_SIZE;
        for (int s = 0; s < it2; ++s) {
            int idx = s * TL_TOPK_BLOCK_SIZE + tx;
            if (idx < l_num_input) {
                int in_idx = s_input_idx[r_idx][idx];
                float v = input[(size_t)bx * seq_len + in_idx];
                l_bin_id32 = (int)((tl_convert_to_uint32(v) >> (24 - round * 8)) & 0xFFu);
                atomicAdd(&s_histogram[l_bin_id32], 1);
            }
        }
        __syncthreads();

        // cumsum over RADIX bins (suffix-style), TileLang parity
        if (tx < TL_TOPK_RADIX) {
            for (int i = 0; i < 8; ++i) {
                int offset = 1 << i;
                named_sync<3>(TL_TOPK_RADIX);
                if (tx < TL_TOPK_RADIX - offset) {
                    l_val = s_histogram[tx] + s_histogram[tx + offset];
                }
                named_sync<3>(TL_TOPK_RADIX);
                if (tx < TL_TOPK_RADIX - offset) {
                    s_histogram[tx] = l_val;
                }
            }
            // find threshold bin id
            named_sync<3>(TL_TOPK_RADIX);
            if (s_histogram[tx] > l_new_topk && s_histogram[tx + 1] <= l_new_topk) {
                s_threshold_bin_id[0] = tx;
            }
        }
        __syncthreads();

        l_threshold_bin_id = s_threshold_bin_id[0];
        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1];
        __syncthreads();

        for (int s = 0; s < it2; ++s) {
            __syncthreads();
            int idx = s * TL_TOPK_BLOCK_SIZE + tx;
            if (idx < l_num_input) {
                int in_idx = s_input_idx[r_idx][idx];
                float v = input[(size_t)bx * seq_len + in_idx];
                l_bin_id32 = (int)((tl_convert_to_uint32(v) >> (24 - round * 8)) & 0xFFu);
                if (l_bin_id32 > l_threshold_bin_id) {
                    int pos = atomicAdd(&s_histogram[l_bin_id32 + 1], 1) + l_start_pos;
                    if (pos < topk) index[bx * topk + pos] = in_idx;
                } else if (l_bin_id32 == l_threshold_bin_id && l_new_topk > 0) {
                    if (round == 3) {
                        int l_out_pos = atomicAdd(&s_histogram[l_bin_id32 + 1], 1) + l_start_pos;
                        if (l_out_pos < topk) index[bx * topk + l_out_pos] = in_idx;
                    } else {
                        int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (pos < TL_TOPK_SMEM_INPUT_SIZE) {
                            s_input_idx[r_idx ^ 1][pos] = in_idx;
                        }
                    }
                }
            }
        }
    }
}


// Optional host wrapper for future wiring/testing; not used now
extern "C" void ggml_cuda_topk_tilelang_port_host(const float * input_h, int batch, int seq_len, int topk,
                                                   int * index_h, const int * starts_h, const int * ends_h) {
    // Intentionally left as a placeholder; do not wire yet.
    (void)input_h; (void)batch; (void)seq_len; (void)topk; (void)index_h; (void)starts_h; (void)ends_h;
}


void ggml_cuda_topk_tilelang_port_device(ggml_backend_cuda_context & ctx,
                                         const float * scores_d, int N, int T, int k,
                                         int * idx_d) {
    cudaStream_t stream = ctx.stream();
    // Synthesize starts/ends as [0, N)
    int * d_starts = nullptr, * d_ends = nullptr;
    cudaMalloc(&d_starts, sizeof(int) * (size_t)T);
    cudaMalloc(&d_ends,   sizeof(int) * (size_t)T);
    std::vector<int> h_starts(T, 0), h_ends(T, N);

    cudaMemcpyAsync(d_starts, h_starts.data(), sizeof(int) * (size_t)T, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ends,   h_ends.data(),   sizeof(int) * (size_t)T, cudaMemcpyHostToDevice, stream);

    // Derive per-column ends from masked scores to avoid scanning beyond kv_end
    {
        int threads = 128;
        int blocks = (T + threads - 1)/threads;
        k_derive_ends_from_scores<<<blocks, threads, 0, stream>>>(scores_d, N, T, /*ld=*/N, -1.0e29f, d_ends);
    }

    // Directly launch the ported kernel on [N, T] row-major (batch=T, seq_len=N)
    dim3 grid(T);
    dim3 block(TL_TOPK_BLOCK_SIZE);
    CUDA_CHECK(cudaMemsetAsync(idx_d, 0xFF, sizeof(int) * (size_t)k * T, stream));
    const char * __prof_env = getenv("LLAMA_SPARSE_PROF");

    auto * __prof_each_env = getenv("LLAMA_SPARSE_PROF_EACH");
    if (__prof_env && *__prof_env) {
        cudaEvent_t __e0, __e1;
        cudaEventCreate(&__e0);
        cudaEventCreate(&__e1);
        cudaEventRecord(__e0, stream);
        k_tl_topk_port<<<grid, block, 0, stream>>>(scores_d, T, N, k, idx_d, d_starts, d_ends);
        cudaEventRecord(__e1, stream);
        cudaEventSynchronize(__e1);
        float __ms = 0.0f;
        cudaEventElapsedTime(&__ms, __e0, __e1);
        static int __cnt_idx_cuda = 0;
        static double __sum_idx_cuda = 0.0;
        __sum_idx_cuda += __ms;
        __cnt_idx_cuda++;
        if (__prof_each_env && *__prof_each_env) {
            fprintf(stderr, "[PROFILE_TL_ONLY] TILELANG_TOPK N=%d T=%d k=%d ms=%.3f\n", N, T, k, __ms);
        } else {
            if (__cnt_idx_cuda % 50 == 0) {
                fprintf(stderr, "[PROFILE_TL_ONLY] TILELANG_TOPK N=%d T=%d k=%d avg_ms=%.3f over 50 calls\n",
                        N, T, k,  (float)(__sum_idx_cuda/50.0));
                __sum_idx_cuda = 0.0;
            }
        }
        cudaEventDestroy(__e0);
        cudaEventDestroy(__e1);
    } else {
        k_tl_topk_port<<<grid, block, 0, stream>>>(scores_d, T, N, k, idx_d, d_starts, d_ends);
    }

    // Debug: validate first column indices against threshold when profiling is enabled (SEL_DEBUG only)
    if (SEL_DEBUG) {
        const char * __prof_env2 = getenv("LLAMA_SPARSE_PROF");
        if (__prof_env2 && *__prof_env2) {
            // Copy first column t=0 of input and first k indices
            std::vector<int> idx0(k, -1);
            CUDA_CHECK(cudaMemcpyAsync(idx0.data(), idx_d, sizeof(int) * (size_t)k, cudaMemcpyDeviceToHost, stream));
            std::vector<float> col0(N);
            CUDA_CHECK(cudaMemcpyAsync(col0.data(), scores_d, sizeof(float) * (size_t)N, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // Compute Kth threshold
            // Host compute sgt0 for t=0 for debugging
            {
                unsigned int hist[256] = {0};
                for (int i = 0; i < N; ++i) {
                    uint8_t b = host_convert_to_uint16_bin(col0[i]);
                    hist[b]++;
                }
                unsigned int S[257];
                S[256] = 0;
                for (int b = 255; b >= 0; --b) S[b] = S[b+1] + hist[b];
                int thr0 = 0;
                for (int b = 255; b >= 0; --b) {
                    if (S[b] > (unsigned)k && S[b+1] <= (unsigned)k) { thr0 = b; break; }
                }
                unsigned int sgt0 = S[thr0+1];
                if (SEL_DEBUG) fprintf(stderr, "[TL_HOST_DBG] thr0=%d sgt0=%u\n", thr0, sgt0);
            }

            std::vector<float> sorted = col0;
            if (k > 0 && k <= N) {
                std::nth_element(sorted.begin(), sorted.begin() + (k-1), sorted.end(), std::greater<float>());
                float thresh = sorted[k-1];
                // compute MSB threshold bin for host check
                int thr_bin = host_convert_to_uint16_bin(thresh);
                int below = 0; int bad_idx = -1; float bad_v = 0.0f; int bad_bin = -1;
                std::vector<char> seen(N, 0);
                for (int i = 0; i < k; ++i) {
                    int ix = idx0[i];
                    if (ix < 0 || ix >= N) { below++; bad_idx = ix; bad_v = NAN; bad_bin = -1; break; }
                    if (!seen[ix]) seen[ix] = 1;
                    float v = col0[ix];
                    int vb = host_convert_to_uint16_bin(v);
                    if (!(v >= thresh)) { below++; bad_idx = ix; bad_v = v; bad_bin = vb; break; }
                }
                if (SEL_DEBUG) fprintf(stderr, "[TL_DEBUG] t=0 thresh=%.6f (bin=%d) below=%d bad_idx=%d bad_v=%g bad_bin=%d\n", thresh, thr_bin, below, bad_idx, bad_v, bad_bin);
            }
            if (SEL_DEBUG) {
                fprintf(stderr, "[TL_DEBUG_IDX] idx0: ");
                for (int i = 0; i < k; ++i) fprintf(stderr, "%d ", idx0[i]);
                fprintf(stderr, "\n");
            }
        }

    }

    cudaFree(d_starts);
    cudaFree(d_ends);
}
