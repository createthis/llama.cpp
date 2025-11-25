#include "vendor_config.h"
#include "tl_templates/cuda/cuda_fp8.h"
#include "tl_templates/cuda/gemm.h"
#include "tl_templates/cuda/copy.h"
#include "tl_templates/cuda/reduce.h"
#include "tl_templates/cuda/ldsm.h"
#include "tl_templates/cuda/threadblock_swizzle.h"
#include "tl_templates/cuda/debug.h"
#ifdef ENABLE_BF16
#include "tl_templates/cuda/cuda_bf16_fallbacks.cuh"
#endif

extern "C" __global__ void mqa_attn_return_logits_kernel_kernel(int* __restrict__ CuSeqLenKE, int* __restrict__ CuSeqLenKS, float* __restrict__ IndexKScale, __grid_constant__ const CUtensorMap IndexK_desc, __grid_constant__ const CUtensorMap IndexQ_desc, float* __restrict__ Logits, float* __restrict__ Weights, int seq_len, int seq_len_kv);
extern "C" __global__ void __launch_bounds__(640, 1) mqa_attn_return_logits_kernel_kernel(int* __restrict__ CuSeqLenKE, int* __restrict__ CuSeqLenKS, float* __restrict__ IndexKScale, __grid_constant__ const CUtensorMap IndexK_desc, __grid_constant__ const CUtensorMap IndexQ_desc, float* __restrict__ Logits, float* __restrict__ Weights, int seq_len, int seq_len_kv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int cu_k_s_min[1];
  int cu_k_e_max[1];
  float weights[2];
  float index_k_scale_fragment[32];
  float s[64];
  float s_reshaped[64];
  float logits[32];
  __shared__ uint64_t mbarrier_mem[7];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(IndexQ_desc);
    tl::prefetch_tma_descriptor(IndexK_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(512);
    mbarrier[4].init(512);
    mbarrier[5].init(512);
    mbarrier[6].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (512 <= ((int)threadIdx.x)) {
    cu_k_s_min[0] = 2147483647;
    cu_k_e_max[0] = -2147483648;
    for (int bq_i = 0; bq_i < 32; ++bq_i) {
      if (((((int)blockIdx.x) * 32) + bq_i) < seq_len) {
        cu_k_s_min[0] = min(cu_k_s_min[0], min(CuSeqLenKS[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)bq_i))], seq_len_kv));
      } else {
        cu_k_s_min[0] = min(cu_k_s_min[0], 0);
      }
    }
    for (int bq_i_1 = 0; bq_i_1 < 32; ++bq_i_1) {
      if (((((int)blockIdx.x) * 32) + bq_i_1) < seq_len) {
        cu_k_e_max[0] = max(cu_k_e_max[0], min(CuSeqLenKE[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)bq_i_1))], seq_len_kv));
      } else {
        cu_k_e_max[0] = max(cu_k_e_max[0], 0);
      }
    }
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[6].expect_transaction(8192);
      tl::fence_proxy_async();
      tl::tma_load(IndexQ_desc, mbarrier[6], (&(((fp8_e4_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.x) * 128));
    }
    mbarrier[6].arrive();
    for (int nbn_i = 0; nbn_i < (((cu_k_e_max[0] + 255) - cu_k_s_min[0]) >> 8); ++nbn_i) {
      mbarrier[((nbn_i % 3) + 3)].wait((((nbn_i % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(nbn_i % 3)].expect_transaction(16384);
        tl::fence_proxy_async();
        tl::tma_load(IndexK_desc, mbarrier[(nbn_i % 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((nbn_i % 3) * 16384) + 8192)])), 0, ((nbn_i * 256) + cu_k_s_min[0]));
      }
      tl::mbarrier_cp_async_arrive(mbarrier[(nbn_i % 3)]);
      mbarrier[(nbn_i % 3)].arrive();
    }
  } else {
    cu_k_s_min[0] = 2147483647;
    cu_k_e_max[0] = -2147483648;
    for (int bq_i_2 = 0; bq_i_2 < 32; ++bq_i_2) {
      if (((((int)blockIdx.x) * 32) + bq_i_2) < seq_len) {
        cu_k_s_min[0] = min(cu_k_s_min[0], min(CuSeqLenKS[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)bq_i_2))], seq_len_kv));
      } else {
        cu_k_s_min[0] = min(cu_k_s_min[0], 0);
      }
    }
    for (int bq_i_3 = 0; bq_i_3 < 32; ++bq_i_3) {
      if (((((int)blockIdx.x) * 32) + bq_i_3) < seq_len) {
        cu_k_e_max[0] = max(cu_k_e_max[0], min(CuSeqLenKE[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)bq_i_3))], seq_len_kv));
      } else {
        cu_k_e_max[0] = max(cu_k_e_max[0], 0);
      }
    }
    float2 condval;
    if (((((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) >> 5) * 2)) + ((((int)threadIdx.x) & 3) >> 1)) < seq_len)) {
      condval = *(float2*)(Weights + (((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    } else {
      condval = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    *(float2*)(weights + 0) = condval;
    mbarrier[6].wait(0);
    for (int nbn_i_1 = 0; nbn_i_1 < (((cu_k_e_max[0] + 255) - cu_k_s_min[0]) >> 8); ++nbn_i_1) {
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        if (((((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv) && (0 <= ((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]))) && (((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv)) {
          index_k_scale_fragment[i] = IndexKScale[((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((int64_t)cu_k_s_min[(int64_t)0]))];
        } else {
          float condval_1;
          if ((((((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv) && (0 <= ((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]))) && (((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv))) {
            condval_1 = IndexKScale[((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((int64_t)cu_k_s_min[(int64_t)0]))];
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          index_k_scale_fragment[i] = condval_1;
        }
      }
      mbarrier[(nbn_i_1 % 3)].wait(((nbn_i_1 % 6) / 3));
      tl::fence_proxy_async();
      tl::gemm_ss<256, 128, 64, 1, 16, 0, 1, 1, 64, 64, 0, 0>((&(((fp8_e4_t*)buf_dyn_shmem)[(((nbn_i_1 % 3) * 16384) + 8192)])), (&(((fp8_e4_t*)buf_dyn_shmem)[0])), (&(s[0])));
      mbarrier[((nbn_i_1 % 3) + 3)].arrive();
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        s_reshaped[i_1] = ((max(s[i_1], 0x0p+0f/*0.000000e+00*/) * weights[(i_1 & 1)]) * index_k_scale_fragment[(i_1 >> 1)]);
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        logits[i_2] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          logits[i_2] = (logits[i_2] + s_reshaped[((i_2 * 2) + rv)]);
        }
        logits[i_2] = tl::AllReduce<tl::SumOp, 2, 1, 0>::run(logits[i_2]);
      }
      if ((((int)threadIdx.x) % 2) == 0) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 32; ++i_3) {
          if (((0 <= ((((nbn_i_1 * 256) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0])) && (((((nbn_i_1 * 256) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv)) && ((((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) >> 5) * 2)) + ((((int)threadIdx.x) & 3) >> 1)) < seq_len)) {
            if (((((nbn_i_1 * 256) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv) {
              Logits[(((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i_3) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) >> (int64_t)1)) * ((int64_t)seq_len_kv))) + ((int64_t)cu_k_s_min[(int64_t)0]))] = logits[i_3];
            }
          } else {
            if (0 <= ((((nbn_i_1 * 256) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0])) {
              if (((((nbn_i_1 * 256) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min[0]) < seq_len_kv) {
                if ((((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) >> 5) * 2)) + ((((int)threadIdx.x) & 3) >> 1)) < seq_len) {
                  Logits[(((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i_3) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) >> (int64_t)1)) * ((int64_t)seq_len_kv))) + ((int64_t)cu_k_s_min[(int64_t)0]))] = logits[i_3];
                }
              }
            }
          }
        }
      }
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_mqa_attn_return_logits_kernel_kernel = cudaFuncSetAttribute(mqa_attn_return_logits_kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 57344);
    if (result_mqa_attn_return_logits_kernel_kernel != cudaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 57344, cudaGetErrorString(result_mqa_attn_return_logits_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(fp8_e4_t* __restrict__ IndexQ, fp8_e4_t* __restrict__ IndexK, float* __restrict__ IndexKScale, float* __restrict__ Logits, float* __restrict__ Weights, int* __restrict__ CuSeqLenKS, int* __restrict__ CuSeqLenKE, int seq_len_kv, int seq_len, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap IndexK_desc;
	CUtensorMapDataType IndexK_desc_type= (CUtensorMapDataType)0;
	cuuint32_t IndexK_desc_tensorRank= 2;
	void *IndexK_desc_globalAddress= IndexK;
	cuuint64_t IndexK_desc_globalDim[2]= {64,seq_len_kv};
	cuuint64_t IndexK_desc_globalStride[2]= {1,64};
	cuuint32_t IndexK_desc_boxDim[2]= {64,256};
	cuuint32_t IndexK_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave IndexK_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle IndexK_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion IndexK_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill IndexK_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult IndexK_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &IndexK_desc, IndexK_desc_type, IndexK_desc_tensorRank, IndexK_desc_globalAddress, IndexK_desc_globalDim, IndexK_desc_globalStride + 1, IndexK_desc_boxDim, IndexK_desc_elementStrides, IndexK_desc_interleave, IndexK_desc_swizzle, IndexK_desc_l2Promotion, IndexK_desc_oobFill);

	if (IndexK_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor IndexK_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap IndexQ_desc;
	CUtensorMapDataType IndexQ_desc_type= (CUtensorMapDataType)0;
	cuuint32_t IndexQ_desc_tensorRank= 2;
	void *IndexQ_desc_globalAddress= IndexQ;
	cuuint64_t IndexQ_desc_globalDim[2]= {64,seq_len * 4};
	cuuint64_t IndexQ_desc_globalStride[2]= {1,64};
	cuuint32_t IndexQ_desc_boxDim[2]= {64,128};
	cuuint32_t IndexQ_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave IndexQ_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle IndexQ_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion IndexQ_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill IndexQ_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult IndexQ_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &IndexQ_desc, IndexQ_desc_type, IndexQ_desc_tensorRank, IndexQ_desc_globalAddress, IndexQ_desc_globalDim, IndexQ_desc_globalStride + 1, IndexQ_desc_boxDim, IndexQ_desc_elementStrides, IndexQ_desc_interleave, IndexQ_desc_swizzle, IndexQ_desc_l2Promotion, IndexQ_desc_oobFill);

	if (IndexQ_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor IndexQ_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	mqa_attn_return_logits_kernel_kernel<<<dim3((seq_len + 31) / 32, 1, 1), dim3(640, 1, 1), 57344, stream>>>(CuSeqLenKE, CuSeqLenKS, IndexKScale, IndexK_desc, IndexQ_desc, Logits, Weights, seq_len, seq_len_kv);
	TILELANG_CHECK_LAST_ERROR("mqa_attn_return_logits_kernel_kernel");

	return 0;
}

