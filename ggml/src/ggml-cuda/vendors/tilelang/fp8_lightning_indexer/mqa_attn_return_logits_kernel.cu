#include "vendor_config.h"
#include "tl_templates/cuda/instruction/mma.h"
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

extern "C" __global__ void mqa_attn_return_logits_kernel_kernel(const int* __restrict__ CuSeqLenKE, const int* __restrict__ CuSeqLenKS, const float* __restrict__ IndexKScale, __grid_constant__ const CUtensorMap IndexK_desc, __grid_constant__ const CUtensorMap IndexQ_desc, float* __restrict__ Logits, const float* __restrict__ Weights, int seq_len, int seq_len_kv);
extern "C" __global__ void __launch_bounds__(640, 1) mqa_attn_return_logits_kernel_kernel(const int* __restrict__ CuSeqLenKE, const int* __restrict__ CuSeqLenKS, const float* __restrict__ IndexKScale, __grid_constant__ const CUtensorMap IndexK_desc, __grid_constant__ const CUtensorMap IndexQ_desc, float* __restrict__ Logits, const float* __restrict__ Weights, int seq_len, int seq_len_kv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int cu_k_s_min = 0;
  int cu_k_e_max = 0;
  float weights[2];
  float index_k_scale_fragment[32];
  float s_reshaped[64];
  float logits[32];
  fp8_e4_t A_local[256];
  fp8_e4_t B_local[8];
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
    cu_k_s_min = 2147483647;
    cu_k_e_max = -2147483648;
    for (int bq_i = 0; bq_i < 2; ++bq_i) {
      int condval;
      if ((((((int)blockIdx.x) * 2) + bq_i) < seq_len)) {
        condval = CuSeqLenKS[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + ((int64_t)bq_i))];
      } else {
        condval = 0;
      }
      cu_k_s_min = min(cu_k_s_min, min(condval, seq_len_kv));
    }
    for (int bq_i_1 = 0; bq_i_1 < 2; ++bq_i_1) {
      int condval_1;
      if ((((((int)blockIdx.x) * 2) + bq_i_1) < seq_len)) {
        condval_1 = CuSeqLenKE[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + ((int64_t)bq_i_1))];
      } else {
        condval_1 = 0;
      }
      cu_k_e_max = max(cu_k_e_max, min(condval_1, seq_len_kv));
    }
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[6].expect_transaction(8192);
      tl::fence_proxy_async();
      tl::tma_load(IndexQ_desc, mbarrier[6], (&(((fp8_e4_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.x) * 128));
    }
    mbarrier[6].arrive();
    for (int nbn_i = 0; nbn_i < (((cu_k_e_max + 255) - cu_k_s_min) >> 8); ++nbn_i) {
      mbarrier[((nbn_i % 3) + 3)].wait((((nbn_i % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(nbn_i % 3)].expect_transaction(16384);
        tl::fence_proxy_async();
        tl::tma_load(IndexK_desc, mbarrier[(nbn_i % 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((nbn_i % 3) * 16384) + 8192)])), 0, ((nbn_i * 256) + cu_k_s_min));
      }
      tl::mbarrier_cp_async_arrive(mbarrier[(nbn_i % 3)]);
      mbarrier[(nbn_i % 3)].arrive();
    }
  } else {
    cu_k_s_min = 2147483647;
    cu_k_e_max = -2147483648;
    for (int bq_i_2 = 0; bq_i_2 < 2; ++bq_i_2) {
      int condval_2;
      if ((((((int)blockIdx.x) * 2) + bq_i_2) < seq_len)) {
        condval_2 = CuSeqLenKS[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + ((int64_t)bq_i_2))];
      } else {
        condval_2 = 0;
      }
      cu_k_s_min = min(cu_k_s_min, min(condval_2, seq_len_kv));
    }
    for (int bq_i_3 = 0; bq_i_3 < 2; ++bq_i_3) {
      int condval_3;
      if ((((((int)blockIdx.x) * 2) + bq_i_3) < seq_len)) {
        condval_3 = CuSeqLenKE[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + ((int64_t)bq_i_3))];
      } else {
        condval_3 = 0;
      }
      cu_k_e_max = max(cu_k_e_max, min(condval_3, seq_len_kv));
    }
    float2 condval_4;
    if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 8)) < seq_len)) {
      condval_4 = *(float2*)(Weights + (((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    } else {
      condval_4 = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    *(float2*)(weights + 0) = condval_4;
    mbarrier[6].wait(0);
    for (int nbn_i_1 = 0; nbn_i_1 < (((cu_k_e_max + 255) - cu_k_s_min) >> 8); ++nbn_i_1) {
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        float condval_5;
        if ((((((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min) < seq_len_kv) && (0 <= ((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min))) && (((((nbn_i_1 * 256) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min) < seq_len_kv))) {
          condval_5 = IndexKScale[((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + ((int64_t)cu_k_s_min))];
        } else {
          condval_5 = 0x0p+0f/*0.000000e+00*/;
        }
        index_k_scale_fragment[i] = condval_5;
      }
      mbarrier[(nbn_i_1 % 3)].wait(((nbn_i_1 % 6) / 3));
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        *(float2*)(s_reshaped + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      tl::__sync_thread_partial<3, 512>();
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_2 = 0; i_2 < 16; ++i_2) {
          tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)buf_dyn_shmem)[(((((((nbn_i_1 % 3) * 16384) + (i_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + 8192)])) + 0, A_local + (i_2 * 16));
        }
        tl::ptx_ldmatrix_x2((&(((fp8_e4_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 15) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])) + 0, B_local + 0);
        for (int i_3 = 0; i_3 < 16; ++i_3) {
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(s_reshaped + (i_3 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 16)), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
      mbarrier[((nbn_i_1 % 3) + 3)].arrive();
      #pragma unroll
      for (int i_4 = 0; i_4 < 64; ++i_4) {
        s_reshaped[i_4] = ((max(s_reshaped[i_4], 0x0p+0f/*0.000000e+00*/) * weights[(i_4 & 1)]) * index_k_scale_fragment[(i_4 >> 1)]);
      }
      tl::__sync_thread_partial<3, 512>();
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        logits[i_5] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          logits[i_5] = (logits[i_5] + s_reshaped[((i_5 * 2) + rv)]);
        }
        logits[i_5] = tl::AllReduce<tl::SumOp, 256, 32, 0, 512>::run_hopper(logits[i_5], (&(((float*)buf_dyn_shmem)[14336])));
        logits[i_5] = tl::AllReduce<tl::SumOp, 4, 1, 0, 512>::run_hopper(logits[i_5]);
      }
      if ((((((int)threadIdx.x) & 3) * 8) + ((((int)threadIdx.x) & 255) >> 5)) == 0) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (0 <= ((((nbn_i_1 * 256) + (i_6 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min)) {
            if (((((nbn_i_1 * 256) + (i_6 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + cu_k_s_min) < seq_len_kv) {
              if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 8)) < seq_len) {
                Logits[(((((((int64_t)nbn_i_1) * (int64_t)256) + (((int64_t)i_6) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + (((((int64_t)((int)blockIdx.x)) * (int64_t)2) + (((int64_t)((int)threadIdx.x)) >> (int64_t)8)) * ((int64_t)seq_len_kv))) + ((int64_t)cu_k_s_min))] = logits[i_6];
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
    
    // Allow up to 64 KiB of dynamic shared memory for this kernel. The
    // TileLang-generated implementation uses a reduction buffer at the end
    // of buf_dyn_shmem, so we must provision enough space for all segments.
    const int dyn_smem_bytes = 65536; // 64 KiB, enough for tiles + AllReduce buffer
    cudaError_t result_mqa_attn_return_logits_kernel_kernel = cudaFuncSetAttribute(
        mqa_attn_return_logits_kernel_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dyn_smem_bytes);
    if (result_mqa_attn_return_logits_kernel_kernel != cudaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE,
                 "Failed to set the allowed dynamic shared memory size to %d with error: %s",
                 dyn_smem_bytes, cudaGetErrorString(result_mqa_attn_return_logits_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(const fp8_e4_t* IndexQ,
                    const fp8_e4_t* IndexK,
                    const float* IndexKScale,
                    float* Logits,
                    const float* Weights,
                    const int* CuSeqLenKS,
                    const int* CuSeqLenKE,
                    int seq_len_kv, int seq_len,
                    int num_q_heads,
                    cudaStream_t stream=cudaStreamDefault) {
	CUtensorMap IndexK_desc;
	CUtensorMapDataType IndexK_desc_type= (CUtensorMapDataType)0;
	cuuint32_t IndexK_desc_tensorRank= 2;
	const void *IndexK_desc_globalAddress= IndexK;
	cuuint64_t IndexK_desc_globalDim[2]= {64,seq_len_kv};
	cuuint64_t IndexK_desc_globalStride[2]= {1,64};
	cuuint32_t IndexK_desc_boxDim[2]= {64,256};
	cuuint32_t IndexK_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave IndexK_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle IndexK_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion IndexK_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill IndexK_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult IndexK_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &IndexK_desc, IndexK_desc_type, IndexK_desc_tensorRank, const_cast<void*>(IndexK_desc_globalAddress), IndexK_desc_globalDim, IndexK_desc_globalStride + 1, IndexK_desc_boxDim, IndexK_desc_elementStrides, IndexK_desc_interleave, IndexK_desc_swizzle, IndexK_desc_l2Promotion, IndexK_desc_oobFill);

	if (IndexK_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor IndexK_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap IndexQ_desc;
	CUtensorMapDataType IndexQ_desc_type= (CUtensorMapDataType)0;
	cuuint32_t IndexQ_desc_tensorRank= 2;
	const void *IndexQ_desc_globalAddress= IndexQ;
	cuuint64_t IndexQ_desc_globalDim[2]= {64, (uint64_t)seq_len * (uint64_t)num_q_heads};
	cuuint64_t IndexQ_desc_globalStride[2]= {1,64};
	cuuint32_t IndexQ_desc_boxDim[2]= {64,128};
	cuuint32_t IndexQ_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave IndexQ_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle IndexQ_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion IndexQ_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill IndexQ_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult IndexQ_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &IndexQ_desc, IndexQ_desc_type, IndexQ_desc_tensorRank, const_cast<void*>(IndexQ_desc_globalAddress), IndexQ_desc_globalDim, IndexQ_desc_globalStride + 1, IndexQ_desc_boxDim, IndexQ_desc_elementStrides, IndexQ_desc_interleave, IndexQ_desc_swizzle, IndexQ_desc_l2Promotion, IndexQ_desc_oobFill);

	if (IndexQ_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor IndexQ_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	mqa_attn_return_logits_kernel_kernel<<<dim3((seq_len + 31) / 32, 1, 1), dim3(640, 1, 1), 65536, stream>>>(CuSeqLenKE, CuSeqLenKS, IndexKScale, IndexK_desc, IndexQ_desc, Logits, Weights, seq_len, seq_len_kv);
	TILELANG_CHECK_LAST_ERROR("mqa_attn_return_logits_kernel_kernel");

	return 0;
}

