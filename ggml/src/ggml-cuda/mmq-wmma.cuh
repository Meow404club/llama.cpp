#include "common.cuh"

// RDNA3 (gfx11) f16-WMMA GEMM path for quantized weights (currently IQ3_S), opt-in via GGML_MMQ_WMMA.
void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_should_use_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc, const int64_t ne11);
