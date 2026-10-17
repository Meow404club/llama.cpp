#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

struct ggml_cuda_rms_batch_item_t {
    const float * src;
    float       * dst;
    const float * mul;
    float         scale_out;
    float         eps;
    int           ncols;
    int           nrows;
    int           row_beg;
};

struct ggml_cuda_rms_batch_t {
    ggml_cuda_rms_batch_item_t it[8];
    int n;
};

void ggml_cuda_op_rms_norm_batch(ggml_backend_cuda_context & ctx, const ggml_cuda_rms_batch_t & b, cudaStream_t stream);
void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
