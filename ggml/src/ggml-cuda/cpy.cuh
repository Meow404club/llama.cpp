#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// meow: batched copy of a run of CPY nodes (see ggml-cuda.cu try_fuse). 2D-pitchable chunks
// (src views are strided row slices of conv_input); fully contiguous degenerates to height 1.
#define MEOW_CPY_BATCH_MAX 16
struct ggml_cuda_cpy_batch_chunk {
    const char * src;
    char       * dst;
    uint32_t     width;   // contiguous row bytes
    uint32_t     height;  // rows
    uint32_t     spitch;
    uint32_t     dpitch;
};
struct ggml_cuda_cpy_batch_params2 {
    ggml_cuda_cpy_batch_chunk c[MEOW_CPY_BATCH_MAX];
    int n;
};
bool ggml_cuda_cpy_batch_chunk_of(const ggml_tensor * s, const ggml_tensor * d, ggml_cuda_cpy_batch_chunk & c);
void ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n);
