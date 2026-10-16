#pragma once

#include "common.cuh"
#include "ggml.h"

// Chunked (WY / UT transform) gated delta net. Handles the scalar-gate case only; callers must
// check ggml_cuda_gdn_chunked_supported() and fall back to the token-by-token kernel otherwise.
// Ported from AMD-Ecosystem/llama.cpp gfx11 (PR #54, MIT/upstream licensed) with the RDNA3
// (gfx1100) gate and the PR #79 CU-mode dispatch added.
struct ggml_cuda_gdn_chunked_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state_in;
    float *       dst;
    float *       state_out;

    int64_t S_v;
    int64_t H;
    int64_t n_tokens;
    int64_t n_seqs;

    // strides in elements
    int64_t sq1, sq2, sq3;
    int64_t sv1, sv2, sv3;
    int64_t sb1, sb2, sb3;

    int64_t neqk1;   // q/k head count, for the GQA head mapping
    int64_t rq3;     // sequences per q/k sequence

    int64_t state_seq_stride = 0; // fused-cache write: per-sequence stride (0 = plain layout)

    float scale;
};

bool ggml_cuda_gdn_chunked_supported(bool kda, bool keep_rs, int64_t S_v, int64_t n_tokens);

void ggml_cuda_gdn_chunked(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunked_args & args);
