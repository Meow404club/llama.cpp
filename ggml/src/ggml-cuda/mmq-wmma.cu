// RDNA3 (gfx11) f16-WMMA GEMM for quantized weights (Q4_K / IQ3_S), replacing the dp4a
// MMQ path for large batches. Opt-in via GGML_MMQ_WMMA.
// Fragment layouts derived empirically on gfx1100 (wave32, 16x16x16):
//   A frag (1st arg): row m carried by lane m (m even, half 0) / lane 16+m (m odd, half 1);
//                     16 f16 per lane = the m-th row of the 16-wide k-slice (j == k).
//   B frag (2nd arg): lane l holds column n = l%16 (lanes n and n+16 duplicate the data;
//                     half-0 lanes feed even-m accumulators, half-1 lanes feed odd-m ones).
//   D/C: lane l holds D[m = 2r + l/16][n = l%16] in 8 f32 registers r = 0..7.
#include "common.cuh"
#include "mmq-wmma.cuh"

#if defined(GGML_USE_HIP)
// The wmma builtins below are gfx11-only; this fork only builds gfx11 targets on HIP.
// Host entry points must NOT be behind per-arch guards: the host compile pass defines
// no __gfx*__ macros and would otherwise bind to the always-false stubs (that bug cost
// a full afternoon of "identical performance" false negatives).
#include "ggml-common.h"

typedef _Float16 wmma_half;
typedef wmma_half wmma_half16 __attribute__((ext_vector_type(16)));
typedef float      wmma_acc8  __attribute__((ext_vector_type(8)));

#define WMMMA_KC 64       // k elements staged per chunk
#define WMMMA_TM 64       // workgroup m tile
#define WMMMA_TN 128      // workgroup n tile
#define WMMMA_LDSTRIDE 72 // smem row stride in halves: 16B-aligned rows, 4-word bank rotation

// ---- dequant functors: 16 consecutive elements starting at j0 (j0 % 16 == 0) ----

static __device__ __forceinline__ void wmma_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

struct WQ4K {
    static constexpr ggml_type type_id = GGML_TYPE_Q4_K;
    using block_t = block_q4_K;
    static __device__ __forceinline__ void frag16(const block_t * __restrict__ blk, const int j0, wmma_half * __restrict__ out) {
        const int g = j0 >> 5;
        const float dall = __low2half(blk->dm);
        const float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        wmma_get_scale_min_k4(g, blk->scales, sc, m);
        const float d = dall * sc;
        const float mf = dmin * m;
        const uint8_t * qs = blk->qs + (j0 >> 1);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t byte = qs[k];
            out[2*k + 0] = (wmma_half)(d * (byte & 0xF) - mf);
            out[2*k + 1] = (wmma_half)(d * (byte >> 4) - mf);
        }
    }
};

struct WIQ3S {
    static constexpr ggml_type type_id = GGML_TYPE_IQ3_S;
    using block_t = block_iq3_s;
    static __device__ __forceinline__ void frag16(const block_t * __restrict__ blk, const int j0, wmma_half * __restrict__ out) {
        const int ib = j0 >> 5;
        const int il = (j0 >> 3) & 3;
        const float d = (float)blk->d * (1 + 2*((blk->scales[ib >> 1] >> 4*(ib & 1)) & 0xf));
        const uint32_t qh = blk->qh[ib];
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int l = il + g;
            const uint32_t g1 = iq3s_grid[blk->qs[8*ib + 2*l + 0] | ((qh << (8 - 2*l)) & 256)];
            const uint32_t g2 = iq3s_grid[blk->qs[8*ib + 2*l + 1] | ((qh << (7 - 2*l)) & 256)];
            const uint8_t s  = blk->signs[4*ib + l];
            const int8_t * G1 = (const int8_t *) &g1;
            const int8_t * G2 = (const int8_t *) &g2;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[8*g + j + 0] = (wmma_half)(d * G1[j] * ((s & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
                out[8*g + j + 4] = (wmma_half)(d * G2[j] * ((s & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
            }
        }
    }
};

// dst[m][n] = sum_k x[m][k] * w[n][k]; w rows = blocks of 256 k, x = f32 [M][K].
// Workgroup: 256 threads (8 wave32), tile 64m x 128n; warp tile 16m x 64n (wy = warp>>1, wx = warp&1).
template<bool need_m_guard, bool need_n_guard, typename DQ>
static __global__ void __launch_bounds__(256)
mul_mat_wmma_kernel(const char * __restrict__ wq,  const size_t w_stride,
                    const char * __restrict__ xf,  const size_t x_stride,
                    char * __restrict__ dst,       const size_t dst_stride_m, const size_t dst_stride_n,
                    const int M, const int N, const int K) {
    __shared__ wmma_half xs[WMMMA_TM][WMMMA_LDSTRIDE];
    __shared__ wmma_half ws[WMMMA_TN][WMMMA_LDSTRIDE];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wy   = warp >> 1;
    const int wx   = warp & 1;

    const int mw = blockIdx.x * WMMMA_TM;
    const int nw = blockIdx.y * WMMMA_TN;

    const int m0 = mw + wy*16;                 // wave tile base row
    const int n0 = nw + wx*64;                 // wave tile base col

    const char * x_base = xf + (size_t)mw * x_stride;

    wmma_acc8 acc[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;

    float4   pre_x[4];
    wmma_half16 pre_w[2];
    const float4 * xp = nullptr;
    const char * wp[2] = {nullptr, nullptr};
    int wstep[2] = {0, 0}; // uniform per-thread counter: j0 = (wstep & 15)*16, block = wstep >> 4

    auto load_regs_ptr = [&]() {
        const int r0 = tid >> 2;
        const int xoff4 = (tid & 3) << 2; // this thread's 16-column slot, in float4 units
        if (need_m_guard ? (mw + r0 < M) : true) {
            pre_x[0] = xp[xoff4 + 0]; pre_x[1] = xp[xoff4 + 1]; pre_x[2] = xp[xoff4 + 2]; pre_x[3] = xp[xoff4 + 3];
        } else {
            const float4 z = {0.f, 0.f, 0.f, 0.f};
            pre_x[0] = pre_x[1] = pre_x[2] = pre_x[3] = z;
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*256;
            const int r = task >> 2;
            if (need_n_guard ? (nw + r < N) : true) {
                DQ::frag16((const typename DQ::block_t *)(wp[rep]) + (wstep[rep] >> 4), (wstep[rep] & 15)*16, (wmma_half *)&pre_w[rep]);
            } else {
                pre_w[rep] = (wmma_half16)0.f;
            }
        }
        xp += WMMMA_KC / 4;            // advance one k-chunk (float4 units)
        wstep[0] += WMMMA_KC / 16;     // +4 per 64-k chunk (uniform for all threads)
        wstep[1] += WMMMA_KC / 16;
    };

    auto store_regs = [&]() {
        {
            const int r  = tid >> 2;
            const int cb = (tid & 3) << 4;
            wmma_half * d = &xs[r][cb];
            _Float16 x0 = (_Float16)pre_x[0].x, x1 = (_Float16)pre_x[0].y, x2 = (_Float16)pre_x[0].z, x3 = (_Float16)pre_x[0].w;
            _Float16 x4 = (_Float16)pre_x[1].x, x5 = (_Float16)pre_x[1].y, x6 = (_Float16)pre_x[1].z, x7 = (_Float16)pre_x[1].w;
            _Float16 x8 = (_Float16)pre_x[2].x, x9 = (_Float16)pre_x[2].y, xa = (_Float16)pre_x[2].z, xb = (_Float16)pre_x[2].w;
            _Float16 xc = (_Float16)pre_x[3].x, xd = (_Float16)pre_x[3].y, xe = (_Float16)pre_x[3].z, xf = (_Float16)pre_x[3].w;
            d[0] = x0; d[1] = x1; d[2] = x2; d[3] = x3;
            d[4] = x4; d[5] = x5; d[6] = x6; d[7] = x7;
            d[8] = x8; d[9] = x9; d[10] = xa; d[11] = xb;
            d[12] = xc; d[13] = xd; d[14] = xe; d[15] = xf;
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*256;
            const int r  = task >> 2;
            const int ks = task & 3;
            *(wmma_half16 *)&ws[r][ks*16] = pre_w[rep];
        }
    };

    // pointer-increment staging: all addresses advance by fixed strides per chunk
    {
        const int r0 = tid >> 2;
        xp = (const float4 *)(x_base + (size_t)r0 * x_stride);
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*256;
            const int r = task >> 2;
            wp[rep] = (const char *)(wq + (size_t)(nw + r) * w_stride);
            wstep[rep] = task & 3;
        }
    }
    load_regs_ptr(); // chunk 0 - without this the first store_regs() writes uninit registers
    for (int k0 = 0; k0 < K; k0 += WMMMA_KC) {
        __syncthreads();
        store_regs();
        if (k0 + WMMMA_KC < K) load_regs_ptr();
        __syncthreads();


        const wmma_half16 * arow_p = (const wmma_half16 *)&xs[wy*16 + (lane & 15)][0];
        const wmma_half16 * brow_p0 = (const wmma_half16 *)&ws[wx*64 + 0*16 + (lane & 15)][0];
        const wmma_half16 * brow_p1 = (const wmma_half16 *)&ws[wx*64 + 1*16 + (lane & 15)][0];
        const wmma_half16 * brow_p2 = (const wmma_half16 *)&ws[wx*64 + 2*16 + (lane & 15)][0];
        const wmma_half16 * brow_p3 = (const wmma_half16 *)&ws[wx*64 + 3*16 + (lane & 15)][0];
        #pragma unroll
        for (int ks = 0; ks < WMMMA_KC / 16; ++ks) {
            const wmma_half16 a = arow_p[ks];
            const wmma_half16 b0 = brow_p0[ks];
            const wmma_half16 b1 = brow_p1[ks];
            const wmma_half16 b2 = brow_p2[ks];
            const wmma_half16 b3 = brow_p3[ks];
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[3]);
        }
    }

    // store: lane holds D[m = m0 + 2r + (lane>=16)][n = n0 + t*16 + lane&15].
    // ggml MUL_MAT dst is [ne0 = N, ne1 = M]: feature n advances by nb[0], token m by nb[1].
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int n = n0 + t*16 + (lane & 15);
        if (need_n_guard ? (n < N) : true) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = m0 + 2*r + (lane >= 16);
                if (need_m_guard ? (m < M) : true) {
                    *(float *)(dst + (size_t)n * dst_stride_m + (size_t)m * dst_stride_n) = acc[t][r];
                }
            }
        }
    }
}


template<typename DQ, bool mg, bool ng>
static void launch_wmma(cudaStream_t stream, const char * wq, size_t w_stride, const char * xf, size_t x_stride,
                        char * dst, size_t dsm, size_t dsn, int M, int N, int K) {
    const dim3 block(256);
    const dim3 grid((M + WMMMA_TM - 1) / WMMMA_TM, (N + WMMMA_TN - 1) / WMMMA_TN);
    mul_mat_wmma_kernel<mg, ng, DQ><<<grid, block, 0, stream>>>(wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
}

// ---- small-batch variant (M <= 16, decode / speculative verify batches) ----
// 128 threads = 4 wave32; n tile 64 (one 16-wide wmma n-segment per warp); m is a single
// 16-wide wmma segment with batch rows zero-padded in xs. Each CTA streams 64 contiguous
// weight rows (like MMVQ's row-major streaming) while dequantizing every weight exactly
// once (like MMQ), so neither the ALU-per-batch-row nor the strided-tile DRAM pattern of
// the two stock kernels applies.
#define WMMAS_TN 64

template<bool need_n_guard, typename DQ>
static __global__ void __launch_bounds__(128)
mul_mat_wmma_small_kernel(const char * __restrict__ wq,  const size_t w_stride,
                          const char * __restrict__ xf,  const size_t x_stride,
                          char * __restrict__ dst,       const size_t dst_stride_m, const size_t dst_stride_n,
                          const int M, const int N, const int K) {
    __shared__ wmma_half xs[16][WMMMA_LDSTRIDE];
    __shared__ wmma_half ws[WMMAS_TN][WMMMA_LDSTRIDE];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int nw = blockIdx.x * WMMAS_TN;
    const int n0 = nw + warp*16;

    wmma_acc8 acc;
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.f;

    const size_t x_stride4 = x_stride / sizeof(float4);
    const float4 * xk = nullptr;                 // start of the current 64-wide k chunk
    float4   pre_x[2];                           // one x task per thread: 8 f32 (2 float4)
    wmma_half16 pre_w[2];
    const char * wp[2] = {nullptr, nullptr};
    int wstep[2] = {0, 0};                       // j0 = (wstep & 15)*16, block = wstep >> 4

    auto load_regs_ptr = [&]() {
        {
            const int r  = tid >> 3;             // x row (0..15)
            const int co = (tid & 7) * 2;        // 8-f32 slot within the chunk, float4 units
            if (r < M) {
                const float4 * src = xk + (size_t)r * x_stride4 + co;
                pre_x[0] = src[0];
                pre_x[1] = src[1];
            } else {
                const float4 z = {0.f, 0.f, 0.f, 0.f};
                pre_x[0] = pre_x[1] = z;
            }
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*128;
            const int r = task >> 2;             // weight row (0..63)
            if (need_n_guard ? (nw + r < N) : true) {
                DQ::frag16((const typename DQ::block_t *)(wp[rep]) + (wstep[rep] >> 4), (wstep[rep] & 15)*16, (wmma_half *)&pre_w[rep]);
            } else {
                pre_w[rep] = (wmma_half16)0.f;
            }
        }
        xk += WMMMA_KC / 4;
        wstep[0] += WMMMA_KC / 16;
        wstep[1] += WMMMA_KC / 16;
    };

    auto store_regs = [&]() {
        {
            const int r  = tid >> 3;
            const int cb = (tid & 7) << 3;
            wmma_half * d = &xs[r][cb];
            d[0] = (wmma_half)pre_x[0].x; d[1] = (wmma_half)pre_x[0].y;
            d[2] = (wmma_half)pre_x[0].z; d[3] = (wmma_half)pre_x[0].w;
            d[4] = (wmma_half)pre_x[1].x; d[5] = (wmma_half)pre_x[1].y;
            d[6] = (wmma_half)pre_x[1].z; d[7] = (wmma_half)pre_x[1].w;
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*128;
            const int r  = task >> 2;
            const int ks = task & 3;
            *(wmma_half16 *)&ws[r][ks*16] = pre_w[rep];
        }
    };

    {
        xk = (const float4 *)xf;
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*128;
            const int r = task >> 2;
            wp[rep] = (const char *)(wq + (size_t)(nw + r) * w_stride);
            wstep[rep] = task & 3;
        }
    }
    load_regs_ptr(); // chunk 0 - without this the first store_regs() writes uninit registers
    for (int k0 = 0; k0 < K; k0 += WMMMA_KC) {
        __syncthreads();
        store_regs();
        if (k0 + WMMMA_KC < K) load_regs_ptr();
        __syncthreads();

        const wmma_half16 * arow_p = (const wmma_half16 *)&xs[lane & 15][0];
        const wmma_half16 * brow_p = (const wmma_half16 *)&ws[warp*16 + (lane & 15)][0];
        #pragma unroll
        for (int ks = 0; ks < WMMMA_KC / 16; ++ks) {
            const wmma_half16 a = arow_p[ks];
            const wmma_half16 b = brow_p[ks];
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
    }

    // lane holds D[m = 2r + (lane>=16)][n = n0 + lane&15]
    const int n = n0 + (lane & 15);
    if (need_n_guard ? (n < N) : true) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int m = 2*r + (lane >= 16);
            if (m < M) {
                *(float *)(dst + (size_t)n * dst_stride_m + (size_t)m * dst_stride_n) = acc[r];
            }
        }
    }
}

template<typename DQ>
static void mul_mat_wmma_small_dispatch(cudaStream_t stream, const char * wq, size_t w_stride, const char * xf, size_t x_stride,
                                        char * dst, size_t dsm, size_t dsn, int M, int N, int K) {
    const dim3 block(128);
    const dim3 grid((N + WMMAS_TN - 1) / WMMAS_TN);
    if ((N % WMMAS_TN) != 0) {
        mul_mat_wmma_small_kernel<true, DQ><<<grid, block, 0, stream>>>(wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
    } else {
        mul_mat_wmma_small_kernel<false, DQ><<<grid, block, 0, stream>>>(wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
    }
}


template<typename DQ>
static void mul_mat_wmma_dispatch(cudaStream_t stream, const char * wq, size_t w_stride, const char * xf, size_t x_stride,
                                  char * dst, size_t dsm, size_t dsn, int M, int N, int K) {
    const bool mg = (M % WMMMA_TM) != 0;
    const bool ng = (N % WMMMA_TN) != 0;
    if (mg && ng)      launch_wmma<DQ, true,  true >(stream, wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
    else if (mg)       launch_wmma<DQ, true,  false>(stream, wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
    else if (ng)       launch_wmma<DQ, false, true >(stream, wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
    else               launch_wmma<DQ, false, false>(stream, wq, w_stride, xf, x_stride, dst, dsm, dsn, M, N, K);
}

void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t M = src1->ne[1];
    const int64_t N = src0->ne[1];
    const int64_t K = src0->ne[0];

    cudaStream_t stream = ctx.stream();


    if (src0->type == GGML_TYPE_Q4_K) {
        if (M <= 16) {
            mul_mat_wmma_small_dispatch<WQ4K>(stream, (const char *)src0->data, src0->nb[1], (const char *)src1->data, src1->nb[1],
                                              (char *)dst->data, dst->nb[0], dst->nb[1], M, N, K);
        } else {
            mul_mat_wmma_dispatch<WQ4K>(stream, (const char *)src0->data, src0->nb[1], (const char *)src1->data, src1->nb[1],
                                        (char *)dst->data, dst->nb[0], dst->nb[1], M, N, K);
        }
    } else {
        if (M <= 16) {
            mul_mat_wmma_small_dispatch<WIQ3S>(stream, (const char *)src0->data, src0->nb[1], (const char *)src1->data, src1->nb[1],
                                               (char *)dst->data, dst->nb[0], dst->nb[1], M, N, K);
        } else {
            mul_mat_wmma_dispatch<WIQ3S>(stream, (const char *)src0->data, src0->nb[1], (const char *)src1->data, src1->nb[1],
                                         (char *)dst->data, dst->nb[0], dst->nb[1], M, N, K);
        }
    }
}

bool ggml_cuda_should_use_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc, const int64_t ne11) {
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) return false;
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_IQ3_S) return false;
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float)) return false;
    if (!ggml_is_contiguous(src0)) return false;
    if (src0->ne[0] % 256 != 0) return false;
    // ggml dst layout [N, M]: feature stride nb[0], token stride nb[1] == N*4 for plain outputs.
    if (dst->nb[0] != sizeof(float) || dst->nb[1] != sizeof(float)*src0->ne[1]) return false;
    if (ne11 < 64) {
        // small-batch variant: decode / speculative verify batches. The env value is the
        // batch floor (GGML_WMMA_SMALL=1 also covers greedy single-row decode for testing).
        static const int small_min = []() {
            const char * e = getenv("GGML_WMMA_SMALL");
            return e ? atoi(e) : 0;
        }();
        if (small_min < 1 || ne11 < small_min || ne11 > 16) return false;
        // bisect switch: 1 = both types, 4 = Q4_K only, 3 = IQ3_S only
        static const int small_types = []() {
            const char * e = getenv("GGML_WMMA_SMALL_TYPES");
            return e ? atoi(e) : 1;
        }();
        if (small_types == 4 && src0->type != GGML_TYPE_Q4_K) return false;
        if (small_types == 3 && src0->type != GGML_TYPE_IQ3_S) return false;
        return true;
    }
    static const bool enabled = getenv("GGML_MMQ_WMMA") != nullptr;
    return enabled;
}

#else // non-HIP fallback stubs

void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("wmma mul_mat called on unsupported target");
}

bool ggml_cuda_should_use_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc, const int64_t ne11) {
    return false;
}

#endif
