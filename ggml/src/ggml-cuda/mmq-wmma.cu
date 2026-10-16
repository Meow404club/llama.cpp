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
        const int g = j0 >> 5; // 8 groups of 32 (verified: 12B scales = 8x6bit; the bq8_offset/2 in vecdot combines with jm/hi bit tricks to index all 8)
        const float dall = __low2half(blk->dm);
        const float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        wmma_get_scale_min_k4(g, blk->scales, sc, m);
        const float d = dall * sc;
        const float mf = dmin * m;
        // meow: nibble layout is BLOCKED, not interleaved (dequantize_row_q4_K):
        // within each 64-element sub-block, elements 0..31 use the LOW nibbles of bytes 0..31,
        // elements 32..63 use the HIGH nibbles of the same bytes. The old interleaved read
        // (out[2k]=low, out[2k+1]=high) produced maxerr ~36 vs the MMVQ reference.
        const int r0  = (j0 & 31) + ((g & 1) << 5);
        const bool hi = r0 >= 32;
        const uint8_t * qs = blk->qs + (g >> 1)*32 + (r0 & 31);
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t byte = qs[k];
            out[k] = (wmma_half)(hi ? (d * (byte >> 4) - mf) : (d * (byte & 0xF) - mf));
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
#define WMMAS_TN 32 // meow: 32 cols/CTA -> 160 CTAs at N=5120, fills 84 CUs x2 (64 threads each)

template<bool need_n_guard, typename DQ>
static __global__ void __launch_bounds__(64)
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
    float4   pre_x[4];                           // one x task per thread: 16 f32 (4 float4)
    wmma_half16 pre_w[2];
    const char * wp[2] = {nullptr, nullptr};
    int wstep[2] = {0, 0};                       // j0 = (wstep & 15)*16, block = wstep >> 4

    auto load_regs_ptr = [&]() {
        {
            const int r  = tid >> 2;             // x row (0..15)
            const int co = (tid & 3) * 4;        // 16-f32 slot within the chunk, float4 units
            if (r < M) {
                const float4 * src = xk + (size_t)r * x_stride4 + co;
                pre_x[0] = src[0]; pre_x[1] = src[1]; pre_x[2] = src[2]; pre_x[3] = src[3];
            } else {
                const float4 z = {0.f, 0.f, 0.f, 0.f};
                pre_x[0] = pre_x[1] = pre_x[2] = pre_x[3] = z;
            }
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*64;
            const int r = task >> 2;             // weight row (0..31)
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
            const int r  = tid >> 2;
            const int cb = (tid & 3) << 4;
            wmma_half * d = &xs[r][cb];
            d[0]  = (wmma_half)pre_x[0].x; d[1]  = (wmma_half)pre_x[0].y;
            d[2]  = (wmma_half)pre_x[0].z; d[3]  = (wmma_half)pre_x[0].w;
            d[4]  = (wmma_half)pre_x[1].x; d[5]  = (wmma_half)pre_x[1].y;
            d[6]  = (wmma_half)pre_x[1].z; d[7]  = (wmma_half)pre_x[1].w;
            d[8]  = (wmma_half)pre_x[2].x; d[9]  = (wmma_half)pre_x[2].y;
            d[10] = (wmma_half)pre_x[2].z; d[11] = (wmma_half)pre_x[2].w;
            d[12] = (wmma_half)pre_x[3].x; d[13] = (wmma_half)pre_x[3].y;
            d[14] = (wmma_half)pre_x[3].z; d[15] = (wmma_half)pre_x[3].w;
        }
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*64;
            const int r  = task >> 2;
            const int ks = task & 3;
            *(wmma_half16 *)&ws[r][ks*16] = pre_w[rep];
        }
    };

    {
        xk = (const float4 *)xf;
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int task = tid + rep*64;
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

// ---- v3: direct-read small-batch variant - no smem, no __syncthreads ----
// Each warp owns a 16-column output segment; lane (lane&15) feeds A row m (x row) and
// B column n0+(lane%16) (weight row), both dequantized straight from global into
// registers. The smem staging round trip of the v1/v2 kernels is pure overhead in the
// bandwidth-bound small-batch regime (measured: MMVQ's direct-read structure wins).
template<bool need_n_guard, typename DQ>
static __global__ void __launch_bounds__(128)
mul_mat_wmma_small_direct(const char * __restrict__ wq,  const size_t w_stride,
                          const char * __restrict__ xf,  const size_t x_stride,
                          char * __restrict__ dst,       const size_t dst_stride_m, const size_t dst_stride_n,
                          const int M, const int N, const int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int nw = blockIdx.x * 64;
    const int n0 = nw + warp*16;

    wmma_acc8 acc;
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.f;

    const int mrow = lane & 15;                    // this lane's A row (0..15)
    const int wrow = n0 + (lane & 15);             // this lane's B column = weight row
    const float * xr = (const float *)(xf) + (size_t)mrow * (x_stride / sizeof(float));
    const char * wr = (const char *)(wq + (size_t)wrow * w_stride);

    wmma_half16 a_cur, b_cur, a_nxt, b_nxt;

    auto load_k = [&](const int k0, wmma_half16 & a, wmma_half16 & b) {
        if (mrow < M) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) ((wmma_half *)&a)[i] = (wmma_half)xr[k0 + i];
        } else {
            a = (wmma_half16)0.f;
        }
        if (need_n_guard ? (wrow < N) : true) {
            DQ::frag16((const typename DQ::block_t *)wr + (k0 >> 8), k0 & 255, (wmma_half *)&b);
        } else {
            b = (wmma_half16)0.f;
        }
    };

    load_k(0, a_cur, b_cur);
    for (int k0 = 0; k0 < K; k0 += 16) {
        if (k0 + 16 < K) load_k(k0 + 16, a_nxt, b_nxt);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_cur, b_cur, acc);
        a_cur = a_nxt;
        b_cur = b_nxt;
    }

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
    // meow: NT32 smem variant measured best of the small kernels (pp8 146 vs 143 NT64 / 100 direct);
    // still 12% behind MMVQ (165), so GGML_WMMA_SMALL stays opt-in.
    const dim3 block(64);
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

// ---- v4: large-batch path spliced from github.com/adelj88/rocm_wmma_gemm (MIT, (c) 2024 Adel Johar) ----
// The upstream gemm_impl skeleton (double-buffered LDS, register-staged prefetch, snake+XOR tile
// mapping, s_setprio'd wmma fold) with the B side replaced by a Q4_K dequant fragment that reads
// ggml weight blocks directly. Tuned config on gfx1100: 1x4 warps, warp tile 8x2 (wide-A),
// k_slices=1 (block_k 16), double buffer, swizzle 8, 256-bit loads -> 65.4 TF @ M2048/N5120/K5120
// with C row-major (llama dst layout), vs ~62 TF dp4a. A = x must be pre-staged as f16 col-major
// [K][M] by x_stage_kernel (0.095 ms at 2048x5120, ~296/N % of GEMM time -> gate by N).
namespace rwg {

enum class m_layout { row_major, col_major };
enum class m_input  { matrix_a, matrix_b };

template<class T> struct type_selector { using type = T; };
template<> struct type_selector<half>  { using type = _Float16; };

constexpr int wmma_tile = 16;
constexpr int warp_size = 32;

static constexpr unsigned buffer_rsrc_config = 0x31004000u;

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr, unsigned num_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), 0, num_bytes, static_cast<int>(buffer_rsrc_config));
}

template<typename V>
static __device__ __forceinline__ V buffer_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset) {
    if constexpr(sizeof(V) == 32) {
        V     result;
        char* res_ptr = reinterpret_cast<char*>(&result);
        using part_t  = float __attribute__((ext_vector_type(4)));
        part_t raw0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
        part_t raw1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset + 16, 0, 0);
        __builtin_memcpy(res_ptr, &raw0, 16);
        __builtin_memcpy(res_ptr + 16, &raw1, 16);
        return result;
    } else if constexpr(sizeof(V) == 16) {
        using payload_t = float __attribute__((ext_vector_type(4)));
        payload_t raw = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
        V result; __builtin_memcpy(&result, &raw, 16); return result;
    } else if constexpr(sizeof(V) == 8) {
        using payload_t = float __attribute__((ext_vector_type(2)));
        payload_t raw = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
        V result; __builtin_memcpy(&result, &raw, 8); return result;
    } else {
        static_assert(sizeof(V) == 4);
        float raw = __builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0);
        V result; __builtin_memcpy(&result, &raw, 4); return result;
    }
}

template<typename V>
static __device__ __forceinline__ void buffer_store(__amdgpu_buffer_rsrc_t rsrc, int byte_offset, const V& value) {
    if constexpr(sizeof(V) == 32) {
        using part_t = unsigned int __attribute__((ext_vector_type(4)));
        part_t      p0, p1;
        const char* val_ptr = reinterpret_cast<const char*>(&value);
        __builtin_memcpy(&p0, val_ptr, 16);
        __builtin_memcpy(&p1, val_ptr + 16, 16);
        __builtin_amdgcn_raw_buffer_store_b128(p0, rsrc, byte_offset, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b128(p1, rsrc, byte_offset + 16, 0, 0);
    } else if constexpr(sizeof(V) == 16) {
        using payload_t = unsigned int __attribute__((ext_vector_type(4)));
        payload_t p; __builtin_memcpy(&p, &value, 16);
        __builtin_amdgcn_raw_buffer_store_b128(p, rsrc, byte_offset, 0, 0);
    } else if constexpr(sizeof(V) == 8) {
        using payload_t = unsigned int __attribute__((ext_vector_type(2)));
        payload_t p; __builtin_memcpy(&p, &value, 8);
        __builtin_amdgcn_raw_buffer_store_b64(p, rsrc, byte_offset, 0, 0);
    } else {
        static_assert(sizeof(V) == 4);
        unsigned int p; __builtin_memcpy(&p, &value, 4);
        __builtin_amdgcn_raw_buffer_store_b32(p, rsrc, byte_offset, 0, 0);
    }
}

template<class B, int W, bool IS_POW2 = ((W & (W - 1)) == 0)>
struct load_vector { using type = B __attribute__((ext_vector_type(W))); };
template<class B, int W>
struct load_vector<B, W, false> { struct type { B data[W]; }; };

// fragment / load_matrix / store_matrix / wmma: upstream fragment.hpp + wmma.hpp (element-wise
// LDS addressing; the gfx11 fragment layouts match the ones derived at the top of this file).
template<class T, int TILE>
class fragment {
public:
    static constexpr int NEW_TILE = (sizeof(T) == sizeof(float)) ? TILE / 2 : TILE;
    using type     = typename type_selector<T>::type;
    using frag_vec = type __attribute__((ext_vector_type(NEW_TILE)));
    __device__ __forceinline__ frag_vec& get() { return _fragment; }
    __device__ __forceinline__ const frag_vec& get() const { return _fragment; }
private:
    frag_vec _fragment = {};
};

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major), void>::type {
    const T* tmp = data;
    auto& f = frag.get();
    #pragma unroll
    for (int i = 0; i < TILE; ++i) { f[i] = tmp[0]; tmp += M; }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major), void>::type {
    const T* tmp = data;
    auto& f = frag.get();
    #pragma unroll
    for (int i = 0; i < TILE; ++i) { f[i] = tmp[0]; ++tmp; }
}

// C store: lane l holds D[m = row + 2i][n = col] for i = 0..7; the lane-dependent half-warp
// offset (l/16, l%16) is folded into the caller's row/col arguments, matching upstream.
template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && !USE_SHARED, void>::type {
    auto& f = frag.get();
    #pragma unroll
    for (int i = 0; i < TILE / 2; ++i) {
        if ((row + 2*i) < M && col < N) data[(row + 2*i) * N + col] = f[i];
    }
}

template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && USE_SHARED, void>::type {
    auto& f = frag.get();
    #pragma unroll
    for (int i = 0; i < TILE / 2; ++i) {
        data[(row + 2*i) * N + col] = f[i];
    }
}

template<class T1, class T2, int TILE>
__device__ __forceinline__ auto wmma(fragment<T1, TILE>& frag1, fragment<T1, TILE>& frag2, fragment<T2, TILE>& frag3) ->
    typename std::enable_if<(std::is_same<T1, half>::value && std::is_same<T2, float>::value), void>::type {
    frag3.get() = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag1.get(), frag2.get(), frag3.get());
}

// global->register staging for the A side (upstream load.hpp, col-major path; our config has
// 1 vector per thread so the checked/remainder paths constant-fold away).
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
class prefetch_fragment {
    static constexpr int contig_dim   = (ACCESS == m_layout::row_major) ? BLOCK_N : BLOCK_M;
    static constexpr int iter_dim     = (ACCESS == m_layout::row_major) ? BLOCK_M : BLOCK_N;
    static constexpr int contig_bytes = contig_dim * sizeof(T);
    static constexpr int max_bytes    = MAX_BITS / 8;

    static constexpr int vector_bytes = []() constexpr {
        constexpr auto fits = [](int vb) constexpr -> bool { return vb <= max_bytes && (contig_bytes % vb) == 0; };
        if constexpr(fits(32)) return 32;
        else if constexpr(fits(16)) return 16;
        else if constexpr(fits(8)) return 8;
        else if constexpr(fits(4)) return 4;
        else return 2;
    }();

    static constexpr int vector_width = vector_bytes / sizeof(T);

    static constexpr int total_elements     = BLOCK_M * BLOCK_N;
    static constexpr int total_vectors      = total_elements / vector_width;
    static constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iters   = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iters    = vectors_per_thread - guaranteed_iters;

    using base_type   = typename type_selector<T>::type;
    using vector_type = typename load_vector<base_type, vector_width>::type;

    vector_type regs[vectors_per_thread];

    static constexpr int step_elems = BLOCK_SIZE * vector_width;
    static constexpr int iter_inc   = step_elems / contig_dim;
    static constexpr int off_inc    = step_elems % contig_dim;

    __amdgpu_buffer_rsrc_t rsrc_;
    const T*               base_ptr_;

public:
    __device__ __forceinline__ prefetch_fragment(const T* base, unsigned alloc_elems)
        : rsrc_(make_buffer_rsrc(base, alloc_elems * static_cast<unsigned>(sizeof(T)))), base_ptr_(base) {}

    __device__ __forceinline__ void advance_global(int lead, int& iter, int& off, int& curr) {
        curr += iter_inc * lead;
        iter += iter_inc;
        if constexpr(off_inc != 0) {
            off += off_inc;
            curr += off_inc;
            if(off >= contig_dim) { off -= contig_dim; iter += 1; curr += lead - contig_dim; }
        }
    }

    __device__ __forceinline__ void load_global_unchecked(int tile_byte_offset, size_t reg, int lead, int& iter, int& off, int& curr) {
        regs[reg] = buffer_load<vector_type>(rsrc_, tile_byte_offset + curr * static_cast<int>(sizeof(T)));
        advance_global(lead, iter, off, curr);
    }

    __device__ __forceinline__ void load_global_checked(int tile_byte_offset, size_t reg, int lead, int& iter, int& off, int& curr) {
        if(iter < iter_dim) regs[reg] = buffer_load<vector_type>(rsrc_, tile_byte_offset + curr * static_cast<int>(sizeof(T)));
        advance_global(lead, iter, off, curr);
    }

    __device__ __forceinline__ void prefetch(const T* input, int lead, int tid) {
        const int tile_byte_offset = static_cast<int>(reinterpret_cast<const char*>(input) - reinterpret_cast<const char*>(base_ptr_));
        const int base_idx = tid * vector_width;
        int iter = base_idx / contig_dim, off = base_idx % contig_dim, curr = iter * lead + off;
        if constexpr(guaranteed_iters > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (load_global_unchecked(tile_byte_offset, i, lead, iter, off, curr), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }
        if constexpr(remainder_iters > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (load_global_checked(tile_byte_offset, guaranteed_iters + i, lead, iter, off, curr), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    template<size_t STEP, size_t TOTAL_STEPS>
    __device__ __forceinline__ void partial_prefetch(const T* input, int lead, int tid) {
        constexpr size_t vpt = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;
        constexpr size_t count = (end > start) ? (end - start) : 0;
        if constexpr(count > 0) {
            const int tile_byte_offset = static_cast<int>(reinterpret_cast<const char*>(input) - reinterpret_cast<const char*>(base_ptr_));
            const int base_idx = tid * vector_width;
            const int seed = base_idx + static_cast<int>(start) * step_elems;
            int iter = seed / contig_dim, off = seed % contig_dim, curr = iter * lead + off;
            [&]<size_t... i>(std::index_sequence<i...>) {
                (
                    [&]() {
                        constexpr size_t reg = start + i;
                        if constexpr(reg < static_cast<size_t>(guaranteed_iters))
                            load_global_unchecked(tile_byte_offset, reg, lead, iter, off, curr);
                        else
                            load_global_checked(tile_byte_offset, reg, lead, iter, off, curr);
                    }(),
                    ...);
            }(std::make_index_sequence<count>{});
        }
    }

    static constexpr int lds_pitch = contig_dim + PADDING;

    __device__ __forceinline__ void advance_lds(int& iter, int& off, int& curr) {
        curr += iter_inc * lds_pitch;
        iter += iter_inc;
        if constexpr(off_inc != 0) {
            off += off_inc; curr += off_inc;
            if(off >= contig_dim) { off -= contig_dim; iter += 1; curr += lds_pitch - contig_dim; }
        }
    }

    __device__ __forceinline__ void store_lds_unchecked(T* output, size_t reg, int& iter, int& off, int& curr) {
        *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
        advance_lds(iter, off, curr);
    }

    __device__ __forceinline__ void store_lds_checked(T* output, size_t reg, int& iter, int& off, int& curr) {
        if(iter < iter_dim) *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
        advance_lds(iter, off, curr);
    }

    template<size_t STEP, size_t TOTAL_STEPS>
    __device__ __forceinline__ void partial_commit(T* output, int tid) {
        constexpr size_t vpt = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;
        constexpr size_t count = (end > start) ? (end - start) : 0;
        if constexpr(count > 0) {
            const int base_idx = tid * vector_width;
            const int seed = base_idx + static_cast<int>(start) * step_elems;
            int iter = seed / contig_dim, off = seed % contig_dim, curr = iter * lds_pitch + off;
            [&]<size_t... i>(std::index_sequence<i...>) {
                (
                    [&]() {
                        constexpr size_t reg = start + i;
                        if constexpr(reg < static_cast<size_t>(guaranteed_iters))
                            store_lds_unchecked(output, reg, iter, off, curr);
                        else
                            store_lds_checked(output, reg, iter, off, curr);
                    }(),
                    ...);
            }(std::make_index_sequence<count>{});
        }
    }

    __device__ __forceinline__ void commit(T* output, int tid) {
        const int base_idx = tid * vector_width;
        int iter = base_idx / contig_dim, off = base_idx % contig_dim, curr = iter * lds_pitch + off;
        if constexpr(guaranteed_iters > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_lds_unchecked(output, i, iter, off, curr), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }
        if constexpr(remainder_iters > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_lds_checked(output, guaranteed_iters + i, iter, off, curr), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }
};

// LDS->global C epilogue (upstream load.hpp; row-major variant only, checked stores).
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         bool     IS_ALIGNED,
         class T,
         int LDS_PAD = 0>
class shared_to_global_store {
    static_assert(ACCESS == m_layout::row_major, "row-major C only");
    static constexpr int contig_dim   = BLOCK_N;
    static constexpr int iter_dim     = BLOCK_M;
    static constexpr int lds_pitch    = contig_dim + LDS_PAD;
    static constexpr int contig_bytes = contig_dim * sizeof(T);
    static constexpr int max_bytes    = MAX_BITS / 8;

    static constexpr int vector_bytes = []() constexpr {
        constexpr auto fits = [](int vb) constexpr -> bool { return vb <= max_bytes && (contig_bytes % vb) == 0; };
        if constexpr(fits(32)) return 32;
        else if constexpr(fits(16)) return 16;
        else if constexpr(fits(8)) return 8;
        else if constexpr(fits(4)) return 4;
        else return 2;
    }();

    static constexpr int actual_load_width = vector_bytes / sizeof(T);

    using type                        = typename type_selector<T>::type;
    using vector_type                 = typename load_vector<type, actual_load_width>::type;
    static constexpr int vector_width = actual_load_width;

    static constexpr int total_elements        = BLOCK_M * BLOCK_N;
    static constexpr int total_vectors         = total_elements / vector_width;
    static constexpr int vectors_per_thread    = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    static constexpr int step_elems = BLOCK_SIZE * vector_width;
    static constexpr int iter_inc   = step_elems / contig_dim;
    static constexpr int off_inc    = step_elems % contig_dim;

    __amdgpu_buffer_rsrc_t out_rsrc_;
    T*                     output_;

public:
    __device__ __forceinline__ shared_to_global_store(T* output, unsigned alloc_elems)
        : out_rsrc_(make_buffer_rsrc(output, alloc_elems * static_cast<unsigned>(sizeof(T))))
        , output_(output) {}

    __device__ __forceinline__ void store(const T* input, int row, int col, int M, int N, int tid) {
        const int lead_dim = N;
        const int base_idx = tid * vector_width;
        int       liter    = base_idx / contig_dim;
        int       lcontig  = base_idx % contig_dim;
        int       curr_sload = liter * lds_pitch + lcontig;
        int       grow        = row + liter;
        int       gcol        = col + lcontig;
        int       curr_gstore = grow * N + gcol;

        auto do_store = [&](int gstore, int sload, int local_grow, int local_gcol) {
            const auto& val      = *reinterpret_cast<const vector_type*>(input + sload);
            const int   byte_off = gstore * static_cast<int>(sizeof(T));
            const int   g_contig = local_gcol;
            const int   g_iter   = local_grow;
            const int   contig_max = N;
            const int   iter_max   = M;
            if constexpr(IS_ALIGNED) {
                buffer_store<vector_type>(out_rsrc_, byte_off, val);
            } else if(g_iter < iter_max && (g_contig + vector_width - 1) < contig_max) {
                buffer_store<vector_type>(out_rsrc_, byte_off, val);
            } else if(g_iter < iter_max && g_contig < contig_max) {
                const int valid = contig_max - g_contig;
                [&]<size_t... v>(std::index_sequence<v...>) {
                    (((static_cast<int>(v) < valid) ? (void)(output_[gstore + static_cast<int>(v)] = input[sload + static_cast<int>(v)])
                                                    : (void)0),
                     ...);
                }(std::make_index_sequence<vector_width>{});
            }
        };

        auto store_vector_unchecked = [&]<size_t>() {
            const int local_grow = row + liter;
            const int local_gcol = col + lcontig;
            do_store(curr_gstore, curr_sload, local_grow, local_gcol);
            curr_sload += iter_inc * lds_pitch;
            curr_gstore += iter_inc * lead_dim;
            liter += iter_inc;
            if constexpr(off_inc != 0) {
                lcontig += off_inc;
                curr_sload += off_inc;
                curr_gstore += off_inc;
                if(lcontig >= contig_dim) {
                    lcontig -= contig_dim;
                    liter += 1;
                    curr_sload += lds_pitch - contig_dim;
                    curr_gstore += lead_dim - contig_dim;
                }
            }
        };

        auto store_vector_checked = [&]<size_t>() {
            if(liter < iter_dim) {
                const int local_grow = row + liter;
                const int local_gcol = col + lcontig;
                do_store(curr_gstore, curr_sload, local_grow, local_gcol);
            }
            curr_sload += iter_inc * lds_pitch;
            curr_gstore += iter_inc * lead_dim;
            liter += iter_inc;
            if constexpr(off_inc != 0) {
                lcontig += off_inc;
                curr_sload += off_inc;
                curr_gstore += off_inc;
                if(lcontig >= contig_dim) {
                    lcontig -= contig_dim;
                    liter += 1;
                    curr_sload += lds_pitch - contig_dim;
                    curr_gstore += lead_dim - contig_dim;
                }
            }
        };

        if constexpr(guaranteed_iterations > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_vector_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iterations>{});
        }
        if constexpr(remainder_iterations > 0) {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_vector_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iterations>{});
        }
    }
};

// Snake + XOR-skew col swizzle tile mapping (upstream mapping.hpp, col-band variant selected
// by the col,col layout pair).
template<int BLOCK_M, int BLOCK_N, int SWIZZLE_N = 8>
class xor_snake_col_swizzle_mapping {
    static_assert((SWIZZLE_N & (SWIZZLE_N - 1)) == 0, "SWIZZLE_N must be a power of 2");
public:
    static __device__ __forceinline__ void map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col) {
        const int idx_m = tile_id % grid_m;
        const int idx_n = tile_id / grid_m;
        const int n_outer = idx_n / SWIZZLE_N;
        const int n_inner = idx_n % SWIZZLE_N;
        int final_m, final_n;
        if(n_outer < grid_n / SWIZZLE_N) {
            const int local = idx_m + n_inner * grid_m;
            final_m         = local / SWIZZLE_N;
            int inner_n = local % SWIZZLE_N;
            inner_n     = inner_n ^ (final_m & (SWIZZLE_N - 1));
            final_n = inner_n + n_outer * SWIZZLE_N;
        } else {
            const int n_rem = grid_n % SWIZZLE_N;
            const int local = idx_m + n_inner * grid_m;
            final_m         = local / n_rem;
            final_n         = (local % n_rem) + n_outer * SWIZZLE_N;
        }
        if(n_outer & 1) final_m = grid_m - 1 - final_m;
        if(final_m >= grid_m || final_n >= grid_n) {
            *block_row = (grid_m - 1) * BLOCK_M;
            *block_col = (grid_n - 1) * BLOCK_N;
            return;
        }
        *block_row = final_m * BLOCK_M;
        *block_col = final_n * BLOCK_N;
    }
};

// B-side dequant fragment (probe/dq_fragment): stages whole 16-k segments of weight rows,
// one WQ4K::frag16 dequant per 16 halves; LDS tile layout [n][k] = col-major B convention.
template<int BLOCK_K, int BLOCK_N, int PADDING, int BLOCK_SIZE>
struct dq_fragment {
    static constexpr int segs_per_row       = BLOCK_K / 16;
    static constexpr int vectors_per_thread = (BLOCK_N * segs_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int lds_pitch          = BLOCK_K + PADDING;

    static_assert((BLOCK_N * segs_per_row) % BLOCK_SIZE == 0 || BLOCK_N * segs_per_row <= BLOCK_SIZE,
                  "tile must divide evenly or be smaller than the block");

    wmma_half16 regs[vectors_per_thread];
    const char * row_base;
    size_t wstride;
    int wrow0, Nlim;
    int seg; // next tile's first 16-elem segment index within the row

    __device__ dq_fragment(const char* wq, size_t ws, int block_col, int N_)
        : row_base(wq + (size_t)block_col * ws), wstride(ws), wrow0(block_col), Nlim(N_), seg(0) {}

    __device__ __forceinline__ void load_vec(int v, int vec_idx) {
        const int r  = vec_idx / segs_per_row;
        const int s  = vec_idx % segs_per_row;
        const int gr = wrow0 + r;
        if (r < BLOCK_N && gr < Nlim) {
            const int j0 = (seg + s) * 16;
            WQ4K::frag16((const block_q4_K *)(row_base + (size_t)r * wstride) + (j0 >> 8), j0 & 255, (wmma_half *)&regs[v]);
        } else {
            regs[v] = (wmma_half16)0.f;
        }
    }

    __device__ __forceinline__ void prefetch(int tid) {
        #pragma unroll
        for (int v = 0; v < vectors_per_thread; ++v) load_vec(v, tid + v * BLOCK_SIZE);
    }

    template<size_t STEP, size_t TOTAL_STEPS>
    __device__ __forceinline__ void partial_prefetch(int tid) {
        // one tile's vectors spread across TOTAL_STEPS calls; step 0 advances to the next tile
        if constexpr(STEP == 0) seg += segs_per_row;
        constexpr size_t lps = (vectors_per_thread + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start = STEP * lps;
        constexpr size_t endc = (start + lps > (size_t)vectors_per_thread) ? (size_t)vectors_per_thread : start + lps;
        #pragma unroll
        for (size_t v = start; v < endc; ++v) load_vec((int)v, tid + (int)v * BLOCK_SIZE);
    }

    __device__ __forceinline__ void commit(__half * output_, int tid) {
        wmma_half * output = (wmma_half *)output_;
        #pragma unroll
        for (int v = 0; v < vectors_per_thread; ++v) {
            const int vec = tid + v * BLOCK_SIZE;
            const int r   = vec / segs_per_row;
            const int s   = vec % segs_per_row;
            if (r < BLOCK_N) *reinterpret_cast<wmma_half16 *>(&output[r * lds_pitch + s * 16]) = regs[v];
        }
    }
};

// gemm_impl with the B side wired to dq_fragment (upstream kernel.hpp structure; the B-side
// prefetch_fragment/pointer arithmetic replaced by the dequant fragment above).
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      k_slices,
         int      single_buffer,
         int      swizzle,
         int      bits,
         int      is_aligned>
__device__ __forceinline__ void gemm_impl_dq(
    T* __restrict__ C, const U* __restrict__ A, const char* wq, int M, int N, int K, size_t w_stride)
{
    static_assert(warps_m != 0 && warps_n != 0);
    static_assert((warp_tile_m * warp_tile_n) > 1);
    static_assert(k_slices > 0);

    constexpr int padding_a = (LAYOUT_A == m_layout::row_major) ? 8 : 0;
    constexpr int padding_b = (LAYOUT_B == m_layout::col_major) ? 8 : 0;

    constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    constexpr int block_n = warps_n * warp_tile_n * wmma_tile;
    constexpr int block_k  = k_slices * wmma_tile;
    constexpr int stride_a = block_k + padding_a;
    constexpr int stride_b = block_k + padding_b;
    constexpr int lds_size = (block_m * stride_a) + (stride_b * block_n);

    constexpr bool use_single_buffer = (single_buffer != 0);
    constexpr int  lds_buffers       = use_single_buffer ? 1 : 2;

    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    using mapper = xor_snake_col_swizzle_mapping<block_m, block_n, swizzle>;

    int block_row, block_col;
    mapper::map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    __shared__ U lds_mem[lds_buffers * lds_size];

    U* a_tiles_0 = lds_mem;
    U* b_tiles_0 = lds_mem + (block_m * stride_a);

    constexpr int full_block = warp_size * warps_m * warps_n;
    const int     tid        = threadIdx.x;

    A += blockIdx.y * M * K;
    C += blockIdx.y * M * N;

    const U* A_base = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);

    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = tid % warp_size;
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    const int warp_m_base = warp_row * warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * warp_tile_n * wmma_tile;

    constexpr int a_frag_size = k_slices * warp_tile_m;
    constexpr int b_frag_size = k_slices * warp_tile_n;

    fragment<T, wmma_tile> c_frags[warp_tile_m][warp_tile_n];
    fragment<U, wmma_tile> a_frag[a_frag_size];
    fragment<U, wmma_tile> b_frag[b_frag_size];

    const int lead_a = (LAYOUT_A == m_layout::col_major) ? M : K;

    // Constructor builds the SRD once from the matrix base + allocation size.
    prefetch_fragment<LAYOUT_A, bits, full_block, block_m, block_k, padding_a, U> regs_a(
        A, static_cast<unsigned>(M) * static_cast<unsigned>(K));
    dq_fragment<block_k, block_n, padding_b, warps_m * warps_n * 32> regs_b(wq, w_stride, block_col, N);

    const U* A_tile_ptr = A_base;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);

    constexpr int frag_mult_A   = (LAYOUT_A == m_layout::col_major) ? 1 : stride_a;
    constexpr int frag_mult_B   = (LAYOUT_B == m_layout::col_major) ? stride_b : 1;
    constexpr int frag_offset_A = wmma_tile * frag_mult_A;
    constexpr int frag_offset_B = wmma_tile * frag_mult_B;

    const int warp_offset_A = (warp_m_base + half_lane) * frag_mult_A;
    const int warp_offset_B = (warp_n_base + half_lane) * frag_mult_B;

    constexpr int slice_stride_A
        = (LAYOUT_A == m_layout::row_major) ? wmma_tile : wmma_tile * block_m;
    constexpr int slice_stride_B
        = (LAYOUT_B == m_layout::col_major) ? wmma_tile : wmma_tile * block_n;

    regs_a.prefetch(A_tile_ptr, lead_a, tid);
    regs_b.prefetch(tid);
    regs_a.commit(a_tiles_0, tid);
    regs_b.commit(b_tiles_0, tid);
    __syncthreads();

    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = use_single_buffer ? a_tiles_0 : (lds_mem + lds_size);
    U* next_b    = use_single_buffer ? b_tiles_0 : (lds_mem + lds_size + (block_m * stride_a));

    constexpr bool   warp_m_is_major = warp_tile_m >= warp_tile_n;
    constexpr size_t warp_inner_max  = warp_m_is_major ? warp_tile_n : warp_tile_m;
    constexpr size_t warp_outer_max
        = k_slices * static_cast<size_t>(warp_m_is_major ? warp_tile_m : warp_tile_n);
    constexpr size_t total_combos = warp_outer_max * warp_inner_max;
    constexpr size_t prefetch_steps = total_combos / 2;

    constexpr auto get_wm = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_o : w_i_s;
    };

    constexpr auto get_wn = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_i_s : w_o;
    };

    auto run_fold
        = [&]<bool do_prefetch>(const U* ca0, const U* cb0, const U* next_A)
    {
        auto a_ptr = [&]<size_t WM>() constexpr -> const U*
        { return ca0 + (WM / warp_tile_m) * slice_stride_A + (WM % warp_tile_m) * frag_offset_A; };
        auto b_ptr = [&]<size_t WN>() constexpr -> const U*
        { return cb0 + (WN / warp_tile_n) * slice_stride_B + (WN % warp_tile_n) * frag_offset_B; };

        if constexpr(warp_m_is_major)
        {
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0], a_ptr.template operator()<0>(), block_m, stride_a);
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0], b_ptr.template operator()<0>(), stride_b, block_n);
        }
        else
        {
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0], b_ptr.template operator()<0>(), stride_b, block_n);
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0], a_ptr.template operator()<0>(), block_m, stride_a);
        }

        [&]<size_t... i>(std::index_sequence<i...>)
        {
            (
                [&]()
                {
                    if constexpr(do_prefetch)
                    {
                        constexpr size_t step = i / 2;
                        if constexpr((i % 2) == 0)
                        {
                            regs_a.template partial_prefetch<step, prefetch_steps>(next_A, lead_a, tid);
                        }
                        else
                        {
                            regs_b.template partial_prefetch<step, prefetch_steps>(tid);
                        }
                    }

                    constexpr size_t wm = get_wm(i);
                    constexpr size_t wn = get_wn(i);

                    constexpr size_t next_i = i + 1;
                    if constexpr(next_i < total_combos)
                    {
                        constexpr size_t next_wm = get_wm(next_i);
                        constexpr size_t next_wn = get_wn(next_i);
                        if constexpr(warp_m_is_major)
                        {
                            if constexpr(next_wm != wm)
                            {
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[next_wm], a_ptr.template operator()<next_wm>(), block_m, stride_a);
                            }
                            if constexpr(next_wm % warp_tile_m == 0)
                            {
                                constexpr size_t b_idx = next_wn + (next_wm / warp_tile_m) * warp_tile_n;
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[b_idx], b_ptr.template operator()<b_idx>(), stride_b, block_n);
                            }
                        }
                        else
                        {
                            if constexpr(next_wn != wn)
                            {
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[next_wn], b_ptr.template operator()<next_wn>(), stride_b, block_n);
                            }
                            if constexpr(next_wn % warp_tile_n == 0)
                            {
                                constexpr size_t a_idx = next_wm + (next_wn / warp_tile_n) * warp_tile_m;
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[a_idx], a_ptr.template operator()<a_idx>(), block_m, stride_a);
                            }
                        }
                    }

                    constexpr size_t a_slice = warp_m_is_major ? 0 : (wn / warp_tile_n);
                    constexpr size_t b_slice = warp_m_is_major ? (wm / warp_tile_m) : 0;
                    constexpr size_t a_wmma  = wm + a_slice * warp_tile_m;
                    constexpr size_t b_wmma  = wn + b_slice * warp_tile_n;

                    __builtin_amdgcn_s_setprio(1);
                    wmma(a_frag[a_wmma], b_frag[b_wmma], c_frags[wm % warp_tile_m][wn % warp_tile_n]);
                    __builtin_amdgcn_s_setprio(0);
                }(),
                ...);
        }(std::make_index_sequence<total_combos>{});
    };

    for(int k_tile = 0; k_tile < K - block_k; k_tile += block_k)
    {
        const U* ca0    = current_a + warp_offset_A;
        const U* cb0    = current_b + warp_offset_B;
        const U* next_A = A_tile_ptr + global_mult_A;

        run_fold.template operator()<true>(ca0, cb0, next_A);

        A_tile_ptr += global_mult_A;

        if constexpr(use_single_buffer)
        {
            __syncthreads();
            regs_a.commit(current_a, tid);
            regs_b.commit(current_b, tid);
            __syncthreads();
        }
        else
        {
            regs_a.commit(next_a, tid);
            regs_b.commit(next_b, tid);

            U* temp_a = current_a;
            U* temp_b = current_b;
            current_a = next_a;
            current_b = next_b;
            next_a    = temp_a;
            next_b    = temp_b;
            __syncthreads();
        }
    }

    const U* ca0 = current_a + warp_offset_A;
    const U* cb0 = current_b + warp_offset_B;

    run_fold.template operator()<false>(ca0, cb0, nullptr);

    __syncthreads();

    if constexpr(use_single_buffer)
    {
        auto store_row = [&]<size_t wm>()
        {
            [&]<size_t... wn>(std::index_sequence<wn...>)
            {
                (store_matrix<LAYOUT_C, false>(
                     C, c_frags[wm][wn],
                     block_row + warp_m_base + static_cast<int>(wm) * wmma_tile + half_warp_id,
                     block_col + warp_n_base + static_cast<int>(wn) * wmma_tile + half_lane,
                     M, N),
                 ...);
            }(std::make_index_sequence<warp_tile_n>{});
        };

        [&]<size_t... wm>(std::index_sequence<wm...>)
        { (store_row.template operator()<wm>(), ...); }(std::make_index_sequence<warp_tile_m>{});
    }
    else
    {
        constexpr bool is_col_major = (LAYOUT_C == m_layout::col_major);
        static_assert(!is_col_major, "row-major C only in this splice");

        constexpr int chunk_rows = wmma_tile;
        constexpr int chunk_cols = wmma_tile;

        constexpr int num_chunks = is_col_major ? (warps_n * warp_tile_n) : (warps_m * warp_tile_m);
        constexpr int available_bytes = lds_buffers * lds_size * static_cast<int>(sizeof(U));
        constexpr int chunk_unit_bytes
            = is_col_major ? (0) : (chunk_rows * block_n * static_cast<int>(sizeof(T)));

        constexpr int chunk_multiple = []() constexpr
        {
            for(int m = num_chunks; m >= 1; --m)
            {
                if(num_chunks % m == 0 && m * chunk_unit_bytes <= available_bytes)
                {
                    return m;
                }
            }
            return 1;
        }();

        constexpr int num_groups = num_chunks / chunk_multiple;
        constexpr int group_cols = is_col_major ? chunk_multiple * chunk_cols : chunk_cols;
        constexpr int group_rows = is_col_major ? chunk_rows : chunk_multiple * chunk_rows;
        constexpr int group_step = is_col_major ? group_cols : group_rows;
        constexpr int transfer_m = is_col_major ? block_m : group_step;
        constexpr int transfer_n = is_col_major ? group_step : block_n;

        T* c_buf = reinterpret_cast<T*>(lds_mem);
        shared_to_global_store<LAYOUT_C, bits, full_block, transfer_m, transfer_n, is_aligned, T, 0>
            store_c(C, static_cast<unsigned>(M) * static_cast<unsigned>(N));

        auto store_chunk = [&]<size_t ci>(T* buf, int local_offset)
        {
            constexpr int responsible_warp_row = static_cast<int>(ci) / warp_tile_m;
            constexpr int wm_idx               = static_cast<int>(ci) % warp_tile_m;

            if(warp_row == responsible_warp_row)
            {
                [&]<size_t... wn>(std::index_sequence<wn...>)
                {
                    (store_matrix<LAYOUT_C, true>(buf,
                                                  c_frags[wm_idx][wn],
                                                  local_offset + half_warp_id,
                                                  warp_n_base + static_cast<int>(wn) * wmma_tile + half_lane,
                                                  group_rows,
                                                  block_n),
                     ...);
                }(std::make_index_sequence<warp_tile_n>{});
            }
        };

        auto flush_group = [&]<size_t gi>(T* buf)
        {
            store_c.store(buf, block_row + static_cast<int>(gi) * group_step, block_col, M, N, tid);
        };

        auto store_group = [&]<size_t g>()
        {
            [&]<size_t... li>(std::index_sequence<li...>)
            {
                (store_chunk.template operator()<g * chunk_multiple + li>(c_buf, static_cast<int>(li) * wmma_tile), ...);
            }(std::make_index_sequence<chunk_multiple>{});
        };

        [&]<size_t... gi>(std::index_sequence<gi...>)
        {
            (
                [&]()
                {
                    store_group.template operator()<gi>();
                    __syncthreads();
                    flush_group.template operator()<gi>(c_buf);
                    __syncthreads();
                }(),
                ...);
        }(std::make_index_sequence<num_groups>{});
    }
}


} // namespace rwg

// x f32 row-major [M][K] -> f16 col-major [K][M] staging (32x32 LDS tiles). m >= M entries in
// the covered tiles are zeroed; the launcher allocates +128 halves of headroom so the GEMM's
// padded A-tile reads never touch unmapped memory.
__device__ unsigned long long g_x16_overflow = 0; // |x| >= 65504 would become Inf in f16
static __global__ void __launch_bounds__(256)
x_stage_f16col(const float* __restrict__ x, size_t x_stride, __half * __restrict__ xc, int M, int K) {
    __shared__ __half tile[32][33];
    const int k0 = blockIdx.x * 32;
    const int m0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int tr = tid >> 5, tc = tid & 31;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int r  = tr + (tc >> 3) * 8;   // m row 0..31
        const int c  = q * 8 + (tc & 7);     // k col 0..31
        const int mr = m0 + r, kr = k0 + c;
        if (kr < K && mr < M) {
            const float v = x[(size_t)mr * (x_stride / 4) + kr];
            if (!(fabsf(v) < 65504.f)) atomicAdd(&g_x16_overflow, 1ULL); // count, still store Inf
            tile[r][c] = (__half)v;
        } else {
            tile[r][c] = (__half)0.f;
        }
    }
    __syncthreads();
    const int wr = tid >> 3;   // k row 0..31
    const int wc = tid & 7;
    if (k0 + wr < K) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int mc = wc * 4 + q;
            const int m  = m0 + mc;
            if (m < M) xc[(size_t)(k0 + wr) * M + m] = tile[mc][wr];
        }
    }
}

// tuned on gfx1100 (probe/scan_rwg_dq): 2x4 warps (block 256m x 128n, 256 threads), warp tile
// 8x2, k_slices 1, double buffer, swizzle 8, 256-bit vectors; C row-major, A f16 col-major
// (staged), B Q4_K native. block_m 256 halves the B(weight)-side re-reads: 69.5-72.8 TF
// across the model's real shapes vs 63-67 at block_m 128 (A-side re-reads are cheaper: the
// staged x16 panels are contiguous 256B bursts).
template<bool IS_ALIGNED>
static __global__ void __launch_bounds__(256)
mul_mat_wmma_dq_kernel(float * __restrict__ dst, const __half * __restrict__ x16,
                       const char * __restrict__ wq, size_t w_stride,
                       int M, int N, int K) {
    rwg::gemm_impl_dq<float, __half,
                      rwg::m_layout::row_major,  // C: ggml dst (m,n) at m*N+n
                      rwg::m_layout::col_major,  // A: x staged as [K][M] f16
                      rwg::m_layout::col_major,  // B: Q4_K weights [N][K]
                      2, 4, 8, 2, 1, 0, 8, 256, IS_ALIGNED>(dst, x16, wq, M, N, K, w_stride);
}

// v4 entry: stage x -> f16 [K][M], then the spliced gemm. The staging pass costs ~296/N % of
// GEMM time (measured 0.095 ms @ M2048/K5120 vs 1.61 ms GEMM), so the dispatcher only routes
// wide-output GEMMs here; at N >= ~6k the kernel's 65 TF beats dp4a's 62 net of staging.
static void mul_mat_wmma_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            int M, int N, int K) {
    cudaStream_t stream = ctx.stream();
    // +128 halves headroom: the last A tile reads up to 127 halves past M in the final k
    // column; those rows are masked out of the C epilogue, but must not fault.
    ggml_cuda_pool_alloc<__half> x16(ctx.pool(), (size_t)K * M + 128);
    const dim3 gstage((K + 31) / 32, (M + 31) / 32);
    x_stage_f16col<<<gstage, 256, 0, stream>>>((const float *)src1->data, src1->nb[1], x16.ptr, M, K);
    {
        static const bool dbg = getenv("GGML_WMMA_DQ_DEBUG") != nullptr;
        static int calls = 0;
        if (dbg && calls < 24) {
            unsigned long long h = 0;
            hipMemcpyFromSymbol(&h, HIP_SYMBOL(g_x16_overflow), sizeof(h));
            fprintf(stderr, "[dq] call %d M=%d N=%d K=%d overflow=%llu\n", calls++, M, N, K, h);
        }
    }

    const int grid = ((M + 255) / 256) * ((N + 127) / 128);
    if ((M % 256) == 0 && (N % 128) == 0) {
        mul_mat_wmma_dq_kernel<true><<<grid, 256, 0, stream>>>((float *)dst->data, x16.ptr, (const char *)src0->data, src0->nb[1], M, N, K);
    } else {
        mul_mat_wmma_dq_kernel<false><<<grid, 256, 0, stream>>>((float *)dst->data, x16.ptr, (const char *)src0->data, src0->nb[1], M, N, K);
    }
}

void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t M = src1->ne[1];
    const int64_t N = src0->ne[1];
    const int64_t K = src0->ne[0];

    cudaStream_t stream = ctx.stream();

    // GGML_MMQ_WMMA=2: v4 Q4_K dequant path for all 2D Q4_K GEMMs (NMIN defaults to 0: full
    // routing measured best end-to-end, +0.8-1.3% across pp512..pp131072).
    static const int dq_mode   = []() { const char * e = getenv("GGML_MMQ_WMMA"); return e ? atoi(e) : 0; }();
    static const int dq_nmin   = []() { const char * e = getenv("GGML_MMQ_WMMA_DQ_NMIN"); return e ? atoi(e) : 0; }();
    if (dq_mode == 2 && src0->type == GGML_TYPE_Q4_K && M > 16 && N >= dq_nmin) {
        mul_mat_wmma_dq(ctx, src0, src1, dst, (int)M, (int)N, (int)K);
        return;
    }

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
    // mode 2 = v4 Q4_K dq path only; IQ3_S stays on dp4a (the v1 big kernel loses to it).
    static const int big_mode = []() {
        const char * e = getenv("GGML_MMQ_WMMA");
        return e ? atoi(e) : 0;
    }();
    if (big_mode == 2) return src0->type == GGML_TYPE_Q4_K;
    return big_mode != 0;
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
