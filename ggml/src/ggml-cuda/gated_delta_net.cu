#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // meow: with scalar decay the per-column recurrences are fully independent,
    // so for latency-bound long prefills more (smaller) CTAs beat fewer wide ones:
    // 2 columns per single-warp CTA -> H * n_seqs * 64 CTAs. Env-gated for A/B.
    static const bool fine_split = (getenv("GGML_GDN_SPLIT") != nullptr);
    if (fine_split && S_v == 128 && n_tokens >= 64) {
        dim3 g(H, n_seqs, S_v / 2);
        dim3 b(warp_size <= S_v ? warp_size : S_v, 1, 1);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream);
        if (!KDA) {
            if (keep_rs_t) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, true, 2>, lp,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            } else {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, 2>, lp,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            }
        } else {
            if (keep_rs_t) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, true, true, 2>, lp,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            } else {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, true, false, 2>, lp,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            }
        }
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ============================================================================
// meow: chunked (WY-representation) prefill for the gated delta rule.
//
// Recurrence (scalar decay g_t, per token t):
//   S_t = g_t S_{t-1} + k_t d_t^T,   d_t = b_t (v_t - g_t S_{t-1}^T k_t),
//   o_t = S_t^T q_t
// With cl_t = sum_{tau<=t} g_tau inside a chunk (Lambda_t = e^{cl_t}):
//   d_t + sum_{tau<t} L[t,tau] d_tau = b_t (v_t - Lambda_t S_0^T k_t),
//     L[t,tau] = b_tau e^{cl_t-cl_tau} (k_tau . k_t)
//   o_t = Lambda_t S_0^T q_t + sum_{tau<=t} e^{cl_t-cl_tau} (k_tau . q_t) d_tau
//   S_C = Lambda_C S_0 + sum_tau e^{cl_C-cl_tau} k_tau d_tau^T
// Decay ratios are factored e^{cl-mid} * e^{mid-cl} with mid = cl_C/2 so each
// stored scale stays bounded. The chunk update is affine in S_0:
//   S_C = (Lambda_C I - e^{mid} P Khat^T-ish) S_0 + P V  with P = Klc^T W_b,
//   W_b = (I+L)^{-1} diag(b)  (state independent!)
// so all chunks precompute their linear map, a chain kernel propagates the
// chunk-start states, and outputs for every chunk are computed in parallel.
// Single sequence only; K>1 / KDA / S_k != S_v / short inputs fall back.
// ============================================================================

#define GDN_CHUNK 64

// row-major strided-batched fp32 GEMMs mapped onto cublas col-major:
//   NN: C[M,N] = A[M,K] @ B[K,N]
//   NT: C[M,N] = A[M,K] @ B^T   (B row-major [N,K])
//   TN: C[M,N] = A^T @ B        (A row-major [K,M])
static void gdn_gemm(cublasHandle_t handle, int mode,
        int M, int N, int K, int batch,
        const float * A, long long sA, const float * B, long long sB,
        float * C, long long sC) {
    const float one = 1.0f, zero = 0.0f;
    // row-major C[M,N] == col-major Cc[N,M] (ld=N); Cc = Bc @ Ac
    if (mode == 0) {        // NN: Bc = OP_N(B, m=N, k=K, ld=N); Ac = OP_N(A, k=K, n=M, ld=K)
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &one, B, N, sB, A, K, sA, &zero, C, N, sC, batch));
    } else if (mode == 1) { // NT: Bc = OP_T(B, ldb=K); Ac = OP_N(A, ld=K)
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &one, B, K, sB, A, K, sA, &zero, C, N, sC, batch));
    } else {                // TN: Bc = OP_N(B, ld=N); Ac = OP_T(A, lda=M)
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            &one, B, N, sB, A, M, sA, &zero, C, N, sC, batch));
    }
}

// per (head, chunk): cumulative log-decay (thread 0), scaled q/k rows, packed v
__global__ void gdn_chunk_prepare_kernel(
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ v, const float * __restrict__ g, const float * __restrict__ beta,
        float * __restrict__ cl_buf,     // (C+1) per (chunk,head): cl_t ..., mid at slot C
        float * __restrict__ qh,         // [C,128]: q_t e^{cl_t - mid}
        float * __restrict__ kh,         // [C,128]: k_t e^{cl_t - mid}
        float * __restrict__ kl0,        // [C,128]: k_t e^{mid - cl_t}
        float * __restrict__ klb,        // [C,128]: k_t e^{mid - cl_t} b_t
        float * __restrict__ klc,        // [C,128]: k_t e^{cl_C - cl_t}
        float * __restrict__ vbuf,       // [C,128]: v_t (packed)
        int64_t H,
        int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2,
        uint3 neqk1_magic,
        int64_t sb1, int64_t sb2) {
    constexpr int C = GDN_CHUNK;
    const int h    = blockIdx.x;
    const int ch   = blockIdx.y;
    const int t0   = ch * C;
    const int lane = threadIdx.x;          // == token-in-chunk

    __shared__ float cl_sh[C + 1];

    if (lane == 0) {
        float acc = 0.0f;
        for (int t = 0; t < C; ++t) {
            acc += g[(t0 + t) * sb2 + h * sb1];
            cl_sh[t] = acc;
        }
        cl_sh[C] = acc * 0.5f;
    }
    __syncthreads();

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);

    const int64_t mat = ((int64_t)ch * H + h) * ((int64_t)C * 128);
    const int64_t row = (int64_t)lane * 128;

    const float cl  = cl_sh[lane];
    const float mid = cl_sh[C];
    const float clC = cl_sh[C - 1];

    const float e_hi = __expf(cl - mid);
    const float e_lo = __expf(mid - cl);
    const float e_c  = __expf(clC - cl);

    const float * q_t = q + (t0 + lane) * sq2 + iq1 * sq1;
    const float * k_t = k + (t0 + lane) * sq2 + iq1 * sq1;
    const float * v_t = v + (t0 + lane) * sv2 + h * sv1;
    const float b_val = beta[(t0 + lane) * sb2 + h * sb1];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {   // 128 elems over 64 lanes
        const int idx = lane * 2 + i;
        const float qv = q_t[idx];
        const float kv = k_t[idx];
        qh [mat + row + idx] = qv * e_hi;
        kh [mat + row + idx] = kv * e_hi;
        kl0[mat + row + idx] = kv * e_lo;
        klb[mat + row + idx] = kv * e_lo * b_val;
        klc[mat + row + idx] = kv * e_c;
        vbuf[mat + row + idx] = v_t[idx];
    }
    cl_buf[((int64_t)ch * H + h) * (C + 1) + lane] = cl;
    if (lane == 0) {
        cl_buf[((int64_t)ch * H + h) * (C + 1) + C] = mid;
    }
}

// forward substitution solving (I + L) X = RHS for X, rows written in order.
// IDENT: RHS[t, j] = (t == j) * beta_t          -> X = W_b = (I+L)^{-1} diag(beta), C columns
// else:   RHS[t, j] = beta_t * ubuf[t, j]       -> X = D, 128 columns
template <bool IDENT>
__global__ void gdn_chunk_solve_kernel(
        const float * __restrict__ lmat,   // [C, C] per (chunk,head), raw Kh Klb^T
        const float * __restrict__ ubuf,   // [C,128] per (chunk,head): V - e^{mid} Kh S_start
        const float * __restrict__ beta,
        float * __restrict__ xmat,         // out: [C, C] (IDENT) or [C,128]
        int64_t H, int64_t chunk_len,
        int64_t sb1, int64_t sb2) {
    constexpr int C = GDN_CHUNK;
    constexpr int NC = IDENT ? C : 128;
    const int h   = blockIdx.x;
    const int ch  = blockIdx.y;
    const int col = threadIdx.x;           // 0..NC-1
    const int64_t base = (int64_t)ch * H + h;

    extern __shared__ float smem[];        // C*C (L) + C*NC (X)
    float * Lh = smem;
    float * Xsh = smem + C * C;

    for (int idx = col; idx < C * C; idx += NC) {
        const int t = idx / C;
        const int s = idx - t * C;
        Lh[idx] = (s < t && t < chunk_len && s < chunk_len) ? lmat[base * (C * C) + idx] : 0.0f;
    }
    __syncthreads();

    const int64_t t_base = (int64_t)ch * C;

    for (int t = 0; t < chunk_len; ++t) {
        float acc = 0.0f;
        const float * Lrow = Lh + t * C;
        for (int s2 = 0; s2 < t; ++s2) {
            acc += Lrow[s2] * Xsh[s2 * NC + col];
        }
        float rhs;
        if constexpr (IDENT) {
            rhs = (t == col) ? beta[(t_base + t) * sb2 + h * sb1] : 0.0f;
        } else {
            rhs = beta[(t_base + t) * sb2 + h * sb1] * ubuf[base * (C * 128) + t * 128 + col];
        }
        const float x = rhs - acc;
        Xsh[t * NC + col] = x;
        xmat[base * ((int64_t)C * NC) + t * NC + col] = x;
        __syncthreads();
    }
}

// W2 = Lambda_C * I - e^{mid} * (P @ Kh): the affine chunk map's linear part
__global__ void gdn_w2_build_kernel(
        const float * __restrict__ gmat, const float * __restrict__ cl_buf,
        float * __restrict__ w2, int64_t H) {
    constexpr int C = GDN_CHUNK;
    const int h  = blockIdx.x;
    const int ch = blockIdx.y;
    const int64_t base = (int64_t)ch * H + h;
    const float clC = cl_buf[base * (C + 1) + C - 1];
    const float e_mid = __expf(cl_buf[base * (C + 1) + C]);
    for (int idx = threadIdx.x; idx < 128 * 128; idx += blockDim.x) {
        const int i = idx / 128;
        const int j = idx - i * 128;
        w2[base * (128 * 128) + idx] = (i == j ? __expf(clC) : 0.0f) - e_mid * gmat[base * (128 * 128) + idx];
    }
}

// chunk-start state chain: S_start[0] = input state; S_start[c+1] = W2_c S_start[c] + R_c.
// grid (H, 4 col-tiles of 32); block 256; each launch advances STEPS chunks with
// the tile's state slice resident in shared memory and W2 streamed in 32-wide slabs.
__global__ void gdn_chunk_chain_kernel(
        const float * __restrict__ w2, const float * __restrict__ rmat,
        const float * __restrict__ s_in,
        float * __restrict__ s_starts,      // [n_chunk+1, H, 128, 128]
        float * __restrict__ state_out,
        int64_t H, int n_chunk, int chunk_base, int n_steps) {
    constexpr int TILE = 32;
    constexpr int STEPS = 4;
    const int h    = blockIdx.x;
    const int tile = blockIdx.y;
    const int c0   = tile * TILE;

    __shared__ float Ssh[128 * TILE];
    __shared__ float Wsh[128 * 32];

    // load the state slice for chunk chunk_base (from s_starts or the transposed input)
    if (chunk_base == 0) {
        for (int j = threadIdx.x; j < 128 * TILE; j += blockDim.x) {
            const int kk = j / TILE;
            const int jj = j - kk * TILE;
            Ssh[j] = s_in[(int64_t)h * 128 * 128 + (int64_t)(c0 + jj) * 128 + kk];
        }
    } else {
        const int64_t b0 = ((int64_t)chunk_base * H + h) * (128 * 128);
        for (int j = threadIdx.x; j < 128 * TILE; j += blockDim.x) {
            const int i = j / TILE;
            const int jj = j - i * TILE;
            Ssh[j] = s_starts[b0 + (int64_t)i * 128 + c0 + jj];
        }
    }
    __syncthreads();

    // thread t: 16 outputs = (row r = t/2) x (cols half*16 .. +16)
    const int r    = threadIdx.x >> 1;
    const int half = (threadIdx.x & 1) * 16;

    for (int st = 0; st < n_steps; ++st) {
        const int c = chunk_base + st;
        if (c >= n_chunk) {
            break;
        }
        const int64_t base = ((int64_t)c * H + h);

        // store the chunk-start slice
        for (int j = threadIdx.x; j < 128 * TILE; j += blockDim.x) {
            const int i = j / TILE;
            const int jj = j - i * TILE;
            s_starts[base * (128 * 128) + (int64_t)i * 128 + c0 + jj] = Ssh[j];
        }

        float acc[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j] = 0.0f;
        }
        for (int slab = 0; slab < 4; ++slab) {
            for (int idx = threadIdx.x; idx < 128 * 32; idx += blockDim.x) {
                const int i  = idx / 32;
                const int kk = idx - i * 32;
                Wsh[idx] = w2[base * (128 * 128) + (int64_t)i * 128 + slab * 32 + kk];
            }
            __syncthreads();
            const float * wrow = Wsh + r * 32;
            const float * s0p  = Ssh + (slab * 32) * TILE + half;
            #pragma unroll
            for (int kk = 0; kk < 32; ++kk) {
                const float w = wrow[kk];
                const float * srow = s0p + kk * TILE;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    acc[j] += w * srow[j];
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            Ssh[r * TILE + half + j] = acc[j] + rmat[base * (128 * 128) + (int64_t)r * 128 + c0 + half + j];
        }
        __syncthreads();
    }

    // store the final slice of this launch (chunk_base + n_steps) + final state
    {
        const int c = chunk_base + n_steps;
        const int64_t b1 = ((int64_t)c * H + h) * (128 * 128);
        const bool last = (c >= n_chunk);
        for (int j = threadIdx.x; j < 128 * TILE; j += blockDim.x) {
            const int i  = j / TILE;
            const int jj = j - i * TILE;
            s_starts[b1 + (int64_t)i * 128 + c0 + jj] = Ssh[j];
            if (last) {
                state_out[(int64_t)h * 128 * 128 + (int64_t)(c0 + jj) * 128 + i] = Ssh[j];
            }
        }
    }
}

// causal mask: zero strictly-upper triangle of the (Qh Kl0^T) product
__global__ void gdn_chunk_bmask_kernel(float * __restrict__ bm, int64_t H) {
    constexpr int C = GDN_CHUNK;
    const int h  = blockIdx.x;
    const int ch = blockIdx.y;
    const int64_t base = ((int64_t)ch * H + h) * (C * C);
    for (int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
        const int t = idx / C;
        const int s = idx - t * C;
        if (s > t) {
            bm[base + idx] = 0.0f;
        }
    }
}

// ubuf = vbuf - e^{mid} * ks0
__global__ void gdn_chunk_ufold_kernel(
        const float * __restrict__ vbuf, const float * __restrict__ ks0,
        const float * __restrict__ cl_buf, float * __restrict__ ubuf, int64_t H) {
    constexpr int C = GDN_CHUNK;
    const int h  = blockIdx.x;
    const int ch = blockIdx.y;
    const int64_t base = (int64_t)ch * H + h;
    const float e_mid = __expf(cl_buf[base * (C + 1) + C]);
    for (int idx = threadIdx.x; idx < C * 128; idx += blockDim.x) {
        ubuf[base * (C * 128) + idx] = vbuf[base * (C * 128) + idx] - e_mid * ks0[base * (C * 128) + idx];
    }
}

// dst[(t0+t) * H*128 + h*128 + col] = (obin + e^{mid} oex) * scale
__global__ void gdn_chunk_scatter_kernel(
        const float * __restrict__ obin, const float * __restrict__ oex,
        const float * __restrict__ cl_buf, float * __restrict__ dst,
        int64_t H, float scale) {
    constexpr int C = GDN_CHUNK;
    const int h   = blockIdx.x;
    const int ch  = blockIdx.y;
    const int col = threadIdx.x;
    const int64_t base = (int64_t)ch * H + h;
    const int64_t t0  = (int64_t)ch * C;
    const float e_mid = __expf(cl_buf[base * (C + 1) + C]);

    for (int t = 0; t < C; ++t) {
        const int64_t idx = (int64_t)t * 128 + col;
        const float val = obin[base * (C * 128) + idx] + e_mid * oex[base * (C * 128) + idx];
        dst[((t0 + t) * H + h) * 128 + col] = val * scale;
    }
}

// meow: env-gated phase profiler for the chunked path.
// Ring-buffers one event per phase boundary per layer and drains once every 48
// layers, so measurement does not stall the pipeline between layers.
struct gdn_phase_prof {
    static constexpr int RING = 48;
    static constexpr int MAXE = 16;
    cudaEvent_t ring[RING][MAXE];
    int n[RING];
    int w = 0;
    const char * names[MAXE] = {};
    int n_names = 0;
    double totals[MAXE] = {};
    long long layers = 0;

    static gdn_phase_prof & get() {
        static gdn_phase_prof inst;
        return inst;
    }

    void mark(const char * name, cudaStream_t stream) {
        if (n_names < MAXE) {
            names[n_names] = name;
            n_names = 0; // filled lazily below
        }
        // record names in order on first layer
        static const char * ordered[MAXE];
        static int n_ordered = 0;
        if (n_ordered < MAXE && name != nullptr) {
            bool seen = false;
            for (int i = 0; i < n_ordered; ++i) {
                if (ordered[i] == name) { seen = true; break; }
            }
            if (!seen) { ordered[n_ordered++] = name; }
            for (int i = 0; i < MAXE; ++i) { names[i] = (i < n_ordered) ? ordered[i] : nullptr; }
        }
        if (n[w] < MAXE) {
            if (ring[w][n[w]] == nullptr) {
                cudaEventCreateWithFlags(&ring[w][n[w]], 0);
            }
            cudaEventRecord(ring[w][n[w]], stream);
            n[w]++;
        }
    }

    void layer_done(cudaStream_t stream) {
        layers++;
        w = (w + 1) % RING;
        if (layers % RING != 0) {
            return;
        }
        cudaStreamSynchronize(stream);
        for (int l = 0; l < RING; ++l) {
            for (int i = 0; i + 1 < n[l]; ++i) {
                float ms = 0.0f;
                if (ring[l][i] != nullptr && ring[l][i + 1] != nullptr) {
                    hipEventElapsedTime(&ms, ring[l][i], ring[l][i + 1]);
                    totals[i] += ms;
                }
            }
            n[l] = 0;
        }
        fprintf(stderr, "[gdn-prof] mean phase ms per layer (%lld layers):\n", layers);
        double total = 0.0;
        for (int i = 0; i + 1 < 16; ++i) {
            if (names[i + 1] == nullptr && totals[i] == 0.0) { break; }
            total += totals[i] / layers;
        }
        for (int i = 0; i + 1 < 16; ++i) {
            if (names[i + 1] == nullptr && totals[i] == 0.0) { break; }
            fprintf(stderr, "[gdn-prof]   %-14s %7.3f  (%4.1f%%)\n",
                    names[i] ? names[i] : "?", totals[i] / layers, 100.0 * (totals[i] / layers) / (total > 0 ? total : 1));
        }
        fprintf(stderr, "[gdn-prof]   %-14s %7.3f\n", "TOTAL", total);
        for (int i = 0; i < MAXE; ++i) { totals[i] = 0.0; }
    }
};

#define GDN_PHASE(name) if (phase_prof) phase_prof->mark(name, stream)

// host orchestration (single sequence; full chunks + optional tail via recurrent kernel)
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens,
        int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2,
        int64_t neqk1,
        float scale, cudaStream_t stream, ggml_backend_cuda_context & ctx) {
    constexpr int C = GDN_CHUNK;
    GGML_ASSERT(n_tokens >= 2 * C);

    const int n_chunk = (int)(n_tokens / C);
    const int tail    = (int)(n_tokens % C);
    const int batch   = n_chunk * (int)H;
    const int64_t CS  = C * 128;
    const int64_t CC  = C * C;
    const int64_t SS  = 128 * 128;

    ggml_cuda_pool_alloc<float> cl_a  (ctx.pool(), (size_t)((n_chunk * H) * (C + 1)));
    ggml_cuda_pool_alloc<float> qh_a  (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> kh_a  (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> kl0_a (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> klb_a (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> klc_a (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> vb_a  (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> l_a   (ctx.pool(), (size_t)(batch * CC));
    ggml_cuda_pool_alloc<float> wb_a  (ctx.pool(), (size_t)(batch * CC));
    ggml_cuda_pool_alloc<float> bm_a  (ctx.pool(), (size_t)(batch * CC));
    ggml_cuda_pool_alloc<float> p_a   (ctx.pool(), (size_t)(batch * CS));   // [128, C] per ch-h
    ggml_cuda_pool_alloc<float> r_a   (ctx.pool(), (size_t)(batch * SS));
    ggml_cuda_pool_alloc<float> g_a   (ctx.pool(), (size_t)(batch * SS));
    ggml_cuda_pool_alloc<float> w2_a  (ctx.pool(), (size_t)(batch * SS));
    ggml_cuda_pool_alloc<float> ss_a  (ctx.pool(), (size_t)((n_chunk + 1) * H * SS));
    ggml_cuda_pool_alloc<float> ks0_a (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> ub_a  (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> d_a   (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> ob_a  (ctx.pool(), (size_t)(batch * CS));
    ggml_cuda_pool_alloc<float> oe_a  (ctx.pool(), (size_t)(batch * CS));

    float * cl  = cl_a.get();  float * qh = qh_a.get(); float * kh  = kh_a.get();
    float * kl0 = kl0_a.get(); float * klb = klb_a.get(); float * klc = klc_a.get();
    float * vb  = vb_a.get();  float * lm = l_a.get();  float * wb  = wb_a.get();
    float * bm  = bm_a.get();  float * pm = p_a.get();  float * rm  = r_a.get();
    float * gm  = g_a.get();   float * w2 = w2_a.get(); float * sst = ss_a.get();
    float * ks0 = ks0_a.get(); float * ub = ub_a.get(); float * dm  = d_a.get();
    float * ob  = ob_a.get();  float * oe = oe_a.get();

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const dim3 g_hc((unsigned)H, (unsigned)n_chunk);
    static const bool prof_on = (getenv("GGML_GDN_PHASE_PROF") != nullptr);
    gdn_phase_prof * phase_prof = prof_on ? &gdn_phase_prof::get() : nullptr;
    if (phase_prof) { phase_prof->mark("start", stream); }

    GDN_PHASE("prepare");
    gdn_chunk_prepare_kernel<<<g_hc, C, 0, stream>>>(
        q_d, k_d, v_d, g_d, b_d, cl, qh, kh, kl0, klb, klc, vb,
        H, sq1, sq2, sv1, sv2, neqk1_magic, sb1, sb2);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("gemm_L");

    // 2) L = Kh @ Klb^T ; W_b = (I+L)^{-1} diag(beta)
    gdn_gemm(handle, 1, C, C, 128, batch, kh, CS, klb, CS, lm, CC);
    {
        static bool cfg = false;
        if (!cfg) {
            CUDA_CHECK(cudaFuncSetAttribute((const void *) gdn_chunk_solve_kernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (C * C + C * C) * (int)sizeof(float)));
            CUDA_CHECK(cudaFuncSetAttribute((const void *) gdn_chunk_solve_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (C * C + C * 128) * (int)sizeof(float)));
            cfg = true;
        }
        gdn_chunk_solve_kernel<true><<<g_hc, C, (C * C + C * C) * sizeof(float), stream>>>(
            lm, nullptr, b_d, wb, H, C, sb1, sb2);
        CUDA_CHECK(cudaGetLastError());
    }

    GDN_PHASE("solve_wb");

    // 3) P = Klc^T @ W_b ; R = P @ V ; G' = P @ Kh ; W2 = Lambda_C I - e^{mid} G'
    gdn_gemm(handle, 2, 128, C, C, batch, klc, CS, wb, CC, pm, CS);        // TN: [128,C]
    gdn_gemm(handle, 0, 128, 128, C, batch, pm, CS, vb, CS, rm, SS);       // NN
    gdn_gemm(handle, 0, 128, 128, C, batch, pm, CS, kh, CS, gm, SS);       // NN
    GDN_PHASE("gemm_PRG");

    gdn_w2_build_kernel<<<g_hc, 256, 0, stream>>>(gm, cl, w2, H);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("w2build");

    // 4) chunk-start states + final state (4 chunks advanced per launch)
    for (int cb = 0; cb < n_chunk; cb += 4) {
        gdn_chunk_chain_kernel<<<dim3((unsigned)H, 4), 256, 0, stream>>>(
            w2, rm, s_d, sst, state_d, H, n_chunk, cb, 4);
    }
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("chain");

    // 5) outputs: KS0 = Kh @ S_start[c] ; Oex = Qh @ S_start[c] ; U = V - e^{mid} KS0 ;
    //    D = subst(I+L, beta.U) ; B = Qh @ Kl0^T (masked) ; O = B @ D
    gdn_gemm(handle, 0, C, 128, 128, batch, kh, CS, sst, SS, ks0, CS);
    gdn_gemm(handle, 0, C, 128, 128, batch, qh, CS, sst, SS, oe, CS);
    GDN_PHASE("gemm_ks0oex");

    gdn_chunk_ufold_kernel<<<g_hc, 256, 0, stream>>>(vb, ks0, cl, ub, H);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("ufold");
    gdn_chunk_solve_kernel<false><<<g_hc, 128, (C * C + C * 128) * sizeof(float), stream>>>(
        lm, ub, b_d, dm, H, C, sb1, sb2);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("solve_d");
    gdn_gemm(handle, 1, C, C, 128, batch, qh, CS, kl0, CS, bm, CC);
    GDN_PHASE("gemm_b");

    gdn_chunk_bmask_kernel<<<g_hc, 256, 0, stream>>>(bm, H);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("bmask");
    gdn_gemm(handle, 0, C, 128, C, batch, bm, CC, dm, CS, ob, CS);
    GDN_PHASE("gemm_o");

    gdn_chunk_scatter_kernel<<<g_hc, 128, 0, stream>>>(ob, oe, cl, dst_d, H, scale);
    CUDA_CHECK(cudaGetLastError());
    GDN_PHASE("scatter");
    if (phase_prof) { phase_prof->layer_done(stream); }

    (void) tail; // tail handled by the caller (recurrent kernel) if nonzero
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t ,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // meow: chunked prefill path (WY representation) for long single-sequence
    // batches without rollback snapshots; env kill switch for A/B + fallback.
    // The gdn->cpy fusion passes a cache even for K==1 (final-state scatter only,
    // slot_stride == 0) - honor it as the state destination.
    static const bool chunked_enabled = (getenv("GGML_GDN_CHUNKED") != nullptr); // meow: opt-in, slower than recurrent today
    const bool cache_simple = (cache == nullptr) || (cache->slot_stride == 0);
    float * state_out = cache ? cache->data : state_d;
    if (chunked_enabled && cache_simple && !kda && !keep_rs && n_seqs == 1 &&
            S_v == 128 && neq0 == nev0 && n_tokens >= 2 * GDN_CHUNK) {
        const int64_t n_full = n_tokens - (n_tokens % GDN_CHUNK);

        launch_gated_delta_net_chunked(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out,
            H, n_full,
            sq1, sq2, sv1, sv2, sb1, sb2,
            neqk1, scale, stream, ctx);

        const int64_t tail = n_tokens - n_full;
        if (tail > 0) {
            // tail tokens run through the recurrent kernel, seeded by the chunked
            // path's final state (already stored, in-place state buffer)
            launch_gated_delta_net<false, false>(
                q_d + n_full * sq2, k_d + n_full * sq2, v_d + n_full * sv2,
                g_d + n_full * sb2, b_d + n_full * sb2,
                state_out, dst_d + n_full * S_v * H, state_out,
                S_v, H, tail, 1, sq1, sq2, sq2, sv1, sv2, sv2, sb1, sb2, sb3,
                neqk1, 1, scale, S_v * S_v * H, 1, stream);
        }
        return;
    }

    static const bool serial_prof_on = (getenv("GGML_GDN_PHASE_PROF") != nullptr);
    gdn_phase_prof * serial_prof = serial_prof_on ? &gdn_phase_prof::get() : nullptr;
    if (serial_prof) { serial_prof->mark("serial_start", stream); }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
    if (serial_prof) { serial_prof->mark("serial_end", stream); serial_prof->layer_done(stream); }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
