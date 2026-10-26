// Revision-2 deterministic M tiling of the exact BF16 projections. Never compiled
// alone: mtile_bf16.get_module appends this file after the full, unchanged
// compressed_kernels.cu (NVRTC, no headers) and reuses its Cbf16Words4,
// cbf16_weights8/2, cbf16_widen8, cbf16_lo/hi, cbf16_fma/add and cbf16_warp_sum.
// One warp per output row and up to MT tokens (grid.y = ceil(M / MT)): each
// weight vector is loaded and decoded once into FP32 registers, then reused by
// every token of the tile. Per token the column loop (stride and order), the
// explicit fma.rn.f32 chain per accumulator, the V8 pairwise tree / V2 a + b,
// cbf16_warp_sum and the single cvt.rn.bf16.f32 are exactly cbf16_linear's, so
// out[token, row] matches R1 bitwise for the same (THREADS, VECTOR). No shared
// memory, atomics, FTZ or fast-math. The host validates shapes, alignment,
// K % 64 == 0 for coded weights, M <= 65535 and 32-bit indexing.

// Column-loop unroll per tile: a fixed register-pressure rule (MT * VECTOR
// accumulators are live), not a tuning dimension. Unrolling never reorders
// any accumulator's FMA chain.
template<int MT>
struct MtBf16Unroll { static constexpr int value = MT <= 2 ? 4 : (MT == 4 ? 2 : 1); };

// Tile sums for tokens 0..tokens-1 of xt (token-major, stride k). GUARD=false
// requires tokens >= MT; GUARD=true never touches x of tokens >= `tokens`.
template<bool CODED, int VECTOR, int MT, int UNROLL, bool GUARD>
__device__ __forceinline__ void mtbf16_tile(
    const unsigned short* __restrict__ xt, const unsigned char* __restrict__ weights,
    const unsigned* __restrict__ metadata, int row, int lane, int k, int tokens,
    float (&sum)[MT]) {
    if constexpr (VECTOR == 8) {
        float acc[MT][8];
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;
        }
        #pragma unroll (UNROLL)
        for (int col = lane * 8; col < k; col += 256) {
            float wv[8];
            cbf16_weights8<CODED>(weights, metadata, row, col, k, wv);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (!GUARD || t < tokens) {
                    float xv[8];
                    cbf16_widen8(*reinterpret_cast<const Cbf16Words4*>(xt + t * k + col), xv);
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] = cbf16_fma(xv[e], wv[e], acc[t][e]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < MT; ++t)
            sum[t] = cbf16_add(cbf16_add(cbf16_add(acc[t][0], acc[t][1]), cbf16_add(acc[t][2], acc[t][3])),
                               cbf16_add(cbf16_add(acc[t][4], acc[t][5]), cbf16_add(acc[t][6], acc[t][7])));
    } else {
        float a[MT], b[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t) a[t] = b[t] = 0.0f;
        #pragma unroll (UNROLL)
        for (int col = lane * 2; col < k; col += 64) {
            float w0, w1;
            cbf16_weights2<CODED>(weights, metadata, row, col, k, w0, w1);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (!GUARD || t < tokens) {
                    unsigned xv = *reinterpret_cast<const unsigned*>(xt + t * k + col);
                    a[t] = cbf16_fma(cbf16_lo(xv), w0, a[t]);
                    b[t] = cbf16_fma(cbf16_hi(xv), w1, b[t]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < MT; ++t) sum[t] = cbf16_add(a[t], b[t]);
    }
}

// Warp-reduce and store tokens 0..tokens-1; the guard is uniform per warp, so
// every __shfl_down_sync sees all 32 lanes.
template<bool BF16_OUT, int MT, bool GUARD>
__device__ __forceinline__ void mtbf16_store(
    float (&sum)[MT], void* __restrict__ out, int n, int row, int lane, int token0, int tokens) {
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (!GUARD || t < tokens) {
            float total = cbf16_warp_sum(sum[t]);
            if (!lane) {
                int index = (token0 + t) * n + row;
                if constexpr (BF16_OUT) {
                    unsigned short bits;
                    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(total));
                    static_cast<unsigned short*>(out)[index] = bits;
                } else {
                    static_cast<float*>(out)[index] = total;
                }
            }
        }
    }
}

// out[token, row] = x[token] . w[row] for tokens blockIdx.y*MT .. min(M, +MT)-1.
// Full tiles run unguarded; only the ragged last tile row takes the guarded path.
template<bool CODED, bool BF16_OUT, int THREADS, int VECTOR, int MT>
__device__ __forceinline__ void mtbf16_linear(
    const unsigned short* __restrict__ x, const unsigned char* __restrict__ weights,
    const unsigned* __restrict__ metadata, void* __restrict__ out, int n, int k, int m) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    if (row >= n) return;  // Uniform within each warp; no block synchronization.
    int token0 = blockIdx.y * MT;
    int tokens = m - token0;
    if (tokens <= 0) return;  // Uniform per block; unreachable with grid.y = ceil(M / MT).
    const unsigned short* xt = x + token0 * k;
    float sum[MT];
    if (tokens >= MT) {
        mtbf16_tile<CODED, VECTOR, MT, MtBf16Unroll<MT>::value, false>(
            xt, weights, metadata, row, lane, k, MT, sum);
        mtbf16_store<BF16_OUT, MT, false>(sum, out, n, row, lane, token0, MT);
    } else {
        mtbf16_tile<CODED, VECTOR, MT, MtBf16Unroll<MT>::value, true>(
            xt, weights, metadata, row, lane, k, tokens, sum);
        mtbf16_store<BF16_OUT, MT, true>(sum, out, n, row, lane, token0, tokens);
    }
}

#define MTBF16_KERNELS(T, V, MT) \
extern "C" __global__ __launch_bounds__(T) void mtbf16_raw_fp32_t##T##_v##V##_m##MT( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k, int m) { \
    mtbf16_linear<false, false, T, V, MT>(x, reinterpret_cast<const unsigned char*>(w), nullptr, out, n, k, m); \
} \
extern "C" __global__ __launch_bounds__(T) void mtbf16_raw_bf16_t##T##_v##V##_m##MT( \
    const unsigned short* x, const unsigned short* w, unsigned short* out, int n, int k, int m) { \
    mtbf16_linear<false, true, T, V, MT>(x, reinterpret_cast<const unsigned char*>(w), nullptr, out, n, k, m); \
} \
extern "C" __global__ __launch_bounds__(T) void mtbf16_coded_fp32_t##T##_v##V##_m##MT( \
    const unsigned short* x, const unsigned char* payload, const unsigned* metadata, \
    float* out, int n, int k, int m) { \
    mtbf16_linear<true, false, T, V, MT>(x, payload, metadata, out, n, k, m); \
} \
extern "C" __global__ __launch_bounds__(T) void mtbf16_coded_bf16_t##T##_v##V##_m##MT( \
    const unsigned short* x, const unsigned char* payload, const unsigned* metadata, \
    unsigned short* out, int n, int k, int m) { \
    mtbf16_linear<true, true, T, V, MT>(x, payload, metadata, out, n, k, m); \
}
#define MTBF16_CONFIG(T, V) \
    MTBF16_KERNELS(T, V, 2) MTBF16_KERNELS(T, V, 4) MTBF16_KERNELS(T, V, 8) MTBF16_KERNELS(T, V, 16)
MTBF16_CONFIG(32, 2) MTBF16_CONFIG(64, 2) MTBF16_CONFIG(128, 2) MTBF16_CONFIG(256, 2)
MTBF16_CONFIG(32, 8) MTBF16_CONFIG(64, 8) MTBF16_CONFIG(128, 8) MTBF16_CONFIG(256, 8)
