// Exact BF16 projections over docs/bf16-format.md version 1. Compiled alone by
// cuda_module.CudaModule (NVRTC, no headers); never part of CUDA_SOURCES.
// x and weights are native BF16 bits widened exactly (bits << 16). Every
// product/accumulation is one explicit fma.rn.f32 or add.rn.f32: no FTZ,
// fast-math, Tensor Cores, shared memory or atomics. Raw and coded kernels share
// cbf16_linear and differ only in the weight load, so within one launch
// configuration their FMA order and reduction tree are identical. FP32 output
// stores the sum; BF16 output rounds it once with cvt.rn.bf16.f32 (SM80+).
// Packed buffers are CPU-validated and immutable; the host checks shapes,
// alignment and K % 64 == 0 for coded projections. 1D blocks, 2D grid.

struct __align__(16) Cbf16Words4 { unsigned a, b, c, d; };  // Eight BF16 values.

__device__ __forceinline__ float cbf16_fma(float x, float w, float acc) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(x), "f"(w), "f"(acc));
    return result;
}

__device__ __forceinline__ float cbf16_add(float a, float b) {
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ __forceinline__ float cbf16_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
        value = cbf16_add(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float cbf16_lo(unsigned word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float cbf16_hi(unsigned word) { return __uint_as_float(word & 0xffff0000u); }

// Bits above the low twelve of `code` are ignored. `base` is metadata & 0x7f800000:
// the block's base exponent, already at FP32 exponent bits 23..30.
__device__ __forceinline__ float cbf16_decode(unsigned code, unsigned base) {
    return __uint_as_float(((code & 0x7fu) << 16) | ((code & 0x80u) << 24) |
                           (base + ((code & 0xf00u) << 15)));
}

__device__ __forceinline__ void cbf16_widen8(Cbf16Words4 v, float (&f)[8]) {
    f[0] = cbf16_lo(v.a); f[1] = cbf16_hi(v.a);
    f[2] = cbf16_lo(v.b); f[3] = cbf16_hi(v.b);
    f[4] = cbf16_lo(v.c); f[5] = cbf16_hi(v.c);
    f[6] = cbf16_lo(v.d); f[7] = cbf16_hi(v.d);
}

// Weights col..col+7 of `row`, col % 8 == 0. Coded: chunk (col % 64) / 8 is
// twelve bytes (three aligned words) of a coded block or sixteen of a raw one.
template<bool CODED>
__device__ __forceinline__ void cbf16_weights8(
    const unsigned char* __restrict__ weights, const unsigned* __restrict__ metadata,
    int row, int col, int k, float (&w)[8]) {
    if constexpr (CODED) {
        unsigned meta = metadata[row * (k >> 6) + (col >> 6)];
        const unsigned char* block = weights + ((meta & 0x7fffffu) << 5);
        int chunk = (col >> 3) & 7;
        if (meta >> 31) {
            cbf16_widen8(*reinterpret_cast<const Cbf16Words4*>(block + 16 * chunk), w);
        } else {
            const unsigned* p = reinterpret_cast<const unsigned*>(block + 12 * chunk);
            unsigned a = p[0], b = p[1], c = p[2], base = meta & 0x7f800000u;
            w[0] = cbf16_decode(a, base);
            w[1] = cbf16_decode(a >> 12, base);
            w[2] = cbf16_decode((a >> 24) | (b << 8), base);
            w[3] = cbf16_decode(b >> 4, base);
            w[4] = cbf16_decode(b >> 16, base);
            w[5] = cbf16_decode((b >> 28) | (c << 4), base);
            w[6] = cbf16_decode(c >> 8, base);
            w[7] = cbf16_decode(c >> 20, base);
        }
    } else {
        const unsigned short* w16 = reinterpret_cast<const unsigned short*>(weights);
        cbf16_widen8(*reinterpret_cast<const Cbf16Words4*>(w16 + row * k + col), w);
    }
}

// Weights col, col+1 of `row`, col % 2 == 0. A coded pair is three bytes, read
// bytewise; a raw pair is one aligned word.
template<bool CODED>
__device__ __forceinline__ void cbf16_weights2(
    const unsigned char* __restrict__ weights, const unsigned* __restrict__ metadata,
    int row, int col, int k, float& w0, float& w1) {
    if constexpr (CODED) {
        unsigned meta = metadata[row * (k >> 6) + (col >> 6)];
        const unsigned char* block = weights + ((meta & 0x7fffffu) << 5);
        int pair = (col & 63) >> 1;
        if (meta >> 31) {
            unsigned v = *reinterpret_cast<const unsigned*>(block + 4 * pair);
            w0 = cbf16_lo(v);
            w1 = cbf16_hi(v);
        } else {
            const unsigned char* p = block + 3 * pair;
            unsigned b0 = p[0], b1 = p[1], b2 = p[2], base = meta & 0x7f800000u;
            w0 = cbf16_decode(b0 | (b1 << 8), base);
            w1 = cbf16_decode((b1 >> 4) | (b2 << 4), base);
        }
    } else {
        const unsigned short* w16 = reinterpret_cast<const unsigned short*>(weights);
        unsigned v = *reinterpret_cast<const unsigned*>(w16 + row * k + col);
        w0 = cbf16_lo(v);
        w1 = cbf16_hi(v);
    }
}

// One warp per (output row, token). VECTOR=2: 4-byte x loads, two accumulators.
// VECTOR=8: 16-byte x loads, eight accumulators. out[token, row] = x[token] . w[row].
template<bool CODED, bool BF16_OUT, int THREADS, int VECTOR>
__device__ __forceinline__ void cbf16_linear(
    const unsigned short* __restrict__ x, const unsigned char* __restrict__ weights,
    const unsigned* __restrict__ metadata, void* __restrict__ out, int n, int k) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    if (row >= n) return;  // Uniform within each warp; no block synchronization.
    const unsigned short* xr = x + blockIdx.y * k;
    float sum;
    if constexpr (VECTOR == 8) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 4
        for (int col = lane * 8; col < k; col += 256) {
            float xv[8], wv[8];
            cbf16_widen8(*reinterpret_cast<const Cbf16Words4*>(xr + col), xv);
            cbf16_weights8<CODED>(weights, metadata, row, col, k, wv);
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = cbf16_fma(xv[e], wv[e], acc[e]);
        }
        sum = cbf16_add(cbf16_add(cbf16_add(acc[0], acc[1]), cbf16_add(acc[2], acc[3])),
                        cbf16_add(cbf16_add(acc[4], acc[5]), cbf16_add(acc[6], acc[7])));
    } else {
        float a = 0.0f, b = 0.0f;
        #pragma unroll 4
        for (int col = lane * 2; col < k; col += 64) {
            unsigned xv = *reinterpret_cast<const unsigned*>(xr + col);
            float w0, w1;
            cbf16_weights2<CODED>(weights, metadata, row, col, k, w0, w1);
            a = cbf16_fma(cbf16_lo(xv), w0, a);
            b = cbf16_fma(cbf16_hi(xv), w1, b);
        }
        sum = cbf16_add(a, b);
    }
    sum = cbf16_warp_sum(sum);
    if (!lane) {
        int index = blockIdx.y * n + row;
        if constexpr (BF16_OUT) {
            unsigned short bits;
            asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(sum));
            static_cast<unsigned short*>(out)[index] = bits;
        } else {
            static_cast<float*>(out)[index] = sum;
        }
    }
}

#define CBF16_CONFIG(T, V) \
extern "C" __global__ __launch_bounds__(T) void cbf16_raw_fp32_t##T##_v##V( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k) { \
    cbf16_linear<false, false, T, V>(x, reinterpret_cast<const unsigned char*>(w), nullptr, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void cbf16_raw_bf16_t##T##_v##V( \
    const unsigned short* x, const unsigned short* w, unsigned short* out, int n, int k) { \
    cbf16_linear<false, true, T, V>(x, reinterpret_cast<const unsigned char*>(w), nullptr, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void cbf16_coded_fp32_t##T##_v##V( \
    const unsigned short* x, const unsigned char* payload, const unsigned* metadata, \
    float* out, int n, int k) { \
    cbf16_linear<true, false, T, V>(x, payload, metadata, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void cbf16_coded_bf16_t##T##_v##V( \
    const unsigned short* x, const unsigned char* payload, const unsigned* metadata, \
    unsigned short* out, int n, int k) { \
    cbf16_linear<true, true, T, V>(x, payload, metadata, out, n, k); \
}
CBF16_CONFIG(32, 2) CBF16_CONFIG(64, 2) CBF16_CONFIG(128, 2) CBF16_CONFIG(256, 2)
CBF16_CONFIG(32, 8) CBF16_CONFIG(64, 8) CBF16_CONFIG(128, 8) CBF16_CONFIG(256, 8)

// Integer-only decode to the original uint16 bits, one thread per weight, for
// bitwise validation. Any K (ragged blocks included); reads stay inside each
// block's 96 or 128 payload bytes. Launch 256 threads per block, N*K < 2^31.
extern "C" __global__ __launch_bounds__(256) void cbf16_unpack(
    const unsigned char* __restrict__ payload, const unsigned* __restrict__ metadata,
    unsigned short* __restrict__ out, int n, int k) {
    unsigned index = blockIdx.x * 256u + threadIdx.x;
    if (index >= static_cast<unsigned>(n) * static_cast<unsigned>(k)) return;
    unsigned columns = static_cast<unsigned>(k);
    unsigned row = index / columns, col = index - row * columns;
    unsigned meta = metadata[row * ((columns + 63u) >> 6) + (col >> 6)];
    const unsigned char* block = payload + ((meta & 0x7fffffu) << 5);
    unsigned j = col & 63u, bits;
    if (meta >> 31) {
        bits = block[2 * j] | (static_cast<unsigned>(block[2 * j + 1]) << 8);
    } else {
        const unsigned char* p = block + ((3 * j) >> 1);
        unsigned lo = p[0], hi = p[1];
        unsigned code = (j & 1u) ? (lo >> 4) | (hi << 4) : lo | ((hi & 15u) << 8);
        bits = (code & 0x7fu) | ((code & 0x80u) << 8) | ((((meta >> 23) & 255u) + (code >> 8)) << 7);
    }
    out[index] = static_cast<unsigned short>(bits);
}
