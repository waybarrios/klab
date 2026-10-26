// FP16/BF16-input projections. The supplied 16-bit values are widened exactly
// to FP32 (cvt.f32.f16, or BF16 bits << 16); FMA accumulation and output are
// FP32. No half arithmetic, Tensor Cores, TF32, fast-math, or atomics.
// Concatenated after kernels.cu, which defines warp_sum.

struct __align__(16) HalfWords4 { unsigned a, b, c, d; };  // Eight 16-bit values.

template<bool BF16>
__device__ __forceinline__ float half_widen(unsigned short bits) {
    if constexpr (BF16) {
        return __uint_as_float(static_cast<unsigned>(bits) << 16);
    } else {
        float value;
        asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
        return value;
    }
}

// Low/high 16-bit halves feed two independent FMA chains.
template<bool BF16>
__device__ __forceinline__ void half_fma2(unsigned x, unsigned w, float& lo, float& hi) {
    lo = fmaf(half_widen<BF16>(static_cast<unsigned short>(x & 0xffffu)),
              half_widen<BF16>(static_cast<unsigned short>(w & 0xffffu)), lo);
    hi = fmaf(half_widen<BF16>(static_cast<unsigned short>(x >> 16)),
              half_widen<BF16>(static_cast<unsigned short>(w >> 16)), hi);
}

// One warp per output row. VECTOR=2: 4-byte loads (K%2==0, 4-byte aligned).
// VECTOR=8: 16-byte loads (K%8==0, 16-byte aligned). Host code checks both.
template<bool BF16, int THREADS, int VECTOR>
__device__ __forceinline__ void half_warp_impl(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
    float* __restrict__ out, int n, int k) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    if (row >= n) return;  // Uniform within each warp; no block synchronization.
    const unsigned* xw = reinterpret_cast<const unsigned*>(x + blockIdx.y * k);
    const unsigned* ww = reinterpret_cast<const unsigned*>(w + row * k);
    int words = k / 2;
    float a = 0.0f, b = 0.0f;
    if constexpr (VECTOR == 8) {
        float c = 0.0f, d = 0.0f;
        #pragma unroll 4
        for (int i = lane * 4; i < words; i += 128) {
            HalfWords4 xv = *reinterpret_cast<const HalfWords4*>(xw + i);
            HalfWords4 wv = *reinterpret_cast<const HalfWords4*>(ww + i);
            half_fma2<BF16>(xv.a, wv.a, a, b);
            half_fma2<BF16>(xv.b, wv.b, c, d);
            half_fma2<BF16>(xv.c, wv.c, a, b);
            half_fma2<BF16>(xv.d, wv.d, c, d);
        }
        a += c;
        b += d;
    } else {
        #pragma unroll 4
        for (int i = lane; i < words; i += 32)
            half_fma2<BF16>(xw[i], ww[i], a, b);
    }
    float sum = warp_sum(a + b);
    if (!lane) out[blockIdx.y * n + row] = sum;
}

#define HALF_WARP(T) \
extern "C" __global__ __launch_bounds__(T) void half_warp_f16_t##T( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k) { \
    half_warp_impl<false, T, 2>(x, w, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void half_warp_bf16_t##T( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k) { \
    half_warp_impl<true, T, 2>(x, w, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void half_warp_f16_t##T##_v8( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k) { \
    half_warp_impl<false, T, 8>(x, w, out, n, k); \
} \
extern "C" __global__ __launch_bounds__(T) void half_warp_bf16_t##T##_v8( \
    const unsigned short* x, const unsigned short* w, float* out, int n, int k) { \
    half_warp_impl<true, T, 8>(x, w, out, n, k); \
}
HALF_WARP(32) HALF_WARP(64) HALF_WARP(128) HALF_WARP(256)
