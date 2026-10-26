// Software E2M1 weights + per-16 E4M3 scales + global FP32 scale on SM 8.6.
// This is a custom row-major NVFP4-style value encoding, not a native FP4 MMA.

__device__ __forceinline__ float fp4_decode(unsigned code) {
    unsigned magnitude = code & 7u;
    unsigned bits = magnitude < 2u ? magnitude * 0x3f000000u
                                  : 0x3f000000u + (magnitude << 22);
    return __uint_as_float(bits | ((code & 8u) << 28));
}

__device__ __forceinline__ float fp8_positive_decode(unsigned code) {
    unsigned exponent = code >> 3;
    unsigned mantissa = code & 7u;
    return exponent ? __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20))
                    : mantissa * (1.0f / 512.0f);
}

template<bool HALF>
__device__ __forceinline__ float fp4_activation(const void* x, int index) {
    if constexpr (HALF) {
        unsigned short bits = static_cast<const unsigned short*>(x)[index];
        float result;
        asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
        return result;
    } else {
        return static_cast<const float*>(x)[index];
    }
}

template<int THREADS, bool HALF>
__device__ void fp4_linear_impl(
    const void* __restrict__ x, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales, const float* __restrict__ global_scale,
    float* __restrict__ out, int n, int k) {
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    int token = blockIdx.y;
    packed += row * (k / 2);
    scales += row * (k / 16);
    float acc[4] = {};
    for (int col = lane * 8; col < k; col += 256) {
        unsigned bits = *reinterpret_cast<const unsigned*>(packed + col / 2);
        float scale = fp8_positive_decode(scales[col / 16]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float weight = fp4_decode((bits >> (4 * i)) & 15u) * scale;
            float value = fp4_activation<HALF>(x, token * k + col + i);
            acc[i % 4] = fmaf(value, weight, acc[i % 4]);
        }
    }
    float sum = warp_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
    if (!lane) out[token * n + row] = sum * global_scale[0];
}

#define FP4(T, A, H) \
extern "C" __global__ __launch_bounds__(T) void fp4_a##A##_t##T( \
    const void* x, const unsigned char* packed, const unsigned char* scales, \
    const float* global_scale, float* out, int n, int k) { \
    fp4_linear_impl<T, H>(x, packed, scales, global_scale, out, n, k); \
}
FP4(32, 16, true) FP4(64, 16, true) FP4(128, 16, true) FP4(256, 16, true)
FP4(32, 32, false) FP4(64, 32, false) FP4(128, 32, false) FP4(256, 32, false)
