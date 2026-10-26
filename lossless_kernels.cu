// Lossless FP32 storage. Preserve all 23 mantissa bits and the sign bit;
// store block-relative exponents as 4 bits, or raw 8-bit exponents as fallback.
// Every decoded weight has exactly the original IEEE-754 bit pattern.
__device__ __forceinline__ Packed4 lossless_load4(
    const unsigned char* mantissa, const unsigned char* exponent,
    const unsigned* metadata, int index) {
    const unsigned* p = reinterpret_cast<const unsigned*>(mantissa + index * 3);
    unsigned a = p[0], b = p[1], c = p[2];
    unsigned m[4] = {a & 0xffffffu, (a >> 24) | ((b & 0xffffu) << 8),
                     (b >> 16) | ((c & 255u) << 16), c >> 8};
    int group = index / 64;
    int local = index % 64;
    unsigned meta = metadata[group];
    bool is_wide = (meta & 1u) != 0;
    unsigned offset = (meta >> 9) << 5;
    const unsigned char* e = exponent + offset + (is_wide ? local : local / 2);
    unsigned exps = is_wide ? *reinterpret_cast<const unsigned*>(e)
                            : *reinterpret_cast<const unsigned short*>(e);
    unsigned bias = is_wide ? 0u : ((meta >> 1) & 255u);
    float values[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned exp = is_wide ? ((exps >> (8 * i)) & 255u)
                               : bias + ((exps >> (4 * i)) & 15u);
        unsigned bits = (m[i] & 0x7fffffu) | ((m[i] & 0x800000u) << 8) | (exp << 23);
        values[i] = __uint_as_float(bits);
    }
    return Packed4{values[0], values[1], values[2], values[3]};
}

extern "C" __global__ void lossless_unpack(
    const unsigned char* mantissa, const unsigned char* exponent,
    const unsigned* metadata, float* out, int count) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (index < count)
        *reinterpret_cast<Packed4*>(out + index) = lossless_load4(mantissa, exponent, metadata, index);
}

template<int THREADS>
__device__ void lossless_linear_impl(
    const float* x, const unsigned char* mantissa, const unsigned char* exponent,
    const unsigned* metadata, float* out, int n, int k) {
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    float a = 0, b = 0, c = 0, d = 0;
    x += blockIdx.y * k;
    for (int col = lane * 4; col < k; col += 128) {
        Packed4 weight = lossless_load4(mantissa, exponent, metadata, row * k + col);
        Packed4 value = *reinterpret_cast<const Packed4*>(x + col);
        a = fmaf(value.a, weight.a, a); b = fmaf(value.b, weight.b, b);
        c = fmaf(value.c, weight.c, c); d = fmaf(value.d, weight.d, d);
    }
    float sum = warp_sum((a + b) + (c + d));
    if (!lane) out[blockIdx.y * n + row] = sum;
}

#define LOSSLESS(T) \
extern "C" __global__ __launch_bounds__(T) void lossless_t##T( \
    const float* x, const unsigned char* mantissa, const unsigned char* exponent, \
    const unsigned* metadata, \
    float* out, int n, int k) { \
    lossless_linear_impl<T>(x, mantissa, exponent, metadata, out, n, k); \
}
LOSSLESS(32) LOSSLESS(64) LOSSLESS(128) LOSSLESS(256)
