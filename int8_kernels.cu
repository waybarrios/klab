// Signed 8-bit weights, FP32 scale per 64 values, FP16/FP32 activations.
// All multiplication and accumulation is FP32. No integer dot approximation.
template<int THREADS, bool HALF>
__device__ void int8_linear_impl(
    const void* __restrict__ x, const signed char* __restrict__ weight,
    const float* __restrict__ scales, float* __restrict__ out, int n, int k) {
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    weight += row * k;
    scales += row * (k / 64);
    float acc[4] = {};
    for (int col = lane * 4; col < k; col += 128) {
        unsigned bits = *reinterpret_cast<const unsigned*>(weight + col);
        float scale = scales[col / 64];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int code = static_cast<signed char>(bits >> (8 * i));
            float w = static_cast<float>(code) * scale;
            float v = fp4_activation<HALF>(x, blockIdx.y * k + col + i);
            acc[i] = fmaf(v, w, acc[i]);
        }
    }
    float sum = warp_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
    if (!lane) out[blockIdx.y * n + row] = sum;
}

#define INT8(T, A, H) \
extern "C" __global__ __launch_bounds__(T) void int8_a##A##_t##T( \
    const void* x, const signed char* w, const float* scales, float* out, int n, int k) { \
    int8_linear_impl<T, H>(x, w, scales, out, n, k); \
}
INT8(32, 16, true) INT8(64, 16, true) INT8(128, 16, true) INT8(256, 16, true)
INT8(32, 32, false) INT8(64, 32, false) INT8(128, 32, false) INT8(256, 32, false)
