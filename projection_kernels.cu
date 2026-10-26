// Large-projection experiments, concatenated after kernels.cu by NVRTC.
// These keep FP32 inputs, accumulation, and outputs.

template<bool CG>
__device__ __forceinline__ Packed4 projection_load4(const float* ptr) {
    Packed4 v;
    if constexpr (CG) {
        asm("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(v.a), "=f"(v.b), "=f"(v.c), "=f"(v.d) : "l"(ptr));
    } else {
        v = *reinterpret_cast<const Packed4*>(ptr);
    }
    return v;
}

template<int THREADS, bool CG, int PREFETCH>
__device__ void projection_stream_impl(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int n, int k) {
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    x += blockIdx.y * k;
    w += row * k;
    float a = 0, b = 0, c = 0, d = 0;
    #pragma unroll 4
    for (int col = lane * 4; col < k; col += 128) {
        if constexpr (PREFETCH) {
            if (col + PREFETCH * 128 < k)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(w + col + PREFETCH * 128));
        }
        Packed4 xv = projection_load4<false>(x + col);
        Packed4 wv = projection_load4<CG>(w + col);
        a = fmaf(xv.a, wv.a, a); b = fmaf(xv.b, wv.b, b);
        c = fmaf(xv.c, wv.c, c); d = fmaf(xv.d, wv.d, d);
    }
    float sum = warp_sum((a + b) + (c + d));
    if (!lane) y[blockIdx.y * n + row] = sum;
}

#define STREAM(T, C, P) \
extern "C" __global__ __launch_bounds__(T) void proj_stream_t##T##_c##C##_p##P( \
    const float* x, const float* w, float* y, int n, int k) { \
    projection_stream_impl<T, C, P>(x, w, y, n, k); \
}
STREAM(64, 0, 0) STREAM(64, 1, 0) STREAM(64, 1, 4)
STREAM(128, 0, 0) STREAM(128, 1, 0) STREAM(128, 1, 4)
STREAM(256, 0, 0) STREAM(256, 1, 0) STREAM(256, 1, 4)

// Physical weights are packed as wt[K,N]. This packing is performed once,
// outside the operation, and is also offered to cuBLAS for a fair comparison.
template<int THREADS>
__device__ void projection_packed_impl(
    const float* __restrict__ x, const float* __restrict__ wt,
    float* __restrict__ partial, int n, int k, int split) {
    int col = blockIdx.x * THREADS + threadIdx.x;
    int chunk = ((k + split * 4 - 1) / (split * 4)) * 4;
    int begin = blockIdx.y * chunk;
    int end = begin + chunk < k ? begin + chunk : k;
    if (col >= n) return;
    float a = 0, b = 0, c = 0, d = 0;
    int q = begin;
    for (; q + 3 < end; q += 4) {
        a = fmaf(x[q],     wt[q * n + col], a);
        b = fmaf(x[q + 1], wt[(q + 1) * n + col], b);
        c = fmaf(x[q + 2], wt[(q + 2) * n + col], c);
        d = fmaf(x[q + 3], wt[(q + 3) * n + col], d);
    }
    float sum = (a + b) + (c + d);
    for (; q < end; ++q) sum = fmaf(x[q], wt[q * n + col], sum);
    partial[blockIdx.y * n + col] = sum;
}

#define PACKED(T) \
extern "C" __global__ void proj_packed_t##T( \
    const float* x, const float* wt, float* partial, int n, int k, int split) { \
    projection_packed_impl<T>(x, wt, partial, n, k, split); \
}
PACKED(64) PACKED(128) PACKED(256)

extern "C" __global__ void proj_reduce(
    const float* partial, float* y, int n, int split) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0;
        for (int s = 0; s < split; ++s) sum += partial[s * n + col];
        y[col] = sum;
    }
}
