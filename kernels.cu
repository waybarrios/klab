// Independently written GPU experiments inspired by jart/matmul's register
// blocking and reuse of input vectors. This is CUDA C++, launched through C++.
// Inputs, accumulation, and outputs are FP32. No fast-math or TF32.

extern "C" __global__ void vector_add(
    const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// A deliberately simple starting point for comparing our own iterations.
// PyTorch/cuBLAS remains the performance baseline for every reported speedup.
extern "C" __global__ void gemv_serial(
    const float* x, const float* w, float* y, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.0f;
    for (int col = 0; col < k; ++col)
        acc = fmaf(x[col], w[row * k + col], acc);
    y[row] = acc;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template<int ROWS, int THREADS>
__device__ void gemv_blocked(
    const float* x, const float* w, float* y, int n, int k) {
    int first_row = blockIdx.x * ROWS;
    int token = blockIdx.y;
    float acc[ROWS] = {};
    for (int col = threadIdx.x; col < k; col += THREADS) {
        float v = x[token * k + col];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
            if (first_row + r < n)
                acc[r] = fmaf(v, w[(first_row + r) * k + col], acc[r]);
    }
    constexpr int WARPS = THREADS / 32;
    __shared__ float partial[ROWS][WARPS];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = warp_sum(acc[r]);
        if (lane == 0) partial[r][warp] = acc[r];
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = lane < WARPS ? partial[r][lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0 && first_row + r < n)
                y[token * n + first_row + r] = v;
        }
    }
}

#define GEMV(R, T) \
extern "C" __global__ void gemv_r##R##_t##T( \
    const float* x, const float* w, float* y, int n, int k) { \
    gemv_blocked<R, T>(x, w, y, n, k); \
}
GEMV(1, 64)  GEMV(1, 128) GEMV(1, 256)
GEMV(2, 64)  GEMV(2, 128) GEMV(2, 256)
GEMV(4, 64)  GEMV(4, 128) GEMV(4, 256)
GEMV(8, 64)  GEMV(8, 128) GEMV(8, 256)

struct __align__(16) Packed4 { float a, b, c, d; };

// Each warp owns an output row. No shared memory or cross-warp barrier.
// The vector path uses four independent FMA chains and 16-byte loads.
template<int THREADS, int VECTOR>
__device__ void gemv_warp_impl(
    const float* x, const float* w, float* y, int n, int k) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    if (row >= n) return; // Uniform within each warp; no block synchronization.
    x += blockIdx.y * k;
    w += row * k;
    float acc = 0.0f;
    if constexpr (VECTOR == 4) {
        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        for (int col = lane * 4; col < k; col += 128) {
            Packed4 xv = *reinterpret_cast<const Packed4*>(x + col);
            Packed4 wv = *reinterpret_cast<const Packed4*>(w + col);
            a = fmaf(xv.a, wv.a, a);
            b = fmaf(xv.b, wv.b, b);
            c = fmaf(xv.c, wv.c, c);
            d = fmaf(xv.d, wv.d, d);
        }
        acc = (a + b) + (c + d);
    } else {
        for (int col = lane; col < k; col += 32)
            acc = fmaf(x[col], w[col], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[blockIdx.y * n + row] = acc;
}

#define WARP_GEMV(T, V) \
extern "C" __global__ void gemv_warp_t##T##_v##V( \
    const float* x, const float* w, float* y, int n, int k) { \
    gemv_warp_impl<T, V>(x, w, y, n, k); \
}
WARP_GEMV(32, 1) WARP_GEMV(64, 1) WARP_GEMV(128, 1) WARP_GEMV(256, 1)
WARP_GEMV(32, 4) WARP_GEMV(64, 4) WARP_GEMV(128, 4) WARP_GEMV(256, 4)

// Reuse each weight tile across multiple tokens. Multiple output elements
// live in each thread's register accumulators, analogous to a CPU microkernel.
template<int BM, int BN, int TM, int TN>
__device__ void gemm_tiled_impl(
    const float* x, const float* w, float* y, int m, int n, int k) {
    constexpr int BK = 32;
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float xs[BM][BK + 1];
    __shared__ float ws[BN][BK + 1];
    int local_m = (threadIdx.x / (BN / TN)) * TM;
    int local_n = (threadIdx.x % (BN / TN)) * TN;
    int first_m = blockIdx.y * BM;
    int first_n = blockIdx.x * BN;
    float acc[TM][TN] = {};
    for (int base_k = 0; base_k < k; base_k += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            xs[row][col] = first_m + row < m && base_k + col < k
                ? x[(first_m + row) * k + base_k + col] : 0.0f;
        }
        for (int i = threadIdx.x; i < BN * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            ws[row][col] = first_n + row < n && base_k + col < k
                ? w[(first_n + row) * k + base_k + col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int col = 0; col < BK; ++col) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = xs[local_m + i][col];
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a, ws[local_n + j][col], acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            if (first_m + local_m + i < m && first_n + local_n + j < n)
                y[(first_m + local_m + i) * n + first_n + local_n + j] = acc[i][j];
}

#define GEMM(BM, BN, TM, TN) \
extern "C" __global__ void gemm_m##BM##_n##BN( \
    const float* x, const float* w, float* y, int m, int n, int k) { \
    gemm_tiled_impl<BM, BN, TM, TN>(x, w, y, m, n, k); \
}
GEMM(8, 32, 1, 2)
GEMM(16, 64, 1, 4)
GEMM(32, 32, 2, 2)

// One block per row. Fuse residual addition, RMS reduction, and scale.
// Keep the residual result in registers rather than a global scratch tensor.
template<int THREADS, int FIXED_K = 0>
__device__ void residual_rmsnorm_impl(
    const float* x, const float* residual, const float* weight,
    float* y, int k, float eps) {
    constexpr int WARPS = THREADS / 32;
    // Wrapper limits K to 8192, so this fixed register tile cannot overflow.
    constexpr int MAX_K = FIXED_K ? FIXED_K : 8192;
    constexpr int VALUES = (MAX_K + THREADS - 1) / THREADS;
    if constexpr (FIXED_K) k = FIXED_K;
    float values[VALUES];
    int row = blockIdx.x;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        int col = threadIdx.x + i * THREADS;
        float v = col < k ? x[row * k + col] + residual[row * k + col] : 0.0f;
        values[i] = v;
        sum += v * v;
    }
    sum = warp_sum(sum);
    __shared__ float partial[WARPS];
    __shared__ float inv_rms;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < WARPS ? partial[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(sum / k + eps);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        int col = threadIdx.x + i * THREADS;
        if (col < k) y[row * k + col] = (values[i] * inv_rms) * weight[col];
    }
}

#define RMS(T) \
extern "C" __global__ void residual_rmsnorm_t##T( \
    const float* x, const float* residual, const float* weight, \
    float* y, int k, float eps) { \
    residual_rmsnorm_impl<T>(x, residual, weight, y, k, eps); \
}
RMS(128) RMS(256)

#define RMS_FIXED(T) \
extern "C" __global__ void residual_rmsnorm_k4096_t##T( \
    const float* x, const float* residual, const float* weight, \
    float* y, int k, float eps) { \
    residual_rmsnorm_impl<T, 4096>(x, residual, weight, y, k, eps); \
}
RMS_FIXED(128) RMS_FIXED(256)
