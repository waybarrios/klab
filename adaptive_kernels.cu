// Small-shape FP32 candidates. No TF32, reduced mantissas, or fast-math.
template<int K, int THREADS, int VECTOR>
__device__ __forceinline__ void tiny_fp32_impl(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int n) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    if (row >= n) return;
    x += blockIdx.y * K;
    w += row * K;
    float sum = 0.0f;
    if constexpr (VECTOR == 1) {
        #pragma unroll
        for (int col = lane; col < K; col += 32)
            sum = fmaf(x[col], w[col], sum);
    } else {
        float a = 0, b = 0, c = 0, d = 0;
        #pragma unroll
        for (int col = lane * 4; col < K; col += 128) {
            Packed4 xv = *reinterpret_cast<const Packed4*>(x + col);
            Packed4 wv = *reinterpret_cast<const Packed4*>(w + col);
            a = fmaf(xv.a, wv.a, a); b = fmaf(xv.b, wv.b, b);
            c = fmaf(xv.c, wv.c, c); d = fmaf(xv.d, wv.d, d);
        }
        sum = (a + b) + (c + d);
    }
    sum = warp_sum(sum);
    if (!lane) y[blockIdx.y * n + row] = sum;
}

#define TINY(K,T,V) \
extern "C" __global__ __launch_bounds__(T) void tiny_k##K##_t##T##_v##V( \
    const float* x, const float* w, float* y, int n) { \
    tiny_fp32_impl<K,T,V>(x,w,y,n); \
}
#define TINY_K(K) \
TINY(K,32,1) TINY(K,64,1) TINY(K,128,1) TINY(K,256,1) \
TINY(K,32,4) TINY(K,64,4) TINY(K,128,4) TINY(K,256,4)
TINY_K(64) TINY_K(128) TINY_K(256) TINY_K(512) TINY_K(1024)

// Exact FP32 significands and four-bit exponent deltas share one 28-bit
// payload stream. Wide-exponent groups retain their original FP32 bits.
__device__ __forceinline__ float unpack_fp28(unsigned code, unsigned base) {
    unsigned bits = (code & 0x7fffffu) | ((code & 0x800000u) << 8)
                  | ((base + ((code >> 24) & 15u)) << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ Packed4 fp28_load4(
    const unsigned* payload, const unsigned* metadata, int index) {
    unsigned meta = metadata[index / 64];
    const unsigned* group = payload + ((meta >> 9) << 3);
    unsigned local = index & 63;
    if (meta & 1u) {
        const float* raw = reinterpret_cast<const float*>(group + local);
        return Packed4{raw[0], raw[1], raw[2], raw[3]};
    }
    const unsigned* p = group + (local * 7 / 8);
    unsigned a=p[0], b=p[1], c=p[2], d=p[3];
    unsigned shift = (local & 4u) * 4u;
    unsigned u0 = __funnelshift_r(a,b,shift);
    unsigned u1 = shift ? __funnelshift_r(b,c,12) : __funnelshift_r(a,b,28);
    unsigned u2 = shift ? __funnelshift_r(c,d,8) : __funnelshift_r(b,c,24);
    unsigned u3 = shift ? (d >> 4) : __funnelshift_r(c,d,20);
    unsigned base = (meta >> 1) & 255u;
    return Packed4{unpack_fp28(u0,base), unpack_fp28(u1,base),
                   unpack_fp28(u2,base), unpack_fp28(u3,base)};
}

extern "C" __global__ void fp28_unpack(
    const unsigned* payload, const unsigned* metadata, float* out, int count) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (index < count)
        *reinterpret_cast<Packed4*>(out + index) = fp28_load4(payload,metadata,index);
}

template<int THREADS>
__device__ __forceinline__ void fp28_linear_impl(
    const float* __restrict__ x, const unsigned* __restrict__ payload,
    const unsigned* __restrict__ metadata, float* __restrict__ out, int n, int k) {
    int row = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    x += blockIdx.y * k;
    float a=0,b=0,c=0,d=0;
    for (int col=lane*4; col<k; col+=128) {
        Packed4 wv=fp28_load4(payload,metadata,row*k+col);
        Packed4 xv=*reinterpret_cast<const Packed4*>(x+col);
        a=fmaf(xv.a,wv.a,a); b=fmaf(xv.b,wv.b,b);
        c=fmaf(xv.c,wv.c,c); d=fmaf(xv.d,wv.d,d);
    }
    float sum=warp_sum((a+b)+(c+d));
    if (!lane) out[blockIdx.y*n+row]=sum;
}

#define FP28(T) \
extern "C" __global__ __launch_bounds__(T) void fp28_t##T( \
    const float* x, const unsigned* payload, const unsigned* metadata, \
    float* out, int n, int k) { \
    fp28_linear_impl<T>(x,payload,metadata,out,n,k); \
}
FP28(32) FP28(64) FP28(128) FP28(256)
