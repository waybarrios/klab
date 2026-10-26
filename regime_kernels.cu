// Observational memory-regime probes, compiled alone by cuda_module.CudaModule
// (NVRTC, no headers). Integer only: uint32 sums wrap mod 2^32; no atomics,
// shared memory or floating point. 1D blocks, 2D grid, zero dynamic shared
// memory. input/output must be 16-byte aligned; the host checks this.

struct __align__(16) RegimeWords4 { unsigned a, b, c, d; };

__device__ __forceinline__ unsigned long long regime_thread() {
    return (static_cast<unsigned long long>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x
           + threadIdx.x;
}

__device__ __forceinline__ unsigned long long regime_stride() {
    return static_cast<unsigned long long>(gridDim.x) * gridDim.y * blockDim.x;
}

__device__ __forceinline__ unsigned regime_sum(RegimeWords4 v) { return v.a + v.b + v.c + v.d; }

// Thread t of T sums 16-byte vectors t, t+T, ... (four independent loads in
// flight), then tail words 4*(words/4)+t, +T, ..., and stores the exact sum to
// sink[t]. Every launch reads all `words` input words once; the host checks
// every sink entry against a closed form.
extern "C" __global__ void regime_read(const unsigned* __restrict__ input,
                                       unsigned* __restrict__ sink, unsigned long long words) {
    const RegimeWords4* in = reinterpret_cast<const RegimeWords4*>(input);
    unsigned long long t = regime_thread(), stride = regime_stride(), vectors = words / 4;
    unsigned s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
    unsigned long long v = t;
    for (; v + 3 * stride < vectors; v += 4 * stride) {
        RegimeWords4 a = in[v], b = in[v + stride], c = in[v + 2 * stride], d = in[v + 3 * stride];
        s0 += regime_sum(a);
        s1 += regime_sum(b);
        s2 += regime_sum(c);
        s3 += regime_sum(d);
    }
    for (; v < vectors; v += stride) s0 += regime_sum(in[v]);
    for (unsigned long long i = vectors * 4 + t; i < words; i += stride) s1 += input[i];
    sink[t] = s0 + s1 + s2 + s3;
}

// Same traversal as regime_read: copies all `words` words, 16 bytes at a time
// plus a scalar tail.
extern "C" __global__ void regime_copy(const unsigned* __restrict__ input,
                                       unsigned* __restrict__ output, unsigned long long words) {
    const RegimeWords4* in = reinterpret_cast<const RegimeWords4*>(input);
    RegimeWords4* out = reinterpret_cast<RegimeWords4*>(output);
    unsigned long long t = regime_thread(), stride = regime_stride(), vectors = words / 4;
    unsigned long long v = t;
    for (; v + 3 * stride < vectors; v += 4 * stride) {
        RegimeWords4 a = in[v], b = in[v + stride], c = in[v + 2 * stride], d = in[v + 3 * stride];
        out[v] = a;
        out[v + stride] = b;
        out[v + 2 * stride] = c;
        out[v + 3 * stride] = d;
    }
    for (; v < vectors; v += stride) out[v] = in[v];
    for (unsigned long long i = vectors * 4 + t; i < words; i += stride) output[i] = input[i];
}

// Near-empty launch-floor probe. One thread increments an observable counter,
// so the host can check that every stream-ordered launch executed.
extern "C" __global__ void regime_launch(unsigned* counter) {
    if (regime_thread() == 0) counter[0] += 1u;
}
