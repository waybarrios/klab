// Single-kernel GQA decode attention for one new token (M=1), BF16 in/out, FP32 softmax and accumulation.
// Fuses the KV-cache append: the first query head of each KV group writes k_new/v_new at position `len`;
// every head uses k_new/v_new from the arguments for that position, so no block reads a slot being written.
// Cache layout [kv_heads, max_len, 64]. Requires len + 1 <= 4096 (shared score buffer), checked on the host.

#define HD 64
#define MAXT 4096

__device__ __forceinline__ float bf(unsigned short b) { return __uint_as_float(static_cast<unsigned>(b) << 16); }

__device__ __forceinline__ float warp_sum_f(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_f(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

extern "C" __global__ __launch_bounds__(256) void decode_attn_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, int len, int max_len, int heads, int kv_heads, float scale) {
    __shared__ float s[MAXT];
    __shared__ float red[8];
    __shared__ float part[4][HD];
    const int head = blockIdx.x, group = heads / kv_heads, kvh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    if (head % group == 0 && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    // Each lane owns dims 2*lane, 2*lane+1 of the query.
    const unsigned qw = reinterpret_cast<const unsigned*>(q + head * HD)[lane];
    const float q0 = bf(qw & 0xffffu) * scale, q1 = bf(qw >> 16) * scale;
    // Scores for cached tokens: warp w takes t = w, w+8, ...; two tokens per step for ILP.
    int t = warp;
    for (; t + 8 < len; t += 16) {
        unsigned a = reinterpret_cast<const unsigned*>(K + (size_t)t * HD)[lane];
        unsigned b = reinterpret_cast<const unsigned*>(K + (size_t)(t + 8) * HD)[lane];
        float sa = warp_sum_f(q0 * bf(a & 0xffffu) + q1 * bf(a >> 16));
        float sb = warp_sum_f(q0 * bf(b & 0xffffu) + q1 * bf(b >> 16));
        if (!lane) { s[t] = sa; s[t + 8] = sb; }
    }
    if (t < len) {
        unsigned a = reinterpret_cast<const unsigned*>(K + (size_t)t * HD)[lane];
        float sa = warp_sum_f(q0 * bf(a & 0xffffu) + q1 * bf(a >> 16));
        if (!lane) s[t] = sa;
    }
    if (warp == 7) {  // the new token, from the argument
        unsigned a = reinterpret_cast<const unsigned*>(kn)[lane];
        float sa = warp_sum_f(q0 * bf(a & 0xffffu) + q1 * bf(a >> 16));
        if (!lane) s[len] = sa;
    }
    __syncthreads();
    const int total = len + 1;
    float m = __int_as_float(0xff800000);  // -inf; NVRTC has no INFINITY macro
    for (int i = tid; i < total; i += 256) m = fmaxf(m, s[i]);
    m = warp_max_f(m);
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int i = tid; i < total; i += 256) { float p = __expf(s[i] - m); s[i] = p; l += p; }
    l = warp_sum_f(l);
    if (!lane) red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int i = 0; i < 8; ++i) l += red[i];
    // Output: thread = (partition p in 0..3, dim d in 0..63); partition p sums tokens t = p, p+4, ...
    const int d = tid & (HD - 1), p = tid >> 6;
    float acc0 = 0.0f, acc1 = 0.0f;
    int u = p;
    for (; u + 4 < len; u += 8) {
        acc0 = fmaf(s[u], bf(V[(size_t)u * HD + d]), acc0);
        acc1 = fmaf(s[u + 4], bf(V[(size_t)(u + 4) * HD + d]), acc1);
    }
    if (u < len) acc0 = fmaf(s[u], bf(V[(size_t)u * HD + d]), acc0);
    if (p == (len & 3)) acc0 = fmaf(s[len], bf(vn[d]), acc0);
    part[p][d] = acc0 + acc1;
    __syncthreads();
    if (tid < HD) {
        float o = (part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid]) / l;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);  // round to nearest even (finite values)
        out[head * HD + tid] = static_cast<unsigned short>(bits >> 16);
    }
}

// v2: no per-token shuffles. Each thread computes whole dot products for its tokens (8 x 16-byte loads per token,
// all independent), and the V pass is unrolled so many loads are in flight. Same contract as decode_attn_bf16.
struct __align__(16) U4 { unsigned a, b, c, d; };

__device__ __forceinline__ float dot8(const U4& k, const float* q) {
    return q[0] * bf(k.a & 0xffffu) + q[1] * bf(k.a >> 16) + q[2] * bf(k.b & 0xffffu) + q[3] * bf(k.b >> 16)
         + q[4] * bf(k.c & 0xffffu) + q[5] * bf(k.c >> 16) + q[6] * bf(k.d & 0xffffu) + q[7] * bf(k.d >> 16);
}

extern "C" __global__ __launch_bounds__(256) void decode_attn_v2_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, int len, int max_len, int heads, int kv_heads, float scale) {
    __shared__ float s[MAXT];
    __shared__ float qs[HD];
    __shared__ float red[8];
    __shared__ float part[8][HD];
    const int head = blockIdx.x, group = heads / kv_heads, kvh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    if (head % group == 0 && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    if (tid < HD) qs[tid] = bf(q[head * HD + tid]) * scale;
    __syncthreads();
    float qr[HD];
    #pragma unroll
    for (int i = 0; i < HD; ++i) qr[i] = qs[i];
    const int total = len + 1;
    #pragma unroll 2
    for (int t = tid; t < total; t += 256) {
        const U4* row = reinterpret_cast<const U4*>(t < len ? K + (size_t)t * HD : kn);
        U4 k0 = row[0], k1 = row[1], k2 = row[2], k3 = row[3], k4 = row[4], k5 = row[5], k6 = row[6], k7 = row[7];
        s[t] = dot8(k0, qr) + dot8(k1, qr + 8) + dot8(k2, qr + 16) + dot8(k3, qr + 24)
             + dot8(k4, qr + 32) + dot8(k5, qr + 40) + dot8(k6, qr + 48) + dot8(k7, qr + 56);
    }
    __syncthreads();
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < total; i += 256) m = fmaxf(m, s[i]);
    m = warp_max_f(m);
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int i = tid; i < total; i += 256) { float p = __expf(s[i] - m); s[i] = p; l += p; }
    l = warp_sum_f(l);
    if (!lane) red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int i = 0; i < 8; ++i) l += red[i];
    // V pass: lane owns dims 2*lane, 2*lane+1; warp w sums tokens w, w+8, ...
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 16
    for (int u = warp; u < len; u += 8) {
        unsigned v = reinterpret_cast<const unsigned*>(V + (size_t)u * HD)[lane];
        a0 = fmaf(s[u], bf(v & 0xffffu), a0);
        a1 = fmaf(s[u], bf(v >> 16), a1);
    }
    if (warp == (len & 7)) {
        unsigned v = reinterpret_cast<const unsigned*>(vn)[lane];
        a0 = fmaf(s[len], bf(v & 0xffffu), a0);
        a1 = fmaf(s[len], bf(v >> 16), a1);
    }
    part[warp][2 * lane] = a0;
    part[warp][2 * lane + 1] = a1;
    __syncthreads();
    if (tid < HD) {
        float o = 0.0f;
        for (int w = 0; w < 8; ++w) o += part[w][tid];
        o /= l;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        out[head * HD + tid] = static_cast<unsigned short>(bits >> 16);
    }
}

// v3 (exploratory): 512 threads, q read from shared memory (broadcast), and during the score pass every thread
// issues L2 prefetches for the V rows of its tokens so the V pass hits L2 instead of DRAM.
extern "C" __global__ __launch_bounds__(512) void decode_attn_v3_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, int len, int max_len, int heads, int kv_heads, float scale) {
    __shared__ float s[MAXT];
    __shared__ float qs[HD];
    __shared__ float red[16];
    __shared__ float part[16][HD];
    const int head = blockIdx.x, group = heads / kv_heads, kvh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    if (head % group == 0 && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    if (tid < HD) qs[tid] = bf(q[head * HD + tid]) * scale;
    __syncthreads();
    const int total = len + 1;
    for (int t = tid; t < total; t += 512) {
        if (t < len) {
            const char* vrow = reinterpret_cast<const char*>(V + (size_t)t * HD);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow + 64));
        }
        const U4* row = reinterpret_cast<const U4*>(t < len ? K + (size_t)t * HD : kn);
        U4 k0 = row[0], k1 = row[1], k2 = row[2], k3 = row[3], k4 = row[4], k5 = row[5], k6 = row[6], k7 = row[7];
        s[t] = dot8(k0, qs) + dot8(k1, qs + 8) + dot8(k2, qs + 16) + dot8(k3, qs + 24)
             + dot8(k4, qs + 32) + dot8(k5, qs + 40) + dot8(k6, qs + 48) + dot8(k7, qs + 56);
    }
    __syncthreads();
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < total; i += 512) m = fmaxf(m, s[i]);
    m = warp_max_f(m);
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 16; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int i = tid; i < total; i += 512) { float p = __expf(s[i] - m); s[i] = p; l += p; }
    l = warp_sum_f(l);
    if (!lane) red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int i = 0; i < 16; ++i) l += red[i];
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 8
    for (int u = warp; u < len; u += 16) {
        unsigned v = reinterpret_cast<const unsigned*>(V + (size_t)u * HD)[lane];
        a0 = fmaf(s[u], bf(v & 0xffffu), a0);
        a1 = fmaf(s[u], bf(v >> 16), a1);
    }
    if (warp == (len & 15)) {
        unsigned v = reinterpret_cast<const unsigned*>(vn)[lane];
        a0 = fmaf(s[len], bf(v & 0xffffu), a0);
        a1 = fmaf(s[len], bf(v >> 16), a1);
    }
    part[warp][2 * lane] = a0;
    part[warp][2 * lane + 1] = a1;
    __syncthreads();
    if (tid < HD) {
        float o = 0.0f;
        for (int w = 0; w < 16; ++w) o += part[w][tid];
        o /= l;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        out[head * HD + tid] = static_cast<unsigned short>(bits >> 16);
    }
}

// v4 (exploratory): split-KV inside one launch. grid.x = heads * splits. Block (head, c) handles tokens
// [c*chunk, min(len, (c+1)*chunk)); the last split also owns the new token. Each block writes (m, l, acc[64]) to
// `scratch`, then the last block of its head (atomic ticket) merges the partials and resets the ticket to zero so
// the kernel can be replayed in a CUDA graph. scratch: heads*splits*(HD+2) floats; tickets: heads ints (zeroed).
extern "C" __global__ __launch_bounds__(256) void decode_attn_v4_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, float* __restrict__ scratch, unsigned* __restrict__ tickets,
    int len, int max_len, int heads, int kv_heads, float scale, int splits) {
    __shared__ float s[MAXT];
    __shared__ float qs[HD];
    __shared__ float red[8];
    __shared__ float part[8][HD];
    __shared__ unsigned ticket;
    const int head = blockIdx.x / splits, c = blockIdx.x % splits;
    const int group = heads / kv_heads, kvh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    const int chunk = (len + splits - 1) / splits;
    const int lo = c * chunk, hi = min(len, lo + chunk);
    const bool owns_new = (c == splits - 1);
    if (owns_new && head % group == 0 && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    if (tid < HD) qs[tid] = bf(q[head * HD + tid]) * scale;
    __syncthreads();
    const int n = max(hi - lo, 0) + (owns_new ? 1 : 0);   // local token count; local index j -> token lo + j
    for (int j = tid; j < n; j += 256) {
        const int t = lo + j;
        if (t < hi) {
            const char* vrow = reinterpret_cast<const char*>(V + (size_t)t * HD);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow + 64));
        }
        const U4* row = reinterpret_cast<const U4*>(t < hi ? K + (size_t)t * HD : kn);
        U4 k0 = row[0], k1 = row[1], k2 = row[2], k3 = row[3], k4 = row[4], k5 = row[5], k6 = row[6], k7 = row[7];
        s[j] = dot8(k0, qs) + dot8(k1, qs + 8) + dot8(k2, qs + 16) + dot8(k3, qs + 24)
             + dot8(k4, qs + 32) + dot8(k5, qs + 40) + dot8(k6, qs + 48) + dot8(k7, qs + 56);
    }
    __syncthreads();
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < n; i += 256) m = fmaxf(m, s[i]);
    m = warp_max_f(m);
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int i = tid; i < n; i += 256) { float p = __expf(s[i] - m); s[i] = p; l += p; }
    l = warp_sum_f(l);
    if (!lane) red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int i = 0; i < 8; ++i) l += red[i];
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 8
    for (int j = warp; j < hi - lo; j += 8) {
        unsigned v = reinterpret_cast<const unsigned*>(V + (size_t)(lo + j) * HD)[lane];
        a0 = fmaf(s[j], bf(v & 0xffffu), a0);
        a1 = fmaf(s[j], bf(v >> 16), a1);
    }
    if (owns_new && warp == 0) {
        unsigned v = reinterpret_cast<const unsigned*>(vn)[lane];
        a0 = fmaf(s[n - 1], bf(v & 0xffffu), a0);
        a1 = fmaf(s[n - 1], bf(v >> 16), a1);
    }
    part[warp][2 * lane] = a0;
    part[warp][2 * lane + 1] = a1;
    __syncthreads();
    float* mine = scratch + (size_t)(head * splits + c) * (HD + 2);
    if (tid < HD) {
        float o = 0.0f;
        for (int w = 0; w < 8; ++w) o += part[w][tid];
        mine[2 + tid] = o;
    }
    if (tid == 0) { mine[0] = m; mine[1] = l; }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(tickets + head, 1u);
    __syncthreads();
    if (ticket != (unsigned)(splits - 1)) return;
    __threadfence();
    // Last block for this head: merge partials (empty splits have l = 0 and m = -inf).
    float gm = __int_as_float(0xff800000);
    for (int k = 0; k < splits; ++k) gm = fmaxf(gm, __ldcg(scratch + (size_t)(head * splits + k) * (HD + 2)));
    if (tid < HD) {
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < splits; ++k) {
            const float* p = scratch + (size_t)(head * splits + k) * (HD + 2);
            float lk = __ldcg(p + 1);
            if (lk == 0.0f) continue;
            float w = __expf(__ldcg(p) - gm);
            num = fmaf(w, __ldcg(p + 2 + tid), num);
            den = fmaf(w, lk, den);
        }
        float o = num / den;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        out[head * HD + tid] = static_cast<unsigned short>(bits >> 16);
    }
    if (tid == 0) tickets[head] = 0u;
}

// v5 (exploratory): GQA-packed split-KV. grid.x = kv_heads * splits, chunk = 64 tokens, group = heads/kv_heads <= 8.
// Each K/V row is loaded once per block and used for every query head of its group. The last block of a KV head
// (atomic ticket) merges the per-split partials of all heads in the group and resets the ticket.
// scratch: heads*splits*(HD+2) floats; tickets: kv_heads ints (zeroed).
#define CH 64
#define GMAX 8
extern "C" __global__ __launch_bounds__(256) void decode_attn_v5_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, float* __restrict__ scratch, unsigned* __restrict__ tickets,
    int len, int max_len, int heads, int kv_heads, float scale, int splits) {
    __shared__ float qs[GMAX][HD];
    __shared__ float s[GMAX][CH + 1];
    __shared__ float ml[GMAX][2];
    __shared__ float part[8][GMAX][HD];
    __shared__ unsigned ticket;
    const int kvh = blockIdx.x / splits, c = blockIdx.x % splits, group = heads / kv_heads, h0 = kvh * group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    const int lo = c * CH, hi = min(len, lo + CH);
    const bool owns_new = (c == splits - 1);
    const int nc = max(hi - lo, 0), n = nc + (owns_new ? 1 : 0);
    if (owns_new && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    for (int i = tid; i < group * HD; i += 256) qs[i / HD][i % HD] = bf(q[(h0 + i / HD) * HD + i % HD]) * scale;
    __syncthreads();
    for (int j = tid; j < n; j += 256) {
        const int t = lo + j;
        if (j < nc) {
            const char* vrow = reinterpret_cast<const char*>(V + (size_t)t * HD);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow + 64));
        }
        const U4* row = reinterpret_cast<const U4*>(j < nc ? K + (size_t)t * HD : kn);
        U4 k0 = row[0], k1 = row[1], k2 = row[2], k3 = row[3], k4 = row[4], k5 = row[5], k6 = row[6], k7 = row[7];
        for (int hh = 0; hh < group; ++hh) {
            const float* qq = qs[hh];
            s[hh][j] = dot8(k0, qq) + dot8(k1, qq + 8) + dot8(k2, qq + 16) + dot8(k3, qq + 24)
                     + dot8(k4, qq + 32) + dot8(k5, qq + 40) + dot8(k6, qq + 48) + dot8(k7, qq + 56);
        }
    }
    __syncthreads();
    if (warp < group) {  // one warp per query head: local softmax statistics over this chunk
        float m = __int_as_float(0xff800000);
        for (int j = lane; j < n; j += 32) m = fmaxf(m, s[warp][j]);
        m = warp_max_f(m);
        float l = 0.0f;
        for (int j = lane; j < n; j += 32) { float p = __expf(s[warp][j] - m); s[warp][j] = p; l += p; }
        l = warp_sum_f(l);
        if (!lane) { ml[warp][0] = m; ml[warp][1] = l; }
    }
    __syncthreads();
    float a[GMAX][2];
    #pragma unroll
    for (int hh = 0; hh < GMAX; ++hh) a[hh][0] = a[hh][1] = 0.0f;
    for (int j = warp; j < n; j += 8) {
        unsigned v = reinterpret_cast<const unsigned*>(j < nc ? V + (size_t)(lo + j) * HD : vn)[lane];
        const float v0 = bf(v & 0xffffu), v1 = bf(v >> 16);
        #pragma unroll
        for (int hh = 0; hh < GMAX; ++hh) {
            if (hh < group) { a[hh][0] = fmaf(s[hh][j], v0, a[hh][0]); a[hh][1] = fmaf(s[hh][j], v1, a[hh][1]); }
        }
    }
    #pragma unroll
    for (int hh = 0; hh < GMAX; ++hh) {
        if (hh < group) { part[warp][hh][2 * lane] = a[hh][0]; part[warp][hh][2 * lane + 1] = a[hh][1]; }
    }
    __syncthreads();
    for (int i = tid; i < group * HD; i += 256) {
        const int hh = i / HD, d = i % HD;
        float o = 0.0f;
        for (int w = 0; w < 8; ++w) o += part[w][hh][d];
        float* mine = scratch + (size_t)((h0 + hh) * splits + c) * (HD + 2);
        mine[2 + d] = o;
        if (d == 0) { mine[0] = ml[hh][0]; mine[1] = ml[hh][1]; }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(tickets + kvh, 1u);
    __syncthreads();
    if (ticket != (unsigned)(splits - 1)) return;
    __threadfence();
    for (int i = tid; i < group * HD; i += 256) {
        const int hh = i / HD, d = i % HD;
        const float* base = scratch + (size_t)(h0 + hh) * splits * (HD + 2);
        float gm = __int_as_float(0xff800000);
        for (int k = 0; k < splits; ++k) if (__ldcg(base + k * (HD + 2) + 1) > 0.0f) gm = fmaxf(gm, __ldcg(base + k * (HD + 2)));
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < splits; ++k) {
            const float* p = base + k * (HD + 2);
            float lk = __ldcg(p + 1);
            if (lk == 0.0f) continue;
            float w = __expf(__ldcg(p) - gm);
            num = fmaf(w, __ldcg(p + 2 + d), num);
            den = fmaf(w, lk, den);
        }
        float o = num / den;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        out[(h0 + hh) * HD + d] = static_cast<unsigned short>(bits >> 16);
    }
    if (tid == 0) tickets[kvh] = 0u;
}

// Fused elementwise kernels for decode (M rows, one block per row, 256 threads, d % 2 == 0).
// add_rmsnorm: x = res + y (rounded to BF16, stored to x_out), xn = x * rsqrt(mean(x^2) + eps) * w (BF16).
extern "C" __global__ __launch_bounds__(256) void add_rmsnorm_bf16(
    const unsigned short* __restrict__ res, const unsigned short* __restrict__ y, const unsigned short* __restrict__ w,
    unsigned short* __restrict__ x_out, unsigned short* __restrict__ xn_out, int d, float eps) {
    __shared__ float red[8];
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned* r2 = reinterpret_cast<const unsigned*>(res + (size_t)row * d);
    const unsigned* y2 = reinterpret_cast<const unsigned*>(y + (size_t)row * d);
    unsigned* x2 = reinterpret_cast<unsigned*>(x_out + (size_t)row * d);
    float vals[8];
    int cnt = 0;
    float ss = 0.0f;
    for (int i = tid; i < d / 2; i += 256, cnt += 2) {
        unsigned a = r2[i], b = y2[i];
        float x0 = bf(a & 0xffffu) + bf(b & 0xffffu), x1 = bf(a >> 16) + bf(b >> 16);
        unsigned u0 = __float_as_uint(x0), u1 = __float_as_uint(x1);
        u0 = (u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16; u1 = (u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16;
        x2[i] = u0 | (u1 << 16);
        x0 = bf(u0); x1 = bf(u1);
        vals[cnt] = x0; vals[cnt + 1] = x1;
        ss = fmaf(x0, x0, fmaf(x1, x1, ss));
    }
    ss = warp_sum_f(ss);
    if (!lane) red[warp] = ss;
    __syncthreads();
    ss = 0.0f;
    for (int i = 0; i < 8; ++i) ss += red[i];
    const float inv = rsqrtf(ss / d + eps);
    const unsigned* w2 = reinterpret_cast<const unsigned*>(w);
    unsigned* o2 = reinterpret_cast<unsigned*>(xn_out + (size_t)row * d);
    cnt = 0;
    for (int i = tid; i < d / 2; i += 256, cnt += 2) {
        unsigned ww = w2[i];
        float o0 = vals[cnt] * inv * bf(ww & 0xffffu), o1 = vals[cnt + 1] * inv * bf(ww >> 16);
        unsigned u0 = __float_as_uint(o0), u1 = __float_as_uint(o1);
        u0 = (u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16; u1 = (u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16;
        o2[i] = u0 | (u1 << 16);
    }
}

// silu_mul: gu = [gate | up] (2*inter per row) -> out = silu(gate) * up, FP32 math, one BF16 rounding.
extern "C" __global__ void silu_mul_bf16(const unsigned short* __restrict__ gu, unsigned short* __restrict__ out,
                                         int inter, int total_pairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // pair index over rows*inter/2
    if (i >= total_pairs) return;
    const int per_row = inter / 2, row = i / per_row, j = i % per_row;
    const unsigned g = reinterpret_cast<const unsigned*>(gu + (size_t)row * 2 * inter)[j];
    const unsigned u = reinterpret_cast<const unsigned*>(gu + (size_t)row * 2 * inter + inter)[j];
    float g0 = bf(g & 0xffffu), g1 = bf(g >> 16);
    float o0 = g0 / (1.0f + __expf(-g0)) * bf(u & 0xffffu), o1 = g1 / (1.0f + __expf(-g1)) * bf(u >> 16);
    unsigned u0 = __float_as_uint(o0), u1 = __float_as_uint(o1);
    u0 = (u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16; u1 = (u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16;
    reinterpret_cast<unsigned*>(out)[i] = u0 | (u1 << 16);  // out is [rows, inter], pair i = row*per_row + j
}

// v4d: v4 with the cache length read from device memory (len_ptr). Splits are sized by capacity (max_len), so the
// launch shape is fixed and the same CUDA graph can be replayed while len grows; empty splits contribute l = 0.
extern "C" __global__ __launch_bounds__(256) void decode_attn_v4d_bf16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    unsigned short* __restrict__ out, float* __restrict__ scratch, unsigned* __restrict__ tickets,
    const int* __restrict__ len_ptr, int max_len, int heads, int kv_heads, float scale, int splits) {
    const int len = *len_ptr;  // current cache length, read on device so one captured graph serves a growing sequence
    __shared__ float s[MAXT];
    __shared__ float qs[HD];
    __shared__ float red[8];
    __shared__ float part[8][HD];
    __shared__ unsigned ticket;
    const int head = blockIdx.x / splits, c = blockIdx.x % splits;
    const int group = heads / kv_heads, kvh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned short* kn = k_new + kvh * HD;
    const unsigned short* vn = v_new + kvh * HD;
    unsigned short* K = kc + (size_t)kvh * max_len * HD;
    unsigned short* V = vc + (size_t)kvh * max_len * HD;
    const int chunk = (max_len + splits - 1) / splits;  // split by capacity; splits past len are empty (l = 0)
    const int lo = c * chunk, hi = min(len, lo + chunk);
    const bool owns_new = (c == splits - 1);
    if (owns_new && head % group == 0 && tid < 2 * HD) {
        if (tid < HD) K[(size_t)len * HD + tid] = kn[tid];
        else V[(size_t)len * HD + tid - HD] = vn[tid - HD];
    }
    if (tid < HD) qs[tid] = bf(q[head * HD + tid]) * scale;
    __syncthreads();
    const int n = max(hi - lo, 0) + (owns_new ? 1 : 0);   // local token count; local index j -> token lo + j
    for (int j = tid; j < n; j += 256) {
        const int t = lo + j;
        if (t < hi) {
            const char* vrow = reinterpret_cast<const char*>(V + (size_t)t * HD);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(vrow + 64));
        }
        const U4* row = reinterpret_cast<const U4*>(t < hi ? K + (size_t)t * HD : kn);
        U4 k0 = row[0], k1 = row[1], k2 = row[2], k3 = row[3], k4 = row[4], k5 = row[5], k6 = row[6], k7 = row[7];
        s[j] = dot8(k0, qs) + dot8(k1, qs + 8) + dot8(k2, qs + 16) + dot8(k3, qs + 24)
             + dot8(k4, qs + 32) + dot8(k5, qs + 40) + dot8(k6, qs + 48) + dot8(k7, qs + 56);
    }
    __syncthreads();
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < n; i += 256) m = fmaxf(m, s[i]);
    m = warp_max_f(m);
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int i = tid; i < n; i += 256) { float p = __expf(s[i] - m); s[i] = p; l += p; }
    l = warp_sum_f(l);
    if (!lane) red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int i = 0; i < 8; ++i) l += red[i];
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 8
    for (int j = warp; j < hi - lo; j += 8) {
        unsigned v = reinterpret_cast<const unsigned*>(V + (size_t)(lo + j) * HD)[lane];
        a0 = fmaf(s[j], bf(v & 0xffffu), a0);
        a1 = fmaf(s[j], bf(v >> 16), a1);
    }
    if (owns_new && warp == 0) {
        unsigned v = reinterpret_cast<const unsigned*>(vn)[lane];
        a0 = fmaf(s[n - 1], bf(v & 0xffffu), a0);
        a1 = fmaf(s[n - 1], bf(v >> 16), a1);
    }
    part[warp][2 * lane] = a0;
    part[warp][2 * lane + 1] = a1;
    __syncthreads();
    float* mine = scratch + (size_t)(head * splits + c) * (HD + 2);
    if (tid < HD) {
        float o = 0.0f;
        for (int w = 0; w < 8; ++w) o += part[w][tid];
        mine[2 + tid] = o;
    }
    if (tid == 0) { mine[0] = m; mine[1] = l; }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(tickets + head, 1u);
    __syncthreads();
    if (ticket != (unsigned)(splits - 1)) return;
    __threadfence();
    // Last block for this head: merge partials (empty splits have l = 0 and m = -inf).
    float gm = __int_as_float(0xff800000);
    for (int k = 0; k < splits; ++k) gm = fmaxf(gm, __ldcg(scratch + (size_t)(head * splits + k) * (HD + 2)));
    if (tid < HD) {
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < splits; ++k) {
            const float* p = scratch + (size_t)(head * splits + k) * (HD + 2);
            float lk = __ldcg(p + 1);
            if (lk == 0.0f) continue;
            float w = __expf(__ldcg(p) - gm);
            num = fmaf(w, __ldcg(p + 2 + tid), num);
            den = fmaf(w, lk, den);
        }
        float o = num / den;
        unsigned bits = __float_as_uint(o);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        out[head * HD + tid] = static_cast<unsigned short>(bits >> 16);
    }
    if (tid == 0) tickets[head] = 0u;
}

