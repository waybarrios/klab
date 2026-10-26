// Latency floor probes for graph-replayed single launches. Throwaway, not a measured kernel family.
extern "C" __global__ void floor_empty(const unsigned* w, float* out) {}

// One thread, one dependent DRAM load, one store: launch + single round trip.
extern "C" __global__ void floor_touch(const unsigned* w, float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = __uint_as_float(w[0]);
}

// N warps, each reads one 16-byte chunk per lane of its row's first 512 bytes, then reduces:
// every row is touched once, so this is the floor for a one-pass row kernel with no K loop.
extern "C" __global__ void floor_rows(const unsigned* w, float* out, int n, int k_words) {
    int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= n) return;
    uint4 v = (lane * 4 < k_words) ? reinterpret_cast<const uint4*>(w + (size_t)row * k_words)[lane] : make_uint4(0, 0, 0, 0);
    float s = __uint_as_float(v.x ^ v.y ^ v.z ^ v.w);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (!lane) out[row] = s;
}
