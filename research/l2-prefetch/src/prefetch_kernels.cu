// Spike: pull a tensor's bytes into L2 from a side stream while another op runs.
// Never writes the tensor; only a dead-branch store keeps loads observable.
extern "C" __global__ void l2_hint(const char* p, long long bytes, int stride, unsigned* sink) {
    long long step = (long long)gridDim.x * blockDim.x * stride;
    for (long long off = (long long)(blockIdx.x * blockDim.x + threadIdx.x) * stride; off < bytes; off += step)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + off));
}

extern "C" __global__ void l2_load(const char* p, long long bytes, int stride, unsigned* sink) {
    long long step = (long long)gridDim.x * blockDim.x * stride;
    unsigned acc = 0;
    for (long long off = (long long)(blockIdx.x * blockDim.x + threadIdx.x) * stride; off < bytes; off += step) {
        unsigned v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + off));
        acc ^= v;
    }
    if (acc == 0x9e3779b9u && bytes < 0) sink[0] = acc;  // Never true; prevents dead-code removal.
}

// Same as l2_hint, but the prefetched lines are marked evict_last (SM80+).
extern "C" __global__ void l2_hint_last(const char* p, long long bytes, int stride, unsigned* sink) {
    long long step = (long long)gridDim.x * blockDim.x * stride;
    for (long long off = (long long)(blockIdx.x * blockDim.x + threadIdx.x) * stride; off < bytes; off += step)
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + off));
}
