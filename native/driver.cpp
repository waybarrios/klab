// C ABI over the CUDA Driver API. Callers own device-buffer lifetimes,
// current contexts, stream ordering, and synchronization before unloading.
#include <dlfcn.h>
#include <mutex>

namespace {
struct Driver {
    using Init = int (*)(unsigned);
    using Context = int (*)(void**);
    using Load = int (*)(void**, const void*);
    using Function = int (*)(void**, void*, const char*);
    using Unload = int (*)(void*);
    using Launch = int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                          unsigned, unsigned, void*, void**, void**);
    using Error = int (*)(int, const char**);
    Init init = nullptr;
    Context context = nullptr;
    Load load = nullptr;
    Function function = nullptr;
    Unload unload = nullptr;
    Launch launch = nullptr;
    Error error = nullptr;
    bool ready = false;
};

Driver& driver() {
    static Driver d;
    static std::once_flag once;
    std::call_once(once, [] {
        // Intentionally retain the driver handle for the wrapper's lifetime.
        void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle) return;
        d.init = reinterpret_cast<Driver::Init>(dlsym(handle, "cuInit"));
        d.context = reinterpret_cast<Driver::Context>(dlsym(handle, "cuCtxGetCurrent"));
        d.load = reinterpret_cast<Driver::Load>(dlsym(handle, "cuModuleLoadData"));
        d.function = reinterpret_cast<Driver::Function>(dlsym(handle, "cuModuleGetFunction"));
        d.unload = reinterpret_cast<Driver::Unload>(dlsym(handle, "cuModuleUnload"));
        d.launch = reinterpret_cast<Driver::Launch>(dlsym(handle, "cuLaunchKernel"));
        d.error = reinterpret_cast<Driver::Error>(dlsym(handle, "cuGetErrorString"));
        d.ready = d.init && d.context && d.load && d.function && d.unload && d.launch && d.error;
    });
    return d;
}
}

extern "C" int lab_load_ptx(const char* ptx, void** out) {
    if (!ptx || !out) return -1002;
    auto& d = driver();
    if (!d.ready) return -1000;
    int status = d.init(0);
    if (status) return status;
    void* context = nullptr;
    status = d.context(&context);
    if (status) return status;
    if (!context) return -1001;
    return d.load(out, ptx);
}

extern "C" int lab_get_kernel(void* module, const char* name, void** out) {
    if (!module || !name || !out) return -1002;
    auto& d = driver();
    return d.ready ? d.function(out, module, name) : -1000;
}

extern "C" int lab_launch(void* kernel, unsigned grid_x, unsigned grid_y,
                           unsigned block_x, void* stream, void** args) {
    if (!kernel || !args || !grid_x || !grid_y || !block_x) return -1002;
    auto& d = driver();
    return d.ready ? d.launch(kernel, grid_x, grid_y, 1, block_x, 1, 1, 0,
                              stream, args, nullptr) : -1000;
}

extern "C" int lab_unload(void* module) {
    if (!module) return -1002;
    auto& d = driver();
    return d.ready ? d.unload(module) : -1000;
}

extern "C" const char* lab_error_string(int code) {
    switch (code) {
    case -1000: return "Cannot load CUDA driver or required symbols";
    case -1001: return "No current CUDA context; initialize PyTorch CUDA first";
    case -1002: return "Invalid null argument or launch dimensions";
    default:
        auto& d = driver();
        const char* message = nullptr;
        if (d.ready && d.error(code, &message) == 0 && message) return message;
        return "Unknown CUDA error";
    }
}
