// klab_native_v1: C++ Torch dispatch for the frozen BF16 raw/coded projection kernels.
//
// The registry owns launch metadata plus Driver API modules loaded from the
// exact NVRTC PTX bytes of compressed_bf16 (R1, M == 1) and mtile_bf16 (R2).
// It never owns weight tensors and never exposes CUfunction pointers: callers
// pass every tensor explicitly and refer to plans by opaque monotonic IDs.
// Modules are deduplicated by (device, PTX SHA-256) and retained for the
// process lifetime, so retiring a plan never unloads code a graph captured.
//
// Forward validates the plan scalars and tensor contracts, allocates with ATen
// on the current stream and enqueues one cuLaunchKernel. No synchronization,
// readback, arithmetic, Python or context switching happens here.
#include <cuda.h>
#include <dlfcn.h>

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Mirrors of the frozen wrapper limits (compressed_bf16 / mtile_bf16).
constexpr int64_t kBlock = 64;
constexpr int64_t kCodedBytes = 96;
constexpr int64_t kRawBytes = 128;
constexpr int64_t kPayloadAlignment = 32;
constexpr int64_t kMetadataAlignment = 4;
constexpr int64_t kMaxM = 65535;
constexpr int64_t kMaxTile = 16;
constexpr int64_t kIndexLimit = int64_t{1} << 31;

// ---------------------------------------------------------------- driver

struct Driver {
    decltype(&cuInit) init = nullptr;
    decltype(&cuDeviceGet) device_get = nullptr;
    decltype(&cuCtxGetCurrent) context_current = nullptr;
    decltype(&cuCtxGetDevice) context_device = nullptr;
    decltype(&cuModuleLoadData) module_load = nullptr;
    decltype(&cuModuleGetFunction) module_function = nullptr;
    decltype(&cuFuncGetAttribute) function_attribute = nullptr;
    decltype(&cuLaunchKernel) launch = nullptr;
    decltype(&cuStreamIsCapturing) is_capturing = nullptr;
    decltype(&cuGetErrorName) error_name = nullptr;
    decltype(&cuGetErrorString) error_string = nullptr;
    bool ready = false;
};

template <typename F>
void load_symbol(void* handle, F& function, const char* name) {
    function = reinterpret_cast<F>(dlsym(handle, name));
}

const Driver& driver() {
    static const Driver loaded = [] {
        Driver d;
        // Intentionally retained for the process lifetime, like native/driver.cpp.
        void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle) return d;
        load_symbol(handle, d.init, "cuInit");
        load_symbol(handle, d.device_get, "cuDeviceGet");
        load_symbol(handle, d.context_current, "cuCtxGetCurrent");
        load_symbol(handle, d.context_device, "cuCtxGetDevice");
        load_symbol(handle, d.module_load, "cuModuleLoadData");
        load_symbol(handle, d.module_function, "cuModuleGetFunction");
        load_symbol(handle, d.function_attribute, "cuFuncGetAttribute");
        load_symbol(handle, d.launch, "cuLaunchKernel");
        load_symbol(handle, d.is_capturing, "cuStreamIsCapturing");
        load_symbol(handle, d.error_name, "cuGetErrorName");
        load_symbol(handle, d.error_string, "cuGetErrorString");
        d.ready = d.init && d.device_get && d.context_current && d.context_device && d.module_load &&
                  d.module_function && d.function_attribute && d.launch && d.is_capturing &&
                  d.error_name && d.error_string;
        return d;
    }();
    return loaded;
}

void check(CUresult status, const char* call) {
    if (status == CUDA_SUCCESS) return;
    const Driver& d = driver();
    const char* name = nullptr;
    const char* text = nullptr;
    if (d.error_name) d.error_name(status, &name);
    if (d.error_string) d.error_string(status, &text);
    TORCH_CHECK(false, call, " failed with CUDA error ", static_cast<int>(status), " (",
                name ? name : "unknown", "): ", text ? text : "no description");
}

// ---------------------------------------------------------------- registry

struct Plan {
    int64_t id = 0;
    bool coded = false;
    std::string kind, output;
    at::ScalarType output_dtype = at::kBFloat16;
    int64_t device = 0, m = 0, n = 0, k = 0, threads = 0, vector = 0, tile = 1;
    unsigned grid_x = 0, grid_y = 0;
    std::string kernel, ptx_sha256;
    bool module_reused = false;
    CUcontext context = nullptr;
    CUfunction function = nullptr;  // Private: never leaves this file.
    std::vector<std::pair<std::string, int>> attributes;
};

struct Module {
    std::string ptx;  // Exact registered bytes, compared on every same-digest registration.
    CUcontext context = nullptr;
    CUmodule module = nullptr;
};

struct Registry {
    std::mutex modules_mutex;  // Serializes PTX loads; never held by forward.
    std::map<std::pair<int64_t, std::string>, Module> modules;  // Never erased or unloaded.
    std::shared_mutex plans_mutex;
    std::unordered_map<int64_t, std::shared_ptr<const Plan>> plans;
    int64_t next_id = 1;
};

Registry& registry() {
    // Leaked on purpose: modules (and captured graphs using them) outlive static destruction.
    static Registry* instance = new Registry();
    return *instance;
}

std::string text(c10::string_view value) {
    return std::string(value.data(), value.size());
}

bool is_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

// Launch metadata from the frozen wrapper formulas; validates every limit itself.
Plan make_plan(int64_t m, int64_t n, int64_t k, int64_t threads, int64_t vector,
               const std::string& kind, const std::string& output) {
    TORCH_CHECK_VALUE(kind == "raw" || kind == "coded", "kind must be 'raw' or 'coded'");
    TORCH_CHECK_VALUE(output == "bf16" || output == "fp32", "output_dtype must be 'bf16' or 'fp32'");
    TORCH_CHECK_VALUE((threads == 32 || threads == 64 || threads == 128 || threads == 256) &&
                          (vector == 2 || vector == 8),
                      "Unsupported launch configuration; threads in (32, 64, 128, 256), vector in (2, 8)");
    // Bounds on N and K first, so the products below cannot overflow int64.
    TORCH_CHECK_VALUE(m >= 1 && m <= kMaxM && n >= 1 && n < kIndexLimit && k >= 1 && k < kIndexLimit &&
                          m * k < kIndexLimit && n * k < kIndexLimit && m * n < kIndexLimit,
                      "Dimensions must be positive, M<=", kMaxM, ", and fit signed 32-bit indexing");
    TORCH_CHECK_VALUE(k % vector == 0, "Require K divisible by ", vector);
    Plan plan;
    plan.coded = kind == "coded";
    TORCH_CHECK_VALUE(!plan.coded || k % kBlock == 0, "Coded projections require K divisible by ", kBlock);
    plan.kind = kind;
    plan.output = output;
    plan.output_dtype = output == "fp32" ? at::kFloat : at::kBFloat16;
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.threads = threads;
    plan.vector = vector;
    // mtile_bf16.tile_for_m: 1 for M == 1, else min(16, next power of two >= M).
    while (plan.tile < m && plan.tile < kMaxTile) plan.tile *= 2;
    const int64_t rows = threads / 32;
    plan.grid_x = static_cast<unsigned>((n + rows - 1) / rows);
    const std::string base = kind + "_" + output + "_t" + std::to_string(threads) + "_v" + std::to_string(vector);
    if (m == 1) {
        // compressed_bf16 R1: grid (ceil(N/rows), M); args (x, w | payload, metadata, out, n, k).
        plan.grid_y = static_cast<unsigned>(m);
        plan.kernel = "cbf16_" + base;
    } else {
        // mtile_bf16 R2: grid (ceil(N/rows), ceil(M/tile)); the same args plus m.
        plan.grid_y = static_cast<unsigned>((m + plan.tile - 1) / plan.tile);
        plan.kernel = "mtbf16_" + base + "_m" + std::to_string(plan.tile);
    }
    return plan;
}

const std::pair<const char*, CUfunction_attribute> kAttributes[] = {
    {"num_regs", CU_FUNC_ATTRIBUTE_NUM_REGS},
    {"local_size_bytes", CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES},
    {"shared_size_bytes", CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES},
    {"const_size_bytes", CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES},
    {"max_threads_per_block", CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
    {"ptx_version", CU_FUNC_ATTRIBUTE_PTX_VERSION},
    {"binary_version", CU_FUNC_ATTRIBUTE_BINARY_VERSION},
};

int64_t register_plan(const at::Tensor& ptx_cpu_u8, int64_t device, int64_t m, int64_t n, int64_t k,
                      int64_t threads, int64_t vector, c10::string_view kind, c10::string_view output_dtype,
                      c10::string_view ptx_sha256) {
    const at::Tensor& ptx = ptx_cpu_u8;
    TORCH_CHECK_VALUE(ptx.defined() && ptx.layout() == at::kStrided && ptx.device().is_cpu() &&
                          ptx.scalar_type() == at::kByte && ptx.dim() == 1 && ptx.is_contiguous() &&
                          ptx.numel() > 0,
                      "ptx_cpu_u8 must be a nonempty contiguous 1-D CPU uint8 tensor");
    const std::string digest = text(ptx_sha256);
    TORCH_CHECK_VALUE(is_sha256(digest), "ptx_sha256 must be a lowercase SHA-256 hex digest");
    TORCH_CHECK_VALUE(device >= 0 && device < c10::cuda::device_count(), "CUDA device ", device,
                      " is not available");
    Plan plan = make_plan(m, n, k, threads, vector, text(kind), text(output_dtype));
    plan.device = device;
    plan.ptx_sha256 = digest;

    // std::string keeps the exact bytes plus a terminator; cuModuleLoadData reads
    // PTX as a C string, so an embedded NUL would silently load only a prefix.
    std::string bytes(reinterpret_cast<const char*>(ptx.const_data_ptr<uint8_t>()),
                      static_cast<size_t>(ptx.numel()));
    TORCH_CHECK_VALUE(bytes.find('\0') == std::string::npos, "PTX bytes must not contain NUL characters");

    const Driver& d = driver();
    TORCH_CHECK(d.ready, "Cannot load libcuda.so.1 or its required Driver API symbols");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    check(d.init(0), "cuInit");
    check(d.context_current(&plan.context), "cuCtxGetCurrent");
    TORCH_CHECK(plan.context, "No current CUDA context on device ", device,
                "; initialize Torch CUDA on this thread before registering (no context is created here)");
    CUdevice expected = 0, actual = 0;
    check(d.device_get(&expected, static_cast<int>(device)), "cuDeviceGet");
    check(d.context_device(&actual), "cuCtxGetDevice");
    TORCH_CHECK(actual == expected, "The current CUDA context does not belong to device ", device);

    Registry& r = registry();
    CUmodule module = nullptr;
    {
        std::lock_guard<std::mutex> lock(r.modules_mutex);
        const auto key = std::make_pair(device, digest);
        const auto found = r.modules.find(key);
        if (found != r.modules.end()) {
            TORCH_CHECK_VALUE(found->second.ptx == bytes, "PTX SHA-256 ", digest,
                              " was already registered on device ", device, " with different bytes");
            TORCH_CHECK(found->second.context == plan.context, "Device ", device,
                        " has a different CUDA context than when this PTX was loaded; restart Python");
            module = found->second.module;
            plan.module_reused = true;
        } else {
            check(d.module_load(&module, bytes.c_str()), "cuModuleLoadData");
            r.modules.emplace(key, Module{std::move(bytes), plan.context, module});
        }
    }
    check(d.module_function(&plan.function, module, plan.kernel.c_str()), "cuModuleGetFunction");
    for (const auto& [name, attribute] : kAttributes) {
        int value = 0;
        check(d.function_attribute(&value, attribute, plan.function), "cuFuncGetAttribute");
        plan.attributes.emplace_back(name, value);
        if (attribute == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) {
            TORCH_CHECK_VALUE(plan.threads <= value, plan.kernel, " allows at most ", value, " threads per block");
        }
    }

    std::unique_lock<std::shared_mutex> lock(r.plans_mutex);
    plan.id = r.next_id++;
    const int64_t id = plan.id;
    r.plans.emplace(id, std::make_shared<const Plan>(std::move(plan)));
    return id;
}

std::shared_ptr<const Plan> find_plan(int64_t id) {
    TORCH_CHECK_VALUE(id > 0, "plan must be a positive registered plan ID");
    Registry& r = registry();
    std::shared_lock<std::shared_mutex> lock(r.plans_mutex);
    const auto found = r.plans.find(id);
    TORCH_CHECK_VALUE(found != r.plans.end(), "Unknown or retired native plan ", id);
    return found->second;
}

void retire_plan(int64_t plan) {
    TORCH_CHECK_VALUE(plan > 0, "plan must be a positive registered plan ID");
    Registry& r = registry();
    std::unique_lock<std::shared_mutex> lock(r.plans_mutex);
    TORCH_CHECK_VALUE(r.plans.erase(plan) == 1, "Unknown or retired native plan ", plan);
}

// JSON-ready static metadata; no addresses and no GPU work.
std::string plan_info(int64_t plan) {
    const auto p = find_plan(plan);
    std::ostringstream json;
    json << std::boolalpha << "{\"plan\":" << p->id << ",\"kind\":\"" << p->kind << "\",\"output_dtype\":\""
         << p->output << "\",\"device\":" << p->device << ",\"shape_mnk\":[" << p->m << "," << p->n << ","
         << p->k << "],\"threads\":" << p->threads << ",\"vector\":" << p->vector << ",\"tile\":" << p->tile
         << ",\"route\":\"" << (p->m == 1 ? "R1" : "R2") << "\",\"kernel\":\"" << p->kernel << "\",\"grid\":["
         << p->grid_x << "," << p->grid_y << "],\"block\":[" << p->threads
         << ",1,1],\"dynamic_shared_bytes\":0,\"ptx_sha256\":\"" << p->ptx_sha256
         << "\",\"module_reused\":" << p->module_reused << ",\"function_attributes\":{";
    for (size_t i = 0; i < p->attributes.size(); ++i) {
        json << (i ? "," : "") << "\"" << p->attributes[i].first << "\":" << p->attributes[i].second;
    }
    json << "}}";
    return json.str();
}

// ---------------------------------------------------------------- forward

std::shared_ptr<const Plan> checked_plan(int64_t id, bool coded, int64_t m, int64_t n, int64_t k,
                                         int64_t threads, int64_t vector, c10::string_view output) {
    auto plan = find_plan(id);
    TORCH_CHECK_VALUE(plan->coded == coded, "Plan ", id, " is a ", plan->kind, " plan");
    TORCH_CHECK_VALUE(plan->m == m && plan->n == n && plan->k == k && plan->threads == threads &&
                          plan->vector == vector && plan->output == output,
                      "Arguments do not match plan ", id, ": expected M,N,K=", plan->m, ",", plan->n, ",",
                      plan->k, " threads=", plan->threads, " vector=", plan->vector, " output_dtype=",
                      plan->output);
    return plan;
}

bool aligned(const at::Tensor& tensor, int64_t bytes) {
    return reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr()) % static_cast<std::uintptr_t>(bytes) == 0;
}

void check_operand(const at::Tensor& tensor, const char* label, at::ScalarType dtype, int64_t dim,
                   const at::Device& device) {
    TORCH_CHECK_VALUE(tensor.defined() && tensor.layout() == at::kStrided && tensor.device() == device &&
                          tensor.scalar_type() == dtype && tensor.dim() == dim,
                      label, " must be a ", dim, "-D strided ", dtype, " tensor on ", device);
    TORCH_CHECK_VALUE(tensor.is_contiguous() && !tensor.requires_grad(), label,
                      " must be contiguous and inference-only (no requires_grad)");
    TORCH_CHECK_VALUE(!tensor.is_neg() && !tensor.is_conj(), label,
                      " must not be a lazily negated or conjugated view");
}

void check_x(const Plan& plan, const at::Tensor& x) {
    TORCH_CHECK_VALUE(x.defined() && x.is_cuda() && x.get_device() == plan.device, "x must be a CUDA tensor on ",
                      "device ", plan.device, ", where plan ", plan.id, " was registered");
    check_operand(x, "x", at::kBFloat16, 2, x.device());
    TORCH_CHECK_VALUE(x.size(0) == plan.m && x.size(1) == plan.k, "x must have the prepared shape [M,K]=[",
                      plan.m, ",", plan.k, "]");
    TORCH_CHECK_VALUE(aligned(x, 2 * plan.vector), "Require ", 2 * plan.vector, "-byte aligned inputs");
}

// Enqueue on the current stream in the wrappers' argument order:
// inputs (x first), out, n, k, plus m for the R2 M-tile kernels.
at::Tensor launch(const Plan& plan, std::initializer_list<const at::Tensor*> inputs) {
    const Driver& d = driver();
    const at::Tensor& x = **inputs.begin();
    c10::cuda::CUDAGuard guard(x.device());
    CUcontext context = nullptr;
    check(d.context_current(&context), "cuCtxGetCurrent");
    TORCH_CHECK(context == plan.context, "The current CUDA context is not the one plan ", plan.id,
                " was registered in; initialize Torch CUDA on this thread (no context switch is attempted)");

    at::Tensor out = at::empty({plan.m, plan.n}, x.options().dtype(plan.output_dtype));
    const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    std::array<void*, 4> pointers{};
    std::array<void*, 7> args{};
    size_t used = 0, count = 0;
    for (const at::Tensor* input : inputs) {
        pointers[used] = const_cast<void*>(input->const_data_ptr());
        args[count++] = &pointers[used++];
    }
    pointers[used] = out.mutable_data_ptr();
    args[count++] = &pointers[used++];
    int dims[3] = {static_cast<int>(plan.n), static_cast<int>(plan.k), static_cast<int>(plan.m)};
    args[count++] = &dims[0];
    args[count++] = &dims[1];
    if (plan.m > 1) args[count++] = &dims[2];
    check(d.launch(plan.function, plan.grid_x, plan.grid_y, 1, static_cast<unsigned>(plan.threads), 1, 1, 0,
                   stream.stream(), args.data(), nullptr),
          "cuLaunchKernel");

    // Raw pointers bypass the caching allocator's stream bookkeeping (as in
    // CudaModule.launch). During capture the caller keeps static inputs alive.
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
    check(d.is_capturing(stream.stream(), &capture), "cuStreamIsCapturing");
    if (capture == CU_STREAM_CAPTURE_STATUS_NONE) {
        for (const at::Tensor* input : inputs) {
            c10::cuda::CUDACachingAllocator::recordStream(input->storage().data_ptr(), stream);
        }
    }
    return out;
}

at::Tensor raw(const at::Tensor& x, const at::Tensor& w, int64_t plan, int64_t m, int64_t n, int64_t k,
               int64_t threads, int64_t vector, c10::string_view output_dtype) {
    const auto p = checked_plan(plan, false, m, n, k, threads, vector, output_dtype);
    check_x(*p, x);
    check_operand(w, "w", at::kBFloat16, 2, x.device());
    TORCH_CHECK_VALUE(w.size(0) == n && w.size(1) == k, "w must have the prepared shape [N,K]=[", n, ",", k,
                      "]");
    TORCH_CHECK_VALUE(aligned(w, 2 * vector), "Require ", 2 * vector, "-byte aligned inputs");
    return launch(*p, {&x, &w});
}

at::Tensor coded(const at::Tensor& x, const at::Tensor& payload, const at::Tensor& metadata, int64_t plan,
                 int64_t m, int64_t n, int64_t k, int64_t threads, int64_t vector,
                 c10::string_view output_dtype) {
    const auto p = checked_plan(plan, true, m, n, k, threads, vector, output_dtype);
    check_x(*p, x);
    check_operand(payload, "payload", at::kByte, 1, x.device());
    check_operand(metadata, "metadata", at::kInt, 1, x.device());
    const int64_t blocks = n * (k / kBlock);
    TORCH_CHECK_VALUE(metadata.numel() == blocks && payload.numel() >= kCodedBytes * blocks &&
                          payload.numel() <= kRawBytes * blocks,
                      "Packed buffers do not match N,K");
    TORCH_CHECK_VALUE(aligned(payload, kPayloadAlignment) && aligned(metadata, kMetadataAlignment),
                      "Packed buffers must keep their 32-byte payload alignment");
    return launch(*p, {&x, &payload, &metadata});
}

}  // namespace

TORCH_LIBRARY(klab_native_v1, library) {
    // The Fake kernels live in native_prepared.py; Torch requires this pairing.
    library.set_python_module("native_prepared");
    library.def(
        "raw(Tensor x, Tensor w, int plan, int m, int n, int k, int threads, int vector, "
        "str output_dtype) -> Tensor");
    library.def(
        "coded(Tensor x, Tensor payload, Tensor metadata, int plan, int m, int n, int k, int threads, "
        "int vector, str output_dtype) -> Tensor");
    library.def(
        "_register_plan(Tensor ptx_cpu_u8, int device, int m, int n, int k, int threads, int vector, "
        "str kind, str output_dtype, str ptx_sha256) -> int");
    library.def("_retire_plan(int plan) -> ()", &retire_plan);
    library.def("plan_info(int plan) -> str", &plan_info);
}

TORCH_LIBRARY_IMPL(klab_native_v1, CPU, library) {
    library.impl("_register_plan", &register_plan);
}

TORCH_LIBRARY_IMPL(klab_native_v1, CUDA, library) {
    library.impl("raw", &raw);
    library.impl("coded", &coded);
}
