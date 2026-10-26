"""Python -> C++ C ABI -> CUDA Driver API, sharing PyTorch buffers/streams.

Experimental inference-only operations. Device sources are CUDA C++.
Compilation and module loading happen at construction, outside timing/capture.
"""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key, directory in {
    "TMPDIR": ".scratch",
    "CUDA_CACHE_PATH": ".cache/cuda",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/inductor",
    "TRITON_CACHE_DIR": ".cache/triton",
}.items():
    path = ROOT / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(path))

# Minimal systems may have Python but not its development headers. The README
# shows how to extract the matching distro package here without changing /usr.
headers = ROOT / ".cache/python-dev/usr/include"
if (headers / "python3.12/Python.h").exists():
    for key in ("C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        paths = [str(headers / "python3.12"), str(headers)]
        if os.environ.get(key):
            paths.append(os.environ[key])
        os.environ[key] = os.pathsep.join(paths)

import torch
from experiment_control import cache_key, cached_ptx, project_path
from experiment_provenance import CUDA_SOURCES

COMPILATION_INFO = {}


def _compile_ptx_uncached(source: bytes, architecture: str) -> bytes:
    """Use PyTorch's local NVRTC package, without a system nvcc install."""
    import nvidia.cuda_nvrtc

    base = Path(next(iter(nvidia.cuda_nvrtc.__path__)))
    nvrtc = ct.CDLL(str(base / "lib/libnvrtc.so.12"))
    program = ct.c_void_p()
    nvrtc.nvrtcCreateProgram.argtypes = [
        ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p,
        ct.c_int, ct.c_void_p, ct.c_void_p,
    ]
    nvrtc.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
    for name in ("nvrtcGetPTXSize", "nvrtcGetProgramLogSize"):
        getattr(nvrtc, name).argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    for name in ("nvrtcGetPTX", "nvrtcGetProgramLog"):
        getattr(nvrtc, name).argtypes = [ct.c_void_p, ct.c_void_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
    status = nvrtc.nvrtcCreateProgram(
        ct.byref(program), source, b"kernels.cu", 0, None, None
    )
    if status:
        raise RuntimeError(f"nvrtcCreateProgram failed: {status}")
    options = (ct.c_char_p * 2)(
        f"--gpu-architecture=compute_{architecture}".encode(), b"--std=c++17"
    )
    try:
        status = nvrtc.nvrtcCompileProgram(program, len(options), options)
        size = ct.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
        log = ct.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        if status:
            raise RuntimeError(f"NVRTC failed ({status}):\n{log.value.decode()}")
        status = nvrtc.nvrtcGetPTXSize(program, ct.byref(size))
        if status:
            raise RuntimeError(f"nvrtcGetPTXSize failed: {status}")
        ptx = ct.create_string_buffer(size.value)
        status = nvrtc.nvrtcGetPTX(program, ptx)
        if status:
            raise RuntimeError(f"nvrtcGetPTX failed: {status}")
        return ptx.value
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))


def compile_ptx() -> bytes:
    import nvidia.cuda_nvrtc

    library = Path(next(iter(nvidia.cuda_nvrtc.__path__))) / "lib/libnvrtc.so.12"
    nvrtc = ct.CDLL(str(library))
    nvrtc.nvrtcVersion.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
    major, minor = ct.c_int(), ct.c_int()
    if nvrtc.nvrtcVersion(ct.byref(major), ct.byref(minor)):
        raise RuntimeError("Cannot query NVRTC version")
    stat = library.stat()
    compiler = f"NVRTC {major.value}.{minor.value}:{stat.st_size}:{stat.st_mtime_ns}"
    architecture = "".join(map(str, torch.cuda.get_device_capability()))
    options = [f"--gpu-architecture=compute_{architecture}", "--std=c++17"]
    source = b"\n".join((ROOT / name).read_bytes() for name in CUDA_SOURCES)
    key = cache_key(source, architecture, compiler, options)
    directory = project_path(ROOT, ".cache/ptx")
    data, hit = cached_ptx(directory, key, lambda: _compile_ptx_uncached(source, architecture),
                           force=os.environ.get("KERNEL_LAB_REBUILD") == "1")
    COMPILATION_INFO.update(cache_key=key, cache_hit=hit, compiler=compiler,
                            architecture=architecture, cache_file=str(directory / f"{key}.ptx"))
    return data


class KernelLab:
    """Single-device FP32 kernel experiment with explicit close().

    Keep this object alive until all launches AND captured graphs using it
    have finished. close() synchronizes and unloads; never replay a graph
    after close(). Calls enqueue asynchronously on PyTorch's current stream.
    """

    def __init__(self):
        torch.cuda.init()
        self.device = torch.cuda.current_device()
        self.lib = ct.CDLL(str(ROOT / "native/build/libkernel_lab.so"))
        pointer = ct.c_void_p
        self.lib.lab_load_ptx.argtypes = [ct.c_char_p, ct.POINTER(pointer)]
        self.lib.lab_get_kernel.argtypes = [pointer, ct.c_char_p, ct.POINTER(pointer)]
        self.lib.lab_launch.argtypes = [
            pointer, ct.c_uint, ct.c_uint, ct.c_uint, pointer, ct.POINTER(pointer)
        ]
        self.lib.lab_unload.argtypes = [pointer]
        self.lib.lab_error_string.argtypes = [ct.c_int]
        self.lib.lab_error_string.restype = ct.c_char_p
        self.module = pointer()
        with torch.cuda.device(self.device):
            # Force the runtime's primary context to be current before loading PTX.
            torch.empty(1, device="cuda")
            self.check(self.lib.lab_load_ptx(compile_ptx(), ct.byref(self.module)))
        self.compilation = dict(COMPILATION_INFO)
        self.functions = {}
        names = ["vector_add", "gemv_serial"]
        names += [f"gemv_r{r}_t{t}" for r in (1, 2, 4, 8) for t in (64, 128, 256)]
        names += [f"residual_rmsnorm_t{t}" for t in (128, 256)]
        names += [f"residual_rmsnorm_k4096_t{t}" for t in (128, 256)]
        names += ["gemm_m8_n32", "gemm_m16_n64", "gemm_m32_n32"]
        names += [f"gemv_warp_t{t}_v{v}" for t in (32, 64, 128, 256) for v in (1, 4)]
        names += [f"proj_stream_t{t}_c{c}_p{p}" for t in (64, 128, 256)
                  for c, p in ((0, 0), (1, 0), (1, 4))]
        names += [f"proj_packed_t{t}" for t in (64, 128, 256)] + ["proj_reduce"]
        names += [f"fp4_a{a}_t{t}" for a in (16, 32) for t in (32, 64, 128, 256)]
        names += [f"int8_a{a}_t{t}" for a in (16, 32) for t in (32, 64, 128, 256)]
        names += [f"lossless_t{t}" for t in (32, 64, 128, 256)] + ["lossless_unpack"]
        names += [f"tiny_k{k}_t{t}_v{v}" for k in (64,128,256,512,1024)
                  for t in (32,64,128,256) for v in (1,4)]
        names += [f"fp28_t{t}" for t in (32,64,128,256)] + ["fp28_unpack"]
        names += [f"half_warp_{s}_t{t}{'_v8' if v==8 else ''}" for s in ("f16","bf16")
                  for t in (32,64,128,256) for v in (2,8)]
        for name in names:
            function = pointer()
            self.check(self.lib.lab_get_kernel(self.module, name.encode(), ct.byref(function)))
            self.functions[name] = function

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lab_error_string(status).decode())

    def validate(self, *tensors):
        if not self.module.value:
            raise RuntimeError("Kernel module is closed")
        for tensor in tensors:
            if tensor.device != torch.device("cuda", self.device):
                raise ValueError("All tensors must be on this KernelLab's CUDA device")
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise ValueError("Only contiguous FP32 tensors are supported")
            if tensor.requires_grad:
                raise ValueError("This experiment supports inference only")

    def launch(self, name, grid, threads, *args):
        values = [ct.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
        params = (ct.c_void_p * len(values))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in values])
        stream = torch.cuda.current_stream(self.device)
        with torch.cuda.device(self.device):
            self.check(self.lib.lab_launch(
                self.functions[name], grid[0], grid[1], threads,
                ct.c_void_p(stream.cuda_stream), params,
            ))
        # ctypes bypasses the caching allocator's normal stream bookkeeping.
        for value in args:
            if isinstance(value, torch.Tensor):
                value.record_stream(stream)

    def add(self, a, b, out=None):
        self.validate(a, b)
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError("add requires equal, nonempty shapes")
        out = self.output(out, a.shape, a, b)
        self.launch("vector_add", ((a.numel() + 255) // 256, 1), 256,
                    a, b, out, ct.c_int(a.numel()))
        return out

    def tiled_matmul(self, x, w, tile="m32_n32", out=None):
        self.validate(x, w)
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Expected x=[M,K], w=[N,K]")
        m, k = x.shape
        n = w.shape[0]
        if min(m, n, k) <= 0 or max(m * k, n * k, m * n) >= 2**31:
            raise ValueError("Dimensions must be positive and fit signed 32-bit indexing")
        configs = {"m8_n32": (8, 32, 128), "m16_n64": (16, 64, 256),
                   "m32_n32": (32, 32, 256)}
        if tile not in configs:
            raise ValueError("Unsupported GEMM tile")
        bm, bn, threads = configs[tile]
        out = self.output(out, (m, n), x, w)
        self.launch(f"gemm_{tile}", ((n + bn - 1) // bn, (m + bm - 1) // bm), threads,
                    x, w, out, ct.c_int(m), ct.c_int(n), ct.c_int(k))
        return out

    def warp_matmul(self, x, w, threads=128, vector=4, out=None):
        self.validate(x, w)
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Expected x=[M,K], w=[N,K]")
        m, k = x.shape
        n = w.shape[0]
        if min(m, n, k) <= 0 or max(m * k, n * k, m * n) >= 2**31:
            raise ValueError("Dimensions must be positive and fit signed 32-bit indexing")
        if threads not in (32, 64, 128, 256) or vector not in (1, 4):
            raise ValueError("Unsupported warp configuration")
        if vector == 4 and (k % 4 or x.data_ptr() % 16 or w.data_ptr() % 16):
            raise ValueError("Vector loads require K divisible by 4 and 16-byte aligned inputs")
        out = self.output(out, (m, n), x, w)
        rows = threads // 32
        self.launch(f"gemv_warp_t{threads}_v{vector}", ((n + rows - 1) // rows, m), threads,
                    x, w, out, ct.c_int(n), ct.c_int(k))
        return out

    def tiny_matmul(self, x, w, threads=32, vector=1, out=None):
        self.validate(x,w)
        if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1]:
            raise ValueError("Expected x=[M,K], w=[N,K]")
        m,k=x.shape
        n=w.shape[0]
        if k not in (64,128,256,512,1024) or min(m,n)<=0 or max(m*k,n*k,m*n)>=2**31:
            raise ValueError("Unsupported fixed-K shape or signed 32-bit indexing")
        if threads not in (32,64,128,256) or vector not in (1,4):
            raise ValueError("Unsupported fixed-K configuration")
        if vector==4 and (x.data_ptr()%16 or w.data_ptr()%16):
            raise ValueError("Vector loads require 16-byte alignment")
        out=self.output(out,(m,n),x,w)
        rows=threads//32
        self.launch(f"tiny_k{k}_t{threads}_v{vector}",((n+rows-1)//rows,m),threads,
                    x,w,out,ct.c_int(n))
        return out

    def output(self, out, shape, *inputs):
        if out is None:
            return torch.empty(shape, device=inputs[0].device, dtype=torch.float32)
        self.validate(out)
        if tuple(out.shape) != tuple(shape):
            raise ValueError("Output shape mismatch")
        # Reject storage aliasing, including views with different data_ptr values.
        if any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in inputs):
            raise ValueError("Output must not share storage with inputs")
        return out

    def matmul(self, x, w, rows=4, threads=128, out=None, serial=False):
        """Compute x @ w.T, where x=[tokens,K] and w=[outputs,K]."""
        self.validate(x, w)
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Expected x=[M,K], w=[N,K]")
        m, k = x.shape
        n = w.shape[0]
        if min(m, n, k) <= 0 or max(m * k, n * k, m * n) >= 2**31:
            raise ValueError("Dimensions must be positive and fit signed 32-bit indexing")
        if rows not in (1, 2, 4, 8) or threads not in (64, 128, 256):
            raise ValueError("Unsupported launch configuration")
        if serial and m != 1:
            raise ValueError("Serial starting kernel supports one token")
        out = self.output(out, (m, n), x, w)
        name = "gemv_serial" if serial else f"gemv_r{rows}_t{threads}"
        grid = ((n + threads - 1) // threads, 1) if serial else ((n + rows - 1) // rows, m)
        self.launch(name, grid, threads, x, w, out, ct.c_int(n), ct.c_int(k))
        return out

    def residual_rmsnorm(self, x, residual, weight, eps=1e-6, threads=256, out=None):
        self.validate(x, residual, weight)
        if x.ndim != 2 or x.shape != residual.shape or weight.shape != (x.shape[1],):
            raise ValueError("Expected x/residual=[M,K], weight=[K]")
        m, k = x.shape
        if not (m > 0 and 0 < k <= 8192 and m * k < 2**31 and eps > 0):
            raise ValueError("Require M>0, 1<=K<=8192, FP32 indexing, eps>0")
        if threads not in (128, 256):
            raise ValueError("threads must be 128 or 256")
        out = self.output(out, x.shape, x, residual, weight)
        kernel = f"residual_rmsnorm_k4096_t{threads}" if k == 4096 else f"residual_rmsnorm_t{threads}"
        self.launch(kernel, (m, 1), threads,
                    x, residual, weight, out, ct.c_int(k), ct.c_float(eps))
        return out

    def close(self):
        if self.module.value:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self.check(self.lib.lab_unload(self.module))
            self.module = ct.c_void_p()


_modules: dict[int, KernelLab] = {}


def default_lab() -> KernelLab:
    device = torch.cuda.current_device()
    if device not in _modules:
        _modules[device] = KernelLab()
    return _modules[device]


@torch.library.custom_op("kernel_lab::matmul", mutates_args=(), device_types="cuda")
def matmul(x: torch.Tensor, w: torch.Tensor, rows: int = 4, threads: int = 128) -> torch.Tensor:
    """torch.compile-compatible boundary around the C++/CUDA implementation."""
    with torch.cuda.device(x.device):
        lab = default_lab()
        if x.shape[0] > 1:
            return lab.tiled_matmul(x, w)
        return lab.matmul(x, w, rows=rows, threads=threads)


@matmul.register_fake
def _matmul_fake(x, w, rows=4, threads=128):
    return x.new_empty((x.shape[0], w.shape[0]))


@torch.library.custom_op("kernel_lab::warp_matmul", mutates_args=(), device_types="cuda")
def warp_matmul(
    x: torch.Tensor, w: torch.Tensor, threads: int = 128, vector: int = 4
) -> torch.Tensor:
    """Explicit warp kernel; select settings with measured workload results."""
    with torch.cuda.device(x.device):
        return default_lab().warp_matmul(x, w, threads, vector)


@warp_matmul.register_fake
def _warp_matmul_fake(x, w, threads=128, vector=4):
    return x.new_empty((x.shape[0], w.shape[0]))


@torch.library.custom_op("kernel_lab::residual_rmsnorm", mutates_args=(), device_types="cuda")
def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    with torch.cuda.device(x.device):
        return default_lab().residual_rmsnorm(x, residual, weight, eps=eps)


@residual_rmsnorm.register_fake
def _rmsnorm_fake(x, residual, weight, eps=1e-6):
    return torch.empty_like(x)
