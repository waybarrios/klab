"""Thin Torch adapters for the frozen R2 raw/coded BF16 projection kernels.

INTERNAL BENCHMARK CANDIDATES, not trusted policy selection. Nothing here reads
evidence, consumes or relaxes kernel/dispatch gates, or claims a speedup; every
candidate reports custom_win_validated=False. dynamic_projection.prepare_projection
remains vendor-only.

prepared_bf16::coded_r2 and prepared_bf16::raw_r2 are allocation-only custom
ops. Their CUDA kernels check the exact prepared M,N,K and then call the
unchanged mtile_bf16.fused_linear/raw_linear, which delegate M == 1 to the R1
compressed_bf16 arithmetic. Launches, arithmetic and buffer/alignment/alias
validation stay in those wrappers. The coded op rebuilds a temporary
compressed_bf16.PackedBF16 header around the retained compressed module; its
source_sha256 field is unused by the kernel, and real provenance lives on the
owning candidate. Fake kernels check only logical contracts and work on CPU
FakeTensors: no data_ptr, CUDA query, module load or launch.

The real ops never compile, pack, fingerprint or search: both CUDA modules must
already be retained here for the device (candidate preparation or
preload_modules, outside capture). Candidates are device-bound plans:
preparation rejects CPU weights before touching CUDA, runs outside capture,
requires TF32 and reduced-precision BF16 reductions to be off, and retains the
modules plus only the original buffers (uploaded payload/metadata, or the raw
weight). Module.to()/cuda()/half() and similar relocation is refused; reprepare
instead. Keep candidates alive for every graph that captured them, and never
mutate their buffers.

Importing this file imports neither wrapper and loads no CUDA module.
"""

import time

import torch

# Mirrors of the frozen wrapper limits, so Fake kernels import nothing.
# preload_modules checks them against compressed_bf16 and mtile_bf16.
_THREADS = (32, 64, 128, 256)
_VECTORS = (2, 8)
_MAX_M = 65535
_INDEX_LIMIT = 2**31
_BLOCK, _CODED_BYTES, _RAW_BYTES = 64, 96, 128
_OUTPUTS = {"bf16": torch.bfloat16, "fp32": torch.float32}
_HEADER_SOURCE = "prepared_kernels op header; provenance is on the owning candidate"

_MODULES = {}  # CUDA device index -> (compressed module, M-tile module), retained for graph lifetime.


# ---------------------------------------------------------------- contracts


def _output_name(value):
    for name, dtype in _OUTPUTS.items():
        if value is dtype or (isinstance(value, str) and value == name):
            return name
    raise ValueError("output_dtype must be 'bf16'/torch.bfloat16 or 'fp32'/torch.float32")


def _check_plan(m, n, k, threads, vector, coded):
    """Prepared M,N,K and launch configuration, with the wrappers' index limits."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (m, n, k, threads, vector)):
        raise ValueError("M, N, K, threads and vector must be integers")
    if threads not in _THREADS or vector not in _VECTORS:
        raise ValueError(f"Unsupported launch configuration; threads in {_THREADS}, vector in {_VECTORS}")
    if min(m, n, k) <= 0 or m > _MAX_M or max(m * k, n * k, m * n) >= _INDEX_LIMIT:
        raise ValueError(f"Dimensions must be positive, M<={_MAX_M}, and fit signed 32-bit indexing")
    if k % vector:
        raise ValueError(f"Require K divisible by {vector}")
    if coded and k % _BLOCK:
        raise ValueError(f"Coded projections require K divisible by {_BLOCK}")


def _check_operand(tensor, label, dtype, ndim, shape=None):
    """Logical tensor contract; device placement is checked by the caller."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D {dtype} tensor")
    if not tensor.is_contiguous() or tensor.requires_grad:
        raise ValueError(f"{label} must be contiguous and inference-only (no requires_grad)")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{label} must have the prepared shape {list(shape)}")


def _check_coded_buffers(payload, metadata, n, k):
    blocks = n * (k // _BLOCK)
    if metadata.numel() != blocks or not _CODED_BYTES * blocks <= payload.numel() <= _RAW_BYTES * blocks:
        raise ValueError("Packed buffers do not match N,K")


def _check_mirrors(cb, mt):
    if ((mt.THREADS, mt.VECTORS, mt.MAX_M, mt.OUTPUTS) != (_THREADS, _VECTORS, _MAX_M, tuple(_OUTPUTS))
            or (cb.THREADS, cb.VECTORS, cb.OUTPUT_DTYPES) != (_THREADS, _VECTORS, _OUTPUTS)
            or (cb.BLOCK, cb.CODED_BYTES, cb.RAW_BYTES) != (_BLOCK, _CODED_BYTES, _RAW_BYTES)):
        raise RuntimeError("prepared_kernels limits no longer match the frozen wrappers")


# ---------------------------------------------------------------- modules


def preload_modules(device):
    """Load and retain this device's compressed and M-tile modules, outside capture."""
    import compressed_bf16 as cb
    import mtile_bf16 as mt

    _check_mirrors(cb, mt)
    index = cb._device_index(device)
    with torch.cuda.device(index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Preload modules before CUDA graph capture")
        modules = (cb.get_module(index), mt.get_module(index))
    _MODULES[index] = modules
    return modules


def _retained(x, m, k):
    """Check x against the prepared M,K; return the retained modules without compiling."""
    if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    if x.ndim != 2 or tuple(x.shape) != (m, k):
        raise ValueError(f"x must have the prepared shape [M,K]=[{m},{k}]")
    modules = _MODULES.get(x.device.index)
    if modules is None or any(module.closed for module in modules):
        raise RuntimeError("Prepare a candidate or call preload_modules on this device outside capture first")
    return modules


# ---------------------------------------------------------------- operators


@torch.library.custom_op("prepared_bf16::coded_r2", mutates_args=(), device_types="cuda")
def coded_r2(x: torch.Tensor, payload: torch.Tensor, metadata: torch.Tensor, m: int, n: int, k: int,
             threads: int, vector: int, output_dtype: str) -> torch.Tensor:
    import compressed_bf16 as cb
    import mtile_bf16 as mt

    module = _retained(x, m, k)[0]
    packed = cb.PackedBF16(payload, metadata, n, k, _HEADER_SOURCE, cb.FORMAT_VERSION, module)
    with torch.cuda.device(x.device):
        return mt.fused_linear(x, packed, threads=threads, vector=vector, output_dtype=output_dtype)


@coded_r2.register_fake
def _coded_r2_fake(x, payload, metadata, m, n, k, threads, vector, output_dtype):
    dtype = _OUTPUTS[_output_name(output_dtype)]
    _check_plan(m, n, k, threads, vector, coded=True)
    _check_operand(x, "x", torch.bfloat16, 2, (m, k))
    _check_operand(payload, "payload", torch.uint8, 1)
    _check_operand(metadata, "metadata", torch.int32, 1)
    if payload.device != x.device or metadata.device != x.device:
        raise ValueError("x, payload and metadata must share a device")
    _check_coded_buffers(payload, metadata, n, k)
    return x.new_empty((m, n), dtype=dtype)


@torch.library.custom_op("prepared_bf16::raw_r2", mutates_args=(), device_types="cuda")
def raw_r2(x: torch.Tensor, w: torch.Tensor, m: int, n: int, k: int,
           threads: int, vector: int, output_dtype: str) -> torch.Tensor:
    import mtile_bf16 as mt

    _retained(x, m, k)
    if tuple(w.shape) != (n, k):
        raise ValueError(f"w must have the prepared shape [N,K]=[{n},{k}]")
    with torch.cuda.device(x.device):
        return mt.raw_linear(x, w, threads=threads, vector=vector, output_dtype=output_dtype)


@raw_r2.register_fake
def _raw_r2_fake(x, w, m, n, k, threads, vector, output_dtype):
    dtype = _OUTPUTS[_output_name(output_dtype)]
    _check_plan(m, n, k, threads, vector, coded=False)
    _check_operand(x, "x", torch.bfloat16, 2, (m, k))
    _check_operand(w, "w", torch.bfloat16, 2, (n, k))
    if w.device != x.device:
        raise ValueError("x and w must share a device")
    return x.new_empty((m, n), dtype=dtype)


_CODED_OP = torch.ops.prepared_bf16.coded_r2.default
_RAW_OP = torch.ops.prepared_bf16.raw_r2.default


# ---------------------------------------------------------------- candidates


def _check_cuda_operand(tensor, label, dtype, ndim):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        raise ValueError(f"{label} must be a CUDA tensor")
    _check_operand(tensor, label, dtype, ndim)


class _Candidate(torch.nn.Module):
    """Shared preparation for one device-bound (M,N,K, config, output) plan."""

    custom_win_validated = False  # Benchmark candidate only; never a policy decision.

    def _contract(self, kind, tokens, n, k, threads, vector, output_dtype):
        """Logical checks only; runs before anything touches CUDA."""
        self.output_dtype = _output_name(output_dtype)
        _check_plan(tokens, n, k, threads, vector, coded=kind == "coded")
        self.kind = kind
        self.m, self.n, self.k = tokens, n, k
        self.shape_mnk = (tokens, n, k)
        self.threads, self.vector = threads, vector

    def _bind(self, device):
        """Outside capture, under the FP32-accumulation flag contract, modules retained."""
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare candidates before CUDA graph capture")
            self.precision_flags = {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction":
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
            if any(self.precision_flags.values()):
                raise ValueError("Set allow_tf32 and allow_bf16_reduced_precision_reduction to False "
                                 "before preparing this FP32-accumulation contract")
            self._kernel_modules = preload_modules(device)
        self.device = device
        return self._kernel_modules

    def _finish(self, started):
        import mtile_bf16 as mt

        self.config = {"revision": mt.REVISION, "threads": self.threads, "vector": self.vector,
                       "tile": mt.tile_for_m(self.m),
                       "route": "R1 delegation (M == 1)" if self.m == 1 else "R2 M-tile"}
        with torch.cuda.device(self.device):
            self.implementation = mt.implementation_fingerprint(self.device)
            # One launch before any capture; it also runs the wrappers' alignment checks.
            self.linear(torch.zeros((self.m, self.k), dtype=torch.bfloat16, device=self.device))
        self.setup_seconds = time.perf_counter() - started

    def _apply(self, fn, recurse=True):
        raise RuntimeError(f"{type(self).__name__} is a device-bound prepared plan; "
                           "reprepare it instead of moving or casting it")

    def forward(self, x):
        return self.linear(x)

    def describe(self):
        """JSON-ready setup record for the dispatch benchmark; not a performance claim."""
        return {"kind": self.kind, "shape_mnk": list(self.shape_mnk), "config": dict(self.config),
                "output_dtype": self.output_dtype, "storage": dict(self.storage),
                "provenance": dict(self.provenance), "implementation": self.implementation,
                "precision_flags": dict(self.precision_flags), "device": str(self.device),
                "setup_seconds": self.setup_seconds, "custom_win_validated": self.custom_win_validated}


class CodedCandidate(_Candidate):
    """Coded R2 over a compressed_bf16.upload() result; retains only its packed buffers."""

    def __init__(self, packed, tokens=1, *, threads, vector, output_dtype):
        super().__init__()
        started = time.perf_counter()
        import compressed_bf16 as cb

        if not isinstance(packed, cb.PackedBF16):
            raise ValueError("packed must come from compressed_bf16.upload()")
        payload, metadata = packed.payload, packed.metadata
        _check_cuda_operand(payload, "payload", torch.uint8, 1)
        _check_cuda_operand(metadata, "metadata", torch.int32, 1)
        if metadata.device != payload.device:
            raise ValueError("payload and metadata must share a device")
        if packed.format_version != cb.FORMAT_VERSION:
            raise ValueError(f"Unsupported packed format version; expected {cb.FORMAT_VERSION}")
        self._contract("coded", tokens, packed.n, packed.k, threads, vector, output_dtype)
        _check_coded_buffers(payload, metadata, self.n, self.k)
        if packed.module is not self._bind(payload.device)[0]:
            raise ValueError("packed must come from compressed_bf16.upload() on this device")
        self.register_buffer("payload", payload, persistent=False)
        self.register_buffer("metadata", metadata, persistent=False)
        self._packed = packed  # Uploaded header for linear_out; shares the buffers above.
        self.storage = {"format": f"compressed_bf16_v{packed.format_version}",
                        "bytes": packed.storage_bytes, "raw_bf16_bytes": 2 * self.n * self.k}
        self.provenance = {"source_sha256": packed.source_sha256, "format_version": packed.format_version}
        self._finish(started)

    def linear(self, x):
        """Allocating registered op (Fake/torch.compile-safe)."""
        return _CODED_OP(x, self.payload, self.metadata, self.m, self.n, self.k,
                         self.threads, self.vector, self.output_dtype)

    def linear_out(self, x, *, out):
        """Write into `out` through the unchanged wrapper and its alias/device/dtype guards."""
        import mtile_bf16 as mt

        _retained(x, self.m, self.k)
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a tensor; use linear(x) to allocate")
        with torch.cuda.device(x.device):
            return mt.fused_linear(x, self._packed, threads=self.threads, vector=self.vector,
                                   output_dtype=self.output_dtype, out=out)


class RawCandidate(_Candidate):
    """Raw R2 over the caller's original BF16 weight [N,K], retained without copying."""

    def __init__(self, w, tokens=1, *, threads, vector, output_dtype):
        super().__init__()
        started = time.perf_counter()
        _check_cuda_operand(w, "w", torch.bfloat16, 2)
        self._contract("raw", tokens, *w.shape, threads, vector, output_dtype)
        self._bind(w.device)
        self.register_buffer("w", w, persistent=False)
        self.storage = {"format": "raw_bf16", "bytes": 2 * self.n * self.k}
        self.provenance = {"weights": "caller's original tensor; not copied or hashed"}
        self._finish(started)

    def linear(self, x):
        """Allocating registered op (Fake/torch.compile-safe)."""
        return _RAW_OP(x, self.w, self.m, self.n, self.k, self.threads, self.vector, self.output_dtype)

    def linear_out(self, x, *, out):
        """Write into `out` through the unchanged wrapper and its alias/device/dtype guards."""
        import mtile_bf16 as mt

        _retained(x, self.m, self.k)
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a tensor; use linear(x) to allocate")
        with torch.cuda.device(x.device):
            return mt.raw_linear(x, self.w, threads=self.threads, vector=self.vector,
                                 output_dtype=self.output_dtype, out=out)
