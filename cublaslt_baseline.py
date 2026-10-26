"""cuBLASLt vendor baseline: bounded heuristic pool, frozen and restorable plans.

C[M,N] = x[M,K] @ w[N,K].T, computed column-major as D[N,M] = op(A) @ B where
B = x viewed col-major [K,M] (ld K) and C = D = out viewed col-major [N,M] (ld N).
  layout "original":   A = w viewed col-major [K,N] (ld K), transa = T
  layout "transposed": A = w.T.contiguous() viewed col-major [N,K] (ld N), transa = N
Compute is CUBLAS_COMPUTE_32F (68, never FAST_TF32) with FP32 host alpha=1, beta=0.
Constants and struct layouts follow the installed cublasLt.h, cublas_api.h and
library_types.h (cuBLAS 12.8.4).

Eligibility: compute 68, reduction scheme NONE (0) or COMPUTE_TYPE (2) as queried
from the returned algorithm, workspace <= 64 MiB.  INPLACE (1) and OUTPUT_TYPE (4)
reduce split-K partials in the output type and are excluded.

Lifetimes.  One LtWorkspace is shared by every plan of a serialized runner on one
device.  Concurrent execution of plans sharing a workspace (two streams, overlapping
graph replays) is unsupported.  Plans reference their workspace and weights; callers
must keep plans and the workspace open while any CUDA graph that captured them can
replay.  close() synchronizes the device before releasing anything.

Weights.  Every plan binds the original weight; a "transposed" plan also owns the
packed copy made at prepare/restore time.  plan.run(x, w, out) requires `w` to be that
original weight (same pointer/shape/dtype, and for transposed plans an unchanged
version counter when available), so nothing is packed or allocated at run time.
Inference-mode tensors have no version counter; their weights must remain
immutable as required by the caller contract.

Diagnostics.  prepare_lt_plans returns an LtPlanList (a list of LtPlan) whose
.search_diagnostics holds one dict per layout: heuristic status, returned count,
selected candidate indices and every rejected candidate with its reason.  An empty
list means no eligible Lt algorithm; nothing falls back to cublasGemmEx.

The CPU helpers (validate_frozen, select_heuristics) import neither torch nor the
library.  GPU entry points raise CublasLtUnavailable if either cannot be loaded.
"""

import base64
import binascii
import copy
import ctypes as ct
import math
from pathlib import Path

MAX_WORKSPACE = 64 << 20
MAX_CANDIDATES = 8
FORMAT = "cublaslt-plan-v1"
_INT32_MAX = 2**31 - 1

# cublas_api.h
CUBLAS_STATUS_SUCCESS = 0
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_COMPUTE_32F = 68
# library_types.h
CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF = 0, 2, 14
# cublasLt.h
MATMUL_DESC_TRANSA, MATMUL_DESC_TRANSB = 3, 4                      # int32_t
PREF_MAX_WORKSPACE_BYTES = 1                                       # uint64_t
PREF_REDUCTION_SCHEME_MASK = 3                                     # uint32_t
PREF_MIN_ALIGNMENT = (5, 6, 7, 8)                                  # A, B, C, D; uint32_t
REDUCTION_NONE, REDUCTION_COMPUTE_TYPE = 0, 2
CONFIG_ID, CONFIG_TILE_ID, CONFIG_SPLITK_NUM, CONFIG_REDUCTION_SCHEME = 0, 1, 2, 3
CONFIG_CTA_SWIZZLING, CONFIG_CUSTOM_OPTION, CONFIG_STAGES_ID = 4, 5, 6
CAP_NUMERICAL_IMPL_FLAGS = 15                                      # uint64_t
CAP_MIN_ALIGNMENT = (16, 17, 18, 19)                               # A, B, C, D; uint32_t

ELIGIBLE_REDUCTION = (REDUCTION_NONE, REDUCTION_COMPUTE_TYPE)
LAYOUTS = ("original", "transposed")
CONTRACTS = (("bf16", "bf16"), ("bf16", "fp32"), ("fp16", "fp32"), ("fp32", "fp32"))
CONTEXT_FIELDS = ("library_version", "device_uuid", "input_dtype", "output_dtype", "shape_mnk", "layout")
_REQUIRED = CONTEXT_FIELDS + ("compute_type", "reduction_scheme", "workspace_bytes",
                              "minimum_alignment", "algorithm_b64")
_CUDA_TYPE = {"bf16": CUDA_R_16BF, "fp16": CUDA_R_16F, "fp32": CUDA_R_32F}


# ---------------------------------------------------------------- CPU helpers

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _contract(record):
    """Context-free eligibility checks; returns the 64 algorithm bytes."""
    if not isinstance(record, dict):
        raise ValueError("frozen record must be a dict")
    missing = [f for f in _REQUIRED if f not in record]
    if missing:
        raise ValueError(f"frozen record is missing {missing}")
    if not _is_int(record["library_version"]) or record["library_version"] <= 0:
        raise ValueError("library_version must be a positive integer")
    if not isinstance(record["device_uuid"], str) or not record["device_uuid"]:
        raise ValueError("device_uuid must be a non-empty string")
    if (record["input_dtype"], record["output_dtype"]) not in CONTRACTS:
        raise ValueError(f"unsupported input/output dtypes; expected one of {CONTRACTS}")
    shape = record["shape_mnk"]
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or not all(_is_int(v) and 0 < v <= _INT32_MAX for v in shape)):
        raise ValueError("shape_mnk must be three positive 32-bit integers")
    if record["layout"] not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if not _is_int(record["compute_type"]) or record["compute_type"] != CUBLAS_COMPUTE_32F:
        raise ValueError("compute_type must be CUBLAS_COMPUTE_32F (68)")
    if not _is_int(record["reduction_scheme"]) or record["reduction_scheme"] not in ELIGIBLE_REDUCTION:
        raise ValueError("reduction_scheme must be NONE (0) or COMPUTE_TYPE (2)")
    if not _is_int(record["workspace_bytes"]) or not 0 <= record["workspace_bytes"] <= MAX_WORKSPACE:
        raise ValueError(f"workspace_bytes must be in [0, {MAX_WORKSPACE}]")
    align = record["minimum_alignment"]
    if (not isinstance(align, (list, tuple)) or len(align) != 4
            or not all(_is_int(v) and v > 0 and v & (v - 1) == 0 for v in align)):
        raise ValueError("minimum_alignment must be four positive powers of two (A, B, C, D)")
    if not isinstance(record["algorithm_b64"], str):
        raise ValueError("algorithm_b64 must be a base64 string")
    try:
        blob = base64.b64decode(record["algorithm_b64"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"algorithm_b64 is not valid base64: {error}") from None
    if len(blob) != 64:
        raise ValueError(f"algorithm must be exactly 64 bytes (cublasLtMatmulAlgo_t), got {len(blob)}")
    return blob


def _same(a, b):
    norm = lambda v: list(v) if isinstance(v, tuple) else v
    return type(norm(a)) is type(norm(b)) and norm(a) == norm(b)


def validate_frozen(record, *, context):
    """Raise ValueError unless `record` is eligible and bound to exactly `context`.

    `context` supplies CONTEXT_FIELDS for the runtime (library version, device UUID,
    dtypes, [M, N, K], layout).  Returns the decoded 64-byte algorithm.  Opaque content
    is not inspected here; GPU restore runs cublasLtMatmulAlgoCheck.
    """
    blob = _contract(record)
    if not isinstance(context, dict) or any(f not in context for f in CONTEXT_FIELDS):
        raise ValueError(f"context must provide {CONTEXT_FIELDS}")
    for field in CONTEXT_FIELDS:
        if not _same(record[field], context[field]):
            raise ValueError(f"frozen {field} {record[field]!r} does not match runtime {context[field]!r}")
    return blob


def select_heuristics(records, *, max_candidates=MAX_CANDIDATES, workspace_limit=MAX_WORKSPACE):
    """First `max_candidates` eligible records in input order, as deep copies.

    Ineligible records are skipped, never relaxed; an empty result means no eligible
    algorithm.  Rejection reasons for an actual search live in search_diagnostics.
    """
    if not _is_int(max_candidates) or not 0 < max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be in [1, {MAX_CANDIDATES}]")
    if not _is_int(workspace_limit) or not 0 <= workspace_limit <= MAX_WORKSPACE:
        raise ValueError(f"workspace_limit must be in [0, {MAX_WORKSPACE}]")
    chosen = []
    for record in records:
        if len(chosen) == max_candidates:
            break
        try:
            _contract(record)
        except ValueError:
            continue
        if record["workspace_bytes"] <= workspace_limit:
            chosen.append(copy.deepcopy(record))
    return chosen


# ---------------------------------------------------------------- library binding

class CublasLtUnavailable(RuntimeError):
    """torch CUDA or libcublasLt could not be loaded; no Lt route exists."""


class CublasLtError(RuntimeError):
    def __init__(self, what, status):
        super().__init__(f"{what} failed: {_status_name(status)} ({status})")
        self.status = status


class _Algo(ct.Structure):
    _fields_ = [("data", ct.c_uint64 * 8)]


class _Heuristic(ct.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ct.c_size_t), ("state", ct.c_int),
                ("wavesCount", ct.c_float), ("reserved", ct.c_int * 4)]


if ct.sizeof(_Algo) != 64 or ct.sizeof(_Heuristic) != 96:
    raise ImportError("cublasLt struct layout mismatch with cublasLt.h")

_LIB = None
_LIB_PATH = None
_HANDLES = {}


def _lib():
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB
    errors = []
    try:
        import nvidia.cublas
        paths = [Path(p) / "lib/libcublasLt.so.12" for p in nvidia.cublas.__path__]
    except ImportError as error:
        raise CublasLtUnavailable(f"nvidia.cublas package unavailable: {error}") from error
    lib = None
    for path in paths:
        try:
            lib = ct.CDLL(str(path))
            break
        except OSError as error:
            errors.append(f"{path}: {error}")
    if lib is None:
        raise CublasLtUnavailable("libcublasLt.so.12 unavailable: " + ("; ".join(errors) or "no candidate path"))
    P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t  # C enums are 32-bit int
    signatures = {
        "cublasLtCreate": [ct.POINTER(P)],
        "cublasLtMatmulDescCreate": [ct.POINTER(P), I, I],
        "cublasLtMatmulDescDestroy": [P],
        "cublasLtMatmulDescSetAttribute": [P, I, P, S],
        "cublasLtMatrixLayoutCreate": [ct.POINTER(P), I, ct.c_uint64, ct.c_uint64, ct.c_int64],
        "cublasLtMatrixLayoutDestroy": [P],
        "cublasLtMatmulPreferenceCreate": [ct.POINTER(P)],
        "cublasLtMatmulPreferenceDestroy": [P],
        "cublasLtMatmulPreferenceSetAttribute": [P, I, P, S],
        "cublasLtMatmulAlgoGetHeuristic": [P] * 7 + [I, ct.POINTER(_Heuristic), ct.POINTER(I)],
        "cublasLtMatmulAlgoCheck": [P] * 6 + [ct.POINTER(_Algo), ct.POINTER(_Heuristic)],
        "cublasLtMatmulAlgoConfigGetAttribute": [ct.POINTER(_Algo), I, P, S, ct.POINTER(S)],
        "cublasLtMatmulAlgoCapGetAttribute": [ct.POINTER(_Algo), I, P, S, ct.POINTER(S)],
        # 16 parameters in this header: handle, desc, alpha, A, Adesc, B, Bdesc, beta,
        # C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream.
        "cublasLtMatmul": [P] * 12 + [ct.POINTER(_Algo), P, S, P],
    }
    try:
        for name, argtypes in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = argtypes, ct.c_int
        lib.cublasLtGetVersion.argtypes, lib.cublasLtGetVersion.restype = [], ct.c_size_t
        lib.cublasLtGetStatusName.argtypes, lib.cublasLtGetStatusName.restype = [I], ct.c_char_p
    except AttributeError as error:
        raise CublasLtUnavailable(f"libcublasLt is missing a required symbol: {error}") from error
    _LIB, _LIB_PATH = lib, str(path)
    return lib


def _status_name(status):
    try:
        return _lib().cublasLtGetStatusName(status).decode()
    except Exception:
        return "CUBLAS_STATUS_UNKNOWN"


def _check(status, what):
    if status != CUBLAS_STATUS_SUCCESS:
        raise CublasLtError(what, status)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise CublasLtUnavailable(f"torch unavailable: {error}") from error
    if not torch.cuda.is_available():
        raise CublasLtUnavailable("CUDA is unavailable")
    return torch


def _library_version():
    return int(_lib().cublasLtGetVersion())


def _handle(device):
    """One process-lifetime cublasLt handle per device, created outside capture."""
    if device not in _HANDLES:
        handle = ct.c_void_p()
        with _torch().cuda.device(device):
            _check(_lib().cublasLtCreate(ct.byref(handle)), "cublasLtCreate")
        _HANDLES[device] = handle
    return _HANDLES[device]


def _device_uuid(torch, device):
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return uuid if uuid.startswith("GPU-") else "GPU-" + uuid


def _dtype_names(torch):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def _align(ptr):
    return ptr & -ptr if ptr else 1 << 62


def _overlap(a, b):
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


def _query(fn, algo, attr, ctype):
    value, written = ctype(), ct.c_size_t()
    status = fn(ct.byref(algo), attr, ct.addressof(value), ct.sizeof(value), ct.byref(written))
    return value.value if status == CUBLAS_STATUS_SUCCESS and written.value == ct.sizeof(value) else None


def _describe(algo):
    cfg, cap = _lib().cublasLtMatmulAlgoConfigGetAttribute, _lib().cublasLtMatmulAlgoCapGetAttribute
    return {"algo_id": _query(cfg, algo, CONFIG_ID, ct.c_int32),
            "tile_id": _query(cfg, algo, CONFIG_TILE_ID, ct.c_uint32),
            "splitk_num": _query(cfg, algo, CONFIG_SPLITK_NUM, ct.c_int32),
            "reduction_scheme": _query(cfg, algo, CONFIG_REDUCTION_SCHEME, ct.c_uint32),
            "cta_swizzling": _query(cfg, algo, CONFIG_CTA_SWIZZLING, ct.c_uint32),
            "custom_option": _query(cfg, algo, CONFIG_CUSTOM_OPTION, ct.c_uint32),
            "stages_id": _query(cfg, algo, CONFIG_STAGES_ID, ct.c_uint32),
            "numerical_impl_flags": _query(cap, algo, CAP_NUMERICAL_IMPL_FLAGS, ct.c_uint64),
            "algo_minimum_alignment": [_query(cap, algo, a, ct.c_uint32) for a in CAP_MIN_ALIGNMENT]}


def _check_tensor(torch, t, name, shape, dtype, device):
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.device.index != device:
        raise ValueError(f"{name} must be a tensor on cuda:{device}")
    if t.dtype != dtype or tuple(t.shape) != tuple(shape) or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype} with shape {tuple(shape)}")


def _inputs(x, w, output_dtype, workspace):
    """Shared prepare/restore/constructor input checks."""
    torch = _torch()
    if not isinstance(workspace, LtWorkspace) or workspace.closed:
        raise ValueError("workspace must be an open LtWorkspace")
    device = workspace.device
    for name, t in (("x", x), ("w", w)):
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise ValueError(f"{name} must be a 2-D tensor")
        _check_tensor(torch, t, name, t.shape, t.dtype, device)
    names = _dtype_names(torch)
    in_name, out_name = names.get(x.dtype), names.get(output_dtype)
    if w.dtype != x.dtype or (in_name, out_name) not in CONTRACTS:
        raise ValueError(f"unsupported dtypes x={x.dtype} w={w.dtype} out={output_dtype}")
    (m, k), (n, k_w) = x.shape, w.shape
    if k != k_w or not all(0 < v <= _INT32_MAX for v in (m, n, k)) or max(m * k, n * k, m * n) > _INT32_MAX:
        raise ValueError(f"expected x[M,K], w[N,K] within 32-bit bounds, got {tuple(x.shape)}, {tuple(w.shape)}")
    if torch.cuda.current_device() != device:
        raise ValueError(f"cuda:{device} must be the current device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare/restore cuBLASLt plans outside CUDA graph capture")
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        raise ValueError("inference only: disable grad or detach inputs")
    return torch, device, (m, n, k), in_name, out_name


# ---------------------------------------------------------------- GPU wrapper

class LtWorkspace:
    """One shared uint8 workspace (<= 64 MiB) for a serialized runner on one device.

    Allocated once, outside capture.  Plans pass its full size to cublasLtMatmul.
    close() synchronizes then releases the tensor; never close it while a captured
    graph that uses it may replay.  Torch's own cuBLAS workspace is separate.
    """

    def __init__(self, device, bytes=MAX_WORKSPACE):
        if not _is_int(device) or device < 0:
            raise ValueError("device must be a non-negative CUDA device index")
        if not _is_int(bytes) or not 0 <= bytes <= MAX_WORKSPACE:
            raise ValueError(f"workspace bytes must be in [0, {MAX_WORKSPACE}]")
        torch = _torch()
        if device >= torch.cuda.device_count():
            raise ValueError(f"cuda:{device} does not exist")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("allocate LtWorkspace outside CUDA graph capture")
            self.tensor = torch.empty(bytes, dtype=torch.uint8, device=f"cuda:{device}")
        self.device, self.nbytes = device, bytes
        self.ptr = self.tensor.data_ptr() if bytes else None
        # cuBLAS documents 256-byte workspace alignment; not stated in these headers.
        if self.ptr is not None and self.ptr % 256:
            raise RuntimeError("workspace allocation is not 256-byte aligned")
        self._streams = set()

    @property
    def closed(self):
        return self.tensor is None

    def record_stream(self, stream):
        """Tell the caching allocator `stream` uses the workspace (skipped while capturing)."""
        if self.tensor is None or not self.nbytes or stream.cuda_stream in self._streams:
            return
        if not _torch().cuda.is_current_stream_capturing():
            self.tensor.record_stream(stream)
            self._streams.add(stream.cuda_stream)

    def close(self):
        if self.tensor is not None:
            _torch().cuda.synchronize(self.device)
            self.tensor, self.ptr = None, None


class _Op:
    """Matmul descriptor plus A, B and shared C/D layouts (host-side objects)."""

    def __init__(self, layout, in_name, out_name, m, n, k):
        lib = self.lib = _lib()
        self.desc, self.a, self.b, self.cd = (ct.c_void_p() for _ in range(4))
        try:
            _check(lib.cublasLtMatmulDescCreate(ct.byref(self.desc), CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "cublasLtMatmulDescCreate")
            transa = ct.c_int32(CUBLAS_OP_T if layout == "original" else CUBLAS_OP_N)
            transb = ct.c_int32(CUBLAS_OP_N)
            for attr, value in ((MATMUL_DESC_TRANSA, transa), (MATMUL_DESC_TRANSB, transb)):
                _check(lib.cublasLtMatmulDescSetAttribute(self.desc, attr, ct.addressof(value), ct.sizeof(value)),
                       f"cublasLtMatmulDescSetAttribute({attr})")
            rows, cols, ld = (k, n, k) if layout == "original" else (n, k, n)
            for target, dtype, shape in ((self.a, in_name, (rows, cols, ld)), (self.b, in_name, (k, m, k)),
                                         (self.cd, out_name, (n, m, n))):
                _check(lib.cublasLtMatrixLayoutCreate(ct.byref(target), _CUDA_TYPE[dtype], *shape),
                       "cublasLtMatrixLayoutCreate")
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        for ptr in (self.cd, self.b, self.a):
            if ptr.value:
                self.lib.cublasLtMatrixLayoutDestroy(ptr)
                ptr.value = None
        if self.desc.value:
            self.lib.cublasLtMatmulDescDestroy(self.desc)
            self.desc.value = None


def _heuristics(op, handle, workspace_bytes, alignment):
    lib = op.lib
    pref = ct.c_void_p()
    _check(lib.cublasLtMatmulPreferenceCreate(ct.byref(pref)), "cublasLtMatmulPreferenceCreate")
    try:
        settings = [(PREF_MAX_WORKSPACE_BYTES, ct.c_uint64(workspace_bytes)),
                    (PREF_REDUCTION_SCHEME_MASK, ct.c_uint32(REDUCTION_COMPUTE_TYPE))]
        settings += [(attr, ct.c_uint32(a)) for attr, a in zip(PREF_MIN_ALIGNMENT, alignment)]
        for attr, value in settings:
            _check(lib.cublasLtMatmulPreferenceSetAttribute(pref, attr, ct.addressof(value), ct.sizeof(value)),
                   f"cublasLtMatmulPreferenceSetAttribute({attr})")
        results, count = (_Heuristic * MAX_CANDIDATES)(), ct.c_int(0)
        status = lib.cublasLtMatmulAlgoGetHeuristic(handle, op.desc, op.a, op.b, op.cd, op.cd, pref,
                                                    MAX_CANDIDATES, results, ct.byref(count))
    finally:
        lib.cublasLtMatmulPreferenceDestroy(pref)
    returned = min(max(count.value, 0), MAX_CANDIDATES) if status == CUBLAS_STATUS_SUCCESS else 0
    return status, [_Heuristic.from_buffer_copy(results[i]) for i in range(returned)]


class LtPlan:
    """One frozen cuBLASLt algorithm bound to (x shape, weight, output dtype, layout).

    LtPlan(spec, algorithm, workspace, *, x, weight, packed=None) validates `spec`
    against the runtime, rebuilds descriptors and runs cublasLtMatmulAlgoCheck; no
    heuristics.  Use prepare_lt_plans or LtPlan.restore rather than calling it directly.
    """

    def __init__(self, spec, algorithm, workspace, *, x, weight, packed=None):
        spec = {f: v for f, v in copy.deepcopy(dict(spec)).items() if f != "algorithm_b64"}
        self._algorithm = bytes(algorithm)
        record = {**spec, "algorithm_b64": base64.b64encode(self._algorithm).decode()}
        _contract(record)
        torch = _torch()
        out_dtype = {v: k for k, v in _dtype_names(torch).items()}[spec["output_dtype"]]
        torch, device, (m, n, k), in_name, out_name = _inputs(x, weight, out_dtype, workspace)
        context = {"library_version": _library_version(), "device_uuid": _device_uuid(torch, device),
                   "input_dtype": in_name, "output_dtype": out_name, "shape_mnk": [m, n, k],
                   "layout": spec["layout"]}
        validate_frozen(record, context=context)
        if spec.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unsupported plan format {spec.get('format')!r}")
        if spec["workspace_bytes"] > workspace.nbytes:
            raise ValueError(f"plan needs {spec['workspace_bytes']} workspace bytes, have {workspace.nbytes}")
        transposed = spec["layout"] == "transposed"
        if transposed:
            _check_tensor(torch, packed, "packed weight", (k, n), weight.dtype, device)
        elif packed is not None:
            raise ValueError("original-layout plans do not take a packed weight")
        a = packed if transposed else weight
        need = spec["minimum_alignment"]
        for t, required, name in ((a, need[0], "A"), (x, need[1], "B")):
            if _align(t.data_ptr()) < required:
                raise ValueError(f"{name} pointer is not {required}-byte aligned")
        self._algo = _Algo.from_buffer_copy(self._algorithm)
        scheme = _query(_lib().cublasLtMatmulAlgoConfigGetAttribute, self._algo, CONFIG_REDUCTION_SCHEME, ct.c_uint32)
        if scheme != spec["reduction_scheme"]:
            raise ValueError(f"algorithm reduction scheme {scheme} does not match frozen {spec['reduction_scheme']}")
        self.spec, self.workspace, self.device = spec, workspace, device
        self.weight, self.packed, self._a = weight, packed, a
        self._weight_key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
        self._weight_version = None if weight.is_inference() else weight._version
        self._dtypes = (x.dtype, out_dtype)
        self._alpha, self._beta = ct.c_float(1.0), ct.c_float(0.0)
        self._streams = set()
        self._handle = _handle(device)
        self._op = _Op(spec["layout"], in_name, out_name, m, n, k)
        try:
            check = _Heuristic()
            _check(_lib().cublasLtMatmulAlgoCheck(self._handle, self._op.desc, self._op.a, self._op.b,
                                                  self._op.cd, self._op.cd, ct.byref(self._algo), ct.byref(check)),
                   "cublasLtMatmulAlgoCheck")
            if check.workspaceSize > workspace.nbytes:
                raise ValueError(f"AlgoCheck needs {check.workspaceSize} workspace bytes, have {workspace.nbytes}")
            self.checked_workspace_bytes = int(check.workspaceSize)
        except BaseException:
            self._op.destroy()
            self._op = None
            raise

    @classmethod
    def restore(cls, record, x, w, *, output_dtype, workspace, layout=None):
        """Rebuild a plan from freeze() output; rejects stale or mismatched records."""
        if layout is not None and isinstance(record, dict) and record.get("layout") != layout:
            raise ValueError(f"frozen layout {record.get('layout')!r} is not {layout!r}")
        torch, device, shape, in_name, out_name = _inputs(x, w, output_dtype, workspace)
        validate_frozen(record, context={
            "library_version": _library_version(), "device_uuid": _device_uuid(torch, device),
            "input_dtype": in_name, "output_dtype": out_name, "shape_mnk": list(shape),
            "layout": record.get("layout") if isinstance(record, dict) else None,
        })
        packed = None
        if record["layout"] == "transposed":
            with torch.no_grad():
                packed = w.t().contiguous()
        return cls(record, base64.b64decode(record["algorithm_b64"]), workspace, x=x, weight=w, packed=packed)

    def freeze(self):
        """JSON-serializable record with the exact 64 algorithm bytes."""
        return {**copy.deepcopy(self.spec), "algorithm_b64": base64.b64encode(self._algorithm).decode()}

    def run(self, x, w, out):
        """Enqueue out = x @ w.T on the current stream; capture-safe (no allocation/search)."""
        if self._op is None or self.workspace.closed:
            raise RuntimeError("plan or workspace is closed")
        torch = _torch()
        m, n, k = self.spec["shape_mnk"]
        _check_tensor(torch, x, "x", (m, k), self._dtypes[0], self.device)
        _check_tensor(torch, out, "out", (m, n), self._dtypes[1], self.device)
        if not isinstance(w, torch.Tensor) or (w.data_ptr(), tuple(w.shape), w.dtype) != self._weight_key \
                or w.device != self.weight.device:
            raise ValueError("w is not the weight this plan was prepared for")
        if (self.packed is not None and self._weight_version is not None
                and self.weight._version != self._weight_version):
            raise ValueError("weight was modified after packing; prepare or restore a new plan")
        if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
            raise ValueError("inference only: disable grad or detach inputs")
        if torch.cuda.current_device() != self.device:
            raise ValueError(f"cuda:{self.device} must be the current device")
        need = self.spec["minimum_alignment"]
        for t, required, name in ((self._a, need[0], "A"), (x, need[1], "B"), (out, need[2], "C"), (out, need[3], "D")):
            if _align(t.data_ptr()) < required:
                raise ValueError(f"{name} pointer is not {required}-byte aligned")
        if _overlap(out, x) or _overlap(out, self._a) or _overlap(out, self.weight):
            raise ValueError("out must not alias x or the weight")
        stream = torch.cuda.current_stream(self.device)
        self._record_stream(torch, stream)
        if not torch.cuda.is_current_stream_capturing():
            # Unlike persistent plan weights, these buffers can be temporary.
            x.record_stream(stream)
            out.record_stream(stream)
        op = self._op
        _check(_lib().cublasLtMatmul(self._handle, op.desc, ct.addressof(self._alpha), self._a.data_ptr(), op.a,
                                     x.data_ptr(), op.b, ct.addressof(self._beta), out.data_ptr(), op.cd,
                                     out.data_ptr(), op.cd, ct.byref(self._algo), self.workspace.ptr,
                                     self.workspace.nbytes, stream.cuda_stream),
               "cublasLtMatmul")
        return out

    def _record_stream(self, torch, stream):
        self.workspace.record_stream(stream)
        if stream.cuda_stream in self._streams or torch.cuda.is_current_stream_capturing():
            return
        for t in (self.weight, self.packed):
            if t is not None:
                t.record_stream(stream)
        self._streams.add(stream.cuda_stream)

    def close(self):
        """Synchronize the device, then destroy descriptors.  The workspace stays open."""
        if self._op is not None:
            _torch().cuda.synchronize(self.device)
            self._op.destroy()
            self._op = None


restore_lt_plan = LtPlan.restore


class LtPlanList(list):
    """list[LtPlan] with .search_diagnostics: one dict per searched layout."""

    def __init__(self, plans=(), search_diagnostics=()):
        super().__init__(plans)
        self.search_diagnostics = list(search_diagnostics)


def _finite(value):
    return float(value) if math.isfinite(value) else None


def prepare_lt_plans(x, w, *, output_dtype, workspace, max_candidates=MAX_CANDIDATES):
    """Search both weight layouts and return up to `max_candidates` eligible plans per layout.

    Requests 8 heuristics per layout with the shared workspace size, reduction mask
    COMPUTE_TYPE (NONE still returned) and the actual pointer alignments of A, x and a
    freshly allocated [M,N] output.  No GPU timing.  Every returned or failed candidate
    appears in the result's .search_diagnostics.
    """
    if not _is_int(max_candidates) or not 0 < max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be in [1, {MAX_CANDIDATES}]")
    torch, device, (m, n, k), in_name, out_name = _inputs(x, w, output_dtype, workspace)
    props = torch.cuda.get_device_properties(device)
    handle = _handle(device)
    base = {"format": FORMAT, "library": "cublasLt", "library_version": _library_version(),
            "library_path": _LIB_PATH, "device_uuid": _device_uuid(torch, device), "device_name": props.name,
            "compute_capability": [props.major, props.minor], "input_dtype": in_name, "output_dtype": out_name,
            "shape_mnk": [m, n, k], "compute_type": CUBLAS_COMPUTE_32F, "scale_type": CUDA_R_32F,
            "workspace_capacity_bytes": workspace.nbytes}
    with torch.no_grad():
        packed = w.t().contiguous()
    probe = torch.empty((m, n), dtype=output_dtype, device=x.device)
    plans, diagnostics = [], []
    for layout in LAYOUTS:
        a = w if layout == "original" else packed
        alignment = [min(_align(t.data_ptr()), 256) for t in (a, x, probe, probe)]
        error_text = None
        try:
            op = _Op(layout, in_name, out_name, m, n, k)
            try:
                status, found = _heuristics(op, handle, workspace.nbytes, alignment)
            finally:
                op.destroy()
        except CublasLtError as error:
            status, found, error_text = error.status, [], str(error)
        diag = {"layout": layout, "status": status, "status_name": _status_name(status), "error": error_text,
                "requested": MAX_CANDIDATES, "returned": len(found), "max_candidates": max_candidates,
                "search_alignment": alignment, "reduction_scheme_mask": REDUCTION_COMPUTE_TYPE,
                "selected": [], "rejected": []}
        for idx, result in enumerate(found):
            if len(diag["selected"]) == max_candidates:
                diag["rejected"].append({"candidate_idx": idx, "reason": "beyond max_candidates"})
                continue
            if result.state != CUBLAS_STATUS_SUCCESS:
                diag["rejected"].append({"candidate_idx": idx, "reason": f"heuristic state {_status_name(result.state)}"})
                continue
            info = _describe(result.algo)
            queried = info.pop("algo_minimum_alignment")
            spec = {**base, **info, "layout": layout, "transa": CUBLAS_OP_T if layout == "original" else CUBLAS_OP_N,
                    "candidate_idx": idx, "workspace_bytes": int(result.workspaceSize),
                    "waves_count": _finite(result.wavesCount), "search_alignment": alignment,
                    "minimum_alignment": [q if q else s for q, s in zip(queried, alignment)],
                    "alignment_source": ["algo_cap" if q else "search_preference" for q in queried]}
            try:
                plan = LtPlan(spec, ct.string_at(ct.addressof(result.algo), ct.sizeof(_Algo)), workspace,
                              x=x, weight=w, packed=packed if layout == "transposed" else None)
            except (ValueError, CublasLtError) as error:
                diag["rejected"].append({"candidate_idx": idx, "reason": str(error),
                                         "reduction_scheme": info["reduction_scheme"],
                                         "workspace_bytes": int(result.workspaceSize)})
                continue
            plans.append(plan)
            diag["selected"].append(idx)
        diag["eligible"] = len(diag["selected"])
        diagnostics.append(diag)
    return LtPlanList(plans, diagnostics)
