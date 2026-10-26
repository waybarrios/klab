"""Exact BF16 block storage (docs/bf16-format.md, version 1) and paired projections.

CPU codec: pack_bits/unpack_bits operate on uint16 bit patterns only; no value
is ever converted to floating point. Work is vectorized over bounded row
chunks. unpack_bits/validate check the canonical layout, exponent range,
tail padding and the source SHA-256 before anything reaches the GPU.

GPU: upload() copies a validated snapshot to one device and preloads that
device's cached CudaModule (compressed_kernels.cu only; never CUDA_SOURCES or
the legacy KernelLab). fused_linear and raw_linear launch the same template, so
compressed and raw outputs of one launch configuration are bit-identical. BF16
inputs, FP32 accumulation, BF16 or FP32 output; inference only. Coded
projections require K % 64 == 0; ragged K still round-trips on CPU and through
unpack_weights. Keep PackedBF16 objects (and the cached module) alive for as
long as any captured graph uses them; there is no __del__ GPU cleanup.

The storage format is an implementation of known ideas; related lossless BF16
formats already exist (see the joint design's prior-work section).
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cuda_module import CudaModule

FORMAT_VERSION = 1
BLOCK = 64
CODED_BYTES = 96
RAW_BYTES = 128
UNIT = 32  # Payload offsets are stored in 32-byte units.
MAX_PAYLOAD_BYTES = 1 << 28  # 23-bit offset field * 32 bytes = 256 MiB.
CHUNK_BLOCKS = 1 << 14  # Blocks per vectorized CPU chunk (about 2 MiB of uint16 input).
OFFSET_MASK = 0x7fffff
RAW_FLAG = 0x80000000

KERNEL_SOURCES = ("compressed_kernels.cu",)
THREADS = (32, 64, 128, 256)
VECTORS = (2, 8)
OUTPUT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
UNPACK_THREADS = 256
IMPLEMENTATION_FILE = Path(__file__).name
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class PackedCPU:
    payload: np.ndarray  # uint8, coded/raw blocks in row-major block order.
    metadata: np.ndarray  # uint32, one word per block.
    n: int
    k: int
    source_sha256: str  # SHA-256 of the original little-endian uint16 bytes.
    format_version: int = FORMAT_VERSION

    @property
    def storage_bytes(self):
        return int(self.payload.nbytes + self.metadata.nbytes)


# ---------------------------------------------------------------- CPU codec


def _row_chunks(n, blocks_per_row):
    rows = max(1, CHUNK_BLOCKS // blocks_per_row)
    for start in range(0, n, rows):
        yield start, min(n, start + rows)


def _blocks(chunk, k, padded_k):
    """[rows, K] uint16 -> [rows * K/64, 64], repeating each row's last weight."""
    if padded_k != k:
        chunk = np.pad(chunk, ((0, 0), (0, padded_k - k)), mode="edge")
    return chunk.reshape(-1, BLOCK)


def _codes(blocks, bases):
    sign_mantissa = (blocks & 0x7f) | ((blocks >> 8) & 0x80)
    delta = ((blocks >> 7) & 0xff) - bases[:, None].astype(np.uint16)
    return sign_mantissa | (delta << 8)


def _pair_bytes(codes):
    first, second = codes[:, 0::2], codes[:, 1::2]
    out = np.empty((len(codes), BLOCK // 2, 3), dtype=np.uint8)
    out[..., 0] = first & 0xff
    out[..., 1] = ((first >> 8) & 0xf) | ((second & 0xf) << 4)
    out[..., 2] = second >> 4
    return out.reshape(len(codes), CODED_BYTES)


def pack_bits(bits: np.ndarray) -> PackedCPU:
    """Pack a row-major [N,K] uint16 BF16 bit matrix. Every bit is preserved."""
    if not isinstance(bits, np.ndarray) or bits.dtype.kind != "u" or bits.dtype.itemsize != 2:
        raise ValueError("Expected a numpy uint16 array of BF16 bit patterns")
    if bits.ndim != 2:
        raise ValueError("Expected a two-dimensional [N,K] array")
    n, k = (int(size) for size in bits.shape)
    if min(n, k) <= 0 or n * k >= 2**31:
        raise ValueError("N and K must be positive with N*K < 2^31")
    blocks_per_row = (k + BLOCK - 1) // BLOCK
    padded_k = blocks_per_row * BLOCK
    count = n * blocks_per_row

    # Pass 1: source digest, block bases and raw flags (small per-block arrays).
    digest = hashlib.sha256()
    raw = np.empty(count, dtype=bool)
    bases = np.empty(count, dtype=np.uint32)
    for start, stop in _row_chunks(n, blocks_per_row):
        chunk = np.ascontiguousarray(bits[start:stop], dtype="<u2")
        digest.update(chunk)
        exponents = (_blocks(chunk, k, padded_k) >> 7) & 0xff
        low, high = exponents.min(axis=1), exponents.max(axis=1)
        selected = slice(start * blocks_per_row, stop * blocks_per_row)
        raw[selected] = (high - low) > 15
        bases[selected] = low
    bases[raw] = 0
    units = np.where(raw, RAW_BYTES // UNIT, CODED_BYTES // UNIT).astype(np.int64)
    ends = np.cumsum(units)
    total = int(ends[-1]) * UNIT
    if total > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Payload of {total} bytes exceeds the {MAX_PAYLOAD_BYTES}-byte offset range; "
                         "pack smaller tensors separately")
    starts = ends - units
    metadata = (starts.astype(np.uint32) | (bases << np.uint32(23))
                | (raw.astype(np.uint32) << np.uint32(31))).astype(np.uint32)

    # Pass 2: write each chunk's blocks into their 32-byte payload units.
    payload = np.empty(total, dtype=np.uint8)
    payload_units = payload.reshape(-1, UNIT)
    for start, stop in _row_chunks(n, blocks_per_row):
        blocks = _blocks(np.ascontiguousarray(bits[start:stop], dtype="<u2"), k, padded_k)
        selected = slice(start * blocks_per_row, stop * blocks_per_row)
        chunk_raw, chunk_starts = raw[selected], starts[selected]
        coded = ~chunk_raw
        if coded.any():
            data = _pair_bytes(_codes(blocks[coded], bases[selected][coded]))
            index = chunk_starts[coded, None] + np.arange(CODED_BYTES // UNIT)
            payload_units[index] = data.reshape(-1, CODED_BYTES // UNIT, UNIT)
        if chunk_raw.any():
            data = np.ascontiguousarray(blocks[chunk_raw], dtype="<u2").view(np.uint8)
            index = chunk_starts[chunk_raw, None] + np.arange(RAW_BYTES // UNIT)
            payload_units[index] = data.reshape(-1, RAW_BYTES // UNIT, UNIT)
    return PackedCPU(payload, metadata, n, k, digest.hexdigest(), FORMAT_VERSION)


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _layout(packed):
    """Validate types and the canonical sequential block layout."""
    if getattr(packed, "format_version", None) != FORMAT_VERSION:
        raise ValueError(f"Unsupported packed format version; expected {FORMAT_VERSION}")
    n, k = _dimension(packed.n, "n"), _dimension(packed.k, "k")
    if min(n, k) <= 0 or n * k >= 2**31:
        raise ValueError("N and K must be positive with N*K < 2^31")
    digest = packed.source_sha256
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
    payload, metadata = packed.payload, packed.metadata
    if not isinstance(payload, np.ndarray) or payload.dtype != np.uint8 or payload.ndim != 1:
        raise ValueError("Payload must be a one-dimensional numpy uint8 array")
    if not isinstance(metadata, np.ndarray) or metadata.dtype != np.dtype("<u4") or metadata.ndim != 1:
        raise ValueError("Metadata must be a one-dimensional little-endian uint32 array")
    blocks_per_row = (k + BLOCK - 1) // BLOCK
    if metadata.size != n * blocks_per_row:
        raise ValueError("Metadata must contain one word per 64-weight block")
    if payload.size > MAX_PAYLOAD_BYTES:
        raise ValueError("Payload exceeds the version-1 offset range")
    raw = (metadata & np.uint32(RAW_FLAG)) != 0
    bases = (metadata >> np.uint32(23)) & np.uint32(0xff)
    if np.any(raw & (bases != 0)):
        raise ValueError("Raw blocks must have base exponent zero")
    units = np.where(raw, RAW_BYTES // UNIT, CODED_BYTES // UNIT).astype(np.int64)
    ends = np.cumsum(units)
    starts = ends - units
    if not np.array_equal((metadata & np.uint32(OFFSET_MASK)).astype(np.int64), starts):
        raise ValueError("Block offsets are not the canonical sequential layout")
    if int(ends[-1]) * UNIT != payload.size:
        raise ValueError("Payload length does not match its block metadata")
    return n, k, blocks_per_row, raw, bases, starts


def _decoded_chunks(packed):
    """Yield (start, stop, rows) of decoded uint16 bits, validating as it goes."""
    n, k, blocks_per_row, raw, bases, starts = _layout(packed)
    padded_k = blocks_per_row * BLOCK
    payload_units = np.ascontiguousarray(packed.payload).reshape(-1, UNIT)
    for start, stop in _row_chunks(n, blocks_per_row):
        selected = slice(start * blocks_per_row, stop * blocks_per_row)
        chunk_raw, chunk_starts = raw[selected], starts[selected]
        coded = ~chunk_raw
        blocks = np.empty((len(chunk_raw), BLOCK), dtype=np.uint16)
        if coded.any():
            index = chunk_starts[coded, None] + np.arange(CODED_BYTES // UNIT)
            data = payload_units[index].reshape(-1, BLOCK // 2, 3).astype(np.uint16)
            first = data[..., 0] | ((data[..., 1] & 0xf) << 8)
            second = (data[..., 1] >> 4) | (data[..., 2] << 4)
            codes = np.stack((first, second), axis=-1).reshape(-1, BLOCK).astype(np.uint32)
            exponents = bases[selected][coded, None] + (codes >> 8)
            if np.any(exponents > 0xff):
                raise ValueError("Decoded exponent overflows BF16")
            blocks[coded] = ((codes & 0x7f) | ((codes & 0x80) << 8) | (exponents << 7)).astype(np.uint16)
        if chunk_raw.any():
            index = chunk_starts[chunk_raw, None] + np.arange(RAW_BYTES // UNIT)
            blocks[chunk_raw] = payload_units[index].reshape(-1, RAW_BYTES).view("<u2")
        rows = blocks.reshape(stop - start, padded_k)
        if padded_k != k and np.any(rows[:, k:] != rows[:, k - 1:k]):
            raise ValueError("Tail padding must repeat each row's last weight")
        yield start, stop, rows[:, :k]


def validate(packed: PackedCPU) -> None:
    """Fully decode in bounded chunks and check layout, exponents and source digest."""
    digest = hashlib.sha256()
    for _, _, rows in _decoded_chunks(packed):
        digest.update(np.ascontiguousarray(rows, dtype="<u2"))
    if digest.hexdigest() != packed.source_sha256:
        raise ValueError("Decoded bits do not match source_sha256")


def unpack_bits(packed: PackedCPU) -> np.ndarray:
    """Decode to the original [N,K] uint16 bits, after full validation."""
    out = None
    digest = hashlib.sha256()
    for start, stop, rows in _decoded_chunks(packed):
        if out is None:
            out = np.empty((int(packed.n), int(packed.k)), dtype=np.uint16)
        out[start:stop] = rows
        digest.update(np.ascontiguousarray(rows, dtype="<u2"))
    if digest.hexdigest() != packed.source_sha256:
        raise ValueError("Decoded bits do not match source_sha256")
    return out


# ---------------------------------------------------------------- GPU module

_MODULES = {}
_MODULES_LOCK = threading.Lock()


def kernel_names():
    """Every kernel in compressed_kernels.cu; the cached module loads all of them."""
    names = [f"cbf16_{kind}_{output}_t{threads}_v{vector}"
             for kind in ("raw", "coded") for output in OUTPUT_DTYPES
             for vector in VECTORS for threads in THREADS]
    return tuple(names + ["cbf16_unpack"])


def _device_index(device):
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, bool):
        raise ValueError("device must be a CUDA device")
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    return torch.cuda.current_device() if device.index is None else device.index


def get_module(device=None) -> CudaModule:
    """This device's cached module. Load it outside capture; never close it while
    graphs or PackedBF16 objects that use it remain alive."""
    index = _device_index(device)
    with _MODULES_LOCK:
        module = _MODULES.get(index)
        if module is not None and not module.closed:
            return module
        if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is not available")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Load the compressed module (upload/get_module) before CUDA graph capture")
        if torch.cuda.get_device_capability(index) < (8, 0):
            raise RuntimeError("Compressed BF16 kernels require SM80+ (cvt.rn.bf16.f32)")
        module = CudaModule(KERNEL_SOURCES, kernel_names(), index)
        _MODULES[index] = module
        return module


def _fingerprint(module):
    fingerprint, compilation = module.fingerprint, module.compilation
    identity = {
        "format_version": FORMAT_VERSION,
        "implementation": IMPLEMENTATION_FILE,
        "implementation_sha256": _IMPLEMENTATION_SHA256,
        "kernel_names": list(kernel_names()),
        "module_fingerprint": fingerprint,
        "ptx_sha256": compilation["ptx_sha256"],
        "compiler": compilation["compiler"],
        "architecture": compilation["architecture"],
        "options": compilation["options"],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {**identity, "fingerprint_sha256": hashlib.sha256(text.encode()).hexdigest()}


def implementation_fingerprint(device=None):
    """Packer/wrapper SHA-256 plus module fingerprint and PTX hash. CudaModule's
    own fingerprint covers only its sources and shared compiler/bridge files."""
    return _fingerprint(get_module(device))


@dataclass(frozen=True, eq=False)
class PackedBF16:
    payload: torch.Tensor  # uint8 [bytes]
    metadata: torch.Tensor  # int32 [blocks], raw uint32 bits.
    n: int
    k: int
    source_sha256: str
    format_version: int
    module: CudaModule  # Retained for graph lifetime and fingerprinting.

    @property
    def storage_bytes(self):
        return int(self.payload.numel() + 4 * self.metadata.numel())

    @property
    def implementation_fingerprint(self):
        return _fingerprint(self.module)

    def linear(self, x, *, threads, vector, output_dtype):
        """Registered custom op compressed_bf16::compressed_linear (torch.compile-safe)."""
        return compressed_linear(x, self.payload, self.metadata, self.n, self.k,
                                 threads, vector, _output_name(output_dtype))

    def unpack(self):
        return unpack_weights(self)


def upload(packed: PackedCPU, device) -> PackedBF16:
    """Validate a private snapshot of `packed`, preload the module, then copy to `device`."""
    index = _device_index(device)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("upload must run outside CUDA graph capture")
    _layout(packed)  # Type checks before copying, so no silent dtype conversion.
    snapshot = PackedCPU(packed.payload.copy(), packed.metadata.copy(), int(packed.n), int(packed.k),
                         packed.source_sha256, packed.format_version)
    validate(snapshot)
    module = get_module(index)
    target = torch.device("cuda", index)
    payload = torch.from_numpy(snapshot.payload).to(target)  # Blocking copies from pageable memory.
    metadata = torch.from_numpy(snapshot.metadata.view(np.int32)).to(target)
    return PackedBF16(payload, metadata, snapshot.n, snapshot.k, snapshot.source_sha256,
                      FORMAT_VERSION, module)


# ---------------------------------------------------------------- projections


def _output_name(value):
    if isinstance(value, str):
        if value in OUTPUT_DTYPES:
            return value
    else:
        for name, dtype in OUTPUT_DTYPES.items():
            if value is dtype:
                return name
    raise ValueError("output_dtype must be torch.bfloat16/'bf16' or torch.float32/'fp32'")


def _check_tensor(tensor, label, dtype, device, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{label} must be a tensor")
    if tensor.device != device or tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D {dtype} tensor on {device}")
    if not tensor.is_contiguous() or tensor.requires_grad:
        raise ValueError(f"{label} must be contiguous and inference-only (no requires_grad)")


def _overlaps(a, b):
    if a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr():
        return True
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


def _prepare(x, n, k, threads, vector, output_dtype, out, inputs, aligned=()):
    """Shared validation; returns (output name, device, output tensor). `inputs`
    may not overlap `out`; `aligned` tensors need the same alignment as x."""
    name = _output_name(output_dtype)
    if isinstance(threads, bool) or threads not in THREADS or isinstance(vector, bool) or vector not in VECTORS:
        raise ValueError(f"Unsupported launch configuration; threads in {THREADS}, vector in {VECTORS}")
    if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    device = x.device
    _check_tensor(x, "x", torch.bfloat16, device, 2)
    m = x.shape[0]
    if x.shape[1] != k:
        raise ValueError("Expected x=[M,K] matching the weights")
    if min(m, n, k) <= 0 or m > 65535 or max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("Dimensions must be positive, M<=65535, and fit signed 32-bit indexing")
    align = 2 * vector
    if k % vector or any(tensor.data_ptr() % align for tensor in (x, *aligned)):
        raise ValueError(f"Require K divisible by {vector} and {align}-byte aligned inputs")
    dtype = OUTPUT_DTYPES[name]
    if out is None:
        out = torch.empty((m, n), dtype=dtype, device=device)
    else:
        _check_tensor(out, "out", dtype, device, 2)
        if tuple(out.shape) != (m, n):
            raise ValueError("Output shape mismatch")
        if any(_overlaps(out, tensor) for tensor in (x, *inputs)):
            raise ValueError("Output must not share storage with x or the weights")
    return name, device, out


def _coded(x, payload, metadata, n, k, threads, vector, output_dtype, out=None):
    n, k = _dimension(n, "n"), _dimension(k, "k")
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a CUDA tensor")
    _check_tensor(payload, "payload", torch.uint8, x.device, 1)
    _check_tensor(metadata, "metadata", torch.int32, x.device, 1)
    if k <= 0 or k % BLOCK:
        raise ValueError("Fused compressed projections require K divisible by 64; "
                         "use unpack_weights for ragged K")
    blocks = n * (k // BLOCK)
    if metadata.numel() != blocks or not (CODED_BYTES * blocks <= payload.numel() <= RAW_BYTES * blocks):
        raise ValueError("Packed buffers do not match N,K")
    if payload.data_ptr() % UNIT or metadata.data_ptr() % 4:
        raise ValueError("Packed buffers must keep their 32-byte payload alignment")
    name, device, out = _prepare(x, n, k, threads, vector, output_dtype, out, (payload, metadata))
    get_module(device).launch(f"cbf16_coded_{name}_t{threads}_v{vector}",
                              ((n + threads // 32 - 1) // (threads // 32), x.shape[0]), threads,
                              x, payload, metadata, out, ct.c_int(n), ct.c_int(k))
    return out


def fused_linear(x, packed: PackedBF16, *, threads, vector, output_dtype, out=None):
    """out[M,N] = x[M,K] @ W.T, decoding exact weights in registers."""
    if not isinstance(packed, PackedBF16):
        raise ValueError("packed must come from upload()")
    return _coded(x, packed.payload, packed.metadata, packed.n, packed.k,
                  threads, vector, output_dtype, out)


def raw_linear(x, w, *, threads, vector, output_dtype, out=None):
    """Same template as fused_linear over an uncompressed BF16 w[N,K]."""
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a CUDA tensor")
    _check_tensor(w, "w", torch.bfloat16, x.device, 2)
    n, k = w.shape
    name, device, out = _prepare(x, n, k, threads, vector, output_dtype, out, (w,), aligned=(w,))
    get_module(device).launch(f"cbf16_raw_{name}_t{threads}_v{vector}",
                              ((n + threads // 32 - 1) // (threads // 32), x.shape[0]), threads,
                              x, w, out, ct.c_int(n), ct.c_int(k))
    return out


def unpack_weights(packed: PackedBF16) -> torch.Tensor:
    """Integer-only GPU decode to a BF16 [N,K] tensor, for bitwise validation."""
    if not isinstance(packed, PackedBF16):
        raise ValueError("packed must come from upload()")
    device = packed.payload.device
    out = torch.empty((packed.n, packed.k), dtype=torch.bfloat16, device=device)
    count = packed.n * packed.k
    get_module(device).launch("cbf16_unpack", ((count + UNPACK_THREADS - 1) // UNPACK_THREADS, 1),
                              UNPACK_THREADS, packed.payload, packed.metadata, out,
                              ct.c_int(packed.n), ct.c_int(packed.k))
    return out


@torch.library.custom_op("compressed_bf16::compressed_linear", mutates_args=(), device_types="cuda")
def compressed_linear(x: torch.Tensor, payload: torch.Tensor, metadata: torch.Tensor, n: int, k: int,
                      threads: int, vector: int, output_dtype: str) -> torch.Tensor:
    with torch.cuda.device(x.device):
        return _coded(x, payload, metadata, n, k, threads, vector, output_dtype)


@compressed_linear.register_fake
def _compressed_linear_fake(x, payload, metadata, n, k, threads, vector, output_dtype):
    dtype = OUTPUT_DTYPES[_output_name(output_dtype)]
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or payload.dtype != torch.uint8
            or metadata.dtype != torch.int32 or payload.ndim != 1 or metadata.ndim != 1):
        raise ValueError("Expected BF16 x=[M,K], uint8 payload and int32 metadata")
    if x.shape[1] != k or k % BLOCK or metadata.shape[0] != n * (k // BLOCK):
        raise ValueError("Packed buffers do not match x or N,K")
    return x.new_empty((x.shape[0], n), dtype=dtype)
