"""Revision-2 deterministic M tiling for the exact BF16 projections (CUDA performance work).

Importing this file is CPU-only: torch, compressed_bf16 and cuda_module load
when a GPU function is called. tile_for_m fixes the token tile from M alone
(min(16, smallest power of two >= M)); it is not a search dimension. Launch
configurations are the same eight (threads, vector) pairs as compressed_bf16.

M == 1 delegates exactly to compressed_bf16.fused_linear/raw_linear. M > 1
reuses compressed_bf16's validation (_check_tensor/_prepare plus the coded
buffer checks copied below) and launches mtile_kernels.cu, which reuses every
weight vector across the tile's tokens while keeping each token's FMA order,
reduction tree and rounding identical to R1. PackedBF16 and its module are
unchanged and unused for M-tile launches.

get_module compiles compressed_kernels.cu followed by mtile_kernels.cu as one
NVRTC source and loads only the mtbf16_* kernels. Load it (and compressed_bf16's
module, used for M == 1) outside CUDA graph capture, and keep both alive for as
long as any launch or captured graph uses them.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import numbers
import threading
from pathlib import Path

REVISION = 2
KERNEL_SOURCES = ("compressed_kernels.cu", "mtile_kernels.cu")
THREADS = (32, 64, 128, 256)
VECTORS = (2, 8)
OUTPUTS = ("bf16", "fp32")
TILES = (2, 4, 8, 16)
MAX_M = 65535
TILE_RULE = "M == 1 delegates to compressed_bf16; else min(16, smallest power of two >= M)"
IMPLEMENTATION_FILE = Path(__file__).name
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def tile_for_m(m):
    """Deterministic token tile: 1 for M == 1, else min(16, next power of two >= M)."""
    if isinstance(m, bool) or not isinstance(m, numbers.Integral):
        raise ValueError("m must be an integer")
    m = int(m)
    if not 1 <= m <= MAX_M:
        raise ValueError(f"m must be in [1, {MAX_M}]")
    tile = 1
    while tile < m and tile < TILES[-1]:
        tile *= 2
    return tile


def kernel_names():
    """Every M-tile kernel in mtile_kernels.cu; the cached module loads only these."""
    return tuple(f"mtbf16_{kind}_{output}_t{threads}_v{vector}_m{tile}"
                 for kind in ("raw", "coded") for output in OUTPUTS
                 for vector in VECTORS for threads in THREADS for tile in TILES)


# ---------------------------------------------------------------- GPU module

_MODULES = {}
_MODULES_LOCK = threading.Lock()


def get_module(device=None):
    """This device's cached M-tile module. Load it outside capture; never close it
    while graphs that use it remain alive."""
    import torch

    import compressed_bf16 as cb
    from cuda_module import CudaModule

    index = cb._device_index(device)
    with _MODULES_LOCK:
        module = _MODULES.get(index)
        if module is not None and not module.closed:
            return module
        if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is not available")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Load the M-tile module (get_module) before CUDA graph capture")
        if torch.cuda.get_device_capability(index) < (8, 0):
            raise RuntimeError("M-tile BF16 kernels require SM80+ (cvt.rn.bf16.f32)")
        module = CudaModule(KERNEL_SOURCES, kernel_names(), index)
        _MODULES[index] = module
        return module


def implementation_fingerprint(device=None):
    """Wrapper SHA-256, M-tile module fingerprint/PTX/options/Torch CUDA, and
    compressed_bf16's fingerprint (codec plus the M == 1 path) as a dependency."""
    import torch

    import compressed_bf16 as cb

    module = get_module(device)
    fingerprint, compilation = module.fingerprint, module.compilation
    identity = {
        "revision": REVISION,
        "implementation": IMPLEMENTATION_FILE,
        "implementation_sha256": _IMPLEMENTATION_SHA256,
        "kernel_sources": list(KERNEL_SOURCES),
        "kernel_names": list(kernel_names()),
        "tile_rule": TILE_RULE,
        "module_fingerprint": fingerprint,
        "ptx_sha256": compilation["ptx_sha256"],
        "compiler": compilation["compiler"],
        "architecture": compilation["architecture"],
        "options": compilation["options"],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "compressed_bf16": cb.implementation_fingerprint(device),
    }
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {**identity, "fingerprint_sha256": hashlib.sha256(text.encode()).hexdigest()}


# ---------------------------------------------------------------- projections


def _single_token(x):
    import torch

    return isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == 1


def _prepare_coded(cb, x, payload, metadata, n, k):
    """compressed_bf16._coded's buffer validation, without its launch."""
    import torch

    n, k = cb._dimension(n, "n"), cb._dimension(k, "k")
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a CUDA tensor")
    cb._check_tensor(payload, "payload", torch.uint8, x.device, 1)
    cb._check_tensor(metadata, "metadata", torch.int32, x.device, 1)
    if k <= 0 or k % cb.BLOCK:
        raise ValueError("Fused compressed projections require K divisible by 64; "
                         "use unpack_weights for ragged K")
    blocks = n * (k // cb.BLOCK)
    if metadata.numel() != blocks or not (cb.CODED_BYTES * blocks <= payload.numel() <= cb.RAW_BYTES * blocks):
        raise ValueError("Packed buffers do not match N,K")
    if payload.data_ptr() % cb.UNIT or metadata.data_ptr() % 4:
        raise ValueError("Packed buffers must keep their 32-byte payload alignment")
    return n, k


def _launch(prefix, device, x, n, k, threads, vector, tensors):
    m = x.shape[0]
    tile = tile_for_m(m)
    if tile not in TILES:
        raise ValueError("M-tile kernels require M > 1")
    rows = threads // 32
    get_module(device).launch(f"{prefix}_t{threads}_v{vector}_m{tile}",
                              ((n + rows - 1) // rows, (m + tile - 1) // tile), threads,
                              *tensors, ct.c_int(n), ct.c_int(k), ct.c_int(m))


def fused_linear(x, packed, *, threads, vector, output_dtype, out=None):
    """out[M,N] = x[M,K] @ W.T over a compressed_bf16.PackedBF16, tiled over M."""
    import compressed_bf16 as cb

    if not isinstance(packed, cb.PackedBF16):
        raise ValueError("packed must come from compressed_bf16.upload()")
    if _single_token(x):
        return cb.fused_linear(x, packed, threads=threads, vector=vector,
                               output_dtype=output_dtype, out=out)
    payload, metadata = packed.payload, packed.metadata
    n, k = _prepare_coded(cb, x, payload, metadata, packed.n, packed.k)
    name, device, out = cb._prepare(x, n, k, threads, vector, output_dtype, out, (payload, metadata))
    _launch(f"mtbf16_coded_{name}", device, x, n, k, threads, vector, (x, payload, metadata, out))
    return out


def raw_linear(x, w, *, threads, vector, output_dtype, out=None):
    """Same tiling as fused_linear over an uncompressed BF16 w[N,K]."""
    import torch

    import compressed_bf16 as cb

    if _single_token(x):
        return cb.raw_linear(x, w, threads=threads, vector=vector, output_dtype=output_dtype, out=out)
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a CUDA tensor")
    cb._check_tensor(w, "w", torch.bfloat16, x.device, 2)
    n, k = w.shape
    name, device, out = cb._prepare(x, n, k, threads, vector, output_dtype, out, (w,), aligned=(w,))
    _launch(f"mtbf16_raw_{name}", device, x, n, k, threads, vector, (x, w, out))
    return out
