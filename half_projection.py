"""FP16/BF16-input projection: exact widening, FP32 accumulation, FP32 output.

x[M,K] and w[N,K] share one 16-bit dtype. The supplied values are read as raw
bits and never rounded or converted lossily inside this operation. Separate
from the FP32 contract in kernel_lab. Inference-only experiment.
"""

import ctypes as ct

import kernel_lab
import torch

SUFFIX = {torch.float16: "f16", torch.bfloat16: "bf16"}
THREADS = (32, 64, 128, 256)
VECTORS = (2, 8)


def kernel_names():
    """Kernels in half_kernels.cu that KernelLab must register."""
    return [f"half_warp_{s}_t{t}{'_v8' if v == 8 else ''}"
            for s in ("f16", "bf16") for v in VECTORS for t in THREADS]


def fused_linear(x, w, threads=32, out=None, vector=2):
    lab = kernel_lab.default_lab()
    if not lab.module.value:
        raise RuntimeError("Kernel module is closed")
    for tensor in (x, w):
        if tensor.device != torch.device("cuda", lab.device) or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("Inference-only contiguous tensors on the current CUDA device required")
    if x.dtype not in SUFFIX or w.dtype != x.dtype:
        raise ValueError("x and w must both be FP16 or both BF16")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected x=[M,K], w=[N,K]")
    m, k = x.shape
    n = w.shape[0]
    if min(m, n, k) <= 0 or m > 65535 or max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("Dimensions must be positive, M<=65535, and fit signed 32-bit indexing")
    if threads not in THREADS or vector not in VECTORS:
        raise ValueError("Unsupported launch configuration")
    align = 2 * vector
    if k % vector or x.data_ptr() % align or w.data_ptr() % align:
        raise ValueError(f"Require K divisible by {vector} and {align}-byte aligned inputs")
    name = f"half_warp_{SUFFIX[x.dtype]}_t{threads}" + ("_v8" if vector == 8 else "")
    if name not in lab.functions:
        raise RuntimeError(f"{name} is not registered in KernelLab")
    out = lab.output(out, (m, n), x, w)  # FP32, contiguous, same device, no aliasing.
    rows = threads // 32
    lab.launch(name, ((n + rows - 1) // rows, m), threads, x, w, out, ct.c_int(n), ct.c_int(k))
    return out


@torch.library.custom_op("kernel_lab::half_linear", mutates_args=(), device_types="cuda")
def linear(x: torch.Tensor, w: torch.Tensor, threads: int = 32, vector: int = 2) -> torch.Tensor:
    with torch.cuda.device(x.device):
        return fused_linear(x, w, threads, vector=vector)


@linear.register_fake
def _linear_fake(x, w, threads=32, vector=2):
    if x.dtype not in SUFFIX or w.dtype != x.dtype or x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected same-dtype FP16/BF16 x=[M,K], w=[N,K]")
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)
