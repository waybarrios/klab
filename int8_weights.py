"""Conservative bandwidth experiment: signed INT8 weights, FP32 accumulation."""

import ctypes as ct
from dataclasses import dataclass

import kernel_lab
import torch


@dataclass(frozen=True)
class Int8Weights:
    codes: torch.Tensor
    scales: torch.Tensor

    def dequantize(self):
        n, k = self.codes.shape
        return (self.codes.float().reshape(n, k // 64, 64) * self.scales.unsqueeze(-1)).reshape(n, k)

    def linear(self, x, threads=128):
        return int8_linear(x, self.codes, self.scales, threads)


def quantize_weights(w):
    if w.device.type != "cuda" or w.dtype != torch.float32 or w.ndim != 2 or not w.is_contiguous():
        raise ValueError("Expected contiguous CUDA FP32 weights [N,K]")
    if w.requires_grad or min(w.shape) <= 0 or w.shape[1] % 64:
        raise ValueError("Inference only; positive N and K divisible by 64 required")
    if not torch.isfinite(w).all():
        raise ValueError("Finite weights required")
    n, k = w.shape
    blocks = w.reshape(n, k // 64, 64)
    amax = blocks.abs().amax(-1)
    scales = torch.where(amax > 0, (amax / 127).clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(amax))
    codes = (blocks / scales.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(n, k)
    return Int8Weights(codes.contiguous(), scales.contiguous())


def fused_linear(x, codes, scales, threads=128, out=None):
    lab = kernel_lab.default_lab()
    for tensor in (x, codes, scales):
        if tensor.device != torch.device("cuda", lab.device) or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("Inference-only contiguous tensors on the current CUDA device required")
    if x.dtype not in (torch.float16, torch.float32) or x.ndim != 2:
        raise ValueError("Expected FP16/FP32 activations [M,K]")
    if codes.dtype != torch.int8 or codes.ndim != 2 or scales.dtype != torch.float32:
        raise ValueError("Expected INT8 codes and FP32 scales")
    m, k = x.shape
    n = codes.shape[0]
    if min(m, n, k) <= 0 or k % 64 or codes.shape[1] != k or scales.shape != (n, k // 64):
        raise ValueError("Invalid 64-value group dimensions")
    if max(m * k, n * k, m * n) >= 2**31 or codes.data_ptr() % 4:
        raise ValueError("Require 32-bit indices and aligned codes")
    if threads not in (32, 64, 128, 256):
        raise ValueError("Unsupported block size")
    if out is None:
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    else:
        lab.validate(out)
        if out.shape != (m, n):
            raise ValueError("Output shape mismatch")
    if any(out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in (x, codes, scales)):
        raise ValueError("Output may not alias inputs")
    bits = 16 if x.dtype == torch.float16 else 32
    rows = threads // 32
    lab.launch(f"int8_a{bits}_t{threads}", ((n + rows - 1) // rows, m), threads,
               x, codes, scales, out, ct.c_int(n), ct.c_int(k))
    return out


@torch.library.custom_op("kernel_lab::int8_linear", mutates_args=(), device_types="cuda")
def int8_linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, threads: int = 128) -> torch.Tensor:
    return fused_linear(x, codes, scales, threads)


@int8_linear.register_fake
def _int8_fake(x, codes, scales, threads=128):
    return torch.empty((x.shape[0], codes.shape[0]), device=x.device, dtype=torch.float32)
