"""Experimental software 4-bit weights with FP32 accumulation on Ampere.

E2M1 elements, positive E4M3 scale per 16 values, and a global FP32 scale.
Custom linear row-major packing, not a Transformer Engine checkpoint ABI.
Only the weights are quantized. Quantization is an explicit accuracy tradeoff.
"""

import ctypes as ct
from dataclasses import dataclass

import kernel_lab
import torch


@dataclass(frozen=True)
class FP4Weights:
    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor

    def dequantize(self):
        n, half_k = self.packed.shape
        codes = torch.stack((self.packed & 15, self.packed >> 4), dim=-1).reshape(n, half_k * 2)
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=self.packed.device)
        values = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
        scale = self.scales.view(torch.float8_e4m3fn).float() * self.global_scale
        return (values.reshape(n, -1, 16) * scale.unsqueeze(-1)).reshape(n, -1)

    def linear(self, x, threads=128):
        return fp4_linear(x, self.packed, self.scales, self.global_scale, threads)


def quantize_weights(w):
    if w.device.type != "cuda" or w.dtype != torch.float32 or w.ndim != 2 or not w.is_contiguous():
        raise ValueError("Weights must be contiguous CUDA FP32 [N,K]")
    if w.requires_grad or min(w.shape) <= 0 or w.shape[1] % 16:
        raise ValueError("Inference only; positive N and K divisible by 16 required")
    if not torch.isfinite(w).all():
        raise ValueError("Quantization requires finite weights")
    n, k = w.shape
    blocks = w.reshape(n, k // 16, 16)
    amax = w.abs().amax()
    global_scale = torch.where(
        amax > 0,
        (amax / (448.0 * 6.0)).clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(amax),
    )
    block_scale = (blocks.abs().amax(-1) / (6.0 * global_scale)).clamp(max=448)
    scales = block_scale.to(torch.float8_e4m3fn)
    decoded_scale = scales.float() * global_scale
    normalized = blocks / decoded_scale.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=w.device)
    codes = torch.bucketize(magnitude, boundaries).to(torch.uint8)
    # Correct the lower-bin bucket rule at exact round-to-nearest-even ties.
    for lower, boundary in ((1, .75), (3, 1.75), (5, 3.5)):
        codes = torch.where(magnitude == boundary, lower + 1, codes)
    codes = (codes | (torch.signbit(normalized).to(torch.uint8) << 3)).reshape(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return FP4Weights(packed, scales.view(torch.uint8).contiguous(), global_scale.contiguous())


def fused_linear(x, packed, scales, global_scale, threads=128, out=None):
    lab = kernel_lab.default_lab()
    tensors = (x, packed, scales, global_scale)
    for tensor in tensors:
        if tensor.device != torch.device("cuda", lab.device) or not tensor.is_contiguous():
            raise ValueError("All tensors must be contiguous on the current CUDA device")
        if tensor.requires_grad:
            raise ValueError("Inference only")
    if x.dtype not in (torch.float16, torch.float32) or x.ndim != 2:
        raise ValueError("Expected FP16 or FP32 activations [M,K]")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("Packed weights and FP8 scale bytes must be uint8")
    m, k = x.shape
    n = packed.shape[0]
    if min(m, n, k) <= 0 or k % 16 or packed.shape[1] != k // 2 or scales.shape != (n, k // 16):
        raise ValueError("Invalid 16-value block dimensions")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("Dimensions exceed signed 32-bit indexing")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("Expected one FP32 global scale")
    if packed.data_ptr() % 4 or threads not in (32, 64, 128, 256):
        raise ValueError("Packed weights require 4-byte alignment and supported block size")
    if out is None:
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    else:
        lab.validate(out)
        if out.shape != (m, n):
            raise ValueError("Output shape mismatch")
    if any(out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in tensors):
        raise ValueError("Output may not share input storage")
    activation_bits = 16 if x.dtype == torch.float16 else 32
    rows = threads // 32
    lab.launch(f"fp4_a{activation_bits}_t{threads}", ((n + rows - 1) // rows, m), threads,
               x, packed, scales, global_scale, out, ct.c_int(n), ct.c_int(k))
    return out


@torch.library.custom_op("kernel_lab::fp4_linear", mutates_args=(), device_types="cuda")
def fp4_linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               global_scale: torch.Tensor, threads: int = 128) -> torch.Tensor:
    return fused_linear(x, packed, scales, global_scale, threads)


@fp4_linear.register_fake
def _fp4_fake(x, packed, scales, global_scale, threads=128):
    return torch.empty((x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32)
