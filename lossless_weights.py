"""Lossless FP32 block encoding. No mantissa truncation or numeric conversion."""

import ctypes as ct
from dataclasses import dataclass

import kernel_lab
import torch


@dataclass(frozen=True)
class LosslessWeights:
    mantissa: torch.Tensor
    exponents: torch.Tensor
    metadata: torch.Tensor

    @property
    def shape(self):
        return self.mantissa.shape[0], self.mantissa.shape[1] // 3

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.mantissa, self.exponents, self.metadata))

    def unpack(self):
        n, k = self.shape
        out = torch.empty(n, k, device=self.mantissa.device, dtype=torch.float32)
        lab = kernel_lab.default_lab()
        lab.launch("lossless_unpack", (((n * k // 4) + 255) // 256, 1), 256,
                   self.mantissa, self.exponents, self.metadata,
                   out, ct.c_int(n * k))
        return out

    def linear(self, x, threads=128):
        return lossless_linear(x, self.mantissa, self.exponents, self.metadata, threads)


def pack_weights(w):
    if w.device.type != "cuda" or w.dtype != torch.float32 or w.ndim != 2 or not w.is_contiguous():
        raise ValueError("Expected contiguous CUDA FP32 [N,K]")
    if w.requires_grad or min(w.shape) <= 0 or w.shape[1] % 64 or w.numel() * 3 >= 2**31:
        raise ValueError("Inference-only, K divisible by 64, positive 32-bit-addressable storage required")
    n, k = w.shape
    bits = w.view(torch.int32)
    significand = (bits & 0x7fffff) | ((bits >> 8) & 0x800000)
    mantissa = torch.stack((significand & 255, (significand >> 8) & 255,
                            (significand >> 16) & 255), dim=-1).to(torch.uint8).reshape(n, k * 3)
    exps = ((bits >> 23) & 255).to(torch.uint8).reshape(-1, 64)
    base = exps.amin(-1)
    delta = exps - base[:, None]
    wide = delta.amax(-1) > 15
    sizes = torch.where(wide, 64, 32).to(torch.int32)
    offsets = sizes.cumsum(0) - sizes
    total = int(sizes.sum().item())  # Setup-only allocation sizing.
    if total >= 2**28:
        raise ValueError("Exponent payload exceeds this experimental metadata format's 256 MiB limit")
    exponent_bytes = torch.empty(total, dtype=torch.uint8, device=w.device)
    packed_delta = delta[:, ::2] | (delta[:, 1::2] << 4)
    padded = torch.cat((packed_delta, torch.zeros_like(packed_delta)), dim=-1)
    source = torch.where(wide[:, None], exps, padded)
    columns = torch.arange(64, device=w.device)
    valid = columns[None, :] < sizes[:, None]
    positions = offsets[:, None] + columns[None, :]
    exponent_bytes.scatter_(0, positions[valid], source[valid])
    # Offsets are multiples of 32. Use those low zero bits to fit offset,
    # exponent base, and fallback flag into a single 32-bit metadata load.
    metadata = ((offsets >> 5) << 9) | (base.to(torch.int64) << 1) | wide.to(torch.int64)
    return LosslessWeights(mantissa.contiguous(), exponent_bytes,
                           metadata.to(torch.int32).reshape(n, k // 64).contiguous())


def fused_linear(x, packed, threads=128, out=None):
    lab = kernel_lab.default_lab()
    tensors = (x, packed.mantissa, packed.exponents, packed.metadata)
    for tensor in tensors:
        if tensor.device != torch.device("cuda", lab.device) or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("Inference-only contiguous tensors on the current CUDA device required")
    if x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError("FP32 activations required")
    n, k = packed.shape
    if min(x.shape[0], n, k) <= 0 or x.shape[1] != k or k % 64:
        raise ValueError("Invalid dimensions")
    if max(x.numel(), n * k * 3, x.shape[0] * n) >= 2**31 or x.data_ptr() % 16 or packed.mantissa.data_ptr() % 4:
        raise ValueError("Require 32-bit indexing and aligned FP32 input")
    if threads not in (32, 64, 128, 256):
        raise ValueError("Unsupported block size")
    expected_shape = (n, k // 64)
    if packed.metadata.shape != expected_shape:
        raise ValueError("Invalid metadata dimensions")
    if any(v.dtype != torch.uint8 for v in (packed.mantissa, packed.exponents)):
        raise ValueError("Expected byte payloads")
    if packed.metadata.dtype != torch.int32 or packed.exponents.ndim != 1:
        raise ValueError("Invalid exponent offsets")
    if out is None:
        out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.float32)
    else:
        lab.validate(out)
        if out.shape != (x.shape[0], n):
            raise ValueError("Output shape mismatch")
    if any(out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in tensors):
        raise ValueError("Output cannot alias inputs")
    rows = threads // 32
    lab.launch(f"lossless_t{threads}", ((n + rows - 1) // rows, x.shape[0]), threads,
               x, packed.mantissa, packed.exponents, packed.metadata, out, ct.c_int(n), ct.c_int(k))
    return out


@torch.library.custom_op("kernel_lab::lossless_linear", mutates_args=(), device_types="cuda")
def lossless_linear(x: torch.Tensor, mantissa: torch.Tensor, exponents: torch.Tensor,
                    metadata: torch.Tensor,
                    threads: int = 128) -> torch.Tensor:
    return fused_linear(x, LosslessWeights(mantissa, exponents, metadata), threads)


@lossless_linear.register_fake
def _lossless_fake(x, mantissa, exponents, metadata, threads=128):
    return x.new_empty((x.shape[0], mantissa.shape[0]))
