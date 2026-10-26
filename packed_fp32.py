"""Lossless FP32 with interleaved 28-bit codes and original-bit fallback.

28 describes storage only. Every original FP32 weight bit is retained.
"""

import ctypes as ct
from dataclasses import dataclass

import torch
import kernel_lab


@dataclass(frozen=True)
class PackedFP32Weights:
    payload: torch.Tensor
    metadata: torch.Tensor

    @property
    def shape(self):
        return self.metadata.shape[0], self.metadata.shape[1] * 64

    @property
    def storage_bytes(self):
        return (self.payload.numel() + self.metadata.numel()) * 4

    def unpack(self):
        n, k = self.shape
        out = torch.empty(n, k, dtype=torch.float32, device=self.payload.device)
        kernel_lab.default_lab().launch("fp28_unpack", (((n*k//4)+255)//256,1),256,
                                        self.payload,self.metadata,out,ct.c_int(n*k))
        return out

    def linear(self, x, threads=128):
        return linear(x, self.payload, self.metadata, threads)


def pack_weights(w):
    if w.device.type != "cuda" or w.dtype != torch.float32 or w.ndim != 2 or not w.is_contiguous():
        raise ValueError("Expected contiguous CUDA FP32 [N,K]")
    n, k = w.shape
    if w.requires_grad or min(n,k) <= 0 or k % 64 or w.numel() * 4 >= 2**31:
        raise ValueError("Inference only; positive 32-bit-addressable sizes and K divisible by 64 required")
    bits = w.view(torch.int32).reshape(-1,64)
    exponents = (bits >> 23) & 255
    base = exponents.amin(-1)
    delta = exponents - base[:,None]
    wide = delta.amax(-1) > 15
    significands = (bits & 0x7fffff) | ((bits >> 8) & 0x800000)
    codes = (significands.to(torch.int64) | (delta.to(torch.int64) << 24)).reshape(-1,8)
    words = []
    for j in range(7):
        index, shift = divmod(j*32,28)
        word = (codes[:,index] >> shift) | (codes[:,index+1] << (28-shift))
        words.append(word.to(torch.int32))
    compact = torch.stack(words,-1).reshape(-1,56)
    source = torch.where(wide[:,None],bits,
                         torch.cat((compact,torch.zeros_like(compact[:,:8])),dim=-1))
    sizes = torch.where(wide,64,56).to(torch.int64)
    offsets = sizes.cumsum(0)-sizes
    total = int(sizes.sum().item())
    if total * 4 >= 2**28:
        raise ValueError("Payload exceeds experimental 256 MiB metadata limit")
    payload = torch.empty(total,dtype=torch.int32,device=w.device)
    columns = torch.arange(64,device=w.device)
    valid = columns[None,:] < sizes[:,None]
    positions = offsets[:,None] + columns[None,:]
    payload.scatter_(0,positions[valid],source[valid])
    metadata = (((offsets >> 3) << 9) | (base.to(torch.int64)<<1) | wide.to(torch.int64)).to(torch.int32)
    return PackedFP32Weights(payload,metadata.reshape(n,k//64).contiguous())


def fused_linear(x, packed, threads=128, out=None):
    lab = kernel_lab.default_lab()
    lab.validate(x)
    n,k = packed.shape
    for tensor in (packed.payload,packed.metadata):
        if tensor.device != x.device or not tensor.is_contiguous() or tensor.dtype != torch.int32:
            raise ValueError("Expected intact contiguous int32 packed buffers on the input device")
    if x.ndim != 2 or x.shape[1] != k or min(x.shape[0],n,k)<=0 or k%64:
        raise ValueError("Expected x=[M,K], matching packed weights")
    if max(x.numel(),n*k,x.shape[0]*n)>=2**31 or x.data_ptr()%16 or packed.payload.ndim!=1:
        raise ValueError("Aligned input and positive signed 32-bit indexing required")
    if threads not in (32,64,128,256):
        raise ValueError("Unsupported block size")
    out=lab.output(out,(x.shape[0],n),x,packed.payload,packed.metadata)
    lab.launch(f"fp28_t{threads}",((n+threads//32-1)//(threads//32),x.shape[0]),threads,
               x,packed.payload,packed.metadata,out,ct.c_int(n),ct.c_int(k))
    return out


@torch.library.custom_op("kernel_lab::packed_fp32_linear", mutates_args=(),device_types="cuda")
def linear(x: torch.Tensor, payload: torch.Tensor, metadata: torch.Tensor,
           threads: int=128) -> torch.Tensor:
    return fused_linear(x,PackedFP32Weights(payload,metadata),threads)


@linear.register_fake
def _fake(x,payload,metadata,threads=128):
    return x.new_empty((x.shape[0],metadata.shape[0]))
