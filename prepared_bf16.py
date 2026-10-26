"""Dtype-correct prepared BF16 vendor fallback, with no codec in the call path.

The custom dispatch cohort is not complete yet. This branch deliberately uses
the current native vendor route, without restoring a serialized Lt algorithm or
promoting a training result. A fallback is never reported as a custom win.

``linear`` binds an allocation-only native operator. Native Torch owns output
allocation, input validation, FakeTensor and torch.compile behavior. The separate
``linear_out`` path checks aliasing. Ordinary calls contain only the native
matmul; route selection, output-type selection and inspection happen once.
Preparation and CUDA handle warmup happen once, before capture. Keep the object
and its weights alive for every graph that uses it. Native vendor calls retain
ordinary Torch shape/stride semantics; shape_mnk describes the prepared case,
not a validated custom performance claim for arbitrary subsequent inputs.
"""
from functools import partial
import time

import torch

from projection_policy import compressed_vendor_fallback


def vendor_callable(w,output_dtype):
    """Bind immutable BF16 weights, preserving the requested output contract."""
    if output_dtype not in (torch.bfloat16,torch.float32):
        raise ValueError("Expected BF16 or FP32 output")
    if (not isinstance(w,torch.Tensor) or w.ndim!=2 or min(w.shape)<=0
            or w.dtype!=torch.bfloat16 or w.requires_grad):
        raise ValueError("Expected nonempty inference-only BF16 weights [N,K]")
    w_t=w.T
    if output_dtype==torch.bfloat16:
        def linear(x):
            return torch.mm(x,w_t)
    else:
        def linear(x):
            return torch.mm(x,w_t,out_dtype=torch.float32)
    # Fixed signatures reject out= rather than bypassing the alias checks.
    # Public Torch calls avoid the keyword-bound descriptor/boxed ATen costs
    # observed in the preserved revision2 eager measurements.
    return linear


def vendor_out_callable(w,output_dtype):
    """Bind the separately checked out= route; preparation validates weights."""
    vendor_callable(w,output_dtype)
    return partial(_vendor_linear,w.T,output_dtype)


def _vendor_linear(w_t,output_dtype,x,*,out=None):
    if out is not None:
        if (not isinstance(out,torch.Tensor) or out.dtype!=output_dtype
                or out.device!=x.device or out.ndim!=2
                or tuple(out.shape)!=(x.shape[0],w_t.shape[1])
                or not out.is_contiguous() or out.requires_grad):
            raise RuntimeError("out must be a contiguous inference tensor of the expected dtype, device and shape")
        if torch._C._overlaps(out,x) or torch._C._overlaps(out,w_t):
            raise ValueError("out must not overlap the input or prepared weights")
    if output_dtype==torch.float32:
        return torch.mm(x,w_t,out_dtype=torch.float32,out=out)
    return torch.mm(x,w_t,out=out)


class PreparedBF16Projection:
    def __init__(self,w,tokens=1,profile=None,*,output_dtype=torch.float32):
        started=time.perf_counter()
        if (not isinstance(w,torch.Tensor) or w.device.type!="cuda" or w.ndim!=2
                or min(w.shape)<=0 or not w.is_contiguous() or w.requires_grad
                or w.dtype!=torch.bfloat16):
            raise ValueError("Expected nonempty contiguous inference-only BF16 CUDA weights [N,K]")
        if isinstance(tokens,bool) or not isinstance(tokens,int) or tokens<=0:
            raise ValueError("Expected a positive integer token count")
        if output_dtype not in (torch.bfloat16,torch.float32):
            raise ValueError("Expected BF16 or FP32 output")
        with torch.cuda.device(w.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("Prepare weights before CUDA graph capture")
            self.w=w
            self.dtype=w.dtype
            self.output_dtype=output_dtype
            self.precision_flags={"allow_tf32":torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
            if any(self.precision_flags.values()):
                raise ValueError("Set allow_tf32 and allow_bf16_reduced_precision_reduction to False before preparing this FP32-accumulation contract")
            self.m,self.n,self.k=tokens,*w.shape
            self.shape_mnk=(tokens,*w.shape)
            name="bf16" if output_dtype==torch.bfloat16 else "fp32"
            self.plan=compressed_vendor_fallback(profile or {},self.shape_mnk,name)
            self.strategy=self.plan["config"]
            self.packed=None
            self.packing_seconds=0.
            self.linear=vendor_callable(w,output_dtype)
            self.linear_out=vendor_out_callable(w,output_dtype)
            # Native handles and workspaces must already exist before capture.
            warmup=torch.zeros((tokens,self.k),device=w.device,dtype=w.dtype)
            self.linear(warmup)
        self.setup_seconds=time.perf_counter()-started

    def __call__(self,x,out=None):
        return self.linear(x) if out is None else self.linear_out(x,out=out)
