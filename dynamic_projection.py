"""Prepare a shape-specific plain/lossless FP32 projection outside the hot path."""

import hashlib
import json
import threading
from pathlib import Path

import torch
import kernel_lab
import lossless_weights
import packed_fp32
import half_projection
from cublas_baseline import CuBlas
from experiment_control import project_path
from experiment_provenance import CUDA_SOURCES
from projection_policy import choose

ROOT=Path(__file__).resolve().parent
_vendors={}
_vendor_lock=threading.Lock()


def vendor_call(x,w,config,out=None):
    with torch.cuda.device(x.device),_vendor_lock:
        lab=kernel_lab.default_lab()
        for tensor in (x,w):
            if tensor.device!=torch.device("cuda",lab.device) or not tensor.is_contiguous() or tensor.requires_grad:
                raise ValueError("Inference-only contiguous same-device tensors required")
        if x.dtype!=w.dtype or x.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError("Projection input dtypes differ or are unsupported")
        packed="packed" in config
        n=w.shape[1] if packed else w.shape[0]
        if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=(w.shape[0] if packed else w.shape[1]):
            raise ValueError("Projection dimensions differ")
        out=lab.output(out,(x.shape[0],n),x,w)
        if config=="torch_mm": return torch.mm(x,w.T,out=out)
        if config=="torch_mv": return torch.mv(w,x[0],out=out[0]).view(1,n)
        device=x.device.index
        if device not in _vendors: _vendors[device]=CuBlas()
        if config in ("cublas_compute32","cublas_packed_compute32"):
            return _vendors[device].mm_compute32(x,w,out,"fp16" if x.dtype==torch.float16 else "bf16","packed" in config)
        if config=="torch_mm_out_fp32": return torch.mm(x,w.T,out_dtype=torch.float32,out=out)
        functions={"cublas_mm":"mm","cublas_mv":"mv","cublas_packed_mm":"mm_packed","cublas_packed_mv":"mv_packed"}
        return getattr(_vendors[device],functions[config])(x,w,out)


@torch.library.custom_op("kernel_lab::dynamic_vendor",mutates_args=(),device_types="cuda")
def _vendor_op(x:torch.Tensor,w:torch.Tensor,config:str)->torch.Tensor:
    return vendor_call(x,w,config)


@_vendor_op.register_fake
def _vendor_fake(x,w,config):
    return x.new_empty((x.shape[0],w.shape[1] if "packed" in config else w.shape[0]),dtype=torch.float32)


@torch.library.custom_op("kernel_lab::dynamic_tiny",mutates_args=(),device_types="cuda")
def _tiny_op(x:torch.Tensor,w:torch.Tensor,threads:int,vector:int)->torch.Tensor:
    with torch.cuda.device(x.device):
        return kernel_lab.default_lab().tiny_matmul(x,w,threads,vector)


@_tiny_op.register_fake
def _tiny_fake(x,w,threads,vector):
    return x.new_empty((x.shape[0],w.shape[0]))


class PreparedProjection:
    def __init__(self,w,tokens=1,profile=None):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Prepare weights before CUDA graph capture")
        self.lab=kernel_lab.default_lab()
        if (w.device!=torch.device("cuda",self.lab.device) or not w.is_contiguous() or w.requires_grad
                or w.dtype not in (torch.float32,torch.float16,torch.bfloat16)):
            raise ValueError("Expected contiguous inference-only FP32, FP16, or BF16 CUDA weights")
        if w.ndim!=2 or min(w.shape)<=0 or tokens<=0: raise ValueError("Expected nonempty w[N,K] and positive tokens")
        self.dtype=w.dtype
        dtype_name={torch.float32:"fp32",torch.float16:"fp16",torch.bfloat16:"bf16"}[w.dtype]
        path=project_path(ROOT,profile or f"results/stock-{dtype_name}.json")
        data=json.loads(path.read_text())
        if ((w.dtype==torch.float32 and data.get("kind")!="streaming-projection")
                or (w.dtype!=torch.float32 and (data.get("kind")!="half-projection" or data.get("dtype")!=dtype_name))):
            raise ValueError("Validation precision contract does not match the supplied weight dtype")
        hashes=data.get("source_sha256",{})
        required=set(CUDA_SOURCES)|{"native/driver.cpp","native/build/libkernel_lab.so","kernel_lab.py",
                                     "lossless_weights.py","packed_fp32.py"}
        if w.dtype!=torch.float32: required.add("half_projection.py")
        if not required.issubset(hashes) or any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=hashes[name] for name in required):
            raise ValueError("Validation fingerprint is stale; rerun klab stream before using this profile")
        env=data.get("environment",{})
        if env.get("compilation_cache_key")!=self.lab.compilation["cache_key"] or env.get("torch")!=str(torch.__version__):
            raise ValueError("Validation compiler/runtime differs")
        actual_uuid=str(torch.cuda.get_device_properties(self.lab.device).uuid)
        if f"GPU-{actual_uuid}" not in env.get("inventory",{}).get("gpu",""):
            raise ValueError("Validation belongs to a different GPU")
        if env["inventory"].get("module_loaded")!=Path("/sys/module/gpu_unlock").exists():
            raise ValueError("GPU unlock state changed; validate this state separately")
        self.m,self.n,self.k=tokens,*w.shape
        self.plan=choose(data,(tokens,self.n,self.k))
        self.strategy=self.plan["config"]
        self.packed=None
        self.w=None
        if self.strategy.startswith("lossless_"):
            self.packed=lossless_weights.pack_weights(w)
        elif self.strategy.startswith("fp28_"):
            self.packed=packed_fp32.pack_weights(w)
        else:
            self.w=w.T.contiguous() if "packed" in self.strategy else w
        if self.packed is not None:
            cases=[c for c in data["cases"] if tuple(c["shape_mnk"])==(tokens,self.n,self.k)]
            calibrated=max(c["lossless_ring_bytes"]/c["raw_ring_bytes"] for c in cases)
            if self.packed.storage_bytes/(w.numel()*w.element_size())>calibrated*1.01:
                self.plan=choose(data,(tokens,self.n,self.k),force_vendor=True)
                self.plan["reason"]="Weight compression does not meet the calibrated storage ratio"
                self.strategy=self.plan["config"]
                self.packed=None
                self.w=w.T.contiguous() if "packed" in self.strategy else w
        self.threads=int(self.strategy.split("_")[1][1:]) if self.plan["backend"]=="custom" else None
        self.vector=int(self.strategy.rsplit("v",1)[1]) if self.strategy.startswith(("tiny_","warp_","half_")) else None
        self.profile=str(path.relative_to(ROOT))
        if self.plan["backend"]=="vendor":
            # Allocate vendor handles/workspace before any later graph capture.
            warmup=torch.zeros(tokens,self.k,device=w.device,dtype=w.dtype)
            vendor_call(warmup,self.w,self.strategy)

    def linear(self,x,out=None):
        if (x.device!=torch.device("cuda",self.lab.device) or x.dtype!=self.dtype
                or not x.is_contiguous() or x.requires_grad):
            raise ValueError("Expected contiguous inference-only input matching the prepared weight dtype/device")
        if x.ndim!=2 or tuple(x.shape)!=(self.m,self.k):
            raise ValueError("Input shape differs; prepare a plan for this token count")
        if self.plan["backend"]=="vendor":
            return vendor_call(x,self.w,self.strategy,out) if out is not None else _vendor_op(x,self.w,self.strategy)
        if self.strategy.startswith("half_"):
            return (half_projection.fused_linear(x,self.w,self.threads,out,self.vector) if out is not None
                    else half_projection.linear(x,self.w,self.threads,self.vector))
        if self.strategy.startswith("tiny_"):
            return (self.lab.tiny_matmul(x,self.w,self.threads,self.vector,out) if out is not None
                    else _tiny_op(x,self.w,self.threads,self.vector))
        if self.strategy.startswith("warp_"):
            return (self.lab.warp_matmul(x,self.w,self.threads,self.vector,out) if out is not None
                    else kernel_lab.warp_matmul(x,self.w,self.threads,self.vector))
        if self.strategy.startswith("lossless_"):
            return lossless_weights.fused_linear(x,self.packed,self.threads,out) if out is not None else self.packed.linear(x,self.threads)
        return packed_fp32.fused_linear(x,self.packed,self.threads,out) if out is not None else self.packed.linear(x,self.threads)

    __call__=linear


def prepare_projection(w,tokens=1,profile=None,*,output_dtype=torch.float32):
    """Weights must remain immutable; select/pack once, then call with x[tokens,K]."""
    if w.dtype==torch.bfloat16:
        data=None
        if profile is not None:
            path=project_path(ROOT,profile)
            data=json.loads(path.read_text())
        compressed=data is not None and data.get("kind") in (
            "compressed-projection","compressed-projection-tuning","compressed-projection-validation")
        if compressed or (output_dtype==torch.bfloat16 and profile is None):
            from prepared_bf16 import PreparedBF16Projection
            return PreparedBF16Projection(w,tokens,data,output_dtype=output_dtype)
    if output_dtype!=torch.float32:
        raise ValueError("Legacy projection profiles require FP32 output")
    return PreparedProjection(w,tokens,profile)
