"""Direct vendor-library baseline, using the SAME tensors and CUDA stream.

C = x @ w.T = column-major transpose(w) @ column-major(x).
Use default high-performance FP32 math, not the slower pedantic mode.
"""

import ctypes as ct
from pathlib import Path

import nvidia.cublas
import torch


class CuBlas:
    def __init__(self):
        base = Path(next(iter(nvidia.cublas.__path__)))
        self.lib = ct.CDLL(str(base / "lib/libcublas.so.12"))
        pointer = ct.c_void_p
        self.lib.cublasCreate_v2.argtypes = [ct.POINTER(pointer)]
        self.lib.cublasDestroy_v2.argtypes = [pointer]
        self.lib.cublasSetStream_v2.argtypes = [pointer, pointer]
        self.lib.cublasSetMathMode.argtypes = [pointer, ct.c_int]
        self.lib.cublasSgemm_v2.argtypes = [
            pointer, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
            ct.POINTER(ct.c_float), pointer, ct.c_int, pointer, ct.c_int,
            ct.POINTER(ct.c_float), pointer, ct.c_int,
        ]
        self.lib.cublasSgemv_v2.argtypes = [
            pointer, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_float),
            pointer, ct.c_int, pointer, ct.c_int, ct.POINTER(ct.c_float), pointer, ct.c_int,
        ]
        self.lib.cublasGemmEx.argtypes = [
            pointer, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
            pointer, pointer, ct.c_int, ct.c_int, pointer, ct.c_int, ct.c_int,
            pointer, pointer, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ]
        self.handle = pointer()
        self.check(self.lib.cublasCreate_v2(ct.byref(self.handle)))
        self.check(self.lib.cublasSetMathMode(self.handle, 0))
        self.lib.cublasGetVersion_v2.argtypes=[pointer,ct.POINTER(ct.c_int)]
        version=ct.c_int()
        self.check(self.lib.cublasGetVersion_v2(self.handle,ct.byref(version)))
        self.version=version.value
        self.alpha = ct.c_float(1)
        self.beta = ct.c_float(0)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"cuBLAS error {status}")

    def mm(self, x, w, out):
        # This is a benchmark helper. KernelLab does the public API validation.
        m, k = x.shape
        n = w.shape[0]
        self.check(self.lib.cublasSetStream_v2(
            self.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        ))
        self.check(self.lib.cublasSgemm_v2(
            self.handle, 1, 0, n, m, k, ct.byref(self.alpha),
            ct.c_void_p(w.data_ptr()), k, ct.c_void_p(x.data_ptr()), k,
            ct.byref(self.beta), ct.c_void_p(out.data_ptr()), n,
        ))
        return out

    def mv(self, x, w, out):
        if x.shape[0] != 1:
            raise ValueError("GEMV requires one input vector")
        k = x.shape[1]
        n = w.shape[0]
        self.check(self.lib.cublasSetStream_v2(
            self.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        ))
        self.check(self.lib.cublasSgemv_v2(
            self.handle, 1, k, n, ct.byref(self.alpha), ct.c_void_p(w.data_ptr()), k,
            ct.c_void_p(x.data_ptr()), 1, ct.byref(self.beta), ct.c_void_p(out.data_ptr()), 1,
        ))
        return out

    def mm_packed(self, x, wt, out):
        m, k = x.shape
        n = wt.shape[1]
        self.check(self.lib.cublasSetStream_v2(
            self.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        ))
        self.check(self.lib.cublasSgemm_v2(
            self.handle, 0, 0, n, m, k, ct.byref(self.alpha),
            ct.c_void_p(wt.data_ptr()), n, ct.c_void_p(x.data_ptr()), k,
            ct.byref(self.beta), ct.c_void_p(out.data_ptr()), n,
        ))
        return out

    def mm_half(self, x, w, out):
        """FP16 A/B, FP32 compute and output, default GEMM algorithm."""
        m, k = x.shape
        n = w.shape[0]
        self.check(self.lib.cublasSetStream_v2(
            self.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        ))
        self.check(self.lib.cublasGemmEx(
            self.handle, 1, 0, n, m, k, ct.byref(self.alpha),
            ct.c_void_p(w.data_ptr()), 2, k, ct.c_void_p(x.data_ptr()), 2, k,
            ct.byref(self.beta), ct.c_void_p(out.data_ptr()), 0, n, 68, -1,
        ))
        return out

    def mm_compute32(self, x, w, out, mode="fp32", packed=False):
        """Explicit input/compute contract for the Tensor Core diagnostic.

        fp16/bf16 round inputs before this call. tf32 explicitly permits
        internal input rounding. All paths accumulate and return FP32.
        """
        formats = {"fp32": (torch.float32, 0, 68), "fp16": (torch.float16, 2, 68),
                   "bf16": (torch.bfloat16, 14, 68), "tf32": (torch.float32, 0, 77)}
        if mode not in formats:
            raise ValueError("Choose fp32, fp16, bf16, or tf32")
        dtype, storage, compute = formats[mode]
        if (x.dtype != dtype or w.dtype != dtype or out.dtype != torch.float32
                or not all(t.is_cuda and t.is_contiguous() for t in (x,w,out))
                or len({t.device for t in (x,w,out)}) != 1):
            raise ValueError("Expected contiguous same-device CUDA tensors matching the requested input format")
        if (x.ndim != 2 or w.ndim != 2 or x.shape[1] != (w.shape[0] if packed else w.shape[1])
                or out.shape != (x.shape[0],w.shape[1] if packed else w.shape[0])):
            raise ValueError("Expected x[M,K], w[N,K], out[M,N]")
        m,k=x.shape
        n=w.shape[1] if packed else w.shape[0]
        self.check(self.lib.cublasSetStream_v2(self.handle,ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)))
        self.check(self.lib.cublasGemmEx(
            self.handle,0 if packed else 1,0,n,m,k,ct.byref(self.alpha),ct.c_void_p(w.data_ptr()),storage,n if packed else k,
            ct.c_void_p(x.data_ptr()),storage,k,ct.byref(self.beta),ct.c_void_p(out.data_ptr()),0,n,compute,-1))
        return out

    def mv_packed(self, x, wt, out):
        if x.shape[0] != 1:
            raise ValueError("GEMV requires one input vector")
        k, n = wt.shape
        self.check(self.lib.cublasSetStream_v2(
            self.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        ))
        self.check(self.lib.cublasSgemv_v2(
            self.handle, 0, n, k, ct.byref(self.alpha), ct.c_void_p(wt.data_ptr()), n,
            ct.c_void_p(x.data_ptr()), 1, ct.byref(self.beta), ct.c_void_p(out.data_ptr()), 1,
        ))
        return out

    def close(self):
        torch.cuda.synchronize()
        self.check(self.lib.cublasDestroy_v2(self.handle))
