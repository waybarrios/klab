"""Side-stream L2 prefetch for chains of PyTorch ops, meant for CUDA graph capture.

prefetch() never writes the tensor it reads. ChainRunner runs op i on the current stream and, at
the same time, prefetches the tensors op i+distance will read on a side stream.
"""
import ctypes as ct
from pathlib import Path

import torch

SOURCE = "research/l2-prefetch/src/prefetch_kernels.cu"
KERNELS = ("l2_hint", "l2_load", "l2_hint_last")
_module = None


def module():
    global _module
    if _module is None:
        from cuda_module import CudaModule
        _module = CudaModule((SOURCE,), KERNELS, torch.cuda.current_device())
    return _module


_sink = {}


def prefetch(tensor, blocks, kernel="l2_hint", stride=128, threads=256, nbytes=None):
    """Enqueue an L2 prefetch of `tensor`'s bytes (or the first `nbytes`) on the current stream."""
    if blocks <= 0:
        return
    device = tensor.device
    if device not in _sink:
        _sink[device] = torch.zeros(1, dtype=torch.int32, device=device)
    size = tensor.numel() * tensor.element_size() if nbytes is None else min(nbytes, tensor.numel() * tensor.element_size())
    module().launch(kernel, (blocks, 1), threads, tensor, ct.c_longlong(size), ct.c_int(stride), _sink[device])


class ChainRunner:
    """ops[i](y) -> y; reads[i] = tensors op i reads that are worth prefetching; plans[i] = kwargs or None."""

    def __init__(self, ops, reads, plans=None, distance=1):
        self.ops, self.reads, self.distance = ops, reads, distance
        self.plans = plans if plans is not None else [None] * len(ops)
        self.side = torch.cuda.Stream()

    def __call__(self, y):
        main = torch.cuda.current_stream()
        used = False
        for i, op in enumerate(self.ops):
            j = i + self.distance
            if j < len(self.ops) and self.plans[j]:
                self.side.wait_stream(main)  # starts when op i starts
                with torch.cuda.stream(self.side):
                    for t in self.reads[j]:
                        prefetch(t, **self.plans[j])
                used = True
            y = op(y)
        if used:
            main.wait_stream(self.side)
        return y
