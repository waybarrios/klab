"""Spike: does prefetching layer i+1's weights into L2 during layer i cut chain latency? Plain F.linear consumers."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import ctypes as ct, math, os, sys
from pathlib import Path
ROOT = Path(KLAB); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
import torch, torch.nn.functional as F
from bench import measure
from cuda_module import CudaModule
from experiment_control import GpuLock

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
with GpuLock(ROOT / ".cache/gpu.lock", wait=True):
    mod = CudaModule((".scratch/latency/prefetch_kernels.cu",), ("l2_hint", "l2_load"), 0)
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    for d in (256, 512, 896):
        layers = max(16, math.ceil((80 << 20) / (d * d * 2)))
        W = torch.randn(layers, d, d, dtype=torch.bfloat16, device="cuda") / math.sqrt(d)
        ws = W.unbind()
        for m in (1, 16):
            x0 = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
            outs = {}
            def chain(kernel=None, blocks=0, stride=128, name=None):
                def run():
                    main = torch.cuda.current_stream(); y = x0
                    for i in range(layers):
                        if kernel and i + 1 < layers:
                            side.wait_stream(main)  # start alongside layer i
                            with torch.cuda.stream(side):
                                mod.launch(kernel, (blocks, 1), 256, ws[i + 1], ct.c_longlong(d * d * 2), ct.c_int(stride), sink)
                        y = F.linear(y, ws[i])
                    if kernel: main.wait_stream(side)
                    outs[name] = y
                return run
            fns = {"torch": chain(name="torch"), "torch_aa": chain(name="torch_aa")}
            for kernel in ("l2_hint", "l2_load"):
                for blocks in (4, 12, 46):
                    for stride in ((128, 32) if kernel == "l2_hint" else (32,)):
                        n = f"{kernel}_b{blocks}_s{stride}"; fns[n] = chain(kernel, blocks, stride, n)
            t = measure(fns, rounds=15, repeats=1)
            base = t["torch"]["median_us"] / layers
            ref = outs["torch"]
            print(f"\n== d={d} M={m} layers={layers}  torch {base:.3f} us/layer  (A/A {t['torch_aa']['median_us']/layers:.3f})")
            for n, v in sorted(t.items(), key=lambda kv: kv[1]["median_us"]):
                if n.startswith("torch"): continue
                per = v["median_us"] / layers
                same = torch.equal(outs[n], ref)
                print(f"  {n:18s} {per:7.3f} us/layer  saved {100*(1-per/base):+6.2f}%  bit-identical={same}")
            del outs
        del W, ws; torch.cuda.empty_cache()
    torch.cuda.synchronize(); mod.close()
