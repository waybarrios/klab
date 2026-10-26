"""Per-launch latency floor in the same streaming graph harness (80 MiB ring, per-call median)."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import ctypes as ct, math, os, sys
from pathlib import Path
ROOT = Path(KLAB); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
import torch
from bench import measure
from cuda_module import CudaModule
from experiment_control import GpuLock

with GpuLock(ROOT / ".cache/gpu.lock", wait=True):
    mod = CudaModule((".scratch/latency/floor_kernels.cu",), ("floor_empty", "floor_touch", "floor_rows"), 0)
    funcs, keep = {}, []
    for n, k in ((64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)):
        copies = max(2, math.ceil((80 << 20) / (n * k * 2)))
        ring = torch.randint(0, 2**15, (copies, n, k), dtype=torch.int16, device="cuda")
        out = torch.zeros(copies, n, device="cuda"); keep += [ring, out]
        ws, os_ = ring.unbind(), out.unbind()
        if n == 64:
            funcs["empty"] = lambda ws=ws, os_=os_, c=copies: [mod.launch("floor_empty", (1, 1), 32, ws[i], os_[i]) for i in range(c)]
            funcs["touch"] = lambda ws=ws, os_=os_, c=copies: [mod.launch("floor_touch", (1, 1), 32, ws[i], os_[i]) for i in range(c)]
        funcs[f"rows_{n}x{k}"] = (lambda ws=ws, os_=os_, c=copies, n=n, k=k:
            [mod.launch("floor_rows", ((n + 3) // 4, 1), 128, ws[i], os_[i], ct.c_int(n), ct.c_int(k // 2)) for i in range(c)])
        funcs[f"rows_{n}x{k}"].copies = copies
    copies = {name: getattr(f, "copies", math.ceil((80 << 20) / (64 * 64 * 2))) for name, f in funcs.items()}
    timing = measure(funcs, rounds=11, repeats=1)
    for name, t in timing.items():
        print(f"{name:16s} {t['median_us'] / copies[name]:6.2f} us/launch")
    torch.cuda.synchronize(); mod.close()
