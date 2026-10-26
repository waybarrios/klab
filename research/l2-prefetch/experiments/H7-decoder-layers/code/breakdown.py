"""EXPLORATORY: GPU kernel-time breakdown of one decode step (eager, profiler), by kernel category."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys, collections
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, models, torch
from torch.profiler import profile, ProfilerActivity
torch.set_grad_enabled(False)
with H.lock():
    H.require_p0()
    for name, fused in (("smollm2-135m", False), ("qwen2.5-0.5b", True)):
        ops, reads, x0, period, keep = models.build(name, 1, fused, seed=1)
        run = lambda: [y := x0] and [y := op(y) for op in ops]
        for _ in range(3): run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(); torch.cuda.synchronize()
        cat = collections.Counter(); cnt = collections.Counter()
        for e in prof.events():
            if e.device_type.name != "CUDA": continue
            n = e.name.lower()
            k = ("gemv/gemm" if any(s in n for s in ("gemv", "gemm", "cutlass", "sm80", "ampere", "splitk", "cublas")) else
                 "attention" if any(s in n for s in ("attention", "fmha", "flash", "softmax", "bmm")) else
                 "cat/copy" if any(s in n for s in ("cat", "copy", "memcpy")) else "elementwise/norm/other")
            cat[k] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total; cnt[k] += 1
        tot = sum(cat.values())
        print(f"== {name} fused={fused}: total kernel time {tot:.0f} us, {sum(cnt.values())} kernels")
        for k, v in cat.most_common(): print(f"   {k:24s} {v:8.0f} us {100*v/tot:5.1f}%  kernels={cnt[k]}")
        top = collections.Counter()
        for e in prof.events():
            if e.device_type.name == "CUDA": top[e.name[:70]] += e.device_time_total
        for n, v in top.most_common(8): print(f"      {v:8.0f} us  {n}")
        del ops, reads, keep; torch.cuda.empty_cache()
