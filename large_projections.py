"""Explore FP32 streaming/layout variants for large single-token projections.

Both original and prepacked layouts are available to vendor baselines. All
prepacking costs are recorded separately. No quantization or TF32 is used.
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path

import kernel_lab
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas
from validate_challenge import cold_measure

ROOT = Path(__file__).resolve().parent


def stream(lab, x, w, out, threads, cg, prefetch):
    n, k = w.shape
    lab.launch(f"proj_stream_t{threads}_c{cg}_p{prefetch}",
               ((n + threads // 32 - 1) // (threads // 32), x.shape[0]), threads,
               x, w, out, ct.c_int(n), ct.c_int(k))
    return out


def packed(lab, x, wt, out, partial, threads, split):
    k, n = wt.shape
    lab.launch(f"proj_packed_t{threads}", ((n + threads - 1) // threads, split), threads,
               x, wt, partial, ct.c_int(n), ct.c_int(k), ct.c_int(split))
    lab.launch("proj_reduce", ((n + 255) // 256, 1), 256,
               partial, out, ct.c_int(n), ct.c_int(split))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/large-projections.json")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must remain inside workspace")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab, vendor = kernel_lab.default_lab(), CuBlas()
    report = {"status": "running", "gpu_before": gpu_state(), "cases": [],
              "precision": "FP32 input/accumulation/output; TF32 off",
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("projection_kernels.cu", "kernels.cu", "kernel_lab.py",
                                          "cublas_baseline.py", "large_projections.py")}}
    for n, k in ((4096, 4096), (11008, 4096), (4096, 11008)):
        torch.manual_seed(901)
        x = torch.randn(1, k, device="cuda") / math.sqrt(k)
        w = torch.randn(n, k, device="cuda")
        wt = w.T.contiguous()
        reference = (x.double() @ w.double().T).float()
        outputs, functions = {}, {}
        for name in ("torch_mm", "torch_mv", "cublas_mm", "cublas_mv",
                     "torch_packed_mm", "cublas_packed_mm", "cublas_packed_mv"):
            outputs[name] = torch.empty(1, n, device="cuda")
        functions.update({
            "torch_mm": lambda: torch.mm(x, w.T, out=outputs["torch_mm"]),
            "torch_mv": lambda: torch.mv(w, x[0], out=outputs["torch_mv"][0]).view(1, n),
            "cublas_mm": lambda: vendor.mm(x, w, outputs["cublas_mm"]),
            "cublas_mv": lambda: vendor.mv(x, w, outputs["cublas_mv"]),
            "torch_packed_mm": lambda: torch.mm(x, wt, out=outputs["torch_packed_mm"]),
            "cublas_packed_mm": lambda: vendor.mm_packed(x, wt, outputs["cublas_packed_mm"]),
            "cublas_packed_mv": lambda: vendor.mv_packed(x, wt, outputs["cublas_packed_mv"]),
        })
        baselines = list(functions)
        for threads in (64, 128, 256):
            for cg, prefetch in ((0, 0), (1, 0), (1, 4)):
                name = f"stream_t{threads}_c{cg}_p{prefetch}"
                outputs[name] = torch.empty(1, n, device="cuda")
                functions[name] = lambda t=threads, c=cg, p=prefetch, o=outputs[name]: stream(lab, x, w, o, t, c, p)
            for split in (8, 16, 32, 64):
                name = f"packed_t{threads}_s{split}"
                outputs[name] = torch.empty(1, n, device="cuda")
                partial = torch.empty(split, n, device="cuda")
                functions[name] = lambda t=threads, s=split, o=outputs[name], tmp=partial: packed(lab, x, wt, o, tmp, t, s)
        errors = {name: check(fn(), reference) for name, fn in functions.items()}
        sweep = measure(functions, rounds=3, repeats=8)
        selected = min((name for name in sweep if name not in baselines), key=lambda name: sweep[name]["median_us"])
        torch.manual_seed(902)
        x.normal_().div_(math.sqrt(k))
        w.normal_()
        wt.copy_(w.T)
        reference = (x.double() @ w.double().T).float()
        final_functions = {name: functions[name] for name in (*baselines, selected)}
        errors = {name: check(fn(), reference) for name, fn in final_functions.items()}
        warm = measure(final_functions, rounds=9, repeats=16)
        cold = cold_measure(final_functions, rounds=15)
        captured = {name: check(outputs[name], reference) for name in final_functions}
        copy_target = torch.empty_like(w)
        memory = measure({"copy": lambda: copy_target.copy_(w),
                          "row_sum": lambda: torch.sum(w, dim=1)}, rounds=5, repeats=8)
        pack_target = torch.empty_like(wt)
        packing = measure({"pack_weights": lambda: pack_target.copy_(w.T)}, rounds=5, repeats=8)
        def speedup(timing):
            best = min(timing[name]["median_us"] for name in baselines)
            return best / timing[selected]["median_us"]
        case = {"shape_mnk": [1, n, k], "selected": selected, "sweep": sweep,
                "warm": warm, "cold": cold, "warm_speedup": speedup(warm),
                "cold_speedup": speedup(cold), "correctness": errors, "captured_correctness": captured,
                "memory_diagnostics": memory, "weight_packing": packing,
                "effective_copy_GBs": 2 * w.numel() * 4 / memory["copy"]["median_us"] / 1e3,
                "effective_reduction_read_GBs": w.numel() * 4 / memory["row_sum"]["median_us"] / 1e3,
                "gpu_after_case": gpu_state()}
        report["cases"].append(case)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"1x{n}x{k} {selected}: warm {case['warm_speedup']:.3f}x, cold {case['cold_speedup']:.3f}x; "
              f"copy {case['effective_copy_GBs']:.1f}GB/s, reduction read {case['effective_reduction_read_GBs']:.1f}GB/s", flush=True)
    vendor.close()
    report["status"] = "complete"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
