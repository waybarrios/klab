"""Find and recheck FP32 small-M kernels against cuBLAS and PyTorch.

This search records every case, including losses. Selection uses one seed;
final timing and FP64 validation use another. Run one process at a time.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import kernel_lab
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cublas-challenge.json")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must stay in the workspace")
    destination.parent.mkdir(exist_ok=True, parents=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab = kernel_lab.default_lab()
    vendor = CuBlas()
    report = {
        "status": "running", "precision": "FP32 input/accumulation/output; TF32 off",
        "vendor_math": "cublasSgemm_v2, CUBLAS_DEFAULT_MATH (not pedantic)",
        "timing": "CUDA graphs, 7 randomized rounds; hot fixed inputs, no output caching",
        "gpu_before": gpu_state(), "torch": torch.__version__,
        "sources_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                           for p in ("kernels.cu", "kernel_lab.py", "challenge.py", "cublas_baseline.py")},
        "cases": [],
    }
    # Small attention/MLP projections plus the earlier, memory-streaming control.
    shapes = [(1, 256, 1024), (1, 1024, 1024), (1, 3072, 1024),
              (1, 768, 768), (1, 2304, 768), (1, 4096, 4096),
              (2, 1024, 1024), (4, 1024, 1024), (8, 1024, 1024),
              (1, 256, 4096), (1, 1024, 4096), (1, 1536, 1536)]
    for m, n, k in shapes:
        torch.manual_seed(501)
        x = torch.randn(m, k, device="cuda") / math.sqrt(k)
        w = torch.randn(n, k, device="cuda")
        reference = (x.double() @ w.double().T).float()
        torch_out = torch.empty(m, n, device="cuda")
        vendor_out = torch.empty_like(torch_out)
        functions = {"torch_mm": lambda: torch.mm(x, w.T, out=torch_out),
                     "cublas_sgemm": lambda: vendor.mm(x, w, vendor_out)}
        if m == 1:
            mv_out = torch.empty_like(torch_out)
            functions["cublas_sgemv"] = lambda: vendor.mv(x, w, mv_out)
            functions["torch_mv"] = lambda: torch.mv(w, x[0], out=mv_out[0]).view(1, n)
        baseline_names = list(functions)
        for threads in (32, 64, 128, 256):
            for vector in (1, 4):
                out = torch.empty_like(torch_out)
                functions[f"warp_t{threads}_v{vector}"] = (
                    lambda t=threads, v=vector, o=out: lab.warp_matmul(x, w, t, v, o)
                )
        for rows in (1, 4):
            out = torch.empty_like(torch_out)
            functions[f"blocked_r{rows}"] = lambda r=rows, o=out: lab.matmul(x, w, r, 128, o)
        if m > 1:
            for tile in ("m8_n32", "m16_n64", "m32_n32"):
                out = torch.empty_like(torch_out)
                functions[f"tile_{tile}"] = lambda t=tile, o=out: lab.tiled_matmul(x, w, t, o)
        for fn in functions.values():
            check(fn(), reference)
        sweep = measure(functions, rounds=3, repeats=16)
        candidates = [name for name in sweep if name not in baseline_names]
        best = min(candidates, key=lambda name: sweep[name]["median_us"])
        torch.manual_seed(502)
        x.normal_().div_(math.sqrt(k))
        w.normal_()
        reference = (x.double() @ w.double().T).float()
        final_functions = {name: functions[name] for name in (*baseline_names, best)}
        errors = {name: check(fn(), reference) for name, fn in final_functions.items()}
        final = measure(final_functions, rounds=9, repeats=128 if k * n < 4194304 else 16)
        baseline = min(final[name]["median_us"] for name in baseline_names)
        speedup = baseline / final[best]["median_us"]
        case = {"shape_mnk": [m, n, k], "selected": best, "sweep": sweep,
                "final": final, "correctness": errors, "speedup_vs_fastest_vendor": speedup,
                "gpu_after_case": gpu_state()}
        report["cases"].append(case)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{m}x{n}x{k}: torch {final['torch_mm']['median_us']:.2f}us, "
              f"cuBLAS {final['cublas_sgemm']['median_us']:.2f}us, "
              f"{best} {final[best]['median_us']:.2f}us -> {speedup:.3f}x", flush=True)
    report["status"] = "complete"
    report["gpu_after"] = gpu_state()
    destination.write_text(json.dumps(report, indent=2) + "\n")
    vendor.close()


if __name__ == "__main__":
    main()
