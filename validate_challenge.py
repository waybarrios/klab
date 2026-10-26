"""Recheck frozen candidates on fresh seeds, with warm and evicted caches.

Evicted-cache timing captures a 64 MiB read/write eviction before external
CUDA start/stop events. The eviction itself is excluded from the measured
interval. This approximates cold inputs; it is not a hardware cache flush API.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import kernel_lab
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas

ROOT = Path(__file__).resolve().parent


def cold_measure(functions, rounds=15):
    eviction = torch.zeros(64 * 1024 * 1024 // 4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captures = {}
    for name, fn in functions.items():
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph, stream=stream):
            eviction.add_(1)
            start.record()
            fn()
            end.record()
        stream.synchronize()
        captures[name] = (graph, start, end)
    samples = {name: [] for name in functions}
    rng = random.Random(901)
    try:
        for _ in range(rounds):
            order = list(functions)
            rng.shuffle(order)
            for name in order:
                graph, start, end = captures[name]
                graph.replay()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000)
    finally:
        torch.cuda.synchronize()
        for graph, _, _ in captures.values():
            graph.reset()
    return {name: {"median_us": statistics.median(values), "min_us": min(values),
                   "max_us": max(values), "samples_us": values}
            for name, values in samples.items()}


def candidate(lab, name, x, w, out):
    if name.startswith("warp_"):
        threads, vector = name.removeprefix("warp_t").split("_v")
        return lab.warp_matmul(x, w, int(threads), int(vector), out)
    if name.startswith("blocked_"):
        return lab.matmul(x, w, int(name.removeprefix("blocked_r")), 128, out)
    if name.startswith("tile_"):
        return lab.tiled_matmul(x, w, name.removeprefix("tile_"), out)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/cublas-challenge-gemv.json")
    parser.add_argument("--output", default="results/cublas-validation.json")
    args = parser.parse_args()
    source = json.loads((ROOT / args.input).read_text())
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must stay inside workspace")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab, vendor = kernel_lab.default_lab(), CuBlas()
    report = {"status": "running", "seeds": [803, 829, 862], "cases": [],
              "gpu_before": gpu_state(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("kernels.cu", "kernel_lab.py", "cublas_baseline.py",
                                             "validate_challenge.py", "bench.py")},
              "warm_timing": "CUDA graph replay + CUDA events",
              "cold_timing": "64 MiB eviction then external graph events around operation only",
              "precision": "FP32; TF32 off; cuBLAS default math"}
    # Include wins AND a large streaming case where no sizable gain was found.
    shapes = {(1, 256, 1024), (1, 768, 768), (1, 1024, 1024),
              (1, 256, 4096), (4, 1024, 1024), (1, 4096, 4096)}
    for case in source["cases"]:
        m, n, k = case["shape_mnk"]
        if (m, n, k) not in shapes:
            continue
        selected = case["selected"]
        for seed in report["seeds"]:
            torch.manual_seed(seed)
            x = torch.randn(m, k, device="cuda") / math.sqrt(k)
            w = torch.randn(n, k, device="cuda")
            out = {name: torch.empty(m, n, device="cuda") for name in
                   ("torch_mm", "cublas_sgemm", "cublas_sgemv", "torch_mv", "candidate")}
            functions = {
                "torch_mm": lambda: torch.mm(x, w.T, out=out["torch_mm"]),
                "cublas_sgemm": lambda: vendor.mm(x, w, out["cublas_sgemm"]),
                "candidate": lambda: candidate(lab, selected, x, w, out["candidate"]),
            }
            if m == 1:
                functions["cublas_sgemv"] = lambda: vendor.mv(x, w, out["cublas_sgemv"])
                functions["torch_mv"] = lambda: torch.mv(w, x[0], out=out["torch_mv"][0]).view(1, n)
            reference = (x.double() @ w.double().T).float()
            for tensor in out.values():
                tensor.fill_(float("nan"))
            errors = {name: check(fn(), reference) for name, fn in functions.items()}
            warm = measure(functions, rounds=11, repeats=128 if n * k < 4194304 else 16)
            cold = cold_measure(functions)
            # Validate captured results, not only the eager call before capture.
            captured_errors = {name: check(out[name], reference) for name in functions}
            def speedup(timing):
                best = min(timing[name]["median_us"] for name in functions if name != "candidate")
                return best / timing["candidate"]["median_us"]
            item = {"shape_mnk": [m, n, k], "selected": selected, "seed": seed,
                    "correctness": errors, "captured_correctness": captured_errors,
                    "warm": warm, "cold": cold, "warm_speedup": speedup(warm),
                    "cold_speedup": speedup(cold), "gpu_after_case": gpu_state()}
            report["cases"].append(item)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{m}x{n}x{k} seed={seed}: warm {item['warm_speedup']:.3f}x, "
                  f"cold {item['cold_speedup']:.3f}x", flush=True)
    report["status"] = "complete"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    vendor.close()


if __name__ == "__main__":
    main()
