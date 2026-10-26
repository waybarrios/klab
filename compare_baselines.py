"""Controlled baseline table on common original FP32 weights and activations.

Lossy formats are diagnostics and explicitly fail the active precision
requirement. Selection of the lossless launch configuration is frozen from
a separate earlier run. All methods use FP32 accumulation/output.
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from experiment_control import parse_shape, project_path, with_run_metadata, write_json
from result_metrics import percentage_gains

ROOT = Path(__file__).resolve().parent


def make_tables(report, destination=None, raw_source=None):
    destination = project_path(ROOT, destination or "results/COMPARISON.md")
    raw_source = project_path(ROOT, raw_source or "results/controlled-comparison.json")
    raw_link = os.path.relpath(raw_source, destination.parent)
    groups = defaultdict(list)
    for case in report["cases"]:
        groups[tuple(case["shape_mnk"])].append(case)
    if not groups:
        raise ValueError("The report has no completed comparison cases")
    lines = ["# Controlled GPU baseline comparison", "",
             "Active requirement: retain the original FP32 weight values without",
             "lossy conversion; FP32 activations, accumulation, and output.", "",
             "The same original weights and activation vector are used for all",
             "methods within each case. The FP16 diagnostic explicitly rounds both.",
             "INT8 and FP4 keep FP32 activations but quantize weights. They are",
             "excluded from the active acceptance criterion.", "",
             "## Accepted lossless path", "",
             "Median GPU times after a 64 MiB eviction-buffer operation. The buffer",
             "operation is excluded from timing. Ranges span independent input seeds,",
             "not confidence intervals. Baseline is the fastest tested FP32 vendor",
             "path, including both original and transposed/packed layouts.", "",
             "| M × N × K | Best FP32 baseline (ms) | Lossless (ms) | Time saved (%) | Throughput gain (%) | Output relative L2 error |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for shape, cases in groups.items():
        baselines = [c["best_vendor_us"] / 1000 for c in cases]
        times = [c["methods"]["lossless_fp32"]["cold"]["median_us"] / 1000 for c in cases]
        gains = [percentage_gains(c["best_vendor_us"], c["methods"]["lossless_fp32"]["cold"]["median_us"]) for c in cases]
        saved = [g["latency_reduction_pct"] for g in gains]
        throughput = [g["throughput_gain_pct"] for g in gains]
        errors = [c["methods"]["lossless_fp32"]["output_relative_l2_error"] for c in cases]
        lines.append(f"| {' × '.join(map(str, shape))} | {min(baselines):.3f}–{max(baselines):.3f} | "
                     f"{min(times):.3f}–{max(times):.3f} | {min(saved):.2f} to {max(saved):.2f} | "
                     f"{min(throughput):.2f} to {max(throughput):.2f} | "
                     f"{max(errors):.2e} |")
    lines += ["", "Time saved = 100 × (1 - kernel time / baseline time). Throughput gain",
              "= 100 × (baseline time / kernel time - 1). Negative values indicate a slowdown.", "",
              "Weight reconstruction was checked bit for bit across every weight",
              "in each case. Output differences are ordinary FP32 arithmetic rounding",
              "against the FP64 mathematical reference, not quantization error.", "",
              "## Full precision/latency tradeoff", "",
              "All rows below use the 4096 × 4096 projection from each seed. Lossy",
              "rows explain the earlier speed/accuracy tradeoff and are not accepted",
              "solutions for the active requirement.", "",
              "| Method | Weight storage | Activation | GPU time (ms) | Time saved vs FP32 (%) | Output L2 error | Meets requirement |",
              "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    example_shape = (1, 4096, 4096) if (1, 4096, 4096) in groups else next(iter(groups))
    cases = groups[example_shape]
    descriptions = {
        "vendor_best": ("Best FP32 vendor", "FP32", "FP32", True),
        "custom_fp32": ("Custom warp", "FP32", "FP32", True),
        "lossless_fp32": ("Lossless exponent packing", "All FP32 bits", "FP32", True),
        "fp16_diagnostic": ("FP16 diagnostic", "FP16 rounded", "FP16 rounded", False),
        "int8_diagnostic": ("INT8 diagnostic", "INT8 + FP32 scales", "FP32", False),
        "fp4_diagnostic": ("FP4 diagnostic", "E2M1 + FP8/FP32 scales", "FP32", False),
    }
    for key, (name, storage, activation, accepted) in descriptions.items():
        if key != "vendor_best" and key not in cases[0]["methods"]:
            continue
        rows = [c["methods"][c["best_vendor_method"] if key == "vendor_best" else key] for c in cases]
        times = [r["cold"]["median_us"] / 1000 for r in rows]
        saved = [percentage_gains(c["best_vendor_us"], r["cold"]["median_us"])["latency_reduction_pct"]
                 for c, r in zip(cases, rows)]
        errors = [r["output_relative_l2_error"] * 100 for r in rows]
        lines.append(f"| {name} | {storage} | {activation} | {min(times):.3f}–{max(times):.3f} | "
                     f"{min(saved):.2f} to {max(saved):.2f} | {min(errors):.6g}–{max(errors):.6g}% | "
                     f"{'Yes' if accepted else 'No'} |")
    lines += ["", "All outputs and accumulators in this table are FP32. Lossy speedups",
              "are precision tradeoffs, not improvements at equivalent FP32 accuracy.", "",
              "## Setup and limits", "",
              "| M × N × K | Lossless packing setup (ms) | Packed/original bytes | Amortization (calls) |",
              "| --- | ---: | ---: | ---: |"]
    for shape, cases in groups.items():
        setup = [c["packing_setup_ms"] for c in cases]
        ratio = [c["lossless_bytes"] / c["original_bytes"] * 100 for c in cases]
        amortization = [c["packing_amortization_calls"] for c in cases]
        payback = (f"{min(amortization)}–{max(amortization)}" if min(amortization) > 0 else "No payback in some/all cases")
        lines.append(f"| {' × '.join(map(str, shape))} | {min(setup):.1f}–{max(setup):.1f} | "
                     f"{min(ratio):.2f}–{max(ratio):.2f}% | {payback} |")
    lines += ["", "Packing is setup work for constant weights. Timing above includes its",
              "CPU/GPU wall time and excludes the separate bitwise validation. The",
              "operation latency excludes setup, compilation, and Python/FFI overhead.",
              "The amortization estimate uses the measured cold-condition time saved.",
              "It is workload dependent; real model traces still need measurement.", "",
              "GPU: RTX 3070 Ti Laptop, SM 8.6. Under earlier benchmark load, memory",
              "ran at 810 MHz with software power/thermal limits. Per-case GPU state",
              "is retained in the JSON. No system power, clock, driver, or firmware",
              "setting was modified. No end-to-end LLM result is claimed.", "",
              "The packed codec currently requires K divisible by 64, contiguous",
              "aligned FP32 tensors, signed 32-bit addressing, and an exponent payload",
              "under 256 MiB. The GPU decoder has a raw-exponent fallback when a block",
              "cannot use four-bit deltas; it never truncates weight information.", "",
              "Run: `./bin/klab compare` (add `--include-lossy` for precision diagnostics)", "",
              "Raw timings, source hashes, seeds, errors, and byte counts are in",
              f"[{raw_source.name}]({raw_link}). All methods run on CUDA. C++ is the host",
              "launch bridge; the custom device kernels are independently written CUDA C++."]
    text = "\n".join(lines) + "\n"
    text = text.replace("All rows below use the 4096 × 4096 projection from each seed.",
                        f"All rows below use M×N×K={'×'.join(map(str, example_shape))} from each seed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def run_comparison(shapes, seeds, rounds, include_lossy, output, threads=None, run_metadata=None):
    import kernel_lab
    import lossless_weights
    import torch
    from bench import check, gpu_state, measure
    from cublas_baseline import CuBlas
    from validate_challenge import cold_measure
    from experiment_provenance import environment,source_hashes

    output = project_path(ROOT, output)
    if rounds < 3 or not seeds or any(min(s) <= 0 or s[2] % 64 for s in shapes):
        raise ValueError("Require rounds>=3, seeds, positive M,N,K and K divisible by 64")
    if threads is not None and threads not in (32, 64, 128, 256):
        raise ValueError("Unsupported lossless block size")
    if include_lossy:
        import fp4
        import int8_weights
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab, vendor = kernel_lab.default_lab(), CuBlas()
    frozen_path = ROOT / "results/lossless-compact.json"
    frozen = ({tuple(c["shape_mnk"]): c["selected_threads"] for c in json.loads(frozen_path.read_text())["cases"]}
              if frozen_path.is_file() else {})
    report = {"status": "running", "kind": "controlled-comparison", "cases": [], "seeds": list(seeds),
              "parameters": {"shapes": list(shapes), "rounds": rounds,
                             "include_lossy": include_lossy, "threads": threads},
              "source_sha256":source_hashes(("compare_baselines.py","fp4.py","int8_weights.py","results/lossless-compact.json")),
              "environment":environment(lab,vendor),"schema_version":2}
    report = with_run_metadata(report, run_metadata)
    for seed in report["seeds"]:
        for m, n, k in shapes:
            torch.manual_seed(seed)
            w = torch.randn(n, k, device="cuda")
            x = torch.randn(m, k, device="cuda") / math.sqrt(k)
            reference64 = x.double() @ w.double().T
            torch.cuda.synchronize()
            start = time.perf_counter()
            packed = lossless_weights.pack_weights(w)
            torch.cuda.synchronize()
            setup_ms = (time.perf_counter() - start) * 1000
            bitwise = torch.equal(packed.unpack().view(torch.int32), w.view(torch.int32))
            if not bitwise:
                raise AssertionError("Lossless round trip failed")
            wt = w.T.contiguous()
            baselines = ["torch_mm", "cublas_mm", "cublas_packed_mm"]
            if m == 1:
                baselines += ["torch_mv", "cublas_mv", "cublas_packed_mv"]
            labels = baselines + ["custom_fp32", "lossless_fp32"]
            if include_lossy:
                labels += ["fp16_diagnostic", "int8_diagnostic", "fp4_diagnostic"]
                q4, q8 = fp4.quantize_weights(w), int8_weights.quantize_weights(w)
                x16, w16 = x.half(), w.half()
            out = {name: torch.empty(m, n, device="cuda") for name in labels}
            functions = {
                "torch_mm": lambda: torch.mm(x, w.T, out=out["torch_mm"]),
                "cublas_mm": lambda: vendor.mm(x, w, out["cublas_mm"]),
                "cublas_packed_mm": lambda: vendor.mm_packed(x, wt, out["cublas_packed_mm"]),
                "custom_fp32": lambda: lab.warp_matmul(x, w, 32, 4, out["custom_fp32"]),
                "lossless_fp32": lambda: lossless_weights.fused_linear(x, packed, threads or frozen.get((m, n, k), 128), out["lossless_fp32"]),
            }
            if m == 1:
                functions.update(torch_mv=lambda: torch.mv(w, x[0], out=out["torch_mv"][0]).view(1, n),
                                 cublas_mv=lambda: vendor.mv(x, w, out["cublas_mv"]),
                                 cublas_packed_mv=lambda: vendor.mv_packed(x, wt, out["cublas_packed_mv"]))
            if include_lossy:
                functions.update(fp16_diagnostic=lambda: vendor.mm_half(x16, w16, out["fp16_diagnostic"]),
                                 int8_diagnostic=lambda: int8_weights.fused_linear(x, q8.codes, q8.scales, 128, out["int8_diagnostic"]),
                                 fp4_diagnostic=lambda: fp4.fused_linear(x, q4.packed, q4.scales, q4.global_scale, 128, out["fp4_diagnostic"]))
            out["torch_mm_aa"]=torch.empty(m,n,device="cuda")
            functions["torch_mm_aa"]=lambda:torch.mm(x,w.T,out=out["torch_mm_aa"])
            for name, fn in functions.items():
                actual = fn()
                if name not in ("fp16_diagnostic", "int8_diagnostic", "fp4_diagnostic"):
                    check(actual, reference64.float())
            warm = measure(functions, rounds=max(3, rounds // 2), repeats=8)
            cold = cold_measure(functions, rounds=rounds)
            best = min(baselines, key=lambda name: cold[name]["median_us"])
            baseline_us = cold[best]["median_us"]
            methods = {}
            for name in functions:
                error = ((out[name].double() - reference64).norm() / reference64.norm()).item()
                accepted = name not in ("fp16_diagnostic", "int8_diagnostic", "fp4_diagnostic")
                if accepted:
                    check(out[name], reference64.float())
                methods[name] = {"warm": warm[name], "cold": cold[name], "output_relative_l2_error": error,
                                 "speedup_vs_best_fp32": baseline_us / cold[name]["median_us"],
                                 "meets_precision_requirement": accepted}
            saved_ms = (baseline_us - cold["lossless_fp32"]["median_us"]) / 1000
            case = {"seed": seed, "shape_mnk": [m, n, k], "best_vendor_method": best,
                    "best_vendor_us": baseline_us, "methods": methods, "bitwise_round_trip": bitwise,
                    "packing_setup_ms": setup_ms, "original_bytes": w.numel() * 4,
                    "lossless_bytes": packed.storage_bytes,
                    "packing_amortization_calls": math.ceil(setup_ms / saved_ms) if saved_ms > 0 else -1,
                    "gpu_after_case": gpu_state(),"selected_threads":threads or frozen.get((m,n,k),128),
                    "aa_time_saved_pct":percentage_gains(cold["torch_mm"]["median_us"],cold["torch_mm_aa"]["median_us"])["latency_reduction_pct"]}
            report["cases"].append(case)
            write_json(output, report)
            gains = percentage_gains(baseline_us, cold["lossless_fp32"]["median_us"])
            print(f"seed={seed} {m}x{n}x{k}: baseline {baseline_us:.1f}us, lossless {cold['lossless_fp32']['median_us']:.1f}us, "
                  f"time saved {gains['latency_reduction_pct']:+.2f}%, "
                  f"throughput {gains['throughput_gain_pct']:+.2f}%; exact weights={bitwise}", flush=True)
    vendor.close()
    report["status"] = "complete"
    write_json(output, report)
    make_tables(report, output.with_suffix(".md"), raw_source=output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=parse_shape, action="append")
    p.add_argument("--seeds", default="857,859")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--include-lossy", action="store_true")
    p.add_argument("--output", default="results/controlled-comparison.json")
    a = p.parse_args()
    run_comparison(a.shape or [(1, 4096, 4096), (1, 11008, 4096), (1, 4096, 11008)],
                   [int(s) for s in a.seeds.split(",")], a.rounds, a.include_lossy, a.output)


if __name__ == "__main__":
    main()
