"""Reproducible, bounded experiments on the local GPU. Run with .venv/bin/python.

Use CUDA graph replay for ALL paths, randomized method order, and multiple
timing samples. These are warm, resident-input operator/subgraph benchmarks,
not LLM token-rate measurements or a run of the full FastKernels suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import kernel_lab
import torch

ROOT = Path(__file__).resolve().parent


def norm_reference(x, residual, weight):
    z = x + residual
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)) * weight


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,"
        "power.limit,clocks.current.sm,clocks.current.memory,"
        "clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown",
        "--format=csv,noheader",
    ], text=True).strip()


def check(actual, reference, atol=1e-5, rtol=1e-4):
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    delta = actual.double() - reference.double()
    return {
        "max_abs_error": delta.abs().max().item(),
        "relative_l2_error": (delta.norm() / reference.double().norm().clamp_min(1e-30)).item(),
    }


def measure(functions, rounds=7, repeats=32, phase_callback=None, round_callback=None):
    """Graph replay excludes Python/FFI dispatch and allocation overhead equally.

    Compilation and warmup finish before capture. Fixed addresses keep inputs
    resident; every captured operation still computes and overwrites output.
    The timer reports per-call GPU latency, including gaps between GPU kernels.
    """
    phase = phase_callback or (lambda _: None)
    phase("capture")
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name, fn in functions.items():
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(repeats):
                fn()
        stream.synchronize()
        graphs[name] = graph
    timings = {name: [] for name in functions}
    generator = random.Random(2026)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        phase("warmup")
        for _ in range(2):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        phase("measurement")
        for round_index in range(rounds):
            order = list(graphs)
            generator.shuffle(order)
            for name in order:
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) * 1000 / repeats)
            if round_callback is not None:
                round_callback(round_index)
    finally:
        torch.cuda.synchronize()
        phase("finished")
        for graph in graphs.values():
            graph.reset()
    return {
        name: {
            "median_us": statistics.median(samples),
            "min_us": min(samples), "max_us": max(samples), "samples_us": samples,
        }
        for name, samples in timings.items()
    }


def correctness(lab):
    """Held-out values and ragged dimensions exercise masks and reductions."""
    count = 0
    for seed in (17, 89):
        torch.manual_seed(seed)
        for m, n, k in ((1, 1, 1), (3, 65, 513), (1, 129, 1025), (2, 17, 8192), (3, 33, 4096)):
            x = torch.randn(m, k, device="cuda") / math.sqrt(k)
            w = torch.randn(n, k, device="cuda")
            reference = (x.double() @ w.double().T).float()
            out = torch.empty(m, n, device="cuda")
            for rows in (1, 2, 4, 8):
                for threads in (64, 128, 256):
                    out.fill_(float("nan"))
                    check(lab.matmul(x, w, rows, threads, out), reference)
                    count += 1
            if m == 1:
                out.fill_(float("nan"))
                check(lab.matmul(x, w, out=out, serial=True), reference)
                count += 1
            for tile in ("m8_n32", "m16_n64", "m32_n32"):
                out.fill_(float("nan"))
                check(lab.tiled_matmul(x, w, tile, out), reference)
                count += 1
            for threads in (32, 64, 128, 256):
                for vector in ((1, 4) if k % 4 == 0 else (1,)):
                    out.fill_(float("nan"))
                    check(lab.warp_matmul(x, w, threads, vector, out), reference)
                    count += 1
            residual = torch.randn_like(x)
            weight = torch.randn(k, device="cuda")
            expected = norm_reference(x, residual, weight)
            for threads in (128, 256):
                out_norm = torch.full_like(x, float("nan"))
                check(lab.residual_rmsnorm(x, residual, weight, threads=threads, out=out_norm),
                      expected, atol=1e-5, rtol=1e-5)
                count += 1
        for length in (1, 31, 1025, 1048576):
            a = torch.randn(length, device="cuda")
            b = torch.randn_like(a)
            check(lab.add(a, b), a + b, atol=0, rtol=0)
            count += 1
    # Verify Python's guard rails before exposing arbitrary device pointers.
    x = torch.ones(1, 16, device="cuda")
    w = torch.ones(1, 16, device="cuda")
    invalid_calls = (
        lambda: lab.matmul(x.cpu(), w),
        lambda: lab.matmul(x.half(), w.half()),
        lambda: lab.matmul(x[:, ::2], w[:, ::2]),
        lambda: lab.add(x, x, out=x),
        lambda: lab.matmul(x.requires_grad_(), w),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            count += 1
        else:
            raise AssertionError("Invalid input accepted")
    # Cross-stream execution: no host copy and no global synchronize in wrapper.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(1025, device="cuda")
        b = torch.randn_like(a)
        out = lab.add(a, b)
        expected = a + b
    torch.cuda.current_stream().wait_stream(stream)
    check(out, expected, atol=0, rtol=0)
    return count + 1


def custom_matmul(lab, x, w, config, out=None):
    if config.startswith("tile_"):
        return lab.tiled_matmul(x, w, config.removeprefix("tile_"), out)
    rows, threads = config.split("_t")
    return lab.matmul(x, w, int(rows[1:]), int(threads), out)


def bench_matmul(lab, m, n, k):
    torch.manual_seed(100)
    x = torch.randn(m, k, device="cuda") / math.sqrt(k)
    w = torch.randn(n, k, device="cuda")
    reference = (x.double() @ w.double().T).float()
    outputs = {}
    functions = {}
    configs = []
    if m <= 16:
        configs += [f"r{r}_t{t}" for r in (1, 2, 4, 8) for t in (64, 128, 256)]
    if m > 1:
        configs += [f"tile_{tile}" for tile in ("m8_n32", "m16_n64", "m32_n32")]
    for name in configs:
        outputs[name] = torch.empty(m, n, device="cuda")
        functions[name] = lambda c=name, o=outputs[name]: custom_matmul(lab, x, w, c, o)
        check(functions[name](), reference)
    sweep = measure(functions, rounds=3, repeats=8 if m > 16 else 16)
    selected = min(sweep, key=lambda name: sweep[name]["median_us"])
    # New values for final correctness and timing; never cache computed outputs.
    torch.manual_seed(101)
    x.normal_().div_(math.sqrt(k))
    w.normal_()
    reference = (x.double() @ w.double().T).float()
    baseline_out = torch.empty(m, n, device="cuda")
    initial_out = torch.empty_like(baseline_out)
    methods = {"torch_mm": lambda: torch.mm(x, w.T, out=baseline_out),
               "cuda_initial": lambda: lab.matmul(x, w, rows=1, threads=128, out=initial_out),
               "cuda_tuned": functions[selected]}
    errors = {name: check(fn(), reference) for name, fn in methods.items()}
    timing = measure(methods, repeats=2 if m > 16 else 32)
    speedup = timing["torch_mm"]["median_us"] / timing["cuda_tuned"]["median_us"]
    print(f"matmul M={m} N={n} K={k}: torch={timing['torch_mm']['median_us']:.2f}us "
          f"CUDA={timing['cuda_tuned']['median_us']:.2f}us ({speedup:.3f}x), {selected}", flush=True)
    return {"operation": "matmul", "shape_mnk": [m, n, k], "dtype": "float32",
            "selected_config": selected, "tuning": sweep, "timing": timing,
            "speedup_vs_torch": speedup, "correctness": errors}


def bench_norm(lab, m, k, compiled):
    torch.manual_seed(303)
    x = torch.randn(m, k, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(k, device="cuda")
    out = torch.empty_like(x)
    functions = {
        "torch_eager": lambda: norm_reference(x, residual, weight),
        "torch_compile": lambda: compiled(x, residual, weight),
        "cuda_cpp": lambda: lab.residual_rmsnorm(x, residual, weight, out=out),
    }
    expected = norm_reference(x, residual, weight)
    errors = {name: check(fn(), expected, atol=1e-5, rtol=1e-5) for name, fn in functions.items()}
    timing = measure(functions)
    speedup = timing["torch_eager"]["median_us"] / timing["cuda_cpp"]["median_us"]
    vs_compile = timing["torch_compile"]["median_us"] / timing["cuda_cpp"]["median_us"]
    print(f"residual RMSNorm M={m} K={k}: eager={timing['torch_eager']['median_us']:.2f}us "
          f"compiled={timing['torch_compile']['median_us']:.2f}us "
          f"C++/CUDA={timing['cuda_cpp']['median_us']:.2f}us ({vs_compile:.3f}x vs compiled)", flush=True)
    return {"operation": "residual_rmsnorm", "shape_mk": [m, k], "dtype": "float32",
            "timing": timing, "speedup_vs_eager": speedup,
            "speedup_vs_compiled": vs_compile, "correctness": errors}


def bench_composed(lab, m, n, k, config):
    """Measure a norm+linear subgraph; this is not full-model inference."""
    torch.manual_seed(404)
    x = torch.randn(m, k, device="cuda") / math.sqrt(k)
    residual = torch.randn_like(x) / math.sqrt(k)
    gamma = torch.randn(k, device="cuda")
    w = torch.randn(n, k, device="cuda") / math.sqrt(k)
    intermediate = torch.empty_like(x)
    out = torch.empty(m, n, device="cuda")

    def reference():
        return norm_reference(x, residual, gamma) @ w.T

    def fused_norm():
        norm = lab.residual_rmsnorm(x, residual, gamma, out=intermediate)
        return torch.mm(norm, w.T, out=out)

    def both_custom():
        norm = lab.residual_rmsnorm(x, residual, gamma, out=intermediate)
        return custom_matmul(lab, norm, w, config, out)

    compiled_reference = torch.compile(reference, fullgraph=True)
    functions = {"torch_eager": reference, "torch_compile": compiled_reference,
                 "custom_norm_torch_mm": fused_norm, "custom_norm_custom_mm": both_custom}
    expected = reference()
    errors = {name: check(fn(), expected, atol=2e-5, rtol=1e-4) for name, fn in functions.items()}
    timing = measure(functions, repeats=16)
    print(f"norm+linear M={m}: " + ", ".join(
        f"{name}={value['median_us']:.2f}us" for name, value in timing.items()), flush=True)
    return {"operation": "norm_plus_linear_subgraph", "shape_mnk": [m, n, k],
            "timing": timing, "correctness": errors}


def tensor_diagnostic(size):
    """Compare compute-heavy and decode-like matmuls at fixed accumulation policy.

    Dtypes are separate diagnostic workloads, never counted as optimization
    speedups over FP32. This does not inspect or change GPU speed selectors.
    """
    result = []
    for m in (1, size):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(m, size, device="cuda", dtype=dtype)
            w = torch.randn(size, size, device="cuda", dtype=dtype)
            out = torch.empty(m, size, device="cuda", dtype=dtype)
            timing = measure({"torch_mm": lambda: torch.mm(x, w.T, out=out)}, rounds=5, repeats=8)
            us = timing["torch_mm"]["median_us"]
            result.append({"shape_mnk": [m, size, size], "dtype": str(dtype),
                           "tflops": 2 * m * size * size / us / 1e6, "timing": timing})
            print(f"tensor diagnostic M={m} K=N={size} {dtype}: {us:.2f}us", flush=True)
    return {"operation": "tensor_core_diagnostic", "cases": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/baseline.json")
    parser.add_argument("--quick", action="store_true", help="Use fewer large shapes")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the workspace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "compute_capability": torch.cuda.get_device_capability(),
        "gpu_before": gpu_state(), "timing_method": "CUDA graph replay + CUDA events",
        "timing_scope": "GPU execution only; Python/FFI and compilation excluded",
        "input_mode": "warm fixed addresses; recomputed outputs; no explicit L2 flush",
        "matmul_policy": "FP32 without TF32; FP16/BF16 reduced-precision reduction disabled",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels.cu", "kernel_lab.py", "native/driver.cpp", "bench.py")},
        "results": [],
        "status": "running",
    }

    def checkpoint(result=None):
        if result is not None:
            result["gpu_after_case"] = gpu_state()
            report["results"].append(result)
        destination.write_text(json.dumps(report, indent=2) + "\n")

    lab = kernel_lab.default_lab()
    report["correctness_checks"] = correctness(lab)
    checkpoint()
    print(f"Correctness: {report['correctness_checks']} checks passed", flush=True)
    # Warm clocks using actual GPU work before comparing implementations.
    warm = torch.randn(1024, 1024, device="cuda")
    start = time.monotonic()
    while time.monotonic() - start < 0.5:
        torch.mm(warm, warm)
        torch.cuda.synchronize()
    shapes = [(1, 4096, 4096), (4, 4096, 4096), (128, 4096, 4096)]
    if not args.quick:
        shapes += [(1, 11008, 4096), (1, 4096, 11008)]
    for shape in shapes:
        checkpoint(bench_matmul(lab, *shape))
    compiled_norm = torch.compile(norm_reference, fullgraph=True)
    for m in ((1, 128) if args.quick else (1, 128, 1024)):
        checkpoint(bench_norm(lab, m, 4096, compiled_norm))
    for result in list(report["results"]):
        if result["operation"] == "matmul" and result["shape_mnk"][0] in (1, 128) and result["shape_mnk"][1:] == [4096, 4096]:
            checkpoint(bench_composed(lab, *result["shape_mnk"], result["selected_config"]))
    # Prove that Python's custom-op boundary survives fullgraph compilation.
    x = torch.randn(3, 513, device="cuda")
    w = torch.randn(65, 513, device="cuda")
    eager = kernel_lab.matmul(x, w)
    compiled = torch.compile(kernel_lab.matmul, fullgraph=True)
    check(compiled(x, w), eager)
    report["torch_compile_custom_op"] = "passed fullgraph=True, ragged dimensions"
    checkpoint(tensor_diagnostic(2048 if args.quick else 4096))
    report["gpu_after"] = gpu_state()
    report["status"] = "complete"
    checkpoint()
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
