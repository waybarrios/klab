"""Test the INT8 bandwidth/accuracy frontier against native FP16 cuBLAS."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import int8_weights
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas
from validate_challenge import cold_measure
from experiment_control import with_run_metadata

ROOT = Path(__file__).resolve().parent


def verify_int8():
    checks = 0
    for seed in (761, 781):
        torch.manual_seed(seed)
        for n, k in ((5, 64), (65, 256), (17, 512)):
            q = int8_weights.quantize_weights(torch.randn(n, k, device="cuda"))
            decoded = q.dequantize()
            for dtype in (torch.float16, torch.float32):
                x = (torch.randn(3, k, device="cuda") / math.sqrt(k)).to(dtype)
                expected = (x.double() @ decoded.double().T).float()
                for threads in (32, 64, 128, 256):
                    check(q.linear(x, threads), expected)
                    checks += 1
    compiled = torch.compile(int8_weights.int8_linear, fullgraph=True)
    for dtype in (torch.float16, torch.float32):
        check(compiled(x.to(dtype), q.codes, q.scales, 128), q.linear(x.to(dtype)))
        checks += 1
    q = int8_weights.quantize_weights(torch.zeros(5, 64, device="cuda"))
    check(q.linear(torch.ones(1, 64, device="cuda")), torch.zeros(1, 5, device="cuda"), atol=0, rtol=0)
    return checks + 1


def main(run_metadata=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--output", default="results/int8-projections.json")
    parser.add_argument("--frozen-from")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must remain inside workspace")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    checks = verify_int8()
    print(f"INT8 verification: {checks} checks passed", flush=True)
    frozen = {}
    if args.frozen_from:
        frozen = {tuple(c["shape_mnk"]): c["selected_threads"] for c in json.loads((ROOT / args.frozen_from).read_text())["cases"]}
    vendor = CuBlas()
    report = {"status": "running", "seed": args.seed, "frozen_from": args.frozen_from,
              "checks": checks, "cases": [], "gpu_before": gpu_state(),
              "format": "signed INT8 + FP32 scale/64; FP16 activations; FP32 accumulation/output",
              "quantization_in_timing": False,
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("int8_kernels.cu", "int8_weights.py", "bench_int8.py", "kernel_lab.py")}}
    report = with_run_metadata(report, run_metadata)
    for n, k in ((4096, 4096), (11008, 4096), (4096, 11008)):
        torch.manual_seed(args.seed)
        original = torch.randn(n, k, device="cuda")
        q = int8_weights.quantize_weights(original)
        decoded = q.dequantize()
        x16 = (torch.randn(1, k, device="cuda") / math.sqrt(k)).half()
        original16 = original.half()
        outputs, candidates = {}, {}
        for threads in (32, 64, 128, 256):
            outputs[threads] = torch.empty(1, n, device="cuda")
            candidates[threads] = lambda t=threads: int8_weights.fused_linear(x16, q.codes, q.scales, t, outputs[t])
        expected = (x16.double() @ decoded.double().T).float()
        for fn in candidates.values():
            check(fn(), expected)
        sweep = {} if frozen else measure(candidates, rounds=3, repeats=16)
        best = frozen.get((1, n, k)) or min(sweep, key=lambda t: sweep[t]["median_us"])
        out_native = torch.empty(1, n, device="cuda")
        out_half = torch.empty(1, n, device="cuda", dtype=torch.float16)
        out_fp32 = torch.empty(1, n, device="cuda")
        x32 = x16.float()
        functions = {
            "cublas_fp16_acc32": lambda: vendor.mm_half(x16, original16, out_native),
            "torch_fp16": lambda: torch.mm(x16, original16.T, out=out_half).float(),
            "cublas_fp32": lambda: vendor.mm(x32, original, out_fp32),
            "int8_acc32": candidates[best],
        }
        warm = measure(functions, rounds=9, repeats=16)
        cold = cold_measure(functions, rounds=15)
        correctness = check(outputs[best], expected)
        check(out_native, (x16.double() @ original16.double().T).float())
        original_ref = (x16.double() @ original.double().T).float()
        output_error = ((outputs[best] - original_ref).norm() / original_ref.norm()).item()
        fp16_error = ((out_native - original_ref).norm() / original_ref.norm()).item()
        def speedup(timing):
            baseline = min(timing[s]["median_us"] for s in ("cublas_fp16_acc32", "torch_fp16"))
            return baseline / timing["int8_acc32"]["median_us"]
        item = {"shape_mnk": [1, n, k], "selected_threads": best, "sweep": sweep,
                "warm": warm, "cold": cold, "warm_speedup_vs_fp16": speedup(warm),
                "cold_speedup_vs_fp16": speedup(cold), "arithmetic_correctness": correctness,
                "output_relative_l2_error": output_error,
                "fp16_output_relative_l2_error": fp16_error,
                "weight_relative_l2_error": ((decoded - original).norm() / original.norm()).item(),
                "weight_bytes": q.codes.numel() + q.scales.numel() * 4,
                "gpu_after_case": gpu_state()}
        report["cases"].append(item)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"1x{n}x{k}: INT8 {cold['int8_acc32']['median_us']:.2f}us, {speedup(cold):.3f}x vs FP16; "
              f"output relative L2 error {output_error:.3%}", flush=True)
    vendor.close()
    report["status"] = "complete"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
