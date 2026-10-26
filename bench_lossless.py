"""Strict FP32 benchmark: bitwise-lossless weights and FP32 arithmetic."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import lossless_weights
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas
from validate_challenge import cold_measure

ROOT = Path(__file__).resolve().parent


def verify():
    count = 0
    for seed in (751, 761):
        torch.manual_seed(seed)
        for n, k in ((5, 64), (65, 256), (17, 512)):
            w = torch.randn(n, k, device="cuda")
            q = lossless_weights.pack_weights(w)
            if not torch.equal(q.unpack().view(torch.int32), w.view(torch.int32)):
                raise AssertionError("Lossless weight round trip failed")
            count += 1
            x = torch.randn(3, k, device="cuda") / math.sqrt(k)
            expected = (x.double() @ w.double().T).float()
            for threads in (32, 64, 128, 256):
                check(q.linear(x, threads), expected)
                count += 1
    # Exercise raw-wide exponent fallback, signed zeros, infinities, NaN payloads,
    # subnormals, and arbitrary significand bits. No floating-point arithmetic.
    special_bits = torch.randint(-(2**31), 2**31 - 1, (5, 256), device="cuda", dtype=torch.int32)
    special_bits[0, :8] = torch.tensor([0, -(2**31), 1, 0x007fffff, 0x7f800000,
                                      -8388608, 0x7fc12345, -0x003abcde], device="cuda", dtype=torch.int32)
    special = special_bits.view(torch.float32)
    q_special = lossless_weights.pack_weights(special)
    if not torch.equal(q_special.unpack().view(torch.int32), special_bits):
        raise AssertionError("Wide exponent/special-value round trip failed")
    count += 1
    compiled = torch.compile(lossless_weights.lossless_linear, fullgraph=True)
    check(compiled(x, q.mantissa, q.exponents, q.metadata, 128), q.linear(x))
    return count + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/lossless-projections.json")
    parser.add_argument("--seed", type=int, default=851)
    parser.add_argument("--frozen-from")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must remain inside workspace")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = verify()
    print(f"Lossless verification: {checks} checks passed", flush=True)
    frozen = {}
    if args.frozen_from:
        frozen = {tuple(c["shape_mnk"]): c["selected_threads"] for c in json.loads((ROOT / args.frozen_from).read_text())["cases"]}
    vendor = CuBlas()
    report = {"status": "running", "seed": args.seed, "checks": checks, "cases": [],
              "precision": "Original FP32 weight bits, FP32 activations/accumulation/output; TF32 off",
              "gpu_before": gpu_state(),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("lossless_kernels.cu", "lossless_weights.py", "bench_lossless.py", "kernel_lab.py")}}
    for n, k in ((4096, 4096), (11008, 4096), (4096, 11008)):
        torch.manual_seed(args.seed)
        w = torch.randn(n, k, device="cuda")
        x = torch.randn(1, k, device="cuda") / math.sqrt(k)
        q = lossless_weights.pack_weights(w)
        if not torch.equal(q.unpack().view(torch.int32), w.view(torch.int32)):
            raise AssertionError("Full-size bitwise round trip failed")
        reference = (x.double() @ w.double().T).float()
        outputs = {t: torch.empty(1, n, device="cuda") for t in (32, 64, 128, 256)}
        candidates = {t: (lambda t=t: lossless_weights.fused_linear(x, q, t, outputs[t])) for t in outputs}
        for fn in candidates.values():
            check(fn(), reference)
        sweep = {} if frozen else measure(candidates, rounds=3, repeats=8)
        best = frozen.get((1, n, k)) or min(sweep, key=lambda t: sweep[t]["median_us"])
        outs = {s: torch.empty(1, n, device="cuda") for s in ("torch_mm", "torch_mv", "cublas_mm", "cublas_mv")}
        functions = {"torch_mm": lambda: torch.mm(x, w.T, out=outs["torch_mm"]),
                     "torch_mv": lambda: torch.mv(w, x[0], out=outs["torch_mv"][0]).view(1, n),
                     "cublas_mm": lambda: vendor.mm(x, w, outs["cublas_mm"]),
                     "cublas_mv": lambda: vendor.mv(x, w, outs["cublas_mv"]),
                     "lossless": candidates[best]}
        warm = measure(functions, rounds=9, repeats=16)
        cold = cold_measure(functions, rounds=15)
        correctness = check(outputs[best], reference)
        relative_error = ((outputs[best].double() - reference.double()).norm() / reference.double().norm()).item()
        def speedup(timing):
            baseline = min(v["median_us"] for s, v in timing.items() if s != "lossless")
            return baseline / timing["lossless"]["median_us"]
        case = {"shape_mnk": [1, n, k], "selected_threads": best, "sweep": sweep,
                "warm": warm, "cold": cold, "warm_speedup": speedup(warm), "cold_speedup": speedup(cold),
                "bitwise_weight_round_trip": True, "arithmetic_correctness": correctness,
                "relative_l2_arithmetic_error_vs_fp64": relative_error,
                "storage_bytes": q.storage_bytes, "original_bytes": w.numel() * 4,
                "compression_ratio": w.numel() * 4 / q.storage_bytes,
                "wide_block_fraction": (q.metadata & 1).float().mean().item(), "gpu_after_case": gpu_state()}
        report["cases"].append(case)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"1x{n}x{k}: lossless {cold['lossless']['median_us']:.2f}us, {speedup(cold):.3f}x, "
              f"storage {case['compression_ratio']:.3f}x, arithmetic L2 {relative_error:.3g}", flush=True)
    vendor.close()
    report["status"] = "complete"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
