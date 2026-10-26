"""GPU-only FP4 projection experiment: speed and quantization error separately."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import fp4
import torch
from bench import check, gpu_state, measure
from cublas_baseline import CuBlas
from validate_challenge import cold_measure
from experiment_control import with_run_metadata

ROOT = Path(__file__).resolve().parent


def verify_fp4():
    count = 0
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    codes = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(1, 16)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    q = fp4.FP4Weights(packed.contiguous(), torch.tensor([[56]], dtype=torch.uint8, device="cuda"),
                       torch.ones((), device="cuda"))
    x = torch.eye(16, device="cuda")
    for dtype in (torch.float16, torch.float32):
        for threads in (32, 64, 128, 256):
            actual = q.linear(x.to(dtype), threads)
            check(actual[:, 0], table, atol=0, rtol=0)
            count += 1
    for seed in (761, 781):
        torch.manual_seed(seed)
        for n, k in ((5, 16), (65, 256), (17, 512)):
            w = torch.randn(n, k, device="cuda")
            q = fp4.quantize_weights(w)
            decoded = q.dequantize()
            for dtype in (torch.float16, torch.float32):
                x = (torch.randn(3, k, device="cuda") / math.sqrt(k)).to(dtype)
                expected = (x.double() @ decoded.double().T).float()
                for threads in (32, 64, 128, 256):
                    check(q.linear(x, threads), expected, atol=1e-5, rtol=1e-4)
                    count += 1
    zero = fp4.quantize_weights(torch.zeros(5, 16, device="cuda"))
    check(zero.linear(torch.ones(1, 16, device="cuda")), torch.zeros(1, 5, device="cuda"), atol=0, rtol=0)
    count += 1
    compiled = torch.compile(fp4.fp4_linear, fullgraph=True)
    check(compiled(x, q.packed, q.scales, q.global_scale, 128), q.linear(x, 128))
    count += 1
    check(compiled(x.half(), q.packed, q.scales, q.global_scale, 128), q.linear(x.half(), 128))
    count += 1
    tiny = fp4.quantize_weights(torch.full((5, 16), 1e-40, device="cuda"))
    if not torch.isfinite(tiny.dequantize()).all():
        raise AssertionError("Subnormal input produced non-finite decoded weights")
    count += 1
    return count


def main(run_metadata=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/fp4-projections.json")
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--frozen-from", help="Use previously selected block settings without retuning")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must stay inside workspace")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    checks = verify_fp4()
    print(f"FP4 verification: {checks} checks passed", flush=True)
    vendor = CuBlas()
    frozen = {}
    if args.frozen_from:
        previous = json.loads((ROOT / args.frozen_from).read_text())
        frozen = {tuple(c["shape_mnk"]): c["selected"] for c in previous["cases"]}
    report = {"status": "running", "device": torch.cuda.get_device_name(), "correctness_checks": checks,
              "gpu_before": gpu_state(), "format": "E2M1 + E4M3 scale/16 + FP32 global; custom layout",
              "compute": "Software decode; FP16/FP32 activations; FP32 FMA accumulation/output",
              "quantization_in_timing": False, "seed": args.seed,
              "frozen_from": args.frozen_from, "cases": [],
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("kernels.cu", "projection_kernels.cu", "fp4_kernels.cu",
                                          "kernel_lab.py", "fp4.py", "bench_fp4.py", "cublas_baseline.py")}}
    report = with_run_metadata(report, run_metadata)
    for n, k in ((4096, 4096), (11008, 4096), (4096, 11008)):
        torch.manual_seed(args.seed)
        original = torch.randn(n, k, device="cuda")
        q = fp4.quantize_weights(original)
        decoded = q.dequantize()
        x16 = (torch.randn(1, k, device="cuda") / math.sqrt(k)).half()
        x32 = x16.float()  # Identical activation values in the FP16/FP32 controls.
        original16, decoded16 = original.half(), decoded.half()
        reference = (x16.double() @ decoded.double().T).float()
        functions, outputs = {}, {}
        for bits, x in ((16, x16), (32, x32)):
            for threads in (32, 64, 128, 256):
                name = f"w4a{bits}_t{threads}"
                outputs[name] = torch.empty(1, n, device="cuda")
                functions[name] = lambda xx=x, t=threads, o=outputs[name]: fp4.fused_linear(xx, q.packed, q.scales, q.global_scale, t, o)
                expected = (x.double() @ decoded.double().T).float()
                check(functions[name](), expected, atol=1e-5, rtol=1e-4)
        sweep = {} if frozen else measure(functions, rounds=3, repeats=16)
        best = frozen.get((1, n, k)) or min(
            (s for s in functions if s.startswith("w4a16")), key=lambda s: sweep[s]["median_us"])
        selected = functions[best]
        out32, out16, outq16 = [torch.empty(1, n, device="cuda") for _ in range(3)]
        torch_out16 = torch.empty(1, n, device="cuda", dtype=torch.float16)
        methods = {
            "original_fp32_cublas": lambda: vendor.mm(x32, original, out32),
            "original_fp16_cublas_acc32": lambda: vendor.mm_half(x16, original16, out16),
            "decoded_fp16_cublas_acc32": lambda: vendor.mm_half(x16, decoded16, outq16),
            "original_fp16_torch": lambda: torch.mm(x16, original16.T, out=torch_out16).float(),
            "fused_fp4_acc32": selected,
        }
        for fn in methods.values():
            fn()
        fp16_reference = (x16.double() @ original16.double().T).float()
        check(vendor.mm_half(x16, original16, out16), fp16_reference, atol=1e-5, rtol=1e-4)
        decoded16_reference = (x16.double() @ decoded16.double().T).float()
        check(vendor.mm_half(x16, decoded16, outq16), decoded16_reference, atol=1e-5, rtol=1e-4)
        warm = measure(methods, rounds=9, repeats=16)
        cold = cold_measure(methods, rounds=15)
        captured_checks = {
            "fp4": check(outputs[best], reference, atol=1e-5, rtol=1e-4),
            "fp16_cublas": check(out16, fp16_reference, atol=1e-5, rtol=1e-4),
            "decoded_fp16_cublas": check(outq16, decoded16_reference, atol=1e-5, rtol=1e-4),
        }
        fused = selected()
        arithmetic_error = check(fused, reference, atol=1e-5, rtol=1e-4)
        original_reference = (x16.double() @ original.double().T).float()
        quant_output_error = (fused - original_reference).norm() / original_reference.norm()
        weight_error = (decoded - original).norm() / original.norm()
        size_bytes = q.packed.numel() + q.scales.numel() + 4
        def speedup(timings):
            base = min(timings[s]["median_us"] for s in ("original_fp16_cublas_acc32", "original_fp16_torch"))
            return base / timings["fused_fp4_acc32"]["median_us"]
        item = {"shape_mnk": [1, n, k], "selected": best, "sweep": sweep, "warm": warm, "cold": cold,
                "warm_speedup_vs_native_fp16": speedup(warm), "cold_speedup_vs_native_fp16": speedup(cold),
                "cold_speedup_vs_fp32": cold["original_fp32_cublas"]["median_us"] / cold["fused_fp4_acc32"]["median_us"],
                "arithmetic_correctness_vs_decoded": arithmetic_error,
                "captured_correctness": captured_checks,
                "weight_relative_l2_quantization_error": weight_error.item(),
                "output_relative_l2_quantization_error": quant_output_error.item(),
                "packed_weight_bytes": size_bytes, "fp16_weight_bytes": original16.numel() * 2,
                "gpu_after_case": gpu_state()}
        report["cases"].append(item)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"1x{n}x{k}: FP4 {cold['fused_fp4_acc32']['median_us']:.2f}us, "
              f"{item['cold_speedup_vs_native_fp16']:.3f}x vs FP16, {item['cold_speedup_vs_fp32']:.3f}x vs FP32; "
              f"output quantization L2 error {item['output_relative_l2_quantization_error']:.2%}", flush=True)
    vendor.close()
    report["status"] = "complete"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
