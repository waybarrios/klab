"""FP16/BF16-input projection: tune on seed 901, freeze, validate on rotating rings.

Every path receives the same supplied 16-bit x and weights, accumulates in FP32,
and writes FP32. Errors are measured against FP64 arithmetic on those values.
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path

from experiment_control import GpuLock, parse_shape, project_path, report_output, run_output, with_run_metadata, write_json
from result_metrics import percentage_gains

ROOT = Path(__file__).resolve().parent
SHAPES = [(1,64,64),(1,128,128),(1,256,256),(1,512,512),(1,1024,1024),(4,256,256)]
TUNING_SEED = 901
MIN_ROUNDS = 31
FORMATS = {"fp16": ("float16", 2), "bf16": ("bfloat16", 14)}  # torch dtype, cudaDataType
CANDIDATES = [(t, v) for v in (2, 8) for t in (32, 64, 128, 256)]
FIXED = "cublas_compute32"
ATOL, RTOL = 1e-5, 1e-4
SOURCES = ("bench_half.py", "half_projection.py", "half_kernels.cu", "kernels.cu", "kernel_lab.py",
           "native/driver.cpp", "cublas_baseline.py", "bench.py", "bench_streaming.py", "gpu_monitor.py")
PRECISION = ("x[M,K], w[N,K]: identical supplied FP16 or BF16 values, unchanged by every path; exact widening "
             "to FP32, FP32 accumulation, FP32 output [M,N]; errors vs FP64 on the same values. Vendor routes use "
             "CUBLAS_COMPUTE_32F with FP32 output. TF32, FP16 accumulation and reduced-precision reductions disabled.")
INPUTS = ("Test-data preparation, outside the operation: weights sampled by torch.randn directly in the target "
          "dtype; x sampled in FP32, scaled by 1/sqrt(K), then rounded once to the target dtype. The FP32-source "
          "rounding diagnostic is separate and labelled.")
PROTOCOL = ("Each path cycles through a ring of >=max(80 MiB, 20x L2) of 16-bit weights (cuBLAS packed route: "
            "bitwise-identical transposed ring of the same size); the whole CUDA graph is timed and divided by "
            "the true call count. Random method order, >=31 rounds, no eviction kernel; GPU telemetry sampled.")
SELECTION = "Lowest rotating-ring median among the threads x vector candidates on seed 901; frozen afterwards."
ACCEPTANCE = ("Noise-gated win: numerics pass, |A/A| <= 1%, bootstrap 95% lower bound of time saved vs best "
              "equivalent vendor > max(0, |A/A|). User target: same gates and lower bound > 10%. Only held-out "
              "validation cases count; tuning cases are in-sample.")
UNLOCK = "Stock configuration: this benchmark does not load or query gpu-unlock; no unlock effects are claimed."


def _name(threads, vector):
    return f"t{threads}_v{vector}"


def _sha(name):
    path = ROOT / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _packed_gemm(vendor, x, wt, out, storage):
    """GemmEx with pre-transposed wt[K,N]: column-major C[N,M] = wt @ x, FP32 compute/output."""
    import torch
    m, k = x.shape
    n = wt.shape[1]
    vendor.check(vendor.lib.cublasSetStream_v2(
        vendor.handle, ct.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)))
    vendor.check(vendor.lib.cublasGemmEx(
        vendor.handle, 0, 0, n, m, k, ct.byref(vendor.alpha), ct.c_void_p(wt.data_ptr()), storage, n,
        ct.c_void_p(x.data_ptr()), storage, k, ct.byref(vendor.beta), ct.c_void_p(out.data_ptr()), 0, n, 68, -1))
    return out


def _torch_out_fp32_status(torch, dtype):
    """One eager check, outside timing, that torch.mm can write FP32 from 16-bit inputs in place."""
    x = torch.randn(2, 64, device="cuda").to(dtype)
    w = torch.randn(8, 64, device="cuda").to(dtype)
    out = torch.empty(2, 8, device="cuda")
    try:
        result = torch.mm(x, w.T, out_dtype=torch.float32, out=out)
        torch.cuda.synchronize()
    except (TypeError, RuntimeError, NotImplementedError) as error:
        return f"unavailable: {type(error).__name__}: {(str(error).splitlines() or [''])[0][:200]}"
    ref = x.double() @ w.double().T
    if (result.data_ptr() != out.data_ptr() or result.dtype != torch.float32
            or not torch.allclose(out.double(), ref, atol=ATOL, rtol=RTOL)):
        return "unavailable: did not produce matching in-place FP32 output"
    return "available"


def _load_frozen(frozen_from, dtype, shapes):
    path = project_path(ROOT, frozen_from)
    data = json.loads(path.read_text())
    if data.get("kind") != "half-projection" or data.get("mode") != "tune" or data.get("status") != "complete":
        raise ValueError("frozen_from must be a complete half-projection tuning run")
    if data.get("dtype") != dtype or data.get("selection_seed") != TUNING_SEED:
        raise ValueError(f"Frozen profile must be a {dtype} selection made on seed {TUNING_SEED}")
    from experiment_provenance import CUDA_SOURCES
    for name in (*CUDA_SOURCES,"half_projection.py","native/driver.cpp","native/build/libkernel_lab.so","kernel_lab.py"):
        if data.get("source_sha256",{}).get(name)!=_sha(name):
            raise ValueError(f"Frozen custom-kernel implementation changed: {name}; retune before validation")
    frozen = {tuple(c["shape_mnk"]): (c["selected_threads"], c["selected_vector"]) for c in data["cases"]}
    missing = [shape for shape in shapes if shape not in frozen]
    if missing:
        raise ValueError(f"No frozen selection for {missing}")
    return path, frozen


def _check_outputs(torch, x, weights, outputs):
    """Every ring copy and output element, vs FP64 on the same 16-bit values, in bounded chunks."""
    copies, n, k = weights.shape
    chunk = max(1, (32 * 1024**2) // (n * k * 8))
    stats = {name: {"max_abs_error": 0.0, "err_sq": 0.0, "violations": 0, "nonfinite": 0} for name in outputs}
    ref_sq = 0.0
    xd = x.double()
    for start in range(0, copies, chunk):
        ref = xd @ weights[start:start + chunk].double().transpose(1, 2)
        ref_sq += ref.square().sum().item()
        limit = ATOL + RTOL * ref.abs()
        for name, out in outputs.items():
            got = out[start:start + chunk].double()
            delta = (got - ref).abs().nan_to_num(nan=math.inf)
            s = stats[name]
            s["max_abs_error"] = max(s["max_abs_error"], delta.max().item())
            s["err_sq"] += delta.square().sum().item()
            s["violations"] += int((~(delta <= limit)).sum().item())
            s["nonfinite"] += int((~torch.isfinite(got)).sum().item())
    result = {}
    for name, s in stats.items():
        result[name] = {"max_abs_error": s["max_abs_error"],
                        "relative_l2_error": math.sqrt(s["err_sq"] / max(ref_sq, 1e-300)),
                        "elements_checked": outputs[name].numel(), "violations": s["violations"],
                        "nonfinite": s["nonfinite"], "passed": s["violations"] == 0}
    return result


def run(shapes, seeds, rounds, output, dtype="fp16", frozen_from=None, run_metadata=None):
    shapes = [tuple(int(v) for v in shape) for shape in shapes]
    seeds = [int(seed) for seed in seeds]
    if dtype not in FORMATS:
        raise ValueError("dtype must be 'fp16' or 'bf16'")
    if rounds < MIN_ROUNDS:
        raise ValueError(f"At least {MIN_ROUNDS} measured rounds are required")
    if not shapes or any(len(s) != 3 or min(s) <= 0 or s[2] % 2 for s in shapes):
        raise ValueError("Shapes must be positive M,N,K with even K")
    tuning = frozen_from is None
    if tuning and seeds != [TUNING_SEED]:
        raise ValueError(f"Tuning uses seed {TUNING_SEED} only; validate other seeds with frozen_from")
    if not tuning and (not seeds or TUNING_SEED in seeds):
        raise ValueError(f"Held-out validation needs seeds other than {TUNING_SEED}")
    frozen_path, frozen = (None, {}) if tuning else _load_frozen(frozen_from, dtype, shapes)

    import torch
    import half_projection
    import kernel_lab
    from bench import measure
    from bench_streaming import interval
    from cublas_baseline import CuBlas
    from gpu_monitor import GpuMonitor
    from experiment_provenance import environment,source_hashes
    torch.set_grad_enabled(False)
    policy = torch.backends.cuda.matmul
    policy.allow_tf32 = False
    policy.allow_fp16_reduced_precision_reduction = False
    policy.allow_bf16_reduced_precision_reduction = False
    if hasattr(policy, "allow_fp16_accumulation"):
        policy.allow_fp16_accumulation = False
    torch_dtype, storage = getattr(torch, FORMATS[dtype][0]), FORMATS[dtype][1]
    lab, vendor = kernel_lab.default_lab(), CuBlas()
    l2 = int(torch.cuda.get_device_properties(lab.device).L2_cache_size)
    ring_target = max(80 * 1024**2, 20 * l2)
    torch_out = _torch_out_fp32_status(torch, torch_dtype)
    vendor_routes = [FIXED, "cublas_packed_compute32"] + (["torch_mm_out_fp32"] if torch_out == "available" else [])
    report = with_run_metadata({
        "kind": "half-projection", "status": "running", "mode": "tune" if tuning else "validate",
        "dtype": dtype, "precision_contract": PRECISION, "input_preparation": INPUTS,
        "selection_seed": TUNING_SEED, "selection_rule": SELECTION,
        "candidate_space": [_name(t, v) for t, v in CANDIDATES],
        "frozen_from": str(frozen_path.relative_to(ROOT)) if frozen_path else None,
        "frozen_profile_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest() if frozen_path else None,
        "seeds": seeds, "rounds": rounds, "l2_bytes": l2, "ring_target_bytes": ring_target,
        "device": torch.cuda.get_device_name(lab.device),
        "compute_capability": list(torch.cuda.get_device_capability(lab.device)),
        "torch": torch.__version__, "cuda": torch.version.cuda, "compilation": lab.compilation,
        "baselines": {"fixed_route": FIXED, "equivalent_vendor_routes": vendor_routes,
                      "torch_mm_out_fp32": torch_out, "aa_twin": f"{FIXED}_aa",
                      "excluded": "torch.mm with 16-bit output and output-cast chains are not equivalent; not timed"},
        "protocol": PROTOCOL, "acceptance": ACCEPTANCE, "gpu_unlock": UNLOCK,
        "source_sha256":source_hashes(SOURCES),"environment":environment(lab,vendor),
        "schema_version":2,"cases": []}, run_metadata)
    report["gpu_unlock"]="No unlock effect is claimed. Actual module/boot state is recorded in environment.inventory."

    def case(seed, m, n, k):
        copies = max(2, math.ceil(ring_target / (n * k * 2)))
        calls = copies * max(1, math.ceil(32 / copies))
        torch.manual_seed(seed)
        weights = torch.randn(copies, n, k, device="cuda", dtype=torch_dtype)
        x = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(torch_dtype)
        wt = weights.transpose(1, 2).contiguous()
        if not torch.equal(wt.transpose(1, 2).view(torch.int16), weights.view(torch.int16)):
            raise AssertionError("Transposed vendor ring is not bitwise identical")
        ws, tw = weights.unbind(), wt.unbind()
        functions, outputs = {}, {}

        def add(name, fn):
            outputs[name] = torch.full((copies, m, n), float("nan"), device="cuda")
            outs = outputs[name].unbind()

            def cycle():
                for j in range(calls):
                    fn(j % copies, outs[j % copies])
            functions[name] = cycle
        add(FIXED, lambda i, o: vendor.mm_compute32(x, ws[i], o, dtype))
        add("cublas_packed_compute32", lambda i, o: _packed_gemm(vendor, x, tw[i], o, storage))
        if torch_out == "available":
            add("torch_mm_out_fp32", lambda i, o: torch.mm(x, ws[i].T, out_dtype=torch.float32, out=o))
        add(f"{FIXED}_aa", lambda i, o: vendor.mm_compute32(x, ws[i], o, dtype))
        options = [frozen[(m, n, k)]] if frozen else [c for c in CANDIDATES if k % c[1] == 0]
        for t, v in options:
            add(f"half_{_name(t, v)}", lambda i, o, t=t, v=v: half_projection.fused_linear(x, ws[i], t, o, v))
        with GpuMonitor() as telemetry:
            timing = measure(functions, rounds=rounds, repeats=1)
        for stats in timing.values():
            for key in ("median_us", "min_us", "max_us"):
                stats[key] /= calls
            stats["samples_us"] = [v / calls for v in stats["samples_us"]]
        numerics = _check_outputs(torch, x, weights, outputs)
        best = min(vendor_routes, key=lambda name: timing[name]["median_us"])
        base = timing[best]["median_us"]
        candidates = {name: {"median_us": timing[name]["median_us"],
                             "time_saved_vs_best_vendor_pct": percentage_gains(base, timing[name]["median_us"])["latency_reduction_pct"]}
                      for name in functions if name.startswith("half_")}
        threads, vector = frozen.get((m, n, k)) or next(
            c for c in options if f"half_{_name(*c)}" == min(candidates, key=lambda name: candidates[name]["median_us"]))
        chosen = f"half_{_name(threads, vector)}"
        gains = percentage_gains(base, timing[chosen]["median_us"])
        ci = interval(timing[best]["samples_us"], timing[chosen]["samples_us"])
        fixed = percentage_gains(timing[FIXED]["median_us"], timing[chosen]["median_us"])
        fixed_ci = interval(timing[FIXED]["samples_us"], timing[chosen]["samples_us"])
        aa = percentage_gains(timing[FIXED]["median_us"], timing[f"{FIXED}_aa"]["median_us"])["latency_reduction_pct"]
        aa_ok = abs(aa) <= 1
        numerics_ok = all(v["passed"] for v in numerics.values())
        # Separate diagnostic: cost of rounding an FP32 source model; not this operation's error.
        g = torch.Generator(device="cuda").manual_seed(seed)
        w32 = torch.randn(n, k, device="cuda", generator=g)
        x32 = torch.randn(m, k, device="cuda", generator=g) / math.sqrt(k)
        exact = x32.double() @ w32.double().T
        rounded = x32.to(torch_dtype).double() @ w32.to(torch_dtype).double().T
        return {"shape_mnk": [m, n, k], "seed": seed, "held_out": not tuning, "dtype": dtype,
                "selected": _name(threads, vector), "selected_threads": threads, "selected_vector": vector,
                "copies": copies, "calls_per_graph": calls,
                "ring_bytes": {"weights": weights.numel() * 2, "transposed_weights": wt.numel() * 2},
                "ring_over_l2": weights.numel() * 2 / l2, "bitwise_layout_check": True,
                "candidates": candidates, "best_vendor": best, "fixed_route": FIXED,
                "methods": {name: {**timing[name], "numerics": numerics[name]} for name in functions},
                "numerical_check_scope": (f"All {copies} ring copies x {m}x{n} outputs for every method, read after "
                                          f"timing, vs FP64 on the same 16-bit values (atol={ATOL}, rtol={RTOL})"),
                "numerics_passed": numerics_ok, "aa_time_saved_pct": aa, "aa_passed": aa_ok,
                "candidate_time_saved_pct": gains["latency_reduction_pct"], "candidate_gain_ci95_pct": ci,
                "fixed_route_time_saved_pct": fixed["latency_reduction_pct"], "fixed_route_gain_ci95_pct": fixed_ci,
                "noise_gated_win": numerics_ok and aa_ok and ci[0] > max(0.0, abs(aa)),
                "accepted_10pct": not tuning and numerics_ok and aa_ok and ci[0] > 10,
                "above_10pct_in_this_run":numerics_ok and aa_ok and ci[0]>10,
                "fp32_source_rounding_diagnostic": {
                    "label": f"Separate diagnostic: rounding an independent FP32 x/w sample to {dtype}; "
                             "not part of this operation's precision contract",
                    "relative_l2_error": ((rounded - exact).norm() / exact.norm()).item()},
                "gpu_telemetry": telemetry.samples}

    try:
        for seed in seeds:
            for m, n, k in shapes:
                c = case(seed, m, n, k)
                report["cases"].append(c)
                write_json(output, report)
                lo, hi = c["candidate_gain_ci95_pct"]
                print(f"{dtype} {m}x{n}x{k} seed={seed}: half_{c['selected']} vs {c['best_vendor']} "
                      f"{c['candidate_time_saved_pct']:+.2f}% CI [{lo:+.2f},{hi:+.2f}]%; A/A {c['aa_time_saved_pct']:+.2f}%; "
                      f"numerics={c['numerics_passed']} noise-gated={c['noise_gated_win']} >10%={c['accepted_10pct']}",
                      flush=True)
    except Exception as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        write_json(output, report)
        raise
    finally:
        vendor.close()
    report["status"] = "complete"
    write_json(output, report)
    return report


def verify():
    import torch
    import half_projection
    from bench import check
    torch.set_grad_enabled(False)
    checks = 0
    for dtype in (torch.float16, torch.bfloat16):
        for seed in (953, 967):
            torch.manual_seed(seed)
            for m, n, k in ((1, 1, 2), (3, 65, 130), (1, 17, 520), (4, 33, 4096), (2, 7, 8)):
                x = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(dtype)
                w = torch.randn(n, k, device="cuda").to(dtype)
                ref = (x.double() @ w.double().T).float()
                for threads in half_projection.THREADS:
                    for vector in ((2, 8) if k % 8 == 0 else (2,)):
                        out = torch.full((m, n), float("nan"), device="cuda")
                        check(half_projection.fused_linear(x, w, threads, out, vector), ref)
                        checks += 1
        # One-hot rows return each finite 16-bit weight widened exactly, incl. subnormals and zeros.
        bits = (torch.randint(0, 2**16, (33, 64), device="cuda", dtype=torch.int32) - 2**15).to(torch.int16)
        bits[0, :6] = torch.tensor([0, -32768, 1, -32767, 0x03ff, 0x0400], device="cuda", dtype=torch.int16)
        bits = torch.where(torch.isfinite(bits.view(dtype)), bits, torch.zeros_like(bits))
        w = bits.view(dtype)
        eye = torch.eye(64, device="cuda", dtype=dtype)
        for threads in half_projection.THREADS:
            for vector in half_projection.VECTORS:
                if not torch.equal(half_projection.fused_linear(eye, w, threads, None, vector), w.float().T):
                    raise AssertionError("Exact 16-bit widening failed")
                checks += 1
        w = torch.ones(2, 64, device="cuda", dtype=dtype)
        w[0, 3], w[1, 5] = float("inf"), float("nan")
        out = half_projection.fused_linear(torch.ones(1, 64, device="cuda", dtype=dtype), w)
        if not (out[0, 0].item() == math.inf and math.isnan(out[0, 1].item())):
            raise AssertionError("Non-finite propagation failed")
        checks += 1
    x = (torch.randn(3, 512, device="cuda") / math.sqrt(512)).half()
    w = torch.randn(65, 512, device="cuda").half()
    compiled = torch.compile(half_projection.linear, fullgraph=True)
    check(compiled(x, w, 64), half_projection.fused_linear(x, w, 64))
    check(half_projection.linear(x.bfloat16(), w.bfloat16(), 128, 8),
          (x.bfloat16().double() @ w.bfloat16().double().T).float())
    checks += 2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = half_projection.fused_linear(x, w)
        expected = (x.double() @ w.double().T).float()
    torch.cuda.current_stream().wait_stream(stream)
    check(out, expected)
    checks += 1
    x = torch.ones(8, 16, device="cuda", dtype=torch.float16)
    w = torch.ones(8, 16, device="cuda", dtype=torch.float16)
    invalid = (
        lambda: half_projection.fused_linear(x.float(), w.float()),  # FP32 is a separate contract.
        lambda: half_projection.fused_linear(x, w.bfloat16()),
        lambda: half_projection.fused_linear(x[:, :15].contiguous(), w[:, :15].contiguous()),
        lambda: half_projection.fused_linear(x[:, ::2], w[:, ::2]),
        lambda: half_projection.fused_linear(x.cpu(), w.cpu()),
        lambda: half_projection.fused_linear(x.clone().requires_grad_(), w),
        lambda: half_projection.fused_linear(x, w, out=w.view(torch.float32)),
        lambda: half_projection.fused_linear(x, w, out=torch.empty(8, 8, device="cuda", dtype=torch.float16)),
        lambda: half_projection.fused_linear(x, w, out=torch.empty(8, 9, device="cuda")),
        lambda: half_projection.fused_linear(x, w, threads=48),
        lambda: half_projection.fused_linear(x[:, :12].contiguous(), w[:, :12].contiguous(), vector=8),
        lambda: half_projection.fused_linear(x.view(-1)[1:15].view(1, 14), w[:1, :14].contiguous()),
    )
    for call in invalid:
        try:
            call()
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid input accepted")
    torch.cuda.synchronize()
    return checks


def make_report(report, source, destination=None):
    destination = report_output(ROOT, source, destination)
    source = Path(source)
    held = report.get("mode") == "validate"
    cases = report["cases"]
    lines = [f"# {report['dtype'].upper()}-input projection: "
             + ("held-out rotating-ring validation" if held else f"seed-{TUNING_SEED} tuning (in-sample)"), "",
             report["precision_contract"], "", report["input_preparation"], "", report["protocol"], "",
             f"Selection: {report['selection_rule']} Frozen source: `{report.get('frozen_from') or 'none: this run selects'}`.",
             f"torch.mm FP32-output route: {report['baselines']['torch_mm_out_fp32']}.", "",
             "| M × N × K | Seed | Kernel | Best vendor | Time saved (%) | 95% CI (%) | vs fixed cuBLAS (%) | A/A (%) | Numerics | Noise-gated win | >10% accepted |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for c in cases:
        lo, hi = c["candidate_gain_ci95_pct"]
        lines.append("| " + " × ".join(map(str, c["shape_mnk"])) + f" | {c['seed']} | {c['selected']} | {c['best_vendor']} | "
                     f"{c['candidate_time_saved_pct']:+.2f} | {lo:+.2f} to {hi:+.2f} | {c['fixed_route_time_saved_pct']:+.2f} | "
                     f"{c['aa_time_saved_pct']:+.2f} | {'pass' if c['numerics_passed'] else 'FAIL'} | "
                     f"{'Yes' if c['noise_gated_win'] else 'No'} | {'Yes' if c['accepted_10pct'] else 'No'} |")
    if not held:
        lines += ["", "## All candidates (% time saved vs best vendor; negative = loss)", ""]
        for c in cases:
            lines.append("- " + " × ".join(map(str, c["shape_mnk"])) + ": " + ", ".join(
                f"{name.removeprefix('half_')} {v['time_saved_vs_best_vendor_pct']:+.2f}" for name, v in c["candidates"].items()))
    wins = sum(c["noise_gated_win"] for c in cases)
    accepted = sum(c["accepted_10pct"] for c in cases)
    lines += ["", report["acceptance"], "",
              f"Noise-gated custom wins: {wins}/{len(cases)}. >10% latency-reduction target met: {accepted}/{len(cases)}."
              + ("" if held else " These are in-sample tuning results and do not count toward acceptance."),
              "Best vendor is chosen per case after timing (conservative for the custom kernel); the fixed cuBLAS column uses one route throughout.",
              "Intervals reflect round-to-round variation within one process; independent process replication is still needed.",
              report["gpu_unlock"], "", f"Raw run: [{source.name}]({source.name}).", ""]
    destination.write_text("\n".join(lines))
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=parse_shape, action="append")
    p.add_argument("--seeds", default=str(TUNING_SEED))
    p.add_argument("--rounds", type=int, default=MIN_ROUNDS)
    p.add_argument("--dtype", choices=tuple(FORMATS), default="fp16")
    p.add_argument("--frozen-from")
    p.add_argument("--output", default="results/half-projection.json")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()
    with GpuLock(ROOT / ".cache/gpu.lock", wait=a.wait):
        print(f"PASS: {verify()} half-projection checks", flush=True)
        if not a.verify_only:
            output = run_output(ROOT, a.output)
            report = run(a.shape or SHAPES, [int(s) for s in a.seeds.split(",")], a.rounds, output, a.dtype, a.frozen_from)
            print(f"Report: {make_report(report, output)}", flush=True)


if __name__ == "__main__":
    main()
