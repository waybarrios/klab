"""Frozen cuBLASLt/vendor pool on real weights, with streaming A/A controls.

This is baseline preparation and a headroom screen, not a custom-kernel win.
The caller owns the GPU lock. A screen-only run does not cover the full grid.
"""

import base64
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from experiment_control import project_path, with_run_metadata, write_json

ROOT = Path(__file__).resolve().parent


def optimistic_gain_pct(weight_bytes, ratio, read_gb_per_s, launch_floor_us, vendor_us):
    values = (weight_bytes, ratio, read_gb_per_s, launch_floor_us, vendor_us)
    if (not all(math.isfinite(v) for v in values) or weight_bytes <= 0 or not 0 < ratio <= 1
            or read_gb_per_s <= 0 or launch_floor_us < 0 or vendor_us <= 0):
        raise ValueError("Invalid storage/bandwidth/latency screen inputs")
    predicted_us = ratio * weight_bytes / (read_gb_per_s * 1000) + launch_floor_us
    return 100 * (vendor_us - predicted_us) / vendor_us


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(calls):
    def invoke():
        for call in calls:
            call()
    return invoke


def _case(entry, manifest, m, input_name, output_name, workspace, vendor, rounds, control=False):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from bench import measure
    from cublaslt_baseline import LtPlan, prepare_lt_plans
    from gpu_monitor import GpuMonitor
    from projection_accuracy import check_outputs
    from real_weights import read_tensor

    torch.manual_seed(901)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[input_name]
    output_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[output_name]
    setup_start = time.monotonic()
    if control:
        n = k = 256
        weight = torch.randn((n, k), dtype=dtype, device="cuda")
        weight_sha = hashlib.sha256(weight.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    else:
        bits = np.array(read_tensor(manifest, entry["name"]), copy=True)
        weight = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
        if not np.array_equal(weight.view(torch.uint16).cpu().numpy(), bits):
            raise RuntimeError("Original checkpoint bits changed during upload")
        n, k = weight.shape
        weight_sha = entry["sha256"]
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    x_sha = hashlib.sha256(x.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    target = max(80 * 1024**2, 20 * l2)
    copies = max(1, math.ceil(target / (n * k * weight.element_size())))
    ring = weight.unsqueeze(0).repeat(copies, 1, 1)
    packed_ring = ring.transpose(1, 2).contiguous()
    if (not torch.equal(ring.view(torch.int16), weight.view(torch.int16).expand(copies, n, k))
            or not torch.equal(packed_ring.view(torch.int16).transpose(1, 2), ring.view(torch.int16))):
        raise RuntimeError("Raw/transposed ring layout changed weight bits")
    torch.cuda.reset_peak_memory_stats()
    functions, outputs, routes, all_plans = {}, {}, {}, []
    selected_control = None
    diagnostics = []

    def add(name, builder, route):
        routes[name] = route
        for label in (name, name + "_aa"):
            out = torch.empty((copies, m, n), dtype=output_dtype, device="cuda")
            outputs[label] = out
            functions[label] = _batch([builder(i, out[i]) for i in range(copies)])

    try:
        pool = prepare_lt_plans(x, ring[0], output_dtype=output_dtype, workspace=workspace)
        diagnostics = pool.search_diagnostics
        all_plans.extend(pool)
        for plan in pool:
            record = plan.freeze()
            clones = []
            for index in range(copies):
                clone = LtPlan(record, base64.b64decode(record["algorithm_b64"]), workspace,
                               x=x, weight=ring[index],
                               packed=packed_ring[index] if record["layout"] == "transposed" else None)
                clones.append(clone)
                all_plans.append(clone)
            name = f"lt_{record['layout']}_h{record['candidate_idx']}"
            add(name, lambda i, out, plans=clones: lambda: plans[i].run(x, ring[i], out), record)
        if output_name == "bf16":
            add("torch_mm", lambda i, out: lambda: torch.mm(x, ring[i].T, out=out), {"kind": "torch.mm"})
            # F.linear has no out=. Capture its native allocation and retain
            # the actual captured output tensors, without inserting a copy.
            for label in ("torch_linear", "torch_linear_aa"):
                holders = [None] * copies
                def linear_batch(holders=holders):
                    for i in range(copies):
                        holders[i] = functional.linear(x, ring[i])
                functions[label] = linear_batch
                outputs[label] = holders
            routes["torch_linear"] = {"kind": "torch.nn.functional.linear", "allocation": "native graph allocation"}
        else:
            add("cublas_compute32", lambda i, out: lambda: vendor.mm_compute32(x, ring[i], out, input_name),
                {"kind": "cublasGemmEx", "layout": "original"})
            add("cublas_packed_compute32", lambda i, out: lambda: vendor.mm_compute32(x, packed_ring[i], out, input_name, True),
                {"kind": "cublasGemmEx", "layout": "transposed"})
            try:
                torch.mm(x, ring[0].T, out_dtype=torch.float32)
            except (TypeError, RuntimeError) as error:
                diagnostics.append({"route": "torch_mm_out_fp32", "unavailable": str(error)})
            else:
                add("torch_mm_out_fp32", lambda i, out: lambda: torch.mm(x, ring[i].T, out_dtype=torch.float32, out=out),
                    {"kind": "torch.mm", "out_dtype": "fp32"})
        if control:
            import half_projection
            frozen = json.loads((ROOT / f"results/{input_name}-search.json").read_text())
            chosen = next(c for c in frozen["cases"] if c["shape_mnk"] == [4, 256, 256])
            threads, vector = chosen["selected_threads"], chosen["selected_vector"]
            selected_control = f"half_t{threads}_v{vector}"
            add(selected_control,
                lambda i, out: lambda: half_projection.fused_linear(x, ring[i], threads, out, vector),
                {"kind": "existing-custom-control", "threads": threads, "vector": vector})
        setup_seconds = time.monotonic() - setup_start
        with GpuMonitor() as monitor:
            start = time.monotonic()
            while time.monotonic() - start < 2:
                next(iter(functions.values()))()
                torch.cuda.synchronize()
            measured = measure(functions, rounds=rounds, repeats=1)
        for method in measured.values():
            for key in ("median_us", "min_us", "max_us"):
                method[key] /= copies
            method["samples_us"] = [v / copies for v in method["samples_us"]]
        reference = x.double() @ weight.double().T
        checks = {}
        for name, out in outputs.items():
            if isinstance(out, list):
                out = torch.stack(out)
            checks[name] = check_outputs(reference.expand(copies, m, n), out, output_dtype=output_dtype)
            measured[name]["numerics"] = checks[name]
        eligible = [name for name in routes if name != selected_control
                    and checks[name]["passed"] and checks[name + "_aa"]["passed"]]
        if not eligible:
            raise RuntimeError("No vendor route passed the output precision contract")
        winner = min(eligible, key=lambda name: measured[name]["median_us"])
        aa = 100 * (measured[winner]["median_us"] - measured[winner + "_aa"]["median_us"]) / measured[winner]["median_us"]
        return {"shape_mnk": [m, n, k], "input_dtype": input_name, "output_dtype": output_name,
                "weight_tensor": "synthetic-control" if control else entry["name"],
                "weight_sha256": weight_sha, "activation_sha256": x_sha, "seed": 901,
                "copies": copies, "calls_per_graph": copies, "ring_bytes": ring.numel() * ring.element_size(),
                "ring_over_l2": ring.numel() * ring.element_size() / l2,
                "bitwise_layout_check": True, "setup_seconds": setup_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "shared_workspace_bytes": workspace.nbytes, "vendor_pool": routes,
                "search_diagnostics": diagnostics, "methods": measured,
                "best_vendor": winner, "best_vendor_us": measured[winner]["median_us"],
                "aa_time_saved_pct": aa, "aa_passed": abs(aa) <= 1,
                "existing_custom_control": selected_control,
                "control_time_saved_pct": (100 * (1 - measured[selected_control]["median_us"] / measured[winner]["median_us"])) if selected_control else None,
                "gpu_telemetry": monitor.samples}
    finally:
        for plan in all_plans:
            plan.close()


def run(spec_path, corpus_path, output, *, regime_path, storage_path, rounds=31, screen_only=False, run_metadata=None):
    import torch
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from gpu_unlock_status import inspect

    paths = [project_path(ROOT, p) for p in (spec_path, corpus_path, regime_path, storage_path)]
    spec, manifest, regime, storage = [json.loads(p.read_text()) for p in paths]
    if (manifest.get("status") != "complete" or not manifest.get("source_digest_verified")
            or manifest["revision"] != spec["revision"] or manifest["source_sha256"] != spec["safetensors_sha256"]):
        raise ValueError("Corpus does not match the preregistered immutable input")
    if rounds < 31 or regime.get("status") != "complete" or not regime["verification"]["passed"]:
        raise ValueError("Need at least31 rounds and a verified complete regime probe")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    device = torch.cuda.current_device()
    if device != 0:
        raise ValueError("This preregistered runner currently targets CUDA device0")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if regime["device"]["uuid"] != uuid:
        raise ValueError("Regime probe belongs to a different GPU")
    selected = []
    for shape in spec["shapes_nk"]:
        selected.append(next(t for t in manifest["tensors"] if t["shape"] == shape and ".layers.0." in t["name"]))
    if not screen_only:
        head = next((t for t in manifest["tensors"] if t["name"] == spec["head"]["tensor"]), None)
        if head:
            selected.append(head)
    report = {"kind": "vendor-tuning", "status": "running", "schema_version": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(), "screen_only": screen_only,
              "spec_sha256": _sha(paths[0]), "corpus_sha256": _sha(paths[1]),
              "regime_path": str(paths[2].relative_to(ROOT)), "regime_sha256": _sha(paths[2]),
              "storage_path": str(paths[3].relative_to(ROOT)), "storage_sha256": _sha(paths[3]),
              "source_sha256": {name: _sha(ROOT / name) for name in ("bench_vendor.py", "cublaslt_baseline.py", "cublas_baseline.py", "projection_accuracy.py", "bench.py", "gpu_monitor.py")},
              "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "inventory": inspect()},
              "protocol": "distinct physical weight copies >=max(80MiB,20L2); graph executes one full ring; report per projection; all outputs checked; per-route A/A",
              "rounds": rounds, "cases": []}
    report = with_run_metadata(report, run_metadata)
    write_json(output, report)
    vendor, workspace = CuBlas(), LtWorkspace(0)
    try:
        for entry in selected:
            ms = [1] if screen_only else spec["m_values"] + ([] if entry["name"] == spec["head"]["tensor"] else spec["baseline_diagnostic_m"])
            for m in ms:
                for out_dtype in ("bf16", "fp32"):
                    case = _case(entry, manifest, m, "bf16", out_dtype, workspace, vendor, rounds)
                    report["cases"].append(case)
                    write_json(output, report)
                    print(f"vendor {case['shape_mnk']} bf16->{out_dtype}: {case['best_vendor']} {case['best_vendor_us']:.3f} us; A/A {case['aa_time_saved_pct']:+.3f}%", flush=True)
        for dtype in ("fp16", "bf16"):
            case = _case(None, manifest, 4, dtype, "fp32", workspace, vendor, rounds, control=True)
            report["cases"].append(case)
            write_json(output, report)
            print(f"old control {dtype}: time saved {case['control_time_saved_pct']:.3f}%; A/A {case['aa_time_saved_pct']:+.3f}%", flush=True)
    finally:
        workspace.close()
        vendor.close()
    median = regime["median"]["steady"]
    screen = []
    for shape in spec["large_primary_shapes_nk"]:
        case = next(c for c in report["cases"] if c["shape_mnk"] == [1, *shape] and c["output_dtype"] == "bf16")
        packed = next(c for c in storage["cases"] if c["name"] == case["weight_tensor"])
        ratio = packed[storage["selected_format"]]["ratio"]
        gain = optimistic_gain_pct(packed["original_bytes"], ratio, median["read_input_gb_per_s"], median["launch_floor_us"], case["best_vendor_us"])
        screen.append({"shape_nk": shape, "storage_ratio": ratio, "optimistic_time_saved_pct": gain,
                       "aa_passed": case["aa_passed"], "passed": ratio <= .8 and gain >= 15 and case["aa_passed"]})
    report.update(status="complete", headroom_screen=screen, codec_gpu_gate_passed=all(c["passed"] for c in screen),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  warning="Optimistic screening estimates are not measured compressed-kernel speedups. Tuning observations are not held-out acceptance.")
    write_json(output, report)
    return report


def make_report(data, source, output):
    lines = ["# Vendor baseline and optimistic headroom screen", "",
             f"Raw data: [{Path(source).name}]({Path(source).name})", "",
             f"Scope: {'M=1 screen plus synthetic controls' if data['screen_only'] else 'full preregistered baseline grid'}. "
             "These are tuning measurements, not held-out custom-kernel acceptance.", "",
             "| M,N,K | Input/output | Best eligible vendor | Median us | A/A % | Existing control saved % |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for case in data["cases"]:
        control = case.get("control_time_saved_pct")
        control_text = f"{control:.3f}" if control is not None else "n/a"
        lines.append(f"| {','.join(map(str,case['shape_mnk']))} | {case['input_dtype']} / {case['output_dtype']} | "
                     f"{case['best_vendor']} | {case['best_vendor_us']:.3f} | {case['aa_time_saved_pct']:+.3f} | {control_text} |")
    lines.extend(["", "## Screening estimates", "", "Storage and roofline estimates only; compressed-kernel latency has not been measured.", "",
                  "| N,K | Packed/original bytes | Optimistic latency saved % | Screen passed |",
                  "| --- | ---: | ---: | --- |"])
    for case in data["headroom_screen"]:
        lines.append(f"| {','.join(map(str,case['shape_nk']))} | {case['storage_ratio']:.4f} | "
                     f"{case['optimistic_time_saved_pct']:.3f} | {case['passed']} |")
    lines.extend(["", f"Codec GPU experiment gate: {data['codec_gpu_gate_passed']}.", "",
                  "All vendor algorithms use FP32 compute and no BF16 partial reduction. "
                  "Timings use a physical rotating ring of at least 80 MiB and 20xL2. "
                  "Packing/setup are outside replay and recorded separately. All captured outputs are checked against FP64.", ""])
    Path(output).write_text("\n".join(lines))
    return Path(output)
