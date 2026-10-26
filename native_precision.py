"""Frozen FP32/FP16/BF16 primitives against a strong streaming vendor pool.

No kernel edits/search, no lossy inputs and no eager/full-model claim. Imports
and protocol checks are CPU-only. CLI owns the GPU lock and restores settings.
"""
import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROTOCOL = "experiments/native_precision_refresh.json"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_cases(protocol, dtype, *, root=ROOT):
    from experiment_control import project_path
    if (dtype not in ("fp32", "fp16", "bf16") or protocol.get("configuration_search_allowed") is not False
            or protocol.get("code_changes_allowed") is not False):
        raise ValueError("Only registered native precisions and frozen configuration replay are supported")
    cases = copy.deepcopy(protocol["cases_by_dtype"][dtype])
    loaded = {}
    for origin in {case["origin"] for case in cases}:
        reference = protocol["source_profiles"][origin]
        path = project_path(root, reference["path"])
        if _sha(path) != reference["sha256"]:
            raise ValueError("Historical configuration source bytes changed")
        loaded[origin] = json.loads(path.read_text())
    seen = set()
    for case in cases:
        shape = tuple(case["shape_mnk"])
        if (len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
                or shape in seen):
            raise ValueError("Frozen shape grid must be positive, unique integer M,N,K")
        seen.add(shape)
        matching = [row for row in loaded[case["origin"]]["cases"] if tuple(row["shape_mnk"]) == shape]
        if len(matching) != 1:
            raise ValueError("Frozen shape has no unique historical choice")
        old = matching[0]
        expected = (f"lossless_t{old['selected_threads']}" if case["origin"] == "fp32_large"
                    else old["selected"] if dtype == "fp32" else "half_" + old["selected"])
        pattern = (r"(?:tiny|warp)_t(?:32|64|128|256)_v(?:1|4)|(?:lossless|fp28)_t(?:32|64|128|256)"
                   if dtype == "fp32" else r"half_t(?:32|64|128|256)_v(?:2|8)")
        if case["config"] != expected or re.fullmatch(pattern, expected) is None:
            raise ValueError("Configuration differs from its historical precision/shape choice")
    return cases


def _case(spec, dtype, seed, workspace, vendor, protocol):
    import numpy as np
    import torch
    import half_projection
    import kernel_lab
    import lossless_weights
    import packed_fp32
    from bench import measure
    from bench_dispatch import current_context
    from compressed_campaign import check_ring, summarize_methods
    from cublaslt_baseline import LtPlan, prepare_lt_plans
    from gpu_monitor import GpuMonitor

    m, n, k = spec["shape_mnk"]
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    integer = torch.int32 if dtype == "fp32" else torch.int16
    torch.manual_seed(seed)
    started = time.monotonic()
    weight = torch.randn((n, k), dtype=torch_dtype, device="cuda")
    x = torch.randn((m, k), dtype=torch_dtype, device="cuda") / math.sqrt(k)
    original = weight.view(integer).cpu().numpy().copy()
    xbits = x.view(integer).cpu().numpy().copy()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    copies = max(2, math.ceil(max(80 << 20, 20 * l2) / (n * k * weight.element_size())))
    ring = weight.unsqueeze(0).repeat(copies, 1, 1)
    transposed = ring.transpose(1, 2).contiguous()
    if not (torch.equal(ring.view(integer), weight.view(integer).expand_as(ring))
            and torch.equal(transposed.view(integer).transpose(1, 2), ring.view(integer))):
        raise RuntimeError("Original/transposed ring changed supplied weight bits")
    weights, packed_views = ring.unbind(), transposed.unbind()
    functions, outputs, routes, plans = {}, {}, {}, []

    def add(name, fn, route=None):
        labels = (name,) if route is None else (name, name + "_aa")
        if route is not None:
            routes[name] = route
        for label in labels:
            out = torch.full((copies, m, n), float("nan"), device="cuda", dtype=torch.float32)
            outputs[label] = out
            outs = out.unbind()
            def sweep(fn=fn, outs=outs):
                for i in range(copies):
                    fn(i, outs[i])
            functions[label] = sweep

    packed = []
    packing_started = time.monotonic()
    config = spec["config"]
    threads = int(re.search(r"_t(\d+)", config).group(1))
    if config.startswith("lossless_"):
        packed = [lossless_weights.pack_weights(w) for w in weights]
        call = lambda i, out: lossless_weights.fused_linear(x, packed[i], threads, out)
    elif config.startswith("fp28_"):
        packed = [packed_fp32.pack_weights(w) for w in weights]
        call = lambda i, out: packed_fp32.fused_linear(x, packed[i], threads, out)
    elif config.startswith("half_"):
        vector = int(config.rsplit("v", 1)[1])
        call = lambda i, out: half_projection.fused_linear(x, weights[i], threads, out, vector)
    else:
        vector = int(config.rsplit("v", 1)[1])
        method = kernel_lab.default_lab().tiny_matmul if config.startswith("tiny_") else kernel_lab.default_lab().warp_matmul
        call = lambda i, out: method(x, weights[i], threads, vector, out)
    exact = all(torch.equal(p.unpack().view(torch.int32), weights[i].view(torch.int32)) for i, p in enumerate(packed))
    if not exact:
        raise RuntimeError("Packed FP32 weights failed original-bit roundtrip")
    torch.cuda.synchronize()
    packing_seconds = time.monotonic() - packing_started if packed else 0.0
    add("custom", call)
    try:
        pool = prepare_lt_plans(x, weights[0], output_dtype=torch.float32, workspace=workspace)
        plans.extend(pool)
        diagnostics = list(pool.search_diagnostics)
        for plan in pool:
            record = plan.freeze()
            holders = []
            for i in range(copies):
                holder = LtPlan(record, base64.b64decode(record["algorithm_b64"]), workspace, x=x,
                                weight=weights[i], packed=packed_views[i] if record["layout"] == "transposed" else None)
                holders.append(holder)
                plans.append(holder)
            name = f"lt_{record['layout']}_h{record['candidate_idx']}"
            add(name, lambda i, out, holders=holders: holders[i].run(x, weights[i], out), record)
        add("cublas_compute32", lambda i, out: vendor.mm_compute32(x, weights[i], out, dtype),
            {"kind": "cublasGemmEx", "compute_type": 68, "layout": "original"})
        add("cublas_packed_compute32", lambda i, out: vendor.mm_compute32(x, packed_views[i], out, dtype, True),
            {"kind": "cublasGemmEx", "compute_type": 68, "layout": "transposed"})
        if dtype == "fp32":
            add("torch_mm", lambda i, out: torch.mm(x, weights[i].T, out=out), {"kind": "torch.mm"})
            if m == 1:
                add("torch_mv", lambda i, out: torch.mv(weights[i], x[0], out=out[0]), {"kind": "torch.mv"})
                add("cublas_mv", lambda i, out: vendor.mv(x, weights[i], out), {"kind": "cublasSgemv", "layout": "original"})
                add("cublas_packed_mv", lambda i, out: vendor.mv_packed(x, packed_views[i], out),
                    {"kind": "cublasSgemv", "layout": "transposed"})
        else:
            try:
                torch.mm(x, weights[0].T, out_dtype=torch.float32)
            except (TypeError, RuntimeError) as error:
                diagnostics.append({"method": "torch_mm_out_fp32", "unavailable": str(error)})
            else:
                add("torch_mm_out_fp32", lambda i, out: torch.mm(x, weights[i].T, out_dtype=torch.float32, out=out),
                    {"kind": "torch.mm", "out_dtype": "fp32"})
        reference = x.double() @ weight.double().T
        before = {}
        for name, function in functions.items():
            function()
            before[name] = check_ring(reference, outputs[name], output_dtype="fp32")
        torch.cuda.synchronize()
        context = current_context(scope="graph")

        class PhaseMonitor(GpuMonitor):
            phase = "preparation"
            def __init__(self):
                import threading
                super().__init__()
                self.sample_lock = threading.Lock()
            def sample(self):
                with self.sample_lock:
                    phase, start = self.phase, len(self.samples)
                    super().sample()
                    for item in self.samples[start:]:
                        item["phase"] = phase if phase == self.phase else "boundary"

        with PhaseMonitor() as monitor:
            warming = time.monotonic()
            while time.monotonic() - warming < 2:
                for function in functions.values():
                    function()
                torch.cuda.synchronize()
            measured = measure(functions, rounds=protocol["minimum_rounds"], repeats=1,
                               phase_callback=lambda phase: setattr(monitor, "phase", phase),
                               round_callback=lambda index: monitor.sample() if index == 0 else None)
        after_context = current_context(scope="graph")
        for name, timing in measured.items():
            for field in ("median_us", "min_us", "max_us"):
                timing[field] /= copies
            timing["samples_us"] = [v / copies for v in timing["samples_us"]]
            timing["numerics"] = check_ring(reference, outputs[name], output_dtype="fp32")
            timing["numerics"]["passed"] = bool(timing["numerics"]["passed"] and before[name]["passed"])
        preserved = bool(np.array_equal(weight.view(integer).cpu().numpy(), original)
                         and np.array_equal(x.view(integer).cpu().numpy(), xbits)
                         and torch.equal(ring.view(integer), weight.view(integer).expand_as(ring))
                         and torch.equal(transposed.view(integer).transpose(1, 2), ring.view(integer)))
        summary = summarize_methods(measured, list(routes), "custom")
        clocks = {item["state"].split(",")[5].strip() for item in monitor.samples
                  if item.get("ok") and item.get("phase") == "measurement"}
        clock_stable = len(clocks) == 1
        accepted = bool(summary["passes_10pct_gate"] and exact and preserved and context == after_context and clock_stable)
        return {**spec, "seed": seed, "input_dtype": dtype, "weight_dtype": dtype,
                "output_dtype": "fp32", "accumulation_dtype": "fp32", "copies": copies,
                "calls_per_graph": copies, "raw_ring_bytes": ring.numel() * ring.element_size(),
                "packed_ring_bytes": sum(p.storage_bytes for p in packed), "packing_seconds": packing_seconds,
                "weight_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                "activation_sha256": hashlib.sha256(xbits.tobytes()).hexdigest(),
                "original_bits_preserved": preserved, "bitwise_round_trip": exact, "bitwise_layout_check": True,
                "numerics_before": before, "methods": measured, "vendor_routes": list(routes), "vendor_pool": routes,
                "search_diagnostics": diagnostics, "summary": summary, "accepted_10pct_target": accepted,
                "observed_memory_clocks": sorted(clocks), "memory_clock_regime_stable": clock_stable,
                "context_before": context, "context_after": after_context, "gpu_telemetry": monitor.samples,
                "setup_and_measurement_seconds": time.monotonic() - started}
    finally:
        for plan in plans:
            plan.close()


def run(dtype, seed, profile, output):
    from experiment_control import project_path, write_json
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    cases = frozen_cases(protocol, dtype)
    if seed not in protocol["held_out_seeds"] or profile not in protocol["profiles"]:
        raise ValueError("Seed/profile is not registered")
    output = project_path(ROOT, output)
    if output.exists() or output.suffix != ".json":
        raise ValueError("Use a new project-local JSON report")
    import torch
    import kernel_lab
    import nvidia.cublas
    from bench_dispatch import current_context
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from cuda_module import _nvrtc_identity
    from experiment_provenance import environment, source_hashes

    torch.set_grad_enabled(False)
    policy = torch.backends.cuda.matmul
    policy.allow_tf32 = False
    policy.allow_fp16_reduced_precision_reduction = False
    policy.allow_bf16_reduced_precision_reduction = False
    if hasattr(policy, "allow_fp16_accumulation"):
        policy.allow_fp16_accumulation = False
    context = current_context(scope="graph")
    if not context["ac_online"] or context["system76_power_profile"].lower() != "power profile: " + profile:
        raise ValueError("Registered profile and AC condition do not match")
    lab, vendor, workspace = kernel_lab.default_lab(), CuBlas(), LtWorkspace(0)
    ptx = project_path(ROOT, lab.compilation["cache_file"])
    sources = source_hashes((PROTOCOL, "native_precision.py", "frozen_transfer.py", "half_projection.py",
                              "projection_accuracy.py", "compressed_campaign.py", "cublaslt_baseline.py",
                              "bench_dispatch.py", "bench_compressed.py", "power_context.py", "gpu_unlock_status.py"))
    inputs = {ref["path"]: ref["sha256"] for ref in protocol["source_profiles"].values()}
    base = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    libraries = {str(base / name): _sha(base / name) for name in ("libcublas.so.12", "libcublasLt.so.12")}
    compiler, compiler_path = _nvrtc_identity()
    report = {"kind": "native-precision-projection", "status": "running", "dtype": dtype,
              "seed": seed, "pid": os.getpid(), "run_id": str(uuid.uuid4()),
              "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "rounds": protocol["minimum_rounds"], "operator_context": context,
              "source_sha256": sources, "source_profiles_sha256": inputs,
              "environment": environment(lab, vendor), "libraries_sha256": libraries,
              "ptx_path": str(ptx.relative_to(ROOT)), "ptx_sha256": _sha(ptx), "nvrtc_identity": compiler,
              "nvrtc_path": compiler_path, "compilation": lab.compilation,
              "precision_flags": {name: getattr(policy, name) for name in ("allow_tf32", "allow_fp16_reduced_precision_reduction",
                                                                          "allow_bf16_reduced_precision_reduction")},
              "scope": protocol["measurement"], "custom_win_validated": False,
              "unlocked_comparison": "pending", "energy_claim": False, "cases": []}
    write_json(output, report)
    try:
        for spec in cases:
            report["active_case"] = spec
            write_json(output, report)
            case = _case(spec, dtype, seed, workspace, vendor, protocol)
            report["cases"].append(case)
            write_json(output, report)
            gain = case["summary"]["time_saved_pct"]
            print(f"{dtype} {spec['shape_mnk']} {spec['config']}: "
                  f"{gain:+.3f}% accepted={case['accepted_10pct_target']}", flush=True)
            if case["context_after"] != context:
                raise RuntimeError("Profile/AC changed; failed case retained")
            torch.cuda.empty_cache()
        for name, digest in {**sources, **inputs, **libraries, str(ptx): report["ptx_sha256"]}.items():
            path = project_path(ROOT, name) if not Path(name).is_absolute() else Path(name)
            if (path.exists() and digest is None) or (digest is not None and _sha(path) != digest):
                raise ValueError(f"Measured dependency changed: {name}")
        if _nvrtc_identity()[0] != compiler:
            raise ValueError("NVRTC changed during the study")
        report.pop("active_case", None)
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output, report)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(output, report)
        raise
    finally:
        workspace.close()
        vendor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--power-profile", choices=("performance", "battery"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch"),
                           ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor")):
        os.environ.setdefault(key, str(ROOT / directory))
    from experiment_control import GpuLock, project_path, write_json
    from frozen_transfer import temporary_profile, _interrupt
    output = project_path(ROOT, args.output)
    state_path = output.with_name(output.stem + "-state.json")
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    frozen_cases(protocol, args.dtype)
    if output.exists() or state_path.exists() or args.seed not in protocol["held_out_seeds"]:
        raise ValueError("Use a fresh registered seed/profile/dtype output")
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    state = {"pid": os.getpid(), "status": "running"}
    try:
        with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
            with temporary_profile(args.power_profile, state):
                run(args.dtype, args.seed, args.power_profile, output)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        write_json(state_path, state)


if __name__ == "__main__":
    main()
