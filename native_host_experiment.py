"""Registered scoped API experiment for the frozen CUDA native host binding.

No CUDA import at import/help, no kernel search and no production promotion.
This measures allocating functional APIs against direct Torch, not the full
vendor pool. The older kernel evidence retains that separate comparison.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROTOCOL = "experiments/native_prepared_scope.json"


def case_specs(protocol, corpus):
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    result = []
    for output in protocol["output_contracts"]:
        for case in protocol["cases"]:
            if "projection" in case:
                name = f"model.layers.{protocol['weights_layer']}.{case['projection']}"
                shape = [case["m"], *entries[name]["shape"]]
            else:
                shape = case["synthetic_shape_mnk"]
                name = "synthetic-control." + ",".join(map(str, shape))
            result.append({"weight_tensor": name, "shape_mnk": shape,
                           "output_dtype": output, "control": "projection" not in case})
    return result


def summarize_scope(methods, *, selected_custom, numerics_passed, context_stable):
    from bench_compressed import paired_gain_interval
    values = {key: method["samples_us"] for key, method in methods.items()}
    if (len({len(v) for v in values.values()}) != 1 or min(map(len, values.values())) < 31
            or any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
                   for row in values.values() for v in row)):
        raise ValueError("Need complete paired positive finite raw rounds")
    medians = {key: statistics.median(row) for key, row in values.items()}
    aa = 100 * (1 - medians["vendor_aa"] / medians["vendor"])
    common = bool(abs(aa) <= 1 and numerics_passed and context_stable)
    result = {"aa_time_saved_pct": aa, "common_gates_passed": common,
              "custom_win_validated": False, "eager_promoted": False}
    for key in ("native", "python_registered", "selected"):
        interval = paired_gain_interval({"vendor": values["vendor"]}, values[key])
        selected_gain = key == "selected" and selected_custom
        result[key] = {"median_us": medians[key],
                       "time_saved_pct": 100 * (1 - medians[key] / medians["vendor"]),
                       "ci95_pct": interval,
                       "case_gate_passed": bool(common and (interval[0] > 10 if selected_gain else interval[0] >= -1)),
                       "custom_gain_claim": bool(selected_gain and common and interval[0] > 10)}
    result["vendor_median_us"] = medians["vendor"]
    return result


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_path, *, seed):
    import numpy as np
    import torch
    import compressed_bf16 as cb
    import mtile_bf16 as mt
    import native_prepared as native
    import prepared_experiment as pe
    from bench import measure, gpu_state
    from bench_dispatch import _eager, current_context
    from compressed_campaign import _identity, check_ring
    from compressed_case import launch_config, _reference
    from cublas_baseline import CuBlas
    from experiment_control import project_path, write_json
    from gpu_monitor import GpuMonitor
    from gpu_unlock_status import inspect
    from prepared_kernels import coded_r2, preload_modules
    from real_weights import read_tensor

    output = project_path(ROOT, output_path)
    if output.exists():
        raise ValueError("Preserve previous results; choose a new output path")
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if seed not in protocol["seeds"]:
        raise ValueError("Seed is not registered in the native host protocol")
    rounds = protocol["rounds"]
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(seed)
    evidence, verified, inputs = {}, {}, {}
    for name in protocol["output_contracts"]:
        path = ROOT / f"results/compressed-performance-{name}-evidence-r2.json"
        evidence[name] = json.loads(path.read_text())
        # Replay the old evidence with its registered seed, without pretending
        # the new native-host activation seed was part of that kernel study.
        verified[name] = pe.verify_artifacts(path, evidence[name], seed=919, rounds=rounds)
        refs = evidence[name]["inputs"]
        for ref in [refs[key] for key in ("spec", "corpus", "profile", "training")] + refs["reports"]:
            inputs[ref["path"]] = ref["sha256"]
        inputs[str(path.relative_to(ROOT))] = _sha(path)
    corpus = verified["bf16"]["corpus"]
    inventory = inspect()
    vendor = CuBlas()
    identity = _identity(vendor, inventory, revision=2)
    for name in evidence:
        if identity != evidence[name]["bindings"]["identity"]:
            raise ValueError("Current frozen CUDA/runtime/device identity differs from the kernel evidence")
    build = native.ensure_loaded()
    preload_modules(0)
    for name, digest in build["runtime_library_sha256"].items():
        if _sha(build["identity"]["runtime_libraries"][name]["path"]) != digest:
            raise ValueError("Native linked runtime bytes changed after compilation")
    context = current_context(scope="graph")
    if not context["ac_online"] or context["system76_power_profile"] != "Power Profile: Performance":
        raise ValueError("The registered native-host context is Performance on AC")
    sources = set(pe.SOURCES) | {PROTOCOL, "native_host_experiment.py", "native/prepared.cpp", "native_prepared.py"}
    hashes = {name: _sha(ROOT / name) for name in sorted(sources)}
    report = {"kind": "native-host-projection-api", "status": "running", "seed": seed, "rounds": rounds,
              "pid": os.getpid(), "run_id": str(uuid.uuid4()), "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Allocating functional APIs versus direct Torch; not full vendor pool superiority",
              "protocol": PROTOCOL, "source_sha256": hashes, "input_files_sha256": inputs,
              "inventory": inventory, "identity": identity, "native_build": build,
              "operator_context": context, "cases": [], "custom_win_validated": False,
              "eager_promoted": False, "unlocked_comparison": "pending", "energy_claim": False}
    write_json(output, report)

    def freeze_check():
        for name, digest in {**hashes, **inputs}.items():
            if _sha(project_path(ROOT, name)) != digest:
                raise ValueError(f"Study source/input changed: {name}")
        for name, digest in build["runtime_library_sha256"].items():
            if _sha(build["identity"]["runtime_libraries"][name]["path"]) != digest:
                raise ValueError("Native linked runtime changed during the study")

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
                for sample in self.samples[start:]:
                    sample["phase"] = phase if phase == self.phase else "boundary"

    try:
        for spec in case_specs(protocol, corpus):
            started = time.monotonic()
            m, n, k = spec["shape_mnk"]
            name = spec["output_dtype"]
            if spec["control"]:
                w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
                bits = w.view(torch.uint16).cpu().numpy().copy()
                config, selected_custom = "coded_t64_v8", False
            else:
                bits = np.array(read_tensor(corpus, spec["weight_tensor"]), copy=True)
                w = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
                config = verified[name]["profile"]["choices"][f"{n},{k}"]
                selected_custom = pe.candidate_plan(evidence[name], [m, n, k], output_dtype=name)["route"] == "coded"
            threads, vector = launch_config(config)
            copies = max(2, math.ceil(max(protocol["physical_original_weight_ring_bytes_min"],
                                           20 * torch.cuda.get_device_properties(0).L2_cache_size) / (n * k * 2)))
            weights = [w] + [w.clone() for _ in range(copies - 1)]
            packed_cpu = cb.pack_bits(bits)
            packed = [cb.upload(packed_cpu, 0) for _ in range(copies)]
            storage_ratio = packed_cpu.storage_bytes / (n * k * 2)
            if storage_ratio >= 1:
                selected_custom = False
            candidate = native.NativeCodedCandidate(packed[0], tokens=m, threads=threads,
                                                     vector=vector, output_dtype=name)
            plan = candidate.plan_id
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
            views = [weight.T for weight in weights]
            payloads, metadata = [p.payload for p in packed], [p.metadata for p in packed]
            native_op = torch.ops.klab_native_v1.coded.default
            python_op = coded_r2
            if name == "fp32":
                direct = lambda i: torch.mm(x, views[i % copies], out_dtype=torch.float32)
            else:
                direct = lambda i: torch.mm(x, views[i % copies])
            functions = {"vendor": direct, "vendor_aa": direct,
                         "native": lambda i: native_op(x, payloads[i % copies], metadata[i % copies], plan,
                                                        m, n, k, threads, vector, name),
                         "python_registered": lambda i: python_op(x, payloads[i % copies], metadata[i % copies],
                                                                  m, n, k, threads, vector, name)}
            functions["selected"] = functions["native"] if selected_custom else direct
            reference = _reference(x, w)
            integer = torch.int16 if name == "bf16" else torch.int32
            expected = mt.fused_linear(x, packed[0], threads=threads, vector=vector, output_dtype=name)
            bit_parity = torch.equal(functions["native"](0).view(integer), expected.view(integer))
            if not bit_parity:
                raise RuntimeError("Native output bits differ from the frozen original wrapper")
            before_numerics = {}
            for arm, function in functions.items():
                allocated = torch.stack([function(i) for i in range(copies)])
                before_numerics[arm] = check_ring(reference, allocated, output_dtype=name)
                del allocated
            torch.cuda.synchronize()
            eager_before, eager_gpu_before = current_context(scope="eager"), gpu_state()
            eager = _eager(functions, rounds=rounds,
                           calls=max(protocol["eager_calls_per_round_min"], copies))
            eager_after, eager_gpu_after = current_context(scope="eager"), gpu_state()
            eager_numerics = {}
            for arm, function in functions.items():
                allocated = torch.stack([function(i) for i in range(copies)])
                eager_numerics[arm] = check_ring(reference, allocated, output_dtype=name)
                del allocated
            captured_outputs = {}

            def sweep(arm):
                captured_outputs[arm] = [functions[arm](i) for i in range(copies)]

            graph_before = current_context(scope="graph")
            with PhaseMonitor() as monitor:
                # Same two-second sustained warmup for all allocating arms.
                warmup = time.monotonic()
                while time.monotonic() - warmup < 2:
                    for arm in functions:
                        sweep(arm)
                    torch.cuda.synchronize()
                graph = measure({arm: lambda arm=arm: sweep(arm) for arm in functions},
                                rounds=rounds, repeats=1,
                                phase_callback=lambda phase: setattr(monitor, "phase", phase),
                                round_callback=lambda index: monitor.sample() if index == 0 else None)
            graph_after = current_context(scope="graph")
            for method in graph.values():
                for field in ("median_us", "min_us", "max_us"):
                    method[field] /= copies
                method["samples_us"] = [v / copies for v in method["samples_us"]]
            graph_numerics = {arm: check_ring(reference, torch.stack(values), output_dtype=name)
                              for arm, values in captured_outputs.items()}
            preserved = np.array_equal(w.view(torch.uint16).cpu().numpy(), bits)
            numeric = bool(bit_parity and preserved and all(c["passed"] for pool in
                           (before_numerics, eager_numerics, graph_numerics) for c in pool.values()))
            expected_clocks = set(verified[name]["profile"]["observed_memory_clocks"])
            clocks = {sample["state"].split(",")[5].strip() for sample in monitor.samples
                      if sample.get("ok") and sample.get("phase") == "measurement"}
            clock_match = bool(clocks and clocks <= expected_clocks)
            eager_clock_match = all(s.split(",")[6].strip() in expected_clocks
                                    for s in (eager_gpu_before, eager_gpu_after))
            case = {**spec, "copies": copies, "raw_ring_bytes": copies * n * k * 2,
                    "packed_ring_bytes": sum(p.storage_bytes for p in packed), "storage_ratio": storage_ratio,
                    "config": config, "selected_route": "native" if selected_custom else "vendor",
                    "native_plan": candidate.describe(), "native_original_bit_parity": bool(bit_parity),
                    "original_bits_preserved": bool(preserved), "weight_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                    "activation_sha256": hashlib.sha256(x.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                    "numerics_before": before_numerics,
                    "eager": {"methods": eager, "numerics_after": eager_numerics,
                              "context_before": eager_before, "context_after": eager_after,
                              "gpu_before": eager_gpu_before, "gpu_after": eager_gpu_after,
                              "summary": summarize_scope(eager, selected_custom=selected_custom,
                                   numerics_passed=numeric, context_stable=eager_before == eager_after and eager_clock_match
                                       and eager_before["ac_online"]
                                       and eager_before["system76_power_profile"].startswith("Power Profile: Performance\n"))},
                    "graph": {"methods": graph, "numerics_after": graph_numerics,
                              "context_before": graph_before, "context_after": graph_after,
                              "gpu_telemetry": monitor.samples, "observed_memory_clocks": sorted(clocks),
                              "summary": summarize_scope(graph, selected_custom=selected_custom, numerics_passed=numeric,
                                   context_stable=graph_before == graph_after == context and clock_match)},
                    "setup_and_measurement_seconds": time.monotonic() - started}
            report["cases"].append(case)
            write_json(output, report)
            print(f"{spec['weight_tensor']} M={m} {name} route={case['selected_route']}: "
                  f"eager {case['eager']['summary']['selected']['time_saved_pct']:+.2f}%, "
                  f"graph {case['graph']['summary']['selected']['time_saved_pct']:+.2f}%", flush=True)
            candidate.close()
            captured_outputs.clear()
            functions.clear()
            del candidate, weights, views, packed, payloads, metadata, w, x, reference, packed_cpu
            torch.cuda.empty_cache()
        freeze_check()
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      scope_controls={scope: all(c[scope]["summary"]["selected"]["case_gate_passed"]
                                               for c in report["cases"] if c["control"])
                                      for scope in ("eager", "graph")})
        write_json(output, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch"),
                           ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor")):
        os.environ.setdefault(key, str(ROOT / directory))
    from experiment_control import GpuLock
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        run(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
