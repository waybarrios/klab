"""Frozen, preregistered lossless-BF16 projection experiments."""
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def experiment_layers(spec, mode, seed):
    if mode == "tune" and seed == spec["tuning_seed"]:
        return list(spec["tuning_layers"])
    if mode == "validate" and seed in spec["held_out_seeds"]:
        return list(spec["held_out_layers"])
    raise ValueError("Tuning and held-out seeds/layers are separate")


def summarize_methods(methods, routes, selected):
    from bench_compressed import paired_gain_interval
    eligible = [name for name in routes if not name.endswith("_aa")
                and methods[name]["numerics"]["passed"]
                and methods[name + "_aa"]["numerics"]["passed"]]
    if not eligible:
        return {"comparison_status": "no-eligible-vendor", "best_vendor": None, "best_vendor_us": None,
                "eligible_vendors": [], "aa_time_saved_pct": None, "aa_passed": False,
                "time_saved_pct": None, "ci95_pct": None, "passes_10pct_gate": False}
    best = min(eligible, key=lambda name: methods[name]["median_us"])
    baseline = methods[best]["median_us"]
    aa = 100 * (baseline - methods[best + "_aa"]["median_us"]) / baseline
    interval = paired_gain_interval({name: methods[name]["samples_us"] for name in eligible},
                                    methods[selected]["samples_us"])
    aa_passed = abs(aa) <= 1
    return {"comparison_status": "compared", "best_vendor": best, "best_vendor_us": baseline, "eligible_vendors": eligible,
            "aa_time_saved_pct": aa, "aa_passed": aa_passed,
            "time_saved_pct": 100 * (1 - methods[selected]["median_us"] / baseline),
            "ci95_pct": interval,
            "passes_10pct_gate": bool(aa_passed and interval[0] > 10 and methods[selected]["numerics"]["passed"])}


def check_ring(reference64, output, *, output_dtype):
    from projection_accuracy import check_outputs
    if output.ndim != 3 or output.shape[0] < 1:
        raise ValueError("Expected output[distinct weight copies,M,N]")
    if hasattr(output, "detach"):
        import torch
        expected = torch.bfloat16 if output_dtype == "bf16" else torch.float32
        if output.dtype != expected:
            raise ValueError("Output dtype differs from its declared contract")
        integer = torch.int16 if output_dtype == "bf16" else torch.int32
        bits = output.view(integer)
        equal = torch.equal(bits, bits[0].expand_as(bits))
    else:
        array = np.ascontiguousarray(output)
        bits = array.view(np.uint8).reshape(array.shape[0], -1)
        equal = np.array_equal(bits, np.broadcast_to(bits[0], bits.shape))
    if equal:
        result = check_outputs(reference64, output[0], output_dtype=output_dtype)
        oracle_elements = int(np.prod(output.shape[1:]))
    else:
        if hasattr(reference64, "detach"):
            expanded = reference64.expand(output.shape)
        else:
            expanded = np.broadcast_to(reference64, tuple(output.shape))
        result = check_outputs(expanded, output, output_dtype=output_dtype)
        oracle_elements = int(np.prod(output.shape))
    return {**result, "elements_checked": int(np.prod(output.shape)),
            "oracle_elements_checked": oracle_elements, "all_copies_bitwise_equal": bool(equal),
            "coverage": "Every output bit compared across copies; independent FP64 oracle on identical representative, otherwise all elements"}


def freeze_choices(cases, spec, *, output_dtype):
    from compressed_case import launch_config
    expected = {f"model.layers.{layer}.{suffix}" for layer in spec["tuning_layers"]
                for suffix in spec["projection_weights"]}
    actual = [case["weight_tensor"] for case in cases]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Freeze requires each registered training layer/projection exactly once")
    groups = {}
    for case in cases:
        if (case["layer"] not in spec["tuning_layers"] or case["shape_mnk"][0] != 1
                or case["output_dtype"] != output_dtype):
            raise ValueError("Only M1 training cases can choose a configuration")
        key = ",".join(map(str, case["shape_mnk"][1:]))
        groups.setdefault(key, []).append(case)
    choices = {}
    for key, group in groups.items():
        candidates = set.intersection(*(set(case["methods"]) for case in group))
        eligible = []
        for name in sorted(candidates):
            if not name.startswith("coded_"):
                continue
            launch_config(name)
            if all(case["methods"][name]["numerics"]["passed"]
                   and case.get("raw_template_parity_by_config", {}).get(name, True) for case in group):
                eligible.append(name)
        if not eligible:
            raise ValueError(f"No numerically valid configuration for all training weights at {key}")
        choices[key] = min(eligible, key=lambda name: (float(np.median([case["methods"][name]["median_us"]
                                                                       for case in group])), name))
    return choices


def _expected_identities(spec, corpus, mode, seed):
    layers = experiment_layers(spec, mode, seed)
    ms = [1] if mode == "tune" else spec["m_values"] + spec["baseline_diagnostic_m"]
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    expected = set()
    for layer in layers:
        for suffix in spec["projection_weights"]:
            name = f"model.layers.{layer}.{suffix}"
            if name not in entries:
                raise ValueError(f"Missing registered tensor {name}")
            for m in ms:
                expected.add((name, m, *entries[name]["shape"]))
    head = spec["head"]["tensor"]
    if mode == "validate" and head in entries:
        expected.update((head, m, *entries[head]["shape"]) for m in ms)
    return expected


def assert_complete(report, spec, corpus):
    expected = _expected_identities(spec, corpus, report["mode"], report["seed"])
    if not report.get("pid") or not report.get("run_id") or report.get("output_dtype") not in ("bf16", "fp32"):
        raise ValueError("A complete experiment needs process identity and output contract")
    observed = []
    for case in report["cases"]:
        if case["seed"] != report["seed"] or case["output_dtype"] != report["output_dtype"]:
            raise ValueError("Case seed/output contract differs from its process")
        if case.get("role") != "untuned-training-diagnostic":
            observed.append((case["weight_tensor"], *case["shape_mnk"]))
        samples = case.get("gpu_telemetry", [])
        if not samples or not any((s.get("ok") is True or s.get("returncode") == 0)
                                  and s.get("phase") == "measurement" for s in samples):
            raise ValueError("Missing successful telemetry within timed rounds")
        if not all(isinstance(case.get(key), bool) for key in ("original_bits_preserved", "raw_template_bitwise")):
            raise ValueError("Missing weight-bit or raw/compressed parity evidence")
        if not case.get("methods"):
            raise ValueError("Missing measured methods")
        for method in case["methods"].values():
            rounds = np.asarray(method.get("samples_us", []), dtype=np.float64)
            if (rounds.ndim != 1 or len(rounds) < spec["minimum_rounds"]
                    or not np.isfinite(rounds).all() or not (rounds > 0).all()
                    or "passed" not in method.get("numerics", {})
                    or method["numerics"].get("elements_checked", 0) <= 0):
                raise ValueError("Missing paired raw rounds or complete numerical checks")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Missing, duplicated or unexpected registered cases")
    if report["mode"] == "validate" and spec["head"]["tensor"] not in {t["name"] for t in corpus["tensors"]}:
        head = report.get("head_diagnostic", {})
        if head.get("status") != "unavailable" or not head.get("reason"):
            raise ValueError("Head omission requires an explicit storage/unavailable reason")


def validate_process_set(reports, spec):
    if len(reports) != len(spec["held_out_seeds"]) or {r["seed"] for r in reports} != set(spec["held_out_seeds"]):
        raise ValueError("Need exactly the three registered held-out seeds")
    for field in ("pid", "run_id"):
        if any(not r.get(field) for r in reports) or len({r[field] for r in reports}) != len(reports):
            raise ValueError("Validation requires distinct fresh process identities")
    for report in reports:
        if report["status"] != "complete" or report["mode"] != "validate":
            raise ValueError("Incomplete or non-validation result")
        for field in ("profile_sha256", "output_dtype", "operator_context"):
            if not report.get(field) or report[field] != reports[0][field]:
                raise ValueError(f"Validation cohorts differ: {field}")
    return all(report["accepted_10pct"] for report in reports)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(vendor, inventory, revision=1):
    import torch
    import nvidia.cublas
    import compressed_bf16 as cb
    if revision == 2:
        import mtile_bf16 as kernel
    else:
        kernel = cb
    from cublaslt_baseline import _library_version
    base = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    sources = ("compressed_case.py", "compressed_campaign.py", "bench_compressed.py", "bench.py",
               "projection_accuracy.py", "bf16_reference.py", "cublaslt_baseline.py", "cublas_baseline.py",
               "gpu_monitor.py", "real_weights.py")
    sources += ("power_context.py",)
    if revision == 2:
        sources += ("mtile_bf16.py", "mtile_kernels.cu")
    return {"implementation": kernel.implementation_fingerprint(0),
            "source_sha256": {name: _sha(ROOT / name) for name in sources},
            "libraries_sha256": {name: _sha(base / name) for name in ("libcublas.so.12", "libcublasLt.so.12")},
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cublas_version": vendor.version, "cublaslt_version": _library_version(),
            "gpu": inventory["gpu"], "kernel": inventory["kernel"],
            "unlock_evidence": {key: inventory[key] for key in ("module_loaded", "unlock_boot_option_present",
                                                                "unlock_success_logged", "post_flr_selectors_logged")},
            "allow_tf32": False, "reduced_precision_reduction": False}


def _memory_clocks(cases):
    return sorted({sample["state"].split(",")[5].strip() for case in cases
                   for sample in case["gpu_telemetry"] if sample.get("ok") and sample.get("phase") == "measurement"
                   and len(sample.get("state", "").split(",")) >= 6})


def run(spec_path, corpus_path, vendor_path, *, mode, seed, profile_path, output_path,
        output_dtype="bf16", revision=1, rounds=31, run_metadata=None, diagnostic_training=False):
    """Caller owns GpuLock. Validation only replays a complete frozen profile."""
    import torch
    from bench_compressed import _power_context, validate_probe_inputs
    from compressed_case import measure_case
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from experiment_control import project_path, with_run_metadata, write_json
    from gpu_unlock_status import inspect
    from power_context import execution_context

    campaign_start = time.monotonic()
    spec_path, corpus_path, vendor_path, profile_path, output_path = [project_path(ROOT, p) for p in
        (spec_path, corpus_path, vendor_path, profile_path, output_path)]
    if len({spec_path, corpus_path, vendor_path, profile_path, output_path}) != 5:
        raise ValueError("Experiment inputs, profile and raw output must be distinct")
    if output_path.exists() or (mode == "tune" and profile_path.exists()):
        raise ValueError("Preserve prior measurements; use new output/profile paths")
    spec, corpus, frozen = [json.loads(path.read_text()) for path in (spec_path, corpus_path, vendor_path)]
    layers = experiment_layers(spec, mode, seed)
    validate_probe_inputs(spec, corpus, frozen, seed=spec["tuning_seed"])
    if (frozen.get("spec_sha256") != _sha(spec_path)
            or frozen["source_sha256"].get("cublaslt_baseline.py") != _sha(ROOT / "cublaslt_baseline.py")):
        raise ValueError("Frozen vendor experiment contract or precision/ABI dependency differs")
    if output_dtype not in ("bf16", "fp32") or revision not in (1, 2) or rounds < spec["minimum_rounds"]:
        raise ValueError("Invalid output contract, performance revision or paired round count")
    if diagnostic_training and mode != "tune":
        raise ValueError("Training diagnostics cannot run on held-out data")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    power = _power_context()
    compute_context = execution_context(power)
    inventory = inspect()
    vendor = CuBlas()
    workspace = LtWorkspace(0)
    identity = _identity(vendor, inventory, revision)
    bindings = {"spec_sha256": _sha(spec_path), "corpus_sha256": _sha(corpus_path),
                "vendor_profile_sha256": _sha(vendor_path), "output_dtype": output_dtype,
                "context_protocol_sha256": _sha(ROOT / "experiments/consumer_ampere_execution_context.json"),
                "revision": revision, "operator_context": compute_context, "identity": identity}
    profile = None
    if mode == "validate":
        profile = json.loads(profile_path.read_text())
        if profile.get("kind") != "compressed-projection-tuning" or profile.get("status") != "frozen":
            raise ValueError("Need a completed frozen compressed tuning profile")
        if profile.get("bindings") != bindings:
            raise ValueError("Frozen profile inputs, sources, runtime, output or operating context are stale")
        training_path = project_path(ROOT, profile["training_report"])
        if profile.get("training_report_sha256") != _sha(training_path):
            raise ValueError("Frozen tuning evidence was changed after configuration selection")
        training_report = json.loads(training_path.read_text())
        if (training_report.get("status") != "complete" or training_report.get("mode") != "tune"
                or training_report.get("choices") != profile["choices"]):
            raise ValueError("Frozen profile does not match its completed tuning evidence")
        assert_complete(training_report, spec, corpus)
        if not profile.get("minimum_12pct_tuning_screen"):
            raise ValueError("Training did not pass the preregistered large-shape screen")
        if seed == profile.get("seed") or os.getpid() == profile.get("pid"):
            raise ValueError("Validation must run in a fresh held-out process")
    metadata = {"pid": os.getpid(), "run_id": str(uuid.uuid4()), "created_utc": datetime.now(timezone.utc).isoformat()}
    if run_metadata:
        metadata.update(run_metadata)
    if metadata["pid"] != os.getpid():
        raise ValueError("Report PID must be the actual experiment process")
    report = with_run_metadata({"kind": "compressed-projection", "status": "running", "mode": mode,
        "seed": seed, **bindings, "inventory": inventory, "operator_display_snapshot": power,
        "profile_sha256": _sha(profile_path) if profile else None,
        "vendor_tuning_oracle_sha256": frozen["source_sha256"].get("projection_accuracy.py"),
        "cases": [], "accepted_10pct": False,
        "scope": "Streaming CUDA graph GPU latency, real native BF16 weights and synthetic seeded activations",
        "full_model_speedup_claim": False, "unlocked_comparison": "pending"}, metadata)
    write_json(output_path, report)
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    vendor_cases = {(tuple(case["shape_mnk"]), case["output_dtype"]): case for case in frozen["cases"]
                    if case["input_dtype"] == "bf16"}
    configs = [f"coded_t{threads}_v{vector}" for threads in spec["kernel_search"]["threads"]
               for vector in spec["kernel_search"]["vectors"]]
    choices = profile["choices"] if profile else None

    def execute(name, m, layer, role, settings):
        entry = entries[name]
        n, k = entry["shape"]
        vendor_case = vendor_cases.get(((m, n, k), output_dtype))
        if vendor_case is None and role != "head-transfer-diagnostic":
            raise ValueError(f"Missing frozen vendor operation M,N,K={m,n,k}, output={output_dtype}")
        case = measure_case(entry, corpus, m, output_dtype, seed, settings, vendor_case, workspace, vendor,
                            rounds=rounds, kernel_revision=revision,
                            decode_diagnostic=mode == "tune" and layer == 0 and m == 1
                            and [n, k] in spec["large_primary_shapes_nk"])
        case.update(layer=layer, role=role)
        report["cases"].append(case)
        if len(settings) == 1:
            case["selected"] = settings[0]
            case.update(summarize_methods(case["methods"], case["vendor_routes"], settings[0]))
        write_json(output_path, report)
        gain = f"{case['time_saved_pct']:+.3f}%" if case.get("time_saved_pct") is not None else "8 configurations / comparison pending"
        print(f"{mode} {name} M={m} {output_dtype}: {gain}", flush=True)
        if not case["operating_context_stable"] or execution_context(_power_context()) != compute_context:
            raise RuntimeError("Power profile/AC changed during the GPU-graph campaign; measured case retained")

    try:
        for layer in layers:
            for suffix in spec["projection_weights"]:
                name = f"model.layers.{layer}.{suffix}"
                n, k = entries[name]["shape"]
                for m in ([1] if mode == "tune" else spec["m_values"] + spec["baseline_diagnostic_m"]):
                    execute(name, m, layer, "training" if mode == "tune" else "held-out",
                            configs if choices is None else [choices[f"{n},{k}"]])
        if mode == "tune":
            choices = freeze_choices(report["cases"], spec, output_dtype=output_dtype)
            for case in report["cases"]:
                n, k = case["shape_mnk"][1:]
                case["selected"] = choices[f"{n},{k}"]
                case.update(summarize_methods(case["methods"], case["vendor_routes"], case["selected"]))
            if diagnostic_training:
                for suffix in spec["projection_weights"]:
                    name = f"model.layers.0.{suffix}"
                    n, k = entries[name]["shape"]
                    for m in spec["m_values"] + spec["baseline_diagnostic_m"]:
                        if m != 1:
                            execute(name, m, 0, "untuned-training-diagnostic", [choices[f"{n},{k}"]])
        else:
            head = spec["head"]["tensor"]
            if head in entries:
                report["head_diagnostic"] = {"status": "included", "config_source": "frozen gate/up winner"}
                for m in spec["m_values"] + spec["baseline_diagnostic_m"]:
                    execute(head, m, None, "head-transfer-diagnostic", [choices["4864,896"]])
            else:
                report["head_diagnostic"] = {"status": "unavailable", "reason": "Head omitted from verified retained corpus"}
        assert_complete(report, spec, corpus)
        clocks = _memory_clocks(report["cases"])
        if not clocks:
            raise ValueError("Missing memory-clock telemetry")
        if profile and clocks != profile["observed_memory_clocks"]:
            raise ValueError("Memory-clock regime differs from frozen training profile")
        primary = [case for case in report["cases"] if case["shape_mnk"][0] == 1
                   and case["shape_mnk"][1:] in spec["large_primary_shapes_nk"]]
        if mode == "tune":
            screen = all(case["aa_passed"] and case["time_saved_pct"] >= 12
                         and case["methods"][case["selected"]]["numerics"]["passed"]
                         and case["storage_ratio"] <= .8 for case in primary)
            report["minimum_12pct_tuning_screen"] = screen
        else:
            report["accepted_10pct"] = bool(primary) and all(case["passes_10pct_gate"]
                and case["raw_template_bitwise"] and case["original_bits_preserved"]
                and case["storage_ratio"] <= .8 for case in primary)
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      choices=choices, observed_memory_clocks=clocks,
                      elapsed_seconds=time.monotonic() - campaign_start)
        write_json(output_path, report)
        if mode == "tune":
            write_json(profile_path, {"kind": "compressed-projection-tuning", "status": "frozen",
                "seed": seed, "pid": os.getpid(), "run_id": report["run_id"], "bindings": bindings,
                "choices": choices, "minimum_12pct_tuning_screen": screen,
                "training_report": str(output_path.relative_to(ROOT)), "training_report_sha256": _sha(output_path),
                "observed_memory_clocks": clocks, "accepted_10pct": False})
        return report
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_path, report)
        raise
    finally:
        workspace.close()
        vendor.close()
