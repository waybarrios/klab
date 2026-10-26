"""R2 frozen-choice transfer to the Battery profile: calibration, then held-out replay.

No configuration search. Every shape runs exactly the registered Performance R2
choice through the frozen measure_case/compressed_campaign paths; this wrapper
only binds provenance and the normal, restored system76-power profile.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = "experiments/consumer_ampere_frozen_transfer.json"
CORPUS = ".cache/real-weights/manifest.json"
VENDOR = "results/vendor-stock.json"
BACKLIGHT = Path("/sys/class/backlight/nvidia_wmi_ec_backlight/brightness")
AC = Path("/sys/class/power_supply/AC0/online")
PROFILES = ("performance", "battery", "balanced")
SELECTION_RULE = "fixed-choice calibration, no search"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_choices(protocol, source, *, source_sha256, output_dtype):
    """Deep copy of the registered parent choices; timings are never consulted."""
    from compressed_case import launch_config
    matches = [entry for entry in protocol["source_profiles"]
               if entry["sha256"] == source_sha256 and entry["output_dtype"] == output_dtype]
    if len(matches) != 1:
        raise ValueError("Source profile bytes/output contract are not the registered R2 parent")
    registered = matches[0]["choices"]
    if (source.get("kind") != "compressed-projection-tuning" or source.get("status") != "frozen"
            or source.get("bindings", {}).get("output_dtype") != output_dtype
            or source.get("choices") != registered):
        raise ValueError("Source profile kind/status/output/choices differ from the registration")
    for config in registered.values():
        launch_config(config)
    return copy.deepcopy(registered)


def calibration_grid(spec, corpus):
    """Exactly the registered training layers x projections, M1 only."""
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    cases = []
    for layer in spec["tuning_layers"]:
        for suffix in spec["projection_weights"]:
            name = f"model.layers.{layer}.{suffix}"
            if name not in entries:
                raise ValueError(f"Missing registered training tensor {name}")
            n, k = entries[name]["shape"]
            cases.append({"weight_tensor": name, "shape_mnk": [1, n, k], "layer": layer, "role": "training"})
    return cases


def read_settings():
    result = subprocess.run(["system76-power", "profile"], capture_output=True, text=True,
                            timeout=15, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"system76-power profile query failed: {result.stderr.strip()}")
    text = result.stdout.strip()
    line = next((line.strip() for line in text.splitlines() if line.startswith("Power Profile: ")), "")
    profile = line.removeprefix("Power Profile: ").strip().lower()
    if profile not in PROFILES:
        raise RuntimeError(f"Unrecognized power profile: {text!r}")
    ac = AC.read_text().strip() == "1" if AC.exists() else None
    return {"profile": profile, "brightness": int(BACKLIGHT.read_text()), "ac_online": ac,
            "system76_power_profile": text}


def select_profile(name):
    if name not in PROFILES:
        raise ValueError(f"Only normal system76-power profiles: {PROFILES}")
    result = subprocess.run(["system76-power", "profile", name], capture_output=True, text=True,
                            timeout=30, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"system76-power profile {name} failed: {result.stderr.strip()}")


def set_brightness(value):
    result = subprocess.run(["busctl", "call", "org.freedesktop.login1", "/org/freedesktop/login1/session/auto",
                             "org.freedesktop.login1.Session", "SetBrightness", "ssu", "backlight",
                             "nvidia_wmi_ec_backlight", str(int(value))],
                            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"login1 SetBrightness failed: {result.stderr.strip()}")


def _restore(initial, state):
    errors = []
    # Termination signals stay pending until restoration finishes, then are delivered.
    masked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        # Profile first: selecting a profile may itself change the brightness.
        for key, apply in (("profile", select_profile), ("brightness", set_brightness)):
            try:
                if read_settings()[key] != initial[key]:
                    apply(initial[key])
            except Exception as error:
                errors.append(f"{key}: {error}")
        final = read_settings()
        state["after_restore"] = final
        state["restored"] = all(final[key] == initial[key] for key in ("profile", "brightness"))
        if not state["restored"]:
            raise RuntimeError(f"Initial profile/brightness not restored {initial} -> {final}: {errors}")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, masked)


@contextlib.contextmanager
def temporary_profile(name, state=None):
    """Snapshot profile+brightness, select `name`, and always restore both."""
    state = {} if state is None else state
    initial = read_settings()
    state.update(initial=initial, restored=False)
    try:
        select_profile(name)
        current = read_settings()
        if current["profile"] != name:
            raise RuntimeError(f"Requested {name} profile was not applied: {current}")
        if current["brightness"] != initial["brightness"]:
            set_brightness(initial["brightness"])
            current = read_settings()
            if current["brightness"] != initial["brightness"]:
                raise RuntimeError("Could not preserve the user's brightness during profile testing")
        state["during"] = current
        yield state
    finally:
        _restore(initial, state)


def _load_inputs(source_profile_path, output_dtype):
    """CPU verification of registration, parent profile, base spec, corpus and vendor pool."""
    from bench_compressed import validate_probe_inputs
    from experiment_control import project_path
    paths = {"protocol": project_path(ROOT, PROTOCOL), "source": project_path(ROOT, source_profile_path),
             "corpus": project_path(ROOT, CORPUS), "vendor": project_path(ROOT, VENDOR)}
    protocol = json.loads(paths["protocol"].read_text())
    paths["spec"] = project_path(ROOT, protocol["base_spec"]["path"])
    data = {key: json.loads(path.read_text()) for key, path in paths.items() if key != "protocol"}
    spec, corpus, frozen, source = data["spec"], data["corpus"], data["vendor"], data["source"]
    digests = {key: _sha(path) for key, path in paths.items()}
    registered = [entry for entry in protocol["source_profiles"] if entry["sha256"] == digests["source"]]
    if registered and project_path(ROOT, registered[0]["path"]) != paths["source"]:
        raise ValueError("Source profile is not at its registered path")
    choices = fixed_choices(protocol, source, source_sha256=digests["source"], output_dtype=output_dtype)
    bindings = source["bindings"]
    calibration = protocol["calibration"]
    if (digests["spec"] != protocol["base_spec"]["sha256"] or digests["spec"] != bindings["spec_sha256"]
            or digests["corpus"] != bindings["corpus_sha256"] or digests["vendor"] != bindings["vendor_profile_sha256"]
            or _sha(ROOT / "experiments/consumer_ampere_execution_context.json") != bindings["context_protocol_sha256"]
            or bindings["revision"] != 2 or protocol["kernel_revision"] != 2):
        raise ValueError("Base spec/corpus/vendor/context/revision differ from the R2 parent bindings")
    if (protocol["custom_configuration_search_allowed"] or protocol["code_or_codec_revision_allowed"]
            or calibration["seed"] != spec["tuning_seed"] or calibration["layers"] != spec["tuning_layers"]
            or calibration["m"] != 1 or len(spec["projection_weights"]) != 7
            or calibration["custom_configurations_per_shape"] != 1
            or protocol["validation"]["layers"] != spec["held_out_layers"]
            or protocol["validation"]["seeds"] != spec["held_out_seeds"]
            or protocol["validation"]["m_values"] != spec["m_values"] + spec["baseline_diagnostic_m"]):
        raise ValueError("Transfer protocol and base spec disagree")
    validate_probe_inputs(spec, corpus, frozen, seed=spec["tuning_seed"])
    if (frozen.get("spec_sha256") != digests["spec"]
            or frozen["source_sha256"].get("cublaslt_baseline.py") != _sha(ROOT / "cublaslt_baseline.py")):
        raise ValueError("Frozen vendor experiment contract or precision/ABI dependency differs")
    calibration_grid(spec, corpus)
    provenance = {"frozen_choice_origin_sha256": digests["source"],
                  "frozen_choice_origin_path": str(paths["source"].relative_to(ROOT)),
                  "inherited_choices": copy.deepcopy(choices),
                  "protocol_path": PROTOCOL, "protocol_sha256": digests["protocol"],
                  "wrapper_source_sha256": _sha(Path(__file__)), "selection_rule": SELECTION_RULE,
                  "parent_observed_memory_clocks": source.get("observed_memory_clocks"),
                  "parent_operator_context": bindings["operator_context"]}
    watched = {str(path.relative_to(ROOT)): digests[key] for key, path in paths.items()}
    watched["frozen_transfer.py"] = provenance["wrapper_source_sha256"]
    return {"paths": paths, "protocol": protocol, "spec": spec, "corpus": corpus, "vendor": frozen,
            "source": source, "choices": choices, "provenance": provenance, "watched": watched}


def check_transfer_profile(profile, inputs):
    """A replayable transferred profile matches this wrapper, protocol, parent and passed its screen."""
    expected = {key: inputs["provenance"][key] for key in ("frozen_choice_origin_sha256",
                "frozen_choice_origin_path", "protocol_sha256", "wrapper_source_sha256", "selection_rule")}
    transfer = profile.get("transfer", {})
    if any(transfer.get(key) != value for key, value in expected.items()):
        raise ValueError("Transferred profile provenance differs from the current origin/protocol/wrapper")
    if (profile.get("kind") != "compressed-projection-tuning" or profile.get("status") != "frozen"
            or profile.get("choices") != inputs["choices"] or profile.get("minimum_12pct_tuning_screen") is not True
            or profile["bindings"]["identity"] != inputs["source"]["bindings"]["identity"]):
        raise ValueError("Transferred profile is not a passed, inherited, same-identity calibration")


def check_sources(identity, watched):
    """Re-hash files named by the frozen identity (nested compiler/bridge, PTX, libraries) and inputs."""
    import nvidia.cublas
    from experiment_control import project_path

    def same(path, digest):
        return path.is_file() and _sha(path) == digest

    changed = [name for name, digest in watched.items() if not same(project_path(ROOT, name), digest)]

    compilers = set()
    def visit(node):
        if not isinstance(node, dict):
            return
        if isinstance(node.get("compiler"), str) and node["compiler"].startswith("NVRTC "):
            compilers.add(node["compiler"])
        for name, digest in (node.get("source_sha256") or {}).items():
            if not same(project_path(ROOT, name), digest):
                changed.append(name)
        if isinstance(node.get("implementation"), str) and node.get("implementation_sha256"):
            if not same(project_path(ROOT, node["implementation"]), node["implementation_sha256"]):
                changed.append(node["implementation"])
        if node.get("ptx_sha256") and isinstance(node.get("module_fingerprint"), dict):
            ptx = ROOT / ".cache/ptx" / f"{node['module_fingerprint']['fingerprint_sha256']}.ptx"
            if not same(ptx, node["ptx_sha256"]):
                changed.append(str(ptx.relative_to(ROOT)))
        for value in node.values():
            visit(value)

    visit(identity)
    if compilers:
        from cuda_module import _nvrtc_identity
        current_compiler, _ = _nvrtc_identity()
        if compilers != {current_compiler}:
            changed.append("NVRTC compiler binary/version")
    base = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    changed += [name for name, digest in identity["libraries_sha256"].items() if not same(base / name, digest)]
    if changed:
        raise RuntimeError(f"Sources/inputs changed during the study: {sorted(set(changed))}")


def calibrate(source_profile_path, profile_path, output_path, *, output_dtype, rounds=31, seed=901,
              run_metadata=None):
    """Caller owns GpuLock and the Battery profile context. One inherited config per case."""
    import torch
    from bench_compressed import _power_context
    from compressed_campaign import _identity, _memory_clocks, assert_complete, freeze_choices, summarize_methods
    from compressed_case import measure_case
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from experiment_control import project_path, with_run_metadata, write_json
    from gpu_unlock_status import inspect
    from power_context import execution_context

    started = time.monotonic()
    inputs = _load_inputs(source_profile_path, output_dtype)
    protocol, spec, corpus, source = inputs["protocol"], inputs["spec"], inputs["corpus"], inputs["source"]
    profile_path, output_path = project_path(ROOT, profile_path), project_path(ROOT, output_path)
    if profile_path == output_path or {profile_path, output_path} & set(inputs["paths"].values()):
        raise ValueError("Calibration report/profile must be new, distinct files")
    if profile_path.exists() or output_path.exists():
        raise ValueError("Preserve prior measurements; use new output/profile paths")
    if seed != protocol["calibration"]["seed"] or rounds != protocol["calibration"]["rounds"]:
        raise ValueError("Calibration uses the registered seed and round count only")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    power = _power_context()
    compute_context = execution_context(power)
    parent_context = source["bindings"]["operator_context"]
    if (compute_context["system76_power_profile"] != protocol["destination_profile"]
            or compute_context["scope"] != "graph" or compute_context["ac_online"] is not True
            or compute_context["ac_online"] != parent_context["ac_online"]):
        raise ValueError(f"Need Battery graph context with the parent's AC condition: {compute_context}")
    inventory = inspect()
    vendor = CuBlas()
    workspace = LtWorkspace(0)
    try:
        identity = _identity(vendor, inventory, 2)
        paths = inputs["paths"]
        bindings = {"spec_sha256": _sha(paths["spec"]), "corpus_sha256": _sha(paths["corpus"]),
                    "vendor_profile_sha256": _sha(paths["vendor"]), "output_dtype": output_dtype,
                    "context_protocol_sha256": _sha(ROOT / "experiments/consumer_ampere_execution_context.json"),
                    "revision": 2, "operator_context": compute_context, "identity": identity}
        def base(values):
            return {key: value for key, value in values.items() if key != "operator_context"}
        if base(bindings) != base(source["bindings"]):
            raise ValueError("Runtime/source identity differs from the R2 parent; only the profile may change")
        metadata = {"pid": os.getpid(), "run_id": str(uuid.uuid4()),
                    "created_utc": datetime.now(timezone.utc).isoformat(), **(run_metadata or {})}
        if metadata["pid"] != os.getpid():
            raise ValueError("Report PID must be the actual experiment process")
        report = with_run_metadata({"kind": "compressed-projection", "status": "running", "mode": "tune",
            "seed": seed, **bindings, "inventory": inventory, "operator_display_snapshot": power,
            "profile_sha256": None, "vendor_tuning_oracle_sha256": inputs["vendor"]["source_sha256"].get("projection_accuracy.py"),
            "selection_rule": SELECTION_RULE, "choices_origin": "inherited", "transfer": inputs["provenance"],
            "cases": [], "accepted_10pct": False,
            "scope": "Streaming CUDA graph GPU latency, real native BF16 weights and synthetic seeded activations",
            "full_model_speedup_claim": False, "unlocked_comparison": "pending"}, metadata)
        write_json(output_path, report)
    except BaseException:
        workspace.close()
        vendor.close()
        raise
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    vendor_cases = {(tuple(case["shape_mnk"]), case["output_dtype"]): case for case in inputs["vendor"]["cases"]
                    if case["input_dtype"] == "bf16"}
    choices = inputs["choices"]
    try:
        for planned in calibration_grid(spec, corpus):
            m, n, k = planned["shape_mnk"]
            vendor_case = vendor_cases.get(((m, n, k), output_dtype))
            if vendor_case is None:
                raise ValueError(f"Missing frozen vendor operation M,N,K={m,n,k}, output={output_dtype}")
            config = choices[f"{n},{k}"]
            case = measure_case(entries[planned["weight_tensor"]], corpus, m, output_dtype, seed, [config],
                                vendor_case, workspace, vendor, rounds=rounds, kernel_revision=2,
                                decode_diagnostic=planned["layer"] == 0 and [n, k] in spec["large_primary_shapes_nk"])
            case.update(layer=planned["layer"], role="training", selected=config)
            case.update(summarize_methods(case["methods"], case["vendor_routes"], config))
            report["cases"].append(case)
            write_json(output_path, report)
            gain = f"{case['time_saved_pct']:+.3f}%" if case.get("time_saved_pct") is not None else "no comparison"
            print(f"calibrate {planned['weight_tensor']} M=1 {output_dtype} {config}: {gain}", flush=True)
            if not case["operating_context_stable"] or execution_context(_power_context()) != compute_context:
                raise RuntimeError("Power profile/AC changed during the GPU-graph calibration; measured case retained")
        assert_complete(report, spec, corpus)
        clocks = _memory_clocks(report["cases"])
        if not clocks:
            raise ValueError("Missing memory-clock telemetry")
        try:  # Replay compatibility for the old evidence builders, not a selection.
            reproduces = freeze_choices(report["cases"], spec, output_dtype=output_dtype) == choices
        except ValueError:
            reproduces = False
        primary = [case for case in report["cases"] if case["shape_mnk"][1:] in spec["large_primary_shapes_nk"]]
        gates = {case["weight_tensor"]: {
            "point_gain_pct": case["time_saved_pct"], "aa_passed": case["aa_passed"],
            "minimum_12pct": case["time_saved_pct"] is not None
                and case["time_saved_pct"] >= protocol["calibration"]["minimum_primary_point_gain_pct"],
            "numerics": case["methods"][case["selected"]]["numerics"]["passed"],
            "storage_ratio": case["storage_ratio"],
            "exact_bits": case["raw_template_bitwise"] and case["original_bits_preserved"]} for case in primary}
        screen = bool(primary) and reproduces and all(
            gate["aa_passed"] and gate["minimum_12pct"] and gate["numerics"] and gate["exact_bits"]
            and gate["storage_ratio"] <= .8 for gate in gates.values())
        check_sources(identity, inputs["watched"])
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), choices=choices,
                      observed_memory_clocks=clocks, elapsed_seconds=time.monotonic() - started,
                      minimum_12pct_tuning_screen=screen, screen_cases=gates, choices_reproduce_from_rounds=reproduces)
        write_json(output_path, report)
        write_json(profile_path, {"kind": "compressed-projection-tuning", "status": "frozen",
            "seed": seed, "pid": os.getpid(), "run_id": report["run_id"], "bindings": bindings,
            "choices": choices, "minimum_12pct_tuning_screen": screen,
            "training_report": str(output_path.relative_to(ROOT)), "training_report_sha256": _sha(output_path),
            "observed_memory_clocks": clocks, "accepted_10pct": False,
            "transfer": {**inputs["provenance"], "calibration_report_sha256": _sha(output_path)}})
        return report
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_path, report)
        raise
    finally:
        workspace.close()
        vendor.close()


def validate(source_profile_path, profile_path, output_path, *, output_dtype, seed, rounds=31, run_metadata=None):
    """Caller owns GpuLock and Battery context. Unchanged compressed_campaign.run replay."""
    from compressed_campaign import run
    from experiment_control import project_path, write_json
    inputs = _load_inputs(source_profile_path, output_dtype)
    profile_path, output_path = project_path(ROOT, profile_path), project_path(ROOT, output_path)
    if output_path.exists():
        raise ValueError("Preserve prior measurements; use a new output path")
    profile = json.loads(profile_path.read_text())
    check_transfer_profile(profile, inputs)
    if seed not in inputs["protocol"]["validation"]["seeds"] or rounds != inputs["protocol"]["validation"]["rounds"]:
        raise ValueError("Validation uses the registered held-out seeds and round count only")
    training = project_path(ROOT, profile["training_report"])
    watched = {**inputs["watched"], str(profile_path.relative_to(ROOT)): _sha(profile_path),
               str(training.relative_to(ROOT)): profile["training_report_sha256"]}
    paths = inputs["paths"]
    try:
        run(paths["spec"], paths["corpus"], paths["vendor"], mode="validate", seed=seed, profile_path=profile_path,
            output_path=output_path, output_dtype=output_dtype, revision=2, rounds=rounds, run_metadata=run_metadata)
    finally:
        if output_path.exists():
            report = json.loads(output_path.read_text())
            report["transfer"] = {**inputs["provenance"], "transferred_profile_sha256": watched[
                str(profile_path.relative_to(ROOT))]}
            try:
                check_sources(profile["bindings"]["identity"], watched)
            except RuntimeError as error:
                report.update(status="failed", error=str(error), transfer_source_check=False)
                write_json(output_path, report)
                raise
            report["transfer_source_check"] = True
            write_json(output_path, report)


def _interrupt(signum, _frame):
    for number in (signal.SIGTERM, signal.SIGINT):  # Let restoration finish uninterrupted.
        signal.signal(number, signal.SIG_IGN)
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None):
    for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch"),
                           ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor"), ("TRITON_CACHE_DIR", ".cache/triton")):
        os.environ.setdefault(key, str(ROOT / directory))
    from experiment_control import GpuLock, project_path, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "validate"), required=True)
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--source-profile", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    profile, output = project_path(ROOT, args.profile), project_path(ROOT, args.output)
    state_path = output.with_name(output.stem + "-state.json")
    inputs = _load_inputs(args.source_profile, args.output_dtype)  # All input bytes before any settings change.
    if (output.suffix != ".json" or profile.suffix != ".json"
            or len({profile, output, state_path, *inputs["paths"].values()}) != 3 + len(inputs["paths"])):
        raise ValueError("Profile, report and state must be distinct new .json files apart from inputs")
    if output.exists() or state_path.exists() or (args.mode == "calibrate" and profile.exists()):
        raise ValueError("Refusing to overwrite prior results")
    if args.mode == "calibrate" and args.seed != inputs["protocol"]["calibration"]["seed"]:
        raise ValueError("Calibration seed is 901")
    if args.mode == "validate":
        if args.seed not in inputs["protocol"]["validation"]["seeds"]:
            raise ValueError("Validation seeds are 919/929/937")
        check_transfer_profile(json.loads(profile.read_text()), inputs)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    metadata = {"command": [sys.executable, *sys.argv]}
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "seed": args.seed,
             "pid": os.getpid(), "output": str(output.relative_to(ROOT)), "status": "running",
             "scope": "normal system76-power profile and login1 brightness only; no clocks/voltage/limits",
             "energy": "not measured", "unlocked_comparison": "pending"}
    try:
        with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
            with temporary_profile("battery", state):
                if args.mode == "calibrate":
                    calibrate(args.source_profile, profile, output, output_dtype=args.output_dtype,
                              rounds=args.rounds, seed=args.seed, run_metadata=metadata)
                else:
                    validate(args.source_profile, profile, output, output_dtype=args.output_dtype,
                             seed=args.seed, rounds=args.rounds, run_metadata=metadata)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if "initial" in state and not state.get("restored"):
            state["status"] = "restore-failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)


if __name__ == "__main__":
    main()
