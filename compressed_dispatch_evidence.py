"""Per-shape kernel evidence from three raw held-out compressed-projection processes.

CPU only: no torch import or GPU initialization. Every statistic is recomputed from the
raw paired rounds through the frozen campaign helpers; saved gates, intervals, savings and
report-level acceptance are ignored. Structural defects raise ValueError, while valid
negative outcomes stay in the evidence with readable reasons. Kernel evidence never
validates a custom prepared-dispatch win by itself. File loading and SHA binding of the
inputs belong to a separate layer.
"""
import copy
import math

import numpy as np

from compressed_campaign import _memory_clocks, assert_complete, summarize_methods, validate_process_set
from compressed_case import launch_config

BINDING_FIELDS = ("output_dtype", "spec_sha256", "corpus_sha256", "vendor_profile_sha256",
                  "context_protocol_sha256", "revision", "operator_context", "identity")
HELD_OUT = "held-out"
HEAD = "head-transfer-diagnostic"
# summarize_methods hardcodes this predicate; the preregistered spec must still agree.
FROZEN_MAX_AA_PCT = 1
FROZEN_CI95_LOWER_PCT = 10


def build_evidence(reports, spec, corpus, profile, *, profile_sha256):
    try:
        return _build(reports, spec, corpus, profile, profile_sha256)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed compressed validation evidence: {error!r}") from error


def _build(reports, spec, corpus, profile, profile_sha256):
    acceptance = spec["acceptance"]
    if (acceptance["maximum_absolute_aa_pct"] != FROZEN_MAX_AA_PCT
            or acceptance["latency_reduction_ci95_lower_bound_strictly_above_pct"] != FROZEN_CI95_LOWER_PCT):
        raise ValueError("Preregistered acceptance differs from the frozen summarize_methods gate")
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    _check_profile(profile, spec)
    _check_processes(reports, spec, corpus, profile, profile_sha256)
    head_config = _head_config(spec, entries, profile["choices"])
    cases = []
    for report in reports:
        for case in report["cases"]:
            _check_case(case, spec, entries, profile["choices"], head_config)
            cases.append(_summarize_case(case, report, spec))

    shapes = {}
    for summary in cases:
        key = ",".join(map(str, summary["shape_mnk"]))
        shape = shapes.setdefault(key, {
            "shape_mnk": list(summary["shape_mnk"]), "selected": summary["selected"], "roles": [],
            "cases": 0, "failed_cases": 0, "kernel_gate_passed": True, "execution_mode": "graph",
            "eager_gate": "pending", "custom_win_validated": False, "dispatch_scope_gate": "pending"})
        if summary["role"] not in shape["roles"]:
            shape["roles"].append(summary["role"])
        shape["cases"] += 1
        if not summary["kernel_gate_passed"]:
            shape["failed_cases"] += 1
            shape["kernel_gate_passed"] = False
    primary_m = acceptance.get("primary_m", 1)
    primary = [f"{primary_m},{n},{k}" for n, k in spec["large_primary_shapes_nk"]]
    bindings = profile["bindings"]
    return {
        "kind": "compressed-dispatch-evidence", "status": "complete",
        "contract": {"input_dtype": "bf16", "weight_dtype": "bf16", "accumulation_dtype": "fp32",
                     "output_dtype": bindings["output_dtype"], "lossy_weights": False,
                     "allow_tf32": False, "reduced_precision_reduction": False},
        "scope": "Streaming CUDA graph GPU kernel latency; prepared dispatch and eager calls need separate evidence",
        "execution_mode": "graph", "eager_promoted": False, "full_model_speedup_claim": False,
        "profile_sha256": profile_sha256, "bindings": copy.deepcopy(bindings),
        "choices": copy.deepcopy(profile["choices"]),
        "observed_memory_clocks": list(profile["observed_memory_clocks"]),
        "tuning_process": {"seed": profile["seed"], "pid": profile["pid"], "run_id": profile["run_id"]},
        "processes": [{"seed": r["seed"], "pid": r["pid"], "run_id": r["run_id"]} for r in reports],
        "cases": cases, "shapes": shapes, "primary_shapes": primary,
        "kernel_milestone_passed": bool(primary) and all(key in shapes and shapes[key]["kernel_gate_passed"]
                                                         for key in primary),
    }


def _check_profile(profile, spec):
    if profile.get("kind") != "compressed-projection-tuning" or profile.get("status") not in ("frozen", "complete"):
        raise ValueError("Need a completed frozen compressed tuning profile")
    bindings = profile["bindings"]
    if not isinstance(bindings, dict) or any(field not in bindings for field in BINDING_FIELDS):
        raise ValueError("Frozen profile lacks its input, runtime, output or context bindings")
    if bindings["output_dtype"] not in ("bf16", "fp32"):
        raise ValueError("Frozen profile has no valid output contract")
    if bindings["operator_context"].get("scope") != "graph":
        raise ValueError("Raw kernel evidence requires the registered graph execution scope")
    if any(bindings["identity"].get(flag) is not False
           for flag in ("allow_tf32", "reduced_precision_reduction")):
        raise ValueError("Evidence must retain the frozen FP32-accumulation vendor contract")
    if profile.get("seed") != spec["tuning_seed"] or not profile.get("pid") or not profile.get("run_id"):
        raise ValueError("Frozen profile needs its tuning seed and training process identity")
    if profile.get("minimum_12pct_tuning_screen") is not True:
        raise ValueError("Training did not pass the preregistered large-shape screen")
    if not isinstance(profile["choices"], dict) or not profile["choices"]:
        raise ValueError("Frozen profile has no configuration choices")
    for config in profile["choices"].values():
        launch_config(config)
    if not isinstance(profile["observed_memory_clocks"], list) or not profile["observed_memory_clocks"]:
        raise ValueError("Frozen profile lacks its memory-clock regime")


def _check_processes(reports, spec, corpus, profile, profile_sha256):
    if not isinstance(reports, (list, tuple)) or any(r.get("kind") != "compressed-projection" for r in reports):
        raise ValueError("Expected raw compressed-projection validation reports")
    validate_process_set(reports, spec)  # Structure only; its primary-M1 accepted_10pct aggregate is ignored.
    for report in reports:
        for field in BINDING_FIELDS:
            if report.get(field) != profile["bindings"][field]:
                raise ValueError(f"Validation seed {report['seed']} differs from the frozen profile: {field}")
        if report["profile_sha256"] != profile_sha256:
            raise ValueError("Validation process replayed a different frozen profile")
        if report["pid"] == profile["pid"] or report["run_id"] == profile["run_id"]:
            raise ValueError("Validation reused the tuning process identity")
        if "choices" in report and report["choices"] != profile["choices"]:
            raise ValueError("Validation process recorded a different frozen selection")
        assert_complete(report, spec, corpus)
        clocks = report.get("observed_memory_clocks")
        telemetry = _memory_clocks(report["cases"])
        if clocks != profile["observed_memory_clocks"] or telemetry != clocks:
            raise ValueError("Memory-clock regime differs from the frozen training profile")
        for case in report["cases"]:
            timed_clocks = _memory_clocks([case])
            if not timed_clocks or not set(timed_clocks).issubset(clocks):
                raise ValueError("Every case needs parseable clocks within its timed rounds")


def _head_config(spec, entries, choices):
    """compressed_campaign replays the frozen gate/up winner on the head; there is no head tuning."""
    if spec["head"]["tensor"] not in entries:
        return None
    keys = {",".join(map(str, entry["shape"])) for name, entry in entries.items()
            if name.endswith(".mlp.gate_proj.weight")}
    if len(keys) != 1 or next(iter(keys)) not in choices:
        raise ValueError("Head transfer needs exactly one frozen gate/up winner")
    return choices[next(iter(keys))]


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_case(case, spec, entries, choices, head_config):
    name = case["weight_tensor"]
    entry = entries.get(name)
    m, n, k = case["shape_mnk"]
    if entry is None or [n, k] != list(entry["shape"]) or case["weight_sha256"] != entry["sha256"]:
        raise ValueError(f"{name} weight name/shape/SHA256 differs from the corpus")
    if name == spec["head"]["tensor"]:
        if case["role"] != HEAD or case["layer"] is not None:
            raise ValueError("The head is only an untuned transfer diagnostic")
        expected = head_config
    else:
        if (case["role"] != HELD_OUT or case["layer"] != int(name.split(".")[2])
                or case["layer"] not in spec["held_out_layers"]):
            raise ValueError(f"{name} is not a registered held-out layer case")
        expected = choices.get(f"{n},{k}")
    if expected is None or case["selected"] != expected:
        raise ValueError(f"{name} selection differs from the frozen profile choice")
    if case["input_dtype"] != "bf16":
        raise ValueError(f"{name} inputs are not native BF16")
    copies = case["copies"]
    if not all(_positive_int(value) for value in (m, n, k, copies)) or case["calls_per_graph"] != copies:
        raise ValueError(f"{name} has an invalid shape or rotating-ring copy count")
    measurement = spec["measurement"]
    if (case["raw_ring_bytes"] != copies * n * k * 2
            or case["raw_ring_bytes"] < measurement["minimum_uncompressed_ring_bytes"]
            or not case["raw_ring_over_l2"] >= measurement["minimum_ring_l2_multiple"]):
        raise ValueError(f"{name} raw ring is not the registered streaming regime")
    storage = case["storage_bytes"]
    if not _positive_int(storage) or not math.isclose(storage / (n * k * 2), case["storage_ratio"], rel_tol=1e-9):
        raise ValueError(f"{name} storage ratio does not match its storage bytes")

    methods, routes = case["methods"], case["vendor_routes"]
    raw = expected.replace("coded_", "raw_")
    if (not routes or len(set(routes)) != len(routes)
            or any(route.endswith("_aa") or route + "_aa" not in methods for route in routes)
            or expected not in methods or raw not in methods):
        raise ValueError(f"{name} lacks a vendor AA twin or the selected coded/raw pair")
    if (not isinstance(case["raw_template_parity_by_config"].get(expected), bool)
            or not isinstance(case["operating_context_stable"], bool)):
        raise ValueError(f"{name} lacks selected raw parity or operating-context evidence")
    lengths = set()
    for label, method in methods.items():
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in method["samples_us"]):
            raise ValueError(f"{name} {label} timings are not numeric measured rounds")
        samples = np.asarray(method["samples_us"], dtype=np.float64)
        lengths.add(samples.shape)
        if not math.isclose(method["median_us"], float(np.median(samples)), rel_tol=1e-9):
            raise ValueError(f"{name} {label} median differs from its raw rounds")
        numerics = method["numerics"]
        oracle = m * n if numerics["all_copies_bitwise_equal"] is True else copies * m * n
        if (not isinstance(numerics["passed"], bool) or numerics["elements_checked"] != copies * m * n
                or numerics["oracle_elements_checked"] != oracle):
            raise ValueError(f"{name} {label} does not check every output element")
    if len(lengths) != 1:
        raise ValueError(f"{name} methods were not timed in paired rounds")


def _summarize_case(case, report, spec):
    methods, selected = case["methods"], case["selected"]
    raw = selected.replace("coded_", "raw_")
    stats = summarize_methods(methods, case["vendor_routes"], selected)
    reasons = []
    if stats["comparison_status"] != "compared":
        reasons.append("No vendor route passed numerics together with its AA twin")
    else:
        if not stats["aa_passed"]:
            reasons.append(f"Winner {stats['best_vendor']} AA control {stats['aa_time_saved_pct']:+.3f}% "
                           f"exceeds ±{FROZEN_MAX_AA_PCT}%")
        if not stats["ci95_pct"][0] > FROZEN_CI95_LOWER_PCT:
            reasons.append(f"CI95 lower bound {stats['ci95_pct'][0]:+.3f}% is not strictly above "
                           f"{FROZEN_CI95_LOWER_PCT}%")
    if not methods[selected]["numerics"]["passed"]:
        reasons.append(f"Selected {selected} failed numerics")
    if not methods[raw]["numerics"]["passed"]:
        reasons.append(f"Paired raw {raw} failed numerics")
    if not case["original_bits_preserved"]:
        reasons.append("Original checkpoint bits were not preserved")
    if not case["raw_template_bitwise"]:
        reasons.append("Raw and compressed template outputs differ")
    if not case["raw_template_parity_by_config"][selected]:
        reasons.append(f"Selected {selected} output differs from its raw twin")
    limit = spec["storage_limits"]["promotion_max_ratio"]
    if case["storage_ratio"] > limit:
        reasons.append(f"Storage ratio {case['storage_ratio']:.4f} exceeds {limit}")
    if not case["operating_context_stable"] or case["execution_context"] != report["operator_context"]:
        reasons.append("Operator power/AC context was not stable")
    summary = {
        "seed": report["seed"], "pid": report["pid"], "run_id": report["run_id"],
        "weight_tensor": case["weight_tensor"], "weight_sha256": case["weight_sha256"],
        "layer": case["layer"], "role": case["role"], "shape_mnk": list(case["shape_mnk"]),
        "selected": selected, "raw_method": raw, "copies": case["copies"],
        "raw_ring_bytes": case["raw_ring_bytes"], "raw_ring_over_l2": case["raw_ring_over_l2"],
        "storage_bytes": case["storage_bytes"], "storage_ratio": case["storage_ratio"],
        "selected_median_us": methods[selected]["median_us"],
        "selected_numerics_passed": methods[selected]["numerics"]["passed"],
        "raw_numerics_passed": methods[raw]["numerics"]["passed"],
        "original_bits_preserved": case["original_bits_preserved"],
        "raw_template_bitwise": case["raw_template_bitwise"],
        "selected_raw_parity": case["raw_template_parity_by_config"][selected],
        "operating_context_stable": case["operating_context_stable"],
    }
    # Recomputed stats only; the helper's combined flag is superseded by the full gate below.
    summary.update({key: value for key, value in stats.items() if key != "passes_10pct_gate"})
    summary.update(kernel_gate_passed=not reasons, failure_reasons=reasons)
    return summary
