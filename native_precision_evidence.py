"""CPU-only audit of the frozen native-precision experiments.

Recompute every comparison from raw rounds. All registered cases, including
negative results, remain visible. A complete cohort is not a blanket success.
This module does not import Torch, launch work or modify measured sources.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from compressed_campaign import summarize_methods

ROOT = Path(__file__).resolve().parent
FLAGS = ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
UNLOCK = ("module_loaded", "unlock_boot_option_present", "unlock_success_logged", "post_flr_selectors_logged")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{label}: fields differ")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), f"{label}: lengths differ")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{label}[{index}]")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), f"{label}: value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: value differs")


def _identity(report):
    result = {key: report.get(key) for key in ("source_sha256", "source_profiles_sha256", "libraries_sha256",
                                              "ptx_sha256", "nvrtc_identity")}
    for key in ("source_sha256", "source_profiles_sha256", "libraries_sha256"):
        require(isinstance(result[key], dict) and result[key]
                and all(_digest(v) for v in result[key].values()), f"Missing {key} identities")
    require(_digest(result["ptx_sha256"]), "Missing PTX identity")
    require(isinstance(result["nvrtc_identity"], str) and result["nvrtc_identity"], "Missing NVRTC identity")
    compilation = report.get("compilation", {})
    result["compilation"] = {key: compilation.get(key) for key in ("cache_key", "architecture", "compiler")}
    require(all(result["compilation"].values()), "Missing compiler/cache identity")
    environment = report.get("environment", {})
    result["environment"] = {key: environment.get(key) for key in ("torch", "cuda", "cublas_version")}
    require(all(result["environment"].values()), "Missing runtime identity")
    inventory = environment.get("inventory", {})
    result["inventory"] = {key: inventory.get(key) for key in ("gpu", "kernel") + UNLOCK}
    require(result["inventory"]["gpu"] and result["inventory"]["kernel"]
            and all(type(result["inventory"][key]) is bool for key in UNLOCK), "Missing hardware/unlock identity")
    return result


def _numerics(record, copies, m, n, *, before_passed=None):
    require(isinstance(record, dict), "Missing numerical record")
    equal = record.get("all_copies_bitwise_equal")
    require(type(equal) is bool and record.get("elements_checked") == copies * m * n
            and record.get("oracle_elements_checked") == (m * n if equal else copies * m * n),
            "Missing complete numerical coverage")
    for key in ("nonfinite", "violations"):
        require(_integer(record.get(key)) and record[key] <= record["oracle_elements_checked"], "Invalid numerical counts")
    if record["nonfinite"]:
        require(record["violations"] == record["nonfinite"], "Nonfinite failure counts differ")
    if not record["nonfinite"] or "atol" in record or "rtol" in record:
        require(record.get("atol") == 1e-5 and record.get("rtol") == 1e-4, "Numerical tolerances changed")
    passed = record["nonfinite"] == 0 and record["violations"] == 0
    if before_passed is not None:
        passed = passed and before_passed
    require(type(record.get("passed")) is bool and record["passed"] == passed, "Numerical pass flag disagrees with counts")
    return passed


def _pool(case, dtype):
    from cublaslt_baseline import _contract
    routes, pool, methods = case.get("vendor_routes"), case.get("vendor_pool"), case.get("methods")
    require(isinstance(routes, list) and routes and len(set(routes)) == len(routes)
            and all(isinstance(r, str) and not r.endswith("_aa") for r in routes), "Invalid vendor route list")
    require(isinstance(pool, dict) and set(pool) == set(routes), "Vendor metadata is incomplete")
    require(isinstance(methods, dict) and set(methods) == {"custom", *routes, *(r + "_aa" for r in routes)},
            "Missing vendor own A/A, custom method or unexpected method")
    mandatory = {"cublas_compute32", "cublas_packed_compute32"}
    if dtype == "fp32":
        mandatory.add("torch_mm")
        if case["shape_mnk"][0] == 1:
            mandatory.update(("torch_mv", "cublas_mv", "cublas_packed_mv"))
    else:
        # The producer records a Torch API failure rather than silently omitting it.
        if "torch_mm_out_fp32" not in routes:
            require(any(d.get("method") == "torch_mm_out_fp32" and d.get("unavailable")
                        for d in case.get("search_diagnostics", [])), "Torch route omission has no diagnostic")
    require(mandatory <= set(routes), "Strong vendor pool lost a mandatory route")
    for name, record in pool.items():
        if name.startswith("lt_"):
            _contract(record)
            require(record["input_dtype"] == dtype and record["output_dtype"] == "fp32"
                    and record["shape_mnk"] == case["shape_mnk"], "Lt shape/dtype contract changed")
            require(name == f"lt_{record['layout']}_h{record['candidate_idx']}", "Lt route identity differs")
        elif name in ("cublas_compute32", "cublas_packed_compute32"):
            require(record.get("kind") == "cublasGemmEx" and record.get("compute_type") == 68
                    and record.get("layout") == ("original" if name == "cublas_compute32" else "transposed"),
                    "Strict cuBLAS GEMM contract changed")
        elif name in ("cublas_mv", "cublas_packed_mv"):
            require(record.get("kind") == "cublasSgemv", "cuBLAS GEMV contract changed")
        else:
            require(name in ("torch_mm", "torch_mv", "torch_mm_out_fp32")
                    and record.get("kind") == ("torch.mv" if name == "torch_mv" else "torch.mm"), "Unexpected vendor route")
            if name == "torch_mm_out_fp32":
                require(record.get("out_dtype") == "fp32", "Torch output contract changed")
    diagnostics = case.get("search_diagnostics", [])
    for layout in ("original", "transposed"):
        records = [d for d in diagnostics if d.get("layout") == layout]
        require(len(records) == 1, "Missing unique Lt layout search diagnostic")
        diag = records[0]
        selected = diag.get("selected")
        require(isinstance(selected, list) and len(set(selected)) == len(selected)
                and all(_integer(i) and i < 8 for i in selected)
                and diag.get("requested") == 8 and diag.get("max_candidates") == 8
                and diag.get("eligible") == len(selected), "Invalid Lt discovery diagnostic")
        expected = {f"lt_{layout}_h{i}" for i in selected}
        require(expected == {r for r in routes if r.startswith(f"lt_{layout}_")}, "Discovered Lt methods were omitted")


def audit_process(report, protocol, *, dtype, profile):
    """Validate one raw process without implying a three-process acceptance."""
    require(dtype in ("fp16", "bf16", "fp32") and profile in ("performance", "battery"), "Unsupported cohort")
    require(report.get("kind") == "native-precision-projection" and report.get("status") == "complete"
            and report.get("dtype") == dtype, "Incomplete or mismatched native-precision report")
    require(report.get("seed") in protocol["held_out_seeds"] and _integer(report.get("pid"), 1)
            and isinstance(report.get("run_id"), str) and report["run_id"], "Missing fresh process identity")
    require(report.get("rounds") == protocol["minimum_rounds"] >= 31, "Round count changed")
    require(all(report.get("precision_flags", {}).get(key) is False for key in FLAGS), "Reduced precision math enabled")
    identity = _identity(report)
    context = report.get("operator_context", {})
    require(context.get("system76_power_profile") == "Power Profile: " + profile.title()
            and context.get("ac_online") is True and context.get("scope") == "graph", "Wrong power or measurement context")
    specs = protocol["cases_by_dtype"][dtype]
    cases = report.get("cases", [])
    require(len(cases) == len(specs) and [c.get("shape_mnk") for c in cases] == [s["shape_mnk"] for s in specs],
            "Incomplete, duplicated or reordered shape grid")
    rows = []
    for case, spec in zip(cases, specs):
        require(all(case.get(key) == value for key, value in spec.items()), "Frozen shape configuration changed")
        m, n, k = spec["shape_mnk"]
        require(case.get("seed") == report["seed"] and case.get("input_dtype") == dtype
                and case.get("weight_dtype") == dtype and case.get("output_dtype") == "fp32"
                and case.get("accumulation_dtype") == "fp32", "Case dtype/seed contract differs")
        copies = case.get("copies")
        require(_integer(copies, 2) and case.get("calls_per_graph") == copies, "Missing physical streaming ring")
        raw_bytes = copies * n * k * (4 if dtype == "fp32" else 2)
        require(case.get("raw_ring_bytes") == raw_bytes and raw_bytes >= 80 << 20, "Streaming ring is too small or miscounted")
        require(all(_digest(case.get(key)) for key in ("weight_sha256", "activation_sha256")), "Missing original input identities")
        bit_keys = ("original_bits_preserved", "bitwise_round_trip", "bitwise_layout_check")
        require(all(type(case.get(key)) is bool for key in bit_keys), "Missing original-bit checks")
        _pool(case, dtype)
        for name, route in case["vendor_pool"].items():
            if name.startswith("lt_"):
                require(route["device_uuid"] in identity["inventory"]["gpu"]
                        and route.get("library_path") in report["libraries_sha256"], "Lt runtime identity differs")
        methods = case["methods"]
        require(set(case.get("numerics_before", {})) == set(methods), "Missing pre-timing numerical checks")
        for name, method in methods.items():
            samples = method.get("samples_us")
            require(isinstance(samples, list) and len(samples) == report["rounds"]
                    and all(type(s) in (int, float) and math.isfinite(s) and s > 0 for s in samples), "Invalid raw timing rounds")
            _same(method.get("median_us"), float(statistics.median(samples)), name + ".median_us")
            for field, value in (("min_us", min(samples)), ("max_us", max(samples))):
                if field in method:
                    _same(method[field], float(value), name + "." + field)
            pre = _numerics(case["numerics_before"][name], copies, m, n)
            _numerics(method.get("numerics"), copies, m, n, before_passed=pre)
        summary = summarize_methods(methods, case["vendor_routes"], "custom")
        _same(case.get("summary"), summary, "summary")
        observations = []
        for sample in case.get("gpu_telemetry", []):
            if sample.get("ok") is True and sample.get("phase") == "measurement":
                fields = [s.strip() for s in sample.get("state", "").split(",")]
                require(len(fields) == 8 and fields[5], "Malformed timed telemetry")
                observations.append(fields)
        clocks = sorted({fields[5] for fields in observations})
        stable = len(clocks) == 1
        require(case.get("observed_memory_clocks") == clocks and case.get("memory_clock_regime_stable") is stable,
                "Declared clock regime differs from raw timed telemetry")
        context_ok = case.get("context_before") == case.get("context_after") == context
        accepted = bool(summary["passes_10pct_gate"] and all(case[key] for key in bit_keys) and context_ok and stable)
        require(type(case.get("accepted_10pct_target")) is bool and case["accepted_10pct_target"] == accepted,
                "Cached case acceptance disagrees with raw evidence")
        rows.append({"seed": report["seed"], "pid": report["pid"], "run_id": report["run_id"],
                     "shape_mnk": list(spec["shape_mnk"]), "config": spec["config"], **summary,
                     "accepted_10pct": accepted, "observed_memory_clocks": clocks,
                     "observed_pstates": sorted({fields[1] for fields in observations}),
                     "observed_throttle_flags": sorted({(fields[6], fields[7]) for fields in observations}),
                     "failed_numerical_methods": [name for name, method in methods.items() if not method["numerics"]["passed"]],
                     "original_bits_preserved": all(case[key] for key in bit_keys), "context_matched": context_ok})
    return {"identity": identity, "cases": rows}


def build_cohort(reports, protocol, *, dtype, profile):
    """Require every registered process, then qualify each shape independently."""
    seeds = protocol["held_out_seeds"]
    require(len(seeds) == 3 and len(set(seeds)) == 3, "Expected three registered seeds")
    require(len(reports) == len(seeds) and {r.get("seed") for r in reports} == set(seeds), "Need exactly three registered fresh processes")
    for field in ("pid", "run_id"):
        require(len({r.get(field) for r in reports}) == len(reports), "Duplicate fresh process identity")
    audits = [audit_process(r, protocol, dtype=dtype, profile=profile) for r in sorted(reports, key=lambda r: seeds.index(r["seed"]))]
    require(all(a["identity"] == audits[0]["identity"] for a in audits), "Runtime, measured source or hardware identity changed")
    cases = [case for audit in audits for case in audit["cases"]]
    shapes = []
    for spec in protocol["cases_by_dtype"][dtype]:
        group = [c for c in cases if c["shape_mnk"] == spec["shape_mnk"]]
        original_cases = [r["cases"][index] for r in reports
                          for index, candidate in enumerate(protocol["cases_by_dtype"][dtype]) if candidate == spec]
        for key in ("weight_sha256", "activation_sha256"):
            require(len({c[key] for c in original_cases}) == len(seeds), "Fresh processes reused identical seeded inputs")
        clocks_match = all(c["observed_memory_clocks"] == group[0]["observed_memory_clocks"] for c in group)
        gains = [c["time_saved_pct"] for c in group if c["time_saved_pct"] is not None]
        lowers = [c["ci95_pct"][0] for c in group if c["ci95_pct"] is not None]
        shapes.append({"shape_mnk": list(spec["shape_mnk"]), "config": spec["config"], "cases": len(group),
                       "passing_cases": sum(c["accepted_10pct"] for c in group),
                       "latency_saved_pct_range": [min(gains), max(gains)] if gains else None,
                       "minimum_per_case_ci95_lower_pct": min(lowers) if lowers else None,
                       "same_memory_clock_regime": clocks_match,
                       "accepted_10pct": bool(clocks_match and all(c["accepted_10pct"] for c in group))})
    return {"kind": "native-precision-evidence", "status": "complete", "dtype": dtype, "power_profile": profile,
            "seeds": list(seeds), "identity": copy.deepcopy(audits[0]["identity"]), "cases": cases, "shapes": shapes,
            "qualified_shapes": sum(shape["accepted_10pct"] for shape in shapes),
            "scope": "Captured streaming GPU work; native requested inputs/weights, FP32 accumulation/output. Synthetic inputs. No eager/model claim.",
            "evidence_limits": [
                "Discovery covers currently available parseable project-local reports; deleted or damaged unknown attempts cannot be detected.",
                "The frozen producer sets allow_fp16_accumulation=False but does not record that flag; this is source evidence.",
                "Input hashes establish distinct bytes across processes, not an independent CPU proof of GPU seed generation.",
                "FP32 accumulation is established by the frozen source and eligible vendor contracts, not measured from output values."],
            "energy_claim": False, "unlocked_comparison": "pending"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(report, root=ROOT):
    """Read recorded dependencies, never initialize CUDA or load the libraries."""
    from experiment_control import project_path
    expected = {}
    for key in ("source_sha256", "source_profiles_sha256", "libraries_sha256"):
        for name, digest in report[key].items():
            require(name not in expected or expected[name] == digest, "Conflicting file identities")
            expected[name] = digest
    expected[report["ptx_path"]] = report["ptx_sha256"]
    compiler_digest = report["nvrtc_identity"].rsplit("sha256=", 1)[-1]
    require(_digest(compiler_digest), "Missing compiler binary SHA")
    expected[report["nvrtc_path"]] = compiler_digest
    for name, digest in expected.items():
        path = project_path(root, name)
        require(path.is_file() and _sha(path) == digest, f"Measured file changed or missing: {name}")
    return expected


def discover_attempts(root, *, dtype, profile):
    """Include every matching attempt and its restoration state, not best-of-N."""
    root = Path(root).resolve()
    prefix = f"native-precision-{profile}-{dtype}-"
    # The producer accepts any project-local path, including .scratch/.cache.
    # Read contents before interpreting names: a raw output may end in -state.
    paths = sorted(path for path in root.rglob("*.json") if path.is_file() and path.resolve().is_relative_to(root))
    reports, states, artifacts = [], {}, []
    for path in paths:
        try:
            row = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            require(not path.name.startswith(prefix), f"Unreadable matching attempt artifact: {path}: {error}")
            continue
        if not isinstance(row, dict):
            require(not path.name.startswith(prefix), "Unknown matching non-object artifact")
            continue
        matching_header = (row.get("kind") == "native-precision-projection" and row.get("dtype") == dtype
                           and row.get("operator_context", {}).get("system76_power_profile")
                           == "Power Profile: " + profile.title())
        if row.get("kind") == "native-precision-projection" and (matching_header or path.name.startswith(prefix)):
            reports.append((path, row))
        elif path.stem.endswith("-state"):
            if path.name.startswith(prefix):
                states[path.with_name(path.stem[:-6] + ".json")] = row
                artifacts.append({"path": str(path.relative_to(root)), "sha256": _sha(path)})
            continue
        elif path.name.startswith(prefix) and row.get("kind") not in ("native-precision-evidence", "native-precision-process-audit"):
            raise ValueError(f"Unknown matching attempt artifact: {path.name}")
        else:
            continue
        artifacts.append({"path": str(path.relative_to(root)), "sha256": _sha(path)})
    for path, report in reports:
        state = states.pop(path, None)
        if state is None:
            state_path = path.with_name(path.stem + "-state.json")
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                artifacts.append({"path": str(state_path.relative_to(root)), "sha256": _sha(state_path)})
        require(state is not None and state.get("status") == "complete" and state.get("restored") is True
                and state.get("pid") == report.get("pid"), f"Unfinished or unrestored attempt: {path.name}")
        _restoration(state)
    # The producer emits the raw header before timed cases. A failed state with no
    # raw report proves preflight-only; retain it. Any raw failed report is rejected.
    for path, state in states.items():
        require(not path.exists() and state.get("status") == "failed" and isinstance(state.get("error"), str)
                and _integer(state.get("pid"), 1), "Unclassified or unfinished preflight attempt")
        if "initial" in state:
            require(state.get("restored") is True, "Preflight settings were not restored")
            _restoration(state)
        else:
            require(not {"during", "after_restore", "restored"}.intersection(state), "Incomplete preflight restoration record")
    require(reports, "No raw process reports found")
    return [report for _, report in reports], artifacts


def _restoration(state):
    initial, after = state.get("initial", {}), state.get("after_restore", {})
    require(initial.get("profile") == after.get("profile") and initial.get("brightness") == after.get("brightness")
            and initial.get("profile") in ("performance", "battery", "balanced") and _integer(initial.get("brightness")),
            "Restored settings differ from original settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--power-profile", required=True, choices=("performance", "battery"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from experiment_control import project_path, write_json
    from native_precision import frozen_cases
    output = project_path(ROOT, args.output)
    require(output.suffix == ".json" and not output.exists(), "Use a new project-local JSON output")
    protocol_path = ROOT / "experiments/native_precision_refresh.json"
    protocol = json.loads(protocol_path.read_text())
    frozen_cases(protocol, args.dtype)
    reports, artifacts = discover_attempts(ROOT, dtype=args.dtype, profile=args.power_profile)
    result = build_cohort(reports, protocol, dtype=args.dtype, profile=args.power_profile)
    for report in reports:
        verify_files(report, root=ROOT)
        require(report["source_profiles_sha256"] == {r["path"]: r["sha256"] for r in protocol["source_profiles"].values()},
                "Report source-profile set differs from registration")
    result["artifacts"] = artifacts
    result["protocol_sha256"] = _sha(protocol_path)
    result["auditor_sha256"] = _sha(Path(__file__))
    write_json(output, result)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "qualified_shapes": result["qualified_shapes"],
                      "registered_shapes": len(result["shapes"])}, indent=2))


if __name__ == "__main__":
    main()
