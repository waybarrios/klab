"""Independent CPU audit of a completed campaign's raw evidence."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(path):
    data = json.loads(path.read_text())
    if data.get("status") != "complete" or data.get("kind") != "compressed-projection":
        raise ValueError("Audit requires a complete compressed campaign")
    for name, digest in data["identity"]["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Current execution dependency differs: {name}")
    numerical_failures = []
    aa_failures = []
    for case in data["cases"]:
        tag = (case["weight_tensor"], tuple(case["shape_mnk"]))
        if case["raw_ring_bytes"] < 80 * 1024**2 or case["raw_ring_over_l2"] < 20:
            raise ValueError(f"Insufficient physical ring: {tag}")
        if not case["original_bits_preserved"] or not case["raw_template_bitwise"]:
            raise ValueError(f"Bit identity failed: {tag}")
        if not case["operating_context_stable"]:
            raise ValueError(f"Profile or AC changed: {tag}")
        if not any(s.get("ok") and s.get("phase") == "measurement" for s in case["gpu_telemetry"]):
            raise ValueError(f"Missing telemetry wholly inside timing: {tag}")
        for name, method in case["methods"].items():
            samples = method["samples_us"]
            if len(samples) < 31 or not all(np.isfinite(x) and x > 0 for x in samples):
                raise ValueError(f"Invalid raw rounds: {tag} {name}")
            if not np.isclose(statistics.median(samples), method["median_us"], rtol=1e-12, atol=1e-10):
                raise ValueError(f"Median differs from raw rounds: {tag} {name}")
            expected_elements = case["copies"] * case["shape_mnk"][0] * case["shape_mnk"][1]
            if method["numerics"]["elements_checked"] != expected_elements:
                raise ValueError(f"Missing output coverage: {tag} {name}")
            if not method["numerics"]["passed"]:
                numerical_failures.append([*tag, name])
        eligible = [name for name in case["vendor_routes"]
                    if case["methods"][name]["numerics"]["passed"]
                    and case["methods"][name + "_aa"]["numerics"]["passed"]]
        if not eligible:
            if case["best_vendor"] is not None or case["passes_10pct_gate"]:
                raise ValueError(f"Invalid no-eligible-vendor gate: {tag}")
            continue
        winner = min(eligible, key=lambda name: case["methods"][name]["median_us"])
        if winner != case["best_vendor"] or winner.endswith("_aa"):
            raise ValueError(f"Incorrect vendor denominator: {tag}")
        baseline = case["methods"][winner]["median_us"]
        aa = 100 * (baseline - case["methods"][winner + "_aa"]["median_us"]) / baseline
        gain = 100 * (1 - case["methods"][case["selected"]]["median_us"] / baseline)
        if not np.isclose(aa, case["aa_time_saved_pct"]) or not np.isclose(gain, case["time_saved_pct"]):
            raise ValueError(f"Incorrect percentage or winner/control mapping: {tag}")
        if abs(aa) > 1:
            aa_failures.append(tag)
        candidate = np.asarray(case["methods"][case["selected"]]["samples_us"])
        indices = np.random.default_rng(2026).integers(0, len(candidate), size=(2000, len(candidate)))
        vendor_draws = np.min([np.median(np.asarray(case["methods"][name]["samples_us"])[indices], axis=1)
                               for name in eligible], axis=0)
        boot = 100 * (vendor_draws - np.median(candidate[indices], axis=1)) / vendor_draws
        interval = np.quantile(boot, [.025, .975])
        if not np.allclose(interval, case["ci95_pct"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"CI does not reproduce from raw paired rounds: {tag}")
        passes = bool(abs(aa) <= 1 and interval[0] > 10 and case["methods"][case["selected"]]["numerics"]["passed"])
        if passes != case["passes_10pct_gate"]:
            raise ValueError(f"Incorrect strict confidence/precision/A-A gate: {tag}")
    return {"status": "audit-passed", "run": str(path.relative_to(ROOT)), "pid": data["pid"],
            "run_id": data["run_id"], "cases": len(data["cases"]),
            "numerical_failures_retained": numerical_failures, "aa_failures_retained": aa_failures,
            "acceptance_claim": data["accepted_10pct"],
            "scope": "Evidence audit, not a substitute for complete held-out coverage or three-process acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    path = (ROOT / args.run).resolve()
    if not path.is_relative_to(ROOT):
        parser.error("Run path must stay in the project")
    try:
        result = audit(path)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "audit-failed", "run": str(path.relative_to(ROOT)), "error": str(error)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
