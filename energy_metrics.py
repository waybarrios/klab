"""CPU-only energy units and paired preliminary comparisons."""
import math
import numbers

import numpy as np


def batch_metrics(start_mj, end_mj, projection_count, *, wall_seconds, gpu_ms):
    if (any(isinstance(x, bool) or not isinstance(x, numbers.Integral) for x in
            (start_mj, end_mj, projection_count)) or min(start_mj, end_mj) < 0
            or end_mj <= start_mj or projection_count <= 0):
        raise ValueError("Require increasing integer mJ counters and a positive exact projection count")
    if (not all(math.isfinite(x) and x > 0 for x in (wall_seconds, gpu_ms))
            or gpu_ms > wall_seconds * 1000 * 1.02 + 1):
        raise ValueError("Invalid wall/CUDA durations or inconsistent units")
    joules = (int(end_mj) - int(start_mj)) / 1000
    per_projection = joules / int(projection_count)
    return {"projection_count": int(projection_count), "gpu_energy_j": joules,
            "j_per_projection": per_projection, "mj_per_projection": 1000 * per_projection,
            "kwh_per_projection": per_projection / 3600000,
            "kwh_per_million_projections": per_projection / 3.6,
            "projections_per_j": 1 / per_projection,
            "average_board_w": joules / wall_seconds,
            "wall_seconds": wall_seconds, "gpu_ms": gpu_ms,
            "gpu_us_per_projection": gpu_ms * 1000 / projection_count,
            "scope": "NVML-reported GPU board energy; no idle subtraction"}


def summarize_energy(rounds, *, draws=2000):
    if len(rounds) < 3:
        raise ValueError("Need at least three paired energy rounds")
    counts = {r[name]["projection_count"] for r in rounds for name in ("vendor", "vendor_aa", "coded")}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("Every energy arm/round must execute the same exact operation count")
    arrays = {name: np.asarray([r[name]["j_per_projection"] for r in rounds], dtype=np.float64)
              for name in ("vendor", "vendor_aa", "coded")}
    if any(not np.isfinite(a).all() or not (a > 0).all() for a in arrays.values()):
        raise ValueError("Require finite positive measured energy per projection")
    median = {name: float(np.median(values)) for name, values in arrays.items()}
    paired_gain = 100 * (arrays["vendor"] - arrays["coded"]) / arrays["vendor"]
    paired_aa = 100 * (arrays["vendor"] - arrays["vendor_aa"]) / arrays["vendor"]
    gain, aa = float(np.median(paired_gain)), float(np.median(paired_aa))
    idx = np.random.default_rng(2026).integers(0, len(rounds), size=(draws, len(rounds)))
    interval = np.quantile(np.median(paired_gain[idx], axis=1), [.025, .975]).tolist()
    aa_passed = bool((np.abs(paired_aa) <= 3).all())
    consistent = bool((paired_gain > 0).all())
    exceeds_aa = bool(np.min(paired_gain) > np.max(np.abs(paired_aa)))
    return {"paired_rounds": len(rounds), "energy_saved_pct": gain, "bootstrap_interval_pct": interval,
            "interval_scope": "Exploratory bootstrap interval; n3 does not give calibrated 95% coverage",
            "paired_energy_saved_pct": paired_gain.tolist(), "paired_aa_energy_saved_pct": paired_aa.tolist(),
            "aa_energy_saved_pct": aa, "aa_passed": aa_passed,
            "median_j_per_projection": median,
            "all_rounds_agree_in_direction": consistent,
            "preliminary_reduction_supported": bool(aa_passed and consistent and exceeds_aa),
            "scope": "Preliminary paired GPU-board energy; not sensor calibration or whole-system efficiency"}
