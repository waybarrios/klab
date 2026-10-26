"""Paper-style native-precision figures from all six audited process cohorts.

Every dot is one process, with its own paired interval. No pooled confidence
interval or full-model, nominal-clock, energy or unlocked claim is made.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (971, 977, 983)
DTYPES = ("fp32", "fp16", "bf16")
PROFILES = ("performance", "battery")
COLORS = {"fp32": "#0072B2", "fp16": "#D55E00", "bf16": "#009E73"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohorts():
    protocol_path = ROOT / "experiments/native_precision_refresh.json"
    protocol = json.loads(protocol_path.read_text())
    inputs, cases, identity = [], [], None
    for profile in PROFILES:
        for dtype in DTYPES:
            path = ROOT / f"results/native-precision-{profile}-{dtype}-evidence-r1.json"
            data = json.loads(path.read_text())
            expected = protocol["cases_by_dtype"][dtype]
            if (data.get("kind") != "native-precision-evidence" or data.get("status") != "complete"
                    or data.get("dtype") != dtype or data.get("power_profile") != profile
                    or data.get("seeds") != list(SEEDS) or data.get("protocol_sha256") != digest(protocol_path)
                    or len(data.get("cases", [])) != len(expected) * len(SEEDS)
                    or len(data.get("shapes", [])) != len(expected)):
                raise ValueError(f"Need the entire registered audited cohort: {path.name}")
            if identity is None:
                identity = data["identity"]
            if data["identity"] != identity:
                raise ValueError("Source/runtime/device identity differs across native precision/profile cohorts")
            inventory = data["identity"]["inventory"]
            if ("RTX 3070 Ti Laptop" not in inventory["gpu"]
                    or any(inventory.get(flag) is not False for flag in
                           ("module_loaded", "unlock_boot_option_present", "unlock_success_logged", "post_flr_selectors_logged"))):
                raise ValueError("These figures require the registered stock laptop observations")
            # Bind the figure to every raw attempt and restoration artifact used
            # by the auditor, including any retained preflight-only failures.
            raw_cases = {}
            for reference in data["artifacts"]:
                source = (ROOT / reference["path"]).resolve()
                if not source.is_relative_to(ROOT) or digest(source) != reference["sha256"]:
                    raise ValueError(f"Audited source file changed: {reference['path']}")
                raw = json.loads(source.read_text())
                if raw.get("kind") == "native-precision-projection":
                    for item in raw["cases"]:
                        key = (raw["seed"], tuple(item["shape_mnk"]))
                        if key in raw_cases:
                            raise ValueError("Duplicate raw case behind the audited figure")
                        raw_cases[key] = item
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
            shapes = {tuple(shape["shape_mnk"]): shape for shape in data["shapes"]}
            observed = {(case["seed"], tuple(case["shape_mnk"])) for case in data["cases"]}
            required = {(seed, tuple(spec["shape_mnk"])) for seed in SEEDS for spec in expected}
            if observed != required:
                raise ValueError("Audited seed/shape grid differs from registration")
            for case in data["cases"]:
                shapespec = shapes[tuple(case["shape_mnk"])]
                interval = case["ci95_pct"]
                original = raw_cases[(case["seed"], tuple(case["shape_mnk"]))]
                custom_us = original["methods"]["custom"]["median_us"]
                vendor_us = case["best_vendor_us"]
                cases.append({"profile": profile, "native_input_weight_dtype": dtype, "seed": case["seed"],
                              "m": case["shape_mnk"][0], "n": case["shape_mnk"][1], "k": case["shape_mnk"][2],
                              "config": case["config"], "best_vendor": case["best_vendor"],
                              "best_vendor_us": case["best_vendor_us"], "latency_saved_pct": case["time_saved_pct"],
                              "custom_us": custom_us,
                              "latency_saved_us": vendor_us - custom_us if vendor_us is not None else None,
                              "ci_low_pct": interval[0] if interval else None,
                              "ci_high_pct": interval[1] if interval else None,
                              "vendor_aa_pct": case["aa_time_saved_pct"], "case_gate_passed": case["accepted_10pct"],
                              "shape_three_process_gate_passed": shapespec["accepted_10pct"],
                              "memory_clocks": json.dumps(case["observed_memory_clocks"]),
                              "pstates": json.dumps(case["observed_pstates"]),
                              "throttle_flags": json.dumps(case["observed_throttle_flags"]),
                              "failed_numerical_methods": json.dumps(case["failed_numerical_methods"]),
                              "cohort_sha256": inputs[-1]["sha256"]})
    return protocol, inputs, cases


def figure(rows, shapes, stem, *, large=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.6 if not large else 4.3), sharex=True, sharey=True)
    finite = [row for row in rows if row["ci_low_pct"] is not None]
    lower = min([0., *[min(row["ci_low_pct"], row["latency_saved_pct"]) for row in finite]])
    upper = max([10., *[max(row["ci_high_pct"], row["latency_saved_pct"]) for row in finite]])
    margin = max(1., (upper - lower) * .06)
    for axis, profile in zip(axes, PROFILES):
        for row in [item for item in rows if item["profile"] == profile]:
            shape = (row["m"], row["n"], row["k"])
            precision_offset = 0. if large else (DTYPES.index(row["native_input_weight_dtype"]) - 1) * .28
            seed_offset = (SEEDS.index(row["seed"]) - 1) * (.13 if large else .085)
            y = shapes.index(shape) + precision_offset + seed_offset
            color = COLORS[row["native_input_weight_dtype"]]
            if row["latency_saved_pct"] is None:
                axis.text(.02, y, "no eligible vendor", color=color, fontsize=6,
                          transform=axis.get_yaxis_transform())
                continue
            axis.hlines(y, row["ci_low_pct"], row["ci_high_pct"], color=color, linewidth=.9)
            axis.plot(row["latency_saved_pct"], y,
                      "o" if row["shape_three_process_gate_passed"] else "x", color=color, markersize=3)
        axis.axvline(0, color=".4", linewidth=.75)
        axis.axvline(10, color=".4", linewidth=.75, linestyle="--")
        axis.set_title(f"{profile.capitalize()} profile on AC", fontsize=10)
        axis.set_yticks(range(len(shapes)), [f"{m} × {n} × {k}" for m, n, k in shapes])
        axis.grid(axis="x", alpha=.18)
        axis.set_xlim(lower - margin, upper + margin)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Projection shape M × N × K", fontsize=9)
    title = "Native FP32 large projections" if large else "Native precision: small projections"
    fig.suptitle(title, fontsize=12, y=.985)
    fig.text(.5, .925 if not large else .91,
             "Original requested-dtype inputs/weights · FP32 accumulation/output · strongest eligible tested vendor",
             ha="center", fontsize=7.5)
    handles = [Line2D([], [], color=COLORS[dtype], marker="o", linewidth=.9, label=dtype.upper())
               for dtype in (("fp32",) if large else DTYPES)]
    handles.append(Line2D([], [], color=".25", marker="x", linestyle="none", label="Shape lacks all-three-process qualification"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .09), ncol=len(handles), fontsize=7.5)
    fig.supxlabel("Latency saved versus best eligible vendor (%)", y=.16, fontsize=9)
    fig.text(.5, .052, "Each dot/interval: one process and its paired 95% CI; seeds 971, 977, 983 from top to bottom within each dtype.",
             ha="center", fontsize=7.2)
    clocks = sorted({clock for row in rows for clock in json.loads(row["memory_clocks"])})
    pstates = sorted({pstate for row in rows for pstate in json.loads(row["pstates"])})
    regime = f"observed {', '.join(pstates)} / memory {', '.join(clocks)}"
    fig.text(.5, .023, f"Streaming CUDA graphs · stock RTX 3070 Ti Laptop · {regime}; throttle flags retained in CSV.",
             ha="center", fontsize=7.2)
    fig.subplots_adjust(left=.18, right=.985, top=.84 if not large else .8, bottom=.245, wspace=.13)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    with stem.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


def main():
    # Do not create a partly complete figure set when a required cohort is absent.
    protocol, inputs, cases = load_cohorts()
    stems = [ROOT / "figures/native_precision_small", ROOT / "figures/native_fp32_large"]
    metadata = ROOT / "figures/native_precision_provenance.json"
    outputs = [stem.with_suffix(suffix) for stem in stems for suffix in (".png", ".pdf", ".csv")] + [metadata]
    if any(path.exists() for path in outputs):
        raise ValueError("Preserve existing figure exports")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "legend.frameon": False})
    small = [tuple(spec["shape_mnk"]) for spec in protocol["cases_by_dtype"]["fp16"]]
    large = [tuple(spec["shape_mnk"]) for spec in protocol["cases_by_dtype"]["fp32"] if tuple(spec["shape_mnk"]) not in small]
    figure([row for row in cases if (row["m"], row["n"], row["k"]) in small], small, stems[0])
    figure([row for row in cases if (row["m"], row["n"], row["k"]) in large], large, stems[1], large=True)
    metadata.write_text(json.dumps({"kind": "native-precision-figure-provenance", "inputs": inputs,
                                    "generator_sha256": digest(Path(__file__)), "raw_cases": len(cases),
                                    "scope": "Every registered process shown. Three-process shape gates, no pooled CI. No model/energy/unlocked claim."}, indent=2) + "\n")
    print(f"Exported native-precision PNG/PDF/CSV with all {len(cases)} raw case summaries")


if __name__ == "__main__":
    main()
