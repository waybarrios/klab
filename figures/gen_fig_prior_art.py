"""Measured training comparisons, including unstable prior-art controls."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source = ROOT / "results/prior-art-nwc-performance-r1.json"
data = json.loads(source.read_text())
assert data["status"] == "complete" and not data["accepted_10pct"]
rows = []
for case in data["cases"]:
    prior = case["against_prior_art"]
    rows.append({"output": case["output_dtype"], "n": case["shape_mnk"][1], "k": case["shape_mnk"][2],
                 "vendor_us": case["best_vendor_us"], "ours_us": case["methods"][case["selected"]]["median_us"],
                 "nwc_us": prior["best_vendor_us"], "vs_vendor_pct": case["time_saved_pct"],
                 "vs_vendor_low": case["ci95_pct"][0], "vs_vendor_high": case["ci95_pct"][1],
                 "vs_nwc_pct": prior["time_saved_pct"], "vs_nwc_low": prior["ci95_pct"][0],
                 "vs_nwc_high": prior["ci95_pct"][1], "vendor_aa_pct": case["aa_time_saved_pct"],
                 "nwc_aa_pct": prior["aa_time_saved_pct"], "nwc_aa_passed": prior["aa_passed"]})
with (ROOT / "figures/prior_art_training.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), sharex=True)
x = np.arange(len(rows))
for ax, prefix, title, color in ((axes[0], "vs_vendor", "Against best eligible vendor", "#0e8879"),
                                  (axes[1], "vs_nwc", "Against best eligible NWC layout", "#335c81")):
    for i, row in enumerate(rows):
        value, lo, hi = (row[prefix + suffix] for suffix in ("_pct", "_low", "_high"))
        valid = prefix == "vs_vendor" or row["nwc_aa_passed"]
        ax.errorbar(i, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]],
                    fmt="o" if valid else "x", capsize=4, color=color if valid else "#999999",
                    markersize=7)
        ax.annotate(f"{value:+.1f}%" if valid else "A/A failed", (i, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, linewidth=.8, color="#555555")
    ax.axhline(10, linewidth=.8, linestyle="--", color="#b67878")
    ax.grid(axis="y", alpha=.2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x, [f"{r['output'].upper()} out\nN={r['n']}\nK={r['k']}" for r in rows], fontsize=8)
    ax.set_ylim(-10, 27)
axes[0].set_ylabel("Our CUDA kernel: GPU latency saved (%)")
fig.suptitle("Real layer-0 weights, M=1, Performance profile", fontsize=12)
fig.text(.5, .012, "Training only, seed 901. Exact BF16 weights and FP32 accumulation. Error bars: paired 95% intervals.",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, .04, 1, .95))
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/prior_art_training.{ext}", dpi=220)
plt.close(fig)

lines = ["# NWC comparison on this GPU", "",
         "Preliminary training evidence, not held-out or model-level acceptance. "
         "Real Qwen layer-0 checkpoint weights, seeded synthetic BF16 activations, M=1, FP32 accumulation. "
         "All arms use identical inputs and an original-weight ring of at least 80 MiB. "
         "GPU graph timing includes NWC's required activation widening, fused decode and final output kernel.", "",
         "| Output | N,K | Vendor µs | Our CUDA µs | NWC µs | Ours vs vendor % saved | Ours vs NWC % saved | NWC A/A % |",
         "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |"]
for r in rows:
    valid = f"{r['vs_nwc_pct']:+.2f}" if r["nwc_aa_passed"] else "unstable; no claim"
    lines.append(f"| {r['output']} | {r['n']},{r['k']} | {r['vendor_us']:.3f} | {r['ours_us']:.3f} | "
                 f"{r['nwc_us']:.3f} | {r['vs_vendor_pct']:.2f} | {valid} | {r['nwc_aa_pct']:+.3f} |")
lines += ["", "Negative percentages mean our kernel is slower. NWC wins the three stable comparisons by 2.19–5.15%. "
          "The fourth comparison has 12.12% A/A disagreement and its interval crosses zero, so its positive point estimate does not count. "
          "All measured arms passed their numerical checks and every decoded weight bit matched the original. "
          "Both NWC layouts were retained; layout 0 was the fastest eligible route in these four cases.", "",
          "NWC's official 0.10.1 Linux wheel was downloaded, SHA256-verified and extracted inside the project, not installed. "
          "Its format is version 10, distinct from the current source checkout's version 11. "
          "Binary SHA256: `8ca545ce6ad5a3f4eb52cbf37932b6f9b2428b8191094b59ce92a6827a5cb2b0`. "
          "The stock driver was used, with memory telemetry near 810 MHz. No gpu-unlock activation.", "",
          "![Training comparison](../figures/prior_art_training.png)", "",
          "Raw measurements and all intervals: [prior-art-nwc-performance-r1.json](prior-art-nwc-performance-r1.json). "
          "Compatibility checks: [nwc-prebuilt-compatibility.json](nwc-prebuilt-compatibility.json). "
          "Prior art: [NWC](https://github.com/parda21/NWC), Apache-2.0. "
          "This establishes no advantage over NWC. Held-out processes, full M/output coverage, eager dispatch and unlocked tests remain pending.", ""]
(ROOT / "results/prior-art-nwc-performance-r1.md").write_text("\n".join(lines))
print("Saved prior-art table, PNG, PDF and CSV; unstable fourth comparison explicitly labelled")
