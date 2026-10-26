"""Plot actual prototype latency reductions, kept separate from acceptance."""
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

rows = []
for profile in ("battery", "performance"):
    source = ROOT / f"results/compressed-bf16-{profile}-revision1.json"
    data = json.loads(source.read_text())
    assert data["status"] == "complete" and not data["accepted_10pct"]
    for case in data["cases"]:
        selected = case["selected"]
        candidate = case["candidates"][selected]
        assert candidate["raw_template_bitwise"] and candidate["numerics_passed"]
        rows.append({"profile": profile, "m": 1, "n": case["shape_mnk"][1], "k": case["shape_mnk"][2],
                     "vendor_us": case["best_vendor_us"], "kernel_us": case["methods"][selected]["median_us"],
                     "latency_saved_pct": candidate["time_saved_pct"], "ci95_low_pct": candidate["ci95_pct"][0],
                     "ci95_high_pct": candidate["ci95_pct"][1], "aa_pct": case["aa_time_saved_pct"],
                     "config": selected, "weight_sha256": case["weight_sha256"]})

csv_path = ROOT / "figures/compressed_preliminary.csv"
with csv_path.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 4.0))
positions = np.arange(2)
for offset, profile, color in ((-.18, "battery", "#335c81"), (.18, "performance", "#0e8879")):
    selected = [row for row in rows if row["profile"] == profile]
    values = np.array([row["latency_saved_pct"] for row in selected])
    errors = np.array([[row["latency_saved_pct"] - row["ci95_low_pct"] for row in selected],
                       [row["ci95_high_pct"] - row["latency_saved_pct"] for row in selected]])
    bars = ax.bar(positions + offset, values, width=.32, color=color,
                  yerr=errors, capsize=4, label=profile.title() + " profile (AC)")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=7, fontsize=10)
ax.axhline(10, color="#9b6262", linestyle="--", linewidth=1, label="10% target")
ax.set_xticks(positions, ["N=4864, K=896", "N=896, K=4864"])
ax.set_ylabel("GPU latency saved vs best eligible vendor (%)")
ax.set_ylim(0, 25)
ax.set_title("Exact BF16 projections, FP32 accumulation, M=1")
ax.yaxis.grid(True, alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=.95, fontsize=8)
fig.text(.5, .015, "Preliminary: layer 0, seed 901, selected among 8 configs. Held-out validation pending.",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, .05, 1, 1))
for suffix in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/compressed_preliminary.{suffix}", dpi=240)
plt.close(fig)

lines = ["# Compressed BF16 prototype results", "",
         "Real Qwen2.5-0.5B layer-0 weights, synthetic seed901 activations, M=1. Original BF16 bits preserved; "
         "FP32 accumulation and native BF16 output. Tuning results, not held-out or full-model acceptance.", "",
         "| Profile on AC | N,K | Best vendor us | Compressed CUDA us | Latency saved % | Paired 95% interval % |",
         "| --- | --- | ---: | ---: | ---: | --- |"]
for row in rows:
    lines.append(f"| {row['profile']} | {row['n']},{row['k']} | {row['vendor_us']:.3f} | {row['kernel_us']:.3f} | "
                 f"{row['latency_saved_pct']:.3f} | [{row['ci95_low_pct']:.3f}, {row['ci95_high_pct']:.3f}] |")
lines.extend(["", "All four cases selected t128/v8. Each profile retimed the same frozen vendor pool, including tuned cuBLASLt, "
              "with per-route A/A controls and every captured output checked. Original weight bits and raw/coded output bits matched.", "",
              "The normal profile control was used. The test harness restored the initial profile and brightness; afterward the user selected Performance and asked to keep it. "
              "Both profiles stayed near40.35GB/s effective read bandwidth; profile selection did not remove that memory regime.", "",
              "![Prototype latency reductions](../figures/compressed_preliminary.png)", "",
              "Pending: all training layers, held-out layers8/16/23 in three fresh processes, full M/output grid, eager performance, "
              "prepared dispatch, external implementation comparisons and required stock/unlocked measurements. "
              "Intervals shown here are exploratory after candidate selection and do not replace held-out validation.", ""])
(ROOT / "results/compressed-bf16-preliminary.md").write_text("\n".join(lines))
print("\n".join(lines[:10]))
print("Saved figures/compressed_preliminary.png, .pdf and .csv")
