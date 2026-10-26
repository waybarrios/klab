"""Full raw revision2 training grid, including every slowdown."""
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np

data = json.loads((ROOT / "results/compressed-performance-bf16-tune-r2.json").read_text())
assert data["status"] == "complete" and not data["accepted_10pct"]
suffixes = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
ms = [1, 2, 4, 8, 16, 64]
grid, labels, rows = [], [], []
for suffix in suffixes:
    cases = [c for c in data["cases"] if c["weight_tensor"] == f"model.layers.0.{suffix}"]
    selected = [next(c for c in cases if c["shape_mnk"][0] == m) for m in ms]
    grid.append([c["time_saved_pct"] for c in selected])
    n, k = selected[0]["shape_mnk"][1:]
    labels.append(f"{suffix.split('.')[-2].replace('_proj', '')}  ({n}×{k})")
    for m, case in zip(ms, selected):
        rows.append({"projection": suffix, "m": m, "n": n, "k": k,
                     "time_saved_pct": case["time_saved_pct"], "ci95_low_pct": case["ci95_pct"][0],
                     "ci95_high_pct": case["ci95_pct"][1], "aa_passed": case["aa_passed"],
                     "numerics_passed": case["methods"][case["selected"]]["numerics"]["passed"]})
with (ROOT / "figures/mtile_training_grid.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
values = np.asarray(grid)
fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(values, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=25), aspect="auto")
for i in range(len(suffixes)):
    for j in range(len(ms)):
        row = rows[i * len(ms) + j]
        marker = "" if row["aa_passed"] and row["numerics_passed"] else "*"
        ax.text(j, i, f"{values[i,j]:+.1f}%{marker}", ha="center", va="center", fontsize=9,
                color="white" if values[i,j] < -70 or values[i,j] > 18 else "black")
ax.set_xticks(range(len(ms)), [f"M={m}" for m in ms])
ax.set_yticks(range(len(labels)), labels)
ax.set_title("Revision2 CUDA: GPU latency saved versus best eligible vendor")
bar = fig.colorbar(im, ax=ax, extend="min", fraction=.04, pad=.03)
bar.set_label("Latency saved (%)")
fig.text(.5, .035, "Training only: layer0, seed901, frozen M1 configuration. Native BF16 inputs/output, FP32 accumulation.",
         ha="center", fontsize=8)
fig.text(.5, .008, "Negative values are slowdowns. Colors saturate below −100%; labels show actual values. No dispatch or model-speed claim.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, .07, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/mtile_training_grid.{ext}", dpi=220)
plt.close(fig)
print("Saved full grid PNG/PDF/CSV, with all slowdowns retained")
