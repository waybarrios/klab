"""Pair fusion and custom comparison remain separate, with every failed gate."""
import csv
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="replace this generator's existing figure exports")
    args = parser.parse_args()

    data = json.loads((ROOT / "results/kv-pair-performance-evidence-r1.json").read_text())
    composition = json.loads((ROOT / "results/kv-large-composition-evidence-r1.json").read_text())
    if data["status"] != "complete" or composition["status"] != "complete":
        raise ValueError("Need complete independently audited pair/composition results")
    seeds, layers = (1013, 1019, 1021), (8, 16, 23)
    colors = ("#0072B2", "#009E73", "#D55E00")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.0), sharey=True)
    rows = []
    for column, output in enumerate(("bf16", "fp32")):
        cases = [c for c in data["cases"] if c["output_dtype"] == output]
        for row, comparison in enumerate(("fusion", "custom")):
            axis = axes[row, column]
            passed = 0
            for case in cases:
                summary = case[comparison]
                low, high = summary["ci95_pct"]
                point = summary["time_saved_pct"]
                gate = case["fusion_case_passed"] if comparison == "fusion" else summary["passes_10pct_gate"]
                passed += gate
                index = seeds.index(case["seed"])
                y = layers.index(case["layer"]) + (index - 1) * .18
                axis.hlines(y, low, high, color=colors[index], linewidth=1.2)
                axis.plot(point, y, "o" if gate else "x", color=colors[index], markersize=4)
                rows.append({"seed": case["seed"], "layer": case["layer"], "output_dtype": output,
                             "comparison": comparison, "time_saved_pct": point,
                             "ci_low_pct": low, "ci_high_pct": high, "gate_passed": gate})
            axis.set_title(f"{output.upper()} output · {passed}/9 cases pass", fontsize=10)
            axis.set_yticks(range(3), [f"Layer {layer}" for layer in layers])
            axis.grid(axis="x", alpha=.18)
            all_intervals = [case[comparison]["ci95_pct"] for case in data["cases"]]
            lower = min(interval[0] for interval in all_intervals)
            upper = max(interval[1] for interval in all_intervals)
            axis.set_xlim((min(19.5, lower - .15), max(21.3, upper + .15)) if comparison == "fusion"
                          else (min(-15, lower - .15), max(-11, upper + .15)))
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.set_xlabel("Vendor fusion vs separate Torch (%)" if comparison == "fusion"
                            else "Custom vs best fused vendor (%)", fontsize=9)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Exact K/V pair: operation fusion and custom-kernel comparison", fontsize=12, y=.985)
    fig.text(.5, .935, "M=1 · original BF16 weights/inputs · FP32 accumulation · 3 fresh processes · 31 rounds/case",
             ha="center", fontsize=8)
    handles = [Line2D([], [], color=color, marker="o", label=f"Seed {seed}") for color, seed in zip(colors, seeds)]
    handles.append(Line2D([], [], color=".25", marker="x", linestyle="none", label="Case gate failed"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .07), ncol=4, fontsize=8)
    fig.text(.5, .044, "Intervals: paired 95%. BF16 fusion fails layer 23 A/A; custom CUDA still loses to fused vendor.",
             ha="center", fontsize=7.6)
    fig.text(.5, .02, "Bias-free matmul pairs · stock RTX 3070 Ti Laptop · Performance on AC · memory 810 MHz",
             ha="center", fontsize=7.6)
    fig.subplots_adjust(left=.12, right=.96, top=.84, bottom=.205, hspace=.65, wspace=.18)
    stem = ROOT / "figures/kv_pair_fusion"
    for suffix in (".png", ".pdf", ".csv"):
        if stem.with_suffix(suffix).exists() and not args.replace:
            raise ValueError("Preserve existing figure artifacts")
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)
    print("Exported audited K/V fusion PNG/PDF/CSV; composition remains in its separate table")


if __name__ == "__main__":
    main()
