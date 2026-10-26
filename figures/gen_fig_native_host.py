"""Paper-style intervals for every registered native-host API case and seed."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    aggregate = ROOT / "results/native-host-performance-evidence-r1.json"
    evidence = json.loads(aggregate.read_text())
    if evidence["kind"] != "native-host-api-evidence" or evidence["status"] != "complete":
        raise ValueError("Need the complete audited three-process API cohort")
    audits = []
    for reference in evidence["processes"]:
        path = (ROOT / reference["path"]).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Raw process report changed")
        audit = json.loads(path.with_name(path.stem + "-audit.json").read_text())
        if audit["status"] != "audit-passed" or audit["report"] != reference:
            raise ValueError("Missing independent audit for this process")
        audits.append(audit)
    audits.sort(key=lambda item: item["seed"])
    if len(audits) != 3:
        raise ValueError("Need three fresh processes")
    categories = ("q_proj", "gate_proj", "down_proj", "k_proj", "v_proj", "synthetic-control")

    def category(case):
        if case["control"]:
            return "synthetic-control"
        return case["weight_tensor"].split(".")[-2]

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.6), sharex=True, sharey=True)
    colors = ("#0072B2", "#009E73", "#D55E00")
    rows, lows, highs = [], [], []
    for column, output in enumerate(("bf16", "fp32")):
        for row, scope in enumerate(("eager", "graph")):
            axis = axes[row, column]
            for index, audit in enumerate(audits):
                selected = [c for c in audit["case_summaries"] if c["output_dtype"] == output]
                for case in selected:
                    values = case[scope]["selected"]
                    point, (low, high) = values["time_saved_pct"], values["ci95_pct"]
                    y = categories.index(category(case)) + (index - 1) * .19
                    axis.hlines(y, low, high, color=colors[index], linewidth=1.1)
                    marker = "o" if values["case_gate_passed"] else "x"
                    axis.plot(point, y, marker, color=colors[index], markersize=4)
                    lows.append(low)
                    highs.append(high)
                    rows.append({"seed": audit["seed"], "tensor": case["weight_tensor"],
                                 "m": case["shape_mnk"][0], "n": case["shape_mnk"][1], "k": case["shape_mnk"][2],
                                 "output_dtype": output, "scope": scope, "route": case["selected_route"],
                                 "time_saved_pct": point, "ci_low_pct": low, "ci_high_pct": high,
                                 "vendor_aa_pct": case[scope]["aa_time_saved_pct"],
                                 "case_gate_passed": values["case_gate_passed"],
                                 "scope_controls_passed": audit["scope_controls"][scope],
                                 "raw_report_sha256": audit["report"]["sha256"]})
            controls = sum(a["scope_controls"][scope] for a in audits)
            axis.axvline(0, color=".35", linewidth=.8)
            axis.axvline(10, color=".45", linestyle="--", linewidth=.8)
            axis.set_title(f"{scope.capitalize()} · {output.upper()} output\nScope controls: {controls}/3 runs pass", fontsize=9)
            axis.set_yticks(range(len(categories)), ("Q · M=2", "Gate · M=1", "Down · M=1",
                                                    "K · M=1", "V · M=1", "64×64 control"))
            axis.grid(axis="x", alpha=.18)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(min(-5, min(lows) - 2), max(65, max(highs) + 2))
    fig.suptitle("Native C++ host dispatch: selected route versus direct Torch", fontsize=12, y=.985)
    fig.text(.5, .94, "BF16 inputs/weights · FP32 accumulation · three fresh processes · 31 rounds/case",
             ha="center", fontsize=8)
    handles = [Line2D([], [], color=color, marker="o", linewidth=1, label=f"Seed {a['seed']}")
               for color, a in zip(colors, audits)]
    handles += [Line2D([], [], color=".25", marker="x", linestyle="none", label="Case gate failed")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .073), ncol=4, fontsize=8)
    fig.supxlabel("Latency saved versus direct allocating Torch (%)", y=.145, fontsize=10)
    fig.text(.5, .047, "Bars: paired 95% intervals. Eager includes allocations/sync; graph replay excludes host and allocation cost.",
             ha="center", fontsize=7.7)
    fig.text(.5, .022, "Stock RTX 3070 Ti Laptop · Performance on AC · memory 810 MHz · failed controls prevent general API acceptance",
             ha="center", fontsize=7.7)
    fig.subplots_adjust(left=.15, right=.985, top=.84, bottom=.225, hspace=.3, wspace=.12)
    stem = ROOT / "figures/native_host_performance"
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        if path.exists():
            raise ValueError("Preserve previous figure artifacts")
        fig.savefig(path, dpi=300)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)
    print("Exported native host PNG, vector PDF and all-case CSV")


if __name__ == "__main__":
    main()
