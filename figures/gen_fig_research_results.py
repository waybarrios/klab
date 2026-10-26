"""Paper-style README figures, generated only from recorded experiment JSON."""
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": .15, "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, GREEN, ORANGE, SKY, GREY = "#0072B2", "#009E73", "#D55E00", "#56B4E9", "#777777"


def read(name):
    data = json.loads((ROOT / "results" / name).read_text())
    assert data["status"] == "complete", name
    return data


def save(fig, name, rows):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/research_{name}.{ext}")
    plt.close(fig)
    with (ROOT / f"figures/research_{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profiles():
    fig, ax = plt.subplots(figsize=(6.75, 2.7))
    rows = []
    for offset, profile, color in ((-.16, "battery", SKY), (.16, "performance", GREEN)):
        data = read(f"compressed-bf16-{profile}-revision1.json")
        for i, case in enumerate(data["cases"]):
            point = case["candidates"][case["selected"]]
            value, (low, high) = point["time_saved_pct"], point["ci95_pct"]
            ax.errorbar(i+offset, value, yerr=[[value-low], [high-value]], fmt="o", color=color,
                        capsize=3, label=f"{profile.title()} profile, on AC" if i == 0 else None)
            ax.annotate(f"{value:.2f}%", (i+offset, value), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8)
            rows.append({"profile": profile, "ac_online": True, "n": case["shape_mnk"][1], "k": case["shape_mnk"][2],
                         "latency_saved_pct": value, "ci95_low_pct": low, "ci95_high_pct": high})
    ax.axhline(10, ls="--", color=GREY, lw=.8, label="10% latency target")
    ax.set_xticks([0, 1], ["N=4864, K=896", "N=896, K=4864"])
    ax.set_ylim(0, 25)
    ax.set_ylabel("GPU latency saved (%)")
    ax.set_title("Power-profile comparison: M=1, BF16 output")
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=7)
    fig.text(.5, .005, "Training only: seed 901/layer 0; FP32 accumulation. Both profiles on AC; memory 810 MHz.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "profiles", rows)


def nwc():
    single = read("prior-art-nwc-performance-r1.json")
    multi = read("prior-art-nwc-performance-multi-r2.json")
    panels = [[c for c in single["cases"] if c["output_dtype"] == "bf16"], multi["cases"]]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), sharey=True, gridspec_kw={"width_ratios": [1, 2]})
    for ax, cases, title in zip(axes, panels, ["M=1: NWC fused token path", "M=2,4: NWC dequantize + F.linear"]):
        labels = []
        for i, case in enumerate(cases):
            stat = case["against_prior_art"]
            value, (low, high) = stat["time_saved_pct"], stat["ci95_pct"]
            valid = stat["aa_passed"]
            ax.errorbar(i, value, yerr=[[value-low], [high-value]], fmt="o" if valid else "x",
                        color=ORANGE if valid else GREY, capsize=3)
            ax.annotate(f"{value:+.1f}%" if valid else "A/A failed", (i, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)
            m, n, k = case["shape_mnk"]
            labels.append(f"M={m}\n{n}×{k}")
            rows.append({"m": m, "n": n, "k": k, "latency_saved_vs_nwc_pct": value,
                         "ci95_low_pct": low, "ci95_high_pct": high, "aa_passed": valid})
        ax.axhline(0, color=GREY, lw=.8)
        ax.set_xticks(range(len(cases)), labels, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(-12, 86)
        ax.grid(axis="y")
    axes[0].set_ylabel("Our GPU latency saved vs NWC (%)")
    fig.text(.5, .005, "Training only; NWC 0.10.1 official APIs. Native BF16 input/output, FP32 accumulation. Not model throughput.", ha="center", fontsize=6.7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "nwc", rows)


def energy_qc():
    data = read("energy-performance-pilot-r2.json")
    rows = []
    fig, ax = plt.subplots(figsize=(6.75, 2.8))
    for i, method in enumerate(("vendor", "vendor_aa", "coded")):
        counter, reported = [], []
        for j, round_data in enumerate(data["rounds"]):
            arm = round_data[method]
            watts = [float(s["state"].split(",")[3].strip().split()[0]) for s in arm["gpu_telemetry"] if s.get("ok")]
            mean_power = float(np.mean(watts))
            counter.append(arm["average_board_w"])
            reported.append(mean_power)
            rows.append({"round": j, "method": method, "counter_implied_w": arm["average_board_w"],
                         "reported_power_trace_mean_w": mean_power,
                         "counter_j_per_projection_unqualified": arm["j_per_projection"],
                         "counter_kwh_per_million_unqualified": arm["kwh_per_million_projections"],
                         "qualified_energy_measurement": False})
        for offset, values, color, label in ((-.17, counter, ORANGE, "Counter delta / wall time"),
                                               (.17, reported, SKY, "Reported power trace mean")):
            ax.bar(i+offset, np.mean(values), width=.28, color=color, label=label if i == 0 else None)
            ax.scatter(np.full(len(values), i+offset), values, s=12, color="black", zorder=3)
    ax.set_xticks([0, 1, 2], ["Vendor", "Vendor A/A twin", "Our CUDA"])
    ax.set_ylabel("Power (W)")
    ax.set_ylim(0, 180)
    ax.set_title("Energy sensor cross-check failed; efficiency unproven")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=7)
    fig.text(.5, .005, "3 fixed-work batches per method, Performance on AC. Power trace spans warmup + measurement; dots are batches.", ha="center", fontsize=6.7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "energy_qc", rows)


def shapes():
    rows = list(csv.DictReader((ROOT / "figures/mtile_training_grid.csv").open()))
    values = np.asarray([float(row["time_saved_pct"]) for row in rows]).reshape(7, 6)
    cmap = LinearSegmentedColormap.from_list("okabe_diverging", [ORANGE, "#FFFFFF", BLUE])
    fig, ax = plt.subplots(figsize=(6.75, 3.9))
    plot = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=25), aspect="auto")
    for i in range(7):
        for j in range(6):
            ax.text(j, i, f"{values[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i,j] < -75 or values[i,j] > 18 else "black")
    ax.set_xticks(range(6), ["M=1", "2", "4", "8", "16", "64"])
    ax.set_yticks(range(7), [f"{rows[i*6]['projection'].split('.')[-2]}\n{rows[i*6]['n']}×{rows[i*6]['k']}" for i in range(7)], fontsize=7)
    ax.set_title("Raw revision 2 CUDA: latency saved vs best vendor (%)")
    fig.colorbar(plot, ax=ax, extend="min", fraction=.04, pad=.03)
    fig.text(.5, .005, "Training layer 0, BF16 output; no dispatch. Colors saturate below −100%; numeric cells retain all slowdowns.", ha="center", fontsize=6.7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "shapes", rows)


if __name__ == "__main__":
    profiles()
    nwc()
    energy_qc()
    shapes()
    print("Saved research_{profiles,nwc,energy_qc,shapes}.{png,pdf,csv}")
