"""Regenerate preliminary plots: .venv/bin/python -m figures.gen_fig_preliminary."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiment_control import project_path
from result_metrics import case_metrics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"


def load_rows(path):
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or not report.get("cases"):
        raise ValueError(f"Expected a completed comparison: {path}")
    rows = []
    for case in report["cases"]:
        if not case["bitwise_round_trip"]:
            raise ValueError("Lossless round trip failed")
        for condition in ("cold", "warm"):
            rows.append({"source": str(path.relative_to(ROOT)), "condition": condition,
                         "seed": case["seed"], "shape": tuple(case["shape_mnk"]),
                         "exact_weights": True,
                         "output_relative_l2_error": case["methods"]["lossless_fp32"]["output_relative_l2_error"],
                         **case_metrics(case, condition)})
    return rows


def grouped(rows, condition):
    groups = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            groups[row["shape"]].append(row)
    return groups


def labels(groups):
    return [" × ".join(f"{v:,}" for v in shape) for shape in groups]


def bars(ax, positions, groups, field, width, color, label, scale=1, percentage=False):
    samples = [[row[field] / scale for row in group] for group in groups.values()]
    centers = np.array([mean(values) for values in samples])
    low = np.array([min(values) for values in samples])
    high = np.array([max(values) for values in samples])
    errors = np.maximum(np.vstack((centers - low, high - centers)), 0)
    artists = ax.bar(positions, centers, width=width, color=color, label=label,
                     yerr=errors, capsize=3, error_kw={"linewidth": 1}, zorder=3)
    for artist, value, lo, hi in zip(artists, centers, low, high):
        if percentage and value < 0:
            artist.set_hatch("//")
        ax.annotate(f"{value:+.1f}%" if percentage else f"{value:.3f}",
                    (artist.get_x() + artist.get_width() / 2, hi if value >= 0 else lo),
                    xytext=(0, 7 if value >= 0 else -9), textcoords="offset points",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.margins(y=0.25)


def save(fig, directory, name):
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        print(path.relative_to(ROOT))
    plt.close(fig)


def plot_large(rows, directory):
    groups = grouped(rows, "cold")
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    fig.suptitle("Preliminary FP32 projections: large tensors", fontsize=15, fontweight="bold")
    bars(axes[0], x - 0.19, groups, "baseline_us", 0.35, BLUE, "Best FP32 vendor", scale=1000)
    bars(axes[0], x + 0.19, groups, "kernel_us", 0.35, GREEN, "Lossless FP32", scale=1000)
    axes[0].set_ylabel("GPU operation time (ms, lower is better)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    bars(axes[1], x - 0.19, groups, "latency_reduction_pct", 0.35, GREEN, "Time saved", percentage=True)
    bars(axes[1], x + 0.19, groups, "throughput_gain_pct", 0.35, BLUE, "Throughput gain", percentage=True)
    axes[1].set_ylabel("Improvement over best FP32 vendor (%)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xticks(x, labels(groups), fontsize=8)
        ax.set_xlabel("M × N × K")
    fig.text(0.5, 0.025, "RTX 3070 Ti Laptop | 2 seeds; mean of per-seed medians; whiskers = seed range, not a confidence interval.\n"
             "64 MiB eviction-buffer condition; eviction, packing, compilation, and Python overhead excluded.",
             ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(top=0.83, bottom=0.22, wspace=0.25)
    save(fig, directory, "fig_preliminary_large")


def plot_small(rows, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
    fig.suptitle("Preliminary FP32 projections: smaller tensors", fontsize=15, fontweight="bold")
    for ax, condition, title in zip(axes, ("warm", "cold"),
                                    ("Resident-weight replay", "After 64 MiB eviction buffer")):
        groups = grouped(rows, condition)
        x = np.arange(len(groups))
        bars(ax, x, groups, "latency_reduction_pct", 0.6, GREEN, "Lossless FP32", percentage=True)
        ax.axhline(0, color="#333333", linewidth=0.9, zorder=4)
        ax.set_title(title)
        ax.set_xticks(x, labels(groups), rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("M × N × K")
    axes[0].set_ylabel("Time saved vs best FP32 vendor (%)")
    fig.text(0.5, 0.025, "Positive = less time; negative / hatched = slower. Each condition selects its own fastest FP32 vendor baseline.\n"
             "2 seeds; 2–4 cases/shape; bars = mean of paired percentages; whiskers = observed range. Setup/eviction excluded.",
             ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(top=0.83, bottom=0.34, wspace=0.12)
    save(fig, directory, "fig_preliminary_small")


def range_text(values, decimals=2):
    low, high = min(values), max(values)
    return f"{low:.{decimals}f}" if round(low, decimals) == round(high, decimals) else f"{low:.{decimals}f} to {high:.{decimals}f}"


def write_summary(large, small, directory):
    lines = ["# Preliminary GPU results in percentages", "",
             "Time saved = `100 × (baseline_time - kernel_time) / baseline_time`.", "",
             "Throughput gain = `100 × (baseline_time / kernel_time - 1)` for repeated independent calls.", "",
             "For example, 100 µs becoming 80 µs saves 20% of execution time and gives 25% more calls per second.",
             "Negative percentages indicate a regression. These are kernel measurements, not end-to-end LLM gains.", ""]
    for title, rows, condition in (("Large tensors: eviction-buffer condition", large, "cold"),
                                   ("Smaller tensors: eviction-buffer condition", small, "cold"),
                                   ("Smaller tensors: resident-weight replay", small, "warm")):
        lines += [f"## {title}", "",
                  "| M × N × K | Best FP32 vendor (µs) | Lossless (µs) | Time saved (%) | Throughput gain (%) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for shape, group in grouped(rows, condition).items():
            values = [range_text([r[field] for r in group]) for field in
                      ("baseline_us", "kernel_us", "latency_reduction_pct", "throughput_gain_pct")]
            lines.append(f"| {' × '.join(map(str, shape))} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["Ranges span seeds 857 and 859, using 15 rounds per cold case. They are not confidence intervals.",
              "Large shapes have 2 cases each. Small shapes have 2 cases each, plus a Python repeat for 64 × 64 and 256 × 256 (4 cases each).",
              "Each percentage uses the fastest FP32 vendor median from its own case and cache condition.",
              "The eviction buffer approximates cold caches; its execution is excluded from the measured interval.", "",
              "Every measured weight reconstructed bit for bit. FP32 arithmetic rounding is checked against FP64.",
              "Packing, compilation, Python dispatch, and allocations are excluded from device latency.",
              "Small timings have visible event-clock granularity and execution noise; these runs establish preliminary observations.",
              "The fixed 128-thread fallback is used for new shapes; small-shape launch configurations were not tuned.", "",
              "At 1024 × 1024, resident replay saves 54.74–72.17% of time, but the eviction-buffer condition takes 4.72–5.51% more time.",
              "That seed-sensitive resident benefit does not establish a gain when streaming weights in a model.", "",
              "Measured on the RTX 3070 Ti Laptop. Memory clocks were about 810 MHz under the existing power/thermal limits.",
              "There is no universal gain: small tensors can lose because decoding work, launch overhead, and cache reuse outweigh saved bytes.", "",
              "[Large-tensor plot](../figures/fig_preliminary_large.png) · [Small-tensor plot](../figures/fig_preliminary_small.png)", "",
              "The Python repeat found 0.00–5.88% time saved at 256 × 256, so small-shape gains are not stable across runs.", "",
              "[Large raw run](cli-reproduction.json) · [Small raw run](small-tensors.json) · "
              "[Python repeat](python-small-example.json) · "
              "[Per-seed plotting data](../figures/preliminary_percentages.csv)", "",
              "Regenerate: `.venv/bin/python -m figures.gen_fig_preliminary`", ""]
    (ROOT / "results/PRELIMINARY_PERCENTAGES.md").write_text("\n".join(lines))
    fields = ["source", "condition", "seed", "shape", "baseline_method", "baseline_us", "kernel_us",
              "latency_reduction_pct", "throughput_gain_pct", "exact_weights", "output_relative_l2_error"]
    with (directory / "preliminary_percentages.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(large + small)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", default="results/cli-reproduction.json")
    parser.add_argument("--small", action="append", help="Repeat to combine small-tensor runs")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    directory = project_path(ROOT, args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    large = load_rows(project_path(ROOT, args.large))
    small_paths = args.small or ["results/small-tensors.json", "results/python-small-example.json"]
    small = [row for path in small_paths for row in load_rows(project_path(ROOT, path))]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "legend.fontsize": 8, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
                         "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_large(large, directory)
    plot_small(small, directory)
    write_summary(large, small, directory)


if __name__ == "__main__":
    main()
