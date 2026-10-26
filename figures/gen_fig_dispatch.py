"""Paper-style plot/table of measured prepared-vendor dispatch overhead."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from bench_dispatch import summarize_dispatch

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    args=parser.parse_args()
    source=(ROOT/args.source).resolve()
    if not source.is_relative_to(ROOT):
        parser.error("Source must stay in the project")
    data=json.loads(source.read_text())
    if data.get("status")!="complete" or len(data.get("cases",[]))!=12:
        parser.error("Need the complete 12-case vendor-fallback study")
    rows=[]
    for case in data["cases"]:
        for scope in ("graph","eager"):
            result=case[scope]
            recomputed=summarize_dispatch(result["methods"])
            if not np.allclose(recomputed["ci95_pct"],result["ci95_pct"]):
                raise ValueError("Stored interval does not reproduce from raw rounds")
            rows.append({"case":case["label"],"shape_mnk":str(case["shape_mnk"]),
                "output_dtype":case["output_dtype"],"scope":scope,
                "vendor_us":result["methods"]["vendor"]["median_us"],
                "prepared_us":result["methods"]["prepared"]["median_us"],
                "time_saved_pct":result["time_saved_pct"],"ci_low_pct":result["ci95_pct"][0],
                "ci_high_pct":result["ci95_pct"][1],"aa_pct":result["aa_time_saved_pct"],
                "overhead_gate_passed":result["overhead_gate_passed"],
                "context_stable":case["operating_context_stable"] if scope=="graph" else
                    case["operating_context_stable"] and case["eager_operating_context_stable"]})
    stem=ROOT/"figures/prepared_dispatch"
    with stem.with_suffix(".csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,
        "ps.fonttype":42,"savefig.dpi":300})
    fig,axes=plt.subplots(1,2,figsize=(9,5.8),sharey=True)
    labels=[f"{case['label']} ({case['output_dtype']} out)" for case in data["cases"]]
    colors={"bf16":"#0072B2","fp32":"#D55E00"}
    for axis,scope in zip(axes,("graph","eager")):
        axis.axvline(0,color="0.35",linewidth=.8)
        axis.axvline(-1,color="0.65",linewidth=.7,linestyle=":")
        subset=[row for row in rows if row["scope"]==scope]
        for i,row in enumerate(subset):
            point=row["time_saved_pct"]
            axis.hlines(i,row["ci_low_pct"],row["ci_high_pct"],color=colors[row["output_dtype"]],linewidth=1.3)
            axis.plot(point,i,"o",color=colors[row["output_dtype"]],markersize=4)
            if not row["overhead_gate_passed"] or not row["context_stable"]:
                axis.plot(point,i,"x",color="black",markersize=7,markeredgewidth=1)
        axis.set_title("CUDA graph replay" if scope=="graph" else "Synchronized eager, allocation included")
        axis.set_xlabel("Latency saved vs direct vendor (%)")
        axis.grid(axis="x",alpha=.18,linewidth=.5)
        axis.set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    fig.suptitle("Prepared vendor fallback overhead",fontsize=12,y=.98)
    fig.text(.5,.923,"Native BF16 weights · FP32 accumulation · 31 paired rounds · stock GPU",ha="center",fontsize=8)
    fig.legend(handles=[Line2D([],[],color=colors["bf16"],marker="o",linestyle="",label="BF16 output"),
        Line2D([],[],color=colors["fp32"],marker="o",linestyle="",label="FP32 output"),
        Line2D([],[],color="black",marker="x",linestyle="",label="Overhead/context gate not passed")],
        loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=8)
    fig.text(.5,.085,"Separate horizontal scales. Vendor fallback is not a custom-kernel speedup.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.285,right=.985,top=.865,bottom=.19,wspace=.12)
    fig.savefig(stem.with_suffix(".png"),dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    lines=["# Prepared vendor fallback timing","",
        "Stock GPU, Performance on AC. BF16 input/weights, FP32 accumulation. Graph uses checked `linear_out`; eager uses allocation-only `linear`. No fallback is labelled a custom-kernel win.","",
        "| Case | Output | Scope | Vendor us | Prepared us | Latency saved | 95% CI | A/A | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for row in rows:
        passed=row["overhead_gate_passed"] and row["context_stable"]
        lines.append(f"| {row['case']} | {row['output_dtype']} | {row['scope']} | {row['vendor_us']:.3f} | "
            f"{row['prepared_us']:.3f} | {row['time_saved_pct']:+.3f}% | [{row['ci_low_pct']:+.3f}, "
            f"{row['ci_high_pct']:+.3f}]% | {row['aa_pct']:+.3f}% | {'pass' if passed else 'fail'} |")
    lines += ["",f"Raw evidence: [{source.name}]({source.name}). All failed controls remain visible.",
        "The initial wrapper run is preserved in `prepared-small-vendor-performance-r1.json`; the revised run removes repeated dtype/output branching and boxed dispatch, and samples GPU telemetry outside synchronized eager timing."]
    source.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print("Wrote prepared_dispatch PNG/PDF/CSV and complete result table")


if __name__=="__main__":
    main()
