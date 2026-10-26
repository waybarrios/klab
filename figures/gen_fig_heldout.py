"""Paper-style primary-case intervals and complete held-out shape summary."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    import numpy as np

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--prefix",default="heldout_bf16_performance")
    args=parser.parse_args()
    source=(ROOT/args.source).resolve()
    if not source.is_relative_to(ROOT) or "/" in args.prefix or "\\" in args.prefix:
        parser.error("Use a project source and a filename prefix")
    data=json.loads(source.read_text())
    if data.get("kind")!="compressed-dispatch-evidence" or data.get("status")!="complete":
        parser.error("Need complete three-process raw-kernel evidence")
    seeds=sorted(process["seed"] for process in data["processes"])
    if len(seeds)!=3:
        parser.error("Need all three fresh processes")
    primary_keys=set(data["primary_shapes"])
    primary=[case for case in data["cases"] if ",".join(map(str,case["shape_mnk"])) in primary_keys]
    context=data["bindings"]["operator_context"]
    profile=context["system76_power_profile"].removeprefix("Power Profile: ")
    power="AC online" if context["ac_online"] else "AC offline"
    clocks=", ".join(data["observed_memory_clocks"])
    identity=data["bindings"]["identity"]
    gpu=identity["gpu"].split(",")[0]
    unlock=identity["unlock_evidence"]
    state="stock" if unlock and all(value is False for value in unlock.values()) else "unlock state: see source"
    conditions=f"{profile} profile · {power} · {state} · {gpu} · memory {clocks}"
    order={"gate_proj":0,"up_proj":1,"down_proj":2}
    def key(case):
        return case["layer"],order[case["weight_tensor"].split(".")[-2]]
    rows=sorted({key(case) for case in primary})
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,
        "ps.fonttype":42,"savefig.dpi":300})
    stem=ROOT/"figures"/args.prefix
    fig,axes=plt.subplots(1,3,figsize=(9.5,4.9),sharex=True,sharey=True)
    exported=[]
    for axis,seed in zip(axes,seeds):
        cases=sorted((case for case in primary if case["seed"]==seed),key=key)
        if [key(case) for case in cases]!=rows:
            raise ValueError("Primary layers/projections differ across processes")
        for i,case in enumerate(cases):
            point=case["time_saved_pct"]
            low,high=case["ci95_pct"]
            color="#D55E00" if "down_proj" in case["weight_tensor"] else "#0072B2"
            axis.hlines(i,low,high,color=color,linewidth=1.4)
            axis.plot(point,i,"o",color=color,markersize=4)
            if not case["kernel_gate_passed"]:
                axis.plot(point,i,"x",color="black",markersize=7)
            exported.append({"seed":seed,"layer":case["layer"],"weight":case["weight_tensor"],
                "shape_mnk":str(case["shape_mnk"]),"selected":case["selected"],
                "vendor_us":case["best_vendor_us"],"custom_us":case["selected_median_us"],
                "time_saved_pct":point,"ci_low_pct":low,"ci_high_pct":high,
                "aa_pct":case["aa_time_saved_pct"],"kernel_gate_passed":case["kernel_gate_passed"]})
        axis.axvline(10,color="0.4",linestyle="--",linewidth=.8)
        axis.set_xlim(8,23)
        axis.set_xticks((10,15,20))
        passed=sum(case["kernel_gate_passed"] for case in cases)
        axis.set_title(f"Seed {seed}: {passed}/{len(cases)} cases pass")
        axis.set_yticks(range(len(rows)),[f"Layer {layer}  {('gate','up','down')[projection]}"
                                        for layer,projection in rows])
        axis.grid(axis="x",alpha=.18,linewidth=.5)
    axes[0].invert_yaxis()
    contract=data["contract"]
    output=contract["output_dtype"].upper()
    fig.suptitle("Held-out large M=1 projections",fontsize=12,y=.975)
    fig.text(.5,.918,f"Native BF16 input/weights · FP32 accumulation · {output} output · 31 paired rounds/case",
             ha="center",fontsize=8)
    fig.supxlabel("GPU latency saved vs fastest eligible tested vendor (%)",y=.135,fontsize=9)
    fig.text(.5,.065,"Bars: 95% paired bootstrap intervals. Dashed line: >10% target. Black x: failed kernel gate.",ha="center",fontsize=8)
    fig.text(.5,.025,conditions,ha="center",fontsize=7.5)
    fig.subplots_adjust(left=.14,right=.985,top=.825,bottom=.22,wspace=.12)
    fig.savefig(stem.with_suffix(".png"),dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    write_csv(stem.with_suffix(".csv"),exported)

    # Every shape, including unqualified positive points and large slowdowns.
    ms=sorted({case["shape_mnk"][0] for case in data["cases"]})
    nks=sorted({tuple(case["shape_mnk"][1:]) for case in data["cases"]},key=lambda nk:(nk[0]*nk[1],nk))
    values=np.empty((len(ms),len(nks)))
    grid=[]
    for i,m in enumerate(ms):
        for j,(n,k) in enumerate(nks):
            shape=data["shapes"][f"{m},{n},{k}"]
            cases=[case for case in data["cases"] if case["shape_mnk"]==[m,n,k]]
            points=[case["time_saved_pct"] for case in cases if case["time_saved_pct"] is not None]
            values[i,j]=np.median(points) if points else np.nan
            grid.append({"m":m,"n":n,"k":k,"median_time_saved_pct":float(values[i,j]),
                "passing_cases":shape["cases"]-shape["failed_cases"],"cases":shape["cases"],
                "all_kernel_gates_passed":shape["kernel_gate_passed"]})
    cmap=LinearSegmentedColormap.from_list("latency",["#D55E00","white","#0072B2"])
    norm=TwoSlopeNorm(vmin=min(-1,float(np.nanmin(values))),vcenter=0,vmax=max(25,float(np.nanmax(values))))
    fig,axis=plt.subplots(figsize=(8.8,5.9))
    shown=axis.imshow(values,cmap=cmap,norm=norm,aspect="auto")
    for row in grid:
        i,j=ms.index(row["m"]),nks.index((row["n"],row["k"]))
        marker="*" if row["all_kernel_gates_passed"] else ""
        axis.text(j,i,f"{row['median_time_saved_pct']:+.1f}%{marker}\n{row['passing_cases']}/{row['cases']}",
                  ha="center",va="center",fontsize=8,color="black")
    axis.set_xticks(range(len(nks)),[f"N={n}\nK={k}" for n,k in nks])
    axis.set_yticks(range(len(ms)),[f"M={m}" for m in ms])
    axis.set_title("Compressed kernel before fallback: complete held-out grid",pad=14,fontsize=11)
    ticks=[value for value in (-300,-200,-100,-50,0,10,20) if norm.vmin<=value<=norm.vmax]
    fig.colorbar(shown,ax=axis,pad=.025,label="Median GPU latency saved (%)",ticks=ticks)
    fig.text(.5,.075,"* Every registered case passes all kernel gates; counts show passing/total cases.",ha="center",fontsize=8)
    fig.text(.5,.035,"Positive medians alone do not qualify. Prepared dispatch and eager promotion require separate evidence.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.09,right=.94,top=.88,bottom=.19)
    full=stem.with_name(stem.name+"_grid")
    fig.savefig(full.with_suffix(".png"),dpi=300)
    fig.savefig(full.with_suffix(".pdf"))
    plt.close(fig)
    write_csv(full.with_suffix(".csv"),grid)
    def number(value):
        return "n/a" if value is None else f"{value:+.3f}%"
    lines=["# Held-out compressed BF16 kernel evidence","",
        f"Primary large-M1 kernel milestone: **{'accepted' if data['kernel_milestone_passed'] else 'not accepted'}**.","",
        f"{conditions}. Native BF16 inputs/weights, FP32 accumulation, {output} output. "
        f"Three fresh processes, {len(data['cases'])} registered cases and 31 paired rounds per case.","",
        ("This accepts GPU kernel performance in the recorded regime. " if data["kernel_milestone_passed"] else
         "The combined primary milestone is not accepted; failed cases remain visible. ")+
        "Prepared dispatch, eager calls, other output/profile contracts and full-model performance require their own evidence.","",
        "| Seed | Primary cases | Latency saved range | Lowest per-case CI lower bound |",
        "| --- | ---: | ---: | ---: |"]
    for seed in seeds:
        cases=[case for case in primary if case["seed"]==seed]
        points=[case["time_saved_pct"] for case in cases if case["time_saved_pct"] is not None]
        lows=[case["ci95_pct"][0] for case in cases if case["ci95_pct"] is not None]
        gain=f"{min(points):.3f}–{max(points):.3f}%" if points else "n/a"
        lines.append(f"| {seed} | {sum(case['kernel_gate_passed'] for case in cases)}/{len(cases)} | {gain} | {number(min(lows) if lows else None)} |")
    lines += ["","Every shape is retained below. Medians are descriptive; acceptance requires every registered case to pass, including A/A, numerics, exact bits/parity and storage.","",
        "| M,N,K | Median saved | Passing cases | All kernel gates |",
        "| --- | ---: | ---: | --- |"]
    for row in grid:
        lines.append(f"| {row['m']},{row['n']},{row['k']} | {row['median_time_saved_pct']:+.3f}% | "
            f"{row['passing_cases']}/{row['cases']} | {'pass' if row['all_kernel_gates_passed'] else 'fail'} |")
    lines += ["",f"Raw aggregate: [{source.name}]({source.name}). Input hashes link all three raw reports and frozen training.",
        f"Plots: [primary PNG](../figures/{stem.name}.png), [primary PDF](../figures/{stem.name}.pdf), "
        f"[complete-grid PNG](../figures/{full.name}.png), [complete-grid PDF](../figures/{full.name}.pdf)."]
    source.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print(f"Wrote primary and complete-grid PNG/PDF/CSV from {source.name}")


if __name__=="__main__":
    main()
