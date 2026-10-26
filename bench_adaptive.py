"""Tune custom FP32 projections once, then validate frozen choices on new inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiment_control import GpuLock, parse_shape, project_path, report_output, with_run_metadata, write_json
from result_metrics import percentage_gains

ROOT = Path(__file__).resolve().parent
SHAPES = [(1,64,64),(1,128,128),(1,256,256),(1,512,512),(1,1024,1024),(4,256,256)]


def build_functions(x,w,lab,vendor,selected=None):
    import torch
    import lossless_weights
    import packed_fp32
    m,k=x.shape
    n=w.shape[0]
    wt=w.T.contiguous()
    old=lossless_weights.pack_weights(w)
    packed=packed_fp32.pack_weights(w)
    assert torch.equal(packed.unpack().view(torch.int32),w.view(torch.int32))
    assert torch.equal(old.unpack().view(torch.int32),w.view(torch.int32))
    outputs={}
    functions={}
    def add(name,fn):
        out=torch.empty(m,n,device="cuda")
        outputs[name]=out
        functions[name]=lambda: fn(out)
    add("torch_mm",lambda out: torch.mm(x,w.T,out=out))
    add("cublas_mm",lambda out: vendor.mm(x,w,out))
    add("cublas_packed_mm",lambda out: vendor.mm_packed(x,wt,out))
    if m==1:
        add("torch_mv",lambda out: torch.mv(w,x[0],out=out[0]).view(1,n))
        add("cublas_mv",lambda out: vendor.mv(x,w,out))
        add("cublas_packed_mv",lambda out: vendor.mv_packed(x,wt,out))
    vendors=list(functions)
    for t in (32,64,128,256):
        for v in (1,4):
            name=f"warp_t{t}_v{v}"
            if selected is None or name==selected:
                add(name,lambda out,t=t,v=v: lab.warp_matmul(x,w,t,v,out))
            name=f"tiny_t{t}_v{v}"
            if k in (64,128,256,512,1024) and (selected is None or name==selected):
                add(name,lambda out,t=t,v=v: lab.tiny_matmul(x,w,t,v,out))
        name=f"lossless_t{t}"
        if selected is None or name==selected or t==128:
            add(name,lambda out,t=t: lossless_weights.fused_linear(x,old,t,out))
        name=f"fp28_t{t}"
        if selected is None or name==selected:
            add(name,lambda out,t=t: packed_fp32.fused_linear(x,packed,t,out))
    return functions,outputs,vendors


def run(shapes,seeds,rounds,output,frozen_from=None,run_metadata=None):
    import torch
    import kernel_lab
    from bench import check,measure,gpu_state
    from cublas_baseline import CuBlas
    from validate_challenge import cold_measure
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    lab,vendor=kernel_lab.default_lab(),CuBlas()
    frozen=({tuple(c["shape_mnk"]):c["selected"] for c in json.loads(project_path(ROOT,frozen_from).read_text())["cases"]}
            if frozen_from else {})
    if frozen_from and any(tuple(shape) not in frozen for shape in shapes):
        raise ValueError("Every validation shape needs a frozen selection")
    report={"status":"running","kind":"adaptive-fp32","cases":[],"seeds":list(seeds),
            "selection_rule":"maximize minimum warm/cold speed relative to each condition's fastest FP32 vendor",
            "selection_seed":901,"frozen_from":str(frozen_from) if frozen_from else None,
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                              ("kernels.cu","adaptive_kernels.cu","lossless_kernels.cu","packed_fp32.py",
                               "bench_adaptive.py","kernel_lab.py","native/driver.cpp")}}
    report=with_run_metadata(report,run_metadata)
    for seed in seeds:
        for m,n,k in shapes:
            torch.manual_seed(seed)
            x=torch.randn(m,k,device="cuda")/math.sqrt(k)
            w=torch.randn(n,k,device="cuda")
            functions,outputs,vendors=build_functions(x,w,lab,vendor,frozen.get((m,n,k)))
            ref=x.double()@w.double().T
            for fn in functions.values(): check(fn(),ref.float())
            warm=measure(functions,rounds=max(7,rounds//2),repeats=64)
            cold=cold_measure(functions,rounds=rounds)
            timing={"warm":warm,"cold":cold}
            baseline={c:min(vendors,key=lambda name:timing[c][name]["median_us"]) for c in timing}
            candidates=[name for name in functions if name not in vendors]
            scores={name:min(timing[c][baseline[c]]["median_us"]/timing[c][name]["median_us"] for c in timing)
                    for name in candidates}
            selected=frozen.get((m,n,k)) or max(scores,key=scores.get)
            methods={}
            for name in functions:
                check(outputs[name],ref.float())
                error=((outputs[name].double()-ref).norm()/ref.norm()).item()
                methods[name]={"warm":warm[name],"cold":cold[name],"output_relative_l2_error":error,
                               "percentages":{c:percentage_gains(timing[c][baseline[c]]["median_us"],timing[c][name]["median_us"])
                                              for c in timing}}
            case={"shape_mnk":[m,n,k],"seed":seed,"selected":selected,"selection_scores":scores,
                  "best_vendor_method":baseline["cold"],"best_vendor_us":cold[baseline["cold"]]["median_us"],
                  "baseline_by_condition":baseline,"methods":methods,"bitwise_round_trip":True,
                  "gpu_after_case":gpu_state()}
            report["cases"].append(case)
            write_json(output,report)
            gains=methods[selected]["percentages"]
            print(f"{m}x{n}x{k} seed={seed}: {selected}, time saved warm={gains['warm']['latency_reduction_pct']:+.2f}% "
                  f"cold={gains['cold']['latency_reduction_pct']:+.2f}%",flush=True)
    vendor.close()
    report["status"]="complete"
    write_json(output,report)
    return report


def make_report(report,source,destination=None):
    from collections import defaultdict
    destination=report_output(ROOT,source,destination)
    groups=defaultdict(list)
    for case in report["cases"]: groups[tuple(case["shape_mnk"])].append(case)
    lines=["# Adaptive CUDA candidate experiment","",
           "Original FP32 weights, FP32 inputs/accumulation/output; no quantized weights.",
           "These are experimental candidates, not a validated no-regression dispatcher.",
           f"Frozen selection source: `{report.get('frozen_from') or 'none: tuning-only results'}`.","",
           "| M × N × K | Selected kernel | Resident time saved (%) | Eviction time saved (%) |",
           "| --- | --- | ---: | ---: |"]
    for shape,cases in groups.items():
        values={condition:[c["methods"][c["selected"]]["percentages"][condition]["latency_reduction_pct"]
                           for c in cases] for condition in ("warm","cold")}
        lines.append("| "+" × ".join(map(str,shape))+f" | {cases[0]['selected']} | "+
                     " | ".join(f"{min(values[c]):+.2f} to {max(values[c]):+.2f}" for c in values)+" |")
    lines += ["","Each cache condition uses its own fastest tested FP32 vendor baseline.",
              "Negative values are losses. One timer tick can be a large percentage for tiny tensors.",
              "Tuning results are not held-out validation. A/A and rotating-weight measurements are required before promotion.",
              "All original timing samples and FP64 numerical comparisons are retained in the raw run.","",
              f"Raw run: [{source.name}]({source.name}).", ""]
    destination.write_text("\n".join(lines))
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",type=parse_shape,action="append")
    p.add_argument("--seeds",default="901")
    p.add_argument("--rounds",type=int,default=15)
    p.add_argument("--frozen-from")
    p.add_argument("--output",default="results/adaptive-search.json")
    a=p.parse_args()
    shapes=a.shape or SHAPES
    if a.rounds<3 or any(k%64 for _,_,k in shapes): p.error("rounds>=3 and K divisible by 64 required")
    with GpuLock(ROOT/".cache/gpu.lock"):
        run(shapes,[int(s) for s in a.seeds.split(",")],a.rounds,project_path(ROOT,a.output),a.frozen_from)


if __name__=="__main__":
    main()
