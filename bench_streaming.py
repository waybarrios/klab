"""Rotating-weight FP32 benchmark with A/A control and longer timing intervals."""

import hashlib
import json
import math
from pathlib import Path

from experiment_control import report_output,with_run_metadata,write_json
from result_metrics import percentage_gains

ROOT=Path(__file__).resolve().parent


def interval(baseline,candidate):
    import numpy as np
    b,c=np.array(baseline),np.array(candidate)
    if b.size!=c.size or b.size<7:
        raise ValueError("At least seven paired timing rounds required")
    ids=np.random.default_rng(997).integers(0,b.size,size=(2000,b.size))
    draws=100*(1-np.median(c[ids],axis=1)/np.median(b[ids],axis=1))
    return [float(v) for v in np.percentile(draws,[2.5,97.5])]


def run(shapes,seeds,rounds,output,profile,run_metadata=None):
    import torch
    import kernel_lab
    import lossless_weights
    import packed_fp32
    from bench import check,measure
    from cublas_baseline import CuBlas
    from gpu_monitor import GpuMonitor
    from experiment_provenance import environment,source_hashes
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    lab,vendor=kernel_lab.default_lab(),CuBlas()
    l2=int(torch.cuda.get_device_properties(0).L2_cache_size)
    choices={tuple(c["shape_mnk"]):c["selected"] for c in json.loads(profile.read_text())["cases"]}
    report=with_run_metadata({"kind":"streaming-projection","status":"running","cases":[],"seeds":list(seeds),
            "profile":str(profile.relative_to(ROOT)),"rounds":rounds,"l2_bytes":l2,
            "torch":torch.__version__,"cuda":torch.version.cuda,
            "profile_sha256":hashlib.sha256(profile.read_bytes()).hexdigest(),
            "precision":"Original FP32 weights; FP32 activations, accumulation, output. TF32 disabled.",
            "protocol":"Cycle through >=16x L2 of weights; time entire CUDA graph; divide by real call count. No eviction kernel.",
            "source_sha256":source_hashes(("bench_streaming.py",str(profile.relative_to(ROOT)))),
            "environment":environment(lab,vendor),"schema_version":2,
            "parameters":{"shapes":[list(s) for s in shapes],"rounds":rounds,"ring_target_bytes":max(80*1024**2,l2*20)}},run_metadata)
    for seed in seeds:
        for m,n,k in shapes:
            selected=choices.get((m,n,k),"warp_t64_v1" if k==768 else "warp_t32_v4")
            copies=max(2,math.ceil(max(80*1024**2,l2*20)/(n*k*4)))
            calls=copies*max(1,math.ceil(32/copies))
            torch.manual_seed(seed)
            weights=torch.randn(copies,n,k,device="cuda")
            x=torch.randn(m,k,device="cuda")/math.sqrt(k)
            wt=weights.transpose(1,2).contiguous()
            all_old=lossless_weights.pack_weights(weights.reshape(copies*n,k))
            old=[lossless_weights.LosslessWeights(all_old.mantissa[i*n:(i+1)*n],all_old.exponents,
                                                all_old.metadata[i*n:(i+1)*n]) for i in range(copies)]
            exact=torch.equal(all_old.unpack().view(torch.int32),weights.reshape(copies*n,k).view(torch.int32))
            packed=None
            if selected.startswith("fp28_"):
                all_packed=packed_fp32.pack_weights(weights.reshape(copies*n,k))
                exact=exact and torch.equal(all_packed.unpack().view(torch.int32),weights.reshape(copies*n,k).view(torch.int32))
                packed=[packed_fp32.PackedFP32Weights(all_packed.payload,all_packed.metadata[i*n:(i+1)*n]) for i in range(copies)]
            if not exact: raise AssertionError("Packed weight reconstruction failed")
            ws,tw=tuple(weights.unbind()),tuple(wt.unbind())
            functions,outputs={},{}
            def add(name,fn):
                outputs[name]=torch.empty(copies,m,n,device="cuda")
                outs=tuple(outputs[name].unbind())
                def cycle():
                    for j in range(calls):
                        i=j%copies
                        fn(i,outs[i])
                functions[name]=cycle
            add("torch_mm",lambda i,o:torch.mm(x,ws[i].T,out=o))
            add("cublas_mm",lambda i,o:vendor.mm(x,ws[i],o))
            add("cublas_packed_mm",lambda i,o:vendor.mm_packed(x,tw[i],o))
            if m==1:
                add("torch_mv",lambda i,o:torch.mv(ws[i],x[0],out=o[0]).view(1,n))
                add("cublas_mv",lambda i,o:vendor.mv(x,ws[i],o))
                add("cublas_packed_mv",lambda i,o:vendor.mv_packed(x,tw[i],o))
            vendors=list(functions)
            add("torch_mm_aa",lambda i,o:torch.mm(x,ws[i].T,out=o))
            parts=selected.split("_")
            threads=int(parts[1][1:])
            if selected.startswith("tiny_"):
                vector=int(parts[2][1:])
                add("candidate",lambda i,o:lab.tiny_matmul(x,ws[i],threads,vector,o))
            elif selected.startswith("warp_"):
                vector=int(parts[2][1:])
                add("candidate",lambda i,o:lab.warp_matmul(x,ws[i],threads,vector,o))
            elif selected.startswith("lossless_"):
                add("candidate",lambda i,o:lossless_weights.fused_linear(x,old[i],threads,o))
            elif selected.startswith("fp28_"):
                add("candidate",lambda i,o:packed_fp32.fused_linear(x,packed[i],threads,o))
            else: raise ValueError("Unknown selected configuration")
            add("forced_lossless",lambda i,o:lossless_weights.fused_linear(x,old[i],128,o))
            with GpuMonitor() as telemetry:
                timing=measure(functions,rounds=rounds,repeats=1)
            for stats in timing.values():
                for key in ("median_us","min_us","max_us"): stats[key]/=calls
                stats["samples_us"]=[v/calls for v in stats["samples_us"]]
            ref=x.double().unsqueeze(0)@weights.double().transpose(1,2)
            methods={}
            for name,values in timing.items():
                check(outputs[name],ref.float())
                error=((outputs[name].double()-ref).norm()/ref.norm()).item()
                methods[name]={**values,"output_relative_l2_error":error,"meets_precision_requirement":True}
            best=min(vendors,key=lambda name:timing[name]["median_us"])
            base=timing[best]
            gains=percentage_gains(base["median_us"],timing["candidate"]["median_us"])
            ci=interval(base["samples_us"],timing["candidate"]["samples_us"])
            aa=percentage_gains(timing["torch_mm"]["median_us"],timing["torch_mm_aa"]["median_us"])["latency_reduction_pct"]
            accepted=abs(aa)<=1 and ci[0]>max(1,abs(aa))
            case={"shape_mnk":[m,n,k],"seed":seed,"selected":selected,"copies":copies,"calls_per_graph":calls,
                  "raw_ring_bytes":weights.numel()*4,"lossless_ring_bytes":all_old.storage_bytes,
                  "best_vendor":best,"methods":methods,"aa_time_saved_pct":aa,
                  "candidate_time_saved_pct":gains["latency_reduction_pct"],"candidate_gain_ci95_pct":ci,
                  "accepted_custom_win":accepted,"bitwise_round_trip":exact,"gpu_telemetry":telemetry.samples}
            case["accepted_10pct_target"]=accepted and ci[0]>10
            report["cases"].append(case)
            write_json(output,report)
            print(f"{m}x{n}x{k} seed={seed}: {selected}, time saved {gains['latency_reduction_pct']:+.2f}% "
                  f"CI [{ci[0]:+.2f},{ci[1]:+.2f}]%; A/A {aa:+.2f}%; accepted={accepted}",flush=True)
    vendor.close()
    report["status"]="complete"
    write_json(output,report)
    return report


def make_report(data,source,destination=None):
    destination=report_output(ROOT,source,destination)
    lines=["# Rotating-weight FP32 validation","",data["protocol"],"",
           "Frozen choices are validated on fresh input values. Device timings exclude host dispatch and setup.",
           "The same original weights and x are used by every implementation in each case.","",
           "| M × N × K | Seed | Kernel | Time saved (%) | 95% bootstrap interval (%) | A/A (%) | Noise gate passed | >10% target |",
           "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |"]
    for c in data["cases"]:
        lo,hi=c["candidate_gain_ci95_pct"]
        lines.append("| "+" × ".join(map(str,c["shape_mnk"]))+f" | {c['seed']} | {c['selected']} | "
                     f"{c['candidate_time_saved_pct']:+.2f} | {lo:+.2f} to {hi:+.2f} | {c['aa_time_saved_pct']:+.2f} | "
                     f"{'Yes' if c['accepted_custom_win'] else 'No'} | {'Yes' if c.get('accepted_10pct_target',False) else 'No'} |")
    lines += ["","Acceptance requires an identical-operation A/A median within 1%, and a paired bootstrap lower bound above the larger of 1% and observed A/A spread.",
              "The intervals quantify variation among rounds in this process; independent process replication is still needed.",
              "All ring weights were checked bitwise after unpacking, and every captured output was checked against FP64.",
              "GPU telemetry was sampled during timing. This is stock-GPU validation; gpu-unlock was not loaded.","",
              f"Raw run: [{source.name}]({source.name}).", ""]
    destination.write_text("\n".join(lines))
    return destination
