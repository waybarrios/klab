"""Measure prepared vendor overhead separately for graphs and synchronized eager.

The initial small-regression study is a vendor-fallback check, not acceptance of
the custom kernel or the complete Task6 grid. No kernel configuration is tuned.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import sys
import uuid
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent


def current_context(*,scope):
    """Read a power snapshot and bind the correct graph/eager context."""
    from bench_compressed import _power_context
    from power_context import execution_context
    return execution_context(_power_context(),scope=scope)


def summarize_dispatch(methods):
    """Recompute the overhead gate from paired raw rounds, never a cached flag."""
    from bench_compressed import paired_gain_interval
    samples={name:methods[name]["samples_us"] for name in ("vendor","vendor_aa","prepared")}
    lengths={len(values) for values in samples.values()}
    if (len(lengths)!=1 or min(lengths)<31
            or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
                   for values in samples.values() for v in values)):
        raise ValueError("Need at least 31 complete paired positive finite rounds")
    medians={name:statistics.median(values) for name,values in samples.items()}
    aa=100*(1-medians["vendor_aa"]/medians["vendor"])
    interval=paired_gain_interval({"vendor":samples["vendor"]},samples["prepared"])
    return {"time_saved_pct":100*(1-medians["prepared"]/medians["vendor"]),
            "ci95_pct":interval,"aa_time_saved_pct":aa,
            "overhead_gate_passed":bool(abs(aa)<=1 and interval[0]>=-1),
            "custom_win_validated":False,"accepted_10pct_target":False}


def _eager(functions,*,rounds=31,calls=32):
    import torch
    timings={name:[] for name in functions}
    generator=random.Random(2026)
    for fn in functions.values():
        for i in range(8):
            fn(i)
        torch.cuda.synchronize()
    for _ in range(rounds):
        order=list(functions)
        generator.shuffle(order)
        for name in order:
            elapsed=0
            for i in range(calls):
                torch.cuda.synchronize()
                start=time.perf_counter_ns()
                result=functions[name](i)
                torch.cuda.synchronize()
                elapsed+=time.perf_counter_ns()-start
                del result
            timings[name].append(elapsed/calls/1000)
    return {name:{"median_us":statistics.median(values),"samples_us":values}
            for name,values in timings.items()}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_small_fallback(output_path,*,rounds=31):
    import numpy as np
    import torch
    from bench import measure
    from dynamic_projection import prepare_projection
    from experiment_control import project_path,write_json,with_run_metadata
    from gpu_monitor import GpuMonitor
    from gpu_unlock_status import inspect
    from projection_accuracy import check_outputs
    from real_weights import read_tensor

    output_path=project_path(ROOT,output_path)
    if output_path.exists():
        raise ValueError("Preserve historical results; choose a new output path")
    if rounds<31:
        raise ValueError("Need at least 31 rounds")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.manual_seed(901)
    corpus_path=ROOT/".cache/real-weights/manifest.json"
    corpus=json.loads(corpus_path.read_text())
    before=current_context(scope="graph")
    l2=torch.cuda.get_device_properties(0).L2_cache_size
    report=with_run_metadata({"kind":"prepared-bf16-vendor-dispatch","status":"running",
        "scope":"Small-regression vendor-only check; not full-grid custom dispatch validation",
        "cases":[],"rounds":rounds,"seed":901,"inventory":inspect(),
        "operator_context":before,"l2_bytes":l2,"source_sha256":{name:_hash(ROOT/name) for name in
            ("prepared_bf16.py","dynamic_projection.py","projection_policy.py","bench_dispatch.py","bench.py")},
        "corpus_sha256":_hash(corpus_path),"custom_win_validated":False,
        "full_model_speedup_claim":False,"unlocked_comparison":"pending"},
        {"pid":os.getpid(),"run_id":str(uuid.uuid4()),"command":sys.argv,
         "created_utc":datetime.now(timezone.utc).isoformat()})
    write_json(output_path,report)
    try:
        entries=[("synthetic-1x64",None,1,64,64),("synthetic-1x128",None,1,128,128),
                 ("synthetic-1x256",None,1,256,256),("synthetic-4x256",None,4,256,256),
                 ("real-small-k","model.layers.0.self_attn.k_proj.weight",1,128,896),
                 ("real-small-v","model.layers.0.self_attn.v_proj.weight",1,128,896)]
        for output_name in ("bf16","fp32"):
            dtype=torch.bfloat16 if output_name=="bf16" else torch.float32
            profile=ROOT/f"results/compressed-performance-{output_name}-search-r2.json"
            for label,tensor_name,m,n,k in entries:
                started=time.monotonic()
                if tensor_name:
                    bits=np.array(read_tensor(corpus,tensor_name),copy=True)
                    weight=torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
                else:
                    weight=torch.randn((n,k),device="cuda",dtype=torch.bfloat16)
                    bits=weight.view(torch.uint16).cpu().numpy().copy()
                count=max(2,math.ceil(max(80<<20,20*l2)/(n*k*2)))
                weights=[weight]+[weight.clone() for _ in range(count-1)]
                x=torch.randn((m,k),device="cuda",dtype=torch.bfloat16)
                plans=[prepare_projection(w,tokens=m,profile=profile,output_dtype=dtype) for w in weights]
                if not all(p.packed is None and not p.plan["requires_packing"]
                           and not p.plan["custom_win_validated"] for p in plans):
                    raise RuntimeError("Vendor-only study unexpectedly packed or promoted a custom route")
                outputs={arm:[torch.empty((m,n),device="cuda",dtype=dtype) for _ in weights]
                         for arm in ("vendor","vendor_aa","prepared")}
                views=[w.T for w in weights]
                def direct(i,*,out=None):
                    if output_name=="fp32":
                        return torch.mm(x,views[i],out_dtype=torch.float32,out=out)
                    return torch.mm(x,views[i],out=out)
                def sweep(arm):
                    for i in range(count):
                        if arm=="prepared":
                            plans[i].linear_out(x,out=outputs[arm][i])
                        else:
                            direct(i,out=outputs[arm][i])
                reference=x.double()@weight.double().T
                numerics={}
                for arm in outputs:
                    sweep(arm)
                    bit_type=torch.int16 if dtype==torch.bfloat16 else torch.int32
                    if not all(torch.equal(value.view(bit_type),outputs[arm][0].view(bit_type))
                               for value in outputs[arm]):
                        raise RuntimeError("Numerical ring shortcut requires every output bit to match")
                    numerics[arm]=check_outputs(reference,outputs[arm][0],output_dtype=output_name)
                    if not numerics[arm]["passed"]:
                        raise RuntimeError(f"Numerical gate failed: {label} {output_name} {arm} {numerics[arm]}")
                torch.cuda.synchronize()
                eager_before=current_context(scope="eager")
                with GpuMonitor() as monitor:
                    # Each callable traverses distinct original weights >=80 MiB.
                    graph=measure({arm:lambda arm=arm:sweep(arm) for arm in outputs},
                                  rounds=rounds,repeats=1)
                    for method in graph.values():
                        method["samples_us"]=[v/count for v in method["samples_us"]]
                        for key in ("median_us","min_us","max_us"):
                            method[key]/=count
                samples=monitor.samples
                # No nvidia-smi subprocess/driver queries during host timings.
                # Eager allocation and synchronization are included equally.
                if output_name=="fp32":
                    vendor=lambda i:torch.mm(x,views[i],out_dtype=torch.float32)
                    twin=lambda i:torch.mm(x,views[i],out_dtype=torch.float32)
                else:
                    vendor=lambda i:torch.mm(x,views[i])
                    twin=lambda i:torch.mm(x,views[i])
                eager=_eager({"vendor":vendor,"vendor_aa":twin,
                    "prepared":lambda i:plans[i].linear(x)},rounds=rounds,calls=count)
                eager_after=current_context(scope="eager")
                original_bits_preserved=bool(np.array_equal(weight.view(torch.uint16).cpu().numpy(),bits))
                if not original_bits_preserved:
                    raise RuntimeError("Prepared vendor call modified original weight bits")
                after=current_context(scope="graph")
                case={"label":label,"weight_tensor":tensor_name,"shape_mnk":[m,n,k],
                    "input_dtype":"bf16","output_dtype":output_name,
                    "weight_sha256":hashlib.sha256(bits.tobytes()).hexdigest(),
                    "activation_sha256":hashlib.sha256(x.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                    "copies":count,"raw_ring_bytes":count*n*k*2,"plan":plans[0].plan,
                    "packing_seconds":0.,"packed_bytes":0,"numerics":numerics,
                    "original_bits_preserved":original_bits_preserved,"graph":{"methods":graph,**summarize_dispatch(graph)},
                    "eager":{"methods":eager,**summarize_dispatch(eager)},"gpu_telemetry":samples,
                    "operator_context_after":after,"operating_context_stable":before==after,
                    "eager_context_before":eager_before,"eager_context_after":eager_after,
                    "eager_operating_context_stable":eager_before==eager_after,
                    "setup_and_measurement_seconds":time.monotonic()-started,
                    "setup_seconds_per_copy":[p.setup_seconds for p in plans]}
                report["cases"].append(case)
                write_json(output_path,report)
                print(f"{label} {output_name}: graph {case['graph']['time_saved_pct']:+.3f}% "
                      f"gate={case['graph']['overhead_gate_passed']}, eager {case['eager']['time_saved_pct']:+.3f}% "
                      f"gate={case['eager']['overhead_gate_passed']}",flush=True)
                del plans,outputs,views,weights,weight,x,reference
                torch.cuda.empty_cache()
        report["status"]="complete"
        report["all_graph_overhead_gates_passed"]=all(c["graph"]["overhead_gate_passed"]
            and c["operating_context_stable"] for c in report["cases"])
        report["all_eager_overhead_gates_passed"]=all(c["eager"]["overhead_gate_passed"]
            and c["operating_context_stable"] and c["eager_operating_context_stable"]
            for c in report["cases"])
        write_json(output_path,report)
        return report
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        write_json(output_path,report)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--rounds",type=int,default=31)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    os.environ.setdefault("CUDA_CACHE_PATH",str(ROOT/".cache/cuda"))
    os.environ.setdefault("TMPDIR",str(ROOT/".scratch"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(ROOT/".cache/torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/".cache/triton"))
    from experiment_control import GpuLock
    with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
        run_small_fallback(args.output,rounds=args.rounds)


if __name__=="__main__":
    main()
