"""Separate Tensor Core input-precision diagnostic for before/after comparisons."""

import hashlib
import json
import math
import time
from pathlib import Path

from experiment_control import report_output,with_run_metadata,write_json

ROOT=Path(__file__).resolve().parent
DEFAULT_SHAPES=[(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(5120,2048,4096)]


def run(shapes,rounds,stage,output,run_metadata=None):
    import torch
    from bench import gpu_state,measure
    from cublas_baseline import CuBlas
    from gpu_unlock_status import inspect
    from gpu_monitor import GpuMonitor
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    inventory=inspect()
    software_unlocked=inventory["module_loaded"] or inventory["unlock_boot_option_present"]
    if stage=="before" and software_unlocked:
        raise ValueError("A before-install label requires no loaded unlock module and no unlock boot option")
    if stage=="after" and not (inventory["module_loaded"] and inventory["unlock_boot_option_present"]):
        raise ValueError("An after label requires a loaded unlock module and unlock boot option; neither proves selector state alone")
    vendor=CuBlas()
    report=with_run_metadata({"status":"running","kind":"tensorcore-diagnostic","stage":stage,
            "precision":"FP32 accumulation/output. FP16/BF16/TF32 inputs are separate relaxed-precision diagnostics.",
            "unlock_gain_claimed":False,"inventory":inventory,"torch":torch.__version__,"cuda":torch.version.cuda,
            "rounds":rounds,"cublas_version":vendor.version,"cases":[],
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("bench_tensorcore.py","cublas_baseline.py","bench.py","gpu_monitor.py")}},run_metadata)
    for m,n,k in shapes:
        torch.manual_seed(971)
        x=torch.randn(m,k,device="cuda")/math.sqrt(k)
        w=torch.randn(n,k,device="cuda")
        inputs={"fp32":(x,w),"tf32":(x,w),"fp16":(x.half(),w.half()),"bf16":(x.bfloat16(),w.bfloat16())}
        outputs={name:torch.empty(m,n,device="cuda") for name in inputs}
        functions={name:(lambda name=name:vendor.mm_compute32(*inputs[name],outputs[name],name)) for name in inputs}
        # Identical A/A path measures variation without changing input precision.
        aa=torch.empty(m,n,device="cuda")
        functions["bf16_aa"]=lambda:vendor.mm_compute32(*inputs["bf16"],aa,"bf16")
        before=gpu_state()
        with GpuMonitor() as telemetry:
            warmup_start=time.monotonic()
            warmup_count=0
            while warmup_count<20 or time.monotonic()-warmup_start<2:
                for fn in functions.values(): fn()
                torch.cuda.synchronize()
                warmup_count+=1
            warmup_elapsed=time.monotonic()-warmup_start
            timing=measure(functions,rounds=rounds,repeats=1)
        rows=torch.linspace(0,m-1,min(m,32),device="cuda").round().long()
        cols=torch.linspace(0,n-1,min(n,32),device="cuda").round().long()
        reference=x[rows].double()@w[cols].double().T
        methods={}
        for name in inputs:
            actual=outputs[name][rows[:,None],cols[None,:]].double()
            quantized_reference=inputs[name][0][rows].double()@inputs[name][1][cols].double().T
            if name=="tf32":
                torch.testing.assert_close(actual.float(),reference.float(),atol=5e-3,rtol=5e-3)
            else:
                torch.testing.assert_close(actual.float(),quantized_reference.float(),atol=3e-5,rtol=2e-4)
            methods[name]={**timing[name],"tflops":2*m*n*k/(timing[name]["median_us"]*1e6),
                           "sample_output_l2_error_vs_original_fp64":((actual-reference).norm()/reference.norm()).item(),
                           "sample_arithmetic_l2_error_vs_input_fp64":((actual-quantized_reference).norm()/quantized_reference.norm()).item(),
                           "meets_original_fp32_input_requirement":name=="fp32"}
        case={"shape_mnk":[m,n,k],"methods":methods,"aa_bf16":timing["bf16_aa"],
              "aa_bf16_difference_pct":100*(timing["bf16_aa"]["median_us"]/timing["bf16"]["median_us"]-1),
              "gpu_before":before,"gpu_after":gpu_state(),
              "gpu_telemetry":telemetry.samples,"warmup_iterations":warmup_count,"warmup_seconds":warmup_elapsed,
              "correctness_scope":"up to 32 rows/columns spread across the output, full K; relaxed TF32 tolerance recorded",
              "sample_rows":rows.tolist(),"sample_columns":cols.tolist()}
        report["cases"].append(case)
        write_json(output,report)
        print(f"{m}x{n}x{k}: "+", ".join(f"{name} {values['tflops']:.2f} TFLOPS" for name,values in methods.items())+
              f"; BF16 A/A {case['aa_bf16_difference_pct']:+.2f}%",flush=True)
    vendor.close()
    report["status"]="complete"
    write_json(output,report)
    return report


def make_report(report,source,destination=None):
    destination=report_output(ROOT,source,destination)
    lines=["# Tensor Core input-precision diagnostic", "",f"Stage label: `{report['stage']}`.","",
           "This measures existing cuBLAS binaries. It does not establish an unlock gain.",
           "Every output and compute type is FP32; FP16/BF16/TF32 input rounding is a separate precision tradeoff.","",
           "| M × N × K | FP32 TFLOPS | FP16 TFLOPS | BF16 TFLOPS | TF32 TFLOPS | BF16 A/A difference (%) |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        values=[case["methods"][name]["tflops"] for name in ("fp32","fp16","bf16","tf32")]
        lines.append("| "+" × ".join(map(str,case["shape_mnk"]))+" | "+" | ".join(f"{v:.2f}" for v in values)+
                     f" | {case['aa_bf16_difference_pct']:+.2f} |")
    lines += ["","Times are median GPU operation latency through CUDA graph replay after warmup.",
              "A/A compares the identical BF16 operation under two names; it estimates variation in this run.",
              "GPU clocks, power, limit observations, all timing samples, and sampled numerical errors are retained in JSON.",
              "The correctness check samples up to 32 rows and columns across output tiles, using the full reduction dimension.","",
              f"Raw run: [{source.name}]({source.name}).", ""]
    destination.write_text("\n".join(lines))
    return destination
