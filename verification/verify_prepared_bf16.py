"""Correctness and compilation checks for the new BF16 vendor fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH",str(ROOT/".cache/cuda"))
os.environ.setdefault("TMPDIR",str(ROOT/".scratch"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(ROOT/".cache/torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/".cache/triton"))


def main(output="results/prepared-bf16-verification-r2.json"):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from dynamic_projection import prepare_projection
    from experiment_control import project_path,write_json
    from projection_accuracy import check_outputs
    from prepared_bf16 import vendor_callable

    output=project_path(ROOT,output)
    if output.exists():
        raise ValueError("Preserve previous verification; choose a new output path")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.manual_seed(991)
    checks=0
    compiled_out=[]
    def require(condition,message):
        nonlocal checks
        if not condition:
            raise RuntimeError(message)
        checks+=1
    for dtype,name in ((torch.bfloat16,"bf16"),(torch.float32,"fp32")):
        profile=ROOT/f"results/compressed-performance-{name}-search-r2.json"
        for m,n,k in ((1,64,64),(1,128,896),(4,256,256)):
            w=torch.randn((n,k),device="cuda",dtype=torch.bfloat16)
            x=torch.randn((m,k),device="cuda",dtype=torch.bfloat16)
            p=prepare_projection(w,tokens=m,profile=profile,output_dtype=dtype)
            require(p.packed is None and p.packing_seconds==0 and not p.plan["custom_win_validated"],"Fallback packed/promoted weights")
            y=p.linear(x)
            require(y.shape==(m,n) and y.dtype==dtype,"Shape/output contract mismatch")
            require(check_outputs(x.double()@w.double().T,y,output_dtype=name)["passed"],"Numerical gate failed")
            out=torch.empty_like(y)
            require(p.linear_out(x,out=out) is out,"out was not reused")
            require(torch.equal(y.view(torch.int16 if name=="bf16" else torch.int32),
                               out.view(torch.int16 if name=="bf16" else torch.int32)),"Allocation/out parity failed")
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    p.linear_out(x,out=out)
                # A stale output or an empty graph must fail this proof.
                x.copy_(torch.randn_like(x)*2)
                out.fill_(float("nan"))
                graph.replay()
                expected=p.linear(x)
            stream.synchronize()
            require(not torch.equal(expected,y),"New input did not change the expected output")
            require(torch.equal(out,expected),"Graph replay failed on modified input/NaN output")
            graph.reset()
            # AOT tests functionalization too, without replacing the vendor GEMM
            # with an Inductor/autotuned implementation.
            compiled=torch.compile(p.linear,backend="aot_eager",fullgraph=True)
            actual=compiled(x)
            require(actual.dtype==dtype and torch.equal(actual,expected),"Compiled allocation path mismatch")
            try:
                compiled_with_out=torch.compile(p.linear_out,backend="aot_eager",fullgraph=True)
                actual_out=compiled_with_out(x,out=out)
            except Exception as error:
                compiled_out.append({"shape_mnk":[m,n,k],"output_dtype":name,
                    "status":"unsupported","error":f"{type(error).__name__}: {error}"[:1500]})
            else:
                require(actual_out is out and torch.equal(out,expected),"Compiled out path mismatch")
                compiled_out.append({"shape_mnk":[m,n,k],"output_dtype":name,"status":"passed"})
        with FakeTensorMode():
            fw=torch.empty((128,896),device="cuda",dtype=torch.bfloat16)
            fx=torch.empty((1,896),device="cuda",dtype=torch.bfloat16)
            fake=vendor_callable(fw,dtype)(fx)
            require(fake.dtype==dtype and fake.shape==(1,128),"Fake output mismatch")
    write_json(output,{"status":"passed","checks":checks,"scope":"Vendor-only fallback correctness, not a speed claim",
        "compiled_out":compiled_out,"torch":str(torch.__version__),"cuda":torch.version.cuda,
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("prepared_bf16.py","dynamic_projection.py","projection_policy.py","verification/verify_prepared_bf16.py")}})
    print(f"PASS {checks} GPU/stream/graph/compile/Fake checks",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--output",default="results/prepared-bf16-verification-r2.json")
    args=parser.parse_args()
    from experiment_control import GpuLock
    with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
        main(args.output)
