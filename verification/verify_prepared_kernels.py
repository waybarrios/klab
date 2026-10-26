"""Actual GPU checks for the R2 candidate adapters; no performance search."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name,directory in (("CUDA_CACHE_PATH",".cache/cuda"),("TMPDIR",".scratch"),
                       ("TORCHINDUCTOR_CACHE_DIR",".cache/torchinductor"),
                       ("TRITON_CACHE_DIR",".cache/triton")):
    os.environ.setdefault(name,str(ROOT/directory))


def main(output_path):
    import numpy as np
    import torch
    import compressed_bf16 as cb
    import mtile_bf16 as mt
    from experiment_control import project_path,write_json
    from prepared_kernels import CodedCandidate,RawCandidate,coded_r2,raw_r2
    from projection_accuracy import check_outputs

    output_path=project_path(ROOT,output_path)
    if output_path.exists():
        raise ValueError("Preserve previous verification; use a new output path")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.manual_seed(991)
    checks=0
    def require(condition,message):
        nonlocal checks
        if not condition:
            raise RuntimeError(message)
        checks+=1

    for m,n,k in ((1,13,64),(3,17,896),(4,128,896),(17,13,64),(16,17,4864)):
        w=torch.randn((n,k),device="cuda",dtype=torch.bfloat16)
        original=w.view(torch.uint16).cpu().numpy().copy()
        packed=cb.upload(cb.pack_bits(original),0)
        require(torch.equal(cb.unpack_weights(packed).view(torch.int16),w.view(torch.int16)),"Upload changed bits")
        x=torch.randn((m,k),device="cuda",dtype=torch.bfloat16)
        for name,dtype,integer in (("bf16",torch.bfloat16,torch.int16),("fp32",torch.float32,torch.int32)):
            raw=RawCandidate(w,tokens=m,threads=128,vector=8,output_dtype=name)
            coded=CodedCandidate(packed,tokens=m,threads=128,vector=8,output_dtype=name)
            expected=mt.raw_linear(x,w,threads=128,vector=8,output_dtype=name)
            for candidate in (raw,coded):
                actual=candidate.linear(x)
                require(actual.dtype==dtype and actual.shape==(m,n),"Output contract mismatch")
                require(torch.equal(actual.view(integer),expected.view(integer)),"Adapter changed kernel output bits")
                require(check_outputs(x.double()@w.double().T,actual,output_dtype=name)["passed"],"Finite FP64 gate failed")
                out=torch.empty_like(actual)
                require(candidate.linear_out(x,out=out) is out,"out was not reused")
                require(torch.equal(out.view(integer),actual.view(integer)),"out path differs")
                # Both allocation and checked output reuse on a nondefault stream.
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        captured=candidate.linear(x)
                        candidate.linear_out(x,out=out)
                    x.copy_(torch.randn_like(x)*2)
                    out.fill_(float("nan"))
                    captured.fill_(float("nan"))
                    graph.replay()
                    fresh=mt.raw_linear(x,w,threads=128,vector=8,output_dtype=name)
                stream.synchronize()
                require(torch.equal(captured.view(integer),fresh.view(integer)),"Allocating graph did not replay new inputs")
                require(torch.equal(out.view(integer),fresh.view(integer)),"out graph did not replay new inputs")
                graph.reset()
                # Each candidate has an intentionally fixed shape/output
                # contract. Isolate its compile proof rather than exhausting
                # Dynamo's default eight-specialization cache on twenty plans.
                torch._dynamo.reset()
                compiled=torch.compile(candidate.linear,backend="aot_eager",fullgraph=True)
                result=compiled(x)
                require(result.dtype==dtype and torch.equal(result.view(integer),fresh.view(integer)),"AOT/Fake output mismatch")
                try:
                    candidate.linear(torch.randn((m+1,k),device="cuda",dtype=torch.bfloat16))
                except (ValueError,RuntimeError):
                    checks+=1
                else:
                    raise RuntimeError("A prepared shape was silently reused for a different M")
                try:
                    candidate.to("cuda")
                except (ValueError,RuntimeError):
                    checks+=1
                else:
                    raise RuntimeError("Device-bound candidate allowed unchecked module relocation")
                # Reset comparison input, since each graph proof changes it.
                expected=mt.raw_linear(x,w,threads=128,vector=8,output_dtype=name)
            require(np.array_equal(w.view(torch.uint16).cpu().numpy(),original),"Adapter changed original weights")
        print(f"checked M,N,K={m},{n},{k}, both output contracts",flush=True)

    # Dispatcher schema and Fake tests use cloned encoded tensors too.
    w=torch.randn((13,64),device="cuda",dtype=torch.bfloat16)
    packed=cb.upload(cb.pack_bits(w.view(torch.uint16).cpu().numpy()),0)
    x=torch.randn((3,64),device="cuda",dtype=torch.bfloat16)
    for name in ("bf16","fp32"):
        for op,args in ((coded_r2,(x,packed.payload,packed.metadata,3,13,64,128,8,name)),
                        (raw_r2,(x,w,3,13,64,128,8,name))):
            results=torch.library.opcheck(op,args,test_utils=("test_schema","test_faketensor"))
            require(all(value=="SUCCESS" for value in results.values()),f"opcheck failed: {results}")

    # A normal compiled caller contains multiple plans in one graph. Prove that
    # composition as well, rather than relying only on isolated compile caches.
    w2=torch.randn((17,896),device="cuda",dtype=torch.bfloat16)
    p2=cb.upload(cb.pack_bits(w2.view(torch.uint16).cpu().numpy()),0)
    a=RawCandidate(w,tokens=3,threads=128,vector=8,output_dtype="bf16")
    b=CodedCandidate(p2,tokens=4,threads=128,vector=8,output_dtype="fp32")
    x2=torch.randn((4,896),device="cuda",dtype=torch.bfloat16)
    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a,self.b=a,b
        def forward(self,left,right):
            return self.a(left),self.b(right)
    torch._dynamo.reset()
    pair=torch.compile(Pair(),backend="aot_eager",fullgraph=True)
    left,right=pair(x,x2)
    require(torch.equal(left.view(torch.int16),mt.raw_linear(x,w,threads=128,vector=8,
            output_dtype="bf16").view(torch.int16)),"Composed BF16 plan mismatch")
    require(torch.equal(right.view(torch.int32),mt.raw_linear(x2,w2,threads=128,vector=8,
            output_dtype="fp32").view(torch.int32)),"Composed FP32 plan mismatch")

    # Output reuse must retain the frozen alias guards through both adapters.
    square=torch.randn((64,64),device="cuda",dtype=torch.bfloat16)
    ps=cb.upload(cb.pack_bits(square.view(torch.uint16).cpu().numpy()),0)
    xs=torch.randn((3,64),device="cuda",dtype=torch.bfloat16)
    r=RawCandidate(square,tokens=3,threads=128,vector=8,output_dtype="bf16")
    c=CodedCandidate(ps,tokens=3,threads=128,vector=8,output_dtype="bf16")
    alias=ps.payload[:3*64*2].view(torch.bfloat16).reshape(3,64)
    for candidate,out in ((r,xs),(c,alias)):
        try:
            candidate.linear_out(xs,out=out)
        except (ValueError,RuntimeError):
            checks+=1
        else:
            raise RuntimeError("Adapter bypassed an input/packed-buffer output alias guard")
    require(torch.equal(cb.unpack_weights(ps).view(torch.int16),square.view(torch.int16)),
            "Rejected alias call changed packed data")
    torch.cuda.synchronize()
    report={"status":"passed","checks":checks,"scope":"Adapter correctness only; no speed or dispatch promotion",
        "torch":str(torch.__version__),"cuda":torch.version.cuda,"implementation":mt.implementation_fingerprint(0),
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("prepared_kernels.py","verification/verify_prepared_kernels.py")}}
    write_json(output_path,report)
    print(f"PASS {checks} GPU/stream/graph/AOT/schema checks",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/prepared-kernels-verification-r1.json")
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    from experiment_control import GpuLock
    with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
        main(args.output)
