"""Use the prepared vendor fallback for a small real BF16 projection.

This verifies routing/correctness. It makes no custom-kernel speedup claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH",str(ROOT/".cache/cuda"))
os.environ.setdefault("TMPDIR",str(ROOT/".scratch"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens",type=int,default=1)
    parser.add_argument("--output-dtype",choices=("bf16","fp32"),default="bf16")
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    if args.tokens<=0:
        parser.error("--tokens must be positive")
    from experiment_control import GpuLock
    with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
        import numpy as np
        import torch
        from dynamic_projection import prepare_projection
        from projection_accuracy import check_outputs
        from real_weights import read_tensor

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.manual_seed(991)
        manifest=json.loads((ROOT/".cache/real-weights/manifest.json").read_text())
        bits=np.array(read_tensor(manifest,"model.layers.0.self_attn.k_proj.weight"),copy=True)
        w=torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
        x=torch.randn((args.tokens,w.shape[1]),device="cuda",dtype=torch.bfloat16)
        dtype=torch.bfloat16 if args.output_dtype=="bf16" else torch.float32
        prepared=prepare_projection(w,tokens=args.tokens,
            profile=f"results/compressed-performance-{args.output_dtype}-search-r2.json",
            output_dtype=dtype)
        y=prepared.linear(x)
        assert y.dtype==dtype and y.shape==(args.tokens,w.shape[0])
        assert check_outputs(x.double()@w.double().T,y,output_dtype=args.output_dtype)["passed"]
        assert prepared.packed is None and not prepared.plan["custom_win_validated"]
        print(json.dumps({"shape_mnk":prepared.shape_mnk,"output_dtype":str(y.dtype),
            "route":prepared.strategy,"reason":prepared.plan["reason"],
            "packing_seconds":prepared.packing_seconds,"setup_seconds":prepared.setup_seconds,
            "custom_win_validated":False,"numerics_passed":True},indent=2))


if __name__=="__main__":
    main()
