"""Exact M1 bias-free K/V graph example, FP32 output; no global route changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_pair_graph(wk, wv):
    """Prepare once; return graph callable(x, joint_out)->(k_view,v_view)."""
    import torch
    if (wk.device.type != "cuda" or wv.device != wk.device or wk.dtype != torch.bfloat16
            or wv.dtype != wk.dtype or wk.ndim != 2 or wv.ndim != 2 or wk.shape[1] != wv.shape[1]
            or not wk.is_contiguous() or not wv.is_contiguous() or wk.requires_grad or wv.requires_grad):
        raise ValueError("Require original contiguous inference BF16 CUDA K/V weights")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError("Disable TF32 and BF16 reduced-precision reductions for this contract")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Prepare concatenated weights before graph capture")
    joined = torch.cat((wk, wv), dim=0)
    nk, n, width = wk.shape[0], wk.shape[0] + wv.shape[0], wk.shape[1]
    wt = joined.T
    def pair_graph(x, out):
        if (tuple(x.shape) != (1, width) or x.device != joined.device or x.dtype != joined.dtype
                or tuple(out.shape) != (1, n) or out.device != x.device or out.dtype != torch.float32
                or not out.is_contiguous() or x.requires_grad or out.requires_grad):
            raise ValueError("Require M1 input and contiguous FP32 joint output on the prepared device")
        if torch._C._overlaps(out, x) or torch._C._overlaps(out, joined):
            raise ValueError("Output must not overlap input or prepared weights")
        torch.mm(x, wt, out_dtype=torch.float32, out=out)
        return out[:, :nk], out[:, nk:]
    return pair_graph


def main(output, layer):
    import numpy as np
    import torch
    from experiment_control import write_json
    from projection_accuracy import check_outputs
    from real_weights import read_tensor
    if output.exists():
        raise ValueError("Preserve previous example output")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    corpus = json.loads((ROOT / ".cache/real-weights/manifest.json").read_text())
    bits = [np.array(read_tensor(corpus, f"model.layers.{layer}.self_attn.{name}_proj.weight"), copy=True) for name in ("k", "v")]
    wk, wv = [torch.from_numpy(value).to("cuda").view(torch.bfloat16) for value in bits]
    paired = prepare_pair_graph(wk, wv)
    torch.manual_seed(1013)
    x = torch.randn((1, wk.shape[1]), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1, wk.shape[0] + wv.shape[0]), device="cuda", dtype=torch.float32)
    paired(x, out)  # Native handles warm before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            k, v = paired(x, out)
        x.copy_(torch.randn_like(x))
        graph.replay()
        reference = x.double() @ torch.cat((wk, wv)).double().T
    stream.synchronize()
    checks = check_outputs(reference, out, output_dtype="fp32")
    if not checks["passed"] or not k.is_contiguous() or not v.is_contiguous():
        raise RuntimeError("Pair example failed FP64 or contiguous-output checks")
    preserved = all(np.array_equal(w.view(torch.uint16).cpu().numpy(), original) for w, original in zip((wk, wv), bits))
    if not preserved:
        raise RuntimeError("Original weight bits changed")
    report = {"status": "passed", "scope": "Numerical graph example only, no performance or model claim",
              "operation": "bias-free M1 K/V pair", "layer": layer, "output_dtype": "fp32",
              "k_shape": list(k.shape), "v_shape": list(v.shape), "original_weight_bits_preserved": preserved,
              "numerics": checks, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output, report)
    graph.reset()
    print("PASS paired K/V graph, both contiguous outputs, original bits and FP64 oracle", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=(8, 16, 23), type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch")):
        os.environ.setdefault(key, str(ROOT / directory))
    from experiment_control import GpuLock, project_path
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        main(project_path(ROOT, args.output), args.layer)
