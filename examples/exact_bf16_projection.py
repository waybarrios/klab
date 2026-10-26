"""Call the lossless CUDA C++ kernel from Python. Correctness example, not timing.

./.venv/bin/python examples/exact_bf16_projection.py --output-dtype bf16
./.venv/bin/python examples/exact_bf16_projection.py --tokens 4 --output-dtype fp32

Uses native BF16 checkpoint bits. It does not convert FP32 weights to BF16.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--revision", type=int, choices=(1, 2), default=2)
    parser.add_argument("--wait", action="store_true", help="Queue behind another GPU experiment")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    from experiment_control import GpuLock
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        import numpy as np
        import compressed_bf16 as cb
        import torch
        if args.revision == 2:
            import mtile_bf16 as kernel
        else:
            kernel = cb
        from projection_accuracy import check_outputs
        from real_weights import read_tensor

        torch.set_grad_enabled(False)
        corpus = json.loads((ROOT / ".cache/real-weights/manifest.json").read_text())
        bits = np.array(read_tensor(corpus, "model.layers.0.mlp.gate_proj.weight"), copy=True)
        weight = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
        torch.manual_seed(901)
        x = torch.randn((args.tokens, weight.shape[1]), dtype=torch.bfloat16, device="cuda")

        # Preparation is explicit and happens before graph capture or hot calls.
        start = time.monotonic()
        packed_cpu = cb.pack_bits(bits)
        packed = cb.upload(packed_cpu, device=0)
        kernel.get_module(0)
        torch.cuda.synchronize()
        preparation_seconds = time.monotonic() - start

        # The body is CUDA C++, called through the C++ CUDA Driver API bridge.
        # This configuration is illustrative, not a validated dynamic choice.
        y = kernel.fused_linear(x, packed, threads=128, vector=8, output_dtype=args.output_dtype)
        raw = kernel.raw_linear(x, weight, threads=128, vector=8, output_dtype=args.output_dtype)
        integer = torch.int16 if args.output_dtype == "bf16" else torch.int32
        assert torch.equal(y.view(integer), raw.view(integer))
        assert torch.equal(cb.unpack_weights(packed).view(torch.int16), weight.view(torch.int16))
        check = check_outputs(x.double() @ weight.double().T, y, output_dtype=args.output_dtype)
        assert check["passed"], check
        print(f"{torch.cuda.get_device_name()}: {tuple(x.shape)} @ {tuple(weight.T.shape)} -> {tuple(y.shape)}, {y.dtype}")
        print(f"CUDA revision {args.revision}")
        print(f"All original weight bits and raw/coded output bits match. Numerical gate passed.")
        print(f"Prepared storage: {packed.storage_bytes:,} / {weight.numel() * weight.element_size():,} bytes")
        print(f"Preparation, including upload/compilation if needed: {preparation_seconds:.3f}s")
        print("This verifies a Python call. It makes no latency, eager-dispatch or model-speed claim.")


if __name__ == "__main__":
    main()
