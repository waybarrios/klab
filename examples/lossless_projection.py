"""Run from the project root: .venv/bin/python -m examples.lossless_projection."""

import argparse
import json
import math
from pathlib import Path

from experiment_control import GpuLock, parse_shape

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, default=(1, 64, 64), help="M,N,K; K divisible by 64")
    parser.add_argument("--seed", type=int, default=857)
    args = parser.parse_args()
    m, n, k = args.shape
    if k % 64:
        parser.error("K must be divisible by 64")

    with GpuLock(ROOT / ".cache/gpu.lock"):
        import torch
        import lossless_weights

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(args.seed)
        x = torch.randn(m, k, device="cuda", dtype=torch.float32) / math.sqrt(k)
        w = torch.randn(n, k, device="cuda", dtype=torch.float32)

        packed = lossless_weights.pack_weights(w)  # Setup once for constant weights.
        assert torch.equal(packed.unpack().view(torch.int32), w.view(torch.int32))
        y = packed.linear(x)                      # Custom CUDA, FP32 accumulation/output.
        y_torch = x @ w.T
        reference64 = x.double() @ w.double().T
        torch.testing.assert_close(y, reference64.float(), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(y_torch, reference64.float(), atol=1e-5, rtol=1e-4)

        print(json.dumps({"x_shape": list(x.shape), "w_shape": list(w.shape),
                          "y_shape": list(y.shape), "exact_weights": True,
                          "output_relative_l2_error": ((y.double() - reference64).norm() / reference64.norm()).item(),
                          "original_bytes": w.numel() * 4, "packed_bytes": packed.storage_bytes,
                          "output_first_row_first_8": y[0, :8].tolist()}, indent=2))


if __name__ == "__main__":
    main()
