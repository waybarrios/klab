"""Correctness only for revision2. Never selects a performance configuration."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))


def main():
    import numpy as np
    import torch
    import compressed_bf16 as cb
    import mtile_bf16 as mt
    from experiment_control import write_json
    from projection_accuracy import check_outputs

    torch.set_grad_enabled(False)
    torch.manual_seed(315)
    cb.get_module(0)
    module = mt.get_module(0)
    checks = 0
    finite_checks = 0
    for n, k in ((13, 64), (17, 896), (5, 4864), (128, 896)):
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        packed = cb.upload(cb.pack_bits(w.view(torch.uint16).cpu().numpy()), 0)
        for m in (1, 2, 3, 4, 7, 8, 15, 16, 17, 64):
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            reference = x.double() @ w.double().T
            for t in (32, 64, 128, 256):
                for v in (2, 8):
                    for dtype in ("bf16", "fp32"):
                        raw = cb.raw_linear(x, w, threads=t, vector=v, output_dtype=dtype)
                        tiled_raw = mt.raw_linear(x, w, threads=t, vector=v, output_dtype=dtype)
                        tiled = mt.fused_linear(x, packed, threads=t, vector=v, output_dtype=dtype)
                        integer = torch.int16 if dtype == "bf16" else torch.int32
                        assert torch.equal(raw.view(integer), tiled_raw.view(integer)), (m, n, k, t, v, dtype, "raw")
                        assert torch.equal(raw.view(integer), tiled.view(integer)), (m, n, k, t, v, dtype, "coded")
                        checks += 2
                        numerical = check_outputs(reference, tiled, output_dtype=dtype)
                        assert numerical["passed"], (m, n, k, t, v, dtype, numerical)
                        finite_checks += 1
        print(f"finite {n}x{k}: all token counts, configurations and output contracts passed", flush=True)
    # Actual large projection sizes with deterministic synthetic finite inputs.
    for n, k in ((4864, 896), (896, 4864)):
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        p = cb.upload(cb.pack_bits(w.view(torch.uint16).cpu().numpy()), 0)
        x = torch.randn((16, k), dtype=torch.bfloat16, device="cuda")
        reference = x.double() @ w.double().T
        for t in (128, 256):
            for dtype in ("bf16", "fp32"):
                a = cb.raw_linear(x, w, threads=t, vector=8, output_dtype=dtype)
                b = mt.raw_linear(x, w, threads=t, vector=8, output_dtype=dtype)
                c = mt.fused_linear(x, p, threads=t, vector=8, output_dtype=dtype)
                integer = torch.int16 if dtype == "bf16" else torch.int32
                assert torch.equal(a.view(integer), b.view(integer))
                assert torch.equal(a.view(integer), c.view(integer))
                assert check_outputs(reference, c, output_dtype=dtype)["passed"]
                checks += 2
                finite_checks += 1
    # Nonfinite/subnormal stress is parity only, separate from performance accuracy.
    bits = np.resize(np.array([0, 0x8000, 1, 0x7f, 0x80, 0x3f80, 0xbf80,
                               0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7f81, 0x7f7f, 0xff7f], dtype=np.uint16), (13, 64))
    w = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
    p = cb.upload(cb.pack_bits(bits), 0)
    x = torch.from_numpy(np.resize(bits[0], (17, 64)).copy()).to("cuda").view(torch.bfloat16)
    for m in (3, 16, 17):
        for t in (32, 64, 128, 256):
            for v in (2, 8):
                for dtype in ("bf16", "fp32"):
                    a = cb.raw_linear(x[:m], w, threads=t, vector=v, output_dtype=dtype)
                    b = mt.raw_linear(x[:m], w, threads=t, vector=v, output_dtype=dtype)
                    c = mt.fused_linear(x[:m], p, threads=t, vector=v, output_dtype=dtype)
                    integer = torch.int16 if dtype == "bf16" else torch.int32
                    assert torch.equal(a.view(integer), b.view(integer))
                    assert torch.equal(a.view(integer), c.view(integer))
                    checks += 2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    finite_w = torch.randn((13, 64), dtype=torch.bfloat16, device="cuda")
    finite_p = cb.upload(cb.pack_bits(finite_w.view(torch.uint16).cpu().numpy()), 0)
    finite_x = torch.randn((3, 64), dtype=torch.bfloat16, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = cb.raw_linear(finite_x, finite_w, threads=128, vector=8, output_dtype="fp32")
        out = mt.fused_linear(finite_x, finite_p, threads=128, vector=8, output_dtype="fp32")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            mt.fused_linear(finite_x, finite_p, threads=128, vector=8, output_dtype="fp32", out=out)
        graph.replay()
    stream.synchronize()
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
    graph.reset()
    checks += 1
    for bad_x in (finite_x.float(), finite_x.cpu(), finite_x[:, :63], finite_x[:0], finite_x.clone().requires_grad_()):
        try:
            mt.fused_linear(bad_x, finite_p, threads=128, vector=8, output_dtype="fp32")
        except (ValueError, RuntimeError):
            checks += 1
        else:
            raise AssertionError("Invalid tensor accepted")
    torch.cuda.synchronize()
    report = {"status": "passed", "checks": checks, "finite_fp64_checks": finite_checks,
              "zero_mismatches": True, "scope": "Correctness only, no performance measurements",
              "implementation": mt.implementation_fingerprint(0), "compilation": module.compilation}
    write_json(ROOT / "results/mtile-bf16-verification.json", report)
    print(f"PASS {checks} bit/runtime/guard checks and {finite_checks} finite FP64 checks; zero mismatches", flush=True)


if __name__ == "__main__":
    from experiment_control import GpuLock
    with GpuLock(ROOT / ".cache/gpu.lock"):
        main()
