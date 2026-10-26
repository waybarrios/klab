"""Bounded compatibility check of the official NWC wheel, no installation."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))

from experiment_control import GpuLock, write_json


def main():
    import numpy as np
    import torch
    from projection_accuracy import check_outputs
    from real_weights import read_tensor

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    directory = ROOT / ".cache/prior-art/nwc-0.10.1"
    wrapper = directory / "nwc/nwc_torch.py"
    binary = directory / "nwc/lib/nwc_ops.so"
    spec = importlib.util.spec_from_file_location("reviewed_nwc_prebuilt", wrapper)
    nwc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nwc)
    corpus = json.loads((ROOT / ".cache/real-weights/manifest.json").read_text())
    report = {"kind": "prior-art-compatibility", "implementation": "NWC official wheel 0.10.1",
              "status": "running", "package": json.loads((directory / "verified-download.json").read_text()),
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
              "format_version": nwc.VERSION, "layout": 0,
              "scope": "Compatibility and numerics only; no timing or superiority claim", "cases": []}
    destination = ROOT / "results/nwc-prebuilt-compatibility.json"
    if destination.exists():
        raise ValueError("Preserve existing compatibility result")
    write_json(destination, report)
    try:
        for n, k in ((13, 40), (4864, 896), (896, 4864)):
            torch.manual_seed(901)
            if n == 13:
                weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
                name = "synthetic-odd-shape"
            else:
                entry = next(t for t in corpus["tensors"] if t["shape"] == [n, k] and ".layers.0." in t["name"])
                bits = np.array(read_tensor(corpus, entry["name"]), copy=True)
                weight = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
                name = entry["name"]
            x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
            packed = nwc.NWCWeight(weight, elem="bf16", layout=0)
            decoded = packed.dequant()
            exact = torch.equal(decoded.view(torch.int16), weight.view(torch.int16))
            if not exact:
                raise RuntimeError("Prior-art codec did not recover every original bit")
            reference = weight.double() @ x.double()
            outputs = {"bf16": packed.linear_bf16(x, None), "fp32": packed.matvec(x.float())}
            checks = {dtype: check_outputs(reference, output, output_dtype=dtype) for dtype, output in outputs.items()}
            physical = sum(getattr(packed, field).numel() * getattr(packed, field).element_size()
                           for field in ("data", "bases", "hdr", "lut", "low"))
            case = {"weight_tensor": name, "shape_nk": [n, k], "all_weight_bits_preserved": exact,
                    "reported_storage_bytes": packed.bytes, "physical_storage_bytes": physical,
                    "scratch_bytes": packed.scratch.numel() * packed.scratch.element_size(),
                    "numerics": checks, "fp32_activation_widening": "exact BF16 to FP32, outside this correctness check"}
            report["cases"].append(case)
            write_json(destination, report)
            print(f"NWC {n}x{k}: bits exact, BF16 gate {checks['bf16']['passed']}, FP32 gate {checks['fp32']['passed']}", flush=True)
        torch.cuda.synchronize()
        report["status"] = "complete"
        report["all_numerics_passed"] = all(check["passed"] for case in report["cases"] for check in case["numerics"].values())
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(destination, report)


if __name__ == "__main__":
    with GpuLock(ROOT / ".cache/gpu.lock"):
        main()
