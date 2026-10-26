"""Training-only NWC binary comparison on identical rotating checkpoint weights.

No installation. Uses the hash-verified official wheel and retains both its
default layout and its optional Tensor Core layout, with independent controls.
"""
import hashlib
import importlib.util
import itertools
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))


def external_factory(nwc):
    def prepare(x, raw, output_dtype):
        import torch
        functions, outputs, owners, infos = {}, {}, [], {}
        for layout in (0, 1) if nwc.LAYOUT1 else (0,):
            weights = [nwc.NWCWeight(weight, elem="bf16", layout=layout) for weight in raw]
            owners.extend(weights)
            multi = x.shape[0] > 1
            if multi and output_dtype != "bf16":
                raise ValueError("This comparison uses the official NWC multi-token BF16 output API")
            layers = [nwc.NWCLinear(w=item) for item in weights] if multi else []
            owners.extend(layers)
            for i, item in enumerate(weights):
                if not torch.equal(item.dequant().view(torch.int16), raw[i].view(torch.int16)):
                    raise RuntimeError(f"NWC layout {layout} failed original-bit reconstruction")
            physical = sum(sum(t.numel() * t.element_size() for t in item.tensors().values()) for item in weights)
            infos[f"nwc_layout{layout}"] = {
                "layout": layout, "all_weight_bits_preserved": True,
                "physical_ring_bytes": physical,
                "scratch_bytes": sum(item.scratch.numel() * item.scratch.element_size() for item in weights),
                "operation_scope": "official NWCLinear.forward: dequantize + F.linear" if multi else "native NWC token path",
                "activation_conversion": "Any required exact BF16-to-FP32 widening is inside measured call"}
            for name in (f"nwc_layout{layout}", f"nwc_layout{layout}_aa"):
                holders = [None] * len(weights)
                def invoke(holders=holders, weights=weights, layers=layers):
                    for i, item in enumerate(weights):
                        if multi:
                            holders[i] = layers[i](x)
                        else:
                            y = item.linear_bf16(x[0], None) if output_dtype == "bf16" else item.matvec(x[0].float())
                            holders[i] = y.view(1, -1)
                functions[name], outputs[name] = invoke, holders
        return functions, outputs, owners, infos
    return prepare


def run(output_path, *, multi=False):
    import torch
    from bench_compressed import _power_context
    from compressed_campaign import summarize_methods
    from compressed_case import measure_case
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from experiment_control import project_path, write_json

    output_path = project_path(ROOT, output_path)
    if output_path.exists():
        raise ValueError("Preserve existing prior-art measurements")
    directory = ROOT / ".cache/prior-art/nwc-0.10.1"
    wrapper = directory / "nwc/nwc_torch.py"
    binary = directory / "nwc/lib/nwc_ops.so"
    verified = json.loads((ROOT / "results/nwc-prebuilt-compatibility.json").read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest() != verified["binary_sha256"]:
        raise ValueError("NWC binary changed after compatibility validation")
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != verified["wrapper_sha256"]:
        raise ValueError("NWC wrapper changed after compatibility validation")
    spec = importlib.util.spec_from_file_location("reviewed_nwc_prebuilt", wrapper)
    nwc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nwc)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    corpus = json.loads((ROOT / ".cache/real-weights/manifest.json").read_text())
    frozen = json.loads((ROOT / "results/vendor-stock.json").read_text())
    if hashlib.sha256((ROOT / "cublaslt_baseline.py").read_bytes()).hexdigest() != frozen["source_sha256"]["cublaslt_baseline.py"]:
        raise ValueError("Frozen cuBLASLt ABI changed")
    report = {"kind": "prior-art-comparison", "status": "running", "mode": "tuning-diagnostic",
              "seed": 901, "layer": 0, "pid": os.getpid(), "run_id": str(uuid.uuid4()),
              "created_utc": datetime.now(timezone.utc).isoformat(), "operator_context": _power_context(),
              "nwc_binary": verified, "cases": [], "accepted_10pct": False,
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ("bench_prior_art.py", "compressed_case.py", "compressed_campaign.py", "bench.py",
                                 "projection_accuracy.py", "cublaslt_baseline.py")},
              "scope": "Training-only graph GPU latency; no held-out or model-level superiority claim",
              "custom_config_source": "previous layer0 seed901 prototype winner coded_t128_v8; no new search"}
    profile = None
    if multi:
        import mtile_bf16
        from power_context import execution_context
        profile_path = ROOT / "results/compressed-performance-bf16-search-r2.json"
        profile = json.loads(profile_path.read_text())
        if profile.get("status") != "frozen" or profile["bindings"]["identity"]["implementation"] != mtile_bf16.implementation_fingerprint(0):
            raise ValueError("Need the completed current revision2 training configuration")
        if profile["bindings"]["operator_context"] != execution_context(_power_context()):
            raise ValueError("Operating profile differs from the frozen revision2 choice")
        report.update(custom_config_source="frozen revision2 M1 training choices; no multi-token tuning",
                      profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                      output_contract="native BF16 inputs/output, FP32 accumulation",
                      external_scope="official NWCLinear.forward multi-token path, including full decompression and F.linear",
                      source_sha256={**report["source_sha256"], **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                                                for name in ("mtile_bf16.py", "mtile_kernels.cu")}})
    write_json(output_path, report)
    workspace, vendor = LtWorkspace(0), CuBlas()
    try:
        for output_dtype in (("bf16",) if multi else ("bf16", "fp32")):
            for m, (n, k) in itertools.product((2, 4) if multi else (1,), ((4864, 896), (896, 4864))):
                entry = next(t for t in corpus["tensors"] if t["shape"] == [n, k] and ".layers.0." in t["name"])
                vendor_case = next(c for c in frozen["cases"] if c["shape_mnk"] == [m, n, k] and c["output_dtype"] == output_dtype)
                selected = profile["choices"][f"{n},{k}"] if multi else "coded_t128_v8"
                case = measure_case(entry, corpus, m, output_dtype, 901, [selected],
                                    vendor_case, workspace, vendor, external_factory=external_factory(nwc),
                                    kernel_revision=2 if multi else 1)
                case["selected"] = selected
                case.update(summarize_methods(case["methods"], case["vendor_routes"], case["selected"]))
                external_routes = [name for name in case["methods"] if name.startswith("nwc_") and not name.endswith("_aa")]
                case["against_prior_art"] = summarize_methods(case["methods"], external_routes, case["selected"])
                case["passes_10pct_gate"] = False  # This is training evidence, never acceptance.
                report["cases"].append(case)
                write_json(output_path, report)
                print(f"{output_dtype} M={m} {n}x{k}: ours vs best eligible NWC {case['against_prior_art']['time_saved_pct']:+.3f}% latency saved", flush=True)
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        workspace.close()
        vendor.close()
        write_json(output_path, report)
    return report


if __name__ == "__main__":
    import argparse
    from experiment_control import GpuLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multi", action="store_true", help="Compare revision2 against NWC's official multi-token path at M=2,4")
    args = parser.parse_args()
    with GpuLock(ROOT / ".cache/gpu.lock"):
        run(args.output, multi=args.multi)
