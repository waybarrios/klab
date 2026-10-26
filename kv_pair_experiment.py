"""Isolated exact K/V pair fusion; existing kernels and production routes stay unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROTOCOL = "experiments/kv_pair_fusion.json"


def concatenate_bits(k_bits, v_bits):
    import numpy as np
    if (not isinstance(k_bits, np.ndarray) or not isinstance(v_bits, np.ndarray)
            or k_bits.dtype != np.uint16 or v_bits.dtype != np.uint16
            or k_bits.ndim != 2 or v_bits.ndim != 2 or min(*k_bits.shape, *v_bits.shape) <= 0
            or k_bits.shape[1] != v_bits.shape[1]):
        raise ValueError("Require nonempty original uint16 K/V matrices with the same input width")
    return np.ascontiguousarray(np.concatenate((k_bits, v_bits), axis=0))


def summarize_pair(methods, vendor_routes, *, output_dtype):
    from bench_compressed import paired_gain_interval
    from compressed_campaign import summarize_methods
    import statistics
    fused = "torch_mm" if output_dtype == "bf16" else "torch_mm_out_fp32"
    labels = ("separate_torch", "separate_torch_aa", fused, fused + "_aa")
    if any(label not in methods for label in labels):
        raise ValueError("Need the fixed separate/fused native Torch arms and both A/A twins")
    values = {label: methods[label]["samples_us"] for label in labels}
    interval = paired_gain_interval({"separate": values["separate_torch"]}, values[fused])
    medians = {label: statistics.median(samples) for label, samples in values.items()}
    aa = 100 * (1 - medians["separate_torch_aa"] / medians["separate_torch"])
    fused_aa = 100 * (1 - medians[fused + "_aa"] / medians[fused])
    numeric = all(methods[label]["numerics"]["passed"] for label in labels)
    return {"fusion": {"time_saved_pct": 100 * (1 - medians[fused] / medians["separate_torch"]),
                       "ci95_pct": interval, "separate_aa_pct": aa, "fused_aa_pct": fused_aa,
                       "numerics_passed": numeric,
                       "passes_10pct_gate": bool(numeric and abs(aa) <= 1 and abs(fused_aa) <= 1 and interval[0] > 10),
                       "scope": "One fixed fused native Torch projection vs two separate native Torch projections"},
            "custom": summarize_methods(methods, vendor_routes, "coded_t64_v8"),
            "custom_win_validated": False, "production_promotion": False}


def _separate_factory(k_rows):
    def factory(x, raw, output_dtype):
        import torch
        from compressed_case import _batch
        dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float32
        copies, n, _ = raw.shape
        if x.shape[0] != 1:
            raise ValueError("This pair experiment requires M1 contiguous output views")
        k_views, v_views = [w[:k_rows].T for w in raw], [w[k_rows:].T for w in raw]
        functions, outputs, retained = {}, {}, [k_views, v_views]
        for label in ("separate_torch", "separate_torch_aa"):
            out = torch.empty((copies, 1, n), device=x.device, dtype=dtype)
            outputs[label] = out
            k_out, v_out = [o[:, :k_rows] for o in out], [o[:, k_rows:] for o in out]
            if not all(t.is_contiguous() for t in (*k_out, *v_out)):
                raise RuntimeError("Separate K/V output contract needs contiguous views")
            calls = []
            for i in range(copies):
                if output_dtype == "bf16":
                    def call(i=i, ko=k_out[i], vo=v_out[i]):
                        torch.mm(x, k_views[i], out=ko)
                        torch.mm(x, v_views[i], out=vo)
                else:
                    def call(i=i, ko=k_out[i], vo=v_out[i]):
                        torch.mm(x, k_views[i], out_dtype=torch.float32, out=ko)
                        torch.mm(x, v_views[i], out_dtype=torch.float32, out=vo)
                calls.append(call)
            functions[label] = _batch(calls)
            retained.extend((k_out, v_out))
        return functions, outputs, retained, {"matmuls_per_pair": 2, "packing_in_timed_path": False,
            "output_copy_in_timed_path": False, "contiguous_outputs": True}
    return factory


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(mode, seed, output, *, screen_report=None):
    import numpy as np
    import torch
    from bench_dispatch import current_context
    from compressed_campaign import _identity, _memory_clocks
    from compressed_case import measure_case
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from experiment_control import atomic_write, project_path, write_json
    from frozen_transfer import check_sources
    from gpu_unlock_status import inspect
    from real_weights import read_tensor

    output = project_path(ROOT, output)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if output.exists() or output.suffix != ".json":
        raise ValueError("Preserve previous results; choose a new project JSON report")
    expected_seeds = [protocol["screen"]["seed"]] if mode == "screen" else protocol["validation"]["seeds"]
    if seed not in expected_seeds:
        raise ValueError("Seed is not registered for this experiment mode")
    corpus_path = project_path(ROOT, protocol["corpus"])
    corpus = json.loads(corpus_path.read_text())
    parent = json.loads((ROOT / "results/compressed-performance-bf16-search-r2.json").read_text())
    if _sha(corpus_path) != parent["bindings"]["corpus_sha256"]:
        raise ValueError("Pair sources differ from the verified original checkpoint corpus")
    watched = {PROTOCOL: _sha(ROOT / PROTOCOL), "kv_pair_experiment.py": _sha(Path(__file__)),
               "frozen_transfer.py": _sha(ROOT / "frozen_transfer.py"),
               protocol["corpus"]: _sha(corpus_path)}
    if mode == "validate":
        if screen_report is None:
            raise ValueError("Confirmation requires a complete passed screen")
        screen_path = project_path(ROOT, screen_report)
        screen = json.loads(screen_path.read_text())
        if (screen.get("status") != "complete" or screen.get("screen_passed") is not True
                or screen.get("source_sha256") != watched):
            raise ValueError("Screen did not pass under these exact source/protocol/corpus bytes")
        watched[screen_path.relative_to(ROOT).as_posix()] = _sha(screen_path)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    context = current_context(scope="graph")
    if not context["ac_online"] or context["system76_power_profile"] != "Power Profile: Performance":
        raise ValueError("Pair comparison is registered for Performance on AC")
    vendor, workspace = CuBlas(), LtWorkspace(0)
    inventory = inspect()
    identity = _identity(vendor, inventory, revision=2)
    if identity != parent["bindings"]["identity"]:
        workspace.close()
        vendor.close()
        raise ValueError("Original CUDA/runtime identity changed")
    run_id = str(uuid.uuid4())
    report = {"kind": "kv-pair-fusion", "status": "running", "mode": mode, "seed": seed,
              "pid": os.getpid(), "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": watched, "runtime_identity": identity, "operator_context": context,
              "inventory": inventory, "cases": [], "production_promotion": False, "existing_routes_changed": False,
              "unlocked_comparison": "pending", "energy_claim": False, "input_files_sha256": {}}
    write_json(output, report)
    layers = protocol["screen" if mode == "screen" else "validation"]["layers"]
    directory = ROOT / ".scratch/kv-pairs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        for layer in layers:
            names = [f"model.layers.{layer}.{suffix}" for suffix in protocol["weights"]]
            k_bits, v_bits = [read_tensor(corpus, name) for name in names]
            bits = concatenate_bits(k_bits, v_bits)
            path = directory / f"layer-{layer}.u16"
            raw = bits.astype("<u2", copy=False).tobytes()
            atomic_write(path, raw)
            entry = {"name": f"model.layers.{layer}.kv_pair.weight", "dtype": "BF16", "shape": list(bits.shape),
                     "path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            pair_corpus = {"status": "complete", "tensors": [entry]}
            report["input_files_sha256"][entry["path"]] = entry["sha256"]
            for name in names:
                original = next(item for item in corpus["tensors"] if item["name"] == name)
                report["input_files_sha256"][original["path"]] = original["sha256"]
            for output_dtype in protocol["output_contracts"]:
                case = measure_case(entry, pair_corpus, 1, output_dtype, seed, [protocol["custom_config"]], None,
                                    workspace, vendor, rounds=protocol["rounds"], kernel_revision=2,
                                    external_factory=_separate_factory(k_bits.shape[0]))
                case.update(layer=layer, original_tensors=names,
                            original_tensor_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in (k_bits, v_bits)],
                            pair_comparison=summarize_pair(case["methods"], case["vendor_routes"], output_dtype=output_dtype))
                case["observed_memory_clocks"] = _memory_clocks([case])
                checks = bool(case["original_bits_preserved"] and case["raw_template_bitwise"]
                              and case["operating_context_stable"] and case["execution_context"] == context
                              and len(case["observed_memory_clocks"]) == 1)
                case["fusion_case_passed"] = bool(checks and case["pair_comparison"]["fusion"]["passes_10pct_gate"])
                case["custom_case_passed"] = bool(checks and case["storage_ratio"] <= .8
                                                   and case["pair_comparison"]["custom"]["passes_10pct_gate"])
                report["cases"].append(case)
                write_json(output, report)
                print(f"layer{layer} {output_dtype}: vendor fusion {case['pair_comparison']['fusion']['time_saved_pct']:+.2f}%, "
                      f"custom vs fused vendor {case['pair_comparison']['custom']['time_saved_pct']:+.2f}%", flush=True)
                if current_context(scope="graph") != context:
                    raise RuntimeError("Profile/AC changed during pair study")
        check_sources(identity, {**watched, **report["input_files_sha256"]})
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        if mode == "screen":
            report["screen_passed"] = all(c["fusion_case_passed"] and c["pair_comparison"]["fusion"]["time_saved_pct"]
                                           >= protocol["screen"]["minimum_point_saved_pct"] for c in report["cases"])
        write_json(output, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise
    finally:
        workspace.close()
        vendor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "validate"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-report")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch")):
        os.environ.setdefault(key, str(ROOT / directory))
    from experiment_control import GpuLock, project_path, write_json
    from frozen_transfer import temporary_profile, _interrupt
    output = project_path(ROOT, args.output)
    state_path = output.with_name(output.stem + "-state.json")
    if output.exists() or state_path.exists():
        raise ValueError("Refusing to overwrite previous study/state")
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    state = {}
    try:
        with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
            with temporary_profile("performance", state):
                run(args.mode, args.seed, output, screen_report=args.screen_report)
    finally:
        write_json(state_path, state)


if __name__ == "__main__":
    main()
