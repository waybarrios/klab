"""Bounded revision-one probe. Tuning data only, never held-out acceptance.

Uses real layer-0 weights, eight preregistered raw/coded launch settings and
the full frozen vendor pool. Each graph traverses distinct weight copies.
"""

import base64
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment_control import project_path, with_run_metadata, write_json

ROOT = Path(__file__).resolve().parent


def paired_gain_interval(vendors, candidate, draws=2000):
    values = [np.asarray(v, dtype=np.float64) for v in vendors.values()]
    candidate = np.asarray(candidate, dtype=np.float64)
    if (not values or candidate.ndim != 1 or len(candidate) < 31
            or any(v.shape != candidate.shape for v in values)
            or not all(np.isfinite(v).all() and (v > 0).all() for v in (*values, candidate))):
        raise ValueError("Need matching finite positive paired timing rounds (at least31)")
    indices = np.random.default_rng(2026).integers(0, len(candidate), size=(draws, len(candidate)))
    best = np.min([np.median(v[indices], axis=1) for v in values], axis=0)
    gain = 100 * (best - np.median(candidate[indices], axis=1)) / best
    return np.quantile(gain, [.025, .975]).tolist()


def validate_probe_inputs(spec, corpus, vendor, *, seed):
    from projection_accuracy import ATOL, RTOL
    if (corpus.get("status") != "complete" or corpus.get("revision") != spec.get("revision")
            or corpus.get("source_sha256") != spec.get("safetensors_sha256")):
        raise ValueError("Corpus differs from the preregistered source")
    if (vendor.get("kind") != "vendor-tuning" or vendor.get("status") != "complete"
            or vendor.get("screen_only") or not vendor.get("codec_gpu_gate_passed")):
        raise ValueError("Need the complete, frozen vendor grid and passed headroom gate")
    if seed != spec["tuning_seed"]:
        raise ValueError("Prototype probes use tuning seed901 only, never held-out seeds")
    if (spec.get("numerics", {}).get("fp32_output_atol") != ATOL
            or spec.get("numerics", {}).get("fp32_output_rtol") != RTOL):
        raise ValueError("Numerical tolerances differ from the preregistered precision contract")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _power_context():
    result = subprocess.run(["system76-power", "profile"], capture_output=True, text=True, timeout=10)
    return {"system76_power_profile": result.stdout.strip() if result.returncode == 0 else "unavailable",
            "ac_online": Path("/sys/class/power_supply/AC0/online").read_text().strip() == "1"}


def _batch(calls):
    def invoke():
        for call in calls:
            call()
    return invoke


def run_probe(spec_path, corpus_path, vendor_path, output_path, *, revision=1, run_metadata=None):
    import torch
    import torch.nn.functional as functional
    import compressed_bf16 as cb
    from bench import measure
    from bf16_reference import decode_bf16
    from cublaslt_baseline import LtPlan, LtWorkspace
    from gpu_monitor import GpuMonitor
    from projection_accuracy import check_outputs
    from real_weights import read_tensor

    spec_path, corpus_path, vendor_path, output_path = [project_path(ROOT, p) for p in
                                                     (spec_path, corpus_path, vendor_path, output_path)]
    spec, corpus, frozen = [json.loads(p.read_text()) for p in (spec_path, corpus_path, vendor_path)]
    validate_probe_inputs(spec, corpus, frozen, seed=901)
    if not 1 <= revision <= 3:
        raise ValueError("At most three preregistered performance revisions")
    # Vendor replay binds its ABI/device operation, while the validator is
    # checked against the immutable mathematical contract and rechecks every
    # current output. Record both validator revisions without rewriting old data.
    if frozen.get("spec_sha256") != _hash(spec_path):
        raise ValueError("Frozen vendor pool used a different experiment contract")
    for name in ("cublaslt_baseline.py",):
        if frozen["source_sha256"].get(name) != _hash(ROOT / name):
            raise ValueError(f"Frozen vendor precision/ABI dependency changed: {name}")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    power = _power_context()
    report = {"kind": "compressed-projection-probe", "status": "running", "mode": "tuning-probe",
              "seed": 901, "layer": 0, "output_dtype": "bf16", "revision": revision,
              "spec_sha256": _hash(spec_path), "corpus_sha256": _hash(corpus_path),
              "vendor_profile_sha256": _hash(vendor_path), "operator_context": power,
              "vendor_tuning_oracle_sha256": frozen["source_sha256"].get("projection_accuracy.py"),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: _hash(ROOT / name) for name in
                                 ("bench_compressed.py", "compressed_bf16.py", "compressed_kernels.cu", "bf16_reference.py",
                                  "projection_accuracy.py", "cublaslt_baseline.py", "bench.py", "gpu_monitor.py")},
              "cases": [], "accepted_10pct": False,
              "scope": "Layer0, seed901, M1, nativeBF16-output prototype. Not held-out/model acceptance."}
    report = with_run_metadata(report, run_metadata)
    write_json(output_path, report)
    workspace = LtWorkspace(0)
    try:
        for n, k in spec["large_primary_shapes_nk"]:
            entry = next(t for t in corpus["tensors"] if t["shape"] == [n, k] and ".layers.0." in t["name"])
            source_bits = np.array(read_tensor(corpus, entry["name"]), copy=True)
            encoded = cb.pack_bits(source_bits)
            if not np.array_equal(decode_bf16(encoded.payload.tobytes(), encoded.metadata, n, k), source_bits):
                raise RuntimeError("Independent CPU decode changed weights")
            torch.manual_seed(901)
            weight = torch.from_numpy(source_bits).to("cuda").view(torch.bfloat16)
            x = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
            l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
            copies = max(1, math.ceil(max(80 * 1024**2, 20 * l2) / (2 * n * k)))
            raw = weight.unsqueeze(0).repeat(copies, 1, 1)
            transposed = raw.transpose(1, 2).contiguous()
            packed = [cb.upload(encoded, 0) for _ in range(copies)]
            for item in packed:
                if not torch.equal(cb.unpack_weights(item).view(torch.int16), weight.view(torch.int16)):
                    raise RuntimeError("GPU decode changed original weight bits")
            vendor_case = next(c for c in frozen["cases"] if c["shape_mnk"] == [1, n, k]
                               and c["output_dtype"] == "bf16")
            functions, outputs, routes, plans = {}, {}, [], []

            def add(name, builder):
                out = torch.empty((copies, 1, n), dtype=torch.bfloat16, device="cuda")
                outputs[name] = out
                functions[name] = _batch([builder(i, out[i]) for i in range(copies)])

            try:
                for name, record in vendor_case["vendor_pool"].items():
                    if not vendor_case["methods"][name]["numerics"]["passed"]:
                        continue
                    routes.append(name)
                    if record.get("library") == "cublasLt":
                        group = [LtPlan(record, base64.b64decode(record["algorithm_b64"]), workspace,
                                        x=x, weight=raw[i], packed=transposed[i] if record["layout"] == "transposed" else None)
                                 for i in range(copies)]
                        plans.extend(group)
                        for label in (name, name + "_aa"):
                            add(label, lambda i, out, group=group: lambda: group[i].run(x, raw[i], out))
                    elif name == "torch_mm":
                        for label in (name, name + "_aa"):
                            add(label, lambda i, out: lambda: torch.mm(x, raw[i].T, out=out))
                    elif name == "torch_linear":
                        for label in (name, name + "_aa"):
                            holders = [None] * copies
                            def invoke(holders=holders):
                                for index in range(copies):
                                    holders[index] = functional.linear(x, raw[index])
                            outputs[label], functions[label] = holders, invoke
                    else:
                        raise ValueError(f"Unsupported native-BF16 vendor route {name}")
                for threads in (32, 64, 128, 256):
                    for vector in (2, 8):
                        for kind in ("raw", "coded"):
                            name = f"{kind}_t{threads}_v{vector}"
                            if kind == "raw":
                                add(name, lambda i, out, t=threads, v=vector:
                                    lambda: cb.raw_linear(x, raw[i], threads=t, vector=v, output_dtype="bf16", out=out))
                            else:
                                add(name, lambda i, out, t=threads, v=vector:
                                    lambda: cb.fused_linear(x, packed[i], threads=t, vector=v, output_dtype="bf16", out=out))
                with GpuMonitor() as monitor:
                    begin = time.monotonic()
                    while time.monotonic() - begin < 2:
                        functions[routes[0]]()
                        torch.cuda.synchronize()
                    timings = measure(functions, rounds=31, repeats=1)
                for method in timings.values():
                    for field in ("median_us", "min_us", "max_us"):
                        method[field] /= copies
                    method["samples_us"] = [v / copies for v in method["samples_us"]]
                reference = (x.double() @ weight.double().T).expand(copies, 1, n)
                for name, out in list(outputs.items()):
                    if isinstance(out, list):
                        outputs[name] = out = torch.stack(out)
                    timings[name]["numerics"] = check_outputs(reference, out, output_dtype="bf16")
                eligible = [name for name in routes if timings[name]["numerics"]["passed"]
                            and timings[name + "_aa"]["numerics"]["passed"]]
                best = min(eligible, key=lambda name: timings[name]["median_us"])
                baseline = timings[best]["median_us"]
                aa = 100 * (baseline - timings[best + "_aa"]["median_us"]) / baseline
                candidates = {}
                for threads in (32, 64, 128, 256):
                    for vector in (2, 8):
                        label = f"coded_t{threads}_v{vector}"
                        raw_label = f"raw_t{threads}_v{vector}"
                        parity = torch.equal(outputs[label].view(torch.int16), outputs[raw_label].view(torch.int16))
                        candidates[label] = {"time_saved_pct": 100 * (1 - timings[label]["median_us"] / baseline),
                                             "ci95_pct": paired_gain_interval({r: timings[r]["samples_us"] for r in eligible}, timings[label]["samples_us"]),
                                             "raw_template_bitwise": parity,
                                             "numerics_passed": timings[label]["numerics"]["passed"]}
                        if not parity:
                            raise RuntimeError(f"Raw/compressed arithmetic changed for {label}")
                selected = min(candidates, key=lambda name: timings[name]["median_us"])
                case = {"shape_mnk": [1, n, k], "weight_tensor": entry["name"], "weight_sha256": entry["sha256"],
                        "copies": copies, "raw_ring_bytes": raw.numel() * 2,
                        "packed_ring_bytes": sum(p.storage_bytes for p in packed), "original_bits_preserved": True,
                        "best_vendor": best, "best_vendor_us": baseline, "aa_time_saved_pct": aa,
                        "aa_passed": abs(aa) <= 1, "selected": selected,
                        "selected_time_saved_pct": candidates[selected]["time_saved_pct"],
                        "candidates": candidates, "methods": timings, "gpu_telemetry": monitor.samples,
                        "implementation": packed[0].implementation_fingerprint}
                report["cases"].append(case)
                write_json(output_path, report)
                print(f"prototype {n}x{k}: {selected}, {case['selected_time_saved_pct']:+.3f}% time saved; vendor {best} {baseline:.3f}us; A/A {aa:+.3f}%", flush=True)
            finally:
                for plan in plans:
                    plan.close()
    finally:
        workspace.close()
    if _power_context() != power:
        raise RuntimeError("Operating profile changed during the probe")
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                  minimum_12pct_tuning_screen=all(c["aa_passed"] and c["selected_time_saved_pct"] >= 12
                                                and c["candidates"][c["selected"]]["numerics_passed"] for c in report["cases"]))
    write_json(output_path, report)
    return report


def run(spec_path, corpus_path, vendor_path, *, mode, seed, profile_path, output_path,
        output_dtype="bf16", revision=1, rounds=31, run_metadata=None, diagnostic_training=False):
    """Run a preregistered campaign. The caller serializes GPU access."""
    from compressed_campaign import run as campaign
    return campaign(spec_path, corpus_path, vendor_path, mode=mode, seed=seed,
                    profile_path=profile_path, output_path=output_path, output_dtype=output_dtype,
                    revision=revision, rounds=rounds, run_metadata=run_metadata,
                    diagnostic_training=diagnostic_training)


def make_campaign_report(data, source, output):
    relative = os.path.relpath(source, Path(output).parent)
    lines = ["# Exact BF16 projection campaign", "", f"Raw data: [{Path(source).name}](<{relative}>)", "",
             f"Mode: {data['mode']}; seed: {data['seed']}; output: {data['output_dtype']}; "
             f"primary target passed in this process: {data.get('accepted_10pct', False)}.", "",
             "Latency saved is 100 × (vendor latency − custom latency) / vendor latency. "
             "Graph GPU timings exclude setup and Python dispatch. "
             "Tuning results are exploratory; acceptance requires three fresh validation processes.", "",
             "| Tensor | M,N,K | Best vendor µs | Coded CUDA µs | Latency saved % | 95% interval % | A/A % | Numerical gate |",
             "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for case in data["cases"]:
        summary = case.get("summary", case)
        selected = case.get("selected")
        if not selected or "time_saved_pct" not in summary:
            continue
        if summary.get("ci95_pct") is None:
            lines.append(f"| {case['weight_tensor']} | {','.join(map(str, case['shape_mnk']))} | unavailable | "
                         f"{case['methods'][selected]['median_us']:.3f} | unavailable | unavailable | unavailable | no eligible vendor |")
            continue
        low, high = summary["ci95_pct"]
        lines.append(f"| {case['weight_tensor']} | {','.join(map(str, case['shape_mnk']))} | "
                     f"{summary['best_vendor_us']:.3f} | {case['methods'][selected]['median_us']:.3f} | "
                     f"{summary['time_saved_pct']:.3f} | [{low:.3f}, {high:.3f}] | "
                     f"{summary['aa_time_saved_pct']:+.3f} | {case['methods'][selected]['numerics']['passed']} |")
    lines += ["", "Every failed case remains in the raw data. Fallback, full-model and unlocked gains are not measured by this table.", ""]
    Path(output).write_text("\n".join(lines))
    return Path(output)


def make_report(data, source, output):
    relative = os.path.relpath(source, Path(output).parent)
    lines = ["# Exact BF16 projection probe", "", f"Raw data: [{Path(source).name}](<{relative}>)", "",
             "Tuning only: layer0, seed901, M=1, native BF16 inputs/output and FP32 accumulation. "
             "Candidates were selected from eight configurations; these intervals do not replace held-out acceptance.", "",
             data["operator_context"]["system76_power_profile"].splitlines()[0], "",
             "| N,K | Best vendor us | Coded CUDA us | Latency saved % | Paired 95% interval % | A/A % |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for case in data["cases"]:
        selected = case["selected"]
        candidate = case["candidates"][selected]
        low, high = candidate["ci95_pct"]
        lines.append(f"| {case['shape_mnk'][1]},{case['shape_mnk'][2]} | {case['best_vendor_us']:.3f} | "
                     f"{case['methods'][selected]['median_us']:.3f} | {candidate['time_saved_pct']:.3f} | "
                     f"[{low:.3f}, {high:.3f}] | {case['aa_time_saved_pct']:+.3f} |")
    lines += ["", "Original weight bits and matching raw/coded arithmetic were checked independently. "
              "GPU graph latency excludes Python dispatch and setup. Held-out processes, the full M/output grid, "
              "eager calls, prepared dispatch and unlocked measurements remain pending.", ""]
    Path(output).write_text("\n".join(lines))
    return Path(output)


if __name__ == "__main__":
    import argparse
    from experiment_control import GpuLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    with GpuLock(ROOT / ".cache/gpu.lock"):
        run_probe("experiments/consumer_ampere_bf16.json", ".cache/real-weights/manifest.json",
                  "results/vendor-stock.json", project_path(ROOT, args.output), revision=args.revision)
