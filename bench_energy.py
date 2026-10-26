"""Fixed-work GPU board energy pilot, using frozen latency configurations.

No kernel or operating-setting changes. Caller selects the matching profile.
Energy is measured from NVML's cumulative mJ counter, not power times latency.
"""
import base64
import hashlib
import json
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))
os.environ.setdefault("TMPDIR", str(ROOT / ".scratch"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(profile_path, output_path, *, rounds=3, batch_seconds=10, warm_seconds=5):
    import torch
    import torch.nn.functional as functional
    import compressed_bf16 as cb
    import mtile_bf16 as mt
    from bench_compressed import _power_context
    from compressed_campaign import check_ring
    from compressed_case import launch_config
    from cublaslt_baseline import LtPlan, LtWorkspace
    from energy_metrics import batch_metrics, summarize_energy
    from experiment_control import project_path, write_json
    from gpu_monitor import GpuMonitor
    from nvml_energy import NvmlEnergy
    from power_context import execution_context
    from real_weights import read_tensor

    if rounds < 3 or batch_seconds < 10 or warm_seconds < 5:
        raise ValueError("Pilot requires >=3 paired rounds, >=10s batches and >=5s warmup")
    profile_path, output_path = [project_path(ROOT, p) for p in (profile_path, output_path)]
    if output_path.exists() or output_path == profile_path:
        raise ValueError("Preserve existing profiles and measurements")
    profile = json.loads(profile_path.read_text())
    if profile.get("status") != "frozen" or profile["bindings"]["revision"] != 2 or profile["bindings"]["output_dtype"] != "bf16":
        raise ValueError("Pilot requires a frozen revision2 BF16-output profile")
    for name, digest in profile["bindings"]["identity"]["source_sha256"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Frozen latency dependency changed: {name}")
    training_path = project_path(ROOT, profile["training_report"])
    if _sha(training_path) != profile["training_report_sha256"]:
        raise ValueError("Frozen training evidence changed")
    training = json.loads(training_path.read_text())
    if training.get("status") != "complete" or not training.get("minimum_12pct_tuning_screen"):
        raise ValueError("Need completed training with passed screen")
    power = _power_context()
    if execution_context(power) != profile["bindings"]["operator_context"]:
        raise ValueError("Select the power profile used for the frozen latency baseline")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if mt.implementation_fingerprint(0) != profile["bindings"]["identity"]["implementation"]:
        raise ValueError("GPU implementation/runtime differs from frozen latency profile")
    case = next(c for c in training["cases"] if c["weight_tensor"] == "model.layers.0.mlp.gate_proj.weight"
                and c["shape_mnk"] == [1, 4864, 896])
    selected = profile["choices"]["4864,896"]
    vendor_name = case["best_vendor"]
    if not all(case["methods"][name]["numerics"]["passed"] for name in (vendor_name, vendor_name + "_aa", selected)):
        raise ValueError("Frozen pilot methods did not pass the precision contract")
    record = case["vendor_pool"][vendor_name]
    corpus_path = ROOT / ".cache/real-weights/manifest.json"
    if _sha(corpus_path) != profile["bindings"]["corpus_sha256"]:
        raise ValueError("Verified corpus manifest changed")
    corpus = json.loads(corpus_path.read_text())
    bits = np.array(read_tensor(corpus, case["weight_tensor"]), copy=True)
    n, k = bits.shape
    encoded = cb.pack_bits(bits)
    weight = torch.from_numpy(bits).to("cuda").view(torch.bfloat16)
    torch.manual_seed(901)
    x = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    x_sha = hashlib.sha256(x.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    if x_sha != case["activation_sha256"] or encoded.source_sha256 != case["weight_sha256"]:
        raise ValueError("Pilot inputs differ from the frozen latency inputs")
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    copies = math.ceil(max(80 * 1024**2, 20 * l2) / bits.nbytes)
    raw = weight.unsqueeze(0).repeat(copies, 1, 1)
    transposed = raw.transpose(1, 2).contiguous()
    packed = [cb.upload(encoded, 0) for _ in range(copies)]
    if not torch.equal(raw.view(torch.int16), weight.view(torch.int16).expand_as(raw)):
        raise RuntimeError("Raw ring changed weight bits")
    if not torch.equal(transposed.view(torch.int16).transpose(1, 2), raw.view(torch.int16)):
        raise RuntimeError("Vendor layout changed weight bits")
    for item in packed:
        if not torch.equal(cb.unpack_weights(item).view(torch.int16), weight.view(torch.int16)):
            raise RuntimeError("Encoded ring changed weight bits")
    threads, vector = launch_config(selected)
    workspace, plans, graphs, sensor = LtWorkspace(0), [], {}, None
    outputs = {name: torch.empty((copies, 1, n), dtype=torch.bfloat16, device="cuda")
               for name in ("vendor", "vendor_aa", "coded")}
    report = {"kind": "projection-energy", "status": "running", "mode": "preliminary-pilot",
              "pid": os.getpid(), "run_id": str(uuid.uuid4()), "created_utc": datetime.now(timezone.utc).isoformat(),
              "profile_sha256": _sha(profile_path), "training_report_sha256": _sha(training_path),
              "shape_mnk": [1, n, k], "weight_tensor": case["weight_tensor"], "seed": 901,
              "weight_sha256": case["weight_sha256"], "activation_sha256": x_sha,
              "input_dtype": "bf16", "output_dtype": "bf16", "accumulation": "fp32",
              "selected": selected, "vendor_name": vendor_name, "vendor_record": record,
              "baseline_selection": "Frozen fastest eligible latency route; not minimum-energy search",
              "operator_context": power, "execution_context": execution_context(power),
              "copies": copies, "raw_ring_bytes": raw.numel() * 2, "packed_ring_bytes": sum(p.storage_bytes for p in packed),
              "implementation": mt.implementation_fingerprint(0),
              "source_sha256": {name: _sha(ROOT / name) for name in ("bench_energy.py", "energy_metrics.py", "nvml_energy.py")},
              "sensor_scope": "NVML GPU board energy, not process-exclusive or whole-laptop/wall energy",
              "idle_subtraction": False, "rounds": [], "requested_rounds": rounds,
              "minimum_batch_seconds": batch_seconds, "minimum_warm_seconds": warm_seconds,
              "whole_laptop_energy_measured": False, "unplugged_battery_measured": False}
    write_json(output_path, report)
    try:
        if record.get("library") == "cublasLt":
            plans = [LtPlan(record, base64.b64decode(record["algorithm_b64"]), workspace,
                            x=x, weight=raw[i], packed=transposed[i] if record["layout"] == "transposed" else None)
                     for i in range(copies)]
        elif vendor_name not in ("torch_mm", "torch_linear"):
            raise ValueError(f"Unsupported frozen pilot vendor route: {vendor_name}")

        def vendor_function(label):
            if vendor_name == "torch_linear":
                outputs[label] = [None] * copies
            def invoke():
                for i in range(copies):
                    if plans:
                        plans[i].run(x, raw[i], outputs[label][i])
                    elif vendor_name == "torch_mm":
                        torch.mm(x, raw[i].T, out=outputs[label][i])
                    else:
                        outputs[label][i] = functional.linear(x, raw[i])
            return invoke

        def coded():
            for i in range(copies):
                mt.fused_linear(x, packed[i], threads=threads, vector=vector, output_dtype="bf16", out=outputs["coded"][i])

        functions = {"vendor": vendor_function("vendor"), "vendor_aa": vendor_function("vendor_aa"), "coded": coded}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for name, fn in functions.items():
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn()
            stream.synchronize()
            graphs[name] = graph
        reference = (x.double() @ weight.double().T).cpu().numpy()

        def verify_outputs():
            checks = {name: check_ring(reference, torch.stack(out) if isinstance(out, list) else out, output_dtype="bf16")
                      for name, out in outputs.items()}
            if not all(check["passed"] for check in checks.values()):
                raise RuntimeError(f"Energy pilot output failed precision: {checks}")
            return checks

        report["numerics_before"] = verify_outputs()
        raw_reference = mt.raw_linear(x, weight, threads=threads, vector=vector, output_dtype="bf16")
        if not torch.equal(outputs["coded"].view(torch.int16), raw_reference.view(torch.int16).expand(copies, 1, n)):
            raise RuntimeError("Pilot raw/coded output parity failed")
        report["all_weight_bits_preserved"] = report["raw_template_bitwise"] = True
        # The execution loop follows below; preparation and graph capture are done.
        sensor = NvmlEnergy(record.get("device_uuid", str(torch.cuda.get_device_properties(0).uuid)))
        calibration = {}
        for name, graph in graphs.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            calibration[name] = start.elapsed_time(end) / 1000 / 32
        replay_count = math.ceil(batch_seconds * 1.2 / min(calibration.values()))
        projection_count = replay_count * copies
        report.update(calibration_seconds_per_graph=calibration, graph_replays_per_batch=replay_count,
                      exact_projections_per_batch=projection_count)
        write_json(output_path, report)
        generator = random.Random(2026)
        for index in range(rounds):
            order = list(graphs)
            generator.shuffle(order)
            row = {"round_index": index, "order": order}
            report["rounds"].append(row)
            for name in order:
                before_context = _power_context()
                if execution_context(before_context) != report["execution_context"]:
                    raise RuntimeError("Operating profile/AC changed before energy batch")
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                end.record()
                end.synchronize()
                with GpuMonitor() as monitor:
                    warm_start = time.monotonic()
                    while time.monotonic() - warm_start < warm_seconds:
                        for _ in range(16):
                            graphs[name].replay()
                        torch.cuda.synchronize()
                    before = sensor.snapshot()
                    start.record()
                    for _ in range(replay_count):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    after = sensor.snapshot()
                wall = after["query_started_seconds"] - before["query_started_seconds"]
                metrics = batch_metrics(before["energy_mj"], after["energy_mj"], projection_count,
                                        wall_seconds=wall, gpu_ms=start.elapsed_time(end))
                after_context = _power_context()
                row[name] = {**metrics, "counter_before": before, "counter_after": after,
                             "operator_context_before": before_context, "operator_context_after": after_context,
                             "gpu_telemetry": monitor.samples}
                write_json(output_path, report)
                if metrics["gpu_ms"] < batch_seconds * 1000 or execution_context(after_context) != report["execution_context"]:
                    raise RuntimeError("Energy batch too short or operating context changed; raw data retained")
                print(f"energy round{index} {name}: {metrics['mj_per_projection']:.6f}mJ/projection, "
                      f"{metrics['average_board_w']:.2f}W, {metrics['gpu_ms']/1000:.2f}s CUDA", flush=True)
        report["numerics_after"] = verify_outputs()
        report["summary"] = summarize_energy(report["rounds"])
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_path, report)
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_path, report)
        raise
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        for plan in plans:
            plan.close()
        workspace.close()
        if sensor is not None:
            sensor.close()


if __name__ == "__main__":
    import argparse
    from experiment_control import GpuLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="results/compressed-performance-bf16-search-r2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-seconds", type=float, default=10)
    parser.add_argument("--warm-seconds", type=float, default=5)
    args = parser.parse_args()
    with GpuLock(ROOT / ".cache/gpu.lock"):
        run(args.profile, args.output, rounds=args.rounds, batch_seconds=args.batch_seconds, warm_seconds=args.warm_seconds)
