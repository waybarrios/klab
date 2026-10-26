"""Observational memory regime: streaming read, copy and launch floor.

CPU import is torch-free. measure_regime() runs on the current CUDA device and
the caller owns the GPU lock. Nothing here changes clocks, power, driver or
boot settings. Rates are effective rates from logical bytes moved, not DRAM
counters. "burst" means immediately after setup/verification, which already
ran GPU work; it is not a guaranteed cold-GPU state.
"""

import ctypes as ct
import hashlib
import math
import numbers
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from cuda_module import CudaModule

ROOT = Path(__file__).resolve().parent
MIB = 1024**2
OPERATIONS = ("read", "copy", "launch")
SOURCES = ("regime_kernels.cu",)
KERNELS = ("regime_read", "regime_copy", "regime_launch")
BLOCKS_PER_SM = 4
BLOCK_THREADS = 256
LAUNCH_REPEATS = 512
SAMPLE_TRAFFIC_BYTES = 2 * 1024**3  # Per timed read/copy graph: several ms at expected rates.
SUSTAIN_PAIRS = 4  # read+copy pairs per sustained-load graph replay.
ROUNDS = 15
SEED = 2026
VERIFY_CHUNK_WORDS = 16 * MIB
SMI_FIELDS = ("name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,"
              "clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,"
              "clocks_event_reasons.active")


def summarize_probe(operation, logical_bytes, samples_us, *, sink_bytes=0):
    """Median rate of one probe from per-call GPU times.

    read traffic = input + sink writes; input_gb_per_s counts the input alone
    (a weight-read proxy). copy traffic = input read + output write. launch is
    a latency floor, never a bandwidth.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown probe operation {operation!r}")
    for value in (logical_bytes, sink_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Byte counts must be nonnegative integers")
    if operation == "launch" and (logical_bytes or sink_bytes):
        raise ValueError("Launch probes carry no memory traffic")
    if operation != "launch" and not logical_bytes:
        raise ValueError("Read/copy probes need positive logical bytes")
    if operation == "copy" and sink_bytes:
        raise ValueError("Only read probes have a sink")
    samples = list(samples_us)
    if not samples or not all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                              and math.isfinite(v) and v > 0 for v in samples):
        raise ValueError("Timings must be a nonempty list of finite positive microseconds")
    samples = [float(v) for v in samples]
    median = statistics.median(samples)
    traffic = {"read": logical_bytes + sink_bytes, "copy": 2 * logical_bytes, "launch": 0}[operation]
    return {"operation": operation, "logical_bytes": logical_bytes, "sink_bytes": sink_bytes,
            "traffic_bytes": traffic, "samples_us": samples, "median_us": median,
            "min_us": min(samples), "max_us": max(samples),
            "gb_per_s": None if operation == "launch" else traffic / (median * 1e3),
            "input_gb_per_s": logical_bytes / (median * 1e3) if operation == "read" else None}


def expected_read_sink(words, threads):
    """Closed-form regime_read sinks over arange(words) as uint32, per thread."""
    import numpy as np

    if words < 0 or threads < 1:
        raise ValueError("Need nonnegative words and positive threads")
    vectors = words // 4
    t = np.arange(threads, dtype=np.int64)
    count = np.maximum(0, (vectors - t + threads - 1) // threads)  # Vectors t, t+T, ... < V.
    pairs = (count * (count - 1) // 2).astype(np.uint64)
    count, t = count.astype(np.uint64), t.astype(np.uint64)
    # Vector v holds 4v..4v+3, summing to 16v+6; uint64 wraparound preserves mod 2^32.
    total = np.uint64(16) * (count * t + np.uint64(threads) * pairs) + np.uint64(6) * count
    for j in range(words - 4 * vectors):
        total[j % threads] += np.uint64(4 * vectors + j)
    return (total & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def smi_snapshot(uuid=None):
    """Read-only raw nvidia-smi state; failures are recorded, not hidden."""
    query = [f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader"]
    attempts = [("all_gpus", ["nvidia-smi", *query])]
    if uuid:
        selector = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
        attempts.insert(0, ("device_uuid", ["nvidia-smi", "-i", selector, *query]))
    error = None
    for scope, command in attempts:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exception:
            error = str(exception)
            continue
        if result.returncode == 0:
            raw = result.stdout.strip()
            values, fields = [v.strip() for v in raw.split(",")], SMI_FIELDS.split(",")
            return {"ok": True, "scope": scope, "raw": raw,
                    "fields": dict(zip(fields, values)) if len(values) == len(fields) else None}
        error = result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    return {"ok": False, "error": error}


def _driver_api_version():
    try:
        value = ct.c_int()
        return value.value if ct.CDLL("libcuda.so.1").cuDriverGetVersion(ct.byref(value)) == 0 else None
    except (OSError, AttributeError):
        return None


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sink_mismatches(sink, words, threads):
    import numpy as np

    return int(np.count_nonzero(sink.cpu().numpy().view(np.uint32) != expected_read_sink(words, threads)))


def _arange_mismatches(tensor, words):
    """Bounded-memory GPU comparison with arange(words)."""
    import torch

    count = 0
    for start in range(0, words, VERIFY_CHUNK_WORDS):
        stop = min(words, start + VERIFY_CHUNK_WORDS)
        reference = torch.arange(start, stop, dtype=torch.int32, device=tensor.device)
        count += int((tensor[start:stop] != reference).sum().item())
    return count


def _time_rounds(names, repeats, replay, rng):
    """CUDA-event GPU time of whole graph replays (host dispatch excluded), per call."""
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rounds = []
    for index in range(ROUNDS):
        order = list(names)
        rng.shuffle(order)
        graph_ms, per_call = {}, {}
        for name in order:
            start.record()
            replay(name)
            end.record()
            end.synchronize()
            graph_ms[name] = start.elapsed_time(end)
            per_call[name] = graph_ms[name] * 1000 / repeats[name]
        rounds.append({"round": index, "order": order, "graph_ms": graph_ms, "per_call_us": per_call})
    return rounds


def _sustain(graph, seconds):
    """Replay one read+copy graph back to back until `seconds` of wall time pass."""
    import torch

    begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    pending, replays = [], 0
    start = time.monotonic()
    begin.record()
    while True:
        graph.replay()
        replays += 1
        marker = torch.cuda.Event(blocking=True)  # Sleep, not spin, while waiting.
        marker.record()
        pending.append(marker)
        if len(pending) > 2:  # Keep the GPU queue nonempty without unbounded backlog.
            pending.pop(0).synchronize()
        if time.monotonic() - start >= seconds:
            break
    finish.record()
    finish.synchronize()
    return {"elapsed_seconds": time.monotonic() - start,
            "gpu_elapsed_seconds": begin.elapsed_time(finish) / 1000, "replays": replays}


def _stage_summary(rounds, logical_bytes, sink_bytes):
    samples = {name: [r["per_call_us"][name] for r in rounds] for name in OPERATIONS}
    return {"read": summarize_probe("read", logical_bytes, samples["read"], sink_bytes=sink_bytes),
            "copy": summarize_probe("copy", logical_bytes, samples["copy"]),
            "launch": summarize_probe("launch", 0, samples["launch"])}


def measure_regime(bytes_mib=256, sustain_seconds=30):
    """Burst rounds, >= sustain_seconds of read+copy load, then steady rounds.

    The caller must hold the GPU lock. Buffers smaller than 20x L2 are
    enlarged to ceil(20*L2) MiB and flagged; requests above 8191 MiB are
    rejected because the arange input uses int32 values.
    """
    if isinstance(bytes_mib, bool) or not isinstance(bytes_mib, int) or not 0 < bytes_mib < 8192:
        raise ValueError("bytes_mib must be an integer from 1 to 8191")
    if (isinstance(sustain_seconds, bool) or not isinstance(sustain_seconds, numbers.Real)
            or not math.isfinite(sustain_seconds) or sustain_seconds <= 0):
        raise ValueError("sustain_seconds must be a finite positive number")
    import torch
    from gpu_monitor import GpuMonitor

    started_utc, wall_start = _utc(), time.monotonic()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, "uuid", "") or "") or None
    smi_before = smi_snapshot(uuid)
    l2 = int(props.L2_cache_size)
    requested = bytes_mib * MIB
    actual = max(requested, math.ceil(20 * l2 / MIB) * MIB)
    words = actual // 4
    if words >= 2**31:
        raise ValueError("Buffer exceeds int32 arange indexing")
    blocks = BLOCKS_PER_SM * props.multi_processor_count
    threads, grid = blocks * BLOCK_THREADS, (blocks, 1)
    sink_bytes = threads * 4
    repeats = {"read": max(1, math.ceil(SAMPLE_TRAFFIC_BYTES / (actual + sink_bytes))),
               "copy": max(1, math.ceil(SAMPLE_TRAFFIC_BYTES / (2 * actual))), "launch": LAUNCH_REPEATS}

    compile_start = time.monotonic()
    module = CudaModule(SOURCES, KERNELS, device)
    compile_seconds = time.monotonic() - compile_start
    graphs, launches = {}, 0
    try:
        with torch.cuda.device(device):
            source = torch.arange(words, dtype=torch.int32, device=device)
            output = torch.empty_like(source)
            sink = torch.empty(threads, dtype=torch.int32, device=device)
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            if source.data_ptr() % 16 or output.data_ptr() % 16:
                raise RuntimeError("Probe buffers must be 16-byte aligned")
            ops = {
                "read": lambda count=words: module.launch("regime_read", grid, BLOCK_THREADS,
                                                          source, sink, ct.c_ulonglong(count)),
                "copy": lambda count=words: module.launch("regime_copy", grid, BLOCK_THREADS,
                                                          source, output, ct.c_ulonglong(count)),
                "launch": lambda: module.launch("regime_launch", (1, 1), 32, counter),
            }

            def direct(name):
                nonlocal launches
                ops[name]()
                if name == "launch":
                    launches += 1

            def replay(name):
                nonlocal launches
                graphs[name].replay()
                launches += repeats[name] if name == "launch" else 0

            def verify(run_full):
                """Sentinel-fill, run, then check every sink entry, copied word and launch."""
                checks = {}
                sink.fill_(-1)
                ops["read"](words - 3)
                checks["read_tail_sink_mismatches"] = _sink_mismatches(sink, words - 3, threads)
                sink.fill_(-1)
                run_full("read")
                checks["read_full_sink_mismatches"] = _sink_mismatches(sink, words, threads)
                output.fill_(-1)
                ops["copy"](words - 3)
                checks["copy_tail_mismatches"] = (_arange_mismatches(output, words - 3)
                                                  + int((output[words - 3:] != -1).sum().item()))
                output.fill_(-1)
                run_full("copy")
                checks["copy_full_mismatches"] = _arange_mismatches(output, words)
                run_full("launch")
                checks["launch_counter_mismatches"] = int(counter.item() != launches)
                result = {"checks": checks, "expected_launch_count": launches,
                          "passed": not any(checks.values())}
                if not result["passed"]:
                    raise RuntimeError(f"Regime probe verification failed: {result}")
                return result

            verification = {"direct_before_timing": verify(direct)}
            for name in OPERATIONS:
                graphs[name] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):
                    for _ in range(repeats[name]):
                        ops[name]()
            graphs["sustain"] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs["sustain"]):
                for _ in range(SUSTAIN_PAIRS):
                    ops["read"]()
                    ops["copy"]()
            # Doubles as the untimed first replay (graph upload) of each timed graph.
            verification["graphs_before_timing"] = verify(replay)
            graphs["sustain"].replay()
            torch.cuda.synchronize(device)
            setup_seconds = time.monotonic() - wall_start

            rng = random.Random(SEED)
            with GpuMonitor() as monitor:
                def mark():
                    return time.monotonic() - monitor.start

                t0 = mark()
                burst = _time_rounds(OPERATIONS, repeats, replay, rng)
                t1 = mark()
                sustained = _sustain(graphs["sustain"], sustain_seconds)
                t2 = mark()
                steady = _time_rounds(OPERATIONS, repeats, replay, rng)
                t3 = mark()
            phases = [{"phase": name, "start_seconds": a, "end_seconds": b}
                      for name, a, b in (("burst", t0, t1), ("sustained", t1, t2), ("steady", t2, t3))]
            samples = monitor.samples
            for sample in samples:
                sample["phase"] = next((p["phase"] for p in phases
                                        if p["start_seconds"] <= sample["elapsed_seconds"] <= p["end_seconds"]),
                                       "outside_timed_stages")
            verification["graphs_after_sustained"] = verify(replay)
            torch.cuda.synchronize(device)
    finally:
        torch.cuda.synchronize(device)
        for graph in graphs.values():
            graph.reset()
        module.close()

    summaries = {"burst": _stage_summary(burst, actual, sink_bytes),
                 "steady": _stage_summary(steady, actual, sink_bytes)}
    medians = {stage: {"read_input_gb_per_s": s["read"]["input_gb_per_s"],
                       "read_total_gb_per_s": s["read"]["gb_per_s"],
                       "copy_gb_per_s": s["copy"]["gb_per_s"], "launch_floor_us": s["launch"]["median_us"]}
               for stage, s in summaries.items()}
    sustain_traffic = SUSTAIN_PAIRS * (actual + sink_bytes + 2 * actual)
    harness = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ("bench_bandwidth.py", "gpu_monitor.py")}
    return {
        "kind": "bandwidth-regime", "schema_version": 1, "status": "complete",
        "created_utc": started_utc, "finished_utc": _utc(), "wall_seconds": time.monotonic() - wall_start,
        "parameters": {"bytes_mib": bytes_mib, "sustain_seconds": sustain_seconds,
                       "rounds_per_stage": ROUNDS, "order_seed": SEED, "grid": list(grid),
                       "block_threads": BLOCK_THREADS, "dynamic_shared_bytes": 0,
                       "graph_repeats": repeats, "sustain_graph": f"{SUSTAIN_PAIRS} x (read, copy)"},
        "buffer": {"requested_bytes": requested, "actual_bytes": actual, "l2_bytes": l2,
                   "actual_over_l2": actual / l2 if l2 else None, "enlarged_to_20x_l2": actual > requested,
                   "rule": "actual = max(requested, ceil(20*L2 to MiB)); read and copy each use this size",
                   "sink_bytes": sink_bytes, "sink_entries": threads},
        "precision": "int32 arange input read as uint32; exact per-thread uint32 sums (mod 2^32); "
                     "bitwise integer copy; no floating point",
        "device": {"index": device, "name": props.name, "uuid": uuid,
                   "compute_capability": f"{props.major}.{props.minor}",
                   "multiprocessors": props.multi_processor_count, "total_memory_bytes": props.total_memory,
                   "pci_bus_id": str(getattr(props, "pci_bus_id", "")) or None},
        "runtime": {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                    "driver_api_version": _driver_api_version(),
                    "driver_version": (smi_before.get("fields") or {}).get("driver_version")},
        "module": {"compilation": module.compilation, "fingerprint": module.fingerprint,
                   "compile_seconds": compile_seconds},
        "harness_sha256": harness,
        "operator_context": {"ac_power": "unknown", "power_profile": "unknown",
                             "settings_changed": False,
                             "note": "Not supplied by the operator; this probe only observes."},
        "clocks_raw": {"before_setup": smi_before, "after_measurement": smi_snapshot(uuid)},
        "setup_seconds": setup_seconds,
        "verification": {**verification, "passed": all(v["passed"] for v in verification.values())},
        "stages": {"burst": {"rounds": burst, "summary": summaries["burst"]},
                   "steady": {"rounds": steady, "summary": summaries["steady"]}},
        "median": medians,
        "steady_over_burst": {key: medians["steady"][key] / medians["burst"][key] for key in medians["burst"]},
        "sustained": {**sustained, "requested_seconds": sustain_seconds,
                      "traffic_bytes_per_replay": sustain_traffic,
                      "average_traffic_gb_per_s": sustained["replays"] * sustain_traffic
                      / (sustained["gpu_elapsed_seconds"] * 1e9),
                      "load": "back-to-back CUDA graph replays of read+copy, <=3 in flight; no sleep load"},
        "telemetry": {"interval_seconds": monitor.interval, "phases": phases, "samples": samples,
                      "ok_samples": sum(bool(s.get("ok")) for s in samples)},
        "metric_definitions": {
            "read_input_gb_per_s": "input bytes / median per-call GPU time; weight-read proxy",
            "read_total_gb_per_s": "(input + sink bytes) / median per-call GPU time",
            "copy_gb_per_s": "(read + write bytes) / median per-call GPU time",
            "launch_floor_us": f"graph of {LAUNCH_REPEATS} near-empty launches / {LAUNCH_REPEATS}; "
                               "excludes host dispatch",
            "scope": "Effective logical-byte rates, not DRAM counters. Burst follows setup and "
                     "verification GPU work; it is not a guaranteed cold state.",
        },
    }


def main(argv=None):
    import argparse
    from experiment_control import GpuLock, run_output, write_json

    parser = argparse.ArgumentParser(description="Observational read/copy/launch regime probe")
    parser.add_argument("--bytes-mib", type=int, default=256)
    parser.add_argument("--sustain-seconds", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    output = run_output(ROOT, args.output)
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        report = measure_regime(args.bytes_mib, args.sustain_seconds)
    write_json(output, report)
    for stage, values in report["median"].items():
        print(f"{stage}: read {values['read_input_gb_per_s']:.1f} GB/s input, "
              f"copy {values['copy_gb_per_s']:.1f} GB/s, launch {values['launch_floor_us']:.2f} us")


if __name__ == "__main__":
    main()
