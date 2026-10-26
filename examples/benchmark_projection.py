"""Use the same baseline suite from Python. Run as -m examples.benchmark_projection."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from compare_baselines import run_comparison
from experiment_control import GpuLock, parse_shape, run_output
from result_metrics import case_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, action="append", help="Repeat M,N,K; K divisible by 64")
    parser.add_argument("--seeds", default="857,859")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--output", default="results/python-example.json")
    args = parser.parse_args()
    shapes = args.shape or [(1, 64, 64), (1, 256, 256)]
    if args.rounds < 3 or any(k % 64 for _, _, k in shapes):
        parser.error("Use rounds >= 3 and K divisible by 64")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    output = run_output(ROOT, args.output)
    metadata = {"run_id": output.stem, "pid": os.getpid(),
                "command": ["python", "-m", "examples.benchmark_projection"],
                "created_utc": datetime.now(timezone.utc).isoformat()}
    with GpuLock(ROOT / ".cache/gpu.lock"):
        report = run_comparison(shapes, seeds, args.rounds, False, output, run_metadata=metadata)
    for case in report["cases"]:
        row = case_metrics(case)
        print(f"{case['shape_mnk']} seed={case['seed']}: "
              f"latency reduction {row['latency_reduction_pct']:+.2f}%; "
              f"throughput gain {row['throughput_gain_pct']:+.2f}%")


if __name__ == "__main__":
    main()
