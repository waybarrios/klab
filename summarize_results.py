"""Build a short report from saved measurements, without running GPU work."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    groups = defaultdict(list)
    inputs = ["cublas-validation.json", "cublas-validation-repeat.json"]
    for filename in inputs:
        data = json.loads((ROOT / "results" / filename).read_text())
        if data["status"] != "complete":
            raise RuntimeError(f"Incomplete report: {filename}")
        for case in data["cases"]:
            groups[tuple(case["shape_mnk"])].append(case)
    lines = [
        "# Measured results on the RTX 3070 Ti Laptop GPU",
        "",
        "FP32 inputs, FP32 accumulation/output, TF32 disabled. The baseline is the",
        "fastest measured path among PyTorch mm/mv and direct cuBLAS SGEMM/SGEMV",
        "(SGEMV/mv only for M=1). Vendor default math, not pedantic mode.",
        "",
        "Frozen configurations were checked with three fresh seeds in two separate",
        "processes. Every entry below is the range across those six trials; these",
        "ranges are not confidence intervals. Values above 1x favor our kernel.",
        "",
        "| M × N × K | Configuration | Resident-weight speedup | Eviction-condition speedup |",
        "| --- | --- | --- | --- |",
    ]
    for shape, cases in groups.items():
        warm = [c["warm_speedup"] for c in cases]
        cold = [c["cold_speedup"] for c in cases]
        lines.append(f"| {' × '.join(map(str, shape))} | {cases[0]['selected']} | "
                     f"{min(warm):.3f}–{max(warm):.3f}x | {min(cold):.3f}–{max(cold):.3f}x |")
    lines += [
        "", "## Interpretation", "",
        "The 1 × 4096 × 4096 projection gives a small, repeatable win in both",
        "conditions. Large resident-weight wins on smaller shapes disappear after",
        "eviction. The four-token candidate regresses in the validation trials.",
        "These are workload-specific results, not a general replacement for cuBLAS.",
        "", "CUDA graph replay excludes Python/FFI dispatch and compilation.",
        "The eviction condition runs a 64 MiB read/write operation before each",
        "measurement, outside external start/stop events. It approximates cold",
        "caches; it is not a hardware cache invalidation API. Warm and eviction",
        "conditions use different timing protocols, with identical treatment of",
        "all methods within each condition.",
        "", "Under benchmark load, nvidia-smi reported a memory clock of 810 MHz",
        "and active software power/thermal limits. The device advertises a maximum",
        "memory clock of 6001 MHz. Results describe this machine's observed state.",
        "No GPU clock, power, firmware, boot, or driver configuration was modified.",
        "", "Rust is the host launch bridge; the device kernels are CUDA C++ compiled",
        "to PTX. The warp implementation removes cross-warp synchronization and",
        "optionally uses aligned vector loads. Numerical references use FP64.",
        "Captured outputs are checked after replay, not just before capture.",
        "", "No full LLM integration or tokens-per-second improvement has been tested.",
        "Native Rust device compilation and a FastKernels adapter remain separate",
        "integration work.",
        "", "## Other experiments", "",
        "`full.json` records the earlier GEMV/GEMM tiling, fused residual RMSNorm,",
        "norm-plus-linear subgraph, and dtype diagnostics. Its source snapshot is",
        "in `iteration1/`. `cublas-challenge-gemv.json` contains all 12 search cases,",
        "including losses. The validation JSON files retain every timing sample.",
    ]
    (ROOT / "results/SUMMARY.md").write_text("\n".join(lines) + "\n")
    observation = ROOT / "results/gpu-observation.json"
    if observation.exists():
        data = json.loads(observation.read_text())
        data["scope"] = "Idle snapshot after the quick benchmark; not the under-load observation"
        observation.write_text(json.dumps(data, indent=2) + "\n")
    print("Saved results/SUMMARY.md")


if __name__ == "__main__":
    main()
