"""CPU-only percentage summaries of saved, paired GPU measurements."""

import math


def percentage_gains(baseline_us, kernel_us):
    if not all(math.isfinite(value) and value > 0 for value in (baseline_us, kernel_us)):
        raise ValueError("GPU times must be finite and positive")
    return {
        "latency_reduction_pct": 100 * (1 - kernel_us / baseline_us),
        "throughput_gain_pct": 100 * (baseline_us / kernel_us - 1),
    }


def case_metrics(case, condition="cold", method="lossless_fp32"):
    if condition not in ("warm", "cold"):
        raise ValueError("Choose warm or cold timings")
    methods = case["methods"]
    vendors = [name for name in methods if name.startswith(("torch_", "cublas_"))]
    if not vendors:
        raise ValueError("No FP32 vendor baseline in this case")
    best = min(vendors, key=lambda name: methods[name][condition]["median_us"])
    baseline_us = methods[best][condition]["median_us"]
    kernel_us = methods[method][condition]["median_us"]
    return {"baseline_method": best, "baseline_us": baseline_us, "kernel_us": kernel_us,
            **percentage_gains(baseline_us, kernel_us)}
