"""harness.run_variants plus a time-based warm-up so short graphs are not timed during the P8->P0 clock ramp.

harness.py stays unchanged because earlier protocols hash it.
"""
import time

import torch

import harness as H


def warm(variants, x0, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for f in variants.values():
            f(x0)
        torch.cuda.synchronize()


def run_variants(variants, x0, calls, rounds=15, baseline="torch", warm_seconds=2.0):
    with torch.no_grad():
        warm(variants, x0, warm_seconds)
    return H.run_variants(variants, x0, calls, rounds=rounds, baseline=baseline)
