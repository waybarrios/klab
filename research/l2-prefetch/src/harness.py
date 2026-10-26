"""Paired graph timing of chain variants with bit-identity checks and bootstrap CIs."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import json, math, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(KLAB)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research/l2-prefetch/src"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache/cuda"))

import torch
from bench import measure


def regime():
    powerd = subprocess.run(["systemctl", "is-active", "nvidia-powerd"], capture_output=True, text=True).stdout.strip()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.mem,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ac = Path("/sys/class/power_supply/AC0/online").read_text().strip()
    return {"nvidia_powerd": powerd, "smi": smi, "ac_online": ac}


def require_p0():
    r = regime()
    if r["nvidia_powerd"] == "active" or r["ac_online"] != "1":
        raise SystemExit(f"Not in the P0 study regime: {r}")
    return r


def paired_saving(base, cand, draws=2000, seed=7):
    """Median per-round saving (%) and bootstrap 95% CI; rounds are paired by index."""
    ratios = [1 - c / b for b, c in zip(base, cand)]
    rng = random.Random(seed)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    boots = sorted(med([rng.choice(ratios) for _ in ratios]) for _ in range(draws))
    return 100 * med(ratios), (100 * boots[int(0.025 * draws)], 100 * boots[int(0.975 * draws)])


def run_variants(variants, x0, calls, rounds=15, baseline="torch"):
    """variants: name -> callable(y)->y over the whole chain. calls = ops per chain (for per-op us)."""
    outs = {}
    fns = {name: (lambda f=f, name=name: outs.__setitem__(name, f(x0))) for name, f in variants.items()}
    t = measure(fns, rounds=rounds, repeats=1)
    ref = outs[baseline]
    rows = {}
    for name, v in t.items():
        s, ci = paired_saving(t[baseline]["samples_us"], v["samples_us"])
        rows[name] = {"us_per_op": v["median_us"] / calls, "saved_pct": s, "ci95": ci,
                      "bit_identical": bool(torch.equal(outs[name], ref)), "samples_us": v["samples_us"]}
    return rows


def lock():
    from experiment_control import GpuLock
    return GpuLock(ROOT / ".cache/gpu.lock", wait=True)
