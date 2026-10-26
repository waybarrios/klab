"""H7 confirmatory run on decoder-layer sequences."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import autotune, models
import torch
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H7-decoder-layers/results/decoder.json")
CONFIGS = [(n, f, m) for n in models.DIMS for f in (False, True) for m in (1, 4)]


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() and OUT.stat().st_size else []
    done = {(r["model"], r["fused"], r["m"]) for r in results}
    with H.lock():
        for name, fused, m in CONFIGS:
            if (name, fused, m) in done:
                continue
            before = H.require_p0()
            print(f"== {name} fused={fused} M={m}", flush=True)
            t0 = time.monotonic()
            ops, reads, x0, period, keep = models.build(name, m, fused, seed=1)
            plans, hist = autotune.tune_positions(ops, reads, x0, period)
            calib = time.monotonic() - t0
            del ops, reads, keep; torch.cuda.empty_cache()
            ops, reads, x0, period, keep = models.build(name, m, fused, seed=2)
            full = [plans[i % period] for i in range(len(ops))]
            variants = {"torch": ChainRunner(ops, reads), "torch_aa": ChainRunner(ops, reads),
                        "tuned": ChainRunner(ops, reads, full if any(plans) else None)}
            rows = H.run_variants(variants, x0, 1)  # per decode step
            results.append({"model": name, "fused": fused, "m": m, "plans": plans, "history": hist,
                            "calibration_seconds": calib, "regime_before": before, "regime_after": H.regime(),
                            "variants": rows})
            OUT.write_text(json.dumps(results, indent=1))
            r = rows["tuned"]
            print(f"   step torch={rows['torch']['us_per_op']:.1f}us tuned={r['us_per_op']:.1f}us saved {r['saved_pct']:+.2f}% "
                  f"ci={r['ci95'][0]:+.2f}..{r['ci95'][1]:+.2f} aa={rows['torch_aa']['saved_pct']:+.2f} bit={r['bit_identical']} "
                  f"calib={calib:.0f}s", flush=True)
            del ops, reads, keep, variants; torch.cuda.empty_cache()
    ratios = [r["variants"]["tuned"]["us_per_op"] / r["variants"]["torch"]["us_per_op"] for r in results]
    geo = 100 * (1 - math.exp(sum(map(math.log, ratios)) / len(ratios)))
    a, b = geo >= 8, all(r["variants"]["tuned"]["ci95"][1] >= -1 for r in results)
    c = all(v["bit_identical"] for r in results for v in r["variants"].values())
    print(f"\nH7: n={len(results)} geomean {geo:.2f}% a={a} b={b} c={c} -> {'SUPPORTED' if a and b and c else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
