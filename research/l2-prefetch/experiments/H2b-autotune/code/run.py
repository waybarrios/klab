"""H2b confirmatory: autotune on calibration weights, evaluate on fresh weights."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import autotune
import torch, torch.nn.functional as F
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H2b-autotune/results/autotune.json")
SHAPES = [("bf16", d, m) for d in (320, 576, 832, 1152) for m in (1, 3, 12)] + \
         [("fp32", d, m) for d in (448, 704) for m in (2, 24)] + [("fp16", d, m) for d in (512, 960) for m in (1, 8)]
DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def chain(dt, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nbytes = d * d * torch.finfo(DT[dt]).bits // 8
    layers = max(16, math.ceil((80 << 20) / nbytes))
    W = torch.randn(layers, d, d, generator=g, device="cuda").div_(math.sqrt(d)).to(DT[dt])
    ws = W.unbind()
    return W, [lambda y, w=w: F.linear(y, w) for w in ws], [[w] for w in ws], nbytes, layers


def main():
    torch.set_grad_enabled(False)
    mm = torch.backends.cuda.matmul
    mm.allow_tf32 = False; mm.allow_bf16_reduced_precision_reduction = False; mm.allow_fp16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() and OUT.stat().st_size else []
    done = {(r["dtype"], r["d"], r["m"]) for r in results}
    with H.lock():
        for dt, d, m in SHAPES:
            if (dt, d, m) in done:
                continue
            before = H.require_p0()
            x0 = torch.randn(m, d, device="cuda").to(DT[dt])
            t0 = time.monotonic()
            Wc, ops, reads, nbytes, layers = chain(dt, d, seed=1)
            plan, report = autotune.tune(ops, reads, x0, nbytes)
            calib_s = time.monotonic() - t0
            del Wc, ops, reads; torch.cuda.empty_cache()
            We, ops, reads, nbytes, layers = chain(dt, d, seed=2)
            variants = {"torch": ChainRunner(ops, reads), "torch_aa": ChainRunner(ops, reads),
                        "autotuned": ChainRunner(ops, reads, [plan] * layers if plan else None)}
            rows = H.run_variants(variants, x0, layers)
            results.append({"dtype": dt, "d": d, "m": m, "bytes": nbytes, "layers": layers, "plan": plan,
                            "calibration": report, "calibration_seconds": calib_s,
                            "regime_before": before, "regime_after": H.regime(), "variants": rows})
            OUT.write_text(json.dumps(results, indent=1))
            r = rows["autotuned"]
            print(f"{dt} d={d:5d} M={m:3d} torch={rows['torch']['us_per_op']:.3f}us pick={report['best']} kept={report['kept']} "
                  f"eval {r['saved_pct']:+.2f}% ci={r['ci95'][0]:+.2f}..{r['ci95'][1]:+.2f} aa={rows['torch_aa']['saved_pct']:+.2f} "
                  f"bit={r['bit_identical']} calib={calib_s:.1f}s", flush=True)
            del We, ops, reads, variants; torch.cuda.empty_cache()
    ratios = [r["variants"]["autotuned"]["us_per_op"] / r["variants"]["torch"]["us_per_op"] for r in results]
    geo = 100 * (1 - math.exp(sum(map(math.log, ratios)) / len(ratios)))
    a, b = geo >= 8, all(r["variants"]["autotuned"]["ci95"][1] >= -1 for r in results)
    c = all(v["bit_identical"] for r in results for v in r["variants"].values())
    print(f"\nH2b: n={len(results)} geomean {geo:.2f}% a={a} b={b} c={c} -> {'SUPPORTED' if a and b and c else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
