"""H2 confirmatory run of the frozen static policy on held-out shapes."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import policy
import torch, torch.nn.functional as F
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H2-heldout-policy/results/heldout.json")
SHAPES = [("bf16", d, m) for d in (384, 640, 896, 1280) for m in (2, 8, 32)] + \
         [("fp32", d, m) for d in (384, 896) for m in (1, 8)]


def main():
    torch.set_grad_enabled(False)
    mm = torch.backends.cuda.matmul
    mm.allow_tf32 = False; mm.allow_bf16_reduced_precision_reduction = False; mm.allow_fp16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["dtype"], r["d"], r["m"]) for r in results}
    with H.lock():
        for dt, d, m in SHAPES:
            if (dt, d, m) in done:
                continue
            before = H.require_p0()
            tdt = torch.bfloat16 if dt == "bf16" else torch.float32
            nbytes = d * d * (2 if dt == "bf16" else 4)
            layers = max(16, math.ceil((80 << 20) / nbytes))
            W = torch.randn(layers, d, d, dtype=tdt, device="cuda") / math.sqrt(d)
            ws = W.unbind()
            ops = [lambda y, w=w: F.linear(y, w) for w in ws]
            reads = [[w] for w in ws]
            p = policy.plan(nbytes, m)
            variants = {"torch": ChainRunner(ops, reads), "torch_aa": ChainRunner(ops, reads),
                        "policy": ChainRunner(ops, reads, [p] * layers if p else None)}
            for b in (1, 2, 4, 8):
                variants[f"x_hint_b{b}"] = ChainRunner(ops, reads, [dict(blocks=b, kernel="l2_hint", stride=128)] * layers)
            x0 = torch.randn(m, d, dtype=tdt, device="cuda")
            rows = H.run_variants(variants, x0, layers)
            results.append({"dtype": dt, "d": d, "m": m, "bytes": nbytes, "layers": layers, "plan": p,
                            "regime_before": before, "regime_after": H.regime(), "variants": rows})
            OUT.write_text(json.dumps(results, indent=1))
            r = rows["policy"]
            print(f"{dt} d={d:5d} M={m:3d} plan={p and p['blocks']} torch={rows['torch']['us_per_op']:.3f}us "
                  f"policy {r['saved_pct']:+.2f}% ci={r['ci95'][0]:+.2f}..{r['ci95'][1]:+.2f} bit={r['bit_identical']} | "
                  + " ".join(f"{k}:{v['saved_pct']:+.1f}" for k, v in rows.items() if k.startswith(("x_", "torch_aa"))), flush=True)
            del W, ws, ops, reads, variants; torch.cuda.empty_cache()
    ratios = [r["variants"]["policy"]["us_per_op"] / r["variants"]["torch"]["us_per_op"] for r in results]
    geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
    a = (1 - geo) * 100 >= 8
    b = all(r["variants"]["policy"]["ci95"][1] >= -1 for r in results)
    c = all(v["bit_identical"] for r in results for v in r["variants"].values())
    print(f"\nH2: n={len(results)} geomean saving {100*(1-geo):.2f}% (a={a}) no-regression b={b} bit-identical c={c} -> "
          f"{'SUPPORTED' if a and b and c else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
