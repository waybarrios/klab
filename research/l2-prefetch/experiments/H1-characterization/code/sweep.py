"""H1 characterization sweep (protocol sha256 c92ec6b9...)."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import torch, torch.nn.functional as F
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H1-characterization/results/sweep.json")


def consumer_kernels(x, w):
    from torch.profiler import profile, ProfilerActivity
    import tempfile, os
    F.linear(x, w); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        F.linear(x, w); torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json", dir=KLAB + "/.scratch")
    prof.export_chrome_trace(path)
    events = json.load(open(path))["traceEvents"]; os.unlink(path)
    return [{"name": e["name"][:90], **{k: e["args"].get(k) for k in ("grid", "block", "blocks per SM", "est. achieved occupancy %")}}
            for e in events if e.get("cat") == "kernel"]


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["d"], r["m"]) for r in results}
    variants_spec = {"torch": None, "torch_aa": None}
    for b in (1, 2, 4, 8, 12, 23):
        variants_spec[f"hint_b{b}"] = dict(blocks=b, kernel="l2_hint", stride=128)
    for b in (4, 12):
        variants_spec[f"last_b{b}"] = dict(blocks=b, kernel="l2_hint_last", stride=128)
    variants_spec["load_b12"] = dict(blocks=12, kernel="l2_load", stride=32)
    with H.lock():
        for d in (256, 512, 768, 1024, 1536):
            nbytes = d * d * 2
            layers = max(16, math.ceil((80 << 20) / nbytes))
            W = torch.randn(layers, d, d, dtype=torch.bfloat16, device="cuda") / math.sqrt(d)
            ws = W.unbind()
            ops = [lambda y, w=w: F.linear(y, w) for w in ws]
            reads = [[w] for w in ws]
            for m in (1, 4, 16, 64):
                if (d, m) in done:
                    continue
                before = H.require_p0()
                x0 = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
                kernels = consumer_kernels(x0, ws[0])
                variants = {name: ChainRunner(ops, reads, [plan] * layers if plan else None) for name, plan in variants_spec.items()}
                rows = H.run_variants(variants, x0, layers)
                after = H.regime()
                results.append({"d": d, "m": m, "bytes": nbytes, "layers": layers, "kernels": kernels,
                                "regime_before": before, "regime_after": after, "variants": rows})
                OUT.write_text(json.dumps(results, indent=1))
                best = max((k for k in rows if k not in ("torch", "torch_aa")), key=lambda k: rows[k]["saved_pct"])
                print(f"d={d:5d} M={m:3d} torch={rows['torch']['us_per_op']:.3f}us aa={rows['torch_aa']['saved_pct']:+.2f}% "
                      f"best={best} {rows[best]['saved_pct']:+.2f}% ci={rows[best]['ci95'][0]:+.1f}..{rows[best]['ci95'][1]:+.1f} "
                      f"grid={[k['grid'] for k in kernels]} | " +
                      " ".join(f"{k}:{v['saved_pct']:+.1f}{'' if v['bit_identical'] else '!'}" for k, v in rows.items() if k != "torch"),
                      flush=True)
            del W, ws, ops, reads; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
