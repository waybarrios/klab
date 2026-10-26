"""H5: stack cross-op prefetch with klab FP32 small-GEMV kernels."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, autotune
import torch, torch.nn.functional as F
import kernel_lab
from bench import measure
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H5-stack-klab/results/stack.json")
CONFIGS = [("tiny", 32, 4), ("tiny", 64, 4), ("tiny", 128, 1), ("warp", 32, 4), ("warp", 128, 4)]


def chain(d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    layers = max(16, math.ceil((80 << 20) / (d * d * 4)))
    W = torch.randn(layers, d, d, generator=g, device="cuda") / math.sqrt(d)
    return W, W.unbind(), layers


def klab_ops(ws, kind, t, v):
    lab = kernel_lab.default_lab()
    f = lab.tiny_matmul if kind == "tiny" else lab.warp_matmul
    return [lambda y, w=w: f(y, w, t, v) for w in ws]


def timed(variants, x0, calls, rounds):
    outs = {}
    fns = {n: (lambda f=f, n=n: outs.__setitem__(n, f(x0))) for n, f in variants.items()}
    t = measure(fns, rounds=rounds, repeats=1)
    return t, outs


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with H.lock():
        for d in (256, 384, 512):
            for m in (1, 4):
                before = H.require_p0()
                x0 = torch.randn(m, d, device="cuda") / math.sqrt(d)
                W, ws, layers = chain(d, 1)
                reads = [[w] for w in ws]
                torch_ops = [lambda y, w=w: F.linear(y, w) for w in ws]
                cands = {f"{k}_t{t}_v{v}": (k, t, v) for k, t, v in CONFIGS if k == "warp" or d in (64, 128, 256, 512, 1024)}
                tc, _ = timed({n: ChainRunner(klab_ops(ws, *c), reads) for n, c in cands.items()}, x0, layers, 9)
                kbest = min(cands, key=lambda n: tc[n]["median_us"])
                kops = klab_ops(ws, *cands[kbest])
                pf_t, rep_t = autotune.tune(torch_ops, reads, x0, d * d * 4)
                pf_k, rep_k = autotune.tune(kops, reads, x0, d * d * 4)
                del W, ws, reads, torch_ops, kops; torch.cuda.empty_cache()
                W, ws, layers = chain(d, 2)
                reads = [[w] for w in ws]
                torch_ops = [lambda y, w=w: F.linear(y, w) for w in ws]
                kops = klab_ops(ws, *cands[kbest])
                arms = {"torch": ChainRunner(torch_ops, reads), "torch_aa": ChainRunner(torch_ops, reads),
                        "torch_pf": ChainRunner(torch_ops, reads, [pf_t] * layers if pf_t else None),
                        "klab": ChainRunner(kops, reads), "klab_pf": ChainRunner(kops, reads, [pf_k] * layers if pf_k else None)}
                t, outs = timed(arms, x0, layers, 15)
                rows = {}
                for n in arms:
                    s, ci = H.paired_saving(t["torch"]["samples_us"], t[n]["samples_us"])
                    rows[n] = {"us_per_op": t[n]["median_us"] / layers, "saved_pct": s, "ci95": ci}
                rel = (outs["klab"].double() - outs["torch"].double()).norm() / outs["torch"].double().norm()
                res = {"d": d, "m": m, "layers": layers, "klab_config": kbest, "pf_torch": pf_t, "pf_klab": pf_k,
                       "bit_torch_pair": bool(torch.equal(outs["torch_pf"], outs["torch"])),
                       "bit_klab_pair": bool(torch.equal(outs["klab_pf"], outs["klab"])), "klab_vs_torch_rel_l2": float(rel),
                       "regime_before": before, "regime_after": H.regime(), "arms": rows}
                results.append(res); OUT.write_text(json.dumps(results, indent=1))
                print(f"fp32 d={d} M={m} torch={rows['torch']['us_per_op']:.2f}us | torch_pf {rows['torch_pf']['saved_pct']:+.1f}% "
                      f"klab({kbest}) {rows['klab']['saved_pct']:+.1f}% klab_pf {rows['klab_pf']['saved_pct']:+.1f}% "
                      f"aa {rows['torch_aa']['saved_pct']:+.2f}% bits {res['bit_torch_pair']}/{res['bit_klab_pair']} rel={rel:.1e} "
                      f"pf_t={pf_t and (pf_t['blocks'], pf_t['nbytes'] >> 10)} pf_k={pf_k and (pf_k['blocks'], pf_k['nbytes'] >> 10)}", flush=True)
                del W, ws, reads, torch_ops, kops, arms, outs; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
