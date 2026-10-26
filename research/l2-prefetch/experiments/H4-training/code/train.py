"""H4 (EXPLORATORY): forward+backward of a deep Linear+GELU stack in a CUDA graph, with/without prefetch."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import torch
from bench import measure
from train_prefetch import Stack

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H4-training/results/train.json")
CASES = [(dt, d, m) for dt in ("bf16",) for d in (256, 512, 768) for m in (8, 32, 128)] + [("fp32", 384, 8), ("fp32", 384, 32)]
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def step_fn(model, x, grads_out):
    def run():
        for w in model.ws:
            w.grad = None
        loss = model(x).float().square().mean()
        loss.backward()
        model.join()
        grads_out[:] = [w.grad for w in model.ws] + [loss.detach()]
    return run


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() and OUT.stat().st_size else []
    done = {(r["dtype"], r["d"], r["m"]) for r in results}
    with H.lock():
        for dt, d, m in CASES:
            if (dt, d, m) in done:
                continue
            before = H.require_p0()
            nbytes = d * d * torch.finfo(DT[dt]).bits // 8
            layers = max(16, math.ceil((80 << 20) / nbytes))
            x = torch.randn(m, d, device="cuda").to(DT[dt])
            plans = {"torch": None, "torch_aa": None}
            for b in (1, 2, 4, 8):
                plans[f"b{b}"] = {"blocks": b, "kernel": "l2_hint", "stride": 128}
            models_, grads, fns = {}, {}, {}
            for name, plan in plans.items():
                models_[name] = Stack(d, layers, DT[dt], plan, seed=3)
                grads[name] = []
                fns[name] = step_fn(models_[name], x, grads[name])
            t = measure(fns, rounds=15, repeats=1)
            ref = grads["torch"]
            rows = {}
            for name in plans:
                s, ci = H.paired_saving(t["torch"]["samples_us"], t[name]["samples_us"])
                same = all(torch.equal(a, b) for a, b in zip(grads[name], ref))
                rows[name] = {"us_per_layer": t[name]["median_us"] / layers, "saved_pct": s, "ci95": ci, "bit_identical": same}
            results.append({"dtype": dt, "d": d, "m": m, "layers": layers, "bytes": nbytes, "regime_before": before,
                            "regime_after": H.regime(), "variants": rows})
            OUT.write_text(json.dumps(results, indent=1))
            print(f"{dt} d={d} M={m} fwd+bwd torch={rows['torch']['us_per_layer']:.2f}us/layer | " +
                  " ".join(f"{k}:{v['saved_pct']:+.1f}{'' if v['bit_identical'] else '!'}" for k, v in rows.items() if k != "torch"), flush=True)
            del models_, grads, fns; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
