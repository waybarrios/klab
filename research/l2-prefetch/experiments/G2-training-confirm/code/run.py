import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, torch
from bench import measure
from train_prefetch import Stack
OUT = Path(KLAB + "/research/l2-prefetch/experiments/G2-training-confirm/results/g2.json")
SHAPES = [("bf16", d, m) for d in (320, 640, 896) for m in (4, 8)] + [("fp32", 448, m) for m in (4, 8)]
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def cool(limit=60, wait=180):
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        t = int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout)
        if t <= limit: return t
        time.sleep(5)
    return t


def timed(models, x, rounds):
    grads = {n: [] for n in models}
    def step(model, out):
        def run():
            for w in model.ws: w.grad = None
            loss = model(x).float().square().mean(); loss.backward(); model.join()
            out[:] = [w.grad for w in model.ws] + [loss.detach()]
        return run
    fns = {n: step(m, grads[n]) for n, m in models.items()}
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        for f in fns.values(): f()
        torch.cuda.synchronize()
    return measure(fns, rounds=rounds, repeats=1), grads


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    res = {"shapes": []}
    with H.lock():
        res["regime_before"] = H.require_p0()
        for dt, d, m in SHAPES:
            layers = max(16, math.ceil((80 << 20) / (d * d * torch.finfo(DT[dt]).bits // 8)))
            x = torch.randn(m, d, device="cuda").to(DT[dt])
            cands = {"plain": None, **{f"b{b}": {"blocks": b, "kernel": "l2_hint", "stride": 128} for b in (1, 2, 4, 8)}}
            temp_c = cool()
            t, _ = timed({n: Stack(d, layers, DT[dt], p, seed=1) for n, p in cands.items()}, x, 9)
            best = min((n for n in cands if n != "plain"), key=lambda n: t[n]["median_us"])
            lo = H.paired_saving(t["plain"]["samples_us"], t[best]["samples_us"])[1][0]
            plan = cands[best] if lo > 1.0 else None
            torch.cuda.empty_cache()
            temp_e = cool()
            plain = Stack(d, layers, DT[dt], None, seed=2)
            models = {"plain": plain, "plain_aa": plain, "tuned": Stack(d, layers, DT[dt], plan, seed=2)}
            t, grads = timed(models, x, 15)
            s, ci = H.paired_saving(t["plain"]["samples_us"], t["tuned"]["samples_us"])
            aa = H.paired_saving(t["plain"]["samples_us"], t["plain_aa"]["samples_us"])[0]
            same = all(torch.equal(a, b) for a, b in zip(grads["tuned"], grads["plain"]))
            row = {"dtype": dt, "d": d, "m": m, "layers": layers, "plan": plan, "calib_best": best, "calib_lower_ci": lo,
                   "plain_us_per_layer": t["plain"]["median_us"] / layers, "tuned_us_per_layer": t["tuned"]["median_us"] / layers,
                   "saved_pct": s, "ci95": ci, "aa_pct": aa, "bit_identical": same, "temp_calib": temp_c, "temp_eval": temp_e}
            res["shapes"].append(row); OUT.write_text(json.dumps(res, indent=1))
            print(f"{dt} d={d} M={m}: plan {plan and plan['blocks']} saved {s:+.2f}% ci {ci[0]:+.2f}..{ci[1]:+.2f} aa {aa:+.2f}% bits {same} temp {temp_e}C", flush=True)
            del models, plain; torch.cuda.empty_cache()
        res["regime_after"] = H.regime()
    rs = res["shapes"]
    geo = 100 * (1 - math.exp(sum(math.log(r["tuned_us_per_layer"] / r["plain_us_per_layer"]) for r in rs) / len(rs)))
    crit = {"a": geo >= 8, "b": all(r["ci95"][1] >= -1 for r in rs), "c": all(r["bit_identical"] for r in rs)}
    res.update(geomean=geo, criteria=crit); OUT.write_text(json.dumps(res, indent=1))
    print(f"G2 geomean {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")


if __name__ == "__main__":
    main()
