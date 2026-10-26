"""H17 step 2 (CONFIRMATORY): 3-way attention dispatch (SDPA / v4 / FlashInfer) vs 2-way and vs base, held-out configs."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import fi_env  # noqa: F401
import flashinfer, harness as H, harness2, models5, attn_dispatch, torch
from l2prefetch import ChainRunner
from run import reference, sdpa_path
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H17-flashinfer/results/step2.json")
MODELS, LENS = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "tinyllama-1.1b"), (320, 960, 1792)


def cool(limit=60, wait=300):
    t0 = time.monotonic()
    while True:
        t = int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout)
        if t <= limit or time.monotonic() - t0 > wait: return t
        time.sleep(5)


torch.set_grad_enabled(False)
res = {"steps": []}
with H.lock():
    res["regime_before"] = H.require_p0()
    for name in MODELS:
        h, kvh = models5.DIMS[name][3], models5.DIMS[name][4]
        for L in LENS:
            cool()
            p2, r2 = attn_dispatch.choose(h, kvh, L)
            p3, r3 = attn_dispatch.choose3(h, kvh, L)
            torch.cuda.empty_cache()
            fi_ok = True
            if p3 == "flashinfer":
                for seed in (1, 2, 3):
                    g = torch.Generator(device="cuda").manual_seed(seed); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
                    q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, L + 1, 64), r(kvh, L + 1, 64)
                    ref = reference(q, kn, vn, K, V, L, h, kvh)
                    Kc, Vc = K.clone(), V.clone(); Kc[:, L] = kn.view(kvh, 64); Vc[:, L] = vn.view(kvh, 64)
                    e = (flashinfer.single_decode_with_kv_cache(q.view(h, 64), Kc, Vc, kv_layout="HND").reshape(-1).double() - ref).abs().max().item()
                    es = (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
                    fi_ok &= e <= 2 * es + 1e-3
            builds = {}
            for impl in {"sdpa", p2, p3}:
                builds[impl] = models5.build(name, 1, kv_len=L, seed=2, impl="compile", attention_impl=impl)
            run = {impl: ChainRunner(b[0], b[1]) for impl, b in builds.items()}
            x0 = builds["sdpa"][2]
            temp = cool()
            rows = harness2.run_variants({"base": run["sdpa"], "base_aa": run["sdpa"], "two_way": run[p2], "three_way": run[p3]}, x0, 1, baseline="base")
            vs2 = harness2.run_variants({"two_way": run[p2], "three_way": run[p3]}, x0, 1, baseline="two_way")
            out = {"model": name, "kv_len": L, "pick2": p2, "pick3": p3, "choose3": r3, "fi_numerics_ok": fi_ok, "temp": temp,
                   "base_us": rows["base"]["us_per_op"], "two_us": rows["two_way"]["us_per_op"], "three_us": rows["three_way"]["us_per_op"],
                   "three_vs_base": rows["three_way"]["saved_pct"], "three_vs_base_ci": rows["three_way"]["ci95"],
                   "two_vs_base": rows["two_way"]["saved_pct"], "three_vs_two": vs2["three_way"]["saved_pct"],
                   "three_vs_two_ci": vs2["three_way"]["ci95"], "aa": rows["base_aa"]["saved_pct"]}
            res["steps"].append(out); OUT.write_text(json.dumps(res, indent=1))
            print(f"{name} L={L}: pick2 {p2} pick3 {p3} | 3-way vs base {out['three_vs_base']:+.2f}% (2-way {out['two_vs_base']:+.2f}%) "
                  f"3-way vs 2-way {out['three_vs_two']:+.2f}% ci {out['three_vs_two_ci'][0]:+.2f}..{out['three_vs_two_ci'][1]:+.2f} "
                  f"aa {out['aa']:+.2f}% fi-num {fi_ok} temp {temp}C", flush=True)
            del builds, run; torch.cuda.empty_cache()
    res["regime_after"] = H.regime()
s = res["steps"]
geo = 100 * (1 - math.exp(sum(math.log(x["three_us"] / x["base_us"]) for x in s) / len(s)))
crit = {"a": all(x["three_vs_two_ci"][1] >= -1 for x in s), "b": geo >= 10, "c": all(x["fi_numerics_ok"] for x in s)}
res.update(geomean_three_vs_base=geo, criteria=crit); OUT.write_text(json.dumps(res, indent=1))
print(f"H17 step2: 3-way vs base geomean {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")
