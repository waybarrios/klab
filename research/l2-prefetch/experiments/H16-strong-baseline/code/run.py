import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, harness2, models4, attn_dispatch, autotune, torch
from decode_attn import decode_attention
from l2prefetch import ChainRunner
from run import reference, sdpa_path
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H16-strong-baseline/results/h16.json")
MODELS, LENS = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "tinyllama-1.1b"), (192, 640, 1280)
torch.set_grad_enabled(False)
res = {"numerics": [], "steps": []}
with H.lock():
    res["regime_before"] = H.require_p0()
    ok_all = True
    for name in MODELS:
        h, kvh = models4.DIMS[name][3], models4.DIMS[name][4]
        for L in LENS:
            for seed in (1, 2, 3):
                g = torch.Generator(device="cuda").manual_seed(seed); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
                q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, 1300, 64), r(kvh, 1300, 64)
                ref = reference(q, kn, vn, K, V, L, h, kvh); Kc, Vc = K.clone(), V.clone()
                e = (decode_attention(q, kn, vn, Kc, Vc, L).double() - ref).abs().max().item()
                es = (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
                ok = e <= 2 * es + 1e-3 and torch.equal(Kc[:, L], kn.view(kvh, 64)) and torch.equal(Vc[:, L], vn.view(kvh, 64))
                ok_all &= ok
    print("numerics ok:", ok_all, flush=True)
    for name in MODELS:
        h, kvh = models4.DIMS[name][3], models4.DIMS[name][4]
        for L in LENS:
            pick, rep = attn_dispatch.choose(h, kvh, L); torch.cuda.empty_cache()
            c = models4.build(name, 1, kv_len=L, seed=1, impl="compile", attention_impl=pick)
            plans, hist = autotune.tune_positions(c[0], c[1], c[2], c[3], log=lambda *_: None)
            del c; torch.cuda.empty_cache()
            b = models4.build(name, 1, kv_len=L, seed=2, impl="compile", attention_impl="sdpa")
            arms = {"base": ChainRunner(b[0], b[1])}; arms["base_aa"] = arms["base"]
            d = b if pick == "sdpa" else models4.build(name, 1, kv_len=L, seed=2, impl="compile", attention_impl="custom")
            arms["dispatch"] = ChainRunner(d[0], d[1])
            full = [plans[i % d[3]] for i in range(len(d[0]))]
            arms["full"] = ChainRunner(d[0], d[1], full if any(plans) else None)
            rows = harness2.run_variants(arms, b[2], 1, baseline="base")
            f, dd = rows["full"], rows["dispatch"]
            res["steps"].append({"model": name, "kv_len": L, "pick": pick, "plans": plans, "base_us": rows["base"]["us_per_op"],
                                 "dispatch_us": dd["us_per_op"], "full_us": f["us_per_op"], "dispatch_saved": dd["saved_pct"],
                                 "full_saved": f["saved_pct"], "full_ci": f["ci95"], "aa": rows["base_aa"]["saved_pct"],
                                 "full_vs_dispatch_bit_identical": f["bit_identical"] == dd["bit_identical"]})
            OUT.write_text(json.dumps(res, indent=1))
            print(f"{name} L={L}: pick {pick} prefetch-positions {sum(p is not None for p in plans)}/4 | base {rows['base']['us_per_op']:.0f}us "
                  f"dispatch {dd['saved_pct']:+.2f}% full {f['saved_pct']:+.2f}% (ci {f['ci95'][0]:+.2f}..{f['ci95'][1]:+.2f}) aa {rows['base_aa']['saved_pct']:+.2f}%", flush=True)
            del b, d, arms; torch.cuda.empty_cache()
    res["regime_after"] = H.regime()
geo = 100 * (1 - math.exp(sum(math.log(s["full_us"] / s["base_us"]) for s in res["steps"]) / len(res["steps"])))
geo_d = 100 * (1 - math.exp(sum(math.log(s["dispatch_us"] / s["base_us"]) for s in res["steps"]) / len(res["steps"])))
crit = {"a": geo >= 5, "b": all(s["full_ci"][1] >= -1 for s in res["steps"]), "c": ok_all}
res.update(geomean_full=geo, geomean_dispatch=geo_d, criteria=crit); OUT.write_text(json.dumps(res, indent=1))
print(f"H16 geomean full {geo:.2f}% (dispatch only {geo_d:.2f}%) criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")
