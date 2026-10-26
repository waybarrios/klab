import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, harness2, models3, attn_dispatch, torch
from decode_attn import decode_attention
from l2prefetch import ChainRunner
from run import reference, sdpa_path
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H14-attention-dispatch/results/h14.json")
MODELS, LENS = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "tinyllama-1.1b"), (128, 384, 768, 1536)
torch.set_grad_enabled(False)
res = {"numerics": [], "steps": []}
with H.lock():
    res["regime_before"] = H.require_p0()
    ok_all = True
    for name in MODELS:
        h, kvh = models3.DIMS[name][3], models3.DIMS[name][4]
        for L in LENS:
            for seed in (1, 2, 3):
                g = torch.Generator(device="cuda").manual_seed(seed); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
                q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, 1600, 64), r(kvh, 1600, 64)
                ref = reference(q, kn, vn, K, V, L, h, kvh); Kc, Vc = K.clone(), V.clone()
                e = (decode_attention(q, kn, vn, Kc, Vc, L).double() - ref).abs().max().item()
                es = (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
                ok = e <= 2 * es + 1e-3 and torch.equal(Kc[:, L], kn.view(kvh, 64)) and torch.equal(Vc[:, L], vn.view(kvh, 64))
                ok_all &= ok; res["numerics"].append({"model": name, "len": L, "seed": seed, "err": e, "err_sdpa": es, "ok": ok})
    print("numerics ok:", ok_all, flush=True)
    for name in MODELS:
        h, kvh = models3.DIMS[name][3], models3.DIMS[name][4]
        for L in LENS:
            pick, rep = attn_dispatch.choose(h, kvh, L); torch.cuda.empty_cache()
            a = models3.build(name, 1, kv_len=L, seed=2, attention_impl="sdpa")
            arms = {"sdpa": ChainRunner(a[0], a[1])}
            arms["sdpa_aa"] = arms["sdpa"]
            if pick == "custom":
                b = models3.build(name, 1, kv_len=L, seed=2, attention_impl="custom"); arms["dispatch"] = ChainRunner(b[0], b[1])
            else:
                b = None; arms["dispatch"] = arms["sdpa"]
            rows = harness2.run_variants(arms, a[2], 1, baseline="sdpa")
            d = rows["dispatch"]
            res["steps"].append({"model": name, "kv_len": L, "pick": pick, "choose": rep, "sdpa_us": rows["sdpa"]["us_per_op"],
                                 "dispatch_us": d["us_per_op"], "saved_pct": d["saved_pct"], "ci95": d["ci95"], "aa_pct": rows["sdpa_aa"]["saved_pct"]})
            OUT.write_text(json.dumps(res, indent=1))
            print(f"{name} L={L}: pick {pick} (micro {rep['saved_pct']:+.1f}%) step sdpa {rows['sdpa']['us_per_op']:.0f}us -> {d['us_per_op']:.0f}us "
                  f"saved {d['saved_pct']:+.2f}% ci {d['ci95'][0]:+.2f}..{d['ci95'][1]:+.2f} aa {rows['sdpa_aa']['saved_pct']:+.2f}%", flush=True)
            del a, b, arms; torch.cuda.empty_cache()
    res["regime_after"] = H.regime()
geo = 100 * (1 - math.exp(sum(math.log(s["dispatch_us"] / s["sdpa_us"]) for s in res["steps"]) / len(res["steps"])))
crit = {"a": geo >= 4, "b": all(s["ci95"][1] >= -1 for s in res["steps"]), "c": ok_all}
res["geomean_saved_pct"], res["criteria"] = geo, crit; OUT.write_text(json.dumps(res, indent=1))
print(f"H14 geomean {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")
