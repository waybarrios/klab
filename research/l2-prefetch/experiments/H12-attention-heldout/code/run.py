import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, harness2, models3, torch
from decode_attn import decode_attention
from l2prefetch import ChainRunner
from run import reference, sdpa_path
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H12-attention-heldout/results/h12.json")
MODELS, LENS = ("smollm2-360m", "qwen2.5-0.5b", "tinyllama-1.1b"), (256, 1024, 2048)
torch.set_grad_enabled(False)
res = {"numerics": [], "steps": []}
with H.lock():
    res["regime_before"] = H.require_p0()
    num_ok = True
    for name in MODELS:
        _, _, _, h, kvh, _ = models3.DIMS[name]
        for L in LENS:
            for seed in (1, 2, 3):
                g = torch.Generator(device="cuda").manual_seed(seed); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
                q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, 2049, 64), r(kvh, 2049, 64)
                ref = reference(q, kn, vn, K, V, L, h, kvh); Kc, Vc = K.clone(), V.clone()
                e = (decode_attention(q, kn, vn, Kc, Vc, L).double() - ref).abs().max().item()
                es = (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
                ok = e <= 2 * es + 1e-3 and torch.equal(Kc[:, L], kn.view(kvh, 64)) and torch.equal(Vc[:, L], vn.view(kvh, 64))
                num_ok &= ok; res["numerics"].append({"model": name, "len": L, "seed": seed, "err": e, "err_sdpa": es, "ok": ok})
    print("numerics ok:", num_ok, flush=True)
    for name in MODELS:
        for L in LENS:
            a = models3.build(name, 1, kv_len=L, seed=2, attention_impl="sdpa")
            b = models3.build(name, 1, kv_len=L, seed=2, attention_impl="custom")
            sd, cu = ChainRunner(a[0], a[1]), ChainRunner(b[0], b[1])
            rows = harness2.run_variants({"sdpa": sd, "sdpa_aa": sd, "custom": cu}, a[2], 1, baseline="sdpa")
            c = rows["custom"]
            res["steps"].append({"model": name, "kv_len": L, "sdpa_us": rows["sdpa"]["us_per_op"], "custom_us": c["us_per_op"],
                                 "saved_pct": c["saved_pct"], "ci95": c["ci95"], "aa_pct": rows["sdpa_aa"]["saved_pct"]})
            OUT.write_text(json.dumps(res, indent=1))
            print(f"{name} L={L}: sdpa {rows['sdpa']['us_per_op']:.0f}us custom {c['us_per_op']:.0f}us saved {c['saved_pct']:+.2f}% "
                  f"ci {c['ci95'][0]:+.2f}..{c['ci95'][1]:+.2f} aa {rows['sdpa_aa']['saved_pct']:+.2f}%", flush=True)
            del a, b, sd, cu; torch.cuda.empty_cache()
    res["regime_after"] = H.regime()
geo = 100 * (1 - math.exp(sum(math.log(s["custom_us"] / s["sdpa_us"]) for s in res["steps"]) / len(res["steps"])))
crit = {"a": geo >= 10, "b": all(s["ci95"][1] >= -1 for s in res["steps"]), "c": num_ok}
res["geomean_saved_pct"], res["criteria"] = geo, crit; OUT.write_text(json.dumps(res, indent=1))
print(f"H12 geomean {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")
