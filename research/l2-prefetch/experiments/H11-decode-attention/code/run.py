"""H11: numerics, microbenchmark and step-level effect of the custom decode attention."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, models3, torch, torch.nn.functional as F
from decode_attn import decode_attention
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/results/h11.json")
SHAPES = {"smollm2-135m": (9, 3), "qwen2.5-0.5b": (14, 2)}


def reference(q, kn, vn, K, V, L, h, kvh):
    Kf = torch.cat([K[:, :L].double(), kn.view(kvh, 1, 64).double()], 1)
    Vf = torch.cat([V[:, :L].double(), vn.view(kvh, 1, 64).double()], 1)
    qf = q.view(h, 64).double(); g = h // kvh
    out = []
    for i in range(h):
        p = torch.softmax(qf[i] @ Kf[i // g].T / 8.0, -1)
        out.append(p @ Vf[i // g])
    return torch.stack(out).reshape(-1)


def sdpa_path(q, kn, vn, K, V, L, h, kvh):
    K[:, L].copy_(kn.view(kvh, 64)); V[:, L].copy_(vn.view(kvh, 64))
    o = F.scaled_dot_product_attention(q.view(1, h, 1, 64), K[None, :, :L + 1], V[None, :, :L + 1], enable_gqa=True)
    return o.reshape(-1)


def main():
    torch.set_grad_enabled(False)
    report = {"numerics": [], "micro": {}, "step": {}}
    ok = True
    with H.lock():
        report["regime_before"] = H.require_p0()
        for name, (h, kvh) in SHAPES.items():
            for L in (1, 7, 100, 512, 2047):
                for seed in (1, 2, 3):
                    g = torch.Generator(device="cuda").manual_seed(seed)
                    r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
                    q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64)
                    K, V = r(kvh, 2048, 64), r(kvh, 2048, 64)
                    ref = reference(q, kn, vn, K, V, L, h, kvh)
                    Kc, Vc = K.clone(), V.clone()
                    ours = decode_attention(q, kn, vn, Kc, Vc, L).double()
                    written = torch.equal(Kc[:, L], kn.view(kvh, 64)) and torch.equal(Vc[:, L], vn.view(kvh, 64)) \
                        and torch.equal(Kc[:, :L], K[:, :L])
                    sd = sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double()
                    e_o, e_s = (ours - ref).abs().max().item(), (sd - ref).abs().max().item()
                    passed = e_o <= 2 * e_s + 1e-3 and written
                    ok &= passed
                    report["numerics"].append({"model": name, "len": L, "seed": seed, "err_custom": e_o, "err_sdpa": e_s,
                                               "cache_written_exactly": written, "passed": passed})
            worst = max((n for n in report["numerics"] if n["model"] == name), key=lambda n: n["err_custom"] - 2 * n["err_sdpa"])
            print(f"numerics {name}: worst custom err {worst['err_custom']:.2e} vs sdpa {worst['err_sdpa']:.2e} (len {worst['len']})", flush=True)
        print(f"(a) numerics all passed: {ok}", flush=True)
        for name, (h, kvh) in SHAPES.items():
            L, layers = 512, 30
            qs = torch.randn(layers, h * 64, device="cuda").to(torch.bfloat16)
            ks = torch.randn(layers, kvh * 64, device="cuda").to(torch.bfloat16)
            Ks = torch.randn(layers, kvh, L + 1, 64, device="cuda").to(torch.bfloat16)
            Vs = torch.randn_like(Ks)
            def sd_chain(x, qs=qs, ks=ks, Ks=Ks, Vs=Vs):
                for i in range(layers):
                    x = sdpa_path(qs[i] + x[: h * 64] * 0, ks[i], ks[i], Ks[i], Vs[i], L, h, kvh)
                return x
            def cu_chain(x, qs=qs, ks=ks, Ks=Ks, Vs=Vs):
                for i in range(layers):
                    x = decode_attention(qs[i] + x[: h * 64] * 0, ks[i], ks[i], Ks[i], Vs[i], L)
                return x
            x0 = torch.zeros(h * 64, device="cuda", dtype=torch.bfloat16)
            rows = harness2.run_variants({"sdpa": sd_chain, "sdpa_aa": sd_chain, "custom": cu_chain}, x0, layers, baseline="sdpa")
            sp = rows["sdpa"]["us_per_op"] / rows["custom"]["us_per_op"]
            report["micro"][name] = {"sdpa_us": rows["sdpa"]["us_per_op"], "custom_us": rows["custom"]["us_per_op"],
                                     "speedup": sp, "aa_pct": rows["sdpa_aa"]["saved_pct"]}
            print(f"(b) micro {name}: sdpa {rows['sdpa']['us_per_op']:.2f}us custom {rows['custom']['us_per_op']:.2f}us "
                  f"speedup {sp:.2f}x (aa {rows['sdpa_aa']['saved_pct']:+.2f}%)  [includes the x*0 add per call]", flush=True)
            del qs, ks, Ks, Vs; torch.cuda.empty_cache()
        for name in SHAPES:
            arms, keep = {}, []
            for arm, impl in (("sdpa", "sdpa"), ("sdpa_aa", "sdpa"), ("custom", "custom")):
                ops, reads, x0, period, k, wb, kvb = models3.build(name, 1, seed=2, attention_impl=impl)
                arms[arm] = ChainRunner(ops, reads); keep.append(k)
            rows = harness2.run_variants(arms, x0, 1, baseline="sdpa")
            r = rows["custom"]
            report["step"][name] = {"sdpa_us": rows["sdpa"]["us_per_op"], "custom_us": r["us_per_op"], "saved_pct": r["saved_pct"],
                                    "ci95": r["ci95"], "aa_pct": rows["sdpa_aa"]["saved_pct"]}
            print(f"(c) step {name}: sdpa {rows['sdpa']['us_per_op']:.0f}us custom {r['us_per_op']:.0f}us saved {r['saved_pct']:+.2f}% "
                  f"ci {r['ci95'][0]:+.2f}..{r['ci95'][1]:+.2f} aa {rows['sdpa_aa']['saved_pct']:+.2f}%", flush=True)
            del arms, keep; torch.cuda.empty_cache()
        report["regime_after"] = H.regime()
    m = report["micro"]; s = report["step"]["smollm2-135m"]
    report["criteria"] = {"a": ok, "b": all(v["speedup"] >= 3 for v in m.values()),
                          "c": s["saved_pct"] >= 15 and s["ci95"][0] > 10}
    OUT.write_text(json.dumps(report, indent=1))
    print("H11 criteria:", report["criteria"], "->", "SUPPORTED" if all(report["criteria"].values()) else "NOT SUPPORTED")


if __name__ == "__main__":
    main()
