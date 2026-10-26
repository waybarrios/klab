"""EXPLORATORY: v4 vs v5 vs SDPA path, kernel alone, across GQA groups and KV lengths (30 layer caches)."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, harness2, torch, torch.nn.functional as F
from decode_attn import decode_attention
from run import reference, sdpa_path
torch.set_grad_enabled(False)
with H.lock():
    H.require_p0()
    for (h, kvh) in ((9, 3), (14, 2), (32, 4)):
        worst = 0.0
        for L in (1, 63, 64, 65, 300, 2047):
            g = torch.Generator(device="cuda").manual_seed(4); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
            q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, 2048, 64), r(kvh, 2048, 64)
            ref = reference(q, kn, vn, K, V, L, h, kvh); Kc, Vc = K.clone(), V.clone()
            for _ in range(2): o = decode_attention(q, kn, vn, Kc, Vc, L, kernel="decode_attn_v5_bf16").double()
            e, es = (o - ref).abs().max().item(), (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
            assert e <= 2 * es + 1e-3 and torch.equal(Kc[:, L], kn.view(kvh, 64)), (h, kvh, L, e, es)
        for L in (256, 512, 1024, 2048):
            n = 24
            qs = torch.randn(n, h * 64, device="cuda").to(torch.bfloat16); ks = torch.randn(n, kvh * 64, device="cuda").to(torch.bfloat16)
            Ks = torch.randn(n, kvh, L + 1, 64, device="cuda").to(torch.bfloat16); Vs = torch.randn_like(Ks)
            outs = torch.empty(n, h * 64, device="cuda", dtype=torch.bfloat16)
            def sdpa(_):
                for i in range(n):
                    Ks[i][:, L].copy_(ks[i].view(kvh, 64)); Vs[i][:, L].copy_(ks[i].view(kvh, 64))
                    F.scaled_dot_product_attention(qs[i].view(1, h, 1, 64), Ks[i][None], Vs[i][None], enable_gqa=True)
                return outs
            mk = lambda k: (lambda _: [decode_attention(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, out=outs[i], kernel=k) for i in range(n)] and outs)
            rows = harness2.run_variants({"sdpa": sdpa, "v4": mk("decode_attn_v4_bf16"), "v5": mk("decode_attn_v5_bf16")}, None, n, baseline="sdpa")
            print(f"h={h} kvh={kvh} L={L}: sdpa {rows['sdpa']['us_per_op']:.2f}us v4 {rows['v4']['us_per_op']:.2f}us v5 {rows['v5']['us_per_op']:.2f}us "
                  f"| v5 {rows['sdpa']['us_per_op']/rows['v5']['us_per_op']:.2f}x", flush=True)
            del qs, ks, Ks, Vs, outs; torch.cuda.empty_cache()
