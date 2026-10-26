"""EXPLORATORY: per-call time of each attention implementation alone (30 distinct layer caches, len 512)."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, torch, torch.nn.functional as F
from decode_attn import decode_attention
torch.set_grad_enabled(False)
with H.lock():
    H.require_p0()
    for name, (h, kvh) in {"smollm2-135m": (9, 3), "qwen2.5-0.5b": (14, 2)}.items():
        L, n = 512, 30
        qs = torch.randn(n, h * 64, device="cuda").to(torch.bfloat16); ks = torch.randn(n, kvh * 64, device="cuda").to(torch.bfloat16)
        Ks = torch.randn(n, kvh, L + 1, 64, device="cuda").to(torch.bfloat16); Vs = torch.randn_like(Ks)
        outs = torch.empty(n, h * 64, device="cuda", dtype=torch.bfloat16)
        def sdpa(_):
            for i in range(n):
                Ks[i][:, L].copy_(ks[i].view(kvh, 64)); Vs[i][:, L].copy_(ks[i].view(kvh, 64))
                F.scaled_dot_product_attention(qs[i].view(1, h, 1, 64), Ks[i][None], Vs[i][None], enable_gqa=True)
            return outs
        mk = lambda k: (lambda _: [decode_attention(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, out=outs[i], kernel=k) for i in range(n)] and outs)
        arms = {"sdpa": sdpa, "sdpa_aa": sdpa, "v1": mk("decode_attn_bf16"), "v2": mk("decode_attn_v2_bf16"), "v3": mk("decode_attn_v3_bf16"), "v4": mk("decode_attn_v4_bf16")}
        rows = harness2.run_variants(arms, None, n, baseline="sdpa")
        print(name, " ".join(f"{k}={v['us_per_op']:.2f}us" for k, v in rows.items()),
              "| v4 speedup %.2fx" % (rows["sdpa"]["us_per_op"] / rows["v4"]["us_per_op"]), flush=True)
