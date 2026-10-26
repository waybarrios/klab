"""Capture-time choice between the v4 single-launch decode attention and PyTorch SDPA (no-harm fallback)."""
import torch
import torch.nn.functional as F

import harness2
from decode_attn import decode_attention


def choose(heads, kv_heads, length, layers=24, min_lower_ci=1.0):
    """Time both implementations on synthetic caches of this shape; return ("custom"|"sdpa", report)."""
    n, L = layers, length
    qs = torch.randn(n, heads * 64, device="cuda").to(torch.bfloat16)
    ks = torch.randn(n, kv_heads * 64, device="cuda").to(torch.bfloat16)
    Ks = torch.randn(n, kv_heads, L + 1, 64, device="cuda").to(torch.bfloat16)
    Vs = torch.randn_like(Ks)
    outs = torch.empty(n, heads * 64, device="cuda", dtype=torch.bfloat16)

    def sdpa(_):
        for i in range(n):
            Ks[i][:, L].copy_(ks[i].view(kv_heads, 64)); Vs[i][:, L].copy_(ks[i].view(kv_heads, 64))
            F.scaled_dot_product_attention(qs[i].view(1, heads, 1, 64), Ks[i][None], Vs[i][None], enable_gqa=True)
        return outs

    def custom(_):
        for i in range(n):
            decode_attention(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, out=outs[i])
        return outs

    with torch.no_grad():
        rows = harness2.run_variants({"sdpa": sdpa, "custom": custom}, None, n, rounds=9, baseline="sdpa", warm_seconds=1.0)
    pick = "custom" if rows["custom"]["ci95"][0] > min_lower_ci else "sdpa"
    return pick, {"sdpa_us": rows["sdpa"]["us_per_op"], "custom_us": rows["custom"]["us_per_op"],
                  "saved_pct": rows["custom"]["saved_pct"], "ci95": rows["custom"]["ci95"]}


def choose3(heads, kv_heads, length, layers=24, min_lower_ci=1.0):
    """Like choose(), with FlashInfer single decode as a third option when it supports the GQA group size."""
    import fi_env  # noqa: F401
    import flashinfer
    n, L = layers, length
    qs = torch.randn(n, heads * 64, device="cuda").to(torch.bfloat16)
    ks = torch.randn(n, kv_heads * 64, device="cuda").to(torch.bfloat16)
    Ks = torch.randn(n, kv_heads, L + 1, 64, device="cuda").to(torch.bfloat16)
    Vs = torch.randn_like(Ks)
    outs = torch.empty(n, heads * 64, device="cuda", dtype=torch.bfloat16)

    def append(i):
        Ks[i][:, L].copy_(ks[i].view(kv_heads, 64)); Vs[i][:, L].copy_(ks[i].view(kv_heads, 64))

    def sdpa(_):
        for i in range(n):
            append(i)
            F.scaled_dot_product_attention(qs[i].view(1, heads, 1, 64), Ks[i][None], Vs[i][None], enable_gqa=True)
        return outs

    def custom(_):
        for i in range(n):
            decode_attention(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, out=outs[i])
        return outs

    def fi(_):
        for i in range(n):
            append(i)
            flashinfer.single_decode_with_kv_cache(qs[i].view(heads, 64), Ks[i], Vs[i], kv_layout="HND")
        return outs

    variants = {"sdpa": sdpa, "custom": custom}
    try:
        fi(None)
        torch.cuda.synchronize()
        variants["flashinfer"] = fi
    except RuntimeError:
        pass  # e.g. unsupported GQA group size
    with torch.no_grad():
        rows = harness2.run_variants(variants, None, n, rounds=9, baseline="sdpa", warm_seconds=1.0)
    best = min((k for k in rows if k != "sdpa"), key=lambda k: rows[k]["us_per_op"])
    pick = best if rows[best]["ci95"][0] > min_lower_ci else "sdpa"
    return pick, {k: {"us": v["us_per_op"], "saved_pct": v["saved_pct"], "ci95": v["ci95"]} for k, v in rows.items()}
