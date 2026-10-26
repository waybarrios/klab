"""Fused decoder layers with a static, in-place KV cache and ablation switches (H10). models.py stays frozen."""
import math
import torch
import torch.nn.functional as F
from models import DIMS


def build(name, m, kv_len=512, dtype=torch.bfloat16, seed=0, attention=True, elementwise=True, static=True):
    """attention=False replaces SDPA by a slice; elementwise=False drops norms/activations/residuals."""
    d, inter, L, h, kvh, hd = DIMS[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s, fan: (torch.randn(*s, generator=g, device="cuda") / math.sqrt(fan)).to(dtype)
    ops, reads, keep, state = [], [], [], {}
    norm = torch.ones(d, dtype=dtype, device="cuda")
    for _ in range(L):
        Wqkv = rnd((h + 2 * kvh) * hd, d, fan=d); Wo = rnd(d, h * hd, fan=h * hd)
        Wgu = rnd(2 * inter, d, fan=d); Wd = rnd(d, inter, fan=inter)
        if static:
            K = rnd(1, kvh, kv_len + m, hd, fan=1); V = rnd(1, kvh, kv_len + m, hd, fan=1)
        else:
            K = rnd(1, kvh, kv_len, hd, fan=1); V = rnd(1, kvh, kv_len, hd, fan=1)
        keep += [Wqkv, Wo, Wgu, Wd, K, V]

        def attn(q, k, v, K=K, V=V):
            if not attention:
                return q
            q = q.view(m, h, hd).transpose(0, 1).unsqueeze(0)
            k = k.view(m, kvh, hd).transpose(0, 1).unsqueeze(0)
            v = v.view(m, kvh, hd).transpose(0, 1).unsqueeze(0)
            if static:
                K[:, :, kv_len:].copy_(k); V[:, :, kv_len:].copy_(v)
                kk, vv = K, V
            else:
                kk, vv = torch.cat([K, k], 2), torch.cat([V, v], 2)
            o = F.scaled_dot_product_attention(q, kk, vv, enable_gqa=True)
            return o.squeeze(0).transpose(0, 1).reshape(m, h * hd)

        ln = (lambda x: F.rms_norm(x, (d,), norm)) if elementwise else (lambda x: x)

        def s_qkv(x, W=Wqkv, ln=ln):
            state["res"] = x
            return F.linear(ln(x), W)

        def s_o(qkv, Wo=Wo, attn=attn):
            q, k, v = qkv.split([h * hd, kvh * hd, kvh * hd], -1)
            y = F.linear(attn(q, k, v), Wo)
            x = state["res"] + y if elementwise else y
            state["res"] = x
            return x

        def s_gu(x, W=Wgu, ln=ln):
            return F.linear(ln(x), W)

        def s_down(gu, Wd=Wd):
            gt, up = gu.chunk(2, -1)
            a = F.silu(gt) * up if elementwise else gt
            y = F.linear(a, Wd)
            return state["res"] + y if elementwise else y

        ops += [s_qkv, s_o, s_gu, s_down]
        reads += [[Wqkv], [Wo, K, V], [Wgu], [Wd]]
    x0 = torch.randn(m, d, generator=g, device="cuda").to(dtype)
    weight_bytes = L * sum(t.numel() * t.element_size() for t in (Wqkv, Wo, Wgu, Wd))
    kv_bytes = L * 2 * K.numel() * K.element_size()
    return ops, reads, x0, 4, keep, weight_bytes, kv_bytes
