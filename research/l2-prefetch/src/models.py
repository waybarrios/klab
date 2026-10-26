"""Pure-PyTorch decoder layers split into weight-owning stages for ChainRunner (decode, static KV cache)."""
import math
import torch
import torch.nn.functional as F

DIMS = {  # hidden, intermediate, layers, heads, kv_heads, head_dim
    "smollm2-135m": (576, 1536, 30, 9, 3, 64),
    "qwen2.5-0.5b": (896, 4864, 24, 14, 2, 64),
}


def build(name, m, fused, kv_len=512, dtype=torch.bfloat16, seed=0):
    d, inter, L, h, kvh, hd = DIMS[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s, fan: (torch.randn(*s, generator=g, device="cuda") / math.sqrt(fan)).to(dtype)
    ops, reads, keep, state = [], [], [], {}
    norm = torch.ones(d, dtype=dtype, device="cuda")
    for _ in range(L):
        Wq, Wk, Wv = rnd(h * hd, d, fan=d), rnd(kvh * hd, d, fan=d), rnd(kvh * hd, d, fan=d)
        Wo = rnd(d, h * hd, fan=h * hd)
        Wg, Wu, Wd = rnd(inter, d, fan=d), rnd(inter, d, fan=d), rnd(d, inter, fan=inter)
        K = rnd(1, kvh, kv_len, hd, fan=1); V = rnd(1, kvh, kv_len, hd, fan=1)
        keep += [Wq, Wk, Wv, Wo, Wg, Wu, Wd, K, V]

        def attn(q, k, v, K=K, V=V):
            q = q.view(m, h, hd).transpose(0, 1).unsqueeze(0)          # [1,h,m,hd]; m new tokens vs cache
            k = k.view(m, kvh, hd).transpose(0, 1).unsqueeze(0)
            v = v.view(m, kvh, hd).transpose(0, 1).unsqueeze(0)
            kk, vv = torch.cat([K, k], 2), torch.cat([V, v], 2)
            o = F.scaled_dot_product_attention(q, kk, vv, enable_gqa=True)
            return o.squeeze(0).transpose(0, 1).reshape(m, h * hd)

        if fused:
            Wqkv = torch.cat([Wq, Wk, Wv]); Wgu = torch.cat([Wg, Wu]); keep += [Wqkv, Wgu]
            def s_qkv(x, W=Wqkv):
                state["res"] = x
                return F.linear(F.rms_norm(x, (d,), norm), W)
            def s_o(qkv, Wo=Wo, attn=attn):
                q, k, v = qkv.split([h * hd, kvh * hd, kvh * hd], -1)
                x = state["res"] + F.linear(attn(q, k, v), Wo); state["res"] = x
                return x
            def s_gu(x, W=Wgu):
                return F.linear(F.rms_norm(x, (d,), norm), W)
            def s_down(gu, Wd=Wd):
                gt, up = gu.chunk(2, -1)
                return state["res"] + F.linear(F.silu(gt) * up, Wd)
            ops += [s_qkv, s_o, s_gu, s_down]
            reads += [[Wqkv], [Wo, K, V], [Wgu], [Wd]]
        else:
            def s_q(x, Wq=Wq):
                state["res"] = x; state["xn"] = F.rms_norm(x, (d,), norm)
                state["q"] = F.linear(state["xn"], Wq); return x
            def s_k(x, Wk=Wk):
                state["k"] = F.linear(state["xn"], Wk); return x
            def s_v(x, Wv=Wv):
                state["v"] = F.linear(state["xn"], Wv); return x
            def s_o(x, Wo=Wo, attn=attn):
                x = state["res"] + F.linear(attn(state["q"], state["k"], state["v"]), Wo); state["res"] = x
                state["xn"] = F.rms_norm(x, (d,), norm); return x
            def s_g(x, Wg=Wg):
                state["g"] = F.linear(state["xn"], Wg); return x
            def s_u(x, Wu=Wu):
                state["a"] = F.silu(state["g"]) * F.linear(state["xn"], Wu); return x
            def s_down(x, Wd=Wd):
                return state["res"] + F.linear(state["a"], Wd)
            ops += [s_q, s_k, s_v, s_o, s_g, s_u, s_down]
            reads += [[Wq], [Wk], [Wv], [Wo, K, V], [Wg], [Wu], [Wd]]
    period = 4 if fused else 7
    x0 = (torch.randn(m, d, generator=g, device="cuda")).to(dtype)
    return ops, reads, x0, period, keep
