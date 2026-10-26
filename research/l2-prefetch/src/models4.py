"""Decoder step with the residual add fused into the next stage's norm, three elementwise implementations:
"eager" (PyTorch ops), "compile" (torch.compile / Inductor-fused), "cuda" (add_rmsnorm_bf16, silu_mul_bf16).
Attention uses the v4 single-launch kernel or SDPA. Weights random; norm weights random positive."""
import math
import torch
import torch.nn.functional as F
from decode_attn import add_rmsnorm, decode_attention, silu_mul
from models3 import DIMS

EPS = 1e-6


def _eager_add_norm(res, y, w):
    x = res + y
    return x, F.rms_norm(x, (x.shape[-1],), w, EPS)


def _eager_silu_mul(gu):
    g, u = gu.chunk(2, -1)
    return F.silu(g) * u


_compiled = {}


def elementwise(impl):
    if impl == "eager":
        return _eager_add_norm, _eager_silu_mul
    if impl == "compile":
        if not _compiled:
            _compiled["an"] = torch.compile(_eager_add_norm, dynamic=False)
            _compiled["sm"] = torch.compile(_eager_silu_mul, dynamic=False)
        return _compiled["an"], _compiled["sm"]
    if impl == "cuda":
        return (lambda res, y, w: add_rmsnorm(res, y, w, EPS)), silu_mul
    raise ValueError(impl)


def build(name, m, kv_len=512, dtype=torch.bfloat16, seed=0, impl="eager", attention_impl="custom"):
    d, inter, L, h, kvh, hd = DIMS[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s, fan: (torch.randn(*s, generator=g, device="cuda") / math.sqrt(fan)).to(dtype)
    add_norm, act = elementwise(impl)
    ops, reads, keep, state = [], [], [], {}
    for layer in range(L):
        Wqkv = rnd((h + 2 * kvh) * hd, d, fan=d); Wo = rnd(d, h * hd, fan=h * hd)
        Wgu = rnd(2 * inter, d, fan=d); Wd = rnd(d, inter, fan=inter)
        n1 = (torch.rand(d, generator=g, device="cuda") + 0.5).to(dtype); n2 = (torch.rand(d, generator=g, device="cuda") + 0.5).to(dtype)
        K = rnd(1, kvh, kv_len + m, hd, fan=1); V = rnd(1, kvh, kv_len + m, hd, fan=1)
        keep += [Wqkv, Wo, Wgu, Wd, n1, n2, K, V]

        def attn(q, k, v, K=K, V=V):
            if attention_impl == "custom":
                return decode_attention(q.reshape(-1), k.reshape(-1), v.reshape(-1), K[0], V[0], kv_len).view(1, h * hd)
            q = q.view(m, h, hd).transpose(0, 1).unsqueeze(0)
            K[:, :, kv_len:].copy_(k.view(m, kvh, hd).transpose(0, 1).unsqueeze(0))
            V[:, :, kv_len:].copy_(v.view(m, kvh, hd).transpose(0, 1).unsqueeze(0))
            o = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
            return o.squeeze(0).transpose(0, 1).reshape(m, h * hd)

        def s_qkv(y, W=Wqkv, n1=n1, first=(layer == 0)):
            if first:
                state["res"] = y
                xn = F.rms_norm(y, (d,), n1, EPS)
            else:
                state["res"], xn = add_norm(state["res"], y, n1)
            return F.linear(xn, W)

        def s_o(qkv, Wo=Wo, attn=attn):
            q, k, v = qkv.split([h * hd, kvh * hd, kvh * hd], -1)
            return F.linear(attn(q, k, v), Wo)

        def s_gu(y, W=Wgu, n2=n2):
            state["res"], xn = add_norm(state["res"], y, n2)
            return F.linear(xn, W)

        def s_down(gu, Wd=Wd):
            return F.linear(act(gu), Wd)

        ops += [s_qkv, s_o, s_gu, s_down]
        reads += [[Wqkv], [Wo, K, V], [Wgu], [Wd]]
    x0 = torch.randn(m, d, generator=g, device="cuda").to(dtype)
    return ops, reads, x0, 4, keep
