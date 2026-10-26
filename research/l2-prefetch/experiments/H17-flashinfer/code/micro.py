"""EXPLORATORY: FlashInfer single-request decode vs our v4 vs copy_+SDPA, kernel path alone, 24 layer caches."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import fi_env  # noqa: F401
import flashinfer, harness as H, harness2, torch, torch.nn.functional as F
from decode_attn import decode_attention
from run import reference, sdpa_path
torch.set_grad_enabled(False)


def fi_path(q, kn, vn, K, V, L, h, kvh):
    K[:, L].copy_(kn.view(kvh, 64)); V[:, L].copy_(vn.view(kvh, 64))
    return flashinfer.single_decode_with_kv_cache(q.view(h, 64), K[:, :L + 1], V[:, :L + 1], kv_layout="HND").reshape(-1)


with H.lock():
    H.require_p0()
    for h, kvh in ((9, 3), (15, 5), (14, 2), (32, 4)):
        for L in (192, 640, 1280):
            g = torch.Generator(device="cuda").manual_seed(1); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
            q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64); K, V = r(kvh, L + 1, 64), r(kvh, L + 1, 64)
            ref = reference(q, kn, vn, K, V, L, h, kvh)
            try:
                e_fi = (fi_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
            except RuntimeError as err:
                print(f"h={h} kvh={kvh} L={L}: flashinfer unsupported ({str(err).splitlines()[0][-40:]})", flush=True)
                continue
            e_sd = (sdpa_path(q, kn, vn, K.clone(), V.clone(), L, h, kvh).double() - ref).abs().max().item()
            n = 24
            qs = torch.randn(n, h * 64, device="cuda").to(torch.bfloat16); ks = torch.randn(n, kvh * 64, device="cuda").to(torch.bfloat16)
            Ks = torch.randn(n, kvh, L + 1, 64, device="cuda").to(torch.bfloat16); Vs = torch.randn_like(Ks)
            outs = torch.empty(n, h * 64, device="cuda", dtype=torch.bfloat16)
            sd = lambda _: [sdpa_path(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, h, kvh) for i in range(n)] and outs
            fi = lambda _: [fi_path(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, h, kvh) for i in range(n)] and outs
            v4 = lambda _: [decode_attention(qs[i], ks[i], ks[i], Ks[i], Vs[i], L, out=outs[i]) for i in range(n)] and outs
            rows = harness2.run_variants({"sdpa": sd, "flashinfer": fi, "v4": v4}, None, n, baseline="sdpa")
            print(f"h={h} kvh={kvh} L={L}: sdpa {rows['sdpa']['us_per_op']:.2f} flashinfer {rows['flashinfer']['us_per_op']:.2f} "
                  f"v4 {rows['v4']['us_per_op']:.2f} us | fi err {e_fi:.1e} sdpa err {e_sd:.1e}", flush=True)
            del qs, ks, Ks, Vs, outs; torch.cuda.empty_cache()
