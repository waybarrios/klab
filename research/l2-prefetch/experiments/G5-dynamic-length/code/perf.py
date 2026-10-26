"""EXPLORATORY: cost of the dynamic-length kernel (splits sized by capacity) vs static v4 and copy_+SDPA."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, harness2, torch, torch.nn.functional as F
from decode_attn import decode_attention, decode_attention_dynamic
from run import sdpa_path
torch.set_grad_enabled(False)
CAP = 2048
with H.lock():
    H.require_p0()
    for h, kvh in ((9, 3), (32, 4)):
        for L in (192, 640, 1280):
            n = 24
            qs = torch.randn(n, h * 64, device="cuda").to(torch.bfloat16); ks = torch.randn(n, kvh * 64, device="cuda").to(torch.bfloat16)
            Ks = torch.randn(n, kvh, CAP, 64, device="cuda").to(torch.bfloat16); Vs = torch.randn_like(Ks)
            Kt = Ks[:, :, :L + 1].contiguous(); Vt = Vs[:, :, :L + 1].contiguous()   # tight caches for the static arms
            outs = torch.empty(n, h * 64, device="cuda", dtype=torch.bfloat16)
            length = torch.tensor([L], device="cuda", dtype=torch.int32)
            sd = lambda _: [sdpa_path(qs[i], ks[i], ks[i], Kt[i], Vt[i], L, h, kvh) for i in range(n)] and outs
            v4 = lambda _: [decode_attention(qs[i], ks[i], ks[i], Kt[i], Vt[i], L, out=outs[i]) for i in range(n)] and outs
            v4d = lambda _: [decode_attention_dynamic(qs[i], ks[i], ks[i], Ks[i], Vs[i], length, out=outs[i]) for i in range(n)] and outs
            rows = harness2.run_variants({"sdpa": sd, "v4": v4, "v4d_cap2048": v4d}, None, n, baseline="sdpa")
            print(f"h={h} kvh={kvh} L={L}: sdpa {rows['sdpa']['us_per_op']:.2f} v4 {rows['v4']['us_per_op']:.2f} "
                  f"v4d(cap {CAP}) {rows['v4d_cap2048']['us_per_op']:.2f} us", flush=True)
            del qs, ks, Ks, Vs, Kt, Vt, outs; torch.cuda.empty_cache()
