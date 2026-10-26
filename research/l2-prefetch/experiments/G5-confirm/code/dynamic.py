import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import sys
sys.path.insert(0, KLAB + "/research/l2-prefetch/src"); sys.path.insert(0, KLAB + "/research/l2-prefetch/experiments/H11-decode-attention/code")
import harness as H, torch
from decode_attn import decode_attention_dynamic
from run import reference, sdpa_path
torch.set_grad_enabled(False)
ok = True
with H.lock():
    for (h, kvh), cap in (((15, 5), 700), ((9, 3), 1500), ((32, 4), 3000)):
        g = torch.Generator(device="cuda").manual_seed(11); r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
        K, V = r(kvh, cap, 64), r(kvh, cap, 64); Kref, Vref = K.clone(), V.clone()
        q, kn, vn = r(h * 64), r(kvh * 64), r(kvh * 64)
        length = torch.zeros(1, device="cuda", dtype=torch.int32); out = torch.empty(h * 64, device="cuda", dtype=torch.bfloat16)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            decode_attention_dynamic(q, kn, vn, K.clone(), V.clone(), length, out=out)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_attention_dynamic(q, kn, vn, K, V, length, out=out)
        steps = sorted(set(list(range(0, cap - 1, max(1, cap // 33))) + [cap - 2]))
        worst = 0.0
        for L in steps:
            q.copy_(r(h * 64)); kn.copy_(r(kvh * 64)); vn.copy_(r(kvh * 64)); length.fill_(L)
            graph.replay(); torch.cuda.synchronize()
            Kref[:, L] = kn.view(kvh, 64); Vref[:, L] = vn.view(kvh, 64)
            ref = reference(q, kn, vn, Kref, Vref, L, h, kvh)
            e = (out.double() - ref).abs().max().item()
            es = (sdpa_path(q, kn, vn, Kref.clone(), Vref.clone(), L, h, kvh).double() - ref).abs().max().item()
            good = e <= 2 * es + 1e-3 and torch.equal(K[:, L], kn.view(kvh, 64)) and torch.equal(V[:, L], vn.view(kvh, 64))
            ok &= good; worst = max(worst, e)
        print(f"heads {h}/{kvh} capacity {cap}: {len(steps)} replays of one graph, worst err {worst:.2e}", flush=True)
print("G5(b):", "PASS" if ok else "FAIL")
