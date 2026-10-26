"""GPU tests for the decode attention kernels. Run: .venv/bin/python -m unittest research/l2-prefetch/tests/test_decode.py"""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, KLAB)
import torch

CUDA = torch.cuda.is_available()


def reference(q, kn, vn, K, V, L, h, kvh):
    Kf = torch.cat([K[:, :L].double(), kn.view(kvh, 1, 64).double()], 1)
    Vf = torch.cat([V[:, :L].double(), vn.view(kvh, 1, 64).double()], 1)
    qf, g = q.view(h, 64).double(), h // kvh
    return torch.stack([torch.softmax(qf[i] @ Kf[i // g].T / 8.0, -1) @ Vf[i // g] for i in range(h)]).reshape(-1)


@unittest.skipUnless(CUDA, "needs a CUDA device")
class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import harness
        cls.lock = harness.lock()
        cls.lock.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.lock.__exit__(None, None, None)

    def _case(self, h, kvh, L, cap=512, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
        return r(h * 64), r(kvh * 64), r(kvh * 64), r(kvh, cap, 64), r(kvh, cap, 64)

    def test_v4_matches_reference_and_appends(self):
        from decode_attn import decode_attention
        for h, kvh, L in ((9, 3, 0), (14, 2, 7), (32, 4, 300)):
            q, kn, vn, K, V = self._case(h, kvh, L)
            ref = reference(q, kn, vn, K, V, L, h, kvh)
            Kc, Vc = K.clone(), V.clone()
            out = decode_attention(q, kn, vn, Kc, Vc, L).double()
            self.assertLess((out - ref).abs().max().item(), 1e-2 * max(1.0, ref.abs().max().item()))
            self.assertTrue(torch.equal(Kc[:, L], kn.view(kvh, 64)) and torch.equal(Vc[:, L], vn.view(kvh, 64)))
            self.assertTrue(torch.equal(Kc[:, :L], K[:, :L]))

    def test_dynamic_length_one_graph(self):
        from decode_attn import decode_attention_dynamic
        h, kvh = 9, 3
        q, kn, vn, K, V = self._case(h, kvh, 0, cap=256)
        Kref, Vref = K.clone(), V.clone()
        length = torch.zeros(1, device="cuda", dtype=torch.int32)
        out = torch.empty(h * 64, device="cuda", dtype=torch.bfloat16)
        decode_attention_dynamic(q, kn, vn, K.clone(), V.clone(), length, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_attention_dynamic(q, kn, vn, K, V, length, out=out)
        for L in (0, 1, 63, 64, 130, 254):
            length.fill_(L)
            graph.replay()
            torch.cuda.synchronize()
            Kref[:, L], Vref[:, L] = kn.view(kvh, 64), vn.view(kvh, 64)
            ref = reference(q, kn, vn, Kref, Vref, L, h, kvh)
            self.assertLess((out.double() - ref).abs().max().item(), 1e-2 * max(1.0, ref.abs().max().item()))

    def test_fused_elementwise(self):
        import torch.nn.functional as F
        from decode_attn import add_rmsnorm, silu_mul
        res, y = torch.randn(2, 576, device="cuda").bfloat16(), torch.randn(2, 576, device="cuda").bfloat16()
        w = (torch.rand(576, device="cuda") + 0.5).bfloat16()
        x, xn = add_rmsnorm(res, y, w)
        self.assertTrue(torch.equal(x, res + y))
        self.assertLess((xn.float() - F.rms_norm(res + y, (576,), w, 1e-6).float()).abs().max().item(), 5e-2)
        gu = torch.randn(2, 2 * 1536, device="cuda").bfloat16()
        g, u = gu.chunk(2, -1)
        self.assertLess((silu_mul(gu).float() - (F.silu(g.float()) * u.float())).abs().max().item(), 5e-2)


if __name__ == "__main__":
    unittest.main()
