"""GPU correctness tests for cross-op L2 prefetch. Run: .venv/bin/python -m unittest research/l2-prefetch/tests/test_prefetch.py"""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, KLAB)
import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "needs a CUDA device")
class PrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import harness
        cls.lock = harness.lock()
        cls.lock.__enter__()
        torch.manual_seed(0)

    @classmethod
    def tearDownClass(cls):
        cls.lock.__exit__(None, None, None)

    def test_prefetch_never_writes(self):
        from l2prefetch import prefetch
        for kernel in ("l2_hint", "l2_hint_last", "l2_load"):
            t = torch.randn(1 << 20, device="cuda")
            before = t.clone()
            prefetch(t, blocks=4, kernel=kernel, stride=32)
            prefetch(t, blocks=2, kernel=kernel, nbytes=4096)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(t, before), kernel)

    def test_chain_bit_identical_in_graph(self):
        import torch.nn.functional as F
        from l2prefetch import ChainRunner
        ws = [torch.randn(384, 384, device="cuda", dtype=torch.bfloat16) / 20 for _ in range(12)]
        ops, reads = [lambda y, w=w: F.linear(y, w) for w in ws], [[w] for w in ws]
        x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
        plain = ChainRunner(ops, reads)(x)
        plan = {"blocks": 2, "kernel": "l2_hint", "stride": 128}
        runner = ChainRunner(ops, reads, [plan] * len(ops))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            runner(x)  # warm-up outside capture
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner(x)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, plain))

    def test_prefetch_sequential_matches_plain(self):
        from api import PrefetchSequential
        torch.set_grad_enabled(False)
        try:
            layers = [torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU()).cuda().half() for _ in range(64)]
            x = torch.randn(4, 256, device="cuda", dtype=torch.half)
            fast = PrefetchSequential(layers, x, rounds=3)
            plain = torch.nn.Sequential(*layers)
            self.assertTrue(torch.equal(fast(x), plain(x)))
        finally:
            torch.set_grad_enabled(True)

    def test_training_gradients_bit_identical(self):
        from train_prefetch import Stack
        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        grads = []
        for plan in (None, {"blocks": 2, "kernel": "l2_hint", "stride": 128}):
            model = Stack(256, 16, torch.bfloat16, plan, seed=5)
            model(x).float().square().mean().backward()
            model.join()
            torch.cuda.synchronize()
            grads.append([w.grad for w in model.ws])
        for a, b in zip(*grads):
            self.assertTrue(torch.equal(a, b))

    def test_hooks_training_gradients_bit_identical(self):
        from api import PrefetchHooks
        torch.manual_seed(1)
        model = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(384, 384), torch.nn.GELU()) for _ in range(24)]).cuda()
        x = torch.randn(8, 384, device="cuda")
        def step(m, inp):
            for p in m.parameters():
                p.grad = None
            loss = m(inp).square().mean()
            loss.backward()
            return torch.cat([p.grad.reshape(-1) for p in m.parameters()] + [loss.reshape(1)])
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # PyTorch rule for capturing backward: eager warm-up on a side stream
            plain = step(model, x).clone()
        torch.cuda.current_stream().wait_stream(side)
        hooks = PrefetchHooks(model, x, rounds=3, min_lower_ci=-100.0, backward=True, step=step)  # force plans on
        try:
            self.assertTrue(any(hooks.plans.values()))
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(step(model, x), plain))
        finally:
            hooks.remove()


if __name__ == "__main__":
    unittest.main()
