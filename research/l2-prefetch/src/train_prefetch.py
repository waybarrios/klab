"""Training-time prefetch: a Linear whose backward prefetches the previous layer's weight into L2."""
import torch
import torch.nn.functional as F
from l2prefetch import prefetch


class _PrefetchLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, nxt, prv, plan, side):
        ctx.save_for_backward(x, w)
        ctx.prv, ctx.plan, ctx.side = prv, plan, side
        if plan is not None and nxt is not None:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                prefetch(nxt, **plan)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        if ctx.plan is not None and ctx.prv is not None:
            ctx.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ctx.side):
                prefetch(ctx.prv, **ctx.plan)
        gx = g @ w
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return gx, gw, None, None, None, None


class Stack(torch.nn.Module):
    """Deep Linear+GELU stack; plan=None gives the plain PyTorch path (F.linear with autograd)."""

    def __init__(self, d, layers, dtype, plan=None, seed=0):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.ws = torch.nn.ParameterList(
            torch.nn.Parameter((torch.randn(d, d, generator=g, device="cuda") / d ** 0.5).to(dtype)) for _ in range(layers))
        self.plan, self.side = plan, torch.cuda.Stream()

    def forward(self, x):
        n = len(self.ws)
        for i, w in enumerate(self.ws):
            if self.plan is None:
                x = F.linear(x, w)
            else:
                nxt = self.ws[i + 1] if i + 1 < n else None
                prv = self.ws[i - 1] if i > 0 else None
                x = _PrefetchLinear.apply(x, w, nxt, prv, self.plan, self.side)
            x = F.gelu(x)
        return x

    def join(self):
        if self.plan is not None:
            torch.cuda.current_stream().wait_stream(self.side)
