"""H3: prefetch across non-GEMV stages (MLP blocks with norm/activation, conv layers). Plain PyTorch consumers."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H
import policy
import torch, torch.nn.functional as F
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H3-generic-ops/results/generic.json")


def mlp_stages(d, hidden, m, dtype, budget):
    """Pre-norm MLP block: y + W2 gelu(W1 ln(y)). Two stages per block, each owns one weight."""
    per = d * hidden * torch.finfo(dtype).bits // 8
    blocks = max(8, math.ceil(budget / (2 * per)))
    W1 = torch.randn(blocks, hidden, d, dtype=dtype, device="cuda") / math.sqrt(d)
    W2 = torch.randn(blocks, d, hidden, dtype=dtype, device="cuda") / math.sqrt(hidden)
    g = torch.ones(d, dtype=dtype, device="cuda")
    ops, reads, state = [], [], {}
    for b in range(blocks):
        def up(y, w=W1[b]):
            state["res"] = y
            return F.gelu(F.linear(F.layer_norm(y, (d,), g), w))
        def down(h, w=W2[b]):
            return state["res"] + F.linear(h, w)
        ops += [up, down]; reads += [[W1[b]], [W2[b]]]
    x0 = torch.randn(m, d, dtype=dtype, device="cuda")
    return ops, reads, x0, per, (W1, W2, g)


def conv_stages(c, hw, n, dtype, budget):
    """conv3x3(C->C) + bias-free ReLU, NCHW, batch n."""
    per = c * c * 9 * torch.finfo(dtype).bits // 8
    layers = max(16, math.ceil(budget / per))
    W = torch.randn(layers, c, c, 3, 3, dtype=dtype, device="cuda") / math.sqrt(9 * c)
    ops = [lambda y, w=W[i]: F.relu(F.conv2d(y, w, padding=1)) for i in range(layers)]
    reads = [[W[i]] for i in range(layers)]
    x0 = torch.randn(n, c, hw, hw, dtype=dtype, device="cuda")
    return ops, reads, x0, per, (W,)


CASES = [("mlp", 256, 1024, 1), ("mlp", 384, 1536, 1), ("mlp", 512, 1024, 4), ("mlp", 512, 1024, 16),
         ("conv", 128, 14, 1), ("conv", 256, 7, 1), ("conv", 256, 14, 1), ("conv", 64, 28, 1)]


def main():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {tuple(r["case"]) for r in results}
    with H.lock():
        for case in CASES:
            if tuple(case) in done:
                continue
            kind, a, b, n = case
            before = H.require_p0()
            build = mlp_stages if kind == "mlp" else conv_stages
            ops, reads, x0, per, keep = build(a, b, n, torch.bfloat16, 80 << 20)
            m_equiv = n if kind == "mlp" else n * b * b   # conv: GEMM rows = N*H*W
            p = policy.plan(per, m_equiv)
            variants = {"torch": ChainRunner(ops, reads), "torch_aa": ChainRunner(ops, reads),
                        "policy": ChainRunner(ops, reads, [p] * len(ops) if p else None)}
            for blk in (1, 2, 4, 8):
                variants[f"x_hint_b{blk}"] = ChainRunner(ops, reads, [dict(blocks=blk, kernel="l2_hint", stride=128)] * len(ops))
            rows = H.run_variants(variants, x0, len(ops))
            results.append({"case": list(case), "bytes_per_stage": per, "stages": len(ops), "plan": p,
                            "regime_before": before, "regime_after": H.regime(), "variants": rows})
            OUT.write_text(json.dumps(results, indent=1))
            print(f"{case} bytes/stage={per/1024:.0f}KiB torch={rows['torch']['us_per_op']:.3f}us/stage plan={p and p['blocks']} "
                  f"policy {rows['policy']['saved_pct']:+.2f}% | " +
                  " ".join(f"{k}:{v['saved_pct']:+.1f}{'' if v['bit_identical'] else '!'}" for k, v in rows.items() if k not in ('torch', 'policy')),
                  flush=True)
            del ops, reads, keep, variants; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
