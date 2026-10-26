"""H8 exploratory dose-response: prefetch first X MiB of gate_up during the attention stage."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, models, torch
from l2prefetch import ChainRunner

OUT = Path(KLAB + "/research/l2-prefetch/experiments/H8-idle-windows/results/dose.json")
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
with H.lock():
    before = H.require_p0()
    ops, reads, x0, period, keep = models.build("qwen2.5-0.5b", 1, True, seed=2)
    variants = {"torch": ChainRunner(ops, reads), "torch_aa": ChainRunner(ops, reads)}
    for mib in (0.5, 1, 1.5, 2, 2.5, 3, 3.5):
        for b in (2, 4, 8):
            plan = {"blocks": b, "kernel": "l2_hint", "stride": 128, "nbytes": int(mib * (1 << 20))}
            # stage 2 (gate_up) prefetched while stage 1 (attention + o) runs
            variants[f"x{mib}_b{b}"] = ChainRunner(ops, reads, [plan if i % period == 2 else None for i in range(len(ops))])
    rows = H.run_variants(variants, x0, 1)
    OUT.write_text(json.dumps({"regime_before": before, "regime_after": H.regime(),
                               "variants": {k: {kk: vv for kk, vv in v.items() if kk != "samples_us"} for k, v in rows.items()}}, indent=1))
    print(f"torch step {rows['torch']['us_per_op']:.1f}us aa {rows['torch_aa']['saved_pct']:+.2f}%")
    for b in (2, 4, 8):
        print(f"b{b}: " + "  ".join(f"{mib}MiB {rows[f'x{mib}_b{b}']['saved_pct']:+.2f}%{'' if rows[f'x{mib}_b{b}']['bit_identical'] else '!'}"
                                     for mib in (0.5, 1, 1.5, 2, 2.5, 3, 3.5)))
