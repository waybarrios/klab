import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, models2, torch
from l2prefetch import ChainRunner
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H10-decode-gap/results/decompose.json")
torch.set_grad_enabled(False)
res = []
with H.lock():
    for name in ("smollm2-135m", "qwen2.5-0.5b"):
        before = H.require_p0()
        arms, keep, info = {}, [], {}
        for arm, kw in {"A_cat": dict(static=False), "B_static": {}, "C_no_attn": dict(attention=False),
                        "D_gemv_only": dict(attention=False, elementwise=False)}.items():
            ops, reads, x0, period, k, wb, kvb = models2.build(name, 1, seed=2, **kw)
            arms[arm] = ChainRunner(ops, reads); keep.append((ops, reads, k)); info = (wb, kvb)
        rows = harness2.run_variants(arms, x0, 1, baseline="B_static")
        floor = (info[0] + info[1]) / 356e3
        r = {k: v["us_per_op"] for k, v in rows.items()}
        res.append({"model": name, "step_us": r, "floor_us": floor, "weight_bytes": info[0], "kv_bytes": info[1],
                    "regime_before": before, "regime_after": H.regime()})
        OUT.write_text(json.dumps(res, indent=1))
        B, C, D = r["B_static"], r["C_no_attn"], r["D_gemv_only"]
        print(f"{name}: A_cat {r['A_cat']:.0f}us | B_static {B:.0f} | C_no_attn {C:.0f} | D_gemv_only {D:.0f} | floor {floor:.0f}us "
              f"| attention {B-C:.0f}us elementwise {C-D:.0f}us gemv-over-floor {D-floor:.0f}us | gap B/floor {B/floor:.2f}x", flush=True)
        del arms, keep; torch.cuda.empty_cache()
