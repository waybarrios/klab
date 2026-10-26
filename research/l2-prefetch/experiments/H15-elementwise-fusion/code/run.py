import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, models4, torch
from l2prefetch import ChainRunner
OUT = Path(KLAB + "/research/l2-prefetch/experiments/H15-elementwise-fusion/results/h15.json")
MODELS = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "tinyllama-1.1b")
torch.set_grad_enabled(False)
res = {"steps": []}
with H.lock():
    res["regime_before"] = H.require_p0()
    for name in MODELS:
        arms, keep = {}, []
        for impl in ("eager", "compile", "cuda"):
            ops, reads, x0, period, k = models4.build(name, 1, kv_len=256, seed=2, impl=impl)
            arms[impl] = ChainRunner(ops, reads); keep.append(k)
        arms["eager_aa"] = arms["eager"]
        rows = harness2.run_variants(arms, x0, 1, baseline="eager")
        cmp_rows = harness2.run_variants({"compile": arms["compile"], "cuda": arms["cuda"]}, x0, 1, baseline="compile")
        out = {"model": name, **{f"{k}_us": v["us_per_op"] for k, v in rows.items()},
               "cuda_vs_eager": rows["cuda"]["saved_pct"], "cuda_vs_eager_ci": rows["cuda"]["ci95"],
               "compile_vs_eager": rows["compile"]["saved_pct"], "aa": rows["eager_aa"]["saved_pct"],
               "cuda_vs_compile": cmp_rows["cuda"]["saved_pct"], "cuda_vs_compile_ci": cmp_rows["cuda"]["ci95"]}
        res["steps"].append(out); OUT.write_text(json.dumps(res, indent=1))
        print(f"{name}: eager {rows['eager']['us_per_op']:.0f}us compile {rows['compile']['us_per_op']:.0f}us cuda {rows['cuda']['us_per_op']:.0f}us | "
              f"cuda vs eager {out['cuda_vs_eager']:+.2f}% compile vs eager {out['compile_vs_eager']:+.2f}% cuda vs compile {out['cuda_vs_compile']:+.2f}% "
              f"(ci {out['cuda_vs_compile_ci'][0]:+.2f}..{out['cuda_vs_compile_ci'][1]:+.2f}) aa {out['aa']:+.2f}%", flush=True)
        del arms, keep; torch.cuda.empty_cache()
    res["regime_after"] = H.regime()
geo = 100 * (1 - math.exp(sum(math.log(1 - s["cuda_vs_eager"] / 100) for s in res["steps"]) / len(res["steps"])))
crit = {"a": geo >= 5, "b": all(s["cuda_vs_compile_ci"][1] >= -1 for s in res["steps"])}
res["geomean_cuda_vs_eager"], res["criteria"] = geo, crit; OUT.write_text(json.dumps(res, indent=1))
print(f"H15 geomean cuda vs eager {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")
