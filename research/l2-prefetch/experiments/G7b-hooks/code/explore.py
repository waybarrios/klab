"""G7b EXPLORATORY: sub-module prefetch via hooks + noise-relative acceptance, on the G7 models."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, torch
from torch import nn
from api import PrefetchHooks, PrefetchSequential
spec = importlib.util.spec_from_file_location("g7", KLAB + "/research/l2-prefetch/experiments/G7-non-llm-models/code/run.py")
g7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g7)
OUT = Path(KLAB + "/research/l2-prefetch/experiments/G7b-hooks/results/explore.json")
THR = 0.25


def with_fastpath(flag, fn):
    def run(x):
        torch.backends.mha.set_fastpath_enabled(flag)
        try:
            return fn(x)
        finally:
            torch.backends.mha.set_fastpath_enabled(True)
    return run


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.manual_seed(0)
res = json.loads(OUT.read_text()) if OUT.exists() else []
with H.lock():
    H.require_p0()
    only = sys.argv[1:]
    for name, layers, x, dt in g7.models():
        if only and not any(o in name for o in only):
            continue
        layers = [m.cuda().to(dt).eval() for m in layers]; x = x.cuda().to(dt)
        model = nn.Sequential(*layers)
        g7.cool()
        seq = PrefetchSequential(layers, x, min_lower_ci=THR)  # before hooks exist on the same modules
        torch.backends.mha.set_fastpath_enabled(False)
        hooks = PrefetchHooks(model, x, min_lower_ci=THR)
        torch.backends.mha.set_fastpath_enabled(True)
        arms = {"plain": with_fastpath(True, hooks._runner({})), "plain_nofast": with_fastpath(False, hooks._runner({})),
                "hooks": with_fastpath(False, hooks._runner(dict(hooks.plans))), "seq_relaxed": with_fastpath(True, seq)}
        arms["plain_aa"] = arms["plain"]
        g7.cool()
        rows = harness2.run_variants(arms, x, 1, baseline="plain")
        kept = sum(1 for g in hooks.report if g["plan"])
        out = {"model": name, "units": len(hooks.units), "groups": len(hooks.report), "kept": kept,
               **{k: {"us": v["us_per_op"], "saved": v["saved_pct"], "ci": v["ci95"], "bit": v["bit_identical"]} for k, v in rows.items()}}
        res.append(out); OUT.write_text(json.dumps(res, indent=1))
        print(f"{name}: units {len(hooks.units)} groups kept {kept}/{len(hooks.report)} | plain {rows['plain']['us_per_op']:.1f}us | "
              f"nofast {rows['plain_nofast']['saved_pct']:+.2f}% hooks {rows['hooks']['saved_pct']:+.2f}% (ci {rows['hooks']['ci95'][0]:+.2f}..{rows['hooks']['ci95'][1]:+.2f}) "
              f"seq_relaxed {rows['seq_relaxed']['saved_pct']:+.2f}% aa {rows['plain_aa']['saved_pct']:+.2f}%", flush=True)
        hooks.remove(); del layers, model, hooks, seq; torch.cuda.empty_cache()
