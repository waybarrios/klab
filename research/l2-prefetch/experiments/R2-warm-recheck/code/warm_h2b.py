import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness, harness2
_orig = harness.run_variants
def warmed(variants, x0, calls, rounds=15, baseline="torch"):
    import torch
    with torch.no_grad():
        harness2.warm(variants, x0, 2.0)
    return _orig(variants, x0, calls, rounds=rounds, baseline=baseline)
harness.run_variants = warmed
src = Path(KLAB + "/research/l2-prefetch/experiments/H2b-autotune/code/run.py")
spec = importlib.util.spec_from_file_location("h2b_run", src); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = Path(sys.argv[1]); mod.main()
