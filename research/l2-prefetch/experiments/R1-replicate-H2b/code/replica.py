"""Run the frozen H2b main() with a different output path (no change to the frozen file)."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import importlib.util, sys
from pathlib import Path
src = Path(KLAB + "/research/l2-prefetch/experiments/H2b-autotune/code/run.py")
spec = importlib.util.spec_from_file_location("h2b_run", src); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = Path(sys.argv[1])
mod.main()
