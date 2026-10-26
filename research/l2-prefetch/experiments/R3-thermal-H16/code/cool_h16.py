import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import importlib.util, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness2
temps = []
_orig = harness2.run_variants
def temp():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
def cooled(*a, **k):
    t0 = time.monotonic()
    while temp() > 60 and time.monotonic() - t0 < 180:
        time.sleep(5)
    temps.append(temp())
    return _orig(*a, **k)
harness2.run_variants = cooled
src = Path(KLAB + "/research/l2-prefetch/experiments/H16-strong-baseline/code/run.py")
spec = importlib.util.spec_from_file_location("h16_run", src); mod = importlib.util.module_from_spec(spec)
mod_out = Path(sys.argv[1])
code = src.read_text().replace('OUT = Path(KLAB + "/research/l2-prefetch/experiments/H16-strong-baseline/results/h16.json")', f'OUT = Path("{mod_out}")')
exec(compile(code, str(src), "exec"), {"__name__": "__main__", "__file__": str(src)})
Path(str(mod_out).replace(".json", "-temps.json")).write_text(json.dumps(temps))
