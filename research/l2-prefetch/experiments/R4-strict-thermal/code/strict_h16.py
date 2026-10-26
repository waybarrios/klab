import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness2
temps = []
_orig = harness2.run_variants
def temp():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
def cooled(*a, **k):
    t0 = time.monotonic()
    while temp() > 60:
        if time.monotonic() - t0 > 45 * 60:
            raise SystemExit("R4 abort: GPU did not reach <= 60 C within 45 min")
        time.sleep(5)
    temps.append(temp())
    return _orig(*a, **k)
harness2.run_variants = cooled
out = Path(sys.argv[1])
src = Path(KLAB + "/research/l2-prefetch/experiments/H16-strong-baseline/code/run.py")
code = src.read_text().replace('OUT = Path(KLAB + "/research/l2-prefetch/experiments/H16-strong-baseline/results/h16.json")', f'OUT = Path("{out}")')
try:
    exec(compile(code, str(src), "exec"), {"__name__": "__main__", "__file__": str(src)})
finally:
    Path(str(out).replace(".json", "-temps.json")).write_text(json.dumps(temps))
