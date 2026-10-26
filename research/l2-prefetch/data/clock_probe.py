"""Read-only probe: does sustained load raise the memory clock above P5's 810 MHz?"""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[3]) if "__file__" in globals() else "."
import subprocess, sys, time, threading
sys.path.insert(0, KLAB)
import torch
from pathlib import Path
from experiment_control import GpuLock

samples = []
stop = False
def sampler():
    while not stop:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        samples.append((time.time(), out)); time.sleep(0.4)

def phase(name, fn, seconds):
    t0 = time.time(); n = 0
    while time.time() - t0 < seconds:
        fn(); n += 1
    torch.cuda.synchronize(); print(f"--- {name}: {n} iters", flush=True)
    return t0, time.time()

with GpuLock(Path(KLAB + "/.cache/gpu.lock"), wait=True):
    th = threading.Thread(target=sampler); th.start()
    big = torch.empty(256 << 20, dtype=torch.uint8, device="cuda"); dst = torch.empty_like(big)
    W = [torch.randn(512, 512, dtype=torch.bfloat16, device="cuda") for _ in range(160)]  # 80 MiB ring
    x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
    marks = {}
    marks["idle"] = (time.time(), time.time() + 2); time.sleep(2)
    marks["copy"] = phase("copy 256MiB", lambda: dst.copy_(big), 6)
    def gemv():
        for w in W: torch.mm(x, w.t())
        torch.cuda.synchronize()
    marks["gemv"] = phase("ring gemv 1x512x512", gemv, 6)
    # bandwidth during copy
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record();  [dst.copy_(big) for _ in range(20)]; e1.record(); torch.cuda.synchronize()
    print(f"copy bandwidth ~ {2*20*big.numel()/e0.elapsed_time(e1)/1e6:.1f} GB/s")
    stop = True; th.join()
for name, (a, b) in marks.items():
    print("==", name); [print("  ", s) for t, s in samples if a <= t <= b + 0.5]
