"""Bounded, read-only nvidia-smi sampling during a GPU experiment."""

import subprocess
import threading
import time


class GpuMonitor:
    def __init__(self,interval=0.5):
        self.interval=interval
        self.samples=[]
        self.stop=threading.Event()

    def sample(self):
        try:
            result=subprocess.run(["nvidia-smi","--query-gpu=name,pstate,temperature.gpu,power.draw,"
                                   "clocks.current.sm,clocks.current.memory,clocks_event_reasons.sw_power_cap,"
                                   "clocks_event_reasons.sw_thermal_slowdown","--format=csv,noheader"],
                                  capture_output=True,text=True,timeout=5)
            self.samples.append({"elapsed_seconds":time.monotonic()-self.start,
                                 "state":result.stdout.strip(),"ok":result.returncode==0})
        except (OSError,subprocess.TimeoutExpired) as error:
            self.samples.append({"elapsed_seconds":time.monotonic()-self.start,"error":str(error),"ok":False})

    def worker(self):
        self.sample()
        while not self.stop.wait(self.interval): self.sample()

    def __enter__(self):
        self.start=time.monotonic()
        self.thread=threading.Thread(target=self.worker,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*_):
        self.stop.set()
        self.thread.join(timeout=6)
        if not self.thread.is_alive(): self.sample()
