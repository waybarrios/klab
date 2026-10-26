"""GPU integration test: cache invalidation, streams, capture and module lifetime."""
import ctypes as ct
import shutil
import tempfile
from pathlib import Path

import torch
import cuda_module
from experiment_control import GpuLock

ROOT = Path(__file__).resolve().parents[1]

with GpuLock(ROOT / ".cache/gpu.lock"):
    with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
        root = Path(directory)
        for name in (*cuda_module.SHARED_DEPENDENCIES, "regime_kernels.cu"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        original_root = cuda_module.ROOT
        cuda_module.ROOT = root
        try:
            first = cuda_module.CudaModule(("regime_kernels.cu",), ("regime_launch",), 0)
            assert not first.compilation["cache_hit"]
            first.close()
            with (root / "kernel_lab.py").open("a") as target:
                target.write("\n# Changed declared compiler dependency in isolated fixture\n")
            second = cuda_module.CudaModule(("regime_kernels.cu",), ("regime_launch",), 0)
            try:
                assert not second.compilation["cache_hit"], "Changed compiler dependency reused stale PTX cache"
                fingerprint = second.fingerprint
                counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    second.launch("regime_launch", (1, 1), 32, counter)
                torch.cuda.current_stream().wait_stream(stream)
                assert counter.item() == 1
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    second.launch("regime_launch", (1, 1), 32, counter)
                graph.replay()
                assert counter.item() == 2
                graph.reset()
            finally:
                second.close()
            second.close()
            try:
                second.launch("regime_launch", (1, 1), 32, counter)
                raise AssertionError("Closed module accepted a launch")
            except RuntimeError:
                pass
            (root / "unrelated.cu").write_text("unrelated source")
            third = cuda_module.CudaModule(("regime_kernels.cu",), ("regime_launch",), 0)
            try:
                assert third.compilation["cache_hit"]
                assert third.fingerprint == fingerprint
            finally:
                third.close()
        finally:
            cuda_module.ROOT = original_root
print("PASS: isolated cache dependencies, streams, graph, close and unrelated-source stability")
