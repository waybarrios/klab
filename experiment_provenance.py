"""Common source/build and environment fingerprints for causal comparisons."""

import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CUDA_SOURCES=("kernels.cu","projection_kernels.cu","fp4_kernels.cu","int8_kernels.cu",
              "lossless_kernels.cu","adaptive_kernels.cu","half_kernels.cu")


def source_hashes(extra=()):
    paths=set(CUDA_SOURCES)|{"native/driver.cpp","native/build/libkernel_lab.so","kernel_lab.py",
          "bench.py","validate_challenge.py","experiment_control.py","experiment_provenance.py",
          "cublas_baseline.py","lossless_weights.py","packed_fp32.py","gpu_monitor.py","requirements.lock"}|set(extra)
    result={}
    for name in sorted(paths):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT): raise ValueError("Provenance paths must stay in project")
        result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def environment(lab,vendor):
    import torch
    from gpu_unlock_status import inspect
    return {"inventory":inspect(),"torch":str(torch.__version__),"cuda":torch.version.cuda,
            "cublas_version":vendor.version,"compilation_cache_key":lab.compilation["cache_key"]}
