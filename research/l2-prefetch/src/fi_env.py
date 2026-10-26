"""Make a separately installed FlashInfer importable (nothing is installed into the measured .venv).

Configure with environment variables:
  KLAB_FI_SITE    directory holding flashinfer (pip/uv install --target, with --no-deps plus its light deps)
  KLAB_CUDA_HOME  CUDA toolkit with nvcc matching torch's CUDA (e.g. a conda env with cuda-nvcc=12.8)
  FLASHINFER_WORKSPACE_BASE  JIT cache; must be a path without spaces (ninja cannot handle them)
Defaults point inside the repository's ignored .cache directory.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SITE = os.environ.get("KLAB_FI_SITE", str(ROOT / ".cache/flashinfer-site"))
CUDA = os.environ.get("KLAB_CUDA_HOME", str(ROOT / ".cache/cuda-toolkit"))

os.environ.setdefault("CUDA_HOME", CUDA)
os.environ["PATH"] = f"{CUDA}/bin:{SITE}/bin:" + os.environ.get("PATH", "")  # nvcc and ninja
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / ".cache/flashinfer-jit"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
if SITE not in sys.path:
    sys.path.append(SITE)  # appended, so .venv packages (torch, numpy) always win
