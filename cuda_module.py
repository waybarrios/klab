"""Isolated NVRTC module over the existing C++ C ABI (native/build/libkernel_lab.so).

Importing this file is CPU-only: torch, NVRTC and the CUDA driver load when a
CudaModule is constructed. The fingerprint binds only the module's own sources
plus the shared compiler/bridge files, so editing an unrelated legacy .cu file
does not change it. Nothing here touches the legacy KernelLab or CUDA_SOURCES.
"""

from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

from experiment_control import cached_ptx, project_path

ROOT = Path(__file__).resolve().parent
LIBRARY = "native/build/libkernel_lab.so"
SHARED_DEPENDENCIES = ("cuda_module.py", "kernel_lab.py", "experiment_control.py",
                       "native/driver.cpp", LIBRARY)
# Architectural limits for every compute capability this project targets. A
# kernel with tighter __launch_bounds__ still fails loudly in cuLaunchKernel.
MAX_BLOCK_THREADS = 1024
MAX_GRID = (2**31 - 1, 65535)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_names(root, names):
    """Validate project-relative files; return their resolved relative names."""
    if isinstance(names, (str, bytes)):
        raise ValueError("Expected a sequence of project-relative paths")
    resolved = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Paths must be nonempty strings")
        path = project_path(root, name)
        if path == root:
            raise ValueError("Path must name a file inside the project")
        if not path.is_file():
            raise FileNotFoundError(f"Missing module file: {name}")
        resolved.append(path.relative_to(root).as_posix())
    return resolved


def _source_names(root, sources):
    names = _project_names(root, sources)
    if not names or len(set(names)) != len(names):
        raise ValueError("Sources must be a nonempty sequence of distinct files")
    return names


def module_fingerprint(sources, architecture, compiler, options, *, root=ROOT):
    """JSON-safe identity of an isolated module and its shared compiler/bridge.

    Ordered sources, target, compiler identity and options are all bound;
    files outside `sources` and SHARED_DEPENDENCIES are deliberately excluded.
    """
    root = Path(root).resolve()
    if isinstance(options, (str, bytes)):
        raise ValueError("options must be a sequence of strings")
    names = _source_names(root, sources)
    dependencies = _project_names(root, SHARED_DEPENDENCIES)
    identity = {
        "sources": names,
        "dependencies": dependencies,
        "source_sha256": {name: _sha256_file(root / name)
                          for name in dict.fromkeys(names + dependencies)},
        "architecture": str(architecture),
        "compiler": str(compiler),
        "options": [str(option) for option in options],
    }
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {**identity, "fingerprint_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _nvrtc_identity():
    """The same NVRTC library _compile_ptx_uncached loads: version + content hash."""
    import nvidia.cuda_nvrtc

    library = Path(next(iter(nvidia.cuda_nvrtc.__path__))) / "lib/libnvrtc.so.12"
    nvrtc = ct.CDLL(str(library))
    nvrtc.nvrtcVersion.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
    major, minor = ct.c_int(), ct.c_int()
    if nvrtc.nvrtcVersion(ct.byref(major), ct.byref(minor)):
        raise RuntimeError("Cannot query NVRTC version")
    return f"NVRTC {major.value}.{minor.value}:sha256={_sha256_file(library)}", str(library)


class CudaModule:
    """One device's independent PTX module, launched through the existing C ABI.

    Launches enqueue asynchronously on PyTorch's current stream for this device:
    2D grid, 1D block, zero dynamic shared memory. Keep this object alive until
    all launches AND captured graphs using it have finished. close() waits for
    pending work on its device, then unloads; it is idempotent. Never replay a
    captured graph after close(). There is no __del__: call close() explicitly.
    """

    def __init__(self, sources: tuple[str, ...], names: tuple[str, ...], device: int):
        root = ROOT.resolve()
        sources = _source_names(root, sources)
        if isinstance(names, (str, bytes)):
            raise ValueError("names must be a sequence of kernel names")
        names = tuple(names)
        if not names or len(set(names)) != len(names) or not all(
                isinstance(name, str) and name.isidentifier() for name in names):
            raise ValueError("Kernel names must be distinct C identifiers")
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a nonnegative CUDA device index")

        import torch
        from kernel_lab import _compile_ptx_uncached

        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is not available")
        compiler, nvrtc_library = _nvrtc_identity()
        self.device = device
        self._module = ct.c_void_p()
        self._functions = {}
        with torch.cuda.device(device):
            # Make the runtime's primary context current before loading PTX.
            torch.empty(1, device=torch.device("cuda", device))
            architecture = "".join(map(str, torch.cuda.get_device_capability(device)))
            # Exactly the options _compile_ptx_uncached passes to NVRTC.
            options = (f"--gpu-architecture=compute_{architecture}", "--std=c++17")
            fingerprint = module_fingerprint(sources, architecture, compiler, options, root=root)
            parts = [(root / name).read_bytes() for name in fingerprint["sources"]]
            if any(hashlib.sha256(part).hexdigest() != fingerprint["source_sha256"][name]
                   for name, part in zip(fingerprint["sources"], parts)):
                raise RuntimeError("Module source changed while it was being compiled")
            source = b"\n".join(parts)
            # Cache the complete declared build, including the shared compiler
            # helper. Reusing source-only PTX after that helper changes can
            # silently retain old compile behavior despite a new profile hash.
            key = fingerprint["fingerprint_sha256"]
            directory = project_path(root, ".cache/ptx")
            ptx, hit = cached_ptx(directory, key, lambda: _compile_ptx_uncached(source, architecture),
                                  force=os.environ.get("KERNEL_LAB_REBUILD") == "1")
            self._lib = ct.CDLL(str(root / LIBRARY))
            pointer = ct.c_void_p
            self._lib.lab_load_ptx.argtypes = [ct.c_char_p, ct.POINTER(pointer)]
            self._lib.lab_get_kernel.argtypes = [pointer, ct.c_char_p, ct.POINTER(pointer)]
            self._lib.lab_launch.argtypes = [
                pointer, ct.c_uint, ct.c_uint, ct.c_uint, pointer, ct.POINTER(pointer)
            ]
            self._lib.lab_unload.argtypes = [pointer]
            self._lib.lab_error_string.argtypes = [ct.c_int]
            self._lib.lab_error_string.restype = ct.c_char_p
            self._check(self._lib.lab_load_ptx(ptx, ct.byref(self._module)))
            try:
                for name in names:
                    function = pointer()
                    self._check(self._lib.lab_get_kernel(self._module, name.encode(),
                                                         ct.byref(function)))
                    self._functions[name] = function
            except BaseException:
                self._lib.lab_unload(self._module)
                self._module = ct.c_void_p()
                raise
        self._fingerprint = fingerprint
        self._compilation = {
            "ptx_sha256": hashlib.sha256(ptx).hexdigest(), "cache_key": key, "cache_hit": hit,
            "compiler": compiler, "nvrtc_library": nvrtc_library, "architecture": architecture,
            "options": list(options), "ptx_path": (directory / f"{key}.ptx").relative_to(root).as_posix(),
            "sources": list(fingerprint["sources"]), "kernels": list(names), "device": device,
            "launch_abi": f"{LIBRARY} lab_launch: 2D grid, 1D block, zero dynamic shared memory",
        }

    @property
    def compilation(self):
        return copy.deepcopy(self._compilation)

    @property
    def fingerprint(self):
        return copy.deepcopy(self._fingerprint)

    @property
    def closed(self):
        return not self._module.value

    def _check(self, status):
        if status:
            raise RuntimeError(self._lib.lab_error_string(status).decode())

    def launch(self, name, grid, threads, *args):
        """Enqueue `name` on the current stream; args are CUDA tensors or ctypes scalars."""
        import torch

        if self.closed:
            raise RuntimeError("CudaModule is closed")
        if name not in self._functions:
            raise ValueError(f"Kernel {name!r} was not loaded by this module")
        grid = tuple(grid)
        if len(grid) != 2:
            raise ValueError("grid must be (x, y)")
        for value, limit in ((grid[0], MAX_GRID[0]), (grid[1], MAX_GRID[1]),
                             (threads, MAX_BLOCK_THREADS)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
                raise ValueError("Grid/block dimensions must be positive integers within CUDA limits")
        device = torch.device("cuda", self.device)
        values = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.device != device:
                    raise ValueError("All tensors must be on this module's CUDA device")
                values.append(ct.c_void_p(arg.data_ptr()))
            elif isinstance(arg, ct._SimpleCData):
                values.append(arg)
            else:
                raise TypeError("Kernel arguments must be torch tensors or ctypes scalars")
        params = (ct.c_void_p * max(1, len(values)))(
            *[ct.cast(ct.byref(value), ct.c_void_p) for value in values])
        stream = torch.cuda.current_stream(self.device)
        with torch.cuda.device(self.device):
            self._check(self._lib.lab_launch(self._functions[name], grid[0], grid[1], threads,
                                             ct.c_void_p(stream.cuda_stream), params))
        # ctypes bypasses the caching allocator's normal stream bookkeeping.
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg.record_stream(stream)

    def close(self):
        if self._module.value:
            import torch

            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._check(self._lib.lab_unload(self._module))
            self._module = ct.c_void_p()
            self._functions.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
