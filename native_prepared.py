"""Native C++ Torch dispatch (klab_native_v1) for the frozen R1/R2 BF16 projection kernels.

INTERNAL BENCHMARK CANDIDATES, not trusted policy selection. Every candidate
reports custom_win_validated=False and eager_promoted=False; nothing here reads
evidence, touches production routing or claims a speedup.

native/prepared.cpp registers CUDA-key C++ operators klab_native_v1::raw/coded
that launch the SAME cached NVRTC PTX as compressed_bf16 (M == 1) and
mtile_bf16 (M > 1), byte for byte, with the wrappers' kernel names, grids and
argument order. Its registry owns only launch metadata and deduplicated
modules (retained for the process lifetime); weights stay explicit operator
inputs. Plan IDs are opaque and process-local.

Importing this file is CPU-only and loads no native library. fake_raw/fake_coded
are stateless logical checks that work on CPU FakeTensors before any binary
exists; they are registered as the ops' Fake kernels after the library loads.
ensure_loaded() loads a matching cached binary only. Building is explicit:

    python native_prepared.py --build [--wait]

which takes the GPU lock, verifies the static CUDA host headers against their
manifest and runs the existing g++ once (bounded, project-local TMPDIR and
cache, no GPU launch). A changed source/Torch/compiler/header/runtime-library
identity needs a new explicit build, and a different library already loaded
in this process needs a fresh Python process.

Candidates prepare outside capture under the FP32-accumulation flag contract,
retain only the caller's raw weight or the validated uploaded packed buffers,
and refuse relocation/casting. Keep them (and their inputs) alive for every
graph that captured them, and never mutate their buffers. close() retires only
the C++ plan record; it never unloads code a graph may replay.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import prepared_kernels as pk
from experiment_control import GpuLock, project_path, write_json

ROOT = Path(__file__).resolve().parent
NAMESPACE = "klab_native_v1"
OPS = ("raw", "coded", "_register_plan", "_retire_plan", "plan_info")
SOURCES = ("native/prepared.cpp", "native_prepared.py")
CACHE = ".cache/native-prepared"
HEADERS = f"{CACHE}/cuda-12.8.93"
HEADERS_VERSION = "12.8.93"
LIBRARY_NAME = "libklab_native_v1.so"
GPU_LOCK = ".cache/gpu.lock"
COMPILE_TIMEOUT_SECONDS = 120
FLAGS = ("-std=c++17", "-O2", "-fPIC", "-shared")  # As in the completed build smoke.
LINKED = ("torch_cpu", "c10_cuda", "c10")
# Environment that could silently change what g++ includes or links.
_COMPILER_ENVIRONMENT = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
                         "GCC_EXEC_PREFIX", "COMPILER_PATH", "LD_RUN_PATH")

_LOAD_LOCK = threading.Lock()
_LOADED = None  # Verified build record of the library loaded by this module.


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- Fake kernels


def _check_plan_id(plan):
    if isinstance(plan, bool) or not isinstance(plan, int) or plan <= 0:
        raise ValueError("plan must be a positive registered plan ID")


def fake_raw(x, w, plan, m, n, k, threads, vector, output_dtype):
    """Stateless Fake kernel for klab_native_v1::raw: logical contracts only, no CUDA."""
    _check_plan_id(plan)
    dtype = pk._OUTPUTS[pk._output_name(output_dtype)]
    pk._check_plan(m, n, k, threads, vector, coded=False)
    pk._check_operand(x, "x", torch.bfloat16, 2, (m, k))
    pk._check_operand(w, "w", torch.bfloat16, 2, (n, k))
    if w.device != x.device:
        raise ValueError("x and w must share a device")
    return x.new_empty((m, n), dtype=dtype)


def fake_coded(x, payload, metadata, plan, m, n, k, threads, vector, output_dtype):
    """Stateless Fake kernel for klab_native_v1::coded: logical contracts only, no CUDA."""
    _check_plan_id(plan)
    dtype = pk._OUTPUTS[pk._output_name(output_dtype)]
    pk._check_plan(m, n, k, threads, vector, coded=True)
    pk._check_operand(x, "x", torch.bfloat16, 2, (m, k))
    pk._check_operand(payload, "payload", torch.uint8, 1)
    pk._check_operand(metadata, "metadata", torch.int32, 1)
    if payload.device != x.device or metadata.device != x.device:
        raise ValueError("x, payload and metadata must share a device")
    pk._check_coded_buffers(payload, metadata, n, k)
    return x.new_empty((m, n), dtype=dtype)


def _register_fakes():
    from torch._library.simple_registry import singleton

    # prepared.cpp names this module in set_python_module; Torch checks the match.
    if __name__ != "native_prepared":
        raise RuntimeError("Import native_prepared as a module (not __main__) before using its native ops")
    for name, fake in (("raw", fake_raw), ("coded", fake_coded)):
        qualname = f"{NAMESPACE}::{name}"
        if singleton.find(qualname).fake_impl.kernel is None:
            torch.library.register_fake(qualname, fake)


# ---------------------------------------------------------------- original PTX


def read_ptx(compilation):
    """The cached NVRTC PTX of a CudaModule compilation, read once and checked
    against that compilation's ptx_sha256; the returned bytes are what gets loaded."""
    try:
        name, digest = compilation["ptx_path"], compilation["ptx_sha256"]
    except (TypeError, KeyError) as error:
        raise ValueError("compilation must provide ptx_path and ptx_sha256") from error
    if not isinstance(name, str) or not name.strip() or not _is_sha256(digest):
        raise ValueError("compilation needs a project-relative ptx_path and a SHA-256 ptx_sha256")
    path = project_path(ROOT, name)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read cached PTX {name}") from error
    if _sha256_bytes(data) != digest:
        raise ValueError(f"Cached PTX {name} no longer matches the original compilation's ptx_sha256")
    return data


# ---------------------------------------------------------------- build identity


def _verified_headers():
    """Check the static NVIDIA host headers against their manifest; return its digest."""
    directory = project_path(ROOT, HEADERS)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("status") != "complete" or manifest.get("packages_installed") is not False
            or manifest.get("version") != HEADERS_VERSION):
        raise RuntimeError(f"{HEADERS}/manifest.json is not a complete verified {HEADERS_VERSION} header cache")
    listed, total = set(), 0
    for item in manifest["files"]:
        path = project_path(ROOT, item["path"])
        if not path.is_relative_to(directory):
            raise RuntimeError(f"Manifest entry outside the header cache: {item['path']}")
        data = path.read_bytes()
        if len(data) != item["bytes"] or _sha256_bytes(data) != item["sha256"]:
            raise RuntimeError(f"CUDA host header changed since verification: {item['path']}")
        listed.add(path)
        total += len(data)
    if total != manifest["retained_bytes"]:
        raise RuntimeError("CUDA host-header cache size does not match its manifest")
    unlisted = sorted(str(p.relative_to(ROOT)) for p in (directory / "include").rglob("*")
                      if p.is_file() and p.resolve() not in listed)
    if unlisted:
        raise RuntimeError(f"Unverified files in the CUDA host-header cache: {unlisted}")
    return _sha256_bytes(raw)


def _tree_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(directory).as_posix()}\0{_sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _identity():
    """Everything the native binary depends on; its digest names the cache entry."""
    torch_root = Path(torch.__file__).resolve().parent
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("No existing g++ on PATH; the native build uses only the host compiler")
    compiler_path = Path(compiler).resolve()
    runtime_include = torch_root.parent / "nvidia/cuda_runtime/include"
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    libraries = {}
    for name in LINKED:
        path = (torch_root / "lib" / f"lib{name}.so").resolve()
        stat = path.stat()
        libraries[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    identity = {
        "namespace": NAMESPACE,
        "sources": {name: _sha256_file(ROOT / name) for name in SOURCES},
        "compiler": compiler,
        "compiler_path": str(compiler_path),
        "compiler_sha256": _sha256_file(compiler_path),
        "flags": [*FLAGS, f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
        "include_dirs": [str(torch_root / "include"), str(runtime_include),
                         str(project_path(ROOT, HEADERS) / "include")],
        "library_dir": str(torch_root / "lib"),
        "linked_libraries": list(LINKED),
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": abi,
        "cuda_host_headers_manifest_sha256": _verified_headers(),
        "cuda_runtime_include_sha256": _tree_digest(runtime_include),
        "runtime_libraries": libraries,
    }
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return identity, _sha256_bytes(text.encode())


def _entry(key):
    if not _is_sha256(key):
        raise ValueError("Invalid native build key")
    return project_path(ROOT, f"{CACHE}/build/{key}")


def _command(identity, source, library):
    includes = [part for directory in identity["include_dirs"] for part in ("-I", directory)]
    return [identity["compiler"], *identity["flags"], *includes, str(source),
            "-L", identity["library_dir"], f"-Wl,-rpath,{identity['library_dir']}",
            *(f"-l{name}" for name in identity["linked_libraries"]), "-o", str(library)]


def _verified_build(identity, key):
    """The complete cache entry for exactly this identity, with its binary re-hashed."""
    directory = _entry(key)
    record_path = directory / "build.json"
    if not record_path.is_file():
        raise RuntimeError(f"No native build matches the current sources/Torch/compiler/headers (key {key[:16]}); "
                           "run `python native_prepared.py --build --wait` (nothing compiles implicitly)")
    record = json.loads(record_path.read_text())
    if record.get("status") != "complete" or record.get("key") != key or record.get("identity") != identity:
        raise RuntimeError(f"Native build entry {directory.relative_to(ROOT)} is incomplete or does not match "
                           "its identity; inspect it before rebuilding")
    library = project_path(ROOT, record["library_path"])
    if library.parent != directory or _sha256_file(library) != record.get("library_sha256"):
        raise RuntimeError(f"Cached native binary {record['library_path']} changed after its build; refusing to load it")
    return record, library


def _registered():
    return hasattr(getattr(torch.ops, NAMESPACE), "raw")


def _foreign_library_loaded(path=None):
    loaded = {os.path.realpath(p) for p in torch.ops.loaded_libraries}
    others = [p for p in loaded if Path(p).name == LIBRARY_NAME and p != path]
    return bool(others) or (_registered() and path not in loaded)


def ensure_loaded():
    """Load the matching cached native library once per process and register its
    Fake kernels. Never compiles; returns a copy of the verified build record."""
    global _LOADED
    with _LOAD_LOCK:
        if _LOADED is None:
            identity, key = _identity()
            record, library = _verified_build(identity, key)
            path = os.path.realpath(library)
            if _foreign_library_loaded(path):
                raise RuntimeError(f"A different {NAMESPACE} library is already loaded in this process; "
                                   "restart Python to use the current build")
            if path not in {os.path.realpath(p) for p in torch.ops.loaded_libraries}:
                torch.ops.load_library(path)
            _register_fakes()
            _LOADED = record
        return copy.deepcopy(_LOADED)


# ---------------------------------------------------------------- explicit build


def _compiler_environment(tmp):
    environment = {key: value for key, value in os.environ.items() if key not in _COMPILER_ENVIRONMENT}
    environment.update(TMPDIR=str(tmp), LC_ALL="C")
    return environment


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def build(wait=False):
    """Explicit serialized g++ build into .cache/native-prepared; launches no GPU work.

    A complete matching entry is verified and reused. Every attempt keeps its
    copied source, compiler output and metadata under attempts/; only a
    compiled, loaded and registered library is moved to build/<identity key>.
    """
    tmp = project_path(ROOT, f"{CACHE}/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)
    with GpuLock(project_path(ROOT, GPU_LOCK), wait=wait):
        if _foreign_library_loaded():
            raise RuntimeError(f"{NAMESPACE} is already loaded in this process; build from a fresh Python process")
        identity, key = _identity()
        final = _entry(key)
        if final.exists():
            record, _ = _verified_build(identity, key)
            print(f"Native build {key[:16]} is already complete: {record['library_path']}", flush=True)
            return record
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        attempt = project_path(ROOT, f"{CACHE}/attempts/{stamp}-{key[:16]}-{os.getpid()}")
        attempt.mkdir(parents=True, exist_ok=False)
        # Compile a private copy of exactly the bytes the identity hashed.
        source_bytes = (ROOT / SOURCES[0]).read_bytes()
        if _sha256_bytes(source_bytes) != identity["sources"][SOURCES[0]]:
            raise RuntimeError(f"{SOURCES[0]} changed during the build; retry")
        source = attempt / "prepared.cpp"
        source.write_bytes(source_bytes)
        library = attempt / LIBRARY_NAME
        command = _command(identity, source, library)
        environment = _compiler_environment(tmp)
        version = subprocess.run([identity["compiler"], "--version"], cwd=ROOT, env=environment,
                                 capture_output=True, text=True, timeout=30)
        record = {
            "kind": "klab-native-v1-build", "status": "running", "key": key, "identity": identity,
            "pid": os.getpid(), "created_utc": _now(), "command": command,
            "compiler_version": version.stdout.strip(),
            "removed_compiler_environment": sorted(k for k in _COMPILER_ENVIRONMENT if k in os.environ),
            "tmpdir": str(tmp.relative_to(ROOT)),
            "runtime_library_sha256": {name: _sha256_file(info["path"])
                                       for name, info in identity["runtime_libraries"].items()},
            "attempt_path": str(attempt.relative_to(ROOT)),
            "library_path": str((final / LIBRARY_NAME).relative_to(ROOT)),
            "gpu_operations_launched": False, "packages_installed": False,
            "existing_sources_modified": False,
        }
        metadata = attempt / "build.json"
        write_json(metadata, record)
        try:
            try:
                result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                                        timeout=COMPILE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                record.update(build_stdout=_text(error.stdout), build_stderr=_text(error.stderr))
                raise RuntimeError(f"g++ exceeded the {COMPILE_TIMEOUT_SECONDS}s compile bound") from error
            record.update(returncode=result.returncode, build_stdout=result.stdout, build_stderr=result.stderr)
            if result.returncode:
                raise RuntimeError(f"g++ failed; see {metadata.relative_to(ROOT)}")
            data = library.read_bytes()
            record.update(library_sha256=_sha256_bytes(data), library_bytes=len(data))
            torch.ops.load_library(str(library))  # Registration check only; launches nothing.
            ops = getattr(torch.ops, NAMESPACE)
            record.update(schemas={name: str(getattr(ops, name).default._schema) for name in OPS},
                          cuda_initialized_after_load=torch.cuda.is_initialized(),
                          status="complete", finished_utc=_now())
            write_json(metadata, record)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(attempt, final)
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=_now())
            write_json(metadata, record)
            raise
        print(f"Built and registered {record['library_path']}; no GPU operation launched. "
              "Use it from a fresh Python process.", flush=True)
        return record


# ---------------------------------------------------------------- candidates


def _launch_contract(kind, m, n, threads, vector, output):
    """Kernel name and grid from the wrapper formulas, to cross-check the C++ plan."""
    import mtile_bf16 as mt

    tile, rows = mt.tile_for_m(m), threads // 32
    name = f"{kind}_{output}_t{threads}_v{vector}"
    if m == 1:  # compressed_bf16 R1 launch.
        return f"cbf16_{name}", [(n + rows - 1) // rows, m], tile
    return f"mtbf16_{name}_m{tile}", [(n + rows - 1) // rows, (m + tile - 1) // tile], tile


class _NativeCandidate(torch.nn.Module):
    """Shared preparation for one device-bound native (M,N,K, config, output) plan."""

    custom_win_validated = False  # Benchmark candidate only; never a policy decision.
    eager_promoted = False

    def __init__(self):
        super().__init__()
        self.plan_id = None
        self.closed = False

    def _contract(self, kind, tokens, n, k, threads, vector, output_dtype):
        """Logical checks only; runs before anything touches CUDA or the native library."""
        self.output_dtype = pk._output_name(output_dtype)
        pk._check_plan(tokens, n, k, threads, vector, coded=kind == "coded")
        self.kind = kind
        self.m, self.n, self.k = tokens, n, k
        self.shape_mnk = (tokens, n, k)
        self.threads, self.vector = threads, vector

    def _prepare(self, device, packed=None):
        """Outside capture, under the FP32-accumulation flag contract: register the
        original module's exact PTX and launch metadata with the C++ registry."""
        import compressed_bf16 as cb
        import mtile_bf16 as mt

        pk._check_mirrors(cb, mt)
        index = cb._device_index(device)
        r1 = self.m == 1
        with torch.cuda.device(index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare native candidates before CUDA graph capture")
            self.precision_flags = {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction":
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
            if any(self.precision_flags.values()):
                raise ValueError("Set allow_tf32 and allow_bf16_reduced_precision_reduction to False "
                                 "before preparing this FP32-accumulation contract")
            if packed is not None and packed.module is not cb.get_module(index):
                raise ValueError("packed must come from compressed_bf16.upload() on this device")
            record = ensure_loaded()
            module, sources = (cb.get_module(index), cb.KERNEL_SOURCES) if r1 else \
                (mt.get_module(index), mt.KERNEL_SOURCES)
            compilation = module.compilation
            kernel, grid, tile = _launch_contract(self.kind, self.m, self.n, self.threads, self.vector,
                                                  self.output_dtype)
            if (compilation["device"] != index or tuple(compilation["sources"]) != tuple(sources)
                    or kernel not in compilation["kernels"]):
                raise RuntimeError(f"The original {'R1' if r1 else 'R2'} module does not provide {kernel}")
            ptx = read_ptx(compilation)
            # Binds this thread to the device's primary context, which the registry records.
            torch.cuda.synchronize(index)
            ops = getattr(torch.ops, NAMESPACE)
            self.plan_id = ops._register_plan(
                torch.frombuffer(bytearray(ptx), dtype=torch.uint8), index, self.m, self.n, self.k,
                self.threads, self.vector, self.kind, self.output_dtype, compilation["ptx_sha256"])
            self.plan_info = json.loads(ops.plan_info(self.plan_id))
        expected = {"plan": self.plan_id, "kind": self.kind, "output_dtype": self.output_dtype, "device": index,
                    "shape_mnk": list(self.shape_mnk), "threads": self.threads, "vector": self.vector,
                    "tile": tile, "route": "R1" if r1 else "R2", "kernel": kernel, "grid": grid,
                    "block": [self.threads, 1, 1], "dynamic_shared_bytes": 0,
                    "ptx_sha256": compilation["ptx_sha256"]}
        if {key: self.plan_info.get(key) for key in expected} != expected:
            raise RuntimeError("C++ launch metadata disagrees with the frozen wrapper formulas")
        self._op = getattr(ops, self.kind).default
        self.device = torch.device("cuda", index)
        self.native = {"namespace": NAMESPACE, "key": record["key"], "library_path": record["library_path"],
                       "library_sha256": record["library_sha256"]}
        self.config = {"revision": mt.REVISION, "threads": self.threads, "vector": self.vector, "tile": tile,
                       "route": "R1 delegation (M == 1)" if r1 else "R2 M-tile", "kernel": kernel,
                       "grid": grid, "dispatch": f"{NAMESPACE} C++ CUDA-key operator"}
        self.ptx = {key: compilation[key] for key in
                    ("ptx_sha256", "ptx_path", "cache_key", "compiler", "architecture", "options", "sources")}
        self.implementation = mt.implementation_fingerprint(index)

    def _finish(self, started):
        with torch.cuda.device(self.device):
            # One launch before any capture; it also exercises the C++ tensor guards.
            self.linear(torch.zeros((self.m, self.k), dtype=torch.bfloat16, device=self.device))
        self.setup_seconds = time.perf_counter() - started

    def _apply(self, fn, recurse=True):
        raise RuntimeError(f"{type(self).__name__} is a device-bound prepared plan; "
                           "reprepare it instead of moving or casting it")

    def forward(self, x):
        return self.linear(x)

    def close(self):
        """Retire the C++ plan record (idempotent). Its module stays loaded for the
        process, so graphs that captured it remain valid while their buffers live."""
        if not self.closed:
            if self.plan_id is not None:
                getattr(torch.ops, NAMESPACE)._retire_plan(self.plan_id)
            self.closed = True

    def describe(self):
        """JSON-ready setup record for a later native-host study; not a performance claim."""
        return {"kind": self.kind, "shape_mnk": list(self.shape_mnk), "config": dict(self.config),
                "output_dtype": self.output_dtype, "storage": dict(self.storage),
                "provenance": copy.deepcopy(self.provenance), "implementation": self.implementation,
                "precision_flags": dict(self.precision_flags), "device": str(self.device),
                "plan_id": self.plan_id, "plan_info": copy.deepcopy(self.plan_info), "closed": self.closed,
                "native": dict(self.native), "setup_seconds": self.setup_seconds,
                "custom_win_validated": self.custom_win_validated, "eager_promoted": self.eager_promoted}


class NativeCodedCandidate(_NativeCandidate):
    """Coded projection over a compressed_bf16.upload() result; retains only its packed buffers."""

    def __init__(self, packed, tokens=1, *, threads, vector, output_dtype):
        super().__init__()
        started = time.perf_counter()
        import compressed_bf16 as cb

        if not isinstance(packed, cb.PackedBF16):
            raise ValueError("packed must come from compressed_bf16.upload()")
        payload, metadata = packed.payload, packed.metadata
        pk._check_cuda_operand(payload, "payload", torch.uint8, 1)
        pk._check_cuda_operand(metadata, "metadata", torch.int32, 1)
        if metadata.device != payload.device:
            raise ValueError("payload and metadata must share a device")
        if packed.format_version != cb.FORMAT_VERSION:
            raise ValueError(f"Unsupported packed format version; expected {cb.FORMAT_VERSION}")
        self._contract("coded", tokens, packed.n, packed.k, threads, vector, output_dtype)
        pk._check_coded_buffers(payload, metadata, self.n, self.k)
        self.register_buffer("payload", payload, persistent=False)
        self.register_buffer("metadata", metadata, persistent=False)
        self.storage = {"format": f"compressed_bf16_v{packed.format_version}",
                        "bytes": packed.storage_bytes, "raw_bf16_bytes": 2 * self.n * self.k}
        try:
            self._prepare(payload.device, packed)
            self.provenance = {"source_sha256": packed.source_sha256, "format_version": packed.format_version,
                               "ptx": dict(self.ptx)}
            self._finish(started)
        except BaseException:
            self.close()
            raise

    def linear(self, x):
        """Allocating klab_native_v1::coded (Fake/torch.compile-safe)."""
        return self._op(x, self.payload, self.metadata, self.plan_id, self.m, self.n, self.k,
                        self.threads, self.vector, self.output_dtype)


class NativeRawCandidate(_NativeCandidate):
    """Raw projection over the caller's original BF16 weight [N,K], retained without copying."""

    def __init__(self, w, tokens=1, *, threads, vector, output_dtype):
        super().__init__()
        started = time.perf_counter()
        pk._check_cuda_operand(w, "w", torch.bfloat16, 2)
        self._contract("raw", tokens, *w.shape, threads, vector, output_dtype)
        self.register_buffer("w", w, persistent=False)
        self.storage = {"format": "raw_bf16", "bytes": 2 * self.n * self.k}
        try:
            self._prepare(w.device)
            self.provenance = {"weights": "caller's original tensor; not copied or hashed", "ptx": dict(self.ptx)}
            self._finish(started)
        except BaseException:
            self.close()
            raise

    def linear(self, x):
        """Allocating klab_native_v1::raw (Fake/torch.compile-safe)."""
        return self._op(x, self.w, self.plan_id, self.m, self.n, self.k,
                        self.threads, self.vector, self.output_dtype)


# Torch imports this module (prepared.cpp's set_python_module) when a Fake kernel
# is first needed for an already loaded library; register them then.
if __name__ == "native_prepared" and _registered():
    _register_fakes()


# ---------------------------------------------------------------- CLI


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Explicit project-local build of the {NAMESPACE} dispatcher")
    parser.add_argument("--build", action="store_true", help="compile native/prepared.cpp with the existing g++")
    parser.add_argument("--wait", action="store_true", help="queue behind running GPU jobs instead of failing")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("nothing to do; pass --build (candidates never compile implicitly)")
    record = build(wait=args.wait)
    print(json.dumps({key: record.get(key) for key in ("status", "key", "library_path", "library_sha256")},
                     indent=2))


if __name__ == "__main__":
    main()
