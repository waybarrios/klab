"""CPU-only helpers for experiment paths, serialization, and compilation cache."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def project_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must stay inside {root}")
    return path


def parse_shape(value):
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("Shape must be M,N,K, for example 1,4096,4096") from error
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("Shape must contain three positive integers M,N,K")
    return shape


def run_output(root, value):
    path = project_path(root, value)
    if path.suffix.lower() != ".json":
        raise ValueError("Run output must be a .json file, separate from control/cache files")
    return path


def report_output(root, source, value=None):
    source = project_path(root, source)
    path = project_path(root, value or source.with_suffix(".md"))
    if path == source or path.suffix.lower() != ".md":
        raise ValueError("Report output must be a distinct .md file; raw JSON is preserved")
    return path


def with_run_metadata(report, metadata=None):
    result = dict(report)
    for key in ("pid", "run_id", "command", "created_utc"):
        if metadata is not None and key in metadata:
            result[key] = metadata[key]
    return result


class GpuLock:
    def __init__(self, path, wait=False):
        self.path = Path(path)
        self.wait = wait
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | (0 if self.wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            self.file.close()
            self.file = None
            raise RuntimeError("Another GPU experiment is running; use --wait to queue") from error
        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()))
        self.file.flush()
        return self

    def __exit__(self, *_):
        if self.file is not None:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None


def cache_key(source, architecture, compiler, options):
    metadata = json.dumps({"source": hashlib.sha256(source).hexdigest(),
                           "architecture": architecture, "compiler": compiler,
                           "options": list(options)}, sort_keys=True).encode()
    return hashlib.sha256(metadata).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, (json.dumps(data, indent=2) + "\n").encode())


def cached_ptx(directory, key, compile_fn, force=False):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid compilation cache key")
    path = Path(directory) / f"{key}.ptx"
    if path.is_file() and not force:
        data = path.read_bytes()
        if b".version" in data and b".target" in data:
            return data, True
    data = compile_fn()
    if b".version" not in data or b".target" not in data:
        raise ValueError("Compiler did not return valid PTX text")
    atomic_write(path, data)
    return data, False
