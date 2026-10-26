"""Verify a safetensors stream while retaining selected native BF16 tensors.

No model code, pickle loader or GPU dependency is used. Files are uint16
little-endian bit patterns; the manifest records their source and digests.
"""

import hashlib
import json
import math
import re
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

from experiment_control import project_path, run_output, write_json

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = (0, 1, 2, 3, 8, 16, 23)
MAX_RETAINED = 512 * 1024**2
FREE_FLOOR = 768 * 1024**2
WIDTHS = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
          "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors header key")
        result[key] = value
    return result


def _read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise ValueError("Truncated safetensors stream")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _parse_header(stream, expected_size):
    prefix = _read_exact(stream, 8)
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= 16 * 1024**2 or 8 + length > expected_size:
        raise ValueError("Invalid safetensors header length")
    encoded = _read_exact(stream, length)
    try:
        header = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    tensors = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", name):
            raise ValueError("Unsafe tensor name")
        if not isinstance(value, dict):
            raise ValueError("Invalid tensor descriptor")
        shape, offsets, dtype = value.get("shape"), value.get("data_offsets"), value.get("dtype")
        if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets) or dtype not in WIDTHS):
            raise ValueError("Invalid safetensors shape, dtype or offsets")
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * WIDTHS[dtype]:
            raise ValueError("Safetensors shape and byte range disagree")
        tensors[name] = {"name": name, "dtype": dtype, "shape": shape,
                         "start": start, "end": end, "bytes": end - start}
    cursor = 0
    for entry in sorted(tensors.values(), key=lambda e: (e["start"], e["end"])):
        if entry["start"] != cursor:
            raise ValueError("Safetensors ranges overlap or contain holes")
        cursor = entry["end"]
    if 8 + length + cursor != expected_size:
        raise ValueError("Safetensors data length disagrees with source size")
    return tensors, prefix + encoded, cursor


def retain_stream(stream, output, selected_names, *, expected_sha256, expected_size,
                  optional_names=(), max_retained_bytes=MAX_RETAINED,
                  free_floor_bytes=FREE_FLOOR):
    """Consume and hash the entire object; publish only after verification.

    Optional tensors are omitted explicitly if they do not fit. Required
    tensors fail before payload writes if the retention/free-space cap fails.
    The caller supplies the immutable source size and upstream digest.
    """
    output = run_output(ROOT, output)
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected a SHA-256 hex digest")
    if expected_size < 10 or max_retained_bytes <= 0 or free_floor_bytes < 0:
        raise ValueError("Invalid corpus size or storage limits")
    selected_names = tuple(selected_names)
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("Select distinct, nonempty tensor names")
    tensors, header, data_size = _parse_header(stream, expected_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output.parent)[2] - free_floor_bytes
    budget = min(max_retained_bytes, available)
    chosen, omitted = [], []
    total = 0
    for name, optional in [(n, False) for n in selected_names] + [(n, True) for n in optional_names]:
        entry = tensors.get(name)
        if entry is None:
            if optional:
                omitted.append({"name": name, "reason": "not present in source"})
                continue
            raise ValueError(f"Required tensor missing: {name}")
        if entry["dtype"] != "BF16" or len(entry["shape"]) != 2 or min(entry["shape"]) <= 0:
            raise ValueError(f"Expected a native BF16 matrix: {name}")
        if total + entry["bytes"] > budget:
            if optional:
                omitted.append({"name": name, "reason": "retention/free-space budget"})
                continue
            raise ValueError("Corpus exceeds retention/free-space budget")
        total += entry["bytes"]
        chosen.append(entry)
    source_hash = hashlib.sha256(header)
    source_identity = expected_sha256 or hashlib.sha256(header).hexdigest()
    staging = []
    try:
        for entry in chosen:
            stem = hashlib.sha256((source_identity + entry["name"]).encode()).hexdigest()
            final = project_path(ROOT, output.parent / "tensors" / f"{stem}.u16")
            final.parent.mkdir(parents=True, exist_ok=True)
            part = final.with_suffix(".u16.part")
            handle = part.open("xb")
            staging.append({"entry": entry, "part": part, "final": final,
                            "file": handle, "hash": hashlib.sha256(), "written": 0})
        position = 0
        next_progress = 128 * 1024**2
        while position < data_size:
            chunk = stream.read(min(8 * 1024**2, data_size - position))
            if not chunk:
                raise ValueError("Truncated safetensors payload")
            source_hash.update(chunk)
            end = position + len(chunk)
            for item in staging:
                entry = item["entry"]
                lo, hi = max(position, entry["start"]), min(end, entry["end"])
                if hi > lo:
                    piece = memoryview(chunk)[lo - position:hi - position]
                    item["file"].write(piece)
                    item["hash"].update(piece)
                    item["written"] += hi - lo
            position = end
            if shutil.disk_usage(output.parent)[2] < free_floor_bytes:
                raise ValueError("Free-space floor reached during corpus stream")
            if position >= next_progress:
                print(f"corpus: verified stream {position // 1024**2} MiB", file=sys.stderr, flush=True)
                next_progress += 128 * 1024**2
        if stream.read(1):
            raise ValueError("Source has trailing bytes beyond declared size")
        actual_hash = source_hash.hexdigest()
        if expected_sha256 is not None and actual_hash != expected_sha256:
            raise ValueError("Full upstream safetensors SHA-256 mismatch")
        entries = []
        for item in staging:
            item["file"].close()
            entry = item["entry"]
            if item["written"] != entry["bytes"]:
                raise ValueError("Retained tensor is incomplete")
            item["part"].replace(item["final"])
            entries.append({"name": entry["name"], "dtype": "BF16", "shape": entry["shape"],
                            "bytes": entry["bytes"], "source_offsets": [entry["start"], entry["end"]],
                            "path": str(item["final"].relative_to(ROOT)),
                            "sha256": item["hash"].hexdigest()})
        report = {"kind": "real-weight-corpus", "status": "complete", "schema_version": 1,
                  "source_size": expected_size, "source_sha256": actual_hash,
                  "source_digest_verified": expected_sha256 is not None,
                  "digest_scope": "whole upstream object" if expected_sha256 else "local provenance only",
                  "header_sha256": hashlib.sha256(header).hexdigest(), "retained_bytes": total,
                  "max_retained_bytes": max_retained_bytes, "free_floor_bytes": free_floor_bytes,
                  "omitted_tensors": omitted, "tensors": entries}
        write_json(output, report)
        return report
    finally:
        for item in staging:
            item["file"].close()
            item["part"].unlink(missing_ok=True)


def read_tensor(manifest, name):
    import numpy as np
    if manifest.get("status") != "complete":
        raise ValueError("Corpus is not complete")
    entry = next((t for t in manifest["tensors"] if t["name"] == name), None)
    if entry is None or entry.get("dtype") != "BF16":
        raise ValueError("Requested native BF16 tensor is absent")
    path = project_path(ROOT, entry["path"])
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("Retained tensor digest/size mismatch")
    if math.prod(entry["shape"]) * 2 != entry["bytes"]:
        raise ValueError("Manifest shape and size disagree")
    return np.memmap(path, mode="r", dtype="<u2", shape=tuple(entry["shape"]))


def fetch_corpus(model, output):
    """Fetch the bounded Qwen study corpus at one immutable Hub revision."""
    if model != MODEL:
        raise ValueError(f"This preregistered corpus supports {MODEL} only")
    output = run_output(ROOT, output)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}?blobs=true", timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not return an immutable revision")
    entry = next(e for e in metadata["siblings"] if e["rfilename"] == "model.safetensors")
    digest = entry.get("lfs", {}).get("sha256")
    if not digest:
        raise ValueError("Pinned checkpoint has no published full-object SHA-256")
    if output.exists():
        existing = json.loads(output.read_text())
        if (existing.get("revision") == revision and existing.get("source_sha256") == digest
                and existing.get("source_digest_verified")):
            for tensor in existing["tensors"]:
                read_tensor(existing, tensor["name"])
            return existing
    base = f"https://huggingface.co/{model}/resolve/{revision}"
    with urllib.request.urlopen(base + "/config.json", timeout=30) as response:
        config_bytes = response.read(1024 * 1024)
    config = json.loads(config_bytes)
    expected = {"hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24,
                "num_attention_heads": 14, "num_key_value_heads": 2, "tie_word_embeddings": True}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Pinned model config differs from the preregistered model")
    names = []
    for layer in LAYERS:
        prefix = f"model.layers.{layer}."
        names.extend(prefix + "self_attn." + p + ".weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
        names.extend(prefix + "mlp." + p + ".weight" for p in ("gate_proj", "up_proj", "down_proj"))
    request = urllib.request.Request(base + "/model.safetensors", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = retain_stream(response, output, names, optional_names=("model.embed_tokens.weight",),
                               expected_sha256=digest, expected_size=entry["size"])
    result.update(model=model, revision=revision, source_url=base + "/model.safetensors",
                  config=config, config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  tuning_layers=[0, 1, 2, 3], held_out_layers=[8, 16, 23])
    write_json(output, result)
    return result
