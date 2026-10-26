"""CPU storage screen for two exact BF16 exponent encodings."""

import hashlib
import re
from pathlib import Path

import numpy as np

from real_weights import read_tensor


def _row_chunks(bits):
    rows = max(1, (8 * 1024**2) // (bits.shape[1] * 2))
    for start in range(0, bits.shape[0], rows):
        yield np.asarray(bits[start:start + rows])


def analyze_bits(bits):
    if bits.dtype != np.uint16 or bits.ndim != 2 or min(bits.shape) <= 0:
        raise ValueError("Expected a nonempty matrix of original BF16 uint16 bits")
    n, k = bits.shape
    histogram = np.zeros(256, dtype=np.int64)
    for chunk in _row_chunks(bits):
        exponents = ((chunk >> 7) & 255).reshape(-1)
        histogram += np.bincount(exponents, minlength=256)
    table = sorted(range(256), key=lambda exponent: (-int(histogram[exponent]), exponent))[:15]
    membership = np.zeros(256, dtype=bool)
    membership[table] = True
    blocks = n * ((k + 63) // 64)
    raw_delta = raw_table = 0
    for chunk in _row_chunks(bits):
        exponents = (chunk >> 7) & 255
        if k % 64:
            exponents = np.pad(exponents, ((0, 0), (0, -k % 64)), mode="edge")
        grouped = exponents.reshape(-1, 64)
        raw_delta += int(np.count_nonzero(grouped.max(axis=1) - grouped.min(axis=1) > 15))
        raw_table += int(np.count_nonzero(~membership[grouped].all(axis=1)))
    result = {"shape_nk": [n, k], "original_bytes": int(bits.size * 2),
              "exponent_histogram": histogram.tolist(), "block_weights": 64}
    for name, raw in (("delta", raw_delta), ("table", raw_table)):
        payload = (blocks - raw) * 96 + raw * 128
        table_bytes = 32 if name == "table" else 0
        storage = payload + blocks * 4 + table_bytes
        result[name] = {"blocks": blocks, "raw_blocks": raw, "coded_blocks": blocks - raw,
                        "payload_bytes": payload, "metadata_bytes": blocks * 4,
                        "table_bytes": table_bytes, "storage_bytes": storage,
                        "ratio": storage / result["original_bytes"], "qualifies_80pct": storage <= .8 * result["original_bytes"]}
        if name == "table":
            result[name]["exponent_table"] = table
    return result


def analyze(manifest):
    if manifest.get("status") != "complete" or not manifest.get("source_digest_verified"):
        raise ValueError("Use a complete, upstream-verified real-weight corpus")
    results, totals = [], {"delta": 0, "table": 0}
    for tensor in manifest["tensors"]:
        bits = read_tensor(manifest, tensor["name"])
        report = analyze_bits(bits)
        match = re.search(r"\.layers\.(\d+)\.", tensor["name"])
        layer = int(match.group(1)) if match else None
        tuning = layer in manifest["tuning_layers"]
        report.update(name=tensor["name"], layer=layer, tuning=tuning, source_sha256=tensor["sha256"])
        results.append(report)
        if tuning:
            for name in totals:
                totals[name] += report[name]["storage_bytes"]
    if not any(totals.values()):
        raise ValueError("Corpus has no preregistered tuning-layer tensors")
    selected = min(totals, key=lambda name: (totals[name], name != "delta"))
    return {"kind": "bf16-storage-screen", "status": "complete", "schema_version": 1,
            "model": manifest.get("model"), "revision": manifest.get("revision"),
            "source_sha256": manifest["source_sha256"], "selected_format": selected,
            "selection_rule": "Minimum total bytes on layers 0-3; ties prefer delta; held-out timings never used",
            "tuning_storage_totals": totals, "cases": results,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
