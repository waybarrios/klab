"""CPU-only round-trip of every retained real tensor against independent decode."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from bf16_reference import decode_bf16
from compressed_bf16 import pack_bits, unpack_bits
from experiment_control import write_json
from real_weights import read_tensor

ROOT = Path(__file__).resolve().parents[1]
assert not torch.cuda.is_initialized(), "CPU codec audit must not initialize CUDA"
manifest = json.loads((ROOT / ".cache/real-weights/manifest.json").read_text())
screen = json.loads((ROOT / "results/bf16-storage-screen.json").read_text())
started = time.monotonic()
report = {"kind": "real-bf16-codec-verification", "status": "running",
          "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                             ("compressed_bf16.py", "bf16_reference.py", ".cache/real-weights/manifest.json")},
          "cases": []}
destination = ROOT / "results/real-bf16-codec-verification.json"
write_json(destination, report)
for tensor in manifest["tensors"]:
    bits = read_tensor(manifest, tensor["name"])
    packed = pack_bits(bits)
    independent = decode_bf16(packed.payload.tobytes(), packed.metadata, packed.n, packed.k)
    assert np.array_equal(independent, bits), tensor["name"]
    del independent
    assert np.array_equal(unpack_bits(packed), bits), tensor["name"]
    assert packed.source_sha256 == tensor["sha256"], tensor["name"]
    predicted = next(c for c in screen["cases"] if c["name"] == tensor["name"])["delta"]["storage_bytes"]
    assert packed.storage_bytes == predicted, tensor["name"]
    report["cases"].append({"name": tensor["name"], "shape_nk": tensor["shape"], "weight_sha256": tensor["sha256"],
                            "original_bytes": tensor["bytes"], "storage_bytes": packed.storage_bytes,
                            "bitwise_round_trip": True, "independent_decoder": True})
    write_json(destination, report)
    if len(report["cases"]) % 10 == 0:
        print(f"real codec: {len(report['cases'])}/50 tensors passed", flush=True)
    del packed
assert not torch.cuda.is_initialized()
report.update(status="complete", tensors_checked=len(report["cases"]), elapsed_seconds=time.monotonic() - started,
              cuda_initialized=False)
write_json(destination, report)
print(f"PASS: {len(report['cases'])} real tensors, every bit recovered, {report['elapsed_seconds']:.2f}s")
