import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from real_weights import read_tensor, retain_stream


def fixture(change=None):
    header = {"a.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
              "skip.weight": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
              "b.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [16, 20]}}
    if change:
        change(header)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<4H", 0, 0x8000, 0x3f80, 0x7fc1) + b"12345678" + struct.pack("<2H", 1, 0xffff)
    return struct.pack("<Q", len(encoded)) + encoded + payload


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".scratch")
        self.output = Path(self.temp.name) / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def retain(self, data, *, stream=None, digest=None, selected=("a.weight",), **kwargs):
        return retain_stream(stream or io.BytesIO(data), self.output, selected,
                             expected_sha256=digest or hashlib.sha256(data).hexdigest(),
                             expected_size=len(data), free_floor_bytes=0, **kwargs)

    def test_retains_selected_original_bits_and_verifies_whole_object(self):
        data = fixture()
        class ShortReads(io.BytesIO):
            def read(self, n=-1):
                return super().read(min(n, 3) if n >= 0 else 3)
        result = self.retain(data, stream=ShortReads(data), selected=("a.weight", "b.weight"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["source_digest_verified"])
        self.assertEqual(result["retained_bytes"], 12)
        self.assertEqual({t["name"] for t in result["tensors"]}, {"a.weight", "b.weight"})
        np.testing.assert_array_equal(read_tensor(result, "a.weight"),
                                      np.array([[0, 0x8000], [0x3f80, 0x7fc1]], dtype=np.uint16))
        np.testing.assert_array_equal(read_tensor(result, "b.weight"),
                                      np.array([[1, 0xffff]], dtype=np.uint16))

    def test_truncation_and_wrong_full_digest_never_complete(self):
        data = fixture()
        for stream, digest in ((io.BytesIO(data[:-1]), None), (io.BytesIO(data), "0" * 64)):
            with self.assertRaises(ValueError):
                self.retain(data, stream=stream, digest=digest)
            self.assertFalse(self.output.exists())
            self.assertFalse(list(self.output.parent.rglob("*.part")))

    def test_malformed_offsets_dtype_and_names_are_rejected(self):
        changes = (
            lambda h: h["a.weight"].update(data_offsets=[0, 10]),
            lambda h: h["a.weight"].update(dtype="F16"),
            lambda h: h["b.weight"].update(data_offsets=[4, 8]),
            lambda h: h.update({"../escape": h.pop("a.weight")}),
        )
        for change in changes:
            with self.assertRaises(ValueError):
                self.retain(fixture(change))
            self.assertFalse(self.output.exists())

    def test_storage_cap_and_free_space_are_enforced_before_retaining(self):
        data = fixture()
        with self.assertRaises(ValueError):
            self.retain(data, max_retained_bytes=7)
        with patch("real_weights.shutil.disk_usage", return_value=(100, 99, 1)):
            with self.assertRaises(ValueError):
                self.retain(data)
        self.assertFalse(self.output.exists())

    def test_optional_head_is_omitted_explicitly_when_it_exceeds_cap(self):
        result = self.retain(fixture(), max_retained_bytes=8, optional_names=("b.weight",))
        self.assertEqual(result["retained_bytes"], 8)
        self.assertEqual(result["omitted_tensors"][0]["name"], "b.weight")
        self.assertIn("budget", result["omitted_tensors"][0]["reason"])

    def test_modified_retained_tensor_cannot_be_read_as_verified(self):
        result = self.retain(fixture())
        path = ROOT / result["tensors"][0]["path"]
        path.write_bytes(b"x" * 8)
        with self.assertRaises(ValueError):
            read_tensor(result, "a.weight")


if __name__ == "__main__":
    unittest.main()
