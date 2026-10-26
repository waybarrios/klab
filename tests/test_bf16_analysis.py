import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_bf16 import analyze_bits


class StorageScreenTests(unittest.TestCase):
    def test_delta_counts_twelve_bit_payload_and_metadata(self):
        report = analyze_bits(np.full((1, 64), 0x3f80, dtype=np.uint16))
        delta = report["delta"]
        self.assertEqual(delta["payload_bytes"], 96)
        self.assertEqual(delta["metadata_bytes"], 4)
        self.assertEqual(delta["storage_bytes"], 100)
        self.assertEqual(delta["ratio"], 0.78125)

    def test_raw_fallback_and_global_table_storage_are_counted(self):
        bits = (np.arange(64, dtype=np.uint16) << 7).reshape(1, 64)
        report = analyze_bits(bits)
        self.assertEqual(report["delta"]["raw_blocks"], 1)
        self.assertEqual(report["delta"]["storage_bytes"], 132)
        self.assertEqual(report["table"]["raw_blocks"], 1)
        self.assertEqual(report["table"]["table_bytes"], 32)
        self.assertEqual(report["table"]["storage_bytes"], 164)

    def test_tail_storage_is_padded_but_original_size_is_not(self):
        report = analyze_bits(np.full((1, 65), 0x3f80, dtype=np.uint16))
        self.assertEqual(report["original_bytes"], 130)
        self.assertEqual(report["delta"]["storage_bytes"], 200)
        self.assertGreater(report["delta"]["ratio"], 1)

    def test_table_is_deterministic_on_frequency_ties(self):
        bits = (np.arange(64, dtype=np.uint16) << 7).reshape(1, 64)
        self.assertEqual(analyze_bits(bits)["table"]["exponent_table"], list(range(15)))


if __name__ == "__main__":
    unittest.main()
