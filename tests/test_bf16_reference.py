import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bf16_reference import decode_bf16


class IndependentDecoderTests(unittest.TestCase):
    def test_literal_twelve_bit_pair_recovers_sign_and_exponent(self):
        # First codes are 0x080 (-1) and 0x200 (+4), base exponent 127.
        payload = bytes([0x80, 0x00, 0x20]) + bytes(93)
        result = decode_bf16(payload, np.array([127 << 23], dtype=np.uint32), 1, 2)
        np.testing.assert_array_equal(result, np.array([[0xbf80, 0x4080]], dtype=np.uint16))

    def test_raw_block_preserves_nan_payload_signed_zero_and_subnormals(self):
        original = [0x0000, 0x8000, 0x0001, 0x7fc1, 0xff80, 0xffff, 0x3f80, 0x4000] * 8
        raw = struct.pack("<64H", *original)
        payload = bytes(96) + raw
        metadata = np.array([127 << 23, 0x80000003], dtype=np.uint32)
        got = decode_bf16(payload, metadata, 1, 128)
        np.testing.assert_array_equal(got[0, :64], np.full(64, 0x3f80, dtype=np.uint16))
        np.testing.assert_array_equal(got[0, 64:], np.array(original, dtype=np.uint16))

    def test_corrupt_offsets_truncation_and_exponent_overflow_fail(self):
        for payload, metadata in ((bytes(95), [127 << 23]),
                                  (bytes(96), [(127 << 23) | 1]),
                                  (bytes([0, 1, 0]) + bytes(93), [255 << 23])):
            with self.assertRaises(ValueError):
                decode_bf16(payload, np.array(metadata, dtype=np.uint32), 1, 64)


if __name__ == "__main__":
    unittest.main()
