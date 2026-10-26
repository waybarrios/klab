import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bf16_reference import decode_bf16
from compressed_bf16 import pack_bits, unpack_bits


class ExactBF16CodecTests(unittest.TestCase):
    def check_round_trip(self, original):
        packed = pack_bits(original)
        self.assertEqual((packed.n, packed.k), original.shape)
        self.assertEqual(packed.storage_bytes, packed.payload.nbytes + packed.metadata.nbytes)
        self.assertEqual(packed.source_sha256, hashlib.sha256(original.astype("<u2").tobytes()).hexdigest())
        independent = decode_bf16(packed.payload.tobytes(), packed.metadata, packed.n, packed.k)
        np.testing.assert_array_equal(independent, original)
        np.testing.assert_array_equal(unpack_bits(packed), original)
        return packed

    def test_every_bf16_bit_pattern_round_trips_independently(self):
        self.check_round_trip(np.arange(65536, dtype=np.uint16).reshape(1024, 64))

    def test_exact_storage_and_raw_fallback(self):
        coded = self.check_round_trip(np.full((1, 64), 0x3f80, dtype=np.uint16))
        self.assertEqual(coded.storage_bytes, 100)
        raw = self.check_round_trip((np.arange(64, dtype=np.uint16) << 7).reshape(1, 64))
        self.assertEqual(raw.storage_bytes, 132)
        self.assertEqual(int(raw.metadata[0]), 0x80000000)

    def test_mixed_blocks_and_tail_padding(self):
        bits = np.full((3, 129), 0xbf80, dtype=np.uint16)
        bits[0, 64:128] = np.arange(64, dtype=np.uint16) << 7
        bits[1, 0] = 0x7fc1
        bits[2, 128] = 0x8000
        self.check_round_trip(bits)

    def test_corrupted_payload_metadata_and_wrong_input_fail(self):
        packed = pack_bits(np.full((1, 64), 0x3f80, dtype=np.uint16))
        packed.metadata[0] |= np.uint32(1)
        with self.assertRaises(ValueError):
            unpack_bits(packed)
        for value in (np.ones((1, 64), dtype=np.float32), np.empty((0, 64), dtype=np.uint16),
                      np.ones(64, dtype=np.uint16)):
            with self.assertRaises(ValueError):
                pack_bits(value)


if __name__ == "__main__":
    unittest.main()
