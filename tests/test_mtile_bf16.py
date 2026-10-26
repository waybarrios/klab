import unittest

from mtile_bf16 import tile_for_m, kernel_names


class MTileTests(unittest.TestCase):
    def test_deterministic_tiles_cover_registered_and_ragged_tokens(self):
        for m, expected in ((1, 1), (2, 2), (3, 4), (4, 4), (7, 8), (8, 8),
                            (15, 16), (16, 16), (17, 16), (64, 16), (65535, 16)):
            self.assertEqual(tile_for_m(m), expected)
        for bad in (0, -1, 65536, True, 1.5):
            with self.assertRaises(ValueError):
                tile_for_m(bad)

    def test_all_device_specializations_are_declared_without_extra_tuning_choices(self):
        names = kernel_names()
        self.assertEqual(len(names), 128)
        self.assertEqual(len(set(names)), 128)
        self.assertIn("mtbf16_coded_bf16_t128_v8_m16", names)
        self.assertIn("mtbf16_raw_fp32_t256_v2_m2", names)


if __name__ == "__main__":
    unittest.main()
