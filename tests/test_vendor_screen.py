import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench_vendor import optimistic_gain_pct


class HeadroomTests(unittest.TestCase):
    def test_screen_uses_latency_saving_and_read_bandwidth_units(self):
        # 8MB / 40GB/s = 200us; coded read160us + launch2us vs vendor210us.
        self.assertAlmostEqual(optimistic_gain_pct(8_000_000, .8, 40, 2, 210), 100 * 48 / 210)

    def test_invalid_screen_inputs_cannot_create_headroom(self):
        for args in ((0, .8, 40, 2, 210), (8_000_000, 1.1, 40, 2, 210),
                     (8_000_000, .8, 0, 2, 210), (8_000_000, .8, 40, -2, 210),
                     (8_000_000, .8, 40, 2, float("nan"))):
            with self.assertRaises(ValueError):
                optimistic_gain_pct(*args)


if __name__ == "__main__":
    unittest.main()
