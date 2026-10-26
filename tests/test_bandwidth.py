import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench_bandwidth import summarize_probe


class BandwidthAccountingTests(unittest.TestCase):
    def test_copy_counts_reads_and_writes_in_gigabytes_per_second(self):
        result = summarize_probe("copy", 1024, [2.0, 3.0, 1.0])
        self.assertEqual(result["traffic_bytes"], 2048)
        self.assertEqual(result["median_us"], 2.0)
        self.assertAlmostEqual(result["gb_per_s"], 1.024)
        self.assertEqual(result["samples_us"], [2.0, 3.0, 1.0])

    def test_read_reports_input_rate_and_sink_separately(self):
        result = summarize_probe("read", 1024, [2.0], sink_bytes=16)
        self.assertEqual(result["traffic_bytes"], 1040)
        self.assertAlmostEqual(result["gb_per_s"], 0.52)
        self.assertAlmostEqual(result["input_gb_per_s"], 0.512)
        self.assertEqual(result["sink_bytes"], 16)

    def test_launch_is_not_reported_as_memory_bandwidth(self):
        result = summarize_probe("launch", 0, [1.0])
        self.assertIsNone(result["gb_per_s"])

    def test_invalid_or_empty_timings_cannot_form_a_result(self):
        for operation, byte_count, samples in (("read", 1024, []), ("read", 1024, [0.0]),
                                             ("read", -1, [1.0]), ("unknown", 1024, [1.0]),
                                             ("copy", 1024, [float("nan")])):
            with self.assertRaises(ValueError):
                summarize_probe(operation, byte_count, samples)


if __name__ == "__main__":
    unittest.main()
