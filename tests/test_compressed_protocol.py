import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench_compressed import paired_gain_interval, validate_probe_inputs


class CompressedProtocolTests(unittest.TestCase):
    def test_latency_interval_uses_best_vendor_and_paired_rounds(self):
        low, high = paired_gain_interval({"a": [100.] * 31, "b": [80.] * 31}, [60.] * 31)
        self.assertAlmostEqual(low, 25.)
        self.assertAlmostEqual(high, 25.)
        with self.assertRaises(ValueError):
            paired_gain_interval({"a": [100.] * 31}, [60.] * 30)

    def test_probe_cannot_consume_held_out_seed_or_stale_vendor(self):
        spec = {"revision": "rev", "safetensors_sha256": "weights", "tuning_seed": 901,
                "numerics": {"fp32_output_atol": 1e-5, "fp32_output_rtol": 1e-4}}
        corpus = {"revision": "rev", "source_sha256": "weights", "status": "complete"}
        vendor = {"kind": "vendor-tuning", "status": "complete", "screen_only": False,
                  "codec_gpu_gate_passed": True}
        validate_probe_inputs(spec, corpus, vendor, seed=901)
        for changed in ({**vendor, "status": "running"}, {**vendor, "codec_gpu_gate_passed": False},
                        {**vendor, "screen_only": True}):
            with self.assertRaises(ValueError):
                validate_probe_inputs(spec, corpus, changed, seed=901)
        with self.assertRaises(ValueError):
            validate_probe_inputs(spec, corpus, vendor, seed=919)
        with self.assertRaises(ValueError):
            validate_probe_inputs({**spec, "numerics": {"fp32_output_atol": 1e-3, "fp32_output_rtol": 1e-2}},
                                  corpus, vendor, seed=901)


if __name__ == "__main__":
    unittest.main()
