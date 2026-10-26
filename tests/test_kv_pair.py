"""Exact pair layout and honest comparison gates, CPU only."""
import importlib
import unittest
import numpy as np


def method(us, passed=True):
    return {"median_us": us, "samples_us": [us] * 31, "numerics": {"passed": passed}}


class KVPairTests(unittest.TestCase):
    def test_pair_concatenation_preserves_original_uint16_bits(self):
        module = importlib.import_module("kv_pair_experiment")
        k = np.array([[0, 0x8000, 0x7fc1], [1, 0x7f80, 0xff80]], dtype=np.uint16)
        v = k[::-1].copy()
        got = module.concatenate_bits(k, v)
        self.assertEqual(got.dtype, np.uint16)
        self.assertTrue(got.flags.c_contiguous)
        np.testing.assert_array_equal(got[:2], k)
        np.testing.assert_array_equal(got[2:], v)
        got[0, 0] = 17
        self.assertEqual(k[0, 0], 0)

    def test_incompatible_or_converted_weights_are_rejected(self):
        module = importlib.import_module("kv_pair_experiment")
        for left, right in ((np.ones((2, 3), dtype=np.uint16), np.ones((2, 4), dtype=np.uint16)),
                            (np.ones((2, 3)), np.ones((2, 3), dtype=np.uint16))):
            with self.assertRaises(ValueError):
                module.concatenate_bits(left, right)

    def test_vendor_fusion_gain_is_separate_from_custom_advantage(self):
        module = importlib.import_module("kv_pair_experiment")
        methods = {"separate_torch": method(20.), "separate_torch_aa": method(20.),
                   "torch_mm": method(12.), "torch_mm_aa": method(12.),
                   "coded_t64_v8": method(15.)}
        result = module.summarize_pair(methods, ["torch_mm"], output_dtype="bf16")
        self.assertAlmostEqual(result["fusion"]["time_saved_pct"], 40.)
        self.assertTrue(result["fusion"]["passes_10pct_gate"])
        self.assertFalse(result["custom"]["passes_10pct_gate"])
        methods["separate_torch_aa"] = method(21.)
        self.assertFalse(module.summarize_pair(methods, ["torch_mm"], output_dtype="bf16")["fusion"]["passes_10pct_gate"])
        methods["separate_torch_aa"] = method(20.)
        methods["torch_mm"]["numerics"]["passed"] = False
        self.assertFalse(module.summarize_pair(methods, ["torch_mm"], output_dtype="bf16")["fusion"]["passes_10pct_gate"])


if __name__ == "__main__":
    unittest.main()
