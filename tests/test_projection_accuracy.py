import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projection_accuracy import check_outputs, round_bf16_bits


class ProjectionAccuracyTests(unittest.TestCase):
    def test_finite_rounding_works_with_numpy_strict_invalid_policy(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import numpy as np; np.seterr(invalid='raise'); "
                                 "from projection_accuracy import round_bf16_bits; "
                                 "assert int(round_bf16_bits(np.array([1.0]))[0]) == 0x3f80"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rounding_uses_fp64_reference_without_double_rounding(self):
        midpoint = 1.00390625
        got = round_bf16_bits(np.array([midpoint - 1e-12, midpoint, midpoint + 1e-12]))
        np.testing.assert_array_equal(got, [0x3f80, 0x3f80, 0x3f81])

    def test_zero_sign_subnormal_and_overflow_rounding(self):
        got = round_bf16_bits(np.array([0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**128]))
        np.testing.assert_array_equal(got, [0, 0x8000, 1, 0x8001, 0x7f80])

    def test_negative_underflow_keeps_zero_sign(self):
        got = round_bf16_bits(np.array([-2.0**-135, -2.0**-134, 2.0**-135]))
        np.testing.assert_array_equal(got, [0x8000, 0x8000, 0])

    def test_fp32_tolerance_rejects_large_error(self):
        self.assertTrue(check_outputs(np.array([1.0]), np.array([1.00001], np.float32), output_dtype="fp32")["passed"])
        self.assertFalse(check_outputs(np.array([1.0]), np.array([1.01], np.float32), output_dtype="fp32")["passed"])

    def test_bf16_rounded_interval_includes_both_sides_of_midpoint(self):
        report = check_outputs(np.array([1.00390625, 1.00390625]),
                               np.array([1.0, 1.0078125]), output_dtype="bf16")
        self.assertTrue(report["passed"])
        self.assertEqual(report["exact_rounding_fraction"], .5)
        self.assertEqual(report["within_one_ulp_fraction"], 1.0)
        self.assertFalse(check_outputs(np.array([1.0]), np.array([1.0078125]), output_dtype="bf16")["passed"])

    def test_near_zero_uses_absolute_floor_not_one_ulp_filter(self):
        report = check_outputs(np.array([0.0]), np.array([2.0**-20]), output_dtype="bf16")
        self.assertTrue(report["passed"])
        self.assertEqual(report["within_one_ulp_fraction"], 0.0)

    def test_nonfinite_and_unrepresentable_outputs_do_not_pass(self):
        self.assertFalse(check_outputs(np.array([1.0]), np.array([float("nan")]), output_dtype="fp32")["passed"])
        self.assertFalse(check_outputs(np.array([float("inf")]), np.array([float("inf")]), output_dtype="bf16")["passed"])
        with self.assertRaises(ValueError):
            check_outputs(np.array([1.0]), np.array([1.0001]), output_dtype="bf16")


if __name__ == "__main__":
    unittest.main()
