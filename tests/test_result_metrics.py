import math
import unittest

from result_metrics import case_metrics, percentage_gains


class ResultMetricsTests(unittest.TestCase):
    def test_latency_and_throughput_percentages_have_different_denominators(self):
        gains = percentage_gains(100, 80)
        self.assertAlmostEqual(gains["latency_reduction_pct"], 20)
        self.assertAlmostEqual(gains["throughput_gain_pct"], 25)
        regression = percentage_gains(100, 125)
        self.assertAlmostEqual(regression["latency_reduction_pct"], -25)
        self.assertAlmostEqual(regression["throughput_gain_pct"], -20)
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                percentage_gains(100, value)

    def test_each_cache_condition_uses_its_own_fastest_fp32_baseline(self):
        def timing(warm, cold):
            return {"warm": {"median_us": warm}, "cold": {"median_us": cold}}
        case = {"methods": {"torch_mm": timing(100, 200), "cublas_mv": timing(120, 160),
                            "lossless_fp32": timing(80, 120), "fp4_diagnostic": timing(1, 1)}}
        self.assertEqual(case_metrics(case, "warm")["baseline_method"], "torch_mm")
        self.assertEqual(case_metrics(case, "cold")["baseline_method"], "cublas_mv")
        self.assertAlmostEqual(case_metrics(case, "cold")["latency_reduction_pct"], 25)


if __name__ == "__main__":
    unittest.main()
