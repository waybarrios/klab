import unittest

from energy_metrics import batch_metrics, summarize_energy


class EnergyMetricTests(unittest.TestCase):
    def test_energy_units_are_per_projection_and_not_per_flop(self):
        result = batch_metrics(60000, 110000, 10000, wall_seconds=2, gpu_ms=1800)
        self.assertAlmostEqual(result["gpu_energy_j"], 50)
        self.assertAlmostEqual(result["j_per_projection"], .005)
        self.assertAlmostEqual(result["kwh_per_projection"], .005 / 3600000)
        self.assertAlmostEqual(result["kwh_per_million_projections"], .005 / 3.6)
        self.assertAlmostEqual(result["projections_per_j"], 200)
        self.assertAlmostEqual(result["average_board_w"], 25)
        self.assertAlmostEqual(result["gpu_us_per_projection"], 180)

    def test_bad_counters_counts_and_durations_are_rejected(self):
        for start, stop, count, wall, gpu in ((100, 99, 1, 1, 1), (100, 100, 1, 1, 1),
            (1, 1000, 0, 1, 1), (1, 1000, True, 1, 1), (1, 1000, 1.5, 1, 1),
            (1, 1000, 1, 0, 1), (1, 1000, 1, 1, float("nan")), (1, 1000, 1, 1, 5000)):
            with self.assertRaises(ValueError):
                batch_metrics(start, stop, count, wall_seconds=wall, gpu_ms=gpu)

    def test_paired_energy_claim_requires_aa_and_positive_interval(self):
        rounds = [{"vendor": {"j_per_projection": .01, "projection_count": 1000},
                   "vendor_aa": {"j_per_projection": .0101, "projection_count": 1000},
                   "coded": {"j_per_projection": .008, "projection_count": 1000}} for _ in range(3)]
        result = summarize_energy(rounds)
        self.assertAlmostEqual(result["energy_saved_pct"], 20)
        self.assertAlmostEqual(result["aa_energy_saved_pct"], -1)
        self.assertEqual(result["bootstrap_interval_pct"], [20, 20])
        self.assertTrue(result["preliminary_reduction_supported"])
        for row in rounds:
            row["vendor_aa"]["j_per_projection"] = .012
        self.assertFalse(summarize_energy(rounds)["preliminary_reduction_supported"])
        rounds[0]["coded"]["projection_count"] = 999
        with self.assertRaises(ValueError):
            summarize_energy(rounds)

    def test_round_ratios_do_not_mix_different_thermal_rounds(self):
        rounds = [{"vendor": {"j_per_projection": b, "projection_count": 1000},
                   "vendor_aa": {"j_per_projection": b * 1.001, "projection_count": 1000},
                   "coded": {"j_per_projection": c, "projection_count": 1000}}
                  for b, c in ((1, .1), (2, 1.9), (100, 50))]
        result = summarize_energy(rounds)
        self.assertAlmostEqual(result["energy_saved_pct"], 50)
        rounds[0]["vendor_aa"]["j_per_projection"] = 1.2
        self.assertFalse(summarize_energy(rounds)["aa_passed"])


if __name__ == "__main__":
    unittest.main()
