import copy
import unittest

import numpy as np

import compressed_campaign as campaign


SPEC = {
    "tuning_seed": 901, "held_out_seeds": [919, 929, 937],
    "tuning_layers": [0, 1, 2, 3], "held_out_layers": [8, 16, 23],
    "projection_weights": ["mlp.gate_proj.weight", "mlp.down_proj.weight"],
    "large_primary_shapes_nk": [[4864, 896], [896, 4864]],
    "m_values": [1, 2, 4, 8, 16], "baseline_diagnostic_m": [64],
    "head": {"tensor": "model.embed_tokens.weight", "shape_nk": [151936, 896]},
    "minimum_rounds": 31,
}


def method(us, passed=True):
    return {"median_us": us, "samples_us": [float(us)] * 31,
            "numerics": {"passed": passed, "elements_checked": 8}}


class CampaignTests(unittest.TestCase):
    def test_seed_and_layers_are_exclusive_and_head_is_never_tuned(self):
        self.assertEqual(campaign.experiment_layers(SPEC, "tune", 901), [0, 1, 2, 3])
        self.assertEqual(campaign.experiment_layers(SPEC, "validate", 919), [8, 16, 23])
        for mode, seed in (("tune", 919), ("validate", 901), ("validate", 1000)):
            with self.assertRaises(ValueError):
                campaign.experiment_layers(SPEC, mode, seed)

    def test_vendor_winner_uses_its_own_twin_and_excludes_ineligible_routes(self):
        methods = {"slow": method(100), "slow_aa": method(1),
                   "fast": method(80), "fast_aa": method(80.4),
                   "invalid": method(10, False), "invalid_aa": method(10),
                   "coded_t128_v8": method(60)}
        result = campaign.summarize_methods(methods, ["slow", "fast", "invalid"],
                                            "coded_t128_v8")
        self.assertEqual(result["best_vendor"], "fast")
        self.assertAlmostEqual(result["time_saved_pct"], 25)
        self.assertAlmostEqual(result["aa_time_saved_pct"], -.5)
        self.assertEqual(result["ci95_pct"], [25, 25])
        self.assertTrue(result["aa_passed"])
        self.assertTrue(result["passes_10pct_gate"])
        methods["coded_t128_v8"]["numerics"]["passed"] = False
        self.assertFalse(campaign.summarize_methods(methods, ["fast"], "coded_t128_v8")["passes_10pct_gate"])

    def test_ring_shortcut_covers_every_copy_and_late_corruption_fails(self):
        reference = np.array([[1., 2., 3.]])
        outputs = np.repeat(reference[None], 12, axis=0).astype(np.float32)
        result = campaign.check_ring(reference, outputs, output_dtype="fp32")
        self.assertTrue(result["passed"])
        self.assertEqual(result["elements_checked"], 36)
        self.assertEqual(result["oracle_elements_checked"], 3)
        self.assertTrue(result["all_copies_bitwise_equal"])
        outputs[-1, 0, -1] += 1
        result = campaign.check_ring(reference, outputs, output_dtype="fp32")
        self.assertFalse(result["passed"])
        self.assertEqual(result["elements_checked"], 36)
        self.assertEqual(result["oracle_elements_checked"], 36)
        self.assertFalse(result["all_copies_bitwise_equal"])
        zeros = np.zeros((2, 1, 3), dtype=np.float32)
        zeros[1, 0, 0] = -0.
        self.assertFalse(campaign.check_ring(np.zeros((1, 3)), zeros, output_dtype="fp32")["all_copies_bitwise_equal"])

    def test_missing_eligible_vendor_is_a_recorded_negative_outcome(self):
        methods = {"vendor": method(80, False), "vendor_aa": method(80, False),
                   "coded_t128_v8": method(60)}
        result = campaign.summarize_methods(methods, ["vendor"], "coded_t128_v8")
        self.assertFalse(result["passes_10pct_gate"])
        self.assertEqual(result["comparison_status"], "no-eligible-vendor")
        self.assertIsNone(result["best_vendor"])
        self.assertIsNone(result["ci95_pct"])

    def test_freeze_aggregates_all_training_layers_and_rejects_validation(self):
        cases = []
        for layer in SPEC["tuning_layers"]:
            for suffix, shape in zip(SPEC["projection_weights"], SPEC["large_primary_shapes_nk"]):
                cases.append({"layer": layer, "shape_mnk": [1, *shape], "output_dtype": "bf16",
                              "weight_tensor": f"model.layers.{layer}.{suffix}",
                              "methods": {"coded_t32_v2": method(10 if layer == 0 else 90),
                                          "coded_t128_v8": method(60)},
                              "aa_passed": True})
        choices = campaign.freeze_choices(cases, SPEC, output_dtype="bf16")
        self.assertEqual(set(choices), {"4864,896", "896,4864"})
        self.assertEqual(set(choices.values()), {"coded_t128_v8"})
        for bad in (cases[:-1], cases + [{**cases[0], "layer": 8}],
                    cases + [{**cases[0], "shape_mnk": [2, 4864, 896]}]):
            with self.assertRaises(ValueError):
                campaign.freeze_choices(bad, SPEC, output_dtype="bf16")

    def test_completion_requires_rounds_numerics_telemetry_and_coverage(self):
        cases = [{"layer": layer, "weight_tensor": f"model.layers.{layer}.{suffix}",
                  "shape_mnk": [m, *shape], "output_dtype": "bf16", "seed": 919,
                  "methods": {"vendor": method(80), "vendor_aa": method(80),
                              "coded_t128_v8": method(60)},
                  "gpu_telemetry": [{"returncode": 0, "phase": "measurement"}], "original_bits_preserved": True,
                  "raw_template_bitwise": True, "passes_10pct_gate": True}
                 for layer in SPEC["held_out_layers"]
                 for suffix, shape in zip(SPEC["projection_weights"], SPEC["large_primary_shapes_nk"])
                 for m in SPEC["m_values"] + SPEC["baseline_diagnostic_m"]]
        corpus = {"tensors": [{"name": f"model.layers.{layer}.{suffix}", "shape": shape}
                               for layer in SPEC["held_out_layers"]
                               for suffix, shape in zip(SPEC["projection_weights"], SPEC["large_primary_shapes_nk"])]}
        report = {"kind": "compressed-projection", "mode": "validate", "seed": 919,
                  "pid": 1, "run_id": "run1", "output_dtype": "bf16", "cases": cases,
                  "head_diagnostic": {"status": "unavailable", "reason": "not retained"}}
        campaign.assert_complete(report, SPEC, corpus)
        for mutation in ("rounds", "numerics", "telemetry", "timed-telemetry", "coverage"):
            bad = copy.deepcopy(report)
            if mutation == "rounds":
                bad["cases"][0]["methods"]["vendor"]["samples_us"] = [80.] * 30
            elif mutation == "numerics":
                del bad["cases"][0]["methods"]["vendor"]["numerics"]
            elif mutation == "telemetry":
                bad["cases"][0]["gpu_telemetry"] = []
            elif mutation == "timed-telemetry":
                bad["cases"][0]["gpu_telemetry"][0]["phase"] = "capture"
            else:
                bad["cases"].pop()
            with self.assertRaises(ValueError):
                campaign.assert_complete(bad, SPEC, corpus)

    def test_aggregate_requires_three_fresh_processes(self):
        reports = [{"status": "complete", "mode": "validate", "seed": seed,
                    "pid": i + 2, "run_id": f"run{i}", "accepted_10pct": True,
                    "profile_sha256": "same", "output_dtype": "bf16",
                    "operator_context": {"profile": "performance"}}
                   for i, seed in enumerate(SPEC["held_out_seeds"])]
        self.assertTrue(campaign.validate_process_set(reports, SPEC))
        for field in ("pid", "run_id", "seed"):
            bad = copy.deepcopy(reports)
            bad[1][field] = bad[0][field]
            with self.assertRaises(ValueError):
                campaign.validate_process_set(bad, SPEC)
        reports[1]["accepted_10pct"] = False
        self.assertFalse(campaign.validate_process_set(reports, SPEC))


if __name__ == "__main__":
    unittest.main()
