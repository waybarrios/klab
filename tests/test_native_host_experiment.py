"""CPU checks for the preregistered native-host study and its honest verdicts."""
import importlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeHostExperimentTests(unittest.TestCase):
    def test_registered_case_grid_keeps_projection_identity_and_outputs(self):
        module = importlib.import_module("native_host_experiment")
        protocol = json.loads((ROOT / "experiments/native_prepared_scope.json").read_text())
        corpus = {"tensors": [{"name": f"model.layers.8.{c['projection']}", "shape": [128, 896]}
                              for c in protocol["cases"] if "projection" in c]}
        cases = module.case_specs(protocol, corpus)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({(c["weight_tensor"], tuple(c["shape_mnk"]), c["output_dtype"]) for c in cases}), 12)
        self.assertEqual(sum(c["control"] for c in cases), 2)

    def test_gain_uses_raw_rounds_and_noise_gate(self):
        module = importlib.import_module("native_host_experiment")
        methods = {key: {"samples_us": [value] * 31, "median_us": 999}
                   for key, value in (("vendor", 100), ("vendor_aa", 100),
                                      ("native", 80), ("python_registered", 120), ("selected", 80))}
        summary = module.summarize_scope(methods, selected_custom=True, numerics_passed=True, context_stable=True)
        self.assertAlmostEqual(summary["selected"]["time_saved_pct"], 20)
        self.assertTrue(summary["selected"]["case_gate_passed"])
        self.assertFalse(summary["custom_win_validated"])
        methods["vendor_aa"]["samples_us"] = [103] * 31
        summary = module.summarize_scope(methods, selected_custom=True, numerics_passed=True, context_stable=True)
        self.assertFalse(summary["selected"]["case_gate_passed"])

    def test_fallback_is_not_a_custom_win_and_requires_all_gates(self):
        module = importlib.import_module("native_host_experiment")
        methods = {key: {"samples_us": [100] * 31}
                   for key in ("vendor", "vendor_aa", "native", "python_registered", "selected")}
        summary = module.summarize_scope(methods, selected_custom=False, numerics_passed=True, context_stable=True)
        self.assertTrue(summary["selected"]["case_gate_passed"])
        self.assertFalse(summary["selected"]["custom_gain_claim"])
        self.assertFalse(module.summarize_scope(methods, selected_custom=False, numerics_passed=False,
                                              context_stable=True)["selected"]["case_gate_passed"])
        self.assertFalse(module.summarize_scope(methods, selected_custom=False, numerics_passed=True,
                                              context_stable=False)["selected"]["case_gate_passed"])


if __name__ == "__main__":
    unittest.main()
