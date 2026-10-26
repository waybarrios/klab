import unittest
from unittest.mock import patch


class DispatchMetricsTests(unittest.TestCase):
    def methods(self,prepared=100.5,aa=100.):
        return {name:{"samples_us":[value]*31} for name,value in
                (("vendor",100.),("vendor_aa",aa),("prepared",prepared))}

    def test_half_percent_dispatch_cost_passes_one_percent_gate(self):
        from bench_dispatch import summarize_dispatch
        result=summarize_dispatch(self.methods())
        self.assertAlmostEqual(result["time_saved_pct"],-.5)
        self.assertTrue(result["overhead_gate_passed"])
        self.assertFalse(result["custom_win_validated"])

    def test_more_than_one_percent_cost_is_not_hidden(self):
        from bench_dispatch import summarize_dispatch
        result=summarize_dispatch(self.methods(prepared=101.1))
        self.assertFalse(result["overhead_gate_passed"])
        self.assertLess(result["ci95_pct"][0],-1)

    def test_failed_aa_prevents_overhead_promotion(self):
        from bench_dispatch import summarize_dispatch
        result=summarize_dispatch(self.methods(aa=102.))
        self.assertFalse(result["overhead_gate_passed"])

    def test_missing_or_misaligned_raw_rounds_fail(self):
        from bench_dispatch import summarize_dispatch
        methods=self.methods()
        methods["vendor_aa"]["samples_us"].pop()
        with self.assertRaises(ValueError):
            summarize_dispatch(methods)

    def test_power_snapshot_is_passed_to_the_real_context_normalizer(self):
        from bench_dispatch import current_context
        snapshot={"system76_power_profile":"Power Profile: Performance\nCPU: 10% - 100%",
                  "ac_online":True}
        with patch("bench_compressed._power_context",return_value=snapshot):
            graph=current_context(scope="graph")
            eager=current_context(scope="eager")
        self.assertEqual(graph,{"system76_power_profile":"Power Profile: Performance",
                                "ac_online":True,"scope":"graph"})
        self.assertIn("CPU: 10% - 100%",eager["system76_power_profile"])
        with patch("bench_compressed._power_context",return_value={}):
            with self.assertRaises(ValueError):
                current_context(scope="graph")
