import unittest

from power_context import execution_context


class PowerContextTests(unittest.TestCase):
    def test_display_brightness_is_recorded_but_not_a_compute_profile(self):
        bright = {"system76_power_profile": "Power Profile: Performance\nCPU: 10% - 100%, Turbo\nBacklight: 255/255", "ac_online": True}
        dim = {**bright, "system76_power_profile": bright["system76_power_profile"].replace("255/255", "64/255")}
        self.assertEqual(execution_context(bright), execution_context(dim))
        for change in ({**bright, "ac_online": False},
                       {**bright, "system76_power_profile": "Power Profile: Battery\nCPU: 10% - 50%, No Turbo\nBacklight: 64/255"}):
            self.assertNotEqual(execution_context(bright), execution_context(change))

    def test_missing_power_query_is_an_error_not_an_unknown_eligible_regime(self):
        with self.assertRaises(ValueError):
            execution_context({"system76_power_profile": "unavailable", "ac_online": True})

    def test_cpu_limit_variation_is_a_separate_eager_contract(self):
        full = {"system76_power_profile": "Power Profile: Performance\nCPU: 10% - 100%, Turbo", "ac_online": True}
        limited = {**full, "system76_power_profile": full["system76_power_profile"].replace("100%", "90%")}
        self.assertEqual(execution_context(full), execution_context(limited))
        self.assertNotEqual(execution_context(full, scope="eager"), execution_context(limited, scope="eager"))


if __name__ == "__main__":
    unittest.main()
