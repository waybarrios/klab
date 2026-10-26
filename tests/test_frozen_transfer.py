"""CPU checks for fixed-choice transfer; fixtures are not GPU measurements."""
import copy
import importlib
import unittest
from unittest import mock


class FrozenTransferTests(unittest.TestCase):
    def fixture(self):
        spec = {"tuning_layers": [0, 1], "projection_weights": ["mlp.gate_proj.weight", "mlp.down_proj.weight"],
                "large_primary_shapes_nk": [[128, 64], [64, 128]]}
        corpus = {"tensors": [{"name": f"model.layers.{layer}.{suffix}", "shape": shape}
                  for layer in (0, 1, 8) for suffix, shape in zip(spec["projection_weights"], ([128, 64], [64, 128]))]}
        source = {"kind": "compressed-projection-tuning", "status": "frozen", "bindings": {"output_dtype": "bf16"},
                  "choices": {"128,64": "coded_t128_v8", "64,128": "coded_t256_v8"}}
        protocol = {"source_profiles": [{"sha256": "a" * 64, "output_dtype": "bf16", "choices": source["choices"]}]}
        return spec, corpus, source, protocol

    def test_choices_are_exactly_inherited_and_independent_of_timing(self):
        transfer = importlib.import_module("frozen_transfer")
        _, _, source, protocol = self.fixture()
        choices = transfer.fixed_choices(protocol, source, source_sha256="a" * 64, output_dtype="bf16")
        self.assertEqual(choices, source["choices"])
        choices["128,64"] = "coded_t32_v2"
        self.assertEqual(source["choices"]["128,64"], "coded_t128_v8")

    def test_wrong_parent_bytes_output_or_choices_cannot_calibrate(self):
        transfer = importlib.import_module("frozen_transfer")
        _, _, source, protocol = self.fixture()
        with self.assertRaises(ValueError):
            transfer.fixed_choices(protocol, source, source_sha256="b" * 64, output_dtype="bf16")
        with self.assertRaises(ValueError):
            transfer.fixed_choices(protocol, source, source_sha256="a" * 64, output_dtype="fp32")
        changed = copy.deepcopy(source)
        changed["choices"]["128,64"] = "coded_t32_v2"
        with self.assertRaises(ValueError):
            transfer.fixed_choices(protocol, changed, source_sha256="a" * 64, output_dtype="bf16")

    def test_calibration_uses_only_training_layers_and_m1(self):
        transfer = importlib.import_module("frozen_transfer")
        spec, corpus, _, _ = self.fixture()
        cases = transfer.calibration_grid(spec, corpus)
        self.assertEqual(len(cases), 4)
        self.assertEqual({case["layer"] for case in cases}, {0, 1})
        self.assertTrue(all(case["shape_mnk"][0] == 1 for case in cases))
        incomplete = copy.deepcopy(corpus)
        incomplete["tensors"].pop(0)
        with self.assertRaises(ValueError):
            transfer.calibration_grid(spec, incomplete)

    def test_profile_restores_after_exception(self):
        transfer = importlib.import_module("frozen_transfer")
        initial = {"profile": "performance", "brightness": 255}
        state = dict(initial)
        def select(profile):
            state.update(profile=profile, brightness=25 if profile == "battery" else 140)
        def brightness(value):
            state["brightness"] = value
        with mock.patch.object(transfer, "read_settings", side_effect=lambda: dict(state)), \
             mock.patch.object(transfer, "select_profile", side_effect=select) as selected, \
             mock.patch.object(transfer, "set_brightness", side_effect=brightness):
            with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                with transfer.temporary_profile("battery"):
                    self.assertEqual(state["profile"], "battery")
                    self.assertEqual(state["brightness"], 255)
                    raise RuntimeError("measurement failed")
            self.assertEqual(state, initial)
            self.assertEqual([c.args[0] for c in selected.call_args_list], ["battery", "performance"])

    def test_profile_restores_even_if_selection_changed_state_then_raised(self):
        transfer = importlib.import_module("frozen_transfer")
        state = {"profile": "performance", "brightness": 255}
        def select(profile):
            state["profile"] = profile
            if profile == "battery":
                raise RuntimeError("profile command failed")
        with mock.patch.object(transfer, "read_settings", side_effect=lambda: dict(state)), \
             mock.patch.object(transfer, "select_profile", side_effect=select), \
             mock.patch.object(transfer, "set_brightness", side_effect=lambda value: state.update(brightness=value)):
            with self.assertRaisesRegex(RuntimeError, "profile command failed"):
                with transfer.temporary_profile("battery"):
                    self.fail("Measurement cannot start after failed selection")
        self.assertEqual(state, {"profile": "performance", "brightness": 255})

    def test_restoration_is_recorded_before_pending_signals_are_unblocked(self):
        transfer = importlib.import_module("frozen_transfer")
        initial = {"profile": "performance", "brightness": 255}
        state = {}
        def mask(how, _values):
            if how == transfer.signal.SIG_SETMASK:
                self.assertTrue(state.get("restored"), "Signal delivery must see verified restoration")
            return set()
        with mock.patch.object(transfer, "read_settings", return_value=dict(initial)), \
             mock.patch.object(transfer.signal, "pthread_sigmask", side_effect=mask):
            transfer._restore(initial, state)

    def test_changed_nvrtc_bytes_prevent_successful_source_check(self):
        transfer = importlib.import_module("frozen_transfer")
        identity = {"libraries_sha256": {}, "module": {"compiler": "NVRTC 12.8:sha256=" + "a" * 64}}
        with mock.patch("cuda_module._nvrtc_identity", return_value=("NVRTC 12.8:sha256=" + "b" * 64, "unused")):
            with self.assertRaisesRegex(RuntimeError, "NVRTC"):
                transfer.check_sources(identity, {})


if __name__ == "__main__":
    unittest.main()
