"""CPU protocol checks, not hardware performance evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativePrecisionTests(unittest.TestCase):
    def fixture(self, directory):
        source = {"cases": [{"shape_mnk": [4, 256, 256], "selected": "t256_v8"}]}
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(source))
        protocol = {"configuration_search_allowed": False, "code_changes_allowed": False,
                    "source_profiles": {"fp16": {"path": "profile.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
                    "cases_by_dtype": {"fp16": [{"shape_mnk": [4, 256, 256], "config": "half_t256_v8", "origin": "fp16"}]}}
        return protocol, path

    def test_configuration_must_match_exact_historical_source(self):
        module = importlib.import_module("native_precision")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            protocol, path = self.fixture(directory)
            cases = module.frozen_cases(protocol, "fp16", root=Path(directory))
            self.assertEqual(cases[0]["config"], "half_t256_v8")
            path.write_text("changed")
            with self.assertRaises(ValueError):
                module.frozen_cases(protocol, "fp16", root=Path(directory))

    def test_configuration_cannot_be_reselected_or_cross_precision(self):
        module = importlib.import_module("native_precision")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            protocol, _ = self.fixture(directory)
            protocol["cases_by_dtype"]["fp16"][0]["config"] = "half_t32_v2"
            with self.assertRaises(ValueError):
                module.frozen_cases(protocol, "fp16", root=Path(directory))
            protocol["cases_by_dtype"]["fp16"][0]["config"] = "lossless_t256"
            with self.assertRaises(ValueError):
                module.frozen_cases(protocol, "fp16", root=Path(directory))

    def test_unsupported_formats_and_new_search_reject_before_gpu_work(self):
        module = importlib.import_module("native_precision")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            protocol, _ = self.fixture(directory)
            with self.assertRaises(ValueError):
                module.frozen_cases(protocol, "fp8", root=Path(directory))
            protocol["configuration_search_allowed"] = True
            with self.assertRaises(ValueError):
                module.frozen_cases(protocol, "fp16", root=Path(directory))


if __name__ == "__main__":
    unittest.main()
