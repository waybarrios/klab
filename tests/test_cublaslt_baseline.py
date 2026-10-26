import base64
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cublaslt_baseline import select_heuristics, validate_frozen


class FrozenVendorTests(unittest.TestCase):
    def record(self):
        return {"library_version": 120804, "device_uuid": "GPU-test", "input_dtype": "bf16",
                "output_dtype": "fp32", "shape_mnk": [1, 4864, 896], "layout": "original",
                "compute_type": 68, "reduction_scheme": 2, "workspace_bytes": 4096,
                "minimum_alignment": [16, 16, 16, 16],
                "algorithm_b64": base64.b64encode(bytes(64)).decode()}

    def context(self):
        return {k: v for k, v in self.record().items()
                if k in ("library_version", "device_uuid", "input_dtype", "output_dtype", "shape_mnk", "layout")}

    def test_frozen_algorithms_bind_runtime_device_and_operation(self):
        record = self.record()
        validate_frozen(record, context=self.context())
        for field, changed in (("library_version", 120805), ("device_uuid", "GPU-other"),
                               ("output_dtype", "bf16"), ("input_dtype", "fp16"),
                               ("shape_mnk", [4, 4864, 896]), ("layout", "transposed")):
            bad = {**record, field: changed}
            with self.assertRaises(ValueError):
                validate_frozen(bad, context=self.context())

    def test_output_type_split_k_and_excess_workspace_cannot_qualify(self):
        for field, changed in (("compute_type", 77), ("reduction_scheme", 1),
                               ("reduction_scheme", 4), ("workspace_bytes", 67108865),
                               ("algorithm_b64", base64.b64encode(bytes(63)).decode())):
            with self.assertRaises(ValueError):
                validate_frozen({**self.record(), field: changed}, context=self.context())

    def test_heuristic_pool_is_bounded_and_exclusions_are_not_winners(self):
        records = []
        for i in range(12):
            r = self.record()
            r["algorithm_b64"] = base64.b64encode(bytes([i]) * 64).decode()
            records.append(r)
        records[0]["reduction_scheme"] = 4
        records[1]["workspace_bytes"] = 67108865
        before = copy.deepcopy(records)
        eligible = select_heuristics(records, max_candidates=8, workspace_limit=67108864)
        self.assertEqual(len(eligible), 8)
        self.assertEqual(eligible[0]["algorithm_b64"], records[2]["algorithm_b64"])
        self.assertTrue(all(r["reduction_scheme"] in (0, 2) and r["workspace_bytes"] <= 67108864 for r in eligible))
        self.assertEqual(records, before)
        self.assertEqual(select_heuristics([], max_candidates=8, workspace_limit=67108864), [])


if __name__ == "__main__":
    unittest.main()
