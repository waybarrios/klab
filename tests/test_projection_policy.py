import copy
import unittest

from projection_policy import choose


class ProjectionPolicyTests(unittest.TestCase):
    def fixture(self):
        case={"shape_mnk":[1,64,64],"seed":1,"selected":"tiny_t128_v1","bitwise_round_trip":True,
              "accepted_custom_win":True,"aa_time_saved_pct":0.1,"candidate_gain_ci95_pct":[14,16],
              "methods":{"torch_mm":{"median_us":6},"cublas_mm":{"median_us":5.9}}}
        second=copy.deepcopy(case)
        second["seed"]=2
        return {"kind":"streaming-projection","status":"complete","cases":[case,second]}

    def test_tiny_shape_keeps_plain_fp32_kernel(self):
        plan=choose(self.fixture(),(1,64,64))
        self.assertEqual(plan["config"],"tiny_t128_v1")
        self.assertTrue(plan["accepted_10pct_target"])

    def test_one_failed_seed_prevents_custom_promotion(self):
        profile=self.fixture()
        profile["cases"][1]["aa_time_saved_pct"]=1.4
        plan=choose(profile,(1,64,64))
        self.assertEqual(plan["backend"],"vendor")
        self.assertEqual(plan["config"],"cublas_mm")
        self.assertFalse(plan["custom_win_validated"])

    def test_unknown_shape_is_vendor_parity_not_a_custom_win(self):
        plan=choose(self.fixture(),(1,8,64))
        self.assertEqual(plan["backend"],"vendor")
        self.assertFalse(plan["accepted_10pct_target"])

    def test_half_contract_and_in_sample_rejection(self):
        profile=self.fixture()
        profile.update(kind="half-projection",mode="validate",dtype="fp16")
        for case in profile["cases"]:
            case.update(selected="t256_v8",bitwise_layout_check=True,numerics_passed=True,noise_gated_win=True,
                        methods={"cublas_compute32":{"median_us":5}})
        plan=choose(profile,(1,64,64))
        self.assertEqual(plan["config"],"half_t256_v8")
        self.assertTrue(plan["accepted_10pct_target"])
        profile["mode"]="tune"
        with self.assertRaises(ValueError): choose(profile,(1,64,64))

    def test_compressed_small_regression_uses_vendor_without_packing(self):
        profile={"kind":"compressed-projection","status":"complete","mode":"tune",
                 "output_dtype":"bf16","cases":[{"shape_mnk":[1,128,896],
                 "time_saved_pct":-63.966,"selected":"coded_t64_v8"}]}
        plan=choose(profile,(1,128,896),output_dtype="bf16")
        self.assertEqual(plan["backend"],"vendor")
        self.assertEqual(plan["config"],"torch_mm")
        self.assertFalse(plan["requires_packing"])
        self.assertFalse(plan["custom_win_validated"])
        self.assertFalse(plan["accepted_10pct_target"])

    def test_compressed_training_flags_cannot_promote_custom(self):
        profile={"kind":"compressed-projection-tuning","status":"complete",
                 "output_dtype":"bf16","accepted_10pct":True,"cases":[]}
        plan=choose(profile,(1,4864,896),output_dtype="bf16")
        self.assertEqual(plan["backend"],"vendor")
        self.assertFalse(plan["requires_packing"])
        self.assertFalse(plan["custom_win_validated"])

    def test_bf16_training_does_not_match_fp32_output(self):
        profile={"kind":"compressed-projection-tuning","status":"complete",
                 "output_dtype":"bf16","cases":[]}
        plan=choose(profile,(1,128,896),output_dtype="fp32")
        self.assertEqual(plan["config"],"torch_mm_out_fp32")
        self.assertFalse(plan["custom_win_validated"])
        self.assertIn("output",plan["reason"].lower())

    def test_choice_table_output_contract_is_nested_in_bindings(self):
        profile={"kind":"compressed-projection-tuning","status":"complete",
                 "bindings":{"output_dtype":"bf16"},"choices":{"128,896":"coded_t64_v8"}}
        plan=choose(profile,(1,128,896),output_dtype="fp32")
        self.assertIn("output",plan["reason"].lower())
        self.assertEqual(plan["config"],"torch_mm_out_fp32")
