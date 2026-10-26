"""Synthetic protocol fixtures, never hardware performance evidence."""
import copy
import math
import unittest


PROFILE_SHA="f"*64


def fixture():
    spec={"tuning_seed":901,"held_out_seeds":[919,929,937],
          "tuning_layers":[0,1,2,3],"held_out_layers":[8,16,23],
          "projection_weights":["mlp.gate_proj.weight","self_attn.k_proj.weight"],
          "large_primary_shapes_nk":[[64,64]],"m_values":[1],"baseline_diagnostic_m":[],
          "head":{"tensor":"model.embed_tokens.weight"},"minimum_rounds":31,
          "storage_limits":{"promotion_max_ratio":.8},
          "measurement":{"minimum_uncompressed_ring_bytes":80<<20,"minimum_ring_l2_multiple":20},
          "acceptance":{"maximum_absolute_aa_pct":1,
                        "latency_reduction_ci95_lower_bound_strictly_above_pct":10}}
    context={"system76_power_profile":"Power Profile: Performance","ac_online":True,"scope":"graph"}
    bindings={"spec_sha256":"1"*64,"corpus_sha256":"2"*64,"vendor_profile_sha256":"3"*64,
              "context_protocol_sha256":"4"*64,"output_dtype":"bf16","revision":2,
              "operator_context":context,"identity":{"implementation":{"fingerprint_sha256":"5"*64},
                  "allow_tf32":False,"reduced_precision_reduction":False}}
    profile={"kind":"compressed-projection-tuning","status":"complete","seed":901,
             "pid":50,"run_id":"training","bindings":bindings,
             "choices":{"64,64":"coded_t128_v8","128,64":"coded_t64_v8"},
             "minimum_12pct_tuning_screen":True,"observed_memory_clocks":["810 MHz"]}
    corpus={"tensors":[]}
    for layer in spec["held_out_layers"]:
        for suffix,n in (("mlp.gate_proj.weight",64),("self_attn.k_proj.weight",128)):
            corpus["tensors"].append({"name":f"model.layers.{layer}.{suffix}","shape":[n,64],
                                      "sha256":("a" if n==64 else "b")*64})
    reports=[]
    for seed in spec["held_out_seeds"]:
        report={"kind":"compressed-projection","status":"complete","mode":"validate","seed":seed,
                "pid":1000+seed,"run_id":f"process-{seed}","profile_sha256":PROFILE_SHA,
                **copy.deepcopy(bindings),"cases":[],"accepted_10pct":False,
                "observed_memory_clocks":["810 MHz"],
                "head_diagnostic":{"status":"unavailable","reason":"Synthetic fixture omits head"}}
        for entry in corpus["tensors"]:
            n,k=entry["shape"]
            raw=n*k*2
            copies=math.ceil((80<<20)/raw)
            selected=profile["choices"][f"{n},{k}"]
            def method(us):
                return {"median_us":float(us),"samples_us":[float(us)]*31,
                        "numerics":{"passed":True,"elements_checked":copies*n,
                                    "oracle_elements_checked":n,"all_copies_bitwise_equal":True}}
            case={"weight_tensor":entry["name"],"weight_sha256":entry["sha256"],
                  "shape_mnk":[1,n,k],"layer":int(entry["name"].split(".")[2]),"role":"held-out",
                  "seed":seed,"input_dtype":"bf16","output_dtype":"bf16","selected":selected,
                  "copies":copies,"calls_per_graph":copies,"raw_ring_bytes":copies*raw,
                  "raw_ring_over_l2":copies*raw/(4<<20),"storage_bytes":int(raw*.78125),
                  "storage_ratio":.78125,"original_bits_preserved":True,"raw_template_bitwise":True,
                  "raw_template_parity_by_config":{selected:True},"operating_context_stable":True,
                  "operator_context":copy.deepcopy(context),"execution_context":copy.deepcopy(context),
                  "vendor_routes":["vendor"],"methods":{"vendor":method(100),"vendor_aa":method(100),
                      selected:method(80),selected.replace("coded_","raw_",1):method(80)},
                  "gpu_telemetry":[{"ok":True,"phase":"measurement",
                    "state":"Synthetic GPU, P5, 70, 50 W, 1530 MHz, 810 MHz, Not Active, Not Active"}],
                  # These deliberately disagree with recomputed positive gates.
                  "passes_10pct_gate":False,"time_saved_pct":-999.,"ci95_pct":[-999.,-999.]}
            report["cases"].append(case)
        reports.append(report)
    return spec,corpus,profile,reports


class CompressedDispatchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.spec,self.corpus,self.profile,self.reports=fixture()

    def build(self):
        from compressed_dispatch_evidence import build_evidence
        return build_evidence(self.reports,self.spec,self.corpus,self.profile,profile_sha256=PROFILE_SHA)

    def test_raw_three_process_gates_do_not_trust_saved_flags(self):
        result=self.build()
        self.assertEqual(result["kind"],"compressed-dispatch-evidence")
        self.assertTrue(result["shapes"]["1,64,64"]["kernel_gate_passed"])
        self.assertFalse(result["shapes"]["1,64,64"]["custom_win_validated"])
        self.assertEqual(result["shapes"]["1,64,64"]["dispatch_scope_gate"],"pending")
        self.assertEqual(len(result["cases"]),18)

    def test_one_regressed_layer_prevents_only_its_shape_from_passing(self):
        case=self.reports[-1]["cases"][-1]
        case["methods"][case["selected"]].update(median_us=160.,samples_us=[160.]*31)
        case["passes_10pct_gate"]=True
        result=self.build()
        self.assertFalse(result["shapes"]["1,128,64"]["kernel_gate_passed"])
        self.assertTrue(result["shapes"]["1,64,64"]["kernel_gate_passed"])

    def test_numeric_bit_parity_and_storage_failures_remain_negative_outcomes(self):
        original=copy.deepcopy(self.reports)
        for failure in ("numerics","original_bits_preserved","raw_template_bitwise","storage_ratio"):
            with self.subTest(failure=failure):
                self.reports=copy.deepcopy(original)
                case=self.reports[0]["cases"][0]
                if failure=="numerics":
                    case["methods"][case["selected"]]["numerics"]["passed"]=False
                elif failure=="storage_ratio":
                    case["storage_bytes"]=case["shape_mnk"][1]*case["shape_mnk"][2]*2
                    case["storage_ratio"]=1.
                else:
                    case[failure]=False
                result=self.build()
                self.assertFalse(result["shapes"]["1,64,64"]["kernel_gate_passed"])
                self.assertEqual(len(result["cases"]),18)

    def test_vendor_winner_uses_its_own_aa_twin(self):
        case=self.reports[0]["cases"][0]
        case["methods"]["vendor_aa"].update(median_us=110.,samples_us=[110.]*31)
        self.assertFalse(self.build()["shapes"]["1,64,64"]["kernel_gate_passed"])

    def test_exact_ten_percent_is_below_the_strict_acceptance_gate(self):
        for report in self.reports:
            for case in report["cases"]:
                case["methods"][case["selected"]].update(median_us=90.,samples_us=[90.]*31)
        self.assertFalse(self.build()["shapes"]["1,64,64"]["kernel_gate_passed"])

    def test_incomplete_or_reused_process_and_missing_layer_are_rejected(self):
        original=copy.deepcopy(self.reports)
        for failure in ("missing_seed","training","pid","training_pid","layer","context","identity","clock"):
            with self.subTest(failure=failure):
                self.reports=copy.deepcopy(original)
                if failure=="missing_seed": self.reports.pop()
                elif failure=="training": self.reports[0]["mode"]="tune"
                elif failure=="pid": self.reports[1]["pid"]=self.reports[0]["pid"]
                elif failure=="training_pid": self.reports[0]["pid"]=self.profile["pid"]
                elif failure=="layer": self.reports[0]["cases"].pop()
                elif failure=="context": self.reports[0]["operator_context"]["ac_online"]=False
                elif failure=="identity": self.reports[0]["identity"]={"stale":True}
                elif failure=="clock": self.reports[0]["observed_memory_clocks"]=["7000 MHz"]
                with self.assertRaises(ValueError): self.build()

    def test_forged_medians_and_partial_output_coverage_are_rejected(self):
        original=copy.deepcopy(self.reports)
        for failure in ("median","coverage","weight","choice","profile"):
            with self.subTest(failure=failure):
                self.reports=copy.deepcopy(original)
                case=self.reports[0]["cases"][0]
                if failure=="median": case["methods"][case["selected"]]["median_us"]=1.
                elif failure=="coverage": case["methods"][case["selected"]]["numerics"]["elements_checked"]-=1
                elif failure=="weight": case["weight_sha256"]="0"*64
                elif failure=="choice": case["selected"]="coded_t32_v2"
                elif failure=="profile": self.reports[0]["profile_sha256"]="0"*64
                with self.assertRaises(ValueError): self.build()

    def test_numerically_ineligible_fast_vendor_cannot_change_denominator(self):
        for report in self.reports:
            for case in report["cases"]:
                invalid=copy.deepcopy(case["methods"]["vendor"])
                invalid.update(median_us=1.,samples_us=[1.]*31)
                invalid["numerics"]["passed"]=False
                case["methods"].update(invalid=invalid,invalid_aa=copy.deepcopy(invalid))
                case["vendor_routes"].append("invalid")
        self.assertTrue(self.build()["shapes"]["1,64,64"]["kernel_gate_passed"])

    def test_no_eligible_vendor_is_retained_as_a_failed_case(self):
        case=self.reports[0]["cases"][0]
        case["methods"]["vendor"]["numerics"]["passed"]=False
        result=self.build()
        self.assertFalse(result["shapes"]["1,64,64"]["kernel_gate_passed"])
        self.assertEqual(len(result["cases"]),18)

    def test_one_case_without_parseable_timed_clocks_invalidates_the_cohort(self):
        self.reports[0]["cases"][0]["gpu_telemetry"][0]["state"]="missing clock observation"
        with self.assertRaises(ValueError): self.build()

    def test_graph_evidence_cannot_be_relabelled_from_eager_context(self):
        self.profile["bindings"]["operator_context"]["scope"]="eager"
        for report in self.reports:
            report["operator_context"]["scope"]="eager"
            for case in report["cases"]:
                case["execution_context"]["scope"]="eager"
        with self.assertRaises(ValueError): self.build()

    def test_boolean_or_string_timings_are_not_measured_rounds(self):
        original=copy.deepcopy(self.reports)
        for value in (True,"80"):
            with self.subTest(value=value):
                self.reports=copy.deepcopy(original)
                case=self.reports[0]["cases"][0]
                method=case["methods"][case["selected"]]
                method.update(samples_us=[value]*31,median_us=float(value))
                with self.assertRaises(ValueError): self.build()

    def test_relaxed_vendor_math_cannot_match_the_accumulation_contract(self):
        original_profile=copy.deepcopy(self.profile)
        original_reports=copy.deepcopy(self.reports)
        for flag in ("allow_tf32","reduced_precision_reduction"):
            with self.subTest(flag=flag):
                self.profile=copy.deepcopy(original_profile)
                self.reports=copy.deepcopy(original_reports)
                self.profile["bindings"]["identity"][flag]=True
                for report in self.reports:
                    report["identity"][flag]=True
                with self.assertRaises(ValueError): self.build()


if __name__=="__main__":
    unittest.main()
