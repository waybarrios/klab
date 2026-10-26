"""Synthetic experiment-policy fixtures, never hardware performance results."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def evidence():
    return {"kind":"compressed-dispatch-evidence","status":"complete",
        "contract":{"input_dtype":"bf16","weight_dtype":"bf16","accumulation_dtype":"fp32",
                    "output_dtype":"bf16","lossy_weights":False,"allow_tf32":False,
                    "reduced_precision_reduction":False},
        "execution_mode":"graph","eager_promoted":False,"kernel_milestone_passed":True,
        "bindings":{"output_dtype":"bf16","revision":2},
        "shapes":{"1,4864,896":{"shape_mnk":[1,4864,896],"selected":"coded_t128_v8",
            "kernel_gate_passed":True,"custom_win_validated":False,"dispatch_scope_gate":"pending"},
            "1,128,896":{"shape_mnk":[1,128,896],"selected":"coded_t64_v8",
            "kernel_gate_passed":False,"custom_win_validated":False,"dispatch_scope_gate":"pending"}}}


def method(us,passed=True):
    return {"median_us":float(us),"samples_us":[float(us)]*31,
            "numerics":{"passed":passed,"elements_checked":64}}


class PreparedExperimentTests(unittest.TestCase):
    def training_fixture(self):
        from test_compressed_dispatch_evidence import fixture
        spec,corpus,profile,reports=fixture()
        training={"kind":"compressed-projection","status":"complete","mode":"tune",
                  "seed":901,"pid":profile["pid"],"run_id":profile["run_id"],
                  **copy.deepcopy(profile["bindings"]),"choices":copy.deepcopy(profile["choices"]),
                  "cases":[]}
        for layer in spec["tuning_layers"]:
            for template in reports[0]["cases"][:2]:
                case=copy.deepcopy(template)
                suffix=case["weight_tensor"].split(".",3)[-1]
                case.update(weight_tensor=f"model.layers.{layer}.{suffix}",layer=layer,role="training",seed=901)
                corpus["tensors"].append({"name":case["weight_tensor"],"shape":case["shape_mnk"][1:],
                                          "sha256":case["weight_sha256"]})
                training["cases"].append(case)
        return spec,corpus,profile,training

    def test_training_hash_reference_alone_cannot_validate_training_content(self):
        from prepared_experiment import validate_training
        spec,corpus,profile,training=self.training_fixture()
        validate_training(profile,training,spec,corpus)
        for field,value in (("status","running"),("mode","validate"),("pid",9999)):
            altered=copy.deepcopy(training)
            altered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_training(profile,altered,spec,corpus)
        altered=copy.deepcopy(training)
        altered["cases"].pop()
        with self.assertRaises(ValueError):
            validate_training(profile,altered,spec,corpus)

    def test_frozen_choices_must_reproduce_from_training_rounds(self):
        from prepared_experiment import validate_training
        spec,corpus,profile,training=self.training_fixture()
        for case in training["cases"]:
            if case["shape_mnk"][1:]==[64,64]:
                case["methods"]["coded_t256_v8"]=method(70)
        with self.assertRaises(ValueError):
            validate_training(profile,training,spec,corpus)

    def test_source_change_during_campaign_prevents_completion(self):
        import prepared_experiment as pe
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/".scratch") as directory:
            root=Path(directory)
            path=root/"candidate.py"
            path.write_text("before")
            report={"source_sha256":{"candidate.py":hashlib.sha256(path.read_bytes()).hexdigest()}}
            with mock.patch.object(pe,"ROOT",root):
                pe.check_frozen_sources(report)
                path.write_text("after")
                with self.assertRaises(ValueError):
                    pe.check_frozen_sources(report)

    def test_only_shape_gate_allows_a_benchmark_candidate(self):
        from prepared_experiment import candidate_plan
        artifact=evidence()
        large=candidate_plan(artifact,(1,4864,896),output_dtype="bf16")
        self.assertEqual(large["route"],"coded")
        self.assertEqual(large["config"],"coded_t128_v8")
        self.assertFalse(large["custom_win_validated"])
        small=candidate_plan(artifact,(1,128,896),output_dtype="bf16")
        self.assertEqual(small["route"],"vendor")
        self.assertFalse(small["requires_packing"])
        self.assertFalse(small["custom_win_validated"])
        unseen=candidate_plan(artifact,(64,128,896),output_dtype="bf16")
        self.assertEqual(unseen["route"],"vendor")
        self.assertFalse(unseen["requires_packing"])

    def test_wrong_precision_or_non_graph_evidence_cannot_match(self):
        from prepared_experiment import candidate_plan
        with self.assertRaises(ValueError):
            candidate_plan(evidence(),(1,4864,896),output_dtype="fp32")
        for field,value in (("execution_mode","eager"),("status","running")):
            artifact=evidence()
            artifact[field]=value
            with self.assertRaises(ValueError):
                candidate_plan(artifact,(1,4864,896),output_dtype="bf16")
        artifact=evidence()
        artifact["contract"]["lossy_weights"]=True
        with self.assertRaises(ValueError):
            candidate_plan(artifact,(1,4864,896),output_dtype="bf16")

    def test_complete_grid_includes_all_layers_tokens_head_and_controls(self):
        from prepared_experiment import registered_cases
        spec={"held_out_layers":[8,16],"projection_weights":["q.weight","k.weight"],
              "m_values":[1,2],"baseline_diagnostic_m":[64],
              "head":{"tensor":"head.weight"},"synthetic_controls_mnk":[[1,64,64],[4,256,256]]}
        corpus={"tensors":[{"name":f"model.layers.{layer}.{suffix}","shape":[128,896]}
                for layer in spec["held_out_layers"] for suffix in spec["projection_weights"]]
                +[{"name":"head.weight","shape":[151936,896]}]}
        cases=registered_cases(spec,corpus)
        self.assertEqual(len(cases),17)
        self.assertEqual(sum(c["role"]=="synthetic-control" for c in cases),2)
        self.assertEqual(sum(c["role"]=="head-transfer-diagnostic" for c in cases),3)
        real=[c for c in cases if c["role"]=="held-out"]
        self.assertEqual(len(real),12)
        self.assertEqual({c["layer"] for c in real},{8,16})
        self.assertEqual({c["shape_mnk"][0] for c in real},{1,2,64})
        self.assertEqual(len({(c["weight_tensor"],tuple(c["shape_mnk"])) for c in cases}),17)
        incomplete=copy.deepcopy(corpus)
        incomplete["tensors"].pop(0)
        with self.assertRaises(ValueError):
            registered_cases(spec,incomplete)

    def test_vendor_denominator_excludes_ineligible_algorithm(self):
        from prepared_experiment import summarize_graph
        methods={"vendor":method(100),"vendor_aa":method(100),
                 "invalid":method(1,False),"invalid_aa":method(1,False),
                 "prepared_direct":method(80),"prepared_direct_aa":method(80),
                 "prepared":method(80)}
        result=summarize_graph(methods,["vendor","invalid"],route="coded")
        self.assertEqual(result["versus_vendor"]["best_vendor"],"vendor")
        self.assertAlmostEqual(result["versus_vendor"]["time_saved_pct"],20)
        self.assertTrue(result["implementation_overhead"]["overhead_gate_passed"])
        self.assertFalse(result["custom_win_validated"])
        self.assertFalse(result["eager_promoted"])

    def test_fallback_never_claims_a_custom_win_and_own_aa_is_required(self):
        from prepared_experiment import summarize_graph
        methods={"vendor":method(100),"vendor_aa":method(100),
                 "prepared_direct":method(100),"prepared_direct_aa":method(105),
                 "prepared":method(100)}
        result=summarize_graph(methods,["vendor"],route="vendor")
        self.assertFalse(result["implementation_overhead"]["overhead_gate_passed"])
        self.assertFalse(result["custom_win_validated"])
        self.assertFalse(result["eager_promoted"])


if __name__=="__main__":
    unittest.main()
