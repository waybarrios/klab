"""Synthetic causal-report fixtures. No unlocked hardware measurements."""
import copy
import unittest


FLAGS=("module_loaded","unlock_boot_option_present","unlock_success_logged","post_flr_selectors_logged")
CORE=("compressed_case.py","compressed_campaign.py","bench_compressed.py","bench.py",
      "projection_accuracy.py","bf16_reference.py","cublaslt_baseline.py","cublas_baseline.py",
      "gpu_monitor.py","real_weights.py","power_context.py","mtile_bf16.py","mtile_kernels.cu")
HALF=("bench_half.py","half_projection.py","half_kernels.cu","kernels.cu","kernel_lab.py","native/driver.cpp",
      "native/build/libkernel_lab.so","bench.py","cublas_baseline.py","bench_streaming.py","gpu_monitor.py")


def pair(kind="compressed-projection"):
    context={"system76_power_profile":"Power Profile: Performance","ac_online":True,"scope":"graph"}
    inventory={"gpu":"Synthetic GPU, GPU-00000000-0000-0000-0000-000000000000, driver-test",
               "kernel":"synthetic-kernel","pci_address":"0000:01:00.0",**dict.fromkeys(FLAGS,False)}
    identity={"implementation":{"fingerprint_sha256":"a"*64,"ptx_sha256":"b"*64},
        "source_sha256":dict.fromkeys(CORE,"c"*64),
        "libraries_sha256":{"libcublas.so.12":"d"*64,"libcublasLt.so.12":"e"*64},
        "torch":"synthetic-torch","cuda":"synthetic-cuda","cublas_version":120804,
        "cublaslt_version":120804,"gpu":inventory["gpu"],"kernel":inventory["kernel"],
        "unlock_evidence":dict.fromkeys(FLAGS,False),"allow_tf32":False,"reduced_precision_reduction":False}
    def method(us,passed=True):
        return {"median_us":float(us),"samples_us":[float(us)]*31,
                "numerics":{"passed":passed,"elements_checked":655360,"nonfinite":0,"violations":0}}
    case={"shape_mnk":[1,64,64],"weight_tensor":"model.layers.8.mlp.gate_proj.weight",
        "weight_sha256":"1"*64,"activation_sha256":"2"*64,"seed":919,
        "input_dtype":"bf16","output_dtype":"bf16","layer":8,"role":"held-out",
        "selected":"coded_t128_v8","copies":10240,"calls_per_graph":10240,
        "raw_ring_bytes":80<<20,"raw_ring_over_l2":20.,"storage_bytes":6400,"storage_ratio":.78125,
        "original_bits_preserved":True,"raw_template_bitwise":True,
        "raw_template_parity_by_config":{"coded_t128_v8":True},
        "operating_context_stable":True,"execution_context":copy.deepcopy(context),
        "vendor_routes":["vendor_a","vendor_b"],
        "vendor_pool":{"vendor_a":{"kind":"synthetic-only","algorithm":"frozen-a"},
                       "vendor_b":{"kind":"synthetic-only","algorithm":"frozen-b"}},
        "gpu_telemetry":[{"ok":True,"phase":"measurement",
            "state":"Synthetic GPU, P5, 70, 50 W, 1530 MHz, 810 MHz, Not Active, Not Active"}],
        "methods":{"vendor_a":method(100),"vendor_a_aa":method(100),
                   "vendor_b":method(110),"vendor_b_aa":method(110),
                   "coded_t128_v8":method(80),"raw_t128_v8":method(80)},
        "aa_time_saved_pct":0,"passes_10pct_gate":True}
    stock={"kind":kind,"status":"complete","mode":"validate","pid":1001,"run_id":"synthetic-stock",
        "schema_version":2,"synthetic_fixture":True,"seed":919,"cases":[case],"revision":2,
        "output_dtype":"bf16","spec_sha256":"3"*64,"corpus_sha256":"4"*64,
        "vendor_profile_sha256":"5"*64,"context_protocol_sha256":"6"*64,
        "profile_sha256":"7"*64,"operator_context":context,"inventory":inventory,
        "identity":identity,"choices":{"64,64":"coded_t128_v8"},
        "observed_memory_clocks":["810 MHz"],"full_model_speedup_claim":False}
    unlocked=copy.deepcopy(stock)
    unlocked.update(pid=2001,run_id="synthetic-unlocked",profile_sha256="8"*64,
                    frozen_choice_origin_sha256=stock["profile_sha256"])
    unlocked["inventory"].update(dict.fromkeys(FLAGS,True))
    unlocked["identity"]["unlock_evidence"].update(dict.fromkeys(FLAGS,True))
    for name,us in (("vendor_a",80),("vendor_a_aa",80),("vendor_b",70),("vendor_b_aa",70),
                    ("coded_t128_v8",64),("raw_t128_v8",64)):
        unlocked["cases"][0]["methods"][name]=method(us)
    if kind=="half-projection":
        for report in (stock,unlocked):
            inv=report.pop("inventory")
            ident=report.pop("identity")
            report.update(dtype="fp16",output_dtype="fp32",source_sha256=dict.fromkeys(HALF,"c"*64),
                libraries_sha256=copy.deepcopy(ident["libraries_sha256"]),
                frozen_profile_sha256=report.pop("profile_sha256"),
                precision_flags=dict.fromkeys(("allow_tf32","allow_fp16_accumulation",
                    "allow_fp16_reduced_precision_reduction","allow_bf16_reduced_precision_reduction"),False),
                environment={"inventory":inv,"torch":ident["torch"],"cuda":ident["cuda"],
                    "cublas_version":ident["cublas_version"],"compilation_cache_key":"a"*64},
                compilation={"cache_key":"a"*64,"ptx_sha256":"b"*64})
            report.pop("choices")
            for c in report["cases"]:
                c.update(dtype="fp16",input_dtype="fp16",output_dtype="fp32",held_out=True,
                    selected="t128_v8",selected_threads=128,selected_vector=8,
                    bitwise_layout_check=True,ring_bytes={"weights":80<<20,"transposed_weights":80<<20})
                c.pop("original_bits_preserved")
                c.pop("raw_template_bitwise")
                c.pop("raw_template_parity_by_config")
                c["methods"]["half_t128_v8"]=c["methods"].pop("coded_t128_v8")
                c["methods"].pop("raw_t128_v8")
    return stock,unlocked


class ExtendedFactorialTests(unittest.TestCase):
    def compare(self,stock,unlocked):
        from factorial_projection import compare_projection
        return compare_projection(stock,unlocked)

    def test_matched_four_cells_keep_the_stock_vendor_fixed_for_unlock_effect(self):
        row=self.compare(*pair())[0]
        self.assertEqual(row["stock_best_vendor"],"vendor_a")
        self.assertEqual(row["unlocked_best_vendor"],"vendor_b")
        self.assertAlmostEqual(row["our_gain_stock_pct"],20)
        self.assertAlmostEqual(row["our_gain_unlocked_pct"],100*(1-64/70))
        self.assertAlmostEqual(row["unlock_gain_vendor_pct"],20)
        self.assertAlmostEqual(row["combined_gain_pct"],36)
        self.assertAlmostEqual(row["interaction_extra_latency_reduction_pct"],0)
        self.assertTrue(row["attribution_qualified"])
        self.assertEqual(row["weight_tensor"],"model.layers.8.mlp.gate_proj.weight")

    def test_same_shape_different_checkpoint_tensors_are_not_collapsed(self):
        stock,unlocked=pair()
        for report in (stock,unlocked):
            other=copy.deepcopy(report["cases"][0])
            other["weight_tensor"]="model.layers.8.mlp.up_proj.weight"
            other["weight_sha256"]="9"*64
            report["cases"].append(other)
        self.assertEqual(len(self.compare(stock,unlocked)),2)

    def test_head_transfer_diagnostics_remain_in_the_case_table(self):
        stock,unlocked=pair()
        for report in (stock,unlocked):
            head=copy.deepcopy(report["cases"][0])
            head.update(weight_tensor="model.embed_tokens.weight",weight_sha256="9"*64,
                        shape_mnk=[1,128,64],copies=5120,calls_per_graph=5120,
                        storage_bytes=12800,role="head-transfer-diagnostic",layer=None)
            report["head_diagnostic"]={"status":"included","config_source":"frozen gate/up winner"}
            report["cases"].append(head)
        rows=self.compare(stock,unlocked)
        self.assertEqual(len(rows),2)
        self.assertTrue(any(row["weight_tensor"]=="model.embed_tokens.weight" for row in rows))

    def test_invalid_nested_build_digest_cannot_count_as_binary_identity(self):
        stock,unlocked=pair()
        for report in (stock,unlocked):
            report["identity"]["implementation"]["module_fingerprint"]={
                "source_sha256":{"native/build/libkernel_lab.so":"not-a-sha256"}}
        with self.assertRaises(ValueError): self.compare(stock,unlocked)

    def test_noise_failure_is_retained_and_cached_success_cannot_override_it(self):
        stock,unlocked=pair()
        stock["cases"][0]["methods"]["vendor_a_aa"].update(median_us=105.,samples_us=[105.]*31)
        row=self.compare(stock,unlocked)[0]
        self.assertFalse(row["attribution_qualified"])
        self.assertFalse(row["stock_10pct_target"])
        self.assertTrue(row["failure_reasons"])

    def test_numerically_ineligible_fast_route_never_wins(self):
        stock,unlocked=pair()
        for report in (stock,unlocked):
            for name in ("vendor_b","vendor_b_aa"):
                report["cases"][0]["methods"][name].update(median_us=1.,samples_us=[1.]*31)
                report["cases"][0]["methods"][name]["numerics"]["passed"]=False
        self.assertEqual(self.compare(stock,unlocked)[0]["unlocked_best_vendor"],"vendor_a")

    def test_missing_real_unlock_evidence_is_rejected(self):
        for flag in FLAGS:
            stock,unlocked=pair()
            unlocked["inventory"][flag]=False
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                self.compare(stock,unlocked)

    def test_input_precision_binary_algorithm_and_power_confounds_are_rejected(self):
        for mutation in ("weight","activation","output","math","binary","algorithm","profile_origin","power","clock"):
            stock,unlocked=pair()
            if mutation=="weight": unlocked["cases"][0]["weight_sha256"]="f"*64
            if mutation=="activation": unlocked["cases"][0]["activation_sha256"]="f"*64
            if mutation=="output": unlocked["output_dtype"]="fp32"
            if mutation=="math": unlocked["identity"]["allow_tf32"]=True
            if mutation=="binary": unlocked["identity"]["implementation"]["ptx_sha256"]="f"*64
            if mutation=="algorithm": unlocked["cases"][0]["vendor_pool"]["vendor_a"]["algorithm"]="changed"
            if mutation=="profile_origin": unlocked.pop("frozen_choice_origin_sha256")
            if mutation=="power": unlocked["operator_context"]["ac_online"]=False
            if mutation=="clock": unlocked["observed_memory_clocks"]=["6000 MHz"]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.compare(stock,unlocked)

    def test_incomplete_rounds_duplicate_cases_and_reused_process_rejected(self):
        for mutation in ("rounds","median","duplicates","run_id","pid","tune"):
            stock,unlocked=pair()
            if mutation=="rounds": unlocked["cases"][0]["methods"]["vendor_a"]["samples_us"].pop()
            if mutation=="median": unlocked["cases"][0]["methods"]["vendor_a"]["median_us"]=1.
            if mutation=="duplicates": unlocked["cases"]*=2
            if mutation in ("run_id","pid"): unlocked[mutation]=stock[mutation]
            if mutation=="tune": unlocked["mode"]="tune"
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.compare(stock,unlocked)

    def test_native_half_reports_require_exact_inputs_and_explicit_math_flags(self):
        stock,unlocked=pair("half-projection")
        row=self.compare(stock,unlocked)[0]
        self.assertTrue(row["attribution_qualified"])
        self.assertEqual(row["input_dtype"],"fp16")
        self.assertEqual(row["output_dtype"],"fp32")
        unlocked["cases"][0].pop("weight_sha256")
        with self.assertRaises(ValueError): self.compare(stock,unlocked)

    def test_native_half_requires_measurement_source_and_library_binary_hashes(self):
        for omitted in ("measurement_source","libraries"):
            stock,unlocked=pair("half-projection")
            for report in (stock,unlocked):
                if omitted=="measurement_source": report["source_sha256"].pop("bench_half.py")
                else: report.pop("libraries_sha256")
            with self.subTest(omitted=omitted),self.assertRaises(ValueError):
                self.compare(stock,unlocked)
        stock,unlocked=pair("half-projection")
        unlocked.pop("precision_flags")
        with self.assertRaises(ValueError): self.compare(stock,unlocked)

    def test_missing_unlocked_measurements_stay_null_not_zero(self):
        from factorial_projection import pending_projection
        stock,_=pair()
        row=pending_projection(stock)[0]
        self.assertEqual(row["comparison_status"],"pending-unlocked")
        self.assertIsNone(row["unlocked_vendor_us"])
        self.assertIsNone(row["unlocked_custom_us"])
        self.assertIsNone(row["combined_gain_pct"])
        self.assertIsNone(row["unlock_gain_vendor_pct"])
        self.assertFalse(row["attribution_qualified"])


if __name__=="__main__":
    unittest.main()
