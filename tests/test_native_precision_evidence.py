"""Synthetic CPU fixtures only, never hardware claims."""
import copy
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def fixture():
    from compressed_campaign import summarize_methods
    protocol = {"held_out_seeds": [971, 977, 983], "minimum_rounds": 31,
                "cases_by_dtype": {"fp16": [{"shape_mnk": [4, 256, 256], "config": "half_t256_v8"}]}}
    context = {"system76_power_profile": "Power Profile: Performance", "ac_online": True, "scope": "graph"}
    numeric = {"passed": True, "elements_checked": 640 * 4 * 256, "oracle_elements_checked": 4 * 256,
               "all_copies_bitwise_equal": True, "nonfinite": 0, "violations": 0, "atol": 1e-5, "rtol": 1e-4}
    routes = ["cublas_compute32", "cublas_packed_compute32", "torch_mm_out_fp32"]
    methods = {name: {"median_us": value, "samples_us": [value] * 31, "numerics": copy.deepcopy(numeric)}
               for name, value in [(name, 100.) for route in routes for name in (route, route + "_aa")] + [("custom", 80.)]}
    case = {"shape_mnk": [4, 256, 256], "config": "half_t256_v8", "input_dtype": "fp16", "weight_dtype": "fp16",
            "output_dtype": "fp32", "accumulation_dtype": "fp32", "copies": 640, "calls_per_graph": 640,
            "raw_ring_bytes": 80 << 20, "original_bits_preserved": True, "bitwise_round_trip": True,
            "bitwise_layout_check": True, "methods": methods,
            "numerics_before": {key: copy.deepcopy(numeric) for key in methods},
            "search_diagnostics": [{"layout": layout, "selected": [], "returned": 0, "requested": 8,
                                    "eligible": 0, "max_candidates": 8} for layout in ("original", "transposed")],
            "context_before": context, "context_after": context, "observed_memory_clocks": ["810 MHz"],
            "memory_clock_regime_stable": True, "accepted_10pct_target": True,
            "gpu_telemetry": [{"ok": True, "phase": "measurement", "state": "GPU,P5,50,30 W,1400 MHz,810 MHz,Active,Active"}],
            "summary": summarize_methods(methods, routes, "custom")}
    case["vendor_routes"] = routes
    case["vendor_pool"] = {"cublas_compute32": {"kind": "cublasGemmEx", "compute_type": 68, "layout": "original"},
                           "cublas_packed_compute32": {"kind": "cublasGemmEx", "compute_type": 68, "layout": "transposed"},
                           "torch_mm_out_fp32": {"kind": "torch.mm", "out_dtype": "fp32"}}
    reports = []
    for i, seed in enumerate(protocol["held_out_seeds"]):
        row = copy.deepcopy(case)
        row.update(seed=seed, weight_sha256=f"{seed:064x}", activation_sha256=f"{seed+1000:064x}")
        reports.append({"kind": "native-precision-projection", "status": "complete", "dtype": "fp16", "seed": seed,
                        "pid": 100 + i, "run_id": f"synthetic-{i}", "rounds": 31, "operator_context": context,
                        "source_sha256": {"module.py": "a" * 64}, "source_profiles_sha256": {"profile.json": "b" * 64},
                        "libraries_sha256": {"libcublas": "c" * 64}, "ptx_sha256": "d" * 64,
                        "nvrtc_identity": "NVRTC synthetic:sha256=" + "e" * 64,
                        "compilation": {"cache_key": "f" * 64, "architecture": "86", "compiler": "synthetic"},
                        "precision_flags": {"allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
                                            "allow_bf16_reduced_precision_reduction": False},
                        "environment": {"torch": "synthetic", "cuda": "synthetic", "cublas_version": 1,
                            "inventory": {"gpu": "GPU-synthetic", "kernel": "synthetic", "module_loaded": False,
                                "unlock_boot_option_present": False, "unlock_success_logged": False, "post_flr_selectors_logged": False}},
                        "cases": [row]})
    return protocol, reports


class NativePrecisionEvidenceTests(unittest.TestCase):
    def test_requires_exactly_three_fresh_processes_without_duplicate_seed(self):
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        self.assertTrue(build_cohort(reports, protocol, dtype="fp16", profile="performance")["shapes"][0]["accepted_10pct"])
        for invalid in (reports[:2], reports + [copy.deepcopy(reports[0])]):
            with self.assertRaises(ValueError):
                build_cohort(invalid, protocol, dtype="fp16", profile="performance")
        reports[1]["pid"] = reports[0]["pid"]
        with self.assertRaises(ValueError):
            build_cohort(reports, protocol, dtype="fp16", profile="performance")

    def test_valid_negative_case_is_retained_and_does_not_qualify(self):
        from compressed_campaign import summarize_methods
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        case = reports[1]["cases"][0]
        case["methods"]["custom"].update(median_us=110., samples_us=[110.] * 31)
        case["summary"] = summarize_methods(case["methods"], case["vendor_routes"], "custom")
        case["accepted_10pct_target"] = False
        result = build_cohort(reports, protocol, dtype="fp16", profile="performance")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["shapes"][0]["accepted_10pct"])
        self.assertEqual(len(result["cases"]), 3)

    def test_changed_runtime_failed_report_or_missing_numeric_coverage_reject(self):
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        for key, value in (("status", "failed"), ("ptx_sha256", "0" * 64)):
            altered = copy.deepcopy(reports)
            altered[1][key] = value
            with self.assertRaises(ValueError):
                build_cohort(altered, protocol, dtype="fp16", profile="performance")
        reports[1]["cases"][0]["methods"]["custom"]["numerics"]["elements_checked"] -= 1
        with self.assertRaises(ValueError):
            build_cohort(reports, protocol, dtype="fp16", profile="performance")

    def test_cross_process_clock_change_cannot_qualify(self):
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        case = reports[1]["cases"][0]
        case["observed_memory_clocks"] = ["7001 MHz"]
        case["gpu_telemetry"][0]["state"] = case["gpu_telemetry"][0]["state"].replace("810 MHz", "7001 MHz")
        result = build_cohort(reports, protocol, dtype="fp16", profile="performance")
        self.assertFalse(result["shapes"][0]["accepted_10pct"])

    def test_precheck_failure_remains_a_valid_negative_even_if_after_counts_are_zero(self):
        from compressed_campaign import summarize_methods
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        case = reports[1]["cases"][0]
        case["numerics_before"]["custom"].update(passed=False, violations=1)
        case["methods"]["custom"]["numerics"]["passed"] = False
        case["summary"] = summarize_methods(case["methods"], case["vendor_routes"], "custom")
        case["accepted_10pct_target"] = False
        result = build_cohort(reports, protocol, dtype="fp16", profile="performance")
        self.assertEqual(result["cases"][1]["failed_numerical_methods"], ["custom"])
        self.assertFalse(result["shapes"][0]["accepted_10pct"])

    def test_dropped_vendor_forged_acceptance_or_replayed_inputs_reject(self):
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        for mutation in ("missing_vendor", "cached_acceptance", "same_inputs"):
            altered = copy.deepcopy(reports)
            case = altered[1]["cases"][0]
            if mutation == "missing_vendor":
                case["vendor_routes"].remove("cublas_compute32")
            elif mutation == "cached_acceptance":
                case["accepted_10pct_target"] = False
            else:
                case["weight_sha256"] = altered[0]["cases"][0]["weight_sha256"]
            with self.assertRaises(ValueError):
                build_cohort(altered, protocol, dtype="fp16", profile="performance")

    def test_file_discovery_keeps_renamed_attempts_and_rejects_duplicates(self):
        from native_precision_evidence import build_cohort, discover_attempts
        protocol, reports = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            for report in reports:
                self._write_attempt(root, f"native-precision-performance-fp16-{report['seed']}-r1", report)
            found, artifacts = discover_attempts(root, dtype="fp16", profile="performance")
            self.assertEqual(len(found), 3)
            self.assertEqual(len(artifacts), 6)
            duplicate = copy.deepcopy(reports[0])
            duplicate.update(pid=999, run_id="renamed-extra-attempt")
            self._write_attempt(root, "another-output-name", duplicate)
            found, _ = discover_attempts(root, dtype="fp16", profile="performance")
            self.assertEqual(len(found), 4)
            with self.assertRaises(ValueError):
                build_cohort(found, protocol, dtype="fp16", profile="performance")

    @staticmethod
    def _write_attempt(root, stem, report):
        state = {"pid": report["pid"], "status": "complete", "restored": True,
                 "initial": {"profile": "performance", "brightness": 255},
                 "after_restore": {"profile": "performance", "brightness": 255}}
        (root / "results" / (stem + ".json")).write_text(json.dumps(report))
        (root / "results" / (stem + "-state.json")).write_text(json.dumps(state))

    def test_unrestored_or_failed_timed_attempt_cannot_be_ignored(self):
        from native_precision_evidence import build_cohort, discover_attempts
        protocol, reports = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            for report in reports:
                self._write_attempt(root, f"native-precision-performance-fp16-{report['seed']}-r1", report)
            state_path = root / "results/native-precision-performance-fp16-977-r1-state.json"
            state = json.loads(state_path.read_text())
            state["restored"] = False
            state_path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                discover_attempts(root, dtype="fp16", profile="performance")
            state["restored"] = True
            state_path.write_text(json.dumps(state))
            failed_path = root / "results/native-precision-performance-fp16-977-r1.json"
            failed = json.loads(failed_path.read_text())
            failed["status"] = "failed"
            failed_path.write_text(json.dumps(failed))
            found, _ = discover_attempts(root, dtype="fp16", profile="performance")
            with self.assertRaises(ValueError):
                build_cohort(found, protocol, dtype="fp16", profile="performance")

    def test_nonfinite_failure_without_tolerance_fields_is_retained(self):
        from compressed_campaign import summarize_methods
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        case = reports[1]["cases"][0]
        for record in (case["numerics_before"]["custom"], case["methods"]["custom"]["numerics"]):
            record.update(passed=False, nonfinite=1, violations=1)
            record.pop("atol")
            record.pop("rtol")
        case["summary"] = summarize_methods(case["methods"], case["vendor_routes"], "custom")
        case["accepted_10pct_target"] = False
        result = build_cohort(reports, protocol, dtype="fp16", profile="performance")
        self.assertEqual(result["cases"][1]["failed_numerical_methods"], ["custom"])
        self.assertFalse(result["shapes"][0]["accepted_10pct"])

    def test_startup_lock_failure_and_balanced_restoration_are_valid(self):
        from native_precision_evidence import discover_attempts
        _, reports = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            self._write_attempt(root, "native-precision-performance-fp16-971-r1", reports[0])
            state_path = root / "results/native-precision-performance-fp16-971-r1-state.json"
            state = json.loads(state_path.read_text())
            state["initial"]["profile"] = state["after_restore"]["profile"] = "balanced"
            state_path.write_text(json.dumps(state))
            failed_path = root / "results/native-precision-performance-fp16-preflight-state.json"
            failed_path.write_text(json.dumps({"pid": 999, "status": "failed", "error": ""}))
            found, artifacts = discover_attempts(root, dtype="fp16", profile="performance")
            self.assertEqual(len(found), 1)
            self.assertEqual(len(artifacts), 3)

    def test_nested_and_state_named_raw_attempts_are_discovered(self):
        from native_precision_evidence import discover_attempts
        _, reports = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            for folder, stem, report in (("results/sub", "renamed", reports[0]),
                                         (".scratch", "x-state", reports[1]),
                                         (".cache", "hidden", reports[2])):
                (root / folder).mkdir(parents=True, exist_ok=True)
                # _write_attempt expects root/results; preserve the arbitrary path.
                path = root / folder / (stem + ".json")
                path.write_text(json.dumps(report))
                state = {"pid": report["pid"], "status": "complete", "restored": True,
                         "initial": {"profile": "performance", "brightness": 255},
                         "after_restore": {"profile": "performance", "brightness": 255}}
                path.with_name(path.stem + "-state.json").write_text(json.dumps(state))
            found, _ = discover_attempts(root, dtype="fp16", profile="performance")
            self.assertEqual({r["seed"] for r in found}, {971, 977, 983})

    def test_lt_metadata_contract_and_runtime_binding_are_checked(self):
        from compressed_campaign import summarize_methods
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        for report in reports:
            case = report["cases"][0]
            case["vendor_routes"].append("lt_original_h0")
            case["vendor_pool"]["lt_original_h0"] = {
                "library_version": 1, "device_uuid": "GPU-synthetic", "library_path": "libcublas",
                "input_dtype": "fp16", "output_dtype": "fp32", "shape_mnk": case["shape_mnk"],
                "layout": "original", "candidate_idx": 0, "compute_type": 68, "reduction_scheme": 0,
                "workspace_bytes": 0, "minimum_alignment": [4] * 4,
                "algorithm_b64": base64.b64encode(bytes(64)).decode()}
            for name in ("lt_original_h0", "lt_original_h0_aa"):
                case["methods"][name] = copy.deepcopy(case["methods"]["cublas_compute32"])
                case["numerics_before"][name] = copy.deepcopy(case["numerics_before"]["cublas_compute32"])
            case["search_diagnostics"][0].update(selected=[0], returned=1, eligible=1)
            case["summary"] = summarize_methods(case["methods"], case["vendor_routes"], "custom")
        self.assertTrue(build_cohort(reports, protocol, dtype="fp16", profile="performance")["shapes"][0]["accepted_10pct"])
        for key, value in (("device_uuid", "another-gpu"), ("reduction_scheme", 4), ("library_path", "wrong.so")):
            altered = copy.deepcopy(reports)
            altered[0]["cases"][0]["vendor_pool"]["lt_original_h0"][key] = value
            with self.assertRaises(ValueError):
                build_cohort(altered, protocol, dtype="fp16", profile="performance")

    def test_fp32_m1_requires_gemv_routes(self):
        from compressed_campaign import summarize_methods
        from native_precision_evidence import build_cohort
        protocol, reports = fixture()
        protocol["cases_by_dtype"] = {"fp32": [{"shape_mnk": [1, 256, 256], "config": "tiny_t32_v4"}]}
        for report in reports:
            report["dtype"] = "fp32"
            case = report["cases"][0]
            case.update(shape_mnk=[1, 256, 256], config="tiny_t32_v4", input_dtype="fp32", weight_dtype="fp32",
                        raw_ring_bytes=160 << 20)
            case["vendor_pool"].pop("torch_mm_out_fp32")
            case["methods"].pop("torch_mm_out_fp32")
            case["methods"].pop("torch_mm_out_fp32_aa")
            case["numerics_before"].pop("torch_mm_out_fp32")
            case["numerics_before"].pop("torch_mm_out_fp32_aa")
            for name, record in (("torch_mm", {"kind": "torch.mm"}), ("torch_mv", {"kind": "torch.mv"}),
                                 ("cublas_mv", {"kind": "cublasSgemv", "layout": "original"}),
                                 ("cublas_packed_mv", {"kind": "cublasSgemv", "layout": "transposed"})):
                case["vendor_pool"][name] = record
                for label in (name, name + "_aa"):
                    case["methods"][label] = copy.deepcopy(case["methods"]["cublas_compute32"])
                    case["numerics_before"][label] = copy.deepcopy(case["numerics_before"]["cublas_compute32"])
            case["vendor_routes"] = list(case["vendor_pool"])
            for records in (case["numerics_before"].values(), (v["numerics"] for v in case["methods"].values())):
                for record in records:
                    record.update(elements_checked=640 * 256, oracle_elements_checked=256)
            case["summary"] = summarize_methods(case["methods"], case["vendor_routes"], "custom")
        self.assertTrue(build_cohort(reports, protocol, dtype="fp32", profile="performance")["shapes"][0]["accepted_10pct"])
        reports[0]["cases"][0]["vendor_routes"].remove("torch_mv")
        with self.assertRaises(ValueError):
            build_cohort(reports, protocol, dtype="fp32", profile="performance")

    def test_file_audit_checks_compiler_and_runtime_bytes(self):
        from native_precision_evidence import verify_files
        _, reports = fixture()
        report = reports[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            for field, name in (("source_sha256", "module.py"), ("source_profiles_sha256", "profile.json"),
                                ("libraries_sha256", "libcublas")):
                (root / name).write_bytes(name.encode())
                report[field] = {name: hashlib.sha256(name.encode()).hexdigest()}
            for name in ("compiled.ptx", "nvrtc.so"):
                (root / name).write_bytes(name.encode())
            report.update(ptx_path="compiled.ptx", ptx_sha256=hashlib.sha256(b"compiled.ptx").hexdigest(),
                          nvrtc_path="nvrtc.so", nvrtc_identity="synthetic:sha256=" + hashlib.sha256(b"nvrtc.so").hexdigest())
            self.assertEqual(len(verify_files(report, root)), 5)
            (root / "nvrtc.so").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_files(report, root)

    def test_cli_only_writes_evidence_after_discovery_and_file_verification(self):
        import native_precision_evidence as evidence
        protocol, reports = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".scratch") as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            (root / "experiments").mkdir()
            for name in ("module.py", "profile.json", "libcublas", "compiled.ptx", "nvrtc.so"):
                (root / name).write_bytes(name.encode())
            digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ("module.py", "profile.json", "libcublas", "compiled.ptx", "nvrtc.so")}
            protocol["source_profiles"] = {"fixture": {"path": "profile.json", "sha256": digests["profile.json"]}}
            (root / "experiments/native_precision_refresh.json").write_text(json.dumps(protocol))
            for report in reports:
                report.update(source_sha256={"module.py": digests["module.py"]},
                              source_profiles_sha256={"profile.json": digests["profile.json"]},
                              libraries_sha256={"libcublas": digests["libcublas"]},
                              ptx_path="compiled.ptx", ptx_sha256=digests["compiled.ptx"], nvrtc_path="nvrtc.so",
                              nvrtc_identity="fixture:sha256=" + digests["nvrtc.so"])
                self._write_attempt(root, f"native-precision-performance-fp16-{report['seed']}-r1", report)
            args = ["native_precision_evidence.py", "--dtype", "fp16", "--power-profile", "performance",
                    "--output", "results/evidence.json"]
            # Only historical-profile loading is replaced by a tiny synthetic
            # registration. Discovery, statistics, bytes checks and writes run.
            with mock.patch.object(evidence, "ROOT", root), mock.patch("sys.argv", args), \
                    mock.patch("native_precision.frozen_cases", return_value=protocol["cases_by_dtype"]["fp16"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                evidence.main()
                saved = json.loads((root / "results/evidence.json").read_text())
                self.assertEqual(saved["qualified_shapes"], 1)
                self.assertEqual(len(saved["artifacts"]), 6)
                (root / "nvrtc.so").write_bytes(b"changed")
                args[-1] = "results/rejected.json"
                with self.assertRaises(ValueError):
                    evidence.main()
                self.assertFalse((root / "results/rejected.json").exists())


if __name__ == "__main__":
    unittest.main()
