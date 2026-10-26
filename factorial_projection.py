"""Stock/unlocked four-cell attribution for compressed- and half-projection raw reports.

CPU-only: validates two saved reports (or one stock report while the unlocked one is
pending) and recomputes every cell, A/A check and interval from raw paired rounds. Saved
gains, gate booleans, best-vendor names and A/A values are never trusted. Structural
confounds raise ValueError; measured numerical/noise failures are retained as UNQUALIFIED
rows. Per-case device-operation latency only: one stock and one unlocked process are not a
benchmark promotion or a whole-cohort acceptance. No filesystem writes.
"""
import math
import re
import statistics

import numpy as np

from compressed_campaign import _memory_clocks, summarize_methods

KINDS = ("compressed-projection", "half-projection")
UNLOCK_FLAGS = ("module_loaded", "unlock_boot_option_present", "unlock_success_logged",
                "post_flr_selectors_logged")
COMPRESSED_SOURCES = ("compressed_case.py", "compressed_campaign.py", "bench_compressed.py", "bench.py",
                      "projection_accuracy.py", "bf16_reference.py", "cublaslt_baseline.py",
                      "cublas_baseline.py", "gpu_monitor.py", "real_weights.py", "power_context.py",
                      "mtile_bf16.py", "mtile_kernels.cu")  # Revision 1 records the first 11.
HALF_SOURCES = ("bench_half.py", "half_projection.py", "half_kernels.cu", "kernels.cu", "kernel_lab.py",
                "native/driver.cpp", "native/build/libkernel_lab.so", "bench.py", "cublas_baseline.py",
                "bench_streaming.py", "gpu_monitor.py")
HALF_PRECISION_FLAGS = ("allow_tf32", "allow_fp16_accumulation", "allow_fp16_reduced_precision_reduction",
                        "allow_bf16_reduced_precision_reduction")
HALF_REQUIRED = ("output_dtype", "precision_flags", "operator_context", "observed_memory_clocks",
                 "environment", "compilation", "source_sha256", "libraries_sha256", "frozen_profile_sha256")
LIBRARIES = ("libcublas.so.12", "libcublasLt.so.12")
SHARED_BINDINGS = ("spec_sha256", "corpus_sha256", "vendor_profile_sha256", "context_protocol_sha256")
INVENTORY_IDENTITY = ("gpu", "kernel", "pci_address")
INVENTORY_OPTIONAL = ("os", "device_id", "driver_bound", "nvidia_runtime_parameters")
REPORT_FIELDS = ("output_dtype", "operator_context", "observed_memory_clocks", "seed", "seeds", "revision",
                 *SHARED_BINDINGS, "vendor_tuning_oracle_sha256", "choices", "dtype", "precision_flags",
                 "source_sha256", "libraries_sha256", "baselines", "rounds", "l2_bytes", "ring_target_bytes", "device",
                 "compute_capability", "candidate_space", "selection_seed", "torch", "cuda", "schema_version")
CASE_FIELDS = ("shape_mnk", "seed", "weight_tensor", "weight_sha256", "activation_sha256", "input_dtype",
               "output_dtype", "dtype", "layer", "role", "held_out", "selected", "selected_threads",
               "selected_vector", "copies", "calls_per_graph", "raw_ring_bytes", "packed_ring_bytes",
               "transposed_ring_bytes", "raw_ring_over_l2", "ring_bytes", "ring_over_l2", "storage_bytes",
               "storage_ratio", "vendor_routes", "vendor_pool", "execution_context", "implementation",
               "kernel_revision", "token_tile")
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
MIN_ROUNDS = 31
DRAWS = 2000
SCOPE = ("Per-case GPU operation latency, one stock and one unlocked process; no benchmark promotion, "
         "whole-cohort acceptance or full-model claim. Percentages are never added.")


def _hex(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _text(value):
    return isinstance(value, str) and bool(value)


def _saved(baseline, value):
    return None if baseline is None or value is None else 100 * (1 - value / baseline)


def _hash_tree(value, label, key=""):
    """Every *sha256 field must be a 64-hex digest; returns whether a PTX hash is present."""
    found = key == "ptx_sha256"
    if isinstance(value, dict):
        if key.endswith("sha256") and (key not in ("source_sha256", "libraries_sha256", "builder_source_sha256")
                or not value or not all(_hex(digest) for digest in value.values())):
            raise ValueError(f"{label}: invalid nested {key} digest map")
        for child, item in value.items():
            found |= _hash_tree(item, label, str(child))
    elif key.endswith("sha256") and not _hex(value):
        raise ValueError(f"{label}: {key} is not a valid SHA-256")
    return found


def _inventory(inventory, unlocked, label):
    if not isinstance(inventory, dict) or not all(_text(inventory.get(key)) for key in INVENTORY_IDENTITY):
        raise ValueError(f"{label}: inventory lacks GPU/kernel/PCI identity")
    if not UUID.search(inventory["gpu"]):
        raise ValueError(f"{label}: inventory GPU identity lacks a GPU UUID")
    for flag in UNLOCK_FLAGS:
        if inventory.get(flag) is not unlocked:
            raise ValueError(f"{label}: report requires inventory {flag}={unlocked}; "
                             "unlock state must be evidenced, never inferred")


def _compressed_report(report, unlocked, label):
    revision = report.get("revision")
    if revision not in (1, 2) or report.get("output_dtype") not in ("bf16", "fp32"):
        raise ValueError(f"{label}: unknown compressed revision or output contract")
    for field in (*SHARED_BINDINGS, "profile_sha256"):
        if not _hex(report.get(field)):
            raise ValueError(f"{label}: missing or invalid {field}")
    identity = report.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{label}: missing build/runtime identity")
    if identity.get("allow_tf32") is not False or identity.get("reduced_precision_reduction") is not False:
        raise ValueError(f"{label}: TF32/reduced-precision reduction must be explicitly disabled")
    sources = identity.get("source_sha256")
    required = COMPRESSED_SOURCES[:11 if revision == 1 else 13]
    if (not isinstance(sources, dict) or not set(required) <= set(sources)
            or not all(_hex(value) for value in sources.values())):
        raise ValueError(f"{label}: incomplete or invalid source fingerprints")
    libraries = identity.get("libraries_sha256")
    if (not isinstance(libraries, dict) or not set(LIBRARIES) <= set(libraries)
            or not all(_hex(value) for value in libraries.values())):
        raise ValueError(f"{label}: incomplete or invalid cuBLAS library fingerprints")
    implementation = identity.get("implementation")
    if not isinstance(implementation, dict) or not _hash_tree(implementation, label):
        raise ValueError(f"{label}: implementation identity lacks a PTX hash")
    for key in ("torch", "cuda", "cublas_version", "cublaslt_version"):
        if identity.get(key) in (None, ""):
            raise ValueError(f"{label}: identity lacks {key}")
    inventory = report.get("inventory")
    _inventory(inventory, unlocked, label)
    if identity.get("gpu") != inventory["gpu"] or identity.get("kernel") != inventory["kernel"]:
        raise ValueError(f"{label}: identity GPU/kernel differs from inventory")
    if identity.get("unlock_evidence") != {flag: inventory[flag] for flag in UNLOCK_FLAGS}:
        raise ValueError(f"{label}: identity unlock evidence disagrees with inventory")
    if not isinstance(report.get("choices"), dict) or not report["choices"]:
        raise ValueError(f"{label}: missing frozen configuration choices")
    return report["profile_sha256"]


def _half_report(report, unlocked, label):
    missing = [field for field in HALF_REQUIRED if field not in report]
    if missing:
        raise ValueError(f"{label}: half-projection report lacks causal-attribution fields {missing}. "
                         "Current bench_half output is incomplete for attribution; a stronger producer "
                         "must record them. They are not inferred.")
    if report.get("dtype") not in ("fp16", "bf16") or report["output_dtype"] != "fp32":
        raise ValueError(f"{label}: half projection needs FP16/BF16 input and FP32 output")
    flags = report["precision_flags"]
    if (not isinstance(flags, dict) or not set(HALF_PRECISION_FLAGS) <= set(flags)
            or not all(value is False for value in flags.values())):
        raise ValueError(f"{label}: all four precision flags must be explicitly False")
    environment, compilation = report["environment"], report["compilation"]
    if not isinstance(environment, dict) or not isinstance(compilation, dict):
        raise ValueError(f"{label}: missing runtime environment or compilation metadata")
    for key in ("torch", "cuda", "cublas_version"):
        if environment.get(key) in (None, ""):
            raise ValueError(f"{label}: environment lacks {key}")
    if (not _hex(compilation.get("cache_key")) or not _hex(compilation.get("ptx_sha256"))
            or environment.get("compilation_cache_key") != compilation["cache_key"]):
        raise ValueError(f"{label}: compilation cache key/PTX hash missing or inconsistent")
    sources = report["source_sha256"]
    if not isinstance(sources, dict) or not all(_hex(sources.get(name)) for name in HALF_SOURCES):
        raise ValueError(f"{label}: incomplete half source/build fingerprints")
    libraries = report["libraries_sha256"]
    if (not isinstance(libraries, dict) or not set(LIBRARIES) <= set(libraries)
            or not all(_hex(value) for value in libraries.values())):
        raise ValueError(f"{label}: incomplete half cuBLAS library binary fingerprints")
    if not _hex(report["frozen_profile_sha256"]):
        raise ValueError(f"{label}: invalid frozen_profile_sha256")
    for field in SHARED_BINDINGS:
        if field in report and not _hex(report[field]):
            raise ValueError(f"{label}: invalid {field}")
    _inventory(environment.get("inventory"), unlocked, label)
    return report["frozen_profile_sha256"]


def _method(name, method, expected_elements, label):
    if not isinstance(method, dict):
        raise ValueError(f"{label}: {name} is not a measured method")
    samples = method.get("samples_us")
    if not isinstance(samples, list) or len(samples) < MIN_ROUNDS or not all(_positive(v) for v in samples):
        raise ValueError(f"{label}: {name} needs at least {MIN_ROUNDS} positive finite numeric rounds")
    median, recomputed = method.get("median_us"), statistics.median(samples)
    # Producers divide median and rounds separately; even counts may differ by rounding only.
    exact = median == recomputed if len(samples) % 2 else (
        _positive(median) and math.isclose(median, recomputed, rel_tol=1e-12))
    if not _positive(median) or not exact:
        raise ValueError(f"{label}: {name} median_us does not equal the median of its raw rounds")
    numerics = method.get("numerics")
    if not isinstance(numerics, dict) or not isinstance(numerics.get("passed"), bool):
        raise ValueError(f"{label}: {name} lacks a numerical verdict")
    if not all(_count(numerics.get(key)) and numerics[key] >= 0
               for key in ("elements_checked", "nonfinite", "violations")):
        raise ValueError(f"{label}: {name} lacks numerical coverage counts")
    if numerics["elements_checked"] != expected_elements:
        raise ValueError(f"{label}: {name} checked {numerics['elements_checked']} outputs, "
                         f"expected copies*M*N={expected_elements}")
    if numerics["passed"] and (numerics["nonfinite"] or numerics["violations"]):
        raise ValueError(f"{label}: {name} numerical verdict contradicts its violation counts")
    return len(samples)


def _routes(report, case, kind, label):
    if kind == "half-projection" and "baselines" in report:
        routes = (report["baselines"] or {}).get("equivalent_vendor_routes")
        if "vendor_routes" in case and case["vendor_routes"] != routes:
            raise ValueError(f"{label}: case vendor routes differ from declared equivalent routes")
    else:
        routes = case.get("vendor_routes")
    if (not isinstance(routes, list) or not routes or len(set(routes)) != len(routes)
            or not all(_text(r) and not r.endswith("_aa") for r in routes)):
        raise ValueError(f"{label}: invalid declared vendor routes")
    pool = case.get("vendor_pool")
    if (not isinstance(pool, dict) or set(pool) != set(routes)
            or not all(isinstance(record, dict) and record for record in pool.values())):
        raise ValueError(f"{label}: every declared vendor route needs a frozen algorithm record")
    return list(routes)


def _case(report, case, kind, method, state):
    shape = case.get("shape_mnk")
    if not (isinstance(shape, list) and len(shape) == 3 and all(_count(v) and v > 0 for v in shape)):
        raise ValueError(f"{state}: invalid shape_mnk {shape!r}")
    m, n, k = shape
    tensor = case.get("weight_tensor")
    label = f"{state} {tensor or 'unnamed weight'} {m}x{n}x{k} seed {case.get('seed')}"
    seeds = [report["seed"]] if "seed" in report else report.get("seeds")
    if not _count(case.get("seed")) or not isinstance(seeds, list) or case["seed"] not in seeds:
        raise ValueError(f"{label}: case seed differs from its process")
    for field in ("weight_sha256", "activation_sha256"):
        if not _hex(case.get(field)):
            raise ValueError(f"{label}: missing exact {field}; seed equality is not measured input equality")
    if (kind == "compressed-projection" or tensor is not None) and not _text(tensor):
        raise ValueError(f"{label}: invalid weight_tensor")
    if kind == "compressed-projection":
        if case.get("input_dtype") != "bf16" or case.get("output_dtype") != report["output_dtype"]:
            raise ValueError(f"{label}: input/output contract differs from its report")
    elif (case.get("dtype") != report["dtype"] or case.get("input_dtype") != report["dtype"]
          or case.get("output_dtype") != "fp32" or case.get("held_out") is not True):
        raise ValueError(f"{label}: half input/output contract or held-out status differs")
    if case.get("execution_context") != report["operator_context"] or case.get("operating_context_stable") is not True:
        raise ValueError(f"{label}: graph power/AC context unstable or differs from the report")
    if not isinstance(case.get("gpu_telemetry"), list):
        raise ValueError(f"{label}: missing GPU telemetry")
    clocks = _memory_clocks([case])
    if not clocks or not set(clocks) <= set(report["observed_memory_clocks"]):
        raise ValueError(f"{label}: measurement-phase memory clocks disagree with the report clock regime")
    copies, calls = case.get("copies"), case.get("calls_per_graph")
    if not (_count(copies) and copies > 0 and _count(calls) and calls >= copies and calls % copies == 0):
        raise ValueError(f"{label}: invalid ring copy/call counts")
    ring = copies * n * k * 2
    if kind == "compressed-projection":
        if case.get("raw_ring_bytes") != ring or not (_count(case.get("storage_bytes")) and case["storage_bytes"] > 0):
            raise ValueError(f"{label}: physical raw ring or compressed storage size inconsistent")
        selected = case.get("selected")
        from compressed_case import launch_config
        launch_config(selected)
        if case.get("role") == "head-transfer-diagnostic":
            transferred = {item.get("selected") for item in report["cases"]
                           if item.get("role") == "held-out"
                           and item.get("weight_tensor", "").endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight"))
                           and item.get("shape_mnk", [None])[-1] == k}
            if (report.get("head_diagnostic", {}).get("status") != "included"
                    or len(transferred) != 1 or selected not in transferred):
                raise ValueError(f"{label}: head does not replay the unique frozen gate/up configuration")
        elif report["choices"].get(f"{n},{k}") != selected:
            raise ValueError(f"{label}: case configuration is not the frozen choice")
        custom, extra = selected, {"raw_" + selected.removeprefix("coded_")}
        parity = case.get("raw_template_parity_by_config")
        evidence = {"original_bits_preserved": case.get("original_bits_preserved"),
                    "raw_template_bitwise": case.get("raw_template_bitwise"),
                    "raw_template_parity": parity.get(selected) if isinstance(parity, dict) else None}
    else:
        rings = case.get("ring_bytes")
        if not isinstance(rings, dict) or rings.get("weights") != ring or rings.get("transposed_weights") != ring:
            raise ValueError(f"{label}: physical weight ring sizes inconsistent")
        threads, vector = case.get("selected_threads"), case.get("selected_vector")
        if (not (_count(threads) and _count(vector)) or threads not in (32,64,128,256) or vector not in (2,8)
                or k % vector or case.get("selected") != f"t{threads}_v{vector}"):
            raise ValueError(f"{label}: frozen threads/vector disagree with the selected name")
        custom, extra = "half_" + case["selected"], set()
        evidence = {"bitwise_layout_check": case.get("bitwise_layout_check")}
    if method != "auto" and method != custom:
        raise ValueError(f"{label}: method {method!r} is not the frozen selection {custom!r}; no reselection")
    if not all(isinstance(value, bool) for value in evidence.values()):
        raise ValueError(f"{label}: missing weight-bit/parity/layout evidence")
    routes = _routes(report, case, kind, label)
    if custom in routes or extra & set(routes):
        raise ValueError(f"{label}: custom method cannot also be a vendor route")
    methods = case.get("methods")
    if not isinstance(methods, dict):
        raise ValueError(f"{label}: missing measured methods")
    missing = ({custom, *routes, *(route + "_aa" for route in routes)} | extra) - set(methods)
    if missing:
        raise ValueError(f"{label}: missing methods {sorted(missing)} (each vendor route needs its own _aa twin)")
    counts = {_method(name, methods[name], copies * m * n, label) for name in sorted(methods)}
    if len(counts) != 1:
        raise ValueError(f"{label}: methods have different paired round counts")
    summary = summarize_methods(methods, routes, custom)
    failures = []
    if summary["best_vendor"] is None:
        failures.append(f"{state}: no numerically eligible vendor baseline")
    elif not summary["aa_passed"]:
        failures.append(f"{state}: A/A {summary['aa_time_saved_pct']:+.3f}% for {summary['best_vendor']} exceeds 1%")
    if not methods[custom]["numerics"]["passed"]:
        failures.append(f"{state}: custom {custom} numerics failed")
    failures += [f"{state}: {name} is False" for name, value in evidence.items() if not value]
    excluded = [f"{state}: {route} excluded from best-vendor selection (own or A/A-twin numerics failed)"
                for route in routes if route not in summary["eligible_vendors"]]
    key = (case["seed"], tensor or f"sha256:{case['weight_sha256']}:{case['activation_sha256']}", tuple(shape))
    return key, {"case": case, "custom": custom, "routes": routes, "rounds": counts.pop(),
                 "summary": summary, "failures": failures, "excluded": excluded}


def _state(report, unlocked, method):
    label = "unlocked" if unlocked else "stock"
    if not isinstance(report, dict) or report.get("kind") not in KINDS:
        raise ValueError(f"{label}: only compressed-projection/half-projection reports are supported")
    if report.get("status") != "complete" or report.get("mode") != "validate":
        raise ValueError(f"{label}: need a complete held-out validation report")
    if not (_count(report.get("pid")) and report["pid"] > 0) or not _text(report.get("run_id")):
        raise ValueError(f"{label}: missing actual process id or run id")
    kind = report["kind"]
    profile = (_compressed_report if kind == "compressed-projection" else _half_report)(report, unlocked, label)
    context = report.get("operator_context")
    if (not isinstance(context, dict) or not _text(context.get("system76_power_profile"))
            or not isinstance(context.get("ac_online"), bool) or context.get("scope") != "graph"):
        raise ValueError(f"{label}: missing graph power-profile/AC operator context")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{label}: no measured cases")
    clocks = report.get("observed_memory_clocks")
    if not isinstance(clocks, list) or not clocks or clocks != _memory_clocks(cases):
        raise ValueError(f"{label}: observed memory clocks missing or not recomputable from telemetry")
    if kind == "compressed-projection":
        if any(case.get("role") not in ("held-out", "head-transfer-diagnostic") for case in cases):
            raise ValueError(f"{label}: validation report contains non-held-out cases")
    index = {}
    for case in cases:
        key, value = _case(report, case, kind, method, label)
        if key in index:
            raise ValueError(f"{label}: duplicate case {key}")
        index[key] = value
    if not index:
        raise ValueError(f"{label}: no held-out cases")
    return {"kind": kind, "report": report, "profile": profile, "cases": index}


def _without(mapping, key):
    return {k: v for k, v in (mapping or {}).items() if k != key}


def _match(before, after):
    s, u = before["report"], after["report"]
    if s["kind"] != u["kind"]:
        raise ValueError("Stock and unlocked report kinds differ")
    if s["pid"] == u["pid"] or s["run_id"] == u["run_id"]:
        raise ValueError("Distinct stock/unlocked process ids and run ids required")
    if bool(s.get("synthetic_fixture")) != bool(u.get("synthetic_fixture")):
        raise ValueError("Synthetic fixtures cannot be paired with measured reports")
    for field in REPORT_FIELDS:
        if s.get(field) != u.get(field):
            raise ValueError(f"Stock/unlocked contracts differ: {field}")
    if s["kind"] == "compressed-projection":
        inventories = s["inventory"], u["inventory"]
        if _without(s["identity"], "unlock_evidence") != _without(u["identity"], "unlock_evidence"):
            raise ValueError("Source/build/binary/library/runtime/GPU identity differs")
    else:
        inventories = s["environment"]["inventory"], u["environment"]["inventory"]
        if _without(s["environment"], "inventory") != _without(u["environment"], "inventory"):
            raise ValueError("Runtime environment or compilation key differs")
        if _without(s["compilation"], "cache_hit") != _without(u["compilation"], "cache_hit"):
            raise ValueError("Binary compilation metadata differs")
    for key in INVENTORY_IDENTITY + INVENTORY_OPTIONAL:
        if inventories[0].get(key) != inventories[1].get(key):
            raise ValueError(f"GPU/driver/kernel inventory differs: {key}")
    origin = u.get("frozen_choice_origin_sha256")
    if origin is not None and origin != before["profile"]:
        raise ValueError("Unlocked frozen-choice origin is not the stock profile")
    if before["profile"] != after["profile"] and origin != before["profile"]:
        raise ValueError("State-specific profiles differ without explicit frozen-transfer provenance "
                         "(frozen_choice_origin_sha256 == stock profile SHA); equal choices alone do not prove it")
    if not before["cases"] or before["cases"].keys() != after["cases"].keys():
        raise ValueError("Stock and unlocked case sets differ")
    for key, left in before["cases"].items():
        right = after["cases"][key]
        for field in CASE_FIELDS:
            if left["case"].get(field) != right["case"].get(field):
                raise ValueError(f"Case {key} differs between states: {field}")
        if (left["routes"] != right["routes"] or left["rounds"] != right["rounds"]
                or set(left["case"]["methods"]) != set(right["case"]["methods"])):
            raise ValueError(f"Case {key} vendor routes, methods or round counts differ")


def _cross_intervals(stock, unlocked, eligible, fixed, custom, draws=DRAWS):
    """Independent stock/unlocked round resampling; vendor/custom rounds stay paired within a state."""
    if not eligible:
        return dict.fromkeys(("unlock_fixed_vendor", "combined", "interaction"))
    left = {name: np.asarray(stock[name]["samples_us"], dtype=np.float64) for name in (*eligible, custom)}
    right = {name: np.asarray(unlocked[name]["samples_us"], dtype=np.float64)
             for name in ((fixed, custom) if fixed else (custom,))}
    rng = np.random.default_rng(1009)
    si = rng.integers(0, len(left[custom]), (draws, len(left[custom])))
    ui = rng.integers(0, len(right[custom]), (draws, len(right[custom])))
    cs, cu = np.median(left[custom][si], axis=1), np.median(right[custom][ui], axis=1)
    best = np.min([np.median(left[name][si], axis=1) for name in eligible], axis=0)
    quantiles = lambda values: np.quantile(values, [.025, .975]).tolist()
    result = {"combined": quantiles(100 * (1 - cu / best)), "unlock_fixed_vendor": None, "interaction": None}
    if fixed:
        bs, bu = np.median(left[fixed][si], axis=1), np.median(right[fixed][ui], axis=1)
        result.update(unlock_fixed_vendor=quantiles(100 * (1 - bu / bs)),
                      interaction=quantiles(100 * (1 - (cu / cs) / (bu / bs))))
    return result


def _row(kind, key, before, after, synthetic):
    case, custom, summary = before["case"], before["custom"], before["summary"]
    failures, excluded = list(before["failures"]), list(before["excluded"])
    vendor, bs = summary["best_vendor"], summary["best_vendor_us"]
    cs = case["methods"][custom]["median_us"]
    interval = summary["ci95_pct"]
    row = {"kind": kind, "seed": key[0], "weight_tensor": case.get("weight_tensor"), "shape_mnk": list(key[2]),
           "role": case.get("role"), "layer": case.get("layer"),
           "input_dtype": case["input_dtype"], "output_dtype": case["output_dtype"], "custom_method": custom,
           "stock_best_vendor": vendor, "unlock_reference_vendor": vendor,
           "stock_vendor_us": bs, "stock_custom_us": cs, "our_gain_stock_pct": _saved(bs, cs),
           "stock_aa_time_saved_pct": summary["aa_time_saved_pct"],
           "stock_10pct_target": bool(not before["failures"] and summary["passes_10pct_gate"]
                                      and interval is not None and interval[0] > 10),
           "synthetic_fixture": synthetic,
           "scope": SCOPE + (" SYNTHETIC FIXTURE: not a performance measurement." if synthetic else "")}
    cells = dict.fromkeys(("unlocked_best_vendor", "unlocked_vendor_us", "unlocked_custom_us",
                           "unlocked_fixed_vendor_us", "our_gain_unlocked_pct", "unlock_gain_vendor_pct",
                           "combined_gain_pct", "interaction_extra_latency_reduction_pct",
                           "unlocked_aa_time_saved_pct", "unlocked_fixed_vendor_aa_time_saved_pct"))
    intervals = {"our_stock": interval, "our_unlocked": None, "unlock_fixed_vendor": None,
                 "combined": None, "interaction": None}
    if after is None:
        failures.append("unlocked: no actual unlocked-hardware report; comparison pending")
        row.update(cells, ci95_pct=intervals, comparison_status="pending-unlocked", attribution_qualified=False,
                   failure_reasons=failures, excluded_vendor_reasons=excluded)
        return row
    methods, later = after["case"]["methods"], after["summary"]
    failures += after["failures"]
    excluded += after["excluded"]
    bu, cu = later["best_vendor_us"], methods[custom]["median_us"]
    fixed = vendor if vendor in later["eligible_vendors"] else None
    bu_fixed = fixed_aa = None
    if vendor is not None and fixed is None:
        failures.append(f"unlocked: stock-selected route {vendor} is numerically ineligible; "
                        "fixed-route effect unavailable (routes are never switched)")
    if fixed:
        bu_fixed = methods[fixed]["median_us"]
        fixed_aa = 100 * (bu_fixed - methods[fixed + "_aa"]["median_us"]) / bu_fixed
        if fixed != later["best_vendor"] and abs(fixed_aa) > 1:
            failures.append(f"unlocked: A/A {fixed_aa:+.3f}% for fixed route {fixed} exceeds 1%")
    interaction = (None if bu_fixed is None or bs is None
                   else 100 * (1 - (cu / cs) / (bu_fixed / bs)))
    cells.update(unlocked_best_vendor=later["best_vendor"], unlocked_vendor_us=bu, unlocked_custom_us=cu,
                 unlocked_fixed_vendor_us=bu_fixed, our_gain_unlocked_pct=_saved(bu, cu),
                 unlock_gain_vendor_pct=_saved(bs, bu_fixed), combined_gain_pct=_saved(bs, cu),
                 interaction_extra_latency_reduction_pct=interaction,
                 unlocked_aa_time_saved_pct=later["aa_time_saved_pct"],
                 unlocked_fixed_vendor_aa_time_saved_pct=fixed_aa)
    intervals.update(our_unlocked=later["ci95_pct"],
                     **_cross_intervals(case["methods"], methods, summary["eligible_vendors"], fixed, custom))
    row.update(cells, ci95_pct=intervals, comparison_status="compared" if not failures else "unqualified",
               attribution_qualified=not failures, failure_reasons=failures, excluded_vendor_reasons=excluded)
    return row


def compare_projection(stock, unlocked, method="auto"):
    """Matched stock/unlocked four cells plus fixed-route unlock effect and interaction, per case."""
    try:
        before, after = _state(stock, False, method), _state(unlocked, True, method)
        _match(before, after)
        synthetic = bool(stock.get("synthetic_fixture"))
        return [_row(before["kind"], key, value, after["cases"][key], synthetic)
                for key, value in before["cases"].items()]
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed projection report: {error!r}") from error


def pending_projection(stock, method="auto"):
    """Validated stock cells only; every unlocked-dependent value stays None until real data exist."""
    try:
        before = _state(stock, False, method)
        synthetic = bool(stock.get("synthetic_fixture"))
        return [_row(before["kind"], key, value, None, synthetic) for key, value in before["cases"].items()]
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed projection report: {error!r}") from error
