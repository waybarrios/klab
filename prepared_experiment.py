"""Full-grid prepared-candidate EXPERIMENT runner; not production routing.

Binds per-shape benchmark candidates from replayed three-process kernel evidence
and measures them with the unchanged compressed_case.measure_case: graph latency
against the full frozen vendor pool, graph implementation overhead against the
identical direct call, and synchronized eager allocation against direct native
Torch. Only a shape's own replayed kernel_gate_passed can produce a coded
candidate; unseen or failed shapes use the prepared vendor and never pack.
Every record keeps custom_win_validated=False and eager_promoted=False until a
later independent review; dynamic_projection.prepare_projection stays
vendor-only. Numerical checks, rings, telemetry and paired rounds come from the
existing helpers. Importing this file imports neither Torch nor CUDA; main owns
GpuLock.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HELD_OUT, HEAD, CONTROL = "held-out", "head-transfer-diagnostic", "synthetic-control"
BUILDERS = ("compressed_evidence_io.py", "compressed_dispatch_evidence.py")
SOURCES = ("prepared_experiment.py", "prepared_kernels.py", "prepared_bf16.py", "projection_policy.py",
           "bench_dispatch.py", "compressed_bf16.py", "compressed_kernels.cu", "mtile_bf16.py",
           "mtile_kernels.cu", "cuda_module.py", "experiment_control.py",
           "experiments/prepared_projection_scope.json", *BUILDERS)
CONTROL_CONFIG = "coded_t64_v8"  # Predeclared forced diagnostic on controls; never accepted or tuned.
CONTROL_SCOPE = "synthetic wrapper-overhead control, not full vendor-pool superiority"
DIAGNOSTIC_SCOPE = ("Base measure_case storage/packing/packed-ring fields describe its forced raw/coded "
                    "diagnostic arms; selected_candidate describes the measured prepared route")
NO_PROMOTION = {"custom_win_validated": False, "eager_promoted": False}


def _vendor_config(output_dtype):
    return "torch_mm" if output_dtype == "bf16" else "torch_mm_out_fp32"


def candidate_plan(evidence, shape, *, output_dtype):
    """Benchmark candidate for one M,N,K from that shape's own replayed kernel gate."""
    from compressed_case import launch_config
    contract = evidence.get("contract", {})
    if output_dtype not in ("bf16", "fp32"):
        raise ValueError("Expected BF16 or FP32 output")
    if evidence.get("kind") != "compressed-dispatch-evidence" or evidence.get("status") != "complete":
        raise ValueError("Need complete compressed-dispatch-evidence")
    if evidence.get("execution_mode") != "graph" or evidence.get("eager_promoted") is not False:
        raise ValueError("Kernel evidence must be graph-scope and unpromoted for eager")
    if ((contract.get("input_dtype"), contract.get("weight_dtype"), contract.get("accumulation_dtype"))
            != ("bf16", "bf16", "fp32")
            or any(contract.get(flag) is not False
                   for flag in ("lossy_weights", "allow_tf32", "reduced_precision_reduction"))):
        raise ValueError("Evidence is not the exact native-BF16, FP32-accumulation contract")
    if contract.get("output_dtype") != output_dtype or evidence.get("bindings", {}).get("output_dtype") != output_dtype:
        raise ValueError("Evidence output contract differs from the requested output")
    shape = [int(value) for value in shape]
    entry = evidence.get("shapes", {}).get(",".join(map(str, shape)))
    plan = {"shape_mnk": shape, "output_dtype": output_dtype, "evidence_shape": entry,
            "scope": "benchmark candidate only; not production routing", **NO_PROMOTION,
            "dispatch_overhead_validated": False}
    # Only this shape's own gate counts; kernel_milestone_passed never promotes another shape.
    if entry is not None and entry.get("shape_mnk") == shape and entry.get("kernel_gate_passed") is True:
        threads, vector = launch_config(entry["selected"])
        return {**plan, "route": "coded", "config": entry["selected"], "threads": threads, "vector": vector,
                "requires_packing": True, "reason": "This shape passed every replayed kernel gate"}
    reason = "Shape absent from kernel evidence" if entry is None else "Shape failed its kernel gate"
    return {**plan, "route": "vendor", "config": _vendor_config(output_dtype), "threads": None,
            "vector": None, "requires_packing": False, "reason": reason}


def registered_cases(spec, corpus):
    """Exact registered grid: held-out layers x projections x M, optional head, controls."""
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    ms = list(spec["m_values"]) + list(spec["baseline_diagnostic_m"])
    cases = []
    for layer in spec["held_out_layers"]:
        for suffix in spec["projection_weights"]:
            name = f"model.layers.{layer}.{suffix}"
            if name not in entries:
                raise ValueError(f"Missing registered tensor {name}")
            cases += [{"weight_tensor": name, "shape_mnk": [m, *entries[name]["shape"]],
                       "layer": layer, "role": HELD_OUT} for m in ms]
    head = spec.get("head", {}).get("tensor")
    if head in entries:
        cases += [{"weight_tensor": head, "shape_mnk": [m, *entries[head]["shape"]],
                   "layer": None, "role": HEAD} for m in ms]
    for m, n, k in spec["synthetic_controls_mnk"]:
        cases.append({"weight_tensor": f"synthetic-control.m{m}.n{n}.k{k}", "shape_mnk": [m, n, k],
                      "layer": None, "role": CONTROL})
    identities = [(case["weight_tensor"], tuple(case["shape_mnk"])) for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Registered grid contains duplicate case identities")
    return cases


def summarize_graph(methods, vendor_routes, *, route):
    """Graph statistics from raw rounds: full frozen pool, and prepared vs identical direct call."""
    from bench_dispatch import summarize_dispatch
    from compressed_campaign import summarize_methods
    arms = ("prepared", "prepared_direct", "prepared_direct_aa")
    overhead = summarize_dispatch({"vendor": methods["prepared_direct"], "vendor_aa": methods["prepared_direct_aa"],
                                   "prepared": methods["prepared"]})
    numerics = all(methods[arm]["numerics"]["passed"] for arm in arms)
    overhead.update(timing_gate_passed=overhead["overhead_gate_passed"], all_arms_numerics_passed=numerics,
                    overhead_gate_passed=bool(overhead["overhead_gate_passed"] and numerics))
    direct = ("unchanged mtile_bf16.fused_linear on the same uploaded buffers/T/V" if route == "coded"
              else "native torch.mm with the prebound weight.T")
    return {"route": route, "versus_vendor": summarize_methods(methods, vendor_routes, "prepared"),
            "versus_vendor_scope": "full frozen vendor pool, each with its own A/A twin",
            "implementation_overhead": overhead, "direct_baseline": direct, **NO_PROMOTION}


# ---------------------------------------------------------------- artifact integrity


def _sha(path):
    from compressed_campaign import _sha as digest
    return digest(path)


def _load(reference):
    from experiment_control import project_path
    path = project_path(ROOT, reference["path"])
    data = path.read_bytes()
    if len(reference.get("sha256", "")) != 64 or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"Input bytes differ from the evidence reference: {reference['path']}")
    return path, json.loads(data)


def validate_training(profile, training, spec, corpus):
    """A matching file digest must still contain the complete registered training."""
    from compressed_campaign import assert_complete, freeze_choices
    from compressed_dispatch_evidence import BINDING_FIELDS
    bindings = profile["bindings"]
    if (training.get("status") != "complete" or training.get("mode") != "tune"
            or training.get("seed") != spec["tuning_seed"]
            or training.get("pid") != profile["pid"] or training.get("run_id") != profile["run_id"]
            or any(training.get(key) != bindings[key] for key in BINDING_FIELDS)):
        raise ValueError("Training content, identity or bindings differ from the frozen profile")
    assert_complete(training, spec, corpus)
    choices = freeze_choices([case for case in training["cases"] if case["role"] == "training"],
                             spec, output_dtype=bindings["output_dtype"])
    if choices != profile["choices"] or training.get("choices") != choices:
        raise ValueError("Frozen choices do not reproduce from complete training rounds")


def check_frozen_sources(report):
    """Reject mid-run edits, including evidence inputs and the scope protocol."""
    from experiment_control import project_path
    for name, digest in report["source_sha256"].items():
        if _sha(project_path(ROOT, name)) != digest:
            raise ValueError(f"Measured source changed during the campaign: {name}")
    for name, digest in report.get("bindings", {}).get("identity", {}).get("source_sha256", {}).items():
        if _sha(project_path(ROOT, name)) != digest:
            raise ValueError(f"Kernel source changed during the campaign: {name}")
    if "evidence" in report:
        _load(report["evidence"])
        for key, reference in report["evidence_inputs"].items():
            for item in reference if key == "reports" else [reference]:
                _load(item)


def verify_artifacts(evidence_path, evidence, *, seed, rounds):
    """CPU-only binding of the aggregate to its exact inputs, builders and execution sources."""
    from compressed_dispatch_evidence import build_evidence
    inputs = evidence["inputs"]
    refs = [inputs[key] for key in ("spec", "corpus", "profile", "training")] + list(inputs["reports"])
    loaded = [_load(ref) for ref in refs]
    paths = [path for path, _ in loaded]
    if len(inputs["reports"]) != 3 or len(set(paths + [evidence_path])) != len(paths) + 1:
        raise ValueError("Evidence needs distinct spec/corpus/profile/training and three report files")
    (_, spec), (_, corpus), (_, profile), (training_path, training) = loaded[:4]
    reports = [data for _, data in loaded[4:]]
    if evidence["builder_source_sha256"] != {name: _sha(ROOT / name) for name in BUILDERS}:
        raise ValueError("Evidence builders changed since the aggregate was written")
    bindings = evidence["bindings"]
    if (bindings["spec_sha256"] != inputs["spec"]["sha256"] or bindings["corpus_sha256"] != inputs["corpus"]["sha256"]
            or evidence["profile_sha256"] != inputs["profile"]["sha256"] or bindings["revision"] != 2
            or bindings["context_protocol_sha256"] != _sha(ROOT / "experiments/consumer_ampere_execution_context.json")):
        raise ValueError("Evidence bindings differ from its spec/corpus/profile/context protocol")
    if (profile["training_report_sha256"] != inputs["training"]["sha256"]
            or Path(ROOT / profile["training_report"]).resolve() != training_path):
        raise ValueError("Training reference differs from the frozen profile")
    validate_training(profile, training, spec, corpus)
    from experiment_control import project_path
    for name, digest in bindings["identity"]["source_sha256"].items():
        if _sha(project_path(ROOT, name)) != digest:
            raise ValueError(f"Bound execution source changed: {name}")
    rebuilt = json.loads(json.dumps(build_evidence(reports, spec, corpus, profile,
                                                   profile_sha256=inputs["profile"]["sha256"])))
    mismatched = sorted(key for key, value in rebuilt.items() if evidence.get(key) != value)
    if mismatched:
        raise ValueError(f"Aggregate differs from rebuilt kernel evidence: {mismatched}")
    if seed not in spec["held_out_seeds"] or seed not in [p["seed"] for p in rebuilt["processes"]]:
        raise ValueError("Seed must be one of the replayed held-out processes")
    kernel = [(ref, data) for ref, data in zip(inputs["reports"], reports) if data["seed"] == seed]
    if len(kernel) != 1:
        raise ValueError("Need exactly one kernel validation report for this seed")
    (kernel_ref, kernel_report), = kernel
    if rounds < spec["minimum_rounds"] or any(len(method["samples_us"]) != rounds
                                              for case in kernel_report["cases"]
                                              for method in case["methods"].values()):
        raise ValueError("Rounds differ from the kernel validation process")
    return {"spec": spec, "corpus": corpus, "profile": profile, "rebuilt": rebuilt,
            "kernel_ref": kernel_ref, "kernel_report": kernel_report}


# ---------------------------------------------------------------- external arms


def _external(plan, source, rounds, max_ratio, state):
    """measure_case external_factory: three extra graph arms plus allocating eager, pre-monitor."""
    def factory(x, raw, output_dtype):
        import torch
        import compressed_bf16 as cb
        import mtile_bf16 as mt
        from bench_dispatch import _eager, current_context, summarize_dispatch
        from compressed_campaign import check_ring
        from compressed_case import _batch, _reference
        from prepared_bf16 import PreparedBF16Projection
        from prepared_kernels import CodedCandidate

        copies, n, k = raw.shape
        m = x.shape[0]
        dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float32
        if not torch.equal(raw[0].view(torch.uint16).cpu(), torch.from_numpy(source)):
            raise RuntimeError("Ring differs from the source checkpoint bits")
        started = time.perf_counter()
        info = {"planned_route": plan["route"], "route": plan["route"], "config": plan["config"],
                "packing_seconds": 0., "storage": {"format": "raw_bf16", "bytes": int(source.nbytes)},
                "codec_in_selected_call": False, **NO_PROMOTION}
        if plan["route"] == "coded":
            packing = time.perf_counter()
            encoded = cb.pack_bits(source)
            info["packing_seconds"] = time.perf_counter() - packing
            ratio = encoded.storage_bytes / source.nbytes
            info["storage"] = {"format": f"compressed_bf16_v{encoded.format_version}",
                               "bytes": encoded.storage_bytes, "raw_bf16_bytes": int(source.nbytes), "ratio": ratio}
            if ratio > max_ratio:
                info["compression_inspection"] = info["storage"]
                info.update(route="vendor", config=_vendor_config(output_dtype),
                            storage={"format": "raw_bf16", "bytes": int(source.nbytes)},
                            downgrade_reason=f"Actual storage ratio {ratio:.4f} exceeds {max_ratio}; no upload")
        weight_t = [raw[i].T for i in range(copies)]
        if output_dtype == "bf16":
            direct_torch = lambda i, out=None: torch.mm(x, weight_t[i], out=out)
        else:
            direct_torch = lambda i, out=None: torch.mm(x, weight_t[i], out_dtype=torch.float32, out=out)
        if info["route"] == "coded":
            packed = [cb.upload(encoded, x.device) for _ in range(copies)]
            for item in packed:
                if not torch.equal(cb.unpack_weights(item).view(torch.int16), raw[0].view(torch.int16)):
                    raise RuntimeError("Uploaded candidate buffers changed original checkpoint bits")
            owners = [CodedCandidate(item, tokens=m, threads=plan["threads"], vector=plan["vector"],
                                     output_dtype=output_dtype) for item in packed]
            direct = lambda i, out: lambda: mt.fused_linear(x, packed[i], threads=plan["threads"],
                                                            vector=plan["vector"], output_dtype=output_dtype, out=out)
            info.update(candidate=owners[0].describe(), packed_ring_bytes=sum(p.storage_bytes for p in packed))
            retained = owners + packed
        else:
            owners = [PreparedBF16Projection(raw[i], tokens=m, profile=None, output_dtype=dtype)
                      for i in range(copies)]
            direct = lambda i, out: lambda: direct_torch(i, out)
            info.update(candidate={"kind": "PreparedBF16Projection", "plan": owners[0].plan,
                                   "packed": owners[0].packed is not None})
            retained = owners + weight_t
        info.update(setup_seconds=time.perf_counter() - started,
                    setup_seconds_per_copy=[owner.setup_seconds for owner in owners])
        functions, outputs = {}, {}
        builders = (("prepared", lambda i, out: lambda: owners[i].linear_out(x, out=out)),
                    ("prepared_direct", direct), ("prepared_direct_aa", direct))
        for label, builder in builders:
            outputs[label] = out = torch.empty((copies, m, n), dtype=dtype, device=x.device)
            functions[label] = _batch([builder(i, out[i]) for i in range(copies)])

        # Synchronized eager allocation, before measure_case starts its GpuMonitor. i % copies
        # lets _eager's fixed eight warmup calls run on a one-copy ring; every arm pays it equally.
        # Allocation controls call public Torch directly, with no out=None or
        # extra direct_torch forwarding frame that could inflate the baseline.
        if output_dtype == "bf16":
            eager_functions = {"vendor": lambda i: torch.mm(x, weight_t[i % copies]),
                               "vendor_aa": lambda i: torch.mm(x, weight_t[i % copies])}
        else:
            eager_functions = {"vendor": lambda i: torch.mm(x, weight_t[i % copies], out_dtype=torch.float32),
                               "vendor_aa": lambda i: torch.mm(x, weight_t[i % copies], out_dtype=torch.float32)}
        eager_functions["prepared"] = lambda i: owners[i % copies].linear(x)
        torch.cuda.synchronize()
        before = current_context(scope="eager")
        eager = _eager(eager_functions, rounds=rounds, calls=copies)
        after = current_context(scope="eager")
        reference = _reference(x, raw[0])
        for name, function in eager_functions.items():
            allocated = torch.stack([function(i) for i in range(copies)])
            eager[name]["numerics"] = check_ring(reference, allocated, output_dtype=output_dtype)
            del allocated
        summary = summarize_dispatch(eager)
        numerics = all(method["numerics"]["passed"] for method in eager.values())
        summary.update(timing_gate_passed=summary["overhead_gate_passed"], all_arms_numerics_passed=numerics,
                       context_stable=before == after,
                       overhead_gate_passed=bool(summary["overhead_gate_passed"] and numerics and before == after),
                       **NO_PROMOTION)
        state.update(outputs=outputs, info=info, eager={
            "methods": eager, "summary": summary, "context_before": before, "context_after": after,
            "baseline": "direct native torch.mm allocation with prebound weight.T; not the full Lt pool",
            "candidate_call": "allocation-only candidate.linear(x)",
            "coverage": "allocating outputs of every copy checked after timing; no SMI, packing or fingerprint in calls"})
        return functions, outputs, retained, info
    return factory


def _control_manifest(controls, seed, run_id):
    """Deterministic BF16 control weights as raw-u16 files; synthetic storage, not checkpoint evidence."""
    import torch
    from experiment_control import atomic_write
    directory = ROOT / ".scratch/prepared-controls" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    tensors = []
    for case in controls:
        name, (_, n, k) = case["weight_tensor"], case["shape_mnk"]
        generator = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16))
        bits = torch.randn((n, k), dtype=torch.bfloat16, generator=generator).view(torch.uint16).numpy()
        data = bits.astype("<u2").tobytes()
        path = directory / f"{name}.u16"
        atomic_write(path, data)
        tensors.append({"name": name, "dtype": "BF16", "shape": [n, k], "bytes": len(data),
                        "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()})
    return {"kind": "synthetic-control-weights", "status": "complete", "tensors": tensors,
            "storage": "seeded synthetic BF16; never native-checkpoint or native-FP32 evidence"}


def _pct(value):
    return "n/a" if value is None else f"{value:+.3f}%"


def _project_environment():
    """Project-local compiler caches and temporary files; must run before Torch is imported."""
    for variable, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch"),
                                ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor"),
                                ("TRITON_CACHE_DIR", ".cache/triton")):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(ROOT / directory))


def run(evidence_path, output_path, *, seed=919, rounds=31):
    """Caller owns GpuLock. Returns the complete prepared-projection-campaign report."""
    from experiment_control import project_path, run_output, with_run_metadata, write_json
    evidence_path, output_path = project_path(ROOT, evidence_path), run_output(ROOT, output_path)
    if evidence_path.suffix != ".json" or output_path.exists():
        raise ValueError("Preserve prior results; use a new project-local JSON output")
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    bound = verify_artifacts(evidence_path, evidence, seed=seed, rounds=rounds)
    protocol_path = ROOT / "experiments/prepared_projection_scope.json"
    protocol = json.loads(protocol_path.read_text())
    if (seed != 919 or rounds != protocol["rounds"]
            or bound["rebuilt"]["bindings"]["output_dtype"] not in protocol["output_contracts"]
            or bound["spec"]["m_values"] + bound["spec"]["baseline_diagnostic_m"] != protocol["tokens"]
            or bound["spec"]["synthetic_controls_mnk"] != protocol["synthetic_controls_mnk"]
            or protocol["kernel_search_allowed"] is not False):
        raise ValueError("Requested scope differs from the registered fixed-choice experiment")
    referenced = {project_path(ROOT, ref["path"]) for key, ref in evidence["inputs"].items() if key != "reports"}
    referenced |= {project_path(ROOT, ref["path"]) for ref in evidence["inputs"]["reports"]}
    if output_path in referenced | {evidence_path}:
        raise ValueError("Output must not alias evidence or its inputs")
    spec, corpus, rebuilt = bound["spec"], bound["corpus"], bound["rebuilt"]
    output_dtype = rebuilt["bindings"]["output_dtype"]
    entries = {entry["name"]: entry for entry in corpus["tensors"]}
    grid = registered_cases(spec, corpus)
    kernel_cases = {(case["weight_tensor"], tuple(case["shape_mnk"])): case for case in bound["kernel_report"]["cases"]}
    for item in grid:
        if item["role"] != CONTROL and (item["weight_tensor"], tuple(item["shape_mnk"])) not in kernel_cases:
            raise ValueError(f"No kernel validation case for {item['weight_tensor']} {item['shape_mnk']}")

    _project_environment()
    import numpy as np
    import torch
    from bench_compressed import _power_context
    from compressed_campaign import _identity, _memory_clocks
    from compressed_case import measure_case
    from compressed_dispatch_evidence import _head_config
    from cublas_baseline import CuBlas
    from cublaslt_baseline import LtWorkspace
    from gpu_unlock_status import inspect
    from power_context import execution_context
    from real_weights import read_tensor

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    power = _power_context()
    context = execution_context(power)
    inventory = inspect()
    vendor = CuBlas()
    workspace = LtWorkspace(0)
    try:
        identity = json.loads(json.dumps(_identity(vendor, inventory, revision=2)))
        if identity != rebuilt["bindings"]["identity"] or context != rebuilt["bindings"]["operator_context"]:
            raise ValueError("Current runtime identity or graph operating context differs from the kernel evidence")
        run_id = str(uuid.uuid4())
        controls = _control_manifest([c for c in grid if c["role"] == CONTROL], seed, run_id)
        report = with_run_metadata({
            "kind": "prepared-projection-campaign", "status": "running", "seed": seed, "rounds": rounds,
            "output_dtype": output_dtype, "scope": "Prepared benchmark candidates; not production routing "
            "or a compressed-kernel acceptance artifact", "production_routing": False, **NO_PROMOTION,
            "evidence": {"path": str(evidence_path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(evidence_bytes).hexdigest()},
            "evidence_inputs": evidence["inputs"], "builder_source_sha256": evidence["builder_source_sha256"],
            "scope_protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha(protocol_path)},
            "bindings": rebuilt["bindings"], "runtime_identity": identity, "operator_context": context,
            "operator_display_snapshot": power, "inventory": inventory,
            "source_sha256": {name: _sha(ROOT / name) for name in SOURCES},
            "kernel_validation_report": {**bound["kernel_ref"], "run_id": bound["kernel_report"]["run_id"]},
            "observed_memory_clocks_kernel": rebuilt["observed_memory_clocks"],
            "registered_grid": grid, "control_manifest": controls, "cases": [],
            "head_diagnostic": {"status": "included", "config_source": "frozen gate/up winner"}
            if spec["head"]["tensor"] in entries else
            {"status": "unavailable", "reason": "Head omitted from verified retained corpus"}},
            {"pid": os.getpid(), "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException:
        workspace.close()
        vendor.close()
        raise
    write_json(output_path, report)
    choices, head_config = rebuilt["choices"], _head_config(spec, entries, rebuilt["choices"])
    max_ratio = spec["storage_limits"]["promotion_max_ratio"]
    try:
        for item in grid:
            name, (m, n, k), role = item["weight_tensor"], item["shape_mnk"], item["role"]
            record = {**item, "status": "running"}
            report["cases"].append(record)
            plan = candidate_plan(rebuilt, (m, n, k), output_dtype=output_dtype)
            if role == CONTROL:
                manifest, config, kernel_case = controls, CONTROL_CONFIG, None
                entry = next(t for t in controls["tensors"] if t["name"] == name)
                vendor_case = {"shape_mnk": [m, n, k], "output_dtype": output_dtype, "scope": CONTROL_SCOPE,
                               "vendor_pool": {"torch_mm": {"kind": "torch.mm"}} if output_dtype == "bf16" else
                               {"torch_mm_out_fp32": {"kind": "torch.mm", "out_dtype": "fp32"}}}
                if plan["route"] != "vendor":
                    raise ValueError("Synthetic controls are unseen kernel shapes and must stay vendor candidates")
            else:
                manifest, entry = corpus, entries[name]
                config = head_config if role == HEAD else choices[f"{n},{k}"]
                vendor_case = kernel_case = kernel_cases[(name, (m, n, k))]
                if kernel_case["selected"] != config or (plan["route"] == "coded" and plan["config"] != config):
                    raise ValueError(f"{name} frozen configuration differs from kernel evidence")
            source = np.array(read_tensor(manifest, name), copy=True)
            state = {}
            case = measure_case(entry, manifest, m, output_dtype, seed, [config], vendor_case, workspace, vendor,
                                rounds=rounds, kernel_revision=2,
                                external_factory=_external(plan, source, rounds, max_ratio, state))
            integer = torch.int16 if output_dtype == "bf16" else torch.int32
            bits = {label: out.view(integer) for label, out in state.pop("outputs").items()}
            clocks = _memory_clocks([case])
            route = state["info"]["route"]
            record.update(case, **item, status="complete", frozen_config=config, candidate_plan=plan,
                          frozen_config_source=("predeclared control diagnostic, never accepted or tuned"
                                                if role == CONTROL else "head gate/up winner" if role == HEAD
                                                else "frozen profile choices[N,K]"),
                          vendor_case_scope=CONTROL_SCOPE if role == CONTROL else "original kernel validation case",
                          kernel_validation_case=None if kernel_case is None else {
                              "activation_sha256": kernel_case["activation_sha256"],
                              "activation_matches": kernel_case["activation_sha256"] == case["activation_sha256"]},
                          selected_candidate=state["info"], diagnostic_scope=DIAGNOSTIC_SCOPE,
                          prepared_graph=summarize_graph(case["methods"], case["vendor_routes"], route=route),
                          prepared_graph_bitwise={
                              "prepared_vs_direct": torch.equal(bits["prepared"], bits["prepared_direct"]),
                              "direct_vs_direct_aa": torch.equal(bits["prepared_direct"], bits["prepared_direct_aa"])},
                          eager=state["eager"], timed_memory_clocks=clocks,
                          memory_clock_regime_matches=bool(clocks) and set(clocks) <= set(rebuilt["observed_memory_clocks"]),
                          **NO_PROMOTION)
            del bits, state
            write_json(output_path, report)
            graph = record["prepared_graph"]
            print(f"{role} {name} M={m} {output_dtype} route={route}: overhead "
                  f"{_pct(graph['implementation_overhead']['time_saved_pct'])} "
                  f"gate={graph['implementation_overhead']['overhead_gate_passed']}, vs vendor "
                  f"{_pct(graph['versus_vendor']['time_saved_pct'])}, eager "
                  f"{_pct(record['eager']['summary']['time_saved_pct'])} "
                  f"gate={record['eager']['summary']['overhead_gate_passed']}", flush=True)
            if not case["operating_context_stable"] or execution_context(_power_context()) != context:
                record.update(status="failed", error="Graph operating context changed; measurements retained")
                write_json(output_path, report)
                raise RuntimeError("Graph operating context changed; measured case retained, campaign stopped")
            torch.cuda.empty_cache()
        check_frozen_sources(report)
        _finish(report, grid, rebuilt)
        write_json(output_path, report)
        return report
    except BaseException as error:
        if report["cases"] and report["cases"][-1]["status"] == "running":
            report["cases"][-1].update(status="failed", error=f"{type(error).__name__}: {error}")
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_path, report)
        raise
    finally:
        workspace.close()
        vendor.close()


def _finish(report, grid, rebuilt):
    """Completion and separate control/per-shape summaries; failed gates stay data."""
    cases = report["cases"]
    observed = [(c["weight_tensor"], tuple(c["shape_mnk"]), c["role"]) for c in cases if c["status"] == "complete"]
    expected = [(c["weight_tensor"], tuple(c["shape_mnk"]), c["role"]) for c in grid]
    problems = []
    if len(observed) != len(set(observed)) or sorted(observed) != sorted(expected):
        problems.append("Missing, duplicated or unexpected registered cases")
    if not all(c["operating_context_stable"] for c in cases):
        problems.append("Graph operating context changed")
    if not all(c["memory_clock_regime_matches"] for c in cases):
        problems.append("Timed memory clocks differ from the kernel evidence regime")
    if not all(c["kernel_validation_case"]["activation_matches"] for c in cases if c["role"] != CONTROL):
        problems.append("Activations differ from the kernel validation process")
    if problems:
        raise ValueError("; ".join(problems))

    def gates(group):
        graph = [c["prepared_graph"]["implementation_overhead"]["overhead_gate_passed"]
                 and all(c["prepared_graph_bitwise"].values()) for c in group]
        eager = [c["eager"]["summary"]["overhead_gate_passed"] for c in group]
        return {"cases": len(group), "graph_overhead_gates_passed": sum(graph),
                "all_graph_overhead_gates_passed": all(graph), "eager_overhead_gates_passed": sum(eager),
                "all_eager_overhead_gates_passed": all(eager), **NO_PROMOTION}

    controls = [c for c in cases if c["role"] == CONTROL]
    shapes = {}
    for case in cases:
        if case["role"] != CONTROL:
            shapes.setdefault(",".join(map(str, case["shape_mnk"])), []).append(case)
    report["control_summary"] = {**gates(controls), "scope": CONTROL_SCOPE}
    report["shape_summaries"] = {key: {
        **gates(group), "route": sorted({c["selected_candidate"]["route"] for c in group}),
        "shape_kernel_gate_passed": rebuilt["shapes"].get(key, {}).get("kernel_gate_passed", False),
        "versus_vendor_time_saved_pct": [c["prepared_graph"]["versus_vendor"]["time_saved_pct"] for c in group],
        "versus_vendor_ci95_pct": [c["prepared_graph"]["versus_vendor"]["ci95_pct"] for c in group]}
        for key, group in shapes.items()}
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), **NO_PROMOTION,
                  production_routing=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    _project_environment()
    from experiment_control import GpuLock
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        report = run(args.evidence, args.output, seed=args.seed, rounds=args.rounds)
    print(f"{report['status']}: {len(report['cases'])} cases; controls {report['control_summary']}")


if __name__ == "__main__":
    main()
