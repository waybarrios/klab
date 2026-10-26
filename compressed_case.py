"""One serial rotating-ring experiment, shared by tuning and validation.

No imports initialize CUDA until measure_case is called. Caller owns GpuLock
and the shared cuBLASLt workspace. Configuration selection lives elsewhere.
"""
import base64
import hashlib
import math
import time

import numpy as np


def launch_config(name):
    import re
    match = re.fullmatch(r"coded_t(32|64|128|256)_v(2|8)", name)
    if match is None:
        raise ValueError(f"Unknown compressed configuration {name}")
    return tuple(map(int, match.groups()))


def _batch(calls):
    def invoke():
        for call in calls:
            call()
    return invoke


def _reference(x, weight):
    """FP64 reference in bounded weight-row chunks, returned on the CPU."""
    n, k = weight.shape
    result = np.empty((x.shape[0], n), dtype=np.float64)
    xd = x.double()
    rows = max(1, 16 * 1024**2 // (8 * k))
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        result[:, start:stop] = (xd @ weight[start:stop].double().T).cpu().numpy()
    return result


def measure_case(entry, corpus, m, output_dtype, seed, configurations, vendor_case,
                 workspace, vendor, *, rounds=31, decode_diagnostic=False,
                 external_factory=None, kernel_revision=1):
    """Time supplied settings and frozen vendor algorithms, checking all outputs.

    An external_factory may add named prior-art arms using the same physical
    ring and activations. Its retained objects stay alive through graph replay.
    No external arm enters the vendor denominator or changes our configuration.
    """
    import torch
    import torch.nn.functional as functional
    import compressed_bf16 as cb
    from bench import measure
    from bench_compressed import _power_context
    from bf16_reference import decode_bf16
    from compressed_campaign import check_ring
    from cublaslt_baseline import LtPlan, prepare_lt_plans
    from gpu_monitor import GpuMonitor
    from power_context import execution_context
    from real_weights import read_tensor

    if kernel_revision == 1:
        kernel = cb
    elif kernel_revision == 2:
        import mtile_bf16 as kernel
    else:
        raise ValueError("Implemented performance revisions are 1 and 2")
    if rounds < 31 or output_dtype not in ("bf16", "fp32") or not configurations:
        raise ValueError("Require 31 rounds, an explicit output contract and frozen configurations")
    if len(set(configurations)) != len(configurations) or len(configurations) > 8:
        raise ValueError("At most eight distinct launch configurations")
    for name in configurations:
        launch_config(name)
    power = _power_context()
    started = time.monotonic()
    implementation = kernel.implementation_fingerprint(0)
    source = np.array(read_tensor(corpus, entry["name"]), copy=True)
    n, k = source.shape
    packing_start = time.monotonic()
    encoded = cb.pack_bits(source)
    packing_seconds = time.monotonic() - packing_start
    if not np.array_equal(decode_bf16(encoded.payload.tobytes(), encoded.metadata, n, k), source):
        raise RuntimeError("Independent CPU decoder changed original checkpoint bits")
    weight = torch.from_numpy(source).to("cuda").view(torch.bfloat16)
    if not torch.equal(weight.view(torch.uint16).cpu(), torch.from_numpy(source)):
        raise RuntimeError("Original checkpoint bits changed on upload")
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    x_sha = hashlib.sha256(x.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    target = max(80 * 1024**2, 20 * l2)
    copies = max(1, math.ceil(target / source.nbytes))
    raw = weight.unsqueeze(0).repeat(copies, 1, 1)
    transposed = raw.transpose(1, 2).contiguous()
    if (not torch.equal(raw.view(torch.int16), weight.view(torch.int16).expand(copies, n, k))
            or not torch.equal(transposed.view(torch.int16).transpose(1, 2), raw.view(torch.int16))):
        raise RuntimeError("Rotating original/transposed ring changed weight bits")
    # Each item has its own offset address space, including the large tied head.
    packed = [cb.upload(encoded, 0) for _ in range(copies)]
    for item in packed:
        if not torch.equal(cb.unpack_weights(item).view(torch.int16), weight.view(torch.int16)):
            raise RuntimeError("GPU decoding changed original checkpoint bits")
    torch.cuda.reset_peak_memory_stats()
    functions, outputs, routes, records, plans, retained = {}, {}, [], {}, [], []
    diagnostics = []

    def add(name, builder, twin=False):
        for label in (name, name + "_aa") if twin else (name,):
            out = torch.empty((copies, m, n), dtype=dtype, device="cuda")
            outputs[label] = out
            functions[label] = _batch([builder(i, out[i]) for i in range(copies)])

    try:
        if vendor_case is None:
            pool = prepare_lt_plans(x, raw[0], output_dtype=dtype, workspace=workspace)
            plans.extend(pool)
            diagnostics = pool.search_diagnostics
            vendor_pool = {f"lt_{p.spec['layout']}_h{p.spec['candidate_idx']}": p.freeze() for p in pool}
            if output_dtype == "bf16":
                vendor_pool.update(torch_mm={"kind": "torch.mm"}, torch_linear={"kind": "torch.nn.functional.linear"})
            else:
                vendor_pool.update(cublas_compute32={"kind": "cublasGemmEx", "layout": "original"},
                                   cublas_packed_compute32={"kind": "cublasGemmEx", "layout": "transposed"})
                try:
                    torch.mm(x, raw[0].T, out_dtype=torch.float32)
                except (TypeError, RuntimeError) as error:
                    diagnostics.append({"route": "torch_mm_out_fp32", "unavailable": str(error)})
                else:
                    vendor_pool["torch_mm_out_fp32"] = {"kind": "torch.mm", "out_dtype": "fp32"}
        else:
            if vendor_case["shape_mnk"] != [m, n, k] or vendor_case["output_dtype"] != output_dtype:
                raise ValueError("Frozen vendor case differs from the requested operation")
            vendor_pool = vendor_case["vendor_pool"]
        for name, record in vendor_pool.items():
            records[name] = record
            routes.append(name)
            if record.get("library") == "cublasLt":
                group = [LtPlan(record, base64.b64decode(record["algorithm_b64"]), workspace,
                                x=x, weight=raw[i],
                                packed=transposed[i] if record["layout"] == "transposed" else None)
                         for i in range(copies)]
                plans.extend(group)
                add(name, lambda i, out, group=group: lambda: group[i].run(x, raw[i], out), twin=True)
            elif name == "torch_mm" and output_dtype == "bf16":
                add(name, lambda i, out: lambda: torch.mm(x, raw[i].T, out=out), twin=True)
            elif name == "torch_linear" and output_dtype == "bf16":
                for label in (name, name + "_aa"):
                    holders = [None] * copies
                    def invoke(holders=holders):
                        for i in range(copies):
                            holders[i] = functional.linear(x, raw[i])
                    outputs[label], functions[label] = holders, invoke
            elif name == "torch_mm_out_fp32" and output_dtype == "fp32":
                add(name, lambda i, out: lambda: torch.mm(x, raw[i].T, out_dtype=torch.float32, out=out), twin=True)
            elif name in ("cublas_compute32", "cublas_packed_compute32") and output_dtype == "fp32":
                is_packed = name == "cublas_packed_compute32"
                matrix = transposed if is_packed else raw
                add(name, lambda i, out, matrix=matrix, flag=is_packed:
                    lambda: vendor.mm_compute32(x, matrix[i], out, "bf16", flag), twin=True)
            else:
                raise ValueError(f"Unsupported frozen vendor route {name} for {output_dtype}")
        for config in configurations:
            threads, vector = launch_config(config)
            add(config, lambda i, out, t=threads, v=vector:
                lambda: kernel.fused_linear(x, packed[i], threads=t, vector=v, output_dtype=output_dtype, out=out))
            add(config.replace("coded_", "raw_"), lambda i, out, t=threads, v=vector:
                lambda: kernel.raw_linear(x, raw[i], threads=t, vector=v, output_dtype=output_dtype, out=out))
        if external_factory is not None:
            extra_functions, extra_outputs, owners, external_info = external_factory(x, raw, output_dtype)
            if set(functions) & set(extra_functions) or set(extra_functions) != set(extra_outputs):
                raise ValueError("External method names must be distinct and have outputs")
            functions.update(extra_functions)
            outputs.update(extra_outputs)
            retained.extend(owners)
        else:
            external_info = None
        torch.cuda.synchronize()
        setup_seconds = time.monotonic() - started
        class PhaseMonitor(GpuMonitor):
            phase = "preparation"

            def __init__(self):
                import threading
                super().__init__()
                self.sample_lock = threading.Lock()

            def sample(self):
                with self.sample_lock:
                    phase_before = self.phase
                    start = len(self.samples)
                    super().sample()
                    for sample in self.samples[start:]:
                        sample["phase"] = phase_before if phase_before == self.phase else "boundary"

        with PhaseMonitor() as monitor:
            warmup = time.monotonic()
            while time.monotonic() - warmup < 2:
                for function in functions.values():
                    function()
                torch.cuda.synchronize()
            timings = measure(functions, rounds=rounds, repeats=1,
                              phase_callback=lambda phase: setattr(monitor, "phase", phase),
                              round_callback=lambda index: monitor.sample() if index == 0 else None)
        for method in timings.values():
            for field in ("median_us", "min_us", "max_us"):
                method[field] /= copies
            method["samples_us"] = [value / copies for value in method["samples_us"]]
        reference = _reference(x, weight)
        for name, out in list(outputs.items()):
            if isinstance(out, list):
                outputs[name] = out = torch.stack(out)
            timings[name]["numerics"] = check_ring(reference, out, output_dtype=output_dtype)
        parities = {}
        for config in configurations:
            integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
            parities[config] = torch.equal(outputs[config].view(integer), outputs[config.replace("coded_", "raw_")].view(integer))
        decode = None
        if decode_diagnostic:
            # A separate single-weight decode diagnostic, never counted as fused gain.
            decode = measure({"decode_only": lambda: cb.unpack_weights(packed[0])}, rounds=rounds, repeats=1)["decode_only"]
            decode.update(scope="single-weight decode only, not a projection", original_bytes=source.nbytes,
                          packed_bytes=encoded.storage_bytes)
        final_power = _power_context()
        context_stable = execution_context(final_power) == execution_context(power)
        return {"shape_mnk": [m, n, k], "input_dtype": "bf16", "output_dtype": output_dtype,
                "weight_tensor": entry["name"], "weight_sha256": entry["sha256"],
                "activation_sha256": x_sha, "seed": seed, "copies": copies, "calls_per_graph": copies,
                "raw_ring_bytes": raw.numel() * 2, "packed_ring_bytes": sum(p.storage_bytes for p in packed),
                "transposed_ring_bytes": transposed.numel() * 2,
                "raw_ring_over_l2": raw.numel() * 2 / l2,
                "storage_bytes": encoded.storage_bytes, "storage_ratio": encoded.storage_bytes / source.nbytes,
                "maximum_per_copy_payload_bytes": encoded.payload.nbytes,
                "packing_seconds": packing_seconds, "setup_seconds": setup_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "shared_workspace_bytes": workspace.nbytes, "vendor_pool": records,
                "vendor_pool_scope": "frozen" if vendor_case is not None else "head vendor-only search; diagnostic",
                "search_diagnostics": diagnostics, "methods": timings, "vendor_routes": routes,
                "raw_template_bitwise": all(parities.values()), "raw_template_parity_by_config": parities,
                "original_bits_preserved": True, "gpu_telemetry": monitor.samples,
                "implementation": implementation, "kernel_revision": kernel_revision,
                "token_tile": 1 if kernel_revision == 1 else kernel.tile_for_m(m),
                "operator_context": power, "operator_context_after": final_power,
                "execution_context": execution_context(power), "operating_context_stable": context_stable,
                "scope": "streaming CUDA graph GPU latency",
                "decode_only_diagnostic": decode, "external_info": external_info}
    finally:
        for plan in plans:
            plan.close()
