"""Correctness checks for the native host binding, never a performance search."""
import argparse
import ctypes as ct
import gc
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, directory in (("CUDA_CACHE_PATH", ".cache/cuda"), ("TMPDIR", ".scratch"),
                       ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor"),
                       ("TRITON_CACHE_DIR", ".cache/triton")):
    os.environ.setdefault(key, str(ROOT / directory))


def main(output):
    import numpy as np
    import torch
    import compressed_bf16 as cb
    import mtile_bf16 as mt
    import native_prepared as native
    from experiment_control import project_path, write_json
    from projection_accuracy import check_outputs

    output = project_path(ROOT, output)
    if output.exists():
        raise ValueError("Preserve previous evidence; choose a new output")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(1409)
    native_build = native.ensure_loaded()
    checks, plans, records = 0, [], []
    driver = ct.CDLL("libcuda.so.1")
    get_attribute = driver.cuFuncGetAttribute
    get_attribute.argtypes = (ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p)
    get_attribute.restype = ct.c_int
    attribute_ids = {"max_threads_per_block": 0, "shared_size_bytes": 1, "const_size_bytes": 2,
                     "local_size_bytes": 3, "num_regs": 4, "ptx_version": 5, "binary_version": 6}

    def require(condition, message):
        nonlocal checks
        if not condition:
            raise RuntimeError(message)
        checks += 1

    def rejects(function, message):
        nonlocal checks
        try:
            function()
        except (ValueError, RuntimeError, TypeError):
            checks += 1
        else:
            raise RuntimeError(message)

    def same(left, right):
        dtype = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
        return left.dtype == right.dtype and torch.equal(left.view(dtype), right.view(dtype))

    # Ragged N and M exercise the original boundary guards and deterministic
    # M tiles. All eight original launch choices remain unchanged.
    for m, n, k in ((1, 13, 64), (2, 17, 128), (3, 17, 128), (5, 13, 64),
                    (17, 13, 64), (4, 17, 896)):
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        bits = w.view(torch.uint16).cpu().numpy().copy()
        packed = cb.upload(cb.pack_bits(bits), 0)
        require(np.array_equal(cb.unpack_weights(packed).view(torch.uint16).cpu().numpy(), bits),
                "Original uploaded weight bits changed")
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        reference = x.double() @ w.double().T
        for name in ("bf16", "fp32"):
            for threads in mt.THREADS:
                for vector in mt.VECTORS:
                    expected = mt.raw_linear(x, w, threads=threads, vector=vector, output_dtype=name)
                    for cls, weights in ((native.NativeRawCandidate, w),
                                         (native.NativeCodedCandidate, packed)):
                        candidate = cls(weights, tokens=m, threads=threads,
                                        vector=vector, output_dtype=name)
                        actual = candidate.linear(x)
                        require(same(actual, expected), "Native launch changed output bits")
                        require(check_outputs(reference, actual, output_dtype=name)["passed"],
                                "Native output failed unchanged FP64 oracle")
                        require(actual.data_ptr() not in (x.data_ptr(), w.data_ptr(), packed.payload.data_ptr()),
                                "Allocating operator aliased an input")
                        require(not candidate.custom_win_validated, "Verifier silently promoted a candidate")
                        info = candidate.describe()
                        original_module = cb.get_module(0) if m == 1 else mt.get_module(0)
                        function = original_module._functions[info["plan_info"]["kernel"]]
                        for label, number in attribute_ids.items():
                            value = ct.c_int()
                            require(get_attribute(ct.byref(value), number, function) == 0,
                                    "Original function attribute query failed")
                            require(value.value == info["plan_info"]["function_attributes"][label],
                                    "Duplicate JIT load has different function attributes")
                        records.append({"shape_mnk": [m, n, k], "output_dtype": name,
                                        "threads": threads, "vector": vector, "candidate": info})
                        candidate.close()
        require(np.array_equal(w.view(torch.uint16).cpu().numpy(), bits), "Native calls mutated weights")
        print(f"checked {m},{n},{k}: all launch choices and output contracts", flush=True)

    # Nondefault streams, allocation-including capture, changed graph inputs,
    # fixed-shape guards, opcheck and AOT composition are independent checks.
    for m, n, k in ((1, 13, 64), (2, 17, 896), (3, 17, 896), (5, 13, 64)):
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        packed = cb.upload(cb.pack_bits(w.view(torch.uint16).cpu().numpy()), 0)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        for name in ("bf16", "fp32"):
            for cls, weights in ((native.NativeRawCandidate, w), (native.NativeCodedCandidate, packed)):
                candidate = cls(weights, tokens=m, threads=128, vector=8, output_dtype=name)
                if cls is native.NativeRawCandidate:
                    op = torch.ops.klab_native_v1.raw.default
                    args = (x, w, candidate.plan_id, m, n, k, 128, 8, name)
                    invalid_args = (x, w, 2**62, m, n, k, 128, 8, name)
                else:
                    op = torch.ops.klab_native_v1.coded.default
                    args = (x, packed.payload, packed.metadata, candidate.plan_id, m, n, k, 128, 8, name)
                    invalid_args = (x, packed.payload, packed.metadata, 2**62, m, n, k, 128, 8, name)
                    bad_payload = torch.empty(packed.payload.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
                    rejects(lambda: op(x, bad_payload, packed.metadata, candidate.plan_id,
                                       m, n, k, 128, 8, name), "Misaligned payload reached the kernel")
                    rejects(lambda: op(x, packed.payload[:3], packed.metadata, candidate.plan_id,
                                       m, n, k, 128, 8, name), "Truncated payload reached the kernel")
                results = torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))
                require(all(value == "SUCCESS" for value in results.values()), f"Native opcheck failed: {results}")
                rejects(lambda: op(*invalid_args), "Unknown native plan was accepted")
                mismatched = (*args[:-3], 64, 8, name)
                rejects(lambda: op(*mismatched), "Plan accepted mismatched launch scalars")
                misaligned = torch.empty(m * k + 1, device="cuda", dtype=torch.bfloat16)[1:].reshape(m, k)
                rejects(lambda: candidate.linear(misaligned), "Misaligned activation reached the kernel")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = candidate.linear(x)
                    x.copy_(torch.randn_like(x) * 2)
                    captured.fill_(float("nan"))
                    graph.replay()
                    fresh = mt.raw_linear(x, w, threads=128, vector=8, output_dtype=name)
                stream.synchronize()
                require(same(captured, fresh), "Native graph missed current stream or changed input")
                torch.cuda.current_stream().wait_stream(stream)
                torch._dynamo.reset()
                compiled = torch.compile(candidate, backend="aot_eager", fullgraph=True)
                require(same(compiled(x), fresh), "Native AOT/Fake output differs")
                rejects(lambda: candidate.linear(torch.randn((m + 1, k), device="cuda", dtype=torch.bfloat16)),
                        "Native plan accepted a different shape")
                rejects(lambda: candidate.linear(x.float()), "Native plan accepted FP32 input")
                rejects(lambda: candidate.linear(x.cpu()), "Native plan accepted CPU input")
                rejects(lambda: candidate.to("cuda"), "Native module silently relocated")
                candidate.close()
                candidate.close()
                rejects(lambda: candidate.linear(x), "Retired native plan still callable")
                # Retiring metadata never unloads code captured by a live graph.
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                require(same(captured, fresh), "Plan retirement invalidated captured kernel code")
                graph.reset()
                plans.append({"candidate": cls.__name__, "shape_mnk": [m, n, k], "output_dtype": name})

    # Asynchronous temporary input lifetime on a nondefault stream. The native
    # recordStream path must protect buffers after Python references disappear.
    w = torch.randn((17, 128), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16)
    candidate = native.NativeRawCandidate(w, tokens=3, threads=128, vector=8, output_dtype="fp32")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    expected = candidate.linear(x)
    with torch.cuda.stream(stream):
        temporary = x.clone()
        got = candidate.linear(temporary)
        del temporary
    gc.collect()
    pressure = [torch.empty_like(x) for _ in range(32)]
    stream.synchronize()
    require(same(got, expected), "Temporary input lifetime was not protected")
    del pressure
    candidate.close()
    torch.cuda.synchronize()
    report = {"kind": "native-prepared-verification", "status": "passed", "checks": checks,
              "scope": "Correctness only, no speed or dispatch promotion", "configurations": records,
              "native_build": {key: native_build[key] for key in ("key", "library_path", "library_sha256")},
              "stream_graph_aot_cases": plans, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "frozen_implementation": mt.implementation_fingerprint(0),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("native/prepared.cpp", "native_prepared.py",
                                             "verification/verify_native_prepared.py")}}
    write_json(output, report)
    print(f"PASS {checks} native GPU correctness checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    from experiment_control import GpuLock
    with GpuLock(ROOT / ".cache/gpu.lock", wait=args.wait):
        main(args.output)
