"""GPU correctness checks for compressed_bf16 (docs/bf16-format.md, version 1).

Correctness only: no timing or tuning, and no held-out corpus tensors. Inputs
are seeded synthetic BF16 bits. Independent oracles are the parent's
bf16_reference (np.unpackbits decoder) and projection_accuracy (FP64 and
BF16-rounded interval). Raw-template weights are uploaded directly from the
original bits, never from the production unpacker. Finite accuracy gates and
nonfinite/subnormal/cancellation parity stress are counted separately.

Run under the project GPU lock: .venv/bin/python verify_compressed.py
"""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import torch

import compressed_bf16 as cb
from bf16_reference import decode_bf16
from experiment_control import GpuLock
from projection_accuracy import check_outputs

ROOT = Path(__file__).resolve().parent
SEED = 20261024  # Synthetic verification data only; not a preregistered tuning/validation seed.
OUTPUTS = (torch.bfloat16, torch.float32)
CONFIGS = tuple((threads, vector) for vector in cb.VECTORS for threads in cb.THREADS)
FINITE_SHAPES = ((1, 1, 64), (3, 37, 128), (5, 131, 192), (2, 4864, 896), (1, 896, 4864), (4, 896, 4864))
SPECIAL_BITS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f,
                         0x7f80, 0xff80, 0x7fc0, 0x7fc1, 0xffc1, 0x7f81, 0xffff], dtype=np.uint16)


class Checks:
    def __init__(self):
        self.counts = {}
        self.failures = []

    def expect(self, ok, label, group="finite"):
        self.counts[group] = self.counts.get(group, 0) + 1
        if not ok:
            self.failures.append(f"[{group}] {label}")

    def rejects(self, function, label, errors=(ValueError,)):
        try:
            function()
        except errors:
            ok = True
        else:
            ok = False
        self.expect(ok, f"accepted invalid input: {label}", "guards")


def attempt(checks, label, group, function):
    """Count an exception from `function` as a mismatch instead of aborting."""
    try:
        ok, detail = bool(function()), ""
    except Exception as error:
        ok, detail = False, f": {error!r}"
    checks.expect(ok, label + detail, group)


def finite_bits(rng, shape, scale):
    """Seeded normal values truncated to BF16 bit patterns (synthetic data only)."""
    values = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def with_raw_blocks(rng, bits):
    """Lower alternate exponents of one block in ~20% of rows by 20 (an exact
    bit edit), so those blocks need the raw fallback."""
    bits = bits.copy()
    n, k = bits.shape
    rows = np.flatnonzero(rng.random(n) < 0.2)
    rows = rows if len(rows) else np.array([0])
    columns = (rng.integers(0, k // 64, size=len(rows)) * 64)[:, None] + np.arange(0, 64, 2)
    part = bits[rows[:, None], columns]
    bits[rows[:, None], columns] = np.where(((part >> 7) & 0xff) > 40, part - np.uint16(20 << 7), part)
    return bits


def values64(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def to_gpu(bits, device):
    return torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint16).view(np.int16)).to(device).view(torch.bfloat16)


def gpu_bits(tensor):
    return tensor.view(torch.int16).cpu().numpy().view(np.uint16)


def same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.view(view), b.view(view))


def block_counts(packed):
    raw = int(np.count_nonzero(packed.metadata >> np.uint32(31)))
    return packed.metadata.size - raw, raw


def check_codec(checks, device):
    patterns = np.arange(65536, dtype=np.uint16)
    cases = {
        "all patterns 1024x64": patterns.reshape(1024, 64),
        "all patterns 512x128": patterns.reshape(512, 128),
        "all patterns ragged 655x100": patterns[:65500].reshape(655, 100),
        "specials ragged 15x1": SPECIAL_BITS.reshape(15, 1),
        "specials ragged 3x129": np.resize(SPECIAL_BITS, 3 * 129).reshape(3, 129),
    }
    for label, bits in cases.items():
        packed = cb.pack_bits(bits)
        checks.expect(packed.storage_bytes == packed.payload.nbytes + packed.metadata.nbytes,
                      f"{label}: storage accounting", "codec")
        checks.expect(packed.source_sha256 == hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest(),
                      f"{label}: source digest", "codec")
        independent = decode_bf16(packed.payload.tobytes(), packed.metadata, packed.n, packed.k)
        checks.expect(np.array_equal(independent, bits), f"{label}: independent CPU decode", "codec")
        checks.expect(np.array_equal(cb.unpack_bits(packed), bits), f"{label}: CPU unpack", "codec")
        gpu = cb.upload(packed, device)
        checks.expect(gpu.storage_bytes == packed.storage_bytes, f"{label}: GPU storage", "codec")
        checks.expect(np.array_equal(gpu_bits(cb.unpack_weights(gpu)), bits), f"{label}: GPU unpack", "codec")


def check_finite(checks, rng, device):
    coded_blocks = raw_blocks = 0
    for m, n, k in FINITE_SHAPES:
        label = f"M={m},N={n},K={k}"
        w_bits = with_raw_blocks(rng, finite_bits(rng, (n, k), 0.02))
        x_bits = finite_bits(rng, (m, k), 1.0)
        packed_cpu = cb.pack_bits(w_bits)
        coded, raw = block_counts(packed_cpu)
        coded_blocks, raw_blocks = coded_blocks + coded, raw_blocks + raw
        independent = decode_bf16(packed_cpu.payload.tobytes(), packed_cpu.metadata, n, k)
        checks.expect(np.array_equal(independent, w_bits), f"{label}: independent CPU decode", "codec")
        packed = cb.upload(packed_cpu, device)
        checks.expect(np.array_equal(gpu_bits(cb.unpack_weights(packed)), w_bits), f"{label}: GPU unpack", "codec")
        x, w = to_gpu(x_bits, device), to_gpu(w_bits, device)
        reference = values64(x_bits) @ values64(w_bits).T
        for dtype in OUTPUTS:
            for threads, vector in CONFIGS:
                config = f"{label} {dtype} t{threads} v{vector}"
                options = dict(threads=threads, vector=vector, output_dtype=dtype)
                result = cb.fused_linear(x, packed, **options)
                checks.expect(same_bits(result, cb.raw_linear(x, w, **options)), f"{config}: coded/raw bits")
                checks.expect(same_bits(result, packed.linear(x, **options)), f"{config}: custom op bits")
                accuracy = check_outputs(reference, result, output_dtype=dtype)
                checks.expect(accuracy["passed"], f"{config}: FP64 gate {accuracy}")
            out = torch.empty((m, n), dtype=dtype, device=device)
            returned = cb.fused_linear(x, packed, threads=128, vector=8, output_dtype=dtype, out=out)
            checks.expect(returned is out and same_bits(out, cb.fused_linear(
                x, packed, threads=128, vector=8, output_dtype=dtype)), f"{label} {dtype}: out= path")
    checks.expect(coded_blocks > 0 and raw_blocks > 0, f"block coverage coded={coded_blocks} raw={raw_blocks}")

    m, n, k = 65535, 3, 64  # Largest grid.y.
    w_bits, x_bits = finite_bits(rng, (n, k), 0.02), finite_bits(rng, (m, k), 1.0)
    packed, x, w = cb.upload(cb.pack_bits(w_bits), device), to_gpu(x_bits, device), to_gpu(w_bits, device)
    reference = values64(x_bits) @ values64(w_bits).T
    for dtype in OUTPUTS:
        for threads, vector in ((32, 2), (256, 8)):
            options = dict(threads=threads, vector=vector, output_dtype=dtype)
            result = cb.fused_linear(x, packed, **options)
            checks.expect(same_bits(result, cb.raw_linear(x, w, **options)), f"M=65535 {dtype}: coded/raw bits")
            checks.expect(check_outputs(reference, result, output_dtype=dtype)["passed"], f"M=65535 {dtype}: FP64")

    m, n, k = 2, 5, 12  # Raw V2 accepts K % 64 != 0; the coded kernels do not.
    w_bits, x_bits = finite_bits(rng, (n, k), 0.02), finite_bits(rng, (m, k), 1.0)
    reference = values64(x_bits) @ values64(w_bits).T
    for dtype in OUTPUTS:
        result = cb.raw_linear(to_gpu(x_bits, device), to_gpu(w_bits, device),
                               threads=32, vector=2, output_dtype=dtype)
        checks.expect(check_outputs(reference, result, output_dtype=dtype)["passed"], f"raw K=12 {dtype}: FP64")


def stress_cases(rng):
    patterns = np.arange(65536, dtype=np.uint16)
    yield "all weight patterns K=64", patterns.reshape(1024, 64), finite_bits(rng, (3, 64), 1.0)
    yield "all weight patterns K=128", patterns.reshape(512, 128), finite_bits(rng, (2, 128), 1.0)
    x = finite_bits(rng, (4, 256), 1.0)
    x.flat[rng.choice(x.size, 96, replace=False)] = rng.choice(SPECIAL_BITS, 96)
    yield "nonfinite/subnormal activations", with_raw_blocks(rng, finite_bits(rng, (67, 256), 0.02)), x
    w = (rng.integers(0, 128, (40, 128)) | (rng.integers(0, 2, (40, 128)) << 15)).astype(np.uint16)
    x = finite_bits(rng, (3, 128), 1.0)
    x[0] = (x[0] & 0x807f) | (((((x[0] >> 7) & 0xff) + 60) & 0xff) << 7)  # About 2^60.
    x[1] &= 0x807f  # Subnormal activations.
    yield "subnormal weights (coded, base 0)", w, x
    v = finite_bits(rng, (16, 128), 2.0**40)
    w = np.empty((16, 256), dtype=np.uint16)
    w[:, 0::2], w[:, 1::2] = v, v ^ 0x8000  # Exact cancellation pairs.
    w[0] = np.where(np.arange(256) % 2, 0xff7f, 0x7f7f)  # Overflow to +inf and -inf.
    x = finite_bits(rng, (3, 256), 1.0)
    x[0] = 0x3f80
    yield "cancellation and overflow", w, x


def check_stress(checks, rng, device):
    for label, w_bits, x_bits in stress_cases(rng):
        packed_cpu = cb.pack_bits(w_bits)
        checks.expect(np.array_equal(decode_bf16(packed_cpu.payload.tobytes(), packed_cpu.metadata,
                                                 packed_cpu.n, packed_cpu.k), w_bits), f"{label}: decode", "stress")
        packed = cb.upload(packed_cpu, device)
        x, w = to_gpu(x_bits, device), to_gpu(w_bits, device)
        for dtype in OUTPUTS:
            for threads, vector in CONFIGS:
                options = dict(threads=threads, vector=vector, output_dtype=dtype)
                checks.expect(same_bits(cb.fused_linear(x, packed, **options), cb.raw_linear(x, w, **options)),
                              f"{label} {dtype} t{threads} v{vector}: coded/raw bits", "stress")

    # No FTZ and one round-to-nearest-even BF16 store: x=[1,1,0,...].
    w_bits = np.zeros((3, 64), dtype=np.uint16)
    w_bits[0, 0] = 0x0001  # 2^-133: an FP32 subnormal product.
    w_bits[1], w_bits[1, 1] = 0x3f80, 0x3b80  # 1 + 2^-8: tie, rounds to even 0x3f80.
    w_bits[2], w_bits[2, 1] = 0x3f81, 0x3b80  # 1 + 2^-7 + 2^-8: tie, rounds to even 0x3f82.
    x_bits = np.zeros((1, 64), dtype=np.uint16)
    x_bits[0, :2] = 0x3f80
    expected = {torch.float32: np.array([[0x00010000, 0x3f808000, 0x3f818000]], dtype=np.uint32),
                torch.bfloat16: np.array([[0x0001, 0x3f80, 0x3f82]], dtype=np.uint16)}
    packed_cpu = cb.pack_bits(w_bits)
    checks.expect(block_counts(packed_cpu) == (3, 0), "exact-rounding weights use coded blocks", "stress")
    packed, x, w = cb.upload(packed_cpu, device), to_gpu(x_bits, device), to_gpu(w_bits, device)
    for dtype in OUTPUTS:
        view = torch.int32 if dtype == torch.float32 else torch.int16
        for threads, vector in CONFIGS:
            options = dict(threads=threads, vector=vector, output_dtype=dtype)
            for arm, result in (("coded", cb.fused_linear(x, packed, **options)),
                                ("raw", cb.raw_linear(x, w, **options))):
                actual = result.view(view).cpu().numpy().view(expected[dtype].dtype)
                checks.expect(np.array_equal(actual, expected[dtype]),
                              f"{arm} {dtype} t{threads} v{vector}: no-FTZ/RNE bits {actual}", "stress")


def check_streams_graphs_compile(checks, rng, device):
    m, n, k = 3, 131, 192
    w_bits = with_raw_blocks(rng, finite_bits(rng, (n, k), 0.02))
    packed_cpu = cb.pack_bits(w_bits)
    packed, w = cb.upload(packed_cpu, device), to_gpu(w_bits, device)
    x, x2 = to_gpu(finite_bits(rng, (m, k), 1.0), device), to_gpu(finite_bits(rng, (m, k), 1.0), device)
    bf16 = dict(threads=128, vector=8, output_dtype=torch.bfloat16)
    fp32 = dict(threads=64, vector=2, output_dtype=torch.float32)

    def eager(value):
        return (cb.fused_linear(value, packed, **bf16), cb.raw_linear(value, w, **fp32),
                cb.fused_linear(value, packed, **fp32))

    expected = eager(x)
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        side = (cb.fused_linear(x, packed, **bf16), cb.raw_linear(x, w, **fp32),
                packed.linear(x, **fp32), cb.unpack_weights(packed))
    current.wait_stream(stream)
    for index in range(3):
        checks.expect(same_bits(side[index], expected[index]), f"non-default stream output {index}", "runtime")
    checks.expect(np.array_equal(gpu_bits(side[3]), w_bits), "non-default stream unpack", "runtime")

    static_x = x.clone()
    static_coded = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    static_raw = torch.empty((m, n), dtype=torch.float32, device=device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):  # Warm up outside capture.
        packed.linear(static_x, **fp32)
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cb.fused_linear(static_x, packed, **bf16, out=static_coded)
        cb.raw_linear(static_x, w, **fp32, out=static_raw)
        static_op = packed.linear(static_x, **fp32)
    for trial, value in enumerate((x, x2, x)):
        if trial == 2:  # Allocator churn and another upload before a later replay.
            churn = [torch.empty(1 << 20, device=device) for _ in range(8)]
            del churn
            cb.upload(packed_cpu, device)
        static_x.copy_(value)
        graph.replay()
        torch.cuda.synchronize(device)
        reference = eager(value)
        for index, result in enumerate((static_coded, static_raw, static_op)):
            checks.expect(same_bits(result, reference[index]), f"graph replay {trial} output {index}", "runtime")
    checks.expect(cb.get_module(device) is packed.module and not packed.module.closed,
                  "cached module retained", "runtime")
    del graph

    for options in (fp32, bf16):
        compiled = torch.compile(lambda value, options=options: packed.linear(value, **options), fullgraph=True)
        attempt(checks, f"torch.compile packed.linear {options['output_dtype']}", "runtime",
                lambda: same_bits(compiled(x), cb.fused_linear(x, packed, **options)))
    direct = torch.compile(lambda value: cb.compressed_linear(value, packed.payload, packed.metadata,
                                                              packed.n, packed.k, 256, 8, "bf16"), fullgraph=True)
    attempt(checks, "torch.compile compressed_linear", "runtime",
            lambda: same_bits(direct(x2), cb.fused_linear(x2, packed, threads=256, vector=8,
                                                          output_dtype=torch.bfloat16)))

    def opcheck():
        torch.library.opcheck(cb.compressed_linear,
                              (x, packed.payload, packed.metadata, packed.n, packed.k, 128, 8, "bf16"))
        return True

    attempt(checks, "opcheck schema/fake/aot", "runtime", opcheck)


def check_guards(checks, rng, device):
    m, n, k = 3, 131, 192
    w_bits = finite_bits(rng, (n, k), 0.02)
    packed_cpu = cb.pack_bits(w_bits)
    packed, w = cb.upload(packed_cpu, device), to_gpu(w_bits, device)
    x = to_gpu(finite_bits(rng, (m, k), 1.0), device)
    ok = dict(threads=128, vector=8, output_dtype=torch.bfloat16)

    def fused(value=x, target=packed, **changes):
        return lambda: cb.fused_linear(value, target, **{**ok, **changes})

    def raw(value=x, weights=w, **changes):
        return lambda: cb.raw_linear(value, weights, **{**ok, **changes})

    shared = torch.empty(m * k + m * n, dtype=torch.bfloat16, device=device)
    shared[:m * k].copy_(x.reshape(-1))
    misaligned = torch.empty(m * k + 8, dtype=torch.bfloat16, device=device)[1:1 + m * k].view(m, k)
    misaligned.copy_(x)
    ragged = cb.upload(cb.pack_bits(finite_bits(rng, (4, 100), 0.02)), device)
    small = cb.upload(cb.pack_bits(w_bits[:, :64].copy()), device)
    cases = {
        "FP16 x": fused(x.half()),
        "FP32 x": fused(x.float()),
        "CPU x": fused(x.cpu()),
        "1-D x": fused(x[0]),
        "non-contiguous x": fused(torch.cat((x, x), 1)[:, ::2]),
        "requires_grad x": fused(x.clone().requires_grad_(True)),
        "K mismatch": fused(torch.cat((x, x[:, :64]), 1)),
        "threads=48": fused(threads=48),
        "threads=True": fused(threads=True),
        "vector=4": fused(vector=4),
        "FP16 output": fused(output_dtype=torch.float16),
        "string fp16 output": fused(output_dtype="fp16"),
        "misaligned x v8": fused(misaligned),
        "misaligned x v2": fused(misaligned, vector=2),
        "M=65536": fused(torch.zeros((65536, 64), dtype=torch.bfloat16, device=device), small),
        "ragged K coded": fused(torch.zeros((1, 100), dtype=torch.bfloat16, device=device), ragged, vector=2),
        "CPU packed": fused(target=packed_cpu),
        "out aliases x": lambda: cb.fused_linear(shared[:m * k].view(m, k), packed, **ok,
                                                 out=shared[m * k:].view(m, n)),
        "out aliases payload": fused(output_dtype=torch.float32,
                                     out=packed.payload.view(torch.float32)[:m * n].view(m, n)),
        "out aliases metadata": fused(output_dtype=torch.float32,
                                      out=packed.metadata.view(torch.float32)[:m * n].view(m, n)),
        "out wrong dtype": fused(out=torch.empty((m, n), dtype=torch.float32, device=device)),
        "out wrong shape": fused(out=torch.empty((m, n + 1), dtype=torch.bfloat16, device=device)),
        "out non-contiguous": fused(out=torch.empty((n, m), dtype=torch.bfloat16, device=device).t()),
        "raw FP32 w": raw(weights=w.float()),
        "raw K%8 v8": raw(torch.zeros((2, 12), dtype=torch.bfloat16, device=device),
                          torch.zeros((5, 12), dtype=torch.bfloat16, device=device)),
        "raw out aliases w": raw(output_dtype=torch.float32,
                                 out=w.view(-1).view(torch.float32)[:m * n].view(m, n)),
        "raw misaligned x": raw(misaligned),
    }
    before = gpu_bits(cb.unpack_weights(packed))
    for label, function in cases.items():
        checks.rejects(function, label)
    checks.expect(np.array_equal(before, w_bits) and np.array_equal(gpu_bits(cb.unpack_weights(packed)), w_bits),
                  "rejected calls left packed weights intact", "guards")

    coded = cb.pack_bits(np.full((1, 64), 0x3f80, dtype=np.uint16))
    raw_block = cb.pack_bits((np.arange(64, dtype=np.uint16) << 7).reshape(1, 64))
    spread = cb.pack_bits(((np.arange(64, dtype=np.uint16) % 16 + 120) << 7).reshape(1, 64))

    def changed(packed_value, **fields):
        return lambda: cb.upload(dataclasses.replace(packed_value, **fields), device)

    def flipped(values, index, mask):
        values = values.copy()
        values[index] ^= mask
        return values

    exponent_overflow = (spread.metadata & np.uint32(~(0xff << 23) & 0xffffffff)) | np.uint32(250 << 23)
    upload_cases = {
        "metadata offset": changed(coded, metadata=flipped(coded.metadata, 0, np.uint32(1))),
        "payload bit": changed(coded, payload=flipped(coded.payload, 5, np.uint8(1))),
        "source digest": changed(coded, source_sha256="0" * 64),
        "format version": changed(coded, format_version=2),
        "raw base nonzero": changed(raw_block, metadata=flipped(raw_block.metadata, 0, np.uint32(1 << 23))),
        "extra payload": changed(coded, payload=np.concatenate((coded.payload, np.zeros(32, np.uint8)))),
        "int32 metadata": changed(coded, metadata=coded.metadata.view(np.int32)),
        "exponent overflow": changed(spread, metadata=exponent_overflow.astype(np.uint32)),
        "CPU device": lambda: cb.upload(coded, "cpu"),
    }
    checks.expect(block_counts(spread) == (1, 0), "exponent-overflow fixture is coded", "guards")
    for label, function in upload_cases.items():
        checks.rejects(function, f"upload {label}")
    for label, value in {"float32": np.ones((1, 64), np.float32), "int16": np.ones((1, 64), np.int16),
                         "empty": np.empty((0, 64), np.uint16), "1-D": np.ones(64, np.uint16),
                         "3-D": np.ones((1, 1, 64), np.uint16), "list": [[1] * 64]}.items():
        checks.rejects(lambda value=value: cb.pack_bits(value), f"pack_bits {label}")


def check_implementation(checks, device):
    module = cb.get_module(device)
    fingerprint = cb.implementation_fingerprint(device)
    compilation = module.compilation
    checks.expect(fingerprint["implementation_sha256"]
                  == hashlib.sha256((ROOT / "compressed_bf16.py").read_bytes()).hexdigest(),
                  "fingerprint binds compressed_bf16.py", "build")
    checks.expect(fingerprint["ptx_sha256"] == compilation["ptx_sha256"]
                  and compilation["sources"] == ["compressed_kernels.cu"]
                  and compilation["kernels"] == list(cb.kernel_names()), "module sources/kernels", "build")
    ptx = (ROOT / compilation["ptx_path"]).read_bytes()
    checks.expect(hashlib.sha256(ptx).hexdigest() == compilation["ptx_sha256"], "PTX cache hash", "build")
    for needle, present in ((b"fma.rn.f32", True), (b"add.rn.f32", True), (b"cvt.rn.bf16.f32", True),
                            (b".ftz", False), (b"atom.", False), (b".shared", False), (b"approx", False)):
        checks.expect((needle in ptx) == present, f"PTX {'contains' if present else 'omits'} {needle!r}", "build")
    return fingerprint


def verify():
    checks = Checks()
    rng = np.random.default_rng(SEED)
    device = torch.cuda.current_device()
    with torch.no_grad():
        fingerprint = check_implementation(checks, device)
        check_codec(checks, device)
        check_finite(checks, rng, device)
        check_stress(checks, rng, device)
        check_streams_graphs_compile(checks, rng, device)
        check_guards(checks, rng, device)
        torch.cuda.synchronize(device)
        # Last: module loading/upload must refuse to run during capture.
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                cb.upload(cb.pack_bits(np.full((1, 64), 0x3f80, dtype=np.uint16)), device)
            rejected = False
        except RuntimeError:
            rejected = True
        checks.expect(rejected, "upload during capture rejected", "guards")
    return checks, fingerprint


if __name__ == "__main__":
    with GpuLock(ROOT / ".cache/gpu.lock"):
        checks, fingerprint = verify()
    groups = ", ".join(f"{group}={count}" for group, count in checks.counts.items())
    print(f"implementation {fingerprint['fingerprint_sha256'][:16]} ptx {fingerprint['ptx_sha256'][:16]}")
    if checks.failures:
        for failure in checks.failures[:50]:
            print("FAIL:", failure)
        raise SystemExit(f"FAIL: {len(checks.failures)} mismatches in {sum(checks.counts.values())} checks ({groups})")
    print(f"PASS: {sum(checks.counts.values())} checks ({groups}), 0 mismatches")
