"""GPU numerical tests for the adaptive projection candidates."""

import math
from pathlib import Path

from experiment_control import GpuLock

ROOT = Path(__file__).resolve().parent


def verify():
    import torch
    import kernel_lab
    import packed_fp32
    from bench import check

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab = kernel_lab.default_lab()
    checks = 0
    for seed in (941, 947):
        torch.manual_seed(seed)
        for n, k in ((1, 64), (65, 128), (17, 256), (33, 512), (7, 1024)):
            w = torch.randn(n, k, device="cuda")
            packed = packed_fp32.pack_weights(w)
            assert torch.equal(packed.unpack().view(torch.int32), w.view(torch.int32))
            checks += 1
            x = torch.randn(3, k, device="cuda") / math.sqrt(k)
            ref = (x.double() @ w.double().T).float()
            for threads in (32, 64, 128, 256):
                check(packed.linear(x, threads), ref)
                checks += 1
                for vector in (1, 4):
                    check(lab.tiny_matmul(x, w, threads, vector), ref)
                    checks += 1
    bits = torch.randint(-(2**31), 2**31 - 1, (5, 256), device="cuda", dtype=torch.int32)
    bits[0, :8] = torch.tensor([0, -(2**31), 1, 0x007fffff, 0x7f800000,
                               -8388608, 0x7fc12345, -0x003abcde], device="cuda", dtype=torch.int32)
    packed = packed_fp32.pack_weights(bits.view(torch.float32))
    assert torch.equal(packed.unpack().view(torch.int32), bits)
    checks += 1
    q = packed_fp32.pack_weights(w)
    compiled = torch.compile(packed_fp32.linear, fullgraph=True)
    check(compiled(x, q.payload, q.metadata, 128), q.linear(x))
    checks += 1
    for invalid in (lambda: packed_fp32.pack_weights(w.half()),
                    lambda: lab.tiny_matmul(x[:, :-1], w[:, :-1]),
                    lambda: packed_fp32.fused_linear(x, q, out=x)):
        try:
            invalid()
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid input accepted")
    torch.cuda.synchronize()
    return checks


if __name__ == "__main__":
    with GpuLock(ROOT / ".cache/gpu.lock"):
        print(f"PASS: {verify()} adaptive-candidate checks")
