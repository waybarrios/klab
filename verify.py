"""Numerical and Python integration checks, separate from performance runs."""

import kernel_lab
import torch
from bench import check, correctness


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    lab = kernel_lab.default_lab()
    checks = correctness(lab)
    x = torch.randn(3, 513, device="cuda")
    w = torch.randn(65, 513, device="cuda")
    reference = (x.double() @ w.double().T).float()

    def operation(x, w):
        return kernel_lab.warp_matmul(x, w, threads=64, vector=1)

    check(operation(x, w), reference, atol=1e-4)
    check(torch.compile(operation, fullgraph=True)(x, w), reference, atol=1e-4)
    checks += 2
    residual = torch.randn_like(x)
    gamma = torch.randn(513, device="cuda")
    norm = kernel_lab.residual_rmsnorm(x, residual, gamma)
    compiled_norm = torch.compile(kernel_lab.residual_rmsnorm, fullgraph=True)
    check(compiled_norm(x, residual, gamma), norm, atol=1e-5, rtol=1e-5)
    checks += 1
    # Packed vector loads must reject a contiguous but misaligned view.
    shifted = torch.randn(1025, device="cuda")[1:].view(1, 1024)
    weights = torch.randn(17, 1024, device="cuda")
    try:
        lab.warp_matmul(shifted, weights, vector=4)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Misaligned 16-byte vector load accepted")
    check(lab.warp_matmul(shifted, weights, vector=1), shifted @ weights.T, atol=1e-4)
    checks += 1
    print(f"PASS: {checks} numerical, boundary, stream, validation, and fullgraph checks")


if __name__ == "__main__":
    main()
