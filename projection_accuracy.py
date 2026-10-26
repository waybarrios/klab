"""Independent CPU accuracy oracle for exact-weight projection experiments.

BF16 rounding uses the FP64 reference directly, avoiding an intermediate
FP32 rounding. A small table of representable values makes midpoint and
subnormal behavior explicit. No CUDA or PyTorch import is needed.
"""

from functools import lru_cache

import numpy as np

ATOL = 1e-5
RTOL = 1e-4


def _values(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


@lru_cache(maxsize=1)
def _grid():
    bits = np.arange(65536, dtype=np.uint16)
    # Exclude infinities/signalling NaNs before any floating conversion.
    bits = bits[(bits & 0x7f80) != 0x7f80]
    values = _values(bits)
    # Virtual +/-2^128 neighbors implement IEEE overflow midpoint rounding.
    values = np.concatenate((values, [-2.0**128, 2.0**128]))
    bits = np.concatenate((bits, np.array([0xff80, 0x7f80], dtype=np.uint16)))
    order = np.argsort(values, kind="stable")
    return values[order], bits[order]


def round_bf16_bits(reference64):
    reference = np.asarray(reference64, dtype=np.float64)
    if not np.isfinite(reference).all():
        raise ValueError("Reference rounding expects finite values; nonfinite cases are separate")
    grid, codes = _grid()
    index = np.searchsorted(grid, reference).clip(1, len(grid) - 1)
    lower, upper = grid[index - 1], grid[index]
    lo_code, hi_code = codes[index - 1], codes[index]
    lo_distance, hi_distance = reference - lower, upper - reference
    choose_hi = (hi_distance < lo_distance) | ((hi_distance == lo_distance) & ((hi_code & 1) == 0))
    result = np.where(choose_hi, hi_code, lo_code).astype(np.uint16)
    # The sorted grid contains both signed zeros. Preserve the source sign
    # whenever rounding reaches zero, including nonzero underflow values.
    return np.where((result & 0x7fff) == 0, np.where(np.signbit(reference), 0x8000, 0), result).astype(np.uint16)


def _array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _rank(bits):
    bits = np.asarray(bits, dtype=np.int64)
    return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)


def check_outputs(reference64, output, *, output_dtype):
    """Check every finite performance output, with explicit nonfinite failures.

    BF16 values must be representable BF16 and lie in the rounded FP64
    tolerance interval. One-ULP and exact-rounding rates are diagnostics.
    Compression/raw arithmetic equality is checked independently by callers.
    """
    dtype = str(output_dtype).removeprefix("torch.")
    dtype = {"bfloat16": "bf16", "float32": "fp32"}.get(dtype, dtype)
    if dtype not in ("bf16", "fp32"):
        raise ValueError("Expected bf16 or fp32 output contract")
    reference, actual = _array(reference64), _array(output)
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError("Reference and output shapes must be equal and nonempty")
    finite = np.isfinite(reference) & np.isfinite(actual)
    if not finite.all():
        return {"passed": False, "elements_checked": int(reference.size),
                "nonfinite": int(np.count_nonzero(~finite)), "violations": int(np.count_nonzero(~finite)),
                "scope": "Nonfinite/overflow inputs require separately labelled stress checks"}
    tol = ATOL + RTOL * np.abs(reference)
    delta = np.abs(actual - reference)
    extra = {}
    if dtype == "bf16":
        actual_bits = round_bf16_bits(actual)
        if not np.array_equal(_values(actual_bits), actual):
            raise ValueError("Output is not representable in the requested BF16 contract")
        low, high = _values(round_bf16_bits(reference - tol)), _values(round_bf16_bits(reference + tol))
        valid = (actual >= low) & (actual <= high)
        reference_bits = round_bf16_bits(reference)
        ulps = np.abs(_rank(actual_bits) - _rank(reference_bits))
        extra = {"exact_rounding_fraction": float(np.mean(actual == _values(reference_bits))),
                 "within_one_ulp_fraction": float(np.mean(ulps <= 1)), "max_ulp_error": int(ulps.max())}
    else:
        valid = delta <= tol
    denominator = np.linalg.norm(reference.reshape(-1))
    return {"passed": bool(valid.all()), "elements_checked": int(reference.size),
            "violations": int(np.count_nonzero(~valid)), "nonfinite": 0,
            "max_abs_error": float(delta.max()),
            "relative_l2_error": float(np.linalg.norm(delta.reshape(-1)) / denominator) if denominator else None,
            "atol": ATOL, "rtol": RTOL, **extra}
