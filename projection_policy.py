"""CPU-only selection from held-out measurements; no online benchmarking."""

import re
import statistics

VENDORS=("torch_mm","torch_mv","cublas_mm","cublas_mv","cublas_packed_mm","cublas_packed_mv")


def choose(profile,shape,force_vendor=False,*,output_dtype=None):
    if profile.get("kind") in ("compressed-projection","compressed-projection-tuning",
                               "compressed-projection-validation"):
        return compressed_vendor_fallback(profile,shape,output_dtype or "fp32")
    half=profile.get("kind")=="half-projection"
    if profile.get("kind") not in ("streaming-projection","half-projection") or profile.get("status")!="complete" or (half and profile.get("mode")!="validate"):
        raise ValueError("Use a completed held-out streaming validation report")
    cases=[case for case in profile["cases"] if tuple(case["shape_mnk"])==tuple(shape)]
    if not cases:
        return {"backend":"vendor","config":"cublas_compute32" if half else "torch_mm","reason":"Shape has no held-out validation",
                "custom_win_validated":False,"accepted_10pct_target":False}
    configs={case["selected"] for case in cases}
    if len(configs)!=1: raise ValueError("Validation changed the selected kernel between cases")
    config=next(iter(configs))
    pattern=r"t(?:32|64|128|256)_v(?:2|8)" if half else r"(?:(?:tiny|warp)_t(?:32|64|128|256)_v(?:1|4)|(?:lossless|fp28)_t(?:32|64|128|256))"
    if not re.fullmatch(pattern,config):
        raise ValueError("Unrecognized custom configuration")
    enough_inputs=len({case["seed"] for case in cases})>=2
    passed=enough_inputs and all((case.get("bitwise_layout_check") and case.get("numerics_passed") and case.get("noise_gated_win") if half
                                  else case.get("bitwise_round_trip") and case.get("accepted_custom_win"))
            and abs(case.get("aa_time_saved_pct",float("inf")))<=1
            and case["candidate_gain_ci95_pct"][0]>max(1,abs(case["aa_time_saved_pct"])) for case in cases)
    if passed and not force_vendor:
        return {"backend":"custom","config":"half_"+config if half else config,"reason":"All held-out A/A and positive-gain checks passed",
                "custom_win_validated":True,
                "accepted_10pct_target":all(case["candidate_gain_ci95_pct"][0]>10 for case in cases)}
    allowed=("cublas_compute32","cublas_packed_compute32","torch_mm_out_fp32") if half else VENDORS
    candidates=[name for name in allowed if all(name in case["methods"] for case in cases)]
    vendor=min(candidates,key=lambda name:statistics.median(case["methods"][name]["median_us"] for case in cases))
    return {"backend":"vendor","config":vendor,"reason":"Custom candidate did not pass every held-out noise gate",
            "custom_win_validated":False,"accepted_10pct_target":False}


def compressed_vendor_fallback(profile,shape,output_dtype):
    """Safe current-vendor route while the three-process dispatch cohort is pending.

    Training results and individual validation processes cannot promote a custom
    route. In particular, a negative small-shape result must not add codec work.
    No serialized vendor algorithm from the supplied profile is restored.
    """
    if output_dtype not in ("bf16","fp32"):
        raise ValueError("Compressed BF16 projections support BF16 or FP32 output")
    recorded_dtype=profile.get("bindings",{}).get("output_dtype",profile.get("output_dtype",output_dtype))
    if recorded_dtype!=output_dtype:
        reason="Profile output contract differs; use the current vendor for the requested output"
    elif any(tuple(case.get("shape_mnk",()))==tuple(shape)
             and case.get("time_saved_pct") is not None and case["time_saved_pct"]<0
             for case in profile.get("cases",())):
        reason="Compressed kernel regressed on this shape; skip packing and use the current vendor"
    else:
        reason="Custom dispatch requires the complete audited three-process held-out cohort"
    return {"backend":"vendor","config":"torch_mm" if output_dtype=="bf16" else "torch_mm_out_fp32",
            "reason":reason,"requires_packing":False,"custom_win_validated":False,
            "accepted_10pct_target":False,"dispatch_overhead_validated":False}
