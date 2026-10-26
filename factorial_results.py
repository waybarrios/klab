"""Separate kernel and GPU-state effects; reject incomplete causal evidence."""

import json
import math
from pathlib import Path

from experiment_provenance import CUDA_SOURCES
from result_metrics import percentage_gains

ROOT=Path(__file__).resolve().parent
PROJECTION_KINDS=("compressed-projection","half-projection")


def gains(stock_baseline,stock_custom,unlocked_baseline,unlocked_custom):
    return {"our_gain_stock_pct":percentage_gains(stock_baseline,stock_custom)["latency_reduction_pct"],
            "our_gain_unlocked_pct":percentage_gains(unlocked_baseline,unlocked_custom)["latency_reduction_pct"],
            "unlock_gain_vendor_pct":percentage_gains(stock_baseline,unlocked_baseline)["latency_reduction_pct"],
            "combined_gain_pct":percentage_gains(stock_baseline,unlocked_custom)["latency_reduction_pct"]}


def confidence(baseline,candidate,paired=True):
    import numpy as np
    b,c=np.asarray(baseline),np.asarray(candidate)
    if b.size<7 or c.size<7 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)) or np.any(b<=0) or np.any(c<=0):
        raise ValueError("At least seven positive finite samples required for confidence intervals")
    rng=np.random.default_rng(1009)
    bi=rng.integers(0,b.size,(2000,b.size))
    ci=bi if paired and b.size==c.size else rng.integers(0,c.size,(2000,c.size))
    values=100*(1-np.median(c[ci],axis=1)/np.median(b[bi],axis=1))
    return [float(v) for v in np.percentile(values,[2.5,97.5])]


def check_environment(stock,unlocked):
    for data in (stock,unlocked):
        if data.get("schema_version",0)<2:
            raise ValueError("Legacy reports lack causal-comparison evidence; rerun with the current CLI")
    a,b=stock.get("environment",{}),unlocked.get("environment",{})
    for key in ("torch","cuda","cublas_version","compilation_cache_key"):
        if not a.get(key) or a.get(key)!=b.get(key):
            raise ValueError(f"Environment differs or is missing: {key}")
    ai,bi=a.get("inventory",{}),b.get("inventory",{})
    for key in ("gpu","kernel","pci_address"):
        if not ai.get(key) or ai.get(key)!=bi.get(key):
            raise ValueError(f"GPU/driver/kernel identity differs or is missing: {key}")
    if ai.get("module_loaded") is not False or ai.get("unlock_boot_option_present") is not False or ai.get("unlock_success_logged"):
        raise ValueError("Stock report contains an unlock module, boot option, or unlock-success log")
    if not all(bi.get(key) is True for key in
               ("module_loaded","unlock_boot_option_present","unlock_success_logged","post_flr_selectors_logged")):
        raise ValueError("Unlocked report requires module/boot state AND unlock-success/post-FLR selector log evidence")


def indexed(data):
    cases=data["cases"]
    result={(c["seed"],tuple(c["shape_mnk"])):c for c in cases}
    if len(result)!=len(cases): raise ValueError("Duplicate shape/seed cases are not allowed")
    return result


def statistics_for(case,name,kind):
    method=case["methods"][name]
    return method["cold"] if kind=="controlled-comparison" else method


def compare(stock,unlocked,method="auto"):
    if stock.get("kind") in PROJECTION_KINDS or unlocked.get("kind") in PROJECTION_KINDS:
        from factorial_projection import compare_projection
        return compare_projection(stock,unlocked,method)
    if stock.get("status")!="complete" or unlocked.get("status")!="complete":
        raise ValueError("Both reports must be complete")
    kind=stock.get("kind")
    if kind not in ("controlled-comparison","streaming-projection") or kind!=unlocked.get("kind"):
        raise ValueError("Use matching FP32 projection reports; Tensor Core peaks are separate diagnostics")
    if not stock.get("run_id") or not unlocked.get("run_id") or stock["run_id"]==unlocked["run_id"]:
        raise ValueError("Distinct stock/unlocked run IDs required")
    required=set(CUDA_SOURCES)|{"native/driver.cpp","native/build/libkernel_lab.so","bench.py","cublas_baseline.py"}
    hashes=stock.get("source_sha256",{})
    if not required.issubset(hashes) or hashes!=unlocked.get("source_sha256"):
        raise ValueError("Complete source/build fingerprints must match")
    if stock.get("parameters")!=unlocked.get("parameters") or stock.get("profile_sha256")!=unlocked.get("profile_sha256"):
        raise ValueError("Benchmark parameters or frozen profiles differ")
    check_environment(stock,unlocked)
    left,right=indexed(stock),indexed(unlocked)
    if not left or left.keys()!=right.keys(): raise ValueError("Matching nonempty shapes and seeds required")
    method=("candidate" if kind=="streaming-projection" else "lossless_fp32") if method=="auto" else method
    rows=[]
    for key,before in left.items():
        after=right[key]
        if before.get("selected")!=after.get("selected") or before.get("selected_threads")!=after.get("selected_threads"):
            raise ValueError("Frozen custom configurations differ")
        for case in (before,after):
            values=case["methods"][method]
            error=values.get("output_relative_l2_error",math.inf)
            if (not values.get("meets_precision_requirement") or not case.get("bitwise_round_trip")
                    or not math.isfinite(error) or error>1e-5):
                raise ValueError("Original FP32 weight reconstruction or output-accuracy requirement failed")
            if abs(case.get("aa_time_saved_pct",math.inf))>1:
                raise ValueError("A/A noise floor exceeds 1%; attribution is not accepted")
        vendor_key="best_vendor_method" if kind=="controlled-comparison" else "best_vendor"
        original_vendor=before[vendor_key]
        unlocked_vendor=after[vendor_key]
        ab=statistics_for(before,original_vendor,kind)
        ac=statistics_for(before,method,kind)
        ub=statistics_for(after,unlocked_vendor,kind)
        uc=statistics_for(after,method,kind)
        fixed=statistics_for(after,original_vendor,kind)
        values=gains(ab["median_us"],ac["median_us"],ub["median_us"],uc["median_us"])
        values["unlock_gain_vendor_pct"]=percentage_gains(ab["median_us"],fixed["median_us"])["latency_reduction_pct"]
        cis={"our_stock":confidence(ab["samples_us"],ac["samples_us"]),
             "our_unlocked":confidence(ub["samples_us"],uc["samples_us"]),
             "unlock_fixed_vendor":confidence(ab["samples_us"],fixed["samples_us"],False),
             "combined":confidence(ab["samples_us"],uc["samples_us"],False)}
        rows.append({"seed":key[0],"shape_mnk":list(key[1]),"stock_best_vendor":original_vendor,
                     "unlocked_best_vendor":unlocked_vendor,"unlock_reference_vendor":original_vendor,
                     "stock_vendor_us":ab["median_us"],"stock_custom_us":ac["median_us"],
                     "unlocked_vendor_us":ub["median_us"],"unlocked_custom_us":uc["median_us"],
                     "unlocked_fixed_vendor_us":fixed["median_us"],"ci95_pct":cis,
                     "stock_10pct_target":cis["our_stock"][0]>10,**values})
    return rows


def _projection_markdown(rows,*,pending):
    """Render actual cells, retaining pending states and unqualified outcomes."""
    missing="pending" if pending else "unqualified"
    def number(value,digits=3):
        return missing if value is None else f"{value:.{digits}f}"
    def percentage(value):
        return missing if value is None else f"{value:+.2f}%"
    synthetic=any(row.get("synthetic_fixture") for row in rows)
    lines=["# Kernel and GPU unlock comparison","",
        ("**Synthetic fixture: arithmetic and schema checks only, no hardware performance measurements.**" if synthetic else
         "Saved hardware measurements; no missing state is fabricated."),"",
        ("Unlocked measurements are **pending**. Only the supplied stock cells are shown."
         if pending else
         f"{sum(row['attribution_qualified'] for row in rows)}/{len(rows)} paired case comparisons pass the report checks; failures remain visible."),
        "","Inputs retain their supplied FP16/BF16 values, with FP32 accumulation and the stated output contract.",
        "This pairwise report does not establish the separate three-process kernel milestone or full-model speedup.","",
        "| Tensor | M,N,K | Seed | Output | Stock vendor µs | Stock custom µs | Unlocked vendor µs | Unlocked custom µs | Status |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        state=row.get("comparison_status",missing)
        lines.append(f"| {row.get('weight_tensor') or 'synthetic'} | {','.join(map(str,row['shape_mnk']))} | "
            f"{row['seed']} | {row['output_dtype']} | "+" | ".join(number(row.get(key)) for key in
            ("stock_vendor_us","stock_custom_us","unlocked_vendor_us","unlocked_custom_us"))+f" | {state} |")
    lines += ["","Percentages mean latency saved. Unqualified rows are descriptive observations, not accepted attribution.","",
        "| Tensor | M,N,K | Kernel, stock | Kernel, unlocked | Unlock, fixed vendor | Combined | Interaction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row.get('weight_tensor') or 'synthetic'} | {','.join(map(str,row['shape_mnk']))} | "+
            " | ".join(percentage(row.get(key)) for key in ("our_gain_stock_pct","our_gain_unlocked_pct",
                "unlock_gain_vendor_pct","combined_gain_pct","interaction_extra_latency_reduction_pct"))+" |")
    lines += ["","Within-state kernel gains use each state's fastest numerically eligible tested vendor.",
        "Unlock-only gain keeps the stock-selected vendor route and algorithm fixed.",
        "Combined gain compares stock vendor with unlocked custom; percentages are not added.",
        "Interaction compares custom and fixed-vendor unlock ratios; zero means matching relative effects.",
        "Cross-state intervals resample states independently. Raw route names, intervals and failures are returned by --json.",
        "Recorded profile/AC, clocks, software unlock logs, runtime and binary bindings are checked; this is not an energy-efficiency measurement."]
    if any(row.get("role")=="head-transfer-diagnostic" for row in rows):
        lines += ["Output-head rows are untuned transfer diagnostics and cannot replace the primary projection gates."]
    failed=[row for row in rows if row.get("failure_reasons")]
    if failed:
        lines += ["","Case qualifications:"]
        for row in failed:
            lines.append(f"- {row.get('weight_tensor') or row['shape_mnk']}, seed {row['seed']}: "+
                         "; ".join(row["failure_reasons"]))
    return "\n".join(lines)+"\n"


def report(stock_path,unlocked_path,output,method="auto"):
    from experiment_control import project_path
    before=Path(stock_path).resolve()
    after=Path(unlocked_path).resolve() if unlocked_path is not None else None
    output=project_path(ROOT,output)
    if output.suffix.lower()!=".md" or output in (before,after) or output.exists():
        raise ValueError("Preserve raw evidence and existing reports; choose a new project-local Markdown output")
    if before==after:
        raise ValueError("Stock and unlocked reports must be distinct files")
    stock=json.loads(before.read_text())
    if stock.get("kind") in PROJECTION_KINDS:
        if after is None:
            from factorial_projection import pending_projection
            rows=pending_projection(stock,method)
        else:
            rows=compare(stock,json.loads(after.read_text()),method)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(_projection_markdown(rows,pending=after is None))
        return rows
    if after is None:
        raise ValueError("Stock-only pending reports currently require half-projection or compressed-projection evidence")
    unlocked=json.loads(after.read_text())
    rows=compare(stock,unlocked,method)
    lines=["# Separate kernel, unlock, and combined effects","",
           "All percentages mean operation time saved. Code/build fingerprints, GPU UUID/driver/kernel, precision,",
           "software versions, software unlock state, selector log evidence, frozen choices, and A/A checks passed.","",
           "| M × N × K | Seed | Our gain, stock (%) | Our gain, unlocked (%) | Unlock, fixed vendor (%) | Combined (%) |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| "+" × ".join(map(str,row["shape_mnk"]))+f" | {row['seed']} | "+
                     " | ".join(f"{row[key]:+.2f}" for key in
                                ("our_gain_stock_pct","our_gain_unlocked_pct","unlock_gain_vendor_pct","combined_gain_pct"))+" |")
    lines += ["","Our gains compare against the fastest tested vendor in the SAME state.",
              "Unlock-only gain holds the stock-selected vendor route fixed, even if the fastest route changes.",
              "Combined gain compares stock vendor with unlocked custom. Percentages must not be added.",
              "Bootstrap intervals and route names are returned by --json; >10% task acceptance uses the stock CI lower bound.",
              "Manual controls still unverified: matched AC power/profile, thermal starting state, selector persistence through the run.",
              "These are device-operation measurements; setup, Python dispatch and full-model throughput are outside scope.", ""]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("\n".join(lines))
    return rows
