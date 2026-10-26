"""Local command router. Discovery/help/reporting do not import CUDA packages."""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiment_control import GpuLock, parse_shape, project_path, report_output, run_output, write_json

ROOT = Path(__file__).resolve().parent
VERSION = "0.2.0"
PROBLEMS = [
    {"id": "fp32-projection", "precision": "FP32 equivalent, no lossy weights",
     "description": "Lossless exponent packing vs cuBLAS GEMM/GEMV and PyTorch",
     "accepted": True, "constraint": "M,N,K positive; K divisible by 64"},
    {"id": "adaptive-projection", "precision": "FP32 equivalent, experimental candidates",
     "description": "Validate frozen choices across plain and lossless CUDA kernels",
     "accepted": True, "constraint": "K divisible by 64; configuration is selected in a separate tune run"},
    {"id":"half-projection","precision":"FP16/BF16 inputs; FP32 accumulation/output",
     "description":"Custom CUDA exact-widening kernels versus equivalent cuBLAS paths",
     "accepted":False,"constraint":"Separate precision contract; positive even K; tune and held-out validation"},
    {"id": "fp4-projection", "precision": "Lossy 4-bit weights, FP32 accumulation",
     "description": "Software E2M1/FP8 weight decode; accuracy diagnostic",
     "accepted": False, "constraint": "Fixed large-projection diagnostic suite"},
    {"id": "int8-projection", "precision": "Lossy INT8 weights, FP32 accumulation",
     "description": "INT8 weight decode; accuracy diagnostic",
     "accepted": False, "constraint": "Fixed large-projection diagnostic suite"},
]
HELP = """◆ klab · local GPU kernel experiments
  Python × C++ × CUDA · strict FP32 by default

usage  klab <command> [args]

get started
  doctor                        inspect the GPU, runtime and compilation cache
  problems                      list the local experiments
  show <problem>                describe precision, baseline and constraints

experiments
  check [problem]               numerical checks, no performance claims
  run [problem]                 run a benchmark (default: fp32-projection)
  compare                       compare baselines on common inputs
  tune                          search plain/lossless CUDA launch settings
  peak                          separate Tensor Core input-precision diagnostic
  stream                        validate frozen choices with rotating weights and A/A
  regime                        observe sustained bandwidth, launch floor and clocks
  corpus                        verify and retain selected real BF16 checkpoint tensors
  analyze-bf16                  screen exact BF16 storage formats on CPU
  vendor-tune                   freeze a strong vendor pool and screen headroom
  probe-compressed              tune-only real-weight lossless BF16 kernel probe

results
  jobs                          list local runs
  status <run>                  inspect a saved run
  report <run>                  generate a Markdown comparison table
  factorial                     separate kernel gains from GPU-unlock gains

system
  build                         build the C++ launcher and CUDA driver wrapper
  unlock-check                  read-only gpu-unlock compatibility inventory
  version                       print the CLI version

each command's options: klab help <command>
global options: --wait (queue GPU access), --rebuild (recompile PTX), --debug
aliases: bench=run · verify=check · list=jobs
"""


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def parser():
    p = argparse.ArgumentParser(prog="klab", description="Local GPU kernel experiments")
    p.add_argument("--wait", action="store_true", help="Wait for another GPU experiment")
    p.add_argument("--rebuild", action="store_true", help="Force NVRTC recompilation")
    p.add_argument("--debug", action="store_true", help="Print error tracebacks")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "problems", "jobs", "build", "unlock-check"):
        command = sub.add_parser(name, aliases=["list"] if name == "jobs" else [])
        command.add_argument("--json", action="store_true")
        if name == "jobs":
            command.add_argument("--limit", type=positive, default=20)
        if name == "unlock-check":
            command.add_argument("--output", help="Save read-only inventory as project JSON")
    show = sub.add_parser("show")
    show.add_argument("problem", choices=[p["id"] for p in PROBLEMS])
    show.add_argument("--json", action="store_true")
    regime = sub.add_parser("regime")
    regime.add_argument("--bytes-mib", type=positive, default=256)
    regime.add_argument("--sustain-seconds", type=positive, default=30)
    regime.add_argument("--output")
    regime.add_argument("--json", action="store_true")
    regime.set_defaults(rounds=15, shape=None, dtype="fp32")
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    corpus.add_argument("--output", default=".cache/real-weights/manifest.json")
    corpus.add_argument("--json", action="store_true")
    analysis = sub.add_parser("analyze-bf16")
    analysis.add_argument("--corpus", required=True)
    analysis.add_argument("--output", default="results/bf16-storage-screen.json")
    analysis.add_argument("--json", action="store_true")
    vendor_tune = sub.add_parser("vendor-tune")
    vendor_tune.add_argument("--spec", required=True)
    vendor_tune.add_argument("--corpus", required=True)
    vendor_tune.add_argument("--regime", default="results/regime-stock-v2.json")
    vendor_tune.add_argument("--storage", default="results/bf16-storage-screen.json")
    vendor_tune.add_argument("--rounds", type=positive, default=31)
    vendor_tune.add_argument("--screen-only", action="store_true")
    vendor_tune.add_argument("--output")
    vendor_tune.add_argument("--json", action="store_true")
    vendor_tune.set_defaults(shape=None, dtype="fp32")
    probe = sub.add_parser("probe-compressed")
    probe.add_argument("--spec", default="experiments/consumer_ampere_bf16.json")
    probe.add_argument("--corpus", default=".cache/real-weights/manifest.json")
    probe.add_argument("--vendor-profile", default="results/vendor-stock.json")
    probe.add_argument("--revision", type=positive, default=1)
    probe.add_argument("--output")
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(rounds=31, shape=None, dtype="bf16")
    check = sub.add_parser("check", aliases=["verify"])
    check.add_argument("problem", nargs="?", default="fp32-projection", choices=[p["id"] for p in PROBLEMS])
    check.add_argument("--all", action="store_true", help="Also verify lossy diagnostic implementations")
    check.add_argument("--json", action="store_true")
    for name in ("run", "compare", "tune", "peak", "stream"):
        command = sub.add_parser(name, aliases=["bench"] if name == "run" else [])
        if name == "run":
            command.add_argument("problem", nargs="?", default="fp32-projection", choices=[p["id"] for p in PROBLEMS])
            command.add_argument("--seed", type=int, default=857)
            command.add_argument("--profile", help="Frozen tune run; default is selected by dtype")
        elif name in ("compare", "tune", "stream"):
            command.add_argument("--seeds", default="901" if name=="tune" else "919,929" if name=="stream" else "857,859")
        if name in ("tune", "stream"):
            command.add_argument("--profile",help="Frozen tune run; default is selected by dtype")
            command.add_argument("--family", choices=("legacy", "compressed"), default="legacy")
            command.add_argument("--spec", default="experiments/consumer_ampere_bf16.json")
            command.add_argument("--corpus", default=".cache/real-weights/manifest.json")
            command.add_argument("--vendor-profile", default="results/vendor-stock.json")
            command.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
            command.add_argument("--revision", type=positive, default=1)
            command.add_argument("--diagnostic-training", action="store_true",
                                 help="Also measure frozen settings on the remaining M grid in training layer 0")
        if name in ("run","tune","stream"):
            command.add_argument("--dtype",choices=("fp32","fp16","bf16"),default="fp32")
        if name=="compare":
            command.add_argument("--include-lossy", action="store_true", help="Include rejected precision-tradeoff rows")
        if name=="peak":
            command.add_argument("--stage", choices=("before","after","diagnostic"), default="diagnostic")
        command.add_argument("--shape", type=parse_shape, action="append", help="M,N,K; repeat for multiple shapes")
        command.add_argument("--rounds", type=positive, default=31 if name in ("run","tune","stream") else 15,
                             help="At least 3 rounds; half and streaming validation use 31 by default")
        command.add_argument("--threads", type=int, choices=(32, 64, 128, 256))
        command.add_argument("--output", help="JSON output inside the project; default is a unique run ID")
        command.add_argument("--json", action="store_true", help="JSON on stdout, progress on stderr")
    for name in ("status", "report"):
        command = sub.add_parser(name)
        command.add_argument("run", help="Run ID or project-relative JSON path")
        if name == "report":
            command.add_argument("--output", help="Markdown output inside the project")
        command.add_argument("--json", action="store_true")
    sub.add_parser("version")
    factorial=sub.add_parser("factorial")
    factorial.add_argument("--stock",required=True,help="Stock projection JSON")
    factorial.add_argument("--unlocked",help="Unlocked projection JSON; omit for a half/compressed pending report")
    factorial.add_argument("--method",default="auto")
    factorial.add_argument("--output")
    factorial.add_argument("--json",action="store_true")
    return p


def emit(value, as_json=False):
    if as_json:
        print(json.dumps(value, indent=2))
    elif isinstance(value, str):
        print(value)
    else:
        for item in value:
            print(item)


def resolve_run(value):
    path = project_path(ROOT, value)
    if not path.is_file():
        path = project_path(ROOT, f"results/runs/{value}.json")
    if not path.is_file():
        raise ValueError(f"Unknown run: {value}")
    return path


def read_run(path):
    data = json.loads(path.read_text())
    if data.get("status") == "running" and data.get("pid"):
        try:
            os.kill(data["pid"], 0)
        except ProcessLookupError:
            data["status"] = "interrupted"
    return data


def write_report(path, destination=None):
    data = read_run(path)
    if data.get("status") != "complete":
        raise ValueError("A report requires a completed run")
    output = report_output(ROOT, path, destination)
    if data.get("kind") == "vendor-tuning":
        from bench_vendor import make_report
        return make_report(data, path, output)
    if data.get("kind") == "compressed-projection-probe":
        from bench_compressed import make_report
        return make_report(data, path, output)
    if data.get("kind") == "compressed-projection":
        from bench_compressed import make_campaign_report
        return make_campaign_report(data, path, output)
    if data.get("kind")=="tensorcore-diagnostic":
        from bench_tensorcore import make_report
        return make_report(data,path,output)
    if data.get("kind")=="adaptive-fp32":
        from bench_adaptive import make_report
        return make_report(data,path,output)
    if data.get("kind")=="streaming-projection":
        from bench_streaming import make_report
        return make_report(data,path,output)
    if data.get("kind")=="half-projection":
        from bench_half import make_report
        return make_report(data,path,output)
    if data.get("cases") and "methods" in data["cases"][0]:
        from compare_baselines import make_tables
        return make_tables(data, output, raw_source=path)
    raise ValueError("Table generation expects a baseline comparison run (klab run or klab compare)")


def doctor():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this runner does not benchmark on CPU")
    import kernel_lab
    from bench import gpu_state
    module = kernel_lab.default_lab()
    return {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu_state(),
            "native_launcher": "C++", "device_kernels": "CUDA C++",
            "precision_default": "FP32 equivalent; no lossy weights", "compilation": module.compilation}


def perform_check(args):
    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    problems = PROBLEMS if args.all else [p for p in PROBLEMS if p["id"] == args.problem]
    results = []
    for problem in problems:
        if problem["id"] == "fp32-projection":
            from bench_lossless import verify
        elif problem["id"] == "adaptive-projection":
            from verify_adaptive import verify
        elif problem["id"]=="half-projection":
            from bench_half import verify
        elif problem["id"] == "fp4-projection":
            from bench_fp4 import verify_fp4 as verify
        else:
            from bench_int8 import verify_int8 as verify
        results.append({"problem": problem["id"], "status": "passed", "checks": verify()})
    return results


def perform_run(args, output):
    if getattr(args, "family", None) == "compressed":
        from bench_compressed import run
        return run(args.spec, args.corpus, args.vendor_profile,
                   mode="tune" if args.command == "tune" else "validate",
                   seed=int(args.seeds), profile_path=args.profile, output_path=output,
                   output_dtype=args.output_dtype, revision=args.revision, rounds=args.rounds,
                   run_metadata=args.run_metadata, diagnostic_training=args.diagnostic_training)
    if args.command == "probe-compressed":
        from bench_compressed import run_probe
        return run_probe(args.spec, args.corpus, args.vendor_profile, output,
                         revision=args.revision, run_metadata=args.run_metadata)
    if args.command == "vendor-tune":
        from bench_vendor import run
        return run(args.spec, args.corpus, output, regime_path=args.regime,
                   storage_path=args.storage, rounds=args.rounds, screen_only=args.screen_only,
                   run_metadata=args.run_metadata)
    if args.command == "regime":
        from bench_bandwidth import measure_regime
        return measure_regime(args.bytes_mib, args.sustain_seconds)
    if (args.command in ("tune","stream") and args.dtype!="fp32") or (args.command=="run" and args.problem=="half-projection"):
        from bench_half import run,SHAPES
        dtype=args.dtype if args.dtype!="fp32" else "fp16"
        tuning=args.command=="tune"
        seeds=[int(s) for s in args.seeds.split(",")] if args.command in ("tune","stream") else [args.seed]
        if args.rounds<31: raise ValueError("Half precision runs require at least 31 timing rounds")
        profile=project_path(ROOT,args.profile or f"results/{dtype}-search.json") if not tuning else None
        return run(args.shape or SHAPES,seeds,args.rounds,output,dtype,profile,args.run_metadata)
    if args.command=="stream":
        from bench_streaming import run
        from bench_adaptive import SHAPES
        return run(args.shape or SHAPES,[int(s) for s in args.seeds.split(",")],args.rounds,output,
                   project_path(ROOT,args.profile or "results/adaptive-search.json"),args.run_metadata)
    if args.command=="peak":
        from bench_tensorcore import run,DEFAULT_SHAPES
        return run(args.shape or DEFAULT_SHAPES,args.rounds,args.stage,output,args.run_metadata)
    if args.command=="tune" or (args.command=="run" and args.problem=="adaptive-projection"):
        from bench_adaptive import run,SHAPES
        tuning=args.command=="tune"
        seeds=[int(s) for s in args.seeds.split(",")] if tuning else [args.seed]
        return run(args.shape or SHAPES,seeds,args.rounds,output,
                   None if tuning else project_path(ROOT,args.profile or "results/adaptive-search.json"),args.run_metadata)
    if args.command in ("compare",) or args.problem == "fp32-projection":
        from compare_baselines import run_comparison
        comparing = args.command == "compare"
        shapes = args.shape or ([(1, 4096, 4096), (1, 11008, 4096), (1, 4096, 11008)]
                                if comparing else [(1, 4096, 4096)])
        seeds = [int(s) for s in args.seeds.split(",")] if comparing else [args.seed]
        return run_comparison(shapes, seeds, args.rounds, args.include_lossy if comparing else False,
                              output, threads=args.threads, run_metadata=args.run_metadata)
    if args.shape or args.threads:
        raise ValueError("Custom shape/threads currently apply to fp32-projection only")
    import importlib
    module = importlib.import_module("bench_fp4" if args.problem == "fp4-projection" else "bench_int8")
    previous = sys.argv
    try:
        sys.argv = [module.__name__, "--seed", str(args.seed), "--output", str(output)]
        module.main(run_metadata=args.run_metadata)
    finally:
        sys.argv = previous
    return json.loads(output.read_text())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["--help"] or argv == ["-h"] or argv == ["help"]:
        print(HELP)
        return 0
    if argv[0] == "help":
        argv = argv[1:] + ["--help"]
    args = parser().parse_args(argv)
    args.command = {"bench": "run", "verify": "check", "list": "jobs"}.get(args.command, args.command)
    try:
        if args.command == "version":
            print(f"klab {VERSION}")
            return 0
        if args.command in ("corpus", "analyze-bf16"):
            output = run_output(ROOT, args.output)
            if args.command == "corpus":
                from real_weights import fetch_corpus
                result = fetch_corpus(args.model, output)
                summary = f"complete: {len(result['tensors'])} tensors, {result['retained_bytes']/1024**2:.1f} MiB retained"
            else:
                source = project_path(ROOT, args.corpus)
                from analyze_bf16 import analyze
                result = analyze(json.loads(source.read_text()))
                write_json(output, result)
                summary = f"complete: {len(result['cases'])} tensors; selected format {result['selected_format']}"
            emit(result if args.json else summary + f"\n{output}", args.json)
            return 0
        if args.command=="unlock-check":
            from gpu_unlock_status import inspect
            output=run_output(ROOT,args.output) if args.output else None
            with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
                result=inspect()
            if args.output:
                write_json(output,result)
            emit(result if args.json else [f"{key}: {value}" for key,value in result.items()],args.json)
            return 0
        if args.command=="factorial":
            from factorial_results import report
            before=resolve_run(args.stock)
            after=resolve_run(args.unlocked) if args.unlocked else None
            suffix=f"vs-{after.stem}" if after is not None else "pending-unlocked"
            output=report_output(ROOT,before,args.output or f"results/factorial-{before.stem}-{suffix}.md")
            rows=report(before,after,output,args.method)
            emit({"report":str(output.relative_to(ROOT)),"cases":rows} if args.json else str(output),args.json)
            return 0
        if args.command=="build":
            with GpuLock(ROOT/".cache/gpu.lock",wait=args.wait):
                completed=subprocess.run(["make","-j2"],cwd=ROOT,text=True,capture_output=True,check=True)
            if args.json:
                emit({"status":"complete","backend":"C++/CUDA","build_log":completed.stdout},True)
            else:
                print(completed.stdout,end="")
                print("C++ launcher and CUDA driver wrapper ready")
            return 0
        if args.command == "problems":
            emit(PROBLEMS if args.json else [f"{p['id']:20} {p['precision']}" for p in PROBLEMS], args.json)
            return 0
        if args.command == "show":
            p = next(p for p in PROBLEMS if p["id"] == args.problem)
            emit(p if args.json else "\n".join(f"{k}: {v}" for k, v in p.items()), args.json)
            return 0
        if args.command == "jobs":
            jobs = []
            files = sorted((ROOT / "results").rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in files[:args.limit]:
                try:
                    data = read_run(path)
                    jobs.append({"id": path.stem, "path": str(path.relative_to(ROOT)),
                                 "status": data.get("status", "unknown"), "cases": len(data.get("cases", []))})
                except (ValueError, OSError):
                    continue
            emit(jobs if args.json else [f"{j['id']:40} {j['status']:12} {j['cases']} cases" for j in jobs], args.json)
            return 0
        if args.command in ("status", "report"):
            path = resolve_run(args.run)
            if args.command == "report":
                destination = write_report(path, args.output)
                emit({"report": str(destination.relative_to(ROOT))} if args.json else str(destination), args.json)
            else:
                data = read_run(path)
                emit(data if args.json else f"{path.stem}: {data.get('status')} · {len(data.get('cases', []))} cases\n{path}", args.json)
            return 0
        output = None
        if args.command in ("run", "compare", "tune", "peak", "stream", "regime", "vendor-tune", "probe-compressed"):
            output = run_output(ROOT, args.output or ("results/runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"))
            if getattr(args, "family", None) == "compressed":
                for name in ("spec", "corpus", "vendor_profile"):
                    project_path(ROOT, getattr(args, name))
                if args.dtype != "bf16" or args.shape or args.threads:
                    raise ValueError("Compressed experiments require original BF16 inputs and the registered shape grid")
                if args.rounds < 31 or args.revision > 3:
                    raise ValueError("Compressed experiments require at least 31 rounds and at most three revisions")
                if len(args.seeds.split(",")) != 1:
                    raise ValueError("Compressed experiments use one seed per fresh CLI process")
                if args.command == "stream" and not args.profile:
                    raise ValueError("Compressed validation requires an explicit frozen --profile")
                args.profile = str(run_output(ROOT, args.profile or output.with_stem(output.stem + "-profile")))
                if output.exists() or (args.command == "tune" and Path(args.profile).exists()):
                    raise ValueError("Preserve prior measurements: choose new output/profile paths")
                if args.diagnostic_training and args.command != "tune":
                    raise ValueError("--diagnostic-training applies only to tuning")
            if args.command == "vendor-tune":
                for name in ("spec", "corpus", "regime", "storage"):
                    project_path(ROOT, getattr(args, name))
            if args.command == "probe-compressed":
                for name in ("spec", "corpus", "vendor_profile"):
                    project_path(ROOT, getattr(args, name))
                if args.revision > 3:
                    raise ValueError("At most three preregistered performance revisions")
            if args.rounds < 3:
                raise ValueError("Use at least 3 timing rounds")
            if args.command=="stream" and args.rounds<7:
                raise ValueError("Streaming validation needs at least 7 timing rounds")
            half_mode=getattr(args,"dtype","fp32")!="fp32" or (args.command=="run" and args.problem=="half-projection")
            if args.command!="peak" and not half_mode and args.shape and any(k % 64 for _, _, k in args.shape):
                raise ValueError("FP32 lossless comparisons require K divisible by 64")
            if args.command in ("compare","tune","stream"):
                if not args.seeds.strip():
                    raise ValueError("At least one seed is required")
                [int(s) for s in args.seeds.split(",")]
        if args.rebuild:
            os.environ["KERNEL_LAB_REBUILD"] = "1"
        context = contextlib.redirect_stdout(sys.stderr) if getattr(args, "json", False) else contextlib.nullcontext()
        with GpuLock(project_path(ROOT, ".cache/gpu.lock"), wait=args.wait), context:
            if args.command == "doctor":
                result = doctor()
            elif args.command == "check":
                result = perform_check(args)
            else:
                metadata = {"run_id": output.stem, "pid": os.getpid(), "command": argv,
                            "created_utc": datetime.now(timezone.utc).isoformat()}
                args.run_metadata = metadata
                if getattr(args, "family", None) != "compressed":
                    write_json(output, {**metadata, "status": "running", "cases": []})
                print(f"run {output.stem}\n{output}", flush=True)
                start = time.monotonic()
                try:
                    result = perform_run(args, output)
                except BaseException as error:
                    try:
                        partial = json.loads(output.read_text())
                    except (ValueError, OSError):
                        partial = {}
                    partial.update(metadata, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                   error=str(error), elapsed_seconds=time.monotonic() - start)
                    write_json(output, partial)
                    raise
                if result.get("kind") != "compressed-projection":
                    result.update(metadata, elapsed_seconds=time.monotonic() - start,
                                  finished_utc=datetime.now(timezone.utc).isoformat())
                if (args.command not in ("regime", "vendor-tune", "probe-compressed")
                        and result.get("kind") not in ("tensorcore-diagnostic", "compressed-projection")):
                    import kernel_lab
                    result["compilation"] = kernel_lab.default_lab().compilation
                write_json(output, result)
                if result.get("cases") and "methods" in result["cases"][0]:
                    write_report(output)
        if getattr(args, "json", False):
            emit(result, True)
        elif args.command == "doctor":
            emit([f"{k}: {v}" for k, v in result.items()])
        elif args.command == "check":
            emit([f"{r['problem']}: {r['status']} · {r['checks']} checks" for r in result])
        else:
            print(f"complete · {output}")
            if result.get("cases") and "methods" in result["cases"][0]:
                print(f"report: {output.with_suffix('.md')}")
        return 0
    except KeyboardInterrupt:
        print("klab: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(f"klab: {error}", file=sys.stderr)
        return 2 if isinstance(error, ValueError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
