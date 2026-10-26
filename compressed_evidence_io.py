"""Bind offline per-shape evidence to the exact raw files and frozen training.

This does not execute a benchmark or promote a prepared custom route. Runtime
identity and separate dispatch-scope validation remain required before use.
"""
import argparse
import hashlib
import json
from pathlib import Path

from experiment_control import project_path,write_json

ROOT=Path(__file__).resolve().parent


def _digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            result.update(chunk)
    return result.hexdigest()


def _reference(path):
    return {"path":str(path.relative_to(ROOT)),"sha256":_digest(path)}


def create_evidence(*,spec_path,corpus_path,profile_path,report_paths,output_path):
    """Verify all input bytes, replay raw gates, then write one new artifact."""
    from compressed_campaign import assert_complete,freeze_choices
    from compressed_dispatch_evidence import BINDING_FIELDS,build_evidence

    spec_path,corpus_path,profile_path,output_path=[project_path(ROOT,p) for p in
        (spec_path,corpus_path,profile_path,output_path)]
    report_paths=[project_path(ROOT,p) for p in report_paths]
    inputs=[spec_path,corpus_path,profile_path,*report_paths]
    if len(inputs)!=len(set(inputs)) or output_path in inputs or output_path.exists():
        raise ValueError("Use distinct input files and preserve existing output artifacts")
    spec,corpus,profile=[json.loads(path.read_text()) for path in inputs[:3]]
    try:
        bindings=profile["bindings"]
        if _digest(spec_path)!=bindings["spec_sha256"] or _digest(corpus_path)!=bindings["corpus_sha256"]:
            raise ValueError("Spec/corpus bytes differ from the frozen training bindings")
        training_path=project_path(ROOT,profile["training_report"])
        if training_path in inputs or training_path==output_path:
            raise ValueError("Training evidence must have its own preserved file")
        if _digest(training_path)!=profile["training_report_sha256"]:
            raise ValueError("Training report bytes differ from the frozen profile")
        training=json.loads(training_path.read_text())
        if (training.get("status")!="complete" or training.get("mode")!="tune"
                or training.get("seed")!=spec["tuning_seed"]
                or training.get("pid")!=profile["pid"] or training.get("run_id")!=profile["run_id"]):
            raise ValueError("Training report is incomplete or has a different process identity")
        if any(training.get(key)!=bindings[key] for key in BINDING_FIELDS):
            raise ValueError("Training runtime/input/output bindings differ from its frozen profile")
        assert_complete(training,spec,corpus)
        tuning=[case for case in training["cases"] if case["role"]=="training"]
        choices=freeze_choices(tuning,spec,output_dtype=bindings["output_dtype"])
        if choices!=profile["choices"] or training["choices"]!=choices:
            raise ValueError("Frozen choices do not reproduce from the complete training cases")
        sources=bindings["identity"]["source_sha256"]
        if not isinstance(sources,dict) or not sources:
            raise ValueError("Missing execution-dependency fingerprints")
        for name,digest in sources.items():
            if _digest(project_path(ROOT,name))!=digest:
                raise ValueError(f"Execution dependency changed: {name}")
        reports=[json.loads(path.read_text()) for path in report_paths]
        result=build_evidence(reports,spec,corpus,profile,profile_sha256=_digest(profile_path))
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError(f"Malformed evidence inputs: {error!r}") from error

    result["inputs"]={"spec":_reference(spec_path),"corpus":_reference(corpus_path),
        "profile":_reference(profile_path),"training":_reference(training_path),
        "reports":[_reference(path) for path in report_paths]}
    result["builder_source_sha256"]={name:_digest(ROOT/name) for name in
        ("compressed_evidence_io.py","compressed_dispatch_evidence.py")}
    result["runtime_verification"]="Required before any prepared custom promotion"
    write_json(output_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",default="experiments/consumer_ampere_bf16.json")
    parser.add_argument("--corpus",default=".cache/real-weights/manifest.json")
    parser.add_argument("--profile",required=True)
    parser.add_argument("--reports",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    try:
        result=create_evidence(spec_path=args.spec,corpus_path=args.corpus,profile_path=args.profile,
            report_paths=args.reports,output_path=args.output)
    except (OSError,ValueError) as error:
        parser.error(str(error))
    passed=sum(shape["kernel_gate_passed"] for shape in result["shapes"].values())
    print(f"Built {len(result['cases'])} raw-case summaries; {passed}/{len(result['shapes'])} shape kernel gates pass")
    print(f"Primary kernel milestone: {result['kernel_milestone_passed']}; prepared dispatch still pending")


if __name__=="__main__":
    main()
