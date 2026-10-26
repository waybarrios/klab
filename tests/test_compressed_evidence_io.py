"""File-provenance tests use synthetic data, not hardware measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_compressed_dispatch_evidence import fixture

ROOT=Path(__file__).resolve().parents[1]


class CompressedEvidenceIOTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=ROOT/".scratch")
        self.addCleanup(self.directory.cleanup)
        self.base=Path(self.directory.name)
        spec,corpus,profile,reports=fixture()
        training=copy.deepcopy(reports[0])
        training.update(mode="tune",seed=901,pid=profile["pid"],run_id=profile["run_id"],
                        choices=profile["choices"],minimum_12pct_tuning_screen=True,cases=[])
        for layer in spec["tuning_layers"]:
            for entry,case in zip(corpus["tensors"][:2],reports[0]["cases"][:2]):
                entry=copy.deepcopy(entry)
                entry["name"]=entry["name"].replace("layers.8.",f"layers.{layer}.")
                corpus["tensors"].append(entry)
                case=copy.deepcopy(case)
                case.update(weight_tensor=entry["name"],seed=901,layer=layer,role="training")
                training["cases"].append(case)
        self.spec=self.put("spec.json",spec)
        self.corpus=self.put("corpus.json",corpus)
        bindings=profile["bindings"]
        bindings.update(spec_sha256=self.digest(self.spec),corpus_sha256=self.digest(self.corpus))
        # The source is a tiny labelled synthetic dependency, not a CUDA binary.
        source=self.base/"synthetic-source.txt"
        source.write_text("Synthetic file-provenance fixture only\n")
        bindings["identity"]["source_sha256"]={str(source.relative_to(ROOT)):self.digest(source)}
        for report in [training,*reports]:
            report.update(copy.deepcopy(bindings))
        self.training=self.put("training.json",training)
        profile.update(training_report=str(self.training.relative_to(ROOT)),
                       training_report_sha256=self.digest(self.training))
        self.profile=self.put("profile.json",profile)
        self.reports=[]
        for report in reports:
            report["profile_sha256"]=self.digest(self.profile)
            self.reports.append(self.put(f"report-{report['seed']}.json",report))
        self.output=self.base/"evidence.json"

    def put(self,name,data):
        path=self.base/name
        path.write_text(json.dumps(data))
        return path

    def digest(self,path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def create(self,**changes):
        from compressed_evidence_io import create_evidence
        args=dict(spec_path=self.spec,corpus_path=self.corpus,profile_path=self.profile,
                  report_paths=self.reports,output_path=self.output)
        args.update(changes)
        return create_evidence(**args)

    def test_complete_cohort_records_every_input_digest_without_promoting_dispatch(self):
        data=self.create()
        self.assertEqual(data,json.loads(self.output.read_text()))
        self.assertEqual(data["inputs"]["profile"]["sha256"],self.digest(self.profile))
        self.assertEqual(len(data["inputs"]["reports"]),3)
        self.assertTrue(data["shapes"]["1,64,64"]["kernel_gate_passed"])
        self.assertFalse(data["shapes"]["1,64,64"]["custom_win_validated"])

    def test_changed_training_bytes_cannot_retain_old_profile_identity(self):
        self.training.write_text(self.training.read_text()+"\n")
        with self.assertRaises(ValueError): self.create()
        self.assertFalse(self.output.exists())

    def test_changed_spec_bytes_are_rejected_before_writing(self):
        self.spec.write_text(self.spec.read_text()+"\n")
        with self.assertRaises(ValueError): self.create()
        self.assertFalse(self.output.exists())

    def test_current_source_change_invalidates_the_evidence(self):
        (self.base/"synthetic-source.txt").write_text("Changed dependency\n")
        with self.assertRaises(ValueError): self.create()

    def test_existing_output_and_input_alias_are_preserved(self):
        self.output.write_text("Keep this result\n")
        with self.assertRaises(ValueError): self.create()
        self.assertEqual(self.output.read_text(),"Keep this result\n")
        original=self.profile.read_bytes()
        with self.assertRaises(ValueError): self.create(output_path=self.profile)
        self.assertEqual(self.profile.read_bytes(),original)

    def test_paths_cannot_escape_the_project(self):
        with self.assertRaises(ValueError): self.create(output_path=ROOT.parent/"outside-evidence.json")


if __name__=="__main__":
    unittest.main()
