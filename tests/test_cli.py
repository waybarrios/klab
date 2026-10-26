import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments
import experiment_control as control


class CliTests(unittest.TestCase):
    def test_default_run_preserves_strict_precision(self):
        args = experiments.parser().parse_args(["run"])
        self.assertEqual(args.problem, "fp32-projection")
        args = experiments.parser().parse_args(["compare"])
        self.assertFalse(args.include_lossy)

    def test_output_escape_rejected_before_cuda(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiments.main(["run", "--output", "../escape.json"]), 2)

    def test_problem_json_and_help_need_no_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / "experiments.py"), "problems", "--json"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["id"], "fp32-projection")
        result = subprocess.run([sys.executable, str(ROOT / "experiments.py"), "help", "run"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--shape", result.stdout)

    def test_native_launcher_propagates_exit_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            fake = Path(directory)
            (fake / "experiments.py").touch()
            (fake / "kernel_lab.py").touch()
            python = fake / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 23\n")
            python.chmod(0o755)
            result = subprocess.run([str(ROOT / "bin/klab"), "--project", str(fake), "jobs"], capture_output=True)
            self.assertEqual(result.returncode, 23)

    def test_native_launcher_uses_known_project_in_untrusted_cwd(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            fake=Path(directory)
            (fake/"experiments.py").touch()
            (fake/"kernel_lab.py").touch()
            python=fake/".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 23\n")
            python.chmod(0o755)
            env=dict(os.environ)
            env.pop("KERNEL_LAB_ROOT",None)
            result=subprocess.run([str(ROOT/"bin/klab"),"problems","--json"],cwd=fake,env=env,
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)[0]["id"],"fp32-projection")

    def test_checkpoint_retains_pid_and_detects_interruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            path = Path(directory) / "run.json"
            metadata = {"pid": child.pid, "run_id": "test-run", "command": ["run"], "created_utc": "test"}
            checkpoint = control.with_run_metadata({"status": "running", "cases": [{}]}, metadata)
            control.write_json(path, checkpoint)
            data = experiments.read_run(path)
            self.assertEqual(data["pid"], child.pid)
            self.assertEqual(data["status"], "interrupted")

    def test_cli_rejects_lock_and_markdown_destinations(self):
        for output in (".cache/gpu.lock", "results/accidental.md"):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(experiments.main(["run", "--output", output]), 2)

    def test_cli_rejects_overlapping_gpu_command(self):
        with control.GpuLock(ROOT / ".cache/gpu.lock"):
            result = subprocess.run([str(ROOT / "bin/klab"), "doctor", "--json"],
                                    cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Another GPU experiment", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_custom_shape_report_links_original_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            source = Path(directory) / "custom-run.json"
            destination = Path(directory) / "renamed-table.md"
            baseline = {"cold": {"median_us": 4.0}, "speedup_vs_best_fp32": 1.0,
                        "output_relative_l2_error": 1e-7}
            lossless = {**baseline, "cold": {"median_us": 3.0}, "speedup_vs_best_fp32": 4 / 3}
            control.write_json(source, {"status": "complete", "cases": [{
                "shape_mnk": [3, 65, 64], "best_vendor_us": 4.0, "best_vendor_method": "torch_mm",
                "methods": {"torch_mm": baseline, "lossless_fp32": lossless},
                "packing_setup_ms": 1.0, "lossless_bytes": 228, "original_bytes": 256,
                "packing_amortization_calls": 1000}]})
            experiments.write_report(source, destination)
            table = destination.read_text()
            self.assertIn("3 × 65 × 64", table)
            self.assertIn("[custom-run.json](custom-run.json)", table)
            self.assertEqual(json.loads(source.read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
