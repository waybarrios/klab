import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments


class CampaignCliTests(unittest.TestCase):
    def test_regime_parameters_are_exposed_without_importing_gpu_code(self):
        args = experiments.parser().parse_args(["regime", "--bytes-mib", "256", "--sustain-seconds", "30"])
        self.assertEqual((args.bytes_mib, args.sustain_seconds), (256, 30))
        result = subprocess.run([sys.executable, "-c",
                                 "import sys,experiments; experiments.main(['help','regime'])"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sustain-seconds", result.stdout)

    def test_new_outputs_cannot_escape_before_gpu_or_network(self):
        calls = (["regime", "--output", "../escape.json"],
                 ["corpus", "--output", "../escape.json"],
                 ["analyze-bf16", "--corpus", "../manifest.json", "--output", "results/unused.json"],
                 ["vendor-tune", "--spec", "../outside.json", "--corpus", ".cache/real-weights/manifest.json"])
        for command in calls:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(experiments.main(command), 2)

    def test_compressed_probe_defaults_remain_tuning_only(self):
        args = experiments.parser().parse_args(["probe-compressed", "--revision", "1"])
        self.assertEqual(args.revision, 1)
        self.assertEqual(args.dtype, "bf16")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiments.main(["probe-compressed", "--output", "../escape.json"]), 2)

    def test_compressed_tune_and_validation_contracts_are_explicit(self):
        for command, seed in (("tune", "901"), ("stream", "919")):
            args = experiments.parser().parse_args([command, "--family", "compressed", "--dtype", "bf16",
                "--output-dtype", "fp32", "--seeds", seed, "--profile", "results/frozen.json"])
            self.assertEqual((args.family, args.output_dtype, args.seeds), ("compressed", "fp32", seed))
        invalid = (["stream", "--family", "compressed", "--dtype", "fp32", "--seeds", "919"],
                   ["stream", "--family", "compressed", "--dtype", "bf16", "--seeds", "919,929"],
                   ["tune", "--family", "compressed", "--dtype", "bf16", "--spec", "../outside.json"],
                   ["tune", "--family", "compressed", "--dtype", "bf16", "--rounds", "30"],
                   ["stream", "--family", "compressed", "--dtype", "bf16", "--seeds", "919", "--profile", "../outside.json"])
        for args in invalid:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(experiments.main(args), 2)


if __name__ == "__main__":
    unittest.main()
