import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_control import GpuLock, cache_key, cached_ptx, parse_shape, project_path
import experiment_control as control


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".scratch")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_output_path_stays_inside_project(self):
        self.assertEqual(project_path(self.root, "results/run.json"), self.root / "results/run.json")
        with self.assertRaises(ValueError):
            project_path(self.root, "../outside.json")

    def test_symlink_escape_is_rejected(self):
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            project_path(self.root, "escape/outside.json")

    def test_shape_validation(self):
        self.assertEqual(parse_shape("1,4096,4096"), (1, 4096, 4096))
        for value in ("1,2", "0,64,64", "1,-64,64", "one,64,64"):
            with self.assertRaises(ValueError):
                parse_shape(value)

    def test_gpu_lock_prevents_overlap_and_releases(self):
        path = self.root / "gpu.lock"
        with GpuLock(path):
            with self.assertRaises(RuntimeError):
                with GpuLock(path):
                    self.fail("Second lock was accepted")
        with GpuLock(path):
            pass

    def test_cache_key_invalidates_source_target_compiler_options(self):
        original = cache_key(b"source", "86", "12.8", ["--std=c++17"])
        alternatives = (
            cache_key(b"changed", "86", "12.8", ["--std=c++17"]),
            cache_key(b"source", "80", "12.8", ["--std=c++17"]),
            cache_key(b"source", "86", "12.9", ["--std=c++17"]),
            cache_key(b"source", "86", "12.8", ["--std=c++20"]),
        )
        self.assertEqual(len(original), 64)
        for other in alternatives:
            self.assertNotEqual(original, other)

    def test_cache_hit_force_and_corruption(self):
        calls = []
        def compile_it():
            calls.append(True)
            return b".version 8.0\n.target sm_86\n"
        key = cache_key(b"source", "86", "12.8", [])
        _, hit = cached_ptx(self.root, key, compile_it)
        self.assertFalse(hit)
        _, hit = cached_ptx(self.root, key, compile_it)
        self.assertTrue(hit)
        cached_ptx(self.root, key, compile_it, force=True)
        self.assertEqual(len(calls), 2)
        (self.root / f"{key}.ptx").write_bytes(b"broken")
        cached_ptx(self.root, key, compile_it)
        self.assertEqual(len(calls), 3)

    def test_run_output_cannot_replace_lock_or_report(self):
        for value in (".cache/gpu.lock", "results/run.md", "kernels.cu"):
            with self.assertRaises(ValueError):
                control.run_output(self.root, value)
        self.assertEqual(control.run_output(self.root, "results/run.json"), self.root / "results/run.json")

    def test_report_output_preserves_json(self):
        source = self.root / "results/run.json"
        with self.assertRaises(ValueError):
            control.report_output(self.root, source, source)
        with self.assertRaises(ValueError):
            control.report_output(self.root, source, "results/other.json")
        self.assertEqual(control.report_output(self.root, source, None), source.with_suffix(".md"))


if __name__ == "__main__":
    unittest.main()
