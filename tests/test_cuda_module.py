import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda_module import module_fingerprint


class ModuleFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".scratch")
        self.root = Path(self.temp.name)
        for name in ("cuda_module.py", "kernel_lab.py", "experiment_control.py",
                     "native/driver.cpp", "native/build/libkernel_lab.so"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"shared dependency")
        (self.root / "active.cu").write_bytes(b"extern C kernel active")
        (self.root / "legacy.cu").write_bytes(b"unrelated legacy kernel")

    def tearDown(self):
        self.temp.cleanup()

    def fingerprint(self, sources=("active.cu",), architecture="86", compiler="NVRTC12.8",
                    options=("--std=c++17",)):
        return module_fingerprint(sources, architecture, compiler, options, root=self.root)

    def test_fingerprint_binds_actual_source_and_shared_build(self):
        result = self.fingerprint()
        self.assertEqual(len(result["fingerprint_sha256"]), 64)
        self.assertEqual(result["source_sha256"]["active.cu"],
                         hashlib.sha256(b"extern C kernel active").hexdigest())
        self.assertEqual(json.loads(json.dumps(result)), result)
        before = result["fingerprint_sha256"]
        for name in ("active.cu", "kernel_lab.py", "native/driver.cpp", "native/build/libkernel_lab.so"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            self.assertNotEqual(before, self.fingerprint()["fingerprint_sha256"], name)
            path.write_bytes(original)

    def test_unrelated_cuda_edit_preserves_profile_identity(self):
        before = self.fingerprint()["fingerprint_sha256"]
        (self.root / "legacy.cu").write_bytes(b"new unrelated experiment")
        self.assertEqual(before, self.fingerprint()["fingerprint_sha256"])

    def test_target_compiler_options_and_source_order_change_identity(self):
        before = self.fingerprint()["fingerprint_sha256"]
        alternatives = (
            self.fingerprint(architecture="80"), self.fingerprint(compiler="NVRTC12.9"),
            self.fingerprint(options=("--std=c++20",)),
            self.fingerprint(sources=("active.cu", "legacy.cu")),
        )
        for result in alternatives:
            self.assertNotEqual(before, result["fingerprint_sha256"])
        self.assertNotEqual(self.fingerprint(sources=("active.cu", "legacy.cu"))["fingerprint_sha256"],
                            self.fingerprint(sources=("legacy.cu", "active.cu"))["fingerprint_sha256"])

    def test_escape_empty_and_missing_sources_are_rejected(self):
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        for sources in ((), ("../outside.cu",), ("escape/outside.cu",)):
            with self.assertRaises(ValueError):
                self.fingerprint(sources=sources)
        with self.assertRaises(FileNotFoundError):
            self.fingerprint(sources=("missing.cu",))

    def test_import_is_cpu_only(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys,cuda_module; assert 'torch' not in sys.modules"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
