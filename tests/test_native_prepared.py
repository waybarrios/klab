"""CPU logical/Fake and immutable PTX-input tests, not GPU performance evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT=Path(__file__).resolve().parents[1]


class NativePreparedTests(unittest.TestCase):
    def test_fake_coded_preserves_both_output_contracts(self):
        import native_prepared as native
        with FakeTensorMode():
            x=torch.empty((3,64),dtype=torch.bfloat16)
            payload=torch.empty(13*96,dtype=torch.uint8)
            metadata=torch.empty(13,dtype=torch.int32)
            for name,dtype in (("bf16",torch.bfloat16),("fp32",torch.float32)):
                y=native.fake_coded(x,payload,metadata,1,3,13,64,128,8,name)
                self.assertEqual(tuple(y.shape),(3,13))
                self.assertEqual(y.dtype,dtype)
                self.assertEqual(y.device.type,"cpu")

    def test_fake_raw_preserves_shape_and_dtype(self):
        import native_prepared as native
        with FakeTensorMode():
            x=torch.empty((2,64),dtype=torch.bfloat16)
            w=torch.empty((17,64),dtype=torch.bfloat16)
            y=native.fake_raw(x,w,1,2,17,64,64,8,"fp32")
            self.assertEqual(tuple(y.shape),(2,17))
            self.assertEqual(y.dtype,torch.float32)

    def test_fake_rejects_wrong_plan_shape_buffers_and_dtype(self):
        import native_prepared as native
        with FakeTensorMode():
            x=torch.empty((2,64),dtype=torch.bfloat16)
            w=torch.empty((17,64),dtype=torch.bfloat16)
            with self.assertRaises(ValueError): native.fake_raw(x,w,0,2,17,64,64,8,"fp32")
            with self.assertRaises(ValueError): native.fake_raw(x,w,1,3,17,64,64,8,"fp32")
            with self.assertRaises(ValueError): native.fake_raw(x.float(),w,1,2,17,64,64,8,"fp32")
            with self.assertRaises(ValueError): native.fake_raw(x,w,1,2,17,64,64,8,"fp16")
            with self.assertRaises(ValueError):
                native.fake_coded(x,torch.empty(3,dtype=torch.uint8),torch.empty(17,dtype=torch.int32),
                                  1,2,17,64,64,8,"fp32")

    def test_cpu_preparation_rejects_before_build_or_cuda_query(self):
        import native_prepared as native
        with mock.patch.object(native,"ensure_loaded",side_effect=AssertionError("must not build/load")):
            with self.assertRaises(ValueError):
                native.NativeRawCandidate(torch.ones((13,64),dtype=torch.bfloat16),
                                          threads=128,vector=8,output_dtype="bf16")

    def test_ptx_bytes_must_match_the_original_compilation_digest(self):
        import native_prepared as native
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            path=Path(directory)/"fixed.ptx"
            original=b"// synthetic PTX identity fixture, never loaded\x00"
            path.write_bytes(original)
            compiled={"ptx_path":str(path.relative_to(ROOT)),"ptx_sha256":hashlib.sha256(original).hexdigest()}
            self.assertEqual(native.read_ptx(compiled),original)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): native.read_ptx(compiled)


if __name__=="__main__":
    unittest.main()
