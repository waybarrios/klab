"""CPU FakeTensor checks for the frozen raw/coded R2 operator adapters."""
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


class PreparedKernelFakeTests(unittest.TestCase):
    def test_coded_operator_fake_preserves_both_output_contracts(self):
        from prepared_kernels import coded_r2
        with FakeTensorMode():
            for m in (1,3,16):
                for output,dtype in (("bf16",torch.bfloat16),("fp32",torch.float32)):
                    x=torch.empty((m,64),dtype=torch.bfloat16)
                    payload=torch.empty(13*96,dtype=torch.uint8)
                    metadata=torch.empty(13,dtype=torch.int32)
                    y=coded_r2(x,payload,metadata,m,13,64,128,8,output)
                    self.assertEqual(y.shape,(m,13))
                    self.assertEqual(y.dtype,dtype)
                    self.assertEqual(y.device,x.device)

    def test_raw_operator_fake_preserves_both_output_contracts(self):
        from prepared_kernels import raw_r2
        with FakeTensorMode():
            x=torch.empty((4,64),dtype=torch.bfloat16)
            w=torch.empty((13,64),dtype=torch.bfloat16)
            for output,dtype in (("bf16",torch.bfloat16),("fp32",torch.float32)):
                y=raw_r2(x,w,4,13,64,128,8,output)
                self.assertEqual(y.shape,(4,13))
                self.assertEqual(y.dtype,dtype)

    def test_coded_fake_rejects_wrong_shape_configuration_and_buffers(self):
        from prepared_kernels import coded_r2
        with FakeTensorMode():
            x=torch.empty((3,64),dtype=torch.bfloat16)
            payload=torch.empty(13*96,dtype=torch.uint8)
            metadata=torch.empty(13,dtype=torch.int32)
            base=[x,payload,metadata,3,13,64,128,8,"bf16"]
            changes=[(0,x.float()),(0,x.requires_grad_()),(1,payload.to(torch.int32)),
                     (1,payload[:95]),(2,metadata[:12]),(3,4),(4,0),(5,63),
                     (6,96),(7,4),(8,"fp16")]
            # requires_grad_ changed x in place; restore the valid baseline.
            x.requires_grad_(False)
            changes[1]=(0,torch.empty((3,64),dtype=torch.bfloat16,requires_grad=True))
            for index,value in changes:
                with self.subTest(index=index,value=str(value)[:80]):
                    args=list(base)
                    args[index]=value
                    with self.assertRaises((ValueError,RuntimeError)):
                        coded_r2(*args)

    def test_raw_fake_cannot_reuse_a_plan_for_a_different_shape(self):
        from prepared_kernels import raw_r2
        with FakeTensorMode():
            x=torch.empty((4,64),dtype=torch.bfloat16)
            w=torch.empty((13,64),dtype=torch.bfloat16)
            for args in ((x,w,1,13,64,128,8,"bf16"),
                         (x,w,4,14,64,128,8,"bf16"),
                         (x,w.float(),4,13,64,128,8,"fp32")):
                with self.assertRaises((ValueError,RuntimeError)):
                    raw_r2(*args)

    def test_candidate_preparation_rejects_cpu_before_touching_cuda(self):
        from prepared_kernels import CodedCandidate,RawCandidate
        import compressed_bf16 as cb
        before=torch.cuda.is_initialized()
        with self.assertRaises(ValueError):
            RawCandidate(torch.ones((13,64),dtype=torch.bfloat16),tokens=3,
                         threads=128,vector=8,output_dtype="bf16")
        packed=cb.PackedBF16(torch.empty(13*96,dtype=torch.uint8),torch.empty(13,dtype=torch.int32),
                             13,64,"synthetic",cb.FORMAT_VERSION,None)
        with self.assertRaises(ValueError):
            CodedCandidate(packed,tokens=3,threads=128,vector=8,output_dtype="bf16")
        self.assertEqual(torch.cuda.is_initialized(),before)


if __name__=="__main__":
    unittest.main()
