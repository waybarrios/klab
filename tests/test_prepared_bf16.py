import unittest

import torch


class PreparedBF16VendorTests(unittest.TestCase):
    def test_native_vendor_callable_preserves_bf16_and_avoids_codec(self):
        # CPU tensors exercise the actual native operation, without taking the
        # GPU lock from the independent held-out run.
        from prepared_bf16 import vendor_callable,vendor_out_callable
        w=torch.arange(32,dtype=torch.bfloat16).reshape(4,8)
        x=torch.arange(16,dtype=torch.bfloat16).reshape(2,8)
        call=vendor_callable(w,torch.bfloat16)
        result=call(x)
        self.assertEqual(result.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(result,torch.mm(x,w.T)))
        out=torch.empty_like(result)
        self.assertIs(vendor_out_callable(w,torch.bfloat16)(x,out=out),out)
        self.assertTrue(torch.equal(out,result))

    def test_native_vendor_checks_incompatible_input_and_output(self):
        from prepared_bf16 import vendor_callable,vendor_out_callable
        w=torch.ones(4,8,dtype=torch.bfloat16)
        call=vendor_callable(w,torch.bfloat16)
        with self.assertRaises(RuntimeError):
            call(torch.ones(2,7,dtype=torch.bfloat16))
        with self.assertRaises(RuntimeError):
            call(torch.ones(2,8,dtype=torch.float32))
        with self.assertRaises(RuntimeError):
            vendor_out_callable(w,torch.bfloat16)(torch.ones(2,8,dtype=torch.bfloat16),
                out=torch.empty(2,4,dtype=torch.float32))

    def test_unsupported_output_contract_fails(self):
        from prepared_bf16 import vendor_callable
        with self.assertRaises(ValueError):
            vendor_callable(torch.ones(4,8,dtype=torch.bfloat16),torch.float16)

    def test_out_cannot_overwrite_inputs_or_prepared_weights(self):
        from prepared_bf16 import vendor_out_callable
        w=torch.ones(4,4,dtype=torch.bfloat16)
        x=torch.ones(4,4,dtype=torch.bfloat16)
        call=vendor_out_callable(w,torch.bfloat16)
        with self.assertRaises(ValueError):
            call(x,out=w)
        with self.assertRaises(ValueError):
            call(x,out=x)

    def test_allocation_path_rejects_out_instead_of_bypassing_alias_checks(self):
        from prepared_bf16 import vendor_callable
        w=torch.ones(4,4,dtype=torch.bfloat16)
        x=torch.ones(4,4,dtype=torch.bfloat16)
        with self.assertRaises((TypeError,RuntimeError)):
            vendor_callable(w,torch.bfloat16)(x,out=w)


if __name__=="__main__":
    unittest.main()
