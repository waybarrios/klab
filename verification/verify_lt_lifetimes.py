"""Parent regression: input/output stream bookkeeping and inference tensors."""
import torch
import cublaslt_baseline as lt
from experiment_control import GpuLock


class TrackedTensor(torch.Tensor):
    def record_stream(self, stream):
        self.recorded = True
        return super().record_stream(stream)


torch.set_grad_enabled(False)
with GpuLock(".cache/gpu.lock"):
    workspace = lt.LtWorkspace(0)
    plans = []
    failures = []
    try:
        x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        plans = lt.prepare_lt_plans(x, w, output_dtype=torch.bfloat16, workspace=workspace)
        assert plans
        out = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
        tx = torch.Tensor._make_subclass(TrackedTensor, x, False)
        to = torch.Tensor._make_subclass(TrackedTensor, out, False)
        tx.recorded = to.recorded = False
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plans[0].run(tx, w, to)
        stream.synchronize()
        if not tx.recorded or not to.recorded:
            failures.append("input/output caching-allocator stream use was not recorded")
        with torch.inference_mode():
            ix = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
            iw = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
            inferred = []
            try:
                inferred = lt.prepare_lt_plans(ix, iw, output_dtype=torch.bfloat16, workspace=workspace)
                if not inferred:
                    failures.append("inference tensors produced no eligible plans: " + str(inferred.search_diagnostics))
                else:
                    io = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
                    inferred[0].run(ix, iw, io)
                    torch.cuda.synchronize()
            except RuntimeError as error:
                failures.append("inference tensors failed: " + str(error))
            finally:
                for plan in inferred:
                    plan.close()
    finally:
        for plan in plans:
            plan.close()
        workspace.close()
    assert not failures, failures
print("PASS: stream lifetime bookkeeping and inference-mode tensors")
