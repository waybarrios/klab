"""Parent GPU verification of vendor ABI, precision and frozen restoration."""
import json
from pathlib import Path

import torch
import cublaslt_baseline as lt
from experiment_control import GpuLock, write_json
from projection_accuracy import check_outputs

ROOT = Path(__file__).resolve().parents[1]

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
results = []
with GpuLock(ROOT / ".cache/gpu.lock"):
    torch.cuda.init()
    before = torch.cuda.memory_allocated()
    workspace = lt.LtWorkspace(0)
    assert torch.cuda.memory_allocated() - before == 67108864
    try:
        for dtype, out_dtype, shape in ((torch.bfloat16, torch.bfloat16, (4, 256, 256)),
                                        (torch.bfloat16, torch.float32, (4, 256, 256)),
                                        (torch.float16, torch.float32, (4, 256, 256)),
                                        (torch.float32, torch.float32, (4, 256, 256)),
                                        (torch.bfloat16, torch.bfloat16, (1, 4864, 896))):
            m, n, k = shape
            torch.manual_seed(901)
            x = torch.randn((m, k), dtype=dtype, device="cuda")
            w = torch.randn((n, k), dtype=dtype, device="cuda")
            ref = x.double() @ w.double().T
            plans = lt.prepare_lt_plans(x, w, output_dtype=out_dtype, workspace=workspace)
            assert plans, plans.search_diagnostics
            case = {"dtype": str(dtype), "output_dtype": str(out_dtype), "shape_mnk": list(shape),
                    "diagnostics": plans.search_diagnostics, "plans": []}
            try:
                for plan in plans:
                    assert plan.workspace is workspace
                    out = torch.empty((m, n), dtype=out_dtype, device="cuda")
                    plan.run(x, w, out)
                    numeric = check_outputs(ref, out, output_dtype=out_dtype)
                    assert numeric["passed"], (plan.freeze(), numeric)
                    record = plan.freeze()
                    restored = lt.LtPlan.restore(record, x, w, output_dtype=out_dtype, workspace=workspace)
                    try:
                        assert restored.freeze() == record
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            restored.run(x, w, out)
                        stream.synchronize()
                        assert check_outputs(ref, out, output_dtype=out_dtype)["passed"]
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            restored.run(x, w, out)
                        graph.replay()
                        assert check_outputs(ref, out, output_dtype=out_dtype)["passed"]
                        graph.reset()
                    finally:
                        restored.close()
                    case["plans"].append({"frozen": record, "numerics": numeric})
                # Caller-requested output contract must not be silently ignored.
                if dtype == torch.bfloat16:
                    different = torch.float32 if out_dtype == torch.bfloat16 else torch.bfloat16
                    wrongly_restored = None
                    try:
                        wrongly_restored = lt.LtPlan.restore(plans[0].freeze(), x, w,
                                                            output_dtype=different, workspace=workspace)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("restore ignored the requested output dtype")
                    finally:
                        if wrongly_restored is not None:
                            wrongly_restored.close()
            finally:
                for plan in plans:
                    plan.close()
            results.append(case)
            print("PASS", str(dtype), str(out_dtype), shape, len(plans), "plans", flush=True)
    finally:
        workspace.close()
write_json(ROOT / "results/cublaslt-verification.json", {"status": "complete", "cases": results})
print("PASS: shared workspace, dtype contracts, numerical checks, freeze/restore, streams and graphs")
