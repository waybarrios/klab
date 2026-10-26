"""User-facing wrapper: cross-op L2 prefetch for a sequence of nn.Modules, tuned at capture time.

    fast = PrefetchSequential(list_of_modules, example_input)   # measures and picks plans once
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fast(static_x)

Plans are shared between modules with the same parameter signature, so tuning cost grows with the number
of distinct layer shapes, not with depth. A group keeps prefetch only if its measured saving has a bootstrap
95% lower bound above +1%; otherwise that group runs exactly as plain PyTorch. Outputs never change.
"""
import torch

import autotune
import harness as H
from l2prefetch import ChainRunner, prefetch


def _reads(module, min_bytes):
    return [p for p in module.parameters(recurse=True) if p.numel() * p.element_size() >= min_bytes]


def _signature(module, reads):
    return (type(module).__name__, tuple((tuple(p.shape), p.dtype) for p in reads))


class PrefetchSequential(torch.nn.Module):
    def __init__(self, modules, example_input, min_bytes=64 << 10, rounds=7, log=None, min_lower_ci=1.0):
        super().__init__()
        self.layers = torch.nn.ModuleList(modules)
        reads = [_reads(m, min_bytes) for m in self.layers]
        sigs = [_signature(m, r) for m, r in zip(self.layers, reads)]
        groups = {}
        for i, s in enumerate(sigs):
            groups.setdefault(s, []).append(i)
        ops = [m.__call__ for m in self.layers]
        plans = [None] * len(ops)
        self.report = []
        with torch.no_grad():
            for sig, members in groups.items():
                nbytes = autotune.stage_bytes(reads[members[0]])
                if nbytes == 0:
                    continue
                best, best_row = None, None
                variants = {"current": ChainRunner(ops, reads, list(plans))}
                cands = autotune.candidates(nbytes)
                for name, plan in cands.items():
                    trial = list(plans)
                    for i in members:
                        trial[i] = plan
                    variants[name] = ChainRunner(ops, reads, trial)
                rows = H.run_variants(variants, example_input, len(ops), rounds=rounds, baseline="current")
                name = min((k for k in rows if k != "current"), key=lambda k: rows[k]["us_per_op"])
                if rows[name]["ci95"][0] > min_lower_ci and rows[name]["bit_identical"]:
                    for i in members:
                        plans[i] = cands[name]
                    best, best_row = cands[name], rows[name]
                self.report.append({"signature": str(sig), "layers": len(members), "bytes": nbytes, "plan": best,
                                    "saved_pct": best_row and best_row["saved_pct"]})
                if log:
                    log(f"{sig[0]} x{len(members)} ({nbytes >> 10} KiB): {best} {best_row and best_row['saved_pct']}")
        self.runner = ChainRunner(ops, reads, plans if any(plans) else None)

    def forward(self, x):
        return self.runner(x)


class PrefetchHooks:
    """Cross-op L2 prefetch for an unmodified nn.Module, installed with forward pre-hooks.

        hooks = PrefetchHooks(model, example_input)   # trace order, tune per layer-shape group, install hooks
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = model(static_x)
        hooks.remove()                                 # model is plain PyTorch again

    A "unit" is a module called during the trace that owns parameters directly; its reads are those parameters plus
    the parameters of descendants that were never called (e.g. MultiheadAttention.out_proj). When unit i starts, a
    side stream prefetches unit i+1's reads; the root's forward hook joins the side stream. Outputs never change.
    Note: fused fast paths (e.g. nn.TransformerEncoderLayer in eval) call no submodules; disable them
    (torch.backends.mha.set_fastpath_enabled(False)) if the units inside a layer should be visible.
    Training (backward=True): as for any CUDA-graph capture of backward, run eager warm-up steps on a side stream;
    an earlier eager backward on the default stream makes the capture fail (measured, independent of prefetch).
    """

    def __init__(self, model, example_input, min_bytes=64 << 10, rounds=7, log=None, min_lower_ci=1.0,
                 backward=False, step=None):
        """backward=True also prefetches unit i-1's reads when unit i's backward starts (training).
        step(model, x) is what gets timed while tuning (default: forward only); pass a fwd+bwd closure for training."""
        self.model, self.side, self.handles = model, torch.cuda.Stream(), []
        self.backward, self.step = backward, step or (lambda m, x: m(x))
        self.units, self.plans, self.used = self._trace(model, example_input, min_bytes), {}, False
        self.in_root, self.bw_armed = False, False
        self._install()
        groups = {}
        for i, (m, reads) in enumerate(self.units):
            groups.setdefault(_signature(m, reads), []).append(i)
        self.report = []
        with torch.set_grad_enabled(backward):
            for sig, members in groups.items():
                nbytes = autotune.stage_bytes(self.units[members[0]][1])
                cands = autotune.candidates(nbytes)
                current = dict(self.plans)
                variants = {"current": self._runner(current)}
                for name, plan in cands.items():
                    trial = dict(current)
                    trial.update({i: plan for i in members})
                    variants[name] = self._runner(trial)
                rows = H.run_variants(variants, example_input, 1, rounds=rounds, baseline="current")
                name = min((k for k in rows if k != "current"), key=lambda k: rows[k]["us_per_op"])
                kept = rows[name]["ci95"][0] > min_lower_ci and rows[name]["bit_identical"]
                if kept:
                    self.plans.update({i: cands[name] for i in members})
                self.report.append({"signature": str(sig), "units": len(members), "bytes": nbytes,
                                    "plan": cands[name] if kept else None, "saved_pct": rows[name]["saved_pct"] if kept else None})
                if log:
                    log(f"{sig[0]} x{len(members)} ({nbytes >> 10} KiB): {cands[name] if kept else None}")

    @staticmethod
    def _trace(model, x, min_bytes):
        order, handles = [], []
        for m in model.modules():
            handles.append(m.register_forward_pre_hook(lambda mod, args: order.append(mod)))
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()
        called = set(map(id, order))
        units, seen = [], set()
        for m in order:
            if id(m) in seen:
                continue
            seen.add(id(m))
            reads = list(m.parameters(recurse=False))
            for child in m.modules():
                if child is not m and id(child) not in called:
                    reads += list(child.parameters(recurse=False))
            reads = [p for p in reads if p.numel() * p.element_size() >= min_bytes]
            if reads:
                units.append((m, reads))
        return units

    def _install(self):
        index = {id(m): i for i, (m, _) in enumerate(self.units)}

        def root_pre(mod, args):
            self.in_root = True
            self.fwd_stream = torch.cuda.current_stream()  # autograd runs each op's backward on its forward stream

        def pre(mod, args):
            if not self.in_root:  # submodule called outside the root's forward: behave exactly like plain PyTorch
                return
            i = index[id(mod)]
            plan = self.plans.get(i + 1)
            if plan and i + 1 < len(self.units):
                self.side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.side):
                    for t in self.units[i + 1][1]:
                        prefetch(t, **plan)
                self.used = True

        def done(mod, args, out):
            self.in_root = False
            self.bw_armed = self.backward and torch.is_grad_enabled()
            if self.used:
                torch.cuda.current_stream().wait_stream(self.side)
                self.used = False

        def back_pre(mod, grad_output):
            if not self.bw_armed:
                return
            i = index[id(mod)]
            plan = self.plans.get(i - 1)
            if plan and i > 0:
                # The autograd thread's current stream is the legacy default stream, not the capturing one.
                self.side.wait_stream(self.fwd_stream)
                with torch.cuda.stream(self.side):
                    for t in self.units[i - 1][1]:
                        prefetch(t, **plan)
                self.used = True

        def back_done(mod, grad_input, grad_output):  # unit 0 finishes last in backward: join there
            self.bw_armed = False
            if self.used:
                self.fwd_stream.wait_stream(self.side)
                self.used = False

        for m, _ in self.units:
            self.handles.append(m.register_forward_pre_hook(pre))
            if self.backward:
                self.handles.append(m.register_full_backward_pre_hook(back_pre))
        if self.backward and self.units:
            self.handles.append(self.units[0][0].register_full_backward_hook(back_done))
        self.handles.append(self.model.register_forward_pre_hook(root_pre))
        self.handles.append(self.model.register_forward_hook(done))

    def _runner(self, plans):
        def run(x):
            saved, self.plans = self.plans, plans
            try:
                return self.step(self.model, x)
            finally:
                self.plans = saved
        return run

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
