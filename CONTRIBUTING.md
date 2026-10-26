# Contributing to klab

Reproductions, numerical counterexamples and careful negative results are
welcome. A small experiment with a clear baseline is a useful contribution.

Start with an issue describing the operation, tensor shape and precision
contract you want to investigate. Include your GPU model and the behavior you
expect to improve. I am happy to discuss the measurement setup before you
spend time on a kernel.

Keep the supplied weights and inputs in their original dtype. Use FP32
accumulation for the exact-weight experiments and retain the numerical checks.
Lossy formats belong in a separate experiment with an explicit accuracy budget.

A benchmark report should include the tested vendor routes, raw timing
rounds, the winning vendor's A/A control, and source/runtime fingerprints.
State whether timing includes setup, packing, allocations or host dispatch.
Record the power profile, AC condition and observed clock regime. Personal
machine paths and device serial identifiers are unnecessary in an issue.

Keep every registered case, including slowdowns and failed controls. Freeze
the candidate before held-out timing. A failed process or control must remain
visible; another run cannot replace it. New searches need a new prospective
study rather than a revision to an accepted result.

Source and reporting improvements are useful even without GPU access. The
portable CPU checks cover codecs, numerical rules, result schemas and evidence
handling. See the [README](README.md) for the local setup and the open questions.

Please keep changes focused and describe what changed, why it helps, and how
you checked it. If a result depends on a particular driver, layout or power
condition, make that condition part of the report.
