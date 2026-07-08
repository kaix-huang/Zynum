# Performance Documentation

This layer is for benchmark methodology, retained dispatch evidence, and kernel
tuning rules that are portable enough to keep in the public repository.

## Read In Order

| Goal | Read |
| --- | --- |
| Plan optimization order across BLAS levels | [`../common/performance_optimization_process.md`](../common/performance_optimization_process.md) |
| Run reportable benchmarks | [`../common/benchmarking.md`](../common/benchmarking.md) |
| Work on BLAS Level 1 kernels | [`../common/level1_optimization_notes.md`](../common/level1_optimization_notes.md) |
| Work on BLAS Level 2 kernels | [`../common/level2_optimization_notes.md`](../common/level2_optimization_notes.md) |
| Work on GEMM planning or kernels | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md) |
| Work on AArch64 GEMM | [`../aarch64/gemm_aarch64_optimization_notes.md`](../aarch64/gemm_aarch64_optimization_notes.md) |
| Work on x86_64 GEMM | [`../x86_64/gemm_x86_64_optimization_notes.md`](../x86_64/gemm_x86_64_optimization_notes.md) |

## Current Working Lessons

- Optimize in level order: Level 1, then Level 2, then Level 3. Higher-level
  exceptions should not hide a weak lower-level primitive.
- Treat correctness as part of every timing result. Rows marked
  `correctness_failed`, `error`, or unchecked are not performance evidence,
  even when the reported operation rate is high.
- Use fresh processes for comparator claims. In-process multi-library sweeps are
  useful smoke tests, but worker-pool and dispatch state can change the answer.
- Diagnose slow or rejected experiments before closing them. A CSV regression
  should be paired with sampling, tracing, disassembly, task timing, or another
  mechanism-level explanation.
- For threaded Level 2 work, first verify the single-thread leaf and the task
  body selected by dispatch. Only then tune row or column splits, helper count,
  and merge policy.
- On Apple Silicon, `hw.perflevel*` data is a capacity hint, not CPU affinity.
  SME/SM/ZA state costs must be separated from scheduler and wait costs.
- On Linux/x86_64, affinity masks are real but bounded by the job cpuset. Record
  the inherited mask and trace actual CPU placement before relying on it.

## Public Evidence Boundary

Keep public performance notes focused on:

- semantic rules that must not regress,
- retained dispatch predicates,
- required correctness and benchmark commands,
- comparator isolation policy,
- CSV and artifact names when they are part of a curated evidence summary.

Keep raw profiler output, local disassembly notes, one-off failed experiments,
machine-specific comparator paths, and uncurated CSVs in local private notes
outside the committed tree.
