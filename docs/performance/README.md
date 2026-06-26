# Performance Documentation

This layer is for benchmark methodology, retained dispatch evidence, and kernel
tuning rules that are portable enough to keep in the public repository.

## Read In Order

| Goal | Read |
| --- | --- |
| Run reportable benchmarks | [`../common/benchmarking.md`](../common/benchmarking.md) |
| Work on BLAS Level 1 kernels | [`../common/level1_optimization_notes.md`](../common/level1_optimization_notes.md) |
| Work on BLAS Level 2 kernels | [`../common/level2_optimization_notes.md`](../common/level2_optimization_notes.md) |
| Work on GEMM dispatch or kernels | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md) |
| Work on AArch64 GEMM | [`../aarch64/gemm_aarch64_optimization_notes.md`](../aarch64/gemm_aarch64_optimization_notes.md) |
| Work on x86_64 GEMM | [`../x86_64/gemm_x86_64_optimization_notes.md`](../x86_64/gemm_x86_64_optimization_notes.md) |

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
