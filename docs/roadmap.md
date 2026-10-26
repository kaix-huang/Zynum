# Scientific roadmap

Updated 2026-10-26. The [capability map](scientific.md) indexes current delivery.
[Development history](development_history.md) retains earlier plans and dated
evidence without treating their scheduling statements as current acceptance.

## Priorities

1. Stabilize existing BLAS/LAPACK/Sparse/FFT functionality and shared resource
   lifetimes. Cover external calls, numerical boundaries and package ownership
   while consolidating the scientific branch.
2. Retain the ABPLaS dependency path: LP64 `dgesv`, `zpotrf`, `dstev`, `dgelsd`,
   complex CSR SpMV/SpMM, mixed indices and common operators. Application adoption
   is qualified independently from installed interface availability. Updated
   ABPLaS source adaptation permits fresh integration builds.
3. Extend FFT through FFTW API/ABI profiles. Internal tensor layouts, algorithms,
   plans and workspace remain Zynum-owned; optional compile-time scripts generate
   interfaces. Current f32/f64 selections remain finite.
4. Extend complex standard/generalized Hermitian eigensolvers and common operator
   algorithms, then broaden LAPACK and Sparse routine/type/mode coverage.
5. Optimize after functionality and stability gates. The requested local
   LAPACK/Sparse/FFT comparison was measured on its original 2026-10-15 SDK with
   ABPLaS scale models plus general sizes and thread caps 1/4. Those measurements
   do not describe subsequent revisions; consolidation does not retime them.

## Architecture invariants

- One mathematical implementation behind selectable external profiles.
- Shared native data, Context, captured allocator, execution policy, workspace
  and handle accounting within an artifact; explicit cross-artifact boundaries.
- Generated declarations/forwarders contain no new numerical or resource owner.
- Preserve argument precedence, padding, queries and documented mutation on
  failure and partial convergence.
- Qualify numerical results, bounds, lifetime, installed calls and ABPLaS
  regressions separately. Cross-compilation does not fill native rows.

## Open coverage

Complete LAPACK, Sparse, FFTW and FEAST compatibility is unfinished. Foreign
native inventory rows remain pending on their actual environments. Additional
Sparse solver/format/preconditioner families, full material TimeEvolution,
statistical convergence and broad vendor IEEE/failure-payload equivalence need
separate acceptance. ABPLaS uses selected operations and does not exercise every
available generated routine.
