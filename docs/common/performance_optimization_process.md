# Performance Optimization Process

This document describes the default order for optimizing Zynum BLAS. It applies
to Level 1, Level 2, and Level 3 work. Operation-specific notes should record
retained evidence, but this file owns the shared process.

The main rule is to optimize families, not isolated points. A benchmark point is
useful when it represents a broader operation class, shape class, data type,
cache regime, instruction-set capability, or thread-scheduling regime. Do not
turn one lucky coordinate into a permanent dispatch rule.

## Optimization Order

Optimize in this order unless a correctness bug forces a narrower detour:

1. Level 1: vector semantics, contiguous fast paths, reductions, scalar
   shortcuts, architecture vector kernels, and large-vector threading.
2. Level 2: matrix-vector and rank-update kernels, triangular or symmetric
   storage traversal, workspace reuse, task shape, and merge costs.
3. Level 3: matrix-matrix kernels, packing, micro-kernel selection, cache
   blocking, shape dispatch, and multi-thread split policy.

This order matters because higher levels reuse lower levels. A Level 2 or Level
3 route that calls a weak Level 1 primitive can hide the real bottleneck. Fixing
the lower-level operation family first often removes the need for a narrow
higher-level exception.

## Generalization Standard

A retained optimization should be expressed with predicates that describe why it
is expected to work:

- Operation family: copy, scaling, AXPY, reduction, GEMV, GER, SYMV/HEMV, GEMM,
  or complex decomposition.
- Data family: f32, f64, complex f32, complex f64, real-alpha complex work, or
  true complex arithmetic.
- Layout family: unit stride, non-unit stride, column-major contiguous panels,
  vector edge, narrow-N, short-wide, tall-skinny, square-ish, high-K, or small
  odd tails.
- Capability family: portable vector code, ASIMD/FMA, SVE/SVE2, SME/SME2, AMX,
  AVX2/FMA, AVX512/FMA, or generic scalar fallback.
- Resource family: register pressure, stack pack budget, cached workspace,
  performance L2 budget, memory bandwidth, worker handoff cost, and merge cost.
- Thread family: single-thread, low-latency helper path, normal `std.Io`
  concurrency, row split, column split, panel split, and reduction merge.

Exact shape checks may remain only when they describe a boundary of a broader
rule, for example a tile alignment, a descriptor minimum block, or a tested
rollback boundary. Prefer helper predicates such as `isSmallSquareish`,
`isNarrowN`, `fitsWorkspaceBudget`, or `usesWholePanels` over inline
coordinates such as `m == 128 and n == 128`.

## Current Cross-Level Lessons

The 2026 Level 1 through Level 3 passes left a few rules that are more useful
than the individual benchmark coordinates:

- Correctness fixes reset the performance clock. Data collected before a
  partial-task fallback, stale state cleanup, conjugation fix, or layout
  correction is not promotion or rollback evidence for the corrected code.
- Comparator claims need fresh processes even for local experiments. Zynum,
  Accelerate, OpenBLAS, MKL, and BLIS-family libraries all keep process-local
  dispatch or worker state, and mixed-library ordering has repeatedly changed
  small and medium results.
- Threading is a composition layer. A default route should first identify the
  right single-thread leaf, materializer, and storage ownership. Only then tune
  helper count, task shape, wake policy, and merge layout.
- Exact gates are acceptable when they protect a measured hardware or planner
  boundary: tile divisibility, odd-K codegen, row-panel tails, compact
  workspace padding, or a comparator-proven cliff. They should not become a
  substitute for a missing shape-family implementation.
- Complex routines often win by reducing the problem to real kernels, but that
  is not automatically cheaper. The materialization layout, plane spacing,
  scalar/conjugate handling, repeated packing, and number of real planner calls
  are part of the optimization, not incidental setup.
- Rejected experiments should record the mechanism that lost. A slower CSV row
  without a route check, disassembly, sample, task timing, or nearby repeat is a
  weak exclusion and should be reopened when the surrounding implementation
  changes.

## Per-Level Phase Order

Use the same phase order inside Level 1, Level 2, and Level 3:

1. Semantics and baseline: read the semantic owner and benchmark owner, fix
   no-op or alpha/beta behavior if it is wrong, and collect a baseline.
2. Kernel and micro-kernel coverage: implement or complete the candidate
   kernels first, including tails, scalar epilogues, alignment cases, and
   fallback handoff. Do not start by tuning thread splits around missing
   kernels.
3. Feasible range discovery: run focused probes over adjacent sizes to find
   where each kernel is correct, stable, and structurally applicable. This
   includes tile alignment, stride/layout constraints, workspace size, register
   pressure, hardware state setup, and cache residency.
4. Single-thread route selection: choose the best single-thread kernel path for
   each operation family and shape/resource family. Descriptor scores, facade
   gates, and execution plans should be tuned here before any multi-thread
   composition is promoted.
5. Multi-thread split and composition: only after the single-thread route is
   stable, tune row/column/panel splits, task counts, helper selection,
   workspace merging, and packed-buffer reuse.
6. Promotion and documentation: run the relevant full sweep with comparator
   isolation when making a comparator claim.
7. Record the retained predicate, evidence, excluded boundaries, and rollback
   condition in the appropriate notes.

If a change improves only one benchmark row and fails to explain nearby rows, it
is not a default optimization. Keep it experimental or remove it.

Threading is deliberately late in this order. If a single-thread path is using
the wrong micro-kernel, multi-thread tuning can hide the mistake by adding more
workers, repeating packing, or changing cache state. Treat multi-thread wins as
composition evidence, not as proof that the underlying kernel choice is good.

## Investigation Loop

Use the same loop for correctness failures, surprising wins, and regressions:

1. Confirm that the intended dispatch route and task body ran. Use symbols,
   sampling, trace output, or disassembly when routing is not obvious.
2. If correctness fails, fix semantics or state preservation first. Timing from
   the failing run is invalid; rerun the affected benchmarks after the fix and
   check whether the correctness guard introduced a measurable cost.
3. If performance regresses, collect at least one mechanism diagnostic before
   closing the experiment: sampling, task timing, trace/counter data,
   disassembly, or comparator-path inspection.
4. Separate kernel-body cost from composition cost. For threaded paths, account
   for wake/wait overhead, helper placement, merge work, and workspace traffic.
   For stateful ISA paths, account for SM/ZA/SIMD, AMX, or equivalent
   save/restore and mode-transition effects.
5. Record the outcome in the narrowest relevant note with the retained or
   rejected predicate, the artifact path, the mechanism, and the condition that
   would reopen the result.

## Level 1 Procedure

Level 1 is the foundation. Start with semantic shortcuts and contiguous paths:

- Normalize unit-stride copy to byte copy.
- Skip no-op scalar cases such as alpha-one scaling and alpha-zero updates when
  BLAS semantics permit it.
- Reuse real vector kernels for complex operations with real scalar
  coefficients.
- Keep true complex arithmetic in interleaved vector kernels with explicit
  shuffle/sign structure.
- Use multiple accumulators for reductions and measure the merge cost.
- Add architecture kernels only when the instruction sequence is materially
  different from the portable vector loop.
- Gate parallelism by vector length, operation arithmetic intensity, and
  measured worker handoff cost.

Level 1 evidence should include unit stride, non-unit stride fallback, negative
stride coverage where applicable, and real/complex variants. Large-vector
threading gates come after the unit kernels are stable and need both
single-thread and default-thread data.

## Level 2 Procedure

Level 2 should reuse Level 1 only when doing so does not create excessive entry
cost or repeated memory traffic:

- For GEMV, choose between direct matrix-vector microkernels, column AXPY
  accumulation, transpose dot kernels, and workspace-backed task splits.
- For GER/SYR/HER, choose row or column task shapes based on write ownership and
  cache reuse.
- For SYMV/HEMV, account for triangular storage, diagonal semantics, prefix
  traffic, per-task delta buffers, and final merges.
- Prefer fused column kernels when separate Level 1 AXPY/DOT calls cause extra
  passes over the same column.
- Gate low-latency helpers by work and merge cost; use normal `std.Io` for
  coarser work where scheduling overhead is amortized.

Level 2 tuning should be described by matrix size class, storage form, output
ownership, and workspace budget. Avoid route changes that only exist for one
matrix dimension unless they are tile-alignment boundaries. Add or complete the
GEMV/GER/SYMV/HEMV micro-kernel family before changing task counts.

## Level 3 Procedure

Level 3 adds packing, cache blocking, and micro-kernel selection:

- First separate semantic fast paths such as vector-edge GEMM and alpha/beta
  store-only paths.
- Then classify shape families: small square, medium square, large square,
  tall-narrow, short-wide, high-K, odd-tail, and complex decomposition.
- Add descriptor metadata for tile size, packing mode, minimum work, workspace
  budget, and ISA capability before adding policy.
- Select kernels through scoring and execution plans rather than architecture
  files.
- Keep packing reuse explicit. Penalize routes that repeat B packing across too
  many row splits or allocate complex workspaces that exceed cache budgets.
- Tune single-thread selection before multi-thread splitting.
- Tune row/column/panel splits after the single-thread route is stable.

Complex GEMM should be treated as a separate family. Compact complex kernels,
3M real decomposition, expanded-real decomposition, and vector-edge GEMV routes
have different workspace, packing, and cache behavior.

## Evidence And Promotion

A default optimization needs an evidence chain:

- Correctness for the target.
- Focused probe for the operation or shape family.
- Adjacent probes that define the retained boundary.
- Full sweep for the affected level.
- Fresh-process comparator data for no-slower-than claims.
- Recorded environment, thread policy, target tuple, CPU features, comparator
  libraries, raw CSV path, and summary.

Promote a rule only when the default predicate is narrower than the evidence.
Reject, narrow, or keep opt-in when data is noisy, isolated to one point,
contaminated by worker/cache state, or missing correctness coverage.

## Documentation Targets

Record retained details in the narrowest relevant file:

- Shared process and promotion rules: this document.
- Benchmark commands and isolation levels: `benchmarking.md`.
- Level 1 retained rules: `level1_optimization_notes.md`.
- Level 2 retained rules: `level2_optimization_notes.md`.
- Level 3/GEMM retained rules: `gemm_optimization_notes.md`.
- Architecture-specific ISA and ABI rules: the matching architecture note.

Keep local machine instructions, temporary CSVs, profiler captures, and
uncurated disassembly outside the public tree.
