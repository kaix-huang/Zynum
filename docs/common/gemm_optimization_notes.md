# GEMM Optimization Notes

This document records cross-platform GEMM engineering rules for Zynum BLAS.
Architecture-specific details belong in the AArch64 and x86_64 documents.

## Layering

GEMM responsibilities are split across:

- `src/blas/core/level3.zig`: BLAS semantics and fast-path detection.
- `src/blas/gemm/dispatch.zig`: shape policy, task splitting, and threading.
- `src/blas/gemm/pool.zig`: optional worker-pool experiments.
- `src/blas/kernels/matrix_matrix.zig`: target-feature candidate-set selection.
- `src/blas/kernels/matrix_matrix/catalog.zig`: stable kernel descriptors with
  `KernelId`, tile, packing, ISA, and threshold metadata.
- `src/blas/kernels/matrix_matrix/tuning.zig`: candidate scoring, shape/scalar
  matching, and execution-plan parameter selection.
- `src/blas/kernels/matrix_matrix/executor.zig`: runtime bridge from selected `KernelId`
  to implementation module.
- `src/blas/kernels/matrix_matrix/packed_simd.zig`: shared fixed-width packed-B SIMD
  skeleton. It owns panel preparation, K accumulation, row/tail handling, and
  write-back wrapping for ASIMD and x86 SIMD backends.
- `src/blas/kernels/matrix_matrix/epilogue.zig`: shared real-GEMM scalar/vector
  alpha/beta write-back helpers.
- `src/blas/kernels/matrix_matrix/task.zig`: shared task description, selected runtime
  kernel id, and the `ExecutionPlan` consumed by kernels.
- `src/blas/kernels/generic/matrix_matrix/basic.zig`: portable parameterized fallback kernels
  and vector-edge helpers that reuse shared epilogue rules where practical.
- `src/blas/kernels/<arch>/`: architecture-specific kernels. These files may
  check ISA availability, OS support, alignment, tile feasibility, scalar
  correctness preconditions, state cleanup, and plan-selected hardware variants.
  They should configure shared kernel skeletons before duplicating loop bodies,
  and must not choose between implementations for performance reasons.

Keep these boundaries strict. Shape policy does not belong inside ABI wrappers
or public Zig API code. Performance tuning should prefer descriptor parameters
and tuning scores before changing micro-kernel bodies. If a new optimized path
needs a shape gate, pack-layout mode, panel variant, AMX/SME symbol, workspace
budget, or thread assumption, add it to `catalog.zig`, `tuning.zig`,
`dispatch.zig`, or `ExecutionPlan`; do not hide it inside
`src/blas/kernels/<arch>/`.

Reusable kernel mechanics belong in `src/blas/kernels/matrix_matrix/`: shared packed-B
SIMD loops in `packed_simd.zig`, shared alpha/beta write-back in `epilogue.zig`,
and descriptor-facing parameters in `catalog.zig`/`tuning.zig`. Architecture
files should stay thin where possible: feature gates, comptime lane/tile choices,
hard safety caps, and hardware state prologue/cleanup.

## Dispatch Principles

- Dispatch on capabilities, not marketing CPU names.
- Keep exact-shape gates narrow and documented.
- Prefer conservative default behavior over broad gates with mixed data.
- Select concrete kernel combinations through descriptors: tile shape, packing
  mode, ISA, minimum work, scalar epilogue requirements, and thread assumptions.
- Select micro-kernel variants and workspace budgets through `ExecutionPlan`.
  Architecture modules should only execute the requested variant after checking
  it is feasible and correct for the visible task.
- Prefer a new comptime `Config` parameter or shared prologue/epilogue hook over
  cloning a micro-kernel body. Fork the loop only when the instruction sequence
  or hardware state model cannot be represented by the shared skeleton.
- Hardware-state prologue/epilogue code is part of the kernel contract, not a
  benchmark detail. SME, AMX, or similar stateful kernels must first prove that
  ABI-visible registers, scalar return registers, and memory ordering are
  correct before any timing result is used for dispatch decisions.
- Use internal dispatch predicates or explicit non-environment APIs/build options
  for experiments before making them default.
- Keep comparator-library measurements isolated when worker state can persist.

## Dispatch Rule Records

Any shape gate or dispatch rule that remains enabled by default must have a
written record. Put cross-platform rules here and architecture-specific rules in
the matching AArch64 or x86_64 note.

Required fields:

- Target predicate: target triple, required features, and any compile-time CPU
  tier such as ASIMD/FMA, SME, AVX2/FMA, or AVX512F/FMA.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta assumptions,
  and thread-count assumptions.
- Dispatch effect: selected `KernelId`, backend, tile/packing path, split policy,
  and any internal dispatch predicate or explicit non-environment control.
- Evidence chain: correctness command, focused benchmark command, full-sweep CSV,
  isolation level, comparator libraries, runtime environment, and summary.
- Boundary notes: nearby shapes that were tested but excluded, known unstable
  points, and the condition for disabling or narrowing the rule.

A gate can be narrower than the evidence, but it should not be broader. If the
evidence is focused, in-process, or missing comparator isolation, keep the rule
opt-in or label it experimental.

## Shape Classes

At minimum, reason about:

- Small square matrices, where fixed cost dominates.
- Medium square matrices, where packing and kernel startup both matter.
- Large square matrices, where throughput dominates.
- Tall/narrow and short/wide matrices, where task splitting can repeat packing.
- Vector-edge GEMM (`m == 1` or `n == 1`), where BLAS calls behave more like
  many dot products or one matrix-vector update than a reusable packed GEMM.
- High-K matrices, where B packing and cache behavior dominate.
- Complex GEMM, where real-kernel decomposition can multiply workspace and
  dispatch overhead.

A rule that improves one class can easily hurt another.

## Packing

Packing is useful only when the saved kernel work exceeds the packing cost.
Rules:

- Write packed buffers contiguously.
- Treat panel preparation as a reusable kernel prologue when the packed layout is
  the same. ASIMD and x86 fixed-width SIMD packed-B panels use the shared
  `packed_simd.zig` prologue; SME/AMX may keep specialized packers when the
  layout feeds assembly or hardware state directly.
- Keep small-stack pack limits explicit.
- Keep stack/cache pack budgets in descriptors or `ExecutionPlan`; architecture
  files may retain hard safety caps for fixed stack frame sizes, but not tuning
  thresholds that decide which path should be faster.
- Avoid repeated B packing across row splits unless the shape justifies it.
- Measure tail handling separately from the main tile path.
- Keep descriptor `pack.kind`, `tile.n_panel`, and stack-pack budgets aligned
  with the implementation's pack layout. Changing one without the other is a
  correctness risk, not just a tuning change.

## Threading

Thread count policy is part of dispatch, not kernel code. Good defaults must
avoid over-threading small problems and keep process-lifetime worker effects out
of reportable comparator sweeps.

Runtime controls relevant to GEMM evidence:

- `ZYNUM_MAXIMUM_THREADS`: maximum number of threads Zynum may use. Record the
  value for every benchmark, including single-thread runs. When unset, record it
  as `unset` and also record the runtime CPU count.
- `std.Io.Threaded`/`std.Io.Group.concurrent` worker use is internal dispatch
  policy, not an environment-variable mode.
- Instruction-set and AMX/SME selection are internal backend decisions, not
  environment-variable modes.

Threading changes should include single-thread and pinned multi-thread evidence.
Do not infer comparator fairness from default thread settings.

## Current Retained Generic Rules

The portable generic backend owns vector-edge fallbacks that avoid B-packing
when matrix dimensions collapse to a row or column:

- `m == 1, k >= 16`: descriptor matching strongly prefers `generic_basic`.
  The generic backend computes row-vector outputs with contiguous dot products
  when `lda == 1`, falling back to strided-A dots otherwise.
- `n == 1, k >= 16`: descriptor matching strongly prefers `generic_basic`.
  The generic backend computes the single output column by streaming contiguous
  A columns and updating C rows, which avoids the cache-unfriendly row-block dot
  pattern used by the older fallback.
- `m == 1` or `n == 1`, `k >= 128`, and work at least `128 Ki` multiply-add
  pairs: dispatch may request the default GEMM thread limit even though the
  global `runtime.gemmThreadCount` would otherwise return one thread for
  `n < 2`. This is only a task-planner rule; it does not add an environment
  variable or a separate worker strategy.

Focused M5 probes showed these rules improve the worst vector-edge points, but
they do not make those shapes competitive with Accelerate. Treat them as
fallback cleanup, not as a broad comparator-performance claim.

## Alpha And Beta

The `alpha=1,beta=0` path is often worth a dedicated store-only fast path.
General alpha/beta handling must remain correct and should be benchmarked
separately because it changes write-back cost. Real-GEMM scalar/vector write-back
formulas live in `src/blas/kernels/matrix_matrix/epilogue.zig`; kernel files should call
those helpers instead of maintaining local copies. Preserve the `beta=0` path so
kernels do not read old C values when BLAS semantics do not require it.

## Complex GEMM

The current complex paths can use real GEMM transformations. This is useful for
coverage, but it can repeat packing, allocate more workspace, and trigger
multiple real GEMM dispatches.

Long-term complex GEMM should use dedicated packing and micro-kernels where the
target architecture justifies the maintenance cost.

## Retained Policy

Retain an optimization when:

- It passes correctness tests for the relevant target.
- It improves the target shape class repeatedly in focused runs and does not
  broadly regress other shape classes in a full sweep.
- Comparator claims, if any, are backed by fresh-process data.
- It has commands, raw CSV paths, summaries, runtime environment, and isolation
  level recorded.
- The rule is expressed by capabilities and shapes, not a single machine name.
- The default gate is narrower than the measured evidence and has a rollback
  condition.

Reject or keep opt-in when:

- It only wins a single point.
- It relies on long-lived state that pollutes later comparator measurements.
- Its supporting benchmark may have been affected by numerical pollution,
  missing register preservation, stale scalar return state, or invalid inline
  assembly memory ordering. Such data is invalid for both promotion and
  rollback; rerun after the correctness issue is fixed.
- It makes nearby shapes unstable.
- It has only cross-compiled or emulator data for a throughput claim.
- It requires ABI or public API layers to know kernel details.

Do not describe an optimization as faster than a comparator unless the retained
record includes the comparator library, version or path, thread policy, and
fresh-process CSV.

Do not permanently reject an SME/SVE/SME2, AMX, or other stateful instruction
candidate based on historical data collected before its ABI prologue/epilogue
and inline-assembly clobbers were audited. Older pessimistic rollback notes are
only useful as a list of shapes to retest; they are not a reliable explanation
for why the candidate is slow.
