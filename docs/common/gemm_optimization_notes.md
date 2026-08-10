# GEMM And Structured Level 3 Optimization Notes

This page defines long-lived rules for GEMM and structured BLAS Level 3 work.
It replaces public experiment diaries with architecture-neutral contracts,
current implementation boundaries, and explicit validation and rollback gates.
Raw benchmark data, dated investigations, host identities, and scheduler output
belong outside the repository.

## Layering

The matrix-matrix stack separates semantics, selection, and execution:

1. `src/blas/core/matrix_matrix/` validates BLAS arguments, normalizes layouts,
   chooses a whole-operation algorithm, and owns fallback.
2. `src/blas/kernels/shared/matrix_matrix/catalog.zig` describes executable real
   and complex GEMM kernels with stable identities and hard contracts.
3. `src/blas/kernels/shared/matrix_matrix/tuning.zig` filters hard feasibility,
   then applies measured shape preferences.
4. The planner composes packing, bounded workspace, epilogue, and task topology
   into an execution plan.
5. `src/blas/kernels/shared/matrix_matrix/executor.zig` maps the selected ID to a
   shared or architecture-specific implementation.

Public Zig and ABI facades must not import catalog, tuning, planner-leaf, or
executor internals. ABI wrappers normalize calling conventions; they do not
make kernel choices.

## Semantic Contract

Every route must implement the complete operation

`C := alpha * op(A) * op(B) + beta * C`

for the descriptor's advertised scalar, layout, transpose, conjugation, and
epilogue classes. In particular:

- zero dimensions and `k == 0` retain correct beta behavior;
- `alpha == 0` must not read matrix elements unnecessarily;
- arbitrary supported leading dimensions and legal alias cases remain valid;
- row-major CBLAS normalization is equivalent to the column-major core;
- odd M, N, and K tails are complete;
- complex conjugation applies to the correct operand exactly once; and
- workspace or feasibility failure returns to a whole-operation fallback before
  caller output is modified.

A microkernel that supports only unit alpha, zero beta, no transpose, or a
particular packed layout is a sub-operation. Its descriptor and planner must
make those restrictions explicit; tuning cannot widen them.

## Dispatch Principles

Dispatch has two stages:

1. **Feasibility:** compiled capability, scalar domain, layout, epilogue,
   dimension, alignment, architecture state, workspace, and lifecycle.
2. **Preference:** measured shape class, tile geometry, packing mode, batching,
   and parallel topology.

Scoring must never make an infeasible kernel eligible. A failed feasibility
check follows the catalog's total fallback chain. Preference thresholds belong
in named tuning profiles, not inside assembly or hot-loop source.

Kernel IDs are semantic identities, not source paths or processor product names.
They distinguish portable, compact, packed, architecture-specific, vector-edge,
materialized, and experimental algorithms only when those bodies or contracts
are genuinely distinct.

## Shape Classes

No single GEMM route is best for all shapes. At minimum, benchmark and reason
about:

- small square matrices where call and packing overhead dominate;
- medium and large square matrices;
- narrow N and narrow M cases;
- wide and tall rectangles;
- small K and high K;
- exact tile multiples and every relevant remainder;
- transpose combinations NN, NT, TN, and TT; and
- real and complex scalar domains separately.

A shape predicate must be stated exactly and tested immediately inside and
outside each boundary. Do not broaden a rule from one representative point.

## Packing And Workspace

Packing is justified by reuse, locality, or a required microkernel layout. The
plan must account for allocation, copy, padding, and repacking costs.

- Prefer bounded stack or caller-provided workspace for small panels.
- Acquire all required workspace before changing `C`.
- Reuse a packed panel across enough tiles to repay its cost.
- Keep pack layout and tile geometry in descriptors or tuning, not duplicated in
  architecture wrappers.
- Do not materialize a complete operand when a bounded panel is sufficient.
- Make padding semantics explicit so tails cannot read uninitialized data.

If packing is rejected or workspace is unavailable, fall back as a whole call.
Partial execution followed by a portable restart would double-update `C`.

## Real GEMM

`packed_simd.zig` owns the reusable fixed-width packed-B loop shape used by
multiple architecture entrypoints. Architecture wrappers provide lane count,
tile geometry, unroll, and instruction-specific operations.

Transposed inputs should be handled by a deliberate plan:

- direct strided access for small work;
- bounded materialization when reuse repays it; or
- a native packed layout whose executor advertises the transpose contract.

The planner, not an ABI wrapper, chooses among these routes. An NN kernel may be
reused for another transpose class only after materialization and fallback
obligations are fully represented.

Alpha and beta handling belongs in the epilogue contract. Folding special
values into a microkernel is useful only if the generic path remains total and
the special route has independent boundary tests.

## Complex GEMM

Complex GEMM has distinct whole-call algorithms:

- direct interleaved arithmetic;
- compact or vector-edge paths for small dimensions;
- three-real-multiply decomposition;
- four-real-multiply expanded planes; and
- materialized transpose/conjugate variants.

Their descriptors record conjugation, plane count and layout, materialization,
packing, workspace, combine, alpha/beta support, and fallback. A decomposition
that accepts only unit alpha and zero beta must reject other epilogues before
writing caller output.

Measure the full route. Fast real sub-GEMMs do not prove a complex win when
plane extraction, padding, repeated packing, conjugation, and result combine
dominate. Small and narrow shapes commonly prefer direct interleaved kernels;
larger shapes may amortize materialization.

## Vector Edges

When one matrix dimension is one or very small, a GEMV-like route can avoid
general packing and task overhead. The selector must still preserve GEMM
leading dimensions, alpha/beta semantics, transpose/conjugate modes, and output
layout. Treat vector-edge execution as a GEMM kernel identity with a total GEMM
fallback, not as an ABI-level shortcut.

## Threading

All matrix-matrix parallelism uses the process-wide `std.Io` runtime described in
[`zig_0_16_std_io_threading.md`](zig_0_16_std_io_threading.md).

- Partition by disjoint output tiles whenever possible.
- Keep packed-panel ownership and lifetime explicit.
- Bound task count and workspace from the work size and available CPU capacity.
- Submit all fallible work before modifying `C`, or guarantee synchronous
  completion for every unsubmitted tile.
- Avoid nested worker pools in isolated objects or architecture backends.
- Measure packing, compute, waiting, and merge phases separately when diagnosing
  scaling.

Low thread caps are diagnostic. Production gates use the documented default with
`ZYNUM_MAXIMUM_THREADS` unset and record the runtime-observed concurrency.

## Structured Level 3

Structured operations share registry vocabulary with GEMM but have different
ownership and dependency contracts. Their catalog should distinguish:

- portable serial implementations;
- retained column-task implementations;
- experimental blocked or link-isolated implementations;
- rejected research implementations; and
- missing cells.

Rejected and missing rows remain enumerable. A later isolated implementation
gets a new stable ID; it does not inherit evidence from a rejected in-graph body.

### Rank-k and rank-2k updates

SYRK/HERK and SYR2K/HER2K own only the selected triangle of `C`. Parallel tasks
should own disjoint columns or bounded private tiles. Never compute a full dense
result and copy one triangle merely to reuse GEMM unless full-route evidence
proves that the extra work and memory are worthwhile.

Hermitian updates preserve a real diagonal and the correct conjugation. Beta is
applied exactly once to the selected triangle.

### Symmetric and Hermitian matrix multiply

SYMM/HEMM produce a full general `C` while reading one structured operand. A
blocked implementation may pack one active symmetric or Hermitian panel, but it
must not expand the complete operand. Side, triangle, scalar domain, and
Hermitian diagonal behavior are hard descriptor constraints.

Dense-GEMM transformation is experimental unless materialization, workspace,
and code-layout effects have been isolated and the complete call wins across
target and control shapes.

### Triangular matrix multiply and solve

TRMM/TRSM update `B` in place. Side, triangle, transpose/conjugate, unit
diagonal, and traversal order define dependencies. Parallel ownership must
respect those dependencies; a generic right-side row split is not automatically
valid.

Blocked solves must complete diagonal work and trailing updates in the required
order. Workspace denial, a rejected shape, or a capability mismatch must fall
back before any element of `B` changes.

### Isolated experimental objects

An experimental implementation may be linked through a hidden fixed-layout ABI
to prevent its code layout from perturbing the production control. Candidate
and control artifacts should differ only in the enable decision, and the object
must recheck hard predicates before writes. It may use the process-wide task
runtime but cannot instantiate another pool.

Isolation removes one measurement confounder; it does not itself justify default
selection. Promotion still requires native correctness, selected-shape wins,
off-gate controls, and a named tuning rule.

## Registry And Coverage

The catalog records executable facts and resources:

- stable identity and total fallback;
- operation, scalar, layout, transpose, and epilogue coverage;
- lifecycle and required capability/state;
- tile, unroll, tail, and packing contracts;
- workspace and output ownership; and
- complete-operation versus sub-operation status.

Coverage normalizes real GEMM, complex GEMM, and structured Level 3 into an
enumerable evidence matrix. Build, native correctness, and native performance
are independent. Cross-builds and emulation do not count as native throughput.

Forced registry tests use the same executor mapping as production while
bypassing only preference cutoffs. They must not bypass capability, layout,
state, workspace, or semantic restrictions.

## Current Implementation Boundary

The portable core remains the total semantic baseline. Production fast paths
include selected shared fixed-width, packed-SIMD, architecture-specific, and
vector-edge routes represented by active tuning profiles. Complex decompositions
and structured blocked or isolated routes are selected only where their complete
contracts and evidence are present; otherwise the planner follows a total
fallback.

External BLAS libraries are comparators only. They must never become a Zynum
compute route.

## Validation Matrix

For every candidate:

1. Build all changed capability tiers and the portable baseline.
2. Run ordinary API/ABI tests plus per-ID forced registry tests.
3. Cover all transpose, conjugate, side, triangle, diagonal, epilogue, and
   row/column-major variants advertised by the descriptor.
4. Exercise zero dimensions, odd tails, leading-dimension padding, workspace
   denial, and exact dispatch boundaries.
5. Compare candidate and same-target control in interleaved fresh processes.
6. Run representative shape classes and explicit off-gate controls.
7. Confirm the selected path and inspect at least one of disassembly, sampling,
   tracing, task timing, or phase timing.
8. Report sample count, robust statistic, dispersion, correctness status, target
   capability, runtime concurrency, and immutable artifact identity.

Comparator rows marked failed, missing, errored, or unchecked cannot support a
performance conclusion. An anomalous improvement or regression requires
mechanism-level follow-up before a gate changes.

## Retention And Rollback

Retain a path only when full-call correctness passes, selected shapes improve
beyond noise, controls meet the declared regression threshold, workspace is
bounded, and fallback is failure-before-write.

Rollback or narrow it when:

- a hard contract is encoded only as a tuning preference;
- packing, materialization, combine, or synchronization dominates the gain;
- a transpose, epilogue, or structured variant reaches an incomplete body;
- candidate/control code layout is not comparable;
- a task path can return after partial completion;
- a default rule depends on one machine identity or one sample; or
- the mechanism cannot be reproduced with an immutable artifact and recorded
  environment.

Public documentation records the surviving contract and rejection mechanism,
not the chronology of individual experiments.
