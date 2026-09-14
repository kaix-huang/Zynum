# Level 2 Optimization Notes

This document is the durable engineering contract for BLAS Level 2 kernels. It
keeps semantic, ownership, dispatch, and acceptance rules while leaving dated
experiments, host identities, raw reports, and batch-run records in ignored
private storage outside the repository.

## Ownership

- `src/blas/core/matrix_vector/` owns BLAS semantics, storage traversal,
  normalization, whole-operation planning, and portable fallback.
- `src/blas/kernels/shared/matrix_vector/` owns reusable GEMV, GER,
  symmetric/Hermitian, and triangular leaves plus catalogs, coverage, and tuning.
- Architecture directories own only target-specific entrypoints and instruction
  bodies.
- ABI wrappers convert layouts and scalar conventions; they do not select kernels.

A Level 2 leaf may complete a full operation or a precisely described
sub-operation. The descriptor must say which. A private-column delta, one
triangular dependency step, or an unpacked panel update is not a complete BLAS
operation and cannot be exposed as one.

## Semantic Contract

Optimized routes must match the portable implementation for:

- row-major and column-major CBLAS normalization;
- no-transpose, transpose, and conjugate-transpose modes;
- upper and lower storage, unit and non-unit diagonal, and side-specific rules;
- packed, banded, triangular, symmetric, and Hermitian storage;
- positive and negative vector increments;
- complex alpha/beta, conjugation, and real Hermitian diagonals;
- in-place vector updates and permitted matrix/vector aliasing; and
- empty, degenerate, and odd-tail shapes.

Preserve `y := alpha * op(A) * x + beta * y` ordering semantics. A path that
supports only unit alpha or zero beta must reject before writing and fall back as
a whole operation. Hermitian kernels must ignore or normalize the imaginary part
of the stored diagonal as required by BLAS semantics.

Triangular matrix-vector operations have loop-carried dependencies. Parallel or
blocked rewrites must preserve traversal order and unit-diagonal behavior; a
generic row split is usually invalid for in-place `trmv` or `trsv`.

## Storage-First Design

Choose the algorithm from the public storage contract before choosing an ISA.

| Family | Primary ownership unit | Main hazard |
| --- | --- | --- |
| GEMV no-transpose | output row or row block | strided access in column-major storage |
| GEMV transpose | output column/dot | reduction and conjugation |
| GER/GERU/GERC | matrix column or column block | vector stride and write ownership |
| SYMV/HEMV | selected triangle contribution | paired output updates and merge cost |
| TRMV/TRSV | dependency step | in-place ordering |
| Packed/banded variants | logical storage segment | address calculation and edge width |

Do not expand a complete structured matrix simply to call a dense kernel. A
small bounded panel or private output delta is acceptable when it removes enough
irregular traversal to repay materialization and merge costs.

## GEMV

### No-transpose

Column-major no-transpose GEMV naturally streams columns of `A` while updating
all of `y`. Useful implementations either:

- keep a bounded output block in registers and traverse columns;
- assign disjoint row ranges when row access remains efficient; or
- give tasks private deltas and merge once when shared-output updates would race.

Repeated full-length private vectors are not a default strategy. Workspace must
be bounded and acquired before caller output changes.

### Transpose and conjugate-transpose

Each output is a dot product with one stored column, so column ownership is
usually independent. Reuse Level 1 dot leaves only when their coefficient,
conjugation, stride, accumulation, and tail semantics match exactly. Complex
conjugation should be a compile-time or plan-level dimension, not a branch in the
innermost lane loop.

For narrow shapes, direct scalar or fixed-width loops can beat packing. For wide
or high-reuse shapes, panelization may help; record the materialization and
workspace cost in the plan.

## Rank-1 Updates

GER-family operations should assign disjoint matrix columns whenever storage
allows it. Scale one scalar from `y`, then update a contiguous segment of `A`
with an AXPY-like leaf. For GERC, conjugate the correct source operand once per
column rather than per vector lane.

Complex GER gates must distinguish GERU and GERC and must not infer conjugation
from the scalar type. Very narrow or short shapes commonly remain on the direct
portable path because task and dispatch overhead dominate.

## Symmetric And Hermitian Operations

A selected-triangle traversal contributes to two logical output regions.
Concurrent tasks therefore need one of:

- ownership that makes every output write disjoint;
- bounded private deltas with an explicit merge; or
- a serial traversal.

Atomic floating-point accumulation is not a substitute for a designed merge.
Avoid full dense expansion. A fixed-width leaf may compute a private delta for a
column, but the planner remains responsible for combining it exactly once.

Hermitian kernels need separate tests for upper/lower storage, diagonal handling,
conjugated off-diagonal values, and complex beta. Reusing a symmetric real-lane
loop without these checks is invalid.

### Real banded symmetric products

The production profile selects `compact_symmetric_band_fused` for f32/f64
SBMV on AArch64 when `n >= 512`, `k >= 8`, and both vector increments are one.
The stable identities use the `generic` capability: the shared fixed-width body
is portable vector code, while the architecture preference belongs to the named
profile. Both upper and lower storage retain their existing band traversal.

Each contiguous off-diagonal segment uses `symmetricAxpyDotUnitReal` to load its
matrix values once, update the direct output region, and accumulate the reflected
dot product with independent accumulators. The core still owns beta scaling,
alpha-zero handling, diagonal contributions, and the final reflected update.
The fused segment requires no workspace and does not change task scheduling.

A zero segment coefficient computes only the dot product, preserving the existing
SBMV rule that skips the direct AXPY update. Nonzero-coefficient segments reject
matrix/output or input/output overlap before any segment write; the same column
then uses the original AXPY followed by DOT. This is a segment-level rejection:
the enclosing traversal may already have scaled beta or completed earlier
columns, and must not restart the whole operation. Non-unit strides, other scalar
families, and unselected profiles keep their existing implementations. The
registered fallback remains `compact_symmetric_band`.

### Real packed symmetric products

For AArch64 f32/f64 SPMV, the production
`enable_fused_real_packed_single` preference applies only inside the existing
`n >= 512`, unit-increment route when the shared runtime selects exactly one
task. It does not increase the thread limit or turn a multi-task request into a
single-task request. The separate multi-task `enable_fused_real_packed` preference is also enabled
in production after same-source on/off validation. It reuses the existing
private-output task partition and merge; the runtime thread limits and packed
storage layout remain unchanged.

`singlePackedMvUnit` acquires an n-element private delta from the existing
thread-local workspace, initializes it, executes the same
`compact_symmetric_packed_fused` column body, and applies beta during the existing
merge. Storage stays packed; no dense matrix is materialized. The descriptor
covers an output-region contribution with a merge obligation, not a complete
standalone SPMV entrypoint. Its fallback is `compact_symmetric_packed`; production
eligibility covers the validated single-task and existing private-output
multi-task compositions; other compositions still require independent evidence.

The single-task executor rejects caller Y overlapping X or packed A, byte-count
overflow, workspace over-budget, or allocation failure before modifying caller
output. The public operation then retains its serial fallback. Workspace remains
bounded by the existing 64 MiB profile budget and participates in normal cache
cleanup. Alpha zero scales Y without reading A or X; beta zero in the merge does
not read the old Y value.

Packed columns differ from banded columns for a zero scaled X coefficient: their
existing scalar expression still evaluates `A * 0`. Such segments retain the
scalar loop so NaN/Inf behavior is not replaced by the banded AXPY skip. Segment
alias rejection likewise retains the scalar column fallback. Complex packed
products, non-unit strides, and multi-task production paths retain their prior
composition.

## Triangular Operations

For `trmv` and `trsv`, derive loop direction from upper/lower storage and
transpose mode. Do not parallelize across dependency steps. Optimization should
focus on the independent AXPY or dot work within a step, bounded blocking, and
address-calculation reduction.

Packed and banded variants should use logical-index helpers outside the hot loop
where practical. Tests must cover unit diagonals, minimal bandwidth, empty
segments, and both increment signs.

## Reusable Fixed-Width Leaves

`src/blas/kernels/shared/matrix_vector/fixed_simd.zig` owns parameterized bodies
for operation families whose inner loops differ only by lane count, unroll,
conjugation, and copy width. Architecture wrappers pass compile-time geometry and
capability constraints. Add a target-specific body only when a real instruction,
state, or data-layout difference cannot be expressed in the shared skeleton.

Sub-operation descriptors must declare their output ownership and merge
obligation. Tuning cannot promote a leaf whose executor lacks that composition.

### AArch64 assembly call preservation

The eight-column f64 complex transpose GEMV FCMLA body borrows v8-v11. Its naked
entry has no compiler-generated prologue, so
`zgemvTransFcmlaF64M128Cols8BodyAsm` explicitly saves d8-d11 in a 32-byte stack
frame and restores them before returning. These are the ABI-preserved low halves
of the vector registers; both transpose and conjugate-transpose instantiations
must retain the save/restore sequence and stack alignment. Numerical output
checks alone cannot establish this contract: caller register-preservation checks
must accompany changes to the assembly body or its entrypoint.

## Registry And Tuning

The Level 2 catalog uses stable semantic IDs and records:

- operation, scalar type, storage and transpose coverage;
- hard stride, alignment, alias, and shape requirements;
- whole-operation or sub-operation ownership;
- required capability and architecture state;
- workspace and merge obligations;
- lifecycle and total fallback behavior; and
- independent build, native-correctness, and native-performance evidence.

Coverage must enumerate missing and rejected cells rather than hiding them
through selector filtering. A target that cross-builds is not performance
supported. Forced-path tests bypass preference thresholds, not hard feasibility.

Measured thresholds, panel geometry, task count, and preferred SIMD width belong
in a named tuning profile. Kernel files contain only hard constraints. A tuning
change must include boundary controls on both sides of every new predicate.

## Parallel Planning

All new parallel paths use the shared Zig task runtime described in
[`zig_0_16_std_io_threading.md`](zig_0_16_std_io_threading.md).

- Base concurrency on the CPU capacity available to the process.
- Prefer disjoint output ownership; otherwise use bounded private deltas.
- Acquire workspace and submit fallible tasks before modifying caller output.
- Complete unsubmitted work synchronously after partial submission.
- Keep a serial route for small work, narrow dimensions, and constrained
  execution environments.
- Measure task bodies separately from submission, waiting, and merge time.

Thread caps help identify oversplitting, but production acceptance uses the
default runtime with `ZYNUM_MAXIMUM_THREADS` unset. A faster low-cap result is a
diagnostic, not permission to change the public default.

Heterogeneous schedulers may place helpers on different capacity classes. Treat
topology as a hypothesis and confirm it with task timing or tracing. Do not
encode processor numbering or a machine-specific helper identity into dispatch.

## Current Implementation Boundary

The portable core is the complete semantic baseline. Shared fixed-width leaves
cover common contiguous GEMV, GER, symmetric/Hermitian column, and triangular
sub-operations. Architecture entrypoints are selected only for the cells and
shape regions represented by the active tuning profile.

Complex routes commonly reuse real arithmetic, but materialization,
conjugation, plane layout, padding, and result combine remain visible costs.
Structured storage is not converted wholesale to dense form. Experimental
blocked or isolated-object routes remain non-default until their complete
composition has native correctness and performance evidence.

## Validation

For every changed family:

1. Run ordinary API and ABI tests in checked and optimized modes.
2. Force the changed registry cell across all supported storage, transpose,
   scalar, stride, and tail cases.
3. Test failure-before-write by denying workspace or violating a hard predicate.
4. Compare against a trusted implementation with correctness checking enabled.
5. Use fresh-process candidate/control measurements with identical build flags,
   runtime controls, affinity policy, shapes, and samples.
6. Include small, rectangular, boundary, and large shapes plus explicit off-gate
   controls.
7. Inspect disassembly, sampling, tracing, selected-path output, or task timing to
   confirm the intended mechanism.

Report medians or another predeclared robust statistic with sample counts and
dispersion. Rows marked failed, missing, errored, or unchecked cannot support a
performance conclusion.

## Retention And Rollback

Retain a rule only when the full operation is correct, selected shapes improve
beyond noise, and controls meet the declared regression threshold. Record the
stable kernel ID, capability tier, exact predicate, workspace bound, task
topology, comparator set, evidence identity, and rollback condition.

Rollback or narrow the route when:

- a storage or transpose variant reaches the wrong leaf;
- workspace failure occurs after output mutation;
- a merge is missing, duplicated, or races;
- task overhead or materialization dominates the saved work;
- an off-gate control regresses materially;
- a native capability or state assumption is unproven; or
- a result depends on processor numbering, a private filesystem layout, or one
  anomalous benchmark sample.

Public notes should keep the mechanism and decision boundary, not individual run
chronology. Detailed raw evidence belongs in ignored private storage.

### Ordered finite compact triangular paths on macOS AArch64

Real TBSV with n >= 128 and k <= n/4 can solve only the stored band instead of
scanning structural zeros across the full triangle. A bounded private vector
retains the original ascending dependency terms and multiply/subtract/divide
order. Previously solved finite values contribute structural signed-zero terms
through sign representatives when the accumulator is zero. Non-finite results
refuse the candidate before caller writes. Effective lower-triangular solves
with k >= 16 use a four-term unrolled leaf; other cases retain a separate scalar
leaf so they do not inherit its register pressure.

Real TPMV with n >= 64 computes stored triangular rows using recurrent packed
offsets. It checks logical input finiteness and stages output before committing.
Zero or non-finite results use the original implementation. Both candidates
retain arbitrary nonzero strides, avoid reading unit diagonals, check overlap and
workspace bounds, and fall back on allocation failure. Workspace is at most
64 MiB per call. Refusal preserves caller memory, not floating-point trap/flag
atomicity. Their catalog lifecycle remains experimental.

The scalar TPMV fallback is kept in a separate noinline leaf. This prevents the
finite-candidate call from extending register lifetimes and stack saves across
the old triangular loop, including calls below the candidate's size gate.
On macOS AArch64, the real TPMV dispatch entry returns directly to either that
leaf or a separate candidate-attempt helper with the same argument ABI. The
helper owns candidate failure and fallback, so calls below the size gate do not
retain arguments across a candidate attempt, allocate workspace, or scan input.
The x86 isolated dispatch remains unchanged.

The finite TPMV leaf peels the diagonal from each row: first for an effective
upper triangle, last for an effective lower triangle. The remaining loop needs
neither a diagonal test nor a final-offset test. Keep the initial positive-zero
addition, unit-diagonal multiply, and ascending term order; do not replace this
with reassociated partial sums. The unused final packed offset is bounded by
the already checked packed storage size plus n.

Non-transposed packed rows can be processed in adjacent pairs. Each packed
column supplies two adjacent matrix elements and one shared vector element,
with a separate ordered accumulator for each output row. Peel the differing
diagonal boundaries before or after the common column interval; never use a
horizontal reduction across the two outputs. Stage both results privately and
retain the single-row path for an odd tail and for transposed operations.

The macOS AArch64 real legacy TPMV leaf has an explicit 64-byte entry alignment
so changes in the finite leaf do not shift its loop instructions within cache
lines. A compile-time choice retains an unannotated leaf for other types and
platforms. Check off-gate real and complex calls after layout changes, even when
their arithmetic instructions are
unchanged; use long interleaved batches with a same-library control to separate
layout regressions from timing variability. For microsecond-scale calls, also
measure continuous native batches with input resets; more trials of a minimum
single-call timer do not remove its quantization or foreign-call overhead.
