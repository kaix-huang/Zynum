# Sparse linear algebra

The `zynum-sparse` module provides borrowed CSR, CSC and COO storage views,
explicit conversion from CSC/COO to sorted CSR, sparse matrix-vector
multiplication (SpMV), sparse-by-dense matrix multiplication (SpMM), real
triangular solves and real conjugate gradient.
Import `@import("zynum").sparse`, or expose the independent `zynum-sparse`
dependency module. Products and conversions support `f32`/`f64` and `ComplexF32`/`ComplexF64`
values with `i32`/`u32`/`i64`/`u64` indices, with no hidden allocation or external compute backend.

Checked CSR validation now uses a private compressed-pattern layer that
separates offsets, ordinals and numerical values. That layer can describe
independent index widths, one-based indices and split physical ranges; the
public views retain their existing contracts below. CSR SpMV and the odd
SpMM column tail now call one internal compressed-product implementation.
The existing two-column SpMM kernel remains shared and unchanged.

The core executes row- or column-compressed operands with independent offset
and ordinal widths, zero/one index bases, and contiguous or split ranges. It
preserves physical traversal and duplicate contributions, reads only active
entries, and implements normal/transpose/adjoint without sorting or copying.
Callers must validate dimensions, borrowed structure and disjointness before
entry; this internal contract does not replace public per-call validation.
These facilities prepare future adapters; they do not add public mixed-index
operations or establish oneMKL compatibility. Their internal tests run with
`test-sparse` and `test-numerics-contract`.

```zig
const sparse = @import("zynum").sparse;
const offsets = [_]u32{ 0, 2, 3 };
const columns = [_]u32{ 0, 2, 1 };
const values = [_]f64{ 2, 3, 4 }; // [[2,0,3],[0,4,0]].
const a = try sparse.constCsrMatrix(f64, u32, &offsets, &columns, &values, .{
    .row_count = 2,
    .column_count = 3,
});
const x = [_]f64{ 1, 2, 3 };
var y: [2]f64 = undefined;
try sparse.matrixVectorMultiply(.{
    .matrix = a,
    .input_vector = &x,
    .result_vector = &y,
});
// y == [11,8]. Use a.transposed() for the transposed product.
```

## Storage and operations

All indices are zero-based. `ConstCsrMatrix(T, Index)` borrows values, column
indices and row offsets. Row offsets have exactly `row_count + 1` entries,
start at zero, are nondecreasing, and end at the value count. Column indices
and values have equal lengths; every column index must be in range. Empty
rows, explicit zeros, unsorted columns and duplicate entries are accepted.
Each stored duplicate contributes separately in storage order. Construction
and multiplication do not sort, coalesce or convert storage.

`ConstCscMatrix(T, Index)` borrows values, row indices and column offsets with
the corresponding `column_count + 1` offset requirement. Rows within a column
may be unsorted or repeated. `ConstCooMatrix(T, Index)` borrows equal-length
row-index, column-index and value slices in arbitrary coordinate order.
Their constructors require `row_count` and `column_count` options. These views
serve as inputs to the explicit conversion functions below; multiplication
accepts CSR.

`matrixVectorMultiply` computes `y = product_scale * op(A) * x + result_scale * y`.
Input and result are contiguous slices of exactly the effective column and row
counts. Scales default to one and zero. `a.transposed()` changes the logical
operation without moving storage; calling it again keeps the transposed view.

`matrixMultiply` takes `left_matrix` (CSR), `right_matrix` (dense input), and
`result_matrix` (dense output), with the same scale defaults. The dense
`ConstMatrix(T)`/`Matrix(T)` views use column-major storage and `usize`
dimensions; `constMatrix`/`matrix` constructors default the leading dimension
to the row count. Leading dimensions must be at least the row count, including
empty views. Only logical output elements are changed; inter-column padding
and unused trailing storage are preserved.

SpMM reuses sparse indices and values across two dense columns at a time for
both normal and transposed products. An odd final column follows the single
column path. Each output retains the original CSR accumulation order. This
portable serial implementation has no dense packing, multiplication workspace,
worker pool, setup cache or architecture-specific dispatch. Normal products
combine output scaling with the final row store; transposed products scale
output before scattering sparse contributions.

## Convert CSC or COO to sorted CSR

Conversions preserve duplicates as separate additive entries. Columns become
nondecreasing within each row, and entries with equal coordinates retain their
source storage order. This is **sorted CSR**, not a duplicate-free canonical
format. Values are copied without arithmetic, including signed zeros and NaN
payloads. Sorting distinct columns can change later floating-point summation.

```zig
const sparse = @import("zynum").sparse;
const rows = [_]u32{ 1, 0, 1, 0 };
const columns = [_]u32{ 2, 1, 2, 0 };
const values = [_]f64{ 2, 3, 4, 5 };
const coo = try sparse.constCooMatrix(f64, u32, &rows, &columns, &values, .{
    .row_count = 2,
    .column_count = 3,
});
const required = try sparse.cooToCsrRequirements(u32, 2, values.len);
// required: 3 row offsets, 4 column indices, 4 values, 266 usize workspace items.
var offsets_out: [3]u32 = undefined;
var columns_out: [4]u32 = undefined;
var values_out: [4]f64 = undefined;
var workspace: [266]usize = undefined;
const csr = try sparse.cooToCsr(.{
    .matrix = coo,
    .row_offsets = offsets_out[0..required.row_offsets],
    .column_indices = columns_out[0..required.column_indices],
    .values = values_out[0..required.values],
    .workspace = workspace[0..required.workspace],
});
// csr.row_offsets == [0,2,4], csr.column_indices == [0,1,2,2],
// csr.values == [5,3,2,4]. The duplicate values 2 and 4 remain separate.
const x = [_]f64{ 1, 2, 3 };
var y: [2]f64 = undefined;
try sparse.matrixVectorMultiply(.{
    .matrix = csr,
    .input_vector = &x,
    .result_vector = &y,
});
// y == [11,18].
```

For CSC, construct the view with
`constCscMatrix(T, Index, column_offsets, row_indices, values, options)`, query
`cscToCsrRequirements(Index, row_count, nnz)`, then call `cscToCsr` with the
same output/workspace argument fields as above. Both conversions retain the
source scalar and index types and return a normal `ConstCsrMatrix` view.

`ConversionRequirements` reports element counts, not bytes. Callers supply
at least these capacities, using arrays or their own allocator:

| Buffer | CSC to CSR | COO to CSR | Element type |
| --- | --- | --- | --- |
| `row_offsets` | `rows + 1` | `rows + 1` | Source index type |
| `column_indices` | `nnz` | `nnz` | Source index type |
| `values` | `nnz` | `nnz` | Source scalar type |
| `workspace`, if `nnz > 0` | `rows` | `rows + 2 * nnz + 256` | `usize` |
| `workspace`, if `nnz == 0` | `0` | `0` | `usize` |

Returned CSR slices cover exactly the required output prefixes. The result
borrows those output buffers, which must remain alive; input and workspace
can be reused immediately after conversion. Only active prefixes are accessed
through output/workspace operands, and unused capacity is excluded from alias
checks. An unused tail may still change if it is reused as another operand's
active writable region. Active output/workspace ranges must be mutually
disjoint and must not overlap any source array. Read-only source arrays may
overlap one another. In-place conversion is not supported.

Zero dimensions are valid only with no stored entries. Empty conversion writes
`rows + 1` zero offsets and requires no workspace. Requirements check `rows + 1`,
workspace arithmetic and whether the index type can represent terminal `nnz`.
CSC conversion also checks that each nonempty implicit column is representable
in the output index type. Every conversion revalidates the borrowed source,
capacities and active byte ranges before changing any output or workspace,
including in `fast` builds. Invalid structure, insufficient capacity, overflow
or prohibited aliasing returns an error before mutation.

## Validation and cost

CSR construction and `.check()` take O(rows + nnz), CSC takes O(columns + nnz),
and COO takes O(nnz). Structural checks do not read numerical values. CSC to
CSR conversion takes O(columns + rows + nnz); COO uses a stable radix ordering
and row scatter in O(rows + sizeof(Index) * (nnz + 256)) time. Its two `nnz`
permutations and 256 radix counters reside in caller workspace, with no
column-dimension-sized table. Requirements queries perform only checked
arithmetic and do not validate a source matrix.

Every multiplication repeats CSR structural validation before changing output,
including in `fast` builds, because callers may change borrowed buffers between
calls. SpMM validates once per call, independently of the number of dense
columns. Dense view checks take constant time.

Multiplication output may not overlap dense input, sparse values, row offsets
or column indices. Dense overlap checks include inter-column padding and
exclude unused trailing storage. Read-only inputs may overlap each other.
Callers retain ownership and must keep buffers alive and free from concurrent
mutation. Pointer validity is the caller's responsibility.

Invalid structure, dimensions, storage, arithmetic size overflow or prohibited
aliasing returns an error before any output mutation. Structural validation
still runs for zero scales. `product_scale = 0` does not read numerical input
values; `result_scale = 0` does not read old output values, including NaNs.
An empty product only scales the output.

## Complex products and conversion

CSR/CSC/COO and dense views also accept `ComplexF32` and `ComplexF64`, using
`complexF32(re, im)` or `complexF64(re, im)`. Scales have the same scalar type.
`a.adjoint()` selects conjugate transpose relative to stored CSR; it conjugates
matrix values only, not the input or scales. Like `.transposed()`, it sets the
operation rather than composing transforms. For real scalars it is equivalent
to transpose. Normal, transpose and adjoint products support all four index
types, unsorted duplicates and multiple dense columns.

A complex scale is zero when both components compare equal to zero, including
signed zeros. Zero product scale skips numerical input reads after preflight;
zero result scale skips old-output reads. Empty products only scale output.
Complex multiplication uses ordinary component products and sums, so extreme
magnitudes can overflow/underflow and used NaNs/Infs propagate. It is not a
scaled complex multiplication algorithm. Conversion preserves component bits,
including NaN payloads and signed zeros, and retains equal-coordinate order.

## Real CSR triangular solves

`triangularSolveInPlace` takes `matrix`, `triangle` (`.lower`/`.upper`),
`diagonal` (`.unit`/`.non_unit`), column-major mutable `rhs`, and `workspace`.
It solves op(A) X = B for square real CSR matrices; RHS must have matching rows.
Triangle names the original stored matrix. Transpose reverses the solve
direction; real adjoint has the same effect. Complex solves are not supported.

Query `triangularSolveWorkspaceLength(T, n, rhs_count, diagonal)`. Non-unit,
nonempty solves need exactly n T elements for aggregate diagonals; unit,
empty and zero-RHS solves need zero. Strict-triangle duplicates contribute
separately in stored order. Diagonal duplicates sum from positive zero in
stored order. Missing or zero non-unit diagonals are singular; aggregate
overflow is a numerical failure even if a later duplicate would cancel it.
Unit diagonals are one and never read numerically. Opposite-triangle values
are also ignored, though all structural indices remain checked.

```zig
const sparse = @import("zynum").sparse;
const lower = try sparse.constCsrMatrix(f64, u32,
    &.{ 0, 1, 3 }, &.{ 0, 0, 1 }, &.{ 2, 1, 3 },
    .{ .row_count = 2, .column_count = 2 });
var b = [_]f64{ 2, 7 }; // Lower triangular A times [1,2].
var scratch: [2]f64 = undefined;
const rhs = try sparse.matrix(f64, &b, .{ .row_count = 2, .column_count = 1 });
const result = try sparse.triangularSolveInPlace(.{
    .matrix = lower, .triangle = .lower, .diagonal = .non_unit,
    .rhs = rhs, .workspace = &scratch,
});
if (result.info != 0) return error.TriangularSolveFailed;
// b is approximately [1,2].
```

Before any write, the solver validates structure, dimensions, capacity, finite
used values, RHS, and active byte-range aliases, then dry-runs all non-unit
diagonals. RHS and workspace must be mutually disjoint and disjoint from every
CSR buffer. Zero RHS/empty systems perform structural checks only and read no
numerical values. Argument errors and diagonal failures preserve both outputs.

`TriangularSolveResult.info` is zero on success or a one-based logical equation
index. `failure` identifies `singular_diagonal`, `non_finite_diagonal`, or
`non_finite_result`; `rhs_column` is zero on success or diagonal preflight failure, and one-based
only for execution failure. `completed_rhs` counts fully solved columns and
equals the total RHS count on success.
During execution failure the current column may contain partial finite work,
previous columns are complete, and later columns are untouched. Input and
padding are preserved. No rollback, sorting or hidden allocation occurs.

## Real conjugate gradient

`conjugateGradient` solves one real square **caller-supplied symmetric positive
definite** CSR system. Structural validation does not establish symmetry or
positive definiteness. Only `.normal` views are accepted; other operations
return `UnsupportedTransform`. Required fields are `matrix`, read-only `rhs`,
mutable `solution`, `workspace`, `initial_guess` (`.zero`/`.provided`),
`relative_tolerance`, `absolute_tolerance`, and `max_iterations`.

```zig
const sparse = @import("zynum").sparse;
const a = try sparse.constCsrMatrix(f64, u32,
    &.{ 0, 2, 4 }, &.{ 0, 1, 0, 1 }, &.{ 4, 1, 1, 3 },
    .{ .row_count = 2, .column_count = 2 });
const b = [_]f64{ 6, 7 };
var x: [2]f64 = undefined;
const required = try sparse.conjugateGradientWorkspaceLength(f64, 2);
var scratch: [8]f64 = undefined;
const result = try sparse.conjugateGradient(.{
    .matrix = a, .rhs = &b, .solution = &x,
    .workspace = scratch[0..required], .initial_guess = .zero,
    .relative_tolerance = 1e-12, .absolute_tolerance = 0,
    .max_iterations = 20,
});
if (result.status != .converged) return error.IterativeSolveDidNotConverge;
// x is approximately [1,2].
```

Workspace is exactly 4*n T elements. Vectors have exact length n. Solution and
active workspace must be mutually disjoint and disjoint from all inputs;
read-only inputs may overlap. Every call preflights structure, normal operation,
shape, sizes, aliases, finite matrix/RHS values, and finite nonnegative
tolerances. A provided guess must be finite; `.zero` ignores old solution values.
Argument errors preserve solution and workspace. Input must remain unchanged
throughout the call, which does not repeat structure checks per internal product.

`ConjugateGradientResult(T)` reports status, committed update count, RHS norm,
tolerance and optional true-residual norm. The threshold is
`absolute_tolerance + relative_tolerance * ||rhs||₂`. Every convergence candidate
recomputes b - A x in wider precision and tests its norm before narrowing it to
T. A misleading recursive residual triggers a restart within the same update
budget. Any non-null reported true residual belongs to the returned solution.
A zero cap still validates and checks the initial guess; `.zero` initializes x.
Empty systems converge immediately. Zero tolerance requests a zero computed
widened residual, not proof of exact real-arithmetic equality.

| `ConjugateGradientStatus` | Meaning |
| --- | --- |
| `converged` | Fresh true residual meets the additive threshold. |
| `max_iterations` | Last complete iterate fails the threshold at the update cap. |
| `nonpositive_curvature` | pᵀ A p <= 0 before verified convergence; may indicate an invalid SPD assumption or floating-point breakdown. |
| `nonfinite` | A numerical result cannot be represented, or a computed scalar is nonfinite. |

Trial solution and recursive residual are checked before committing an update.
Numerical failure preserves the latest complete iterate, though workspace may
be partial and is not resumable state. Dot products, CSR sums and residuals use
compensated `ConjugateGradientAccumulator(T)` arithmetic (f64 for f32, f128 for
f64); stored vectors remain T. This is finite precision, unscaled and
unpreconditioned. Underflow can erase directions, roundoff can cause stagnation,
and small residual does not guarantee small solution error. No status certifies
SPD. Preflight and each iteration cost O(n+nnz), with additional traversals for
true-residual verification. There are no workers, hidden allocation, callbacks,
preconditioner, complex solve, multiple RHS or performance guarantee.

## Scope and validation

Reverse format conversion, mixed index types, complex triangular/iterative
solvers, preconditioning and reusable analyzed structures remain follow-up
work. The [checked C/Fortran interface](../interop/README.md) exposes real SpMV
with unsigned 32/64-bit indices. The published BLAS charts do not cover this
module; [module comparison tools](../../bench/modules/README.md) separately
measure real CSR products against Accelerate. Comparisons must report view
construction and format conversion as separate setup costs and include repeated validation in the measured checked
operation; competitor setup and indexing conversions must also be disclosed.

The canonical `test/sparse/sparse_test.zig` root includes format, complex, triangular, CG, optimization
boundary and randomized conversion tests. It compares rectangular normal and
transposed products with independent dense references and covers all supported
scalar/index types, duplicate order, two-column blocks and odd tails, zero
scales, empty shapes, padding, malformed or mutated storage, overflow, capacity
and aliasing. Conversion checks include roundtrips and 500 deterministic random
cases compared with an independent coordinate-order reference. After the
public inventory gates pass:

```sh
zig build test-sparse -Dcpu=baseline -Dtest-optimize=debug
zig build test-sparse -Dcpu=baseline -Dtest-optimize=safe
zig build test-sparse -Dcpu=baseline -Dtest-optimize=fast
```

The complete baseline `test` and native correctness paths also include this
root. See the [contributor checks](../contributors/README.md#required-checks)
and [module roadmap](../roadmap.md).

## Measured development snapshot (2026-10-10)

These results belong to the archived foundation source, before the shared
storage-geometry refactor. That refactor retains numerical loops but changes
the source closure; no later binary inherits these timing results. The source
identities below remain the reproduction baseline.

The full real CSR product suite compares the checked Zynum API with Accelerate
on Apple M5, using ReleaseFast, baseline CPU and requested thread caps of one.
It covers both precisions, three shapes, three sparsity patterns, N/T operations,
RHS counts 1/8, and tight/padded dense storage. Zynum uses signed 32-bit indices;
Accelerate uses its native mixed 64-bit row offsets and 32-bit column indices.
Index conversion and setup are reported separately. Timed Zynum calls repeat
structural validation; timed Accelerate calls construct borrowed descriptors.

Each case has four fresh processes per backend, rotating order, with 30 final
samples per process. Ratios are Accelerate time / Zynum time using medians of
process medians and equal log weight per case; values above one favor Zynum.

| Operation | Cases | Geometric mean ratio | Strict Zynum wins |
| --- | ---: | ---: | ---: |
| Normal SpMV | 36 | 0.5001 | 0 |
| Normal SpMM, 8 RHS | 36 | 1.1351 | 28 |
| Transposed SpMV | 36 | 0.4839 | 0 |
| Transposed SpMM, 8 RHS | 36 | 0.9883 | 18 |
| **Complete primary suite** | **144** | **0.7218** | **46** |

Normal multiple-RHS products lead on this set, but single-vector products and
the complete suite remain slower overall. The complete recheck gave 0.7211 and
46/144 strict wins. Both captures are retained separately. Seven of the first
run's 288 backend/case groups had process-median maximum/minimum ratios above
1.2 (maximum 3.035), mainly at the start of the sparse run. The recheck's maximum
was 1.055, with no group above 1.2. No samples were removed. These are descriptive
ranges, not confidence intervals. All numerical, input-preservation, padding
and 100-microsecond batch checks passed. These fixtures do not time complex
products, conversions, triangular solves or CG. See the shared
[source identities and reproduction record](../lapack/README.md#reproduction-and-evidence)
and [measurement contract](../../bench/modules/README.md#sparse-timing-and-correctness).

### Retained and rejected optimizations

Earlier prototype A/B measurements selected the current approach of specialized
normal traversal and two-column RHS blocks: 1.353x over the initial prototype
across 72 cases, winning all 72. That result measures the historical prototype
binaries, not the later integrated snapshot; it shares the historical metadata
identity recorded in the LAPACK document.

A subsequent adjacent-row interleaving candidate was tested against the
unchanged integrated baseline and Accelerate in six rotating process repeats:
72 primary cases plus a separate 24-case edge set, totaling 1,728 processes and
51,840 samples. The primary candidate speedup over baseline was 1.0255, and
normal SpMV improved by 1.0917. The edge set ratio was 0.9930. Three primary and
seven edge cases were over 5% slower with disjoint observed process-median
ranges; the largest primary slowdown was 33.24%. This included normal f64
SpMM cases whose logical traversal was not the intended optimization target.
No code-generation or hardware cause is claimed without further evidence.

The candidate was rejected under the review policy written before measurement;
the existing implementation remains in place. A narrower predicate would be
a new candidate requiring validation and new measurements. The complete
rejected capture is `zig-out/sparse-optimization-stage3-benchmark/results-quiet-stage8`,
metadata SHA256
`522a4be168a834c6cd6349e9af81cf8d1154221daa82df6fb53d9f6f68fb3a2a`.
