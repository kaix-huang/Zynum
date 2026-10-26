# LAPACK

The `zynum-lapack` module provides portable, allocation-free dense linear algebra.
Cholesky, partial-pivoted LU and Householder QR support real and complex
single/double precision. Thin and rectangular SVD also support these scalar
families, with real singular values. The real symmetric eigenproblem interface
supports `f32` and `f64`. The detailed real QR and thin-SVD contracts below
remain specific to those entrypoints; later complex and external interface
contracts are recorded in the [interface guide](../interop/README.md).
Import `@import("zynum").lapack`,
or expose the independent `zynum-lapack` dependency module in the consumer build.

| Operation | Storage and result |
| --- | --- |
| `choleskyFactorize(T, matrix, triangle)` | Square matrix; overwrite the selected `.lower` or `.upper` triangle with L or U. |
| `choleskySolve(T, factor, rhs, triangle)` | Solve A X = B from a successful Cholesky factor; overwrite B with X. |
| `luFactorize(T, matrix, pivots)` | Rectangular matrix; partial row pivoting with P A = L U. L has an implicit unit diagonal. |
| `luSolve(T, factor, pivots, rhs)` | Solve A X = B from a successful square LU factor; overwrite B with X. |
| `qrFactorize(T, matrix, tau)` | Rectangular A = Q R; store R and compact reflector tails in A, with coefficients in caller-owned tau. |
| `qrApplyQ(T, factor, tau, rhs, operation)` | Apply Q or Qᵀ from the left; overwrite the right-hand-side columns. |

All matrices use the existing BLAS column-major `Matrix(T)` and `ConstMatrix(T)`
views, with `i32` dimensions and an explicit leading dimension. Within the
`zynum` facade, LAPACK and BLAS views have the same type. Independent dependency
module instances should be used separately, following the existing
`zynum`/`zynum-blas` import convention.

The real and complex Cholesky and LU implementations use private, target-sized (`usize`)
descriptors after the checked frontend validates its public views. The public
dimensions, validation precedence and numerical failure behavior remain as
documented here. This extraction does not establish LAPACK/LAPACKE semantics
or an ILP64 interface. Shared internal contract tests run with `test-lapack`
and can also be selected with `test-numerics-contract`. Complex Cholesky and LU
share the same pure complex arithmetic helpers and original scalar types;
their accumulator precision and arithmetic order are preserved. LU's borrowed
pivot codecs separate integer width and zero/one-base storage from the core's
zero-based logical row swaps. Each pivot is written directly at the original
factorization step; no conversion buffer or extra workspace is allocated.
The existing LU solve's IEEE triangular substitutions now share the private
`numerics/triangular_ieee.zig` core with the optional ordinary LAPACKE TRTRS
adapter. That core also supports transpose/adjoint and unit/nonunit diagonal
without adding a public checked LAPACK function or new resource owner. The
finite external selections and their separate INFO/NaN contracts are documented
in the [interface guide](../interop/contracts.md#optional-ordinary-trtrs-solve).
The private `numerics/triangular_inverse.zig` core also supplies the optional
ordinary TRTRI boundary. It uses the same native dense views and real/complex
arithmetic, borrowing n widened workspace values for n>1 without owning an
allocator or execution state. Unit diagonals are neither read nor overwritten;
the selected inverse column is completed before narrowing, and the opposite
triangle/padding are preserved. This internal capability does not add a public
checked `zynum-lapack` inverse function. The six-parameter generated high/work
interfaces, allocation/error order and qualified package scope are described
in the [TRTRI guide](../interop/contracts.md#optional-ordinary-trtri-inverse).
Checked pivots remain zero-based i32. Internal codec support does not establish
a standard one-based LAPACK or MKL interface.

```zig
const lapack = @import("zynum").lapack;
var a = [_]f64{ 4, 1, 1, 3 }; // Columns of [[4,1],[1,3]].
var b = [_]f64{ 6, 7 };       // A * [1,2].
const factor = try lapack.matrix(f64, &a, .{ .row_count = 2, .column_count = 2 });
const rhs = try lapack.matrix(f64, &b, .{ .row_count = 2, .column_count = 1 });
const factor_info = try lapack.choleskyFactorize(f64, factor, .lower);
if (factor_info != 0) return error.NotPositiveDefinite;
const solve_info = try lapack.choleskySolve(f64, factor.asConst(), rhs, .lower);
if (solve_info != 0) return error.UnusableFactor;
// b now contains approximately [1,2].
```

For QR, allocate `min(rows, columns)` coefficients. The following factors a
3-by-2 matrix and applies Qᵀ to a three-element right-hand side:

```zig
const lapack = @import("zynum").lapack;
var a = [_]f64{ 1, 2, 3, 4, 5, 6 }; // Two columns, three rows.
var tau: [2]f64 = undefined;
var b = [_]f64{ 7, 8, 9 };
const factor = try lapack.matrix(f64, &a, .{ .row_count = 3, .column_count = 2 });
const rhs = try lapack.matrix(f64, &b, .{ .row_count = 3, .column_count = 1 });
if (try lapack.qrFactorize(f64, factor, &tau) != 0) return error.QrNumericalFailure;
if (try lapack.qrApplyQ(f64, factor.asConst(), &tau, rhs, .transposed) != 0)
    return error.QrNumericalFailure;
// b now contains Q^T times the original b. This is not a least-squares driver.
```

## Checked contract

Factorization, solve and QR operations return `Error!usize`: zero means success; a positive result is
a **one-based numerical failure position**. LU pivot entries are **zero-based
row-swap indices**, stored in caller-owned `[]PivotInt` (`i32`) of length at
least `min(rows, columns)`. Solving requires the complete pivot sequence from a
successful factorization; `pivots[k]` must lie in `[k, n)`.

Dimensions, storage capacity, pivot bounds, prohibited overlap, and finite
input values are checked before mutation in every optimization mode, including
`fast`. The aliased BLAS view constructors keep their existing BLAS check
policy; LAPACK operations independently validate those views. Invalid arguments
return an error and leave outputs unchanged. Cholesky checks only the selected
triangle; the opposite triangle and padding are preserved. LU preserves padding
and the unused pivot tail. Solves preserve factors and pivot storage.

Solve and QR-application factor views must have `.operation = .normal`. Factor,
right-hand-side, and used pivot or tau storage must be pairwise disjoint;
overlap checks include matrix padding between columns. Only the used coefficient
prefixes and declared matrix spans participate in these checks. Unused tails or
trailing storage deliberately reused as another output are not additionally
protected from that output's writes. A zero LU diagonal or nonpositive Cholesky
diagonal is detected before changing the right-hand sides. Empty systems and
zero right-hand-side columns are validated no-ops.

Empty shapes accept any nonnegative leading dimension, including zero; negative
leading dimensions are rejected even when no elements are accessed. LU records
the first zero pivot and continues the remaining factorization steps, completing
the pivot sequence for finite arithmetic. Nonfinite arithmetic terminates early.

Numerical failure can leave a partial factor or partially solved right-hand
side. Do not use a failed result as a successful factor. Nonfinite input is an
argument error; nonfinite arithmetic is reported as a positive numerical status.
Cholesky/LU column updates can leave later rows of the current failing column
partially updated; later right-hand-side columns remain untouched on these
solves' failures. QR application processes reflectors across RHS columns, and
least squares applies Qᵀ to all RHS before back substitution, so their failures
can leave later columns transformed.

## Compact QR contract

For an m-by-n matrix, `k = min(m, n)`. The upper trapezoid stores R; column i
below its diagonal stores the tail of `v_i`, with implicit leading one.
`tau[0..k]` contains coefficients for `H_i = I - tau_i v_i v_i^T`, and
`Q = H_0 H_1 ... H_(k-1)`. Q has m rows and columns but is never materialized by
the implementation. QR writes only the logical matrix and used tau prefix.

`QOperation.normal` applies Q and `.transposed` applies Qᵀ. The right-hand side
must have m rows and may have any number of columns. Applying Q preserves the
factor and used tau prefix; it does not read R, matrix padding, or tau's unused
tail. Those unused values need not be finite. All used reflector tails, tau
coefficients, and logical RHS values are validated before mutation, even with
zero RHS columns. Tails must be finite with magnitude at most one, and tau must
be finite in `[0, 2]`; out-of-range finite values return `InvalidReflector`.
These range checks do not establish orthogonality: callers must supply matching
factors from a successful factorization, not arbitrarily scaled reflectors.

Rank-deficient matrices are accepted, and zero R diagonals are not failures.
QR does not perform column pivoting or estimate rank. A nonzero numerical status
identifies the **one-based reflector index**, including when Q application
executes reflectors in reverse order. It may leave partial matrix/tau or RHS
updates; a failed result must not be used as a successful factor.

QR uses scaled norms and compensated f64 working arithmetic, including for f32
inputs. Reflector application tracks product exponents to avoid avoidable
intermediate range loss. This is finite-precision arithmetic, not an exact dot
product: extreme cancellation can still lose small contributions, and factor
storage can round or underflow. These are initial unblocked routines with no
QR performance claims.

## Complex Cholesky and LU

The same four Cholesky/LU entry points accept `ComplexF32` and `ComplexF64`,
constructed with `complexF32(re, im)` and `complexF64(re, im)`. These are the
existing BLAS extern pair types. Cholesky treats the selected triangle as a
Hermitian matrix: lower storage yields A = L Lᴴ, upper storage A = Uᴴ U.
Both components of every selected entry must be finite, and each diagonal's
imaginary component must be exactly zero (either sign). A nonzero imaginary
diagonal returns `InvalidHermitianDiagonal` before mutation. This deliberately
stricter checked contract differs from [Netlib ZPOTF2](https://www.netlib.org/lapack/complex16/zpotf2.f),
which uses only its real part. Successful factor diagonals have a positive real part and imaginary +0.

Complex LU selects the first maximum `abs(re) + abs(im)` in widened precision.
Pivots retain the zero-based convention; products and the normal A X = B solve
are unconjugated. Cholesky substitution uses the conjugate transpose. Neither
solve accepts a transposed factor view. Scalar sums, products and normalized
complex division use `f64` for complex-f32 and `f128` for complex-f64, followed
by checked stores in the original type. This improves arithmetic range but
does not guarantee correct rounding, unlimited dynamic range or accuracy for
ill-conditioned systems. Stores can underflow; cancellation can lose small
terms. `f128` may use software arithmetic. The real Cholesky/LU implementations
retain their existing arithmetic paths.

## QR least squares

`qrLeastSquaresSolve(T, factor, tau, rhs)` solves real square or overdetermined
systems (m >= n) using a successful matching compact QR factorization. The
caller must establish full column rank and assess conditioning: there is no
pivoting, rank estimate, or near-zero diagonal threshold. The factor and tau
are read-only; RHS has m rows. On success its first n rows contain X, while
rows n through m-1 contain the trailing **coordinates of Qᵀ B**, not the
residual in the original row space. Their squared norm equals the residual
squared norm in exact arithmetic.

All logical R entries, reflector tails, tau, and RHS values must be finite;
reflector range and pairwise used-span checks precede mutation. For nonempty
RHS, an exactly zero R diagonal returns its one-based index without changing
RHS. Zero RHS is a validated no-op; n=0 leaves all of B as the residual tail.
A later numerical failure reports a one-based reflector or substitution row
and may leave partial/transformed RHS. Back substitution uses ordinary T
arithmetic, which can overflow an intermediate even when the exact solution
would be finite.

After factoring A in the QR example above, replace `qrApplyQ` with:

```zig
if (try lapack.qrLeastSquaresSolve(f64, factor.asConst(), &tau, rhs) != 0)
    return error.LeastSquaresNumericalFailure;
// b[0..2] contains approximately [-1, 2]; b[2] is a residual coordinate.
```

## Real symmetric eigenproblems

`symmetricEigenDecompose(T, input, triangle, work, eigenvalues, vectors, options)`
uses row-cyclic Jacobi rotations. Input is a read-only normal n-by-n view; only
the selected triangle must be finite. Work and vectors are separate n-by-n
mutable views, and eigenvalues has capacity at least n. All four used spans
must be pairwise disjoint. Output contents are not read on entry. On convergence,
eigenvalues ascend and matching vector columns approximate an orthonormal basis.

```zig
const lapack = @import("zynum").lapack;
const input_values = [_]f64{ 4, 1, 1, 4 };
var work_values: [4]f64 = undefined;
var vector_values: [4]f64 = undefined;
var eigenvalues: [2]f64 = undefined;
const input = try lapack.constMatrix(f64, &input_values, .{ .row_count = 2, .column_count = 2 });
const work = try lapack.matrix(f64, &work_values, .{ .row_count = 2, .column_count = 2 });
const vectors = try lapack.matrix(f64, &vector_values, .{ .row_count = 2, .column_count = 2 });
const result = try lapack.symmetricEigenDecompose(f64, input, .lower, work, &eigenvalues, vectors, .{});
if (result.status != .converged) return error.EigenproblemDidNotConverge;
// eigenvalues is approximately [3,5]; matching eigenvectors are columns.
```

`SymmetricEigenOptions.max_sweeps` defaults to 64; zero allows an immediate
convergence check. `SymmetricEigenResult` reports status, complete sweeps,
scaling exponent, remaining scaled off-diagonal maximum, and a one-based
failure index where applicable:

| `SymmetricEigenStatus` | Output contract |
| --- | --- |
| `converged` | Eigenvalues and vector columns contain the result. |
| `iteration_limit` | Work/vectors hold an unfinished iterate; eigenvalues remain unchanged. |
| `nonfinite_iteration` | Work/vectors may contain a partial rotation; eigenvalues remain unchanged. |
| `eigenvalue_out_of_range` | Iteration converged, but rescaling an eigenvalue is nonfinite in T; all eigenvalues remain unchanged. |

Argument errors preserve all outputs. Input, padding and unused suffixes are
not directly written. Workspace uses global power-of-two scaling and T stores;
rotation expressions use f64. Convergence requires the maximum off-diagonal
magnitude to be at most `4*eps(T)*max(initial_scaled_max, max_abs_diagonal)`.
This is a normwise stopping rule, not a relative-accuracy guarantee for tiny
eigenvalues. Extreme dynamic range can underflow small entries during scaling.
Eigenvector signs and bases in repeated eigenspaces are unspecified. Each
sweep costs O(n³); this reference-style implementation is not a general,
complex/Hermitian, generalized or partial-spectrum eigensolver.

## Real thin SVD

`svdDecompose(T, a, singular, right, workspace, options)` supports m >= n and
computes A = U diag(s) Vᵀ. On success, thin U overwrites the m-by-n input A,
`singular[0..n]` contains descending nonnegative values, and the n-by-n `right`
matrix contains V (not Vᵀ). All four active byte spans must be disjoint, including
matrix padding. Input A must be finite. The old singular values, V and workspace
are not read. Argument errors preserve all buffers; **every unsuccessful
numerical return also preserves A, singular values and V**, while workspace
may contain partial work.

Query `svdWorkspaceLength(T, m, n)` before allocating. It returns zero for n=0,
otherwise exactly `m*n + n*n + n` elements of `SvdWorkspaceScalar(T)`: f64 for
f32 input, f128 for f64. Dimensions, element counts and byte sizes are checked.
The one-sided row-cyclic Jacobi algorithm keeps iterated columns and V in this
wider workspace and normalizes U before rounding to T. It uses scaled norms
and compensated dot products; no Gram matrix or external backend is formed.

```zig
const lapack = @import("zynum").lapack;
var values = [_]f64{ 3, 0, 0, 0, 2, 0 }; // A is 3-by-2.
var right_values: [4]f64 = undefined;
var singular: [2]f64 = undefined;
const required = try lapack.svdWorkspaceLength(f64, 3, 2);
var scratch: [12]lapack.SvdWorkspaceScalar(f64) = undefined;
const a = try lapack.matrix(f64, &values, .{ .row_count = 3, .column_count = 2 });
const right = try lapack.matrix(f64, &right_values, .{ .row_count = 2, .column_count = 2 });
const result = try lapack.svdDecompose(f64, a, &singular, right, scratch[0..required], .{});
if (result.status != .converged) return error.SvdDidNotConverge;
// singular is approximately [3,2]; values holds U, right_values holds V.
```

`SvdOptions` defaults to 64 sweeps and normalized column-correlation tolerance
`32*eps(T)` (`relative_tolerance = null`). Explicit tolerance must be finite,
positive, and less than one. `SvdResult` reports status, complete sweeps, last
complete maximum correlation, prospective T-output orthogonality error and
count of tiny nonzero columns removed. `SvdStatus` is `converged`,
`no_convergence`, `non_finite_result`, or `range_limit`. The last covers a
stalled rotation, failed zero-vector completion or unattainable output
precision. Very tight tolerance is not silently relaxed: rounded U and V must
also meet the requested maximum error in Qᵀ Q - I before any output is stored.

No relative singular-value cutoff or numerical-rank estimate is used. Tiny
nonzero columns are removed only when an upward-rounded bound for the whole
set, added to previous discarded sets, still rounds to zero in T. This protects
jointly representable directions; it is not a certified global SVD error bound.
Rounded rotations, norms, normalization, and final subnormal conversion still
lose accuracy. Zero-output left directions receive an orthonormal completion.
Signs and bases in repeated singular subspaces are unspecified. This is an
allocation-free correctness foundation; software f128 and O(m*n²+n³) work per
sweep can be expensive. It provides no wide-matrix, complex, relative-accuracy,
rank-revealing, or LAPACK ABI guarantee.

## Scope and validation

These are serial, unblocked implementations using Zynum's own arithmetic.
Real Cholesky and LU use portable contiguous column updates and elementwise
vector operations; real Cholesky selects specialized upper/lower arithmetic after public
preflight validation. Forward substitution preserves each output's ascending
contribution order, and backward substitution retains that order as well.
QR uses an initial scalar Householder implementation. None calls Accelerate,
OpenBLAS, or another computation backend.

The [checked C/Fortran interface](../interop/README.md) exposes real
Cholesky factor/solve with explicit lengths and statuses. The optional Apple
arm64 checked C Cholesky frontend also exposes complex f32/f64 factor/solve
over the same caller-owned factors and admitted internal implementation. It is experimental
and does not provide a conventional LAPACK ABI. Complex QR, left/right Q
application and generation, rectangular complex SVD, and finite complex
standard/generalized Hermitian interfaces have since been implemented; consult
the interface guide for their separate contracts and accepted selections.
QR column pivoting and blocked LAPACK execution remain future work. Published BLAS charts do not
measure this module; the [module comparison tools](../../bench/modules/README.md) separately measure
real Cholesky/LU against eligible Accelerate/OpenBLAS operations.

The canonical root is `test/lapack/lapack_test.zig`, which also imports
QR, least-squares, eigen, SVD, complex and optimized-path tests. It checks factor reconstruction,
solution residuals, Q orthogonality/application, both triangles, rectangular
matrices, multiple right-hand sides, vector boundaries, padding, singular and
nonpositive cases, extreme finite scales, nonfinite values, and invalid inputs.
After the public inventory gates pass, run:

```sh
zig build test-lapack -Dcpu=baseline -Dtest-optimize=debug
zig build test-lapack -Dcpu=baseline -Dtest-optimize=safe
zig build test-lapack -Dcpu=baseline -Dtest-optimize=fast
```

The complete baseline `test` and native correctness paths also include this
root. See the [contributor checks](../contributors/README.md#required-checks)
and [module roadmap](../roadmap.md).

## Measured development snapshot (2026-10-10)

These results belong to the archived foundation source identified below,
before the shared storage-geometry refactor. The refactor preserves numerical
loops but changes the measured source closure; it has correctness evidence,
not a new timing capture. Do not attribute these timings to later binaries.

The full real Cholesky/LU suite was measured on Apple M5, macOS 27.0.1,
Zig 0.17.0, OpenBLAS 0.3.34 and modern Accelerate. This is an uncommitted
development snapshot based on `68dedd42076024dfc46980519ec48763075be49c`,
identified by the source archive below; that commit alone does not contain
these modules. Workers use ReleaseFast, baseline CPU and requested thread
caps of one. The measured algorithms are serial; actual competitor thread
counts were not independently measured.

Each case uses six fresh processes per backend, 30 final timing samples per
process, rotating backend order, and the median of process medians. The ratio
is eligible competitor time / Zynum time, so values above one favor Zynum.
Every case has equal log weight. These are descriptive measurements of the
checked API, including its validation; allocation, input restoration and
solve factorization setup are outside the execution timer.

| Operation | Cases | Geometric mean ratio | Strict Zynum wins |
| --- | ---: | ---: | ---: |
| Cholesky factorization (`POTRF`) | 20 | 0.2812 | 2 |
| Cholesky solve (`POTRS`) | 24 | 0.2014 | 0 |
| LU factorization (`GETRF`) | 18 | 0.4087 | 0 |
| LU solve (`GETRS`) | 12 | 0.1658 | 0 |
| **Complete primary suite** | **74** | **0.2537** | **2** |

The primary comparison selects the fastest eligible library per case:
Accelerate in 58 cases and OpenBLAS in 16. Five upper Accelerate POTRF cases
modified the unused triangle and were ineligible for those timed comparisons;
OpenBLAS remained eligible. Eight POTRS cases also observed this deviation
during untimed factor setup, while timed solves preserved the complete factors.
All backends remain in the raw results. The separately labeled fastest
numerical-output-only ratio, including the timed storage deviations, is 0.2335.
See the [comparison policy](../../bench/modules/README.md#storage-deviations-and-comparison-policy).

A second complete capture gave 0.2540 and 3/74 strict wins. The first capture
remains primary; neither run is discarded or pooled. Six of the first run's
222 backend/case groups had a process-median maximum/minimum ratio above 1.2
(maximum 1.935); the recheck had two (maximum 1.874). These ranges are not
confidence intervals. All numerical/preservation checks and 100-microsecond
batch targets passed in both captures. QR, least squares, eigen, SVD and complex
operations were not timed, and these well-conditioned fixtures do not establish
arbitrary-input numerical stability. The measured dense implementation still
substantially trails the mature libraries; blocked updates and multiple-RHS
solve kernels remain the main performance work.

Earlier prototype A/B measurements selected the contiguous column-update
implementation: 1.415x over its initial prototype across 28 cases, with 23 strict
wins. That historical result belongs to its own archived binaries and source,
not to the current 74-case capture. Its metadata SHA256 is
`afa775aedabac91e451f6a8a023eba727b46ec556019ef8d86ecef4b5829e7f6`.

### Reproduction and evidence

Use the [build and run recipe](../../bench/modules/README.md#build-and-verify)
with `--suite full` and fresh output directories. Each complete combined run
contains 74 LAPACK and 144 Sparse cases, 2,484 fresh worker processes and 74,520
final inner timing samples. Raw captures, process ranges, exact commands,
compiler/header/library identities and source archives stay outside Git:

- Primary: `zig-out/module-comparison-public-full-20261010`;
  metadata SHA256 `b05c2d46a02da779b714f07cd4f85381a3bf58b633c7636a3a9b0c7149039fdf`.
- Complete recheck: `zig-out/module-comparison-public-full-recheck-20261010`;
  metadata SHA256 `e202bd0f9cb1dbc06462352291a0efeff48082e0eec4cb5a6f4b20e17a03cd29`.
- Shared measured source archive SHA256:
  `1f654f72e6c5cb736e761ccb3a09f687a85aa5cab2881fb17d8e8aed8205703b`.
- Shared build manifest SHA256:
  `a773a4a0fe3573090c8614360e162fd838101a449940e0a9b6bf343fc6f7bac0`.

The published BLAS benchmark bundle is separate and unchanged by these runs.

## Ordinary POSV interface composition

The optional `-Dlapacke-posv=true` frontend composes the existing four-scalar
ordinary POTRF/POTRS cores for LP64 and ILP64. It owns one admission and the
POSV argument/NaN/copyback sequence; it does not call the two public managed
entrypoints in sequence. In particular, all column-major arguments are checked
before factorization, `nrhs=0` still factors A, and a positive factorization INFO
leaves B unchanged. Row-major temporary storage uses the same admitted operation
and existing scratch service. See the interface guide for finite selections,
installed client evidence and pending integration boundaries. No new numerical
algorithm or ABP production adoption is implied by this frontend.

## Finite GETRI boundary and canonical acceptance

The selected GETRI high/work/query boundary is enabled with `-Dlapacke-getri=true`. It reuses the shared Core LU inverse, Dense/Pivot data and Context-owned public/private workspaces. Four scalars, two index widths, both layouts, original headers and relocated callers are qualified. Canonical ReleaseSafe passes 1,532 bodies with four original skips across 26 roots. Fresh Main installation passes 83 steps, 86 external client processes and three updated ABPLaS regressions; 288 compared DOS dataset payloads match the accepted baseline. ABPLaS uses no GETRI entry. Complete vendor APIs and full LAPACK coverage remain pending. See the GETRI checkpoint in `docs/interop/README.md` for exact evidence and remaining limits.

## Finite POTRI component and canonical acceptance

The optional ordinary POTRI high/work boundary uses five parameters at both levels. Its generated LP64/ILP64 component forwards to the same Core Cholesky inverse and Context-owned scratch lifecycle. Four scalars, both layouts, valid Hermitian-factor cases, singular/NaN/INFO boundaries, returning diagnostics and close/rebind behavior are qualified on the recorded candidates. Canonical ReleaseSafe passes 1,556 actual Zig bodies with four original skips; the complete selected original 51-method source regression also passes. Fresh Main installation passes 93 steps, 108 C/C++ consumers and 32 separate invalid-alignment guards; three updated ABPLaS regressions pass, including 288 unchanged DOS dataset payloads. ABPLaS adopts no POTRI entry in this phase. See the POTRI checkpoint in `docs/interop/README.md` for precise scope and remaining coverage.
