// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Checked public SVD facade over the shared, allocation-free numerical core.
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const views = @import("../blas/api/views.zig");
const core = @import("core/svd.zig");
const complex_core = @import("core/complex_svd.zig");
const scalar = @import("../blas/core/shared/scalar.zig");
pub const Error = core.Error;
pub const Status = core.Status;
pub const Result = core.Result;
pub const Options = core.Options;
fn implementation(comptime T: type) type {
    return if (scalar.isComplex(T)) complex_core else core;
}
pub fn WorkspaceScalar(comptime T: type) type {
    return implementation(T).WorkspaceScalar(T);
}

/// Exact workspace length in widened elements; signed public dimensions are checked.
pub fn workspaceLength(comptime T: type, m: views.BlasInt, n: views.BlasInt) Error!usize {
    if (m < 0 or n < 0) return error.InvalidLength;
    return implementation(T).workspaceLength(T, @intCast(m), @intCast(n));
}
fn span(a: anytype) Error!usize {
    if (a.row_count < 0 or a.column_count < 0) return error.InvalidLength;
    if (a.leading_dimension < 0) return error.InvalidLeadingDimension;
    if (a.row_count == 0 or a.column_count == 0) return 0;
    if (a.leading_dimension < a.row_count) return error.InvalidLeadingDimension;
    const used = storage.columnMajorExtent(@intCast(a.row_count), @intCast(a.column_count), @intCast(a.leading_dimension)) catch return error.InvalidLength;
    if (a.values.len < used) return error.BufferTooSmall;
    return used;
}
/// Thin SVD A=U*diag(singular)*V^H for m>=n (V^T for real inputs).
/// All four scalar families are supported; singular values use Real(T). U overwrites A, and right is
/// n-by-n V. Singular values descend; vector signs and repeated-subspace bases
/// are unspecified. Supply workspaceLength(T,m,n) WorkspaceScalar(T) elements:
/// contiguous m-by-n iterated A, n-by-n iterated V, and n column norms.
/// Complex workspace uses existing widened real/imaginary pairs. No
/// allocation/backend is used. Output padding/workspace suffix are not written
/// directly. Unused storage may alias a different active operand.
///
/// All dimensions, capacities, finite input, tolerance and mixed-type byte-span
/// aliases are checked before mutation, including fast builds. All four used
/// operand spans are pairwise disjoint; matrix spans include intervening padding.
/// Workspace old contents and right/singular old contents are not read. Empty
/// shapes accept nonnegative leading dimensions. A, right and singular remain
/// unchanged on ANY unsuccessful return; workspace may contain partial work.
///
/// Default tolerance is 32*eps(Real(T)). Each complete sweep is followed by a full
/// working column-correlation check. max_sweeps=0 permits already orthogonal
/// input. Widened storage prevents T-subnormal quantization during iteration;
/// scaled norms and compensated normalized dots limit range loss. Only a whole
/// set of tiny columns whose conservatively bounded cumulative discarded norm
/// rounds to zero in T may be deflated before convergence. No relative rank
/// threshold is used. Computed zero-output directions receive an orthonormal
/// completion. Roundoff can still turn algebraic zeros into small nonzero
/// singular values; this is not a rank estimator or an exact/relative-accuracy
/// guarantee for arbitrary ill-conditioned matrices.
///
/// Success also requires the prospective T-rounded U and V to satisfy
/// max(abs(Q^H*Q-I)) <= tolerance (Q^T for real inputs), reported as output_orthogonality. An
/// unattainable requested tolerance, stalled rotation or failed completion
/// returns range_limit; no_convergence reports an exhausted sweep budget.
/// max_correlation describes the last full working check (infinity for an
/// unrepresentable norm). Out-of-range singular values return non_finite_result.
/// Widened storage, including f128 arithmetic for f64, is an unmeasured reference
/// approach, not LAPACK DGESVJ equivalence or a performance claim.
pub fn decompose(comptime T: type, a: views.Matrix(T), singular: []scalar.Real(T), right: views.Matrix(T), workspace: []WorkspaceScalar(T), options: Options) Error!Result {
    var admission = execution.beginDefault();
    defer admission.end();
    // Preserve public signed validation order before constructing native descriptors.
    _ = try span(a);
    _ = try span(right);
    return implementation(T).decompose(T, .{ .values = a.values, .row_count = @intCast(a.row_count), .column_count = @intCast(a.column_count), .leading_dimension = @intCast(a.leading_dimension) }, singular, .{ .values = right.values, .row_count = @intCast(right.row_count), .column_count = @intCast(right.column_count), .leading_dimension = @intCast(right.leading_dimension) }, workspace, options);
}

/// Rectangular thin SVD workspace in widened scalar/pair elements.
pub fn rectangularWorkspaceLength(comptime T: type, m: views.BlasInt, n: views.BlasInt) Error!usize {
    if (m < 0 or n < 0) return error.InvalidLength;
    return implementation(T).workspaceLength(T, @intCast(@max(m, n)), @intCast(@min(m, n)));
}
/// Additional T elements for the private transpose/conjugate transpose of wide A.
pub fn rectangularScratchLength(comptime T: type, m: views.BlasInt, n: views.BlasInt) Error!usize {
    if (m < 0 or n < 0) return error.InvalidLength;
    return implementation(T).rectangularScratchLength(T, @intCast(m), @intCast(n));
}
/// Thin SVD for every rectangular shape. First k=min(m,n) columns of A become
/// U and right is n-by-k V; trailing A columns remain untouched. Complex inputs
/// use A^H for the wide reduction. The same transactional, finite, tolerance and
/// mixed-byte-span rules apply as decompose. Exactly one existing admission is
/// borrowed by the selected pure Core; no additional Context or workspace owner.
pub fn decomposeRectangular(comptime T: type, a: views.Matrix(T), singular: []scalar.Real(T), right: views.Matrix(T), workspace: []WorkspaceScalar(T), transpose_scratch: []T, options: Options) Error!Result {
    var admission = execution.beginDefault();
    defer admission.end();
    _ = try span(a);
    _ = try span(right);
    return implementation(T).decomposeRectangular(T, .{ .values = a.values, .row_count = @intCast(a.row_count), .column_count = @intCast(a.column_count), .leading_dimension = @intCast(a.leading_dimension) }, singular, .{ .values = right.values, .row_count = @intCast(right.row_count), .column_count = @intCast(right.column_count), .leading_dimension = @intCast(right.leading_dimension) }, workspace, transpose_scratch, options);
}
