// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const views = @import("../blas/api/views.zig");
const v = @import("validation.zig");

const dense = @import("../numerics/dense.zig");
const core = @import("core/lu.zig");
const pivot_codec = @import("../numerics/pivot.zig");

/// Factor an m-by-n matrix with partial row pivoting: P*A=L*U. The diagonal
/// of L is implicit one; U occupies the diagonal and upper trapezoid. P is the
/// ordered sequence of row swaps (k, pivots[k]), with ZERO-BASED pivot values.
/// The first min(m,n) pivot entries are outputs; their unused tail and matrix
/// padding remain untouched. Pivot storage must not overlap the matrix span.
/// Finite input and all argument checks precede mutation in every build.
/// Returns zero on success or the ONE-BASED first failing step. As in GETRF,
/// a zero pivot is recorded and factorization continues without dividing its
/// zero column: all pivot entries are completed and P*A=L*U remains available
/// for finite arithmetic. Nonfinite arithmetic instead terminates immediately,
/// preserving any earlier failure index; the matrix and pivot prefix may then
/// be partial. Do not pass a factor with nonzero information to luSolve.
pub fn factorize(comptime T: type, a: views.Matrix(T), pivots: []views.BlasInt) v.Error!usize {
    const span = try v.matrixSpan(T, a);
    const rows: usize = @intCast(a.row_count);
    const columns: usize = @intCast(a.column_count);
    const steps = @min(rows, columns);
    if (pivots.len < steps) return error.BufferTooSmall;
    try v.noOverlap(T, a.values[0..span], views.BlasInt, pivots[0..steps]);
    try v.finiteMatrix(T, a);
    const operand: dense.Matrix(T) = .{
        .values = a.values,
        .row_count = @intCast(a.row_count),
        .column_count = @intCast(a.column_count),
        .leading_dimension = @intCast(a.leading_dimension),
    };
    // Validated nonnegative i32 dimensions already prove zero-based i32 pivots
    // representable; do not insert a new fallible check into the old preflight.
    return core.factorize(T, operand, pivot_codec.Writer(views.BlasInt, .zero){ .values = pivots[0..steps] });
}

/// Solve A*X=B in place in rhs using a successful square LU factor. Factors
/// must have operation=.normal. pivots contains the ZERO-BASED swap sequence
/// from luFactorize; entry k must be in [k,n). All views and pivot entries,
/// finite inputs, and pairwise disjoint factor/rhs/used-pivot spans are checked
/// before mutation. A zero U diagonal returns its ONE-BASED index with rhs
/// unchanged. Nonfinite substitution arithmetic returns a one-based row and
/// may leave rhs partially solved. Factor, pivots, and padding are untouched.
/// Empty systems and zero RHS columns are validated no-ops.
pub fn solve(comptime T: type, factor: views.ConstMatrix(T), pivots: []const views.BlasInt, rhs: views.Matrix(T)) v.Error!usize {
    const factor_span = try v.normalFactor(T, factor);
    const rhs_span = try v.matrixSpan(T, rhs);
    if (rhs.row_count != factor.row_count) return error.DimensionMismatch;
    const n: usize = @intCast(factor.row_count);
    if (pivots.len < n) return error.BufferTooSmall;
    try v.noOverlap(T, factor.values[0..factor_span], T, rhs.values[0..rhs_span]);
    try v.noOverlap(T, factor.values[0..factor_span], views.BlasInt, pivots[0..n]);
    try v.noOverlap(T, rhs.values[0..rhs_span], views.BlasInt, pivots[0..n]);
    for (0..n) |k| {
        if (pivots[k] < 0) return error.InvalidPivot;
        const pivot: usize = @intCast(pivots[k]);
        if (pivot < k or pivot >= n) return error.InvalidPivot;
    }
    try v.finiteMatrix(T, factor);
    try v.finiteMatrix(T, rhs);
    const factor_operand: dense.ConstMatrix(T) = .{
        .values = factor.values,
        .row_count = @intCast(factor.row_count),
        .column_count = @intCast(factor.column_count),
        .leading_dimension = @intCast(factor.leading_dimension),
    };
    const rhs_operand: dense.Matrix(T) = .{
        .values = rhs.values,
        .row_count = @intCast(rhs.row_count),
        .column_count = @intCast(rhs.column_count),
        .leading_dimension = @intCast(rhs.leading_dimension),
    };
    return core.solve(T, factor_operand, pivot_codec.Reader(views.BlasInt, .zero){ .values = pivots[0..n] }, rhs_operand);
}
