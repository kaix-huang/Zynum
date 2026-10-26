// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const s = @import("complex_support.zig");

const dense = @import("../numerics/dense.zig");
const core = @import("core/lu.zig");
const pivot_codec = @import("../numerics/pivot.zig");

/// Rectangular partial-pivoted LU: P*A=L*U, implicit unit L diagonal. P is the
/// ordered swap sequence (k,pivots[k]); pivot values are ZERO-BASED. Select the
/// first maximum |re|+|im| in each active column, evaluated in widened precision
/// to avoid overflow of the magnitude sum. Matrix and used min(m,n) pivot spans
/// must be disjoint; finite input and all argument checks precede mutation.
/// Return zero or the first ONE-BASED singular/numerical-failure step. Exact
/// zero pivot columns record the first singular step and continue, completing
/// all pivots and factors if later arithmetic stays finite. Nonfinite stored
/// arithmetic instead stops immediately (retaining any earlier singular info)
/// with partial matrix/pivot state. The unused pivot tail/padding is preserved.
pub fn factorize(comptime C: type, a: s.views.Matrix(C), pivots: []s.views.BlasInt) s.Error!usize {
    const span = try s.matrixSpan(C, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const steps = @min(m, n);
    if (pivots.len < steps) return error.BufferTooSmall;
    try s.noOverlap(C, a.values[0..span], s.views.BlasInt, pivots[0..steps]);
    try s.finiteMatrix(C, a);
    const operand: dense.Matrix(C) = .{
        .values = a.values,
        .row_count = @intCast(a.row_count),
        .column_count = @intCast(a.column_count),
        .leading_dimension = @intCast(a.leading_dimension),
    };
    // Validated nonnegative i32 dimensions already prove zero-based i32 pivots
    // representable; do not insert a new fallible check into the old preflight.
    return core.factorize(C, operand, pivot_codec.Writer(s.views.BlasInt, .zero){ .values = pivots[0..steps] });
}

/// Solve normal A*X=B for square successful LU factors. Complex LU products
/// and triangular substitutions are unconjugated; this API has no transpose
/// mode. Pivots must obey k<=pivots[k]<n and all factor/RHS/used-pivot spans
/// must be pairwise disjoint. Full finite/shape/capacity/pivot/alias preflight
/// precedes mutation. For nonempty RHS, the first exactly-zero complex U
/// diagonal returns its one-based index with RHS unchanged. Range failure
/// during substitution returns its one-based row; the current RHS can contain
/// its completed permutation and partial solution, while later RHS columns,
/// factors, pivots and padding remain untouched. Empty/zero-RHS cases are
/// validated no-ops. A nonzero factorization info is not a valid solve input.
pub fn solve(comptime C: type, factor: s.views.ConstMatrix(C), pivots: []const s.views.BlasInt, rhs: s.views.Matrix(C)) s.Error!usize {
    const aspan = try s.squareFactor(C, factor);
    const bspan = try s.matrixSpan(C, rhs);
    if (rhs.row_count != factor.row_count) return error.DimensionMismatch;
    const n: usize = @intCast(factor.row_count);
    if (pivots.len < n) return error.BufferTooSmall;
    try s.noOverlap(C, factor.values[0..aspan], C, rhs.values[0..bspan]);
    try s.noOverlap(C, factor.values[0..aspan], s.views.BlasInt, pivots[0..n]);
    try s.noOverlap(C, rhs.values[0..bspan], s.views.BlasInt, pivots[0..n]);
    for (0..n) |k| {
        if (pivots[k] < 0) return error.InvalidPivot;
        const pivot: usize = @intCast(pivots[k]);
        if (pivot < k or pivot >= n) return error.InvalidPivot;
    }
    try s.finiteMatrix(C, factor);
    try s.finiteMatrix(C, rhs);
    const factor_operand: dense.ConstMatrix(C) = .{
        .values = factor.values,
        .row_count = @intCast(factor.row_count),
        .column_count = @intCast(factor.column_count),
        .leading_dimension = @intCast(factor.leading_dimension),
    };
    const rhs_operand: dense.Matrix(C) = .{
        .values = rhs.values,
        .row_count = @intCast(rhs.row_count),
        .column_count = @intCast(rhs.column_count),
        .leading_dimension = @intCast(rhs.leading_dimension),
    };
    return core.solve(C, factor_operand, pivot_codec.Reader(s.views.BlasInt, .zero){ .values = pivots[0..n] }, rhs_operand);
}
