// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const views = @import("../blas/api/views.zig");
const v = @import("validation.zig");
const dense = @import("../numerics/dense.zig");
const core = @import("core/cholesky.zig");

/// Overwrite the selected triangle with L (lower, A=L*L^T) or U (upper,
/// A=U^T*U). The opposite triangle and all padding remain untouched. Only the
/// selected triangle is read; it must contain finite values. Returns zero on
/// success, otherwise the one-based factorization step with a nonpositive
/// pivot or nonfinite arithmetic. A numerical failure retains completed work
/// and may modify part of the failing column/row; no usable full factor is
/// promised. Invalid arguments and nonfinite input leave the matrix unchanged.
pub fn factorize(comptime T: type, a: views.Matrix(T), triangle: v.Triangle) v.Error!usize {
    _ = try v.matrixSpan(T, a);
    if (a.row_count != a.column_count) return error.DimensionMismatch;
    try v.finiteTriangle(T, a, triangle);
    const operand: dense.Matrix(T) = .{
        .values = a.values,
        .row_count = @intCast(a.row_count),
        .column_count = @intCast(a.column_count),
        .leading_dimension = @intCast(a.leading_dimension),
    };
    return switch (triangle) {
        .lower => core.factorize(T, operand, .lower),
        .upper => core.factorize(T, operand, .upper),
    };
}

/// Solve A*X=B in place in rhs using a successful Cholesky factor. Factor and
/// rhs physical spans must not overlap; factor must have operation=.normal.
/// Finite input, shape, storage, and overlap checks precede mutation in every
/// build. A nonpositive diagonal returns its one-based index before modifying
/// rhs. Arithmetic overflow during substitution returns its one-based row and
/// may leave rhs partially solved. Factor, unused triangle, and padding are
/// untouched. An empty system or zero RHS columns is a validated no-op.
pub fn solve(comptime T: type, factor: views.ConstMatrix(T), rhs: views.Matrix(T), triangle: v.Triangle) v.Error!usize {
    const factor_span = try v.normalFactor(T, factor);
    const rhs_span = try v.matrixSpan(T, rhs);
    if (rhs.row_count != factor.row_count) return error.DimensionMismatch;
    try v.noOverlap(T, factor.values[0..factor_span], T, rhs.values[0..rhs_span]);
    try v.finiteTriangle(T, factor, triangle);
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
    return switch (triangle) {
        .lower => core.solve(T, factor_operand, rhs_operand, .lower),
        .upper => core.solve(T, factor_operand, rhs_operand, .upper),
    };
}
