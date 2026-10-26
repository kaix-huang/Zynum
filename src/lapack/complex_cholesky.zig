// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const s = @import("complex_support.zig");
const dense = @import("../numerics/dense.zig");
const core = @import("core/complex_cholesky.zig");

/// Overwrite only the selected triangle with lower L (A=L*L^H) or upper U
/// (A=U^H*U). Selected entries must be finite in both components; each diagonal
/// imaginary component must be exactly zero (either sign). The other triangle
/// is ignored and preserved, including NaNs. Success writes positive real
/// diagonal entries with imaginary +0. All argument/finite/diagonal-form errors
/// precede mutation. Numeric info is the one-based step where the updated real
/// pivot is nonpositive or a stored result is nonfinite; completed work and a
/// prefix of the failing column/row may remain. Padding is not written.
pub fn factorize(comptime C: type, a: s.views.Matrix(C), triangle: s.Triangle) s.Error!usize {
    _ = try s.matrixSpan(C, a);
    if (a.row_count != a.column_count) return error.DimensionMismatch;
    try s.hermitianTriangle(C, a, triangle);
    const operand: dense.Matrix(C) = .{
        .values = a.values,
        .row_count = @intCast(a.row_count),
        .column_count = @intCast(a.column_count),
        .leading_dimension = @intCast(a.leading_dimension),
    };
    return core.factorize(C, operand, switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    });
}

/// Solve A*X=B using successful L or U factors; factors must be a normal view.
/// Forward substitution uses L, and backward substitution uses its conjugate
/// transpose (upper input is interpreted as L=U^H). Pairwise factor/RHS span
/// overlap, dimensions/capacity, all used finite entries, and exactly-real
/// diagonal form are checked before mutation. With nonempty RHS, a nonpositive
/// real diagonal returns its one-based index without changing RHS. Arithmetic
/// range failure returns a one-based row and can leave the current RHS partially
/// solved; later RHS columns, factors, opposite triangle and padding are intact.
/// Zero RHS columns/empty systems are validated no-ops.
pub fn solve(comptime C: type, factor: s.views.ConstMatrix(C), rhs: s.views.Matrix(C), triangle: s.Triangle) s.Error!usize {
    const aspan = try s.squareFactor(C, factor);
    const bspan = try s.matrixSpan(C, rhs);
    if (rhs.row_count != factor.row_count) return error.DimensionMismatch;
    try s.noOverlap(C, factor.values[0..aspan], C, rhs.values[0..bspan]);
    try s.hermitianTriangle(C, factor, triangle);
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
    return core.solve(C, factor_operand, rhs_operand, switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    });
}
