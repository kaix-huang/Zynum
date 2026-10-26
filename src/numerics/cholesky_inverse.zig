// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Inverse from a borrowed triangular Cholesky factor. The existing triangular
//! inverse computes a wide factor inverse. Its Hermitian Gram product is
//! narrowed only when publishing the selected output triangle. The opposite
//! triangle, padding and workspace suffix are untouched. No Context, allocator,
//! execution, threads, handle or retained workspace is owned here.
const dense = @import("dense.zig");
const arithmetic = @import("complex.zig");
const triangular = @import("triangular_ieee.zig");
const inverse = @import("triangular_inverse.zig");
const scalar = arithmetic.scalar;
pub const Triangle = triangular.Triangle;
pub const WorkValue = inverse.WorkValue;
pub const Error = inverse.WideError;

pub fn workspaceLength(n: usize) Error!usize {
    return if (n <= 1) 0 else inverse.wideWorkspaceLength(n);
}

fn zero(comptime T: type) WorkValue(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = 0, .im = 0 } else 0;
}

fn addProduct(comptime T: type, sum: *WorkValue(T), a: WorkValue(T), b: WorkValue(T), triangle: Triangle) void {
    if (comptime scalar.isComplex(T)) {
        sum.re += a.re * b.re + a.im * b.im;
        // Upper: inv(U) inv(U)^H. Lower: inv(L)^H inv(L).
        sum.im += if (triangle == .upper) a.im * b.re - a.re * b.im else a.re * b.im - a.im * b.re;
    } else sum.* += a * b;
}

fn publish(comptime T: type, value: WorkValue(T), diagonal: bool) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(value.re), .im = if (diagonal) 0 else @floatCast(value.im) } else @floatCast(value);
}

fn gramEntry(comptime T: type, wide: []const WorkValue(T), n: usize, row: usize, column: usize, triangle: Triangle) WorkValue(T) {
    var sum = zero(T);
    for (@max(row, column)..n) |k| {
        const a = if (triangle == .upper) wide[row + k * n] else wide[k + row * n];
        const b = if (triangle == .upper) wide[column + k * n] else wide[k + column * n];
        addProduct(T, &sum, a, b, triangle);
    }
    return sum;
}

/// Numerical leg after exact-zero diagonal preflight. A short or overflowing
/// workspace fails before any operand or workspace access/store. All factor
/// reads finish before output publication. Empty/singleton use no caller work.
pub fn invertNonsingular(comptime T: type, matrix: dense.Matrix(T), triangle: Triangle, workspace: []WorkValue(T)) Error!void {
    const n = matrix.row_count;
    if (n == 0) return;
    const required = try workspaceLength(n);
    if (workspace.len < required) return error.WorkspaceTooSmall;
    if (n == 1) {
        var one: [1]WorkValue(T) = undefined;
        try inverse.invertToWideNonsingular(T, matrix.asConst(), triangle, .non_unit, &one);
        var result = zero(T);
        addProduct(T, &result, one[0], one[0], triangle);
        matrix.values[0] = publish(T, result, true);
        return;
    }
    const wide = workspace[0..required];
    try inverse.invertToWideNonsingular(T, matrix.asConst(), triangle, .non_unit, wide);
    for (0..n) |column| {
        const first: usize = if (triangle == .upper) 0 else column;
        const end = if (triangle == .upper) column + 1 else n;
        for (first..end) |row| {
            matrix.values[row + column * matrix.leading_dimension] = publish(T, gramEntry(T, wide, n, row, column, triangle), row == column);
        }
    }
}

/// First one-based exact-zero factor diagonal returns before matrix or scratch
/// stores. NaN/Inf are not classified as exact singularity. Frontends own valid
/// square Dense geometry, capacities, addresses, aliases and scalar policies.
pub fn invert(comptime T: type, matrix: dense.Matrix(T), triangle: Triangle, workspace: []WorkValue(T)) Error!usize {
    const info = triangular.firstZeroDiagonal(T, matrix.asConst(), .non_unit);
    if (info != 0) return info;
    try invertNonsingular(T, matrix, triangle, workspace);
    return 0;
}
