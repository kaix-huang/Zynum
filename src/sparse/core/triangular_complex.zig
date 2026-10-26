// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Checked complex substitution over validated borrowed CSR/RHS/workspace.
//! No admission, allocation, sorting, cache, dispatch or external interface.
const math = @import("../../numerics/complex.zig");
const scalar = math.scalar;
fn used(row: usize, column: usize, lower: bool) bool {
    return if (lower) column < row else column > row;
}
fn failure(comptime R: type, row: usize, rhs: usize) R {
    return .{ .info = row + 1, .failure = .non_finite_result, .rhs_column = rhs + 1, .completed_rhs = rhs };
}
fn product(comptime T: type, a: T, b: T) ?T {
    const x = math.widen(T, a);
    const y = math.widen(T, b);
    return math.narrow(T, .{ .re = x.re * y.re - x.im * y.im, .im = x.re * y.im + x.im * y.re });
}
fn subtract(comptime T: type, a: T, b: T) ?T {
    const x = math.widen(T, a);
    const y = math.widen(T, b);
    return math.narrow(T, .{ .re = x.re - y.re, .im = x.im - y.im });
}
fn project(comptime T: type, z: math.WideComplex(T), comptime checked: bool) ?T {
    return if (checked) math.narrow(T, z) else T{ .re = @floatCast(z.re), .im = @floatCast(z.im) };
}
fn productWithContract(comptime T: type, a: T, b: T, comptime checked: bool) ?T {
    if (checked) return product(T, a, b);
    if (!math.finite(T, a) or !math.finite(T, b)) return scalar.mul(T, a, b);
    const x = math.widen(T, a);
    const y = math.widen(T, b);
    return project(T, .{ .re = x.re * y.re - x.im * y.im, .im = x.re * y.im + x.im * y.re }, false);
}
fn subtractWithContract(comptime T: type, a: T, b: T, comptime checked: bool) ?T {
    if (checked) return subtract(T, a, b);
    const x = math.widen(T, a);
    const y = math.widen(T, b);
    return project(T, .{ .re = x.re - y.re, .im = x.im - y.im }, false);
}
fn divideWithContract(comptime T: type, a: T, b: T, comptime checked: bool) ?T {
    return if (checked) math.narrow(T, math.divide(T, math.widen(T, a), b)) else math.divideIeee(T, a, b);
}
pub fn execute(comptime T: type, comptime R: type, a: anytype, rhs: anytype, lower: bool, unit: bool, workspace: []T, comptime checked: bool) R {
    const normal = a.operation == .normal;
    const conjugate = a.operation == .adjoint;
    const forward = if (normal) lower else !lower;
    const n = a.row_count;
    for (0..rhs.column_count) |j| {
        const x = rhs.values[j * rhs.leading_dimension ..][0..n];
        for (0..n) |step| {
            const row = if (forward) step else n - 1 - step;
            const diagonal = if (unit) scalar.one(T) else if (conjugate) scalar.conj(T, workspace[row]) else workspace[row];
            const begin: usize = @intCast(a.row_offsets[row]);
            const end: usize = @intCast(a.row_offsets[row + 1]);
            if (normal) {
                var residual = x[row];
                for (begin..end) |entry| {
                    const col: usize = @intCast(a.column_indices[entry]);
                    if (!used(row, col, lower)) continue;
                    const term = productWithContract(T, a.values[entry], x[col], checked) orelse return failure(R, row, j);
                    residual = subtractWithContract(T, residual, term, checked) orelse return failure(R, row, j);
                }
                x[row] = (if (!checked and unit) residual else divideWithContract(T, residual, diagonal, checked)) orelse return failure(R, row, j);
            } else {
                const solution = (if (!checked and unit) x[row] else divideWithContract(T, x[row], diagonal, checked)) orelse return failure(R, row, j);
                x[row] = solution;
                for (begin..end) |entry| {
                    const col: usize = @intCast(a.column_indices[entry]);
                    if (!used(row, col, lower)) continue;
                    const coefficient = if (conjugate) scalar.conj(T, a.values[entry]) else a.values[entry];
                    const term = productWithContract(T, coefficient, solution, checked) orelse return failure(R, col, j);
                    x[col] = subtractWithContract(T, x[col], term, checked) orelse return failure(R, col, j);
                }
            }
        }
    }
    return .{ .completed_rhs = rhs.column_count };
}
