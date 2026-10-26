// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared triangular numerical contracts with independently assembled equations.
const std = @import("std");
const triangular = @import("triangular_ieee.zig");
const scalar = @import("complex.zig").scalar;
const Ref = struct { re: f64, im: f64 };
fn number(comptime T: type, re: f64, im: f64) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn reference(comptime T: type, value: T) Ref {
    return if (comptime scalar.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn times(a: Ref, b: Ref) Ref {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn poison(comptime T: type) T {
    return number(T, std.math.nan(f64), std.math.nan(f64));
}
fn expected(comptime T: type, row: usize, col: usize) T {
    return number(T, 0.25 * @as(f64, @floatFromInt(row + 1)) + 0.5 * @as(f64, @floatFromInt(col)), 0.125 * (@as(f64, @floatFromInt(row)) - @as(f64, @floatFromInt(col))));
}
fn coefficient(comptime T: type, a: []const T, ld: usize, row: usize, col: usize, triangle: triangular.Triangle, operation: triangular.Operation, diagonal: triangular.Diagonal) Ref {
    const i = if (operation == .normal) row else col;
    const j = if (operation == .normal) col else row;
    if ((triangle == .lower and i < j) or (triangle == .upper and i > j)) return .{ .re = 0, .im = 0 };
    if (i == j and diagonal == .unit) return .{ .re = 1, .im = 0 };
    var v = reference(T, a[i + j * ld]);
    if (operation == .adjoint) v.im = -v.im;
    return v;
}
fn near(comptime T: type, want: T, got: T) !void {
    const a = reference(T, want);
    const b = reference(T, got);
    const tolerance: f64 = if (scalar.Real(T) == f32) 2e-5 else 2e-13;
    try std.testing.expectApproxEqAbs(a.re, b.re, tolerance);
    try std.testing.expectApproxEqAbs(a.im, b.im, tolerance);
}

test "shared triangular independently assembled N T H equations cover 432 scalar triangle diagonal shape combinations" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]triangular.Triangle{ .lower, .upper }) |triangle| {
            for ([_]triangular.Operation{ .normal, .transposed, .adjoint }) |operation| {
                for ([_]triangular.Diagonal{ .unit, .non_unit }) |diagonal| {
                    for ([_]usize{ 0, 1, 4 }) |n| {
                        for ([_]usize{ 0, 1, 3 }) |nrhs| {
                            var a: [32]T = @splat(poison(T));
                            for (0..n) |j| for (0..n) |i| {
                                if ((triangle == .lower and i < j) or (triangle == .upper and i > j) or (diagonal == .unit and i == j)) continue;
                                a[i + j * 6] = if (i == j) number(T, 2 + 0.25 * @as(f64, @floatFromInt(i)), 0.125) else number(T, 0.0625 * @as(f64, @floatFromInt(i + j + 1)), 0.125 * (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(j))));
                            };
                            const a_before = a;
                            var b: [32]T = @splat(poison(T));
                            for (0..nrhs) |j| for (0..n) |i| {
                                var sum = Ref{ .re = 0, .im = 0 };
                                for (0..n) |k| {
                                    const term = times(coefficient(T, &a, 6, i, k, triangle, operation, diagonal), reference(T, expected(T, k, j)));
                                    sum.re += term.re;
                                    sum.im += term.im;
                                }
                                b[i + j * 7] = number(T, sum.re, sum.im);
                            };
                            const b_before = b;
                            const info = triangular.solveNonsingular(T, .{ .values = &a, .row_count = n, .column_count = n, .leading_dimension = 6 }, .{ .values = &b, .row_count = n, .column_count = nrhs, .leading_dimension = 7 }, triangle, operation, diagonal);
                            try std.testing.expectEqual(@as(usize, 0), info);
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a_before), std.mem.asBytes(&a));
                            for (0..b.len) |index| {
                                const row = index % 7;
                                const col = index / 7;
                                if (row < n and col < nrhs) try near(T, expected(T, row, col), b[index]) else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&b_before[index]), std.mem.asBytes(&b[index]));
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 432), count);
}

test "shared triangular first exact zero rejects before RHS mutation even with zero RHS columns" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var a: [9]T = @splat(poison(T));
        a[0] = number(T, 1, 0);
        a[4] = number(T, -0.0, 0);
        a[8] = number(T, 0, -0.0);
        var b: [9]T = @splat(number(T, 3, -2));
        const before = b;
        for ([_]usize{ 0, 2 }) |nrhs| {
            for ([_]triangular.Triangle{ .lower, .upper }) |triangle| {
                for ([_]triangular.Operation{ .normal, .transposed, .adjoint }) |operation| {
                    try std.testing.expectEqual(@as(usize, 2), triangular.solveNonsingular(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .{ .values = &b, .row_count = 3, .column_count = nrhs, .leading_dimension = 3 }, triangle, operation, .non_unit));
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&b));
                }
            }
        }
        a[0] = number(T, std.math.inf(f64), 0);
        a[4] = poison(T);
        a[8] = number(T, 1, if (scalar.isComplex(T)) 2 else 0);
        try std.testing.expectEqual(@as(usize, 0), triangular.firstZeroDiagonal(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .non_unit));
    }
}

test "shared triangular unit empty native-width calls and normal IEEE zero skips retain nonreads" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const huge = @as(usize, std.math.maxInt(i32)) + 7;
        var empty: [0]T = .{};
        try std.testing.expectEqual(@as(usize, 0), triangular.solveNonsingular(T, .{ .values = &empty, .row_count = huge, .column_count = huge, .leading_dimension = huge }, .{ .values = &empty, .row_count = huge, .column_count = 0, .leading_dimension = huge }, .upper, .adjoint, .unit));
        const a = [_]T{poison(T)};
        for ([_]triangular.Triangle{ .lower, .upper }) |triangle| {
            var b = [_]T{number(T, -0.0, 0.0)};
            const before = b;
            triangular.solve(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .normal, .non_unit);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&b));
            triangular.solve(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .transposed, .non_unit);
            const got = reference(T, b[0]);
            try std.testing.expect(std.math.isNan(got.re) or std.math.isNan(got.im));
        }
    }
}
