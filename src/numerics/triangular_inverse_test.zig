// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Independent finite nilpotent-series oracle, two-sided products and boundaries.
const std = @import("std");
const dense = @import("dense.zig");
const inverse = @import("triangular_inverse.zig");
const scalar = @import("complex.zig").scalar;
const Z = struct { re: f128, im: f128 };
const Matrix = [8][8]Z;
const zero: Z = .{ .re = 0, .im = 0 };
const one: Z = .{ .re = 1, .im = 0 };
fn number(comptime T: type, re: f128, im: f128) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn reference(comptime T: type, value: T) Z {
    return if (comptime scalar.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn multiply(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn divide(a: Z, b: Z) Z {
    const denominator = b.re * b.re + b.im * b.im;
    return .{ .re = (a.re * b.re + a.im * b.im) / denominator, .im = (a.im * b.re - a.re * b.im) / denominator };
}
fn selected(row: usize, column: usize, triangle: inverse.Triangle) bool {
    return if (triangle == .upper) row <= column else row >= column;
}
fn poison(comptime T: type) T {
    return number(T, std.math.nan(f128), -std.math.nan(f128));
}
fn coefficient(comptime T: type, values: []const T, ld: usize, row: usize, column: usize, triangle: inverse.Triangle, diagonal: inverse.Diagonal) Z {
    if (!selected(row, column, triangle)) return zero;
    if (row == column and diagonal == .unit) return one;
    return reference(T, values[row + column * ld]);
}
fn product(a: Matrix, b: Matrix, n: usize) Matrix {
    var result: Matrix = @splat(@splat(zero));
    for (0..n) |i| for (0..n) |j| for (0..n) |k| {
        result[i][j] = add(result[i][j], multiply(a[i][k], b[k][j]));
    };
    return result;
}
fn seriesInverse(comptime T: type, values: []const T, ld: usize, n: usize, triangle: inverse.Triangle, diagonal: inverse.Diagonal) Matrix {
    // A = D(I+N), with strictly triangular nilpotent N. Dense f128 powers
    // evaluate (I-N+N^2-...)D^-1 without using the candidate recurrence.
    var negative_n: Matrix = @splat(@splat(zero));
    var d: [8]Z = @splat(zero);
    for (0..n) |i| {
        d[i] = divide(one, coefficient(T, values, ld, i, i, triangle, diagonal));
        for (0..n) |j| {
            if (i == j or !selected(i, j, triangle)) continue;
            const v = multiply(d[i], reference(T, values[i + j * ld]));
            negative_n[i][j] = .{ .re = -v.re, .im = -v.im };
        }
    }
    var power: Matrix = @splat(@splat(zero));
    for (0..n) |i| power[i][i] = one;
    var sum = power;
    for (1..@max(@as(usize, 1), n)) |_| {
        power = product(power, negative_n, n);
        for (0..n) |i| for (0..n) |j| {
            sum[i][j] = add(sum[i][j], power[i][j]);
        };
    }
    for (0..n) |i| for (0..n) |j| {
        sum[i][j] = multiply(sum[i][j], d[j]);
    };
    return sum;
}
fn near(comptime T: type, expected: Z, actual: Z, residual: bool) !void {
    const tolerance: f128 = if (scalar.Real(T) == f32) (if (residual) 8e-5 else 2e-5) else (if (residual) 2e-12 else 3e-13);
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance * @max(@as(f128, 1), @abs(expected.re)));
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance * @max(@as(f128, 1), @abs(expected.im)));
}

test "triangular inverse matches independent f128 finite series and both identity products in 864 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var workspace: [8]inverse.WorkValue(T) = undefined;
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]inverse.Diagonal{ .unit, .non_unit }) |diagonal| {
                for ([_]usize{ 0, 1, 2, 3, 5, 8 }) |n| {
                    for ([_]usize{ 0, 1, 3 }) |padding| {
                        for (0..3) |seed| {
                            const ld = @max(@as(usize, 1), n + padding);
                            var a: [104]T = @splat(poison(T));
                            const values = a[3..];
                            for (0..n) |j| for (0..n) |i| {
                                if (!selected(i, j, triangle) or (i == j and diagonal == .unit)) continue;
                                const x: f128 = @floatFromInt(i + 2 * j + seed + 1);
                                const y: f128 = @floatFromInt(i + seed);
                                const z: f128 = @floatFromInt(j);
                                values[i + j * ld] = if (i == j) number(T, 1.5 + 0.25 * x, 0.125 * (y + 1)) else number(T, (if ((i + j + seed) % 2 == 0) @as(f128, 1) else -1) * x / 32, (y - z) / 64);
                            };
                            const before = a;
                            const expected = seriesInverse(T, before[3..], ld, n, triangle, diagonal);
                            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = ld }, triangle, diagonal, &workspace));
                            var original: Matrix = @splat(@splat(zero));
                            var result: Matrix = @splat(@splat(zero));
                            for (0..n) |i| for (0..n) |j| {
                                original[i][j] = coefficient(T, before[3..], ld, i, j, triangle, diagonal);
                                result[i][j] = coefficient(T, values, ld, i, j, triangle, diagonal);
                                try near(T, expected[i][j], result[i][j], false);
                            };
                            const left = product(original, result, n);
                            const right = product(result, original, n);
                            for (0..n) |i| for (0..n) |j| {
                                const target = if (i == j) one else zero;
                                try near(T, target, left[i][j], true);
                                try near(T, target, right[i][j], true);
                            };
                            for (0..a.len) |index| {
                                const row = if (index < 3) n else (index - 3) % ld;
                                const column = if (index < 3) n else (index - 3) / ld;
                                if (row < n and column < n and selected(row, column, triangle) and !(row == column and diagonal == .unit)) continue;
                                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[index]), std.mem.asBytes(&a[index]));
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 864), count);
}

test "triangular inverse rejects every first exact zero before any store and ignores earlier NaNs in the preflight" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]usize{ 1, 2, 5, 8 }) |n| {
                for (0..n) |position| {
                    var a: [80]T = @splat(poison(T));
                    for (0..n) |i| a[i + i * 9] = number(T, 2, 0);
                    if (position != 0) a[0] = poison(T);
                    a[position + position * 9] = number(T, -0.0, 0.0);
                    if (position + 1 < n) a[position + 1 + (position + 1) * 9] = number(T, 0.0, -0.0);
                    const before = a;
                    try std.testing.expectEqual(position + 1, try inverse.invert(T, .{ .values = &a, .row_count = n, .column_count = n, .leading_dimension = 9 }, triangle, .non_unit, &.{}));
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
                }
            }
        }
    }
}

test "triangular inverse unit empty singleton and clipped strict-triangle views do not access unused storage" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var empty: [0]T = .{};
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]usize{ 0, 1 }) |n| {
                try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &empty, .row_count = n, .column_count = n, .leading_dimension = std.math.maxInt(usize) }, triangle, .unit, &.{}));
            }
            var a: [5]T = @splat(poison(T));
            const offset: usize = if (triangle == .upper) 3 else 1;
            a[offset] = number(T, 3, -2);
            const before = a;
            const extent: usize = if (triangle == .upper) 4 else 2;
            var workspace: [2]inverse.WorkValue(T) = undefined;
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = a[0..extent], .row_count = 2, .column_count = 2, .leading_dimension = 3 }, triangle, .unit, &workspace));
            try near(T, reference(T, number(T, -3, 2)), reference(T, a[offset]), false);
            for (0..a.len) |i| if (i != offset) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[i]), std.mem.asBytes(&a[i]));
        }
    }
}

test "triangular inverse finite reciprocal covers signed phases large tiny and subnormal output" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const R = scalar.Real(T);
        const large: R = if (comptime R == f32) 0x1p127 else 0x1p1023;
        const small: R = if (comptime R == f32) 0x1p-126 else 0x1p-1022;
        const cases = [_]T{ number(T, 2, 1), number(T, -4, -2), number(T, large, large), number(T, small, -small) };
        for (cases) |value| for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            var a = [_]T{value};
            const expected = divide(one, reference(T, value));
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .non_unit, &.{}));
            const actual = reference(T, a[0]);
            const tolerance: f128 = if (R == f32) 2e-6 else 3e-15;
            try std.testing.expect(std.math.isFinite(actual.re) and std.math.isFinite(actual.im));
            try std.testing.expect(@abs(actual.re - expected.re) <= tolerance * @abs(expected.re));
            try std.testing.expect(@abs(actual.im - expected.im) <= tolerance * @abs(expected.im));
        };
    }
}

test "triangular inverse ordinary NaN Inf and purely imaginary nonzero diagonals retain own IEEE classification" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            var a = [_]T{poison(T)};
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .non_unit, &.{}));
            const nan_result = reference(T, a[0]);
            try std.testing.expect(std.math.isNan(nan_result.re));
            a[0] = number(T, std.math.inf(f128), 0);
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .non_unit, &.{}));
            try std.testing.expectEqual(@as(f128, 0), reference(T, a[0]).re);
            if (comptime scalar.isComplex(T)) {
                a[0] = number(T, 0, 2);
                try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, triangle, .non_unit, &.{}));
                try near(T, .{ .re = 0, .im = -0.5 }, reference(T, a[0]), false);
            }
        }
    }
}

test "triangular inverse short and exact borrowed workspaces preserve matrices and workspace guards in 48 configurations" {
    const wide = @as(usize, std.math.maxInt(i32)) + 11;
    try std.testing.expectEqual(wide, inverse.workspaceLength(wide));
    try std.testing.expectEqual(@as(usize, 0), inverse.workspaceLength(0));
    try std.testing.expectEqual(@as(usize, 0), inverse.workspaceLength(1));
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]inverse.Diagonal{ .unit, .non_unit }) |diagonal| {
                for ([_]usize{ 2, 3, 8 }) |n| {
                    var a: [80]T = @splat(poison(T));
                    for (0..n) |i| for (0..n) |j| {
                        if (!selected(i, j, triangle) or (i == j and diagonal == .unit)) continue;
                        a[i + j * 9] = if (i == j) number(T, 2, 0) else number(T, 0.25, -0.125);
                    };
                    const before = a;
                    var work = std.mem.zeroes([12]inverse.WorkValue(T));
                    const work_before = work;
                    const matrix: dense.Matrix(T) = .{ .values = &a, .row_count = n, .column_count = n, .leading_dimension = 9 };
                    try std.testing.expectError(error.WorkspaceTooSmall, inverse.invert(T, matrix, triangle, diagonal, work[2 .. n + 1]));
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&work_before), std.mem.asBytes(&work));
                    try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, matrix, triangle, diagonal, work[2 .. n + 2]));
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(work_before[0..2]), std.mem.sliceAsBytes(work[0..2]));
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(work_before[n + 2 ..]), std.mem.sliceAsBytes(work[n + 2 ..]));
                    count += 1;
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 48), count);
}
