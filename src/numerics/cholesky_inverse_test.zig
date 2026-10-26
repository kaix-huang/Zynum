// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const inverse = @import("cholesky_inverse.zig");
const triangular = @import("triangular_inverse.zig");
const dense = @import("dense.zig");
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
fn sub(a: Z, b: Z) Z {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.im * b.re + a.re * b.im };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn div(a: Z, b: Z) Z {
    const d = b.re * b.re + b.im * b.im;
    return .{ .re = (a.re * b.re + a.im * b.im) / d, .im = (a.im * b.re - a.re * b.im) / d };
}
fn selected(row: usize, column: usize, triangle: inverse.Triangle) bool {
    return if (triangle == .upper) row <= column else row >= column;
}
fn near(comptime T: type, expected: Z, actual: Z, identity: bool) !void {
    const tolerance: f128 = if (scalar.Real(T) == f32) (if (identity) 2e-4 else 3e-5) else (if (identity) 8e-12 else 2e-12);
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance * @max(@as(f128, 1), @abs(expected.re)));
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance * @max(@as(f128, 1), @abs(expected.im)));
}
fn factor(comptime T: type, values: []const T, ld: usize, row: usize, column: usize, triangle: inverse.Triangle, diagonal: triangular.Diagonal) Z {
    if (!selected(row, column, triangle)) return zero;
    if (row == column and diagonal == .unit) return one;
    return reference(T, values[row + column * ld]);
}
fn gaussJordan(a: Matrix, n: usize) Matrix {
    // Independent full-matrix pivoted elimination on A, not a triangular
    // recurrence or a product of the candidate's inverse factors.
    var v: [8][16]Z = @splat(@splat(zero));
    for (0..n) |i| for (0..n) |j| {
        v[i][j] = a[i][j];
        v[i][n + j] = if (i == j) one else zero;
    };
    for (0..n) |k| {
        var pivot = k;
        for (k + 1..n) |i| if (@abs(v[i][k].re) + @abs(v[i][k].im) > @abs(v[pivot][k].re) + @abs(v[pivot][k].im)) {
            pivot = i;
        };
        std.mem.swap([16]Z, &v[k], &v[pivot]);
        const d = v[k][k];
        for (0..2 * n) |j| v[k][j] = div(v[k][j], d);
        for (0..n) |i| {
            if (i == k) continue;
            const multiplier = v[i][k];
            for (0..2 * n) |j| v[i][j] = sub(v[i][j], mul(multiplier, v[k][j]));
        }
    }
    var result: Matrix = @splat(@splat(zero));
    for (0..n) |i| for (0..n) |j| {
        result[i][j] = v[i][n + j];
    };
    return result;
}

test "Cholesky inverse matches independent f128 full-matrix elimination and both identities in 432 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]usize{ 0, 1, 2, 3, 5, 8 }) |n| for ([_]usize{ 0, 1, 3 }) |padding| for (0..3) |seed| {
                const ld = @max(@as(usize, 1), n + padding);
                var a: [104]T = @splat(number(T, std.math.nan(f128), -std.math.nan(f128)));
                for (0..n) |j| for (0..n) |i| {
                    if (!selected(i, j, triangle)) continue;
                    const re: f128 = if (i == j) 1.25 + @as(f128, @floatFromInt(i + seed)) / 8 else @as(f128, @floatFromInt(@as(i32, @intCast((i * 5 + j * 3 + seed) % 9)) - 4)) / 32;
                    const im: f128 = if (i == j) 0 else @as(f128, @floatFromInt(@as(i32, @intCast((i * 3 + j * 5 + seed) % 7)) - 3)) / 40;
                    a[i + j * ld] = number(T, re, im);
                };
                const before = a;
                var original: Matrix = @splat(@splat(zero));
                for (0..n) |i| for (0..n) |j| for (0..n) |k| {
                    const left = if (triangle == .upper) conj(factor(T, &before, ld, k, i, triangle, .non_unit)) else factor(T, &before, ld, i, k, triangle, .non_unit);
                    const right = if (triangle == .upper) factor(T, &before, ld, k, j, triangle, .non_unit) else conj(factor(T, &before, ld, j, k, triangle, .non_unit));
                    original[i][j] = add(original[i][j], mul(left, right));
                };
                const expected = gaussJordan(original, n);
                const marker: inverse.WorkValue(T) = if (comptime scalar.isComplex(T)) .{ .re = 777, .im = -123 } else 777;
                var work: [70]inverse.WorkValue(T) = @splat(marker);
                const required = try inverse.workspaceLength(n);
                try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = n, .column_count = n, .leading_dimension = ld }, triangle, work[0 .. required + 2]));
                var actual: Matrix = @splat(@splat(zero));
                for (0..n) |i| for (0..n) |j| {
                    actual[i][j] = if (selected(i, j, triangle)) reference(T, a[i + j * ld]) else conj(reference(T, a[j + i * ld]));
                    try near(T, expected[i][j], actual[i][j], false);
                    if (comptime scalar.isComplex(T)) if (i == j) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&@as(scalar.Real(T), 0)), std.mem.asBytes(&a[i + j * ld].im));
                };
                for (0..n) |i| for (0..n) |j| {
                    var left = zero;
                    var right = zero;
                    for (0..n) |k| {
                        left = add(left, mul(original[i][k], actual[k][j]));
                        right = add(right, mul(actual[i][k], original[k][j]));
                    }
                    const identity = if (i == j) one else zero;
                    try near(T, identity, left, true);
                    try near(T, identity, right, true);
                };
                for (&a, 0..) |*value, i| {
                    if (i / ld < n and i % ld < n and selected(i % ld, i / ld, triangle)) continue;
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[i]), std.mem.asBytes(value));
                }
                for (work[required..]) |*value| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&marker), std.mem.asBytes(value));
                count += 1;
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 432), count);
}

test "wide triangular inverse reuses both scalar diagonal policies without factor mutation in 48 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| for ([_]triangular.Diagonal{ .unit, .non_unit }) |diagonal| for ([_]usize{ 0, 1, 3 }) |padding| {
            const n: usize = 4;
            const ld = n + padding;
            var a: [32]T = @splat(number(T, std.math.nan(f128), std.math.nan(f128)));
            var original: Matrix = @splat(@splat(zero));
            for (0..n) |i| for (0..n) |j| {
                if (selected(i, j, triangle) and !(i == j and diagonal == .unit)) a[i + j * ld] = number(T, if (i == j) 2 else 0.125, if (i == j) 0 else -0.0625);
                original[i][j] = factor(T, &a, ld, i, j, triangle, diagonal);
            };
            const expected = gaussJordan(original, n);
            const before = a;
            const marker: inverse.WorkValue(T) = if (comptime scalar.isComplex(T)) .{ .re = 123, .im = -456 } else 123;
            var work: [18]inverse.WorkValue(T) = @splat(marker);
            try triangular.invertToWideNonsingular(T, .{ .values = &a, .row_count = n, .column_count = n, .leading_dimension = ld }, triangle, diagonal, &work);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
            for (0..n) |i| for (0..n) |j| {
                const value = work[i + j * n];
                const actual: Z = if (comptime scalar.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
                try near(T, expected[i][j], actual, false);
                if (!selected(i, j, triangle)) try std.testing.expectEqual(zero, actual);
            };
            for (work[16..]) |*value| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&marker), std.mem.asBytes(value));
            count += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 48), count);
}

test "Cholesky inverse rejects short overflow and first-zero cases before stores and keeps singleton work untouched" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const marker: inverse.WorkValue(T) = if (comptime scalar.isComplex(T)) .{ .re = 999, .im = -321 } else 999;
        var work: [10]inverse.WorkValue(T) = @splat(marker);
        const before_work = work;
        var a: [9]T = @splat(number(T, 12, -7));
        a[0] = number(T, 1, 0);
        a[4] = number(T, -0.0, 0);
        a[8] = number(T, 0, -0.0);
        var before = a;
        try std.testing.expectEqual(@as(usize, 2), try inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .lower, &.{}));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
        a[4] = number(T, 1, 0);
        a[8] = number(T, 1, 0);
        before = a;
        try std.testing.expectError(error.WorkspaceTooSmall, inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .upper, work[0..8]));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_work), std.mem.asBytes(&work));
        const huge = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, inverse.workspaceLength(huge));
        try std.testing.expectError(error.SizeOverflow, inverse.invertNonsingular(T, .{ .values = &.{}, .row_count = huge, .column_count = huge, .leading_dimension = huge }, .lower, &.{}));
        try std.testing.expectError(error.SizeOverflow, triangular.invertToWideNonsingular(T, .{ .values = &.{}, .row_count = huge, .column_count = huge, .leading_dimension = huge }, .upper, .non_unit, &.{}));
        try std.testing.expectError(error.WorkspaceTooSmall, triangular.invertToWideNonsingular(T, .{ .values = &.{}, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .lower, .unit, work[0..3]));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_work), std.mem.asBytes(&work));
        const unreadable: [*]T = @ptrFromInt(@alignOf(T));
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = unreadable[0..0], .row_count = 0, .column_count = 0, .leading_dimension = huge }, .upper, &work));
        try triangular.invertToWideNonsingular(T, .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = huge }, .upper, .unit, &work);
        var singleton = [_]T{number(T, -2, 0)};
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &singleton, .row_count = 1, .column_count = 1, .leading_dimension = huge }, .lower, &work));
        try std.testing.expectEqual(number(T, 0.25, 0), singleton[0]);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_work), std.mem.asBytes(&work));
        var unit: [1]inverse.WorkValue(T) = undefined;
        try triangular.invertToWideNonsingular(T, .{ .values = &.{}, .row_count = 1, .column_count = 1, .leading_dimension = huge }, .upper, .unit, &unit);
        const expected: inverse.WorkValue(T) = if (comptime scalar.isComplex(T)) .{ .re = 1, .im = 0 } else 1;
        try std.testing.expectEqual(expected, unit[0]);
    }
}

test "Cholesky inverse ordinary nonfinite singletons keep IEEE classes and exact-real complex diagonals" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var a = [_]T{number(T, std.math.inf(f128), 0)};
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .upper, &.{}));
        try std.testing.expectEqual(number(T, 0, 0), a[0]);
        a[0] = number(T, std.math.nan(f128), 0);
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .lower, &.{}));
        const result = reference(T, a[0]);
        try std.testing.expect(std.math.isNan(result.re));
        try std.testing.expectEqual(@as(f128, 0), result.im);
    }
}
