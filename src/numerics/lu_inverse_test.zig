// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dense finite polynomial oracle, explicit permutations and storage guards.
const std = @import("std");
const dense = @import("dense.zig");
const inverse = @import("lu_inverse.zig");
const scalar = @import("complex.zig").scalar;
const Z = struct { re: f128, im: f128 };
const Matrix = [8][8]Z;
const zero: Z = .{ .re = 0, .im = 0 };
const one: Z = .{ .re = 1, .im = 0 };
fn number(comptime T: type, re: f128, im: f128) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn wide(comptime T: type, value: T) Z {
    return if (comptime scalar.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn multiply(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn divide(a: Z, b: Z) Z {
    const den = b.re * b.re + b.im * b.im;
    return .{ .re = (a.re * b.re + a.im * b.im) / den, .im = (a.im * b.re - a.re * b.im) / den };
}
fn product(a: Matrix, b: Matrix, n: usize) Matrix {
    var result: Matrix = @splat(@splat(zero));
    for (0..n) |i| for (0..n) |j| for (0..n) |k| {
        result[i][j] = add(result[i][j], multiply(a[i][k], b[k][j]));
    };
    return result;
}
fn identity(n: usize) Matrix {
    var result: Matrix = @splat(@splat(zero));
    for (0..n) |i| result[i][i] = one;
    return result;
}
fn polynomialInverse(input: Matrix, n: usize) Matrix {
    // D(I+N), where N is strictly triangular and N^n=0. This dense polynomial
    // is independent of the candidate's forward/back substitution recurrences.
    var negative_n: Matrix = @splat(@splat(zero));
    for (0..n) |i| for (0..n) |j| {
        if (i == j) continue;
        const value = divide(input[i][j], input[i][i]);
        negative_n[i][j] = .{ .re = -value.re, .im = -value.im };
    };
    var power = identity(n);
    var result = power;
    for (1..@max(@as(usize, 1), n)) |_| {
        power = product(power, negative_n, n);
        for (0..n) |i| for (0..n) |j| {
            result[i][j] = add(result[i][j], power[i][j]);
        };
    }
    for (0..n) |i| for (0..n) |j| {
        result[i][j] = divide(result[i][j], input[j][j]);
    };
    return result;
}
fn near(comptime T: type, expected: Z, actual: Z, residual: bool) !void {
    const tolerance: f128 = if (scalar.Real(T) == f32) (if (residual) 8e-5 else 2e-5) else (if (residual) 2e-12 else 4e-13);
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance * @max(@as(f128, 1), @abs(expected.re)));
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance * @max(@as(f128, 1), @abs(expected.im)));
}
const Reader = struct {
    values: []const usize,
    used: usize,
    calls: *usize,
    order: *[8]usize,
    pub fn readUnchecked(self: @This(), index: usize) usize {
        if (index >= self.used) @panic("inverse read the ignored pivot tail");
        self.order[self.calls.*] = index;
        self.calls.* += 1;
        return self.values[index];
    }
};
const Unread = struct {
    pub fn readUnchecked(_: @This(), _: usize) usize {
        @panic("inverse read a pivot before admission or for n<=1");
    }
};

test "LU inverse matches independent nilpotent products and explicit permutations in 216 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const sentinel = number(T, 123, -456);
        const work_sentinel: inverse.WorkValue(T) = if (comptime scalar.isComplex(T)) .{ .re = 42, .im = -71 } else 42;
        for ([_]usize{ 0, 1, 2, 3, 5, 8 }) |n| {
            for ([_]usize{ 0, 1, 3 }) |padding| {
                for (0..3) |seed| {
                    const ld = @max(@as(usize, 1), n + padding);
                    var a: [104]T = @splat(sentinel);
                    const values = a[3..];
                    var l = identity(n);
                    var u: Matrix = @splat(@splat(zero));
                    for (0..n) |i| for (0..n) |j| {
                        const x: f128 = @floatFromInt(i + 2 * j + seed + 1);
                        const y: f128 = @floatFromInt(i + seed + 1);
                        const z: f128 = @floatFromInt(j);
                        values[i + j * ld] = if (i == j) number(T, 1.5 + x / 8, y / 16) else number(T, (if ((i + j + seed) % 2 == 0) @as(f128, 1) else -1) * x / 64, (y - z) / 128);
                        if (i > j) l[i][j] = wide(T, values[i + j * ld]) else u[i][j] = wide(T, values[i + j * ld]);
                    };
                    var pivots: [8]usize = @splat(std.math.maxInt(usize));
                    var p = identity(n);
                    for (0..n -| 1) |i| {
                        pivots[i] = if (seed == 0) i else if (seed == 1) n - 1 else i + 1;
                        std.mem.swap([8]Z, &p[i], &p[pivots[i]]);
                    }
                    var pt: Matrix = @splat(@splat(zero));
                    for (0..n) |i| for (0..n) |j| {
                        pt[i][j] = p[j][i];
                    };
                    const original = product(pt, product(l, u, n), n);
                    const expected = product(product(polynomialInverse(u, n), polynomialInverse(l, n), n), p, n);
                    const before = a;
                    var workspace: [66]inverse.WorkValue(T) = @splat(work_sentinel);
                    const work_before = workspace;
                    const required = try inverse.workspaceLength(n);
                    var calls: usize = 0;
                    var order: [8]usize = undefined;
                    const reader: Reader = .{ .values = &pivots, .used = n -| 1, .calls = &calls, .order = &order };
                    try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = ld }, reader, workspace[1..][0..required]));
                    try std.testing.expectEqual(n -| 1, calls);
                    for (0..calls) |i| try std.testing.expectEqual(n - 2 - i, order[i]);
                    var result: Matrix = @splat(@splat(zero));
                    for (0..n) |i| for (0..n) |j| {
                        result[i][j] = wide(T, values[i + j * ld]);
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
                        if (row < n and column < n) continue;
                        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[index]), std.mem.asBytes(&a[index]));
                    }
                    for (0..workspace.len) |index| {
                        if (index > 0 and index <= required) continue;
                        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&work_before[index]), std.mem.asBytes(&workspace[index]));
                    }
                    count += 1;
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 216), count);
}

test "LU inverse rejects short and overflowing workspace before stores and pivot reads" {
    try std.testing.expectEqual(@as(usize, 0), try inverse.workspaceLength(0));
    try std.testing.expectEqual(@as(usize, 0), try inverse.workspaceLength(1));
    try std.testing.expectEqual(@as(usize, 4), try inverse.workspaceLength(2));
    const huge = @as(usize, std.math.maxInt(i32)) + 1;
    try std.testing.expectEqual(huge * huge, try inverse.workspaceLength(huge));
    try std.testing.expectError(error.SizeOverflow, inverse.workspaceLength(std.math.maxInt(usize)));
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var a: [9]T = @splat(number(T, 2, 0));
        const before = a;
        var work: [8]inverse.WorkValue(T) = @splat(if (comptime scalar.isComplex(T)) .{ .re = 99, .im = -55 } else 99);
        const work_before = work;
        try std.testing.expectError(error.WorkspaceTooSmall, inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, Unread{}, &work));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&work_before), std.mem.asBytes(&work));
        var empty: [0]T = .{};
        try std.testing.expectError(error.SizeOverflow, inverse.invertNonsingular(T, .{ .values = &empty, .row_count = std.math.maxInt(usize), .column_count = std.math.maxInt(usize), .leading_dimension = std.math.maxInt(usize) }, Unread{}, &.{}));
    }
}

test "LU inverse first exact zero precedes all mutation and pivot or workspace access" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var a: [16]T = @splat(number(T, 2, 0));
        a[5] = number(T, -0.0, 0);
        a[10] = number(T, 0, -0.0);
        const before = a;
        var work: [9]inverse.WorkValue(T) = undefined;
        @memset(std.mem.asBytes(&work), 0xa5);
        const work_before = work;
        try std.testing.expectEqual(@as(usize, 2), try inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 4 }, Unread{}, &work));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&work_before), std.mem.asBytes(&work));
        try std.testing.expectEqual(@as(usize, 2), try inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 4 }, Unread{}, &.{}));
    }
}

test "LU inverse empty and singleton use no workspace or pivot and retain ordinary IEEE classification" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var empty: [0]T = .{};
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &empty, .row_count = 0, .column_count = 0, .leading_dimension = std.math.maxInt(usize) }, Unread{}, &.{}));
        var a = [_]T{number(T, 2, 0)};
        const matrix: dense.Matrix(T) = .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, matrix, Unread{}, &.{}));
        try near(T, .{ .re = 0.5, .im = 0 }, wide(T, a[0]), false);
        a[0] = number(T, std.math.inf(f128), 0);
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, matrix, Unread{}, &.{}));
        try near(T, zero, wide(T, a[0]), false);
        a[0] = number(T, std.math.nan(f128), 0);
        try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, matrix, Unread{}, &.{}));
        const value = wide(T, a[0]);
        try std.testing.expect(std.math.isNan(value.re));
        if (comptime scalar.isComplex(T)) try std.testing.expect(std.math.isNan(value.im));
    }
}
