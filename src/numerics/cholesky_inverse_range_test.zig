// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const inverse = @import("cholesky_inverse.zig");
const scalar = @import("complex.zig").scalar;
fn number(comptime T: type, re: scalar.Real(T), im: scalar.Real(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn setUpper(comptime T: type, a: []T, row: usize, column: usize, value: T, triangle: inverse.Triangle) void {
    a[if (triangle == .upper) row + column * 5 else column + row * 5] = if (triangle == .upper) value else scalar.conj(T, value);
}
fn setup(comptime T: type, a: []T, triangle: inverse.Triangle) void {
    for (0..4) |j| for (0..4) |i| {
        if (if (triangle == .upper) i <= j else i >= j) a[i + j * 5] = number(T, if (i == j) 1 else 0, 0);
    };
}
fn preserve(comptime T: type, before: []const T, after: []const T, triangle: inverse.Triangle) !void {
    for (after, 0..) |*value, index| {
        const i = index % 5;
        const j = index / 5;
        if (i < 4 and j < 4 and (if (triangle == .upper) i <= j else i >= j)) {
            if (comptime scalar.isComplex(T)) {
                try std.testing.expect(!std.math.isNan(value.re) and !std.math.isNan(value.im));
            } else try std.testing.expect(!std.math.isNan(value.*));
        } else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[index]), std.mem.asBytes(value));
    }
}

test "Cholesky inverse retains Hermitian couplings from inverse-factor entries below storage range in sixteen cases" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const R = scalar.Real(T);
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| for (0..2) |subnormal| {
            const s: R = if (comptime R == f32) (if (subnormal == 0) 0x1p-126 else 0x1p-127) else (if (subnormal == 0) 0x1p-1022 else 0x1p-1023);
            const h: R = if (comptime R == f32) (if (subnormal == 0) 0x1p126 else 0x1p127) else (if (subnormal == 0) 0x1p1022 else 0x1p1023);
            var a: [20]T = @splat(number(T, -777, 333));
            setup(T, &a, triangle);
            setUpper(T, &a, 0, 1, number(T, s, if (scalar.isComplex(T)) s else 0), triangle);
            setUpper(T, &a, 1, 3, number(T, s, if (scalar.isComplex(T)) -s else 0), triangle);
            setUpper(T, &a, 2, 3, number(T, if (scalar.isComplex(T)) h / 2 else h, if (scalar.isComplex(T)) h / 2 else 0), triangle);
            const before = a;
            var work: [16]inverse.WorkValue(T) = undefined;
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 4, .column_count = 4, .leading_dimension = 5 }, triangle, &work));
            const index: usize = if (triangle == .upper) 10 else 2;
            try std.testing.expectEqual(number(T, -s, if (triangle == .upper) s else -s), a[index]);
            const underflowed = a[if (triangle == .upper) 15 else 3];
            try std.testing.expectEqual(number(T, 0, 0), underflowed);
            try preserve(T, &before, &a, triangle);
            count += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 16), count);
}

test "Cholesky inverse cancels individually overflowing Gram products before narrowing in eight cases" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const R = scalar.Real(T);
        const h: R = if (comptime R == f32) 0x1p65 else 0x1p513;
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            var a: [20]T = @splat(number(T, -444, 222));
            setup(T, &a, triangle);
            const positive = number(T, h, if (scalar.isComplex(T)) h else 0);
            const negative = scalar.neg(T, positive);
            setUpper(T, &a, 0, 2, negative, triangle);
            setUpper(T, &a, 0, 3, negative, triangle);
            setUpper(T, &a, 1, 2, negative, triangle);
            setUpper(T, &a, 1, 3, positive, triangle);
            const before = a;
            var work: [16]inverse.WorkValue(T) = undefined;
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 4, .column_count = 4, .leading_dimension = 5 }, triangle, &work));
            try std.testing.expectEqual(number(T, 0, 0), a[if (triangle == .upper) 5 else 1]);
            const diagonal = a[0];
            try std.testing.expect(std.math.isInf(if (comptime scalar.isComplex(T)) diagonal.re else diagonal));
            if (comptime scalar.isComplex(T)) try std.testing.expectEqual(@as(R, 0), diagonal.im);
            try preserve(T, &before, &a, triangle);
            count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 8), count);
}
