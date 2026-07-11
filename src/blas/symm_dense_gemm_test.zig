// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const symmetric = @import("core/matrix_matrix/symmetric.zig");
const dense_gemm = @import("core/matrix_matrix/symmetric_dense_gemm.zig");
const scalar = @import("core/shared/scalar.zig");

const ComplexF32 = scalar.ComplexF32;
const ComplexF64 = scalar.ComplexF64;

fn matIndex(leading_dimension: usize, row: usize, column: usize) usize {
    return row + column * leading_dimension;
}

fn testValue(comptime T: type, real: f64, imaginary: f64) T {
    if (T == f32 or T == f64) return @floatCast(real);
    return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
}

fn nanValue(comptime T: type) T {
    if (T == f32 or T == f64) return std.math.nan(T);
    const R = scalar.Real(T);
    return .{ .re = std.math.nan(R), .im = std.math.nan(R) };
}

fn alphaValue(comptime T: type) T {
    return testValue(T, 0.75, -0.125);
}

fn betaValue(comptime T: type) T {
    return testValue(T, 0.25, 0.0625);
}

fn storedAValue(comptime T: type, row: usize, column: usize, herm: bool) T {
    const real_raw = @as(f64, @floatFromInt((row * 17 + column * 11 + 3) % 37)) - 18.0;
    const imaginary_raw = @as(f64, @floatFromInt((row * 7 + column * 19 + 5) % 31)) - 15.0;
    var value = testValue(T, real_raw / 64.0, imaginary_raw / 80.0);
    if (comptime scalar.isComplex(T)) {
        if (herm and row == column) value.im = std.math.nan(scalar.Real(T));
    }
    return value;
}

fn fillStructuredA(comptime T: type, a: []T, lda: usize, order: usize, uplo: symmetric.Uplo, herm: bool) void {
    @memset(a, nanValue(T));
    for (0..order) |column| {
        const row_begin = if (uplo == .upper) 0 else column;
        const row_end = if (uplo == .upper) column + 1 else order;
        for (row_begin..row_end) |row| {
            a[matIndex(lda, row, column)] = storedAValue(T, row, column, herm);
        }
    }
}

fn structuredValue(comptime T: type, a: []const T, lda: usize, uplo: symmetric.Uplo, row: usize, column: usize, herm: bool) T {
    const stored_direct = (uplo == .upper and row <= column) or (uplo == .lower and row >= column);
    var value = if (stored_direct) a[matIndex(lda, row, column)] else a[matIndex(lda, column, row)];
    if (herm and !stored_direct) value = scalar.conj(T, value);
    if (comptime scalar.isComplex(T)) {
        if (herm and row == column) value.im = 0;
    }
    return value;
}

fn fillB(comptime T: type, b: []T, ldb: usize, n: usize) void {
    for (0..n) |column| {
        for (0..ldb) |row| {
            const real_raw = @as(f64, @floatFromInt((row * 13 + column * 5 + 1) % 41)) - 20.0;
            const imaginary_raw = @as(f64, @floatFromInt((row * 3 + column * 17 + 9) % 43)) - 21.0;
            b[matIndex(ldb, row, column)] = testValue(T, real_raw / 48.0, imaginary_raw / 56.0);
        }
    }
}

fn fillC(comptime T: type, c: []T, ldc: usize, n: usize) void {
    for (0..n) |column| {
        for (0..ldc) |row| {
            const real_raw = @as(f64, @floatFromInt((row * 23 + column * 7 + 4) % 47)) - 23.0;
            const imaginary_raw = @as(f64, @floatFromInt((row * 11 + column * 13 + 2) % 53)) - 26.0;
            c[matIndex(ldc, row, column)] = testValue(T, real_raw / 40.0, imaginary_raw / 72.0);
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 2e-3) else @as(f64, 5e-10);
    if (T == f32 or T == f64) {
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    } else {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
    }
}

fn referenceElement(comptime T: type, side: symmetric.Side, uplo: symmetric.Uplo, herm: bool, m: usize, n: usize, a: []const T, lda: usize, b: []const T, ldb: usize, initial_c: []const T, ldc: usize, row: usize, column: usize) T {
    var sum = scalar.zero(T);
    if (side == .left) {
        for (0..m) |inner| {
            sum = scalar.add(T, sum, scalar.mul(T, structuredValue(T, a, lda, uplo, row, inner, herm), b[matIndex(ldb, inner, column)]));
        }
    } else {
        for (0..n) |inner| {
            sum = scalar.add(T, sum, scalar.mul(T, b[matIndex(ldb, row, inner)], structuredValue(T, a, lda, uplo, inner, column, herm)));
        }
    }
    return scalar.add(T, scalar.mul(T, alphaValue(T), sum), scalar.mul(T, betaValue(T), initial_c[matIndex(ldc, row, column)]));
}

fn expectDenseGemmCase(comptime T: type, m: usize, n: usize, side: symmetric.Side, uplo: symmetric.Uplo, herm: bool) !void {
    const order = if (side == .left) m else n;
    const lda = order + 3;
    const ldb = m + 2;
    const ldc = m + 4;
    const allocator = std.testing.allocator;

    const a = try allocator.alloc(T, lda * order);
    defer allocator.free(a);
    const b = try allocator.alloc(T, ldb * n);
    defer allocator.free(b);
    const initial_c = try allocator.alloc(T, ldc * n);
    defer allocator.free(initial_c);
    const actual = try allocator.alloc(T, ldc * n);
    defer allocator.free(actual);

    fillStructuredA(T, a, lda, order, uplo, herm);
    fillB(T, b, ldb, n);
    fillC(T, initial_c, ldc, n);
    @memcpy(actual, initial_c);

    try std.testing.expect(dense_gemm.trySymm(T, side, uplo, @intCast(m), @intCast(n), alphaValue(T), a.ptr, @intCast(lda), b.ptr, @intCast(ldb), betaValue(T), actual.ptr, @intCast(ldc), herm));

    for (0..n) |column| {
        for (0..m) |row| {
            const expected = referenceElement(T, side, uplo, herm, m, n, a, lda, b, ldb, initial_c, ldc, row, column);
            try expectApprox(T, expected, actual[matIndex(ldc, row, column)]);
        }
        for (m..ldc) |row| {
            try std.testing.expectEqual(initial_c[matIndex(ldc, row, column)], actual[matIndex(ldc, row, column)]);
        }
    }
}

fn expectDenseGemmFamily(comptime T: type, herm: bool) !void {
    inline for (.{ symmetric.Side.left, symmetric.Side.right }) |side| {
        inline for (.{ symmetric.Uplo.upper, symmetric.Uplo.lower }) |uplo| {
            try expectDenseGemmCase(T, 128, 128, side, uplo, herm);
        }
    }
}

fn expectAlphaZeroCase(comptime T: type, beta: T) !void {
    const m: usize = 3;
    const n: usize = 2;
    const ldc: usize = m + 2;
    var a: [m * m]T = undefined;
    var b: [m * n]T = undefined;
    var initial_c: [ldc * n]T = undefined;
    var actual: [ldc * n]T = undefined;
    @memset(&a, nanValue(T));
    @memset(&b, nanValue(T));
    fillC(T, &initial_c, ldc, n);
    @memcpy(actual[0..], initial_c[0..]);

    try std.testing.expect(dense_gemm.trySymm(T, .left, .upper, @intCast(m), @intCast(n), scalar.zero(T), &a, @intCast(m), &b, @intCast(m), beta, &actual, @intCast(ldc), false));
    for (0..n) |column| {
        for (0..m) |row| {
            const expected = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, initial_c[matIndex(ldc, row, column)]);
            try expectApprox(T, expected, actual[matIndex(ldc, row, column)]);
        }
        for (m..ldc) |row| {
            try std.testing.expectEqual(initial_c[matIndex(ldc, row, column)], actual[matIndex(ldc, row, column)]);
        }
    }
}

fn expectDenseBetaZeroCase(comptime T: type, herm: bool) !void {
    const m: usize = 128;
    const n: usize = 128;
    const lda: usize = n + 2;
    const ldb: usize = m + 2;
    const ldc: usize = m + 3;
    const allocator = std.testing.allocator;

    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const b = try allocator.alloc(T, ldb * n);
    defer allocator.free(b);
    const initial_c = try allocator.alloc(T, ldc * n);
    defer allocator.free(initial_c);
    const actual = try allocator.alloc(T, ldc * n);
    defer allocator.free(actual);

    @memset(a, nanValue(T));
    for (0..n) |column| {
        for (column..n) |row| a[matIndex(lda, row, column)] = scalar.zero(T);
        a[matIndex(lda, column, column)] = scalar.one(T);
        if (comptime scalar.isComplex(T)) {
            if (herm) a[matIndex(lda, column, column)].im = std.math.nan(scalar.Real(T));
        }
    }
    fillB(T, b, ldb, n);
    fillC(T, initial_c, ldc, n);
    for (0..n) |column| {
        for (0..m) |row| initial_c[matIndex(ldc, row, column)] = nanValue(T);
    }
    @memcpy(actual, initial_c);

    try std.testing.expect(dense_gemm.trySymm(T, .right, .lower, @intCast(m), @intCast(n), alphaValue(T), a.ptr, @intCast(lda), b.ptr, @intCast(ldb), scalar.zero(T), actual.ptr, @intCast(ldc), herm));
    for (0..n) |column| {
        for (0..m) |row| {
            try expectApprox(T, scalar.mul(T, alphaValue(T), b[matIndex(ldb, row, column)]), actual[matIndex(ldc, row, column)]);
        }
        for (m..ldc) |row| {
            try std.testing.expectEqual(initial_c[matIndex(ldc, row, column)], actual[matIndex(ldc, row, column)]);
        }
    }
}

test "dense GEMM SYMM preserves s d c z side uplo and leading dimensions" {
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try expectDenseGemmFamily(f32, false);
    try expectDenseGemmFamily(f64, false);
    try expectDenseGemmFamily(ComplexF32, false);
    try expectDenseGemmFamily(ComplexF64, false);
}

test "dense GEMM HEMM ignores unstored values and imaginary diagonals" {
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try expectDenseGemmFamily(ComplexF32, true);
    try expectDenseGemmFamily(ComplexF64, true);
}

test "dense GEMM SYMM and HEMM cover tall and wide order-512 operands" {
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try expectDenseGemmCase(f32, 512, 128, .left, .lower, false);
    try expectDenseGemmCase(ComplexF64, 128, 512, .right, .upper, true);
}

test "SYMM alpha zero scales C without reading A or B" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| {
        try expectAlphaZeroCase(T, scalar.zero(T));
        try expectAlphaZeroCase(T, betaValue(T));
    }
}

test "dense GEMM beta zero does not read C" {
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try expectDenseBetaZeroCase(T, false);
    try expectDenseBetaZeroCase(ComplexF32, true);
    try expectDenseBetaZeroCase(ComplexF64, true);
}

test {
    std.testing.refAllDecls(symmetric);
}
