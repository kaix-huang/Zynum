// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("core/shared/scalar.zig");
const indexing = @import("core/shared/indexing.zig");
const blocked = @import("core/matrix_matrix/structured_blocked.zig");
const symmetric = @import("core/matrix_matrix/symmetric.zig");
const triangular = @import("core/matrix_matrix/triangular.zig");
const packing = @import("kernels/shared/matrix_matrix/structured_packing.zig");

const matIndex = indexing.matIndex;

fn value(comptime T: type, real: f64, imaginary: f64) T {
    if (comptime scalar.isComplex(T)) return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
    return @floatCast(real);
}

fn sample(comptime T: type, index: usize, salt: usize) T {
    const re = (@as(f64, @floatFromInt((index * 17 + salt * 11) % 29)) - 14) / 16;
    const im = (@as(f64, @floatFromInt((index * 7 + salt * 13) % 31)) - 15) / 24;
    return value(T, re, im);
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    const Real = scalar.Real(T);
    const absolute: Real = if (T == f32 or T == scalar.ComplexF32) 5e-3 else 5e-11;
    const relative: Real = if (T == f32 or T == scalar.ComplexF32) 5e-3 else 5e-11;
    if (comptime scalar.isComplex(T)) {
        const scale = @max(@abs(expected.re), @abs(expected.im));
        const tolerance = absolute + relative * scale;
        try std.testing.expect(@abs(expected.re - actual.re) <= tolerance);
        try std.testing.expect(@abs(expected.im - actual.im) <= tolerance);
    } else {
        try std.testing.expect(@abs(expected - actual) <= absolute + relative * @abs(expected));
    }
}

fn expectSlicesClose(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| try expectClose(T, want, got);
}

fn runSymmCase(comptime T: type, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
    const m: usize = 9;
    const n: usize = 7;
    const order = if (side == .left) m else n;
    const lda: scalar.BlasInt = @intCast(order + 2);
    const ldb: scalar.BlasInt = @intCast(m + 3);
    const ldc: scalar.BlasInt = @intCast(m + 4);
    var a: [(m + 2) * m]T = undefined;
    var b: [(m + 3) * n]T = undefined;
    var expected: [(m + 4) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 1);
    for (&b, 0..) |*entry, index| entry.* = sample(T, index, 2);
    for (&expected, 0..) |*entry, index| entry.* = sample(T, index, 3);
    var actual = expected;
    const alpha = value(T, 0.75, if (hermitian) 0 else -0.1875);
    const beta = value(T, -0.25, if (hermitian) 0 else 0.125);

    symmetric.symm(T, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
    try std.testing.expect(blocked.trySymm(T, .{ .block_size = 4 }, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    expectSlicesClose(T, &expected, &actual) catch |err| {
        std.debug.print("SYMM mismatch type={s} side={s} uplo={s} hermitian={}\n", .{ @typeName(T), @tagName(side), @tagName(uplo), hermitian });
        return err;
    };
}

test "blocked SYMM and HEMM pack only active structured panels" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ scalar.Side.left, scalar.Side.right }) |side| {
            inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| try runSymmCase(T, side, uplo, false);
        }
    }
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ scalar.Side.left, scalar.Side.right }) |side| {
            inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| try runSymmCase(T, side, uplo, true);
        }
    }
}

fn runRankCase(comptime T: type, uplo: scalar.Uplo, trans: scalar.Order, hermitian: bool, rank2: bool) !void {
    const n: usize = 9;
    const k: usize = 7;
    const rows = if (trans == .no_trans) n else k;
    const lda: scalar.BlasInt = @intCast(rows + 2);
    const ldb: scalar.BlasInt = @intCast(rows + 3);
    const ldc: scalar.BlasInt = @intCast(n + 4);
    var a: [12 * 9]T = undefined;
    var b: [12 * 9]T = undefined;
    var expected: [(n + 4) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 4);
    for (&b, 0..) |*entry, index| entry.* = sample(T, index, 5);
    for (&expected, 0..) |*entry, index| entry.* = sample(T, index, 6);
    var actual = expected;
    const alpha = value(T, 0.625, if (hermitian) 0 else -0.25);
    const beta = value(T, -0.375, 0);

    if (rank2) {
        symmetric.syr2k(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
        try std.testing.expect(blocked.trySyr2k(T, .{ .block_size = 4 }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    } else {
        symmetric.syrk(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &expected, ldc, hermitian);
        try std.testing.expect(blocked.trySyrk(T, .{ .block_size = 4 }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &actual, ldc, hermitian));
    }
    try expectSlicesClose(T, &expected, &actual);
}

test "blocked rank-k and rank-2k preserve the unstored triangle and Hermitian diagonal" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            inline for (.{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                try runRankCase(T, uplo, trans, false, false);
                try runRankCase(T, uplo, trans, false, true);
            }
        }
    }
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            inline for (.{ scalar.Order.no_trans, scalar.Order.conj_trans }) |trans| {
                try runRankCase(T, uplo, trans, true, false);
                try runRankCase(T, uplo, trans, true, true);
            }
        }
    }
}

fn runTriangularCase(comptime T: type, solve: bool, side: scalar.Side, uplo: scalar.Uplo, trans: scalar.Order, diag: scalar.Diag) !void {
    const m: usize = 9;
    const n: usize = 7;
    const order = if (side == .left) m else n;
    const lda: scalar.BlasInt = @intCast(order + 2);
    const ldb: scalar.BlasInt = @intCast(m + 3);
    var a: [(m + 2) * m]T = undefined;
    var initial: [(m + 3) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 7);
    for (0..order) |j| {
        const index = matIndex(lda, j, j);
        a[index] = value(T, 2.5 + @as(f64, @floatFromInt(j)) / 16, 0.0625);
    }
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 8);
    var expected = initial;
    var actual = initial;
    const alpha = value(T, -0.625, 0.1875);

    if (solve) {
        triangular.trsm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrsm(T, .{ .block_size = 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    } else {
        triangular.trmm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrmm(T, .{ .block_size = 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    }
    try expectSlicesClose(T, &expected, &actual);
}

fn testTriangularType(comptime T: type) !void {
    inline for (.{ scalar.Side.left, scalar.Side.right }) |side| {
        inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            inline for (.{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                inline for (.{ scalar.Diag.non_unit, scalar.Diag.unit }) |diag| {
                    try runTriangularCase(T, false, side, uplo, trans, diag);
                    try runTriangularCase(T, true, side, uplo, trans, diag);
                }
            }
            if (comptime scalar.isComplex(T)) {
                inline for (.{ scalar.Diag.non_unit, scalar.Diag.unit }) |diag| {
                    try runTriangularCase(T, false, side, uplo, .conj_trans, diag);
                    try runTriangularCase(T, true, side, uplo, .conj_trans, diag);
                }
            }
        }
    }
}

test "blocked TRMM and TRSM preserve every side triangle transpose and diagonal mode" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| try testTriangularType(T);
}

test "blocked workspace rejection occurs before caller output changes" {
    var a: [9 * 9]f64 = undefined;
    var b: [9 * 7]f64 = undefined;
    var c: [9 * 7]f64 = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(f64, index, 9);
    for (&b, 0..) |*entry, index| entry.* = sample(f64, index, 10);
    for (&c, 0..) |*entry, index| entry.* = sample(f64, index, 11);
    const original_b = b;
    const original_c = c;
    const denied = blocked.Options{ .block_size = 4, .workspace_available = false };
    try std.testing.expect(!blocked.trySymm(f64, denied, .left, .upper, 9, 7, 1, &a, 9, &b, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    try std.testing.expect(!blocked.trySyrk(f64, denied, .upper, .no_trans, 9, 7, 1, &a, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    try std.testing.expect(!blocked.tryTrmm(f64, denied, .left, .upper, .no_trans, .non_unit, 9, 7, 1, &a, 9, &b, 9));
    try std.testing.expectEqualSlices(f64, &original_b, &b);
    try std.testing.expect(!blocked.tryTrsm(f64, denied, .right, .lower, .trans, .non_unit, 9, 7, 1, &a, 9, &b, 9));
    try std.testing.expectEqualSlices(f64, &original_b, &b);
}

test "packing effective triangle follows transpose" {
    try std.testing.expectEqual(packing.Triangle.upper, packing.effectiveTriangle(.upper, .no_trans));
    try std.testing.expectEqual(packing.Triangle.lower, packing.effectiveTriangle(.upper, .trans));
    try std.testing.expectEqual(packing.Triangle.upper, packing.effectiveTriangle(.lower, .conj_trans));
}
