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
    return runSymmCaseSized(T, 9, 7, side, uplo, hermitian);
}

fn runSymmCaseSized(comptime T: type, comptime m: usize, comptime n: usize, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
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

    if (m >= 256) {
        for (0..n) |j| {
            for (0..m) |i| {
                var sum = scalar.zero(T);
                for (0..order) |p| {
                    const row = if (side == .left) i else p;
                    const col = if (side == .left) p else j;
                    const stored = if (uplo == .upper) row <= col else row >= col;
                    var av = a[if (stored) matIndex(lda, row, col) else matIndex(lda, col, row)];
                    if (hermitian) {
                        if (!stored) av = scalar.conj(T, av);
                        if (comptime scalar.isComplex(T)) {
                            if (row == col) av.im = 0;
                        }
                    }
                    const bv = b[if (side == .left) matIndex(ldb, p, j) else matIndex(ldb, i, p)];
                    sum = scalar.add(T, sum, scalar.mul(T, av, bv));
                }
                const index = matIndex(ldc, i, j);
                expected[index] = scalar.add(T, scalar.mul(T, alpha, sum), scalar.mul(T, beta, expected[index]));
            }
        }
    } else {
        symmetric.symm(T, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
    }
    try std.testing.expect(blocked.trySymm(T, .{ .block_size = 4 }, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    expectSlicesClose(T, &expected, &actual) catch |err| {
        std.debug.print("SYMM mismatch type={s} side={s} uplo={s} hermitian={}\n", .{ @typeName(T), @tagName(side), @tagName(uplo), hermitian });
        return err;
    };
}

test "full structured materialization matches scalar rectangular reference" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                try runSymmCaseSized(T, 263, 257, side, uplo, false);
                if (comptime scalar.isComplex(T)) try runSymmCaseSized(T, 263, 257, side, uplo, true);
            }
        }
    }
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
    return runRankCaseSized(T, 9, 7, 4, uplo, trans, hermitian, rank2);
}

fn runRankCaseSized(comptime T: type, comptime n: usize, comptime k: usize, comptime block_size: usize, uplo: scalar.Uplo, trans: scalar.Order, hermitian: bool, rank2: bool) !void {
    const rows = if (trans == .no_trans) n else k;
    const lda: scalar.BlasInt = @intCast(rows + 2);
    const ldb: scalar.BlasInt = @intCast(rows + 3);
    const ldc: scalar.BlasInt = @intCast(n + 4);
    var a: [(@as(usize, @max(n, k)) + 3) * @as(usize, @max(n, k))]T = undefined;
    var b: [(@as(usize, @max(n, k)) + 3) * @as(usize, @max(n, k))]T = undefined;
    var expected: [(n + 4) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 4);
    for (&b, 0..) |*entry, index| entry.* = sample(T, index, 5);
    for (&expected, 0..) |*entry, index| entry.* = sample(T, index, 6);
    var actual = expected;
    const alpha = value(T, 0.625, if (hermitian and !rank2) 0 else -0.25);
    const beta = value(T, -0.375, 0);

    if (n >= 256) {
        const Reference = struct {
            fn op(matrix: [*]const T, ld: scalar.BlasInt, tr: scalar.Order, i: usize, p: usize) T {
                const v = matrix[if (tr == .no_trans) matIndex(ld, i, p) else matIndex(ld, p, i)];
                return if (tr == .conj_trans) scalar.conj(T, v) else v;
            }
        };
        for (0..n) |j| {
            for (0..n) |i| {
                if (if (uplo == .upper) i > j else i < j) continue;
                var first = scalar.zero(T);
                var second = scalar.zero(T);
                for (0..k) |p| {
                    const ai = Reference.op(&a, lda, trans, i, p);
                    const aj = Reference.op(&a, lda, trans, j, p);
                    const bi = Reference.op(&b, ldb, trans, i, p);
                    const bj = Reference.op(&b, ldb, trans, j, p);
                    const right = if (rank2) bj else aj;
                    first = scalar.add(T, first, scalar.mul(T, ai, if (hermitian) scalar.conj(T, right) else right));
                    if (rank2) second = scalar.add(T, second, scalar.mul(T, bi, if (hermitian) scalar.conj(T, aj) else aj));
                }
                const index = matIndex(ldc, i, j);
                expected[index] = scalar.add(T, scalar.mul(T, alpha, first), scalar.mul(T, beta, expected[index]));
                if (rank2) expected[index] = scalar.add(T, expected[index], scalar.mul(T, if (hermitian) scalar.conj(T, alpha) else alpha, second));
                if (hermitian and i == j) {
                    if (comptime scalar.isComplex(T)) expected[index].im = 0;
                }
            }
        }
    } else if (rank2) {
        symmetric.syr2k(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
    } else {
        symmetric.syrk(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &expected, ldc, hermitian);
    }
    if (rank2) {
        try std.testing.expect(blocked.trySyr2k(T, .{ .block_size = block_size }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    } else {
        try std.testing.expect(blocked.trySyrk(T, .{ .block_size = block_size }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &actual, ldc, hermitian));
    }
    try expectSlicesClose(T, &expected, &actual);
    for (0..n) |j| {
        for (0..n + 4) |i| {
            if (i >= n or (if (uplo == .upper) i > j else i < j)) try std.testing.expectEqual(expected[i + j * (n + 4)], actual[i + j * (n + 4)]);
        }
    }
}

test "queued rank tiles match independent scalar reference including partial last tiles" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            for ([_]scalar.Order{ .no_trans, .trans }) |trans| {
                try runRankCaseSized(T, 257, 73, 64, uplo, trans, false, false);
                try runRankCaseSized(T, 257, 73, 64, uplo, trans, false, true);
            }
            if (comptime scalar.isComplex(T)) {
                for ([_]scalar.Order{ .no_trans, .conj_trans }) |trans| {
                    try runRankCaseSized(T, 257, 73, 64, uplo, trans, true, false);
                    try runRankCaseSized(T, 257, 73, 64, uplo, trans, true, true);
                }
            }
        }
    }
}

test "rank panels and split diagonal preserve full general-coefficient results and storage fringes" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            inline for (.{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                try runRankCaseSized(T, 97, 73, 96, uplo, trans, false, false);
                try runRankCaseSized(T, 97, 73, 96, uplo, trans, false, true);
            }
            if (comptime scalar.isComplex(T)) {
                inline for (.{ scalar.Order.no_trans, scalar.Order.conj_trans }) |trans| {
                    try runRankCaseSized(T, 97, 73, 96, uplo, trans, true, false);
                    try runRankCaseSized(T, 97, 73, 96, uplo, trans, true, true);
                }
            }
        }
    }
}

test "single-thread rank transpose reuse preserves odd full output and padding" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    inline for (.{ f32, f64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            try runRankCaseSized(T, 129, 67, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 129, 67, 64, uplo, .trans, false, true);
            try runRankCaseSized(T, 257, 73, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 257, 73, 64, uplo, .trans, false, true);
        }
    }
}

test "queued real transpose panels preserve partial final blocks at larger depth" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            try runRankCaseSized(T, 385, 257, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 385, 257, 64, uplo, .trans, false, true);
        }
    }
}

test "tiled real transpose packing preserves odd extents and output guards" {
    inline for (.{ f32, f64 }) |T| {
        const m = 65;
        const k = 1027;
        const lda = k + 3;
        var input: [lda * m]T = undefined;
        for (&input, 0..) |*element, i| element.* = @as(T, @floatFromInt(i)) / 137;
        var output: [m * k + 2]T = @splat(-999);
        @import("kernels/shared/matrix_matrix/packing.zig").packTransposedA(T, m, k, &input, lda, output[1 .. m * k + 1]);
        for (0..k) |p| {
            for (0..m) |i| try std.testing.expectEqual(input[p + i * lda], output[1 + p * m + i]);
        }
        try std.testing.expectEqual(@as(T, -999), output[0]);
        try std.testing.expectEqual(@as(T, -999), output[m * k + 1]);
    }
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
    return runTriangularCaseSized(T, 9, 7, solve, side, uplo, trans, diag);
}

fn runTriangularCaseSized(comptime T: type, comptime m: usize, comptime n: usize, solve: bool, side: scalar.Side, uplo: scalar.Uplo, trans: scalar.Order, diag: scalar.Diag) !void {
    const order = if (side == .left) m else n;
    const lda: scalar.BlasInt = @intCast(order + 2);
    const ldb: scalar.BlasInt = @intCast(m + 3);
    var a: [(m + 2) * m]T = undefined;
    var initial: [(m + 3) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 7);
    if (m >= 256) {
        for (&a) |*entry| entry.* = scalar.mul(T, entry.*, value(T, 1.0 / @as(f64, @floatFromInt(m)), 0));
    }
    for (0..order) |j| {
        const index = matIndex(lda, j, j);
        a[index] = value(T, 2.5 + @as(f64, @floatFromInt(j)) / 16, 0.0625);
    }
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 8);
    var expected = initial;
    var actual = initial;
    const alpha = value(T, -0.625, 0.1875);

    if (solve) {
        if (m < 256) triangular.trsm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrsm(T, .{ .block_size = 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    } else {
        if (m < 256) triangular.trmm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrmm(T, .{ .block_size = 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    }
    if (m < 256) return expectSlicesClose(T, &expected, &actual);
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = scalar.zero(T);
            for (0..order) |p| {
                const row = if (side == .left) i else p;
                const col = if (side == .left) p else j;
                const source_row = if (trans == .no_trans) row else col;
                const source_col = if (trans == .no_trans) col else row;
                if (if (uplo == .upper) source_row > source_col else source_row < source_col) continue;
                var av = if (row == col and diag == .unit) scalar.one(T) else a[matIndex(lda, source_row, source_col)];
                if (trans == .conj_trans) av = scalar.conj(T, av);
                const bindex = if (side == .left) matIndex(ldb, p, j) else matIndex(ldb, i, p);
                sum = scalar.add(T, sum, scalar.mul(T, av, if (solve) actual[bindex] else initial[bindex]));
            }
            const index = matIndex(ldb, i, j);
            if (solve) {
                try expectClose(T, scalar.mul(T, alpha, initial[index]), sum);
            } else {
                try expectClose(T, scalar.mul(T, alpha, sum), actual[index]);
            }
        }
        for (m..m + 3) |i| try std.testing.expectEqual(initial[matIndex(ldb, i, j)], actual[matIndex(ldb, i, j)]);
    }
}

test "queued triangular RHS panels preserve full scalar products and solve residuals" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]scalar.Diag{ .unit, .non_unit }) |diag| {
                        try runTriangularCaseSized(T, 259, 257, false, side, uplo, trans, diag);
                        try runTriangularCaseSized(T, 259, 257, true, side, uplo, trans, diag);
                    }
                }
            }
        }
    }
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
