// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const triangular = @import("core/matrix_matrix/triangular.zig");
const right_parallel = @import("core/matrix_matrix/triangular_right_parallel.zig");
const scalar = @import("core/shared/scalar.zig");
const core_pool = @import("core/execution/thread_pool.zig");
const runtime = @import("runtime.zig");

const ComplexF32 = scalar.ComplexF32;
const ComplexF64 = scalar.ComplexF64;

const RightOperation = enum {
    multiply,
    solve,
};

fn matIndex(leading_dimension: usize, row: usize, column: usize) usize {
    return row + column * leading_dimension;
}

fn complexValue(real: f64, imaginary: f64) ComplexF64 {
    return .{ .re = real, .im = imaginary };
}

fn triangularOpValue(uplo: triangular.Uplo, trans_: triangular.Order, diag: triangular.Diag, a: []const ComplexF64, lda: usize, row: usize, column: usize) ComplexF64 {
    const source_row = if (trans_ == .no_trans) row else column;
    const source_column = if (trans_ == .no_trans) column else row;
    if (source_row == source_column and diag == .unit) return scalar.one(ComplexF64);
    if ((uplo == .upper and source_row > source_column) or (uplo == .lower and source_row < source_column)) return scalar.zero(ComplexF64);
    const value = a[matIndex(lda, source_row, source_column)];
    return if (trans_ == .conj_trans) scalar.conj(ComplexF64, value) else value;
}

fn fillTriangularMatrix(a: []ComplexF64, lda: usize, order: usize) void {
    @memset(a, scalar.zero(ComplexF64));
    for (0..order) |column| {
        for (0..order) |row| {
            const raw_real = @as(f64, @floatFromInt((row * 7 + column * 11 + 3) % 19)) - 9.0;
            const raw_imaginary = @as(f64, @floatFromInt((row * 13 + column * 5 + 1) % 17)) - 8.0;
            a[matIndex(lda, row, column)] = if (row == column)
                complexValue(2.0 + raw_real / 64.0, raw_imaginary / 96.0)
            else
                complexValue(raw_real / 32.0, raw_imaginary / 48.0);
        }
    }
}

fn fillDenseMatrix(b: []ComplexF64, ldb: usize, column_count: usize) void {
    for (0..column_count) |column| {
        for (0..ldb) |row| {
            const raw_real = @as(f64, @floatFromInt((row * 5 + column * 3 + 2) % 23)) - 11.0;
            const raw_imaginary = @as(f64, @floatFromInt((row * 3 + column * 7 + 4) % 21)) - 10.0;
            b[matIndex(ldb, row, column)] = complexValue(raw_real / 16.0, raw_imaginary / 24.0);
        }
    }
}

fn expectComplexApprox(expected: ComplexF64, actual: ComplexF64) !void {
    try std.testing.expectApproxEqAbs(expected.re, actual.re, 2e-11);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, 2e-11);
}

fn referenceLeftTrmm(uplo: triangular.Uplo, trans_: triangular.Order, diag: triangular.Diag, m: usize, n: usize, alpha: ComplexF64, a: []const ComplexF64, lda: usize, b: []const ComplexF64, ldb: usize, output: []ComplexF64) void {
    for (0..n) |column| {
        for (0..m) |row| {
            var sum = scalar.zero(ComplexF64);
            for (0..m) |inner| {
                sum = scalar.add(ComplexF64, sum, scalar.mul(ComplexF64, triangularOpValue(uplo, trans_, diag, a, lda, row, inner), b[matIndex(ldb, inner, column)]));
            }
            output[matIndex(ldb, row, column)] = scalar.mul(ComplexF64, alpha, sum);
        }
    }
}

fn expectLeftTrsmResidual(uplo: triangular.Uplo, trans_: triangular.Order, diag: triangular.Diag, m: usize, n: usize, alpha: ComplexF64, a: []const ComplexF64, lda: usize, original_b: []const ComplexF64, result: []const ComplexF64, ldb: usize) !void {
    for (0..n) |column| {
        for (0..m) |row| {
            var recovered = scalar.zero(ComplexF64);
            for (0..m) |inner| {
                recovered = scalar.add(ComplexF64, recovered, scalar.mul(ComplexF64, triangularOpValue(uplo, trans_, diag, a, lda, row, inner), result[matIndex(ldb, inner, column)]));
            }
            try expectComplexApprox(scalar.mul(ComplexF64, alpha, original_b[matIndex(ldb, row, column)]), recovered);
        }
    }
}

fn rightTestValue(comptime T: type, real: f64, imaginary: f64) T {
    if (T == f32 or T == f64) return @floatCast(real);
    return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
}

fn nanValue(comptime T: type) T {
    if (T == f32 or T == f64) return std.math.nan(T);
    const R = scalar.Real(T);
    return .{ .re = std.math.nan(R), .im = std.math.nan(R) };
}

fn expectValueApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 5e-4) else @as(f64, 2e-11);
    if (T == f32 or T == f64) {
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    } else {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
    }
}

fn fillRightTestOperands(comptime T: type, a: []T, lda: usize, order: usize, b: []T, ldb: usize, column_count: usize) void {
    @memset(a, scalar.zero(T));
    for (0..order) |column| {
        for (0..order) |row| {
            const raw_real = @as(f64, @floatFromInt((row * 7 + column * 11 + 3) % 19)) - 9.0;
            const raw_imaginary = @as(f64, @floatFromInt((row * 13 + column * 5 + 1) % 17)) - 8.0;
            a[matIndex(lda, row, column)] = if (row == column)
                rightTestValue(T, 2.0 + raw_real / 64.0, raw_imaginary / 96.0)
            else
                rightTestValue(T, raw_real / 2048.0, raw_imaginary / 3072.0);
        }
    }
    for (0..column_count) |column| {
        for (0..ldb) |row| {
            const raw_real = @as(f64, @floatFromInt((row * 5 + column * 3 + 2) % 23)) - 11.0;
            const raw_imaginary = @as(f64, @floatFromInt((row * 3 + column * 7 + 4) % 21)) - 10.0;
            b[matIndex(ldb, row, column)] = rightTestValue(T, raw_real / 16.0, raw_imaginary / 24.0);
        }
    }
}

fn rightConjugateTransposeValue(comptime T: type, uplo: triangular.Uplo, diag: triangular.Diag, a: []const T, lda: usize, row: usize, column: usize) T {
    const source_row = column;
    const source_column = row;
    if (source_row == source_column and diag == .unit) return scalar.one(T);
    if ((uplo == .upper and source_row > source_column) or (uplo == .lower and source_row < source_column)) return scalar.zero(T);
    return scalar.conj(T, a[matIndex(lda, source_row, source_column)]);
}

fn expectRightConjugateTrsmResidual(comptime T: type, uplo: triangular.Uplo, diag: triangular.Diag, m: usize, n: usize, alpha: T, a: []const T, lda: usize, original_b: []const T, result: []const T, ldb: usize) !void {
    const tolerance = if (T == scalar.ComplexF32) @as(f32, 2e-3) else @as(f64, 2e-10);
    for (0..n) |column| {
        for (0..m) |row| {
            var recovered = scalar.zero(T);
            for (0..n) |inner| {
                recovered = scalar.add(T, recovered, scalar.mul(T, result[matIndex(ldb, row, inner)], rightConjugateTransposeValue(T, uplo, diag, a, lda, inner, column)));
            }
            const expected = scalar.mul(T, alpha, original_b[matIndex(ldb, row, column)]);
            try std.testing.expectApproxEqAbs(expected.re, recovered.re, tolerance);
            try std.testing.expectApproxEqAbs(expected.im, recovered.im, tolerance);
        }
    }
}

fn rightGateDiagonal(comptime T: type, column: usize) T {
    const imaginary = @as(f64, @floatFromInt(column % 7)) / 64.0 - 0.046875;
    return rightTestValue(T, 1.75 + @as(f64, @floatFromInt(column % 5)) / 32.0, imaginary);
}

fn fillRightGateDiagonal(comptime T: type, a: []T, lda: usize, n: usize, diag: triangular.Diag) void {
    @memset(a, scalar.zero(T));
    for (0..n) |column| {
        a[matIndex(lda, column, column)] = if (diag == .unit) nanValue(T) else rightGateDiagonal(T, column);
    }
}

fn fillRightGateDense(comptime T: type, b: []T, ldb: usize, n: usize) void {
    for (0..n) |column| {
        for (0..ldb) |row| {
            const real = @as(f64, @floatFromInt((row * 11 + column * 5 + 3) % 43)) / 29.0 - 0.75;
            const imaginary = @as(f64, @floatFromInt((row * 7 + column * 13 + 1) % 37)) / 31.0 - 0.5625;
            b[matIndex(ldb, row, column)] = rightTestValue(T, real, imaginary);
        }
    }
}

fn rightGateOpDiagonal(comptime T: type, trans_: triangular.Order, diag: triangular.Diag, column: usize) T {
    if (diag == .unit) return scalar.one(T);
    const value = rightGateDiagonal(T, column);
    return if (trans_ == .conj_trans) scalar.conj(T, value) else value;
}

fn expectRightGateResult(comptime T: type, operation: RightOperation, trans_: triangular.Order, diag: triangular.Diag, m: usize, n: usize, alpha: T, initial_b: []const T, actual: []const T, ldb: usize) !void {
    for (0..n) |column| {
        const diagonal = rightGateOpDiagonal(T, trans_, diag, column);
        for (0..ldb) |row| {
            const initial = initial_b[matIndex(ldb, row, column)];
            const expected = if (row >= m)
                initial
            else switch (operation) {
                .multiply => scalar.mul(T, alpha, scalar.mul(T, initial, diagonal)),
                .solve => scalar.divv(T, scalar.mul(T, alpha, initial), diagonal),
            };
            try expectValueApprox(T, expected, actual[matIndex(ldb, row, column)]);
        }
    }
}

fn expectPublicRightGateAllModes(comptime T: type) !void {
    const m = 512;
    const n = 128;
    const lda = n + 1;
    const ldb = m + 2;
    const alpha = rightTestValue(T, -0.75, 0.25);
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const initial_b = try allocator.alloc(T, ldb * n);
    defer allocator.free(initial_b);
    const actual = try allocator.alloc(T, ldb * n);
    defer allocator.free(actual);
    fillRightGateDense(T, initial_b, ldb, n);

    inline for (.{ RightOperation.multiply, RightOperation.solve }) |operation| {
        inline for (.{ triangular.Uplo.upper, triangular.Uplo.lower }) |uplo| {
            inline for (.{ triangular.Order.no_trans, triangular.Order.trans, triangular.Order.conj_trans }) |trans_| {
                inline for (.{ triangular.Diag.non_unit, triangular.Diag.unit }) |diag| {
                    fillRightGateDiagonal(T, a, lda, n, diag);
                    @memcpy(actual, initial_b);
                    if (operation == .multiply) {
                        triangular.trmm(T, .right, uplo, trans_, diag, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
                    } else {
                        triangular.trsm(T, .right, uplo, trans_, diag, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
                    }
                    try expectRightGateResult(T, operation, trans_, diag, m, n, alpha, initial_b, actual, ldb);
                }
            }
        }
    }
}

test {
    std.testing.refAllDecls(triangular);
    std.testing.refAllDecls(right_parallel);
}

test "current source left complex TRMM and TRSM satisfy every transpose and diagonal mode" {
    const m = 9;
    const n = 7;
    const lda = m + 2;
    const ldb = m + 3;
    const alpha = complexValue(-0.625, 0.1875);
    var a: [lda * m]ComplexF64 = undefined;
    var initial_b: [ldb * n]ComplexF64 = undefined;
    var expected: [ldb * n]ComplexF64 = undefined;
    var actual: [ldb * n]ComplexF64 = undefined;
    fillTriangularMatrix(&a, lda, m);
    fillDenseMatrix(&initial_b, ldb, n);

    inline for (.{ triangular.Uplo.upper, triangular.Uplo.lower }) |uplo| {
        inline for (.{ triangular.Order.no_trans, triangular.Order.trans, triangular.Order.conj_trans }) |trans_| {
            inline for (.{ triangular.Diag.non_unit, triangular.Diag.unit }) |diag| {
                @memcpy(expected[0..], initial_b[0..]);
                @memcpy(actual[0..], initial_b[0..]);
                referenceLeftTrmm(uplo, trans_, diag, m, n, alpha, &a, lda, &initial_b, ldb, &expected);
                triangular.trmm(ComplexF64, .left, uplo, trans_, diag, m, n, alpha, &a, lda, &actual, ldb);
                for (0..n) |column| {
                    for (0..m) |row| {
                        try expectComplexApprox(expected[matIndex(ldb, row, column)], actual[matIndex(ldb, row, column)]);
                    }
                }

                @memcpy(actual[0..], initial_b[0..]);
                triangular.trsm(ComplexF64, .left, uplo, trans_, diag, m, n, alpha, &a, lda, &actual, ldb);
                try expectLeftTrsmResidual(uplo, trans_, diag, m, n, alpha, &a, lda, &initial_b, &actual, ldb);
            }
        }
    }
}

test "right row-parallel TRSM-C satisfies c32 and c64 residuals across the gate" {
    const m = 512;
    const n = 128;
    const lda = n + 2;
    const ldb = m + 3;
    const allocator = std.testing.allocator;

    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const alpha = rightTestValue(T, -0.75, 0.25);
        const a = try allocator.alloc(T, lda * n);
        defer allocator.free(a);
        const initial_b = try allocator.alloc(T, ldb * n);
        defer allocator.free(initial_b);
        const actual = try allocator.alloc(T, ldb * n);
        defer allocator.free(actual);
        inline for (.{ triangular.Uplo.upper, triangular.Uplo.lower }) |uplo| {
            inline for (.{ triangular.Diag.non_unit, triangular.Diag.unit }) |diag| {
                fillRightTestOperands(T, a, lda, n, initial_b, ldb, n);
                if (diag == .unit) {
                    for (0..n) |column| a[matIndex(lda, column, column)] = nanValue(T);
                }
                @memcpy(actual, initial_b);
                triangular.trsm(T, .right, uplo, .conj_trans, diag, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
                try expectRightConjugateTrsmResidual(T, uplo, diag, m, n, alpha, a, lda, initial_b, actual, ldb);
            }
        }
    }
}

test "right TRMM and TRSM public gate covers every type and triangular mode" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| {
        try expectPublicRightGateAllModes(T);
    }
}

test "right TRSM uses the serial fallback with one thread" {
    const T = ComplexF64;
    const m = 512;
    const n = 128;
    const lda = n + 1;
    const ldb = m + 2;
    const alpha = rightTestValue(T, -0.75, 0.25);
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const initial_b = try allocator.alloc(T, ldb * n);
    defer allocator.free(initial_b);
    const actual = try allocator.alloc(T, ldb * n);
    defer allocator.free(actual);
    fillRightGateDiagonal(T, a, lda, n, .unit);
    fillRightGateDense(T, initial_b, ldb, n);
    @memcpy(actual, initial_b);

    runtime.setMaxThreads(1);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    triangular.trsm(T, .right, .lower, .conj_trans, .unit, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
    try expectRightGateResult(T, .solve, .conj_trans, .unit, m, n, alpha, initial_b, actual, ldb);
}
