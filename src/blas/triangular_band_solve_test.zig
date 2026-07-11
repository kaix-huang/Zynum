// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const matrix_vector = @import("core/matrix_vector.zig");
const triangular_band_solve = @import("core/matrix_vector/triangular_band_solve.zig");
const scalar = @import("core/shared/scalar.zig");

const ComplexF32 = scalar.ComplexF32;
const ComplexF64 = scalar.ComplexF64;
const Diag = scalar.Diag;
const Order = scalar.Order;
const Uplo = scalar.Uplo;

fn value(comptime T: type, re: f64, im: f64) T {
    if (T == f32 or T == f64) return @floatCast(re);
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn nanValue(comptime T: type) T {
    if (T == f32 or T == f64) return std.math.nan(T);
    const R = scalar.Real(T);
    return .{ .re = std.math.nan(R), .im = std.math.nan(R) };
}

fn matrixValue(comptime T: type, row: usize, column: usize) T {
    const re_seed = @as(f64, @floatFromInt((row * 17 + column * 11 + 5) % 37)) - 18.0;
    const im_seed = @as(f64, @floatFromInt((row * 7 + column * 19 + 3) % 31)) - 15.0;
    if (row == column) return value(T, 1.625 + re_seed / 512.0, im_seed / 768.0);
    return value(T, re_seed / 192.0, im_seed / 224.0);
}

fn solutionValue(comptime T: type, index: usize) T {
    const re_seed = @as(f64, @floatFromInt((index * 13 + 7) % 41)) - 20.0;
    const im_seed = @as(f64, @floatFromInt((index * 5 + 9) % 37)) - 18.0;
    return value(T, re_seed / 23.0, im_seed / 29.0);
}

fn fillBandMatrix(comptime T: type, a: []T, uplo: Uplo, diag: Diag, n: usize, k: usize, lda: usize) void {
    @memset(a, nanValue(T));
    for (0..n) |column| {
        if (uplo == .upper) {
            const first_row = column - @min(column, k);
            var band_row = k - (column - first_row);
            for (first_row..column + 1) |row| {
                if (diag == .non_unit or row != column) a[band_row + column * lda] = matrixValue(T, row, column);
                band_row += 1;
            }
        } else {
            const row_end = @min(n, column + k + 1);
            var band_row: usize = 0;
            for (column..row_end) |row| {
                if (diag == .non_unit or row != column) a[band_row + column * lda] = matrixValue(T, row, column);
                band_row += 1;
            }
        }
    }
}

fn accumulateReferenceEntry(
    comptime T: type,
    trans: Order,
    diag: Diag,
    row: usize,
    column: usize,
    matrix_index: usize,
    a: []const T,
    solution: []const T,
    rhs: []T,
) void {
    const stored = if (diag == .unit and row == column) scalar.one(T) else a[matrix_index];
    if (trans == .no_trans) {
        rhs[row] = scalar.add(T, rhs[row], scalar.mul(T, stored, solution[column]));
    } else {
        const av = if (trans == .conj_trans) scalar.conj(T, stored) else stored;
        rhs[column] = scalar.add(T, rhs[column], scalar.mul(T, av, solution[row]));
    }
}

// Build b = op(A) * solution out of place and by stored columns. This keeps
// the reference traversal independent from the in-place substitution kernels.
fn buildReferenceRhs(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, k: usize, a: []const T, lda: usize, solution: []const T, rhs: []T) void {
    @memset(rhs, scalar.zero(T));
    for (0..n) |column| {
        if (uplo == .upper) {
            const first_row = column - @min(column, k);
            var band_row = k - (column - first_row);
            for (first_row..column + 1) |row| {
                accumulateReferenceEntry(T, trans, diag, row, column, band_row + column * lda, a, solution, rhs);
                band_row += 1;
            }
        } else {
            const row_end = @min(n, column + k + 1);
            var band_row: usize = 0;
            for (column..row_end) |row| {
                accumulateReferenceEntry(T, trans, diag, row, column, band_row + column * lda, a, solution, rhs);
                band_row += 1;
            }
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 2e-3) else @as(f64, 2e-11);
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isFinite(actual));
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    } else {
        try std.testing.expect(std.math.isFinite(actual.re));
        try std.testing.expect(std.math.isFinite(actual.im));
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
    }
}

fn expectNan(comptime T: type, actual: T) !void {
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isNan(actual));
    } else {
        try std.testing.expect(std.math.isNan(actual.re));
        try std.testing.expect(std.math.isNan(actual.im));
    }
}

fn expectFinite(comptime T: type, actual: T) !void {
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isFinite(actual));
    } else {
        try std.testing.expect(std.math.isFinite(actual.re));
        try std.testing.expect(std.math.isFinite(actual.im));
    }
}

fn expectBandSentinels(comptime T: type, a: []const T, uplo: Uplo, diag: Diag, n: usize, k: usize, lda: usize) !void {
    for (0..n) |column| {
        for (0..lda) |band_row| {
            const is_logical = if (uplo == .upper)
                band_row >= k - @min(column, k) and band_row <= k
            else
                band_row <= @min(k, n - column - 1);
            const is_diagonal = band_row == (if (uplo == .upper) k else 0);
            if (!is_logical or (diag == .unit and is_diagonal)) {
                try expectNan(T, a[band_row + column * lda]);
            } else {
                try expectFinite(T, a[band_row + column * lda]);
            }
        }
    }
}

fn expectBytesEqual(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn runCase(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, k: usize, check_facade: bool) !void {
    const lda = k + 4;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const solution = try allocator.alloc(T, n);
    defer allocator.free(solution);
    const rhs = try allocator.alloc(T, n);
    defer allocator.free(rhs);
    const x_storage = try allocator.alloc(T, n + 2);
    defer allocator.free(x_storage);

    fillBandMatrix(T, a, uplo, diag, n, k, lda);
    const a_before = try allocator.dupe(T, a);
    defer allocator.free(a_before);
    for (solution, 0..) |*entry, i| entry.* = solutionValue(T, i);
    buildReferenceRhs(T, uplo, trans, diag, n, k, a, lda, solution, rhs);

    const x = x_storage[1 .. n + 1];
    @memcpy(x, rhs);
    x_storage[0] = value(T, 123.0, -77.0);
    x_storage[n + 1] = value(T, -91.0, 55.0);

    const forced_hit = triangular_band_solve.testing.tryTbsvForX86(
        T,
        uplo,
        trans,
        diag,
        @intCast(n),
        @intCast(k),
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
    );
    try std.testing.expect(forced_hit);
    for (solution, x) |want, got| try expectApprox(T, want, got);
    try expectApprox(T, value(T, 123.0, -77.0), x_storage[0]);
    try expectApprox(T, value(T, -91.0, 55.0), x_storage[n + 1]);

    @memcpy(x, rhs);
    const production_hit = triangular_band_solve.tryTbsv(
        T,
        uplo,
        trans,
        diag,
        @intCast(n),
        @intCast(k),
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
    );
    if (builtin.cpu.arch == .x86_64) {
        try std.testing.expect(production_hit);
        for (solution, x) |want, got| try expectApprox(T, want, got);
    } else {
        try std.testing.expect(!production_hit);
        try expectBytesEqual(T, rhs, x);
    }
    if (check_facade) {
        @memcpy(x, rhs);
        matrix_vector.tbsv(T, uplo, trans, diag, @intCast(n), @intCast(k), a.ptr, @intCast(lda), x.ptr, 1);
        for (solution, x) |want, got| try expectApprox(T, want, got);
    }

    try expectApprox(T, value(T, 123.0, -77.0), x_storage[0]);
    try expectApprox(T, value(T, -91.0, 55.0), x_storage[n + 1]);
    try expectBytesEqual(T, a_before, a);
    try expectBandSentinels(T, a, uplo, diag, n, k, lda);
}

fn checkType(comptime T: type) !void {
    for ([_]usize{ 0, 1, 8 }) |k| {
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                for ([_]Diag{ .unit, .non_unit }) |diag| {
                    try runCase(T, uplo, trans, diag, 512, k, false);
                }
            }
        }
    }
}

fn checkComplexVectorHelpers(comptime T: type) !void {
    try runCase(T, .upper, .no_trans, .non_unit, 1024, 64, false);
    try runCase(T, .lower, .no_trans, .unit, 1024, 64, false);
    try runCase(T, .upper, .conj_trans, .unit, 1024, 64, false);
    try runCase(T, .lower, .conj_trans, .non_unit, 1024, 64, false);
}

fn expectGateMissUnchanged(comptime T: type, n: scalar.BlasInt, k: scalar.BlasInt, incx: scalar.BlasInt) !void {
    var a = [1]T{nanValue(T)};
    var x = [4]T{
        value(T, 1.0, -2.0),
        value(T, 3.0, -4.0),
        value(T, 5.0, -6.0),
        value(T, 7.0, -8.0),
    };
    const before = x;

    try std.testing.expect(!triangular_band_solve.testing.tryTbsvForX86(T, .upper, .no_trans, .unit, n, k, &a, 1, &x, incx));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
    try std.testing.expect(!triangular_band_solve.tryTbsv(T, .upper, .no_trans, .unit, n, k, &a, 1, &x, incx));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
}

fn checkGateMisses(comptime T: type) !void {
    try std.testing.expect(triangular_band_solve.testing.gateAllowsForX86(T, 512, 0, 1));
    try std.testing.expect(triangular_band_solve.testing.gateAllowsForX86(T, 512, 32, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 511, 0, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, -1, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, 33, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, 0, 2));
    try std.testing.expectEqual(
        builtin.cpu.arch == .x86_64,
        triangular_band_solve.testing.productionGateAllows(T, 512, 8, 1),
    );

    try expectGateMissUnchanged(T, 511, 0, 1);
    try expectGateMissUnchanged(T, 512, -1, 1);
    try expectGateMissUnchanged(T, 512, 33, 1);
    try expectGateMissUnchanged(T, 512, 0, 2);
}

test "TBSV band-window solve matches independent s/d/c/z references" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkType(T);
    inline for (.{ ComplexF32, ComplexF64 }) |T| try checkComplexVectorHelpers(T);
}

test "matrix-vector facade routes compact TBSV" {
    try runCase(f64, .lower, .conj_trans, .unit, 512, 8, true);
}

test "TBSV band-window gate misses leave x unchanged" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkGateMisses(T);
}
