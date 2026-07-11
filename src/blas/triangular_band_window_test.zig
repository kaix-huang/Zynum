// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const triangular = @import("core/matrix_vector/triangular.zig");
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
    if (row == column) return value(T, 1.25 + re_seed / 256.0, im_seed / 320.0);
    return value(T, re_seed / 64.0, im_seed / 80.0);
}

fn vectorValue(comptime T: type, index: usize) T {
    const re_seed = @as(f64, @floatFromInt((index * 13 + 7) % 41)) - 20.0;
    const im_seed = @as(f64, @floatFromInt((index * 5 + 9) % 37)) - 18.0;
    return value(T, re_seed / 17.0, im_seed / 19.0);
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
    trans_: Order,
    diag: Diag,
    row: usize,
    column: usize,
    matrix_index: usize,
    a: []const T,
    x: []const T,
    y: []T,
) void {
    const stored = if (diag == .unit and row == column) scalar.one(T) else a[matrix_index];
    if (trans_ == .no_trans) {
        y[row] = scalar.add(T, y[row], scalar.mul(T, stored, x[column]));
    } else {
        const av = if (trans_ == .conj_trans) scalar.conj(T, stored) else stored;
        y[column] = scalar.add(T, y[column], scalar.mul(T, av, x[row]));
    }
}

// This reference is deliberately out of place and column-oriented. The
// production leaf is in place and row-oriented, so dependency or window bugs
// cannot be hidden by sharing its traversal order.
fn referenceTbmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, k: usize, a: []const T, lda: usize, x: []const T, y: []T) void {
    @memset(y, scalar.zero(T));
    for (0..n) |column| {
        if (uplo == .upper) {
            const first_row = column - @min(column, k);
            var band_row = k - (column - first_row);
            for (first_row..column + 1) |row| {
                accumulateReferenceEntry(T, trans_, diag, row, column, band_row + column * lda, a, x, y);
                band_row += 1;
            }
        } else {
            const row_end = @min(n, column + k + 1);
            var band_row: usize = 0;
            for (column..row_end) |row| {
                accumulateReferenceEntry(T, trans_, diag, row, column, band_row + column * lda, a, x, y);
                band_row += 1;
            }
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 3e-3) else @as(f64, 1e-10);
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

fn expectMatrixUnchanged(comptime T: type, before: []const T, after: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(after));
}

fn expectPaddingStillNan(comptime T: type, a: []const T, n: usize, k: usize, lda: usize) !void {
    for (0..n) |column| {
        for (k + 1..lda) |band_row| {
            const entry = a[band_row + column * lda];
            if (T == f32 or T == f64) {
                try std.testing.expect(std.math.isNan(entry));
            } else {
                try std.testing.expect(std.math.isNan(entry.re));
                try std.testing.expect(std.math.isNan(entry.im));
            }
        }
    }
}

fn runCase(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, k: usize, lda: usize) !void {
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const input = try allocator.alloc(T, n);
    defer allocator.free(input);
    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);
    const x_storage = try allocator.alloc(T, n + 2);
    defer allocator.free(x_storage);

    fillBandMatrix(T, a, uplo, diag, n, k, lda);
    const a_before = try allocator.dupe(T, a);
    defer allocator.free(a_before);

    for (input, 0..) |*entry, i| entry.* = vectorValue(T, i);
    const x = x_storage[1 .. n + 1];
    @memcpy(x, input);
    x_storage[0] = value(T, 123.0, -77.0);
    x_storage[n + 1] = value(T, -91.0, 55.0);

    referenceTbmv(T, uplo, trans_, diag, n, k, a, lda, input, expected);
    triangular.tbmv(T, uplo, trans_, diag, @intCast(n), @intCast(k), a.ptr, @intCast(lda), x.ptr, 1);

    for (expected, x) |want, got| try expectApprox(T, want, got);
    try expectApprox(T, value(T, 123.0, -77.0), x_storage[0]);
    try expectApprox(T, value(T, -91.0, 55.0), x_storage[n + 1]);
    try expectMatrixUnchanged(T, a_before, a);
    try expectPaddingStillNan(T, a, n, k, lda);
}

fn runAllOperations(comptime T: type, n: usize, k: usize, lda: usize) !void {
    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        inline for (.{ Diag.non_unit, Diag.unit }) |diag| {
            try runCase(T, uplo, .no_trans, diag, n, k, lda);
            try runCase(T, uplo, .trans, diag, n, k, lda);
            if (comptime T == ComplexF32 or T == ComplexF64) try runCase(T, uplo, .conj_trans, diag, n, k, lda);
        }
    }
}

fn runBothLeadingDimensions(comptime T: type, n: usize, k: usize) !void {
    try runAllOperations(T, n, k, k + 1);
    try runAllOperations(T, n, k, k + 4);
}

fn checkCompactEdges(comptime T: type) !void {
    try runBothLeadingDimensions(T, 1, 0);
    inline for (.{ 0, 1, 3, 8 }) |k| try runBothLeadingDimensions(T, 9, k);
}

fn checkGateBoundary(comptime T: type) !void {
    try runCase(T, .upper, .no_trans, .unit, 511, 3, 4);
    try runCase(T, .lower, .trans, .non_unit, 511, 3, 7);
    if (comptime T == ComplexF32 or T == ComplexF64) try runCase(T, .upper, .conj_trans, .non_unit, 511, 1, 4);

    inline for (.{ 0, 1, 3 }) |k| try runBothLeadingDimensions(T, 512, k);

    // Exercise both sides of k <= n/16 and the wide-band fallback without
    // multiplying the full cross-product of slow O(n^2) fallback cases.
    try runCase(T, .upper, .trans, .unit, 512, 32, 36);
    try runCase(T, .lower, .no_trans, .non_unit, 512, 33, 34);
    try runCase(T, .lower, if (T == ComplexF32 or T == ComplexF64) .conj_trans else .trans, .unit, 512, 511, 515);
}

test "TBMV compact edges match an independent out-of-place reference" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkCompactEdges(T);
}

test "TBMV unit-stride band-window gate preserves all operation variants" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkGateBoundary(T);
}
