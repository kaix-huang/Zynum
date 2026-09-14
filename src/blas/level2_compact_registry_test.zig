// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Boundary differential tests for complete compact-band Level 2 families.

const std = @import("std");

const types = @import("types.zig");
const scalar = @import("core/shared/scalar.zig");
const general = @import("core/matrix_vector/general.zig");
const symmetric = @import("core/matrix_vector/symmetric.zig");
const catalog = @import("kernels/shared/matrix_vector/catalog.zig");

fn value(comptime T: type, index: usize, phase: usize) T {
    const re = @as(f64, @floatFromInt((index * 17 + phase * 11) % 41)) / 29.0 - 0.65;
    if (T == f32 or T == f64) return @floatCast(re);
    const im = @as(f64, @floatFromInt((index * 13 + phase * 7) % 37)) / 31.0 - 0.55;
    return .{ .re = re, .im = im };
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    if (T == f32) return std.testing.expectApproxEqAbs(expected, actual, 3e-5);
    if (T == f64) return std.testing.expectApproxEqAbs(expected, actual, 2e-10);
    try std.testing.expectApproxEqAbs(expected.re, actual.re, 4e-10);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, 4e-10);
}

fn generalBandIndex(kl: usize, ku: usize, lda: usize, row: usize, column: usize) ?usize {
    if (row + ku < column or row > column + kl) return null;
    return ku + row - column + column * lda;
}

fn differentialGeneralBand(comptime T: type, trans: scalar.Order) !void {
    const m: usize = 512;
    const n: usize = 512;
    const kl: usize = 8;
    const ku: usize = 8;
    const lda: usize = 19;
    const input_len = if (trans == .no_trans) n else m;
    const output_len = if (trans == .no_trans) m else n;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, input_len);
    defer allocator.free(x);
    const actual = try allocator.alloc(T, output_len);
    defer allocator.free(actual);
    const expected = try allocator.alloc(T, output_len);
    defer allocator.free(expected);

    for (a, 0..) |*item, i| item.* = value(T, i, 1);
    for (x, 0..) |*item, i| item.* = value(T, i, 2);
    for (actual, expected, 0..) |*got, *want, i| {
        const initial = value(T, i, 3);
        got.* = initial;
        want.* = initial;
    }
    const alpha = value(T, 5, 4);
    const beta = value(T, 7, 5);

    for (0..output_len) |row| {
        var sum = scalar.zero(T);
        for (0..input_len) |column| {
            const matrix_row = if (trans == .no_trans) row else column;
            const matrix_column = if (trans == .no_trans) column else row;
            const index = generalBandIndex(kl, ku, lda, matrix_row, matrix_column) orelse continue;
            var av = a[index];
            if (trans == .conj_trans) av = scalar.conj(T, av);
            sum = scalar.add(T, sum, scalar.mul(T, av, x[column]));
        }
        expected[row] = scalar.add(
            T,
            scalar.mul(T, beta, expected[row]),
            scalar.mul(T, alpha, sum),
        );
    }

    general.gbmv(T, trans, m, n, kl, ku, alpha, a.ptr, lda, x.ptr, 1, beta, actual.ptr, 1);
    for (expected, actual) |want, got| try expectClose(T, want, got);
}

test "compact GBMV boundary gate covers N T and C windows" {
    try differentialGeneralBand(f64, .no_trans);
    try differentialGeneralBand(f64, .trans);
    try differentialGeneralBand(types.ComplexF64, .conj_trans);
}

fn symmetricBandValue(
    comptime T: type,
    uplo: scalar.Uplo,
    n: usize,
    k: usize,
    lda: usize,
    a: []const T,
    row: usize,
    column: usize,
    hermitian: bool,
) T {
    const direct = (uplo == .upper and row <= column) or (uplo == .lower and row >= column);
    if (!direct) {
        var mirrored = symmetricBandValue(T, uplo, n, k, lda, a, column, row, false);
        if (hermitian) mirrored = scalar.conj(T, mirrored);
        return mirrored;
    }
    const distance = if (row > column) row - column else column - row;
    if (distance > k) return scalar.zero(T);
    const index = switch (uplo) {
        .upper => k + row - column + column * lda,
        .lower => row - column + column * lda,
    };
    var result = a[index];
    if (comptime scalar.isComplex(T)) {
        if (hermitian and row == column) result.im = 0;
    }
    return result;
}

fn differentialSymmetricBand(comptime T: type, uplo: scalar.Uplo, hermitian: bool) !void {
    const n: usize = 512;
    const k: usize = 8;
    const lda: usize = 11;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, n);
    defer allocator.free(x);
    const actual = try allocator.alloc(T, n);
    defer allocator.free(actual);
    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);

    for (a, 0..) |*item, i| item.* = value(T, i, 9);
    for (x, 0..) |*item, i| item.* = value(T, i, 10);
    for (actual, expected, 0..) |*got, *want, i| {
        const initial = value(T, i, 11);
        got.* = initial;
        want.* = initial;
    }
    const alpha = value(T, 5, 12);
    const beta = value(T, 7, 13);

    for (0..n) |row| {
        var sum = scalar.zero(T);
        for (0..n) |column| {
            const av = symmetricBandValue(T, uplo, n, k, lda, a, row, column, hermitian);
            sum = scalar.add(T, sum, scalar.mul(T, av, x[column]));
        }
        expected[row] = scalar.add(
            T,
            scalar.mul(T, beta, expected[row]),
            scalar.mul(T, alpha, sum),
        );
    }

    symmetric.sbmv(T, uplo, n, k, alpha, a.ptr, lda, x.ptr, 1, beta, actual.ptr, 1, hermitian);
    for (expected, actual) |want, got| try expectClose(T, want, got);
    if (comptime T == f32 or T == f64) {
        inline for (.{ catalog.Implementation.compact_symmetric_band, catalog.Implementation.compact_symmetric_band_fused }) |implementation| {
            for (actual, 0..) |*item, i| item.* = value(T, i, 11);
            try std.testing.expect(symmetric.testing.symmetricBandImplementation(T, implementation, uplo, n, k, alpha, a.ptr, lda, x.ptr, 1, beta, actual.ptr, 1));
            for (expected, actual) |want, got| try expectClose(T, want, got);
            const snapshot = try allocator.dupe(T, actual);
            defer allocator.free(snapshot);
            try std.testing.expect(!symmetric.testing.symmetricBandImplementation(T, implementation, uplo, n, k, alpha, a.ptr, lda, x.ptr, 2, beta, actual.ptr, 1));
            try std.testing.expectEqualSlices(T, snapshot, actual);
        }
    }
}

fn differentialFusedSymmetricSegment(comptime T: type) !void {
    const fixed = @import("kernels/shared/matrix_vector/fixed_simd.zig");
    const cfg = fixed.Config{ .lane_count = if (T == f32) 4 else 2 };
    var a: [35]T = undefined;
    var x: [35]T = undefined;
    var y: [35]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 17);
    for (&x, 0..) |*item, i| item.* = value(T, i, 18);
    for (0..34) |n| {
        inline for (.{ @as(T, 0), @as(T, -0.625) }) |coefficient| {
            for (&y, 0..) |*item, i| item.* = value(T, i, 19);
            const before = y;
            var expected: T = 0;
            for (0..n) |i| expected += a[i] * x[i];
            const result = fixed.symmetricAxpyDotUnitReal(T, cfg, n, coefficient, &a, &x, &y).?;
            try expectClose(T, expected, result);
            for (0..y.len) |i| {
                const wanted = if (i < n) before[i] + coefficient * a[i] else before[i];
                try expectClose(T, wanted, y[i]);
            }
        }
    }
    // Partially overlapping inputs must reject without changing any output.
    const before = y;
    try std.testing.expect(fixed.symmetricAxpyDotUnitReal(T, cfg, 17, 1, &a, &y, y[1..].ptr) == null);
    try std.testing.expectEqualSlices(T, &before, &y);
    try std.testing.expect(fixed.symmetricAxpyDotUnitReal(T, cfg, 17, 1, &y, &x, y[1..].ptr) == null);
    try std.testing.expectEqualSlices(T, &before, &y);
    // Zero AXPY coefficient cannot contaminate Y from a non-finite matrix lane.
    a[0] = std.math.nan(T);
    const result = fixed.symmetricAxpyDotUnitReal(T, cfg, 17, 0, &a, &x, &y).?;
    try std.testing.expect(std.math.isNan(result));
    try std.testing.expectEqualSlices(T, &before, &y);
}

test "compact SBMV and HBMV boundary gates preserve exact stored windows" {
    try differentialFusedSymmetricSegment(f32);
    try differentialFusedSymmetricSegment(f64);
    try differentialSymmetricBand(f32, .upper, false);
    try differentialSymmetricBand(f32, .lower, false);
    try differentialSymmetricBand(f64, .upper, false);
    try differentialSymmetricBand(f64, .lower, false);
    try differentialSymmetricBand(types.ComplexF64, .lower, true);
}

test "compact registry cells remain explicit execution leaves" {
    const tuning = @import("kernels/shared/matrix_vector/tuning.zig");
    inline for (.{ f32, f64 }) |T| {
        const fused = catalog.findImplementation(.sbmv, tuning.scalarKind(T), .compact_symmetric_band_fused).?;
        try std.testing.expectEqual(catalog.Lifecycle.production, fused.lifecycle);
        try std.testing.expectEqual(catalog.IsaCapability.generic, fused.kernel.capability);
        try std.testing.expectEqual(catalog.Implementation.compact_symmetric_band, fused.fallback.?.implementation);
        try std.testing.expectEqual(catalog.CompletionScope.whole_operation, fused.completion);
        var profile = tuning.production_2026_07_17.symmetric;
        try std.testing.expect(fused.lifecycle.defaultEligible());
        try std.testing.expect(profile.enable_fused_real_band);
        const expected = if (@import("builtin").cpu.arch == .aarch64) catalog.Implementation.compact_symmetric_band_fused else catalog.Implementation.compact_symmetric_band;
        try std.testing.expectEqual(expected, profile.selectRealBandImplementation(T, 512, 8));
        try std.testing.expectEqual(catalog.Implementation.compact_symmetric_band, profile.selectRealBandImplementation(T, 511, 8));
        try std.testing.expectEqual(catalog.Implementation.compact_symmetric_band, profile.selectRealBandImplementation(T, 512, 7));
        profile.enable_fused_real_band = false;
        try std.testing.expectEqual(catalog.Implementation.compact_symmetric_band, profile.selectRealBandImplementation(T, 512, 8));
    }
    try std.testing.expect(catalog.findImplementation(.hbmv, .complex_f64, .compact_symmetric_band_fused) == null);

    const gbmv = catalog.findImplementation(.gbmv, .f64, .compact_general_band).?;
    try std.testing.expectEqual(catalog.StoredWindow.exact_band, gbmv.stored_window);
    const hpmv = catalog.findImplementation(.hpmv, .complex_f64, .compact_symmetric_packed).?;
    try std.testing.expectEqual(
        catalog.TaskFallbackContract.private_results_commit_after_all_tasks,
        hpmv.task_fallback,
    );
}
