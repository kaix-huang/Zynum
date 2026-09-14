// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("core/shared/scalar.zig");
const symmetric = @import("core/matrix_vector/symmetric.zig");
const core_pool = @import("core/execution/thread_pool.zig");
const runtime = @import("runtime.zig");

const Uplo = scalar.Uplo;

fn testValue(comptime T: type, index: usize, phase: usize) T {
    const re = @as(f64, @floatFromInt((index * 17 + phase * 11) % 37)) / 29.0 - 0.625;
    if (T == f32 or T == f64) return @floatCast(re);
    const im = @as(f64, @floatFromInt((index * 13 + phase * 7) % 31)) / 27.0 - 0.5;
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn expectApprox(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    if (T == f32 or T == f64) {
        const tolerance: T = if (T == f32) 3e-3 else 1e-10;
        for (expected, actual) |want, got| try std.testing.expectApproxEqAbs(want, got, tolerance);
    } else {
        const Real = if (T == scalar.ComplexF32) f32 else f64;
        const tolerance: Real = if (T == scalar.ComplexF32) 6e-3 else 2e-10;
        for (expected, actual) |want, got| {
            try std.testing.expectApproxEqAbs(want.re, got.re, tolerance);
            try std.testing.expectApproxEqAbs(want.im, got.im, tolerance);
        }
    }
}

fn expectPackedMvParallelMatchesSingleThread(comptime T: type, uplo: Uplo, herm: bool) !void {
    const n: usize = 512;
    const packed_len = n * (n + 1) / 2;
    const allocator = std.testing.allocator;
    const ap = try allocator.alloc(T, packed_len);
    defer allocator.free(ap);
    const x = try allocator.alloc(T, n);
    defer allocator.free(x);
    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);
    const actual = try allocator.alloc(T, n);
    defer allocator.free(actual);

    for (ap, 0..) |*value, i| value.* = testValue(T, i, 1);
    for (x, 0..) |*value, i| value.* = testValue(T, i, 2);
    for (expected, actual, 0..) |*want, *got, i| {
        const value = testValue(T, i, 3);
        want.* = value;
        got.* = value;
    }
    if (herm and comptime scalar.isComplex(T)) {
        for (0..n) |j| {
            const diagonal = if (uplo == .upper) j * (j + 1) / 2 + j else j * (2 * n - j + 1) / 2;
            ap[diagonal].im = @floatCast(4.0 + @as(f64, @floatFromInt(j % 7)));
        }
    }

    const alpha = if (comptime scalar.isComplex(T))
        T{ .re = 0.625, .im = -0.25 }
    else
        @as(T, 0.625);
    const beta = if (comptime scalar.isComplex(T))
        T{ .re = -0.375, .im = 0.125 }
    else
        @as(T, -0.375);

    runtime.setMaxThreads(1);
    symmetric.spmv(T, uplo, @intCast(n), alpha, ap.ptr, x.ptr, 1, beta, expected.ptr, 1, herm);
    runtime.setMaxThreads(4);
    symmetric.spmv(T, uplo, @intCast(n), alpha, ap.ptr, x.ptr, 1, beta, actual.ptr, 1, herm);
    try expectApprox(T, expected, actual);
}

fn expectForcedPackedColumns(comptime T: type, uplo: Uplo) !void {
    const catalog = @import("kernels/shared/matrix_vector/catalog.zig");
    const tuning = @import("kernels/shared/matrix_vector/tuning.zig");
    const old_id = catalog.Implementation.compact_symmetric_packed;
    const new_id = catalog.Implementation.compact_symmetric_packed_fused;
    const descriptor = catalog.findImplementation(.spmv, tuning.scalarKind(T), new_id).?;
    try std.testing.expectEqual(catalog.Lifecycle.production, descriptor.lifecycle);
    try std.testing.expectEqual(old_id, descriptor.fallback.?.implementation);
    try std.testing.expectEqual(catalog.CompletionScope.output_region, descriptor.completion);
    try std.testing.expect(descriptor.lifecycle.defaultEligible());
    const candidate = tuning.fixed_candidates_2026_07_17.symmetric;
    const expected_id = if (@import("builtin").cpu.arch == .aarch64) new_id else old_id;
    try std.testing.expectEqual(expected_id, candidate.selectRealPackedImplementation(T));
    var production = tuning.production_2026_07_17.symmetric;
    try std.testing.expect(production.enable_fused_real_packed_single);
    try std.testing.expect(production.enable_fused_real_packed);
    try std.testing.expectEqual(expected_id, production.selectRealPackedSingleImplementation(T));
    try std.testing.expectEqual(expected_id, production.selectRealPackedImplementation(T));
    production.enable_fused_real_packed = false;
    try std.testing.expectEqual(old_id, production.selectRealPackedImplementation(T));
    production.enable_fused_real_packed_single = false;
    try std.testing.expectEqual(old_id, production.selectRealPackedSingleImplementation(T));
    try std.testing.expect(candidate.enable_fused_real_packed);
    try std.testing.expect(candidate.enable_fused_real_packed_single);

    const n = 35;
    var ap: [n * (n + 1) / 2]T = undefined;
    var x: [n]T = undefined;
    var expected: [n]T = undefined;
    var actual: [n]T = undefined;
    for (&ap, 0..) |*item, i| item.* = testValue(T, i, 14);
    for (&x, 0..) |*item, i| item.* = testValue(T, i, 15);
    inline for (.{ .{ 0, 1 }, .{ 1, 17 }, .{ 17, n } }) |range| {
        for (&expected, 0..) |*item, i| item.* = testValue(T, i, 16);
        actual = expected;
        try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, old_id, uplo, n, range[0], range[1], 0.625, &ap, &x, &expected));
        try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, new_id, uplo, n, range[0], range[1], 0.625, &ap, &x, &actual));
        try expectApprox(T, &expected, &actual);
    }
    // A private-delta violation must retain the original alias behavior.
    expected = x;
    actual = x;
    try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, old_id, uplo, n, 0, n, 0.625, &ap, &expected, &expected));
    try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, new_id, uplo, n, 0, n, 0.625, &ap, &actual, &actual));
    try std.testing.expectEqualSlices(T, &expected, &actual);
    const snapshot = actual;
    try std.testing.expect(!symmetric.testing.packedSymmetricColumnsImplementation(T, new_id, uplo, n, 0, n + 1, 1, &ap, &x, &actual));
    try std.testing.expectEqualSlices(T, &snapshot, &actual);

    // Packed semantics evaluate A*0: do not inherit the band's zero-AXPY skip.
    inline for (.{ std.math.inf(T), std.math.nan(T) }) |nonfinite| {
        @memset(&ap, 1);
        ap[1] = nonfinite;
        @memset(&x, 0);
        @memset(&expected, 2);
        actual = expected;
        const j: usize = if (uplo == .upper) 1 else 0;
        try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, old_id, uplo, n, j, j + 1, 1, &ap, &x, &expected));
        try std.testing.expect(symmetric.testing.packedSymmetricColumnsImplementation(T, new_id, uplo, n, j, j + 1, 1, &ap, &x, &actual));
        const changed: usize = if (uplo == .upper) 0 else 1;
        try std.testing.expect(std.math.isNan(actual[changed]));
        for (expected, actual) |want, got| {
            if (std.math.isNan(want)) try std.testing.expect(std.math.isNan(got)) else try std.testing.expectEqual(want, got);
        }
    }
}

fn expectPackedSingleThreadReference(comptime T: type, uplo: Uplo) !void {
    const n: usize = 512;
    const packed_len = n * (n + 1) / 2;
    const allocator = std.testing.allocator;
    const ap_buffer = try allocator.alloc(T, packed_len + 2);
    defer allocator.free(ap_buffer);
    const x_buffer = try allocator.alloc(T, n + 2);
    defer allocator.free(x_buffer);
    const y_buffer = try allocator.alloc(T, n + 2);
    defer allocator.free(y_buffer);
    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);
    const ap = ap_buffer[1 .. packed_len + 1];
    const x = x_buffer[1 .. n + 1];
    const y = y_buffer[1 .. n + 1];
    const guard: T = -731;
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    const tuning = @import("kernels/shared/matrix_vector/tuning.zig");
    try std.testing.expect(tuning.active.symmetric.preferPackedParallel(n));
    try std.testing.expectEqual(@as(usize, 1), core_pool.taskCount(n, tuning.active.symmetric.packedMinColumns()));
    @memset(ap, 1);
    @memset(x, 1);
    @memset(y, 2);
    try std.testing.expect(!symmetric.testing.singlePackedSymmetricImplementation(T, uplo, n, 1, ap.ptr, y.ptr, 1, 1, y.ptr, 1));
    try std.testing.expect(!symmetric.testing.singlePackedSymmetricImplementation(T, uplo, n, 1, ap.ptr, x.ptr, 2, 1, y.ptr, 1));
    try std.testing.expect(!symmetric.testing.singlePackedSymmetricImplementation(T, uplo, n - 1, 1, ap.ptr, x.ptr, 1, 1, y.ptr, 1));
    for (y) |item| try std.testing.expectEqual(@as(T, 2), item);
    for (0..6) |scenario| {
        ap_buffer[0] = guard;
        ap_buffer[packed_len + 1] = guard;
        x_buffer[0] = guard;
        x_buffer[n + 1] = guard;
        y_buffer[0] = guard;
        y_buffer[n + 1] = guard;
        for (ap, 0..) |*item, i| item.* = testValue(T, i, 21);
        for (x, 0..) |*item, i| item.* = testValue(T, i, 22);
        for (y, 0..) |*item, i| item.* = testValue(T, i, 23);
        const alpha: T = if (scenario == 2) 0 else 0.625;
        const beta: T = if (scenario == 1) 0 else -0.25;
        if (scenario == 1) @memset(y, std.math.nan(T));
        if (scenario == 2) {
            @memset(ap, std.math.nan(T));
            @memset(x, std.math.nan(T));
        }
        if (scenario == 3) {
            ap[1] = std.math.inf(T);
            @memset(x, 0);
        }
        if (scenario == 4) ap[1] = std.math.nan(T);
        if (scenario == 5) ap[1] = std.math.inf(T);
        for (0..n) |row| {
            expected[row] = if (beta == 0) 0 else beta * y[row];
            if (alpha == 0) continue;
            var sum: T = 0;
            for (0..n) |column| {
                const lo = @min(row, column);
                const hi = @max(row, column);
                const offset = if (uplo == .upper) hi * (hi + 1) / 2 + lo else lo * (2 * n - lo + 1) / 2 + hi - lo;
                sum += ap[offset] * x[column];
            }
            expected[row] += alpha * sum;
        }
        try std.testing.expect(symmetric.testing.singlePackedSymmetricImplementation(T, uplo, n, alpha, ap.ptr, x.ptr, 1, beta, y.ptr, 1));
        for (expected, y) |want, got| {
            if (std.math.isNan(want)) {
                try std.testing.expect(std.math.isNan(got));
            } else if (std.math.isInf(want)) {
                try std.testing.expectEqual(want, got);
            } else {
                try std.testing.expectApproxEqAbs(want, got, if (T == f32) @as(T, 3e-3) else @as(T, 1e-10));
            }
        }
        for ([_]T{ ap_buffer[0], ap_buffer[packed_len + 1], x_buffer[0], x_buffer[n + 1], y_buffer[0], y_buffer[n + 1] }) |actual_guard| {
            try std.testing.expectEqual(guard, actual_guard);
        }
    }
}

test "packed SPMV and HPMV task paths match single-thread fallback" {
    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        try expectForcedPackedColumns(f32, uplo);
        try expectForcedPackedColumns(f64, uplo);
        try expectPackedSingleThreadReference(f32, uplo);
        try expectPackedSingleThreadReference(f64, uplo);
    }
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
        symmetric.freeCurrentThreadCaches();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        try expectPackedMvParallelMatchesSingleThread(f32, uplo, false);
        try expectPackedMvParallelMatchesSingleThread(f64, uplo, false);
        try expectPackedMvParallelMatchesSingleThread(scalar.ComplexF32, uplo, true);
        try expectPackedMvParallelMatchesSingleThread(scalar.ComplexF64, uplo, true);
    }
}
