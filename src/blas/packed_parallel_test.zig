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

test "packed SPMV and HPMV task paths match single-thread fallback" {
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
