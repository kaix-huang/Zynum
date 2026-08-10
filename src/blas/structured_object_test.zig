// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const bridge = @import("kernels/isolated/x86_64_structured_bridge.zig");
const abi = @import("kernels/isolated/x86_64_structured_abi.zig");
const scalar = @import("core/shared/scalar.zig");
const symmetric = @import("core/matrix_matrix/symmetric.zig");
const triangular = @import("core/matrix_matrix/triangular.zig");
const runtime = @import("runtime.zig");
const core_pool = @import("core/execution/thread_pool.zig");

fn sample(comptime T: type, index: usize, salt: usize) T {
    const re = (@as(f64, @floatFromInt((index * 17 + salt * 11) % 29)) - 14) / 32;
    const im = (@as(f64, @floatFromInt((index * 7 + salt * 13) % 31)) - 15) / 48;
    if (comptime scalar.isComplex(T)) return .{ .re = @floatCast(re), .im = @floatCast(im) };
    return @floatCast(re);
}

fn value(comptime T: type, re: f64, im: f64) T {
    if (comptime scalar.isComplex(T)) return .{ .re = @floatCast(re), .im = @floatCast(im) };
    return @floatCast(re);
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    const tolerance: scalar.Real(T) = if (T == f32 or T == scalar.ComplexF32) 2e-3 else 2e-10;
    if (comptime scalar.isComplex(T)) {
        if (@abs(expected.re - actual.re) > tolerance) try std.testing.expectApproxEqRel(expected.re, actual.re, tolerance);
        if (@abs(expected.im - actual.im) > tolerance) try std.testing.expectApproxEqRel(expected.im, actual.im, tolerance);
    } else {
        if (@abs(expected - actual) > tolerance) try std.testing.expectApproxEqRel(expected, actual, tolerance);
    }
}

fn expectCloseSlices(comptime T: type, expected: []const T, actual: []const T) !void {
    for (expected, actual) |want, got| try expectClose(T, want, got);
}

fn testDenseSymm(comptime T: type, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
    const m: usize = 128;
    const n: usize = 128;
    const order = if (side == .left) m else n;
    const lda: i32 = @intCast(order + 1);
    const ldb: i32 = @intCast(m + 2);
    const ldc: i32 = @intCast(m + 3);
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, @as(usize, @intCast(lda)) * order);
    defer allocator.free(a);
    const b = try allocator.alloc(T, @as(usize, @intCast(ldb)) * n);
    defer allocator.free(b);
    const expected = try allocator.alloc(T, @as(usize, @intCast(ldc)) * n);
    defer allocator.free(expected);
    const actual = try allocator.alloc(T, expected.len);
    defer allocator.free(actual);
    for (a, 0..) |*entry, index| entry.* = sample(T, index, 1);
    for (b, 0..) |*entry, index| entry.* = sample(T, index, 2);
    for (expected, 0..) |*entry, index| entry.* = sample(T, index, 3);
    @memcpy(actual, expected);
    const alpha = value(T, 0.75, if (hermitian) 0 else -0.125);
    const beta = value(T, -0.25, if (hermitian) 0 else 0.0625);
    symmetric.symm(T, side, uplo, @intCast(m), @intCast(n), alpha, a.ptr, lda, b.ptr, ldb, beta, expected.ptr, ldc, hermitian);
    try std.testing.expect(bridge.trySymm(T, side, uplo, @intCast(m), @intCast(n), alpha, a.ptr, lda, b.ptr, ldb, beta, actual.ptr, ldc, hermitian));
    try expectCloseSlices(T, expected, actual);
}

fn testRightTriangular(comptime T: type, solve: bool, uplo: scalar.Uplo, transpose: scalar.Order, diagonal: scalar.Diag) !void {
    const m: usize = 512;
    const n: usize = 128;
    const lda: i32 = @intCast(n + 1);
    const ldb: i32 = @intCast(m + 2);
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, @as(usize, @intCast(lda)) * n);
    defer allocator.free(a);
    const expected = try allocator.alloc(T, @as(usize, @intCast(ldb)) * n);
    defer allocator.free(expected);
    const actual = try allocator.alloc(T, expected.len);
    defer allocator.free(actual);
    for (a, 0..) |*entry, index| entry.* = sample(T, index, 4);
    for (0..n) |j| a[j + j * @as(usize, @intCast(lda))] = value(T, 2.5 + @as(f64, @floatFromInt(j)) / 256, 0.03125);
    for (expected, 0..) |*entry, index| entry.* = sample(T, index, 5);
    @memcpy(actual, expected);
    const alpha = value(T, -0.625, 0.1875);
    if (solve) {
        triangular.trsm(T, .right, uplo, transpose, diagonal, @intCast(m), @intCast(n), alpha, a.ptr, lda, expected.ptr, ldb);
        try std.testing.expect(bridge.tryTrsmRight(T, uplo, transpose, diagonal, @intCast(m), @intCast(n), alpha, a.ptr, lda, actual.ptr, ldb));
    } else {
        triangular.trmm(T, .right, uplo, transpose, diagonal, @intCast(m), @intCast(n), alpha, a.ptr, lda, expected.ptr, ldb);
        try std.testing.expect(bridge.tryTrmmRight(T, uplo, transpose, diagonal, @intCast(m), @intCast(n), alpha, a.ptr, lda, actual.ptr, ldb));
    }
    try expectCloseSlices(T, expected, actual);
}

test "isolated dense SYMM and HEMM cross the private object ABI" {
    try testDenseSymm(f32, .left, .upper, false);
    try testDenseSymm(scalar.ComplexF32, .right, .lower, true);
}

test "isolated dense candidate rejects rectangular work before write" {
    const m: usize = 128;
    const n: usize = 129;
    const lda: i32 = 128;
    const ldb: i32 = 130;
    const ldc: i32 = 131;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f32, @as(usize, @intCast(lda)) * m);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, @as(usize, @intCast(ldb)) * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, @as(usize, @intCast(ldc)) * n);
    defer allocator.free(c);
    const original = try allocator.alloc(f32, c.len);
    defer allocator.free(original);
    for (a, 0..) |*entry, index| entry.* = sample(f32, index, 8);
    for (b, 0..) |*entry, index| entry.* = sample(f32, index, 9);
    for (c, 0..) |*entry, index| entry.* = sample(f32, index, 10);
    @memcpy(original, c);

    try std.testing.expect(!bridge.trySymm(f32, .left, .upper, @intCast(m), @intCast(n), 0, a.ptr, lda, b.ptr, ldb, 0.25, c.ptr, ldc, false));
    try std.testing.expectEqualSlices(f32, original, c);
}

test "isolated right TRMM and TRSM use the host task runtime" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    try testRightTriangular(f32, false, .upper, .no_trans, .non_unit);
    try testRightTriangular(scalar.ComplexF32, true, .lower, .conj_trans, .unit);
}

test "isolated right candidate rejects before write when helpers are unavailable" {
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f32, 129 * 128);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 514 * 128);
    defer allocator.free(b);
    const original = try allocator.dupe(f32, b);
    defer allocator.free(original);
    for (a, 0..) |*entry, index| entry.* = sample(f32, index, 6);
    for (b, 0..) |*entry, index| entry.* = sample(f32, index, 7);
    @memcpy(original, b);
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    try std.testing.expect(!bridge.tryTrmmRight(f32, .upper, .no_trans, .non_unit, 512, 128, 0.75, a.ptr, 129, b.ptr, 514));
    try std.testing.expectEqualSlices(f32, original, b);
}

test "structured object ABI layout remains fixed" {
    try std.testing.expectEqual(@as(usize, 88), @sizeOf(abi.Request));
    try std.testing.expectEqual(@as(usize, 88), @sizeOf(abi.GemmRequest));
}
