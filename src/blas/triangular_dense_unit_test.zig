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

fn fillMatrix(comptime T: type, a: []T, lda: usize, n: usize) void {
    for (0..n) |column| {
        for (0..lda) |row| {
            const re_seed = @as(f64, @floatFromInt((row * 17 + column * 11 + 5) % 29)) - 14.0;
            const im_seed = @as(f64, @floatFromInt((row * 7 + column * 19 + 3) % 23)) - 11.0;
            const re = if (row == column) 1.375 + re_seed / 256.0 else re_seed / 1024.0;
            const im = if (row == column) im_seed / 384.0 else im_seed / 1536.0;
            a[row + column * lda] = value(T, re, im);
        }
    }
}

fn fillVectors(comptime T: type, unit: []T, strided: []T) void {
    for (unit, 0..) |*entry, i| {
        const re_seed = @as(f64, @floatFromInt((i * 13 + 7) % 31)) - 15.0;
        const im_seed = @as(f64, @floatFromInt((i * 5 + 9) % 27)) - 13.0;
        entry.* = value(T, re_seed / 17.0, im_seed / 19.0);
        strided[2 * i] = entry.*;
        strided[2 * i + 1] = value(T, 77.0, -33.0);
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 8e-4) else @as(f64, 3e-11);
    if (T == f32 or T == f64) {
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    } else {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
    }
}

fn checkResult(comptime T: type, unit: []const T, strided: []const T) !void {
    for (unit, 0..) |actual, i| {
        try expectApprox(T, strided[2 * i], actual);
        try expectApprox(T, value(T, 77.0, -33.0), strided[2 * i + 1]);
    }
}

fn checkCase(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, a: []const T, lda: usize, unit: []T, strided: []T) !void {
    fillVectors(T, unit, strided);
    triangular.trmv(T, uplo, trans_, diag, @intCast(unit.len), a.ptr, @intCast(lda), unit.ptr, 1);
    triangular.trmv(T, uplo, trans_, diag, @intCast(unit.len), a.ptr, @intCast(lda), strided.ptr, 2);
    try checkResult(T, unit, strided);

    fillVectors(T, unit, strided);
    triangular.trsv(T, uplo, trans_, diag, @intCast(unit.len), a.ptr, @intCast(lda), unit.ptr, 1);
    triangular.trsv(T, uplo, trans_, diag, @intCast(unit.len), a.ptr, @intCast(lda), strided.ptr, 2);
    try checkResult(T, unit, strided);
}

fn checkTranspose(comptime T: type, trans_: Order, a: []const T, lda: usize, unit: []T, strided: []T) !void {
    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        inline for (.{ Diag.non_unit, Diag.unit }) |diag| {
            try checkCase(T, uplo, trans_, diag, a, lda, unit, strided);
        }
    }
}

fn checkType(comptime T: type) !void {
    const n = 129;
    const lda = n + 3;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const unit = try allocator.alloc(T, n);
    defer allocator.free(unit);
    const strided = try allocator.alloc(T, 2 * n);
    defer allocator.free(strided);

    fillMatrix(T, a, lda, n);
    try checkTranspose(T, .no_trans, a, lda, unit, strided);
    try checkTranspose(T, .trans, a, lda, unit, strided);
    if (comptime T == ComplexF32 or T == ComplexF64) {
        try checkTranspose(T, .conj_trans, a, lda, unit, strided);
    }
}

test "dense unit-stride TRMV and TRSV match the strided fallback" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkType(T);
}
