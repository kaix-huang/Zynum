// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const matrix_vector = @import("core/matrix_vector.zig");
const packed_unit = @import("core/matrix_vector/triangular_packed_unit.zig");
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
    if (row == column) return value(T, 1.5 + re_seed / 512.0, im_seed / 640.0);
    return value(T, re_seed / 4096.0, im_seed / 5120.0);
}

fn vectorValue(comptime T: type, index: usize) T {
    const re_seed = @as(f64, @floatFromInt((index * 13 + 7) % 41)) - 20.0;
    const im_seed = @as(f64, @floatFromInt((index * 5 + 9) % 37)) - 18.0;
    return value(T, re_seed / 23.0, im_seed / 29.0);
}

fn fillPacked(comptime T: type, ap: []T, uplo: Uplo, diag: Diag, n: usize) void {
    @memset(ap, nanValue(T));
    var packed_index: usize = 0;
    for (0..n) |column| {
        if (uplo == .upper) {
            for (0..column + 1) |row| {
                if (diag == .non_unit or row != column) ap[packed_index] = matrixValue(T, row, column);
                packed_index += 1;
            }
        } else {
            for (column..n) |row| {
                if (diag == .non_unit or row != column) ap[packed_index] = matrixValue(T, row, column);
                packed_index += 1;
            }
        }
    }
}

fn unpackDense(comptime T: type, dense: []T, ap: []const T, uplo: Uplo, diag: Diag, n: usize) void {
    @memset(dense, scalar.zero(T));
    var packed_index: usize = 0;
    for (0..n) |column| {
        if (uplo == .upper) {
            for (0..column + 1) |row| {
                dense[row + column * n] = if (diag == .unit and row == column) scalar.one(T) else ap[packed_index];
                packed_index += 1;
            }
        } else {
            for (column..n) |row| {
                dense[row + column * n] = if (diag == .unit and row == column) scalar.one(T) else ap[packed_index];
                packed_index += 1;
            }
        }
    }
}

fn opValue(comptime T: type, dense: []const T, trans_: Order, n: usize, row: usize, column: usize) T {
    const stored_row = if (trans_ == .no_trans) row else column;
    const stored_column = if (trans_ == .no_trans) column else row;
    const entry = dense[stored_row + stored_column * n];
    return if (trans_ == .conj_trans) scalar.conj(T, entry) else entry;
}

// TPMV is referenced out of place through a dense matrix, independently of
// the packed-column in-place traversal used by the production leaf.
fn referenceTpmv(comptime T: type, dense: []const T, trans_: Order, n: usize, input: []const T, output: []T) void {
    for (0..n) |row| {
        var sum = scalar.zero(T);
        for (0..n) |column| {
            sum = scalar.add(T, sum, scalar.mul(T, opValue(T, dense, trans_, n, row, column), input[column]));
        }
        output[row] = sum;
    }
}

// TPSV uses row substitution over the dense operator. It neither shares packed
// offsets nor the column-update solve order used by no-transpose production.
fn referenceTpsv(comptime T: type, dense: []const T, uplo: Uplo, trans_: Order, diag: Diag, n: usize, rhs: []const T, output: []T) void {
    @memcpy(output, rhs);
    const op_is_upper = (uplo == .upper and trans_ == .no_trans) or (uplo == .lower and trans_ != .no_trans);
    if (op_is_upper) {
        var row = n;
        while (row > 0) {
            row -= 1;
            var solved = output[row];
            for (row + 1..n) |column| {
                solved = scalar.sub(T, solved, scalar.mul(T, opValue(T, dense, trans_, n, row, column), output[column]));
            }
            if (diag == .non_unit) solved = scalar.divv(T, solved, opValue(T, dense, trans_, n, row, row));
            output[row] = solved;
        }
    } else {
        for (0..n) |row| {
            var solved = output[row];
            for (0..row) |column| {
                solved = scalar.sub(T, solved, scalar.mul(T, opValue(T, dense, trans_, n, row, column), output[column]));
            }
            if (diag == .non_unit) solved = scalar.divv(T, solved, opValue(T, dense, trans_, n, row, row));
            output[row] = solved;
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 2e-3) else @as(f64, 5e-11);
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

fn expectPaddingNan(comptime T: type, storage: []const T, active_start: usize, active_len: usize) !void {
    for (storage[0..active_start]) |entry| try expectNan(T, entry);
    for (storage[active_start + active_len ..]) |entry| try expectNan(T, entry);
}

fn invokeTpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: []const T, input: []const T, x: []T) !void {
    if (comptime builtin.cpu.arch == .x86_64) {
        try std.testing.expect(packed_unit.tryTpmv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
    } else {
        try std.testing.expect(!packed_unit.tryTpmv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input), std.mem.sliceAsBytes(x));
        packed_unit.testing.runTpmvUnit(T, uplo, trans_, diag, n, ap.ptr, x.ptr);
    }
}

fn invokeTpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: []const T, input: []const T, x: []T) !void {
    if (comptime builtin.cpu.arch == .x86_64) {
        try std.testing.expect(packed_unit.tryTpsv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
    } else {
        try std.testing.expect(!packed_unit.tryTpsv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input), std.mem.sliceAsBytes(x));
        packed_unit.testing.runTpsvUnit(T, uplo, trans_, diag, n, ap.ptr, x.ptr);
    }
}

fn runCase(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, check_facade: bool) !void {
    const n = 131;
    const packed_len = n * (n + 1) / 2;
    const guard = 4;
    const allocator = std.testing.allocator;

    const ap_storage = try allocator.alloc(T, packed_len + 2 * guard);
    defer allocator.free(ap_storage);
    const ap = ap_storage[guard .. guard + packed_len];
    @memset(ap_storage, nanValue(T));
    fillPacked(T, ap, uplo, diag, n);
    const ap_before = try allocator.dupe(T, ap_storage);
    defer allocator.free(ap_before);

    const dense = try allocator.alloc(T, n * n);
    defer allocator.free(dense);
    unpackDense(T, dense, ap, uplo, diag, n);

    const input = try allocator.alloc(T, n);
    defer allocator.free(input);
    for (input, 0..) |*entry, i| entry.* = vectorValue(T, i);

    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);
    const x_storage = try allocator.alloc(T, n + 2 * guard);
    defer allocator.free(x_storage);
    const x = x_storage[guard .. guard + n];

    referenceTpmv(T, dense, trans_, n, input, expected);
    @memset(x_storage, nanValue(T));
    @memcpy(x, input);
    try invokeTpmv(T, uplo, trans_, diag, n, ap, input, x);
    for (expected, x) |want, got| try expectApprox(T, want, got);
    try expectPaddingNan(T, x_storage, guard, n);
    if (check_facade) {
        @memset(x_storage, nanValue(T));
        @memcpy(x, input);
        matrix_vector.tpmv(T, uplo, trans_, diag, @intCast(n), ap.ptr, x.ptr, 1);
        for (expected, x) |want, got| try expectApprox(T, want, got);
        try expectPaddingNan(T, x_storage, guard, n);
    }

    referenceTpsv(T, dense, uplo, trans_, diag, n, input, expected);
    @memset(x_storage, nanValue(T));
    @memcpy(x, input);
    try invokeTpsv(T, uplo, trans_, diag, n, ap, input, x);
    for (expected, x) |want, got| try expectApprox(T, want, got);
    try expectPaddingNan(T, x_storage, guard, n);
    if (check_facade) {
        @memset(x_storage, nanValue(T));
        @memcpy(x, input);
        matrix_vector.tpsv(T, uplo, trans_, diag, @intCast(n), ap.ptr, x.ptr, 1);
        for (expected, x) |want, got| try expectApprox(T, want, got);
        try expectPaddingNan(T, x_storage, guard, n);
    }

    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ap_before), std.mem.sliceAsBytes(ap_storage));
    try expectPaddingNan(T, ap_storage, guard, packed_len);
}

fn runAllCases(comptime T: type) !void {
    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        inline for (.{ Order.no_trans, Order.trans, Order.conj_trans }) |trans_| {
            inline for (.{ Diag.non_unit, Diag.unit }) |diag| try runCase(T, uplo, trans_, diag, false);
        }
    }
}

fn expectGateMissUnchanged(comptime T: type, n: usize, incx: scalar.BlasInt) !void {
    const allocator = std.testing.allocator;
    const packed_len = n * (n + 1) / 2;
    const ap = try allocator.alloc(T, packed_len);
    defer allocator.free(ap);
    for (ap, 0..) |*entry, i| entry.* = matrixValue(T, i % n, i % n);

    const x = try allocator.alloc(T, 2 * n + 8);
    defer allocator.free(x);
    for (x, 0..) |*entry, i| entry.* = vectorValue(T, i);
    const before = try allocator.dupe(T, x);
    defer allocator.free(before);

    try std.testing.expect(!packed_unit.tryTpmv(T, .upper, .no_trans, .non_unit, n, ap.ptr, x.ptr, incx));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(x));
    try std.testing.expect(!packed_unit.tryTpsv(T, .lower, .conj_trans, .unit, n, ap.ptr, x.ptr, incx));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(x));
}

test "packed-column unit TPMV and TPSV match independent references" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try runAllCases(T);
}

test "matrix-vector facade routes compact TPMV and TPSV" {
    try runCase(ComplexF64, .lower, .conj_trans, .unit, true);
}

test "packed-column production gates fail without modifying x" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| {
        try std.testing.expectEqual(builtin.cpu.arch == .x86_64, packed_unit.testing.gateAllows(T, 128, 1));
        try std.testing.expect(!packed_unit.testing.gateAllows(T, 127, 1));
        try std.testing.expect(!packed_unit.testing.gateAllows(T, 128, 2));
        try expectGateMissUnchanged(T, 127, 1);
        try expectGateMissUnchanged(T, 128, 2);
    }
}
