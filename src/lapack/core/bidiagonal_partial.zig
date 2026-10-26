// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Borrowed finite bidiagonal factors in one upper orientation, without an
//! allocator, execution admission, interface policy or hidden state.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
pub const Error = dense.Error || error{ DimensionMismatch, AliasingNotAllowed, NonFiniteInput, NonFiniteResult, NotBidiagonal, InvalidTolerance, RangeLimit };
fn re(comptime T: type, comptime R: type, v: T) R {
    return if (T == R) v else v.re;
}
fn im(comptime T: type, comptime R: type, v: T) R {
    return if (T == R) 0 else v.im;
}
fn realValue(comptime T: type, comptime R: type, v: R) T {
    return if (T == R) v else .{ .re = v, .im = 0 };
}
fn finite(comptime T: type, comptime R: type, v: T) bool {
    return std.math.isFinite(re(T, R, v)) and std.math.isFinite(im(T, R, v));
}
fn range(comptime T: type, values: []T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn rotate(comptime T: type, comptime R: type, a: *T, b: *T, c: R, s: R) Error!void {
    const ar = re(T, R, a.*);
    const ai = im(T, R, a.*);
    const br = re(T, R, b.*);
    const bi = im(T, R, b.*);
    const xr = @mulAdd(R, c, ar, s * br);
    const xi = @mulAdd(R, c, ai, s * bi);
    const yr = @mulAdd(R, -s, ar, c * br);
    const yi = @mulAdd(R, -s, ai, c * bi);
    if (!std.math.isFinite(xr) or !std.math.isFinite(xi) or !std.math.isFinite(yr) or !std.math.isFinite(yi)) return error.NonFiniteResult;
    a.* = if (T == R) xr else .{ .re = xr, .im = xi };
    b.* = if (T == R) yr else .{ .re = yr, .im = yi };
}
/// Input is A=L*B*R^H, or A=L*B^H*R^H when adjoint is true. B is stored
/// upper bidiagonal in p-by-k tall storage, in original input units. On return
/// adjoint is false and A=L*B_upper*R^H. Geometry, aliases and finite inputs
/// are checked before writes; arithmetic faults may leave borrowed work partial.
/// The caller owns the orientation flag and must retain it with these views.
/// A diagonal result is normalized and sorted, including its factor columns.
pub fn prepare(comptime T: type, comptime R: type, b: dense.Matrix(T), left: dense.Matrix(T), right: dense.Matrix(T), adjoint: *bool) Error!usize {
    const bs = try b.check();
    const ls = try left.check();
    const rs = try right.check();
    const m = left.row_count;
    const n = right.row_count;
    const k = @min(m, n);
    if (b.row_count != @max(m, n) or b.column_count != k) return error.DimensionMismatch;
    if (k == 0) {
        if (left.column_count != 0 or right.column_count != 0) return error.DimensionMismatch;
        adjoint.* = false;
        return 0;
    }
    if (left.column_count != m or right.column_count != n or (adjoint.* and m >= n)) return error.DimensionMismatch;
    const spans = [_]storage.ByteRange{ try range(T, b.values[0..bs]), try range(T, left.values[0..ls]), try range(T, right.values[0..rs]) };
    for (spans, 0..) |one, i| for (spans[i + 1 ..]) |other| {
        if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    };
    for (0..b.column_count) |j| for (0..b.row_count) |i| {
        const v = b.values[i + j * b.leading_dimension];
        if (!finite(T, R, v)) return error.NonFiniteInput;
        if (im(T, R, v) != 0 or ((i != j and j != i + 1) and re(T, R, v) != 0)) return error.NotBidiagonal;
    };
    inline for (.{ left, right }) |q| for (0..q.column_count) |j| for (0..q.row_count) |i| {
        if (!finite(T, R, q.values[i + j * q.leading_dimension])) return error.NonFiniteInput;
    };
    const ld = b.leading_dimension;
    if (adjoint.*) for (0..k - 1) |i| {
        const d = re(T, R, b.values[i + i * ld]);
        const e = re(T, R, b.values[i + (i + 1) * ld]);
        const next = re(T, R, b.values[i + 1 + (i + 1) * ld]);
        const radius = std.math.hypot(d, e);
        if (!std.math.isFinite(radius)) return error.NonFiniteResult;
        const c: R = if (radius == 0) 1 else d / radius;
        const s: R = if (radius == 0) 0 else e / radius;
        const next_diagonal = c * next;
        const next_edge = s * next;
        if (!std.math.isFinite(next_diagonal) or !std.math.isFinite(next_edge)) return error.NonFiniteResult;
        b.values[i + i * ld] = realValue(T, R, radius);
        b.values[i + (i + 1) * ld] = realValue(T, R, next_edge);
        b.values[i + 1 + (i + 1) * ld] = realValue(T, R, next_diagonal);
        for (0..m) |row| try rotate(T, R, &left.values[row + i * left.leading_dimension], &left.values[row + (i + 1) * left.leading_dimension], c, s);
    };
    var remaining: usize = 0;
    for (0..k - 1) |i| if (re(T, R, b.values[i + (i + 1) * ld]) != 0) {
        remaining += 1;
    };
    if (remaining == 0) {
        for (0..k) |i| if (re(T, R, b.values[i + i * ld]) < 0) {
            b.values[i + i * ld] = realValue(T, R, -re(T, R, b.values[i + i * ld]));
            for (0..m) |row| {
                const v = left.values[row + i * left.leading_dimension];
                left.values[row + i * left.leading_dimension] = if (T == R) -v else .{ .re = -v.re, .im = -v.im };
            }
        };
        for (0..k) |i| {
            var largest = i;
            for (i + 1..k) |j| if (re(T, R, b.values[j + j * ld]) > re(T, R, b.values[largest + largest * ld])) {
                largest = j;
            };
            if (largest == i) continue;
            std.mem.swap(T, &b.values[i + i * ld], &b.values[largest + largest * ld]);
            for (0..m) |row| std.mem.swap(T, &left.values[row + i * left.leading_dimension], &left.values[row + largest * left.leading_dimension]);
            for (0..n) |row| std.mem.swap(T, &right.values[row + i * right.leading_dimension], &right.values[row + largest * right.leading_dimension]);
        }
    }
    adjoint.* = false;
    return remaining;
}

fn rounded(comptime N: type, comptime R: type, v: R) R {
    return @as(R, @as(N, @floatCast(v)));
}
fn checkRoundedBasis(comptime N: type, comptime T: type, comptime R: type, q: dense.Matrix(T), columns: usize, tolerance: R) Error!void {
    for (0..columns) |j| for (0..q.row_count) |i| {
        const v = q.values[i + j * q.leading_dimension];
        if (!finite(T, R, v) or !std.math.isFinite(rounded(N, R, re(T, R, v))) or !std.math.isFinite(rounded(N, R, im(T, R, v)))) return error.NonFiniteResult;
    };
    for (0..columns) |p| for (p..columns) |j| {
        var real: R = 0;
        var imaginary: R = 0;
        for (0..q.row_count) |i| {
            const x = q.values[i + p * q.leading_dimension];
            const y = q.values[i + j * q.leading_dimension];
            const xr = rounded(N, R, re(T, R, x));
            const xi = rounded(N, R, im(T, R, x));
            const yr = rounded(N, R, re(T, R, y));
            const yi = rounded(N, R, im(T, R, y));
            real += xr * yr + xi * yi;
            imaginary += xr * yi - xi * yr;
        }
        const error_norm = std.math.hypot(real - @as(R, if (p == j) 1 else 0), imaginary);
        if (!std.math.isFinite(error_norm) or error_norm > tolerance) return error.RangeLimit;
    };
}
/// Check the entire partial diagonal/edge payload and the requested factor
/// columns after rounding to the caller's real component type. No writes.
/// An edge that would round to zero is rejected so INFO can count the actual
/// returned nonzero edges without silently changing the partial factorization.
pub fn checkNative(comptime N: type, comptime T: type, comptime R: type, b: dense.Matrix(T), left: dense.Matrix(T), right: dense.Matrix(T), left_columns: usize, right_columns: usize, tolerance: f64) Error!void {
    _ = try b.check();
    _ = try left.check();
    _ = try right.check();
    const k = @min(left.row_count, right.row_count);
    if (b.row_count != @max(left.row_count, right.row_count) or b.column_count != k or left_columns > left.column_count or right_columns > right.column_count) return error.DimensionMismatch;
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    for (0..b.column_count) |j| for (0..b.row_count) |i| {
        const v = b.values[i + j * b.leading_dimension];
        if (!finite(T, R, v)) return error.NonFiniteResult;
        const value = re(T, R, v);
        if (im(T, R, v) != 0 or ((i != j and j != i + 1) and value != 0)) return error.NotBidiagonal;
        const native = rounded(N, R, value);
        if (!std.math.isFinite(native)) return error.NonFiniteResult;
        if (j == i + 1 and value != 0 and native == 0) return error.RangeLimit;
    };
    try checkRoundedBasis(N, T, R, left, left_columns, @floatCast(tolerance));
    try checkRoundedBasis(N, T, R, right, right_columns, @floatCast(tolerance));
}
