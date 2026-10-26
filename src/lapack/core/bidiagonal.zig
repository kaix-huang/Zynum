// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Unblocked real upper-bidiagonal reduction with explicit orthogonal factors.
//! A = U * B * V^T, m>=n. All storage is borrowed; no runtime or allocation.
//! Widened callers may use f64/f128. This is a correctness foundation.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
pub const NumericalPolicy = enum { finite, ieee };
pub const Error = dense.Error || error{ DimensionMismatch, AliasingNotAllowed, NonFiniteInput, NonFiniteResult };
fn range(comptime T: type, data: []T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(data.ptr), try storage.byteLength(data.len, @sizeOf(T)));
}
fn identity(comptime T: type, a: dense.Matrix(T)) void {
    for (0..a.column_count) |j| for (0..a.row_count) |i| {
        a.values[i + j * a.leading_dimension] = if (i == j) 1 else 0;
    };
}
fn rotate(comptime T: type, x: *T, y: *T, c: T, s: T, policy: NumericalPolicy) Error!void {
    const old_x = x.*;
    const old_y = y.*;
    const next_x = @mulAdd(T, c, old_x, s * old_y);
    const next_y = @mulAdd(T, -s, old_x, c * old_y);
    if (policy == .finite and (!std.math.isFinite(next_x) or !std.math.isFinite(next_y))) return error.NonFiniteResult;
    x.* = next_x;
    y.* = next_y;
}
fn columns(comptime T: type, a: dense.Matrix(T), p: usize, q: usize, c: T, s: T, policy: NumericalPolicy) Error!void {
    for (0..a.row_count) |i| try rotate(T, &a.values[i + p * a.leading_dimension], &a.values[i + q * a.leading_dimension], c, s, policy);
}

/// B initially holds A, then its upper bidiagonal form. U is m*m and V is n*n.
/// Geometry, aliasing and finite input are validated before mutation. Arithmetic
/// range failure may leave partial work in B/U/V. Padding/suffixes are untouched.
/// Caller chooses widened scratch to keep intermediate norms representable.
pub fn reduce(comptime T: type, b: dense.Matrix(T), u: dense.Matrix(T), v: dense.Matrix(T)) Error!void {
    return reduceWithPolicy(T, b, u, v, .finite);
}

/// IEEE propagation keeps geometry/alias checks and bounded loop structure.
/// Orthogonality/reconstruction promises apply only to finite arithmetic.
pub fn reduceWithPolicy(comptime T: type, b: dense.Matrix(T), u: dense.Matrix(T), v: dense.Matrix(T), policy: NumericalPolicy) Error!void {
    if (T != f32 and T != f64 and T != f128) @compileError("bidiagonal reduction requires a real floating type");
    const bs = try b.check();
    const us = try u.check();
    const vs = try v.check();
    const m = b.row_count;
    const n = b.column_count;
    if (m < n or u.row_count != m or u.column_count != m or v.row_count != n or v.column_count != n) return error.DimensionMismatch;
    const ranges = [_]storage.ByteRange{ try range(T, b.values[0..bs]), try range(T, u.values[0..us]), try range(T, v.values[0..vs]) };
    for (ranges, 0..) |a, i| for (ranges[i + 1 ..]) |other| {
        if (storage.overlaps(a, other)) return error.AliasingNotAllowed;
    };
    if (policy == .finite) for (0..n) |j| for (0..m) |i| {
        if (!std.math.isFinite(b.values[i + j * b.leading_dimension])) return error.NonFiniteInput;
    };
    identity(T, u);
    identity(T, v);
    for (0..n) |k| {
        var row = m;
        while (row > k + 1) {
            row -= 1;
            const x = b.values[k + k * b.leading_dimension];
            const y = b.values[row + k * b.leading_dimension];
            if (y == 0) continue;
            const radius = std.math.hypot(x, y);
            if (policy == .finite and !std.math.isFinite(radius)) return error.NonFiniteResult;
            const c = x / radius;
            const s = y / radius;
            for (k + 1..n) |j| try rotate(T, &b.values[k + j * b.leading_dimension], &b.values[row + j * b.leading_dimension], c, s, policy);
            b.values[k + k * b.leading_dimension] = radius;
            b.values[row + k * b.leading_dimension] = 0;
            try columns(T, u, k, row, c, s, policy);
        }
        if (k + 1 >= n) continue;
        var col = n;
        while (col > k + 2) {
            col -= 1;
            const p = k + 1;
            const x = b.values[k + p * b.leading_dimension];
            const y = b.values[k + col * b.leading_dimension];
            if (y == 0) continue;
            const radius = std.math.hypot(x, y);
            if (policy == .finite and !std.math.isFinite(radius)) return error.NonFiniteResult;
            const c = x / radius;
            const s = y / radius;
            for (k + 1..m) |i| try rotate(T, &b.values[i + p * b.leading_dimension], &b.values[i + col * b.leading_dimension], c, s, policy);
            b.values[k + p * b.leading_dimension] = radius;
            b.values[k + col * b.leading_dimension] = 0;
            try columns(T, v, p, col, c, s, policy);
        }
    }
}

pub const IterationOptions = struct { max_steps: usize = 4096, numerical_policy: NumericalPolicy = .finite };
pub const IterationResult = struct { converged: bool, steps: usize, unconverged: usize };
fn rows(comptime T: type, a: dense.Matrix(T), p: usize, q: usize, c: T, s: T, policy: NumericalPolicy) Error!void {
    for (0..a.column_count) |j| try rotate(T, &a.values[p + j * a.leading_dimension], &a.values[q + j * a.leading_dimension], c, s, policy);
}
fn givens(comptime T: type, x: T, y: T) struct { c: T, s: T } {
    const scale = @max(@abs(x), @abs(y));
    if (scale == 0) return .{ .c = 1, .s = 0 };
    const xx = x / scale;
    const yy = y / scale;
    const radius = std.math.hypot(xx, yy);
    return .{ .c = xx / radius, .s = yy / radius };
}

/// Implicit shifted QR on an upper bidiagonal B, accumulating U and V from reduce.
/// On budget exhaustion B remains bidiagonal and A=U*B*V^T still holds (up to
/// rounding/deflation). unconverged counts its actual nonzero superdiagonal edges.
/// On convergence the diagonal is nonnegative and descending. All operands are
/// borrowed mutable scratch; arithmetic range errors may expose partial work.
pub fn diagonalize(comptime T: type, b: dense.Matrix(T), u: dense.Matrix(T), v: dense.Matrix(T), options: IterationOptions) (Error || error{NotBidiagonal})!IterationResult {
    const policy = options.numerical_policy;
    const bs = try b.check();
    const us = try u.check();
    const vs = try v.check();
    const m = b.row_count;
    const n = b.column_count;
    if (m < n or u.row_count != m or u.column_count != m or v.row_count != n or v.column_count != n) return error.DimensionMismatch;
    const ranges = [_]storage.ByteRange{ try range(T, b.values[0..bs]), try range(T, u.values[0..us]), try range(T, v.values[0..vs]) };
    for (ranges, 0..) |a, i| for (ranges[i + 1 ..]) |other| {
        if (storage.overlaps(a, other)) return error.AliasingNotAllowed;
    };
    if (policy == .finite) inline for (.{ b, u, v }) |a| for (0..a.column_count) |j| for (0..a.row_count) |i| {
        if (!std.math.isFinite(a.values[i + j * a.leading_dimension])) return error.NonFiniteInput;
    };
    for (0..n) |j| for (0..m) |i| {
        if (i != j and j != i + 1 and b.values[i + j * b.leading_dimension] != 0) return error.NotBidiagonal;
    };
    const ld = b.leading_dimension;
    var steps: usize = 0;
    while (true) {
        var count: usize = 0;
        if (n > 1) for (0..n - 1) |i| {
            const edge = &b.values[i + (i + 1) * ld];
            const scale = @max(@abs(b.values[i + i * ld]), @abs(b.values[i + 1 + (i + 1) * ld]));
            if (scale != 0 and @abs(edge.*) / scale <= std.math.floatEps(T)) edge.* = 0;
            if (edge.* != 0) count += 1;
        };
        if (count == 0) break;
        if (steps == options.max_steps) return .{ .converged = false, .steps = steps, .unconverged = count };
        var hi = n - 1;
        while (b.values[hi - 1 + hi * ld] == 0) hi -= 1;
        var lo = hi - 1;
        while (lo > 0 and b.values[lo - 1 + lo * ld] != 0) lo -= 1;
        var scale: T = 0;
        var zero_diagonal = false;
        for (lo..hi + 1) |i| {
            const diagonal = b.values[i + i * ld];
            scale = @max(scale, @abs(diagonal));
            zero_diagonal = zero_diagonal or diagonal == 0;
            if (i < hi) scale = @max(scale, @abs(b.values[i + (i + 1) * ld]));
        }
        const d0 = b.values[lo + lo * ld] / scale;
        const e0 = b.values[lo + (lo + 1) * ld] / scale;
        var f = d0;
        var g = e0;
        if (!zero_diagonal) {
            const dh = b.values[hi + hi * ld] / scale;
            const dp = b.values[hi - 1 + (hi - 1) * ld] / scale;
            const ep = b.values[hi - 1 + hi * ld] / scale;
            const earlier = if (hi - 1 > lo) b.values[hi - 2 + (hi - 1) * ld] / scale else 0;
            const aa = dp * dp + earlier * earlier;
            const cc = dh * dh + ep * ep;
            const cross = dp * ep;
            const delta = (aa - cc) * 0.5;
            const denominator = @abs(delta) + std.math.hypot(delta, cross);
            const shift = @max(@as(T, 0), cc - (if (denominator == 0) 0 else std.math.copysign(cross * cross / denominator, delta)));
            f = d0 * d0 - shift;
            g = d0 * e0;
            // A vanished starting vector admits an unshifted step.
            if (f == 0 and g == 0) {
                f = d0;
                g = e0;
            }
        }
        for (lo..hi) |k| {
            const right_rotation = givens(T, f, g);
            try columns(T, b, k, k + 1, right_rotation.c, right_rotation.s, policy);
            try columns(T, v, k, k + 1, right_rotation.c, right_rotation.s, policy);
            if (k > lo) b.values[k - 1 + (k + 1) * ld] = 0;
            const left_rotation = givens(T, b.values[k + k * ld], b.values[k + 1 + k * ld]);
            try rows(T, b, k, k + 1, left_rotation.c, left_rotation.s, policy);
            try columns(T, u, k, k + 1, left_rotation.c, left_rotation.s, policy);
            b.values[k + 1 + k * ld] = 0;
            if (k + 1 < hi) {
                f = b.values[k + (k + 1) * ld];
                g = b.values[k + (k + 2) * ld];
            }
        }
        steps += 1;
    }
    for (0..n) |i| {
        if (b.values[i + i * ld] < 0) {
            b.values[i + i * ld] = -b.values[i + i * ld];
            for (0..m) |r| u.values[r + i * u.leading_dimension] = -u.values[r + i * u.leading_dimension];
        }
    }
    for (0..n) |i| {
        var largest = i;
        for (i + 1..n) |j| if (b.values[j + j * ld] > b.values[largest + largest * ld]) {
            largest = j;
        };
        if (largest == i) continue;
        std.mem.swap(T, &b.values[i + i * ld], &b.values[largest + largest * ld]);
        for (0..m) |r| std.mem.swap(T, &u.values[r + i * u.leading_dimension], &u.values[r + largest * u.leading_dimension]);
        for (0..n) |r| std.mem.swap(T, &v.values[r + i * v.leading_dimension], &v.values[r + largest * v.leading_dimension]);
    }
    return .{ .converged = true, .steps = steps, .unconverged = 0 };
}
