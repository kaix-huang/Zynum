// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed complex tall reduction A=U*B*V^H with real upper bidiagonal B.
//! Complex unitary rotations and phases feed the existing real bidiagonal QR.
//! No allocation, execution admission, backend, external ABI or state owner.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const cx = @import("../../numerics/complex.zig");
const real = @import("bidiagonal.zig");
pub const NumericalPolicy = real.NumericalPolicy;
pub const IterationOptions = real.IterationOptions;
pub const IterationResult = real.IterationResult;
pub const Error = real.Error || error{NotBidiagonal};
pub const Wide = cx.Wide;
pub const WorkspaceScalar = cx.WideComplex;

fn range(comptime T: type, values: []T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn mul(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conjugate(comptime C: type, a: WorkspaceScalar(C)) WorkspaceScalar(C) {
    return .{ .re = a.re, .im = -a.im };
}
fn identity(comptime C: type, a: dense.Matrix(WorkspaceScalar(C))) void {
    for (0..a.column_count) |j| for (0..a.row_count) |i| {
        a.values[i + j * a.leading_dimension] = .{ .re = if (i == j) 1 else 0, .im = 0 };
    };
}
fn validate(comptime C: type, b: dense.Matrix(WorkspaceScalar(C)), u: dense.Matrix(WorkspaceScalar(C)), v: dense.Matrix(WorkspaceScalar(C)), policy: NumericalPolicy) Error![3]storage.ByteRange {
    const Z = WorkspaceScalar(C);
    const bs = try b.check();
    const us = try u.check();
    const vs = try v.check();
    const m = b.row_count;
    const n = b.column_count;
    if (m < n or u.row_count != m or u.column_count != m or v.row_count != n or v.column_count != n) return error.DimensionMismatch;
    const ranges = [_]storage.ByteRange{ try range(Z, b.values[0..bs]), try range(Z, u.values[0..us]), try range(Z, v.values[0..vs]) };
    for (ranges, 0..) |one, i| for (ranges[i + 1 ..]) |other| {
        if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    };
    if (policy == .finite) for (0..n) |j| for (0..m) |i| {
        if (!cx.finite(Z, b.values[i + j * b.leading_dimension])) return error.NonFiniteInput;
    };
    return ranges;
}
fn finite(comptime C: type, value: WorkspaceScalar(C), policy: NumericalPolicy) Error!WorkspaceScalar(C) {
    if (policy == .finite and !cx.finite(WorkspaceScalar(C), value)) return error.NonFiniteResult;
    return value;
}
fn phase(comptime C: type, value: WorkspaceScalar(C)) WorkspaceScalar(C) {
    const scale = @max(@abs(value.re), @abs(value.im));
    if (scale == 0) return .{ .re = 1, .im = 0 };
    const r = value.re / scale;
    const i = value.im / scale;
    const length = std.math.hypot(r, i);
    return .{ .re = r / length, .im = i / length };
}

/// G*[x,y]^T=[radius,0]^T for nonnegative real x; G=[c,s;-conj(s),c].
fn rotation(comptime C: type, x: Wide(C), y: WorkspaceScalar(C)) struct { c: Wide(C), s: WorkspaceScalar(C), radius: Wide(C) } {
    const scale = @max(@abs(x), @max(@abs(y.re), @abs(y.im)));
    if (scale == 0) return .{ .c = 1, .s = .{ .re = 0, .im = 0 }, .radius = 0 };
    const xx = x / scale;
    const yr = y.re / scale;
    const yi = y.im / scale;
    const length = std.math.hypot(xx, std.math.hypot(yr, yi));
    return .{ .c = xx / length, .s = .{ .re = yr / length, .im = -yi / length }, .radius = scale * length };
}
fn leftPair(comptime C: type, x: *WorkspaceScalar(C), y: *WorkspaceScalar(C), c: Wide(C), s: WorkspaceScalar(C), policy: NumericalPolicy) Error!void {
    const a = x.*;
    const b = y.*;
    const sb = mul(C, s, b);
    const sa = mul(C, conjugate(C, s), a);
    x.* = try finite(C, .{ .re = c * a.re + sb.re, .im = c * a.im + sb.im }, policy);
    y.* = try finite(C, .{ .re = c * b.re - sa.re, .im = c * b.im - sa.im }, policy);
}
/// Right multiplication by G^H accumulates the inverse left rotation.
fn rightPair(comptime C: type, x: *WorkspaceScalar(C), y: *WorkspaceScalar(C), c: Wide(C), s: WorkspaceScalar(C), policy: NumericalPolicy) Error!void {
    const a = x.*;
    const b = y.*;
    const sb = mul(C, conjugate(C, s), b);
    const sa = mul(C, s, a);
    x.* = try finite(C, .{ .re = c * a.re + sb.re, .im = c * a.im + sb.im }, policy);
    y.* = try finite(C, .{ .re = c * b.re - sa.re, .im = c * b.im - sa.im }, policy);
}

/// B receives the real upper bidiagonal form, U/V receive full unitary factors.
/// Validate geometry, disjoint spans and finite B before any write. Arithmetic
/// range errors may expose partial factors. Padding and unused tails are kept.
pub fn reduce(comptime C: type, b: dense.Matrix(WorkspaceScalar(C)), u: dense.Matrix(WorkspaceScalar(C)), v: dense.Matrix(WorkspaceScalar(C))) Error!void {
    return reduceWithPolicy(C, b, u, v, .finite);
}
pub fn reduceWithPolicy(comptime C: type, b: dense.Matrix(WorkspaceScalar(C)), u: dense.Matrix(WorkspaceScalar(C)), v: dense.Matrix(WorkspaceScalar(C)), policy: NumericalPolicy) Error!void {
    _ = try validate(C, b, u, v, policy);
    const m = b.row_count;
    const n = b.column_count;
    identity(C, u);
    identity(C, v);
    for (0..n) |k| {
        const unit = phase(C, b.values[k + k * b.leading_dimension]);
        for (k..n) |j| b.values[k + j * b.leading_dimension] = try finite(C, mul(C, conjugate(C, unit), b.values[k + j * b.leading_dimension]), policy);
        for (0..m) |i| u.values[i + k * u.leading_dimension] = try finite(C, mul(C, u.values[i + k * u.leading_dimension], unit), policy);
        b.values[k + k * b.leading_dimension].im = 0;
        var row = m;
        while (row > k + 1) {
            row -= 1;
            const y = b.values[row + k * b.leading_dimension];
            if (y.re == 0 and y.im == 0) continue;
            const g = rotation(C, b.values[k + k * b.leading_dimension].re, y);
            if (policy == .finite and !std.math.isFinite(g.radius)) return error.NonFiniteResult;
            for (k + 1..n) |j| try leftPair(C, &b.values[k + j * b.leading_dimension], &b.values[row + j * b.leading_dimension], g.c, g.s, policy);
            for (0..m) |i| try rightPair(C, &u.values[i + k * u.leading_dimension], &u.values[i + row * u.leading_dimension], g.c, g.s, policy);
            b.values[k + k * b.leading_dimension] = .{ .re = g.radius, .im = 0 };
            b.values[row + k * b.leading_dimension] = .{ .re = 0, .im = 0 };
        }
        if (k + 1 >= n) continue;
        const p = k + 1;
        const row_unit = conjugate(C, phase(C, b.values[k + p * b.leading_dimension]));
        for (k..m) |i| b.values[i + p * b.leading_dimension] = try finite(C, mul(C, b.values[i + p * b.leading_dimension], row_unit), policy);
        for (0..n) |i| v.values[i + p * v.leading_dimension] = try finite(C, mul(C, v.values[i + p * v.leading_dimension], row_unit), policy);
        b.values[k + p * b.leading_dimension].im = 0;
        var col = n;
        while (col > k + 2) {
            col -= 1;
            const y = b.values[k + col * b.leading_dimension];
            if (y.re == 0 and y.im == 0) continue;
            const g = rotation(C, b.values[k + p * b.leading_dimension].re, conjugate(C, y));
            if (policy == .finite and !std.math.isFinite(g.radius)) return error.NonFiniteResult;
            for (k + 1..m) |i| try rightPair(C, &b.values[i + p * b.leading_dimension], &b.values[i + col * b.leading_dimension], g.c, g.s, policy);
            for (0..n) |i| try rightPair(C, &v.values[i + p * v.leading_dimension], &v.values[i + col * v.leading_dimension], g.c, g.s, policy);
            b.values[k + p * b.leading_dimension] = .{ .re = g.radius, .im = 0 };
            b.values[k + col * b.leading_dimension] = .{ .re = 0, .im = 0 };
        }
    }
}

pub fn realWorkspaceLength(m: usize, n: usize) Error!usize {
    if (m < n) return error.DimensionMismatch;
    const b = std.math.mul(usize, m, n) catch return error.SizeOverflow;
    const u = std.math.mul(usize, m, m) catch return error.SizeOverflow;
    const v = std.math.mul(usize, n, n) catch return error.SizeOverflow;
    return std.math.add(usize, std.math.add(usize, b, u) catch return error.SizeOverflow, v) catch error.SizeOverflow;
}
fn accumulate(comptime C: type, complex: dense.Matrix(WorkspaceScalar(C)), factor: dense.Matrix(Wide(C)), row: []WorkspaceScalar(C), policy: NumericalPolicy) Error!void {
    for (0..complex.row_count) |i| {
        for (0..complex.column_count) |j| row[j] = complex.values[i + j * complex.leading_dimension];
        for (0..complex.column_count) |j| {
            var sum: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            for (0..complex.column_count) |k| {
                const weight = factor.values[k + j * factor.leading_dimension];
                sum.re += row[k].re * weight;
                sum.im += row[k].im * weight;
            }
            complex.values[i + j * complex.leading_dimension] = try finite(C, sum, policy);
        }
    }
}

/// QR uses the same real bidiagonal owner. On exhaustion B remains bidiagonal,
/// A=U*B*V^H remains valid, and unconverged counts its actual active edges.
/// The real arena holds B and real rotation factors; row scratch has m complex
/// elements. All active spans, including mixed scalar spans, must be disjoint.
pub fn diagonalize(comptime C: type, b: dense.Matrix(WorkspaceScalar(C)), u: dense.Matrix(WorkspaceScalar(C)), v: dense.Matrix(WorkspaceScalar(C)), workspace: []Wide(C), row: []WorkspaceScalar(C), options: IterationOptions) Error!IterationResult {
    const ranges = try validate(C, b, u, v, options.numerical_policy);
    const m = b.row_count;
    const n = b.column_count;
    const count = try realWorkspaceLength(m, n);
    if (workspace.len < count or row.len < m) return error.BufferTooSmall;
    const rr = try range(Wide(C), workspace[0..count]);
    const cr = try range(WorkspaceScalar(C), row[0..m]);
    if (storage.overlaps(rr, cr)) return error.AliasingNotAllowed;
    for (ranges) |one| if (storage.overlaps(one, rr) or storage.overlaps(one, cr)) return error.AliasingNotAllowed;
    if (options.numerical_policy == .finite) inline for (.{ u, v }) |factor| for (0..factor.column_count) |j| for (0..factor.row_count) |i| {
        if (!cx.finite(WorkspaceScalar(C), factor.values[i + j * factor.leading_dimension])) return error.NonFiniteInput;
    };
    for (0..n) |j| for (0..m) |i| {
        const z = b.values[i + j * b.leading_dimension];
        if (z.im != 0 or (i != j and j != i + 1 and z.re != 0)) return error.NotBidiagonal;
    };
    const rb: dense.Matrix(Wide(C)) = .{ .values = workspace[0 .. m * n], .row_count = m, .column_count = n, .leading_dimension = m };
    const ru: dense.Matrix(Wide(C)) = .{ .values = workspace[m * n ..][0 .. m * m], .row_count = m, .column_count = m, .leading_dimension = m };
    const rv: dense.Matrix(Wide(C)) = .{ .values = workspace[m * n + m * m ..][0 .. n * n], .row_count = n, .column_count = n, .leading_dimension = n };
    for (0..n) |j| for (0..m) |i| {
        rb.values[i + j * m] = b.values[i + j * b.leading_dimension].re;
    };
    for (0..m) |j| for (0..m) |i| {
        ru.values[i + j * m] = if (i == j) 1 else 0;
    };
    for (0..n) |j| for (0..n) |i| {
        rv.values[i + j * n] = if (i == j) 1 else 0;
    };
    const result = try real.diagonalize(Wide(C), rb, ru, rv, options);
    try accumulate(C, u, ru, row, options.numerical_policy);
    try accumulate(C, v, rv, row, options.numerical_policy);
    for (0..n) |j| for (0..m) |i| {
        b.values[i + j * b.leading_dimension] = .{ .re = rb.values[i + j * m], .im = 0 };
    };
    return result;
}
