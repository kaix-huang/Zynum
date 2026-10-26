// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed complex minimum-norm reconstruction A=U*S*V^H.
//! No admission, allocation, configured owner or external numerical backend.
const std = @import("std");
const cx = @import("../../numerics/complex.zig");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
pub const WorkspaceScalar = cx.WideComplex;
pub const Real = cx.scalar.Real;
pub const Matrix = dense.Matrix;
pub const ConstMatrix = dense.ConstMatrix;
pub const Error = dense.Error || error{ DimensionMismatch, InvalidSingularValues, InvalidTolerance, NonFiniteInput, AliasingNotAllowed };
pub const Result = struct { rank: usize, status: enum { success, non_finite_result } };

pub fn workspaceLength(comptime C: type, columns: usize, rhs: usize) Error!usize {
    comptime cx.expectComplex(C);
    const count = std.math.mul(usize, columns, rhs) catch return error.SizeOverflow;
    _ = try storage.byteLength(count, @sizeOf(WorkspaceScalar(C)));
    return count;
}
fn range(comptime T: type, values: []const T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn add(comptime W: type, sum: *W, correction: *W, term: W) void {
    const y = term - correction.*;
    const next = sum.* + y;
    correction.* = (next - sum.*) - y;
    sum.* = next;
}

/// Checked caller supplies sorted, finite, nonnegative singular values.
/// Strict s>rcond*s[0] determines rank; negative rcond uses eps(Real(C)).
pub fn rankFromWideSingular(comptime C: type, comptime S: type, singular: []const S, rcond: Real(C)) usize {
    comptime cx.expectComplex(C);
    comptime std.debug.assert(S == Real(C) or S == cx.Wide(C));
    const W = cx.Wide(C);
    const cutoff: W = if (singular.len == 0) 0 else @as(W, singular[0]) * @as(W, if (rcond < 0) std.math.floatEps(Real(C)) else rcond);
    var rank: usize = 0;
    for (singular) |value| if (@as(W, value) > cutoff) {
        rank += 1;
    };
    return rank;
}

pub fn solve(comptime C: type, u: dense.ConstMatrix(C), singular: []const Real(C), v: dense.ConstMatrix(C), b: dense.ConstMatrix(C), x: dense.Matrix(C), rcond: Real(C), scratch: []WorkspaceScalar(C)) Error!Result {
    return solveWithSingularType(C, Real(C), u, singular, v, b, x, rcond, scratch);
}

/// U is m-by-k, V n-by-k, k=min(m,n), B m-by-r and X n-by-r.
/// Factors must have orthonormal columns: geometry/finite/singular ordering are
/// checked here; factor orthogonality is the factorization caller's obligation.
/// X commits only when every prospective complex component is finite in C.
/// Active X and scratch byte spans cannot overlap any used input or each other.
/// Read-only inputs may overlap. Padding and unused suffixes are preserved.
/// Singular values remain widened until reconstruction when S=Wide(C).
pub fn solveWithSingularType(comptime C: type, comptime S: type, u: dense.ConstMatrix(C), singular: []const S, v: dense.ConstMatrix(C), b: dense.ConstMatrix(C), x: dense.Matrix(C), rcond: Real(C), scratch: []WorkspaceScalar(C)) Error!Result {
    comptime cx.expectComplex(C);
    comptime std.debug.assert(S == Real(C) or S == cx.Wide(C));
    const W = cx.Wide(C);
    const Z = WorkspaceScalar(C);
    const us = try u.check();
    const vs = try v.check();
    const bs = try b.check();
    const xs = try x.check();
    const k = @min(u.row_count, v.row_count);
    if (u.column_count != k or v.column_count != k or b.row_count != u.row_count or x.row_count != v.row_count or b.column_count != x.column_count) return error.DimensionMismatch;
    if (!std.math.isFinite(rcond)) return error.InvalidTolerance;
    const count = try workspaceLength(C, x.row_count, x.column_count);
    if (singular.len < k or scratch.len < count) return error.BufferTooSmall;
    const outputs = [_]storage.ByteRange{ try range(C, x.values[0..xs]), try range(Z, scratch[0..count]) };
    if (storage.overlaps(outputs[0], outputs[1])) return error.AliasingNotAllowed;
    const inputs = [_]storage.ByteRange{ try range(C, u.values[0..us]), try range(C, v.values[0..vs]), try range(C, b.values[0..bs]), try range(S, singular[0..k]) };
    for (outputs) |out| for (inputs) |in| if (storage.overlaps(out, in)) {
        return error.AliasingNotAllowed;
    };
    for (singular[0..k], 0..) |s, i| if (!std.math.isFinite(s) or s < 0 or (i > 0 and s > singular[i - 1])) {
        return error.InvalidSingularValues;
    };
    inline for (.{ u, v, b }) |a| {
        if (a.row_count != 0) for (0..a.column_count) |j| for (a.values[j * a.leading_dimension ..][0..a.row_count]) |value| if (!cx.finite(C, value)) {
            return error.NonFiniteInput;
        };
    }
    const rank = rankFromWideSingular(C, S, singular[0..k], rcond);
    const cutoff: W = if (k == 0) 0 else @as(W, singular[0]) * @as(W, if (rcond < 0) std.math.floatEps(Real(C)) else rcond);
    if (x.row_count != 0) for (0..b.column_count) |j| for (0..x.row_count) |i| {
        var sum: Z = .{ .re = 0, .im = 0 };
        var compensation: Z = .{ .re = 0, .im = 0 };
        for (0..k) |q| {
            if (@as(W, singular[q]) <= cutoff) continue;
            var dot: Z = .{ .re = 0, .im = 0 };
            var correction: Z = .{ .re = 0, .im = 0 };
            for (0..u.row_count) |r| {
                const uw = cx.widen(C, u.values[r + q * u.leading_dimension]);
                const bw = cx.widen(C, b.values[r + j * b.leading_dimension]);
                add(W, &dot.re, &correction.re, uw.re * bw.re + uw.im * bw.im);
                add(W, &dot.im, &correction.im, uw.re * bw.im - uw.im * bw.re);
            }
            const sigma: W = singular[q];
            const ar = dot.re / sigma;
            const ai = dot.im / sigma;
            const vw = cx.widen(C, v.values[i + q * v.leading_dimension]);
            add(W, &sum.re, &compensation.re, vw.re * ar - vw.im * ai);
            add(W, &sum.im, &compensation.im, vw.re * ai + vw.im * ar);
        }
        if (!cx.finite(Z, sum) or cx.narrow(C, sum) == null) return .{ .rank = rank, .status = .non_finite_result };
        scratch[i + j * x.row_count] = sum;
    };
    if (x.row_count != 0) for (0..x.column_count) |j| for (0..x.row_count) |i| {
        x.values[i + j * x.leading_dimension] = cx.narrow(C, scratch[i + j * x.row_count]).?;
    };
    return .{ .rank = rank, .status = .success };
}
