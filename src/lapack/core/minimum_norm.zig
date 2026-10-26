// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Minimum-norm reconstruction from thin real SVD factors A=U*S*V^T.
//! Borrowed native-width descriptors; no allocation or execution admission.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const svd = @import("svd.zig");
pub const WorkspaceScalar = svd.WorkspaceScalar;
pub const Error = dense.Error || error{ DimensionMismatch, InvalidSingularValues, InvalidTolerance, NonFiniteInput, AliasingNotAllowed };
/// IEEE output permits final conversion overflow. Nonfinite RHS admission is
/// controlled separately, as is explicit nonfinite factor admission.
pub const OutputPolicy = enum { finite, ieee };
pub const RhsPolicy = enum { finite, ieee };
pub const CutoffPolicy = enum { finite, unordered };
pub const FactorPolicy = enum { finite, ieee };
pub const Policies = struct { factors: FactorPolicy = .finite, output: OutputPolicy = .finite, rhs: RhsPolicy = .finite, cutoff: CutoffPolicy = .finite };
pub const Result = struct { rank: usize, status: enum { success, non_finite_result } };

pub fn workspaceLength(comptime T: type, columns: usize, rhs: usize) Error!usize {
    const count = std.math.mul(usize, columns, rhs) catch return error.SizeOverflow;
    _ = try storage.byteLength(count, @sizeOf(WorkspaceScalar(T)));
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

/// Rank for finite, sorted nonnegative singular values and finite rcond.
/// Inputs follow solve's checked preconditions; adapters normalize their own policy.
pub fn rankFromSingular(comptime T: type, singular: []const T, rcond: T) usize {
    return rankFromWideSingular(T, T, singular, rcond);
}

pub fn rankFromWideSingular(comptime T: type, comptime S: type, singular: []const S, rcond: T) usize {
    return rankWithPolicy(T, S, singular, rcond, .finite);
}

/// Unordered comparisons retain directions, including zero singular values.
/// Callers must explicitly admit NaN tolerance before selecting this policy.
pub fn rankWithPolicy(comptime T: type, comptime S: type, singular: []const S, rcond: T, policy: CutoffPolicy) usize {
    const W = WorkspaceScalar(T);
    comptime std.debug.assert(S == T or S == W);
    const cutoff: W = if (singular.len == 0) 0 else @as(W, singular[0]) * @as(W, if (rcond < 0) std.math.floatEps(T) else rcond);
    var rank: usize = 0;
    for (singular) |value| if (if (policy == .unordered) !(@as(W, value) <= cutoff) else @as(W, value) > cutoff) {
        rank += 1;
    };
    return rank;
}

/// U is m-by-k, V is n-by-k, k=min(m,n), B is m-by-r and X is n-by-r.
/// Factors must be an SVD with orthonormal columns; this routine checks geometry,
/// finite entries and descending nonnegative singular values, not orthogonality.
/// Singular directions survive iff s>rcond*s[0]. Negative rcond uses eps(T).
/// Output is committed only after all reconstructed entries fit T. Scratch may
/// change on numerical failure. Active X/scratch spans cannot overlap operands;
/// read-only operands may overlap each other. Padding and suffixes are preserved.
pub fn solve(comptime T: type, u: dense.ConstMatrix(T), singular: []const T, v: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), rcond: T, scratch: []WorkspaceScalar(T)) Error!Result {
    return solveWithSingularType(T, T, u, singular, v, b, x, rcond, scratch, .finite, .finite);
}

/// Keep singular values in the widened factorization type until reconstruction.
/// This permits finite solutions even when exported singular values overflow T.
/// RHS policy permits IEEE propagation independently of final cast policy.
/// Each finite RHS retains widened arithmetic checks; rank depends only on S.
pub fn solveWithSingularType(comptime T: type, comptime S: type, u: dense.ConstMatrix(T), singular: []const S, v: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), rcond: T, scratch: []WorkspaceScalar(T), output_policy: OutputPolicy, rhs_policy: RhsPolicy) Error!Result {
    return solveWithPolicies(T, S, u, singular, v, b, x, rcond, scratch, .{ .output = output_policy, .rhs = rhs_policy });
}

pub fn solveWithPolicies(comptime T: type, comptime S: type, u: dense.ConstMatrix(T), singular: []const S, v: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), rcond: T, scratch: []WorkspaceScalar(T), policies: Policies) Error!Result {
    const W = WorkspaceScalar(T);
    comptime std.debug.assert(S == T or S == W);
    const us = try u.check();
    const vs = try v.check();
    const bs = try b.check();
    const xs = try x.check();
    const k = @min(u.row_count, v.row_count);
    if (u.column_count != k or v.column_count != k or b.row_count != u.row_count or x.row_count != v.row_count or x.column_count != b.column_count) return error.DimensionMismatch;
    const count = try workspaceLength(T, x.row_count, x.column_count);
    if (singular.len < k or scratch.len < count) return error.BufferTooSmall;
    if (!std.math.isFinite(rcond) and !(policies.cutoff == .unordered and std.math.isNan(rcond))) return error.InvalidTolerance;
    const outputs = [_]storage.ByteRange{ try range(T, x.values[0..xs]), try range(W, scratch[0..count]) };
    if (storage.overlaps(outputs[0], outputs[1])) return error.AliasingNotAllowed;
    const inputs = [_]storage.ByteRange{ try range(T, u.values[0..us]), try range(T, v.values[0..vs]), try range(T, b.values[0..bs]), try range(S, singular[0..k]) };
    for (outputs) |out| for (inputs) |in| {
        if (storage.overlaps(out, in)) return error.AliasingNotAllowed;
    };
    var nonfinite_factors = false;
    for (singular[0..k], 0..) |s, i| {
        nonfinite_factors = nonfinite_factors or !std.math.isFinite(s);
        if ((policies.factors == .finite and !std.math.isFinite(s)) or s < 0 or (i > 0 and s > singular[i - 1])) return error.InvalidSingularValues;
    }
    inline for (.{ u, v }) |a| {
        for (0..a.column_count) |j| for (a.values[j * a.leading_dimension ..][0..a.row_count]) |value| {
            if (!std.math.isFinite(value)) {
                if (policies.factors == .finite) return error.NonFiniteInput;
                nonfinite_factors = true;
            }
        };
    }
    if (policies.rhs == .finite) for (0..b.column_count) |j| for (b.values[j * b.leading_dimension ..][0..b.row_count]) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteInput;
    };
    const rank = rankWithPolicy(T, S, singular[0..k], rcond, policies.cutoff);
    for (0..b.column_count) |j| {
        var nonfinite_rhs = false;
        for (b.values[j * b.leading_dimension ..][0..b.row_count]) |value| nonfinite_rhs = nonfinite_rhs or !std.math.isFinite(value);
        for (0..x.row_count) |i| {
            var sum: W = 0;
            var correction: W = 0;
            const cutoff: W = if (k == 0) 0 else @as(W, singular[0]) * @as(W, if (rcond < 0) std.math.floatEps(T) else rcond);
            for (0..k) |q| {
                const keep = if (policies.cutoff == .unordered) !(@as(W, singular[q]) <= cutoff) else @as(W, singular[q]) > cutoff;
                if (!keep) continue;
                var dot: W = 0;
                var dot_correction: W = 0;
                for (0..u.row_count) |r| add(W, &dot, &dot_correction, @as(W, u.values[r + q * u.leading_dimension]) * @as(W, b.values[r + j * b.leading_dimension]));
                add(W, &sum, &correction, @as(W, v.values[i + q * v.leading_dimension]) * (dot / @as(W, singular[q])));
            }
            if ((!nonfinite_factors and !nonfinite_rhs and !std.math.isNan(rcond) and !std.math.isFinite(sum)) or (policies.output == .finite and !std.math.isFinite(@as(T, @floatCast(sum))))) return .{ .rank = rank, .status = .non_finite_result };
            scratch[i + j * x.row_count] = sum;
        }
    }
    for (0..x.column_count) |j| for (0..x.row_count) |i| {
        x.values[i + j * x.leading_dimension] = @floatCast(scratch[i + j * x.row_count]);
    };
    return .{ .rank = rank, .status = .success };
}
