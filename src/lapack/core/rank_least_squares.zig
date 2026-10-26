// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed-workspace real rank-revealing least squares, tall/square/wide.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const svd = @import("bidiagonal_svd.zig");
const Status = @import("svd.zig").Status;
const reconstruction = @import("minimum_norm.zig");
pub const Error = svd.Error || reconstruction.Error;
pub const WorkspaceScalar = svd.WorkspaceScalar;
pub const Plan = struct { scalar_elements: usize, wide_elements: usize, a_elements: usize, v_elements: usize, singular_elements: usize, residual_elements: usize };
pub const Options = struct {
    max_steps: usize = 4096,
    relative_tolerance: ?f64 = null,
    output_policy: reconstruction.OutputPolicy = .finite,
    rhs_policy: reconstruction.RhsPolicy = .finite,
    cutoff_policy: reconstruction.CutoffPolicy = .finite,
    numerical_policy: svd.NumericalPolicy = .finite,
};
pub const Result = struct { status: Status, rank: usize, steps: usize, unconverged: usize = 0 };

pub fn plan(comptime T: type, m: usize, n: usize, rhs: usize) Error!Plan {
    return planWithResidual(T, m, n, rhs, false);
}

pub fn planWithResidual(comptime T: type, m: usize, n: usize, rhs: usize, include_residual: bool) Error!Plan {
    const k = @min(m, n);
    const a = std.math.mul(usize, m, n) catch return error.SizeOverflow;
    const v = std.math.mul(usize, n, k) catch return error.SizeOverflow;
    var total = std.math.add(usize, a, v) catch return error.SizeOverflow;
    total = std.math.add(usize, total, k) catch return error.SizeOverflow;
    const residual = if (include_residual) std.math.mul(usize, m -| n, rhs) catch return error.SizeOverflow else 0;
    total = std.math.add(usize, total, residual) catch return error.SizeOverflow;
    _ = try storage.byteLength(total, @sizeOf(T));
    const wide = std.math.add(usize, k, @max(try svd.workspaceLength(T, m, n), try reconstruction.workspaceLength(T, n, rhs))) catch return error.SizeOverflow;
    _ = try storage.byteLength(wide, @sizeOf(WorkspaceScalar(T)));
    return .{ .scalar_elements = total, .wide_elements = wide, .a_elements = a, .v_elements = v, .singular_elements = k, .residual_elements = residual };
}
fn range(comptime T: type, values: []const T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}

/// A and B are immutable. X and singular values commit only on success; scratch
/// may change on failure. Active writable spans cannot overlap each other or
/// either input. Matrix padding and all unused suffixes are preserved.
/// Uses widened bidiagonal QR SVD, not a claim of LAPACK's divide-and-conquer algorithm.
pub fn solve(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), singular: []T, rcond: T, scalar_scratch: []T, wide_scratch: []WorkspaceScalar(T), options: Options) Error!Result {
    return solveImpl(T, a, b, x, singular, null, rcond, scalar_scratch, wide_scratch, options);
}

/// Residual receives (m-n)-by-nrhs orthogonal residual coordinates only for a
/// full-column-rank tall system. Their squares sum to ||B-A*X||^2 per RHS.
/// For rank-deficient systems it remains untouched. It shares the transactional
/// commit and non-overlap rules of X and singular. Use planWithResidual(...,true).
pub fn solveWithResidual(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), singular: []T, residual: dense.Matrix(T), rcond: T, scalar_scratch: []T, wide_scratch: []WorkspaceScalar(T), options: Options) Error!Result {
    return solveImpl(T, a, b, x, singular, residual, rcond, scalar_scratch, wide_scratch, options);
}

fn solveImpl(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), x: dense.Matrix(T), singular: []T, residual: ?dense.Matrix(T), rcond: T, scalar_scratch: []T, wide_scratch: []WorkspaceScalar(T), options: Options) Error!Result {
    const aspan = try a.check();
    const bspan = try b.check();
    const xspan = try x.check();
    const m = a.row_count;
    const n = a.column_count;
    if (b.row_count != m or x.row_count != n or b.column_count != x.column_count) return error.DimensionMismatch;
    if (!std.math.isFinite(rcond) and !(options.cutoff_policy == .unordered and std.math.isNan(rcond))) return error.InvalidTolerance;
    const residual_span = if (residual) |r| try r.check() else 0;
    if (residual) |r| {
        if (r.row_count != m -| n or r.column_count != b.column_count) return error.DimensionMismatch;
    }
    const needed = try planWithResidual(T, m, n, b.column_count, residual != null);
    const k = needed.singular_elements;
    if (singular.len < k or scalar_scratch.len < needed.scalar_elements or wide_scratch.len < needed.wide_elements) return error.BufferTooSmall;
    const writes = [_]storage.ByteRange{ try range(T, if (residual) |r| r.values[0..residual_span] else &.{}), try range(T, x.values[0..xspan]), try range(T, singular[0..k]), try range(T, scalar_scratch[0..needed.scalar_elements]), try range(WorkspaceScalar(T), wide_scratch[0..needed.wide_elements]) };
    const reads = [_]storage.ByteRange{ try range(T, a.values[0..aspan]), try range(T, b.values[0..bspan]) };
    for (writes, 0..) |out, i| {
        for (writes[i + 1 ..]) |other| if (storage.overlaps(out, other)) return error.AliasingNotAllowed;
        for (reads) |in| if (storage.overlaps(out, in)) return error.AliasingNotAllowed;
    }
    const factor_a = scalar_scratch[0..needed.a_elements];
    const factor_v = scalar_scratch[needed.a_elements..][0..needed.v_elements];
    const values = wide_scratch[0..k];
    const algorithm_scratch = wide_scratch[k..needed.wide_elements];
    if (m != 0) for (0..n) |j| {
        @memcpy(factor_a[j * m ..][0..m], a.values[j * a.leading_dimension ..][0..m]);
    };
    const decomposed = try svd.decomposeWithSingularType(T, WorkspaceScalar(T), .{ .values = factor_a, .row_count = m, .column_count = n, .leading_dimension = m }, values, .{ .values = factor_v, .row_count = n, .column_count = k, .leading_dimension = n }, algorithm_scratch, .{ .max_steps = options.max_steps, .relative_tolerance = options.relative_tolerance, .numerical_policy = options.numerical_policy });
    if (decomposed.status != .converged) return .{ .status = decomposed.status, .rank = 0, .steps = decomposed.steps, .unconverged = decomposed.unconverged };
    if (options.output_policy == .finite) for (values) |value| {
        if (!std.math.isFinite(@as(T, @floatCast(value)))) return .{ .status = .non_finite_result, .rank = 0, .steps = decomposed.steps };
    };
    const expected_rank = reconstruction.rankWithPolicy(T, WorkspaceScalar(T), values, rcond, options.cutoff_policy);
    const residual_rows = m -| n;
    const residual_values = scalar_scratch[needed.a_elements + needed.v_elements + k ..][0..needed.residual_elements];
    const emit_residual = residual != null and m > n and expected_rank == n;
    if (emit_residual) {
        // U is still in widened SVD storage; reconstruction reuses that storage.
        // Empty-column matrices have no U allocation and use the identity basis.
        const W = WorkspaceScalar(T);
        const complement = if (n != 0) try svd.leftComplement(T, m, n, algorithm_scratch) else null;
        for (0..b.column_count) |j| for (0..residual_rows) |i| {
            var sum: W = 0;
            var correction: W = 0;
            var nonfinite_rhs = false;
            var nonfinite_factor = false;
            for (0..m) |r| {
                const input = b.values[r + j * b.leading_dimension];
                if (!std.math.isFinite(input)) {
                    if (options.rhs_policy == .finite) return error.NonFiniteInput;
                    nonfinite_rhs = true;
                }
                const coefficient: W = if (n == 0) (if (r == i) 1 else 0) else complement.?.values[r + i * complement.?.leading_dimension];
                nonfinite_factor = nonfinite_factor or !std.math.isFinite(coefficient);
                const term = coefficient * @as(W, input) - correction;
                const next = sum + term;
                correction = (next - sum) - term;
                sum = next;
            }
            if ((!nonfinite_rhs and !nonfinite_factor and !std.math.isFinite(sum)) or (options.output_policy == .finite and !std.math.isFinite(@as(T, @floatCast(sum))))) return .{ .status = .non_finite_result, .rank = expected_rank, .steps = decomposed.steps };
            residual_values[i + j * residual_rows] = @floatCast(sum);
        };
    }
    const solved = try reconstruction.solveWithPolicies(T, WorkspaceScalar(T), .{ .values = factor_a, .row_count = m, .column_count = k, .leading_dimension = m }, values, .{ .values = factor_v, .row_count = n, .column_count = k, .leading_dimension = n }, b, x, rcond, algorithm_scratch, .{ .output = options.output_policy, .rhs = options.rhs_policy, .cutoff = options.cutoff_policy, .factors = if (options.numerical_policy == .ieee) .ieee else .finite });
    if (solved.status != .success) return .{ .status = .non_finite_result, .rank = solved.rank, .steps = decomposed.steps };
    for (values, singular[0..k]) |value, *out| out.* = @floatCast(value);
    if (emit_residual) {
        const out = residual.?;
        for (0..out.column_count) |j| @memcpy(out.values[j * out.leading_dimension ..][0..residual_rows], residual_values[j * residual_rows ..][0..residual_rows]);
    }
    return .{ .status = .converged, .rank = solved.rank, .steps = decomposed.steps };
}
