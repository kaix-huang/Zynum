// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Widened rectangular SVD over shared bidiagonal reduction and QR iteration.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const bidiagonal = @import("bidiagonal.zig");
const legacy = @import("svd.zig");
const partial = @import("bidiagonal_partial.zig");
pub const WorkspaceScalar = legacy.WorkspaceScalar;
pub const Error = bidiagonal.Error || partial.Error || error{ NotBidiagonal, InvalidTolerance };
pub const NumericalPolicy = bidiagonal.NumericalPolicy;
pub const Options = struct { max_steps: usize = 4096, relative_tolerance: ?f64 = null, numerical_policy: NumericalPolicy = .finite };
pub const Result = struct { status: legacy.Status, steps: usize, unconverged: usize = 0 };

pub fn workspaceLength(comptime T: type, m: usize, n: usize) Error!usize {
    const p = @max(m, n);
    const k = @min(m, n);
    if (k == 0) return 0;
    const b = std.math.mul(usize, p, k) catch return error.SizeOverflow;
    const u = std.math.mul(usize, p, p) catch return error.SizeOverflow;
    const v = std.math.mul(usize, k, k) catch return error.SizeOverflow;
    const bu = std.math.add(usize, b, u) catch return error.SizeOverflow;
    const total = std.math.add(usize, bu, v) catch return error.SizeOverflow;
    _ = try storage.byteLength(total, @sizeOf(WorkspaceScalar(T)));
    return total;
}
fn range(comptime T: type, data: []T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(data.ptr), try storage.byteLength(data.len, @sizeOf(T)));
}
fn roundedOrthogonality(comptime T: type, a: dense.Matrix(WorkspaceScalar(T)), columns: usize) WorkspaceScalar(T) {
    const W = WorkspaceScalar(T);
    var worst: W = 0;
    for (0..columns) |i| for (i..columns) |j| {
        var dot: W = 0;
        for (0..a.row_count) |r| {
            const x: T = @floatCast(a.values[r + i * a.leading_dimension]);
            const y: T = @floatCast(a.values[r + j * a.leading_dimension]);
            dot += @as(W, x) * @as(W, y);
        }
        worst = @max(worst, @abs(dot - @as(W, if (i == j) 1 else 0)));
    };
    return worst;
}

/// First k=min(m,n) columns of A become U; right receives n-by-k V.
/// On any failure A/right/singular remain unchanged. Widened workspace retains
/// partial bidiagonal factors for nonconvergence diagnostics. No admission/allocation.
/// Explicit IEEE mode permits nonfinite factors; convergence then means only
/// that no active bidiagonal edges remain, not that finite SVD accuracy holds.
pub fn decompose(comptime T: type, a: dense.Matrix(T), singular: []T, right: dense.Matrix(T), workspace: []WorkspaceScalar(T), options: Options) Error!Result {
    return decomposeWithSingularType(T, T, a, singular, right, workspace, options);
}

/// Preserve widened singular values for downstream operations before export.
pub fn decomposeWithSingularType(comptime T: type, comptime S: type, a: dense.Matrix(T), singular: []S, right: dense.Matrix(T), workspace: []WorkspaceScalar(T), options: Options) Error!Result {
    const W = WorkspaceScalar(T);
    comptime std.debug.assert(S == T or S == W);
    const aspan = try a.check();
    const vspan = try right.check();
    const m = a.row_count;
    const n = a.column_count;
    const p = @max(m, n);
    const k = @min(m, n);
    if (right.row_count != n or right.column_count != k) return error.DimensionMismatch;
    const used = try workspaceLength(T, m, n);
    if (singular.len < k or workspace.len < used) return error.BufferTooSmall;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(T));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    const ranges = [_]storage.ByteRange{ try range(T, a.values[0..aspan]), try range(T, right.values[0..vspan]), try range(S, singular[0..k]), try range(W, workspace[0..used]) };
    for (ranges, 0..) |one, i| for (ranges[i + 1 ..]) |other| {
        if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    };
    if (k == 0) return .{ .status = .converged, .steps = 0 };
    var finite_input = true;
    var input_scale: W = 0;
    for (0..n) |j| for (0..m) |i| {
        const value = a.values[i + j * a.leading_dimension];
        finite_input = finite_input and std.math.isFinite(value);
        input_scale = @max(input_scale, @abs(@as(W, value)));
    };
    if (!finite_input and options.numerical_policy == .finite) return error.NonFiniteInput;
    // Normalize in widened scratch before reduction/QR. Finite input ratios
    // remain representable across the entire source scalar exponent range.
    // This makes iteration independent of a common input scale while keeping
    // checked output range checks and the explicit IEEE propagation policy.
    if (!finite_input or input_scale == 0) input_scale = 1;
    const b: dense.Matrix(W) = .{ .values = workspace[0 .. p * k], .row_count = p, .column_count = k, .leading_dimension = p };
    const u: dense.Matrix(W) = .{ .values = workspace[p * k ..][0 .. p * p], .row_count = p, .column_count = p, .leading_dimension = p };
    const v: dense.Matrix(W) = .{ .values = workspace[p * k + p * p ..][0 .. k * k], .row_count = k, .column_count = k, .leading_dimension = k };
    for (0..n) |j| for (0..m) |i| {
        b.values[if (m >= n) i + j * p else j + i * p] = @as(W, a.values[i + j * a.leading_dimension]) / input_scale;
    };
    try bidiagonal.reduceWithPolicy(W, b, u, v, options.numerical_policy);
    const result = try bidiagonal.diagonalize(W, b, u, v, .{ .max_steps = options.max_steps, .numerical_policy = options.numerical_policy });
    // Restore singular/partial bidiagonal values to original input units.
    // Factors stay dimensionless and caller-visible arrays still commit only
    // after convergence, range and orthogonality checks.
    for (0..k) |j| for (0..p) |i| {
        b.values[i + j * p] *= input_scale;
    };
    if (!result.converged) return .{ .status = .no_convergence, .steps = result.steps, .unconverged = result.unconverged };
    if (options.numerical_policy == .finite) for (0..k) |j| if (!std.math.isFinite(@as(S, @floatCast(b.values[j + j * p])))) return .{ .status = .non_finite_result, .steps = result.steps };
    const left_factor = if (m >= n) u else v;
    const right_factor = if (m >= n) v else u;
    if (finite_input and (roundedOrthogonality(T, left_factor, k) > tolerance or roundedOrthogonality(T, right_factor, k) > tolerance)) return .{ .status = .range_limit, .steps = result.steps };
    for (0..k) |j| {
        singular[j] = @floatCast(b.values[j + j * p]);
        for (0..m) |i| a.values[i + j * a.leading_dimension] = @floatCast(left_factor.values[i + j * left_factor.leading_dimension]);
        for (0..n) |i| right.values[i + j * right.leading_dimension] = @floatCast(right_factor.values[i + j * right_factor.leading_dimension]);
    }
    return .{ .status = .converged, .steps = result.steps };
}

/// Borrow the full-left-factor complement after successful tall decomposition,
/// before workspace reuse. This accessor owns the internal scratch layout.
/// Nonempty n is required; empty-column callers use the identity complement.
pub fn leftComplement(comptime T: type, m: usize, n: usize, workspace: []WorkspaceScalar(T)) Error!dense.ConstMatrix(WorkspaceScalar(T)) {
    if (m < n or n == 0) return error.DimensionMismatch;
    const used = try workspaceLength(T, m, n);
    if (workspace.len < used) return error.BufferTooSmall;
    const start = m * n + m * n;
    return .{ .values = workspace[start..][0 .. m * (m - n)], .row_count = m, .column_count = m - n, .leading_dimension = m };
}

pub fn Factors(comptime T: type) type {
    return struct {
        b_tall: dense.Matrix(WorkspaceScalar(T)),
        left: dense.Matrix(WorkspaceScalar(T)),
        right: dense.Matrix(WorkspaceScalar(T)),
        adjoint_reduction: bool,
    };
}
/// Normalize caller-owned retained factors to upper orientation, keeping the
/// updated orientation with their views. No allocation or admission occurs.
pub fn preparePartialUpper(comptime T: type, state: *Factors(T)) Error!usize {
    return partial.prepare(WorkspaceScalar(T), WorkspaceScalar(T), state.b_tall, state.left, state.right, &state.adjoint_reduction);
}
pub fn checkPartialNative(comptime T: type, state: Factors(T), left_columns: usize, right_columns: usize, tolerance: f64) Error!void {
    if (state.adjoint_reduction) return error.DimensionMismatch;
    return partial.checkNative(T, WorkspaceScalar(T), WorkspaceScalar(T), state.b_tall, state.left, state.right, left_columns, right_columns, tolerance);
}
/// Borrow retained full widened factors after a converged/no_convergence result.
/// Tall A=left*B*right^T; wide A=left*B^T*right^T, in original input units.
/// This accessor owns the scratch layout; it adds no computation or admission.
pub fn factors(comptime T: type, m: usize, n: usize, workspace: []WorkspaceScalar(T)) Error!Factors(T) {
    const used = try workspaceLength(T, m, n);
    if (workspace.len < used) return error.BufferTooSmall;
    const p = @max(m, n);
    const k = @min(m, n);
    const b: dense.Matrix(WorkspaceScalar(T)) = .{ .values = workspace[0 .. p * k], .row_count = p, .column_count = k, .leading_dimension = p };
    if (k == 0) return .{ .b_tall = b, .left = .{ .values = workspace[0..0], .row_count = m, .column_count = 0, .leading_dimension = 0 }, .right = .{ .values = workspace[0..0], .row_count = n, .column_count = 0, .leading_dimension = 0 }, .adjoint_reduction = m < n };
    const u: dense.Matrix(WorkspaceScalar(T)) = .{ .values = workspace[p * k ..][0 .. p * p], .row_count = p, .column_count = p, .leading_dimension = p };
    const v: dense.Matrix(WorkspaceScalar(T)) = .{ .values = workspace[p * k + p * p ..][0 .. k * k], .row_count = k, .column_count = k, .leading_dimension = k };
    return .{ .b_tall = b, .left = if (m >= n) u else v, .right = if (m >= n) v else u, .adjoint_reduction = m < n };
}
