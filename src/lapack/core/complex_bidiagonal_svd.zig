// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed rectangular complex SVD using shared real bidiagonal QR.
//! All arrays are caller-owned; no execution admission or allocation occurs.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const cx = @import("../../numerics/complex.zig");
const bidiagonal = @import("complex_bidiagonal.zig");
const real_svd = @import("bidiagonal_svd.zig");
const partial = @import("bidiagonal_partial.zig");
pub const WorkspaceScalar = cx.WideComplex;
pub const Wide = cx.Wide;
pub const Real = cx.scalar.Real;
pub const Options = real_svd.Options;
pub const Result = real_svd.Result;
pub const Error = bidiagonal.Error || partial.Error || error{InvalidTolerance};
pub const Plan = struct { complex_elements: usize, real_elements: usize, rows: usize, columns: usize, b_elements: usize, u_elements: usize, v_elements: usize };

pub fn plan(comptime C: type, m: usize, n: usize) Error!Plan {
    comptime cx.expectComplex(C);
    const p = @max(m, n);
    const k = @min(m, n);
    if (k == 0) return .{ .complex_elements = 0, .real_elements = 0, .rows = p, .columns = 0, .b_elements = 0, .u_elements = 0, .v_elements = 0 };
    const b = std.math.mul(usize, p, k) catch return error.SizeOverflow;
    const u = std.math.mul(usize, p, p) catch return error.SizeOverflow;
    const v = std.math.mul(usize, k, k) catch return error.SizeOverflow;
    const real = try bidiagonal.realWorkspaceLength(p, k);
    const complex = std.math.add(usize, real, p) catch return error.SizeOverflow;
    _ = try storage.byteLength(real, @sizeOf(Wide(C)));
    _ = try storage.byteLength(complex, @sizeOf(WorkspaceScalar(C)));
    return .{ .complex_elements = complex, .real_elements = real, .rows = p, .columns = k, .b_elements = b, .u_elements = u, .v_elements = v };
}

pub fn Factors(comptime C: type) type {
    return struct {
        b_tall: dense.Matrix(WorkspaceScalar(C)),
        left: dense.Matrix(WorkspaceScalar(C)),
        right: dense.Matrix(WorkspaceScalar(C)),
        adjoint_reduction: bool,
    };
}
/// Prepare the borrowed retained factors in upper orientation without creating
/// a second owner, execution operation or scratch allocation.
pub fn preparePartialUpper(comptime C: type, state: *Factors(C)) Error!usize {
    return partial.prepare(WorkspaceScalar(C), Wide(C), state.b_tall, state.left, state.right, &state.adjoint_reduction);
}
pub fn checkPartialNative(comptime C: type, state: Factors(C), left_columns: usize, right_columns: usize, tolerance: f64) Error!void {
    if (state.adjoint_reduction) return error.DimensionMismatch;
    return partial.checkNative(Real(C), WorkspaceScalar(C), Wide(C), state.b_tall, state.left, state.right, left_columns, right_columns, tolerance);
}
/// Inspect the retained widened factors after a converged/no_convergence result.
/// Tall/square A=left*B*right^H; wide A=left*B^H*right^H. B uses original input
/// units. Other validation/arithmetic failures do not establish a usable state.
pub fn factors(comptime C: type, m: usize, n: usize, workspace: []WorkspaceScalar(C)) Error!Factors(C) {
    const p = try plan(C, m, n);
    if (workspace.len < p.complex_elements) return error.BufferTooSmall;
    const b: dense.Matrix(WorkspaceScalar(C)) = .{ .values = workspace[0..p.b_elements], .row_count = p.rows, .column_count = p.columns, .leading_dimension = p.rows };
    if (p.columns == 0) return .{ .b_tall = b, .left = .{ .values = workspace[0..0], .row_count = m, .column_count = 0, .leading_dimension = 0 }, .right = .{ .values = workspace[0..0], .row_count = n, .column_count = 0, .leading_dimension = 0 }, .adjoint_reduction = m < n };
    const u: dense.Matrix(WorkspaceScalar(C)) = .{ .values = workspace[p.b_elements..][0..p.u_elements], .row_count = p.rows, .column_count = p.rows, .leading_dimension = p.rows };
    const v: dense.Matrix(WorkspaceScalar(C)) = .{ .values = workspace[p.b_elements + p.u_elements ..][0..p.v_elements], .row_count = p.columns, .column_count = p.columns, .leading_dimension = p.columns };
    return .{ .b_tall = b, .left = if (m >= n) u else v, .right = if (m >= n) v else u, .adjoint_reduction = m < n };
}
fn range(comptime T: type, values: []T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn roundedOrthogonality(comptime C: type, q: dense.Matrix(WorkspaceScalar(C)), columns: usize) Wide(C) {
    var worst: Wide(C) = 0;
    for (0..columns) |p| for (p..columns) |j| {
        var re: Wide(C) = 0;
        var im: Wide(C) = 0;
        for (0..q.row_count) |i| {
            const x = q.values[i + p * q.leading_dimension];
            const y = q.values[i + j * q.leading_dimension];
            const xr: Real(C) = @floatCast(x.re);
            const xi: Real(C) = @floatCast(x.im);
            const yr: Real(C) = @floatCast(y.re);
            const yi: Real(C) = @floatCast(y.im);
            re += @as(Wide(C), xr) * yr + @as(Wide(C), xi) * yi;
            im += @as(Wide(C), xr) * yi - @as(Wide(C), xi) * yr;
        }
        worst = @max(worst, std.math.hypot(re - @as(Wide(C), if (p == j) 1 else 0), im));
    };
    return worst;
}

/// A=U*diag(S)*V^H for every rectangular shape, k=min(m,n). First k columns
/// of A receive thin U, right receives n-by-k V. Trailing A columns, padding,
/// singular suffix and unused scratch tails remain unchanged. Public numerical
/// outputs commit only on convergence, representability and rounded-unitarity
/// success. Workspace may change on failure; retained partial QR has an actual
/// nonzero bidiagonal edge count. Single ownership remains with the caller.
pub fn decompose(comptime C: type, a: dense.Matrix(C), singular: []Real(C), right: dense.Matrix(C), complex_work: []WorkspaceScalar(C), real_work: []Wide(C), options: Options) Error!Result {
    return decomposeWithSingularType(C, Real(C), a, singular, right, complex_work, real_work, options);
}
pub fn decomposeWithSingularType(comptime C: type, comptime S: type, a: dense.Matrix(C), singular: []S, right: dense.Matrix(C), complex_work: []WorkspaceScalar(C), real_work: []Wide(C), options: Options) Error!Result {
    if (S != Real(C) and S != Wide(C)) @compileError("complex bidiagonal SVD singular type must be matching real or widened real");
    const aspan = try a.check();
    const vspan = try right.check();
    const m = a.row_count;
    const n = a.column_count;
    const k = @min(m, n);
    if (right.row_count != n or right.column_count != k) return error.DimensionMismatch;
    const needed = try plan(C, m, n);
    if (singular.len < k or complex_work.len < needed.complex_elements or real_work.len < needed.real_elements) return error.BufferTooSmall;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(Real(C)));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    const ranges = [_]storage.ByteRange{ try range(C, a.values[0..aspan]), try range(C, right.values[0..vspan]), try range(S, singular[0..k]), try range(WorkspaceScalar(C), complex_work[0..needed.complex_elements]), try range(Wide(C), real_work[0..needed.real_elements]) };
    for (ranges, 0..) |one, i| for (ranges[i + 1 ..]) |other| {
        if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    };
    if (k == 0) return .{ .status = .converged, .steps = 0 };
    var input_scale: Wide(C) = 0;
    var finite_input = true;
    for (0..n) |j| for (0..m) |i| {
        const z = a.values[i + j * a.leading_dimension];
        finite_input = finite_input and cx.finite(C, z);
        input_scale = @max(input_scale, @max(@abs(@as(Wide(C), z.re)), @abs(@as(Wide(C), z.im))));
    };
    if (!finite_input and options.numerical_policy == .finite) return error.NonFiniteInput;
    if (!finite_input or input_scale == 0) input_scale = 1;
    const state = try factors(C, m, n, complex_work);
    const b = state.b_tall;
    for (0..n) |j| for (0..m) |i| {
        const z = a.values[i + j * a.leading_dimension];
        b.values[if (m >= n) i + j * b.leading_dimension else j + i * b.leading_dimension] = .{ .re = @as(Wide(C), z.re) / input_scale, .im = (if (m >= n) @as(Wide(C), z.im) else -@as(Wide(C), z.im)) / input_scale };
    };
    const u = if (m >= n) state.left else state.right;
    const v = if (m >= n) state.right else state.left;
    try bidiagonal.reduceWithPolicy(C, b, u, v, options.numerical_policy);
    const row = complex_work[needed.b_elements + needed.u_elements + needed.v_elements ..][0..needed.rows];
    const result = try bidiagonal.diagonalize(C, b, u, v, real_work, row, .{ .max_steps = options.max_steps, .numerical_policy = options.numerical_policy });
    for (0..b.column_count) |j| for (0..b.row_count) |i| {
        b.values[i + j * b.leading_dimension].re *= input_scale;
    };
    if (!result.converged) return .{ .status = .no_convergence, .steps = result.steps, .unconverged = result.unconverged };
    if (options.numerical_policy == .finite) for (0..k) |j| {
        if (!std.math.isFinite(@as(S, @floatCast(b.values[j + j * b.leading_dimension].re)))) return .{ .status = .non_finite_result, .steps = result.steps };
    };
    if (finite_input and (roundedOrthogonality(C, state.left, k) > tolerance or roundedOrthogonality(C, state.right, k) > tolerance)) return .{ .status = .range_limit, .steps = result.steps };
    for (0..k) |j| {
        singular[j] = @floatCast(b.values[j + j * b.leading_dimension].re);
        for (0..m) |i| a.values[i + j * a.leading_dimension] = .{ .re = @floatCast(state.left.values[i + j * state.left.leading_dimension].re), .im = @floatCast(state.left.values[i + j * state.left.leading_dimension].im) };
        for (0..n) |i| right.values[i + j * right.leading_dimension] = .{ .re = @floatCast(state.right.values[i + j * state.right.leading_dimension].re), .im = @floatCast(state.right.values[i + j * state.right.leading_dimension].im) };
    }
    return .{ .status = .converged, .steps = result.steps };
}
