// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Four-scalar SVD least squares over one borrowed admitted operation.
//! Cutoff and checked/IEEE policies are internal, independent of vendor WORK.
const std = @import("std");
const execution = @import("zynum-execution");
const dense = @import("../numerics/dense.zig");
const storage = @import("../numerics/storage.zig");
const cx = @import("../numerics/complex.zig");
const scalar = cx.scalar;
const real_svd = @import("core/bidiagonal_svd.zig");
const complex_svd = @import("core/complex_bidiagonal_svd.zig");
pub const Options = real_svd.Options;
pub const Status = @import("core/svd.zig").Status;
pub const Stage = enum { complete, factorization, reconstruction, publication };
pub const Result = struct { status: Status, rank: usize = 0, steps: usize = 0, unconverged: usize = 0, stage: Stage = .complete };
pub const Error = real_svd.Error || complex_svd.Error || error{InvalidCutoff};
pub const ManagedError = Error || execution.context.AdmissionError || std.mem.Allocator.Error;
pub fn Wide(comptime T: type) type {
    if (T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("SVD least squares requires real/complex f32 or f64");
    return if (scalar.Real(T) == f32) f64 else f128;
}
pub fn WorkspaceScalar(comptime T: type) type {
    return if (scalar.isComplex(T)) cx.WideComplex(T) else Wide(T);
}
pub fn Cutoff(comptime T: type) type {
    return struct { relative: Wide(T) = 0, absolute: Wide(T) = 0 };
}
pub fn Outputs(comptime T: type) type {
    return struct {
        solution: dense.Matrix(T),
        singular: []scalar.Real(T),
        right_vectors: dense.Matrix(T),
        residual: dense.Matrix(T),
    };
}
pub const Plan = struct { scalar_elements: usize, computational_elements: usize, real_elements: usize, a_elements: usize, v_elements: usize, solution_elements: usize, residual_elements: usize, svd_elements: usize, real_svd_elements: usize, singular_elements: usize, coefficient_elements: usize };
fn mulSize(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}
fn addSize(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
pub fn plan(comptime T: type, m: usize, n: usize, rhs: usize) Error!Plan {
    const k = @min(m, n);
    const a = try mulSize(m, n);
    const v = try mulSize(n, k);
    const x = try mulSize(n, rhs);
    const residual = try mulSize(m -| n, rhs);
    const coefficients = try mulSize(k, rhs);
    const elements = try addSize(try addSize(a, v), try addSize(x, residual));
    const complex_plan = if (comptime scalar.isComplex(T)) try complex_svd.plan(T, m, n) else {};
    const svd_elements = if (comptime scalar.isComplex(T)) complex_plan.complex_elements else try real_svd.workspaceLength(T, m, n);
    const real_svd_elements: usize = if (comptime scalar.isComplex(T)) complex_plan.real_elements else 0;
    const computational = try addSize(svd_elements, coefficients);
    const real_elements = try addSize(k, real_svd_elements);
    _ = try storage.byteLength(elements, @sizeOf(T));
    _ = try storage.byteLength(computational, @sizeOf(WorkspaceScalar(T)));
    _ = try storage.byteLength(real_elements, @sizeOf(Wide(T)));
    return .{ .scalar_elements = elements, .computational_elements = computational, .real_elements = real_elements, .a_elements = a, .v_elements = v, .solution_elements = x, .residual_elements = residual, .svd_elements = svd_elements, .real_svd_elements = real_svd_elements, .singular_elements = k, .coefficient_elements = coefficients };
}
fn range(comptime T: type, values: []const T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn finite(comptime T: type, value: T) bool {
    return if (comptime scalar.isComplex(T)) cx.finite(T, value) else std.math.isFinite(value);
}
fn zero(comptime T: type) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = 0, .im = 0 } else 0;
}
fn widen(comptime T: type, value: T) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) cx.widen(T, value) else value;
}
fn conjugateProduct(comptime T: type, a: WorkspaceScalar(T), b: WorkspaceScalar(T)) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = a.re * b.re + a.im * b.im, .im = a.re * b.im - a.im * b.re } else a * b;
}
fn product(comptime T: type, a: WorkspaceScalar(T), b: WorkspaceScalar(T)) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re } else a * b;
}
fn accumulateReal(comptime W: type, sum: *W, correction: *W, term: W) void {
    const adjusted = term - correction.*;
    const next = sum.* + adjusted;
    correction.* = (next - sum.*) - adjusted;
    sum.* = next;
}
fn accumulate(comptime T: type, sum: *WorkspaceScalar(T), correction: *WorkspaceScalar(T), term: WorkspaceScalar(T)) void {
    if (comptime scalar.isComplex(T)) {
        accumulateReal(Wide(T), &sum.re, &correction.re, term.re);
        accumulateReal(Wide(T), &sum.im, &correction.im, term.im);
    } else accumulateReal(Wide(T), sum, correction, term);
}
fn divideReal(comptime T: type, value: WorkspaceScalar(T), divisor: Wide(T)) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = value.re / divisor, .im = value.im / divisor } else value / divisor;
}
fn narrow(comptime T: type, value: WorkspaceScalar(T), options: Options) ?T {
    const result: T = if (comptime scalar.isComplex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else @floatCast(value);
    return if (options.numerical_policy == .finite and !finite(T, result)) null else result;
}
fn validate(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), cutoff: Cutoff(T), options: Options) Error!Plan {
    const aspan = try a.check();
    const bspan = try b.check();
    const xspan = try out.solution.check();
    const vspan = try out.right_vectors.check();
    const rspan = try out.residual.check();
    const m = a.row_count;
    const n = a.column_count;
    const rhs = b.column_count;
    const k = @min(m, n);
    if (b.row_count != m or out.solution.row_count != n or out.solution.column_count != rhs or out.right_vectors.row_count != n or out.right_vectors.column_count != k or out.residual.row_count != m -| n or out.residual.column_count != rhs) return error.DimensionMismatch;
    if (out.singular.len < k) return error.BufferTooSmall;
    if (cutoff.relative < 0 or cutoff.absolute < 0 or (options.numerical_policy == .finite and (std.math.isNan(cutoff.relative) or std.math.isNan(cutoff.absolute)))) return error.InvalidCutoff;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(scalar.Real(T)));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    const writes = [_]storage.ByteRange{ try range(T, out.solution.values[0..xspan]), try range(scalar.Real(T), out.singular[0..k]), try range(T, out.right_vectors.values[0..vspan]), try range(T, out.residual.values[0..rspan]) };
    const reads = [_]storage.ByteRange{ try range(T, a.values[0..aspan]), try range(T, b.values[0..bspan]) };
    for (writes, 0..) |one, i| {
        for (writes[i + 1 ..]) |other| if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
        for (reads) |other| if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    }
    if (options.numerical_policy == .finite) inline for (.{ a, b }) |matrix| for (0..matrix.column_count) |j| for (0..matrix.row_count) |i| {
        if (!finite(T, matrix.values[i + j * matrix.leading_dimension])) return error.NonFiniteInput;
    };
    return plan(T, m, n, rhs);
}

/// A and B remain immutable. Successful publication commits X, descending S and
/// n-by-k V. Only full-column-rank tall problems publish orthogonal residual
/// coordinates. Geometry/alias/cutoff errors and unsuccessful numeric results
/// leave all caller outputs unchanged. Every scratch lease borrows this one op,
/// including allocations and cleanup after its Context starts closing.
pub fn solveAdmitted(operation: *const execution.context.Operation, comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), cutoff: Cutoff(T), options: Options) ManagedError!Result {
    _ = operation.borrowedContext();
    const p = try validate(T, a, b, out, cutoff, options);
    var scalar_lease = try execution.context.ContextScratch(T).acquire(operation, p.scalar_elements);
    defer scalar_lease.deinit();
    var computational_lease = try execution.context.ContextScratch(WorkspaceScalar(T)).acquire(operation, p.computational_elements);
    defer computational_lease.deinit();
    var real_lease = try execution.context.ContextScratch(Wide(T)).acquire(operation, p.real_elements);
    defer real_lease.deinit();
    return solvePrepared(T, a, b, out, cutoff, options, p, .{
        .scalar = scalar_lease.lease.values,
        .computational = computational_lease.lease.values,
        .real = real_lease.lease.values,
    });
}

/// Borrowed mathematical scratch; independent of any external WORK layout.
/// Only the planned prefixes are admitted. Unused suffixes stay untouched.
pub fn Scratch(comptime T: type) type {
    return struct { scalar: []T, computational: []WorkspaceScalar(T), real: []Wide(T) };
}
fn admitScratch(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), p: Plan, buffers: Scratch(T)) Error!Scratch(T) {
    if (buffers.scalar.len < p.scalar_elements or buffers.computational.len < p.computational_elements or buffers.real.len < p.real_elements) return error.BufferTooSmall;
    const accepted: Scratch(T) = .{
        .scalar = buffers.scalar[0..p.scalar_elements],
        .computational = buffers.computational[0..p.computational_elements],
        .real = buffers.real[0..p.real_elements],
    };
    const work_ranges = [_]storage.ByteRange{
        try range(T, accepted.scalar), try range(WorkspaceScalar(T), accepted.computational), try range(Wide(T), accepted.real),
    };
    const operands = [_]storage.ByteRange{
        try range(T, a.values[0..try a.check()]),                                 try range(T, b.values[0..try b.check()]),
        try range(T, out.solution.values[0..try out.solution.check()]),           try range(scalar.Real(T), out.singular[0..p.singular_elements]),
        try range(T, out.right_vectors.values[0..try out.right_vectors.check()]), try range(T, out.residual.values[0..try out.residual.check()]),
    };
    for (work_ranges, 0..) |one, i| {
        for (work_ranges[i + 1 ..]) |other| if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
        for (operands) |other| if (storage.overlaps(one, other)) return error.AliasingNotAllowed;
    }
    return accepted;
}
/// The same transactional mathematics as solveAdmitted, using caller scratch.
/// This entry creates no Context, admission, allocator, executor or handle.
pub fn solveBorrowed(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), cutoff: Cutoff(T), options: Options, buffers: Scratch(T)) Error!Result {
    const p = try validate(T, a, b, out, cutoff, options);
    const accepted = try admitScratch(T, a, b, out, p, buffers);
    return solvePrepared(T, a, b, out, cutoff, options, p, accepted);
}
fn solvePrepared(comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), cutoff: Cutoff(T), options: Options, p: Plan, buffers: Scratch(T)) Error!Result {
    const m = a.row_count;
    const n = a.column_count;
    const rhs = b.column_count;
    const k = @min(m, n);
    const values = buffers.scalar;
    const work = buffers.computational;
    const real_work = buffers.real;
    const factor_a: dense.Matrix(T) = .{ .values = values[0..p.a_elements], .row_count = m, .column_count = n, .leading_dimension = m };
    const factor_v: dense.Matrix(T) = .{ .values = values[p.a_elements..][0..p.v_elements], .row_count = n, .column_count = k, .leading_dimension = n };
    const solution = values[p.a_elements + p.v_elements ..][0..p.solution_elements];
    const residual = values[p.a_elements + p.v_elements + p.solution_elements ..][0..p.residual_elements];
    const singular = real_work[0..k];
    for (0..n) |j| if (m != 0) @memcpy(factor_a.values[j * m ..][0..m], a.values[j * a.leading_dimension ..][0..m]);
    const result = if (comptime scalar.isComplex(T))
        try complex_svd.decomposeWithSingularType(T, Wide(T), factor_a, singular, factor_v, work[0..p.svd_elements], real_work[k..], options)
    else
        try real_svd.decomposeWithSingularType(T, Wide(T), factor_a, singular, factor_v, work[0..p.svd_elements], options);
    if (result.status != .converged) return .{ .status = result.status, .steps = result.steps, .unconverged = result.unconverged, .stage = .factorization };
    const threshold: Wide(T) = if (k == 0) 0 else blk: {
        const relative = cutoff.relative * singular[0];
        break :blk if (std.math.isNan(relative) or std.math.isNan(cutoff.absolute)) std.math.nan(Wide(T)) else @max(relative, cutoff.absolute);
    };
    var rank: usize = 0;
    for (singular) |sigma| if (sigma > threshold) {
        rank += 1;
    };
    const state = if (comptime scalar.isComplex(T)) try complex_svd.factors(T, m, n, work[0..p.svd_elements]) else try real_svd.factors(T, m, n, work[0..p.svd_elements]);
    const coefficients = work[p.svd_elements..][0..p.coefficient_elements];
    for (0..rhs) |j| for (0..k) |q| {
        var dot = zero(T);
        var correction = zero(T);
        if (singular[q] > threshold) {
            for (0..m) |i| accumulate(T, &dot, &correction, conjugateProduct(T, state.left.values[i + q * state.left.leading_dimension], widen(T, b.values[i + j * b.leading_dimension])));
            dot = divideReal(T, dot, singular[q]);
        }
        coefficients[q + j * k] = dot;
    };
    for (0..rhs) |j| for (0..n) |i| {
        var value = zero(T);
        var correction = zero(T);
        for (0..k) |q| accumulate(T, &value, &correction, product(T, state.right.values[i + q * state.right.leading_dimension], coefficients[q + j * k]));
        solution[i + j * n] = narrow(T, value, options) orelse return .{ .status = .non_finite_result, .rank = rank, .steps = result.steps, .stage = .reconstruction };
    };
    const emit_residual = m > n and rank == n;
    if (emit_residual) for (0..rhs) |j| for (0..m - n) |q| {
        var value = zero(T);
        var correction = zero(T);
        if (n == 0) value = widen(T, b.values[q + j * b.leading_dimension]) else {
            for (0..m) |i| accumulate(T, &value, &correction, conjugateProduct(T, state.left.values[i + (n + q) * state.left.leading_dimension], widen(T, b.values[i + j * b.leading_dimension])));
        }
        residual[q + j * (m - n)] = narrow(T, value, options) orelse return .{ .status = .non_finite_result, .rank = rank, .steps = result.steps, .stage = .reconstruction };
    };
    if (options.numerical_policy == .finite) for (singular) |sigma| {
        if (!std.math.isFinite(@as(scalar.Real(T), @floatCast(sigma)))) return .{ .status = .non_finite_result, .rank = rank, .steps = result.steps, .stage = .publication };
    };
    for (0..rhs) |j| {
        if (n != 0) @memcpy(out.solution.values[j * out.solution.leading_dimension ..][0..n], solution[j * n ..][0..n]);
        if (emit_residual) @memcpy(out.residual.values[j * out.residual.leading_dimension ..][0 .. m - n], residual[j * (m - n) ..][0 .. m - n]);
    }
    for (0..k) |j| {
        out.singular[j] = @floatCast(singular[j]);
        if (n != 0) @memcpy(out.right_vectors.values[j * out.right_vectors.leading_dimension ..][0..n], factor_v.values[j * n ..][0..n]);
    }
    return .{ .status = .converged, .rank = rank, .steps = result.steps };
}
pub fn solveManaged(context: *execution.context.Context, comptime T: type, a: dense.ConstMatrix(T), b: dense.ConstMatrix(T), out: Outputs(T), cutoff: Cutoff(T), options: Options) ManagedError!Result {
    var operation = try context.begin();
    defer operation.end();
    return solveAdmitted(&operation, T, a, b, out, cutoff, options);
}
