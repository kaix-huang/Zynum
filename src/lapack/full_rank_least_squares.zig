// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Full-rank QR/LQ least squares and minimum-norm composition.
//! All buffers belong to the caller; all stages borrow one execution operation.
//! No SVD, rank threshold, external backend or new resource owner is introduced.
const std = @import("std");
const execution = @import("zynum-execution");
const views = @import("../blas/api/views.zig");
const dense = @import("../numerics/dense.zig");
const storage = @import("../numerics/storage.zig");
const arithmetic = @import("../numerics/complex.zig");
const scalar = arithmetic.scalar;
const validation = @import("validation.zig");
const complex = @import("complex_support.zig");
const qr = @import("qr.zig");
const lq = @import("lq.zig");
const triangular = @import("../numerics/triangular_ieee.zig");

pub const Error = validation.Error;
pub const Contract = qr.Contract;
pub const Operation = enum { normal, adjoint };
pub const Stage = enum { complete, factorization, q_application, triangular_solve, rescaling };
pub const Failure = enum { singular, arithmetic };
pub const Result = struct {
    info: usize = 0,
    failure: ?Failure = null,
    stage: Stage = .complete,
    rhs_column: usize = 0,
};
pub const Plan = struct { tau_elements: usize = 0, composition_elements: usize = 0, total_elements: usize = 0 };

fn Wide(comptime T: type) type {
    if (T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("full-rank least squares requires real/complex f32 or f64");
    return if (scalar.Real(T) == f32) f64 else f128;
}

/// Internal scratch plan, independent of any external WORK array contract.
/// Empty computations need none. The LQ branch reuses one transpose workspace.
pub fn plan(comptime T: type, m: usize, n: usize, nrhs: usize) Error!Plan {
    _ = Wide(T);
    if (m == 0 or n == 0 or nrhs == 0) return .{};
    const k = @min(m, n);
    const composition = if (m < n) try lq.applyWorkspaceLength(T, m, n) else 0;
    const total = std.math.add(usize, k, composition) catch return error.InvalidLength;
    _ = storage.byteLength(total, @sizeOf(T)) catch return error.InvalidLength;
    return .{ .tau_elements = k, .composition_elements = composition, .total_elements = total };
}

fn span(comptime T: type, a: anytype) Error!usize {
    return if (comptime scalar.isComplex(T)) complex.matrixSpan(T, a) else validation.matrixSpan(T, a);
}

fn finite(comptime T: type, value: T) bool {
    return if (comptime scalar.isComplex(T)) arithmetic.finite(T, value) else std.math.isFinite(value);
}

fn prefix(comptime T: type, a: views.Matrix(T), rows: usize) views.Matrix(T) {
    var result = a;
    result.row_count = @intCast(rows);
    return result;
}

fn magnitude(comptime T: type, value: T) Wide(T) {
    if (comptime scalar.isComplex(T)) {
        const re = @abs(@as(Wide(T), value.re));
        const im = @abs(@as(Wide(T), value.im));
        if (std.math.isNan(re) or std.math.isNan(im)) return std.math.nan(Wide(T));
        const largest = @max(re, im);
        if (largest == 0 or std.math.isInf(largest)) return largest;
        const r = re / largest;
        const i = im / largest;
        return largest * @sqrt(r * r + i * i);
    }
    return @abs(@as(Wide(T), value));
}

fn norm(comptime T: type, a: views.Matrix(T)) Wide(T) {
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const ld: usize = @intCast(a.leading_dimension);
    var largest: Wide(T) = 0;
    for (0..n) |j| for (0..m) |i| {
        const value = magnitude(T, a.values[i + j * ld]);
        if (std.math.isNan(value)) return value;
        largest = @max(largest, value);
    };
    return largest;
}

fn scaleFactor(comptime T: type, value: Wide(T)) Wide(T) {
    if (!(value > 0) or !std.math.isFinite(value)) return 1;
    const R = scalar.Real(T);
    const small = @as(Wide(T), std.math.floatMin(R)) / @as(Wide(T), std.math.floatEps(R));
    const large = 1 / small;
    return if (value < small) small / value else if (value > large) large / value else 1;
}

fn scaled(comptime T: type, value: T, factor: Wide(T)) T {
    return if (comptime scalar.isComplex(T))
        .{ .re = @floatCast(@as(Wide(T), value.re) * factor), .im = @floatCast(@as(Wide(T), value.im) * factor) }
    else
        @floatCast(@as(Wide(T), value) * factor);
}

fn scaleMatrix(comptime T: type, a: views.Matrix(T), factor: Wide(T)) void {
    if (factor == 1) return;
    const ld: usize = @intCast(a.leading_dimension);
    for (0..@as(usize, @intCast(a.column_count))) |j| for (0..@as(usize, @intCast(a.row_count))) |i| {
        a.values[i + j * ld] = scaled(T, a.values[i + j * ld], factor);
    };
}

fn zeroRows(comptime T: type, b: views.Matrix(T), first: usize, last: usize) void {
    const ld: usize = @intCast(b.leading_dimension);
    for (0..@as(usize, @intCast(b.column_count))) |j| for (first..last) |i| {
        b.values[i + j * ld] = scalar.zero(T);
    };
}

fn observedNonFinite(comptime T: type, b: views.Matrix(T), stage: Stage) ?Result {
    const ld: usize = @intCast(b.leading_dimension);
    for (0..@as(usize, @intCast(b.column_count))) |j| for (0..@as(usize, @intCast(b.row_count))) |i| {
        if (!finite(T, b.values[i + j * ld])) return .{ .info = i + 1, .failure = .arithmetic, .stage = stage, .rhs_column = j + 1 };
    };
    return null;
}

fn solveTriangle(comptime T: type, a: views.Matrix(T), b: views.Matrix(T), triangle: triangular.Triangle, operation: Operation, comptime contract: Contract) Result {
    const k: usize = @intCast(b.row_count);
    const factor: dense.ConstMatrix(T) = .{ .values = a.values, .row_count = k, .column_count = k, .leading_dimension = @intCast(a.leading_dimension) };
    const rhs: dense.Matrix(T) = .{ .values = b.values, .row_count = k, .column_count = @intCast(b.column_count), .leading_dimension = @intCast(b.leading_dimension) };
    const info = triangular.solveNonsingular(T, factor, rhs, triangle, if (operation == .normal) .normal else .adjoint, .non_unit);
    if (info != 0) return .{ .info = info, .failure = .singular, .stage = .triangular_solve };
    if (contract == .checked) if (observedNonFinite(T, b, .triangular_solve)) |failure| return failure;
    return .{};
}

fn applyQ(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []const T, b: views.Matrix(T), q_operation: qr.QOperation, scratch: []T, comptime contract: Contract) Error!Result {
    const info = if (a.row_count >= a.column_count)
        try qr.applyQSideWithContractAdmitted(operation, T, a.asConst(), tau, b, .left, q_operation, contract)
    else
        try lq.applyQWithContractAdmitted(operation, T, a.asConst(), tau, b, .left, q_operation, scratch, contract);
    return if (info == 0) .{} else .{ .info = info, .failure = .arithmetic, .stage = .q_application };
}

/// A is m-by-n; B has max(m,n) rows and nrhs columns. Input B uses m rows
/// for normal operation, n for adjoint. On success its first n/m rows hold X.
/// A contains the scaled QR/LQ factor. Least-squares tails remain the transformed,
/// scaled RHS coordinates; only solution rows are restored to original units.
///
/// Shape, capacity, alias and checked finite-input errors precede mutation.
/// Empty computations zero B without numerical reads; all-zero A likewise
/// returns zero B. Numerical failures retain committed stages. Exact singularity
/// is checked at the triangular leg, after Q application in least-squares
/// branches, before zero extension in minimum-norm branches, before rescaling.
/// Checked arithmetic failures may leave nonfinite intermediate RHS values;
/// their explicit failure/stage is distinct from standard positive singular INFO.
pub fn solve(comptime T: type, a: views.Matrix(T), b: views.Matrix(T), problem: Operation, workspace: []T) Error!Result {
    var operation = execution.beginDefault();
    defer operation.end();
    return solveWithContractAdmitted(&operation, T, a, b, problem, workspace, .checked);
}

pub fn solveWithContractAdmitted(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), b: views.Matrix(T), problem: Operation, workspace: []T, comptime contract: Contract) Error!Result {
    _ = operation.borrowedContext();
    const a_span = try span(T, a);
    const b_span = try span(T, b);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const rows = @max(m, n);
    const nrhs: usize = @intCast(b.column_count);
    if (@as(usize, @intCast(b.row_count)) != rows) return error.DimensionMismatch;
    const requirement = try plan(T, m, n, nrhs);
    if (workspace.len < requirement.total_elements) return error.BufferTooSmall;
    try validation.noOverlap(T, a.values[0..a_span], T, b.values[0..b_span]);
    try validation.noOverlap(T, a.values[0..a_span], T, workspace[0..requirement.total_elements]);
    try validation.noOverlap(T, b.values[0..b_span], T, workspace[0..requirement.total_elements]);
    if (m == 0 or n == 0 or nrhs == 0) {
        zeroRows(T, b, 0, rows);
        return .{};
    }
    const input_rows = if (problem == .normal) m else n;
    const input_b = prefix(T, b, input_rows);
    if (contract == .checked) {
        if (comptime scalar.isComplex(T)) {
            try complex.finiteMatrix(T, a);
            try complex.finiteMatrix(T, input_b);
        } else {
            try validation.finiteMatrix(T, a);
            try validation.finiteMatrix(T, input_b);
        }
    }
    const anorm = norm(T, a);
    if (anorm == 0) {
        zeroRows(T, b, 0, rows);
        return .{};
    }
    const a_scale = scaleFactor(T, anorm);
    const b_scale = scaleFactor(T, norm(T, input_b));
    scaleMatrix(T, a, a_scale);
    scaleMatrix(T, input_b, b_scale);
    const tau = workspace[0..requirement.tau_elements];
    const scratch = workspace[requirement.tau_elements..requirement.total_elements];
    const info = if (m >= n)
        try qr.factorizeWithContractAdmitted(operation, T, a, tau, contract)
    else
        try lq.factorizeWithContractAdmitted(operation, T, a, tau, scratch, contract);
    if (info != 0) return .{ .info = info, .failure = .arithmetic, .stage = .factorization };
    const k = @min(m, n);
    if (m >= n) {
        if (problem == .normal) {
            const applied = try applyQ(operation, T, a, tau, b, .conjugate_transposed, scratch, contract);
            if (applied.failure != null) return applied;
            const solved = solveTriangle(T, a, prefix(T, b, k), .upper, .normal, contract);
            if (solved.failure != null) return solved;
        } else {
            const solved = solveTriangle(T, a, prefix(T, b, k), .upper, .adjoint, contract);
            if (solved.failure != null) return solved;
            zeroRows(T, b, k, rows);
            const applied = try applyQ(operation, T, a, tau, b, .normal, scratch, contract);
            if (applied.failure != null) return applied;
        }
    } else if (problem == .normal) {
        const solved = solveTriangle(T, a, prefix(T, b, k), .lower, .normal, contract);
        if (solved.failure != null) return solved;
        zeroRows(T, b, k, rows);
        const applied = try applyQ(operation, T, a, tau, b, .conjugate_transposed, scratch, contract);
        if (applied.failure != null) return applied;
    } else {
        const applied = try applyQ(operation, T, a, tau, b, .normal, scratch, contract);
        if (applied.failure != null) return applied;
        const solved = solveTriangle(T, a, prefix(T, b, k), .lower, .adjoint, contract);
        if (solved.failure != null) return solved;
    }
    const solution_rows = if (problem == .normal) n else m;
    const ld: usize = @intCast(b.leading_dimension);
    const factor = a_scale / b_scale;
    // The widened combined ratio avoids an overflowing intermediate rescale.
    // A checked rescale rejects before any rescaled solution store.
    if (contract == .checked) for (0..nrhs) |j| for (0..solution_rows) |i| {
        if (!finite(T, scaled(T, b.values[i + j * ld], factor))) return .{ .info = i + 1, .failure = .arithmetic, .stage = .rescaling, .rhs_column = j + 1 };
    };
    scaleMatrix(T, prefix(T, b, solution_rows), factor);
    return .{};
}
