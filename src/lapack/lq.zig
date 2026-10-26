// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Caller-workspace LQ composition over the existing QR implementation.
//! A = L*Q uses LAPACK row reflectors: Q = H(k)^H ... H(1)^H;
//! conjugated reflector tails are stored above A's diagonal, tau is unchanged.
//! No allocator, Context, thread pool, cache or external backend is created.

const std = @import("std");
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const views = @import("../blas/api/views.zig");
const scalar = @import("../blas/core/shared/scalar.zig");
const real = @import("validation.zig");
const complex = @import("complex_support.zig");
const qr = @import("qr.zig");
pub const Error = real.Error;
pub const Contract = qr.Contract;
pub const QOperation = qr.QOperation;
pub const QSide = qr.QSide;

fn expectScalar(comptime T: type) void {
    if (T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("LQ supports real/complex f32 and f64");
}

pub fn factorWorkspaceLength(comptime T: type, m: usize, n: usize) Error!usize {
    comptime expectScalar(T);
    const count = std.math.mul(usize, m, n) catch return error.InvalidLength;
    _ = storage.byteLength(count, @sizeOf(T)) catch return error.InvalidLength;
    return count;
}

pub fn applyWorkspaceLength(comptime T: type, m: usize, n: usize) Error!usize {
    const count = std.math.add(usize, try factorWorkspaceLength(T, m, n), @min(m, n)) catch return error.InvalidLength;
    _ = storage.byteLength(count, @sizeOf(T)) catch return error.InvalidLength;
    return count;
}

fn matrixSpan(comptime T: type, a: anytype) Error!usize {
    return if (comptime scalar.isComplex(T)) complex.matrixSpan(T, a) else real.matrixSpan(T, a);
}

fn finiteMatrix(comptime T: type, a: anytype) Error!void {
    if (comptime scalar.isComplex(T)) return complex.finiteMatrix(T, a);
    return real.finiteMatrix(T, a);
}

fn transposedMatrix(comptime T: type, values: []T, m: usize, n: usize) views.Matrix(T) {
    return .{ .values = values, .row_count = @intCast(n), .column_count = @intCast(m), .leading_dimension = @intCast(n) };
}

/// Factor arbitrary rectangular A. Workspace uses exactly m*n scalar elements;
/// tau uses min(m,n). Padding, unused tau and workspace tails stay untouched.
/// Shape, capacity, alias and checked finite-input errors precede all writes.
/// A positive result is the failing QR reflector index; partial factors and tau
/// are committed in LQ representation and are not usable as a completed factor.
pub fn factorize(comptime T: type, a: views.Matrix(T), tau: []T, workspace: []T) Error!usize {
    var operation = execution.beginDefault();
    defer operation.end();
    return factorizeAdmitted(&operation, T, a, tau, workspace);
}

pub fn factorizeAdmitted(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []T, workspace: []T) Error!usize {
    return factorizeWithContractAdmitted(operation, T, a, tau, workspace, .checked);
}

pub fn factorizeWithContractAdmitted(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []T, workspace: []T, comptime contract: Contract) Error!usize {
    _ = operation.borrowedContext();
    const span = try matrixSpan(T, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const k = @min(m, n);
    const count = try factorWorkspaceLength(T, m, n);
    if (tau.len < k or workspace.len < count) return error.BufferTooSmall;
    try real.noOverlap(T, a.values[0..span], T, tau[0..k]);
    try real.noOverlap(T, a.values[0..span], T, workspace[0..count]);
    try real.noOverlap(T, tau[0..k], T, workspace[0..count]);
    if (contract == .checked) try finiteMatrix(T, a);
    const ld: usize = @intCast(a.leading_dimension);
    const temporary = workspace[0..count];
    for (0..n) |column| for (0..m) |row| {
        temporary[column + row * n] = scalar.conj(T, a.values[row + column * ld]);
    };
    const info = try qr.factorizeWithContractAdmitted(operation, T, transposedMatrix(T, temporary, m, n), tau[0..k], contract);
    for (0..n) |column| for (0..m) |row| {
        a.values[row + column * ld] = scalar.conj(T, temporary[column + row * n]);
    };
    return info;
}

// Validate row-stored reflectors before projecting them to QR workspace. These
// are the existing QR canonical reflector bounds, with row/column coordinates
// exchanged. Lower-trapezoidal L coefficients and padding are never read.
fn checkReflectors(comptime T: type, factor: views.ConstMatrix(T), tau: []const T, k: usize, n: usize) Error!void {
    const ld: usize = @intCast(factor.leading_dimension);
    for (0..k) |step| {
        const t = tau[step];
        if (comptime scalar.isComplex(T)) {
            if (!complex.finite(T, t)) return error.NonFiniteInput;
            if (t.re < 0 or t.re > 2 or @abs(t.im) > 1) return error.InvalidReflector;
        } else {
            if (!std.math.isFinite(t)) return error.NonFiniteInput;
            if (t < 0 or t > 2) return error.InvalidReflector;
        }
        for (step + 1..n) |column| {
            const value = factor.values[step + column * ld];
            if (comptime scalar.isComplex(T)) {
                if (!complex.finite(T, value)) return error.NonFiniteInput;
                const wide = complex.widen(T, value);
                if (wide.re * wide.re + wide.im * wide.im > 1) return error.InvalidReflector;
            } else {
                if (!std.math.isFinite(value)) return error.NonFiniteInput;
                if (@abs(value) > 1) return error.InvalidReflector;
            }
        }
    }
}

/// Apply the n-by-n Q of a successful m-by-n compact LQ factor to B from
/// either side, with normal, transpose or adjoint operation. Only tau's k-prefix
/// and strict upper row tails are read; L may contain NaNs. B padding is kept.
/// Workspace needs m*n+k elements, including conjugated tau for complex Q^T.
/// All argument/value validation precedes every B/workspace write. Numerical
/// failure may leave B partially transformed and returns a reflector index.
pub fn applyQ(comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, q_operation: QOperation, workspace: []T) Error!usize {
    var operation = execution.beginDefault();
    defer operation.end();
    return applyQAdmitted(&operation, T, factor, tau, b, side, q_operation, workspace);
}

pub fn applyQAdmitted(operation: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, q_operation: QOperation, workspace: []T) Error!usize {
    return applyQWithContractAdmitted(operation, T, factor, tau, b, side, q_operation, workspace, .checked);
}

pub fn applyQWithContractAdmitted(operation: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, q_operation: QOperation, workspace: []T, comptime contract: Contract) Error!usize {
    _ = operation.borrowedContext();
    const span = try matrixSpan(T, factor);
    const b_span = try matrixSpan(T, b);
    if (factor.operation != .normal) return error.InvalidMatrixOperation;
    if ((if (side == .left) b.row_count else b.column_count) != factor.column_count) return error.DimensionMismatch;
    const m: usize = @intCast(factor.row_count);
    const n: usize = @intCast(factor.column_count);
    const k = @min(m, n);
    const count = try applyWorkspaceLength(T, m, n);
    if (tau.len < k or workspace.len < count) return error.BufferTooSmall;
    const ranges = [_][]const T{ factor.values[0..span], tau[0..k], b.values[0..b_span], workspace[0..count] };
    for (0..ranges.len) |first| for (first + 1..ranges.len) |second| {
        try real.noOverlap(T, ranges[first], T, ranges[second]);
    };
    if (contract == .checked) {
        try checkReflectors(T, factor, tau, k, n);
        try finiteMatrix(T, b);
    }
    const projected_tau = workspace[0..k];
    const projected_factor = workspace[k..count];
    @memset(projected_factor, scalar.zero(T));
    const ld: usize = @intCast(factor.leading_dimension);
    for (0..k) |step| {
        projected_tau[step] = if (q_operation == .transposed) scalar.conj(T, tau[step]) else tau[step];
        for (step + 1..n) |column| {
            const value = factor.values[step + column * ld];
            projected_factor[column + step * n] = if (q_operation == .transposed) value else scalar.conj(T, value);
        }
    }
    const qr_operation: QOperation = if (q_operation == .normal) .conjugate_transposed else .normal;
    return qr.applyQSideWithContractAdmitted(operation, T, transposedMatrix(T, projected_factor, m, n).asConst(), projected_tau, b, side, qr_operation, contract);
}
