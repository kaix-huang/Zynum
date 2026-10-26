// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Prefactored QR least squares: real m>=n systems only.
//! No allocation, column pivoting, rank estimate, or underdetermined solve.

const execution = @import("zynum-execution");
const std = @import("std");
const storage = @import("../numerics/storage.zig");
const views = @import("../blas/api/views.zig");
const validation = @import("validation.zig");
const qr = @import("qr.zig");
pub const Matrix = views.Matrix;
pub const ConstMatrix = views.ConstMatrix;
pub const Error = validation.Error;

fn matrixSpan(comptime T: type, value: anytype) Error!usize {
    if (T != f32 and T != f64) @compileError("least squares supports f32 and f64 only");
    if (value.row_count < 0 or value.column_count < 0) return error.InvalidLength;
    if (value.leading_dimension < 0) return error.InvalidLeadingDimension;
    if (value.row_count == 0 or value.column_count == 0) return 0;
    if (value.leading_dimension < value.row_count) return error.InvalidLeadingDimension;
    const rows: usize = @intCast(value.row_count);
    const columns: usize = @intCast(value.column_count);
    const ld: usize = @intCast(value.leading_dimension);
    const used = storage.columnMajorExtent(rows, columns, ld) catch return error.InvalidLength;
    if (value.values.len < used) return error.BufferTooSmall;
    return used;
}

fn disjoint(comptime T: type, first: []const T, second: []const T) Error!void {
    const first_size = storage.byteLength(first.len, @sizeOf(T)) catch return error.InvalidLength;
    const second_size = storage.byteLength(second.len, @sizeOf(T)) catch return error.InvalidLength;
    const first_start = @intFromPtr(first.ptr);
    const second_start = @intFromPtr(second.ptr);
    const first_range = storage.byteRange(first_start, first_size) catch return error.InvalidLength;
    const second_range = storage.byteRange(second_start, second_size) catch return error.InvalidLength;
    if (storage.overlaps(first_range, second_range)) return error.AliasingNotAllowed;
}

/// Solve min ||A*X-B||_2 using matching successful public QR factors for
/// m-by-n A, m>=n. Square systems are included. factor.operation must be normal.
/// tau[0..n] is the used coefficient prefix; factor and tau are read-only.
/// RHS has m rows: on success rows [0,n) hold X and [n,m) retain the trailing
/// coordinates of Q^T*B. Their squared norm equals the residual squared norm
/// in exact arithmetic; they are NOT residual entries in original row space.
/// Matrix padding and unused trailing storage are not written directly.
///
/// Full shape, storage, finite-input, reflector-range, and pairwise used-span
/// overlap checks precede mutation in every build. Factor span includes
/// intervening padding; only tau's used prefix participates. Full logical R
/// entries must be finite, unlike qrApplyQ which does not use R.
///
/// For nonempty RHS, an exactly zero R diagonal returns its first one-based
/// index before RHS mutation. No near-zero threshold or rank estimate is used;
/// a caller must establish full column rank and assess conditioning. Zero RHS
/// is a validated no-op; n=0 leaves B unchanged as the entire residual tail.
///
/// Zero means success. Nonfinite arithmetic returns a one-based reflector
/// index during Q^T application, or a one-based row during back substitution.
/// Numerical failure may leave partial/transformed RHS and no solution is
/// promised. Argument errors and zero-diagonal rejection leave RHS unchanged.
pub fn solve(comptime T: type, factor: ConstMatrix(T), tau: []const T, rhs: Matrix(T)) Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    const factor_span = try matrixSpan(T, factor);
    const rhs_span = try matrixSpan(T, rhs);
    if (factor.operation != .normal) return error.InvalidMatrixOperation;
    if (factor.row_count < factor.column_count or rhs.row_count != factor.row_count) return error.DimensionMismatch;
    const m: usize = @intCast(factor.row_count);
    const n: usize = @intCast(factor.column_count);
    const nrhs: usize = @intCast(rhs.column_count);
    if (tau.len < n) return error.BufferTooSmall;
    try disjoint(T, factor.values[0..factor_span], tau[0..n]);
    try disjoint(T, factor.values[0..factor_span], rhs.values[0..rhs_span]);
    try disjoint(T, tau[0..n], rhs.values[0..rhs_span]);
    const lda: usize = @intCast(factor.leading_dimension);
    const ldb: usize = @intCast(rhs.leading_dimension);
    if (m == 0) return 0;
    for (0..n) |j| {
        if (!std.math.isFinite(tau[j])) return error.NonFiniteInput;
        if (tau[j] < 0 or tau[j] > 2) return error.InvalidReflector;
        for (0..m) |i| {
            const value = factor.values[i + j * lda];
            if (!std.math.isFinite(value)) return error.NonFiniteInput;
            if (i > j and @abs(value) > 1) return error.InvalidReflector;
        }
    }
    for (0..nrhs) |j| for (0..m) |i| {
        if (!std.math.isFinite(rhs.values[i + j * ldb])) return error.NonFiniteInput;
    };
    if (nrhs == 0 or n == 0) return 0;
    for (0..n) |i| if (factor.values[i + i * lda] == 0) return i + 1;

    // The internal composition repeats the same checked preflight while
    // borrowing this operation lifetime. No unchecked QR path is used.
    const info = try qr.applyQAdmitted(&admission, T, factor, tau, rhs, .transposed);
    if (info != 0) return info;
    for (0..nrhs) |column| {
        var remaining = n;
        while (remaining != 0) {
            remaining -= 1;
            const row = remaining;
            var value = rhs.values[row + column * ldb];
            for (row + 1..n) |k| value -= factor.values[row + k * lda] * rhs.values[k + column * ldb];
            value /= factor.values[row + row * lda];
            if (!std.math.isFinite(value)) return row + 1;
            rhs.values[row + column * ldb] = value;
        }
    }
    return 0;
}
