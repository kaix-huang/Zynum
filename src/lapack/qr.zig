// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Allocation-free unblocked real/complex Householder QR. Compact factors use the
//! borrowed column-major LAPACK views. There is no pivoting or rank decision.
//! Argument errors precede every mutation, including in optimized builds.
//! Numerical failure returns the one-based reflector index and may leave a
//! partial result. Zero means success, including rank-deficient matrices.

const execution = @import("zynum-execution");
const std = @import("std");
const views = @import("../blas/api/views.zig");
const validation = @import("validation.zig");
pub const QOperation = enum { normal, transposed, conjugate_transposed };
pub const QSide = enum { left, right };
/// Closed numerical policy, independent of external signature or resource ownership.
/// IEEE projection may store infinities/NaNs and returns zero on arithmetic range
/// events. It does not promise vendor payload, rounding or exception trajectories.
pub const Contract = enum { checked, lapack_ieee_v1 };

fn compensatedAdd(sum: *f64, correction: *f64, value: f64) void {
    const adjusted = value - correction.*;
    const next = sum.* + adjusted;
    correction.* = (next - sum.*) - adjusted;
    sum.* = next;
}

// Construct H=I-tau*v*v^T with v[0]=1. Divide by the input scale before
// computing both norm and v: neither alpha-beta nor an unscaled norm square
// is formed. Reflector parameters remain accurate for subnormal input even
// when the stored beta must round to subnormal precision. f32 uses f64 working
// scalars; f64 uses its own precision and range, without extended arithmetic.
fn makeReflector(comptime T: type, column: []T, tau: *T, comptime contract: Contract) bool {
    var has_tail = false;
    var scale: f64 = @abs(@as(f64, @floatCast(column[0])));
    for (column[1..]) |entry| {
        has_tail = has_tail or entry != 0;
        scale = @max(scale, @abs(@as(f64, @floatCast(entry))));
    }
    if (!has_tail) {
        tau.* = 0;
        return true;
    }
    var sum: f64 = 0;
    var correction: f64 = 0;
    for (column) |entry| {
        const normalized = @as(f64, @floatCast(entry)) / scale;
        compensatedAdd(&sum, &correction, normalized * normalized);
    }
    const alpha = @as(f64, @floatCast(column[0])) / scale;
    const beta_normalized = -std.math.copysign(@sqrt(sum), @as(f64, @floatCast(column[0])));
    const beta: T = @floatCast(scale * beta_normalized);
    if (contract == .checked and !std.math.isFinite(beta)) return false;
    const coefficient: T = @floatCast(1 - alpha / beta_normalized);
    const denominator = alpha - beta_normalized;
    for (column[1..]) |*entry| entry.* = @floatCast((@as(f64, @floatCast(entry.*)) / scale) / denominator);
    column[0] = beta;
    tau.* = coefficient;
    return true;
}

// Apply a compact reflector to one vector. Products are normalized by their
// binary exponents, not by the largest vector entry: a huge entry multiplied
// by a zero/tiny reflector must not erase an unrelated small contribution.
// The dot product and subsequent correction can exceed/undershoot f64 range
// temporarily. Keep that scale as an exponent and split the final correction
// between two normal FMA operands to avoid premature range loss in that step.
// Significand products and the compensated dot are still rounded; tiny terms
// can be lost when much larger terms later cancel. This is not an exact dot.
fn reflect(comptime T: type, tail: []const T, tau: T, vector: []T) bool {
    return reflectStrided(T, tail, tau, vector, 1);
}

fn reflectStrided(comptime T: type, tail: []const T, tau: T, vector: []T, stride: usize) bool {
    return reflectStridedWithContract(T, tail, tau, vector, stride, .checked);
}

// The IEEE leaf uses widened direct products for nonfinite/unchecked coefficients.
// No integer decomposition of NaN/Inf, allocation, retry or invented INFO occurs.
fn reflectIeee(comptime T: type, tail: []const T, tau: T, vector: []T, stride: usize) bool {
    if (tau == 0) return true;
    const W = if (T == f32) f64 else f128;
    var dot: W = vector[0];
    for (tail, 0..) |v, i| dot += @as(W, v) * @as(W, vector[(i + 1) * stride]);
    const weight = @as(W, tau) * dot;
    for (0..tail.len + 1) |i| {
        const v: W = if (i == 0) 1 else tail[i - 1];
        vector[i * stride] = @floatCast(@as(W, vector[i * stride]) - v * weight);
    }
    return true;
}
fn reflectStridedWithContract(comptime T: type, tail: []const T, tau: T, vector: []T, stride: usize, comptime contract: Contract) bool {
    if (comptime contract == .lapack_ieee_v1) {
        if (!std.math.isFinite(tau)) return reflectIeee(T, tail, tau, vector, stride);
        for (tail) |v| if (!std.math.isFinite(v)) return reflectIeee(T, tail, tau, vector, stride);
        for (0..tail.len + 1) |i| if (!std.math.isFinite(vector[i * stride])) return reflectIeee(T, tail, tau, vector, stride);
    }
    if (tau == 0) return true;
    var largest_exponent: i32 = std.math.minInt(i32);
    for (0..tail.len + 1) |i| {
        const entry = vector[i * stride];
        const v: f64 = if (i == 0) 1 else @floatCast(tail[i - 1]);
        if (v == 0 or entry == 0) continue;
        const left = std.math.frexp(v);
        const right = std.math.frexp(@as(f64, @floatCast(entry)));
        largest_exponent = @max(largest_exponent, left.exponent + right.exponent);
    }
    if (largest_exponent == std.math.minInt(i32)) return true;
    var dot: f64 = 0;
    var correction: f64 = 0;
    for (0..tail.len + 1) |i| {
        const entry = vector[i * stride];
        const v: f64 = if (i == 0) 1 else @floatCast(tail[i - 1]);
        if (v == 0 or entry == 0) continue;
        const left = std.math.frexp(v);
        const right = std.math.frexp(@as(f64, @floatCast(entry)));
        const term = std.math.ldexp(left.significand * right.significand, left.exponent + right.exponent - largest_exponent);
        compensatedAdd(&dot, &correction, term);
    }
    const weight = @as(f64, @floatCast(tau)) * dot;
    if (!std.math.isFinite(weight)) {
        if (comptime contract == .lapack_ieee_v1) return reflectIeee(T, tail, tau, vector, stride);
        return false;
    }
    if (weight == 0) return true;
    const normalized_weight = std.math.frexp(weight);
    for (0..tail.len + 1) |i| {
        const entry = &vector[i * stride];
        const v: f64 = if (i == 0) 1 else @floatCast(tail[i - 1]);
        if (v == 0) continue;
        const normalized_v = std.math.frexp(v);
        const exponent = normalized_v.exponent + normalized_weight.exponent + largest_exponent;
        // Significand product is in [1/4,1). Above this bound, no finite
        // input could cancel enough of the correction to yield finite f64.
        if (exponent > 1026) {
            if (comptime contract == .checked) return false;
            const W = if (T == f32) f64 else f128;
            const change = std.math.ldexp(@as(W, normalized_v.significand) * @as(W, normalized_weight.significand), exponent);
            entry.* = @floatCast(@as(W, entry.*) - change);
            continue;
        }
        // A correction strictly below half the smallest subnormal cannot
        // change a representable entry under round-to-nearest semantics.
        if (exponent <= -1075) continue;
        // The correction is p*2^exponent with |p|<1. For a finite nonzero
        // f64 entry with frexp exponent e, even the smaller spacing toward
        // zero at a power of two is at least 2^(e-54). This bound is strictly
        // below half that spacing, so rounding must preserve the entry.
        // Promoted f32 entries have no finer spacing, making it conservative.
        const widened_entry: f64 = @floatCast(entry.*);
        if (widened_entry != 0 and std.math.isFinite(widened_entry) and
            exponent <= std.math.frexp(widened_entry).exponent - 55) continue;
        const first_exponent = @divFloor(exponent, 2);
        const first = std.math.ldexp(normalized_v.significand * normalized_weight.significand, first_exponent);
        const second = std.math.ldexp(@as(f64, 1), exponent - first_exponent);
        const updated: T = @floatCast(@mulAdd(f64, -first, second, @as(f64, @floatCast(entry.*))));
        if (contract == .checked and !std.math.isFinite(updated)) return false;
        entry.* = updated;
    }
    return true;
}

/// Factor m-by-n A as Q*R, Q=H[0]*...*H[k-1], k=min(m,n). R is the upper
/// trapezoid; strictly lower column i stores H[i]'s tail, with implicit leading
/// one; tau[0..k] holds reflector coefficients. tau's unused tail and matrix
/// padding are preserved. tau must be disjoint from A's used physical span
/// (including intervening padding). Empty matrices accept nonnegative ld,
/// including zero. All logical input entries must be finite before mutation.
/// Rank deficiency is legal. If a result cannot be represented finitely,
/// returns its one-based reflector index; partial factors are not usable.
pub fn factorize(comptime T: type, a: views.Matrix(T), tau: []T) validation.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    return factorizeAdmitted(&admission, T, a, tau);
}

/// Borrow the caller's live operation; do not readmit during internal composition.
pub fn factorizeAdmitted(admission: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []T) validation.Error!usize {
    return factorizeWithContractAdmitted(admission, T, a, tau, .checked);
}
pub fn factorizeWithContractAdmitted(admission: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []T, comptime contract: Contract) validation.Error!usize {
    if (comptime @import("complex_support.zig").scalar.isComplex(T)) return @import("complex_qr.zig").factorizeWithContractAdmitted(admission, T, a, tau, contract);
    _ = admission.borrowedContext();
    const used = try validation.matrixSpan(T, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const k = @min(m, n);
    if (tau.len < k) return error.BufferTooSmall;
    try validation.noOverlap(T, a.values[0..used], T, tau[0..k]);
    if (contract == .checked) try validation.finiteMatrix(T, a);
    const ld: usize = @intCast(a.leading_dimension);
    for (0..k) |step| {
        const column = a.values[step + step * ld ..][0 .. m - step];
        if (!makeReflector(T, column, &tau[step], contract)) return step + 1;
        for (step + 1..n) |j| {
            if (!reflectStridedWithContract(T, column[1..], tau[step], a.values[step + j * ld ..][0 .. m - step], 1, contract)) return step + 1;
        }
    }
    return 0;
}

/// Overwrite B with Q*B, Q^T*B or Q^H*B using a successful compact QR factor, without
/// materializing Q. Left application only: B has m rows and any RHS count.
/// Reflectors/factor must have operation=.normal. Used factor, tau, and B
/// spans are pairwise disjoint; factor/tau and B padding remain unchanged.
/// Validate every argument, used reflector entry, tau, and logical B entry
/// before mutation, even for zero RHS. Upper R entries are not read, and may
/// be nonfinite. Canonical reflector tails must be finite with abs<=1 and tau
/// finite in [0,2] for real scalars; complex reflector preflight is in complex_qr.zig. Callers are responsible for supplying a matching valid QR
/// factor (these range checks do not prove orthogonality). tau's unused tail
/// is ignored. A numerical failure returns the one-based reflector index
/// (not its execution ordinal) and can leave B partially transformed.
pub fn applyQ(comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), operation: QOperation) validation.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    return applyQAdmitted(&admission, T, factor, tau, b, operation);
}

// Internal composition borrows the caller lifetime; checked preflight remains below.
pub fn applyQAdmitted(admission: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), operation: QOperation) validation.Error!usize {
    return applyQSideAdmitted(admission, T, factor, tau, b, .left, operation);
}

/// Apply Q from either side without transposition copies or additional storage.
pub fn applyQSide(comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, operation: QOperation) validation.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    return applyQSideAdmitted(&admission, T, factor, tau, b, side, operation);
}
pub fn applyQSideAdmitted(admission: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, operation: QOperation) validation.Error!usize {
    return applyQSideWithContractAdmitted(admission, T, factor, tau, b, side, operation, .checked);
}
pub fn applyQSideWithContractAdmitted(admission: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), tau: []const T, b: views.Matrix(T), side: QSide, operation: QOperation, comptime contract: Contract) validation.Error!usize {
    if (comptime @import("complex_support.zig").scalar.isComplex(T)) return @import("complex_qr.zig").applyQSideWithContractAdmitted(admission, T, factor, tau, b, side, operation, contract);
    _ = admission.borrowedContext();
    const factor_span = try validation.matrixSpan(T, factor);
    const b_span = try validation.matrixSpan(T, b);
    if (factor.operation != .normal) return error.InvalidMatrixOperation;
    if ((if (side == .left) b.row_count else b.column_count) != factor.row_count) return error.DimensionMismatch;
    const m: usize = @intCast(factor.row_count);
    const n: usize = @intCast(factor.column_count);
    const k = @min(m, n);
    if (tau.len < k) return error.BufferTooSmall;
    try validation.noOverlap(T, factor.values[0..factor_span], T, tau[0..k]);
    try validation.noOverlap(T, factor.values[0..factor_span], T, b.values[0..b_span]);
    try validation.noOverlap(T, tau[0..k], T, b.values[0..b_span]);
    const lda: usize = @intCast(factor.leading_dimension);
    const ldb: usize = @intCast(b.leading_dimension);
    const nrhs: usize = @intCast(if (side == .left) b.column_count else b.row_count);
    if (contract == .checked) for (0..k) |step| {
        if (!std.math.isFinite(tau[step])) return error.NonFiniteInput;
        if (tau[step] < 0 or tau[step] > 2) return error.InvalidReflector;
        for (step + 1..m) |i| {
            const entry = factor.values[i + step * lda];
            if (!std.math.isFinite(entry)) return error.NonFiniteInput;
            if (@abs(entry) > 1) return error.InvalidReflector;
        }
    };
    if (contract == .checked) try validation.finiteMatrix(T, b);
    for (0..k) |ordinal| {
        const step = if ((operation == .normal) == (side == .left)) k - 1 - ordinal else ordinal;
        const tail = factor.values[step + step * lda + 1 ..][0 .. m - step - 1];
        for (0..nrhs) |j| {
            const start = if (side == .left) step + j * ldb else j + step * ldb;
            const stride = if (side == .left) 1 else ldb;
            if (!reflectStridedWithContract(T, tail, tau[step], b.values[start..], stride, contract)) return step + 1;
        }
    }
    return 0;
}

/// Replace compact reflectors by the first n columns of Q. Requires k<=n<=m.
/// Only tau[0..k] and the strictly lower tails of the first k columns are read;
/// upper R, padding and remaining columns are ignored. All checked preflight
/// errors precede writes. Arithmetic failure reports a one-based reflector and
/// may leave partially generated Q. No workspace allocation or readmission.
pub fn generateQ(comptime T: type, a: views.Matrix(T), tau: []const T, k: usize) validation.Error!usize {
    var operation = execution.beginDefault();
    defer operation.end();
    return generateQWithContractAdmitted(&operation, T, a, tau, k, .checked);
}
pub fn generateQWithContractAdmitted(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), tau: []const T, k: usize, comptime contract: Contract) validation.Error!usize {
    if (comptime @import("complex_support.zig").scalar.isComplex(T)) return @import("complex_qr.zig").generateQWithContractAdmitted(operation, T, a, tau, k, contract);
    _ = operation.borrowedContext();
    const span = try validation.matrixSpan(T, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    if (n > m or k > n) return error.DimensionMismatch;
    if (tau.len < k) return error.BufferTooSmall;
    try validation.noOverlap(T, a.values[0..span], T, tau[0..k]);
    const ld: usize = @intCast(a.leading_dimension);
    if (contract == .checked) for (0..k) |step| {
        if (!std.math.isFinite(tau[step])) return error.NonFiniteInput;
        if (tau[step] < 0 or tau[step] > 2) return error.InvalidReflector;
        for (step + 1..m) |i| {
            const v = a.values[i + step * ld];
            if (!std.math.isFinite(v)) return error.NonFiniteInput;
            if (@abs(v) > 1) return error.InvalidReflector;
        }
    };
    for (k..n) |j| for (0..m) |i| {
        a.values[i + j * ld] = if (i == j) 1 else 0;
    };
    for (0..k) |ordinal| {
        const step = k - 1 - ordinal;
        const tail = a.values[step + step * ld + 1 ..][0 .. m - step - 1];
        for (step + 1..n) |j| {
            if (!reflectStridedWithContract(T, tail, tau[step], a.values[step + j * ld ..], 1, contract)) return step + 1;
        }
        for (tail) |*v| v.* = -tau[step] * v.*;
        a.values[step + step * ld] = 1 - tau[step];
        for (0..step) |i| a.values[i + step * ld] = 0;
    }
    return 0;
}
