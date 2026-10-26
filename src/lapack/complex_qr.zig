// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compact complex Householder QR using the shared scalar, storage and lifetime.
//! H = I - tau*v*v^H; H^H annihilates the input tail and beta is real.
//! Checked argument errors precede writes. Numerical failures return a one-based
//! reflector index; earlier reflectors may already have been committed.
const std = @import("std");
const s = @import("complex_support.zig");
const execution = @import("zynum-execution");
const QOperation = @import("qr.zig").QOperation;
const QSide = @import("qr.zig").QSide;
const Contract = @import("qr.zig").Contract;
fn project(comptime C: type, z: s.WideComplex(C), comptime contract: Contract) ?C {
    if (comptime contract == .checked) return s.narrow(C, z);
    return .{ .re = @floatCast(z.re), .im = @floatCast(z.im) };
}

fn reflector(comptime C: type, column: []C, tau: *C, comptime contract: Contract) bool {
    const W = s.Wide(C);
    var scale: W = 0;
    var tail_nonzero = false;
    for (column, 0..) |z, i| {
        scale = @max(scale, @max(@abs(@as(W, z.re)), @abs(@as(W, z.im))));
        if (i != 0) tail_nonzero = tail_nonzero or z.re != 0 or z.im != 0;
    }
    if (!tail_nonzero and column[0].im == 0) {
        tau.* = .{ .re = 0, .im = 0 };
        return true;
    }
    var sum: W = 0;
    for (column) |z| {
        const re = @as(W, z.re) / scale;
        const im = @as(W, z.im) / scale;
        sum += re * re + im * im;
    }
    const alpha = s.widen(C, column[0]);
    const beta_normalized = -std.math.copysign(@sqrt(sum), alpha.re);
    const beta = project(C, .{ .re = scale * beta_normalized, .im = 0 }, contract) orelse return false;
    const ar = alpha.re / scale;
    const ai = alpha.im / scale;
    const coefficient = project(C, .{ .re = 1 - ar / beta_normalized, .im = -ai / beta_normalized }, contract) orelse return false;
    const dr = ar - beta_normalized;
    const di = ai;
    const den = dr * dr + di * di;
    // Preflight all narrowed tails before changing this reflector.
    for (column[1..]) |z| {
        const zr = @as(W, z.re) / scale;
        const zi = @as(W, z.im) / scale;
        _ = project(C, .{ .re = (zr * dr + zi * di) / den, .im = (zi * dr - zr * di) / den }, contract) orelse return false;
    }
    for (column[1..]) |*z| {
        const zr = @as(W, z.re) / scale;
        const zi = @as(W, z.im) / scale;
        z.* = project(C, .{ .re = (zr * dr + zi * di) / den, .im = (zi * dr - zr * di) / den }, contract).?;
    }
    column[0] = beta;
    tau.* = coefficient;
    return true;
}

fn applyReflector(comptime C: type, tail: []const C, tau: C, vector: []C, conjugate_vector: bool) bool {
    return applyReflectorStrided(C, tail, tau, vector, 1, conjugate_vector, .checked);
}

fn applyReflectorStrided(comptime C: type, tail: []const C, tau: C, vector: []C, stride: usize, conjugate_vector: bool, comptime contract: Contract) bool {
    if (tau.re == 0 and tau.im == 0) return true;
    const W = s.Wide(C);
    var dot = s.widen(C, vector[0]);
    for (tail, 0..) |stored, i| {
        const z = vector[(i + 1) * stride];
        const vr: W = stored.re;
        const vi: W = if (conjugate_vector) -@as(W, stored.im) else @as(W, stored.im);
        dot.re += vr * @as(W, z.re) + vi * @as(W, z.im);
        dot.im += vr * @as(W, z.im) - vi * @as(W, z.re);
    }
    const tr: W = tau.re;
    const ti: W = tau.im;
    const wr = tr * dot.re - ti * dot.im;
    const wi = tr * dot.im + ti * dot.re;
    // Two passes make an individual vector update atomic on range failure.
    for (0..2) |pass| for (0..tail.len + 1) |i| {
        const z = &vector[i * stride];
        const vr: W = if (i == 0) 1 else tail[i - 1].re;
        const vi: W = if (i == 0) 0 else if (conjugate_vector) -@as(W, tail[i - 1].im) else @as(W, tail[i - 1].im);
        const result = project(C, .{ .re = @as(W, z.re) - (vr * wr - vi * wi), .im = @as(W, z.im) - (vr * wi + vi * wr) }, contract) orelse return false;
        if (pass == 1) z.* = result;
    };
    return true;
}

pub fn factorize(comptime C: type, a: s.views.Matrix(C), tau: []C) s.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    return factorizeAdmitted(&admission, C, a, tau);
}
pub fn factorizeAdmitted(admission: *const execution.context.Operation, comptime C: type, a: s.views.Matrix(C), tau: []C) s.Error!usize {
    return factorizeWithContractAdmitted(admission, C, a, tau, .checked);
}
pub fn factorizeWithContractAdmitted(admission: *const execution.context.Operation, comptime C: type, a: s.views.Matrix(C), tau: []C, comptime contract: Contract) s.Error!usize {
    _ = admission.borrowedContext();
    const used = try s.matrixSpan(C, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const k = @min(m, n);
    if (tau.len < k) return error.BufferTooSmall;
    try s.noOverlap(C, a.values[0..used], C, tau[0..k]);
    if (contract == .checked) try s.finiteMatrix(C, a);
    const ld: usize = @intCast(a.leading_dimension);
    for (0..k) |step| {
        const column = a.values[step + step * ld ..][0 .. m - step];
        if (!reflector(C, column, &tau[step], contract)) return step + 1;
        const t = s.scalar.conj(C, tau[step]);
        for (step + 1..n) |j| {
            if (!applyReflectorStrided(C, column[1..], t, a.values[step + j * ld ..][0 .. m - step], 1, false, contract)) return step + 1;
        }
    }
    return 0;
}
pub fn applyQSideWithContractAdmitted(admission: *const execution.context.Operation, comptime C: type, factor: s.views.ConstMatrix(C), tau: []const C, b: s.views.Matrix(C), side: QSide, operation: QOperation, comptime contract: Contract) s.Error!usize {
    _ = admission.borrowedContext();
    const factor_span = try s.matrixSpan(C, factor);
    const b_span = try s.matrixSpan(C, b);
    if (factor.operation != .normal) return error.InvalidMatrixOperation;
    if ((if (side == .left) b.row_count else b.column_count) != factor.row_count) return error.DimensionMismatch;
    const m: usize = @intCast(factor.row_count);
    const n: usize = @intCast(factor.column_count);
    const k = @min(m, n);
    if (tau.len < k) return error.BufferTooSmall;
    try s.noOverlap(C, factor.values[0..factor_span], C, tau[0..k]);
    try s.noOverlap(C, factor.values[0..factor_span], C, b.values[0..b_span]);
    try s.noOverlap(C, tau[0..k], C, b.values[0..b_span]);
    const lda: usize = @intCast(factor.leading_dimension);
    const ldb: usize = @intCast(b.leading_dimension);
    if (contract == .checked) for (0..k) |step| {
        const t = tau[step];
        if (!s.finite(C, t)) return error.NonFiniteInput;
        if (t.re < 0 or t.re > 2 or @abs(t.im) > 1) return error.InvalidReflector;
        for (step + 1..m) |i| {
            const z = factor.values[i + step * lda];
            if (!s.finite(C, z)) return error.NonFiniteInput;
            const zw = s.widen(C, z);
            if (zw.re * zw.re + zw.im * zw.im > 1) return error.InvalidReflector;
        }
    };
    if (contract == .checked) try s.finiteMatrix(C, b);
    const nrhs: usize = @intCast(if (side == .left) b.column_count else b.row_count);
    for (0..k) |ordinal| {
        const step = if ((operation == .normal) == (side == .left)) k - 1 - ordinal else ordinal;
        const t = if (operation == .conjugate_transposed) s.scalar.conj(C, tau[step]) else tau[step];
        const tail = factor.values[step + step * lda + 1 ..][0 .. m - step - 1];
        for (0..nrhs) |j| {
            const start = if (side == .left) step + j * ldb else j + step * ldb;
            const stride = if (side == .left) 1 else ldb;
            const conjugate_vector = if (side == .left) operation == .transposed else operation != .transposed;
            if (!applyReflectorStrided(C, tail, t, b.values[start..], stride, conjugate_vector, contract)) return step + 1;
        }
    }
    return 0;
}

pub fn generateQWithContractAdmitted(operation: *const execution.context.Operation, comptime C: type, a: s.views.Matrix(C), tau: []const C, k: usize, comptime contract: Contract) s.Error!usize {
    _ = operation.borrowedContext();
    const span = try s.matrixSpan(C, a);
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    if (n > m or k > n) return error.DimensionMismatch;
    if (tau.len < k) return error.BufferTooSmall;
    try s.noOverlap(C, a.values[0..span], C, tau[0..k]);
    const ld: usize = @intCast(a.leading_dimension);
    if (contract == .checked) for (0..k) |step| {
        const t = tau[step];
        if (!s.finite(C, t)) return error.NonFiniteInput;
        if (t.re < 0 or t.re > 2 or @abs(t.im) > 1) return error.InvalidReflector;
        for (step + 1..m) |i| {
            const v = a.values[i + step * ld];
            if (!s.finite(C, v)) return error.NonFiniteInput;
            const w = s.widen(C, v);
            if (w.re * w.re + w.im * w.im > 1) return error.InvalidReflector;
        }
    };
    for (k..n) |j| for (0..m) |i| {
        a.values[i + j * ld] = .{ .re = if (i == j) 1 else 0, .im = 0 };
    };
    for (0..k) |ordinal| {
        const step = k - 1 - ordinal;
        const tail = a.values[step + step * ld + 1 ..][0 .. m - step - 1];
        for (step + 1..n) |j| {
            if (!applyReflectorStrided(C, tail, tau[step], a.values[step + j * ld ..], 1, false, contract)) return step + 1;
        }
        const t = s.widen(C, tau[step]);
        // Each tail projection is checked before changing this output column.
        for (0..2) |pass| for (tail) |*v| {
            const w = s.widen(C, v.*);
            const z = project(C, .{ .re = -(t.re * w.re - t.im * w.im), .im = -(t.re * w.im + t.im * w.re) }, contract) orelse return step + 1;
            if (pass == 1) v.* = z;
        };
        a.values[step + step * ld] = .{ .re = 1 - tau[step].re, .im = -tau[step].im };
        for (0..step) |i| a.values[i + step * ld] = .{ .re = 0, .im = 0 };
    }
    return 0;
}
