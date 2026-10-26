// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure complex Householder reduction, real phase conversion and the existing
//! real tridiagonal IEEE solver. Borrowed widened dense workspace only.
const std = @import("std");
const cx = @import("../../numerics/complex.zig");
const tri = @import("tridiagonal_eigen.zig");
pub const UnconvergedBlock = tri.UnconvergedBlock;
fn conjugate(z: anytype) @TypeOf(z) {
    return .{ .re = z.re, .im = -z.im };
}
fn product(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn subtract(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn scaled(z: anytype, x: @TypeOf(z.re)) @TypeOf(z) {
    return .{ .re = z.re * x, .im = z.im * x };
}
fn magnitude(z: anytype) @TypeOf(z.re) {
    return std.math.hypot(z.re, z.im);
}
/// h, q, scratch each contain n*n widened complex entries and are disjoint.
/// h is Hermitian with real diagonal. On exit its diagonal/subdiagonal contain
/// the real iteration outputs and q the accumulated complex basis. A positive
/// unconverged result counts actual nonzero tridiagonal off-diagonals.
/// scratch is temporarily reused as native real storage after reduction.
pub fn solve(comptime C: type, h: []cx.WideComplex(C), q: []cx.WideComplex(C), scratch: []cx.WideComplex(C), n: usize, max_iterations: usize) tri.Result {
    const Z = cx.WideComplex(C);
    const W = cx.Wide(C);
    const R = cx.scalar.Real(C);
    for (0..n) |j| for (0..n) |i| {
        q[i + j * n] = .{ .re = if (i == j) 1 else 0, .im = 0 };
    };
    if (n == 0) return .{ .status = .converged };
    if (n == 1) return .{ .status = .converged };
    for (0..n - 1) |k| {
        const m = n - k - 1;
        const u = scratch[0..m];
        const y = scratch[n .. n + m];
        var norm: W = 0;
        for (0..m) |i| {
            u[i] = h[k + 1 + i + k * n];
            norm = std.math.hypot(norm, magnitude(u[i]));
        }
        if (norm == 0) continue;
        const first = magnitude(u[0]);
        const phase: Z = if (first == 0) .{ .re = 1, .im = 0 } else scaled(u[0], 1 / first);
        const alpha = scaled(phase, -norm);
        u[0] = subtract(u[0], alpha);
        var unorm: W = 0;
        for (u) |z| {
            unorm = std.math.hypot(unorm, magnitude(z));
        }
        for (u) |*z| {
            z.* = scaled(z.*, 1 / unorm);
        }
        for (0..m) |i| {
            var value: Z = .{ .re = 0, .im = 0 };
            for (0..m) |j| {
                value = add(value, product(h[k + 1 + i + (k + 1 + j) * n], u[j]));
            }
            y[i] = value;
        }
        var dot: W = 0;
        for (0..m) |i| {
            dot += product(conjugate(u[i]), y[i]).re;
        }
        for (0..m) |i| {
            y[i] = subtract(y[i], scaled(u[i], dot));
        }
        for (0..m) |j| for (0..m) |i| {
            const at = k + 1 + i + (k + 1 + j) * n;
            h[at] = subtract(h[at], scaled(add(product(u[i], conjugate(y[j])), product(y[i], conjugate(u[j]))), 2));
            if (i == j) h[at].im = 0;
        };
        for (0..n) |i| {
            var projection: Z = .{ .re = 0, .im = 0 };
            for (0..m) |j| {
                projection = add(projection, product(q[i + (k + 1 + j) * n], u[j]));
            }
            for (0..m) |j| {
                q[i + (k + 1 + j) * n] = subtract(q[i + (k + 1 + j) * n], scaled(product(projection, conjugate(u[j])), 2));
            }
        }
        h[k + 1 + k * n] = alpha;
        h[k + (k + 1) * n] = conjugate(alpha);
        for (k + 2..n) |i| {
            h[i + k * n] = .{ .re = 0, .im = 0 };
            h[k + i * n] = .{ .re = 0, .im = 0 };
        }
    }
    // Exact workspace geometry is checked by the enclosing Hermitian leaf.
    const pointer: [*]R = @ptrCast(@alignCast(scratch.ptr));
    const real = pointer[0 .. scratch.len * @sizeOf(Z) / @sizeOf(R)];
    const d = real[0..n];
    const e = real[n .. 2 * n - 1];
    const z = real[2 * n - 1 .. 2 * n - 1 + n * n];
    var phase: Z = .{ .re = 1, .im = 0 };
    for (0..n) |j| {
        d[j] = @floatCast(h[j + j * n].re);
        if (j > 0) {
            const lower = h[j + (j - 1) * n];
            const mag = magnitude(lower);
            e[j - 1] = @floatCast(mag);
            if (mag != 0) phase = product(scaled(lower, 1 / mag), phase);
        }
        for (0..n) |i| {
            q[i + j * n] = product(q[i + j * n], phase);
        }
    }
    const result = tri.solveWithContract(R, d, e, .{ .values = z, .row_count = n, .column_count = n, .leading_dimension = n }, .{ .max_iterations = max_iterations }, .stev_ieee_v1);
    for (0..n) |j| for (0..n) |i| {
        var value: Z = .{ .re = 0, .im = 0 };
        for (0..n) |k| {
            value = add(value, scaled(q[i + k * n], @as(W, z[k + j * n])));
        }
        h[i + j * n] = value;
    };
    @memcpy(q, h);
    for (0..n) |j| {
        h[j + j * n] = .{ .re = d[j], .im = 0 };
        if (j + 1 < n) h[j + 1 + j * n] = .{ .re = e[j], .im = 0 };
    }
    return result;
}
