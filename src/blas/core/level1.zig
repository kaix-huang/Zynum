// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("scalar.zig");
const indexing = @import("indexing.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;

const Real = scalar.Real;
const zero = scalar.zero;
const one = scalar.one;
const realScalar = scalar.realScalar;
const add = scalar.add;
const sub = scalar.sub;
const mul = scalar.mul;
const divv = scalar.divv;
const conj = scalar.conj;
const maybeConj = scalar.maybeConj;
const realPart = scalar.realPart;
const imagPart = scalar.imagPart;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;
const abs1 = scalar.abs1;
const abs2 = scalar.abs2;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const vectorGet = indexing.vectorGet;
const vectorSet = indexing.vectorSet;

pub fn scal(comptime T: type, n_: BlasInt, alpha: T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |i| {
        const p = ix(sx, i, incx_);
        x[p] = mul(T, alpha, x[p]);
    }
}

pub fn rscal(comptime T: type, n_: BlasInt, alpha: Real(T), x: [*]T, incx_: BlasInt) void {
    scal(T, n_, realScalar(T, alpha), x, incx_);
}

pub fn copy(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| vectorSet(T, y, sy, i, incy_, vectorGet(T, x, sx, i, incx_));
}

pub fn swap(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const t = x[px];
        x[px] = y[py];
        y[py] = t;
    }
}

pub fn axpy(comptime T: type, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, vectorGet(T, x, sx, i, incx_)));
    }
}

pub fn axpby(comptime T: type, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        const xv = vectorGet(T, x, sx, i, incx_);
        y[py] = add(T, mul(T, alpha, xv), mul(T, beta, y[py]));
    }
}

pub fn dot(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, conjx: bool) T {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return zero(T);
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    var sum = zero(T);
    for (0..n) |i| {
        const a = maybeConj(T, vectorGet(T, x, sx, i, incx_), conjx);
        sum = add(T, sum, mul(T, a, vectorGet(T, y, sy, i, incy_)));
    }
    return sum;
}

pub fn asum(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    var sum: Real(T) = 0;
    for (0..n) |i| sum += abs1(T, vectorGet(T, x, sx, i, incx_));
    return sum;
}

pub fn nrm2(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    var scale: Real(T) = 0;
    var ssq: Real(T) = 1;
    for (0..n) |i| {
        const v = vectorGet(T, x, sx, i, incx_);
        if (comptime isComplex(T)) {
            inline for (.{ realPart(T, v), imagPart(T, v) }) |component| {
                const ax = @abs(component);
                if (ax != 0) {
                    if (scale < ax) {
                        const r = scale / ax;
                        ssq = 1 + ssq * r * r;
                        scale = ax;
                    } else {
                        const r = ax / scale;
                        ssq += r * r;
                    }
                }
            }
        } else {
            const ax = @abs(v);
            if (ax != 0) {
                if (scale < ax) {
                    const r = scale / ax;
                    ssq = 1 + ssq * r * r;
                    scale = ax;
                } else {
                    const r = ax / scale;
                    ssq += r * r;
                }
            }
        }
    }
    return scale * @sqrt(ssq);
}

pub fn iamax(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) BlasInt {
    if (n_ < 1 or incx_ <= 0) return 0;
    const n = toUsize(n_);
    var best: usize = 0;
    var best_abs = abs1(T, x[0]);
    var p: usize = @intCast(incx_);
    for (1..n) |i| {
        const a = abs1(T, x[p]);
        if (a > best_abs) {
            best_abs = a;
            best = i;
        }
        p += @intCast(incx_);
    }
    return @intCast(best + 1);
}

pub fn rot(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt, c: Real(T), s: T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    const cc = realScalar(T, c);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const xv = x[px];
        const yv = y[py];
        x[px] = add(T, mul(T, cc, xv), mul(T, s, yv));
        y[py] = sub(T, mul(T, cc, yv), mul(T, s, xv));
    }
}

pub fn rotgReal(comptime T: type, a: *T, b: *T, c: *T, s: *T) void {
    const aa = a.*;
    const bb = b.*;
    const roe = if (@abs(bb) > @abs(aa)) bb else aa;
    const scale = @abs(aa) + @abs(bb);
    if (scale == 0) {
        c.* = 1;
        s.* = 0;
        a.* = 0;
        b.* = 0;
        return;
    }
    var r = scale * @sqrt((aa / scale) * (aa / scale) + (bb / scale) * (bb / scale));
    r = std.math.copysign(r, roe);
    c.* = aa / r;
    s.* = bb / r;
    var z: T = 1;
    if (@abs(aa) > @abs(bb)) z = s.*;
    if (@abs(bb) >= @abs(aa) and c.* != 0) z = 1 / c.*;
    a.* = r;
    b.* = z;
}

pub fn rotgComplex(comptime T: type, ca: *T, cb: *T, c: *Real(T), s: *T) void {
    const a = ca.*;
    const b = cb.*;
    const abs_a = abs2(T, a);
    if (abs_a == 0) {
        c.* = 0;
        s.* = one(T);
        ca.* = b;
        return;
    }
    const scale = abs_a + abs2(T, b);
    const norm = scale * @sqrt((abs_a / scale) * (abs_a / scale) + (abs2(T, b) / scale) * (abs2(T, b) / scale));
    const alpha = divv(T, a, realScalar(T, abs_a));
    c.* = abs_a / norm;
    s.* = divv(T, mul(T, alpha, conj(T, b)), realScalar(T, norm));
    ca.* = mul(T, alpha, realScalar(T, norm));
}

pub fn rotm(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt, param: [*]const T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const flag = param[0];
    if (flag == -2) return;
    const h11 = param[1];
    const h21 = param[2];
    const h12 = param[3];
    const h22 = param[4];
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const w = x[px];
        const z = y[py];
        if (flag < 0) {
            x[px] = w * h11 + z * h12;
            y[py] = w * h21 + z * h22;
        } else if (flag == 0) {
            x[px] = w + z * h12;
            y[py] = w * h21 + z;
        } else {
            x[px] = w * h11 + z;
            y[py] = -w + z * h22;
        }
    }
}

pub fn rotmg(comptime T: type, d1: *T, d2: *T, x1: *T, y1: *const T, param: [*]T) void {
    const gam: T = 4096;
    const gamsq = gam * gam;
    const rgamsq: T = 1 / gamsq;
    var flag: T = undefined;
    var h11: T = 0;
    var h12: T = 0;
    var h21: T = 0;
    var h22: T = 0;

    if (d1.* < 0) {
        flag = -1;
        d1.* = 0;
        d2.* = 0;
        x1.* = 0;
    } else {
        const p2 = d2.* * y1.*;
        if (p2 == 0) {
            flag = -2;
            param[0] = flag;
            return;
        }
        const p1 = d1.* * x1.*;
        const q2 = p2 * y1.*;
        const q1 = p1 * x1.*;
        if (@abs(q1) > @abs(q2)) {
            h21 = -y1.* / x1.*;
            h12 = p2 / p1;
            const u = 1 - h12 * h21;
            if (u > 0) {
                flag = 0;
                d1.* /= u;
                d2.* /= u;
                x1.* *= u;
            } else {
                flag = -1;
                d1.* = 0;
                d2.* = 0;
                x1.* = 0;
            }
        } else {
            if (q2 < 0) {
                flag = -1;
                d1.* = 0;
                d2.* = 0;
                x1.* = 0;
            } else {
                flag = 1;
                h11 = p1 / p2;
                h22 = x1.* / y1.*;
                const u = 1 + h11 * h22;
                const tmp = d2.* / u;
                d2.* = d1.* / u;
                d1.* = tmp;
                x1.* = y1.* * u;
            }
        }
        if (d1.* != 0) {
            while (d1.* <= rgamsq or d1.* >= gamsq) {
                if (flag == 0) {
                    h11 = 1;
                    h22 = 1;
                    flag = -1;
                } else if (flag > 0) {
                    h21 = -1;
                    h12 = 1;
                    flag = -1;
                }
                if (d1.* <= rgamsq) {
                    d1.* *= gamsq;
                    x1.* /= gam;
                    h11 /= gam;
                    h12 /= gam;
                } else {
                    d1.* /= gamsq;
                    x1.* *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }
        if (d2.* != 0) {
            while (@abs(d2.*) <= rgamsq or @abs(d2.*) >= gamsq) {
                if (flag == 0) {
                    h11 = 1;
                    h22 = 1;
                    flag = -1;
                } else if (flag > 0) {
                    h21 = -1;
                    h12 = 1;
                    flag = -1;
                }
                if (@abs(d2.*) <= rgamsq) {
                    d2.* *= gamsq;
                    h21 /= gam;
                    h22 /= gam;
                } else {
                    d2.* /= gamsq;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }
    param[0] = flag;
    if (flag < 0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == 0) {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
}
