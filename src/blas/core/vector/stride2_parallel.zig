// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const core_pool = @import("../execution/thread_pool.zig");

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
const isOne = scalar.isOne;
const abs1 = scalar.abs1;
const abs2 = scalar.abs2;

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

inline fn asConstRealPtr(comptime T: type, ptr: [*]const T) [*]const Real(T) {
    return @ptrCast(ptr);
}

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 16;
    if (T == f64) return 8;
    @compileError("real Level 1 vector lanes support f32 and f64");
}

fn unroll(comptime T: type) comptime_int {
    return 4 * lanes(T);
}

inline fn loadVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, index: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    return @as(*align(1) const V, @ptrCast(ptr + index)).*;
}

fn stride2Mask(comptime lane_count: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        values[i] = if (i < lane_count / 2)
            @intCast(2 * i)
        else
            ~@as(i32, @intCast(2 * (i - lane_count / 2)));
    }
    return values;
}

inline fn loadStride2Vec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, logical_index: usize) @Vector(lane_count, T) {
    const physical_index = 2 * logical_index;
    const low = loadVec(T, lane_count, ptr, physical_index);
    const high = loadVec(T, lane_count, ptr, physical_index + lane_count);
    return @shuffle(T, low, high, stride2Mask(lane_count));
}

fn Stride2Block(comptime T: type, comptime lane_count: comptime_int) type {
    const V = @Vector(lane_count, T);
    return struct {
        low: V,
        high: V,
        active: V,
    };
}

inline fn loadStride2Block(comptime T: type, comptime lane_count: comptime_int, ptr: [*]align(1) const T, logical_index: usize) Stride2Block(T, lane_count) {
    const V = @Vector(lane_count, T);
    const physical_index = 2 * logical_index;
    const low = @as(*align(1) const V, @ptrCast(ptr + physical_index)).*;
    const high = @as(*align(1) const V, @ptrCast(ptr + physical_index + lane_count)).*;
    return .{
        .low = low,
        .high = high,
        .active = @shuffle(T, low, high, stride2Mask(lane_count)),
    };
}

fn stride2StoreMask(comptime lane_count: comptime_int, comptime part: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        values[i] = @intCast(part * (lane_count / 2) + i / 2);
    }
    return values;
}

fn evenLaneMask(comptime lane_count: comptime_int) @Vector(lane_count, bool) {
    comptime var values: [lane_count]bool = undefined;
    inline for (0..lane_count) |i| values[i] = i % 2 == 0;
    return values;
}

inline fn storeStride2Block(
    comptime T: type,
    comptime lane_count: comptime_int,
    ptr: [*]align(1) T,
    logical_index: usize,
    block: Stride2Block(T, lane_count),
    active: @Vector(lane_count, T),
) void {
    const physical_index = 2 * logical_index;
    const select_mask = evenLaneMask(lane_count);
    const low_values = @shuffle(T, active, undefined, stride2StoreMask(lane_count, 0));
    const high_values = @shuffle(T, active, undefined, stride2StoreMask(lane_count, 1));
    const V = @Vector(lane_count, T);
    @as(*align(1) V, @ptrCast(ptr + physical_index)).* = @select(T, select_mask, low_values, block.low);
    @as(*align(1) V, @ptrCast(ptr + physical_index + lane_count)).* = @select(T, select_mask, high_values, block.high);
}

fn scalStride2Real(comptime T: type, n: usize, alpha: T, x: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        storeStride2Block(T, lane_count, x, i, xb, xb.active * alpha_v);
    }
    while (i < n) : (i += 1) x[2 * i] *= alpha;
}

fn swapStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T) void {
    const lane_count = lanes(T);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        storeStride2Block(T, lane_count, x, i, xb, yb.active);
        storeStride2Block(T, lane_count, y, i, yb, xb.active);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const value = x[index];
        x[index] = y[index];
        y[index] = value;
    }
}

fn swapComplexF32Stride2(n: usize, x: [*]ComplexF32, y: [*]ComplexF32) void {
    const packed_complexes = 8;
    const x_bits: [*]align(1) u64 = @ptrCast(x);
    const y_bits: [*]align(1) u64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadStride2Block(u64, packed_complexes, x_bits, i);
        const yb = loadStride2Block(u64, packed_complexes, y_bits, i);
        storeStride2Block(u64, packed_complexes, x_bits, i, xb, yb.active);
        storeStride2Block(u64, packed_complexes, y_bits, i, yb, xb.active);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const value = x[index];
        x[index] = y[index];
        y[index] = value;
    }
}

fn swapComplexF64Stride2(n: usize, x: [*]ComplexF64, y: [*]ComplexF64) void {
    const lane_count = lanes(f64);
    const packed_complexes = lane_count / 2;
    const real_x: [*]align(1) f64 = @ptrCast(x);
    const real_y: [*]align(1) f64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(f64, lane_count, real_x, i);
        const yb = loadComplexStride2Block(f64, lane_count, real_y, i);
        storeComplexStride2Block(f64, lane_count, real_x, i, xb, yb.active);
        storeComplexStride2Block(f64, lane_count, real_y, i, yb, xb.active);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const value = x[index];
        x[index] = y[index];
        y[index] = value;
    }
}

fn axpyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        const result = @mulAdd(V, alpha_v, xb.active, yb.active);
        storeStride2Block(T, lane_count, y, i, yb, result);
    }
    while (i < n) : (i += 1) y[2 * i] = @mulAdd(T, alpha, x[2 * i], y[2 * i]);
}

fn axpbyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        const result = @mulAdd(V, xb.active, alpha_v, yb.active * beta_v);
        storeStride2Block(T, lane_count, y, i, yb, result);
    }
    while (i < n) : (i += 1) y[2 * i] = @mulAdd(T, alpha, x[2 * i], beta * y[2 * i]);
}

fn dotStride2Real(comptime T: type, n: usize, x: [*]const T, y: [*]const T) T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count < n) : (i += unroll_count) {
        acc0 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i), loadStride2Vec(T, lane_count, y, i), acc0);
        acc1 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + lane_count), loadStride2Vec(T, lane_count, y, i + lane_count), acc1);
        acc2 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + 2 * lane_count), loadStride2Vec(T, lane_count, y, i + 2 * lane_count), acc2);
        acc3 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + 3 * lane_count), loadStride2Vec(T, lane_count, y, i + 3 * lane_count), acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count < n) : (i += lane_count) {
        acc = @mulAdd(V, loadStride2Vec(T, lane_count, x, i), loadStride2Vec(T, lane_count, y, i), acc);
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(T, x[2 * i], y[2 * i], sum);
    return sum;
}

fn dotF32AccF64Stride2(n: usize, x: [*]const f32, y: [*]const f32) f64 {
    const F64V = @Vector(4, f64);
    var acc0: F64V = @splat(0);
    var acc1: F64V = @splat(0);
    var acc2: F64V = @splat(0);
    var acc3: F64V = @splat(0);
    var i: usize = 0;
    while (i + 16 < n) : (i += 16) {
        const x0: F64V = @floatCast(loadStride2Vec(f32, 4, x, i));
        const y0: F64V = @floatCast(loadStride2Vec(f32, 4, y, i));
        const x1: F64V = @floatCast(loadStride2Vec(f32, 4, x, i + 4));
        const y1: F64V = @floatCast(loadStride2Vec(f32, 4, y, i + 4));
        const x2: F64V = @floatCast(loadStride2Vec(f32, 4, x, i + 8));
        const y2: F64V = @floatCast(loadStride2Vec(f32, 4, y, i + 8));
        const x3: F64V = @floatCast(loadStride2Vec(f32, 4, x, i + 12));
        const y3: F64V = @floatCast(loadStride2Vec(f32, 4, y, i + 12));
        acc0 = @mulAdd(F64V, x0, y0, acc0);
        acc1 = @mulAdd(F64V, x1, y1, acc1);
        acc2 = @mulAdd(F64V, x2, y2, acc2);
        acc3 = @mulAdd(F64V, x3, y3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + 4 < n) : (i += 4) {
        const xv: F64V = @floatCast(loadStride2Vec(f32, 4, x, i));
        const yv: F64V = @floatCast(loadStride2Vec(f32, 4, y, i));
        acc = @mulAdd(F64V, xv, yv, acc);
    }
    var sum: f64 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(f64, @as(f64, x[2 * i]), @as(f64, y[2 * i]), sum);
    return sum;
}

fn asumStride2Real(comptime T: type, n: usize, x: [*]const T) T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count < n) : (i += unroll_count) {
        acc0 += @abs(loadStride2Vec(T, lane_count, x, i));
        acc1 += @abs(loadStride2Vec(T, lane_count, x, i + lane_count));
        acc2 += @abs(loadStride2Vec(T, lane_count, x, i + 2 * lane_count));
        acc3 += @abs(loadStride2Vec(T, lane_count, x, i + 3 * lane_count));
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count < n) : (i += lane_count) {
        acc += @abs(loadStride2Vec(T, lane_count, x, i));
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += @abs(x[2 * i]);
    return sum;
}

fn asumStride2Complex(comptime T: type, n: usize, x: [*]const T) Real(T) {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const real_x = asConstRealPtr(T, x);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + 4 * packed_complexes < n) : (i += 4 * packed_complexes) {
        acc0 += @abs(loadComplexStride2RealVec(R, lane_count, real_x, i));
        acc1 += @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + packed_complexes));
        acc2 += @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + 2 * packed_complexes));
        acc3 += @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + 3 * packed_complexes));
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        acc += @abs(loadComplexStride2RealVec(R, lane_count, real_x, i));
    }
    var sum: R = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += abs1(T, x[2 * i]);
    return sum;
}

fn nrm2Stride2RealSsq(comptime T: type, n: usize, x: [*]const T, scale: T, comptime use_reciprocal: bool) T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const factor_v: V = if (use_reciprocal) @splat(1 / scale) else @splat(scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count < n) : (i += unroll_count) {
        const x0 = loadStride2Vec(T, lane_count, x, i);
        const x1 = loadStride2Vec(T, lane_count, x, i + lane_count);
        const x2 = loadStride2Vec(T, lane_count, x, i + 2 * lane_count);
        const x3 = loadStride2Vec(T, lane_count, x, i + 3 * lane_count);
        const v0 = if (use_reciprocal) x0 * factor_v else x0 / factor_v;
        const v1 = if (use_reciprocal) x1 * factor_v else x1 / factor_v;
        const v2 = if (use_reciprocal) x2 * factor_v else x2 / factor_v;
        const v3 = if (use_reciprocal) x3 * factor_v else x3 / factor_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count < n) : (i += lane_count) {
        const xv = loadStride2Vec(T, lane_count, x, i);
        const v = if (use_reciprocal) xv * factor_v else xv / factor_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const v = x[2 * i] / scale;
        ssq = @mulAdd(T, v, v, ssq);
    }
    return ssq;
}

fn nrm2Stride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count < n) : (i += unroll_count) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadStride2Vec(T, lane_count, x, i + k * lane_count)));
        }
    }
    while (i + lane_count < n) : (i += lane_count) {
        max_v = @max(max_v, @abs(loadStride2Vec(T, lane_count, x, i)));
    }
    var scale: T = @reduce(.Max, max_v);
    while (i < n) : (i += 1) scale = @max(scale, @abs(x[2 * i]));
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const ssq = if (scale >= std.math.floatMin(T))
        nrm2Stride2RealSsq(T, n, x, scale, true)
    else
        nrm2Stride2RealSsq(T, n, x, scale, false);
    return scale * @sqrt(ssq);
}

fn nrm2Stride2ComplexSsq(comptime T: type, n: usize, x: [*]const T, scale: Real(T), comptime use_reciprocal: bool) Real(T) {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const real_x = asConstRealPtr(T, x);
    const factor_v: V = if (use_reciprocal) @splat(1 / scale) else @splat(scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + 4 * packed_complexes < n) : (i += 4 * packed_complexes) {
        const x0 = loadComplexStride2RealVec(R, lane_count, real_x, i);
        const x1 = loadComplexStride2RealVec(R, lane_count, real_x, i + packed_complexes);
        const x2 = loadComplexStride2RealVec(R, lane_count, real_x, i + 2 * packed_complexes);
        const x3 = loadComplexStride2RealVec(R, lane_count, real_x, i + 3 * packed_complexes);
        const v0 = if (use_reciprocal) x0 * factor_v else x0 / factor_v;
        const v1 = if (use_reciprocal) x1 * factor_v else x1 / factor_v;
        const v2 = if (use_reciprocal) x2 * factor_v else x2 / factor_v;
        const v3 = if (use_reciprocal) x3 * factor_v else x3 / factor_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xv = loadComplexStride2RealVec(R, lane_count, real_x, i);
        const v = if (use_reciprocal) xv * factor_v else xv / factor_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: R = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const value = x[2 * i];
        const re = realPart(T, value) / scale;
        const im = imagPart(T, value) / scale;
        ssq = @mulAdd(R, re, re, ssq);
        ssq = @mulAdd(R, im, im, ssq);
    }
    return ssq;
}

fn nrm2Stride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const real_x = asConstRealPtr(T, x);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + 4 * packed_complexes < n) : (i += 4 * packed_complexes) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + k * packed_complexes)));
        }
    }
    while (i + packed_complexes < n) : (i += packed_complexes) {
        max_v = @max(max_v, @abs(loadComplexStride2RealVec(R, lane_count, real_x, i)));
    }
    var scale: R = @reduce(.Max, max_v);
    while (i < n) : (i += 1) {
        const value = x[2 * i];
        scale = @max(scale, @max(@abs(realPart(T, value)), @abs(imagPart(T, value))));
    }
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const ssq = if (scale >= std.math.floatMin(R))
        nrm2Stride2ComplexSsq(T, n, x, scale, true)
    else
        nrm2Stride2ComplexSsq(T, n, x, scale, false);
    return scale * @sqrt(ssq);
}

fn pairSwapMask(comptime lane_count: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        values[i] = if (i % 2 == 0) @intCast(i + 1) else @intCast(i - 1);
    }
    return values;
}

fn pairHalfMask(comptime lane_count: comptime_int, comptime part: comptime_int) @Vector(lane_count / 2, i32) {
    comptime var values: [lane_count / 2]i32 = undefined;
    inline for (0..lane_count / 2) |i| values[i] = @intCast(2 * i + part);
    return values;
}

inline fn pairAbsSums(comptime R: type, comptime lane_count: comptime_int, ax: @Vector(lane_count, R)) @Vector(lane_count / 2, R) {
    const even_mask = comptime pairHalfMask(lane_count, 0);
    const odd_mask = comptime pairHalfMask(lane_count, 1);
    const even: @Vector(lane_count / 2, R) = @shuffle(R, ax, undefined, even_mask);
    const odd: @Vector(lane_count / 2, R) = @shuffle(R, ax, undefined, odd_mask);
    return even + odd;
}

fn complexStride2RealMask(comptime lane_count: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        const local = i % (lane_count / 2);
        const source_index: i32 = @intCast(4 * (local / 2) + local % 2);
        values[i] = if (i < lane_count / 2) source_index else ~source_index;
    }
    return values;
}

inline fn loadComplexStride2RealVec(
    comptime R: type,
    comptime lane_count: comptime_int,
    real_x: [*]const R,
    logical_index: usize,
) @Vector(lane_count, R) {
    const physical_index = 4 * logical_index;
    const low = loadVec(R, lane_count, real_x, physical_index);
    const high = loadVec(R, lane_count, real_x, physical_index + lane_count);
    return @shuffle(R, low, high, complexStride2RealMask(lane_count));
}

fn ComplexStride2Block(comptime R: type, comptime lane_count: comptime_int) type {
    const V = @Vector(lane_count, R);
    return struct {
        low: V,
        high: V,
        active: V,
    };
}

inline fn loadComplexStride2Block(
    comptime R: type,
    comptime lane_count: comptime_int,
    real_x: [*]align(1) const R,
    logical_index: usize,
) ComplexStride2Block(R, lane_count) {
    const V = @Vector(lane_count, R);
    const physical_index = 4 * logical_index;
    const low = @as(*align(1) const V, @ptrCast(real_x + physical_index)).*;
    const high = @as(*align(1) const V, @ptrCast(real_x + physical_index + lane_count)).*;
    return .{
        .low = low,
        .high = high,
        .active = @shuffle(R, low, high, complexStride2RealMask(lane_count)),
    };
}

fn complexStride2StoreMask(comptime lane_count: comptime_int, comptime part: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        values[i] = @intCast(part * (lane_count / 2) + 2 * (i / 4) + i % 2);
    }
    return values;
}

fn complexActiveLaneMask(comptime lane_count: comptime_int) @Vector(lane_count, bool) {
    comptime var values: [lane_count]bool = undefined;
    inline for (0..lane_count) |i| values[i] = i % 4 < 2;
    return values;
}

inline fn storeComplexStride2Block(
    comptime R: type,
    comptime lane_count: comptime_int,
    real_x: [*]align(1) R,
    logical_index: usize,
    block: ComplexStride2Block(R, lane_count),
    active: @Vector(lane_count, R),
) void {
    const V = @Vector(lane_count, R);
    const physical_index = 4 * logical_index;
    const select_mask = complexActiveLaneMask(lane_count);
    const low_values = @shuffle(R, active, undefined, complexStride2StoreMask(lane_count, 0));
    const high_values = @shuffle(R, active, undefined, complexStride2StoreMask(lane_count, 1));
    @as(*align(1) V, @ptrCast(real_x + physical_index)).* = @select(R, select_mask, low_values, block.low);
    @as(*align(1) V, @ptrCast(real_x + physical_index + lane_count)).* = @select(R, select_mask, high_values, block.high);
}

fn complexRealScalStride2(comptime T: type, n: usize, alpha: Real(T), x: [*]T) void {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const alpha_v: V = @splat(alpha);
    const real_x: [*]align(1) R = @ptrCast(x);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(R, lane_count, real_x, i);
        storeComplexStride2Block(R, lane_count, real_x, i, xb, xb.active * alpha_v);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        x[index].re *= alpha;
        x[index].im *= alpha;
    }
}

fn pairSignVector(comptime R: type, comptime lane_count: comptime_int, im: R) @Vector(lane_count, R) {
    comptime var signs: [lane_count]R = undefined;
    inline for (0..lane_count) |i| {
        signs[i] = if (i % 2 == 0) -1 else 1;
    }
    const sign_v: @Vector(lane_count, R) = signs;
    return sign_v * @as(@Vector(lane_count, R), @splat(im));
}

fn pairPatternVector(comptime R: type, comptime lane_count: comptime_int, comptime even: R, comptime odd: R) @Vector(lane_count, R) {
    comptime var values: [lane_count]R = undefined;
    inline for (0..lane_count) |i| {
        values[i] = if (i % 2 == 0) even else odd;
    }
    return values;
}

inline fn complexMulPackedF32(value: @Vector(16, f32), alpha_re: f32, alpha_im: f32) @Vector(16, f32) {
    const re_v: @Vector(16, f32) = @splat(alpha_re);
    const im_sign_v = pairSignVector(f32, 16, alpha_im);
    const swapped = @shuffle(f32, value, undefined, pairSwapMask(16));
    return @mulAdd(@Vector(16, f32), value, re_v, swapped * im_sign_v);
}

fn complexScalF32Stride2(n: usize, alpha_re: f32, alpha_im: f32, x: [*]ComplexF32) void {
    const packed_complexes = 8;
    const PackedReal = @Vector(16, f32);
    const PackedBits = @Vector(packed_complexes, u64);
    const x_bits: [*]align(1) u64 = @ptrCast(x);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadStride2Block(u64, packed_complexes, x_bits, i);
        const xv: PackedReal = @bitCast(xb.active);
        const result: PackedBits = @bitCast(complexMulPackedF32(xv, alpha_re, alpha_im));
        storeStride2Block(u64, packed_complexes, x_bits, i, xb, result);
    }
    const alpha: ComplexF32 = .{ .re = alpha_re, .im = alpha_im };
    while (i < n) : (i += 1) x[2 * i] = mul(ComplexF32, alpha, x[2 * i]);
}

fn complexAxpyF32Stride2(n: usize, alpha: ComplexF32, x: [*]const ComplexF32, y: [*]ComplexF32) void {
    const packed_complexes = 8;
    const PackedReal = @Vector(16, f32);
    const PackedBits = @Vector(packed_complexes, u64);
    const x_bits: [*]align(1) const u64 = @ptrCast(x);
    const y_bits: [*]align(1) u64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadStride2Block(u64, packed_complexes, x_bits, i);
        const yb = loadStride2Block(u64, packed_complexes, y_bits, i);
        const xv: PackedReal = @bitCast(xb.active);
        const yv: PackedReal = @bitCast(yb.active);
        const result: PackedBits = @bitCast(complexMulPackedF32(xv, alpha.re, alpha.im) + yv);
        storeStride2Block(u64, packed_complexes, y_bits, i, yb, result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        y[index] = add(ComplexF32, y[index], mul(ComplexF32, alpha, x[index]));
    }
}

fn complexAxpbyF32Stride2(
    n: usize,
    alpha_re: f32,
    alpha_im: f32,
    x: [*]const ComplexF32,
    beta_re: f32,
    beta_im: f32,
    y: [*]ComplexF32,
) void {
    const packed_complexes = 8;
    const PackedReal = @Vector(16, f32);
    const PackedBits = @Vector(packed_complexes, u64);
    const x_bits: [*]align(1) const u64 = @ptrCast(x);
    const y_bits: [*]align(1) u64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadStride2Block(u64, packed_complexes, x_bits, i);
        const yb = loadStride2Block(u64, packed_complexes, y_bits, i);
        const xv: PackedReal = @bitCast(xb.active);
        const yv: PackedReal = @bitCast(yb.active);
        const result: PackedBits = @bitCast(complexMulPackedF32(xv, alpha_re, alpha_im) + complexMulPackedF32(yv, beta_re, beta_im));
        storeStride2Block(u64, packed_complexes, y_bits, i, yb, result);
    }
    const alpha: ComplexF32 = .{ .re = alpha_re, .im = alpha_im };
    const beta: ComplexF32 = .{ .re = beta_re, .im = beta_im };
    while (i < n) : (i += 1) {
        const index = 2 * i;
        y[index] = add(ComplexF32, mul(ComplexF32, alpha, x[index]), mul(ComplexF32, beta, y[index]));
    }
}

inline fn complexMulPackedF64(value: @Vector(8, f64), alpha_re: f64, alpha_im: f64) @Vector(8, f64) {
    const re_v: @Vector(8, f64) = @splat(alpha_re);
    const im_sign_v = pairSignVector(f64, 8, alpha_im);
    const swapped = @shuffle(f64, value, undefined, pairSwapMask(8));
    return @mulAdd(@Vector(8, f64), value, re_v, swapped * im_sign_v);
}

fn complexScalF64Stride2(n: usize, alpha_re: f64, alpha_im: f64, x: [*]ComplexF64) void {
    const lane_count = lanes(f64);
    const packed_complexes = lane_count / 2;
    const real_x: [*]align(1) f64 = @ptrCast(x);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(f64, lane_count, real_x, i);
        const result = complexMulPackedF64(xb.active, alpha_re, alpha_im);
        storeComplexStride2Block(f64, lane_count, real_x, i, xb, result);
    }
    const alpha: ComplexF64 = .{ .re = alpha_re, .im = alpha_im };
    while (i < n) : (i += 1) x[2 * i] = mul(ComplexF64, alpha, x[2 * i]);
}

fn complexAxpyF64Stride2(n: usize, alpha: ComplexF64, x: [*]const ComplexF64, y: [*]ComplexF64) void {
    const lane_count = lanes(f64);
    const packed_complexes = lane_count / 2;
    const real_x: [*]align(1) const f64 = @ptrCast(x);
    const real_y: [*]align(1) f64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(f64, lane_count, real_x, i);
        const yb = loadComplexStride2Block(f64, lane_count, real_y, i);
        const result = complexMulPackedF64(xb.active, alpha.re, alpha.im) + yb.active;
        storeComplexStride2Block(f64, lane_count, real_y, i, yb, result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        y[index] = add(ComplexF64, y[index], mul(ComplexF64, alpha, x[index]));
    }
}

fn complexAxpbyF64Stride2(n: usize, alpha: ComplexF64, x: [*]const ComplexF64, beta: ComplexF64, y: [*]ComplexF64) void {
    const lane_count = lanes(f64);
    const packed_complexes = lane_count / 2;
    const real_x: [*]align(1) const f64 = @ptrCast(x);
    const real_y: [*]align(1) f64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(f64, lane_count, real_x, i);
        const yb = loadComplexStride2Block(f64, lane_count, real_y, i);
        const result = complexMulPackedF64(xb.active, alpha.re, alpha.im) + complexMulPackedF64(yb.active, beta.re, beta.im);
        storeComplexStride2Block(f64, lane_count, real_y, i, yb, result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        y[index] = add(ComplexF64, mul(ComplexF64, alpha, x[index]), mul(ComplexF64, beta, y[index]));
    }
}

fn complexDotStride2(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) T {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const real_x = asConstRealPtr(T, x);
    const real_y = asConstRealPtr(T, y);
    const swap_mask = comptime pairSwapMask(lane_count);
    const re_sign: V = if (conjx) @splat(1) else pairPatternVector(R, lane_count, 1, -1);
    const im_sign: V = if (conjx) pairPatternVector(R, lane_count, 1, -1) else @splat(1);
    var re_acc0: V = @splat(0);
    var re_acc1: V = @splat(0);
    var re_acc2: V = @splat(0);
    var re_acc3: V = @splat(0);
    var im_acc0: V = @splat(0);
    var im_acc1: V = @splat(0);
    var im_acc2: V = @splat(0);
    var im_acc3: V = @splat(0);
    var i: usize = 0;
    while (i + 4 * packed_complexes < n) : (i += 4 * packed_complexes) {
        inline for (0..4) |k| {
            const xv = loadComplexStride2RealVec(R, lane_count, real_x, i + k * packed_complexes);
            const yv = loadComplexStride2RealVec(R, lane_count, real_y, i + k * packed_complexes);
            const y_swap = @shuffle(R, yv, undefined, swap_mask);
            switch (k) {
                0 => {
                    re_acc0 = @mulAdd(V, xv * yv, re_sign, re_acc0);
                    im_acc0 = @mulAdd(V, xv * y_swap, im_sign, im_acc0);
                },
                1 => {
                    re_acc1 = @mulAdd(V, xv * yv, re_sign, re_acc1);
                    im_acc1 = @mulAdd(V, xv * y_swap, im_sign, im_acc1);
                },
                2 => {
                    re_acc2 = @mulAdd(V, xv * yv, re_sign, re_acc2);
                    im_acc2 = @mulAdd(V, xv * y_swap, im_sign, im_acc2);
                },
                3 => {
                    re_acc3 = @mulAdd(V, xv * yv, re_sign, re_acc3);
                    im_acc3 = @mulAdd(V, xv * y_swap, im_sign, im_acc3);
                },
                else => unreachable,
            }
        }
    }
    var re_acc = re_acc0 + re_acc1 + re_acc2 + re_acc3;
    var im_acc = im_acc0 + im_acc1 + im_acc2 + im_acc3;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xv = loadComplexStride2RealVec(R, lane_count, real_x, i);
        const yv = loadComplexStride2RealVec(R, lane_count, real_y, i);
        const y_swap = @shuffle(R, yv, undefined, swap_mask);
        re_acc = @mulAdd(V, xv * yv, re_sign, re_acc);
        im_acc = @mulAdd(V, xv * y_swap, im_sign, im_acc);
    }
    var re_sum: R = @reduce(.Add, re_acc);
    var im_sum: R = @reduce(.Add, im_acc);
    while (i < n) : (i += 1) {
        const xv = x[2 * i];
        const yv = y[2 * i];
        if (conjx) {
            re_sum = @mulAdd(R, xv.im, yv.im, @mulAdd(R, xv.re, yv.re, re_sum));
            im_sum = @mulAdd(R, -xv.im, yv.re, @mulAdd(R, xv.re, yv.im, im_sum));
        } else {
            re_sum = @mulAdd(R, -xv.im, yv.im, @mulAdd(R, xv.re, yv.re, re_sum));
            im_sum = @mulAdd(R, xv.im, yv.re, @mulAdd(R, xv.re, yv.im, im_sum));
        }
    }
    return .{ .re = re_sum, .im = im_sum };
}

fn iamaxStride2Real(comptime T: type, n: usize, x: [*]const T) BlasInt {
    if (n == 0) return 0;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    var best: usize = 0;
    var best_abs: T = @abs(x[0]);
    var i: usize = 1;
    while (i + unroll_count < n) : (i += unroll_count) {
        var max_v: @Vector(lane_count, T) = @splat(0);
        inline for (0..4) |k| {
            const values = @abs(loadStride2Vec(T, lane_count, x, i + k * lane_count));
            const non_nan_candidates = @select(T, values == values, values, @as(@Vector(lane_count, T), @splat(0)));
            max_v = @max(max_v, non_nan_candidates);
        }
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + unroll_count;
            var j = i;
            while (j < end) : (j += 1) {
                const ax = @abs(x[2 * j]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j;
                }
            }
        }
    }
    while (i + lane_count < n) : (i += lane_count) {
        const values = @abs(loadStride2Vec(T, lane_count, x, i));
        const max_v = @select(T, values == values, values, @as(@Vector(lane_count, T), @splat(0)));
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + lane_count;
            var j = i;
            while (j < end) : (j += 1) {
                const ax = @abs(x[2 * j]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j;
                }
            }
        }
    }
    while (i < n) : (i += 1) {
        const ax = @abs(x[2 * i]);
        if (ax > best_abs) {
            best_abs = ax;
            best = i;
        }
    }
    return @intCast(best + 1);
}

fn iamaxStride2Complex(comptime T: type, n: usize, x: [*]const T) BlasInt {
    if (n == 0) return 0;
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const M = @Vector(packed_complexes, R);
    const real_x = asConstRealPtr(T, x);
    var best: usize = 0;
    var best_abs = abs1(T, x[0]);
    var i: usize = 0;

    while (i + 4 * packed_complexes < n) : (i += 4 * packed_complexes) {
        var max_v: M = @splat(0);
        inline for (0..4) |k| {
            const ax = @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + k * packed_complexes));
            const values = pairAbsSums(R, lane_count, ax);
            const non_nan_candidates = @select(R, values == values, values, @as(M, @splat(0)));
            max_v = @max(max_v, non_nan_candidates);
        }
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + 4 * packed_complexes;
            var j = i;
            while (j < end) : (j += 1) {
                const ax = abs1(T, x[2 * j]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j;
                }
            }
        }
    }
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const ax = @abs(loadComplexStride2RealVec(R, lane_count, real_x, i));
        const values = pairAbsSums(R, lane_count, ax);
        const non_nan_candidates = @select(R, values == values, values, @as(M, @splat(0)));
        if (@reduce(.Max, non_nan_candidates) > best_abs) {
            const end = i + packed_complexes;
            var j = i;
            while (j < end) : (j += 1) {
                const scalar_abs = abs1(T, x[2 * j]);
                if (scalar_abs > best_abs) {
                    best_abs = scalar_abs;
                    best = j;
                }
            }
        }
    }
    while (i < n) : (i += 1) {
        const ax = abs1(T, x[2 * i]);
        if (ax > best_abs) {
            best_abs = ax;
            best = i;
        }
    }
    return @intCast(best + 1);
}

fn rotStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const c_v: V = @splat(c);
    const s_v: V = @splat(s);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        const x_result = @mulAdd(V, xb.active, c_v, yb.active * s_v);
        const y_result = @mulAdd(V, -xb.active, s_v, yb.active * c_v);
        storeStride2Block(T, lane_count, x, i, xb, x_result);
        storeStride2Block(T, lane_count, y, i, yb, y_result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const xv = x[index];
        const yv = y[index];
        x[index] = @mulAdd(T, xv, c, yv * s);
        y[index] = @mulAdd(T, -xv, s, yv * c);
    }
}

fn rotComplexF32Stride2(n: usize, x: [*]ComplexF32, y: [*]ComplexF32, c: f32, s: f32) void {
    const packed_complexes = 8;
    const PackedReal = @Vector(2 * packed_complexes, f32);
    const PackedBits = @Vector(packed_complexes, u64);
    const c_v: PackedReal = @splat(c);
    const s_v: PackedReal = @splat(s);
    const x_bits: [*]align(1) u64 = @ptrCast(x);
    const y_bits: [*]align(1) u64 = @ptrCast(y);
    var i: usize = 0;
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadStride2Block(u64, packed_complexes, x_bits, i);
        const yb = loadStride2Block(u64, packed_complexes, y_bits, i);
        const xv: PackedReal = @bitCast(xb.active);
        const yv: PackedReal = @bitCast(yb.active);
        const x_result: PackedBits = @bitCast(@mulAdd(PackedReal, xv, c_v, yv * s_v));
        const y_result: PackedBits = @bitCast(@mulAdd(PackedReal, -xv, s_v, yv * c_v));
        storeStride2Block(u64, packed_complexes, x_bits, i, xb, x_result);
        storeStride2Block(u64, packed_complexes, y_bits, i, yb, y_result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const xv = x[index];
        const yv = y[index];
        x[index] = .{
            .re = @mulAdd(f32, xv.re, c, yv.re * s),
            .im = @mulAdd(f32, xv.im, c, yv.im * s),
        };
        y[index] = .{
            .re = @mulAdd(f32, -xv.re, s, yv.re * c),
            .im = @mulAdd(f32, -xv.im, s, yv.im * c),
        };
    }
}

fn rotComplexF64Stride2(n: usize, x: [*]ComplexF64, y: [*]ComplexF64, c: f64, s: f64) void {
    const lane_count = lanes(f64);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, f64);
    const c_v: V = @splat(c);
    const s_v: V = @splat(s);
    const real_x: [*]align(1) f64 = @ptrCast(x);
    const real_y: [*]align(1) f64 = @ptrCast(y);
    var i: usize = 0;
    while (i + 2 * packed_complexes < n) : (i += 2 * packed_complexes) {
        inline for (0..2) |k| {
            const offset = i + k * packed_complexes;
            const xb = loadComplexStride2Block(f64, lane_count, real_x, offset);
            const yb = loadComplexStride2Block(f64, lane_count, real_y, offset);
            const x_result = @mulAdd(V, xb.active, c_v, yb.active * s_v);
            const y_result = @mulAdd(V, -xb.active, s_v, yb.active * c_v);
            storeComplexStride2Block(f64, lane_count, real_x, offset, xb, x_result);
            storeComplexStride2Block(f64, lane_count, real_y, offset, yb, y_result);
        }
    }
    while (i + packed_complexes < n) : (i += packed_complexes) {
        const xb = loadComplexStride2Block(f64, lane_count, real_x, i);
        const yb = loadComplexStride2Block(f64, lane_count, real_y, i);
        const x_result = @mulAdd(V, xb.active, c_v, yb.active * s_v);
        const y_result = @mulAdd(V, -xb.active, s_v, yb.active * c_v);
        storeComplexStride2Block(f64, lane_count, real_x, i, xb, x_result);
        storeComplexStride2Block(f64, lane_count, real_y, i, yb, y_result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const xv = x[index];
        const yv = y[index];
        x[index] = .{
            .re = @mulAdd(f64, xv.re, c, yv.re * s),
            .im = @mulAdd(f64, xv.im, c, yv.im * s),
        };
        y[index] = .{
            .re = @mulAdd(f64, -xv.re, s, yv.re * c),
            .im = @mulAdd(f64, -xv.im, s, yv.im * c),
        };
    }
}

fn rotmStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const h11_v: V = @splat(h11);
    const h21_v: V = @splat(h21);
    const h12_v: V = @splat(h12);
    const h22_v: V = @splat(h22);
    var i: usize = 0;
    while (i + lane_count < n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        var x_result: V = undefined;
        var y_result: V = undefined;
        if (flag < 0) {
            x_result = @mulAdd(V, xb.active, h11_v, yb.active * h12_v);
            y_result = @mulAdd(V, xb.active, h21_v, yb.active * h22_v);
        } else if (flag == 0) {
            x_result = @mulAdd(V, yb.active, h12_v, xb.active);
            y_result = @mulAdd(V, xb.active, h21_v, yb.active);
        } else {
            x_result = @mulAdd(V, xb.active, h11_v, yb.active);
            y_result = yb.active * h22_v - xb.active;
        }
        storeStride2Block(T, lane_count, x, i, xb, x_result);
        storeStride2Block(T, lane_count, y, i, yb, y_result);
    }
    while (i < n) : (i += 1) {
        const index = 2 * i;
        const w = x[index];
        const z = y[index];
        if (flag < 0) {
            x[index] = @mulAdd(T, w, h11, z * h12);
            y[index] = @mulAdd(T, w, h21, z * h22);
        } else if (flag == 0) {
            x[index] = @mulAdd(T, z, h12, w);
            y[index] = @mulAdd(T, w, h21, z);
        } else {
            x[index] = @mulAdd(T, w, h11, z);
            y[index] = z * h22 - w;
        }
    }
}

fn parallelTaskCount(n: usize, min_items_per_task: usize, max_task_count: usize) usize {
    if (n < 512 * 1024) return 1;
    return @min(core_pool.taskCount(n, min_items_per_task), max_task_count);
}

fn runLevel1Tasks(task_fn: core_pool.TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (comptime builtin.cpu.arch == .x86_64) return core_pool.runLowLatency(task_fn, tasks, count);
    return core_pool.run(task_fn, tasks, count);
}

fn RangeTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        alpha: T,
        beta: T,
        x: [*]const T,
        y: [*]T,
        out: *T,
    };
}

fn SwapTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]T,
        y: [*]T,
    };
}

fn RotTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]T,
        y: [*]T,
        c: T,
        s: T,
    };
}

fn RotmTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]T,
        y: [*]T,
        flag: T,
        h11: T,
        h21: T,
        h12: T,
        h22: T,
    };
}

fn runSwapStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SwapTask(f32) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    swapStride2Real(f32, task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runSwapStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SwapTask(f64) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    swapStride2Real(f64, task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runSwapStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SwapTask(ComplexF32) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    swapComplexF32Stride2(task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runSwapStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SwapTask(ComplexF64) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    swapComplexF64Stride2(task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

pub noinline fn parallelSwapStride2(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]SwapTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
        };
    }
    const runner = if (T == f32)
        runSwapStride2TaskF32
    else if (T == f64)
        runSwapStride2TaskF64
    else if (T == ComplexF32)
        runSwapStride2TaskC32
    else
        runSwapStride2TaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runRotStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (comptime isReal(T)) {
        rotStride2Real(T, task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0, task.c, task.s);
    } else if (comptime T == ComplexF32) {
        rotComplexF32Stride2(
            task.n1 - task.n0,
            task.x + 2 * task.n0,
            task.y + 2 * task.n0,
            task.c.re,
            task.s.re,
        );
    } else {
        rotComplexF64Stride2(
            task.n1 - task.n0,
            task.x + 2 * task.n0,
            task.y + 2 * task.n0,
            task.c.re,
            task.s.re,
        );
    }
}

fn runRotStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runRotStride2Task(f32, raw_tasks, index);
}

fn runRotStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runRotStride2Task(f64, raw_tasks, index);
}

fn runRotStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runRotStride2Task(ComplexF32, raw_tasks, index);
}

fn runRotStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runRotStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelRotStride2(comptime T: type, n: usize, x: [*]T, y: [*]T, c: Real(T), s: T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RotTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .c = realScalar(T, c),
            .s = s,
        };
    }
    const runner = if (T == f32)
        runRotStride2TaskF32
    else if (T == f64)
        runRotStride2TaskF64
    else if (T == ComplexF32)
        runRotStride2TaskC32
    else
        runRotStride2TaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runRotmStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RotmTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    rotmStride2Real(
        T,
        task.n1 - task.n0,
        task.x + 2 * task.n0,
        task.y + 2 * task.n0,
        task.flag,
        task.h11,
        task.h21,
        task.h12,
        task.h22,
    );
}

fn runRotmStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runRotmStride2Task(f32, raw_tasks, index);
}

fn runRotmStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runRotmStride2Task(f64, raw_tasks, index);
}

pub noinline fn parallelRotmStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RotmTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .flag = flag,
            .h11 = h11,
            .h21 = h21,
            .h12 = h12,
            .h22 = h22,
        };
    }
    const runner = if (T == f32) runRotmStride2TaskF32 else runRotmStride2TaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runScalStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    scalStride2Real(T, task.n1 - task.n0, task.alpha, task.y + 2 * task.n0);
}

fn runScalStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runScalStride2Task(f32, raw_tasks, index);
}

fn runScalStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runScalStride2Task(f64, raw_tasks, index);
}

fn runComplexScalStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (comptime T == ComplexF32) {
        complexScalF32Stride2(task.n1 - task.n0, task.alpha.re, task.alpha.im, task.y + 2 * task.n0);
    } else {
        complexScalF64Stride2(task.n1 - task.n0, task.alpha.re, task.alpha.im, task.y + 2 * task.n0);
    }
}

fn runComplexScalStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexScalStride2Task(ComplexF32, raw_tasks, index);
}

fn runComplexScalStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexScalStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelScalStride2Real(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = 0,
            .x = x,
            .y = x,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runScalStride2TaskF32 else runScalStride2TaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

pub noinline fn parallelComplexScalStride2(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = zero(T),
            .x = x,
            .y = x,
            .out = undefined,
        };
    }
    const runner = if (T == ComplexF32) runComplexScalStride2TaskC32 else runComplexScalStride2TaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn ComplexRealScalTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        alpha: Real(T),
        x: [*]T,
    };
}

fn runComplexRealScalStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexRealScalTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexRealScalStride2(T, task.n1 - task.n0, task.alpha, task.x + 2 * task.n0);
}

fn runComplexRealScalStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexRealScalStride2Task(ComplexF32, raw_tasks, index);
}

fn runComplexRealScalStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexRealScalStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelComplexRealScalStride2(comptime T: type, n: usize, alpha: Real(T), x: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]ComplexRealScalTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .x = x,
        };
    }
    const runner = if (T == ComplexF32) runComplexRealScalStride2TaskC32 else runComplexRealScalStride2TaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runAxpyStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpyStride2Real(T, task.n1 - task.n0, task.alpha, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runAxpyStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyStride2Task(f32, raw_tasks, index);
}

fn runAxpyStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyStride2Task(f64, raw_tasks, index);
}

fn runComplexAxpyStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(ComplexF32) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexAxpyF32Stride2(task.n1 - task.n0, task.alpha, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runComplexAxpyStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(ComplexF64) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexAxpyF64Stride2(task.n1 - task.n0, task.alpha, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

pub noinline fn parallelAxpyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = 0,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runAxpyStride2TaskF32 else runAxpyStride2TaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

pub noinline fn parallelComplexAxpyStride2C32(n: usize, alpha: ComplexF32, x: [*]const ComplexF32, y: [*]ComplexF32) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(ComplexF32) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = zero(ComplexF32),
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    return runLevel1Tasks(runComplexAxpyStride2TaskC32, @ptrCast(&tasks), task_count);
}

pub noinline fn parallelComplexAxpyStride2C64(n: usize, alpha: ComplexF64, x: [*]const ComplexF64, y: [*]ComplexF64) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(ComplexF64) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = zero(ComplexF64),
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    return runLevel1Tasks(runComplexAxpyStride2TaskC64, @ptrCast(&tasks), task_count);
}

fn runAxpbyStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpbyStride2Real(T, task.n1 - task.n0, task.alpha, task.x + 2 * task.n0, task.beta, task.y + 2 * task.n0);
}

fn runAxpbyStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyStride2Task(f32, raw_tasks, index);
}

fn runAxpbyStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyStride2Task(f64, raw_tasks, index);
}

fn runComplexAxpbyStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (comptime T == ComplexF32) {
        complexAxpbyF32Stride2(
            task.n1 - task.n0,
            task.alpha.re,
            task.alpha.im,
            task.x + 2 * task.n0,
            task.beta.re,
            task.beta.im,
            task.y + 2 * task.n0,
        );
    } else {
        complexAxpbyF64Stride2(task.n1 - task.n0, task.alpha, task.x + 2 * task.n0, task.beta, task.y + 2 * task.n0);
    }
}

fn runComplexAxpbyStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpbyStride2Task(ComplexF32, raw_tasks, index);
}

fn runComplexAxpbyStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpbyStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelAxpbyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = beta,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runAxpbyStride2TaskF32 else runAxpbyStride2TaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

pub noinline fn parallelComplexAxpbyStride2(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = beta,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == ComplexF32) runComplexAxpbyStride2TaskC32 else runComplexAxpbyStride2TaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runDotStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const DotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = dotStride2Real(T, task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

fn runDotStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runDotStride2Task(f32, raw_tasks, index);
}

fn runDotStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runDotStride2Task(f64, raw_tasks, index);
}

fn DotTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        y: [*]const T,
        out: *T,
    };
}

const DotF32AccF64Task = struct {
    n0: usize,
    n1: usize,
    x: [*]const f32,
    y: [*]const f32,
    out: *f64,
};

fn runDotF32AccF64Stride2Task(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const DotF32AccF64Task = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = dotF32AccF64Stride2(task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0);
}

pub noinline fn parallelDotF32AccF64Stride2(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]f64 = undefined;
    var tasks: [core_pool.max_tasks]DotF32AccF64Task = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .out = &partial[task_index],
        };
    }
    if (!runLevel1Tasks(runDotF32AccF64Stride2Task, @ptrCast(&tasks), task_count)) return null;

    var result: f64 = 0;
    for (partial[0..task_count]) |value| result += value;
    return result;
}

pub noinline fn parallelDotStride2Real(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var tasks: [core_pool.max_tasks]DotTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32) runDotStride2TaskF32 else runDotStride2TaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |value| result += value;
    return result;
}

fn ComplexDotTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        y: [*]const T,
        conjx: bool,
        out: *T,
    };
}

fn runDotComplexStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexDotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = complexDotStride2(T, task.n1 - task.n0, task.x + 2 * task.n0, task.y + 2 * task.n0, task.conjx);
}

fn runDotComplexStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runDotComplexStride2Task(ComplexF32, raw_tasks, index);
}

fn runDotComplexStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runDotComplexStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelDotStride2Complex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var tasks: [core_pool.max_tasks]ComplexDotTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = zero(T);
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .conjx = conjx,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == ComplexF32) runDotComplexStride2TaskC32 else runDotComplexStride2TaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result = zero(T);
    for (partial[0..task_count]) |value| result = add(T, result, value);
    return result;
}

fn runAsumStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = asumStride2Real(T, task.n1 - task.n0, task.x + 2 * task.n0);
}

fn runAsumStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAsumStride2Task(f32, raw_tasks, index);
}

fn runAsumStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAsumStride2Task(f64, raw_tasks, index);
}

fn ComplexAsumTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        out: *Real(T),
    };
}

fn runComplexAsumStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexAsumTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = asumStride2Complex(T, task.n1 - task.n0, task.x + 2 * task.n0);
}

fn runComplexAsumStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAsumStride2Task(ComplexF32, raw_tasks, index);
}

fn runComplexAsumStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAsumStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelAsumStride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = 0,
            .beta = 0,
            .x = x,
            .y = undefined,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32) runAsumStride2TaskF32 else runAsumStride2TaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |value| result += value;
    return result;
}

pub noinline fn parallelAsumStride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const R = Real(T);
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]R = undefined;
    var tasks: [core_pool.max_tasks]ComplexAsumTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == ComplexF32) runComplexAsumStride2TaskC32 else runComplexAsumStride2TaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result: R = 0;
    for (partial[0..task_count]) |value| result += value;
    return result;
}

fn Nrm2Task(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        out: *T,
        ok: *bool,
    };
}

fn runNrm2Stride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const Nrm2Task(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (nrm2Stride2Real(T, task.n1 - task.n0, task.x + 2 * task.n0)) |result| {
        task.out.* = result;
        task.ok.* = true;
    } else {
        task.out.* = 0;
        task.ok.* = false;
    }
}

fn runNrm2Stride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runNrm2Stride2Task(f32, raw_tasks, index);
}

fn runNrm2Stride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runNrm2Stride2Task(f64, raw_tasks, index);
}

fn ComplexNrm2Task(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        out: *Real(T),
        ok: *bool,
    };
}

fn runComplexNrm2Stride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexNrm2Task(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (nrm2Stride2Complex(T, task.n1 - task.n0, task.x + 2 * task.n0)) |result| {
        task.out.* = result;
        task.ok.* = true;
    } else {
        task.out.* = 0;
        task.ok.* = false;
    }
}

fn runComplexNrm2Stride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexNrm2Stride2Task(ComplexF32, raw_tasks, index);
}

fn runComplexNrm2Stride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexNrm2Stride2Task(ComplexF64, raw_tasks, index);
}

fn combineNrm2Partials(comptime T: type, partial: []const T) T {
    var scale: T = 0;
    var ssq: T = 1;
    for (partial) |norm| {
        if (!std.math.isFinite(norm)) return norm;
        if (norm == 0) continue;
        if (scale < norm) {
            const r = scale / norm;
            ssq = 1 + ssq * r * r;
            scale = norm;
        } else {
            const r = norm / scale;
            ssq += r * r;
        }
    }
    return scale * @sqrt(ssq);
}

pub noinline fn parallelNrm2Stride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var ok: [core_pool.max_tasks]bool = undefined;
    var tasks: [core_pool.max_tasks]Nrm2Task(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        ok[task_index] = false;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .out = &partial[task_index],
            .ok = &ok[task_index],
        };
    }
    const runner = if (T == f32) runNrm2Stride2TaskF32 else runNrm2Stride2TaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;
    for (ok[0..task_count]) |task_ok| if (!task_ok) return null;
    return combineNrm2Partials(T, partial[0..task_count]);
}

pub noinline fn parallelNrm2Stride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const R = Real(T);
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]R = undefined;
    var ok: [core_pool.max_tasks]bool = undefined;
    var tasks: [core_pool.max_tasks]ComplexNrm2Task(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        ok[task_index] = false;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .out = &partial[task_index],
            .ok = &ok[task_index],
        };
    }
    const runner = if (T == ComplexF32) runComplexNrm2Stride2TaskC32 else runComplexNrm2Stride2TaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;
    for (ok[0..task_count]) |task_ok| if (!task_ok) return null;
    return combineNrm2Partials(R, partial[0..task_count]);
}

fn IamaxTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        out: *BlasInt,
    };
}

fn iamaxStride2TaskCandidate(comptime T: type, n: usize, x: [*]const T) BlasInt {
    var n0: usize = 0;
    while (n0 < n and std.math.isNan(abs1(T, x[2 * n0]))) : (n0 += 1) {}
    if (n0 == n) return 0;
    const local = if (comptime isReal(T))
        iamaxStride2Real(T, n - n0, x + 2 * n0)
    else
        iamaxStride2Complex(T, n - n0, x + 2 * n0);
    return @intCast(n0 + @as(usize, @intCast(local)));
}

fn runIamaxStride2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const IamaxTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = iamaxStride2TaskCandidate(T, task.n1 - task.n0, task.x + 2 * task.n0);
}

fn runIamaxStride2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxStride2Task(f32, raw_tasks, index);
}

fn runIamaxStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxStride2Task(f64, raw_tasks, index);
}

fn runIamaxStride2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxStride2Task(ComplexF32, raw_tasks, index);
}

fn runIamaxStride2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxStride2Task(ComplexF64, raw_tasks, index);
}

pub noinline fn parallelIamaxStride2(comptime T: type, n: usize, x: [*]const T) ?BlasInt {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    const task_count = parallelTaskCount(n, 32 * 1024, 32);
    if (task_count <= 1) return null;
    if (std.math.isNan(abs1(T, x[0]))) return 1;

    var partial: [core_pool.max_tasks]BlasInt = undefined;
    var tasks: [core_pool.max_tasks]IamaxTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32)
        runIamaxStride2TaskF32
    else if (T == f64)
        runIamaxStride2TaskF64
    else if (T == ComplexF32)
        runIamaxStride2TaskC32
    else
        runIamaxStride2TaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var best: usize = 0;
    var best_abs = abs1(T, x[0]);
    for (tasks[0..task_count], partial[0..task_count]) |task, local_result| {
        if (local_result <= 0) continue;
        const global_index = task.n0 + @as(usize, @intCast(local_result - 1));
        const candidate_abs = abs1(T, x[2 * global_index]);
        if (candidate_abs > best_abs) {
            best_abs = candidate_abs;
            best = global_index;
        }
    }
    return @intCast(best + 1);
}
