// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const core_pool = @import("../execution/thread_pool.zig");
const vector_binary_kernels = @import("../../kernels/dispatch/vector_binary.zig");
const vector_unary_kernels = @import("../../kernels/dispatch/vector_unary.zig");

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

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const vectorGet = indexing.vectorGet;
const vectorSet = indexing.vectorSet;

extern fn memcpy(noalias dest: [*]u8, noalias src: [*]const u8, n: usize) callconv(.c) [*]u8;
extern fn memmove(dest: [*]u8, src: [*]const u8, n: usize) callconv(.c) [*]u8;

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

inline fn asRealPtr(comptime T: type, ptr: [*]T) [*]Real(T) {
    return @ptrCast(ptr);
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

inline fn storeVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]T, index: usize, value: @Vector(lane_count, T)) void {
    const V = @Vector(lane_count, T);
    @as(*align(1) V, @ptrCast(ptr + index)).* = value;
}

inline fn sameAddress(x: [*]const u8, y: [*]const u8) bool {
    return @intFromPtr(x) == @intFromPtr(y);
}

inline fn byteRangesOverlap(x: [*]const u8, y: [*]const u8, n: usize) bool {
    if (n == 0) return false;
    const xp = @intFromPtr(x);
    const yp = @intFromPtr(y);
    if (xp == yp) return true;
    if (xp < yp) return yp - xp < n;
    return xp - yp < n;
}

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) void {
    if (vector_unary_kernels.scalUnitReal(T, n, alpha, x)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            storeVec(T, lane_count, x, offset, loadVec(T, lane_count, x, offset) * alpha_v);
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        storeVec(T, lane_count, x, i, loadVec(T, lane_count, x, i) * alpha_v);
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const alpha_tail: TailV = @splat(alpha);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                storeVec(T, tail_lanes, x, i, loadVec(T, tail_lanes, x, i) * alpha_tail);
            }
        }
    }
    while (i < n) : (i += 1) x[i] *= alpha;
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) void {
    if (n_bytes == 0) return;
    if (sameAddress(x, y)) return;
    if (byteRangesOverlap(x, y, n_bytes)) {
        _ = memmove(y, x, n_bytes);
        return;
    }
    if (vector_binary_kernels.copyBytes(n_bytes, x, y)) return;
    _ = memcpy(y, x, n_bytes);
}

pub fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    copyBytes(n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    copyUnit(T, n, x, y);
}

fn swapUnit(comptime T: type, n: usize, x: [*]T, y: [*]T) void {
    if (comptime isReal(T)) {
        if (vector_binary_kernels.swapUnitReal(T, n, x, y)) return;
        const lane_count = lanes(T);
        const unroll_count = unroll(T);
        var i: usize = 0;
        while (i + unroll_count <= n) : (i += unroll_count) {
            inline for (0..4) |k| {
                const offset = i + k * lane_count;
                const xv = loadVec(T, lane_count, x, offset);
                const yv = loadVec(T, lane_count, y, offset);
                storeVec(T, lane_count, x, offset, yv);
                storeVec(T, lane_count, y, offset, xv);
            }
        }
        while (i + lane_count <= n) : (i += lane_count) {
            const xv = loadVec(T, lane_count, x, i);
            const yv = loadVec(T, lane_count, y, i);
            storeVec(T, lane_count, x, i, yv);
            storeVec(T, lane_count, y, i, xv);
        }
        inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
            if (comptime tail_lanes > 1) {
                while (i + tail_lanes <= n) : (i += tail_lanes) {
                    const xv = loadVec(T, tail_lanes, x, i);
                    const yv = loadVec(T, tail_lanes, y, i);
                    storeVec(T, tail_lanes, x, i, yv);
                    storeVec(T, tail_lanes, y, i, xv);
                }
            }
        }
        while (i < n) : (i += 1) {
            const t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    } else {
        for (0..n) |i| {
            const t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    }
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    if (vector_binary_kernels.axpyUnitReal(T, n, alpha, x, y)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(T, lane_count, x, offset);
            const yv = loadVec(T, lane_count, y, offset);
            storeVec(T, lane_count, y, offset, @mulAdd(V, xv, alpha_v, yv));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, y, i, @mulAdd(V, xv, alpha_v, yv));
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const alpha_tail: TailV = @splat(alpha);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const xv = loadVec(T, tail_lanes, x, i);
                const yv = loadVec(T, tail_lanes, y, i);
                storeVec(T, tail_lanes, y, i, @mulAdd(TailV, xv, alpha_tail, yv));
            }
        }
    }
    while (i < n) : (i += 1) y[i] = @mulAdd(T, alpha, x[i], y[i]);
}

fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) void {
    if (vector_binary_kernels.axpbyUnitReal(T, n, alpha, x, beta, y)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(T, lane_count, x, offset);
            const yv = loadVec(T, lane_count, y, offset);
            storeVec(T, lane_count, y, offset, @mulAdd(V, xv, alpha_v, yv * beta_v));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, y, i, @mulAdd(V, xv, alpha_v, yv * beta_v));
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const alpha_tail: TailV = @splat(alpha);
            const beta_tail: TailV = @splat(beta);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const xv = loadVec(T, tail_lanes, x, i);
                const yv = loadVec(T, tail_lanes, y, i);
                storeVec(T, tail_lanes, y, i, @mulAdd(TailV, xv, alpha_tail, yv * beta_tail));
            }
        }
    }
    while (i < n) : (i += 1) y[i] = @mulAdd(T, alpha, x[i], beta * y[i]);
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) T {
    if (vector_binary_kernels.dotUnitReal(T, n, x, y)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 = @mulAdd(V, loadVec(T, lane_count, x, i), loadVec(T, lane_count, y, i), acc0);
        acc1 = @mulAdd(V, loadVec(T, lane_count, x, i + lane_count), loadVec(T, lane_count, y, i + lane_count), acc1);
        acc2 = @mulAdd(V, loadVec(T, lane_count, x, i + 2 * lane_count), loadVec(T, lane_count, y, i + 2 * lane_count), acc2);
        acc3 = @mulAdd(V, loadVec(T, lane_count, x, i + 3 * lane_count), loadVec(T, lane_count, y, i + 3 * lane_count), acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc = @mulAdd(V, loadVec(T, lane_count, x, i), loadVec(T, lane_count, y, i), acc);
    }
    var sum: T = @reduce(.Add, acc);
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            var tail_acc: TailV = @splat(0);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                tail_acc = @mulAdd(TailV, loadVec(T, tail_lanes, x, i), loadVec(T, tail_lanes, y, i), tail_acc);
            }
            sum += @reduce(.Add, tail_acc);
        }
    }
    while (i < n) : (i += 1) sum = @mulAdd(T, x[i], y[i], sum);
    return sum;
}

fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) T {
    if (vector_unary_kernels.asumUnitReal(T, n, x)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 += @abs(loadVec(T, lane_count, x, i));
        acc1 += @abs(loadVec(T, lane_count, x, i + lane_count));
        acc2 += @abs(loadVec(T, lane_count, x, i + 2 * lane_count));
        acc3 += @abs(loadVec(T, lane_count, x, i + 3 * lane_count));
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc += @abs(loadVec(T, lane_count, x, i));
    }
    var sum: T = @reduce(.Add, acc);
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            var tail_acc: @Vector(tail_lanes, T) = @splat(0);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                tail_acc += @abs(loadVec(T, tail_lanes, x, i));
            }
            sum += @reduce(.Add, tail_acc);
        }
    }
    while (i < n) : (i += 1) sum += @abs(x[i]);
    return sum;
}

fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (vector_unary_kernels.nrm2UnitReal(T, n, x)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadVec(T, lane_count, x, i + k * lane_count)));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        max_v = @max(max_v, @abs(loadVec(T, lane_count, x, i)));
    }
    var scale: T = @reduce(.Max, max_v);
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            var tail_max: @Vector(tail_lanes, T) = @splat(0);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                tail_max = @max(tail_max, @abs(loadVec(T, tail_lanes, x, i)));
            }
            scale = @max(scale, @reduce(.Max, tail_max));
        }
    }
    while (i < n) : (i += 1) scale = @max(scale, @abs(x[i]));
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const inv_scale_v: V = @splat(1 / scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    i = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        const v0 = loadVec(T, lane_count, x, i) * inv_scale_v;
        const v1 = loadVec(T, lane_count, x, i + lane_count) * inv_scale_v;
        const v2 = loadVec(T, lane_count, x, i + 2 * lane_count) * inv_scale_v;
        const v3 = loadVec(T, lane_count, x, i + 3 * lane_count) * inv_scale_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        const v = loadVec(T, lane_count, x, i) * inv_scale_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: T = @reduce(.Add, acc);
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const inv_scale_tail: TailV = @splat(1 / scale);
            var tail_acc: TailV = @splat(0);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const v = loadVec(T, tail_lanes, x, i) * inv_scale_tail;
                tail_acc = @mulAdd(TailV, v, v, tail_acc);
            }
            ssq += @reduce(.Add, tail_acc);
        }
    }
    while (i < n) : (i += 1) {
        const v = x[i] / scale;
        ssq = @mulAdd(T, v, v, ssq);
    }
    return scale * @sqrt(ssq);
}

fn pairSwapMask(comptime lane_count: comptime_int) @Vector(lane_count, i32) {
    comptime var values: [lane_count]i32 = undefined;
    inline for (0..lane_count) |i| {
        values[i] = if (i % 2 == 0) @intCast(i + 1) else @intCast(i - 1);
    }
    return values;
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

fn complexScalUnit(comptime T: type, n: usize, alpha: T, x: [*]T) void {
    if (vector_unary_kernels.scalUnitComplex(T, n, alpha, x)) return;
    const R = Real(T);
    const lane_count = lanes(R);
    const unroll_count = unroll(R);
    const V = @Vector(lane_count, R);
    const real_n = 2 * n;
    const real_x = asRealPtr(T, x);
    const re_v: V = @splat(realPart(T, alpha));
    const im_sign_v = pairSignVector(R, lane_count, imagPart(T, alpha));
    const swap_mask = comptime pairSwapMask(lane_count);
    var i: usize = 0;
    while (i + unroll_count <= real_n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(R, lane_count, real_x, offset);
            const swapped = @shuffle(R, xv, undefined, swap_mask);
            storeVec(R, lane_count, real_x, offset, @mulAdd(V, xv, re_v, swapped * im_sign_v));
        }
    }
    while (i + lane_count <= real_n) : (i += lane_count) {
        const xv = loadVec(R, lane_count, real_x, i);
        const swapped = @shuffle(R, xv, undefined, swap_mask);
        storeVec(R, lane_count, real_x, i, @mulAdd(V, xv, re_v, swapped * im_sign_v));
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, R);
            const tail_re_v: TailV = @splat(realPart(T, alpha));
            const tail_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, alpha));
            const tail_swap_mask = comptime pairSwapMask(tail_lanes);
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                const xv = loadVec(R, tail_lanes, real_x, i);
                const swapped = @shuffle(R, xv, undefined, tail_swap_mask);
                storeVec(R, tail_lanes, real_x, i, @mulAdd(TailV, xv, tail_re_v, swapped * tail_im_sign_v));
            }
        }
    }
    while (i < real_n) : (i += 2) {
        const re = real_x[i];
        const im = real_x[i + 1];
        real_x[i] = realPart(T, alpha) * re - imagPart(T, alpha) * im;
        real_x[i + 1] = realPart(T, alpha) * im + imagPart(T, alpha) * re;
    }
}

fn complexAxpyUnit(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    if (vector_binary_kernels.axpyUnitComplex(T, n, alpha, x, y)) return;
    const R = Real(T);
    const lane_count = lanes(R);
    const unroll_count = unroll(R);
    const V = @Vector(lane_count, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    const real_y = asRealPtr(T, y);
    const re_v: V = @splat(realPart(T, alpha));
    const im_sign_v = pairSignVector(R, lane_count, imagPart(T, alpha));
    const swap_mask = comptime pairSwapMask(lane_count);
    var i: usize = 0;
    while (i + unroll_count <= real_n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(R, lane_count, real_x, offset);
            const yv = loadVec(R, lane_count, real_y, offset);
            const swapped = @shuffle(R, xv, undefined, swap_mask);
            storeVec(R, lane_count, real_y, offset, yv + @mulAdd(V, xv, re_v, swapped * im_sign_v));
        }
    }
    while (i + lane_count <= real_n) : (i += lane_count) {
        const xv = loadVec(R, lane_count, real_x, i);
        const yv = loadVec(R, lane_count, real_y, i);
        const swapped = @shuffle(R, xv, undefined, swap_mask);
        storeVec(R, lane_count, real_y, i, yv + @mulAdd(V, xv, re_v, swapped * im_sign_v));
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, R);
            const tail_re_v: TailV = @splat(realPart(T, alpha));
            const tail_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, alpha));
            const tail_swap_mask = comptime pairSwapMask(tail_lanes);
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                const xv = loadVec(R, tail_lanes, real_x, i);
                const yv = loadVec(R, tail_lanes, real_y, i);
                const swapped = @shuffle(R, xv, undefined, tail_swap_mask);
                storeVec(R, tail_lanes, real_y, i, yv + @mulAdd(TailV, xv, tail_re_v, swapped * tail_im_sign_v));
            }
        }
    }
    while (i < real_n) : (i += 2) {
        const re = real_x[i];
        const im = real_x[i + 1];
        real_y[i] += realPart(T, alpha) * re - imagPart(T, alpha) * im;
        real_y[i + 1] += realPart(T, alpha) * im + imagPart(T, alpha) * re;
    }
}

fn complexAxpbyUnit(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) void {
    const R = Real(T);
    const lane_count = lanes(R);
    const unroll_count = unroll(R);
    const V = @Vector(lane_count, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    const real_y = asRealPtr(T, y);
    const alpha_re_v: V = @splat(realPart(T, alpha));
    const alpha_im_sign_v = pairSignVector(R, lane_count, imagPart(T, alpha));
    const beta_re_v: V = @splat(realPart(T, beta));
    const beta_im_sign_v = pairSignVector(R, lane_count, imagPart(T, beta));
    const swap_mask = comptime pairSwapMask(lane_count);
    var i: usize = 0;
    while (i + unroll_count <= real_n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(R, lane_count, real_x, offset);
            const yv = loadVec(R, lane_count, real_y, offset);
            const x_swapped = @shuffle(R, xv, undefined, swap_mask);
            const y_swapped = @shuffle(R, yv, undefined, swap_mask);
            const x_term = @mulAdd(V, xv, alpha_re_v, x_swapped * alpha_im_sign_v);
            const y_term = @mulAdd(V, yv, beta_re_v, y_swapped * beta_im_sign_v);
            storeVec(R, lane_count, real_y, offset, x_term + y_term);
        }
    }
    while (i + lane_count <= real_n) : (i += lane_count) {
        const xv = loadVec(R, lane_count, real_x, i);
        const yv = loadVec(R, lane_count, real_y, i);
        const x_swapped = @shuffle(R, xv, undefined, swap_mask);
        const y_swapped = @shuffle(R, yv, undefined, swap_mask);
        const x_term = @mulAdd(V, xv, alpha_re_v, x_swapped * alpha_im_sign_v);
        const y_term = @mulAdd(V, yv, beta_re_v, y_swapped * beta_im_sign_v);
        storeVec(R, lane_count, real_y, i, x_term + y_term);
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, R);
            const tail_alpha_re_v: TailV = @splat(realPart(T, alpha));
            const tail_alpha_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, alpha));
            const tail_beta_re_v: TailV = @splat(realPart(T, beta));
            const tail_beta_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, beta));
            const tail_swap_mask = comptime pairSwapMask(tail_lanes);
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                const xv = loadVec(R, tail_lanes, real_x, i);
                const yv = loadVec(R, tail_lanes, real_y, i);
                const x_swapped = @shuffle(R, xv, undefined, tail_swap_mask);
                const y_swapped = @shuffle(R, yv, undefined, tail_swap_mask);
                const x_term = @mulAdd(TailV, xv, tail_alpha_re_v, x_swapped * tail_alpha_im_sign_v);
                const y_term = @mulAdd(TailV, yv, tail_beta_re_v, y_swapped * tail_beta_im_sign_v);
                storeVec(R, tail_lanes, real_y, i, x_term + y_term);
            }
        }
    }
    while (i < real_n) : (i += 2) {
        const xr = real_x[i];
        const xi = real_x[i + 1];
        const yr = real_y[i];
        const yi = real_y[i + 1];
        real_y[i] = realPart(T, alpha) * xr - imagPart(T, alpha) * xi + realPart(T, beta) * yr - imagPart(T, beta) * yi;
        real_y[i + 1] = realPart(T, alpha) * xi + imagPart(T, alpha) * xr + realPart(T, beta) * yi + imagPart(T, beta) * yr;
    }
}

inline fn complexDotAccumulateVec(
    comptime R: type,
    comptime lane_count: comptime_int,
    x: [*]const R,
    y: [*]const R,
    offset: usize,
    comptime swap_mask: @Vector(lane_count, i32),
    re_sign: @Vector(lane_count, R),
    im_sign: @Vector(lane_count, R),
    re_acc: *@Vector(lane_count, R),
    im_acc: *@Vector(lane_count, R),
) void {
    const V = @Vector(lane_count, R);
    const xv = loadVec(R, lane_count, x, offset);
    const yv = loadVec(R, lane_count, y, offset);
    const y_swap = @shuffle(R, yv, undefined, swap_mask);
    re_acc.* = @mulAdd(V, xv * yv, re_sign, re_acc.*);
    im_acc.* = @mulAdd(V, xv * y_swap, im_sign, im_acc.*);
}

fn complexDotUnit(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) T {
    const R = Real(T);
    const lane_count = lanes(R);
    const unroll_count = unroll(R);
    const V = @Vector(lane_count, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    const real_y = asConstRealPtr(T, y);
    const swap_mask = comptime pairSwapMask(lane_count);
    const re_sign: V = if (conjx)
        @splat(1)
    else
        pairPatternVector(R, lane_count, 1, -1);
    const im_sign: V = if (conjx)
        pairPatternVector(R, lane_count, 1, -1)
    else
        @splat(1);
    var re_acc0: V = @splat(0);
    var re_acc1: V = @splat(0);
    var re_acc2: V = @splat(0);
    var re_acc3: V = @splat(0);
    var im_acc0: V = @splat(0);
    var im_acc1: V = @splat(0);
    var im_acc2: V = @splat(0);
    var im_acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= real_n) : (i += unroll_count) {
        complexDotAccumulateVec(R, lane_count, real_x, real_y, i, swap_mask, re_sign, im_sign, &re_acc0, &im_acc0);
        complexDotAccumulateVec(R, lane_count, real_x, real_y, i + lane_count, swap_mask, re_sign, im_sign, &re_acc1, &im_acc1);
        complexDotAccumulateVec(R, lane_count, real_x, real_y, i + 2 * lane_count, swap_mask, re_sign, im_sign, &re_acc2, &im_acc2);
        complexDotAccumulateVec(R, lane_count, real_x, real_y, i + 3 * lane_count, swap_mask, re_sign, im_sign, &re_acc3, &im_acc3);
    }
    var re_acc = re_acc0 + re_acc1 + re_acc2 + re_acc3;
    var im_acc = im_acc0 + im_acc1 + im_acc2 + im_acc3;
    while (i + lane_count <= real_n) : (i += lane_count) {
        complexDotAccumulateVec(R, lane_count, real_x, real_y, i, swap_mask, re_sign, im_sign, &re_acc, &im_acc);
    }
    var re_sum: R = @reduce(.Add, re_acc);
    var im_sum: R = @reduce(.Add, im_acc);
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, R);
            const tail_swap_mask = comptime pairSwapMask(tail_lanes);
            const tail_re_sign: TailV = if (conjx)
                @splat(1)
            else
                pairPatternVector(R, tail_lanes, 1, -1);
            const tail_im_sign: TailV = if (conjx)
                pairPatternVector(R, tail_lanes, 1, -1)
            else
                @splat(1);
            var tail_re_acc: TailV = @splat(0);
            var tail_im_acc: TailV = @splat(0);
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                complexDotAccumulateVec(R, tail_lanes, real_x, real_y, i, tail_swap_mask, tail_re_sign, tail_im_sign, &tail_re_acc, &tail_im_acc);
            }
            re_sum += @reduce(.Add, tail_re_acc);
            im_sum += @reduce(.Add, tail_im_acc);
        }
    }
    while (i < real_n) : (i += 2) {
        const xr = real_x[i];
        const xi = real_x[i + 1];
        const yr = real_y[i];
        const yi = real_y[i + 1];
        if (conjx) {
            re_sum = @mulAdd(R, xi, yi, @mulAdd(R, xr, yr, re_sum));
            im_sum = @mulAdd(R, -xi, yr, @mulAdd(R, xr, yi, im_sum));
        } else {
            re_sum = @mulAdd(R, -xi, yi, @mulAdd(R, xr, yr, re_sum));
            im_sum = @mulAdd(R, xi, yr, @mulAdd(R, xr, yi, im_sum));
        }
    }
    return .{ .re = re_sum, .im = im_sum };
}

fn complexDotUnitBest(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) T {
    return vector_binary_kernels.dotUnitComplex(T, n, x, y, conjx) orelse complexDotUnit(T, n, x, y, conjx);
}

fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) BlasInt {
    if (n == 0) return 0;
    if (vector_unary_kernels.iamaxUnitReal(T, n, x)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    var best: usize = 0;
    var best_abs: T = @abs(x[0]);
    var i: usize = 1;

    while (i + unroll_count <= n) : (i += unroll_count) {
        var max_v: @Vector(lane_count, T) = @splat(0);
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadVec(T, lane_count, x, i + k * lane_count)));
        }
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + unroll_count;
            var j = i;
            while (j < end) : (j += 1) {
                const ax = @abs(x[j]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j;
                }
            }
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const max_v = @abs(loadVec(T, lane_count, x, i));
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + lane_count;
            var j = i;
            while (j < end) : (j += 1) {
                const ax = @abs(x[j]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j;
                }
            }
        }
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const max_v = @abs(loadVec(T, tail_lanes, x, i));
                if (@reduce(.Max, max_v) > best_abs) {
                    const end = i + tail_lanes;
                    var j = i;
                    while (j < end) : (j += 1) {
                        const ax = @abs(x[j]);
                        if (ax > best_abs) {
                            best_abs = ax;
                            best = j;
                        }
                    }
                }
            }
        }
    }
    while (i < n) : (i += 1) {
        const ax = @abs(x[i]);
        if (ax > best_abs) {
            best_abs = ax;
            best = i;
        }
    }
    return @intCast(best + 1);
}

fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) void {
    if (vector_binary_kernels.rotUnitReal(T, n, x, y, c, s)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const c_v: V = @splat(c);
    const s_v: V = @splat(s);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(T, lane_count, x, offset);
            const yv = loadVec(T, lane_count, y, offset);
            storeVec(T, lane_count, x, offset, @mulAdd(V, xv, c_v, yv * s_v));
            storeVec(T, lane_count, y, offset, yv * c_v - xv * s_v);
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, x, i, @mulAdd(V, xv, c_v, yv * s_v));
        storeVec(T, lane_count, y, i, yv * c_v - xv * s_v);
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const c_tail: TailV = @splat(c);
            const s_tail: TailV = @splat(s);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const xv = loadVec(T, tail_lanes, x, i);
                const yv = loadVec(T, tail_lanes, y, i);
                storeVec(T, tail_lanes, x, i, @mulAdd(TailV, xv, c_tail, yv * s_tail));
                storeVec(T, tail_lanes, y, i, yv * c_tail - xv * s_tail);
            }
        }
    }
    while (i < n) : (i += 1) {
        const xv = x[i];
        const yv = y[i];
        x[i] = @mulAdd(T, c, xv, s * yv);
        y[i] = c * yv - s * xv;
    }
}

fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) void {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var i: usize = 0;

    if (flag < 0) {
        const h11_v: V = @splat(h11);
        const h21_v: V = @splat(h21);
        const h12_v: V = @splat(h12);
        const h22_v: V = @splat(h22);
        while (i + unroll_count <= n) : (i += unroll_count) {
            inline for (0..4) |k| {
                const offset = i + k * lane_count;
                const w = loadVec(T, lane_count, x, offset);
                const z = loadVec(T, lane_count, y, offset);
                storeVec(T, lane_count, x, offset, @mulAdd(V, w, h11_v, z * h12_v));
                storeVec(T, lane_count, y, offset, @mulAdd(V, w, h21_v, z * h22_v));
            }
        }
        while (i + lane_count <= n) : (i += lane_count) {
            const w = loadVec(T, lane_count, x, i);
            const z = loadVec(T, lane_count, y, i);
            storeVec(T, lane_count, x, i, @mulAdd(V, w, h11_v, z * h12_v));
            storeVec(T, lane_count, y, i, @mulAdd(V, w, h21_v, z * h22_v));
        }
        inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
            if (comptime tail_lanes > 1) {
                const TailV = @Vector(tail_lanes, T);
                const h11_tail: TailV = @splat(h11);
                const h21_tail: TailV = @splat(h21);
                const h12_tail: TailV = @splat(h12);
                const h22_tail: TailV = @splat(h22);
                while (i + tail_lanes <= n) : (i += tail_lanes) {
                    const w = loadVec(T, tail_lanes, x, i);
                    const z = loadVec(T, tail_lanes, y, i);
                    storeVec(T, tail_lanes, x, i, @mulAdd(TailV, w, h11_tail, z * h12_tail));
                    storeVec(T, tail_lanes, y, i, @mulAdd(TailV, w, h21_tail, z * h22_tail));
                }
            }
        }
    } else if (flag == 0) {
        const h21_v: V = @splat(h21);
        const h12_v: V = @splat(h12);
        while (i + unroll_count <= n) : (i += unroll_count) {
            inline for (0..4) |k| {
                const offset = i + k * lane_count;
                const w = loadVec(T, lane_count, x, offset);
                const z = loadVec(T, lane_count, y, offset);
                storeVec(T, lane_count, x, offset, @mulAdd(V, z, h12_v, w));
                storeVec(T, lane_count, y, offset, @mulAdd(V, w, h21_v, z));
            }
        }
        while (i + lane_count <= n) : (i += lane_count) {
            const w = loadVec(T, lane_count, x, i);
            const z = loadVec(T, lane_count, y, i);
            storeVec(T, lane_count, x, i, @mulAdd(V, z, h12_v, w));
            storeVec(T, lane_count, y, i, @mulAdd(V, w, h21_v, z));
        }
        inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
            if (comptime tail_lanes > 1) {
                const TailV = @Vector(tail_lanes, T);
                const h21_tail: TailV = @splat(h21);
                const h12_tail: TailV = @splat(h12);
                while (i + tail_lanes <= n) : (i += tail_lanes) {
                    const w = loadVec(T, tail_lanes, x, i);
                    const z = loadVec(T, tail_lanes, y, i);
                    storeVec(T, tail_lanes, x, i, @mulAdd(TailV, z, h12_tail, w));
                    storeVec(T, tail_lanes, y, i, @mulAdd(TailV, w, h21_tail, z));
                }
            }
        }
    } else {
        const h11_v: V = @splat(h11);
        const h22_v: V = @splat(h22);
        while (i + unroll_count <= n) : (i += unroll_count) {
            inline for (0..4) |k| {
                const offset = i + k * lane_count;
                const w = loadVec(T, lane_count, x, offset);
                const z = loadVec(T, lane_count, y, offset);
                storeVec(T, lane_count, x, offset, @mulAdd(V, w, h11_v, z));
                storeVec(T, lane_count, y, offset, z * h22_v - w);
            }
        }
        while (i + lane_count <= n) : (i += lane_count) {
            const w = loadVec(T, lane_count, x, i);
            const z = loadVec(T, lane_count, y, i);
            storeVec(T, lane_count, x, i, @mulAdd(V, w, h11_v, z));
            storeVec(T, lane_count, y, i, z * h22_v - w);
        }
        inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
            if (comptime tail_lanes > 1) {
                const TailV = @Vector(tail_lanes, T);
                const h11_tail: TailV = @splat(h11);
                const h22_tail: TailV = @splat(h22);
                while (i + tail_lanes <= n) : (i += tail_lanes) {
                    const w = loadVec(T, tail_lanes, x, i);
                    const z = loadVec(T, tail_lanes, y, i);
                    storeVec(T, tail_lanes, x, i, @mulAdd(TailV, w, h11_tail, z));
                    storeVec(T, tail_lanes, y, i, z * h22_tail - w);
                }
            }
        }
    }

    while (i < n) : (i += 1) {
        const w = x[i];
        const z = y[i];
        if (flag < 0) {
            x[i] = @mulAdd(T, w, h11, z * h12);
            y[i] = @mulAdd(T, w, h21, z * h22);
        } else if (flag == 0) {
            x[i] = @mulAdd(T, z, h12, w);
            y[i] = @mulAdd(T, w, h21, z);
        } else {
            x[i] = @mulAdd(T, w, h11, z);
            y[i] = z * h22 - w;
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

fn parallelScalTaskCount(n: usize) usize {
    if (comptime builtin.cpu.arch == .x86_64) return parallelTaskCount(n, 32 * 1024, 32);
    if (n < 2 * 1024 * 1024) return 1;
    return parallelTaskCount(n, 170 * 1024, 6);
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

fn runScalTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    scalUnitReal(T, task.n1 - task.n0, task.alpha, task.y + task.n0);
}

fn runScalTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runScalTask(f32, raw_tasks, index);
}

fn runScalTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runScalTask(f64, raw_tasks, index);
}

fn runComplexScalTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexScalUnit(T, task.n1 - task.n0, task.alpha, task.y + task.n0);
}

fn runComplexScalTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexScalTask(ComplexF32, raw_tasks, index);
}

fn runComplexScalTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexScalTask(ComplexF64, raw_tasks, index);
}

fn parallelScalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) {
        if (T == f32 and n <= 2 * 1024 * 1024) return false;
    }
    const task_count = parallelScalTaskCount(n);
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
    const runner = if (T == f32) runScalTaskF32 else runScalTaskF64;
    if (comptime builtin.cpu.arch == .aarch64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn parallelComplexScalUnit(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64) return false;
    if (comptime builtin.cpu.arch == .aarch64) {
        if (n < 512 * 1024) return false;
    }
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else
        parallelTaskCount(n, 128 * 1024, 10);
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
    const runner = if (T == ComplexF32) runComplexScalTaskC32 else runComplexScalTaskC64;
    if (comptime builtin.cpu.arch == .aarch64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

const ByteCopyTask = struct {
    n0: usize,
    n1: usize,
    x: [*]const u8,
    y: [*]u8,
};

fn runCopyBytesTask(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ByteCopyTask = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    copyBytes(task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn parallelCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n_bytes < 4 * 1024 * 1024) return false;
    } else if (n_bytes < 4 * 1024 * 1024 or n_bytes > 8 * 1024 * 1024) return false;
    if (byteRangesOverlap(x, y, n_bytes)) return false;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n_bytes, 256 * 1024, 32)
    else
        parallelTaskCount(n_bytes, if (n_bytes == 8 * 1024 * 1024) 2 * 1024 * 1024 else 512 * 1024, 10);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]ByteCopyTask = undefined;
    const block_bytes: usize = 1024;
    const block_count = n_bytes / block_bytes;
    for (0..task_count) |task_index| {
        const n0, const n1 = if (block_count > 0) .{
            (task_index * block_count / task_count) * block_bytes,
            if (task_index + 1 == task_count) n_bytes else ((task_index + 1) * block_count / task_count) * block_bytes,
        } else .{
            task_index * n_bytes / task_count,
            (task_index + 1) * n_bytes / task_count,
        };
        tasks[task_index] = .{
            .n0 = n0,
            .n1 = n1,
            .x = x,
            .y = y,
        };
    }
    if (comptime builtin.cpu.arch == .aarch64) return core_pool.runLowLatency(runCopyBytesTask, @ptrCast(&tasks), task_count);
    return runLevel1Tasks(runCopyBytesTask, @ptrCast(&tasks), task_count);
}

fn runAxpyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpyUnitReal(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.y + task.n0);
}

fn runAxpyTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyTask(f32, raw_tasks, index);
}

fn runAxpyTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyTask(f64, raw_tasks, index);
}

fn runComplexAxpyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexAxpyUnit(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.y + task.n0);
}

fn runComplexAxpyTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpyTask(ComplexF32, raw_tasks, index);
}

fn runComplexAxpyTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpyTask(ComplexF64, raw_tasks, index);
}

fn parallelAxpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64) {
        if (T == f32 and n < 2 * 1024 * 1024) return false;
    }
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else
        parallelTaskCount(n, 170 * 1024, 6);
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
    const runner = if (T == f32) runAxpyTaskF32 else runAxpyTaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn parallelComplexAxpyUnit(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64) return false;
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
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == ComplexF32) runComplexAxpyTaskC32 else runComplexAxpyTaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runAxpbyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpbyUnitReal(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.beta, task.y + task.n0);
}

fn runAxpbyTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyTask(f32, raw_tasks, index);
}

fn runAxpbyTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyTask(f64, raw_tasks, index);
}

fn runComplexAxpbyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexAxpbyUnit(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.beta, task.y + task.n0);
}

fn runComplexAxpbyTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpbyTask(ComplexF32, raw_tasks, index);
}

fn runComplexAxpbyTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexAxpbyTask(ComplexF64, raw_tasks, index);
}

fn parallelAxpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    const task_count = parallelTaskCount(n, 170 * 1024, 6);
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
    const runner = if (T == f32) runAxpbyTaskF32 else runAxpbyTaskF64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn parallelComplexAxpbyUnit(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
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
    const runner = if (T == ComplexF32) runComplexAxpbyTaskC32 else runComplexAxpbyTaskC64;
    return runLevel1Tasks(runner, @ptrCast(&tasks), task_count);
}

fn runDotTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const DotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = dotUnitReal(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn runDotTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runDotTask(f32, raw_tasks, index);
}

fn runDotTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runDotTask(f64, raw_tasks, index);
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

fn parallelDotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime builtin.cpu.arch != .x86_64) {
        if (T == f32 and n <= 2 * 1024 * 1024) return null;
    }
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else
        parallelTaskCount(n, 128 * 1024, 10);
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
    const runner = if (T == f32) runDotTaskF32 else runDotTaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |v| result += v;
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

fn runDotComplexTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexDotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = complexDotUnitBest(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0, task.conjx);
}

fn runDotComplexTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runDotComplexTask(ComplexF32, raw_tasks, index);
}

fn runDotComplexTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runDotComplexTask(ComplexF64, raw_tasks, index);
}

fn parallelDotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime builtin.cpu.arch != .x86_64) {
        if (T == ComplexF32) return null;
    }
    if (n < 512 * 1024) return null;
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
    const runner = if (T == ComplexF32) runDotComplexTaskC32 else runDotComplexTaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result = zero(T);
    for (partial[0..task_count]) |v| result = add(T, result, v);
    return result;
}

fn runAsumTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = asumUnitReal(T, task.n1 - task.n0, task.x + task.n0);
}

fn runAsumTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAsumTask(f32, raw_tasks, index);
}

fn runAsumTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAsumTask(f64, raw_tasks, index);
}

fn parallelAsumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n < 512 * 1024) return null;
    } else {
        if (T == f32 and n <= 2 * 1024 * 1024) return null;
        if (n < 2 * 1024 * 1024) return null;
    }
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else
        parallelTaskCount(n, 96 * 1024, 10);
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
    const runner = if (T == f32) runAsumTaskF32 else runAsumTaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |v| result += v;
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

fn runNrm2Task(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const Nrm2Task(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    if (nrm2UnitReal(T, task.n1 - task.n0, task.x + task.n0)) |result| {
        task.out.* = result;
        task.ok.* = true;
    } else {
        task.out.* = 0;
        task.ok.* = false;
    }
}

fn runNrm2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runNrm2Task(f32, raw_tasks, index);
}

fn runNrm2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runNrm2Task(f64, raw_tasks, index);
}

fn combineNrm2Partials(comptime T: type, partial: []const T) T {
    var scale: T = 0;
    var ssq: T = 1;
    for (partial) |norm| {
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

fn parallelNrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (n < 512 * 1024) return null;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else
        parallelTaskCount(n, 128 * 1024, 10);
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
    const runner = if (T == f32) runNrm2TaskF32 else runNrm2TaskF64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    for (ok[0..task_count]) |task_ok| {
        if (!task_ok) return null;
    }
    return combineNrm2Partials(T, partial[0..task_count]);
}

pub fn scal(comptime T: type, n_: BlasInt, alpha: T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    if (isOne(T, alpha)) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) {
            if (parallelScalUnitReal(T, n, alpha, x)) return;
            return scalUnitReal(T, n, alpha, x);
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1) {
            if (imagPart(T, alpha) == 0) {
                const R = Real(T);
                const real_n = 2 * n;
                const real_alpha = realPart(T, alpha);
                const real_x = asRealPtr(T, x);
                if (parallelScalUnitReal(R, real_n, real_alpha, real_x)) return;
                return scalUnitReal(R, real_n, real_alpha, real_x);
            }
            if (parallelComplexScalUnit(T, n, alpha, x)) return;
            return complexScalUnit(T, n, alpha, x);
        }
    }
    const sx = startIndex(n_, incx_);
    for (0..n) |i| {
        const p = ix(sx, i, incx_);
        x[p] = mul(T, alpha, x[p]);
    }
}

pub fn rscal(comptime T: type, n_: BlasInt, alpha: Real(T), x: [*]T, incx_: BlasInt) void {
    if (alpha == 1) return;
    if (comptime isComplex(T)) {
        if (n_ <= 0 or incx_ == 0) return;
        if (incx_ == 1) {
            const R = Real(T);
            const real_n = 2 * toUsize(n_);
            const real_x = asRealPtr(T, x);
            if (parallelScalUnitReal(R, real_n, alpha, real_x)) return;
            return scalUnitReal(R, real_n, alpha, real_x);
        }
    }
    scal(T, n_, realScalar(T, alpha), x, incx_);
}

pub fn copy(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (incx_ == 1 and incy_ == 1) {
        const n_bytes = n * @sizeOf(T);
        if (parallelCopyBytes(n_bytes, @ptrCast(x), @ptrCast(y))) return;
        return copyBytes(n_bytes, @ptrCast(x), @ptrCast(y));
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| vectorSet(T, y, sy, i, incy_, vectorGet(T, x, sx, i, incx_));
}

pub fn swap(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) return swapUnit(Real(T), 2 * n, asRealPtr(T, x), asRealPtr(T, y));
    }
    if (incx_ == 1 and incy_ == 1) return swapUnit(T, n, x, y);
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
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (parallelAxpyUnitReal(T, n, alpha, x, y)) return;
            return axpyUnitReal(T, n, alpha, x, y);
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (imagPart(T, alpha) == 0) {
                const R = Real(T);
                const real_n = 2 * n;
                const real_alpha = realPart(T, alpha);
                const real_x = asConstRealPtr(T, x);
                const real_y = asRealPtr(T, y);
                if (parallelAxpyUnitReal(R, real_n, real_alpha, real_x, real_y)) return;
                return axpyUnitReal(R, real_n, real_alpha, real_x, real_y);
            }
            if (parallelComplexAxpyUnit(T, n, alpha, x, y)) return;
            return complexAxpyUnit(T, n, alpha, x, y);
        }
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, vectorGet(T, x, sx, i, incx_)));
    }
}

pub fn axpby(comptime T: type, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    if (isZero(T, alpha)) return scal(T, n_, beta, y, incy_);
    if (isOne(T, beta)) return axpy(T, n_, alpha, x, incx_, y, incy_);
    if (isOne(T, alpha) and isZero(T, beta)) return copy(T, n_, x, incx_, y, incy_);
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (parallelAxpbyUnitReal(T, n, alpha, x, beta, y)) return;
            return axpbyUnitReal(T, n, alpha, x, beta, y);
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (imagPart(T, alpha) == 0 and imagPart(T, beta) == 0) {
                const R = Real(T);
                const real_n = 2 * n;
                const real_alpha = realPart(T, alpha);
                const real_beta = realPart(T, beta);
                const real_x = asConstRealPtr(T, x);
                const real_y = asRealPtr(T, y);
                if (parallelAxpbyUnitReal(R, real_n, real_alpha, real_x, real_beta, real_y)) return;
                return axpbyUnitReal(R, real_n, real_alpha, real_x, real_beta, real_y);
            }
            if (parallelComplexAxpbyUnit(T, n, alpha, x, beta, y)) return;
            if (vector_binary_kernels.axpbyUnitComplex(T, n, alpha, x, beta, y)) return;
            return complexAxpbyUnit(T, n, alpha, x, beta, y);
        }
    }
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
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            return parallelDotUnitReal(T, n, x, y) orelse dotUnitReal(T, n, x, y);
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) {
            return parallelDotUnitComplex(T, n, x, y, conjx) orelse complexDotUnitBest(T, n, x, y, conjx);
        }
    }
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
    if (comptime isReal(T)) {
        if (incx_ == 1) return parallelAsumUnitReal(T, n, x) orelse asumUnitReal(T, n, x);
    } else if (comptime isComplex(T)) {
        if (incx_ == 1) {
            const R = Real(T);
            const real_n = 2 * n;
            const real_x = asConstRealPtr(T, x);
            return parallelAsumUnitReal(R, real_n, real_x) orelse asumUnitReal(R, real_n, real_x);
        }
    }
    const sx = startIndex(n_, incx_);
    var sum: Real(T) = 0;
    for (0..n) |i| sum += abs1(T, vectorGet(T, x, sx, i, incx_));
    return sum;
}

pub fn nrm2(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) {
            if (parallelNrm2UnitReal(T, n, x)) |result| return result;
            if (nrm2UnitReal(T, n, x)) |result| return result;
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1) {
            const R = Real(T);
            const real_n = 2 * n;
            const real_x = asConstRealPtr(T, x);
            if (parallelNrm2UnitReal(R, real_n, real_x)) |result| return result;
            if (nrm2UnitReal(R, real_n, real_x)) |result| return result;
        }
    }
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
    if (comptime isReal(T)) {
        if (incx_ == 1) return iamaxUnitReal(T, n, x);
    }
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
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) return rotUnitReal(T, n, x, y, c, s);
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1 and imagPart(T, s) == 0) {
            return rotUnitReal(Real(T), 2 * n, asRealPtr(T, x), asRealPtr(T, y), c, realPart(T, s));
        }
    }
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
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) return rotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22);
    }
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
