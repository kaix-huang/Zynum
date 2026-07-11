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
        const n_bytes = n * @sizeOf(T);
        if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), n_bytes)) {
            if (vector_binary_kernels.swapUnitReal(T, n, x, y)) return;
        }
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

fn swapStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T) void {
    const lane_count = lanes(T);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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

fn axpyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        const xb = loadStride2Block(T, lane_count, x, i);
        const yb = loadStride2Block(T, lane_count, y, i);
        const result = @mulAdd(V, alpha_v, xb.active, yb.active);
        storeStride2Block(T, lane_count, y, i, yb, result);
    }
    while (i < n) : (i += 1) y[2 * i] = @mulAdd(T, alpha, x[2 * i], y[2 * i]);
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

fn dotStride2Real(comptime T: type, n: usize, x: [*]const T, y: [*]const T) T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i), loadStride2Vec(T, lane_count, y, i), acc0);
        acc1 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + lane_count), loadStride2Vec(T, lane_count, y, i + lane_count), acc1);
        acc2 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + 2 * lane_count), loadStride2Vec(T, lane_count, y, i + 2 * lane_count), acc2);
        acc3 = @mulAdd(V, loadStride2Vec(T, lane_count, x, i + 3 * lane_count), loadStride2Vec(T, lane_count, y, i + 3 * lane_count), acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc = @mulAdd(V, loadStride2Vec(T, lane_count, x, i), loadStride2Vec(T, lane_count, y, i), acc);
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(T, x[2 * i], y[2 * i], sum);
    return sum;
}

fn dotF32AccF64Unit(n: usize, x: [*]const f32, y: [*]const f32) f64 {
    const F32V = @Vector(4, f32);
    const F64V = @Vector(4, f64);
    var acc0: F64V = @splat(0);
    var acc1: F64V = @splat(0);
    var acc2: F64V = @splat(0);
    var acc3: F64V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: F64V = @floatCast(loadVec(f32, 4, x, i));
        const y0: F64V = @floatCast(loadVec(f32, 4, y, i));
        const x1: F64V = @floatCast(loadVec(f32, 4, x, i + 4));
        const y1: F64V = @floatCast(loadVec(f32, 4, y, i + 4));
        const x2: F64V = @floatCast(loadVec(f32, 4, x, i + 8));
        const y2: F64V = @floatCast(loadVec(f32, 4, y, i + 8));
        const x3: F64V = @floatCast(loadVec(f32, 4, x, i + 12));
        const y3: F64V = @floatCast(loadVec(f32, 4, y, i + 12));
        acc0 = @mulAdd(F64V, x0, y0, acc0);
        acc1 = @mulAdd(F64V, x1, y1, acc1);
        acc2 = @mulAdd(F64V, x2, y2, acc2);
        acc3 = @mulAdd(F64V, x3, y3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + 4 <= n) : (i += 4) {
        const xv: F64V = @floatCast(@as(*align(1) const F32V, @ptrCast(x + i)).*);
        const yv: F64V = @floatCast(@as(*align(1) const F32V, @ptrCast(y + i)).*);
        acc = @mulAdd(F64V, xv, yv, acc);
    }
    var sum: f64 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(f64, @as(f64, x[i]), @as(f64, y[i]), sum);
    return sum;
}

fn dotF32AccF64Stride2(n: usize, x: [*]const f32, y: [*]const f32) f64 {
    const F64V = @Vector(4, f64);
    var acc0: F64V = @splat(0);
    var acc1: F64V = @splat(0);
    var acc2: F64V = @splat(0);
    var acc3: F64V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
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
    while (i + 4 <= n) : (i += 4) {
        const xv: F64V = @floatCast(loadStride2Vec(f32, 4, x, i));
        const yv: F64V = @floatCast(loadStride2Vec(f32, 4, y, i));
        acc = @mulAdd(F64V, xv, yv, acc);
    }
    var sum: f64 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(f64, @as(f64, x[2 * i]), @as(f64, y[2 * i]), sum);
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

fn asumStride2Real(comptime T: type, n: usize, x: [*]const T) T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 += @abs(loadStride2Vec(T, lane_count, x, i));
        acc1 += @abs(loadStride2Vec(T, lane_count, x, i + lane_count));
        acc2 += @abs(loadStride2Vec(T, lane_count, x, i + 2 * lane_count));
        acc3 += @abs(loadStride2Vec(T, lane_count, x, i + 3 * lane_count));
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc += @abs(loadStride2Vec(T, lane_count, x, i));
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += @abs(x[2 * i]);
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

fn nrm2Stride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadStride2Vec(T, lane_count, x, i + k * lane_count)));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        max_v = @max(max_v, @abs(loadStride2Vec(T, lane_count, x, i)));
    }
    var scale: T = @reduce(.Max, max_v);
    while (i < n) : (i += 1) scale = @max(scale, @abs(x[2 * i]));
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const inv_scale_v: V = @splat(1 / scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    i = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        const v0 = loadStride2Vec(T, lane_count, x, i) * inv_scale_v;
        const v1 = loadStride2Vec(T, lane_count, x, i + lane_count) * inv_scale_v;
        const v2 = loadStride2Vec(T, lane_count, x, i + 2 * lane_count) * inv_scale_v;
        const v3 = loadStride2Vec(T, lane_count, x, i + 3 * lane_count) * inv_scale_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        const v = loadStride2Vec(T, lane_count, x, i) * inv_scale_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const v = x[2 * i] / scale;
        ssq = @mulAdd(T, v, v, ssq);
    }
    return scale * @sqrt(ssq);
}

fn nrm2Stride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    const R = Real(T);
    const lane_count = lanes(R);
    const packed_complexes = lane_count / 2;
    const V = @Vector(lane_count, R);
    const real_x = asConstRealPtr(T, x);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + 4 * packed_complexes <= n) : (i += 4 * packed_complexes) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + k * packed_complexes)));
        }
    }
    while (i + packed_complexes <= n) : (i += packed_complexes) {
        max_v = @max(max_v, @abs(loadComplexStride2RealVec(R, lane_count, real_x, i)));
    }
    var scale: R = @reduce(.Max, max_v);
    while (i < n) : (i += 1) {
        const value = x[2 * i];
        scale = @max(scale, @max(@abs(realPart(T, value)), @abs(imagPart(T, value))));
    }
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const inv_scale_v: V = @splat(1 / scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    i = 0;
    while (i + 4 * packed_complexes <= n) : (i += 4 * packed_complexes) {
        const v0 = loadComplexStride2RealVec(R, lane_count, real_x, i) * inv_scale_v;
        const v1 = loadComplexStride2RealVec(R, lane_count, real_x, i + packed_complexes) * inv_scale_v;
        const v2 = loadComplexStride2RealVec(R, lane_count, real_x, i + 2 * packed_complexes) * inv_scale_v;
        const v3 = loadComplexStride2RealVec(R, lane_count, real_x, i + 3 * packed_complexes) * inv_scale_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + packed_complexes <= n) : (i += packed_complexes) {
        const v = loadComplexStride2RealVec(R, lane_count, real_x, i) * inv_scale_v;
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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

fn iamaxUnitComplex(comptime T: type, n: usize, x: [*]const T) BlasInt {
    if (n == 0) return 0;
    const R = Real(T);
    const lane_count = lanes(R);
    const unroll_count = unroll(R);
    const M = @Vector(lane_count / 2, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    var best: usize = 0;
    var best_abs = @abs(real_x[0]) + @abs(real_x[1]);
    var i: usize = 0;

    while (i + unroll_count <= real_n) : (i += unroll_count) {
        var max_v: M = @splat(0);
        inline for (0..4) |k| {
            const ax = @abs(loadVec(R, lane_count, real_x, i + k * lane_count));
            max_v = @max(max_v, pairAbsSums(R, lane_count, ax));
        }
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + unroll_count;
            var j = i;
            while (j < end) : (j += 2) {
                const ax = @abs(real_x[j]) + @abs(real_x[j + 1]);
                if (ax > best_abs) {
                    best_abs = ax;
                    best = j / 2;
                }
            }
        }
    }
    while (i + lane_count <= real_n) : (i += lane_count) {
        const ax = @abs(loadVec(R, lane_count, real_x, i));
        const max_v = pairAbsSums(R, lane_count, ax);
        if (@reduce(.Max, max_v) > best_abs) {
            const end = i + lane_count;
            var j = i;
            while (j < end) : (j += 2) {
                const scalar_abs = @abs(real_x[j]) + @abs(real_x[j + 1]);
                if (scalar_abs > best_abs) {
                    best_abs = scalar_abs;
                    best = j / 2;
                }
            }
        }
    }
    inline for (.{ lane_count / 2, lane_count / 4, lane_count / 8 }) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                const ax = @abs(loadVec(R, tail_lanes, real_x, i));
                const max_v = pairAbsSums(R, tail_lanes, ax);
                if (@reduce(.Max, max_v) > best_abs) {
                    const end = i + tail_lanes;
                    var j = i;
                    while (j < end) : (j += 2) {
                        const scalar_abs = @abs(real_x[j]) + @abs(real_x[j + 1]);
                        if (scalar_abs > best_abs) {
                            best_abs = scalar_abs;
                            best = j / 2;
                        }
                    }
                }
            }
        }
    }
    while (i < real_n) : (i += 2) {
        const ax = @abs(real_x[i]) + @abs(real_x[i + 1]);
        if (ax > best_abs) {
            best_abs = ax;
            best = i / 2;
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

    while (i + 4 * packed_complexes <= n) : (i += 4 * packed_complexes) {
        var max_v: M = @splat(0);
        inline for (0..4) |k| {
            const ax = @abs(loadComplexStride2RealVec(R, lane_count, real_x, i + k * packed_complexes));
            max_v = @max(max_v, pairAbsSums(R, lane_count, ax));
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
        const ax = @abs(loadComplexStride2RealVec(R, lane_count, real_x, i));
        if (@reduce(.Max, pairAbsSums(R, lane_count, ax)) > best_abs) {
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
            storeVec(T, lane_count, y, offset, @mulAdd(V, -xv, s_v, yv * c_v));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, x, i, @mulAdd(V, xv, c_v, yv * s_v));
        storeVec(T, lane_count, y, i, @mulAdd(V, -xv, s_v, yv * c_v));
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
                storeVec(T, tail_lanes, y, i, @mulAdd(TailV, -xv, s_tail, yv * c_tail));
            }
        }
    }
    while (i < n) : (i += 1) {
        const xv = x[i];
        const yv = y[i];
        x[i] = @mulAdd(T, c, xv, s * yv);
        y[i] = @mulAdd(T, -xv, s, c * yv);
    }
}

fn rotStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const c_v: V = @splat(c);
    const s_v: V = @splat(s);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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
    while (i + 2 * packed_complexes <= n) : (i += 2 * packed_complexes) {
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
    while (i + packed_complexes <= n) : (i += packed_complexes) {
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

fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) void {
    if (vector_binary_kernels.rotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22)) return;
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

fn rotmStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const h11_v: V = @splat(h11);
    const h21_v: V = @splat(h21);
    const h12_v: V = @splat(h12);
    const h22_v: V = @splat(h22);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
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

fn parallelCopyTaskCount(n_bytes: usize) usize {
    if (comptime builtin.cpu.arch == .x86_64) {
        const min_bytes_per_task: usize = if (n_bytes < 1024 * 1024) 64 * 1024 else 256 * 1024;
        return @min(core_pool.taskCount(n_bytes, min_bytes_per_task), 32);
    }
    return parallelTaskCount(n_bytes, if (n_bytes == 8 * 1024 * 1024) 2 * 1024 * 1024 else 512 * 1024, 10);
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

const ComplexScalStride2TaskF64 = struct {
    n0: usize,
    n1: usize,
    alpha_re: f64,
    alpha_im: f64,
    x: [*]ComplexF64,
};

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

fn runSwapTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SwapTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    swapUnit(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn runSwapTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSwapTask(f32, raw_tasks, index);
}

fn runSwapTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSwapTask(f64, raw_tasks, index);
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

fn parallelSwapUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    const n_bytes = n * @sizeOf(T);
    if (byteRangesOverlap(@ptrCast(x), @ptrCast(y), n_bytes)) return false;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else if (comptime builtin.cpu.arch == .aarch64) task_count: {
        if (n_bytes < 4 * 1024 * 1024 or n_bytes > 16 * 1024 * 1024) return false;
        const max_task_count: usize = if (n_bytes <= 8 * 1024 * 1024) 4 else 2;
        break :task_count @min(core_pool.taskCount(n, 128 * 1024), max_task_count);
    } else return false;
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
    const runner = if (T == f32) runSwapTaskF32 else runSwapTaskF64;
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
}

fn parallelSwapStride2F64(n: usize, x: [*]f64, y: [*]f64) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), 2);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]SwapTask(f64) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
        };
    }
    return core_pool.runLowLatency(runSwapStride2TaskF64, @ptrCast(&tasks), task_count);
}

fn parallelSwapStride2C32(n: usize, x: [*]ComplexF32, y: [*]ComplexF32) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), 2);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]SwapTask(ComplexF32) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
        };
    }
    return core_pool.runLowLatency(runSwapStride2TaskC32, @ptrCast(&tasks), task_count);
}

fn runRotTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RotTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    rotUnitReal(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0, task.c, task.s);
}

fn runRotTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runRotTask(f32, raw_tasks, index);
}

fn runRotTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runRotTask(f64, raw_tasks, index);
}

fn parallelRotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    const n_bytes = n * @sizeOf(T);
    if (byteRangesOverlap(@ptrCast(x), @ptrCast(y), n_bytes)) return false;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else if (comptime builtin.cpu.arch == .aarch64) task_count: {
        if (n_bytes < 4 * 1024 * 1024 or n_bytes > 16 * 1024 * 1024) return false;
        const max_task_count: usize = if (n_bytes <= 8 * 1024 * 1024) 4 else 2;
        break :task_count @min(core_pool.taskCount(n, 128 * 1024), max_task_count);
    } else return false;
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RotTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .y = y,
            .c = c,
            .s = s,
        };
    }
    const runner = if (T == f32) runRotTaskF32 else runRotTaskF64;
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
}

fn runRotmTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RotmTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    rotmUnitReal(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0, task.flag, task.h11, task.h21, task.h12, task.h22);
}

fn runRotmTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runRotmTask(f32, raw_tasks, index);
}

fn runRotmTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runRotmTask(f64, raw_tasks, index);
}

fn parallelRotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    const n_bytes = n * @sizeOf(T);
    if (byteRangesOverlap(@ptrCast(x), @ptrCast(y), n_bytes)) return false;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else if (comptime builtin.cpu.arch == .aarch64) task_count: {
        if (n_bytes < 4 * 1024 * 1024 or n_bytes > 8 * 1024 * 1024) return false;
        break :task_count @min(core_pool.taskCount(n, 128 * 1024), 4);
    } else return false;
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
    const runner = if (T == f32) runRotmTaskF32 else runRotmTaskF64;
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
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

fn runComplexScalStride2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexScalStride2TaskF64 = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    complexScalF64Stride2(task.n1 - task.n0, task.alpha_re, task.alpha_im, task.x + 2 * task.n0);
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

fn parallelComplexScalStride2F64(n: usize, alpha_re: f64, alpha_im: f64, x: [*]ComplexF64) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), 2);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]ComplexScalStride2TaskF64 = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha_re = alpha_re,
            .alpha_im = alpha_im,
            .x = x,
        };
    }
    return core_pool.runLowLatency(runComplexScalStride2TaskF64, @ptrCast(&tasks), task_count);
}

const ByteCopyTask = struct {
    n0: usize,
    n1: usize,
    x: [*]const u8,
    y: [*]u8,
    use_fixed: bool,
};

fn runCopyBytesTask(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ByteCopyTask = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    const n_bytes = task.n1 - task.n0;
    if (task.use_fixed and vector_binary_kernels.fixedCopyBytes(n_bytes, task.x + task.n0, task.y + task.n0)) return;
    copyBytes(n_bytes, task.x + task.n0, task.y + task.n0);
}

fn parallelCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n_bytes <= 128 * 1024) return false;
    } else if (n_bytes < 4 * 1024 * 1024 or (n_bytes > 5 * 1024 * 1024 and n_bytes != 8 * 1024 * 1024)) return false;
    if (byteRangesOverlap(x, y, n_bytes)) return false;
    const task_count = parallelCopyTaskCount(n_bytes);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]ByteCopyTask = undefined;
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n_bytes == 256 * 1024 and task_count == 4) {
            const chunk: usize = 64 * 1024;
            const chunk_order = .{ 1, 0, 2, 3 };
            inline for (chunk_order, 0..) |chunk_index, task_index| {
                tasks[task_index] = .{
                    .n0 = chunk_index * chunk,
                    .n1 = (chunk_index + 1) * chunk,
                    .x = x,
                    .y = y,
                    .use_fixed = false,
                };
            }
            return runLevel1Tasks(runCopyBytesTask, @ptrCast(&tasks), task_count);
        }
    }
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
            .use_fixed = builtin.cpu.arch == .aarch64 and n_bytes == 8 * 1024 * 1024,
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

fn parallelAxpyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const max_task_count: usize = if (T == f64) 3 else 2;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), max_task_count);
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
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
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

fn parallelComplexAxpyStride2C32(n: usize, alpha: ComplexF32, x: [*]const ComplexF32, y: [*]ComplexF32) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), 3);
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
    return core_pool.runLowLatency(runComplexAxpyStride2TaskC32, @ptrCast(&tasks), task_count);
}

fn parallelComplexAxpyStride2C64(n: usize, alpha: ComplexF64, x: [*]const ComplexF64, y: [*]ComplexF64) bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    if (n < 512 * 1024) return false;
    const task_count = @min(core_pool.taskCount(n, 256 * 1024), 2);
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
    return core_pool.runLowLatency(runComplexAxpyStride2TaskC64, @ptrCast(&tasks), task_count);
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
    if (byteRangesOverlap(@ptrCast(x), @ptrCast(y), n * @sizeOf(T))) return false;
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else if (comptime builtin.cpu.arch == .aarch64)
        parallelTaskCount(n, 128 * 1024, 4)
    else
        parallelTaskCount(n, 170 * 1024, 6);
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
    if (comptime builtin.cpu.arch == .aarch64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
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

const DotF32AccF64Task = struct {
    n0: usize,
    n1: usize,
    x: [*]const f32,
    y: [*]const f32,
    out: *f64,
};

fn runDotF32AccF64Task(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const DotF32AccF64Task = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = dotF32AccF64Unit(task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn parallelDotF32AccF64Unit(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
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
    if (!runLevel1Tasks(runDotF32AccF64Task, @ptrCast(&tasks), task_count)) return null;

    var result: f64 = 0;
    for (partial[0..task_count]) |v| result += v;
    return result;
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
        if (comptime T == ComplexF32) {
            if (incx_ == 2) return complexScalF32Stride2(n, alpha.re, alpha.im, x);
        }
        if (comptime T == ComplexF64) {
            if (incx_ == 2) {
                if (parallelComplexScalStride2F64(n, alpha.re, alpha.im, x)) return;
                return complexScalF64Stride2(n, alpha.re, alpha.im, x);
            }
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
        if (incx_ == 1 and incy_ == 1) {
            const R = Real(T);
            const real_n = 2 * n;
            const real_x = asRealPtr(T, x);
            const real_y = asRealPtr(T, y);
            const n_bytes = real_n * @sizeOf(R);
            if (comptime builtin.cpu.arch == .aarch64) {
                if (!byteRangesOverlap(@ptrCast(real_x), @ptrCast(real_y), n_bytes)) {
                    // Keep one streaming-mode lifetime around the full swap.
                    if (vector_binary_kernels.swapUnitRealStreaming(R, real_n, real_x, real_y)) return;
                }
            }
            if (parallelSwapUnitReal(R, real_n, real_x, real_y)) return;
            return swapUnit(R, real_n, real_x, real_y);
        }
        if (comptime T == ComplexF32) {
            if (incx_ == 2 and incy_ == 2) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    if (parallelSwapStride2C32(n, x, y)) return;
                    return swapComplexF32Stride2(n, x, y);
                }
            }
        }
    }
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            const n_bytes = n * @sizeOf(T);
            if (comptime builtin.cpu.arch == .aarch64) {
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), n_bytes)) {
                    if (vector_binary_kernels.swapUnitRealStreaming(T, n, x, y)) return;
                }
            }
            if (parallelSwapUnitReal(T, n, x, y)) return;
            return swapUnit(T, n, x, y);
        }
        if (incx_ == 2 and incy_ == 2) {
            const span_bytes = (2 * n - 1) * @sizeOf(T);
            if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                if (comptime T == f64) {
                    if (parallelSwapStride2F64(n, x, y)) return;
                }
                return swapStride2Real(T, n, x, y);
            }
        }
    }
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
        if (incx_ == 2 and incy_ == 2) {
            const span_bytes = (2 * n - 1) * @sizeOf(T);
            if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                if (parallelAxpyStride2Real(T, n, alpha, x, y)) return;
                return axpyStride2Real(T, n, alpha, x, y);
            }
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
        if (comptime T == ComplexF32) {
            if (incx_ == 2 and incy_ == 2) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    if (parallelComplexAxpyStride2C32(n, alpha, x, y)) return;
                    return complexAxpyF32Stride2(n, alpha, x, y);
                }
            }
        }
        if (comptime T == ComplexF64) {
            if (incx_ == 2 and incy_ == 2) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    if (parallelComplexAxpyStride2C64(n, alpha, x, y)) return;
                    return complexAxpyF64Stride2(n, alpha, x, y);
                }
            }
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
            // The large Apple f32 path owns SM/ZA once for the full vector.
            // Do not split it into helpers that each pay a state transition.
            if (comptime builtin.cpu.arch == .aarch64 and T == f32) {
                if (vector_binary_kernels.axpbyUnitReal(T, n, alpha, x, beta, y)) return;
            }
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
        if (comptime T == ComplexF32) {
            if (incx_ == 2 and incy_ == 2) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    return complexAxpbyF32Stride2(n, alpha.re, alpha.im, x, beta.re, beta.im, y);
                }
            }
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
        if (incx_ == 2 and incy_ == 2) return dotStride2Real(T, n, x, y);
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

pub fn dotF32AccF64(n_: BlasInt, x: [*]const f32, incx_: BlasInt, y: [*]const f32, incy_: BlasInt) f64 {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return 0;
    const n = toUsize(n_);
    if (incx_ == 1 and incy_ == 1) return parallelDotF32AccF64Unit(n, x, y) orelse dotF32AccF64Unit(n, x, y);
    if (incx_ == 2 and incy_ == 2) return dotF32AccF64Stride2(n, x, y);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    var sum: f64 = 0;
    for (0..n) |i| {
        sum = @mulAdd(
            f64,
            @as(f64, x[ix(sx, i, incx_)]),
            @as(f64, y[ix(sy, i, incy_)]),
            sum,
        );
    }
    return sum;
}

pub fn asum(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) return parallelAsumUnitReal(T, n, x) orelse asumUnitReal(T, n, x);
        if (incx_ == 2) return asumStride2Real(T, n, x);
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
        if (incx_ == 2) {
            if (nrm2Stride2Real(T, n, x)) |result| return result;
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1) {
            const R = Real(T);
            const real_n = 2 * n;
            const real_x = asConstRealPtr(T, x);
            if (parallelNrm2UnitReal(R, real_n, real_x)) |result| return result;
            if (nrm2UnitReal(R, real_n, real_x)) |result| return result;
        }
        if (incx_ == 2) {
            if (nrm2Stride2Complex(T, n, x)) |result| return result;
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

fn IamaxTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        x: [*]const T,
        out: *BlasInt,
    };
}

fn runIamaxTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const IamaxTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = if (comptime isReal(T))
        iamaxUnitReal(T, task.n1 - task.n0, task.x + task.n0)
    else
        iamaxUnitComplex(T, task.n1 - task.n0, task.x + task.n0);
}

fn runIamaxTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxTask(f32, raw_tasks, index);
}

fn runIamaxTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxTask(f64, raw_tasks, index);
}

fn runIamaxTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxTask(ComplexF32, raw_tasks, index);
}

fn runIamaxTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runIamaxTask(ComplexF64, raw_tasks, index);
}

fn parallelIamaxUnit(comptime T: type, n: usize, x: [*]const T) ?BlasInt {
    const task_count = if (comptime builtin.cpu.arch == .x86_64)
        parallelTaskCount(n, 32 * 1024, 32)
    else if (comptime builtin.cpu.arch == .aarch64 and isComplex(T)) task_count: {
        if (n < 256 * 1024) return null;
        const min_items_per_task: usize = if (T == ComplexF32) 64 * 1024 else 128 * 1024;
        break :task_count @min(core_pool.taskCount(n, min_items_per_task), 4);
    } else return null;
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]BlasInt = undefined;
    var tasks: [core_pool.max_tasks]IamaxTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0 = task_index * n / task_count;
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = n0,
            .n1 = (task_index + 1) * n / task_count,
            .x = x,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32)
        runIamaxTaskF32
    else if (T == f64)
        runIamaxTaskF64
    else if (T == ComplexF32)
        runIamaxTaskC32
    else
        runIamaxTaskC64;
    if (!runLevel1Tasks(runner, @ptrCast(&tasks), task_count)) return null;

    var best: usize = 0;
    var best_abs = abs1(T, x[0]);
    for (tasks[0..task_count], partial[0..task_count]) |task, local_result| {
        if (local_result <= 0) continue;
        const global_index = task.n0 + @as(usize, @intCast(local_result - 1));
        const candidate_abs = abs1(T, x[global_index]);
        if (candidate_abs > best_abs) {
            best_abs = candidate_abs;
            best = global_index;
        }
    }
    return @intCast(best + 1);
}

pub fn iamax(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) BlasInt {
    if (n_ < 1 or incx_ <= 0) return 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) return parallelIamaxUnit(T, n, x) orelse iamaxUnitReal(T, n, x);
    } else if (comptime isComplex(T)) {
        if (incx_ == 1) return parallelIamaxUnit(T, n, x) orelse iamaxUnitComplex(T, n, x);
        if (incx_ == 2) return iamaxStride2Complex(T, n, x);
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
        if (incx_ == 1 and incy_ == 1) {
            // Large SME transforms own SM/ZA once for the complete vector.
            if (comptime builtin.cpu.arch == .aarch64 and T == f32) {
                if (vector_binary_kernels.rotUnitRealStreaming(T, n, x, y, c, s)) return;
            }
            if (parallelRotUnitReal(T, n, x, y, c, s)) return;
            return rotUnitReal(T, n, x, y, c, s);
        }
        if (incx_ == 2 and incy_ == 2) {
            const span_bytes = (2 * n - 1) * @sizeOf(T);
            if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                return rotStride2Real(T, n, x, y, c, s);
            }
        }
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1 and imagPart(T, s) == 0) {
            const R = Real(T);
            const real_n = 2 * n;
            const real_x = asRealPtr(T, x);
            const real_y = asRealPtr(T, y);
            if (comptime builtin.cpu.arch == .aarch64 and R == f32) {
                if (vector_binary_kernels.rotUnitRealStreaming(R, real_n, real_x, real_y, c, realPart(T, s))) return;
            }
            if (parallelRotUnitReal(R, real_n, real_x, real_y, c, realPart(T, s))) return;
            return rotUnitReal(R, real_n, real_x, real_y, c, realPart(T, s));
        }
        if (comptime T == ComplexF32) {
            if (incx_ == 2 and incy_ == 2 and imagPart(T, s) == 0) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    return rotComplexF32Stride2(n, x, y, c, realPart(T, s));
                }
            }
        }
        if (comptime T == ComplexF64) {
            if (incx_ == 2 and incy_ == 2 and imagPart(T, s) == 0) {
                const span_bytes = (2 * n - 1) * @sizeOf(T);
                if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                    return rotComplexF64Stride2(n, x, y, c, realPart(T, s));
                }
            }
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
    const h11 = if (flag < 0 or flag > 0) param[1] else 0;
    const h21 = if (flag <= 0) param[2] else 0;
    const h12 = if (flag <= 0) param[3] else 0;
    const h22 = if (flag < 0 or flag > 0) param[4] else 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            // As with ROT, keep one streaming-mode lifetime around the whole
            // transform instead of paying it once per parallel helper.
            if (comptime builtin.cpu.arch == .aarch64 and T == f32) {
                if (vector_binary_kernels.rotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22)) return;
            }
            if (parallelRotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22)) return;
            return rotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22);
        }
        if (incx_ == 2 and incy_ == 2) {
            const span_bytes = (2 * n - 1) * @sizeOf(T);
            if (!byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) {
                return rotmStride2Real(T, n, x, y, flag, h11, h21, h12, h22);
            }
        }
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
