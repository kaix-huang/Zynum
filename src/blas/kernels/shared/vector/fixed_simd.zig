// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared fixed-width SIMD BLAS Level 1 microkernels.
//!
//! Architecture files select a `Config` from target features; this file owns the
//! reusable lane-width, unroll, tail, prologue, and epilogue mechanics.

const std = @import("std");

const types = @import("../../../types.zig");

pub const Config = struct {
    lane_count: comptime_int,
    unroll_vectors: comptime_int = 4,
    copy_lane_count: comptime_int = 64,
    min_len: usize = 0,
};

const RealAffineMode = enum { scal, axpy, axpby };
const ComplexAffineMode = enum { scal, axpy, axpby };

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn isComplex(comptime T: type) bool {
    return T == types.ComplexF32 or T == types.ComplexF64;
}

fn Real(comptime T: type) type {
    if (T == f32 or T == types.ComplexF32) return f32;
    if (T == f64 or T == types.ComplexF64) return f64;
    @compileError("fixed SIMD vector kernels support f32, f64, ComplexF32, and ComplexF64");
}

inline fn realPart(comptime T: type, x: T) Real(T) {
    if (T == f32 or T == f64) return x;
    return x.re;
}

inline fn imagPart(comptime T: type, x: T) Real(T) {
    if (T == f32 or T == f64) return 0;
    return x.im;
}

fn checkRealConfig(comptime T: type, comptime cfg: Config) void {
    if (!isReal(T)) @compileError("fixed SIMD real vector kernels support f32 and f64");
    if (cfg.lane_count == 0) @compileError("fixed SIMD lane_count must be nonzero");
    if (cfg.unroll_vectors == 0) @compileError("fixed SIMD unroll_vectors must be nonzero");
}

fn checkComplexConfig(comptime T: type, comptime cfg: Config) void {
    if (!isComplex(T)) @compileError("fixed SIMD complex vector kernels support ComplexF32 and ComplexF64");
    if (cfg.lane_count == 0 or cfg.lane_count % 2 != 0) {
        @compileError("fixed SIMD complex vector kernels need an even real lane_count");
    }
    if (cfg.unroll_vectors == 0) @compileError("fixed SIMD unroll_vectors must be nonzero");
}

fn vectorThreshold(comptime cfg: Config) usize {
    return @max(cfg.min_len, @as(usize, cfg.lane_count));
}

fn unrollCount(comptime cfg: Config) comptime_int {
    return cfg.lane_count * cfg.unroll_vectors;
}

fn tailLaneCounts(comptime lanes: comptime_int) [3]comptime_int {
    return .{ lanes / 2, lanes / 4, lanes / 8 };
}

inline fn loadVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]const T, index: usize) @Vector(lanes, T) {
    return @as(*align(1) const @Vector(lanes, T), @ptrCast(ptr + index)).*;
}

inline fn storeVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]T, index: usize, value: @Vector(lanes, T)) void {
    @as(*align(1) @Vector(lanes, T), @ptrCast(ptr + index)).* = value;
}

inline fn asRealPtr(comptime T: type, ptr: [*]T) [*]Real(T) {
    return @ptrCast(ptr);
}

inline fn asConstRealPtr(comptime T: type, ptr: [*]const T) [*]const Real(T) {
    return @ptrCast(ptr);
}

fn pairSwapMask(comptime lanes: comptime_int) @Vector(lanes, i32) {
    comptime var values: [lanes]i32 = undefined;
    inline for (0..lanes) |i| {
        values[i] = if (i % 2 == 0) @intCast(i + 1) else @intCast(i - 1);
    }
    return values;
}

fn pairSignVector(comptime T: type, comptime lanes: comptime_int, im: T) @Vector(lanes, T) {
    comptime var signs: [lanes]T = undefined;
    inline for (0..lanes) |i| {
        signs[i] = if (i % 2 == 0) -1 else 1;
    }
    return @as(@Vector(lanes, T), signs) * @as(@Vector(lanes, T), @splat(im));
}

fn pairPatternVector(comptime T: type, comptime lanes: comptime_int, comptime even: T, comptime odd: T) @Vector(lanes, T) {
    comptime var values: [lanes]T = undefined;
    inline for (0..lanes) |i| {
        values[i] = if (i % 2 == 0) even else odd;
    }
    return values;
}

inline fn complexScaleVec(
    comptime R: type,
    comptime lanes: comptime_int,
    value: @Vector(lanes, R),
    re_v: @Vector(lanes, R),
    im_sign_v: @Vector(lanes, R),
    comptime swap_mask: @Vector(lanes, i32),
) @Vector(lanes, R) {
    const V = @Vector(lanes, R);
    return @mulAdd(V, value, re_v, @shuffle(R, value, undefined, swap_mask) * im_sign_v);
}

inline fn complexZero(comptime T: type) T {
    return .{ .re = 0, .im = 0 };
}

inline fn complexAffineVec(
    comptime R: type,
    comptime lanes: comptime_int,
    comptime mode: ComplexAffineMode,
    xv: @Vector(lanes, R),
    yv: @Vector(lanes, R),
    alpha_re_v: @Vector(lanes, R),
    alpha_im_sign_v: @Vector(lanes, R),
    beta_re_v: @Vector(lanes, R),
    beta_im_sign_v: @Vector(lanes, R),
    comptime swap_mask: @Vector(lanes, i32),
) @Vector(lanes, R) {
    const x_term = complexScaleVec(R, lanes, xv, alpha_re_v, alpha_im_sign_v, swap_mask);
    return switch (mode) {
        .scal => x_term,
        .axpy => yv + x_term,
        .axpby => x_term + complexScaleVec(R, lanes, yv, beta_re_v, beta_im_sign_v, swap_mask),
    };
}

inline fn complexAffineScalar(
    comptime T: type,
    comptime mode: ComplexAffineMode,
    xr: Real(T),
    xi: Real(T),
    yr: Real(T),
    yi: Real(T),
    alpha: T,
    beta: T,
) T {
    const ar = realPart(T, alpha);
    const ai = imagPart(T, alpha);
    const x_re = ar * xr - ai * xi;
    const x_im = ar * xi + ai * xr;
    return switch (mode) {
        .scal => .{ .re = x_re, .im = x_im },
        .axpy => .{ .re = yr + x_re, .im = yi + x_im },
        .axpby => .{
            .re = x_re + realPart(T, beta) * yr - imagPart(T, beta) * yi,
            .im = x_im + realPart(T, beta) * yi + imagPart(T, beta) * yr,
        },
    };
}

inline fn updateIamaxRange(comptime T: type, x: [*]const T, start: usize, end: usize, best: *usize, best_abs: *T) void {
    var j = start;
    while (j < end) : (j += 1) {
        const ax = @abs(x[j]);
        if (ax > best_abs.*) {
            best_abs.* = ax;
            best.* = j;
        }
    }
}

inline fn realAffineVec(
    comptime T: type,
    comptime lanes: comptime_int,
    comptime mode: RealAffineMode,
    xv: @Vector(lanes, T),
    yv: @Vector(lanes, T),
    alpha_v: @Vector(lanes, T),
    beta_v: @Vector(lanes, T),
) @Vector(lanes, T) {
    const V = @Vector(lanes, T);
    return switch (mode) {
        .scal => xv * alpha_v,
        .axpy => @mulAdd(V, xv, alpha_v, yv),
        .axpby => @mulAdd(V, xv, alpha_v, yv * beta_v),
    };
}

inline fn realAffineScalar(comptime T: type, comptime mode: RealAffineMode, xv: T, yv: T, alpha: T, beta: T) T {
    return switch (mode) {
        .scal => xv * alpha,
        .axpy => @mulAdd(T, alpha, xv, yv),
        .axpby => @mulAdd(T, alpha, xv, beta * yv),
    };
}

fn realAffineUnit(
    comptime T: type,
    comptime cfg: Config,
    comptime mode: RealAffineMode,
    n: usize,
    alpha: T,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    if (comptime !isReal(T)) return false;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return false;

    const V = @Vector(cfg.lane_count, T);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            const xv = loadVec(T, cfg.lane_count, x, offset);
            const yv: V = if (comptime mode == .scal) @splat(0) else loadVec(T, cfg.lane_count, y, offset);
            storeVec(T, cfg.lane_count, y, offset, realAffineVec(T, cfg.lane_count, mode, xv, yv, alpha_v, beta_v));
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        const xv = loadVec(T, cfg.lane_count, x, i);
        const yv: V = if (comptime mode == .scal) @splat(0) else loadVec(T, cfg.lane_count, y, i);
        storeVec(T, cfg.lane_count, y, i, realAffineVec(T, cfg.lane_count, mode, xv, yv, alpha_v, beta_v));
    }
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const alpha_tail: TailV = @splat(alpha);
            const beta_tail: TailV = @splat(beta);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const xv = loadVec(T, tail_lanes, x, i);
                const yv: TailV = if (comptime mode == .scal) @splat(0) else loadVec(T, tail_lanes, y, i);
                storeVec(T, tail_lanes, y, i, realAffineVec(T, tail_lanes, mode, xv, yv, alpha_tail, beta_tail));
            }
        }
    }
    while (i < n) : (i += 1) {
        const yv = if (comptime mode == .scal) @as(T, 0) else y[i];
        y[i] = realAffineScalar(T, mode, x[i], yv, alpha, beta);
    }
    return true;
}

inline fn rotVecBlock(
    comptime T: type,
    comptime lanes: comptime_int,
    x: [*]T,
    y: [*]T,
    index: usize,
    c_v: @Vector(lanes, T),
    s_v: @Vector(lanes, T),
) void {
    const V = @Vector(lanes, T);
    const xv = loadVec(T, lanes, x, index);
    const yv = loadVec(T, lanes, y, index);
    storeVec(T, lanes, x, index, @mulAdd(V, xv, c_v, yv * s_v));
    storeVec(T, lanes, y, index, @mulAdd(V, -xv, s_v, yv * c_v));
}

pub fn copyBytes(comptime cfg: Config, n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (n_bytes == 0) return true;
    if (n_bytes < @max(cfg.min_len, @as(usize, cfg.copy_lane_count))) return false;
    if (cfg.copy_lane_count == 0) @compileError("copy_lane_count must be nonzero");

    var i: usize = 0;
    while (i + cfg.copy_lane_count * cfg.unroll_vectors <= n_bytes) : (i += cfg.copy_lane_count * cfg.unroll_vectors) {
        inline for (0..cfg.unroll_vectors) |k| {
            const offset = i + k * cfg.copy_lane_count;
            storeVec(u8, cfg.copy_lane_count, y, offset, loadVec(u8, cfg.copy_lane_count, x, offset));
        }
    }
    while (i + cfg.copy_lane_count <= n_bytes) : (i += cfg.copy_lane_count) {
        storeVec(u8, cfg.copy_lane_count, y, i, loadVec(u8, cfg.copy_lane_count, x, i));
    }
    inline for (tailLaneCounts(cfg.copy_lane_count)) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            while (i + tail_lanes <= n_bytes) : (i += tail_lanes) {
                storeVec(u8, tail_lanes, y, i, loadVec(u8, tail_lanes, x, i));
            }
        }
    }
    while (i < n_bytes) : (i += 1) y[i] = x[i];
    return true;
}

pub fn copyUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T, y: [*]T) bool {
    if (comptime !isReal(T)) return false;
    comptime checkRealConfig(T, cfg);
    return copyBytes(cfg, n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn swapUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime !isReal(T)) return false;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return false;

    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            const xv = loadVec(T, cfg.lane_count, x, offset);
            const yv = loadVec(T, cfg.lane_count, y, offset);
            storeVec(T, cfg.lane_count, x, offset, yv);
            storeVec(T, cfg.lane_count, y, offset, xv);
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        const xv = loadVec(T, cfg.lane_count, x, i);
        const yv = loadVec(T, cfg.lane_count, y, i);
        storeVec(T, cfg.lane_count, x, i, yv);
        storeVec(T, cfg.lane_count, y, i, xv);
    }
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
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
        const tmp = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
    return true;
}

pub fn scalUnitReal(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]T) bool {
    return realAffineUnit(T, cfg, .scal, n, alpha, x, 0, x);
}

pub fn axpyUnitReal(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    return realAffineUnit(T, cfg, .axpy, n, alpha, x, 0, y);
}

pub fn axpbyUnitReal(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    return realAffineUnit(T, cfg, .axpby, n, alpha, x, beta, y);
}

pub fn dotUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime !isReal(T)) return null;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return null;

    const V = @Vector(cfg.lane_count, T);
    var accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            accs[k] = @mulAdd(V, loadVec(T, cfg.lane_count, x, offset), loadVec(T, cfg.lane_count, y, offset), accs[k]);
        }
    }
    var acc: V = @splat(0);
    inline for (0..cfg.unroll_vectors) |k| acc += accs[k];
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        acc = @mulAdd(V, loadVec(T, cfg.lane_count, x, i), loadVec(T, cfg.lane_count, y, i), acc);
    }
    var sum: T = @reduce(.Add, acc);
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
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

pub fn asumUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T) ?T {
    if (comptime !isReal(T)) return null;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return null;

    const V = @Vector(cfg.lane_count, T);
    var accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            accs[k] += @abs(loadVec(T, cfg.lane_count, x, i + k * cfg.lane_count));
        }
    }
    var acc: V = @splat(0);
    inline for (0..cfg.unroll_vectors) |k| acc += accs[k];
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        acc += @abs(loadVec(T, cfg.lane_count, x, i));
    }
    var sum: T = @reduce(.Add, acc);
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
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

pub fn nrm2UnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T) ?T {
    if (comptime !isReal(T)) return null;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return null;

    const V = @Vector(cfg.lane_count, T);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            max_v = @max(max_v, @abs(loadVec(T, cfg.lane_count, x, i + k * cfg.lane_count)));
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        max_v = @max(max_v, @abs(loadVec(T, cfg.lane_count, x, i)));
    }
    var scale: T = @reduce(.Max, max_v);
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
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
    var accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    i = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const v = loadVec(T, cfg.lane_count, x, i + k * cfg.lane_count) * inv_scale_v;
            accs[k] = @mulAdd(V, v, v, accs[k]);
        }
    }
    var acc: V = @splat(0);
    inline for (0..cfg.unroll_vectors) |k| acc += accs[k];
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        const v = loadVec(T, cfg.lane_count, x, i) * inv_scale_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: T = @reduce(.Add, acc);
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
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

pub fn nrm2UnitRealFastF32(comptime cfg: Config, n: usize, x: [*]const f32) ?f32 {
    comptime checkRealConfig(f32, cfg);
    if (n < vectorThreshold(cfg)) return null;

    const V = @Vector(cfg.lane_count, f32);
    var max_v: V = @splat(0);
    var accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const ax = @abs(loadVec(f32, cfg.lane_count, x, i + k * cfg.lane_count));
            max_v = @max(max_v, ax);
            accs[k] = @mulAdd(V, ax, ax, accs[k]);
        }
    }
    var acc: V = @splat(0);
    inline for (0..cfg.unroll_vectors) |k| acc += accs[k];
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        const ax = @abs(loadVec(f32, cfg.lane_count, x, i));
        max_v = @max(max_v, ax);
        acc = @mulAdd(V, ax, ax, acc);
    }
    var max_abs = @reduce(.Max, max_v);
    var ssq = @reduce(.Add, acc);
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, f32);
            var tail_max: TailV = @splat(0);
            var tail_acc: TailV = @splat(0);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                const ax = @abs(loadVec(f32, tail_lanes, x, i));
                tail_max = @max(tail_max, ax);
                tail_acc = @mulAdd(TailV, ax, ax, tail_acc);
            }
            max_abs = @max(max_abs, @reduce(.Max, tail_max));
            ssq += @reduce(.Add, tail_acc);
        }
    }
    while (i < n) : (i += 1) {
        const ax = @abs(x[i]);
        max_abs = @max(max_abs, ax);
        ssq = @mulAdd(f32, ax, ax, ssq);
    }
    if (max_abs == 0) return 0;
    if (!std.math.isFinite(max_abs) or !std.math.isFinite(ssq)) return null;

    const safe_limit = @sqrt(std.math.floatMax(f32) / @as(f32, @floatFromInt(n)));
    if (max_abs > safe_limit) return null;
    return @sqrt(ssq);
}

pub fn iamaxUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !isReal(T)) return null;
    comptime checkRealConfig(T, cfg);
    if (n == 0) return 0;
    if (n < vectorThreshold(cfg)) return null;

    var best: usize = 0;
    var best_abs: T = @abs(x[0]);
    var i: usize = 1;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        var max_v: @Vector(cfg.lane_count, T) = @splat(0);
        inline for (0..cfg.unroll_vectors) |k| {
            max_v = @max(max_v, @abs(loadVec(T, cfg.lane_count, x, i + k * cfg.lane_count)));
        }
        if (@reduce(.Max, max_v) > best_abs) {
            updateIamaxRange(T, x, i, i + unrollCount(cfg), &best, &best_abs);
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        const max_v = @abs(loadVec(T, cfg.lane_count, x, i));
        if (@reduce(.Max, max_v) > best_abs) {
            updateIamaxRange(T, x, i, i + cfg.lane_count, &best, &best_abs);
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

pub fn rotUnitReal(comptime T: type, comptime cfg: Config, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime !isReal(T)) return false;
    comptime checkRealConfig(T, cfg);
    if (n < vectorThreshold(cfg)) return false;

    const V = @Vector(cfg.lane_count, T);
    const c_v: V = @splat(c);
    const s_v: V = @splat(s);
    var i: usize = 0;
    while (i + unrollCount(cfg) <= n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            rotVecBlock(T, cfg.lane_count, x, y, i + k * cfg.lane_count, c_v, s_v);
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        rotVecBlock(T, cfg.lane_count, x, y, i, c_v, s_v);
    }
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
        if (comptime tail_lanes > 1) {
            const TailV = @Vector(tail_lanes, T);
            const tail_c_v: TailV = @splat(c);
            const tail_s_v: TailV = @splat(s);
            while (i + tail_lanes <= n) : (i += tail_lanes) {
                rotVecBlock(T, tail_lanes, x, y, i, tail_c_v, tail_s_v);
            }
        }
    }
    while (i < n) : (i += 1) {
        const xv = x[i];
        const yv = y[i];
        x[i] = @mulAdd(T, c, xv, s * yv);
        y[i] = @mulAdd(T, -xv, s, c * yv);
    }
    return true;
}

pub fn scalUnitComplex(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !isComplex(T)) return false;
    return complexAffineUnit(T, cfg, .scal, n, alpha, x, complexZero(T), x);
}

pub fn axpyUnitComplex(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime !isComplex(T)) return false;
    return complexAffineUnit(T, cfg, .axpy, n, alpha, x, complexZero(T), y);
}

pub fn axpbyUnitComplex(comptime T: type, comptime cfg: Config, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime !isComplex(T)) return false;
    return complexAffineUnit(T, cfg, .axpby, n, alpha, x, beta, y);
}

fn complexAffineUnit(
    comptime T: type,
    comptime cfg: Config,
    comptime mode: ComplexAffineMode,
    n: usize,
    alpha: T,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    if (comptime !isComplex(T)) return false;
    comptime checkComplexConfig(T, cfg);
    if (n * 2 < vectorThreshold(cfg)) return false;

    const R = Real(T);
    const V = @Vector(cfg.lane_count, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    const real_y = asRealPtr(T, y);
    const alpha_re_v: V = @splat(realPart(T, alpha));
    const alpha_im_sign_v = pairSignVector(R, cfg.lane_count, imagPart(T, alpha));
    const beta_re_v: V = @splat(realPart(T, beta));
    const beta_im_sign_v = pairSignVector(R, cfg.lane_count, imagPart(T, beta));
    const swap_mask = comptime pairSwapMask(cfg.lane_count);
    var i: usize = 0;
    while (i + unrollCount(cfg) <= real_n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            const xv = loadVec(R, cfg.lane_count, real_x, offset);
            const yv: V = if (comptime mode == .scal) @splat(0) else loadVec(R, cfg.lane_count, real_y, offset);
            storeVec(R, cfg.lane_count, real_y, offset, complexAffineVec(R, cfg.lane_count, mode, xv, yv, alpha_re_v, alpha_im_sign_v, beta_re_v, beta_im_sign_v, swap_mask));
        }
    }
    while (i + cfg.lane_count <= real_n) : (i += cfg.lane_count) {
        const xv = loadVec(R, cfg.lane_count, real_x, i);
        const yv: V = if (comptime mode == .scal) @splat(0) else loadVec(R, cfg.lane_count, real_y, i);
        storeVec(R, cfg.lane_count, real_y, i, complexAffineVec(R, cfg.lane_count, mode, xv, yv, alpha_re_v, alpha_im_sign_v, beta_re_v, beta_im_sign_v, swap_mask));
    }
    inline for (tailLaneCounts(cfg.lane_count)) |tail_lanes| {
        if (comptime tail_lanes > 1 and tail_lanes % 2 == 0) {
            const TailV = @Vector(tail_lanes, R);
            const tail_alpha_re_v: TailV = @splat(realPart(T, alpha));
            const tail_alpha_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, alpha));
            const tail_beta_re_v: TailV = @splat(realPart(T, beta));
            const tail_beta_im_sign_v = pairSignVector(R, tail_lanes, imagPart(T, beta));
            const tail_swap_mask = comptime pairSwapMask(tail_lanes);
            while (i + tail_lanes <= real_n) : (i += tail_lanes) {
                const xv = loadVec(R, tail_lanes, real_x, i);
                const yv: TailV = if (comptime mode == .scal) @splat(0) else loadVec(R, tail_lanes, real_y, i);
                storeVec(R, tail_lanes, real_y, i, complexAffineVec(R, tail_lanes, mode, xv, yv, tail_alpha_re_v, tail_alpha_im_sign_v, tail_beta_re_v, tail_beta_im_sign_v, tail_swap_mask));
            }
        }
    }
    while (i < real_n) : (i += 2) {
        const out = complexAffineScalar(
            T,
            mode,
            real_x[i],
            real_x[i + 1],
            if (comptime mode == .scal) 0 else real_y[i],
            if (comptime mode == .scal) 0 else real_y[i + 1],
            alpha,
            beta,
        );
        real_y[i] = out.re;
        real_y[i + 1] = out.im;
    }
    return true;
}

inline fn complexDotAccumulateVec(
    comptime R: type,
    comptime lanes: comptime_int,
    x: [*]const R,
    y: [*]const R,
    offset: usize,
    comptime swap_mask: @Vector(lanes, i32),
    re_sign: @Vector(lanes, R),
    im_sign: @Vector(lanes, R),
    re_acc: *@Vector(lanes, R),
    im_acc: *@Vector(lanes, R),
) void {
    const V = @Vector(lanes, R);
    const xv = loadVec(R, lanes, x, offset);
    const yv = loadVec(R, lanes, y, offset);
    const y_swap = @shuffle(R, yv, undefined, swap_mask);
    re_acc.* = @mulAdd(V, xv * yv, re_sign, re_acc.*);
    im_acc.* = @mulAdd(V, xv * y_swap, im_sign, im_acc.*);
}

pub fn dotUnitComplex(comptime T: type, comptime cfg: Config, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime !isComplex(T)) return null;
    comptime checkComplexConfig(T, cfg);
    if (n * 2 < vectorThreshold(cfg)) return null;

    const R = Real(T);
    const V = @Vector(cfg.lane_count, R);
    const real_n = 2 * n;
    const real_x = asConstRealPtr(T, x);
    const real_y = asConstRealPtr(T, y);
    const swap_mask = comptime pairSwapMask(cfg.lane_count);
    const re_sign: V = if (conjx) @splat(1) else pairPatternVector(R, cfg.lane_count, 1, -1);
    const im_sign: V = if (conjx) pairPatternVector(R, cfg.lane_count, 1, -1) else @splat(1);

    var re_accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    var im_accs: [cfg.unroll_vectors]V = [_]V{@splat(0)} ** cfg.unroll_vectors;
    var i: usize = 0;
    while (i + unrollCount(cfg) <= real_n) : (i += unrollCount(cfg)) {
        inline for (0..cfg.unroll_vectors) |k| {
            complexDotAccumulateVec(R, cfg.lane_count, real_x, real_y, i + k * cfg.lane_count, swap_mask, re_sign, im_sign, &re_accs[k], &im_accs[k]);
        }
    }
    var re_acc: V = @splat(0);
    var im_acc: V = @splat(0);
    inline for (0..cfg.unroll_vectors) |k| {
        re_acc += re_accs[k];
        im_acc += im_accs[k];
    }
    while (i + cfg.lane_count <= real_n) : (i += cfg.lane_count) {
        complexDotAccumulateVec(R, cfg.lane_count, real_x, real_y, i, swap_mask, re_sign, im_sign, &re_acc, &im_acc);
    }
    var re_sum: R = @reduce(.Add, re_acc);
    var im_sum: R = @reduce(.Add, im_acc);
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
