// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Parameterized portable GEMM micro-kernels.
//!
//! These are the base no-pack kernels used for tiny, odd, and fallback shapes.
//! Architecture-specific modules provide stronger packed or matrix-extension
//! candidates, but they share the same task contract and planner descriptors.

const epilogue = @import("../epilogue.zig");
const gemm_task = @import("../task.zig");

const matIndex = gemm_task.matIndex;

pub const Config = struct {
    vector_lanes: comptime_int,
    tile_n: comptime_int,
    k_unroll: comptime_int = 1,
};

fn WideLanes(comptime T: type) comptime_int {
    if (T == f32) return 8;
    if (T == f64) return 4;
    @compileError("wide GEMM vector lanes support f32 and f64");
}

inline fn loadVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]const T, ld: gemm_task.BlasInt, row: usize, col: usize) @Vector(lanes, T) {
    var out: @Vector(lanes, T) = undefined;
    inline for (0..lanes) |lane| {
        out[lane] = ptr[matIndex(ld, row + lane, col)];
    }
    return out;
}

inline fn storeVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]T, ld: gemm_task.BlasInt, row: usize, col: usize, value: @Vector(lanes, T)) void {
    inline for (0..lanes) |lane| {
        ptr[matIndex(ld, row + lane, col)] = value[lane];
    }
}

inline fn oldVec(comptime T: type, comptime lanes: comptime_int, task: gemm_task.Task(T), row: usize, col: usize) @Vector(lanes, T) {
    if (task.beta == 0) return @splat(0);
    return loadVec(T, lanes, task.c, task.ldc, row, col);
}

inline fn applyScalar(comptime T: type, task: gemm_task.Task(T), acc: T, row: usize, col: usize) T {
    return epilogue.applyTaskScalar(T, task, acc, row, col);
}

inline fn writeScalarEpilogue(comptime T: type, task: gemm_task.Task(T), comptime direct_store: bool, row: usize, col: usize, acc: T) void {
    task.c[matIndex(task.ldc, row, col)] = if (comptime direct_store) acc else applyScalar(T, task, acc, row, col);
}

fn kernelVec(comptime T: type, comptime cfg: Config, comptime direct_store: bool, task: gemm_task.Task(T), row: usize, col: usize) void {
    const V = @Vector(cfg.vector_lanes, T);
    const zero: V = @splat(0);
    var acc: [cfg.tile_n]V = [_]V{zero} ** cfg.tile_n;

    var p: usize = 0;
    while (p + cfg.k_unroll <= task.k) : (p += cfg.k_unroll) {
        inline for (0..cfg.k_unroll) |u| {
            const av = loadVec(T, cfg.vector_lanes, task.a, task.lda, row, p + u);
            inline for (0..cfg.tile_n) |n_lane| {
                const bv: V = @splat(task.b[matIndex(task.ldb, p + u, col + n_lane)]);
                acc[n_lane] = @mulAdd(V, av, bv, acc[n_lane]);
            }
        }
    }
    while (p < task.k) : (p += 1) {
        const av = loadVec(T, cfg.vector_lanes, task.a, task.lda, row, p);
        inline for (0..cfg.tile_n) |n_lane| {
            const bv: V = @splat(task.b[matIndex(task.ldb, p, col + n_lane)]);
            acc[n_lane] = @mulAdd(V, av, bv, acc[n_lane]);
        }
    }

    inline for (0..cfg.tile_n) |n_lane| {
        const out = if (comptime direct_store)
            acc[n_lane]
        else
            epilogue.applyVector(T, cfg.vector_lanes, task.alpha, task.beta, acc[n_lane], oldVec(T, cfg.vector_lanes, task, row, col + n_lane));
        storeVec(T, cfg.vector_lanes, task.c, task.ldc, row, col + n_lane, out);
    }
}

fn kernelScalarRows(comptime T: type, comptime cfg: Config, comptime direct_store: bool, task: gemm_task.Task(T), row_start: usize, col: usize) void {
    var row = row_start;
    while (row < task.m) : (row += 1) {
        var acc: [cfg.tile_n]T = [_]T{0} ** cfg.tile_n;
        var p: usize = 0;
        while (p + cfg.k_unroll <= task.k) : (p += cfg.k_unroll) {
            inline for (0..cfg.k_unroll) |u| {
                const av = task.a[matIndex(task.lda, row, p + u)];
                inline for (0..cfg.tile_n) |n_lane| {
                    acc[n_lane] = @mulAdd(T, av, task.b[matIndex(task.ldb, p + u, col + n_lane)], acc[n_lane]);
                }
            }
        }
        while (p < task.k) : (p += 1) {
            const av = task.a[matIndex(task.lda, row, p)];
            inline for (0..cfg.tile_n) |n_lane| {
                acc[n_lane] = @mulAdd(T, av, task.b[matIndex(task.ldb, p, col + n_lane)], acc[n_lane]);
            }
        }
        inline for (0..cfg.tile_n) |n_lane| {
            writeScalarEpilogue(T, task, direct_store, row, col + n_lane, acc[n_lane]);
        }
    }
}

fn kernelScalarTailColumns(comptime T: type, comptime direct_store: bool, task: gemm_task.Task(T), col: usize) void {
    var j = col;
    while (j < task.n1) : (j += 1) {
        var row: usize = 0;
        while (row < task.m) : (row += 1) {
            var acc: T = 0;
            for (0..task.k) |p| {
                acc = @mulAdd(T, task.a[matIndex(task.lda, row, p)], task.b[matIndex(task.ldb, p, j)], acc);
            }
            writeScalarEpilogue(T, task, direct_store, row, j, acc);
        }
    }
}

fn dotContiguous(comptime T: type, comptime lanes: comptime_int, a: [*]const T, b: [*]const T, len: usize) T {
    const V = @Vector(lanes, T);
    var acc: V = @splat(0);
    var p: usize = 0;
    while (p + lanes <= len) : (p += lanes) {
        const av = @as(*align(1) const V, @ptrCast(a + p)).*;
        const bv = @as(*align(1) const V, @ptrCast(b + p)).*;
        acc = @mulAdd(V, av, bv, acc);
    }

    var sum: T = @reduce(.Add, acc);
    while (p < len) : (p += 1) {
        sum = @mulAdd(T, a[p], b[p], sum);
    }
    return sum;
}

fn dotStridedA(comptime T: type, a: [*]const T, lda: gemm_task.BlasInt, b: [*]const T, len: usize) T {
    var acc: T = 0;
    var p: usize = 0;
    while (p < len) : (p += 1) {
        acc = @mulAdd(T, a[matIndex(lda, 0, p)], b[p], acc);
    }
    return acc;
}

fn rowVectorColumnsImpl(comptime T: type, comptime direct_store: bool, task: gemm_task.Task(T)) void {
    const lanes = WideLanes(T);
    const a_contiguous = gemm_task.toUsize(task.lda) == 1;
    var j = task.n0;
    while (j < task.n1) : (j += 1) {
        const b_col = task.b + matIndex(task.ldb, 0, j);
        const acc = if (a_contiguous)
            dotContiguous(T, lanes, task.a, b_col, task.k)
        else
            dotStridedA(T, task.a, task.lda, b_col, task.k);
        writeScalarEpilogue(T, task, direct_store, 0, j, acc);
    }
}

pub fn rowVectorColumns(comptime T: type, task: gemm_task.Task(T)) void {
    if (task.alpha == 1 and task.beta == 0) return rowVectorColumnsImpl(T, true, task);
    return rowVectorColumnsImpl(T, false, task);
}

fn noTransRealImpl(comptime T: type, comptime cfg: Config, comptime direct_store: bool, task: gemm_task.Task(T)) void {
    var col = task.n0;
    while (col + cfg.tile_n <= task.n1) : (col += cfg.tile_n) {
        var row: usize = 0;
        while (row + cfg.vector_lanes <= task.m) : (row += cfg.vector_lanes) {
            kernelVec(T, cfg, direct_store, task, row, col);
        }
        kernelScalarRows(T, cfg, direct_store, task, row, col);
    }
    kernelScalarTailColumns(T, direct_store, task, col);
}

pub fn noTransReal(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T)) void {
    if (task.alpha == 1 and task.beta == 0) return noTransRealImpl(T, cfg, true, task);
    return noTransRealImpl(T, cfg, false, task);
}

pub fn f32x4x1(task: gemm_task.Task(f32)) void {
    noTransReal(f32, .{ .vector_lanes = 4, .tile_n = 1, .k_unroll = 4 }, task);
}

pub fn f32x8x1(task: gemm_task.Task(f32)) void {
    noTransReal(f32, .{ .vector_lanes = 8, .tile_n = 1, .k_unroll = 4 }, task);
}

pub fn f32x4x4(task: gemm_task.Task(f32)) void {
    noTransReal(f32, .{ .vector_lanes = 4, .tile_n = 4, .k_unroll = 4 }, task);
}

pub fn f32x4x8(task: gemm_task.Task(f32)) void {
    noTransReal(f32, .{ .vector_lanes = 4, .tile_n = 8, .k_unroll = 4 }, task);
}

pub fn f64x2x1(task: gemm_task.Task(f64)) void {
    noTransReal(f64, .{ .vector_lanes = 2, .tile_n = 1, .k_unroll = 4 }, task);
}

pub fn f64x4x1(task: gemm_task.Task(f64)) void {
    noTransReal(f64, .{ .vector_lanes = 4, .tile_n = 1, .k_unroll = 4 }, task);
}

pub fn f64x2x4(task: gemm_task.Task(f64)) void {
    noTransReal(f64, .{ .vector_lanes = 2, .tile_n = 4, .k_unroll = 4 }, task);
}

pub fn f64x2x6(task: gemm_task.Task(f64)) void {
    noTransReal(f64, .{ .vector_lanes = 2, .tile_n = 6, .k_unroll = 4 }, task);
}
