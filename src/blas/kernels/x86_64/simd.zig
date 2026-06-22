// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const gemm_task = @import("../gemm_task.zig");
const features = @import("features.zig");
const generic = @import("../generic/gemm.zig");

const matIndex = gemm_task.matIndex;

pub const enabled: bool = features.has_sse2;
pub const supports_avx: bool = features.has_avx;
pub const supports_avx2: bool = features.has_avx2;
pub const supports_avx512f: bool = features.has_avx512f;
pub const supports_fma: bool = features.has_fma;

fn lanes(comptime T: type) comptime_int {
    if (T == f32) {
        if (comptime features.has_avx512f) return 16;
        if (comptime features.has_avx) return 8;
        return 4;
    }
    if (T == f64) {
        if (comptime features.has_avx512f) return 8;
        if (comptime features.has_avx) return 4;
        return 2;
    }
    @compileError("x86_64 SIMD GEMM kernels support f32 and f64");
}

fn columnTile(comptime T: type) comptime_int {
    if (T == f32) {
        if (comptime features.has_avx) return 12;
        return 8;
    }
    if (T == f64) {
        if (comptime features.has_avx) return 8;
        return 6;
    }
    @compileError("x86_64 SIMD GEMM kernels support f32 and f64");
}

pub fn preferredColumnBlock(comptime T: type) usize {
    return columnTile(T);
}

inline fn loadVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, lda: gemm_task.BlasInt, row: usize, col: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    const v: *align(1) const V = @ptrCast(ptr + matIndex(lda, row, col));
    return v.*;
}

inline fn storeVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]T, lda: gemm_task.BlasInt, row: usize, col: usize, value: @Vector(lane_count, T)) void {
    const V = @Vector(lane_count, T);
    const out: *align(1) V = @ptrCast(ptr + matIndex(lda, row, col));
    out.* = value;
}

inline fn oldVec(comptime T: type, comptime lane_count: comptime_int, task: gemm_task.Task(T), row: usize, col: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    if (task.beta == 0) return @splat(0);
    const old = loadVec(T, lane_count, task.c, task.ldc, row, col);
    if (task.beta == 1) return old;
    return old * @as(V, @splat(task.beta));
}

fn packBPanel(comptime T: type, comptime tile_n: comptime_int, task: gemm_task.Task(T), j: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile_n;
        inline for (0..tile_n) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

inline fn accumulatePackedP(
    comptime T: type,
    comptime lane_count: comptime_int,
    comptime tile_n: comptime_int,
    task: gemm_task.Task(T),
    b_pack: []const T,
    i: usize,
    p: usize,
    acc: *[tile_n]@Vector(lane_count, T),
) void {
    const V = @Vector(lane_count, T);
    const av = loadVec(T, lane_count, task.a, task.lda, i, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        acc.*[col] = @mulAdd(V, av, @as(V, @splat(b_pack[b_base + col])), acc.*[col]);
    }
}

fn kernelPacked(
    comptime T: type,
    comptime lane_count: comptime_int,
    comptime tile_n: comptime_int,
    task: gemm_task.Task(T),
    b_pack: []const T,
    i: usize,
    j: usize,
) void {
    const V = @Vector(lane_count, T);
    const zero: V = @splat(0);
    var acc: [tile_n]V = [_]V{zero} ** tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulatePackedP(T, lane_count, tile_n, task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) accumulatePackedP(T, lane_count, tile_n, task, b_pack, i, p, &acc);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeVec(T, lane_count, task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: V = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out = @mulAdd(V, acc[col], alpha_v, oldVec(T, lane_count, task, i, j + col));
        storeVec(T, lane_count, task.c, task.ldc, i, j + col, out);
    }
}

fn tailRowsPacked(comptime T: type, comptime tile_n: comptime_int, task: gemm_task.Task(T), b_pack: []const T, row_start: usize, j: usize) void {
    var i = row_start;
    while (i < task.m) : (i += 1) {
        var acc: [tile_n]T = [_]T{0} ** tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * tile_n;
            inline for (0..tile_n) |col| {
                acc[col] = @mulAdd(T, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..tile_n) |col| {
            const idxc = matIndex(task.ldc, i, j + col);
            task.c[idxc] = if (task.alpha == 1 and task.beta == 0) acc[col] else blk: {
                const old = if (task.beta == 0) 0 else if (task.beta == 1) task.c[idxc] else task.beta * task.c[idxc];
                break :blk @mulAdd(T, task.alpha, acc[col], old);
            };
        }
    }
}

fn noTransRealTyped(comptime T: type, task: gemm_task.Task(T)) void {
    const lane_count = lanes(T);
    const tile_n = columnTile(T);

    const b_pack = std.heap.c_allocator.alloc(T, task.k * tile_n) catch {
        generic.noTransReal(T, task);
        return;
    };
    defer std.heap.c_allocator.free(b_pack);

    var j = task.n0;
    while (j + tile_n <= task.n1) : (j += tile_n) {
        packBPanel(T, tile_n, task, j, b_pack);
        var i: usize = 0;
        while (i + lane_count <= task.m) : (i += lane_count) {
            kernelPacked(T, lane_count, tile_n, task, b_pack, i, j);
        }
        tailRowsPacked(T, tile_n, task, b_pack, i, j);
    }

    if (j < task.n1) {
        var tail = task;
        tail.n0 = j;
        generic.noTransReal(T, tail);
    }
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    noTransRealTyped(f32, task);
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    noTransRealTyped(f64, task);
}
