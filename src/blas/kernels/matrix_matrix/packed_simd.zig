// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared packed-B SIMD GEMM micro-kernel body.
//!
//! Architecture modules provide feature gates, lane/panel sizes, and fallback
//! policy. This file owns the reusable K loop, packed-B layout, row tails, and
//! alpha/beta write-back logic for fixed-width SIMD backends.

const std = @import("std");

const epilogue = @import("epilogue.zig");
const generic = @import("../generic/matrix_matrix.zig");
const gemm_task = @import("task.zig");

const matIndex = gemm_task.matIndex;

pub const Config = struct {
    lane_count: comptime_int,
    tile_n: comptime_int,
    row_groups: comptime_int = 1,
    k_unroll: comptime_int = 4,
    tail_vector_lanes: comptime_int = 0,
    max_stack_pack_bytes: comptime_int = 0,
    pack_tail_columns: bool = false,
};

fn checkConfig(comptime T: type, comptime cfg: Config) void {
    if (T != f32 and T != f64) @compileError("packed SIMD GEMM kernels support f32 and f64");
    if (cfg.lane_count == 0) @compileError("packed SIMD GEMM lane_count must be nonzero");
    if (cfg.tile_n == 0) @compileError("packed SIMD GEMM tile_n must be nonzero");
    if (cfg.row_groups == 0) @compileError("packed SIMD GEMM row_groups must be nonzero");
    if (cfg.k_unroll == 0) @compileError("packed SIMD GEMM k_unroll must be nonzero");
    if (cfg.tail_vector_lanes >= cfg.lane_count) @compileError("tail_vector_lanes must be smaller than lane_count");
}

fn withTile(comptime cfg: Config, comptime tile_n: comptime_int) Config {
    return .{
        .lane_count = cfg.lane_count,
        .tile_n = tile_n,
        .row_groups = cfg.row_groups,
        .k_unroll = cfg.k_unroll,
        .tail_vector_lanes = cfg.tail_vector_lanes,
        .max_stack_pack_bytes = cfg.max_stack_pack_bytes,
        .pack_tail_columns = cfg.pack_tail_columns,
    };
}

fn maxStackPackElems(comptime T: type, comptime cfg: Config) comptime_int {
    return cfg.max_stack_pack_bytes / @sizeOf(T);
}

fn plannedStackPackElems(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T)) usize {
    return @min(@as(usize, maxStackPackElems(T, cfg)), task.execution.pack.stack_bytes / @sizeOf(T));
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
    if (task.beta == 0) return @splat(0);
    return loadVec(T, lane_count, task.c, task.ldc, row, col);
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

fn preparePanel(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), j: usize, b_pack: []T) []const T {
    const panel = b_pack[0 .. task.k * cfg.tile_n];
    packBPanel(T, cfg.tile_n, task, j, panel);
    return panel;
}

inline fn writeVectorEpilogue(
    comptime T: type,
    comptime lane_count: comptime_int,
    task: gemm_task.Task(T),
    row: usize,
    col: usize,
    acc: @Vector(lane_count, T),
) void {
    const out = if (task.alpha == 1 and task.beta == 0)
        acc
    else
        epilogue.applyVector(T, lane_count, task.alpha, task.beta, acc, oldVec(T, lane_count, task, row, col));
    storeVec(T, lane_count, task.c, task.ldc, row, col, out);
}

inline fn writeScalarEpilogue(comptime T: type, task: gemm_task.Task(T), row: usize, col: usize, acc: T) void {
    task.c[matIndex(task.ldc, row, col)] = epilogue.applyTaskScalar(T, task, acc, row, col);
}

inline fn accumulatePackedP(
    comptime T: type,
    comptime cfg: Config,
    comptime groups: comptime_int,
    task: gemm_task.Task(T),
    b_pack: []const T,
    i: usize,
    p: usize,
    acc: *[groups][cfg.tile_n]@Vector(cfg.lane_count, T),
) void {
    const V = @Vector(cfg.lane_count, T);
    const b_base = p * cfg.tile_n;
    inline for (0..groups) |group| {
        const av = loadVec(T, cfg.lane_count, task.a, task.lda, i + group * cfg.lane_count, p);
        inline for (0..cfg.tile_n) |col| {
            acc.*[group][col] = @mulAdd(V, av, @as(V, @splat(b_pack[b_base + col])), acc.*[group][col]);
        }
    }
}

fn kernelPacked(
    comptime T: type,
    comptime cfg: Config,
    comptime groups: comptime_int,
    task: gemm_task.Task(T),
    b_pack: []const T,
    i: usize,
    j: usize,
) void {
    const V = @Vector(cfg.lane_count, T);
    const zero: V = @splat(0);
    var acc: [groups][cfg.tile_n]V = undefined;
    inline for (0..groups) |group| {
        inline for (0..cfg.tile_n) |col| {
            acc[group][col] = zero;
        }
    }

    var p: usize = 0;
    while (p + cfg.k_unroll <= task.k) : (p += cfg.k_unroll) {
        inline for (0..cfg.k_unroll) |u| {
            accumulatePackedP(T, cfg, groups, task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) {
        accumulatePackedP(T, cfg, groups, task, b_pack, i, p, &acc);
    }

    inline for (0..groups) |group| {
        inline for (0..cfg.tile_n) |col| {
            const row = i + group * cfg.lane_count;
            writeVectorEpilogue(T, cfg.lane_count, task, row, j + col, acc[group][col]);
        }
    }
}

fn tailRowsScalarPacked(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []const T, row_start: usize, j: usize) void {
    var i = row_start;
    while (i < task.m) : (i += 1) {
        var acc: [cfg.tile_n]T = [_]T{@as(T, 0)} ** cfg.tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * cfg.tile_n;
            inline for (0..cfg.tile_n) |col| {
                acc[col] = @mulAdd(T, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..cfg.tile_n) |col| {
            writeScalarEpilogue(T, task, i, j + col, acc[col]);
        }
    }
}

fn tailRowsPacked(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []const T, row_start: usize, j: usize) void {
    var i = row_start;

    inline for (1..cfg.row_groups) |offset| {
        const groups = cfg.row_groups - offset;
        while (i + cfg.lane_count * groups <= task.m) : (i += cfg.lane_count * groups) {
            kernelPacked(T, cfg, groups, task, b_pack, i, j);
        }
    }

    if (comptime cfg.tail_vector_lanes != 0) {
        const tail_cfg = Config{
            .lane_count = cfg.tail_vector_lanes,
            .tile_n = cfg.tile_n,
            .row_groups = 1,
            .k_unroll = cfg.k_unroll,
        };
        while (i + cfg.tail_vector_lanes <= task.m) : (i += cfg.tail_vector_lanes) {
            kernelPacked(T, tail_cfg, 1, task, b_pack, i, j);
        }
    }

    tailRowsScalarPacked(T, cfg, task, b_pack, i, j);
}

fn runPreparedPanel(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), panel: []const T, j: usize) void {
    var i: usize = 0;
    while (i + cfg.lane_count * cfg.row_groups <= task.m) : (i += cfg.lane_count * cfg.row_groups) {
        kernelPacked(T, cfg, cfg.row_groups, task, panel, i, j);
    }
    tailRowsPacked(T, cfg, task, panel, i, j);
}

fn runPanel(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []T, j: usize) void {
    runPreparedPanel(T, cfg, task, preparePanel(T, cfg, task, j, b_pack), j);
}

fn tailColsPackedN(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []T, j: usize) void {
    runPanel(T, cfg, task, b_pack, j);
}

fn tailColsPacked(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []T, j: usize) void {
    const tile_n = task.n1 - j;
    if (tile_n == 0) return;
    inline for (1..cfg.tile_n) |tail_tile_n| {
        if (tile_n == tail_tile_n) {
            tailColsPackedN(T, withTile(cfg, tail_tile_n), task, b_pack, j);
            return;
        }
    }
    unreachable;
}

fn tailColsGeneric(comptime T: type, task: gemm_task.Task(T), j: usize) void {
    var tail = task;
    tail.n0 = j;
    generic.noTransReal(T, tail);
}

fn noTransRealWithPack(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), b_pack: []T) void {
    var j = task.n0;
    while (j + cfg.tile_n <= task.n1) : (j += cfg.tile_n) {
        runPanel(T, cfg, task, b_pack, j);
    }

    if (j < task.n1) {
        if (comptime cfg.pack_tail_columns) {
            tailColsPacked(T, cfg, task, b_pack, j);
        } else {
            tailColsGeneric(T, task, j);
        }
    }
}

fn noTransRealWithHeapPack(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T), pack_elems: usize) void {
    const b_pack = std.heap.c_allocator.alloc(T, pack_elems) catch {
        generic.noTransReal(T, task);
        return;
    };
    defer std.heap.c_allocator.free(b_pack);
    noTransRealWithPack(T, cfg, task, b_pack);
}

pub fn noTransReal(comptime T: type, comptime cfg: Config, task: gemm_task.Task(T)) void {
    comptime checkConfig(T, cfg);
    const pack_elems = task.k * cfg.tile_n;

    if (comptime cfg.max_stack_pack_bytes != 0) {
        if (pack_elems <= plannedStackPackElems(T, cfg, task)) {
            var stack_pack: [maxStackPackElems(T, cfg)]T = undefined;
            noTransRealWithPack(T, cfg, task, stack_pack[0..pack_elems]);
            return;
        }
    }

    noTransRealWithHeapPack(T, cfg, task, pack_elems);
}
