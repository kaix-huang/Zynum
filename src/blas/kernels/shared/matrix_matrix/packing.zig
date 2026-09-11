// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared GEMM packing helpers for K-major, panel-contiguous B panels.

const gemm_task = @import("task.zig");

const matIndex = gemm_task.matIndex;

inline fn loadVector(comptime T: type, comptime lanes: usize, ptr: [*]const T) @Vector(lanes, T) {
    return @as(*align(1) const @Vector(lanes, T), @ptrCast(ptr)).*;
}

inline fn storeVector(comptime T: type, comptime lanes: usize, ptr: [*]T, value: @Vector(lanes, T)) void {
    @as(*align(1) @Vector(lanes, T), @ptrCast(ptr)).* = value;
}

fn packTransposedAF32(m: usize, k: usize, a: [*]const f32, lda: gemm_task.BlasInt, a_pack: []f32, ld_out: usize) void {
    var p: usize = 0;
    while (p + 4 <= k) : (p += 4) {
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const v0 = loadVector(f32, 4, a + matIndex(lda, p, i + 0));
            const v1 = loadVector(f32, 4, a + matIndex(lda, p, i + 1));
            const v2 = loadVector(f32, 4, a + matIndex(lda, p, i + 2));
            const v3 = loadVector(f32, 4, a + matIndex(lda, p, i + 3));
            const t0 = @shuffle(f32, v0, v1, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
            const t1 = @shuffle(f32, v0, v1, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
            const t2 = @shuffle(f32, v2, v3, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
            const t3 = @shuffle(f32, v2, v3, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
            storeVector(f32, 4, a_pack.ptr + (p + 0) * ld_out + i, @shuffle(f32, t0, t2, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }));
            storeVector(f32, 4, a_pack.ptr + (p + 1) * ld_out + i, @shuffle(f32, t1, t3, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }));
            storeVector(f32, 4, a_pack.ptr + (p + 2) * ld_out + i, @shuffle(f32, t0, t2, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }));
            storeVector(f32, 4, a_pack.ptr + (p + 3) * ld_out + i, @shuffle(f32, t1, t3, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }));
        }
        while (i < m) : (i += 1) {
            inline for (0..4) |lane| a_pack[(p + lane) * ld_out + i] = a[matIndex(lda, p + lane, i)];
        }
    }
    while (p < k) : (p += 1) {
        for (0..m) |i| a_pack[p * ld_out + i] = a[matIndex(lda, p, i)];
    }
}

fn packTransposedAF64(m: usize, k: usize, a: [*]const f64, lda: gemm_task.BlasInt, a_pack: []f64, ld_out: usize) void {
    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        var i: usize = 0;
        while (i + 2 <= m) : (i += 2) {
            const v0 = loadVector(f64, 2, a + matIndex(lda, p, i + 0));
            const v1 = loadVector(f64, 2, a + matIndex(lda, p, i + 1));
            storeVector(f64, 2, a_pack.ptr + (p + 0) * ld_out + i, @shuffle(f64, v0, v1, @Vector(2, i32){ 0, ~@as(i32, 0) }));
            storeVector(f64, 2, a_pack.ptr + (p + 1) * ld_out + i, @shuffle(f64, v0, v1, @Vector(2, i32){ 1, ~@as(i32, 1) }));
        }
        while (i < m) : (i += 1) {
            a_pack[(p + 0) * ld_out + i] = a[matIndex(lda, p + 0, i)];
            a_pack[(p + 1) * ld_out + i] = a[matIndex(lda, p + 1, i)];
        }
    }
    while (p < k) : (p += 1) {
        for (0..m) |i| a_pack[p * ld_out + i] = a[matIndex(lda, p, i)];
    }
}

pub fn packTransposedA(comptime T: type, m: usize, k: usize, a: [*]const T, lda: gemm_task.BlasInt, a_pack: []T) void {
    if (@import("builtin").cpu.arch == .x86_64 and T == f64 and m >= 64 and k >= 1024) {
        // Bound simultaneous input/output pages and keep partial cache lines
        // nearby instead of revisiting all columns for every vector of K.
        var p0: usize = 0;
        while (p0 < k) : (p0 += 32) {
            var row_start: usize = 0;
            while (row_start < m) : (row_start += 32) {
                const rows = @min(@as(usize, 32), m - row_start);
                const depth = @min(@as(usize, 32), k - p0);
                const input = a + matIndex(lda, p0, row_start);
                const output = a_pack[p0 * m + row_start ..];
                if (T == f32) packTransposedAF32(rows, depth, input, lda, output, m) else if (T == f64) packTransposedAF64(rows, depth, input, lda, output, m) else @compileError("transposed A packing supports f32 and f64");
            }
        }
        return;
    }
    if (T == f32) return packTransposedAF32(m, k, a, lda, a_pack, m);
    if (T == f64) return packTransposedAF64(m, k, a, lda, a_pack, m);
    @compileError("transposed A packing supports f32 and f64");
}

pub inline fn isTransposedB(comptime T: type, task: gemm_task.Task(T)) bool {
    return task.b_layout == .trans;
}

pub inline fn bValue(comptime T: type, task: gemm_task.Task(T), row: usize, col: usize) T {
    if (isTransposedB(T, task)) {
        return task.b[matIndex(task.ldb, col, row)];
    }
    return task.b[matIndex(task.ldb, row, col)];
}

pub noinline fn packBPanelF32x6NoTrans(task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        const v0 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 0));
        const v1 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 1));
        const v2 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 2));
        const v3 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 3));
        const v4 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 4));
        const v5 = loadVector(f32, 4, task.b + matIndex(task.ldb, p, j + 5));
        const t0 = @shuffle(f32, v0, v1, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
        const t1 = @shuffle(f32, v0, v1, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
        const t2 = @shuffle(f32, v2, v3, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
        const t3 = @shuffle(f32, v2, v3, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
        const rows = [4]@Vector(4, f32){
            @shuffle(f32, t0, t2, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }),
            @shuffle(f32, t1, t3, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }),
            @shuffle(f32, t0, t2, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }),
            @shuffle(f32, t1, t3, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }),
        };
        inline for (0..4) |lane| {
            const base = (p + lane) * 6;
            storeVector(f32, 4, b_pack.ptr + base, rows[lane]);
            b_pack[base + 4] = v4[lane];
            b_pack[base + 5] = v5[lane];
        }
    }
    while (p < task.k) : (p += 1) {
        const base = p * 6;
        inline for (0..6) |col| b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
    }
}

pub noinline fn packBPanelF64x8NoTrans(task: gemm_task.Task(f64), j: usize, b_pack: []f64) void {
    var p: usize = 0;
    while (p + 2 <= task.k) : (p += 2) {
        inline for (0..4) |pair| {
            const col = pair * 2;
            const v0 = loadVector(f64, 2, task.b + matIndex(task.ldb, p, j + col));
            const v1 = loadVector(f64, 2, task.b + matIndex(task.ldb, p, j + col + 1));
            storeVector(f64, 2, b_pack.ptr + p * 8 + col, @shuffle(f64, v0, v1, @Vector(2, i32){ 0, ~@as(i32, 0) }));
            storeVector(f64, 2, b_pack.ptr + (p + 1) * 8 + col, @shuffle(f64, v0, v1, @Vector(2, i32){ 1, ~@as(i32, 1) }));
        }
    }
    while (p < task.k) : (p += 1) {
        const base = p * 8;
        inline for (0..8) |col| b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
    }
}

pub inline fn packBPanel(comptime T: type, comptime panel_cols: usize, task: gemm_task.Task(T), j: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        inline for (0..panel_cols) |col| {
            b_pack[base + col] = bValue(T, task, p, j + col);
        }
    }
}

pub fn packBPanelDynamic(comptime T: type, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            b_pack[base + col] = bValue(T, task, p, j + col);
        }
    }
}

pub fn packBPanelWithDefault(comptime T: type, comptime default_panel_cols: usize, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    switch (panel_cols) {
        default_panel_cols => packBPanel(T, default_panel_cols, task, j, b_pack),
        else => packBPanelDynamic(T, task, j, panel_cols, b_pack),
    }
}
