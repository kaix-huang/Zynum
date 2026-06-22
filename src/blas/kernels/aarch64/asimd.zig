// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const gemm_task = @import("../gemm_task.zig");
const features = @import("features.zig");
const generic = @import("../generic/gemm.zig");

const matIndex = gemm_task.matIndex;

const f32_tile_n = 8;
const f64_tile_n = 8;
const stack_pack_bytes = 64 * 1024;
const stack_pack_f32_elems = stack_pack_bytes / @sizeOf(f32);
const stack_pack_f64_elems = stack_pack_bytes / @sizeOf(f64);
const F32Vec = @Vector(4, f32);
const F32Pair = @Vector(2, f32);
const F64Vec = @Vector(2, f64);

pub const enabled: bool = features.has_asimd;

pub fn preferredColumnBlock(comptime T: type) usize {
    if (T == f32) return f32_tile_n;
    if (T == f64) return f64_tile_n;
    @compileError("ASIMD GEMM kernels support f32 and f64");
}

inline fn loadF32x4(ptr: [*]const f32, lda: gemm_task.BlasInt, row: usize, col: usize) F32Vec {
    const v: *align(1) const F32Vec = @ptrCast(ptr + matIndex(lda, row, col));
    return v.*;
}

inline fn loadF32x2(ptr: [*]const f32, lda: gemm_task.BlasInt, row: usize, col: usize) F32Pair {
    const v: *align(1) const F32Pair = @ptrCast(ptr + matIndex(lda, row, col));
    return v.*;
}

inline fn loadF64x2(ptr: [*]const f64, lda: gemm_task.BlasInt, row: usize, col: usize) F64Vec {
    const v: *align(1) const F64Vec = @ptrCast(ptr + matIndex(lda, row, col));
    return v.*;
}

inline fn storeF32x4(ptr: [*]f32, lda: gemm_task.BlasInt, row: usize, col: usize, v: F32Vec) void {
    const out: *align(1) F32Vec = @ptrCast(ptr + matIndex(lda, row, col));
    out.* = v;
}

inline fn storeF32x2(ptr: [*]f32, lda: gemm_task.BlasInt, row: usize, col: usize, v: F32Pair) void {
    const out: *align(1) F32Pair = @ptrCast(ptr + matIndex(lda, row, col));
    out.* = v;
}

inline fn storeF64x2(ptr: [*]f64, lda: gemm_task.BlasInt, row: usize, col: usize, v: F64Vec) void {
    const out: *align(1) F64Vec = @ptrCast(ptr + matIndex(lda, row, col));
    out.* = v;
}

inline fn oldF32x4(task: gemm_task.Task(f32), row: usize, col: usize) F32Vec {
    if (task.beta == 0) return @splat(0);
    const old = loadF32x4(task.c, task.ldc, row, col);
    if (task.beta == 1) return old;
    return old * @as(F32Vec, @splat(task.beta));
}

inline fn oldF32x2(task: gemm_task.Task(f32), row: usize, col: usize) F32Pair {
    if (task.beta == 0) return @splat(0);
    const old = loadF32x2(task.c, task.ldc, row, col);
    if (task.beta == 1) return old;
    return old * @as(F32Pair, @splat(task.beta));
}

inline fn oldF64x2(task: gemm_task.Task(f64), row: usize, col: usize) F64Vec {
    if (task.beta == 0) return @splat(0);
    const old = loadF64x2(task.c, task.ldc, row, col);
    if (task.beta == 1) return old;
    return old * @as(F64Vec, @splat(task.beta));
}

fn packBPanelF32(task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * f32_tile_n;
        inline for (0..f32_tile_n) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF32Partial(comptime tile_n: usize, task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile_n;
        inline for (0..tile_n) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF64(task: gemm_task.Task(f64), j: usize, b_pack: []f64) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * f64_tile_n;
        inline for (0..f64_tile_n) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF64Partial(comptime tile_n: usize, task: gemm_task.Task(f64), j: usize, b_pack: []f64) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile_n;
        inline for (0..tile_n) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

inline fn accumulateF32PackedP(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc_lo: *[f32_tile_n]F32Vec, acc_hi: *[f32_tile_n]F32Vec) void {
    const a_lo = loadF32x4(task.a, task.lda, i, p);
    const a_hi = loadF32x4(task.a, task.lda, i + 4, p);
    const b_base = p * f32_tile_n;
    inline for (0..f32_tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc_lo.*[col] = @mulAdd(F32Vec, a_lo, bv, acc_lo.*[col]);
        acc_hi.*[col] = @mulAdd(F32Vec, a_hi, bv, acc_hi.*[col]);
    }
}

inline fn accumulateF32PackedPartialP(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc_lo: *[f32_tile_n]F32Vec, acc_hi: *[f32_tile_n]F32Vec) void {
    const a_lo = loadF32x4(task.a, task.lda, i, p);
    const a_hi = loadF32x4(task.a, task.lda, i + 4, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc_lo.*[col] = @mulAdd(F32Vec, a_lo, bv, acc_lo.*[col]);
        acc_hi.*[col] = @mulAdd(F32Vec, a_hi, bv, acc_hi.*[col]);
    }
}

inline fn accumulateF32PackedP12(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc0: *[f32_tile_n]F32Vec, acc1: *[f32_tile_n]F32Vec, acc2: *[f32_tile_n]F32Vec) void {
    const a0 = loadF32x4(task.a, task.lda, i, p);
    const a1 = loadF32x4(task.a, task.lda, i + 4, p);
    const a2 = loadF32x4(task.a, task.lda, i + 8, p);
    const b_base = p * f32_tile_n;
    inline for (0..f32_tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F32Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F32Vec, a1, bv, acc1.*[col]);
        acc2.*[col] = @mulAdd(F32Vec, a2, bv, acc2.*[col]);
    }
}

inline fn accumulateF32PackedPartialP12(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc0: *[f32_tile_n]F32Vec, acc1: *[f32_tile_n]F32Vec, acc2: *[f32_tile_n]F32Vec) void {
    const a0 = loadF32x4(task.a, task.lda, i, p);
    const a1 = loadF32x4(task.a, task.lda, i + 4, p);
    const a2 = loadF32x4(task.a, task.lda, i + 8, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F32Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F32Vec, a1, bv, acc1.*[col]);
        acc2.*[col] = @mulAdd(F32Vec, a2, bv, acc2.*[col]);
    }
}

inline fn accumulateF32PackedP4(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc: *[f32_tile_n]F32Vec) void {
    const av = loadF32x4(task.a, task.lda, i, p);
    const b_base = p * f32_tile_n;
    inline for (0..f32_tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc.*[col] = @mulAdd(F32Vec, av, bv, acc.*[col]);
    }
}

inline fn accumulateF32PackedPartialP4(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc: *[f32_tile_n]F32Vec) void {
    const av = loadF32x4(task.a, task.lda, i, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F32Vec = @splat(b_pack[b_base + col]);
        acc.*[col] = @mulAdd(F32Vec, av, bv, acc.*[col]);
    }
}

inline fn accumulateF32PackedP2(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc: *[f32_tile_n]F32Pair) void {
    const av = loadF32x2(task.a, task.lda, i, p);
    const b_base = p * f32_tile_n;
    inline for (0..f32_tile_n) |col| {
        const bv: F32Pair = @splat(b_pack[b_base + col]);
        acc.*[col] = @mulAdd(F32Pair, av, bv, acc.*[col]);
    }
}

inline fn accumulateF32PackedPartialP2(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, p: usize, acc: *[f32_tile_n]F32Pair) void {
    const av = loadF32x2(task.a, task.lda, i, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F32Pair = @splat(b_pack[b_base + col]);
        acc.*[col] = @mulAdd(F32Pair, av, bv, acc.*[col]);
    }
}

fn kernelF32x8x12Packed(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc_lo: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc_hi: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedP(task, b_pack, i, p + u, &acc_lo, &acc_hi);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedP(task, b_pack, i, p, &acc_lo, &acc_hi);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f32_tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc_lo[col]);
            storeF32x4(task.c, task.ldc, i + 4, j + col, acc_hi[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..f32_tile_n) |col| {
        const out_lo = @mulAdd(F32Vec, acc_lo[col], alpha_v, oldF32x4(task, i, j + col));
        const out_hi = @mulAdd(F32Vec, acc_hi[col], alpha_v, oldF32x4(task, i + 4, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out_lo);
        storeF32x4(task.c, task.ldc, i + 4, j + col, out_hi);
    }
}

fn kernelF32x8xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc_lo: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc_hi: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedPartialP(tile_n, task, b_pack, i, p + u, &acc_lo, &acc_hi);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedPartialP(tile_n, task, b_pack, i, p, &acc_lo, &acc_hi);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc_lo[col]);
            storeF32x4(task.c, task.ldc, i + 4, j + col, acc_hi[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out_lo = @mulAdd(F32Vec, acc_lo[col], alpha_v, oldF32x4(task, i, j + col));
        const out_hi = @mulAdd(F32Vec, acc_hi[col], alpha_v, oldF32x4(task, i + 4, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out_lo);
        storeF32x4(task.c, task.ldc, i + 4, j + col, out_hi);
    }
}

fn kernelF32x12x8Packed(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc0: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc1: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc2: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedP12(task, b_pack, i, p + u, &acc0, &acc1, &acc2);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedP12(task, b_pack, i, p, &acc0, &acc1, &acc2);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f32_tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc0[col]);
            storeF32x4(task.c, task.ldc, i + 4, j + col, acc1[col]);
            storeF32x4(task.c, task.ldc, i + 8, j + col, acc2[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..f32_tile_n) |col| {
        const out0 = @mulAdd(F32Vec, acc0[col], alpha_v, oldF32x4(task, i, j + col));
        const out1 = @mulAdd(F32Vec, acc1[col], alpha_v, oldF32x4(task, i + 4, j + col));
        const out2 = @mulAdd(F32Vec, acc2[col], alpha_v, oldF32x4(task, i + 8, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out0);
        storeF32x4(task.c, task.ldc, i + 4, j + col, out1);
        storeF32x4(task.c, task.ldc, i + 8, j + col, out2);
    }
}

fn kernelF32x12xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc0: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc1: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;
    var acc2: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedPartialP12(tile_n, task, b_pack, i, p + u, &acc0, &acc1, &acc2);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedPartialP12(tile_n, task, b_pack, i, p, &acc0, &acc1, &acc2);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc0[col]);
            storeF32x4(task.c, task.ldc, i + 4, j + col, acc1[col]);
            storeF32x4(task.c, task.ldc, i + 8, j + col, acc2[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out0 = @mulAdd(F32Vec, acc0[col], alpha_v, oldF32x4(task, i, j + col));
        const out1 = @mulAdd(F32Vec, acc1[col], alpha_v, oldF32x4(task, i + 4, j + col));
        const out2 = @mulAdd(F32Vec, acc2[col], alpha_v, oldF32x4(task, i + 8, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out0);
        storeF32x4(task.c, task.ldc, i + 4, j + col, out1);
        storeF32x4(task.c, task.ldc, i + 8, j + col, out2);
    }
}

fn kernelF32x4x8TailPacked(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedP4(task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedP4(task, b_pack, i, p, &acc);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f32_tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..f32_tile_n) |col| {
        const out = @mulAdd(F32Vec, acc[col], alpha_v, oldF32x4(task, i, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out);
    }
}

fn kernelF32x4xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Vec = @splat(0);
    var acc: [f32_tile_n]F32Vec = [_]F32Vec{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedPartialP4(tile_n, task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedPartialP4(tile_n, task, b_pack, i, p, &acc);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF32x4(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F32Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out = @mulAdd(F32Vec, acc[col], alpha_v, oldF32x4(task, i, j + col));
        storeF32x4(task.c, task.ldc, i, j + col, out);
    }
}

fn kernelF32x2x8TailPacked(task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Pair = @splat(0);
    var acc: [f32_tile_n]F32Pair = [_]F32Pair{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedP2(task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedP2(task, b_pack, i, p, &acc);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f32_tile_n) |col| {
            storeF32x2(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F32Pair = @splat(task.alpha);
    inline for (0..f32_tile_n) |col| {
        const out = @mulAdd(F32Pair, acc[col], alpha_v, oldF32x2(task, i, j + col));
        storeF32x2(task.c, task.ldc, i, j + col, out);
    }
}

fn kernelF32x2xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, i: usize, j: usize) void {
    const zero: F32Pair = @splat(0);
    var acc: [f32_tile_n]F32Pair = [_]F32Pair{zero} ** f32_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF32PackedPartialP2(tile_n, task, b_pack, i, p + u, &acc);
        }
    }
    while (p < task.k) : (p += 1) accumulateF32PackedPartialP2(tile_n, task, b_pack, i, p, &acc);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF32x2(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F32Pair = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out = @mulAdd(F32Pair, acc[col], alpha_v, oldF32x2(task, i, j + col));
        storeF32x2(task.c, task.ldc, i, j + col, out);
    }
}

fn tailRowsF32x8Packed(task: gemm_task.Task(f32), b_pack: []const f32, row_start: usize, j: usize) void {
    var i = row_start;
    while (i + 8 <= task.m) : (i += 8) {
        kernelF32x8x12Packed(task, b_pack, i, j);
    }
    while (i + 4 <= task.m) : (i += 4) {
        kernelF32x4x8TailPacked(task, b_pack, i, j);
    }
    while (i + 2 <= task.m) : (i += 2) {
        kernelF32x2x8TailPacked(task, b_pack, i, j);
    }
    while (i < task.m) : (i += 1) {
        var acc: [f32_tile_n]f32 = [_]f32{0} ** f32_tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * f32_tile_n;
            inline for (0..f32_tile_n) |col| {
                acc[col] = @mulAdd(f32, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..f32_tile_n) |col| {
            const idxc = matIndex(task.ldc, i, j + col);
            task.c[idxc] = if (task.alpha == 1 and task.beta == 0) acc[col] else blk: {
                const old = if (task.beta == 0) 0 else if (task.beta == 1) task.c[idxc] else task.beta * task.c[idxc];
                break :blk @mulAdd(f32, task.alpha, acc[col], old);
            };
        }
    }
}

fn tailRowsF32x8TailPacked(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []const f32, row_start: usize, j: usize) void {
    var i = row_start;
    while (i + 8 <= task.m) : (i += 8) {
        kernelF32x8xTailPacked(tile_n, task, b_pack, i, j);
    }
    while (i + 4 <= task.m) : (i += 4) {
        kernelF32x4xTailPacked(tile_n, task, b_pack, i, j);
    }
    while (i + 2 <= task.m) : (i += 2) {
        kernelF32x2xTailPacked(tile_n, task, b_pack, i, j);
    }
    while (i < task.m) : (i += 1) {
        var acc: [f32_tile_n]f32 = [_]f32{0} ** f32_tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * tile_n;
            inline for (0..tile_n) |col| {
                acc[col] = @mulAdd(f32, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..tile_n) |col| {
            const idxc = matIndex(task.ldc, i, j + col);
            task.c[idxc] = if (task.alpha == 1 and task.beta == 0) acc[col] else blk: {
                const old = if (task.beta == 0) 0 else if (task.beta == 1) task.c[idxc] else task.beta * task.c[idxc];
                break :blk @mulAdd(f32, task.alpha, acc[col], old);
            };
        }
    }
}

fn tailColsF32PackedN(comptime tile_n: usize, task: gemm_task.Task(f32), b_pack: []f32, j: usize) void {
    packBPanelF32Partial(tile_n, task, j, b_pack[0 .. task.k * tile_n]);
    var i: usize = 0;
    while (i + 12 <= task.m) : (i += 12) {
        kernelF32x12xTailPacked(tile_n, task, b_pack, i, j);
    }
    tailRowsF32x8TailPacked(tile_n, task, b_pack, i, j);
}

fn tailColsF32Packed(task: gemm_task.Task(f32), b_pack: []f32, j: usize) void {
    const tile_n = task.n1 - j;
    if (tile_n == 0) return;
    switch (tile_n) {
        1 => tailColsF32PackedN(1, task, b_pack, j),
        2 => tailColsF32PackedN(2, task, b_pack, j),
        3 => tailColsF32PackedN(3, task, b_pack, j),
        4 => tailColsF32PackedN(4, task, b_pack, j),
        5 => tailColsF32PackedN(5, task, b_pack, j),
        6 => tailColsF32PackedN(6, task, b_pack, j),
        7 => tailColsF32PackedN(7, task, b_pack, j),
        else => unreachable,
    }
}

fn noTransRealF32WithPack(task: gemm_task.Task(f32), b_pack: []f32) void {
    var j = task.n0;
    while (j + f32_tile_n <= task.n1) : (j += f32_tile_n) {
        packBPanelF32(task, j, b_pack);
        var i: usize = 0;
        while (i + 12 <= task.m) : (i += 12) {
            kernelF32x12x8Packed(task, b_pack, i, j);
        }
        tailRowsF32x8Packed(task, b_pack, i, j);
    }
    if (j < task.n1) {
        tailColsF32Packed(task, b_pack, j);
    }
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    const pack_elems = task.k * f32_tile_n;
    if (pack_elems <= stack_pack_f32_elems) {
        var stack_pack: [stack_pack_f32_elems]f32 = undefined;
        noTransRealF32WithPack(task, stack_pack[0..pack_elems]);
        return;
    }

    const b_pack = std.heap.c_allocator.alloc(f32, pack_elems) catch {
        generic.noTransRealF32(task);
        return;
    };
    defer std.heap.c_allocator.free(b_pack);
    noTransRealF32WithPack(task, b_pack);
}

inline fn accumulateF64PackedP(task: gemm_task.Task(f64), b_pack: []const f64, i: usize, p: usize, acc0: *[f64_tile_n]F64Vec, acc1: *[f64_tile_n]F64Vec, acc2: *[f64_tile_n]F64Vec) void {
    const a0 = loadF64x2(task.a, task.lda, i, p);
    const a1 = loadF64x2(task.a, task.lda, i + 2, p);
    const a2 = loadF64x2(task.a, task.lda, i + 4, p);
    const b_base = p * f64_tile_n;
    inline for (0..f64_tile_n) |col| {
        const bv: F64Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F64Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F64Vec, a1, bv, acc1.*[col]);
        acc2.*[col] = @mulAdd(F64Vec, a2, bv, acc2.*[col]);
    }
}

inline fn accumulateF64PackedPartialP(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, i: usize, p: usize, acc0: *[f64_tile_n]F64Vec, acc1: *[f64_tile_n]F64Vec, acc2: *[f64_tile_n]F64Vec) void {
    const a0 = loadF64x2(task.a, task.lda, i, p);
    const a1 = loadF64x2(task.a, task.lda, i + 2, p);
    const a2 = loadF64x2(task.a, task.lda, i + 4, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F64Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F64Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F64Vec, a1, bv, acc1.*[col]);
        acc2.*[col] = @mulAdd(F64Vec, a2, bv, acc2.*[col]);
    }
}

fn kernelF64x6x8Packed(task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc0: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc1: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc2: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF64PackedP(task, b_pack, i, p + u, &acc0, &acc1, &acc2);
        }
    }
    while (p < task.k) : (p += 1) accumulateF64PackedP(task, b_pack, i, p, &acc0, &acc1, &acc2);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f64_tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc0[col]);
            storeF64x2(task.c, task.ldc, i + 2, j + col, acc1[col]);
            storeF64x2(task.c, task.ldc, i + 4, j + col, acc2[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..f64_tile_n) |col| {
        const out0 = @mulAdd(F64Vec, acc0[col], alpha_v, oldF64x2(task, i, j + col));
        const out1 = @mulAdd(F64Vec, acc1[col], alpha_v, oldF64x2(task, i + 2, j + col));
        const out2 = @mulAdd(F64Vec, acc2[col], alpha_v, oldF64x2(task, i + 4, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out0);
        storeF64x2(task.c, task.ldc, i + 2, j + col, out1);
        storeF64x2(task.c, task.ldc, i + 4, j + col, out2);
    }
}

fn kernelF64x6xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc0: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc1: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc2: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF64PackedPartialP(tile_n, task, b_pack, i, p + u, &acc0, &acc1, &acc2);
        }
    }
    while (p < task.k) : (p += 1) accumulateF64PackedPartialP(tile_n, task, b_pack, i, p, &acc0, &acc1, &acc2);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc0[col]);
            storeF64x2(task.c, task.ldc, i + 2, j + col, acc1[col]);
            storeF64x2(task.c, task.ldc, i + 4, j + col, acc2[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out0 = @mulAdd(F64Vec, acc0[col], alpha_v, oldF64x2(task, i, j + col));
        const out1 = @mulAdd(F64Vec, acc1[col], alpha_v, oldF64x2(task, i + 2, j + col));
        const out2 = @mulAdd(F64Vec, acc2[col], alpha_v, oldF64x2(task, i + 4, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out0);
        storeF64x2(task.c, task.ldc, i + 2, j + col, out1);
        storeF64x2(task.c, task.ldc, i + 4, j + col, out2);
    }
}

inline fn accumulateF64PackedP4(task: gemm_task.Task(f64), b_pack: []const f64, i: usize, p: usize, acc0: *[f64_tile_n]F64Vec, acc1: *[f64_tile_n]F64Vec) void {
    const a0 = loadF64x2(task.a, task.lda, i, p);
    const a1 = loadF64x2(task.a, task.lda, i + 2, p);
    const b_base = p * f64_tile_n;
    inline for (0..f64_tile_n) |col| {
        const bv: F64Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F64Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F64Vec, a1, bv, acc1.*[col]);
    }
}

inline fn accumulateF64PackedPartialP4(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, i: usize, p: usize, acc0: *[f64_tile_n]F64Vec, acc1: *[f64_tile_n]F64Vec) void {
    const a0 = loadF64x2(task.a, task.lda, i, p);
    const a1 = loadF64x2(task.a, task.lda, i + 2, p);
    const b_base = p * tile_n;
    inline for (0..tile_n) |col| {
        const bv: F64Vec = @splat(b_pack[b_base + col]);
        acc0.*[col] = @mulAdd(F64Vec, a0, bv, acc0.*[col]);
        acc1.*[col] = @mulAdd(F64Vec, a1, bv, acc1.*[col]);
    }
}

fn kernelF64x4x8TailPacked(task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc0: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc1: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF64PackedP4(task, b_pack, i, p + u, &acc0, &acc1);
        }
    }
    while (p < task.k) : (p += 1) accumulateF64PackedP4(task, b_pack, i, p, &acc0, &acc1);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f64_tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc0[col]);
            storeF64x2(task.c, task.ldc, i + 2, j + col, acc1[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..f64_tile_n) |col| {
        const out0 = @mulAdd(F64Vec, acc0[col], alpha_v, oldF64x2(task, i, j + col));
        const out1 = @mulAdd(F64Vec, acc1[col], alpha_v, oldF64x2(task, i + 2, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out0);
        storeF64x2(task.c, task.ldc, i + 2, j + col, out1);
    }
}

fn kernelF64x4xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc0: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;
    var acc1: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            accumulateF64PackedPartialP4(tile_n, task, b_pack, i, p + u, &acc0, &acc1);
        }
    }
    while (p < task.k) : (p += 1) accumulateF64PackedPartialP4(tile_n, task, b_pack, i, p, &acc0, &acc1);

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc0[col]);
            storeF64x2(task.c, task.ldc, i + 2, j + col, acc1[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out0 = @mulAdd(F64Vec, acc0[col], alpha_v, oldF64x2(task, i, j + col));
        const out1 = @mulAdd(F64Vec, acc1[col], alpha_v, oldF64x2(task, i + 2, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out0);
        storeF64x2(task.c, task.ldc, i + 2, j + col, out1);
    }
}

fn kernelF64x2x8TailPacked(task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            const av = loadF64x2(task.a, task.lda, i, p + u);
            const b_base = (p + u) * f64_tile_n;
            inline for (0..f64_tile_n) |col| {
                const bv: F64Vec = @splat(b_pack[b_base + col]);
                acc[col] = @mulAdd(F64Vec, av, bv, acc[col]);
            }
        }
    }
    while (p < task.k) : (p += 1) {
        const av = loadF64x2(task.a, task.lda, i, p);
        const b_base = p * f64_tile_n;
        inline for (0..f64_tile_n) |col| {
            const bv: F64Vec = @splat(b_pack[b_base + col]);
            acc[col] = @mulAdd(F64Vec, av, bv, acc[col]);
        }
    }

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..f64_tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..f64_tile_n) |col| {
        const out = @mulAdd(F64Vec, acc[col], alpha_v, oldF64x2(task, i, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out);
    }
}

fn kernelF64x2xTailPacked(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, i: usize, j: usize) void {
    const zero: F64Vec = @splat(0);
    var acc: [f64_tile_n]F64Vec = [_]F64Vec{zero} ** f64_tile_n;

    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        inline for (0..4) |u| {
            const av = loadF64x2(task.a, task.lda, i, p + u);
            const b_base = (p + u) * tile_n;
            inline for (0..tile_n) |col| {
                const bv: F64Vec = @splat(b_pack[b_base + col]);
                acc[col] = @mulAdd(F64Vec, av, bv, acc[col]);
            }
        }
    }
    while (p < task.k) : (p += 1) {
        const av = loadF64x2(task.a, task.lda, i, p);
        const b_base = p * tile_n;
        inline for (0..tile_n) |col| {
            const bv: F64Vec = @splat(b_pack[b_base + col]);
            acc[col] = @mulAdd(F64Vec, av, bv, acc[col]);
        }
    }

    if (task.alpha == 1 and task.beta == 0) {
        inline for (0..tile_n) |col| {
            storeF64x2(task.c, task.ldc, i, j + col, acc[col]);
        }
        return;
    }

    const alpha_v: F64Vec = @splat(task.alpha);
    inline for (0..tile_n) |col| {
        const out = @mulAdd(F64Vec, acc[col], alpha_v, oldF64x2(task, i, j + col));
        storeF64x2(task.c, task.ldc, i, j + col, out);
    }
}

fn tailRowsF64x8Packed(task: gemm_task.Task(f64), b_pack: []const f64, row_start: usize, j: usize) void {
    var i = row_start;
    while (i + 4 <= task.m) : (i += 4) {
        kernelF64x4x8TailPacked(task, b_pack, i, j);
    }
    while (i + 2 <= task.m) : (i += 2) {
        kernelF64x2x8TailPacked(task, b_pack, i, j);
    }
    while (i < task.m) : (i += 1) {
        var acc: [f64_tile_n]f64 = [_]f64{0} ** f64_tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * f64_tile_n;
            inline for (0..f64_tile_n) |col| {
                acc[col] = @mulAdd(f64, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..f64_tile_n) |col| {
            const idxc = matIndex(task.ldc, i, j + col);
            task.c[idxc] = if (task.alpha == 1 and task.beta == 0) acc[col] else blk: {
                const old = if (task.beta == 0) 0 else if (task.beta == 1) task.c[idxc] else task.beta * task.c[idxc];
                break :blk @mulAdd(f64, task.alpha, acc[col], old);
            };
        }
    }
}

fn tailRowsF64x8TailPacked(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []const f64, row_start: usize, j: usize) void {
    var i = row_start;
    while (i + 4 <= task.m) : (i += 4) {
        kernelF64x4xTailPacked(tile_n, task, b_pack, i, j);
    }
    while (i + 2 <= task.m) : (i += 2) {
        kernelF64x2xTailPacked(tile_n, task, b_pack, i, j);
    }
    while (i < task.m) : (i += 1) {
        var acc: [f64_tile_n]f64 = [_]f64{0} ** f64_tile_n;
        for (0..task.k) |p| {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * tile_n;
            inline for (0..tile_n) |col| {
                acc[col] = @mulAdd(f64, av, b_pack[b_base + col], acc[col]);
            }
        }
        inline for (0..tile_n) |col| {
            const idxc = matIndex(task.ldc, i, j + col);
            task.c[idxc] = if (task.alpha == 1 and task.beta == 0) acc[col] else blk: {
                const old = if (task.beta == 0) 0 else if (task.beta == 1) task.c[idxc] else task.beta * task.c[idxc];
                break :blk @mulAdd(f64, task.alpha, acc[col], old);
            };
        }
    }
}

fn tailColsF64PackedN(comptime tile_n: usize, task: gemm_task.Task(f64), b_pack: []f64, j: usize) void {
    packBPanelF64Partial(tile_n, task, j, b_pack[0 .. task.k * tile_n]);
    var i: usize = 0;
    while (i + 6 <= task.m) : (i += 6) {
        kernelF64x6xTailPacked(tile_n, task, b_pack, i, j);
    }
    tailRowsF64x8TailPacked(tile_n, task, b_pack, i, j);
}

fn tailColsF64Packed(task: gemm_task.Task(f64), b_pack: []f64, j: usize) void {
    const tile_n = task.n1 - j;
    if (tile_n == 0) return;
    switch (tile_n) {
        1 => tailColsF64PackedN(1, task, b_pack, j),
        2 => tailColsF64PackedN(2, task, b_pack, j),
        3 => tailColsF64PackedN(3, task, b_pack, j),
        4 => tailColsF64PackedN(4, task, b_pack, j),
        5 => tailColsF64PackedN(5, task, b_pack, j),
        6 => tailColsF64PackedN(6, task, b_pack, j),
        7 => tailColsF64PackedN(7, task, b_pack, j),
        else => unreachable,
    }
}

fn noTransRealF64WithPack(task: gemm_task.Task(f64), b_pack: []f64) void {
    var j = task.n0;
    while (j + f64_tile_n <= task.n1) : (j += f64_tile_n) {
        packBPanelF64(task, j, b_pack);
        var i: usize = 0;
        while (i + 6 <= task.m) : (i += 6) {
            kernelF64x6x8Packed(task, b_pack, i, j);
        }
        tailRowsF64x8Packed(task, b_pack, i, j);
    }
    if (j < task.n1) {
        tailColsF64Packed(task, b_pack, j);
    }
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    const pack_elems = task.k * f64_tile_n;
    if (pack_elems <= stack_pack_f64_elems) {
        var stack_pack: [stack_pack_f64_elems]f64 = undefined;
        noTransRealF64WithPack(task, stack_pack[0..pack_elems]);
        return;
    }

    const b_pack = std.heap.c_allocator.alloc(f64, pack_elems) catch {
        generic.noTransRealF64(task);
        return;
    };
    defer std.heap.c_allocator.free(b_pack);
    noTransRealF64WithPack(task, b_pack);
}
