// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const amx_ops = @import("amx_ops.zig");
const gemm_task = @import("../../../shared/matrix_matrix/task.zig");

const Vec4f = @Vector(4, f32);
const Vec2d = @Vector(2, f64);
const max_stack_pack_elems: usize = 32768;
const compat_pack_plan = gemm_task.PackWorkspacePlan{
    .stack_bytes = 256 * 1024,
    .cache_bytes = 8 * 1024 * 1024,
};

threadlocal var cached_f32_pack_ptr: ?[*]f32 = null;
threadlocal var cached_f32_pack_len: usize = 0;
threadlocal var cached_f64_pack_ptr: ?[*]f64 = null;
threadlocal var cached_f64_pack_len: usize = 0;

const amxLdx = amx_ops.ldx;
const amxLdy = amx_ops.ldy;
const amxStz = amx_ops.stz;
const amxFma64 = amx_ops.fma64;
const amxFma32 = amx_ops.fma32;
const amxMatfp = amx_ops.matfp;
const amxSet = amx_ops.set;
const amxClr = amx_ops.clr;
const ptrRowFlags = amx_ops.ptrRowFlags;
const amxFma32RowOperand = amx_ops.fma32RowOperand;
const amxFma64RowOperand = amx_ops.fma64RowOperand;
const amxFma64XyOperand = amx_ops.fma64XyOperand;
const amxMatfp32RowOperand = amx_ops.matfp32RowOperand;
const amxFma32XyRowOperand = amx_ops.fma32XyRowOperand;

fn AmxPackWorkspace(comptime T: type) type {
    return struct {
        data: []T,
        cached: bool,

        fn deinit(self: @This()) void {
            if (!self.cached) std.heap.c_allocator.free(self.data);
        }
    };
}

fn acquirePack(comptime T: type, len: usize, cache_bytes: usize) ?AmxPackWorkspace(T) {
    const max_cached_elems = cache_bytes / @sizeOf(T);
    if (len > max_cached_elems) {
        const data = std.heap.c_allocator.alloc(T, len) catch return null;
        return .{ .data = data, .cached = false };
    }

    if (T == f32) {
        if (cached_f32_pack_len < len) {
            const data = std.heap.c_allocator.alloc(f32, len) catch return null;
            if (cached_f32_pack_ptr) |old| std.heap.c_allocator.free(old[0..cached_f32_pack_len]);
            cached_f32_pack_ptr = data.ptr;
            cached_f32_pack_len = len;
        }
        return .{ .data = cached_f32_pack_ptr.?[0..len], .cached = true };
    }
    if (T == f64) {
        if (cached_f64_pack_len < len) {
            const data = std.heap.c_allocator.alloc(f64, len) catch return null;
            if (cached_f64_pack_ptr) |old| std.heap.c_allocator.free(old[0..cached_f64_pack_len]);
            cached_f64_pack_ptr = data.ptr;
            cached_f64_pack_len = len;
        }
        return .{ .data = cached_f64_pack_ptr.?[0..len], .cached = true };
    }
    @compileError("AMX pack workspace supports f32 and f64");
}

pub fn freeCurrentThreadCaches() void {
    if (cached_f32_pack_ptr) |ptr| std.heap.c_allocator.free(ptr[0..cached_f32_pack_len]);
    cached_f32_pack_ptr = null;
    cached_f32_pack_len = 0;
    if (cached_f64_pack_ptr) |ptr| std.heap.c_allocator.free(ptr[0..cached_f64_pack_len]);
    cached_f64_pack_ptr = null;
    cached_f64_pack_len = 0;
}

inline fn canUseStackPack(comptime T: type, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return pack_elems <= max_stack_pack_elems and pack_elems * @sizeOf(T) <= workspace.stack_bytes;
}

noinline fn amxGemmStackPack(
    comptime T: type,
    comptime loop_fn: fn (usize, usize, usize, [*]const T, usize, [*]const T, usize, [*]T, usize, [*]T) void,
    comptime capacity: usize,
    pack_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    a: [*]const T,
    lda: usize,
    b: [*]const T,
    ldb: usize,
    c: [*]T,
    ldc: usize,
) void {
    var stack_pack: [capacity]T = undefined;
    loop_fn(m, n, k, a, lda, b, ldb, c, ldc, stack_pack[0..pack_elems].ptr);
}

fn amxGemmWithPack(
    comptime T: type,
    comptime loop_fn: fn (usize, usize, usize, [*]const T, usize, [*]const T, usize, [*]T, usize, [*]T) void,
    m: usize,
    n: usize,
    k: usize,
    a: [*]const T,
    lda: usize,
    b: [*]const T,
    ldb: usize,
    c: [*]T,
    ldc: usize,
    pack_elems: usize,
    workspace: gemm_task.PackWorkspacePlan,
) bool {
    if (canUseStackPack(T, pack_elems, workspace)) {
        amxGemmStackPack(T, loop_fn, max_stack_pack_elems, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        const heap_workspace = acquirePack(T, pack_elems, workspace.cache_bytes) orelse return false;
        defer heap_workspace.deinit();
        loop_fn(m, n, k, a, lda, b, ldb, c, ldc, heap_workspace.data.ptr);
    }
    return true;
}

inline fn loadF32x4(ptr: [*]const f32) Vec4f {
    return @as(*align(1) const Vec4f, @ptrCast(ptr)).*;
}

inline fn storeF32x4(ptr: [*]f32, value: Vec4f) void {
    @as(*align(1) Vec4f, @ptrCast(ptr)).* = value;
}

inline fn loadF64x2(ptr: [*]const f64) Vec2d {
    return @as(*align(1) const Vec2d, @ptrCast(ptr)).*;
}

inline fn storeF64x2(ptr: [*]f64, value: Vec2d) void {
    @as(*align(1) Vec2d, @ptrCast(ptr)).* = value;
}

inline fn trn1F32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
}

inline fn trn2F32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
}

inline fn combineLowF32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
}

inline fn combineHighF32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
}

inline fn trn1F64(a: Vec2d, b: Vec2d) Vec2d {
    return @shuffle(f64, a, b, @Vector(2, i32){ 0, ~@as(i32, 0) });
}

inline fn trn2F64(a: Vec2d, b: Vec2d) Vec2d {
    return @shuffle(f64, a, b, @Vector(2, i32){ 1, ~@as(i32, 1) });
}

noinline fn amxSgemmMblocksN16(comptime m_blocks: usize, a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        amxLdy(ptrRowFlags(b_pack + (p + 0) * 16, 0, 0));
        const init = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + (p + 0) * lda, 0, 0));
            if (init) {
                amxFma32(amxFma32RowOperand(block, true));
            } else {
                amxFma32(amxFma32RowOperand(block, false));
            }
        }

        amxLdy(ptrRowFlags(b_pack + (p + 1) * 16, 0, 0));
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + (p + 1) * lda, 0, 0));
            amxFma32(amxFma32RowOperand(block, false));
        }
    }
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 16, 0, 0));
        const skip_z = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + p * lda, 0, 0));
            if (skip_z) {
                amxFma32(amxFma32RowOperand(block, true));
            } else {
                amxFma32(amxFma32RowOperand(block, false));
            }
        }
    }

    var j: usize = 0;
    while (j < 16) : (j += 1) {
        inline for (0..m_blocks) |block| {
            amxStz(ptrRowFlags(c + block * 16 + j * ldc, j * 4 + block, 0));
        }
    }
}

inline fn amxSgemmMblocksN16Dispatch(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize, m_blocks: usize) void {
    switch (m_blocks) {
        1 => amxSgemmMblocksN16(1, a, b_pack, c, k, lda, ldc),
        2 => amxSgemmMblocksN16(2, a, b_pack, c, k, lda, ldc),
        3 => amxSgemmMblocksN16(3, a, b_pack, c, k, lda, ldc),
        4 => amxSgemmMblocksN16(4, a, b_pack, c, k, lda, ldc),
        else => unreachable,
    }
}

noinline fn amxSgemmMblocksN16BStride32(comptime m_blocks: usize, a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        amxLdy(ptrRowFlags(b_pack + (p + 0) * 32, 0, 0));
        const init = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + (p + 0) * lda, 0, 0));
            if (init) {
                amxFma32(amxFma32RowOperand(block, true));
            } else {
                amxFma32(amxFma32RowOperand(block, false));
            }
        }

        amxLdy(ptrRowFlags(b_pack + (p + 1) * 32, 0, 0));
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + (p + 1) * lda, 0, 0));
            amxFma32(amxFma32RowOperand(block, false));
        }
    }
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 32, 0, 0));
        const skip_z = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + p * lda, 0, 0));
            if (skip_z) {
                amxFma32(amxFma32RowOperand(block, true));
            } else {
                amxMatfp(amxMatfp32RowOperand(block));
            }
        }
    }

    var j: usize = 0;
    while (j < 16) : (j += 1) {
        inline for (0..m_blocks) |block| {
            amxStz(ptrRowFlags(c + block * 16 + j * ldc, j * 4 + block, 0));
        }
    }
}

inline fn amxSgemmMblocksN16BStride32Dispatch(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize, m_blocks: usize) void {
    switch (m_blocks) {
        1 => amxSgemmMblocksN16BStride32(1, a, b_pack, c, k, lda, ldc),
        2 => amxSgemmMblocksN16BStride32(2, a, b_pack, c, k, lda, ldc),
        3 => amxSgemmMblocksN16BStride32(3, a, b_pack, c, k, lda, ldc),
        4 => amxSgemmMblocksN16BStride32(4, a, b_pack, c, k, lda, ldc),
        else => unreachable,
    }
}

fn amxSgemvMblocksN16(comptime m_blocks: usize, a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 16, 0, 0));
        const skip_z = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 16 + p * lda, 0, 0));
            if (skip_z) {
                amxFma32(amxFma32RowOperand(block, true));
            } else {
                amxMatfp(amxMatfp32RowOperand(block));
            }
        }
    }

    inline for (0..m_blocks) |block| {
        amxStz(ptrRowFlags(c + block * 16, block, 0));
    }
}

inline fn amxSgemvMblocksN16Dispatch(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, m_blocks: usize) void {
    switch (m_blocks) {
        1 => amxSgemvMblocksN16(1, a, b_pack, c, k, lda),
        2 => amxSgemvMblocksN16(2, a, b_pack, c, k, lda),
        3 => amxSgemvMblocksN16(3, a, b_pack, c, k, lda),
        4 => amxSgemvMblocksN16(4, a, b_pack, c, k, lda),
        else => unreachable,
    }
}

noinline fn amxSgemmM32N32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        amxLdx(ptrRowFlags(a + (p + 0) * lda + 0, 0, 0));
        amxLdx(ptrRowFlags(a + (p + 0) * lda + 16, 1, 0));
        amxLdx(ptrRowFlags(a + (p + 1) * lda + 0, 2, 0));
        amxLdx(ptrRowFlags(a + (p + 1) * lda + 16, 3, 0));

        amxLdy(ptrRowFlags(b_pack + (p + 0) * 32 + 0, 0, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 0) * 32 + 16, 1, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 1) * 32 + 0, 2, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 1) * 32 + 16, 3, 0));

        const init = p == 0;
        amxFma32(amxFma32XyRowOperand(0, 0, 0, init));
        amxFma32(amxFma32XyRowOperand(1, 0, 1, init));
        amxFma32(amxFma32XyRowOperand(0, 1, 2, init));
        amxFma32(amxFma32XyRowOperand(1, 1, 3, init));

        amxFma32(amxFma32XyRowOperand(2, 2, 0, false));
        amxFma32(amxFma32XyRowOperand(3, 2, 1, false));
        amxFma32(amxFma32XyRowOperand(2, 3, 2, false));
        amxFma32(amxFma32XyRowOperand(3, 3, 3, false));
    }

    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 32 + 0, 0, 0));
        amxLdy(ptrRowFlags(b_pack + p * 32 + 16, 1, 0));
        amxLdx(ptrRowFlags(a + p * lda + 0, 0, 0));
        amxLdx(ptrRowFlags(a + p * lda + 16, 1, 0));
        const init = p == 0;
        amxFma32(amxFma32XyRowOperand(0, 0, 0, init));
        amxFma32(amxFma32XyRowOperand(1, 0, 1, init));
        amxFma32(amxFma32XyRowOperand(0, 1, 2, init));
        amxFma32(amxFma32XyRowOperand(1, 1, 3, init));
    }

    var j: usize = 0;
    while (j < 16) : (j += 1) {
        amxStz(ptrRowFlags(c + j * ldc + 0, j * 4 + 0, 0));
        amxStz(ptrRowFlags(c + j * ldc + 16, j * 4 + 1, 0));
        amxStz(ptrRowFlags(c + (j + 16) * ldc + 0, j * 4 + 2, 0));
        amxStz(ptrRowFlags(c + (j + 16) * ldc + 16, j * 4 + 3, 0));
    }
}

fn packBF32_4x4(comptime panel_cols: usize, b_pack: [*]f32, b: [*]const f32, j_start: usize, k: usize, ldb: usize) void {
    if (comptime panel_cols == 0 or panel_cols % 4 != 0) @compileError("f32 AMX B panel columns must be a nonzero multiple of 4");

    var p: usize = 0;
    while (p + 4 <= k) : (p += 4) {
        inline for (0..(panel_cols / 4)) |block| {
            const col = block * 4;
            const v0 = loadF32x4(b + p + (j_start + col + 0) * ldb);
            const v1 = loadF32x4(b + p + (j_start + col + 1) * ldb);
            const v2 = loadF32x4(b + p + (j_start + col + 2) * ldb);
            const v3 = loadF32x4(b + p + (j_start + col + 3) * ldb);
            const t0 = trn1F32(v0, v1);
            const t1 = trn2F32(v0, v1);
            const t2 = trn1F32(v2, v3);
            const t3 = trn2F32(v2, v3);
            storeF32x4(b_pack + (p + 0) * panel_cols + col, combineLowF32(t0, t2));
            storeF32x4(b_pack + (p + 1) * panel_cols + col, combineLowF32(t1, t3));
            storeF32x4(b_pack + (p + 2) * panel_cols + col, combineHighF32(t0, t2));
            storeF32x4(b_pack + (p + 3) * panel_cols + col, combineHighF32(t1, t3));
        }
    }
    while (p < k) : (p += 1) {
        const dst = b_pack + p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            dst[col] = b[p + (j_start + col) * ldb];
        }
    }
}

fn amxSgemmN16Loop(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, b_pack: [*]f32) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 16) {
        packBF32_4x4(16, b_pack, b, j, k, ldb);
        var i: usize = 0;
        while (i + 64 <= m) : (i += 64) {
            amxSgemmMblocksN16(4, a + i, b_pack, c + i + j * ldc, k, lda, ldc);
        }
        if (i < m) {
            const m_blocks = (m - i) / 16;
            amxSgemmMblocksN16Dispatch(a + i, b_pack, c + i + j * ldc, k, lda, ldc, m_blocks);
        }
    }
}

fn amxSgemmN32Loop(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, b_pack: [*]f32) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 32) {
        packBF32_4x4(32, b_pack, b, j, k, ldb);
        var i: usize = 0;
        if ((k & 1) != 0) {
            while (i + 64 <= m) : (i += 64) {
                amxSgemmMblocksN16BStride32(4, a + i, b_pack, c + i + j * ldc, k, lda, ldc);
                amxSgemmMblocksN16BStride32(4, a + i, b_pack + 16, c + i + (j + 16) * ldc, k, lda, ldc);
            }
            if (i < m) {
                const m_blocks = (m - i) / 16;
                amxSgemmMblocksN16BStride32Dispatch(a + i, b_pack, c + i + j * ldc, k, lda, ldc, m_blocks);
                amxSgemmMblocksN16BStride32Dispatch(a + i, b_pack + 16, c + i + (j + 16) * ldc, k, lda, ldc, m_blocks);
            }
        } else {
            while (i < m) : (i += 32) {
                amxSgemmM32N32(a + i, b_pack, c + i + j * ldc, k, lda, ldc);
            }
        }
    }
}

fn amxSgemmN16WithPack(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return amxGemmWithPack(f32, amxSgemmN16Loop, m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace);
}

fn amxSgemmN32WithPack(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return amxGemmWithPack(f32, amxSgemmN32Loop, m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace);
}

pub fn sgemmN16(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int, workspace: gemm_task.PackWorkspacePlan) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 15) != 0 or (n_ & 15) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    const pack_elems = k * 16;
    return if (amxSgemmN16WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace)) 1 else 0;
}

pub export fn zynum_blas_amx_sgemm_nn_f32_n16(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int) callconv(.c) c_int {
    return sgemmN16(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, compat_pack_plan);
}

pub fn sgemmN32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int, workspace: gemm_task.PackWorkspacePlan) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 31) != 0 or (n_ & 31) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    const pack_elems = k * 32;
    return if (amxSgemmN32WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace)) 1 else 0;
}

pub export fn zynum_blas_amx_sgemm_nn_f32_n32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int) callconv(.c) c_int {
    return sgemmN32(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, compat_pack_plan);
}

pub export fn zynum_blas_amx_sgemm_nn_f32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int) callconv(.c) c_int {
    return zynum_blas_amx_sgemm_nn_f32_n16(m_, n_, k_, a, lda_, b, ldb_, c, ldc_);
}

pub fn sgemvN16PackedB(m_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b_pack: [*]const f32, c: [*]f32) c_int {
    if (m_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 15) != 0) return 0;

    const m: usize = @intCast(m_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);

    amxSet();
    defer amxClr();

    var i: usize = 0;
    while (i < m) : (i += 64) {
        var m_blocks = (m - i) / 16;
        if (m_blocks > 4) m_blocks = 4;
        amxSgemvMblocksN16Dispatch(a + i, b_pack, c + i, k, lda, m_blocks);
    }
    return 1;
}

fn amxDgemmMblocksN8(comptime m_blocks: usize, a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 8, 0, 0));
        const skip_z = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 8 + p * lda, 0, 0));
            amxFma64(amxFma64RowOperand(block, skip_z));
        }
    }

    var j: usize = 0;
    while (j < 8) : (j += 1) {
        inline for (0..m_blocks) |block| {
            amxStz(ptrRowFlags(c + block * 8 + j * ldc, j * 8 + block, 0));
        }
    }
}

inline fn amxDgemmMblocksN8Dispatch(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, ldc: usize, m_blocks: usize) void {
    switch (m_blocks) {
        1 => amxDgemmMblocksN8(1, a, b_pack, c, k, lda, ldc),
        2 => amxDgemmMblocksN8(2, a, b_pack, c, k, lda, ldc),
        3 => amxDgemmMblocksN8(3, a, b_pack, c, k, lda, ldc),
        4 => amxDgemmMblocksN8(4, a, b_pack, c, k, lda, ldc),
        5 => amxDgemmMblocksN8(5, a, b_pack, c, k, lda, ldc),
        6 => amxDgemmMblocksN8(6, a, b_pack, c, k, lda, ldc),
        7 => amxDgemmMblocksN8(7, a, b_pack, c, k, lda, ldc),
        8 => amxDgemmMblocksN8(8, a, b_pack, c, k, lda, ldc),
        else => unreachable,
    }
}

fn amxDgemvMblocksN8(comptime m_blocks: usize, a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 8, 0, 0));
        const skip_z = p == 0;
        inline for (0..m_blocks) |block| {
            amxLdx(ptrRowFlags(a + block * 8 + p * lda, 0, 0));
            amxFma64(amxFma64RowOperand(block, skip_z));
        }
    }

    inline for (0..m_blocks) |block| {
        amxStz(ptrRowFlags(c + block * 8, block, 0));
    }
}

inline fn amxDgemvMblocksN8Dispatch(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, m_blocks: usize) void {
    switch (m_blocks) {
        1 => amxDgemvMblocksN8(1, a, b_pack, c, k, lda),
        2 => amxDgemvMblocksN8(2, a, b_pack, c, k, lda),
        3 => amxDgemvMblocksN8(3, a, b_pack, c, k, lda),
        4 => amxDgemvMblocksN8(4, a, b_pack, c, k, lda),
        5 => amxDgemvMblocksN8(5, a, b_pack, c, k, lda),
        6 => amxDgemvMblocksN8(6, a, b_pack, c, k, lda),
        7 => amxDgemvMblocksN8(7, a, b_pack, c, k, lda),
        8 => amxDgemvMblocksN8(8, a, b_pack, c, k, lda),
        else => unreachable,
    }
}

noinline fn amxDgemmM32N16(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 16 + 0, 0, 0));
        amxLdy(ptrRowFlags(b_pack + p * 16 + 8, 1, 0));
        amxLdx(ptrRowFlags(a + p * lda + 0, 0, 0));
        amxLdx(ptrRowFlags(a + p * lda + 8, 1, 0));
        amxLdx(ptrRowFlags(a + p * lda + 16, 2, 0));
        amxLdx(ptrRowFlags(a + p * lda + 24, 3, 0));
        const skip_z = p == 0;

        amxFma64(amxFma64XyOperand(0, 0, 0, skip_z));
        amxFma64(amxFma64XyOperand(1, 0, 1, skip_z));
        amxFma64(amxFma64XyOperand(2, 0, 2, skip_z));
        amxFma64(amxFma64XyOperand(3, 0, 3, skip_z));
        amxFma64(amxFma64XyOperand(0, 1, 4, skip_z));
        amxFma64(amxFma64XyOperand(1, 1, 5, skip_z));
        amxFma64(amxFma64XyOperand(2, 1, 6, skip_z));
        amxFma64(amxFma64XyOperand(3, 1, 7, skip_z));
    }

    var j: usize = 0;
    while (j < 8) : (j += 1) {
        amxStz(ptrRowFlags(c + j * ldc + 0, j * 8 + 0, 0));
        amxStz(ptrRowFlags(c + j * ldc + 8, j * 8 + 1, 0));
        amxStz(ptrRowFlags(c + j * ldc + 16, j * 8 + 2, 0));
        amxStz(ptrRowFlags(c + j * ldc + 24, j * 8 + 3, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 0, j * 8 + 4, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 8, j * 8 + 5, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 16, j * 8 + 6, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 24, j * 8 + 7, 0));
    }
}

fn packBF64_2x2(comptime panel_cols: usize, b_pack: [*]f64, b: [*]const f64, j_start: usize, k: usize, ldb: usize) void {
    if (comptime panel_cols == 0 or panel_cols % 2 != 0) @compileError("f64 AMX B panel columns must be a nonzero multiple of 2");

    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        inline for (0..(panel_cols / 2)) |block| {
            const col = block * 2;
            const v0 = loadF64x2(b + p + (j_start + col + 0) * ldb);
            const v1 = loadF64x2(b + p + (j_start + col + 1) * ldb);
            storeF64x2(b_pack + (p + 0) * panel_cols + col, trn1F64(v0, v1));
            storeF64x2(b_pack + (p + 1) * panel_cols + col, trn2F64(v0, v1));
        }
    }
    while (p < k) : (p += 1) {
        const dst = b_pack + p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            dst[col] = b[p + (j_start + col) * ldb];
        }
    }
}

fn amxDgemmN16Loop(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, b_pack: [*]f64) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 16) {
        packBF64_2x2(16, b_pack, b, j, k, ldb);
        var i: usize = 0;
        while (i < m) : (i += 32) {
            amxDgemmM32N16(a + i, b_pack, c + i + j * ldc, k, lda, ldc);
        }
    }
}

fn amxDgemmN8Loop(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, b_pack: [*]f64) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 8) {
        var p: usize = 0;
        while (p < k) : (p += 1) {
            const dst = b_pack + p * 8;
            var col: usize = 0;
            while (col < 8) : (col += 1) {
                dst[col] = b[p + (j + col) * ldb];
            }
        }
        var i: usize = 0;
        while (i < m) : (i += 64) {
            var m_blocks = (m - i) / 8;
            if (m_blocks > 8) m_blocks = 8;
            amxDgemmMblocksN8Dispatch(a + i, b_pack, c + i + j * ldc, k, lda, ldc, m_blocks);
        }
    }
}

pub fn dgemmN8PackedB(m_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b_pack: [*]const f64, c: [*]f64, ldc_: c_int) c_int {
    if (m_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 7) != 0) return 0;

    const m: usize = @intCast(m_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldc: usize = @intCast(ldc_);

    amxSet();
    defer amxClr();

    var i: usize = 0;
    while (i < m) : (i += 64) {
        var m_blocks = (m - i) / 8;
        if (m_blocks > 8) m_blocks = 8;
        amxDgemmMblocksN8Dispatch(a + i, b_pack, c + i, k, lda, ldc, m_blocks);
    }
    return 1;
}

pub fn dgemvN8PackedB(m_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b_pack: [*]const f64, c: [*]f64) c_int {
    if (m_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 7) != 0) return 0;

    const m: usize = @intCast(m_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);

    amxSet();
    defer amxClr();

    var i: usize = 0;
    while (i < m) : (i += 64) {
        var m_blocks = (m - i) / 8;
        if (m_blocks > 8) m_blocks = 8;
        amxDgemvMblocksN8Dispatch(a + i, b_pack, c + i, k, lda, m_blocks);
    }
    return 1;
}

fn packDgemvTransX8(k: usize, alpha: f64, x: [*]const f64, b_pack: [*]f64) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        const value = alpha * x[p];
        const dst = b_pack + p * 8;
        dst[0] = value;
        dst[1] = value;
        dst[2] = value;
        dst[3] = value;
        dst[4] = value;
        dst[5] = value;
        dst[6] = value;
        dst[7] = value;
    }
}

fn packDgemvTransA8(k: usize, a: [*]const f64, lda: usize, j: usize, a_pack: [*]f64) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        const row = a + p;
        const dst = a_pack + p * 8;
        dst[0] = row[(j + 0) * lda];
        dst[1] = row[(j + 1) * lda];
        dst[2] = row[(j + 2) * lda];
        dst[3] = row[(j + 3) * lda];
        dst[4] = row[(j + 4) * lda];
        dst[5] = row[(j + 5) * lda];
        dst[6] = row[(j + 6) * lda];
        dst[7] = row[(j + 7) * lda];
    }
}

pub fn dgemvTransN8(m_: c_int, n_: c_int, alpha: f64, a: [*]const f64, lda_: c_int, x: [*]const f64, y: [*]f64) c_int {
    if (m_ <= 0 or n_ <= 0) return 1;
    if ((n_ & 7) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const lda: usize = @intCast(lda_);
    if (lda == 0) return 0;

    const panel_elems = m * 8;
    const pack_elems = panel_elems * 2;
    const heap_workspace = acquirePack(f64, pack_elems, compat_pack_plan.cache_bytes) orelse return 0;
    defer heap_workspace.deinit();

    const a_pack = heap_workspace.data.ptr;
    const b_pack = a_pack + panel_elems;
    var c_block: [8]f64 = undefined;

    packDgemvTransX8(m, alpha, x, b_pack);

    amxSet();
    defer amxClr();

    var j: usize = 0;
    while (j < n) : (j += 8) {
        packDgemvTransA8(m, a, lda, j, a_pack);
        amxDgemvMblocksN8(1, a_pack, b_pack, &c_block, m, 8);
        y[j + 0] += c_block[0];
        y[j + 1] += c_block[1];
        y[j + 2] += c_block[2];
        y[j + 3] += c_block[3];
        y[j + 4] += c_block[4];
        y[j + 5] += c_block[5];
        y[j + 6] += c_block[6];
        y[j + 7] += c_block[7];
    }
    return 1;
}

fn amxDgemmN16WithPack(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return amxGemmWithPack(f64, amxDgemmN16Loop, m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace);
}

fn amxDgemmN8WithPack(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return amxGemmWithPack(f64, amxDgemmN8Loop, m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace);
}

pub fn dgemmN16(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int, workspace: gemm_task.PackWorkspacePlan) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 31) != 0 or (n_ & 15) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    const pack_elems = k * 16;
    return if (amxDgemmN16WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace)) 1 else 0;
}

pub export fn zynum_blas_amx_dgemm_nn_f64_n16(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    return dgemmN16(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, compat_pack_plan);
}

pub fn dgemmN8(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int, workspace: gemm_task.PackWorkspacePlan) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 7) != 0 or (n_ & 7) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    const pack_elems = k * 8;
    return if (amxDgemmN8WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace)) 1 else 0;
}

pub export fn zynum_blas_amx_dgemm_nn_f64_n8(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    return dgemmN8(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, compat_pack_plan);
}

pub export fn zynum_blas_amx_dgemm_nn_f64(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    return zynum_blas_amx_dgemm_nn_f64_n8(m_, n_, k_, a, lda_, b, ldb_, c, ldc_);
}

noinline fn amxDgemmM16N32(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 32 + 0, 0, 0));
        amxLdy(ptrRowFlags(b_pack + p * 32 + 8, 1, 0));
        amxLdy(ptrRowFlags(b_pack + p * 32 + 16, 2, 0));
        amxLdy(ptrRowFlags(b_pack + p * 32 + 24, 3, 0));
        amxLdx(ptrRowFlags(a + p * lda + 0, 0, 0));
        amxLdx(ptrRowFlags(a + p * lda + 8, 1, 0));
        const skip_z = p == 0;

        amxFma64(amxFma64XyOperand(0, 0, 0, skip_z));
        amxFma64(amxFma64XyOperand(1, 0, 1, skip_z));
        amxFma64(amxFma64XyOperand(0, 1, 2, skip_z));
        amxFma64(amxFma64XyOperand(1, 1, 3, skip_z));
        amxFma64(amxFma64XyOperand(0, 2, 4, skip_z));
        amxFma64(amxFma64XyOperand(1, 2, 5, skip_z));
        amxFma64(amxFma64XyOperand(0, 3, 6, skip_z));
        amxFma64(amxFma64XyOperand(1, 3, 7, skip_z));
    }

    var j: usize = 0;
    while (j < 8) : (j += 1) {
        amxStz(ptrRowFlags(c + j * ldc + 0, j * 8 + 0, 0));
        amxStz(ptrRowFlags(c + j * ldc + 8, j * 8 + 1, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 0, j * 8 + 2, 0));
        amxStz(ptrRowFlags(c + (j + 8) * ldc + 8, j * 8 + 3, 0));
        amxStz(ptrRowFlags(c + (j + 16) * ldc + 0, j * 8 + 4, 0));
        amxStz(ptrRowFlags(c + (j + 16) * ldc + 8, j * 8 + 5, 0));
        amxStz(ptrRowFlags(c + (j + 24) * ldc + 0, j * 8 + 6, 0));
        amxStz(ptrRowFlags(c + (j + 24) * ldc + 8, j * 8 + 7, 0));
    }
}

fn amxDgemmN32Loop(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, b_pack: [*]f64) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 32) {
        packBF64_2x2(32, b_pack, b, j, k, ldb);
        var i: usize = 0;
        while (i < m) : (i += 16) {
            amxDgemmM16N32(a + i, b_pack, c + i + j * ldc, k, lda, ldc);
        }
    }
}

fn amxDgemmN32WithPack(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, pack_elems: usize, workspace: gemm_task.PackWorkspacePlan) bool {
    return amxGemmWithPack(f64, amxDgemmN32Loop, m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace);
}

pub fn dgemmN32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int, workspace: gemm_task.PackWorkspacePlan) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    if ((m & 15) != 0 or (n & 31) != 0) return 0;

    const pack_elems = k * 32;
    return if (amxDgemmN32WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems, workspace)) 1 else 0;
}

pub export fn zynum_blas_amx_dgemm_nn_f64_n32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    return dgemmN32(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, compat_pack_plan);
}

// Legacy exported symbol kept as an ABI alias; shape policy lives in tuning.zig.
pub export fn zynum_blas_amx_dgemm_nn_f64_n32_square(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    return zynum_blas_amx_dgemm_nn_f64_n32(m_, n_, k_, a, lda_, b, ldb_, c, ldc_);
}
