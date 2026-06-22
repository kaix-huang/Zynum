// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const Vec4f = @Vector(4, f32);
const Vec2d = @Vector(2, f64);
const max_stack_pack_elems: usize = 32768;

inline fn amxNopOpImm5(comptime op: usize, comptime imm5: usize) void {
    asm volatile ("nop\nnop\nnop\n.word (0x201000 + (%[op] << 5) + %[imm5])"
        :
        : [op] "i" (op),
          [imm5] "i" (imm5),
        : .{ .memory = true });
}

inline fn amxOpGpr(comptime op: usize, gpr: usize) void {
    asm volatile (".word (0x201000 + (%[op] << 5) + 0%[gpr] - ((0%[gpr] >> 4) * 6))"
        :
        : [op] "i" (op),
          [gpr] "r" (gpr),
        : .{ .memory = true });
}

inline fn amxLdx(gpr: usize) void {
    amxOpGpr(0, gpr);
}

inline fn amxLdy(gpr: usize) void {
    amxOpGpr(1, gpr);
}

inline fn amxStz(gpr: usize) void {
    amxOpGpr(5, gpr);
}

inline fn amxFma64(gpr: usize) void {
    amxOpGpr(10, gpr);
}

inline fn amxFma32(gpr: usize) void {
    amxOpGpr(12, gpr);
}

inline fn amxMatfp(gpr: usize) void {
    amxOpGpr(21, gpr);
}

inline fn amxSet() void {
    amxNopOpImm5(17, 0);
}

inline fn amxClr() void {
    amxNopOpImm5(17, 1);
}

inline fn ptrRowFlags(ptr: anytype, row: usize, flags: usize) usize {
    return @intFromPtr(ptr) + ((row + flags * 64) << 56);
}

inline fn amxFma32Operand(skip_z: bool) usize {
    return if (skip_z) (1 << 27) else 0;
}

inline fn amxFma64Operand(skip_z: bool) usize {
    return if (skip_z) (1 << 27) else 0;
}

inline fn amxFma32RowOperand(row: usize, skip_z: bool) usize {
    return (row << 20) | amxFma32Operand(skip_z);
}

inline fn amxFma64RowOperand(row: usize, skip_z: bool) usize {
    return (row << 20) | amxFma64Operand(skip_z);
}

inline fn amxFma64XyOperand(xrow: usize, yrow: usize, zrow: usize, skip_z: bool) usize {
    return (xrow << 16) | (yrow << 6) | (zrow << 20) | amxFma64Operand(skip_z);
}

inline fn amxMatfp32RowOperand(row: usize) usize {
    return (4 << 42) | (row << 20);
}

inline fn amxFma32XyRowOperand(xrow: usize, yrow: usize, zrow: usize, skip_z: bool) usize {
    return (xrow << 16) | (yrow << 6) | (zrow << 20) | amxFma32Operand(skip_z);
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

noinline fn amxSgemmMblocksN16(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize, m_blocks: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 16, 0, 0));
        const skip_z = p == 0;
        var block: usize = 0;
        while (block < m_blocks) : (block += 1) {
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
        var block: usize = 0;
        while (block < m_blocks) : (block += 1) {
            amxStz(ptrRowFlags(c + block * 16 + j * ldc, j * 4 + block, 0));
        }
    }
}

noinline fn amxSgemmM32N32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, k: usize, lda: usize, ldc: usize) void {
    var p: usize = 0;
    while (p + 4 <= k) : (p += 4) {
        amxLdx(ptrRowFlags(a + (p + 0) * lda + 0, 0, 0));
        amxLdx(ptrRowFlags(a + (p + 0) * lda + 16, 1, 0));
        amxLdx(ptrRowFlags(a + (p + 1) * lda + 0, 2, 0));
        amxLdx(ptrRowFlags(a + (p + 1) * lda + 16, 3, 0));
        amxLdx(ptrRowFlags(a + (p + 2) * lda + 0, 4, 0));
        amxLdx(ptrRowFlags(a + (p + 2) * lda + 16, 5, 0));
        amxLdx(ptrRowFlags(a + (p + 3) * lda + 0, 6, 0));
        amxLdx(ptrRowFlags(a + (p + 3) * lda + 16, 7, 0));

        amxLdy(ptrRowFlags(b_pack + (p + 0) * 32 + 0, 0, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 0) * 32 + 16, 1, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 1) * 32 + 0, 2, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 1) * 32 + 16, 3, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 2) * 32 + 0, 4, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 2) * 32 + 16, 5, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 3) * 32 + 0, 6, 0));
        amxLdy(ptrRowFlags(b_pack + (p + 3) * 32 + 16, 7, 0));

        const init = p == 0;
        amxFma32(amxFma32XyRowOperand(0, 0, 0, init));
        amxFma32(amxFma32XyRowOperand(1, 0, 1, init));
        amxFma32(amxFma32XyRowOperand(0, 1, 2, init));
        amxFma32(amxFma32XyRowOperand(1, 1, 3, init));

        amxFma32(amxFma32XyRowOperand(2, 2, 0, false));
        amxFma32(amxFma32XyRowOperand(3, 2, 1, false));
        amxFma32(amxFma32XyRowOperand(2, 3, 2, false));
        amxFma32(amxFma32XyRowOperand(3, 3, 3, false));

        amxFma32(amxFma32XyRowOperand(4, 4, 0, false));
        amxFma32(amxFma32XyRowOperand(5, 4, 1, false));
        amxFma32(amxFma32XyRowOperand(4, 5, 2, false));
        amxFma32(amxFma32XyRowOperand(5, 5, 3, false));

        amxFma32(amxFma32XyRowOperand(6, 6, 0, false));
        amxFma32(amxFma32XyRowOperand(7, 6, 1, false));
        amxFma32(amxFma32XyRowOperand(6, 7, 2, false));
        amxFma32(amxFma32XyRowOperand(7, 7, 3, false));
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

fn packB16F32_4x4(b_pack: [*]f32, b: [*]const f32, j_start: usize, k: usize, ldb: usize) void {
    var p: usize = 0;
    while (p + 4 <= k) : (p += 4) {
        var col: usize = 0;
        while (col < 16) : (col += 4) {
            const v0 = loadF32x4(b + p + (j_start + col + 0) * ldb);
            const v1 = loadF32x4(b + p + (j_start + col + 1) * ldb);
            const v2 = loadF32x4(b + p + (j_start + col + 2) * ldb);
            const v3 = loadF32x4(b + p + (j_start + col + 3) * ldb);
            const t0 = trn1F32(v0, v1);
            const t1 = trn2F32(v0, v1);
            const t2 = trn1F32(v2, v3);
            const t3 = trn2F32(v2, v3);
            storeF32x4(b_pack + (p + 0) * 16 + col, combineLowF32(t0, t2));
            storeF32x4(b_pack + (p + 1) * 16 + col, combineLowF32(t1, t3));
            storeF32x4(b_pack + (p + 2) * 16 + col, combineHighF32(t0, t2));
            storeF32x4(b_pack + (p + 3) * 16 + col, combineHighF32(t1, t3));
        }
    }
    while (p < k) : (p += 1) {
        const dst = b_pack + p * 16;
        var col: usize = 0;
        while (col < 16) : (col += 1) {
            dst[col] = b[p + (j_start + col) * ldb];
        }
    }
}

fn packB32F32_4x4(b_pack: [*]f32, b: [*]const f32, j_start: usize, k: usize, ldb: usize) void {
    var p: usize = 0;
    while (p + 4 <= k) : (p += 4) {
        var col: usize = 0;
        while (col < 32) : (col += 4) {
            const v0 = loadF32x4(b + p + (j_start + col + 0) * ldb);
            const v1 = loadF32x4(b + p + (j_start + col + 1) * ldb);
            const v2 = loadF32x4(b + p + (j_start + col + 2) * ldb);
            const v3 = loadF32x4(b + p + (j_start + col + 3) * ldb);
            const t0 = trn1F32(v0, v1);
            const t1 = trn2F32(v0, v1);
            const t2 = trn1F32(v2, v3);
            const t3 = trn2F32(v2, v3);
            storeF32x4(b_pack + (p + 0) * 32 + col, combineLowF32(t0, t2));
            storeF32x4(b_pack + (p + 1) * 32 + col, combineLowF32(t1, t3));
            storeF32x4(b_pack + (p + 2) * 32 + col, combineHighF32(t0, t2));
            storeF32x4(b_pack + (p + 3) * 32 + col, combineHighF32(t1, t3));
        }
    }
    while (p < k) : (p += 1) {
        const dst = b_pack + p * 32;
        var col: usize = 0;
        while (col < 32) : (col += 1) {
            dst[col] = b[p + (j_start + col) * ldb];
        }
    }
}

fn amxSgemmN16Loop(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, b_pack: [*]f32) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 16) {
        packB16F32_4x4(b_pack, b, j, k, ldb);
        var i: usize = 0;
        while (i < m) : (i += 64) {
            var m_blocks = (m - i) / 16;
            if (m_blocks > 4) m_blocks = 4;
            amxSgemmMblocksN16(a + i, b_pack, c + i + j * ldc, k, lda, ldc, m_blocks);
        }
    }
}

fn amxSgemmN32Loop(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, b_pack: [*]f32) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 32) {
        packB32F32_4x4(b_pack, b, j, k, ldb);
        var i: usize = 0;
        while (i < m) : (i += 32) {
            amxSgemmM32N32(a + i, b_pack, c + i + j * ldc, k, lda, ldc);
        }
    }
}

noinline fn amxSgemmN16Stack(comptime capacity: usize, pack_elems: usize, m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize) void {
    var stack_pack: [capacity]f32 = undefined;
    amxSgemmN16Loop(m, n, k, a, lda, b, ldb, c, ldc, stack_pack[0..pack_elems].ptr);
}

noinline fn amxSgemmN32Stack(comptime capacity: usize, pack_elems: usize, m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize) void {
    var stack_pack: [capacity]f32 = undefined;
    amxSgemmN32Loop(m, n, k, a, lda, b, ldb, c, ldc, stack_pack[0..pack_elems].ptr);
}

fn amxSgemmN16WithPack(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, pack_elems: usize) bool {
    if (pack_elems <= 2048) {
        amxSgemmN16Stack(2048, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 4096) {
        amxSgemmN16Stack(4096, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 8192) {
        amxSgemmN16Stack(8192, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 16384) {
        amxSgemmN16Stack(16384, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= max_stack_pack_elems) {
        amxSgemmN16Stack(max_stack_pack_elems, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        const b_pack = std.heap.c_allocator.alloc(f32, pack_elems) catch return false;
        defer std.heap.c_allocator.free(b_pack);
        amxSgemmN16Loop(m, n, k, a, lda, b, ldb, c, ldc, b_pack.ptr);
    }
    return true;
}

fn amxSgemmN32WithPack(m: usize, n: usize, k: usize, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, c: [*]f32, ldc: usize, pack_elems: usize) bool {
    if (pack_elems <= 4096) {
        amxSgemmN32Stack(4096, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 8192) {
        amxSgemmN32Stack(8192, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 16384) {
        amxSgemmN32Stack(16384, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= max_stack_pack_elems) {
        amxSgemmN32Stack(max_stack_pack_elems, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        const b_pack = std.heap.c_allocator.alloc(f32, pack_elems) catch return false;
        defer std.heap.c_allocator.free(b_pack);
        amxSgemmN32Loop(m, n, k, a, lda, b, ldb, c, ldc, b_pack.ptr);
    }
    return true;
}

pub export fn zynum_blas_amx_sgemm_nn_f32(m_: c_int, n_: c_int, k_: c_int, a: [*]const f32, lda_: c_int, b: [*]const f32, ldb_: c_int, c: [*]f32, ldc_: c_int) callconv(.c) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 15) != 0 or (n_ & 15) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    const low_k_large_n32 = m >= 512 and n >= 512 and k <= 128;
    const small_square_n32 = m == n and k == n and (m == 192 or m == 256);
    const high_k_chunk_n32 = m == 128 and n == 32 and k >= 4096;
    if ((m & 31) == 0 and (n & 31) == 0 and (low_k_large_n32 or small_square_n32 or high_k_chunk_n32)) {
        const pack_elems = k * 32;
        return if (amxSgemmN32WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems)) 1 else 0;
    }

    const pack_elems = k * 16;
    return if (amxSgemmN16WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems)) 1 else 0;
}

fn amxDgemmMblocksN8(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, k: usize, lda: usize, ldc: usize, m_blocks: usize) void {
    var p: usize = 0;
    while (p < k) : (p += 1) {
        amxLdy(ptrRowFlags(b_pack + p * 8, 0, 0));
        const skip_z = p == 0;
        var block: usize = 0;
        while (block < m_blocks) : (block += 1) {
            amxLdx(ptrRowFlags(a + block * 8 + p * lda, 0, 0));
            amxFma64(amxFma64RowOperand(block, skip_z));
        }
    }

    var j: usize = 0;
    while (j < 8) : (j += 1) {
        var block: usize = 0;
        while (block < m_blocks) : (block += 1) {
            amxStz(ptrRowFlags(c + block * 8 + j * ldc, j * 8 + block, 0));
        }
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

fn packB16F64_2x2(b_pack: [*]f64, b: [*]const f64, j_start: usize, k: usize, ldb: usize) void {
    var p: usize = 0;
    while (p + 2 <= k) : (p += 2) {
        var col: usize = 0;
        while (col < 16) : (col += 2) {
            const v0 = loadF64x2(b + p + (j_start + col + 0) * ldb);
            const v1 = loadF64x2(b + p + (j_start + col + 1) * ldb);
            storeF64x2(b_pack + (p + 0) * 16 + col, trn1F64(v0, v1));
            storeF64x2(b_pack + (p + 1) * 16 + col, trn2F64(v0, v1));
        }
    }
    while (p < k) : (p += 1) {
        const dst = b_pack + p * 16;
        var col: usize = 0;
        while (col < 16) : (col += 1) {
            dst[col] = b[p + (j_start + col) * ldb];
        }
    }
}

fn amxDgemmN16Loop(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, b_pack: [*]f64) void {
    amxSet();
    defer amxClr();
    var j: usize = 0;
    while (j < n) : (j += 16) {
        packB16F64_2x2(b_pack, b, j, k, ldb);
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
            amxDgemmMblocksN8(a + i, b_pack, c + i + j * ldc, k, lda, ldc, m_blocks);
        }
    }
}

noinline fn amxDgemmN16Stack(comptime capacity: usize, pack_elems: usize, m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize) void {
    var stack_pack: [capacity]f64 = undefined;
    amxDgemmN16Loop(m, n, k, a, lda, b, ldb, c, ldc, stack_pack[0..pack_elems].ptr);
}

noinline fn amxDgemmN8Stack(comptime capacity: usize, pack_elems: usize, m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize) void {
    var stack_pack: [capacity]f64 = undefined;
    amxDgemmN8Loop(m, n, k, a, lda, b, ldb, c, ldc, stack_pack[0..pack_elems].ptr);
}

fn amxDgemmN16WithPack(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, pack_elems: usize) bool {
    if (pack_elems <= 2048) {
        amxDgemmN16Stack(2048, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 4096) {
        amxDgemmN16Stack(4096, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 8192) {
        amxDgemmN16Stack(8192, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 16384) {
        amxDgemmN16Stack(16384, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= max_stack_pack_elems) {
        amxDgemmN16Stack(max_stack_pack_elems, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        const b_pack = std.heap.c_allocator.alloc(f64, pack_elems) catch return false;
        defer std.heap.c_allocator.free(b_pack);
        amxDgemmN16Loop(m, n, k, a, lda, b, ldb, c, ldc, b_pack.ptr);
    }
    return true;
}

fn amxDgemmN8WithPack(m: usize, n: usize, k: usize, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, c: [*]f64, ldc: usize, pack_elems: usize) bool {
    if (pack_elems <= 2048) {
        amxDgemmN8Stack(2048, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 4096) {
        amxDgemmN8Stack(4096, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 8192) {
        amxDgemmN8Stack(8192, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= 16384) {
        amxDgemmN8Stack(16384, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else if (pack_elems <= max_stack_pack_elems) {
        amxDgemmN8Stack(max_stack_pack_elems, pack_elems, m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        const b_pack = std.heap.c_allocator.alloc(f64, pack_elems) catch return false;
        defer std.heap.c_allocator.free(b_pack);
        amxDgemmN8Loop(m, n, k, a, lda, b, ldb, c, ldc, b_pack.ptr);
    }
    return true;
}

pub export fn zynum_blas_amx_dgemm_nn_f64(m_: c_int, n_: c_int, k_: c_int, a: [*]const f64, lda_: c_int, b: [*]const f64, ldb_: c_int, c: [*]f64, ldc_: c_int) callconv(.c) c_int {
    if (m_ <= 0 or n_ <= 0 or k_ <= 0) return 1;
    if ((m_ & 7) != 0 or (n_ & 7) != 0) return 0;

    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const lda: usize = @intCast(lda_);
    const ldb: usize = @intCast(ldb_);
    const ldc: usize = @intCast(ldc_);

    if ((m & 31) == 0 and (n & 15) == 0 and
        ((m <= 64 and n >= 512 and k >= 1024) or
            (m == n and n == k and m >= 64 and m <= 384)))
    {
        const pack_elems = k * 16;
        return if (amxDgemmN16WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems)) 1 else 0;
    }

    const pack_elems = k * 8;
    return if (amxDgemmN8WithPack(m, n, k, a, lda, b, ldb, c, ldc, pack_elems)) 1 else 0;
}
