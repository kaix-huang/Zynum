// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 matrix-vector specialized kernels.

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../../../core/shared/scalar.zig");
const amx = @import("matrix_matrix/amx.zig");
const features = @import("features.zig");
const simd_config = @import("simd_config.zig");
const fixed_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const matrix_vector_asm = @import("asm/matrix_vector.zig");

const BlasInt = scalar.BlasInt;
const enable_amx_gemv_n = true;
const enable_amx_gemv_t = true;
const enable_sm_gemv_n = false;
const enable_sme2_gemv_n = true;
const enable_sme2_gemv_t = true;
const enable_sme2_zgemv_n_rows = false;
const enable_fcmla_cgemv_t_m128 = true;
const enable_fcmla_cgemv_n_m128 = true;
const enable_fcmla_cgemv_n_m512_n64_task = true;
const enable_fcmla_zgemv_n_m128 = true;
const enable_fcmla_zgemv_n_m64_n512_rows = false;
const enable_fcmla_zgemv_t_m128 = true;
const enable_fcmla_zgemv_t_m256_n128_task = true;
const enable_fcmla_zgemv_n_m512_n64_task = true;
const enable_fcmla_zgemv_t_m512_n64_task = true;
const enable_asimd_sger = true;
const enable_asimd_dger = true;

const AmxGemvBufferKind = enum { b, c };

fn useFixedSimdComplexBeforeParallel(comptime T: type, m: usize, n: usize) bool {
    if (comptime !features.has_asimd) return false;
    if (T == scalar.ComplexF32) return (m == 128 and n == 128) or n < 128;
    if (T == scalar.ComplexF64) return n < 96;
    return false;
}

fn AmxGemvBuffers(comptime T: type) type {
    return struct {
        threadlocal var b_ptr: ?[*]T = null;
        threadlocal var b_len: usize = 0;
        threadlocal var c_ptr: ?[*]T = null;
        threadlocal var c_len: usize = 0;

        fn get(which: AmxGemvBufferKind, len: usize) ?[]T {
            switch (which) {
                .b => return grow(&b_ptr, &b_len, len),
                .c => return grow(&c_ptr, &c_len, len),
            }
        }

        fn grow(ptr: *?[*]T, capacity: *usize, len: usize) ?[]T {
            if (capacity.* < len) {
                const data = std.heap.c_allocator.alloc(T, len) catch return null;
                if (ptr.*) |old| std.heap.c_allocator.free(old[0..capacity.*]);
                ptr.* = data.ptr;
                capacity.* = len;
            }
            return ptr.*.?[0..len];
        }
    };
}

fn amxGemvBuffer(comptime T: type, which: AmxGemvBufferKind, len: usize) ?[]T {
    return AmxGemvBuffers(T).get(which, len);
}

fn axpbyUnitF32(n: usize, alpha: f32, x: [*]const f32, beta: f32, y: [*]f32) void {
    const V = @Vector(8, f32);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    if (beta == 0) {
        while (i + 8 <= n) : (i += 8) {
            const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
            @as(*align(1) V, @ptrCast(y + i)).* = xv * alpha_v;
        }
        while (i < n) : (i += 1) y[i] = alpha * x[i];
        return;
    }
    while (i + 8 <= n) : (i += 8) {
        const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        const yv: V = @as(*align(1) const V, @ptrCast(y + i)).*;
        @as(*align(1) V, @ptrCast(y + i)).* = @mulAdd(V, yv, beta_v, xv * alpha_v);
    }
    while (i < n) : (i += 1) y[i] = beta * y[i] + alpha * x[i];
}

fn axpbyUnitF64(n: usize, alpha: f64, x: [*]const f64, beta: f64, y: [*]f64) void {
    const V = @Vector(4, f64);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    if (beta == 0) {
        while (i + 4 <= n) : (i += 4) {
            const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
            @as(*align(1) V, @ptrCast(y + i)).* = xv * alpha_v;
        }
        while (i < n) : (i += 1) y[i] = alpha * x[i];
        return;
    }
    while (i + 4 <= n) : (i += 4) {
        const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        const yv: V = @as(*align(1) const V, @ptrCast(y + i)).*;
        @as(*align(1) V, @ptrCast(y + i)).* = @mulAdd(V, yv, beta_v, xv * alpha_v);
    }
    while (i < n) : (i += 1) y[i] = beta * y[i] + alpha * x[i];
}

fn addUnitF64(n: usize, x: [*]const f64, y: [*]f64) void {
    const V = @Vector(4, f64);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        const yv: V = @as(*align(1) const V, @ptrCast(y + i)).*;
        @as(*align(1) V, @ptrCast(y + i)).* = xv + yv;
    }
    while (i < n) : (i += 1) y[i] += x[i];
}

fn scaleUnitF64(n: usize, beta: f64, y: [*]f64) void {
    if (beta == 1) return;
    if (beta == 0) {
        @memset(y[0..n], 0);
        return;
    }
    const V = @Vector(4, f64);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const yv: V = @as(*align(1) const V, @ptrCast(y + i)).*;
        @as(*align(1) V, @ptrCast(y + i)).* = yv * beta_v;
    }
    while (i < n) : (i += 1) y[i] *= beta;
}

fn canUseAmxGemvNoTransF32(m: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enable_amx_gemv_n) return false;
    if (comptime builtin.target.os.tag != .macos) return false;
    if (!features.has_asimd) return false;
    if (m *| n < 128 * 128 or m *| n >= 1536 * 1536) return false;
    if ((m & 15) != 0 or n == 0 or lda <= 0) return false;
    if (!(m == 128 or m == 256 or m == 512)) return false;
    return true;
}

fn gemvNoTransAmxF32(
    m: usize,
    n: usize,
    alpha: f32,
    a: [*]const f32,
    lda: BlasInt,
    x: [*]const f32,
    beta: f32,
    y: [*]f32,
) bool {
    if (!canUseAmxGemvNoTransF32(m, n, lda)) return false;
    const pack_len = n * 16;
    const b = amxGemvBuffer(f32, .b, pack_len) orelse return false;
    const c = amxGemvBuffer(f32, .c, m) orelse return false;
    for (0..n) |i| {
        const value = alpha * x[i];
        @memset(b[i * 16 .. i * 16 + 16], value);
    }
    if (amx.sgemvN16PackedB(@intCast(m), @intCast(n), a, @intCast(lda), b.ptr, c.ptr) == 0) return false;
    axpbyUnitF32(m, 1, c.ptr, beta, y);
    return true;
}

fn gemvNoTransAsimdF32M128(
    m: usize,
    n: usize,
    alpha: f32,
    a: [*]const f32,
    lda: BlasInt,
    x: [*]const f32,
    beta: f32,
    y: [*]f32,
) bool {
    if (comptime !features.has_asimd) return false;
    if (m != 128 or n != 128 or lda < 128) return false;

    const V = @Vector(4, f32);
    const beta_v: V = @splat(beta);
    var row: usize = 0;
    while (row < 128) : (row += 16) {
        var acc0: V = @splat(0);
        var acc1: V = @splat(0);
        var acc2: V = @splat(0);
        var acc3: V = @splat(0);

        var col: usize = 0;
        while (col < 128) : (col += 1) {
            const xv: V = @splat(alpha * x[col]);
            const base = a + @as(usize, @intCast(lda)) * col + row;
            const a0: V = @as(*align(1) const V, @ptrCast(base + 0)).*;
            const a1: V = @as(*align(1) const V, @ptrCast(base + 4)).*;
            const a2: V = @as(*align(1) const V, @ptrCast(base + 8)).*;
            const a3: V = @as(*align(1) const V, @ptrCast(base + 12)).*;
            acc0 = @mulAdd(V, a0, xv, acc0);
            acc1 = @mulAdd(V, a1, xv, acc1);
            acc2 = @mulAdd(V, a2, xv, acc2);
            acc3 = @mulAdd(V, a3, xv, acc3);
        }

        const y0: V = @as(*align(1) const V, @ptrCast(y + row + 0)).*;
        const y1: V = @as(*align(1) const V, @ptrCast(y + row + 4)).*;
        const y2: V = @as(*align(1) const V, @ptrCast(y + row + 8)).*;
        const y3: V = @as(*align(1) const V, @ptrCast(y + row + 12)).*;
        @as(*align(1) V, @ptrCast(y + row + 0)).* = @mulAdd(V, y0, beta_v, acc0);
        @as(*align(1) V, @ptrCast(y + row + 4)).* = @mulAdd(V, y1, beta_v, acc1);
        @as(*align(1) V, @ptrCast(y + row + 8)).* = @mulAdd(V, y2, beta_v, acc2);
        @as(*align(1) V, @ptrCast(y + row + 12)).* = @mulAdd(V, y3, beta_v, acc3);
    }
    return true;
}

fn canUseAmxGemvNoTransF64(m: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enable_amx_gemv_n) return false;
    if (comptime builtin.target.os.tag != .macos) return false;
    if (m *| n < 256 * 256 or m *| n >= 1536 * 1536) return false;
    if (n < 256) return false;
    if (m > 2 * n) return false;
    if ((m & 7) != 0 or n == 0 or lda <= 0) return false;
    return true;
}

fn gemvNoTransPackLenF64(m: usize, n: usize, lda: BlasInt) ?usize {
    if (!canUseAmxGemvNoTransF64(m, n, lda)) return null;
    return n * 8;
}

fn gemvNoTransPackF64(n: usize, alpha: f64, x: [*]const f64, pack: []f64) bool {
    if (pack.len < n * 8) return false;
    for (0..n) |i| {
        const value = alpha * x[i];
        @memset(pack[i * 8 .. i * 8 + 8], value);
    }
    return true;
}

fn gemvNoTransPackedRowsF64(
    row_count: usize,
    n: usize,
    a: [*]const f64,
    lda: BlasInt,
    pack: [*]const f64,
    scratch: [*]f64,
    y: [*]f64,
) bool {
    if ((row_count & 7) != 0 or n == 0 or lda <= 0) return false;
    const ok = amx.dgemvN8PackedB(
        @intCast(row_count),
        @intCast(n),
        a,
        @intCast(lda),
        pack,
        scratch,
    ) != 0;
    if (!ok) return false;
    addUnitF64(row_count, scratch, y);
    return true;
}

fn gemvNoTransAmxF64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    y: [*]f64,
) bool {
    const pack_len = gemvNoTransPackLenF64(m, n, lda) orelse return false;
    const b = amxGemvBuffer(f64, .b, pack_len) orelse return false;
    const c = amxGemvBuffer(f64, .c, m) orelse return false;

    if (!gemvNoTransPackF64(n, alpha, x, b)) return false;
    return gemvNoTransPackedRowsF64(m, n, a, lda, b.ptr, c.ptr, y);
}

fn gemvNoTransAmxF64Full(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    beta: f64,
    y: [*]f64,
) bool {
    const pack_len = gemvNoTransPackLenF64(m, n, lda) orelse return false;
    const b = amxGemvBuffer(f64, .b, pack_len) orelse return false;
    const c = amxGemvBuffer(f64, .c, m) orelse return false;

    if (!gemvNoTransPackF64(n, alpha, x, b)) return false;
    if (amx.dgemvN8PackedB(@intCast(m), @intCast(n), a, @intCast(lda), b.ptr, c.ptr) == 0) return false;
    axpbyUnitF64(m, 1, c.ptr, beta, y);
    return true;
}

inline fn callGemvTransF64Kernel(
    comptime kernel: anytype,
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) void {
    const Kernel = *const fn (usize, usize, f64, [*]const f64, usize, [*]const f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(m, n, alpha, a, lda_bytes, x, y);
}

inline fn callSmGemvF64Kernel(
    comptime kernel: anytype,
    m: usize,
    n: usize,
    alpha_bits: u64,
    beta_bits: u64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) void {
    const Kernel = *const fn (usize, usize, u64, u64, [*]const f64, usize, [*]const f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
}

inline fn callSmGemvC64Kernel(
    comptime kernel: anytype,
    m: usize,
    n: usize,
    alpha_re_bits: u64,
    alpha_im_bits: u64,
    a: [*]const scalar.ComplexF64,
    lda_bytes: usize,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    const Kernel = *const fn (usize, usize, u64, u64, [*]const f64, usize, [*]const f64, [*]f64) callconv(.c) void;
    const a_f64: [*]const f64 = @ptrCast(a);
    const x_f64: [*]const f64 = @ptrCast(x);
    const y_f64: [*]f64 = @ptrCast(y);
    @as(Kernel, @ptrCast(&kernel))(m, n, alpha_re_bits, alpha_im_bits, a_f64, lda_bytes, x_f64, y_f64);
}

inline fn callCgemvTransFcmlaF32M128(
    comptime kernel: anytype,
    alpha: scalar.ComplexF32,
    beta: scalar.ComplexF32,
    a: [*]const scalar.ComplexF32,
    lda_bytes: usize,
    x: [*]const scalar.ComplexF32,
    y: [*]scalar.ComplexF32,
) void {
    const Kernel = *const fn (u32, u32, u32, u32, [*]const f32, usize, [*]const f32, [*]f32) callconv(.c) void;
    const a_f32: [*]const f32 = @ptrCast(a);
    const x_f32: [*]const f32 = @ptrCast(x);
    const y_f32: [*]f32 = @ptrCast(y);
    @as(Kernel, @ptrCast(&kernel))(@bitCast(alpha.re), @bitCast(alpha.im), @bitCast(beta.re), @bitCast(beta.im), a_f32, lda_bytes, x_f32, y_f32);
}

inline fn callZgemvNoTransFcmlaF64M128(
    comptime kernel: anytype,
    alpha: scalar.ComplexF64,
    beta: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda_bytes: usize,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    const Kernel = *const fn (u64, u64, u64, u64, [*]const f64, usize, [*]const f64, [*]f64) callconv(.c) void;
    const a_f64: [*]const f64 = @ptrCast(a);
    const x_f64: [*]const f64 = @ptrCast(x);
    const y_f64: [*]f64 = @ptrCast(y);
    @as(Kernel, @ptrCast(&kernel))(@bitCast(alpha.re), @bitCast(alpha.im), @bitCast(beta.re), @bitCast(beta.im), a_f64, lda_bytes, x_f64, y_f64);
}

inline fn callZgemvNoTransFcmlaF64M512NTask(
    comptime kernel: anytype,
    alpha: scalar.ComplexF64,
    panel_count: usize,
    a: [*]const scalar.ComplexF64,
    lda_bytes: usize,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    const Kernel = *const fn (u64, u64, usize, [*]const f64, usize, [*]const f64, [*]f64) callconv(.c) void;
    const a_f64: [*]const f64 = @ptrCast(a);
    const x_f64: [*]const f64 = @ptrCast(x);
    const y_f64: [*]f64 = @ptrCast(y);
    @as(Kernel, @ptrCast(&kernel))(@bitCast(alpha.re), @bitCast(alpha.im), panel_count, a_f64, lda_bytes, x_f64, y_f64);
}

inline fn callSmGemvF32Kernel(
    comptime kernel: anytype,
    m: usize,
    n: usize,
    alpha_bits: u32,
    beta_bits: u32,
    a: [*]const f32,
    lda_bytes: usize,
    x: [*]const f32,
    y: [*]f32,
) void {
    const Kernel = *const fn (usize, usize, u32, u32, [*]const f32, usize, [*]const f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
}

fn cgemvTransFcmlaF32M128(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF32,
    a: [*]const scalar.ComplexF32,
    lda: BlasInt,
    x: [*]const scalar.ComplexF32,
    beta: scalar.ComplexF32,
    y: [*]scalar.ComplexF32,
    do_conj: bool,
) bool {
    if (comptime !enable_fcmla_cgemv_t_m128 or !features.has_complxnum) return false;
    if (do_conj or m != 128 or n != 128 or lda < 128) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF32);
    callCgemvTransFcmlaF32M128(matrix_vector_asm.cgemvTransFcmlaF32M128, alpha, beta, a, lda_bytes, x, y);
    return true;
}

fn cgemvNoTransFcmlaF32M128(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF32,
    a: [*]const scalar.ComplexF32,
    lda: BlasInt,
    x: [*]const scalar.ComplexF32,
    beta: scalar.ComplexF32,
    y: [*]scalar.ComplexF32,
) bool {
    if (comptime !enable_fcmla_cgemv_n_m128 or !features.has_complxnum) return false;
    if (m != 128 or n != 128 or lda < 128) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF32);
    callCgemvTransFcmlaF32M128(matrix_vector_asm.cgemvNoTransFcmlaF32M128, alpha, beta, a, lda_bytes, x, y);
    return true;
}

fn cgemvNoTransFcmlaF32M512N64Task(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF32,
    a: [*]const scalar.ComplexF32,
    lda: BlasInt,
    x: [*]const scalar.ComplexF32,
    y_delta: [*]scalar.ComplexF32,
) bool {
    if (comptime !enable_fcmla_cgemv_n_m512_n64_task or !features.has_complxnum) return false;
    if (m != 512 or n != 64 or lda < 512) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF32);
    callCgemvTransFcmlaF32M128(matrix_vector_asm.cgemvNoTransFcmlaF32M512N64Task, alpha, scalar.zero(scalar.ComplexF32), a, lda_bytes, x, y_delta);
    return true;
}

fn zgemvNoTransFcmlaF64M128(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    if (comptime !enable_fcmla_zgemv_n_m128 or !features.has_complxnum) return false;
    if (m != 128 or n != 128 or lda < 128) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvNoTransFcmlaF64M128, alpha, beta, a, lda_bytes, x, y);
    return true;
}

fn zgemvNoTransFcmlaF64M512N64Task(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    y_delta: [*]scalar.ComplexF64,
) bool {
    if (comptime !enable_fcmla_zgemv_n_m512_n64_task or !features.has_complxnum) return false;
    if (m != 512 or lda < 512) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    if (n == 64) {
        callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvNoTransFcmlaF64M512N64Task, alpha, scalar.zero(scalar.ComplexF64), a, lda_bytes, x, y_delta);
        return true;
    }
    if (n < 48 or n > 64 or n % 4 != 0) return false;
    callZgemvNoTransFcmlaF64M512NTask(matrix_vector_asm.zgemvNoTransFcmlaF64M512NTask, alpha, n / 4, a, lda_bytes, x, y_delta);
    return true;
}

fn zgemvTransFcmlaF64M128(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
    do_conj: bool,
) bool {
    if (comptime !enable_fcmla_zgemv_t_m128 or !features.has_complxnum) return false;
    if (do_conj or m != 128 or n != 128 or lda < 128) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvTransFcmlaF64M128, alpha, beta, a, lda_bytes, x, y);
    return true;
}

fn zgemvTransFcmlaF64M256N128Task(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    if (comptime !enable_fcmla_zgemv_t_m256_n128_task or !features.has_complxnum) return false;
    if (m != 256 or n != 128 or lda < 256) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvTransFcmlaF64M256N128Task, alpha, beta, a, lda_bytes, x, y);
    return true;
}

fn zgemvTransFcmlaF64M512N64Task(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
    do_conj: bool,
) bool {
    if (!supportsGemvTransTaskFullUnitComplex(scalar.ComplexF64, m, n, lda, do_conj)) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvTransFcmlaF64M512N64Task, alpha, scalar.one(scalar.ComplexF64), a, lda_bytes, x, y);
    return true;
}

fn zgemvTransFcmlaF64M512N64TaskBeta(
    m: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
    do_conj: bool,
) bool {
    if (!supportsGemvTransTaskFullUnitComplex(scalar.ComplexF64, m, n, lda, do_conj)) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvTransFcmlaF64M512N64TaskBeta, alpha, beta, a, lda_bytes, x, y);
    return true;
}

pub fn gemvTransTaskFullUnitComplexC64M512N64(
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callZgemvNoTransFcmlaF64M128(matrix_vector_asm.zgemvTransFcmlaF64M512N64TaskBeta, alpha, beta, a, lda_bytes, x, y);
}

pub fn gemvTransTaskFullUnitComplexC64M256N128(
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    return zgemvTransFcmlaF64M256N128Task(256, 128, alpha, a, lda, x, beta, y);
}

inline fn loadF64x8(ptr: [*]const f64, offset: usize) @Vector(8, f64) {
    return @as(*align(@alignOf(f64)) const @Vector(8, f64), @ptrCast(ptr + offset)).*;
}

inline fn storeF64x8(ptr: [*]f64, offset: usize, value: @Vector(8, f64)) void {
    @as(*align(@alignOf(f64)) @Vector(8, f64), @ptrCast(ptr + offset)).* = value;
}

inline fn loadF32x8(ptr: [*]const f32, offset: usize) @Vector(8, f32) {
    return @as(*align(1) const @Vector(8, f32), @ptrCast(ptr + offset)).*;
}

inline fn storeF32x8(ptr: [*]f32, offset: usize, value: @Vector(8, f32)) void {
    @as(*align(1) @Vector(8, f32), @ptrCast(ptr + offset)).* = value;
}

inline fn gerF32x16Rows8VectorColumn(
    x_vec0: @Vector(8, f32),
    x_vec1: @Vector(8, f32),
    scalar_vec: @Vector(8, f32),
    col: [*]f32,
    i: usize,
) void {
    storeF32x8(col, i, @mulAdd(@Vector(8, f32), x_vec0, scalar_vec, loadF32x8(col, i)));
    storeF32x8(col, i + 8, @mulAdd(@Vector(8, f32), x_vec1, scalar_vec, loadF32x8(col, i + 8)));
}

noinline fn gerF32x16Rows8Vector(
    m: usize,
    n: usize,
    alpha: f32,
    noalias x: [*]const f32,
    noalias y: [*]const f32,
    noalias a: [*]f32,
    lda: BlasInt,
) void {
    const V = @Vector(8, f32);
    const lda_elems: usize = @intCast(lda);
    var j: usize = 0;
    while (j + 8 <= n) : (j += 8) {
        const s0: V = @splat(alpha * y[j]);
        const s1: V = @splat(alpha * y[j + 1]);
        const s2: V = @splat(alpha * y[j + 2]);
        const s3: V = @splat(alpha * y[j + 3]);
        const s4: V = @splat(alpha * y[j + 4]);
        const s5: V = @splat(alpha * y[j + 5]);
        const s6: V = @splat(alpha * y[j + 6]);
        const s7: V = @splat(alpha * y[j + 7]);
        const c0 = a + lda_elems * j;
        const c1 = c0 + lda_elems;
        const c2 = c1 + lda_elems;
        const c3 = c2 + lda_elems;
        const c4 = c3 + lda_elems;
        const c5 = c4 + lda_elems;
        const c6 = c5 + lda_elems;
        const c7 = c6 + lda_elems;

        var i: usize = 0;
        while (i + 16 <= m) : (i += 16) {
            @branchHint(.likely);
            @prefetch(x + i + 32, .{ .rw = .read, .locality = 3, .cache = .data });
            const x_vec0 = loadF32x8(x, i);
            const x_vec1 = loadF32x8(x, i + 8);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s0, c0, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s1, c1, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s2, c2, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s3, c3, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s4, c4, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s5, c5, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s6, c6, i);
            gerF32x16Rows8VectorColumn(x_vec0, x_vec1, s7, c7, i);
        }
    }
}

inline fn gerF64x8Rows8VectorColumn(
    x_vec: @Vector(8, f64),
    scalar_vec: @Vector(8, f64),
    col: [*]f64,
    i: usize,
) void {
    storeF64x8(col, i, @mulAdd(@Vector(8, f64), x_vec, scalar_vec, loadF64x8(col, i)));
}

inline fn gerF64VectorRowsColumns(
    comptime columns: comptime_int,
    m: usize,
    noalias x: [*]const f64,
    scalars: *const [columns]f64,
    cols: *const [columns][*]f64,
) void {
    const V = @Vector(8, f64);
    var scalar_vecs: [columns]V = undefined;
    inline for (0..columns) |col| scalar_vecs[col] = @splat(scalars.*[col]);

    var i: usize = 0;
    while (i + 8 <= m) : (i += 8) {
        @branchHint(.likely);
        @prefetch(x + i + 32, .{ .rw = .read, .locality = 3, .cache = .data });
        const x_vec = loadF64x8(x, i);
        inline for (0..columns) |col| {
            gerF64x8Rows8VectorColumn(x_vec, scalar_vecs[col], cols.*[col], i);
        }
    }
    while (i < m) : (i += 1) {
        const x_scalar = x[i];
        inline for (0..columns) |col| {
            const col_ptr = cols.*[col];
            col_ptr[i] = @mulAdd(f64, x_scalar, scalars.*[col], col_ptr[i]);
        }
    }
}

noinline fn gerF64x8Rows8Vector(
    m: usize,
    n: usize,
    alpha: f64,
    noalias x: [*]const f64,
    noalias y: [*]const f64,
    noalias a: [*]f64,
    lda: BlasInt,
) void {
    const V = @Vector(8, f64);
    const lda_elems: usize = @intCast(lda);
    var j: usize = 0;
    while (j + 8 <= n) : (j += 8) {
        const s0: V = @splat(alpha * y[j]);
        const s1: V = @splat(alpha * y[j + 1]);
        const s2: V = @splat(alpha * y[j + 2]);
        const s3: V = @splat(alpha * y[j + 3]);
        const s4: V = @splat(alpha * y[j + 4]);
        const s5: V = @splat(alpha * y[j + 5]);
        const s6: V = @splat(alpha * y[j + 6]);
        const s7: V = @splat(alpha * y[j + 7]);
        const c0 = a + lda_elems * j;
        const c1 = c0 + lda_elems;
        const c2 = c1 + lda_elems;
        const c3 = c2 + lda_elems;
        const c4 = c3 + lda_elems;
        const c5 = c4 + lda_elems;
        const c6 = c5 + lda_elems;
        const c7 = c6 + lda_elems;

        var i: usize = 0;
        while (i + 8 <= m) : (i += 8) {
            @branchHint(.likely);
            @prefetch(x + i + 32, .{ .rw = .read, .locality = 3, .cache = .data });
            const x_vec = loadF64x8(x, i);
            gerF64x8Rows8VectorColumn(x_vec, s0, c0, i);
            gerF64x8Rows8VectorColumn(x_vec, s1, c1, i);
            gerF64x8Rows8VectorColumn(x_vec, s2, c2, i);
            gerF64x8Rows8VectorColumn(x_vec, s3, c3, i);
            gerF64x8Rows8VectorColumn(x_vec, s4, c4, i);
            gerF64x8Rows8VectorColumn(x_vec, s5, c5, i);
            gerF64x8Rows8VectorColumn(x_vec, s6, c6, i);
            gerF64x8Rows8VectorColumn(x_vec, s7, c7, i);
        }
    }
}

noinline fn gerF64Vector(
    m: usize,
    n: usize,
    alpha: f64,
    noalias x: [*]const f64,
    noalias y: [*]const f64,
    noalias a: [*]f64,
    lda: BlasInt,
) void {
    const lda_elems: usize = @intCast(lda);
    var j: usize = 0;
    while (j + 8 <= n) : (j += 8) {
        const scalars = [_]f64{
            alpha * y[j],
            alpha * y[j + 1],
            alpha * y[j + 2],
            alpha * y[j + 3],
            alpha * y[j + 4],
            alpha * y[j + 5],
            alpha * y[j + 6],
            alpha * y[j + 7],
        };
        const c0 = a + lda_elems * j;
        const cols = [_][*]f64{
            c0,
            c0 + lda_elems,
            c0 + lda_elems * 2,
            c0 + lda_elems * 3,
            c0 + lda_elems * 4,
            c0 + lda_elems * 5,
            c0 + lda_elems * 6,
            c0 + lda_elems * 7,
        };
        gerF64VectorRowsColumns(8, m, x, &scalars, &cols);
    }
    while (j + 4 <= n) : (j += 4) {
        const scalars = [_]f64{
            alpha * y[j],
            alpha * y[j + 1],
            alpha * y[j + 2],
            alpha * y[j + 3],
        };
        const c0 = a + lda_elems * j;
        const cols = [_][*]f64{
            c0,
            c0 + lda_elems,
            c0 + lda_elems * 2,
            c0 + lda_elems * 3,
        };
        gerF64VectorRowsColumns(4, m, x, &scalars, &cols);
    }
    while (j + 2 <= n) : (j += 2) {
        const scalars = [_]f64{
            alpha * y[j],
            alpha * y[j + 1],
        };
        const c0 = a + lda_elems * j;
        const cols = [_][*]f64{
            c0,
            c0 + lda_elems,
        };
        gerF64VectorRowsColumns(2, m, x, &scalars, &cols);
    }
    if (j < n) {
        const scalars = [_]f64{alpha * y[j]};
        const cols = [_][*]f64{a + lda_elems * j};
        gerF64VectorRowsColumns(1, m, x, &scalars, &cols);
    }
}

fn gerUnitRealSmeF64(
    m: usize,
    n: usize,
    alpha: f64,
    x: [*]const f64,
    y: [*]const f64,
    a: [*]f64,
    lda: BlasInt,
) bool {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda;
    return false;
}

fn gerUnitRealAsimdF64(
    m: usize,
    n: usize,
    alpha: f64,
    x: [*]const f64,
    y: [*]const f64,
    a: [*]f64,
    lda: BlasInt,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_asimd_dger or !features.has_asimd) return false;
    const small_shape = m >= 64 and m <= 256 and n >= 16 and n <= 128;
    const dger512_task_shape = m == 512 and n == 128;
    if (!small_shape and !dger512_task_shape) return false;
    if ((m & 7) == 0 and (n & 7) == 0) {
        gerF64x8Rows8Vector(m, n, alpha, x, y, a, lda);
        return true;
    }
    gerF64Vector(m, n, alpha, x, y, a, lda);
    return true;
}

fn gerUnitRealAsimdF32(
    m: usize,
    n: usize,
    alpha: f32,
    x: [*]const f32,
    y: [*]const f32,
    a: [*]f32,
    lda: BlasInt,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_asimd_sger or !features.has_asimd) return false;
    const sger128_shape = m == 128 and n == 128 and lda >= 128;
    const sger512_task_shape = m == 512 and n == 128 and lda >= 512;
    if (!sger128_shape and !sger512_task_shape) return false;
    gerF32x16Rows8Vector(m, n, alpha, x, y, a, lda);
    return true;
}

fn gemvNoTransF64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    y: [*]f64,
) bool {
    if (comptime !features.has_asimd) return false;
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (gemvNoTransAmxF64(m, n, alpha, a, lda, x, y)) return true;

    return false;
}

fn gemvNoTransSme2F64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    beta: f64,
    y: [*]f64,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_sme2_gemv_n or !features.has_sme2 or !features.has_sme_f64f64) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if (n < 128 or n > 1024) return false;
    if (m != 128 and (m < 256 or m > 1024 or (m & 255) != 0)) return false;

    const alpha_bits: u64 = @bitCast(alpha);
    const beta_bits: u64 = @bitCast(beta);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    gemvNoTransSme2F64Bits(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvNoTransSme2F64Bits(
    m: usize,
    n: usize,
    alpha_bits: u64,
    beta_bits: u64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    if (m == 128) {
        callSmGemvF64Kernel(matrix_vector_asm.dgemvNoTransSme2F64128x1, 128, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    } else {
        callSmGemvF64Kernel(matrix_vector_asm.dgemvNoTransSme2F64256x1, m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    }
}

fn gemvNoTransSmF64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    beta: f64,
    y: [*]f64,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_sm_gemv_n or !features.has_sme) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if ((m & 127) != 0 or m > 512 or n < 128 or n > 512 or (n & 3) != 0) return false;

    const alpha_bits: u64 = @bitCast(alpha);
    const beta_bits: u64 = @bitCast(beta);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    gemvNoTransSmF64Bits(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvNoTransSmF64Bits(
    m: usize,
    n: usize,
    alpha_bits: u64,
    beta_bits: u64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSm();
    defer sm_state.stopSm();

    var row: usize = 0;
    while (row < m) : (row += 128) {
        const block_a = a + row;
        callSmGemvF64Kernel(matrix_vector_asm.dgemvNoTransSmF64M128, 128, n, alpha_bits, beta_bits, block_a, lda_bytes, x, y + row);
    }
}

fn gemvNoTransSme2F32(
    m: usize,
    n: usize,
    alpha: f32,
    a: [*]const f32,
    lda: BlasInt,
    x: [*]const f32,
    beta: f32,
    y: [*]f32,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_sme2_gemv_n or !features.has_sme2) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if ((m != 256 and m != 512) or n < 128 or n > 1024) return false;

    const alpha_bits: u32 = @bitCast(alpha);
    const beta_bits: u32 = @bitCast(beta);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f32);
    gemvNoTransSme2F32Bits(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvNoTransSme2F32Bits(
    m: usize,
    n: usize,
    alpha_bits: u32,
    beta_bits: u32,
    a: [*]const f32,
    lda_bytes: usize,
    x: [*]const f32,
    y: [*]f32,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    if (m == 256) {
        callSmGemvF32Kernel(matrix_vector_asm.sgemvNoTransSme2F32256x1, m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    } else {
        callSmGemvF32Kernel(matrix_vector_asm.sgemvNoTransSme2F32512x1, m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    }
}

pub fn gemvNoTransFullUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    if (T == f32) {
        if (gemvNoTransSme2F32(m, n, alpha, a, lda, x, beta, y)) return true;
        if (gemvNoTransAsimdF32M128(m, n, alpha, a, lda, x, beta, y)) return true;
        if (gemvNoTransAmxF32(m, n, alpha, a, lda, x, beta, y)) return true;
        if (comptime features.has_asimd) return fixed_simd.gemvNoTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
        return false;
    }
    if (T == f64) {
        if (gemvNoTransSmF64(m, n, alpha, a, lda, x, beta, y)) return true;
        if (gemvNoTransSme2F64(m, n, alpha, a, lda, x, beta, y)) return true;
        if (comptime features.has_asimd)
            if (gemvNoTransAmxF64Full(m, n, alpha, a, lda, x, beta, y)) return true;
        if (comptime features.has_asimd) return fixed_simd.gemvNoTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
        return false;
    }
    return false;
}

fn gemvTransF64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    y: [*]f64,
) bool {
    if (comptime enable_amx_gemv_t and builtin.target.os.tag == .macos and features.has_asimd) {
        if (m == 1024 and n == 1024) {
            if (amx.dgemvTransN8(@intCast(m), @intCast(n), alpha, a, @intCast(lda), x, y) != 0) return true;
        }
    }
    if (m == 0 or n == 0 or lda <= 0) return false;

    if (comptime features.has_sve) {
        if (m >= 256 and n >= 8 and n <= 1536) {
            const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
            const sve_lanes = features.sveVectorBytes() / @sizeOf(f64);
            if (sve_lanes > 0 and (m % sve_lanes) == 0) {
                const panel_n = n & ~@as(usize, 7);
                if ((m % (2 * sve_lanes)) == 0) {
                    if (panel_n > 0) callGemvTransF64Kernel(matrix_vector_asm.dgemvTransSveF64FullN8Acc2, m, panel_n, alpha, a, lda_bytes, x, y);
                } else {
                    if (panel_n > 0) callGemvTransF64Kernel(matrix_vector_asm.dgemvTransSveF64FullN8, m, panel_n, alpha, a, lda_bytes, x, y);
                }
                if (panel_n < n) {
                    const tail_a = a + panel_n * @as(usize, @intCast(lda));
                    callGemvTransF64Kernel(matrix_vector_asm.dgemvTransSveF64, m, n - panel_n, alpha, tail_a, lda_bytes, x, y + panel_n);
                }
                return true;
            }
            callGemvTransF64Kernel(matrix_vector_asm.dgemvTransSveF64, m, n, alpha, a, lda_bytes, x, y);
            return true;
        }
    }
    return false;
}

fn gemvTransSme2F64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda: BlasInt,
    x: [*]const f64,
    beta: f64,
    y: [*]f64,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_sme2_gemv_t or !features.has_sme2 or !features.has_sme_f64f64) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if (m < 128 or m > 1024 or n < 8 or n > 1024 or (m & 31) != 0 or (n & 7) != 0) return false;

    const alpha_bits: u64 = @bitCast(alpha);
    const beta_bits: u64 = @bitCast(beta);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    gemvTransSme2F64Bits(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvTransSme2F64Bits(
    m: usize,
    n: usize,
    alpha_bits: u64,
    beta_bits: u64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    callSmGemvF64Kernel(matrix_vector_asm.dgemvTransSme2F648x32, m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
}

fn canUseGemvTransF64(m: usize, n: usize, lda: BlasInt) bool {
    if (comptime enable_amx_gemv_t and builtin.target.os.tag == .macos and features.has_asimd) {
        if (m == 1024 and n == 1024) return true;
    }
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (comptime features.has_sve) return m >= 256 and n >= 8 and n <= 1536;
    return false;
}

fn gemvTransSme2F32(
    m: usize,
    n: usize,
    alpha: f32,
    a: [*]const f32,
    lda: BlasInt,
    x: [*]const f32,
    beta: f32,
    y: [*]f32,
) bool {
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (alpha == 0) return false;
    if (comptime !enable_sme2_gemv_t or !features.has_sme2) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if (m < 256 or m > 1024 or n < 128 or n > 1024 or (m & 63) != 0 or (n & 15) != 0) return false;

    const alpha_bits: u32 = @bitCast(alpha);
    const beta_bits: u32 = @bitCast(beta);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f32);
    gemvTransSme2F32Bits(m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvTransSme2F32Bits(
    m: usize,
    n: usize,
    alpha_bits: u32,
    beta_bits: u32,
    a: [*]const f32,
    lda_bytes: usize,
    x: [*]const f32,
    y: [*]f32,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    callSmGemvF32Kernel(matrix_vector_asm.sgemvTransSme2F3216x64, m, n, alpha_bits, beta_bits, a, lda_bytes, x, y);
}

pub fn gemvTransFullUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    if (T == f32) {
        if (gemvTransSme2F32(m, n, alpha, a, lda, x, beta, y)) return true;
        if (m == 512 and n == 512) return false;
        if (comptime features.has_asimd) return fixed_simd.gemvTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
        return false;
    }
    if (T == f64) {
        if (gemvTransSme2F64(m, n, alpha, a, lda, x, beta, y)) return true;
        if (m == 512 and n == 512) return false;
        if (canUseGemvTransF64(m, n, lda)) {
            scaleUnitF64(n, beta, y);
            if (gemvTransF64(m, n, alpha, a, lda, x, y)) return true;
        }
        if (comptime features.has_asimd) return fixed_simd.gemvTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
        return false;
    }
    return false;
}

pub fn gemvTransAmxUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (T != f64) return false;
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (comptime !enable_amx_gemv_t or builtin.target.os.tag != .macos or !features.has_asimd) return false;
    if (m != 1024 or n != 1024) return false;
    return amx.dgemvTransN8(@intCast(m), @intCast(n), alpha, a, @intCast(lda), x, y) != 0;
}

pub fn gemvTransUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (T == f32) {
        if (gemvTransSme2F32(m, n, alpha, a, lda, x, 1, y)) return true;
        if (comptime features.has_asimd) return fixed_simd.gemvTransUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, y);
        return false;
    }
    if (T == f64) {
        if (gemvTransSme2F64(m, n, alpha, a, lda, x, 1, y)) return true;
        if (gemvTransF64(m, n, alpha, a, lda, x, y)) return true;
        if (comptime features.has_asimd) return fixed_simd.gemvTransUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, y);
        return false;
    }
    return false;
}

pub fn gemvNoTransUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (T == f64 and gemvNoTransF64(m, n, alpha, a, lda, x, y)) return true;
    if ((T == f32 or T == f64) and comptime features.has_asimd) return fixed_simd.gemvNoTransUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, y);
    return false;
}

fn canUseGemvNoTransSme2C64Rows(row_count: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enable_sme2_zgemv_n_rows) return false;
    if ((row_count != 64 and row_count != 128) or n != 512 or lda != 512) return false;
    if (comptime !features.has_sme2 or !features.has_sme_f64f64) return false;
    return features.streamingVectorBytes() == 64;
}

fn canUseGemvNoTransFcmlaC64Rows(row_count: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enable_fcmla_zgemv_n_m64_n512_rows or !features.has_complxnum) return false;
    return row_count == 64 and n == 512 and lda == 512;
}

fn gemvNoTransFcmlaC64Rows(
    row_count: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    if (!canUseGemvNoTransFcmlaC64Rows(row_count, n, lda)) return false;

    const alpha_re_bits: u64 = @bitCast(alpha.re);
    const alpha_im_bits: u64 = @bitCast(alpha.im);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    callSmGemvC64Kernel(matrix_vector_asm.zgemvNoTransFcmlaF64M64N512Rows, row_count, n, alpha_re_bits, alpha_im_bits, a, lda_bytes, x, y);
    return true;
}

fn gemvNoTransSme2C64Rows(
    row_count: usize,
    n: usize,
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    if (!canUseGemvNoTransSme2C64Rows(row_count, n, lda)) return false;

    const alpha_re_bits: u64 = @bitCast(alpha.re);
    const alpha_im_bits: u64 = @bitCast(alpha.im);
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(scalar.ComplexF64);
    gemvNoTransSme2C64RowsBits(row_count, n, alpha_re_bits, alpha_im_bits, a, lda_bytes, x, y);
    return true;
}

noinline fn gemvNoTransSme2C64RowsBits(
    row_count: usize,
    n: usize,
    alpha_re_bits: u64,
    alpha_im_bits: u64,
    a: [*]const scalar.ComplexF64,
    lda_bytes: usize,
    x: [*]const scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    if (row_count == 64) {
        callSmGemvC64Kernel(matrix_vector_asm.zgemvNoTransSme2C6464x1, row_count, n, alpha_re_bits, alpha_im_bits, a, lda_bytes, x, y);
    } else {
        callSmGemvC64Kernel(matrix_vector_asm.zgemvNoTransSme2C64512x1, row_count, n, alpha_re_bits, alpha_im_bits, a, lda_bytes, x, y);
    }
}

pub fn supportsGemvNoTransRowsUnitComplex(comptime T: type, row_count: usize, n: usize, lda: BlasInt) bool {
    if (T == scalar.ComplexF64) return canUseGemvNoTransFcmlaC64Rows(row_count, n, lda) or canUseGemvNoTransSme2C64Rows(row_count, n, lda);
    return false;
}

pub fn gemvNoTransRowsUnitComplex(
    comptime T: type,
    row_count: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (T == scalar.ComplexF64 and gemvNoTransFcmlaC64Rows(row_count, n, alpha, a, lda, x, y)) return true;
    if (T == scalar.ComplexF64) return gemvNoTransSme2C64Rows(row_count, n, alpha, a, lda, x, y);
    return false;
}

pub fn gemvNoTransUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (T == scalar.ComplexF32) {
        if (useFixedSimdComplexBeforeParallel(T, m, n)) return fixed_simd.gemvNoTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y);
        return false;
    }
    if (T == scalar.ComplexF64) {
        if (useFixedSimdComplexBeforeParallel(T, m, n)) return fixed_simd.gemvNoTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y);
        return false;
    }
    return false;
}

pub fn gemvNoTransFullUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    if (T == scalar.ComplexF32) return cgemvNoTransFcmlaF32M128(m, n, alpha, a, lda, x, beta, y);
    if (T == scalar.ComplexF64) return zgemvNoTransFcmlaF64M128(m, n, alpha, a, lda, x, beta, y);
    return false;
}

pub fn supportsGemvNoTransFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt) bool {
    if (!features.has_complxnum or m != 128 or n != 128 or lda < 128) return false;
    return (T == scalar.ComplexF32 and enable_fcmla_cgemv_n_m128) or (T == scalar.ComplexF64 and enable_fcmla_zgemv_n_m128);
}

pub fn gemvTransFullUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (T == scalar.ComplexF32) return cgemvTransFcmlaF32M128(m, n, alpha, a, lda, x, beta, y, do_conj);
    if (T == scalar.ComplexF64) return zgemvTransFcmlaF64M128(m, n, alpha, a, lda, x, beta, y, do_conj);
    return false;
}

pub fn supportsGemvTransFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (do_conj or !features.has_complxnum or m != 128 or n != 128 or lda < 128) return false;
    return (T == scalar.ComplexF32 and enable_fcmla_cgemv_t_m128) or (T == scalar.ComplexF64 and enable_fcmla_zgemv_t_m128);
}

pub fn gemvNoTransTaskUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y_delta: [*]T,
) bool {
    if (T == scalar.ComplexF32) return cgemvNoTransFcmlaF32M512N64Task(m, n, alpha, a, lda, x, y_delta);
    if (T == scalar.ComplexF64) return zgemvNoTransFcmlaF64M512N64Task(m, n, alpha, a, lda, x, y_delta);
    return false;
}

pub fn supportsGemvNoTransTaskUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt) bool {
    if (!features.has_complxnum or m != 512 or lda < 512) return false;
    if (T == scalar.ComplexF32) return n == 64 and enable_fcmla_cgemv_n_m512_n64_task;
    if (T == scalar.ComplexF64) return enable_fcmla_zgemv_n_m512_n64_task and n >= 48 and n <= 64 and n % 4 == 0;
    return false;
}

pub fn gemvTransUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (T == scalar.ComplexF32) {
        if (useFixedSimdComplexBeforeParallel(T, m, n)) return fixed_simd.gemvTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y, do_conj);
        return false;
    }
    if (T == scalar.ComplexF64) {
        if (useFixedSimdComplexBeforeParallel(T, m, n)) return fixed_simd.gemvTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y, do_conj);
        return false;
    }
    return false;
}

pub fn gemvTransTaskUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (T == scalar.ComplexF64) return zgemvTransFcmlaF64M512N64Task(m, n, alpha, a, lda, x, y, do_conj);
    return false;
}

pub fn gemvTransTaskFullUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (T == scalar.ComplexF64) return zgemvTransFcmlaF64M512N64TaskBeta(m, n, alpha, a, lda, x, beta, y, do_conj);
    return false;
}

pub fn supportsGemvTransTaskFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (comptime !enable_fcmla_zgemv_t_m512_n64_task or !features.has_complxnum) return false;
    return T == scalar.ComplexF64 and !do_conj and m == 512 and n == 64 and lda >= 512;
}

pub fn supportsGemvNoTransUnitComplex(comptime T: type) bool {
    if (T == scalar.ComplexF32 or T == scalar.ComplexF64) return features.has_asimd;
    return false;
}

pub fn supportsGemvTransUnitComplex(comptime T: type) bool {
    if (T == scalar.ComplexF32 or T == scalar.ComplexF64) return features.has_asimd;
    return false;
}

pub fn gemvNoTransPackLenUnitReal(comptime T: type, m: usize, n: usize, lda: BlasInt) ?usize {
    if (T == f64) return gemvNoTransPackLenF64(m, n, lda);
    return null;
}

pub fn gemvNoTransPackUnitReal(
    comptime T: type,
    n: usize,
    alpha: T,
    x: [*]const T,
    pack: []T,
) bool {
    if (T == f64) return gemvNoTransPackF64(n, alpha, x, pack);
    return false;
}

pub fn gemvNoTransPackedRowsUnitReal(
    comptime T: type,
    row_count: usize,
    n: usize,
    a: [*]const T,
    lda: BlasInt,
    pack: [*]const T,
    scratch: [*]T,
    y: [*]T,
) bool {
    if (T == f64) return gemvNoTransPackedRowsF64(row_count, n, a, lda, pack, scratch, y);
    return false;
}

pub fn supportsGemvNoTransUnitReal(comptime T: type) bool {
    return (T == f32 or T == f64) and features.has_asimd;
}

pub fn gerUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
) bool {
    if (T == f32) {
        if (gerUnitRealAsimdF32(m, n, alpha, x, y, a, lda)) return true;
    }
    if (T == f64) {
        if (gerUnitRealAsimdF64(m, n, alpha, x, y, a, lda)) return true;
        if (gerUnitRealSmeF64(m, n, alpha, x, y, a, lda)) return true;
    }
    if ((T == f32 or T == f64) and comptime features.has_asimd) return fixed_simd.gerUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, x, y, a, lda);
    return false;
}
