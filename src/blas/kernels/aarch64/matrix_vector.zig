// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 matrix-vector specialized kernels.

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../../core/scalar.zig");
const amx = @import("amx_gemm.zig");
const features = @import("features.zig");

const BlasInt = scalar.BlasInt;
const enable_amx_gemv_n = true;
const enable_amx_gemv_t = false;
const enable_sme2_gemv_t = true;
const enable_asimd_gemv_t_n1 = false;
const enable_asimd_gemv_t = false;
const enable_sve_ger = false;
const enable_asimd_ger = false;
const enable_sme_ger = false;

threadlocal var amx_gemv_b_ptr: ?[*]f64 = null;
threadlocal var amx_gemv_b_len: usize = 0;
threadlocal var amx_gemv_c_ptr: ?[*]f64 = null;
threadlocal var amx_gemv_c_len: usize = 0;

fn amxGemvBuffer(which: enum { b, c }, len: usize) ?[]f64 {
    switch (which) {
        .b => {
            if (amx_gemv_b_len < len) {
                const data = std.heap.c_allocator.alloc(f64, len) catch return null;
                if (amx_gemv_b_ptr) |old| std.heap.c_allocator.free(old[0..amx_gemv_b_len]);
                amx_gemv_b_ptr = data.ptr;
                amx_gemv_b_len = len;
            }
            return amx_gemv_b_ptr.?[0..len];
        },
        .c => {
            if (amx_gemv_c_len < len) {
                const data = std.heap.c_allocator.alloc(f64, len) catch return null;
                if (amx_gemv_c_ptr) |old| std.heap.c_allocator.free(old[0..amx_gemv_c_len]);
                amx_gemv_c_ptr = data.ptr;
                amx_gemv_c_len = len;
            }
            return amx_gemv_c_ptr.?[0..len];
        },
    }
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

fn canUseAmxGemvNoTransF64(m: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enable_amx_gemv_n) return false;
    if (comptime builtin.target.os.tag != .macos) return false;
    if (m *| n < 512 * 512 or m *| n >= 1536 * 1536) return false;
    if (n < 512) return false;
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
    const b = amxGemvBuffer(.b, pack_len) orelse return false;
    const c = amxGemvBuffer(.c, m) orelse return false;

    if (!gemvNoTransPackF64(n, alpha, x, b)) return false;
    return gemvNoTransPackedRowsF64(m, n, a, lda, b.ptr, c.ptr, y);
}

extern fn zynum_blas_sve_dgemv_t_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sve_dgemv_t_f64_full_n8(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sve_dgemv_t_f64_full_n8_acc2(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sve_dgemv_t_f64_n16(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sme2_dgemv_n_f64_256x1(
    m: usize,
    n: usize,
    alpha: f64,
    beta: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sme2_dgemv_t_f64_8x32(
    m: usize,
    n: usize,
    alpha: f64,
    beta: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_asimd_dgemv_t_f64_n1(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_asimd_dgemv_t_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_asimd_dgemv_n_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    lda_bytes: usize,
    x: [*]const f64,
    y: [*]f64,
) callconv(.c) void;

extern fn zynum_blas_sve_dger_f64(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.c) void;
extern fn zynum_blas_asimd_dger_f64(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_dger_f64(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.c) void;

fn gerUnitRealSmeF64(
    m: usize,
    n: usize,
    alpha: f64,
    x: [*]const f64,
    y: [*]const f64,
    a: [*]f64,
    lda: BlasInt,
) bool {
    if (comptime !enable_sme_ger) return false;
    if (comptime !features.has_sme_f64f64) return false;
    if (m == 0 or n == 0 or lda <= 0) return false;
    if (m *| n < 1536 * 1536) return false;

    const tile = @max(@as(usize, 1), features.streamingVectorBytes() / @sizeOf(f64));
    if (tile == 0 or (m % tile) != 0 or (n % tile) != 0) return false;

    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    zynum_blas_sme_dger_f64(m, n, alpha, x, y, a, lda_bytes);
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
    if (comptime !enable_sme2_gemv_t or !features.has_sme2 or !features.has_sme_f64f64) return false;
    if (features.streamingVectorBytes() != 64) return false;
    if (m < 512 or n < 512 or m > 1536 or n > 1536 or (m & 255) != 0) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    zynum_blas_sme2_dgemv_n_f64_256x1(m, n, alpha, beta, a, lda_bytes, x, y);
    return true;
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
    if (T == f64) return gemvNoTransSme2F64(m, n, alpha, a, lda, x, beta, y);
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
        if (m >= 512 and n >= 512 and m <= 1536 and n <= 1536 and (n & 7) == 0) {
            if (amx.dgemvTransN8(@intCast(m), @intCast(n), alpha, a, @intCast(lda), x, y) != 0) return true;
        }
    }
    if (m == 0 or n == 0 or lda <= 0) return false;

    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    if (gemvTransSme2F64(m, n, alpha, a, lda, x, 1, y)) return true;
    if (comptime enable_asimd_gemv_t_n1 and features.has_asimd) {
        if (m >= 512 and n >= 8 and m <= 1536 and n <= 1536) {
            zynum_blas_asimd_dgemv_t_f64_n1(m, n, alpha, a, lda_bytes, x, y);
            return true;
        }
    }
    if (comptime features.has_sve) {
        if (m >= 256 and n >= 8 and n <= 1536) {
            const sve_lanes = features.sveVectorBytes() / @sizeOf(f64);
            if (sve_lanes > 0 and (m % sve_lanes) == 0) {
                if ((m % (2 * sve_lanes)) == 0) {
                    zynum_blas_sve_dgemv_t_f64_full_n8_acc2(m, n, alpha, a, lda_bytes, x, y);
                    return true;
                }
                zynum_blas_sve_dgemv_t_f64_full_n8(m, n, alpha, a, lda_bytes, x, y);
                return true;
            }
            zynum_blas_sve_dgemv_t_f64(m, n, alpha, a, lda_bytes, x, y);
            return true;
        }
    }
    if (comptime enable_asimd_gemv_t and features.has_asimd) {
        if (m >= 64 and n >= 8) {
            zynum_blas_asimd_dgemv_t_f64(m, n, alpha, a, lda_bytes, x, y);
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
    if (m < 512 or n < 8 or m > 1536 or n > 1536 or (m & 31) != 0 or (n & 7) != 0) return false;
    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    zynum_blas_sme2_dgemv_t_f64_8x32(m, n, alpha, beta, a, lda_bytes, x, y);
    return true;
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
    if (T == f64) return gemvTransSme2F64(m, n, alpha, a, lda, x, beta, y);
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
    if (m < 512 or n < 512 or m > 1536 or n > 1536 or (n & 7) != 0) return false;
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
    if (T == f64) return gemvTransF64(m, n, alpha, a, lda, x, y);
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
    if (T == f64) return gemvNoTransF64(m, n, alpha, a, lda, x, y);
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
    return T == f64 and features.has_asimd;
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
    if (T == f64) {
        if (gerUnitRealSmeF64(m, n, alpha, x, y, a, lda)) return true;
    }
    if (comptime enable_sve_ger and features.has_sve) {
        const sve_lanes = features.sveVectorBytes() / @sizeOf(f64);
        if (T == f64 and sve_lanes > 0 and (m % sve_lanes) == 0 and m >= 256 and n >= 4 and m *| n >= 512 * 512 and lda > 0) {
            const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
            zynum_blas_sve_dger_f64(m, n, alpha, x, y, a, lda_bytes);
            return true;
        }
    }
    if (comptime !enable_asimd_ger) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (m == 0 or n == 0 or lda <= 0) return false;

    const lda_bytes = @as(usize, @intCast(lda)) * @sizeOf(f64);
    zynum_blas_asimd_dger_f64(m, n, alpha, x, y, a, lda_bytes);
    return true;
}
