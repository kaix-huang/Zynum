// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 1 binary-vector kernel configuration.

const builtin = @import("builtin");

const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");

pub const enabled: bool = simd_config.enabled;

comptime {
    if (builtin.os.tag == .linux and builtin.abi == .gnu) {
        asm (".symver zynum_glibc_memcpy, memcpy@GLIBC_2.14");
    }
}

extern fn zynum_glibc_memcpy(dest: [*]u8, src: [*]const u8, n: usize) callconv(.c) [*]u8;

pub fn fixedCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime !enabled) return false;
    return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
}

inline fn preferCoreCopyBytes(n_bytes: usize) bool {
    return n_bytes >= 32 * 1024 and n_bytes < 128 * 1024;
}

inline fn glibcCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime builtin.os.tag != .linux or builtin.abi != .gnu) return false;
    _ = zynum_glibc_memcpy(y, x, n_bytes);
    return true;
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (n_bytes == 8 * 1024) return glibcCopyBytes(n_bytes, x, y);
    if (preferCoreCopyBytes(n_bytes)) return false;
    return fixedCopyBytes(n_bytes, x, y);
}

pub fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return copyBytes(n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    if (comptime !enabled) return false;
    if (comptime T != f32 and T != f64) return false;
    return copyUnit(T, n, x, y);
}

pub fn swapUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.swapUnitReal(T, simd_config.vectorConfig(T), n, x, y);
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.axpyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, y);
}

pub fn axpyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.axpyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, y);
}

pub fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.axpbyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
}

pub fn axpbyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.axpbyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime !enabled) return null;
    return fixed_simd.dotUnitReal(T, simd_config.vectorConfig(T), n, x, y);
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime !enabled) return null;
    return fixed_simd.dotUnitComplex(T, simd_config.vectorConfig(T), n, x, y, conjx);
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
}
