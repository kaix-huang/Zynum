// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 1 binary-vector kernel configuration.

const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const tuning = @import("../../../shared/vector/tuning.zig");

pub const enabled: bool = simd_config.enabled;
const profile = tuning.active.x86_64;

pub fn fixedCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime !enabled) return false;
    return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (profile.preferCoreCopy(n_bytes)) return false;
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
    const cfg = if (comptime simd_config.has_avx512_width and profile.preferAvx2WidthAxpy())
        simd_config.avx2WidthVectorConfig(T)
    else
        simd_config.vectorConfig(T);
    return fixed_simd.axpyUnitReal(T, cfg, n, alpha, x, y);
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
    const cfg = if (comptime simd_config.has_avx512_width and profile.preferAvx2WidthDot(T))
        simd_config.avx2WidthVectorConfig(T)
    else
        simd_config.vectorConfig(T);
    return fixed_simd.dotUnitReal(T, cfg, n, x, y);
}

pub fn dotF32AccF64Unit(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !enabled) return null;
    if (!profile.preferFixedDotF32AccF64(n)) return null;
    return fixedDotF32AccF64UnitCandidate(n, x, y);
}

pub fn fixedDotF32AccF64UnitCandidate(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !enabled) return null;
    return fixed_simd.dotF32AccF64Unit(simd_config.vectorConfig(f64), n, x, y);
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime !enabled) return null;
    return fixed_simd.dotUnitComplex(T, simd_config.vectorConfig(T), n, x, y, conjx);
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
}

pub fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !enabled or !profile.enable_fixed_rotm) return false;
    return fixedRotmUnitRealCandidate(T, n, x, y, flag, h11, h21, h12, h22);
}

pub fn fixedRotmUnitRealCandidate(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.rotmUnitReal(T, simd_config.vectorConfig(T), n, x, y, flag, h11, h21, h12, h22);
}
