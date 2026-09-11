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
    if (n_bytes >= tuning.streaming_copy_min_bytes and streamCopyBytes(n_bytes, x, y)) return true;
    if (profile.preferCoreCopy(n_bytes)) return false;
    return fixedCopyBytes(n_bytes, x, y);
}

/// Caller guarantees disjoint ranges. Workers inherit the whole-copy policy,
/// rather than independently deciding from their smaller partition size.
pub fn streamCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime simd_config.capability != .x86_64_avx2_fma) return false;
    if (n_bytes < 256) return false;
    const V = @Vector(4, u64);
    var i: usize = 0;
    // Keep temporal prefix/tail stores off the non-temporal cache lines.
    while ((@intFromPtr(y + i) & 63) != 0) : (i += 1) y[i] = x[i];
    while (i + 256 <= n_bytes) : (i += 256) {
        inline for (0..8) |k| {
            const source: *align(1) const V = @ptrCast(x + i + k * 32);
            asm volatile ("vmovntdq %[value], (%[target])"
                :
                : [value] "x" (source.*),
                  [target] "r" (y + i + k * 32),
                : .{ .memory = true });
        }
    }
    while (i + 64 <= n_bytes) : (i += 64) {
        inline for (0..2) |k| {
            const source: *align(1) const V = @ptrCast(x + i + k * 32);
            asm volatile ("vmovntdq %[value], (%[target])"
                :
                : [value] "x" (source.*),
                  [target] "r" (y + i + k * 32),
                : .{ .memory = true });
        }
    }
    while (i < n_bytes) : (i += 1) y[i] = x[i];
    // Includes completion ordering in the call's cost and precedes publishing
    // worker completion through the pool.
    asm volatile ("sfence" ::: .{ .memory = true });
    return true;
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
        simd_config.reductionVectorConfig(T);
    return fixed_simd.dotUnitReal(T, cfg, n, x, y);
}

pub fn dotF32AccF64Unit(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !enabled) return null;
    const preferred = if (comptime simd_config.capability == .x86_64_avx2_fma)
        profile.preferAvx2MixedDot(n)
    else
        profile.preferFixedDotF32AccF64(n);
    if (!preferred) return null;
    return fixedDotF32AccF64UnitCandidate(n, x, y);
}

pub fn fixedDotF32AccF64UnitCandidate(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !enabled) return null;
    if (n > 65536) return fixed_simd.dotF32AccF64Unit(simd_config.vectorConfig(f64), n, x, y);
    return fixed_simd.dotF32AccF64Unit(simd_config.reductionVectorConfig(f64), n, x, y);
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime !enabled) return null;
    // Retain the resident loop and the single-precision streaming path.
    // Long double-complex AVX2 streams defer alternating signs until reduction.
    if (n > 64 * 1024 / @sizeOf(T)) {
        if (comptime @sizeOf(T) == 16 and simd_config.capability == .x86_64_avx2_fma)
            return dotUnitComplexLong(T, n, x, y, conjx);
        return fixed_simd.dotUnitComplex(T, simd_config.vectorConfig(T), n, x, y, conjx);
    }
    const cfg = comptime blk: {
        var config = simd_config.vectorConfig(T);
        config.fuse_complex_dot = simd_config.capability == .x86_64_avx2_fma;
        break :blk config;
    };
    return fixed_simd.dotUnitComplex(T, cfg, n, x, y, conjx);
}

noinline fn dotUnitComplexLong(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    return fixed_simd.dotUnitComplexDeferred(T, simd_config.vectorConfig(T), n, x, y, conjx);
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
}

pub fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !enabled) return false;
    const preferred = if (comptime simd_config.capability == .x86_64_avx2_fma)
        profile.preferAvx2Rotm(n)
    else
        profile.enable_fixed_rotm;
    if (!preferred) return false;
    return fixedRotmUnitRealCandidate(T, n, x, y, flag, h11, h21, h12, h22);
}

pub fn fixedRotmUnitRealCandidate(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.rotmUnitReal(T, simd_config.vectorConfig(T), n, x, y, flag, h11, h21, h12, h22);
}
