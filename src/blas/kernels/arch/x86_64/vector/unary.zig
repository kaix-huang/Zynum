// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 1 unary-vector kernel configuration.

const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const tuning = @import("../../../shared/vector/tuning.zig");
const types = @import("../../../../types.zig");

pub const enabled: bool = simd_config.enabled;
const profile = tuning.active.x86_64;

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.scalUnitReal(T, simd_config.vectorConfig(T), n, alpha, x);
}

pub fn scalUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !enabled) return false;
    return fixed_simd.scalUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x);
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    const cfg = if (comptime simd_config.has_avx512_width and profile.preferAvx2WidthAsum(T))
        simd_config.avx2WidthVectorConfig(T)
    else
        simd_config.asumVectorConfig(T);
    return fixed_simd.asumUnitReal(T, cfg, n, x);
}

/// Complex ASUM reuses a real component stream but has a distinct measured
/// width preference from public real ASUM on AVX-512F/FMA targets.
pub fn asumUnitComplexComponents(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    const cfg = if (comptime simd_config.has_avx512_width and profile.preferAvx2WidthComplexAsum(T))
        simd_config.avx2WidthVectorConfig(T)
    else
        simd_config.asumVectorConfig(T);
    return fixed_simd.asumUnitReal(T, cfg, n, x);
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    if (comptime T == f32) {
        if (fixed_simd.nrm2UnitRealFastF32(simd_config.vectorConfig(T), n, x)) |result| return result;
    }
    return fixed_simd.nrm2UnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    return fixed_simd.iamaxUnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitComplex(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    if (!profile.preferFixedComplexIamax(n)) return null;
    return fixedIamaxUnitComplexCandidate(T, n, x);
}

pub fn fixedIamaxUnitComplexCandidate(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    return fixed_simd.iamaxUnitComplex(T, simd_config.vectorConfig(T), n, x);
}
