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
    return prefetchedReduction(false, T, cfg, n, x);
}

/// Complex ASUM reuses a real component stream but has a distinct measured
/// width preference from public real ASUM on AVX-512F/FMA targets.
pub fn asumUnitComplexComponents(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    const cfg = if (comptime simd_config.has_avx512_width and profile.preferAvx2WidthComplexAsum(T))
        simd_config.avx2WidthVectorConfig(T)
    else
        simd_config.asumVectorConfig(T);
    return prefetchedReduction(false, T, cfg, n, x);
}

fn fastReduction(comptime norm: bool, comptime T: type, comptime cfg: fixed_simd.Config, n: usize, x: [*]const T) ?T {
    if (comptime !norm) return fixed_simd.asumUnitReal(T, cfg, n, x);
    return if (T == f32) fixed_simd.nrm2UnitRealFastF32(cfg, n, x) else fixed_simd.nrm2UnitRealFastF64(cfg, n, x);
}

fn prefetchedReduction(comptime norm: bool, comptime T: type, comptime cfg: fixed_simd.Config, n: usize, x: [*]const T) ?T {
    if (comptime !norm and T == f64 and simd_config.capability == .x86_64_avx2_fma) {
        // Additional independent sums help the small cache-resident stream;
        // larger streams did not show a consistent benefit in isolated sweeps.
        if (n >= 1024 and n <= 8192) {
            const wider = comptime blk: {
                var selected = cfg;
                selected.unroll_vectors = 8;
                break :blk selected;
            };
            return prefetchedReductionConfigured(norm, T, wider, n, x);
        }
    }
    return prefetchedReductionConfigured(norm, T, cfg, n, x);
}

fn prefetchedReductionConfigured(comptime norm: bool, comptime T: type, comptime cfg: fixed_simd.Config, n: usize, x: [*]const T) ?T {
    if (comptime simd_config.capability == .x86_64_avx2_fma) {
        // Single-core cache/DRAM sweeps found T0 helpful in these regimes,
        // while intermediate LLC-resident streams regressed with prefetch.
        if (norm and n >= 256 * 1024 / @sizeOf(T) and n <= 2 * 1024 * 1024 / @sizeOf(T)) {
            const ahead = comptime blk: {
                var selected = cfg;
                selected.prefetch_bytes = 2048;
                break :blk selected;
            };
            return fastReduction(norm, T, ahead, n, x);
        }
        if (n >= 32 * 1024 * 1024 / @sizeOf(T)) {
            const ahead = comptime blk: {
                var selected = cfg;
                selected.prefetch_bytes = 4096;
                break :blk selected;
            };
            return fastReduction(norm, T, ahead, n, x);
        }
    }
    return fastReduction(norm, T, cfg, n, x);
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    const norm_cfg = comptime blk: {
        var cfg = simd_config.vectorConfig(T);
        // Eight independent FMA chains cover latency without exceeding AVX2's
        // sixteen vector registers in the normal-range sum-of-squares loop.
        if (simd_config.capability == .x86_64_avx2_fma) cfg.unroll_vectors = 8;
        break :blk cfg;
    };
    if (comptime T == f32 or (T == f64 and simd_config.capability == .x86_64_avx2_fma)) {
        if (prefetchedReduction(true, T, norm_cfg, n, x)) |result| return result;
    }
    return fixed_simd.nrm2UnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    if (comptime simd_config.capability == .x86_64_avx2_fma) {
        if (fixed_simd.iamaxUnitBatched(T, simd_config.vectorConfig(T), n, x)) |result| return result;
    }
    return fixed_simd.iamaxUnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitComplex(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    if (!profile.preferFixedComplexIamax(n)) return null;
    if (comptime simd_config.capability == .x86_64_avx2_fma) {
        if (fixed_simd.iamaxUnitBatched(T, simd_config.vectorConfig(T), n, x)) |result| return result;
    }
    return fixedIamaxUnitComplexCandidate(T, n, x);
}

pub fn fixedIamaxUnitComplexCandidate(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    return fixed_simd.iamaxUnitComplex(T, simd_config.vectorConfig(T), n, x);
}
