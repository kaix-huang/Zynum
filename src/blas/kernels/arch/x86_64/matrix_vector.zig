// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 2 matrix-vector kernel configuration.

const std = @import("std");
const builtin = @import("builtin");
const simd_config = @import("simd_config.zig");
const isolated_width = @import("../../isolated/x86_64_level2_width_bridge.zig");
const fixed_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const level2_tuning = @import("../../shared/matrix_vector/tuning.zig");
const types = @import("../../../types.zig");

const BlasInt = types.BlasInt;

pub const enabled: bool = simd_config.enabled;

pub fn candidateDispatch(is_test: bool, has_avx512_width: bool, profile_enabled: bool) bool {
    return !is_test and has_avx512_width and profile_enabled;
}

comptime {
    const expected = [_]bool{ false, false, false, true, false, false, false, false };
    var index: usize = 0;
    for (.{ false, true }) |is_test| {
        for (.{ false, true }) |has_avx512_width| {
            for (.{ false, true }) |profile_enabled| {
                if (candidateDispatch(is_test, has_avx512_width, profile_enabled) != expected[index]) {
                    @compileError("x86 Level 2 width candidate dispatch truth table changed");
                }
                index += 1;
            }
        }
    }
}

const isolated_gemv_width_enabled = candidateDispatch(
    builtin.is_test,
    simd_config.has_avx512_width,
    level2_tuning.active.gemv.enable_x86_narrow_width,
);
const isolated_rank_width_enabled = candidateDispatch(
    builtin.is_test,
    simd_config.has_avx512_width,
    level2_tuning.active.rank_update.enable_x86_narrow_width,
);

fn gemvRealConfig(comptime T: type) fixed_simd.Config {
    if (level2_tuning.active.gemv.enable_x86_narrow_width and
        simd_config.has_avx512_width and T == f32)
    {
        return simd_config.matrixNarrowConfig(T);
    }
    return simd_config.matrixConfig(T);
}

fn gemvComplexConfig(comptime T: type) fixed_simd.Config {
    if (level2_tuning.active.gemv.enable_x86_narrow_width and simd_config.has_avx512_width) {
        return simd_config.matrixComplexNarrowConfig(T);
    }
    return simd_config.matrixComplexConfig(T);
}

fn gerRealConfig(comptime T: type) fixed_simd.Config {
    if (level2_tuning.active.rank_update.enable_x86_narrow_width and
        simd_config.has_avx512_width and T == f32)
    {
        return simd_config.matrixNarrowConfig(T);
    }
    return simd_config.matrixConfig(T);
}

fn gerComplexConfig(comptime T: type) fixed_simd.Config {
    if (level2_tuning.active.rank_update.enable_x86_narrow_width and simd_config.has_avx512_width) {
        return simd_config.matrixComplexNarrowConfig(T);
    }
    return simd_config.matrixComplexConfig(T);
}

comptime {
    if (!level2_tuning.active.gemv.enable_x86_narrow_width) {
        if (!std.meta.eql(gemvRealConfig(f32), simd_config.matrixConfig(f32)) or
            !std.meta.eql(gemvRealConfig(f64), simd_config.matrixConfig(f64)) or
            !std.meta.eql(gemvComplexConfig(types.ComplexF32), simd_config.matrixComplexConfig(types.ComplexF32)) or
            !std.meta.eql(gemvComplexConfig(types.ComplexF64), simd_config.matrixComplexConfig(types.ComplexF64)))
        {
            @compileError("default x86 Level 2 GEMV profile must use the production matrix configuration");
        }
    }
    if (!level2_tuning.active.rank_update.enable_x86_narrow_width) {
        if (!std.meta.eql(gerRealConfig(f32), simd_config.matrixConfig(f32)) or
            !std.meta.eql(gerRealConfig(f64), simd_config.matrixConfig(f64)) or
            !std.meta.eql(gerComplexConfig(types.ComplexF32), simd_config.matrixComplexConfig(types.ComplexF32)) or
            !std.meta.eql(gerComplexConfig(types.ComplexF64), simd_config.matrixComplexConfig(types.ComplexF64)))
        {
            @compileError("default x86 Level 2 GER profile must use the production matrix configuration");
        }
    }
}

fn supportsComplexTaskBase(comptime T: type, comptime cfg: fixed_simd.Config, m: usize, n: usize, lda: BlasInt) bool {
    if (comptime !enabled) return false;
    if (T != types.ComplexF32 and T != types.ComplexF64) return false;
    if (m != 512 or lda <= 0 or n == 0) return false;
    const min_rows = (cfg.lane_count + 1) / 2;
    const work = m *| n;
    if (m < min_rows or work < cfg.min_work) return false;
    if (cfg.max_work != 0 and work > cfg.max_work) return false;
    return true;
}

fn supportsComplexNoTransTask(comptime T: type, comptime cfg: fixed_simd.Config, m: usize, n: usize, lda: BlasInt) bool {
    if (!supportsComplexTaskBase(T, cfg, m, n, lda)) return false;
    if (T == types.ComplexF32) return n == 64;
    if (T == types.ComplexF64) return n == 48 or n == 52;
    return false;
}

fn supportsComplexTransTask(comptime T: type, comptime cfg: fixed_simd.Config, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (do_conj) return false;
    if (!supportsComplexTaskBase(T, cfg, m, n, lda)) return false;
    return n == 64;
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
    if (comptime !enabled) return false;
    if (comptime isolated_gemv_width_enabled and T == f32) {
        if (isolated_width.tryGemvTransUnit(T, m, n, alpha, a, lda, x, y, false)) return true;
    }
    return fixed_simd.gemvTransUnitReal(T, gemvRealConfig(T), m, n, alpha, a, lda, x, y);
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
    if (comptime !enabled) return false;
    if (comptime isolated_gemv_width_enabled and T == f32) {
        if (isolated_width.tryGemvTransFull(T, m, n, alpha, a, lda, x, beta, y)) return true;
    }
    return fixed_simd.gemvTransFullUnitReal(T, gemvRealConfig(T), m, n, alpha, a, lda, x, beta, y);
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
    if (comptime !enabled) return false;
    if (comptime isolated_gemv_width_enabled and T == f32) {
        if (isolated_width.tryGemvNoTransUnit(T, m, n, alpha, a, lda, x, y)) return true;
    }
    return fixed_simd.gemvNoTransUnitReal(T, gemvRealConfig(T), m, n, alpha, a, lda, x, y);
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
    if (comptime !enabled) return false;
    if (comptime isolated_gemv_width_enabled and T == f32) {
        if (isolated_width.tryGemvNoTransFull(T, m, n, alpha, a, lda, x, beta, y)) return true;
    }
    return fixed_simd.gemvNoTransFullUnitReal(T, gemvRealConfig(T), m, n, alpha, a, lda, x, beta, y);
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
    if (comptime !enabled) return false;
    if (m == 512 and n == 512) return false;
    if (comptime isolated_gemv_width_enabled) {
        if (m *| n <= 128 * 128 and isolated_width.tryGemvNoTransUnit(T, m, n, alpha, a, lda, x, y)) return true;
    }
    return fixed_simd.gemvNoTransUnitComplex(T, gemvComplexConfig(T), m, n, alpha, a, lda, x, y);
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
    if (T != types.ComplexF32 and T != types.ComplexF64) return false;
    const cfg = comptime gemvComplexConfig(T);
    if (!supportsComplexNoTransTask(T, cfg, m, n, lda)) return false;
    @memset(y_delta[0..m], .{ .re = 0, .im = 0 });
    if (comptime isolated_gemv_width_enabled) {
        if (m *| n <= 128 * 128 and isolated_width.tryGemvNoTransUnit(T, m, n, alpha, a, lda, x, y_delta)) return true;
    }
    return fixed_simd.gemvNoTransUnitComplex(T, cfg, m, n, alpha, a, lda, x, y_delta);
}

pub fn supportsGemvNoTransTaskUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt) bool {
    if (T != types.ComplexF32 and T != types.ComplexF64) return false;
    const cfg = comptime gemvComplexConfig(T);
    return supportsComplexNoTransTask(T, cfg, m, n, lda);
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
    if (comptime !enabled) return false;
    if (m == 512 and n == 512) return false;
    if (comptime isolated_gemv_width_enabled) {
        if (m *| n <= 128 * 128 and isolated_width.tryGemvTransUnit(T, m, n, alpha, a, lda, x, y, do_conj)) return true;
    }
    return fixed_simd.gemvTransUnitComplex(T, gemvComplexConfig(T), m, n, alpha, a, lda, x, y, do_conj);
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
    if (T != types.ComplexF32 and T != types.ComplexF64) return false;
    const cfg = comptime gemvComplexConfig(T);
    if (!supportsComplexTransTask(T, cfg, m, n, lda, do_conj)) return false;
    if (comptime isolated_gemv_width_enabled) {
        if (m *| n <= 128 * 128 and isolated_width.tryGemvTransUnit(T, m, n, alpha, a, lda, x, y, do_conj)) return true;
    }
    return fixed_simd.gemvTransUnitComplex(T, cfg, m, n, alpha, a, lda, x, y, do_conj);
}

pub fn supportsGemvTransTaskUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (T != types.ComplexF32 and T != types.ComplexF64) return false;
    const cfg = comptime gemvComplexConfig(T);
    return supportsComplexTransTask(T, cfg, m, n, lda, do_conj);
}

pub fn supportsGemvNoTransUnitComplex(comptime T: type) bool {
    return enabled and (T == types.ComplexF32 or T == types.ComplexF64);
}

pub fn supportsGemvTransUnitComplex(comptime T: type) bool {
    return enabled and (T == types.ComplexF32 or T == types.ComplexF64);
}

pub fn gemvNoTransPackLenUnitReal(comptime T: type, m: usize, n: usize, lda: BlasInt) ?usize {
    if (comptime !enabled) return null;
    if (T != f32 and T != f64) return null;
    if ((m & 7) != 0 or n == 0 or lda <= 0) return null;
    return n;
}

pub fn gemvNoTransPackUnitReal(
    comptime T: type,
    n: usize,
    alpha: T,
    x: [*]const T,
    pack: []T,
) bool {
    if (comptime !enabled) return false;
    if (T != f32 and T != f64) return false;
    if (pack.len < n) return false;
    for (0..n) |j| pack[j] = alpha * x[j];
    return true;
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
    _ = scratch;
    if (comptime !enabled) return false;
    if (T != f32 and T != f64) return false;
    if ((row_count & 7) != 0 or n == 0 or lda <= 0) return false;
    return fixed_simd.gemvNoTransUnitReal(T, simd_config.matrixPackedRowsConfig(T), row_count, n, 1, a, lda, pack, y);
}

pub fn supportsGemvNoTransUnitReal(comptime T: type) bool {
    return enabled and (T == f32 or T == f64);
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
    if (comptime !enabled) return false;
    if (comptime isolated_rank_width_enabled and T == f32) {
        if (isolated_width.tryGer(T, m, n, alpha, x, y, a, lda, false)) return true;
    }
    return fixed_simd.gerUnitReal(T, gerRealConfig(T), m, n, alpha, x, y, a, lda);
}

pub fn gerUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
    conjugate_y: bool,
) bool {
    if (comptime !enabled) return false;
    return fixed_simd.gerUnitComplex(T, gerComplexConfig(T), m, n, alpha, x, y, a, lda, conjugate_y);
}

pub fn triangularAxpyUnit(
    comptime T: type,
    n: usize,
    alpha: T,
    a: [*]const T,
    x: [*]T,
) bool {
    if (comptime !enabled) return false;
    return fixed_simd.triangularAxpyUnit(T, simd_config.matrixBodyConfig(T), n, alpha, a, x);
}

pub fn triangularDotUnit(
    comptime T: type,
    n: usize,
    a: [*]const T,
    x: [*]const T,
    conjugate_a: bool,
    result: *T,
) bool {
    if (comptime !enabled) return false;
    return fixed_simd.triangularDotUnit(T, simd_config.matrixBodyConfig(T), n, a, x, conjugate_a, result);
}

pub fn symmetricColumnsUnit(
    comptime T: type,
    upper: bool,
    hermitian: bool,
    n: usize,
    j0: usize,
    j1: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y_delta: [*]T,
) bool {
    if (comptime !enabled) return false;
    return fixed_simd.symmetricColumnsUnit(
        T,
        simd_config.matrixBodyConfig(T),
        upper,
        hermitian,
        n,
        j0,
        j1,
        alpha,
        a,
        lda,
        x,
        y_delta,
    );
}
