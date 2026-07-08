// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 2 matrix-vector kernel configuration.

const simd_config = @import("simd_config.zig");
const fixed_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const types = @import("../../../types.zig");

const BlasInt = types.BlasInt;

pub const enabled: bool = simd_config.enabled;

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
    return fixed_simd.gemvTransUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, y);
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
    return fixed_simd.gemvTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
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
    return fixed_simd.gemvNoTransUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, y);
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
    return fixed_simd.gemvNoTransFullUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, a, lda, x, beta, y);
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
    return fixed_simd.gemvNoTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y);
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
    return fixed_simd.gemvTransUnitComplex(T, simd_config.matrixComplexConfig(T), m, n, alpha, a, lda, x, y, do_conj);
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
    return fixed_simd.gerUnitReal(T, simd_config.matrixConfig(T), m, n, alpha, x, y, a, lda);
}
