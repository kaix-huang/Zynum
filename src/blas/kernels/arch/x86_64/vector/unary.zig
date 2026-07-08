// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! x86_64 BLAS Level 1 unary-vector kernel configuration.

const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const types = @import("../../../../types.zig");

pub const enabled: bool = simd_config.enabled;

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
    return fixed_simd.asumUnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !enabled) return null;
    return fixed_simd.nrm2UnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !enabled) return null;
    return fixed_simd.iamaxUnitReal(T, simd_config.vectorConfig(T), n, x);
}
