// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 single-vector specialized kernels.

const features = @import("features.zig");

const enable_sme_dscal = false;
const enable_sve_dscal = true;
const enable_asimd_dscal = false;
const enable_asimd_dasum = false;
const enable_sme_dasum = false;
const asimd_level1_min_len = 256 * 1024;

extern fn zynum_blas_asimd_dscal_f64(n: usize, alpha: f64, x: [*]f64) callconv(.c) void;
extern fn zynum_blas_sve_dscal_f64(n: usize, alpha: f64, x: [*]f64) callconv(.c) void;
extern fn zynum_blas_sme_dscal_f64(n: usize, alpha: f64, x: [*]f64) callconv(.c) void;
extern fn zynum_blas_asimd_dasum_f64(n: usize, x: [*]const f64) callconv(.c) f64;
extern fn zynum_blas_sme_dasum_f64(n: usize, x: [*]const f64) callconv(.c) f64;

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime enable_sme_dscal and features.has_sme) {
        if (T == f64 and n >= 256 * 1024 and features.streamingVectorBytes() == 64) {
            zynum_blas_sme_dscal_f64(n, alpha, x);
            return true;
        }
    }
    if (comptime enable_sve_dscal and features.has_sve) {
        if (T == f64 and n >= 16) {
            zynum_blas_sve_dscal_f64(n, alpha, x);
            return true;
        }
    }
    if (comptime !enable_asimd_dscal) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (n < asimd_level1_min_len) return false;
    zynum_blas_asimd_dscal_f64(n, alpha, x);
    return true;
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime enable_sme_dasum and features.has_sme) {
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            return zynum_blas_sme_dasum_f64(n, x);
        }
    }
    if (comptime !enable_asimd_dasum) return null;
    if (T != f64) return null;
    if (comptime !features.has_asimd) return null;
    if (n < asimd_level1_min_len) return null;
    return zynum_blas_asimd_dasum_f64(n, x);
}
