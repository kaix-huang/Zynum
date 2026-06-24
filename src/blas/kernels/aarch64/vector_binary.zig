// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 two-vector specialized kernels.

const features = @import("features.zig");

const enable_sve_dcopy = false;
const enable_asimd_dcopy = true;
const enable_asimd_daxpy = false;
const enable_asimd_ddot = false;
const enable_sme_ddot = false;
const asimd_level1_min_len = 256 * 1024;

extern fn zynum_blas_asimd_dcopy_f64(n: usize, x: [*]const f64, y: [*]f64) callconv(.c) void;
extern fn zynum_blas_sve_dcopy_f64(n: usize, x: [*]const f64, y: [*]f64) callconv(.c) void;
extern fn zynum_blas_asimd_daxpy_f64(n: usize, alpha: f64, x: [*]const f64, y: [*]f64) callconv(.c) void;
extern fn zynum_blas_asimd_ddot_f64(n: usize, x: [*]const f64, y: [*]const f64) callconv(.c) f64;
extern fn zynum_blas_sme_ddot_f64(n: usize, x: [*]const f64, y: [*]const f64) callconv(.c) f64;

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    if (comptime enable_sve_dcopy and features.has_sve) {
        if (T == f64 and n >= 16) {
            zynum_blas_sve_dcopy_f64(n, x, y);
            return true;
        }
    }
    if (comptime !enable_asimd_dcopy) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (n < 16 or n >= 256 * 1024) return false;
    zynum_blas_asimd_dcopy_f64(n, x, y);
    return true;
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime !enable_asimd_daxpy) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (n < asimd_level1_min_len) return false;
    zynum_blas_asimd_daxpy_f64(n, alpha, x, y);
    return true;
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime enable_sme_ddot and features.has_sme2) {
        if (T == f64 and n >= 512 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSm();
            const result = zynum_blas_sme_ddot_f64(n, x, y);
            const stopped_result_bits = sm_state.stopSmRetU64(@bitCast(result));
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime !enable_asimd_ddot) return null;
    if (T != f64) return null;
    if (comptime !features.has_asimd) return null;
    if (n < asimd_level1_min_len) return null;
    return zynum_blas_asimd_ddot_f64(n, x, y);
}
