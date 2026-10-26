// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Reference LAPACK 3.12.1 GELSD boundary policy; not an exported routine.
//! Numerical cores retain their own checked contracts. NaN/high-work behavior,
//! workspace validation and diagnostic delivery belong to the eventual adapter.
const std = @import("std");

/// DLALSD's finite-input cutoff convention, including its singleton shortcut.
/// Call after shape validation. NaN is deliberately not converted into a finite
/// tolerance; this helper does not authorize a checked core to process NaNs.
pub fn effectiveRcond(comptime T: type, min_dimension: usize, rcond: T) T {
    if (T != f32 and T != f64) @compileError("GELSD requires real f32/f64");
    if (min_dimension == 1) return 0;
    return if (rcond <= 0 or rcond >= 1) std.math.floatEps(T) * 0.5 else rcond;
}

/// First scalar error of LAPACKE_*gelsd_work, before workspace checks.
/// Row-major leading dimensions are checked before the Fortran scalar order,
/// including query calls. Returns C argument numbering; zero means valid here.
pub fn scalarInfo(comptime I: type, layout: c_int, m: I, n: I, nrhs: I, lda: I, ldb: I) I {
    if (I != i32 and I != i64) @compileError("GELSD index ABI must be i32/i64");
    if (layout != 101 and layout != 102) return -1;
    if (layout == 101) {
        if (lda < n) return -6;
        if (ldb < nrhs) return -8;
    }
    if (m < 0) return -2;
    if (n < 0) return -3;
    if (nrhs < 0) return -4;
    if (layout == 102) {
        if (lda < @max(1, m)) return -6;
        if (ldb < @max(1, @max(m, n))) return -8;
    }
    return 0;
}
