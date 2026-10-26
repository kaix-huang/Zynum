// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Benchmark-only ABI root for experimental x86 Level 2 SIMD widths.

pub const zynum_level2_width_candidates = true;

const fortran = @import("zynum-blas-abi").fortran;
const cblas = @import("zynum-blas-abi").cblas;

comptime {
    _ = fortran;
    _ = cblas;
}
