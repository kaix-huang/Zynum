// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Benchmark-only ABI root for experimental x86 Level 2 SIMD widths.

pub const zynum_level2_width_candidates = true;

const fortran = @import("abi/fortran.zig");
const cblas = @import("abi/cblas.zig");

comptime {
    _ = fortran;
    _ = cblas;
}
