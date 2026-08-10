// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Benchmark-only BLAS ABI root for isolated structured Level 3 candidates.

pub const zynum_structured_object_candidates = true;

const fortran = @import("abi/fortran.zig");
const cblas = @import("abi/cblas.zig");

comptime {
    _ = fortran;
    _ = cblas;
}
