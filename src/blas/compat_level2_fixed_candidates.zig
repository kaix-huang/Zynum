// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Benchmark-only ABI export root for experimental fixed-width Level 2 leaves.
//!
//! Select this root with `-Dlevel2-fixed-candidates=true`. It exports the same
//! public BLAS/CBLAS ABI as `compat.zig`; the root declaration is consumed by
//! the named Level 2 tuning profile and never by default library builds.

pub const zynum_level2_fixed_candidates = true;

const fortran = @import("abi/fortran.zig");
const cblas = @import("abi/cblas.zig");

comptime {
    _ = fortran;
    _ = cblas;
}
