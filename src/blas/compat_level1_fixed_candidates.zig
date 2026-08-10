// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Benchmark-only ABI export root for experimental Level 1 candidates.
//!
//! Select this root with `-Dlevel1-fixed-candidates=true`. It exports the same
//! public BLAS/CBLAS ABI as `compat.zig`; the root declaration is consumed by
//! the named Level 1 tuning profile and never by default library builds.

pub const zynum_level1_fixed_candidates = true;

const fortran = @import("abi/fortran.zig");
const cblas = @import("abi/cblas.zig");

comptime {
    _ = fortran;
    _ = cblas;
}
