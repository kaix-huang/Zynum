// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! ABI export root for the `zynum_blas` link library.
//!
//! `build.zig` uses this file as the root module for the shared and static
//! compatibility libraries. Importing both ABI modules here keeps their
//! `pub export` BLAS and CBLAS symbols reachable from the final artifact. Zig
//! tests and module consumers should prefer the testable facades in
//! `compat_fortran.zig` and `compat_cblas.zig`.

const fortran = @import("abi/fortran.zig");
const cblas = @import("abi/cblas.zig");

comptime {
    _ = fortran;
    _ = cblas;
}
