// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Root source file for the testable classic Fortran BLAS compatibility module.
//!
//! The build imports `src/blas/abi/fortran.zig` directly for ABI tests. This
//! compatibility root remains as a small namespace wrapper for consumers that
//! want a stable compatibility import path without making this file the export
//! root.

const compat = @import("compat/fortran.zig");

pub const abi = compat.abi;
