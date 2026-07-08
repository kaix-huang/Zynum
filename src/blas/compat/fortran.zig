// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Compatibility namespace for classic Fortran BLAS ABI declarations.
//!
//! Tests import `src/blas/abi/fortran.zig` directly so the test surface cannot
//! drift from the exported ABI implementation. This leaf remains as a documented
//! namespace for compatibility-focused consumers that prefer an explicit path.

pub const abi = @import("../abi/fortran.zig");
