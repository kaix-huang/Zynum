// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Compatibility namespace for classic Fortran BLAS ABI declarations.
//!
//! Consumers and tests share the named adapter owner. Numerical implementations
//! remain exclusively in its injected Core module.

pub const abi = @import("zynum-blas-abi").fortran;
