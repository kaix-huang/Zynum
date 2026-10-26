// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional generalized real-symmetric exports reuse the already selected LAPACKE owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const generalized = @import("zynum-lapacke-symmetric-generalized-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or generalized.OwnerIdentity != Core) @compileError("SYGV must borrow the configured Core");
    if (WireInt != generalized.WireInt) @compileError("SYGV and base integer profiles differ");
    _ = base;
    _ = generalized;
}
