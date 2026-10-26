// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional real-symmetric exports reuse the already selected LAPACKE owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const symmetric = @import("zynum-lapacke-symmetric-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or symmetric.OwnerIdentity != Core) @compileError("SYEV must borrow the configured Core");
    if (WireInt != symmetric.WireInt) @compileError("SYEV and base integer profiles differ");
    _ = base;
    _ = symmetric;
}
