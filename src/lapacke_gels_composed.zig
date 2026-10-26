// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! GELS exports borrow the already configured mathematical and execution owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const gels = @import("zynum-lapacke-gels-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or gels.OwnerIdentity != Core) @compileError("GELS must borrow the configured Core");
    if (gels.BindingOwner != binding) @compileError("GELS must borrow the existing interface configuration owner");
    if (WireInt != gels.WireInt) @compileError("GELS and base integer profiles differ");
    _ = base;
    _ = gels;
}
