// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! GVD exports borrow the already configured mathematical and execution owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const gvd = @import("zynum-lapacke-gvd-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or gvd.OwnerIdentity != Core) @compileError("GVD must borrow the configured Core");
    if (gvd.BindingOwner != binding) @compileError("GVD must borrow the existing interface configuration owner");
    if (WireInt != gvd.WireInt) @compileError("GVD and base integer profiles differ");
    _ = base;
    _ = gvd;
}
