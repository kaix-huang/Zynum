// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! EVD exports borrow the already configured mathematical and execution owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const evd = @import("zynum-lapacke-evd-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or evd.OwnerIdentity != Core) @compileError("EVD must borrow the configured Core");
    if (evd.BindingOwner != binding) @compileError("EVD must borrow the existing interface configuration owner");
    if (WireInt != evd.WireInt) @compileError("EVD and base integer profiles differ");
    _ = base;
    _ = evd;
}
