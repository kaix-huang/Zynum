// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! GELSS exports borrow the already configured mathematical and execution owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const gelss = @import("zynum-lapacke-gelss-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or gelss.OwnerIdentity != Core) @compileError("GELSS must borrow the configured Core");
    if (gelss.BindingOwner != binding) @compileError("GELSS must borrow the existing interface configuration owner");
    if (WireInt != gelss.WireInt) @compileError("GELSS and base integer profiles differ");
    _ = base;
    _ = gelss;
}
