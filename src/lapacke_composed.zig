// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compose external export modules over the same configured numerical owner.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const extra = @import("zynum-lapacke-component-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or extra.OwnerIdentity != Core) @compileError("LAPACKE components must share the existing Core");
    if (WireInt != extra.WireInt) @compileError("LAPACKE component integer profiles differ");
    _ = base;
    _ = extra;
}
