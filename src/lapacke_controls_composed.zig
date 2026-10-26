// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Configuration forwards share the exact binding used by the math exports.
const Core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");
const base = @import("zynum-lapacke-base-exports");
const controls = @import("zynum-lapacke-controls-exports");
pub const WireInt = base.WireInt;
comptime {
    if (binding.Provider.OwnerIdentity != Core or controls.OwnerIdentity != Core) @compileError("LAPACKE controls must borrow the configured Core");
    if (controls.BindingOwner != binding) @compileError("LAPACKE controls must share the math configuration owner");
    if (WireInt != controls.WireInt) @compileError("LAPACKE controls and base integer profiles differ");
    _ = base;
    _ = controls;
}
