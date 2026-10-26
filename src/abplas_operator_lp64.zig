// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional generated operator frontend over the same combined package Core.
const Core = @import("zynum-core");
const Checked = @import("checked-interface-owner");
const Binding = @import("operator-interface-owner").Binding(Core, Checked);
comptime {
    if (Binding.OwnerIdentity != Core)
        @compileError("operator and LP64 interfaces require the same actual Core module");
    _ = @import("zynum-abplas-base-exports");
    _ = @import("zynum-operator-exports");
}
