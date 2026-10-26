// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional FEAST declarations composed onto the selected unified package.
const Core = @import("zynum-core");
const Feast = @import("feast-interface-owner");
comptime {
    if (Feast.Provider.OwnerIdentity != Core)
        @compileError("unified FEAST frontend requires the same actual Core module");
    _ = @import("zynum-abplas-base-exports");
    _ = @import("zynum-feast-exports");
}
