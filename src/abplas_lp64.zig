// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Finite LP64 LAPACKE and checked Sparse exports over one numerical owner.
//! Select this artifact when both interfaces must share default execution state.
const core = @import("zynum-core");
const binding = @import("lapacke-interface-owner");

comptime {
    if (binding.Provider.OwnerIdentity != core)
        @compileError("combined LAPACKE/Sparse requires one actual Core module");
    _ = @import("zynum-lapacke-exports");
    _ = @import("zynum-checked-sparse-exports");
}
