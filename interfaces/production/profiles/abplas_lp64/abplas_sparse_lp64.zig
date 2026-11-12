// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional generated standard Sparse frontend shares the existing unified Core.
const Core = @import("zynum-core");
const Sparse = @import("sparse-interface-owner");
comptime {
    if (Sparse.Provider.OwnerIdentity != Core) @compileError("standard Sparse and unified interfaces require one actual Core module");
    _ = @import("zynum-abplas-base-exports");
    _ = @import("zynum-standard-sparse-exports");
}
