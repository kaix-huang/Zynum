// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Optional existing LP64 BLAS ABI over the exact combined numerical owner.
//! This root adds declarations only; it creates no execution or resource owner.
const core = @import("zynum-core");
const abi = @import("zynum-blas-abi");

comptime {
    if (abi.OwnerIdentity != core)
        @compileError("combined BLAS requires the same actual Core module");
    _ = abi.fortran;
    _ = abi.cblas;
    _ = @import("zynum-abplas-base-exports");
}
