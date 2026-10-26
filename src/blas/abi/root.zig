// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! One adapter owner per compilation configuration; numerical bodies belong to Core.
pub const OwnerIdentity = @import("zynum-core");
pub const fortran = @import("fortran.zig");
pub const cblas = @import("cblas.zig");
