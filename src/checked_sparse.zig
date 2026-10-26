// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Additive artifact root: both interfaces share ABI types, adapters and owner.
// Select this combined artifact instead of loading two independent runtimes.
comptime {
    _ = @import("checked/exports.zig");
    _ = @import("checked/sparse_exports.zig");
    if (@import("zynum-checked-options").checked_complex_cholesky)
        _ = @import("checked/cholesky_exports.zig");
}

// Shared declaration/conversion owner for optional generated callback frontends.
const implementation = @import("checked/implementation.zig");
pub const Wire = implementation.Wire;
pub const readBuffer = implementation.readBuffer;
pub const writeBuffer = implementation.writeBuffer;
pub const countElements = implementation.countElements;
pub const statusCode = implementation.statusCode;
pub const spmvWithIndexTypesAdmitted = implementation.spmvWithIndexTypesAdmitted;
pub const spmmWithIndexTypesAdmitted = implementation.spmmWithIndexTypesAdmitted;
