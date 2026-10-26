// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Artifact root: relative LAPACK/Sparse imports share one module owner.
// This is not an additional public Zig facade.
comptime {
    _ = @import("checked/exports.zig");
}
