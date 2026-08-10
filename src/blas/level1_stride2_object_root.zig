// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Build root for the separately compiled Level 1 stride-two object.

comptime {
    _ = @import("kernels/isolated/x86_64_stride2_object.zig");
}
