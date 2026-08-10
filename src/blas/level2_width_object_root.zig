// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Build root for the isolated x86_64 Level 2 width object.

const object_format_sections = @import("kernels/isolated/object_format_sections.zig");

comptime {
    _ = @import("kernels/isolated/x86_64_level2_width_object.zig");
}

var enabled: u8 linksection(object_format_sections.writable_data) = 1;

comptime {
    @export(&enabled, .{
        .name = "zynum_internal_x86_64_level2_width_enabled",
        .visibility = .hidden,
    });
}
