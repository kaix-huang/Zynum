// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const object_format_sections = @import("kernels/isolated/object_format_sections.zig");

comptime {
    _ = @import("kernels/isolated/x86_64_structured_object.zig");
}

var enabled: u8 linksection(object_format_sections.writable_data) = 0;

comptime {
    @export(&enabled, .{ .name = "zynum_internal_x86_64_structured_enabled", .visibility = .hidden });
}
