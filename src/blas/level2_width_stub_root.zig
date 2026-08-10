// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! ABI-only disabled root for the isolated x86_64 Level 2 width object.

const object_format_sections = @import("kernels/isolated/object_format_sections.zig");
const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");

var enabled: u8 linksection(object_format_sections.writable_data) = 0;

fn execute(_: *abi.Request) callconv(.c) u8 {
    return 0;
}

comptime {
    @export(&enabled, .{
        .name = "zynum_internal_x86_64_level2_width_enabled",
        .visibility = .hidden,
    });
    @export(&execute, .{
        .name = "zynum_internal_x86_64_level2_width_execute",
        .visibility = .hidden,
    });
}
