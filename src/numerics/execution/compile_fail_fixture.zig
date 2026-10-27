// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Uninstalled negative compilation fixture; never imported by execution/root.
const std = @import("std");
const execution = @import("root.zig");
pub fn main() void {
    const mode = @import("zynum-build-options").negative_mode;
    if (comptime std.mem.eql(u8, mode, "integrated")) {
        std.mem.doNotOptimizeAway(execution.Context.initIntegrated);
    }
    if (comptime std.mem.eql(u8, mode, "factory")) {
        _ = execution.Context.initLegacyBorrowed(undefined);
    }
    if (comptime std.mem.eql(u8, mode, "namespace")) {
        _ = execution.reference.ReferenceContext;
    }
    if (comptime std.mem.eql(u8, mode, "default")) {
        _ = execution.beginDefault();
    }
    if (comptime std.mem.eql(u8, mode, "owner")) {
        _ = execution.owner.Executor;
    }
    if (comptime std.mem.eql(u8, mode, "allocator")) {
        const op: execution.Operation = undefined;
        _ = op.allocator();
    }
}
