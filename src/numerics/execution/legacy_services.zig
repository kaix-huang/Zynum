// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const contracts = @import("contracts.zig");
pub const LegacyLifetime = enum { owned_closable, immortal_default };
pub const LegacyServices = struct {
    userdata: *anyopaque,
    vtable: *const VTable,
    lifetime: LegacyLifetime,
    default_cache_domain: ?*const contracts.WorkerCacheDomain = null,
    pub const VTable = struct {
        allocator: *const fn (*const anyopaque) std.mem.Allocator,
        max_threads: *const fn (*const anyopaque) usize,
        task_count: *const fn (*const anyopaque, usize, usize) usize,
        run: *const fn (*anyopaque, contracts.Batch, contracts.RunMode, usize) bool,
        drain_owned: *const fn (*anyopaque) void,
    };
};
