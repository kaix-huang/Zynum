// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reference-only owner. The integrated root never imports this file.
const std = @import("std");
pub const owner = @import("owner.zig");
pub const policy = @import("policy.zig");
pub const context = @import("context.zig");
pub const test_support = @import("test_support.zig");
const adapter = @import("legacy_adapter.zig");
const legacy = @import("legacy_services.zig");
extern fn zynum_internal_reference_release_worker_caches() callconv(.c) void;
fn releaseWorkerCaches() void {
    zynum_internal_reference_release_worker_caches();
}
const cache_domain: owner.WorkerCacheDomain = .{ .release_current_worker = releaseWorkerCaches };
pub fn defaultCacheDomain() *const owner.WorkerCacheDomain {
    return &cache_domain;
}
var default_executor: owner.Executor = blk: {
    var executor = owner.Executor.init(std.heap.c_allocator, .{ .max_threads = policy.maxThreads, .helper_count = policy.helperThreadCount, .configure_thread = policy.configureWorkerThread, .worker_stack_size = policy.worker_stack_size });
    executor.default_cache_domain = &cache_domain;
    break :blk executor;
};
var default_context = context.Context.initLegacyBorrowed(adapter.services(&default_executor, .immortal_default, &cache_domain));
pub fn services() legacy.LegacyServices {
    return adapter.services(&default_executor, .immortal_default, &cache_domain);
}
pub fn get() *owner.Executor {
    return &default_executor;
}
pub fn getContext() *context.Context {
    return &default_context;
}
pub fn beginDefault() context.Operation {
    return default_context.begin() catch @panic("default Context admission failed");
}
