// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("owner.zig");
const common = @import("context.zig");
const contracts = @import("contracts.zig");
const legacy = @import("legacy_services.zig");
fn mutable(raw: *anyopaque) *owner.Executor {
    return @ptrCast(@alignCast(raw));
}
fn immutable(raw: *const anyopaque) *const owner.Executor {
    return @ptrCast(@alignCast(raw));
}
fn allocator(raw: *const anyopaque) std.mem.Allocator {
    return immutable(raw).allocator;
}
fn maxThreads(raw: *const anyopaque) usize {
    return immutable(raw).policy.max_threads();
}
fn taskCount(raw: *const anyopaque, items: usize, minimum: usize) usize {
    return immutable(raw).taskCount(items, minimum);
}
fn run(raw: *anyopaque, batch: contracts.Batch, mode: contracts.RunMode, limit: usize) bool {
    const executor = mutable(raw);
    var bound = batch;
    if (bound.cache_domain == null) bound.cache_domain = executor.default_cache_domain;
    return switch (mode) {
        .ordinary => executor.run(bound),
        .low_latency => executor.runLowLatency(bound),
        .queued => executor.runQueued(bound, limit),
    };
}
fn drain(raw: *anyopaque) void {
    mutable(raw).drainAndReset();
}
const vtable: legacy.LegacyServices.VTable = .{ .allocator = allocator, .max_threads = maxThreads, .task_count = taskCount, .run = run, .drain_owned = drain };
pub fn services(executor: *owner.Executor, lifetime: legacy.LegacyLifetime, domain: ?*const contracts.WorkerCacheDomain) legacy.LegacyServices {
    return .{ .userdata = executor, .vtable = &vtable, .lifetime = lifetime, .default_cache_domain = domain };
}
pub const ReferenceContext = struct {
    executor: owner.Executor,
    context: common.Context,
    /// Initializes self-pointers in final storage; never move the live wrapper.
    pub fn initInPlace(self: *@This(), alloc: std.mem.Allocator, policy: contracts.Policy, lifetime: legacy.LegacyLifetime, domain: ?*const contracts.WorkerCacheDomain) void {
        self.executor = owner.Executor.init(alloc, policy);
        self.executor.default_cache_domain = domain;
        self.context = common.Context.initLegacyBorrowed(services(&self.executor, lifetime, domain));
    }
    pub fn borrowedContext(self: *@This()) *common.Context {
        return &self.context;
    }
};
