// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const contracts = @import("contracts.zig");
const common = @import("context.zig");
const legacy = @import("legacy_services.zig");
const defaults = @import("default.zig");
pub const ReferenceCallResources = union(enum) {
    admitted: *const common.Operation,
    legacy_default,
    fn services(self: @This()) legacy.LegacyServices {
        return switch (self) {
            .admitted => |op| op.legacyServices(),
            .legacy_default => defaults.services(),
        };
    }
    pub fn scratchAllocator(self: @This()) ?std.mem.Allocator {
        return switch (self) {
            .admitted => |op| op.legacyAllocator(),
            .legacy_default => null,
        };
    }
    pub fn maxThreads(self: @This()) usize {
        const s = self.services();
        return s.vtable.max_threads(s.userdata);
    }
    pub fn taskCount(self: @This(), items: usize, minimum: usize) usize {
        const s = self.services();
        return s.vtable.task_count(s.userdata, items, minimum);
    }
    fn execute(self: @This(), task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize, mode: contracts.RunMode, limit: usize) bool {
        const s = self.services();
        return s.vtable.run(s.userdata, .{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = s.default_cache_domain }, mode, limit);
    }
    pub fn run(self: @This(), task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize) bool {
        return self.execute(task_fn, tasks, count, .ordinary, 0);
    }
    pub fn runLowLatency(self: @This(), task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize) bool {
        return self.execute(task_fn, tasks, count, .low_latency, 0);
    }
    pub fn runQueued(self: @This(), task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize, limit: usize) bool {
        return self.execute(task_fn, tasks, count, .queued, limit);
    }
    pub fn runTyped(self: @This(), comptime T: type, comptime task: fn (T) void, values: []const T) bool {
        const Adapter = struct {
            fn call(raw: *const anyopaque, index: usize) void {
                const p: [*]const T = @ptrCast(@alignCast(raw));
                task(p[index]);
            }
        };
        return self.run(Adapter.call, values.ptr, values.len);
    }
    pub fn runTypedLowLatency(self: @This(), comptime T: type, comptime task: fn (T) void, values: []const T) bool {
        const Adapter = struct {
            fn call(raw: *const anyopaque, index: usize) void {
                const p: [*]const T = @ptrCast(@alignCast(raw));
                task(p[index]);
            }
        };
        return self.runLowLatency(Adapter.call, values.ptr, values.len);
    }
};
