// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const execution_owner = @import("zynum-execution").owner;
const default_owner = @import("zynum-execution");
fn freeWorkerCaches() void {
    // Only SME owns persistent kernel pack allocations. Keep the baseline
    // scheduler independent of the matrix kernel graph (and its test roots).
    const builtin = @import("builtin");
    if (comptime @import("zynum-build-options").dynamic_dispatch or
        (builtin.target.cpu.arch == .aarch64 and builtin.target.cpu.features.isEnabled(@backingInt(std.Target.aarch64.Feature.sme))))
    {
        @import("../../kernels/dispatch/matrix_matrix.zig").freeCurrentThreadCaches();
    }
}

pub const max_tasks = execution_owner.max_tasks;
pub const TaskFn = execution_owner.TaskFn;
const blas_cache_domain: execution_owner.WorkerCacheDomain = .{ .release_current_worker = freeWorkerCaches };

/// Synchronous internal borrow, never an independently owned executor/allocator
/// pair. Admitted calls derive both from the same live Operation. Existing
/// internal callers retain their default services until their entrypoint is
/// migrated; this bridge does not acquire a second operation.
pub const CallResources = union(enum) {
    admitted: *const default_owner.context.Operation,
    legacy_default,

    pub fn executor(self: CallResources) *execution_owner.Executor {
        return switch (self) {
            .admitted => |operation| &operation.borrowedContext().executor,
            .legacy_default => default_owner.get(),
        };
    }

    /// A non-null allocator owns call-local kernel packs. Null preserves the
    /// existing legacy allocation/cache policy; it never selects another Context.
    pub fn scratchAllocator(self: CallResources) ?std.mem.Allocator {
        return switch (self) {
            .admitted => |operation| operation.allocator(),
            .legacy_default => null,
        };
    }

    pub fn maxThreads(self: CallResources) usize {
        return self.executor().policy.max_threads();
    }

    pub fn taskCount(self: CallResources, items: usize, min_items_per_task: usize) usize {
        return self.executor().taskCount(items, min_items_per_task);
    }

    pub fn run(self: CallResources, task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
        return self.executor().run(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain });
    }

    pub fn runLowLatency(self: CallResources, task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
        return self.executor().runLowLatency(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain });
    }

    pub fn runQueued(self: CallResources, task_fn: TaskFn, tasks: *const anyopaque, count: usize, worker_limit: usize) bool {
        return self.executor().runQueued(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain }, worker_limit);
    }

    pub fn runTyped(self: CallResources, comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task) bool {
        const Adapter = struct {
            fn call(raw: *const anyopaque, index: usize) void {
                const typed: [*]const Task = @ptrCast(@alignCast(raw));
                task_fn(typed[index]);
            }
        };
        return self.run(Adapter.call, @ptrCast(tasks.ptr), tasks.len);
    }

    pub fn runTypedLowLatency(self: CallResources, comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task) bool {
        const Adapter = struct {
            fn call(raw: *const anyopaque, index: usize) void {
                const typed: [*]const Task = @ptrCast(@alignCast(raw));
                task_fn(typed[index]);
            }
        };
        return self.runLowLatency(Adapter.call, @ptrCast(tasks.ptr), tasks.len);
    }
};

pub fn taskCount(items: usize, min_items_per_task: usize) usize {
    return default_owner.get().taskCount(items, min_items_per_task);
}
pub fn run(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return default_owner.get().run(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain });
}
pub fn runLowLatency(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return default_owner.get().runLowLatency(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain });
}
pub fn runQueued(task_fn: TaskFn, tasks: *const anyopaque, count: usize, worker_limit: usize) bool {
    return default_owner.get().runQueued(.{ .task_fn = task_fn, .tasks = tasks, .count = count, .cache_domain = &blas_cache_domain }, worker_limit);
}
pub fn runTyped(comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task) bool {
    const Adapter = struct {
        fn run(ctx: *const anyopaque, index: usize) void {
            const typed: [*]const Task = @ptrCast(@alignCast(ctx));
            task_fn(typed[index]);
        }
    };
    return run(Adapter.run, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runTypedLowLatency(comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task) bool {
    const Adapter = struct {
        fn run(ctx: *const anyopaque, index: usize) void {
            const typed: [*]const Task = @ptrCast(@alignCast(ctx));
            task_fn(typed[index]);
        }
    };
    return runLowLatency(Adapter.run, @ptrCast(tasks.ptr), tasks.len);
}

pub fn shutdown() void {
    default_owner.get().drainAndReset();
}

const WrapperTestWork = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
    accepted: bool = false,
    finished: std.atomic.Value(bool) = .init(false),
    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index == 0) {
            self.entered.store(true, .release);
            while (!self.release.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    fn run(self: *@This()) void {
        defer self.finished.store(true, .release);
        self.accepted = runLowLatency(execute, self, 2);
    }
};
test "BLAS wrappers and neutral default share admission and policy" {
    const runtime = @import("../../runtime.zig");
    const policy = default_owner.policy;
    try std.testing.expect(runtime.setMaxThreads == policy.setMaxThreads);
    try std.testing.expect(runtime.configureWorkerThread == policy.configureWorkerThread);
    try std.testing.expect(TaskFn == execution_owner.TaskFn);
    try std.testing.expect(@TypeOf(default_owner.get()) == *execution_owner.Executor);
    const before = runtime.maxThreadsOverride();
    runtime.setMaxThreads(2);
    defer {
        shutdown();
        runtime.setMaxThreads(before);
    }
    try std.testing.expectEqual(@as(usize, 2), policy.maxThreadsOverride());
    policy.setMaxThreads(3);
    try std.testing.expectEqual(@as(usize, 3), runtime.maxThreadsOverride());
    runtime.setMaxThreads(2);
    if (runtime.maxThreads() < 2) return error.SkipZigTest;
    var active: WrapperTestWork = .{};
    const caller = try std.Thread.spawn(.{}, WrapperTestWork.run, .{&active});
    defer {
        active.release.store(true, .release);
        caller.join();
    }
    while (!active.entered.load(.acquire) and !active.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(active.entered.load(.acquire));
    var refused: WrapperTestWork = .{};
    try std.testing.expect(!default_owner.get().run(.{ .task_fn = WrapperTestWork.execute, .tasks = &refused, .count = 2 }));
    for (&refused.visits) |*v| try std.testing.expectEqual(@as(u32, 0), v.load(.acquire));
    active.release.store(true, .release);
}

test "BLAS typed adapters preserve function types and exactly-once task slices" {
    const runtime = @import("../../runtime.zig");
    const Task = struct {
        value: *u32,
        fn call(t: @This()) void {
            t.value.* += 1;
        }
    };
    const before = runtime.maxThreadsOverride();
    runtime.setMaxThreads(2);
    defer {
        shutdown();
        runtime.setMaxThreads(before);
    }
    if (runtime.maxThreads() < 2) return error.SkipZigTest;
    var values = [_]u32{ 0, 0 };
    const tasks = [_]Task{ .{ .value = &values[0] }, .{ .value = &values[1] } };
    try std.testing.expect(runTyped(Task, Task.call, &tasks));
    try std.testing.expect(runTypedLowLatency(Task, Task.call, &tasks));
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, &values);
}
