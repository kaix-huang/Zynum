// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core scheduling names alias the unique execution owner. Legacy forwards are
//! reference-only and fail compilation if reached from an integrated profile.
const std = @import("std");
const execution = @import("zynum-execution");
const default_owner = execution;
const execution_owner = execution.owner;
const options = @import("zynum-build-options");
pub const max_tasks = execution.max_tasks;
pub const TaskFn = execution.TaskFn;
pub const WorkerCacheDomain = execution.WorkerCacheDomain;
pub const AdmittedCallResources = execution.AdmittedCallResources;
pub const admittedResources = execution.admittedResources;
pub const CallResources = execution.CallResources;
fn referenceResources() execution.reference.ReferenceCallResources {
    if (comptime !options.reference_compat) @compileError("legacy scheduling requires reference_compat");
    return .legacy_default;
}
pub fn taskCount(items: usize, minimum: usize) usize {
    return referenceResources().taskCount(items, minimum);
}
pub fn run(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return referenceResources().run(task_fn, tasks, count);
}
pub fn runLowLatency(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return referenceResources().runLowLatency(task_fn, tasks, count);
}
pub fn runQueued(task_fn: TaskFn, tasks: *const anyopaque, count: usize, limit: usize) bool {
    return referenceResources().runQueued(task_fn, tasks, count, limit);
}
pub fn runTyped(comptime T: type, comptime task_fn: fn (T) void, tasks: []const T) bool {
    return referenceResources().runTyped(T, task_fn, tasks);
}
pub fn runTypedLowLatency(comptime T: type, comptime task_fn: fn (T) void, tasks: []const T) bool {
    return referenceResources().runTypedLowLatency(T, task_fn, tasks);
}
pub fn shutdown() void {
    if (comptime !options.reference_compat) @compileError("legacy shutdown requires reference_compat");
    execution.reference.get().drainAndReset();
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
