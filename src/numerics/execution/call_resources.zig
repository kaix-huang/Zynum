// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const common = @import("context.zig");
const contracts = @import("contracts.zig");
const results = @import("results.zig");
const ports = @import("resource_port.zig");
pub const AdmittedCallResources = struct {
    operation: *const common.Operation,
    pub fn maxThreads(self: @This()) usize {
        const port = self.operation.borrowedContext().resourcePort() orelse return 0;
        return port.parallel_limit;
    }
    pub fn taskCount(self: @This(), items: usize, min_items_per_task: usize) usize {
        if (items == 0) return 0;
        const minimum = @max(min_items_per_task, 1);
        return @min(self.maxThreads(), contracts.max_tasks, @max(1, items / minimum));
    }
    pub fn acquireScratch(self: @This(), comptime T: type, count: usize, report: *results.CallReport) results.ExecutionError!common.OperationScratch(T) {
        return common.acquireOperationScratch(self.operation, T, count, report);
    }
    pub fn runChecked(self: @This(), task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize, report: *results.CallReport) results.ExecutionError!void {
        report.* = .{};
        const port = self.operation.borrowedContext().resourcePort() orelse return error.Unsupported;
        _ = std.math.cast(u64, count) orelse return error.ExtentOverflow;
        if (count > contracts.max_tasks) return error.Unsupported;
        if (count == 0) return;
        var control = try self.acquireScratch(ports.TaskBatch, 1, report);
        defer control.release(report);
        control.values[0] = .{ .task_fn = task_fn, .tasks = tasks, .count = count };
        try port.runSync(&control.values[0], report);
    }

    pub fn runTypedChecked(self: @This(), comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task, report: *results.CallReport) results.ExecutionError!void {
        report.* = .{};
        if (self.operation.borrowedContext().resourcePort() == null) return error.Unsupported;
        if (tasks.len == 0) return;
        if (tasks.len > contracts.max_tasks) return error.Unsupported;
        const Adapter = struct {
            fn invoke(raw: *const anyopaque, index: usize) void {
                const values: [*]const Task = @ptrCast(@alignCast(raw));
                task_fn(values[index]);
            }
        };
        var copied = try self.acquireScratch(Task, tasks.len, report);
        defer copied.release(report);
        @memcpy(copied.values, tasks);
        try self.runChecked(Adapter.invoke, @ptrCast(copied.values.ptr), copied.values.len, report);
    }
};
pub fn admittedResources(operation: *const common.Operation) AdmittedCallResources {
    return .{ .operation = operation };
}
