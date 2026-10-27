// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const common = @import("context.zig");
const contracts = @import("contracts.zig");
const results = @import("results.zig");
const provider = @import("heteris_provider.zig");
const c = provider.c;
const Invocation = struct { task_fn: contracts.TaskFn, tasks: *const anyopaque, count: u64 };

/// Ordinary runtime checks are deliberately present in ReleaseFast too.
fn bulkRange(data: ?*anyopaque, begin: u64, end: u64) callconv(.c) i32 {
    const raw = data orelse return c.HETERIS_INVALID_ARGUMENT;
    if (@intFromPtr(raw) % @alignOf(Invocation) != 0) return c.HETERIS_INVALID_ARGUMENT;
    const invocation: *const Invocation = @ptrCast(@alignCast(raw));
    if (begin > end) return c.HETERIS_INVALID_ARGUMENT;
    if (end > invocation.count) return c.HETERIS_OUT_OF_BOUNDS;
    const first = std.math.cast(usize, begin) orelse return c.HETERIS_OVERFLOW;
    const last = std.math.cast(usize, end) orelse return c.HETERIS_OVERFLOW;
    for (first..last) |index| {
        if (index >= invocation.count) return c.HETERIS_OUT_OF_BOUNDS;
        invocation.task_fn(invocation.tasks, index);
    }
    return c.HETERIS_OK;
}
pub const AdmittedCallResources = struct {
    operation: *const common.Operation,
    pub fn maxThreads(self: @This()) usize {
        const binding = self.operation.borrowedContext().integratedBinding() orelse return 0;
        return binding.parallel_limit;
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
        const binding = self.operation.borrowedContext().integratedBinding() orelse return error.Unsupported;
        const count64 = std.math.cast(u64, count) orelse return error.ExtentOverflow;
        if (count > contracts.max_tasks) return error.Unsupported;
        if (count == 0) return;
        var control = try self.acquireScratch(Invocation, 1, report);
        defer control.release(report);
        control.values[0] = .{ .task_fn = task_fn, .tasks = tasks, .count = count64 };
        var outcome: c.heteris_rp_bulk_result_v1 = std.mem.zeroes(c.heteris_rp_bulk_result_v1);
        report.provider.stage = .submit;
        const code = binding.api.cpu_bulk_sync.?(binding.token, 0, count64, 1, binding.parallel_limit, bulkRange, &control.values[0], &outcome);
        report.provider.returned_status = code;
        report.provider.execution_status = outcome.execution_status;
        report.provider.callback_status = outcome.callback_status;
        report.provider.callbacks_started = outcome.callbacks_started;
        report.provider.callbacks_completed = outcome.callbacks_completed;
        // Preserve every raw result field even when the provider completion is malformed.
        if (outcome.abi_version != 1 or outcome.struct_size < @sizeOf(@TypeOf(outcome)) or outcome.reserved != 0 or outcome.callbacks_started > count64 or outcome.callbacks_completed > outcome.callbacks_started) provider.failStop(report, "malformed bulk completion prefix/counts");
        const expected = if (outcome.execution_status != c.HETERIS_OK) outcome.execution_status else outcome.callback_status;
        if (code != expected) provider.failStop(report, "bulk return/status mismatch");
        switch (outcome.state) {
            c.HETERIS_RP_REJECTED => {
                report.provider.state = .rejected;
                if (code == c.HETERIS_OK or outcome.callbacks_started != 0 or outcome.callbacks_completed != 0 or outcome.callback_status != c.HETERIS_OK) provider.failStop(report, "rejected bulk performed callbacks");
                return provider.mapStatus(code);
            },
            c.HETERIS_RP_QUIESCENT => {
                report.provider.state = .quiescent;
                report.output = .partial_or_invalid;
                if (outcome.callbacks_started != outcome.callbacks_completed) provider.failStop(report, "bulk returned before all callbacks completed");
                if (outcome.execution_status != c.HETERIS_OK) return error.ProviderExecutionFailed;
                if (outcome.callback_status != c.HETERIS_OK) return error.ProviderCallbackFailed;
                if (outcome.callbacks_completed != count64) provider.failStop(report, "successful bulk did not cover every task");
                report.output = .complete;
            },
            else => provider.failStop(report, "bulk returned unknown access state"),
        }
    }
    pub fn runTypedChecked(self: @This(), comptime Task: type, comptime task_fn: fn (Task) void, tasks: []const Task, report: *results.CallReport) results.ExecutionError!void {
        report.* = .{};
        if (self.operation.borrowedContext().integratedBinding() == null) return error.Unsupported;
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

test "bulk range rejects malformed bounds before reading tasks" {
    const Test = struct {
        fn call(_: *const anyopaque, _: usize) void {
            @panic("malformed range reached task");
        }
    };
    var invocation: Invocation = .{ .task_fn = Test.call, .tasks = undefined, .count = 3 };
    try std.testing.expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), bulkRange(null, 0, 1));
    try std.testing.expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), bulkRange(&invocation, 2, 1));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OUT_OF_BOUNDS), bulkRange(&invocation, 0, 4));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OK), bulkRange(&invocation, 3, 3));
}
