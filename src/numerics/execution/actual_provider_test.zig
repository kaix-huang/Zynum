// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicitly selected local Heteris integration evidence, not math qualification.
const std = @import("std");
const common = @import("context.zig");
const resources = @import("call_resources.zig");
const results = @import("results.zig");
const c = @import("heteris-provider-c");
test "actual original Runtime owns task writes and shared Context close isolation" {
    const desc: c.heteris_rp_cpu_create_desc_v1 = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_cpu_create_desc_v1), .compute_workers = 1, .managed_thread_limit = 2, .max_tasks = 8, .max_dependencies = 8, .host_bytes_limit = 8192, .task_scratch_bytes_limit = 0, .reserved = .{ 0, 0 } };
    var root: c.heteris_rp_token = undefined;
    try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_create_v1(&desc, &root));
    var source: c.heteris_runtime_provider_v1 = undefined;
    try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_export_provider_v1(root, &source));
    var report: results.CallReport = .{};
    var first = try common.Context.initIntegrated(&source, .{}, &report);
    var second = try common.Context.initIntegrated(&source, .{}, &report);
    const Task = struct {
        value: *u32,
        caller: std.Thread.Id,
        fn execute(self: @This()) void {
            if (std.Thread.getCurrentId() == self.caller) @panic("integrated task executed on submitting caller");
            self.value.* += 1;
        }
    };
    var values = [_]u32{ 0, 0, 0 };
    var operation = try first.begin();
    const call = resources.admittedResources(&operation);
    var tasks = try call.acquireScratch(Task, 3, &report);
    for (tasks.values, 0..) |*task, index| task.* = .{ .value = &values[index], .caller = std.Thread.getCurrentId() };
    const Adapter = struct {
        fn run(raw: *const anyopaque, index: usize) void {
            const list: [*]const Task = @ptrCast(@alignCast(raw));
            Task.execute(list[index]);
        }
    };
    try call.runChecked(Adapter.run, tasks.values.ptr, tasks.values.len, &report);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 1 }, &values);
    operation.end();
    _ = first.requestClose();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, first.tryProgressClose(&report));
    tasks.release(&report);
    try std.testing.expectEqual(common.CloseResult.closed, first.tryProgressClose(&report));
    try std.testing.expectEqual(@as(i32, c.HETERIS_BUSY), c.heteris_cpu_runtime_destroy_v1(root));
    var other = try second.begin();
    const task = [_]Task{.{ .value = &values[0], .caller = std.Thread.getCurrentId() }};
    try resources.admittedResources(&other).runTypedChecked(Task, Task.execute, &task, &report);
    try std.testing.expectEqual(@as(u32, 2), values[0]);
    other.end();
    _ = second.requestClose();
    try std.testing.expectEqual(common.CloseResult.closed, second.tryProgressClose(&report));
    try std.testing.expectEqual(@as(i32, 0), source.api[0].provider_release.?(source.provider));
    try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_destroy_v1(root));
}
