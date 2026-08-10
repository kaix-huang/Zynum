// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Main-module side of the isolated-kernel task bridge.
//!
//! All isolated objects submit work to the one process-wide `std.Io` pool.

const core_pool = @import("../../core/execution/thread_pool.zig");

const Context = struct {
    task_fn: core_pool.TaskFn,
    tasks: *const anyopaque,
};

fn runTask(raw_context: *const anyopaque, index: usize) void {
    const context: *const Context = @ptrCast(@alignCast(raw_context));
    context.task_fn(context.tasks, index);
}

fn taskCount(items: usize, min_items_per_task: usize) callconv(.c) usize {
    return core_pool.taskCount(items, min_items_per_task);
}

fn runLowLatency(task_fn_address: usize, tasks: *const anyopaque, count: usize) callconv(.c) u8 {
    const context: Context = .{ .task_fn = @ptrFromInt(task_fn_address), .tasks = tasks };
    return @intFromBool(core_pool.runLowLatency(runTask, &context, count));
}

fn run(task_fn_address: usize, tasks: *const anyopaque, count: usize) callconv(.c) u8 {
    const context: Context = .{ .task_fn = @ptrFromInt(task_fn_address), .tasks = tasks };
    return @intFromBool(core_pool.run(runTask, &context, count));
}

comptime {
    @export(&taskCount, .{ .name = "zynum_internal_level1_task_count", .visibility = .hidden });
    @export(&runLowLatency, .{ .name = "zynum_internal_level1_run_low_latency", .visibility = .hidden });
    @export(&run, .{ .name = "zynum_internal_level1_run", .visibility = .hidden });
}
