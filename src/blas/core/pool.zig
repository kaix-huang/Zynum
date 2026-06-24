// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const runtime = @import("../runtime.zig");

pub const max_tasks = 64;
pub const TaskFn = *const fn (*const anyopaque, usize) void;

var io_init_state = std.atomic.Value(u8).init(0);
var io_busy = std.atomic.Value(u8).init(0);
var io_threaded: std.Io.Threaded = undefined;

threadlocal var worker_qos_configured = false;

fn configuredHelperCount() usize {
    const limit = runtime.maxThreads();
    if (limit <= 1) return 0;
    return @min(limit - 1, max_tasks - 1);
}

fn ensureWorkerQoS() void {
    if (worker_qos_configured) return;
    worker_qos_configured = true;
    if (comptime builtin.os.tag == .macos) {
        _ = std.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
    }
}

fn ensureIoThreaded() bool {
    const state = io_init_state.load(.acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    if (io_init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        const helper_count = configuredHelperCount();
        if (helper_count == 0) {
            io_init_state.store(3, .release);
            return false;
        }
        io_threaded = std.Io.Threaded.init(std.heap.c_allocator, .{
            .stack_size = 256 * 1024,
            .async_limit = .limited(helper_count),
            .concurrent_limit = .limited(helper_count),
        });
        io_init_state.store(2, .release);
        return true;
    }

    while (true) {
        const current = io_init_state.load(.acquire);
        if (current == 2) return true;
        if (current == 3) return false;
        std.atomic.spinLoopHint();
    }
}

fn runIoTask(task_fn: TaskFn, tasks: *const anyopaque, index: usize) void {
    ensureWorkerQoS();
    task_fn(tasks, index);
}

pub fn taskCount(items: usize, min_items_per_task: usize) usize {
    const limit = runtime.maxThreads();
    if (limit <= 1 or items < min_items_per_task * 2) return 1;
    const by_size = @max(@as(usize, 1), items / min_items_per_task);
    return @min(@min(limit, max_tasks), by_size);
}

pub fn run(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (count <= 1) return false;
    if (!ensureIoThreaded()) return false;
    if (count - 1 > configuredHelperCount()) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    var group: std.Io.Group = .init;
    var submitted = false;
    const io = io_threaded.io();
    for (1..count) |index| {
        group.concurrent(io, runIoTask, .{ task_fn, tasks, index }) catch {
            task_fn(tasks, index);
            continue;
        };
        submitted = true;
    }
    ensureWorkerQoS();
    task_fn(tasks, 0);
    if (submitted) group.await(io) catch {};
    return true;
}
