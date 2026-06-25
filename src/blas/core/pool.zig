// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const runtime = @import("../runtime.zig");

pub const max_tasks = 64;
pub const TaskFn = *const fn (*const anyopaque, usize) void;
const persistent_spin_iters: u32 = 500_000;
const persistent_done_spin_iters: u32 = 100_000;

var io_init_state = std.atomic.Value(u8).init(0);
var io_busy = std.atomic.Value(u8).init(0);
var io_threaded: std.Io.Threaded = undefined;
var persistent_threaded: std.Io.Threaded = undefined;

var persistent_init_state = std.atomic.Value(u8).init(0);
var persistent_group: std.Io.Group = .init;
var persistent_worker_count = std.atomic.Value(u32).init(0);
var persistent_ready_count = std.atomic.Value(u32).init(0);
var persistent_generation = std.atomic.Value(u32).init(0);
var persistent_worker_generation = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(0)} ** max_tasks;
var persistent_active_helpers = std.atomic.Value(u32).init(0);
var persistent_first_helper = std.atomic.Value(u32).init(0);
var persistent_done_target = std.atomic.Value(u32).init(0);
var persistent_done_count = std.atomic.Value(u32).init(0);
var persistent_job_count = std.atomic.Value(u32).init(0);
var persistent_task_fn: TaskFn = undefined;
var persistent_tasks: *const anyopaque = undefined;

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

fn runPersistentWorker(worker_id: usize) void {
    ensureWorkerQoS();
    const io = persistent_threaded.io();
    var seen = persistent_worker_generation[worker_id].load(.acquire);
    _ = persistent_ready_count.fetchAdd(1, .acq_rel);
    io.futexWake(u32, &persistent_ready_count.raw, 1);

    while (true) {
        var current = persistent_worker_generation[worker_id].load(.acquire);
        var spin_count: u32 = 0;
        while (current == seen and spin_count < persistent_spin_iters) : (spin_count += 1) {
            std.atomic.spinLoopHint();
            current = persistent_worker_generation[worker_id].load(.acquire);
        }
        while (current == seen) {
            io.futexWaitUncancelable(u32, &persistent_worker_generation[worker_id].raw, seen);
            current = persistent_worker_generation[worker_id].load(.acquire);
        }
        seen = current;

        const task_fn = persistent_task_fn;
        const tasks = persistent_tasks;
        const count = persistent_job_count.load(.acquire);
        const done_target = persistent_done_target.load(.acquire);
        const active_helpers = persistent_active_helpers.load(.acquire);
        const first_helper = persistent_first_helper.load(.acquire);
        if (worker_id < first_helper or worker_id >= first_helper + active_helpers) continue;
        const index = worker_id - first_helper + 1;
        if (index < count) task_fn(tasks, @intCast(index));

        const done = persistent_done_count.fetchAdd(1, .acq_rel) + 1;
        if (done >= done_target) io.futexWake(u32, &persistent_done_count.raw, 1);
    }
}

fn ensurePersistentWorkers() bool {
    const state = persistent_init_state.load(.acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    if (persistent_init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        const helper_count = configuredHelperCount();
        if (helper_count == 0) {
            persistent_init_state.store(3, .release);
            return false;
        }
        persistent_threaded = std.Io.Threaded.init(std.heap.c_allocator, .{
            .stack_size = 256 * 1024,
            .async_limit = .limited(helper_count),
            .concurrent_limit = .limited(helper_count),
        });

        const io = persistent_threaded.io();
        persistent_ready_count.store(0, .release);
        var submitted: u32 = 0;
        for (0..helper_count) |worker_id| {
            persistent_group.concurrent(io, runPersistentWorker, .{worker_id}) catch break;
            submitted += 1;
        }
        if (submitted == 0) {
            persistent_init_state.store(3, .release);
            return false;
        }
        persistent_worker_count.store(submitted, .release);

        while (true) {
            const ready = persistent_ready_count.load(.acquire);
            if (ready >= submitted) break;
            io.futexWaitUncancelable(u32, &persistent_ready_count.raw, ready);
        }

        persistent_init_state.store(2, .release);
        return true;
    }

    while (true) {
        const current = persistent_init_state.load(.acquire);
        if (current == 2) return true;
        if (current == 3) return false;
        std.atomic.spinLoopHint();
    }
}

pub fn taskCount(items: usize, min_items_per_task: usize) usize {
    const limit = runtime.maxThreads();
    if (limit <= 1 or items < min_items_per_task * 2) return 1;
    const by_size = @max(@as(usize, 1), items / min_items_per_task);
    return @min(@min(limit, max_tasks), by_size);
}

fn runPersistent(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (!ensurePersistentWorkers()) return false;
    const workers = persistent_worker_count.load(.acquire);
    if (workers == 0) return false;
    const task_helpers: u32 = @intCast(count - 1);
    const active_helpers = @min(workers, task_helpers);
    if (active_helpers == 0) return false;
    const first_helper: u32 = if (count == 3 and workers >= active_helpers + 2) 2 else 0;

    persistent_task_fn = task_fn;
    persistent_tasks = tasks;
    persistent_job_count.store(@intCast(count), .release);
    persistent_done_count.store(0, .release);
    persistent_done_target.store(active_helpers, .release);
    persistent_first_helper.store(first_helper, .release);
    persistent_active_helpers.store(active_helpers, .release);

    const io = persistent_threaded.io();
    const generation = persistent_generation.fetchAdd(1, .acq_rel) + 1;
    for (0..active_helpers) |worker_id| {
        const target_worker = first_helper + worker_id;
        persistent_worker_generation[target_worker].store(generation, .release);
        io.futexWake(u32, &persistent_worker_generation[target_worker].raw, 1);
    }

    ensureWorkerQoS();
    task_fn(tasks, 0);
    while (true) {
        var done = persistent_done_count.load(.acquire);
        const done_target = persistent_done_target.load(.acquire);
        var spin_count: u32 = 0;
        while (done < done_target and spin_count < persistent_done_spin_iters) : (spin_count += 1) {
            std.atomic.spinLoopHint();
            done = persistent_done_count.load(.acquire);
        }
        if (done >= done_target) break;
        io.futexWaitUncancelable(u32, &persistent_done_count.raw, done);
    }
    return true;
}

fn runGroup(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (!ensureIoThreaded()) return false;
    if (count - 1 > configuredHelperCount()) return false;

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

pub fn run(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (count <= 1) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    return runGroup(task_fn, tasks, count);
}

pub fn runLowLatency(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    if (count <= 1) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    if (runPersistent(task_fn, tasks, count)) return true;
    return runGroup(task_fn, tasks, count);
}
