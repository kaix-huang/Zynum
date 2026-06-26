// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const runtime = @import("../runtime.zig");
const gemm_kernels = @import("../kernels/matrix_matrix.zig");

const JobKind = enum(u8) {
    f32,
    f64,
};

var io_init_state = std.atomic.Value(u8).init(0);
var io_busy = std.atomic.Value(u8).init(0);
var io_threaded: std.Io.Threaded = undefined;

threadlocal var worker_qos_configured = false;

fn configuredHelperCount() usize {
    const limit = runtime.maxThreads();
    if (limit <= 1) return 0;
    return limit - 1;
}

fn ensureWorkerQoS() void {
    if (worker_qos_configured) return;
    worker_qos_configured = true;
    if (comptime builtin.os.tag == .macos) {
        _ = std.c.pthread_set_qos_class_self_np(.USER_INITIATED, 0);
    }
}

fn runTask(kind: JobKind, tasks: *const anyopaque, index: usize, allow_sme: bool) void {
    switch (kind) {
        .f32 => {
            const typed: [*]const gemm_kernels.Task(f32) = @ptrCast(@alignCast(tasks));
            var task = typed[index];
            task.allow_sme = task.allow_sme and allow_sme;
            gemm_kernels.noTransRealF32(task);
        },
        .f64 => {
            const typed: [*]const gemm_kernels.Task(f64) = @ptrCast(@alignCast(tasks));
            var task = typed[index];
            task.allow_sme = task.allow_sme and allow_sme;
            gemm_kernels.noTransRealF64(task);
        },
    }
}

fn runIoTask(kind: JobKind, tasks: *const anyopaque, index: usize) void {
    ensureWorkerQoS();
    runTask(kind, tasks, index, true);
}

fn ensureIoThreaded() bool {
    if (configuredHelperCount() == 0) return false;

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
            .stack_size = 512 * 1024,
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

fn runIoGroupWithCaller(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
    var group: std.Io.Group = .init;
    var submitted = false;

    for (1..task_count) |index| {
        group.concurrent(io, runIoTask, .{ kind, tasks, index }) catch {
            runTask(kind, tasks, index, true);
            continue;
        };
        submitted = true;
    }

    runTask(kind, tasks, 0, true);
    if (submitted) group.await(io) catch {};
}

fn runIoWorkersOnly(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
    var group: std.Io.Group = .init;
    var submitted = false;

    for (0..task_count) |index| {
        group.concurrent(io, runIoTask, .{ kind, tasks, index }) catch {
            runTask(kind, tasks, index, true);
            continue;
        };
        submitted = true;
    }

    if (submitted) group.await(io) catch {};
}

fn runWithCaller(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count <= 1) return false;
    if (!ensureIoThreaded()) return false;
    if (task_count - 1 > configuredHelperCount()) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    runIoGroupWithCaller(io_threaded.io(), kind, tasks, task_count);
    return true;
}

fn runWorkersOnly(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count == 0) return false;
    if (!ensureIoThreaded()) return false;
    if (task_count > configuredHelperCount()) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    runIoWorkersOnly(io_threaded.io(), kind, tasks, task_count);
    return true;
}

pub fn runF32(tasks: []const gemm_kernels.Task(f32)) bool {
    return runWithCaller(.f32, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runF64(tasks: []const gemm_kernels.Task(f64)) bool {
    return runWithCaller(.f64, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runSingleF32(task: gemm_kernels.Task(f32)) bool {
    const tasks = [_]gemm_kernels.Task(f32){task};
    return runWorkersOnly(.f32, @ptrCast(&tasks), tasks.len);
}

pub fn runSingleF64(task: gemm_kernels.Task(f64)) bool {
    const tasks = [_]gemm_kernels.Task(f64){task};
    return runWorkersOnly(.f64, @ptrCast(&tasks), tasks.len);
}
