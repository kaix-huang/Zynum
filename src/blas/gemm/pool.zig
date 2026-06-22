// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const runtime = @import("../runtime.zig");
const gemm_kernels = @import("../kernels/backend.zig");

const max_helpers = 8;
const pool_env_name = "ZYNUM_BLAS_GEMM_POOL";
const io_env_name = "ZYNUM_BLAS_GEMM_IO";
const io_mode_default: u8 = 1;
const io_mode_disabled: u8 = 2;
const io_mode_group_concurrent: u8 = 3;
const io_mode_group_async: u8 = 4;
const io_mode_future_concurrent: u8 = 5;
const io_mode_future_async: u8 = 6;
const io_mode_persistent_pool: u8 = 7;

const JobKind = enum(u8) {
    f32,
    f64,
};

var init_state = std.atomic.Value(u8).init(0);
var env_state = std.atomic.Value(u8).init(0);
var busy = std.atomic.Value(u8).init(0);

var io_init_state = std.atomic.Value(u8).init(0);
var io_env_state = std.atomic.Value(u8).init(0);
var io_busy = std.atomic.Value(u8).init(0);
var io_threaded: std.Io.Threaded = undefined;
var io_worker_futures: [max_helpers]std.Io.Future(void) = undefined;

var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var ready_cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER;
var done_cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER;

var helper_count: usize = 0;
var generation: u64 = 0;
var done_generation: u64 = 0;
var active_helpers: usize = 0;
var job_kind: JobKind = .f32;
var job_tasks: ?*const anyopaque = null;
var job_task_count: usize = 0;
var next_task = std.atomic.Value(usize).init(0);
var remaining_helpers = std.atomic.Value(usize).init(0);

fn checkPthread(rc: std.c.E) void {
    if (rc != .SUCCESS) unreachable;
}

fn lock() void {
    checkPthread(std.c.pthread_mutex_lock(&mutex));
}

fn unlock() void {
    checkPthread(std.c.pthread_mutex_unlock(&mutex));
}

fn condWait(cond: *std.c.pthread_cond_t) void {
    checkPthread(std.c.pthread_cond_wait(cond, &mutex));
}

fn condSignal(cond: *std.c.pthread_cond_t) void {
    checkPthread(std.c.pthread_cond_signal(cond));
}

fn condBroadcast(cond: *std.c.pthread_cond_t) void {
    checkPthread(std.c.pthread_cond_broadcast(cond));
}

fn poolEnvState() u8 {
    switch (env_state.load(.acquire)) {
        1, 2, 3 => |state| return state,
        else => {},
    }

    const state: u8 = blk: {
        const raw = std.c.getenv(pool_env_name) orelse break :blk 1;
        const value = std.mem.span(raw);
        if (std.mem.eql(u8, value, "1") or
            std.ascii.eqlIgnoreCase(value, "true") or
            std.ascii.eqlIgnoreCase(value, "on"))
        {
            break :blk 3;
        }
        if (std.mem.eql(u8, value, "0") or
            std.ascii.eqlIgnoreCase(value, "false") or
            std.ascii.eqlIgnoreCase(value, "off") or
            std.ascii.eqlIgnoreCase(value, "no"))
        {
            break :blk 2;
        }
        break :blk 2;
    };
    env_state.store(state, .release);
    return state;
}

fn batchPoolEnabled() bool {
    // Batch pool is intentionally opt-in.  It removes pthread create/join
    // overhead for selected probes, but default-on helper threads regressed
    // complete GEMM sweeps by interfering with later small shapes.
    return poolEnvState() == 3;
}

fn ioPoolMode() u8 {
    switch (io_env_state.load(.acquire)) {
        io_mode_default,
        io_mode_disabled,
        io_mode_group_concurrent,
        io_mode_group_async,
        io_mode_future_concurrent,
        io_mode_future_async,
        io_mode_persistent_pool,
        => |cached| return cached,
        else => {},
    }

    const mode: u8 = blk: {
        const raw = std.c.getenv(io_env_name) orelse break :blk io_mode_default;
        const value = std.mem.span(raw);
        if (std.mem.eql(u8, value, "1") or
            std.ascii.eqlIgnoreCase(value, "true") or
            std.ascii.eqlIgnoreCase(value, "on") or
            std.ascii.eqlIgnoreCase(value, "concurrent") or
            std.ascii.eqlIgnoreCase(value, "group-concurrent") or
            std.ascii.eqlIgnoreCase(value, "group_concurrent"))
        {
            break :blk io_mode_group_concurrent;
        }
        if (std.ascii.eqlIgnoreCase(value, "async") or
            std.ascii.eqlIgnoreCase(value, "group-async") or
            std.ascii.eqlIgnoreCase(value, "group_async"))
        {
            break :blk io_mode_group_async;
        }
        if (std.ascii.eqlIgnoreCase(value, "future") or
            std.ascii.eqlIgnoreCase(value, "future-concurrent") or
            std.ascii.eqlIgnoreCase(value, "future_concurrent"))
        {
            break :blk io_mode_future_concurrent;
        }
        if (std.ascii.eqlIgnoreCase(value, "await") or
            std.ascii.eqlIgnoreCase(value, "future-async") or
            std.ascii.eqlIgnoreCase(value, "future_async") or
            std.ascii.eqlIgnoreCase(value, "async-await") or
            std.ascii.eqlIgnoreCase(value, "async_await"))
        {
            break :blk io_mode_future_async;
        }
        if (std.ascii.eqlIgnoreCase(value, "pool") or
            std.ascii.eqlIgnoreCase(value, "persistent") or
            std.ascii.eqlIgnoreCase(value, "persistent-pool") or
            std.ascii.eqlIgnoreCase(value, "persistent_pool") or
            std.ascii.eqlIgnoreCase(value, "worker-pool") or
            std.ascii.eqlIgnoreCase(value, "worker_pool"))
        {
            break :blk io_mode_persistent_pool;
        }
        if (std.mem.eql(u8, value, "0") or
            std.ascii.eqlIgnoreCase(value, "false") or
            std.ascii.eqlIgnoreCase(value, "off") or
            std.ascii.eqlIgnoreCase(value, "no"))
        {
            break :blk io_mode_disabled;
        }
        break :blk io_mode_disabled;
    };
    io_env_state.store(mode, .release);
    return mode;
}

fn singleWorkerPoolEnabled() bool {
    return poolEnvState() != 2;
}

fn configuredHelperCount() usize {
    const p_threads = runtime.performanceThreadCount();
    if (p_threads > 1) return @min(max_helpers, p_threads - 1);

    const limit = runtime.defaultGemmThreadLimit();
    if (limit <= 1) return 0;
    return @min(max_helpers, limit - 1);
}

fn runTask(kind: JobKind, tasks: *const anyopaque, index: usize, allow_sme: bool) void {
    switch (kind) {
        .f32 => {
            const typed: [*]const gemm_kernels.Task(f32) = @ptrCast(@alignCast(tasks));
            var task = typed[index];
            task.allow_sme = allow_sme;
            gemm_kernels.noTransRealF32(task);
        },
        .f64 => {
            const typed: [*]const gemm_kernels.Task(f64) = @ptrCast(@alignCast(tasks));
            var task = typed[index];
            task.allow_sme = allow_sme;
            gemm_kernels.noTransRealF64(task);
        },
    }
}

fn runIoTask(kind: JobKind, tasks: *const anyopaque, index: usize) void {
    runTask(kind, tasks, index, true);
}

fn workerMain(worker_id: usize) void {
    if (comptime builtin.os.tag == .macos) {
        _ = std.c.pthread_set_qos_class_self_np(.USER_INITIATED, 0);
    }

    var seen_generation: u64 = 0;
    while (true) {
        lock();
        while (generation == seen_generation) condWait(&ready_cond);
        seen_generation = generation;
        const kind = job_kind;
        const tasks = job_tasks.?;
        const task_count = job_task_count;
        const helpers = active_helpers;
        unlock();

        if (worker_id < helpers) {
            while (true) {
                const index = next_task.fetchAdd(1, .monotonic);
                if (index >= task_count) break;
                runTask(kind, tasks, index, true);
            }

            if (remaining_helpers.fetchSub(1, .acq_rel) == 1) {
                lock();
                done_generation = seen_generation;
                condSignal(&done_cond);
                unlock();
            }
        }
    }
}

fn ensurePool(comptime require_batch_opt_in: bool) bool {
    if (require_batch_opt_in) {
        if (!batchPoolEnabled()) return false;
    } else if (!singleWorkerPoolEnabled()) return false;

    const state = init_state.load(.acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    if (init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        const count = configuredHelperCount();
        var spawned: usize = 0;
        while (spawned < count) : (spawned += 1) {
            const thread = std.Thread.spawn(.{ .stack_size = 512 * 1024 }, workerMain, .{spawned}) catch break;
            thread.detach();
        }
        helper_count = spawned;
        init_state.store(if (spawned == 0) 3 else 2, .release);
        return spawned != 0;
    }

    while (true) {
        const current = init_state.load(.acquire);
        if (current == 2) return true;
        if (current == 3) return false;
        std.atomic.spinLoopHint();
    }
}

fn publish(kind: JobKind, tasks: *const anyopaque, task_count: usize, helpers: usize, first_worker_task: usize) u64 {
    lock();
    job_kind = kind;
    job_tasks = tasks;
    job_task_count = task_count;
    active_helpers = helpers;
    next_task.store(first_worker_task, .release);
    remaining_helpers.store(helpers, .release);
    generation +%= 1;
    const target_generation = generation;
    condBroadcast(&ready_cond);
    unlock();
    return target_generation;
}

fn waitDone(target_generation: u64) void {
    lock();
    while (done_generation != target_generation) condWait(&done_cond);
    unlock();
}

fn runWithCaller(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count <= 1) return false;
    if (!ensurePool(true)) return false;
    const helpers = task_count - 1;
    if (helpers > helper_count) return false;
    if (busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer busy.store(0, .release);

    // Caller runs task 0 while helper threads take tasks 1..N-1.  This is
    // the only batch mode exposed to normal GEMM dispatch, and only when
    // ZYNUM_BLAS_GEMM_POOL enables it.
    const target_generation = publish(kind, tasks, task_count, helpers, 1);
    runTask(kind, tasks, 0, true);
    waitDone(target_generation);
    return true;
}

fn ensureIoThreadedAllowDefault(allow_default: bool) bool {
    const mode = ioPoolMode();
    if (mode == io_mode_disabled) return false;
    if (!allow_default and mode == io_mode_default) return false;

    const state = io_init_state.load(.acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    if (io_init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        const count = configuredHelperCount();
        if (count == 0) {
            io_init_state.store(3, .release);
            return false;
        }
        io_threaded = std.Io.Threaded.init(std.heap.c_allocator, .{
            .stack_size = 512 * 1024,
            .async_limit = .limited(count),
            .concurrent_limit = .limited(count),
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

fn ensureIoThreaded() bool {
    return ensureIoThreadedAllowDefault(false);
}

fn ensureIoThreadedAuto() bool {
    return ensureIoThreadedAllowDefault(true);
}

fn runIoGroupConcurrent(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
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

fn runIoGroupAsync(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
    var group: std.Io.Group = .init;

    for (1..task_count) |index| {
        group.async(io, runIoTask, .{ kind, tasks, index });
    }

    runTask(kind, tasks, 0, true);
    group.await(io) catch {};
}

fn runIoFutureConcurrent(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
    var futures: [max_helpers]std.Io.Future(void) = undefined;
    var future_count: usize = 0;

    for (1..task_count) |index| {
        futures[future_count] = std.Io.concurrent(io, runIoTask, .{ kind, tasks, index }) catch {
            runTask(kind, tasks, index, true);
            continue;
        };
        future_count += 1;
    }

    runTask(kind, tasks, 0, true);
    for (futures[0..future_count]) |*future| future.await(io);
}

fn runIoFutureAsync(io: std.Io, kind: JobKind, tasks: *const anyopaque, task_count: usize) void {
    var futures: [max_helpers]std.Io.Future(void) = undefined;
    var future_count: usize = 0;

    for (1..task_count) |index| {
        futures[future_count] = std.Io.async(io, runIoTask, .{ kind, tasks, index });
        future_count += 1;
    }

    runTask(kind, tasks, 0, true);
    for (futures[0..future_count]) |*future| future.await(io);
}

fn runIoWithCaller(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count <= 1) return false;
    if (!ensureIoThreaded()) return false;
    if (task_count - 1 > configuredHelperCount()) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    const io = io_threaded.io();
    switch (ioPoolMode()) {
        io_mode_group_concurrent => runIoGroupConcurrent(io, kind, tasks, task_count),
        io_mode_group_async => runIoGroupAsync(io, kind, tasks, task_count),
        io_mode_future_concurrent => runIoFutureConcurrent(io, kind, tasks, task_count),
        io_mode_future_async => runIoFutureAsync(io, kind, tasks, task_count),
        io_mode_persistent_pool => return runIoPersistentWithCaller(kind, tasks, task_count),
        else => return false,
    }
    return true;
}

fn runIoAutoFutureWithCaller(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count <= 1) return false;
    if (ioPoolMode() != io_mode_default) return false;
    if (!ensureIoThreadedAuto()) return false;
    if (task_count - 1 > configuredHelperCount()) return false;
    if (io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer io_busy.store(0, .release);

    runIoFutureConcurrent(io_threaded.io(), kind, tasks, task_count);
    return true;
}

fn ensureIoPersistentPool() bool {
    if (!ensureIoThreaded()) return false;
    const state = init_state.load(.acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    if (init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        const count = configuredHelperCount();
        var spawned: usize = 0;
        const io = io_threaded.io();
        while (spawned < count) : (spawned += 1) {
            io_worker_futures[spawned] = std.Io.concurrent(io, workerMain, .{spawned}) catch break;
        }
        helper_count = spawned;
        init_state.store(if (spawned == 0) 3 else 2, .release);
        return spawned != 0;
    }

    while (true) {
        const current = init_state.load(.acquire);
        if (current == 2) return true;
        if (current == 3) return false;
        std.atomic.spinLoopHint();
    }
}

fn runIoPersistentWithCaller(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count <= 1) return false;
    if (!ensureIoPersistentPool()) return false;
    const helpers = task_count - 1;
    if (helpers > helper_count) return false;
    if (busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer busy.store(0, .release);

    const target_generation = publish(kind, tasks, task_count, helpers, 1);
    runTask(kind, tasks, 0, true);
    waitDone(target_generation);
    return true;
}

fn runWorkersOnly(kind: JobKind, tasks: *const anyopaque, task_count: usize) bool {
    if (task_count == 0) return false;
    if (!ensurePool(false)) return false;
    if (task_count > helper_count) return false;
    if (busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
    defer busy.store(0, .release);

    // Worker-only mode is for direct-SME experiments where the caller
    // should avoid entering streaming mode.  It is not used for default
    // multi-task GEMM dispatch.
    const target_generation = publish(kind, tasks, task_count, task_count, 0);
    waitDone(target_generation);
    return true;
}

pub fn runF32(tasks: []const gemm_kernels.Task(f32)) bool {
    return runWithCaller(.f32, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runF64(tasks: []const gemm_kernels.Task(f64)) bool {
    return runWithCaller(.f64, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runIoF32(tasks: []const gemm_kernels.Task(f32)) bool {
    return runIoWithCaller(.f32, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runIoF64(tasks: []const gemm_kernels.Task(f64)) bool {
    return runIoWithCaller(.f64, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runIoAutoF32(tasks: []const gemm_kernels.Task(f32)) bool {
    return runIoAutoFutureWithCaller(.f32, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runIoAutoF64(tasks: []const gemm_kernels.Task(f64)) bool {
    return runIoAutoFutureWithCaller(.f64, @ptrCast(tasks.ptr), tasks.len);
}

pub fn runSingleF32(task: gemm_kernels.Task(f32)) bool {
    const tasks = [_]gemm_kernels.Task(f32){task};
    return runWorkersOnly(.f32, @ptrCast(&tasks), tasks.len);
}

pub fn runSingleF64(task: gemm_kernels.Task(f64)) bool {
    const tasks = [_]gemm_kernels.Task(f64){task};
    return runWorkersOnly(.f64, @ptrCast(&tasks), tasks.len);
}
