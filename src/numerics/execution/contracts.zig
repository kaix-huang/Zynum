// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
pub const max_tasks: usize = 64;
pub const TaskFn = *const fn (*const anyopaque, usize) void;
pub const WorkerCacheDomain = struct { release_current_worker: *const fn () void };
pub const Batch = struct {
    task_fn: TaskFn,
    tasks: *const anyopaque,
    count: usize,
    cache_domain: ?*const WorkerCacheDomain = null,
};
pub const Policy = struct {
    max_threads: *const fn () usize,
    helper_count: *const fn (usize) usize,
    configure_thread: *const fn (?usize) void,
    worker_stack_size: usize,
};
pub const RunMode = enum { ordinary, low_latency, queued };
