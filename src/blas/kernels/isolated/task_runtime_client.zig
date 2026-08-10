// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Object-side access to the process-wide BLAS task runtime.
//!
//! The callback address is transported as data and reconstructed by the host;
//! the host therefore invokes the original Zig calling convention rather than
//! pretending it is a C callback.

pub const max_tasks = 64;
pub const TaskFn = *const fn (*const anyopaque, usize) void;

extern fn zynum_internal_level1_task_count(items: usize, min_items_per_task: usize) callconv(.c) usize;
extern fn zynum_internal_level1_run_low_latency(task_fn_address: usize, tasks: *const anyopaque, count: usize) callconv(.c) u8;
extern fn zynum_internal_level1_run(task_fn_address: usize, tasks: *const anyopaque, count: usize) callconv(.c) u8;

pub fn taskCount(items: usize, min_items_per_task: usize) usize {
    return zynum_internal_level1_task_count(items, min_items_per_task);
}

pub fn runLowLatency(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return zynum_internal_level1_run_low_latency(@intFromPtr(task_fn), tasks, count) != 0;
}

pub fn run(task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
    return zynum_internal_level1_run(@intFromPtr(task_fn), tasks, count) != 0;
}
