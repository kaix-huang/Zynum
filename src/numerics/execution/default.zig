// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
pub const owner = @import("owner.zig");
pub const policy = @import("policy.zig");
pub const context = @import("context.zig");

pub const test_support = @import("test_support.zig");

// One stable default per shared core module, never per facade or isolated object.
var default_context = context.Context.init(std.heap.c_allocator, .{
    .max_threads = policy.maxThreads,
    .helper_count = policy.helperThreadCount,
    .configure_thread = policy.configureWorkerThread,
    .worker_stack_size = policy.worker_stack_size,
}, .immortal_default);

pub fn get() *owner.Executor {
    return &default_context.executor;
}

/// Same stable default instance; the legacy get() remains an Executor forwarder.
pub fn getContext() *context.Context {
    return &default_context;
}

/// Existing APIs use an immortal default and keep their error/ABI contracts.
/// Counter exhaustion is an explicit internal terminal fault, never an INFO or
/// numerical failure and never permission to run outside lifetime accounting.
pub fn beginDefault() context.Operation {
    return default_context.begin() catch |err| switch (err) {
        error.Closing => @panic("immortal default Context unexpectedly closed"),
        error.CounterOverflow => @panic("default Context operation counter exhausted"),
    };
}
