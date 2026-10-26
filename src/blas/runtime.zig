// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const build_options = @import("zynum-build-options");

comptime {
    if (builtin.target.cpu.arch == .x86_64 and !@hasDecl(build_options, "kernel_entry"))
        _ = @import("kernels/isolated/task_runtime_host.zig");
}

const policy = @import("zynum-execution").policy;

pub const maximum_threads_env_name = policy.maximum_threads_env_name;
pub const worker_stack_size = policy.worker_stack_size;
pub const setMaxThreads = policy.setMaxThreads;
pub const maxThreadsOverride = policy.maxThreadsOverride;
pub const totalThreadCount = policy.totalThreadCount;
pub const maxThreads = policy.maxThreads;
pub const helperThreadCount = policy.helperThreadCount;
pub const hasExplicitThreadLimit = policy.hasExplicitThreadLimit;
pub const performanceThreadCount = policy.performanceThreadCount;
pub const efficiencyThreadCount = policy.efficiencyThreadCount;
pub const performanceL2Bytes = policy.performanceL2Bytes;
pub const cacheLineBytes = policy.cacheLineBytes;
pub const configureWorkerThread = policy.configureWorkerThread;

/// Diagnostic only; specialization does not query the runtime ISA resolver.
pub fn selectedKernelTier() []const u8 {
    if (comptime build_options.dynamic_dispatch) return @tagName(@import("kernels/multiversion/client.zig").selectedTier());
    return "specialized";
}
