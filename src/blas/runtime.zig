// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const build_options = @import("zynum-build-options");

const reference_compat = @hasDecl(build_options, "reference_compat") and build_options.reference_compat;
comptime {
    if (reference_compat) {
        _ = @import("reference_cache_hook.zig");
        if (builtin.target.cpu.arch == .x86_64 and !@hasDecl(build_options, "kernel_entry"))
            _ = @import("kernels/isolated/task_runtime_host.zig");
    }
}

const policy = @import("zynum-execution").reference.policy;
const hardware = @import("hardware_metadata.zig");

pub const maximum_threads_env_name = policy.maximum_threads_env_name;
pub const worker_stack_size = policy.worker_stack_size;
pub const setMaxThreads = policy.setMaxThreads;
pub const maxThreadsOverride = policy.maxThreadsOverride;
pub const totalThreadCount = if (reference_compat) policy.totalThreadCount else hardware.totalThreadCount;
pub const maxThreads = policy.maxThreads;
pub const helperThreadCount = policy.helperThreadCount;
pub const hasExplicitThreadLimit = policy.hasExplicitThreadLimit;
pub const performanceThreadCount = if (reference_compat) policy.performanceThreadCount else hardware.performanceThreadCount;
pub const efficiencyThreadCount = if (reference_compat) policy.efficiencyThreadCount else hardware.efficiencyThreadCount;
pub const performanceL2Bytes = if (reference_compat) policy.performanceL2Bytes else hardware.performanceL2Bytes;
pub const cacheLineBytes = if (reference_compat) policy.cacheLineBytes else hardware.cacheLineBytes;
pub const configureWorkerThread = policy.configureWorkerThread;

/// Diagnostic only; specialization does not query the runtime ISA resolver.
pub fn selectedKernelTier() []const u8 {
    if (comptime !reference_compat) return "portable_generic";
    if (comptime build_options.dynamic_dispatch) return @tagName(@import("kernels/multiversion/client.zig").selectedTier());
    return "specialized";
}
