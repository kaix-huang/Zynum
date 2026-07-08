// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

pub const maximum_threads_env_name = "ZYNUM_MAXIMUM_THREADS";
pub const worker_stack_size: usize = 512 * 1024;

var max_threads_override = std.atomic.Value(usize).init(0);
var total_threads_cache = std.atomic.Value(usize).init(0);
var performance_threads_cache = std.atomic.Value(usize).init(0);
var efficiency_threads_cache = std.atomic.Value(usize).init(0);
var performance_l2_cache = std.atomic.Value(usize).init(0);
var cache_line_cache = std.atomic.Value(usize).init(0);
var env_threads_cache = std.atomic.Value(usize).init(0);

threadlocal var worker_qos_configured = false;
threadlocal var worker_affinity_configured = false;

pub fn setMaxThreads(n: usize) void {
    max_threads_override.store(n, .monotonic);
}

pub fn maxThreadsOverride() usize {
    return max_threads_override.load(.monotonic);
}

fn envThreadLimit() usize {
    const cached = env_threads_cache.load(.monotonic);
    if (cached != 0) return cached - 1;

    const raw = std.c.getenv(maximum_threads_env_name) orelse {
        env_threads_cache.store(1, .monotonic);
        return 0;
    };
    const value = std.mem.span(raw);
    const parsed = if (value.len == 0) 0 else std.fmt.parseUnsigned(usize, value, 10) catch 0;
    env_threads_cache.store(parsed +| 1, .monotonic);
    return parsed;
}

pub fn totalThreadCount() usize {
    const cached = total_threads_cache.load(.monotonic);
    if (cached != 0) return cached;
    const detected = std.Thread.getCpuCount() catch 1;
    const value = @max(@as(usize, 1), detected);
    total_threads_cache.store(value, .monotonic);
    return value;
}

pub fn maxThreads() usize {
    const override = maxThreadsOverride();
    if (override != 0) return @min(override, totalThreadCount());
    const env_limit = envThreadLimit();
    if (env_limit != 0) return @min(env_limit, totalThreadCount());
    return totalThreadCount();
}

pub fn helperThreadCount(max_helpers: usize) usize {
    const limit = maxThreads();
    if (limit <= 1) return 0;
    return @min(limit - 1, max_helpers);
}

pub fn hasExplicitThreadLimit() bool {
    return maxThreadsOverride() != 0 or envThreadLimit() != 0;
}

fn sysctlInt(comptime name: [:0]const u8) usize {
    if (comptime builtin.os.tag != .macos) return 0;

    var value: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (std.c.sysctlbyname(name.ptr, &value, &len, null, 0) == 0 and value > 0) {
        return @intCast(value);
    }

    var value_usize: usize = 0;
    len = @sizeOf(usize);
    if (std.c.sysctlbyname(name.ptr, &value_usize, &len, null, 0) == 0 and value_usize > 0) {
        return value_usize;
    }
    return 0;
}

pub fn performanceThreadCount() usize {
    const cached = performance_threads_cache.load(.monotonic);
    if (cached != 0) return cached;
    const detected = sysctlInt("hw.perflevel0.logicalcpu");
    const value = if (detected != 0) detected else 0;
    performance_threads_cache.store(value, .monotonic);
    return value;
}

pub fn efficiencyThreadCount() usize {
    const cached = efficiency_threads_cache.load(.monotonic);
    if (cached != 0) return cached;
    const detected = sysctlInt("hw.perflevel1.logicalcpu");
    const value = if (detected != 0) detected else 0;
    efficiency_threads_cache.store(value, .monotonic);
    return value;
}

pub fn performanceL2Bytes() usize {
    const cached = performance_l2_cache.load(.monotonic);
    if (cached != 0) return cached;
    const detected = sysctlInt("hw.perflevel0.l2cachesize");
    const value = if (detected != 0) detected else 16 * 1024 * 1024;
    performance_l2_cache.store(value, .monotonic);
    return value;
}

pub fn cacheLineBytes() usize {
    const cached = cache_line_cache.load(.monotonic);
    if (cached != 0) return cached;
    const detected = sysctlInt("hw.cachelinesize");
    const value = if (detected != 0) detected else 128;
    cache_line_cache.store(value, .monotonic);
    return value;
}

fn ensureWorkerQoS() void {
    if (worker_qos_configured) return;
    worker_qos_configured = true;
    if (comptime builtin.os.tag == .macos) {
        _ = std.c.pthread_set_qos_class_self_np(.USER_INITIATED, 0);
    }
}

fn cpuSetContains(set: std.posix.cpu_set_t, cpu: usize) bool {
    const bits_per_word = @bitSizeOf(usize);
    const word_index = cpu / bits_per_word;
    const bit_index: std.math.Log2Int(usize) = @intCast(cpu % bits_per_word);
    return word_index < set.len and ((set[word_index] & (@as(usize, 1) << bit_index)) != 0);
}

fn cpuSetAdd(set: *std.posix.cpu_set_t, cpu: usize) void {
    const bits_per_word = @bitSizeOf(usize);
    const word_index = cpu / bits_per_word;
    const bit_index: std.math.Log2Int(usize) = @intCast(cpu % bits_per_word);
    if (word_index < set.len) set[word_index] |= @as(usize, 1) << bit_index;
}

fn pickAllowedCpu(allowed: std.posix.cpu_set_t, ordinal: usize) ?usize {
    const count = std.os.linux.CPU_COUNT(allowed);
    if (count == 0) return null;
    const target = ordinal % count;
    var seen: usize = 0;
    for (0..std.os.linux.CPU_SETSIZE) |cpu| {
        if (!cpuSetContains(allowed, cpu)) continue;
        if (seen == target) return cpu;
        seen += 1;
    }
    return null;
}

fn ensureWorkerAffinity(ordinal: usize) void {
    if (worker_affinity_configured) return;
    worker_affinity_configured = true;
    if (comptime builtin.os.tag != .linux) return;

    const allowed = std.posix.sched_getaffinity(0) catch return;
    const cpu = pickAllowedCpu(allowed, ordinal) orelse return;
    var pinned: std.posix.cpu_set_t = [_]usize{0} ** (std.os.linux.CPU_SETSIZE / @sizeOf(usize));
    cpuSetAdd(&pinned, cpu);
    std.os.linux.sched_setaffinity(0, &pinned) catch {};
}

pub fn configureWorkerThread(affinity_ordinal: ?usize) void {
    ensureWorkerQoS();
    if (affinity_ordinal) |ordinal| ensureWorkerAffinity(ordinal);
}
