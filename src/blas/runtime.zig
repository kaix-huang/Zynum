// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

pub const gemm_parallel_work_threshold: usize = 192 * 192 * 192;
pub const medium_gemm_work_threshold: usize = 768 * 768 * 768;
pub const maximum_threads_env_name = "ZYNUM_MAXIMUM_THREADS";

var max_threads_override = std.atomic.Value(usize).init(0);
var total_threads_cache = std.atomic.Value(usize).init(0);
var performance_threads_cache = std.atomic.Value(usize).init(0);
var efficiency_threads_cache = std.atomic.Value(usize).init(0);
var performance_l2_cache = std.atomic.Value(usize).init(0);
var cache_line_cache = std.atomic.Value(usize).init(0);
var env_threads_cache = std.atomic.Value(usize).init(0);

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
    if (override != 0) return override;
    const env_limit = envThreadLimit();
    if (env_limit != 0) return env_limit;
    return totalThreadCount();
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

pub fn defaultGemmThreadLimit() usize {
    return maxThreads();
}

pub fn gemmThreadCount(m: usize, n: usize, k: usize) usize {
    if (m == 0 or n < 2 or k == 0) return 1;

    const work = m *| n *| k;
    if (work < gemm_parallel_work_threshold) return 1;

    const limit = maxThreads();
    if (limit <= 1) return 1;
    return @max(@as(usize, 1), @min(limit, n));
}
