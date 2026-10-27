// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Passive host metadata only. No environment budgets, worker startup, affinity,
//! allocator, Context or execution-policy ownership exists in this module.
const std = @import("std");
const builtin = @import("builtin");
fn sysctlInt(comptime name: [:0]const u8) usize {
    if (comptime builtin.target.os.tag != .macos) return 0;
    var value: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (std.c.sysctlbyname(name.ptr, &value, &len, null, 0) == 0 and value > 0) return @intCast(value);
    var wide: usize = 0;
    len = @sizeOf(usize);
    if (std.c.sysctlbyname(name.ptr, &wide, &len, null, 0) == 0 and wide > 0) return wide;
    return 0;
}
pub fn totalThreadCount() usize {
    return @max(@as(usize, 1), std.Thread.getCpuCount() catch 1);
}
pub fn performanceThreadCount() usize {
    return sysctlInt("hw.perflevel0.logicalcpu");
}
pub fn efficiencyThreadCount() usize {
    return sysctlInt("hw.perflevel1.logicalcpu");
}
pub fn performanceL2Bytes() usize {
    const value = sysctlInt("hw.perflevel0.l2cachesize");
    return if (value != 0) value else 16 * 1024 * 1024;
}
pub fn cacheLineBytes() usize {
    const value = sysctlInt("hw.cachelinesize");
    return if (value != 0) value else 128;
}
