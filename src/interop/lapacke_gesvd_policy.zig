// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0- or -later
//! External scalar/workspace policy only. Numerical SVD stays in shared Core.
const std = @import("std");
pub const Job = enum { all, some, overwrite, none };
pub fn job(byte: u8) ?Job {
    return switch (byte) {
        'A', 'a' => .all,
        'S', 's' => .some,
        'O', 'o' => .overwrite,
        'N', 'n' => .none,
        else => null,
    };
}
/// Reference ILAENV6 uses single precision, including its threshold rounding.
pub fn minimumWork(comptime complex: bool, m: usize, n: usize, u: Job, v: Job) error{SizeOverflow}!usize {
    const k = @min(m, n);
    if (k == 0) return 1;
    const threshold_f: f32 = @as(f32, @floatFromInt(k)) * @as(f32, 1.6);
    const threshold: usize = if (threshold_f >= @as(f32, @floatFromInt(std.math.maxInt(usize)))) std.math.maxInt(usize) else @intFromFloat(threshold_f);
    const compressed = if (m >= n) m >= threshold and u == .none else n >= threshold and v == .none;
    if (compressed) return std.math.mul(usize, if (complex) 3 else 5, k) catch error.SizeOverflow;
    const common = std.math.add(usize, std.math.mul(usize, if (complex) 2 else 3, k) catch return error.SizeOverflow, @max(m, n)) catch return error.SizeOverflow;
    return if (complex) common else @max(common, std.math.mul(usize, 5, k) catch return error.SizeOverflow);
}
pub fn scalarInfo(a: anytype) i32 {
    const row = a.layout == 101;
    const uj = job(a.jobu);
    const vj = job(a.jobvt);
    const k = @min(a.m, a.n);
    const uc = if (uj == .all) a.m else if (uj == .some) k else 1;
    const vc = if (vj == .all or vj == .some) a.n else 1;
    if (row) {
        if (a.lda < a.n) return -7;
        if (a.ldu < uc) return -10;
        if (a.ldvt < vc) return -12;
    }
    if (uj == null) return -2;
    if (vj == null or (uj == .overwrite and vj == .overwrite)) return -3;
    if (a.m < 0) return -4;
    if (a.n < 0) return -5;
    if (!row and a.lda < @max(1, a.m)) return -7;
    if (!row and (a.ldu < 1 or ((uj == .all or uj == .some) and a.ldu < a.m))) return -10;
    if (!row and (a.ldvt < 1 or (vj == .all and a.ldvt < a.n) or (vj == .some and a.ldvt < k))) return -12;
    return 0;
}
pub fn numericalInfo(err: anyerror) ?i32 {
    return switch (err) {
        error.NonFiniteInput, error.NonFiniteResult, error.NoConvergence, error.RangeLimit => 1,
        else => null,
    };
}
