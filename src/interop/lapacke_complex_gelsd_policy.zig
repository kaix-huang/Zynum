// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! External finite complex GELSD policy, independent of Core numerical rules.
const std = @import("std");
const real_policy = @import("lapacke_gelsd_policy.zig");
pub const scalarInfo = real_policy.scalarInfo;
pub const effectiveRcond = real_policy.effectiveRcond;
pub const Error = error{SizeOverflow};
/// The generated finite frontend returns an own bounded failure token for
/// unchecked nonfinite operands and unrepresentable numerical results. It is
/// separate from actual QR-exhaustion edge counts, with no vendor NaN payload promise.
pub fn standardNumericalInfo(fault: anyerror) ?i32 {
    return switch (fault) {
        error.NonFiniteInput, error.InvalidTolerance, error.NonFiniteResult, error.RangeLimit => 1,
        else => null,
    };
}
fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}
/// Selected Reference units: B is scaled before reconstruction, then only its
/// first N solution rows are restored. Tall residual rows retain scaled units.
/// Native least-squares inputs and outputs have independent units.
pub fn rightHandSideScale(comptime R: type, comptime W: type, maximum: W) W {
    const small = @as(W, std.math.floatMin(R)) / @as(W, std.math.floatEps(R));
    const large = 1 / small;
    return if (maximum > 0 and maximum < small) small / maximum else if (maximum > large) large / maximum else 1;
}
/// Reference 3.12.1 ILAENV(6) uses INT(REAL(k)*1.6E0), including the f32
/// rounding before truncation. This selected profile is independent of Core.
pub fn minimumWork(comptime I: type, m: usize, n: usize, rhs: usize) Error!I {
    if (I != i32 and I != i64) @compileError("complex GELSD index is i32/i64");
    if (m == 0 or n == 0) return 1;
    const k = @min(m, n);
    const twice = try mul(2, k);
    // The checked 2*k above bounds this conversion inside native usize.
    const threshold_real: f32 = @as(f32, @floatFromInt(k)) * @as(f32, 1.6);
    const threshold = @max(@as(usize, 1), @as(usize, @intFromFloat(threshold_real)));
    const mm = if (m >= n and m >= threshold) n else m;
    const linear = try add(twice, if (m >= n) mm else n);
    const panel = try add(twice, try mul(k, rhs));
    return std.math.cast(I, @max(linear, panel)) orelse error.SizeOverflow;
}
