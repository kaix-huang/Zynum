// SPDX-License-Identifier: LGPL-3.0-or-later
//! Test-only analytic reference extracted from the original padded-grid test.
//! Positive n0/n2 and in-range indices are caller preconditions; no stencil.
const std = @import("std");
pub fn fourierInput(i: usize, k: usize, n0: usize, n2: usize) f64 {
    return @cos(2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n0))) +
        @sin(2 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n2)));
}
pub fn fourierExpected(i: usize, k: usize, n0: usize, n2: usize, h: [3]f64) f64 {
    return 2 * h[0] * (@cos(@as(f64, 2 * std.math.pi) / @as(f64, @floatFromInt(n0))) - 1) * @cos(2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n0))) +
        2 * h[2] * (@cos(@as(f64, 2 * std.math.pi) / @as(f64, @floatFromInt(n2))) - 1) * @sin(2 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n2)));
}
