// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! FEAST4 external parameter translation. No native numerical types or owners.
//! Profile gaps return an explicit extension code, never a successful substitute.
const std = @import("std");
pub const unsupported: i32 = -1000;
pub const Rule = enum { gauss, trapezoidal };
pub const Controls = struct {
    print: i32,
    nodes: usize,
    exponent: i32,
    refinements: usize,
    reuse: bool,
    trace: bool,
    rule: Rule,
    ratio: f64,
    inexact: bool,
    linear_tolerance: f64,
    linear_iterations: usize,
    resolved: [11]i32,
};
pub const Result = union(enum) { invalid: i32, valid: Controls };
const indices = [_]usize{ 0, 1, 2, 3, 4, 5, 15, 17, 42, 44, 45 };
pub fn initialize(values: []i32) !void {
    if (values.len < 64) return error.BufferTooSmall;
    @memset(values[0..64], -111);
}
fn value(values: []const i32, index: usize, fallback: i32) i32 {
    return if (values[index] == -111) fallback else values[index];
}
pub fn decode(values: []const i32) Result {
    if (values.len < 64) return .{ .invalid = unsupported };
    const inexact = value(values, 42, 0) == 1;
    const p = [_]i32{ value(values, 0, 0), value(values, 1, if (inexact) 4 else 8), value(values, 2, 12), value(values, 3, if (inexact) 50 else 20), value(values, 4, 0), value(values, 5, 1), value(values, 15, if (inexact) 1 else 0), value(values, 17, if (inexact) 100 else 30), value(values, 42, 0), value(values, 44, 1), value(values, 45, 40) };
    if (p[1] <= 0) return .{ .invalid = 102 };
    if (p[2] < 0 or p[2] > 16) return .{ .invalid = 103 };
    if (p[3] < 0) return .{ .invalid = 104 };
    if (p[4] != 0 and p[4] != 1) return .{ .invalid = 105 };
    if (p[5] != 0 and p[5] != 1) return .{ .invalid = 106 };
    if (p[6] < 0 or p[6] > 2) return .{ .invalid = 116 };
    if (p[7] < 0) return .{ .invalid = 118 };
    if (p[6] == 0 and p[1] > 20 and p[1] != 24 and p[1] != 32 and p[1] != 40 and p[1] != 48 and p[1] != 56) return .{ .invalid = 102 };
    // These valid FEAST modes still need their own native algorithms/integration.
    if (p[0] != 0 or value(values, 13, 0) != 0 or value(values, 39, 0) != 0 or p[6] == 2 or p[7] == 0 or p[10] < 0) return .{ .invalid = unsupported };
    const linear = std.math.pow(f64, 10, -@as(f64, @floatFromInt(p[9])));
    if (!std.math.isFinite(linear) or linear <= 0) return .{ .invalid = unsupported };
    return .{ .valid = .{ .print = p[0], .nodes = @intCast(p[1]), .exponent = p[2], .refinements = @intCast(p[3]), .reuse = p[4] == 1, .trace = p[5] == 0, .rule = if (p[6] == 0) .gauss else .trapezoidal, .ratio = @as(f64, @floatFromInt(p[7])) / 100, .inexact = inexact, .linear_tolerance = linear, .linear_iterations = @intCast(@max(@as(i32, 1), p[10])), .resolved = p } };
}
pub fn publish(values: []i32, controls: Controls, iterations: usize) !void {
    if (values.len < 64) return error.BufferTooSmall;
    const count = std.math.cast(i32, iterations) orelse return error.SizeOverflow;
    for (indices, controls.resolved) |index, resolved| if (values[index] == -111) {
        values[index] = resolved;
    };
    values[59] = count;
}
