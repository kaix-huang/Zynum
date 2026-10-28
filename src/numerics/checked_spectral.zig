// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Ownerless checked interval-scaled FP64 series and raw moments.
//! Extracted from integrations/heteris/src/spectral.zig at Zynum
//! 98b3a671ceea1c5c9c5d282d16c0dd98bb2e73b0. Its series was adapted from
//! c5cb123e55eaae683de09b0c8030837f5600f140 src/numerics/chebyshev.zig;
//! the moments reduction is Scientific Platform bridge code.
//!
//! The facade validates live, disjoint storage and input finiteness and owns
//! admission, mapping and callback metadata. A trusted panel provides cell(i,j)
//! and panel(start,columns); a synchronous callback provides apply(source,next).
//! Arithmetic is the original recurrence_arithmetic type from the configured
//! Core. No allocation, ownership, Context, provider or C view types live here.
const std = @import("std");

// Private Zig interface: deliberately no cross-language layout promise.
pub const Phase = enum { initialize, apply, recurrence, moment, commit, complete };
pub const Progress = struct {
    phase: Phase = .initialize,
    order_valid: bool = false,
    order: u64 = 0,
    callback_returned: bool = false,
    callback_status: i32 = 0,
    callback_order: u64 = 0,
    callbacks_returned: u64 = 0,

    fn at(self: *Progress, phase: Phase, order: u64) void {
        self.phase = phase;
        self.order_valid = true;
        self.order = order;
    }
    fn returned(self: *Progress, order: u64, status: i32) void {
        self.callback_returned = true;
        self.callback_status = status;
        self.callback_order = order;
        self.callbacks_returned += 1;
    }
};
pub const Result = union(enum) { ok, callback_failure: i32, numerical_failure };

/// Trusted dimensions, count>=1, interval and panels have already been checked.
/// Preserves the legacy W=0 schedule: count-1 synchronous callbacks still run.
pub fn series(comptime A: type, callback: anytype, x: anytype, coeff: anytype, y: anytype, work: anytype, rows: u64, columns: u64, count: u64, lower: f64, upper: f64, progress: *Progress) Result {
    progress.* = .{};
    progress.at(.initialize, 0);
    // Widen before adding/subtracting: even extreme finite intervals are valid.
    const center = (@as(f128, lower) + upper) / 2;
    const half = (@as(f128, upper) - lower) / 2;
    var previous = work.panel(0, columns);
    var current = work.panel(columns, columns);
    var next = work.panel(2 * columns, columns);
    const sum = work.panel(3 * columns, columns);
    for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
        const value = x.cell(i, j).*;
        previous.cell(i, j).* = value;
        sum.cell(i, j).* = A.linear(f64, coeff.cell(0, 0).*, value, 0, 0) catch return .numerical_failure;
    };
    for (1..@intCast(count)) |order| {
        const source = if (order == 1) previous else current;
        progress.at(.apply, order);
        const status = callback.apply(source, next);
        progress.returned(order, status);
        if (status != 0) return .{ .callback_failure = status };
        progress.at(.recurrence, order);
        for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
            const product = next.cell(i, j).*;
            if (!std.math.isFinite(product)) return .numerical_failure;
            const state = A.shifted(f64, product, source.cell(i, j).*, if (order == 1) 0 else previous.cell(i, j).*, (if (order == 1) @as(f128, 1) else 2) / half, center, if (order == 1) 0 else 1) catch return .numerical_failure;
            next.cell(i, j).* = state;
            const accum = sum.cell(i, j);
            accum.* = A.linear(f64, 1, accum.*, coeff.cell(order, 0).*, state) catch return .numerical_failure;
        };
        if (order > 1) std.mem.swap(@TypeOf(work), &previous, &current);
        std.mem.swap(@TypeOf(work), &current, &next);
    }
    progress.phase = .commit;
    for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
        y.cell(i, j).* = sum.cell(i, j).*;
    };
    progress.phase = .complete;
    return .ok;
}

/// Trusted dimensions, count>=1, interval and panels have already been checked.
/// W=0 skips callbacks; N=0,W>0 retains callbacks and writes zero raw moments.
pub fn rawMoments(comptime A: type, callback: anytype, x: anytype, moments: anytype, work: anytype, stage: anytype, rows: u64, columns: u64, count: u64, lower: f64, upper: f64, progress: *Progress) Result {
    progress.* = .{};
    if (columns == 0) {
        progress.phase = .complete;
        return .ok;
    }
    progress.at(.initialize, 0);
    const center = (@as(f128, lower) + upper) / 2;
    const half = (@as(f128, upper) - lower) / 2;
    var previous = work.panel(0, columns);
    var current = work.panel(columns, columns);
    var next = work.panel(2 * columns, columns);
    if (rows != 0) for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
        previous.cell(i, j).* = x.cell(i, j).*;
    };
    progress.at(.moment, 0);
    if (!momentOrder(A, x, previous, stage, rows, columns, 0)) return .numerical_failure;
    for (1..@intCast(count)) |order| {
        const source = if (order == 1) previous else current;
        progress.at(.apply, order);
        const status = callback.apply(source, next);
        progress.returned(order, status);
        if (status != 0) return .{ .callback_failure = status };
        progress.at(.recurrence, order);
        if (rows != 0) for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
            const product = next.cell(i, j).*;
            if (!std.math.isFinite(product)) return .numerical_failure;
            next.cell(i, j).* = A.shifted(f64, product, source.cell(i, j).*, if (order == 1) 0 else previous.cell(i, j).*, (if (order == 1) @as(f128, 1) else 2) / half, center, if (order == 1) 0 else 1) catch return .numerical_failure;
        };
        progress.at(.moment, order);
        if (!momentOrder(A, x, next, stage, rows, columns, order)) return .numerical_failure;
        if (order > 1) std.mem.swap(@TypeOf(work), &previous, &current);
        std.mem.swap(@TypeOf(work), &current, &next);
    }
    progress.phase = .commit;
    for (0..@intCast(columns)) |j| for (0..@intCast(count)) |k| {
        moments.cell(k, j).* = stage.cell(k, j).*;
    };
    progress.phase = .complete;
    return .ok;
}

fn momentOrder(comptime A: type, x: anytype, state: anytype, stage: anytype, rows: u64, columns: u64, order: u64) bool {
    for (0..@intCast(columns)) |j| {
        // Keep products and the entire sum wide, including cancellation.
        var sum: f128 = 0;
        for (0..@intCast(rows)) |i| sum += @as(f128, x.cell(i, j).*) * state.cell(i, j).*;
        stage.cell(order, j).* = A.produced(f64, @floatCast(sum)) catch return false;
    }
    return true;
}
