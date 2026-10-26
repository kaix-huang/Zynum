// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared stable local reductions. No admission, allocation or communication.
const std = @import("std");
const scalar = @import("../blas/core/shared/scalar.zig");
const execution = @import("zynum-execution");
/// Shared borrowed scalar reductions for native operator algorithms.
pub fn Reduction(comptime T: type) type {
    return struct {
        context: *const anyopaque,
        inner_fn: *const fn (*const anyopaque, *const execution.context.Operation, []const T, []const T) anyerror!T,
        norm_fn: *const fn (*const anyopaque, *const execution.context.Operation, []const T) anyerror!scalar.Real(T),
    };
}
pub fn inner(comptime T: type, operation: *const execution.context.Operation, reduction: ?Reduction(T), x: []const T, y: []const T) !T {
    const value = if (reduction) |r| try r.inner_fn(r.context, operation, x, y) else try localInner(T, x, y);
    if (!finite(T, value)) return error.NonFiniteReduction;
    return value;
}
pub fn norm(comptime T: type, operation: *const execution.context.Operation, reduction: ?Reduction(T), x: []const T) !scalar.Real(T) {
    const value = if (reduction) |r| try r.norm_fn(r.context, operation, x) else try localNorm(T, x);
    if (!std.math.isFinite(value) or value < 0) return error.NonFiniteReduction;
    return value;
}
fn wide(comptime T: type) type {
    return if (scalar.Real(T) == f32) f64 else f128;
}
pub fn finite(comptime T: type, x: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(x.re) and std.math.isFinite(x.im) else std.math.isFinite(x);
}
fn checkFinite(comptime T: type, values: []const T) !void {
    for (values) |x| if (!finite(T, x)) return error.NonFiniteInput;
}
/// Stable local reductions are public internal helpers for caller composition.
/// They neither admit execution nor perform communication.
pub fn localInner(comptime T: type, x: []const T, y: []const T) !T {
    if (x.len != y.len) return error.DimensionMismatch;
    try checkFinite(T, x);
    try checkFinite(T, y);
    const W = wide(T);
    var re: W = 0;
    var im: W = 0;
    for (x, y) |a, b| {
        if (comptime scalar.isComplex(T)) {
            re += @as(W, a.re) * b.re + @as(W, a.im) * b.im;
            im += @as(W, a.re) * b.im - @as(W, a.im) * b.re;
        } else re += @as(W, a) * b;
    }
    const result: T = if (comptime scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
    if (!finite(T, result)) return error.NonFiniteReduction;
    return result;
}
pub fn localNorm(comptime T: type, x: []const T) !scalar.Real(T) {
    try checkFinite(T, x);
    const W = wide(T);
    var sum: W = 0;
    for (x) |a| {
        if (comptime scalar.isComplex(T)) sum += @as(W, a.re) * a.re + @as(W, a.im) * a.im else sum += @as(W, a) * a;
    }
    const result: scalar.Real(T) = @floatCast(@sqrt(sum));
    if (!std.math.isFinite(result)) return error.NonFiniteReduction;
    return result;
}
