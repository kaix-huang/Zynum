// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Allocation-free full complex elliptic contour quadrature for
//! (2*pi*i)^-1 integral (zI-A)^-1 dz. Nodes on the upper half are followed
//! by their conjugates; both halves are required for complex right-hand sides.
//! Geometry and counts are native, independent of any vendor FPM or ABI.
//! Publication is transactional, with no allocator, owner or thread state.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const A = @import("recurrence_arithmetic.zig");
const Store = @import("storage.zig");
pub const Rule = enum { midpoint_trapezoidal, gauss_legendre };
pub fn Options(comptime R: type) type {
    if (R != f32 and R != f64) @compileError("unsupported contour precision");
    return struct {
        lower: R,
        upper: R,
        vertical_ratio: R = 1,
        half_nodes: usize,
        rule: Rule = .midpoint_trapezoidal,
        gauss_max_iterations: usize = 64,
    };
}
pub fn Point(comptime T: type) type {
    return struct { node: T, weight: T };
}
fn supported(comptime T: type) void {
    if (T != S.ComplexF32 and T != S.ComplexF64) @compileError("complex contour precision required");
}
pub fn nodeCount(comptime T: type, options: Options(S.Real(T))) !usize {
    supported(T);
    if (options.half_nodes == 0) return error.InvalidNodeCount;
    const count = std.math.mul(usize, options.half_nodes, 2) catch return error.SizeOverflow;
    _ = try Store.byteLength(count, @sizeOf(T));
    return count;
}
fn Geometry(comptime T: type) type {
    return struct { center: A.Wide(T), horizontal: A.Wide(T), vertical: A.Wide(T), count: usize };
}
fn geometry(comptime T: type, options: Options(S.Real(T))) !Geometry(T) {
    const count = try nodeCount(T, options);
    if (!std.math.isFinite(options.lower) or !std.math.isFinite(options.upper) or !std.math.isFinite(options.vertical_ratio)) return error.NonFiniteInput;
    if (!(options.lower < options.upper)) return error.InvalidInterval;
    if (!(options.vertical_ratio > 0)) return error.InvalidRatio;
    const W = A.Wide(T);
    const center = (@as(W, options.lower) + options.upper) / 2;
    const horizontal = (@as(W, options.upper) - options.lower) / 2;
    const vertical = horizontal * options.vertical_ratio;
    if (!std.math.isFinite(center) or !std.math.isFinite(horizontal) or !std.math.isFinite(vertical) or !(horizontal > 0) or !(vertical > 0)) return error.RangeLimit;
    return .{ .center = center, .horizontal = horizontal, .vertical = vertical, .count = count };
}
const Legendre = struct { value: f64, derivative: f64 };
fn legendre(n: usize, x: f64) Legendre {
    var previous: f64 = 1;
    var current = x;
    for (2..n + 1) |k| {
        const order: f64 = @floatFromInt(k);
        const next = ((2 * order - 1) * x * current - (order - 1) * previous) / order;
        previous = current;
        current = next;
    }
    return .{ .value = current, .derivative = @as(f64, @floatFromInt(n)) * (x * current - previous) / (x * x - 1) };
}
const Abscissa = struct { x: f64, weight: f64 };
fn gauss(n: usize, index: usize, limit: usize) !Abscissa {
    const nf: f64 = @floatFromInt(n);
    var x = @cos(std.math.pi * (@as(f64, @floatFromInt(index)) + 0.75) / (nf + 0.5));
    for (0..limit) |_| {
        const p = legendre(n, x);
        const next = x - p.value / p.derivative;
        if (!std.math.isFinite(next) or !(@abs(next) < 1)) return error.QuadratureFailure;
        const stopped = @abs(next - x) <= 4 * std.math.floatEps(f64);
        x = next;
        if (stopped) {
            const derivative = legendre(n, x).derivative;
            const weight = 2 / ((1 - x * x) * derivative * derivative);
            if (!std.math.isFinite(weight) or !(weight > 0)) return error.QuadratureFailure;
            return .{ .x = x, .weight = weight };
        }
    }
    return error.QuadratureIterationLimit;
}
fn evaluated(comptime T: type, options: Options(S.Real(T)), g: Geometry(T), index: usize) !Point(T) {
    if (index >= g.count) return error.InvalidIndex;
    const upper_index = index % options.half_nodes;
    const lower_half = index >= options.half_nodes;
    const angle, const weight_scale = switch (options.rule) {
        .midpoint_trapezoidal => .{ std.math.pi * (@as(f64, @floatFromInt(upper_index)) + 0.5) / @as(f64, @floatFromInt(options.half_nodes)), 1 / @as(f64, @floatFromInt(g.count)) },
        .gauss_legendre => blk: {
            const p = try gauss(options.half_nodes, upper_index, options.gauss_max_iterations);
            break :blk .{ std.math.pi * (1 + p.x) / 2, p.weight / 4 };
        },
    };
    const W = A.Wide(T);
    const cosine: W = @cos(angle);
    const sine: W = @sin(angle);
    const node: T = .{ .re = @floatCast(g.center + g.horizontal * cosine), .im = @floatCast(g.vertical * sine) };
    const weight: T = .{ .re = @floatCast(g.vertical * cosine * @as(W, weight_scale)), .im = @floatCast(g.horizontal * sine * @as(W, weight_scale)) };
    if (!std.math.isFinite(node.re) or !std.math.isFinite(node.im) or !(node.im > 0) or !std.math.isFinite(weight.re) or !std.math.isFinite(weight.im) or S.isZero(T, weight)) return error.RangeLimit;
    return .{ .node = .{ .re = node.re, .im = if (lower_half) -node.im else node.im }, .weight = .{ .re = weight.re, .im = if (lower_half) -weight.im else weight.im } };
}
pub fn point(comptime T: type, options: Options(S.Real(T)), index: usize) !Point(T) {
    return evaluated(T, options, try geometry(T, options), index);
}
pub fn populate(comptime T: type, options: Options(S.Real(T)), nodes: []T, weights: []T) !void {
    const g = try geometry(T, options);
    if (nodes.len < g.count or weights.len < g.count) return error.WorkspaceTooSmall;
    const length = try Store.byteLength(g.count, @sizeOf(T));
    if (Store.overlaps(try Store.byteRange(@intFromPtr(nodes.ptr), length), try Store.byteRange(@intFromPtr(weights.ptr), length))) return error.AliasingNotAllowed;
    // First evaluate every entry without publication. A late representability
    // or Newton-iteration failure cannot expose a partial contour.
    for (0..g.count) |index| _ = try evaluated(T, options, g, index);
    for (nodes[0..g.count], weights[0..g.count], 0..) |*node, *weight, index| {
        const p = try evaluated(T, options, g, index);
        node.* = p.node;
        weight.* = p.weight;
    }
}
