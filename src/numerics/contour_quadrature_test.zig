// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Q = @import("contour_quadrature.zig");
const F = @import("rational_filter.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const E = @import("zynum-execution");
fn z(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn close(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}
fn expectValue(comptime T: type, expected: T, actual: T) !void {
    const tolerance: S.Real(T) = if (S.Real(T) == f32) 3e-5 else 3e-12;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}
test "Contour quadrature two precisions two rules satisfy Cauchy projector and Laurent moments" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| for ([_]Q.Rule{ .midpoint_trapezoidal, .gauss_legendre }) |rule| {
        var nodes: [128]T = undefined;
        var weights: [128]T = undefined;
        try Q.populate(T, .{ .lower = -1, .upper = 3, .vertical_ratio = 0.6, .half_nodes = 64, .rule = rule }, &nodes, &weights);
        for ([_]S.Real(T){ 0.5, 1, 7 }) |lambda| {
            var response = z(T, 0, 0);
            for (nodes, weights) |node, weight| response = S.add(T, response, S.divv(T, weight, S.sub(T, node, z(T, lambda, 0))));
            try expectValue(T, z(T, if (lambda == 7) 0 else 1, 0), response);
        }
        for (0..3) |power| {
            var moment = z(T, 0, 0);
            for (nodes, weights) |node, weight| {
                const centered = S.sub(T, node, z(T, 1, 0));
                var denominator = centered;
                for (0..power) |_| denominator = S.mul(T, denominator, centered);
                moment = S.add(T, moment, S.divv(T, weight, denominator));
            }
            try expectValue(T, z(T, if (power == 0) 1 else 0, 0), moment);
        }
    };
}
test "Contour quadrature conjugate pairs and tails are exact and caller buffers stay separate" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| for ([_]Q.Rule{ .midpoint_trapezoidal, .gauss_legendre }) |rule| {
        var nodes: [10]T = @splat(z(T, 77, -33));
        var weights: [10]T = @splat(z(T, 78, -34));
        try Q.populate(T, .{ .lower = 0, .upper = 2, .vertical_ratio = 0.3, .half_nodes = 4, .rule = rule }, &nodes, &weights);
        for (0..4) |j| {
            try std.testing.expect(nodes[j].im > 0);
            try std.testing.expectEqual(nodes[j].re, nodes[j + 4].re);
            try std.testing.expectEqual(-nodes[j].im, nodes[j + 4].im);
            try std.testing.expectEqual(weights[j].re, weights[j + 4].re);
            try std.testing.expectEqual(-weights[j].im, weights[j + 4].im);
        }
        for (nodes[8..]) |v| try std.testing.expectEqualDeep(z(T, 77, -33), v);
        for (weights[8..]) |v| try std.testing.expectEqualDeep(z(T, 78, -34), v);
    };
}
test "Contour quadrature widened geometry accepts opposite extreme endpoints without overflow" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const largest = std.math.floatMax(R);
        var nodes: [4]T = undefined;
        var weights: [4]T = undefined;
        try Q.populate(T, .{ .lower = -largest, .upper = largest, .half_nodes = 2 }, &nodes, &weights);
        for (nodes, weights) |node, weight| {
            try std.testing.expect(std.math.isFinite(node.re) and std.math.isFinite(node.im));
            try std.testing.expect(std.math.isFinite(weight.re) and std.math.isFinite(weight.im));
            try std.testing.expect(@abs(node.im) > largest / 2);
        }
        const small = try Q.point(T, .{ .lower = largest / 2, .upper = largest, .half_nodes = 2 }, 0);
        try std.testing.expect(std.math.isFinite(small.node.re));
        try std.testing.expect(small.node.re >= largest / 2 and small.node.re <= largest);
    }
}
test "Contour quadrature options capacity aliases overflow and iteration failure preserve outputs" {
    const T = S.ComplexF64;
    var nodes: [8]T = @splat(z(T, 77, -33));
    var weights: [8]T = @splat(z(T, 78, -34));
    const before_nodes = nodes;
    const before_weights = weights;
    var options = Q.Options(f64){ .lower = 0, .upper = 2, .half_nodes = 4 };
    try std.testing.expectError(error.WorkspaceTooSmall, Q.populate(T, options, nodes[0..7], &weights));
    try std.testing.expectError(error.AliasingNotAllowed, Q.populate(T, options, &nodes, &nodes));
    options.half_nodes = 0;
    try std.testing.expectError(error.InvalidNodeCount, Q.populate(T, options, &nodes, &weights));
    options.half_nodes = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, Q.nodeCount(T, options));
    options.half_nodes = 4;
    options.lower = options.upper;
    try std.testing.expectError(error.InvalidInterval, Q.populate(T, options, &nodes, &weights));
    options.lower = 0;
    options.vertical_ratio = 0;
    try std.testing.expectError(error.InvalidRatio, Q.populate(T, options, &nodes, &weights));
    options.vertical_ratio = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, Q.populate(T, options, &nodes, &weights));
    options.vertical_ratio = 1;
    options.rule = .gauss_legendre;
    options.gauss_max_iterations = 0;
    try std.testing.expectError(error.QuadratureIterationLimit, Q.populate(T, options, &nodes, &weights));
    try std.testing.expectError(error.InvalidIndex, Q.point(T, options, 8));
    try std.testing.expectEqualDeep(before_nodes, nodes);
    try std.testing.expectEqualDeep(before_weights, weights);
}
test "Contour quadrature late range failure rolls back both arrays and tiny imaginary geometry rejects" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const largest = std.math.floatMax(R);
        var nodes: [6]T = @splat(z(T, 77, -33));
        var weights: [6]T = @splat(z(T, 78, -34));
        const before_nodes = nodes;
        const before_weights = weights;
        const options = Q.Options(R){ .lower = -largest, .upper = largest, .vertical_ratio = 1.5, .half_nodes = 3 };
        _ = try Q.point(T, options, 0);
        try std.testing.expectError(error.RangeLimit, Q.point(T, options, 1));
        try std.testing.expectError(error.RangeLimit, Q.populate(T, options, &nodes, &weights));
        const tiny = std.math.floatTrueMin(R);
        try std.testing.expectError(error.RangeLimit, Q.populate(T, .{ .lower = 0, .upper = tiny, .vertical_ratio = tiny, .half_nodes = 3 }, &nodes, &weights));
        try std.testing.expectEqualDeep(before_nodes, nodes);
        try std.testing.expectEqualDeep(before_weights, weights);
    }
}
fn HermitianAction(comptime T: type) type {
    return struct {
        calls: usize = 0,
        fn action(self: *@This()) O.Operator(T) {
            return .{ .context = self, .rows = 2, .columns = 2, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, operation: *const E.context.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
            y[0] = S.add(T, S.mul(T, z(T, 2, 0), x[0]), S.mul(T, z(T, 0, 1), x[1]));
            y[1] = S.add(T, S.mul(T, z(T, 0, -1), x[0]), S.mul(T, z(T, 2, 0), x[1]));
        }
    };
}
test "Contour quadrature both rules feed shared rational solver and recover analytic complex projector" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| for ([_]Q.Rule{ .midpoint_trapezoidal, .gauss_legendre }) |rule| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        var backend = HermitianAction(T){};
        var nodes: [64]T = undefined;
        var weights: [64]T = undefined;
        try Q.populate(T, .{ .lower = 0.5, .upper = 1.5, .half_nodes = 32, .rule = rule }, &nodes, &weights);
        const input = [_]T{ z(T, 1, 0.25), z(T, -0.5, 1), z(T, -1, -2), z(T, 0.75, -0.25) };
        var output: [6]T = @splat(z(T, 77, -33));
        const shape: D.PanelShape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 2, .layout = .column_major };
        const result = try F.Managed(E).runScratch(T, owner, .{ .action = backend.action(), .input = .{ .values = &input, .shape = shape }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &output, .shape = shape }, .global_dimension = 2, .max_iterations = 16, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 });
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 128), result.completed_systems);
        try std.testing.expectEqual(backend.calls, result.action_count);
        for (0..2) |column| {
            const a = input[2 * column];
            const b = input[2 * column + 1];
            try expectValue(T, S.mul(T, z(T, 0.5, 0), S.add(T, a, S.mul(T, z(T, 0, -1), b))), output[2 * column]);
            try expectValue(T, S.mul(T, z(T, 0.5, 0), S.add(T, S.mul(T, z(T, 0, 1), a), b)), output[2 * column + 1]);
        }
        try std.testing.expectEqualDeep(z(T, 77, -33), output[4]);
        try std.testing.expectEqualDeep(z(T, 77, -33), output[5]);
    };
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
