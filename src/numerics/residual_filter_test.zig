// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("rational_filter.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const E = @import("zynum-execution");
fn z(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn done(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}
fn Backend(comptime T: type) type {
    return struct {
        order: usize = 3,
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        matrix: [9]T = .{ z(T, 1, 0), z(T, 0.3, 0.2), z(T, -0.2, 0.1), z(T, 0, 0), z(T, 2, 0), z(T, 0.4, -0.3), z(T, 0, 0), z(T, 0, 0), z(T, 4, 0) },
        fn action(self: *@This()) O.Operator(T) {
            return .{ .context = self, .rows = self.order, .columns = self.order, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, op: *const E.context.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            if (self.close_at == self.calls) {
                _ = op.borrowedContext().requestClose();
                try std.testing.expectEqual(E.context.CloseResult.busy, fixtureProgressClose(op.borrowedContext()));
            }
            for (y, 0..) |*out, row| {
                out.* = S.zero(T);
                for (x, 0..) |value, column| out.* = S.add(T, out.*, S.mul(T, self.matrix[row * self.order + column], value));
            }
            if (self.fail_at == self.calls) return error.InjectedResidualFilterAction;
            if (self.nan_at == self.calls) y[0].re = std.math.nan(S.Real(T));
        }
    };
}
fn rhs(comptime T: type, row: usize, column: usize) T {
    return z(T, 1 + @as(S.Real(T), @floatFromInt(row)) * 0.25 + @as(S.Real(T), @floatFromInt(column)) * 0.5, @as(S.Real(T), @floatFromInt(row)) * 0.25 - @as(S.Real(T), @floatFromInt(column)) * 0.5);
}
fn inverse(comptime T: type, node: T, column: usize) [3]T {
    const c = S.divv(T, rhs(T, 2, column), S.sub(T, node, z(T, 4, 0)));
    const b = S.divv(T, S.add(T, rhs(T, 1, column), S.mul(T, z(T, 0.4, -0.3), c)), S.sub(T, node, z(T, 2, 0)));
    const a = S.divv(T, S.add(T, rhs(T, 0, column), S.add(T, S.mul(T, z(T, 0.3, 0.2), b), S.mul(T, z(T, -0.2, 0.1), c))), S.sub(T, node, z(T, 1, 0)));
    return .{ a, b, c };
}
fn near(comptime T: type, a: T, b: T) !void {
    const tolerance: S.Real(T) = if (S.Real(T) == f32) 5e-5 else 8e-13;
    try std.testing.expectApproxEqAbs(a.re, b.re, tolerance);
    try std.testing.expectApproxEqAbs(a.im, b.im, tolerance);
}
fn pos(layout: D.Layout, ld: usize, row: usize, column: usize) usize {
    return if (layout == .column_major) row + column * ld else column + row * ld;
}
test "Residual contour filter two precisions independent triangular resolvent all layouts" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer done(owner) catch unreachable;
        var backend = Backend(T){};
        const nodes = [_]T{ z(T, 2, 0.7), z(T, 5, -0.4), z(T, 1.5, 0) };
        const weights = [_]T{ z(T, 0.3, 0.1), z(T, -0.2, 0.15), z(T, 0, 0) };
        const lambdas = [_]S.Real(T){ 1.5, 2.75 };
        for ([_]D.Layout{ .row_major, .column_major }) |bl| for ([_]D.Layout{ .row_major, .column_major }) |rl| for ([_]D.Layout{ .row_major, .column_major }) |ol| {
            const bld: usize = if (bl == .column_major) 5 else 4;
            const rld: usize = if (rl == .column_major) 5 else 4;
            const old: usize = if (ol == .column_major) 6 else 5;
            var basis: [18]T = @splat(z(T, std.math.nan(S.Real(T)), std.math.nan(S.Real(T))));
            var residual = basis;
            var output: [18]T = @splat(z(T, 77, -33));
            for (0..2) |column| for (0..3) |row| {
                basis[pos(bl, bld, row, column)] = rhs(T, row, column);
                var ax = S.zero(T);
                for (0..3) |k| ax = S.add(T, ax, S.mul(T, backend.matrix[row * 3 + k], rhs(T, k, column)));
                residual[pos(rl, rld, row, column)] = S.sub(T, ax, S.mul(T, z(T, lambdas[column], 0), rhs(T, row, column)));
            };
            const before_basis = basis;
            const before_residual = residual;
            backend.calls = 0;
            const result = try F.Managed(E).runScratch(T, owner, .{ .action = backend.action(), .input = .{ .values = &residual, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = rld, .layout = rl } }, .ritz_correction = .{ .basis = .{ .values = &basis, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = bld, .layout = bl } }, .eigenvalues = &lambdas }, .output = .{ .values = &output, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = old, .layout = ol } }, .nodes = &nodes, .weights = &weights, .global_dimension = 3, .max_iterations = 32, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 });
            try std.testing.expectEqual(F.Status.success, result.status);
            try std.testing.expectEqual(@as(usize, 4), result.completed_systems);
            try std.testing.expectEqual(backend.calls, result.action_count);
            for (0..2) |column| {
                const a = inverse(T, nodes[0], column);
                const b = inverse(T, nodes[1], column);
                for (0..3) |row| try near(T, S.add(T, S.mul(T, weights[0], a[row]), S.mul(T, weights[1], b[row])), output[pos(ol, old, row, column)]);
            }
            for (output, 0..) |value, index| {
                var logical = false;
                for (0..2) |column| for (0..3) |row| {
                    logical = logical or index == pos(ol, old, row, column);
                };
                if (!logical) try std.testing.expectEqualDeep(z(T, 77, -33), value);
            }
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_basis), std.mem.asBytes(&basis));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_residual), std.mem.asBytes(&residual));
        };
    }
}
test "Residual contour filter full input basis output overlap late failures retry and original close" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer done(owner) catch unreachable;
        var backend = Backend(T){ .order = 1 };
        const initial = [_]T{ z(T, 1, 0.5), z(T, -0.3, 0.4), z(T, 77, -33) };
        var shared = initial;
        const nodes = [_]T{ z(T, 2, 1), z(T, 3, -0.5) };
        const weights = [_]T{ z(T, 0.3, 0.1), z(T, -0.2, 0.15) };
        const lambdas = [_]S.Real(T){ 0, 0 };
        const shape = D.PanelShape{ .row_count = 1, .column_count = 2, .leading_dimension = 1, .layout = .column_major };
        const q = F.Request(T){ .action = backend.action(), .input = .{ .values = &shared, .shape = shape }, .ritz_correction = .{ .basis = .{ .values = &shared, .shape = shape }, .eigenvalues = &lambdas }, .output = .{ .values = &shared, .shape = shape }, .nodes = &nodes, .weights = &weights, .global_dimension = 1, .max_iterations = 8, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 };
        const baseline = try F.Managed(E).runScratch(T, owner, q);
        try std.testing.expectEqual(F.Status.success, baseline.status);
        const response = S.add(T, S.divv(T, weights[0], S.sub(T, nodes[0], z(T, 1, 0))), S.divv(T, weights[1], S.sub(T, nodes[1], z(T, 1, 0))));
        for (initial[0..2], shared[0..2]) |a, b| try near(T, S.mul(T, response, a), b);
        try std.testing.expectEqualDeep(initial[2], shared[2]);
        const last = backend.calls;
        backend.calls = 0;
        backend.fail_at = last;
        shared = initial;
        try std.testing.expectError(error.InjectedResidualFilterAction, F.Managed(E).runScratch(T, owner, q));
        try std.testing.expectEqualDeep(initial, shared);
        backend.calls = 0;
        backend.fail_at = 0;
        backend.nan_at = last;
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, q));
        try std.testing.expectEqualDeep(initial, shared);
        backend.calls = 0;
        backend.nan_at = 0;
        backend.close_at = 1;
        try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runScratch(T, owner, q)).status);
        try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
    }
}
test "Residual contour filter correction geometry shifts byte aliases finite policy and scratch tails" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer done(owner) catch unreachable;
    var backend = Backend(T){ .order = 1 };
    const input = [_]T{z(T, 0, 0)};
    var basis = [_]T{z(T, 1, 0)};
    var output = [_]T{ z(T, 77, -33), z(T, 78, -34) };
    var lambdas = [_]f64{1};
    var nodes = [_]T{z(T, 2, 1)};
    var weights = [_]T{z(T, 1, 0)};
    const shape = D.PanelShape{ .row_count = 1, .column_count = 1, .leading_dimension = 1, .layout = .column_major };
    var q = F.Request(T){ .action = backend.action(), .input = .{ .values = &input, .shape = shape }, .ritz_correction = .{ .basis = .{ .values = &basis, .shape = shape }, .eigenvalues = &lambdas }, .output = .{ .values = &output, .shape = shape }, .nodes = &nodes, .weights = &weights, .global_dimension = 1, .max_iterations = 8, .relative_tolerance = 1e-14 };
    var work: [14]T = @splat(z(T, 66, -22));
    var wide: [3]f128 = @splat(44);
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, work[0..12], &wide));
    q.ritz_correction.?.eigenvalues = &.{};
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runBorrowed(T, owner, q, &work, &wide));
    q.ritz_correction.?.eigenvalues = &lambdas;
    lambdas[0] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(T, owner, q, &work, &wide));
    lambdas[0] = 1;
    nodes[0] = z(T, 1, 0);
    try std.testing.expectError(error.ShiftCoincidesWithRitzValue, F.Managed(E).runBorrowed(T, owner, q, &work, &wide));
    nodes[0] = z(T, 2, 1);
    const raw: [*]f64 = @ptrCast(&output);
    q.ritz_correction.?.eigenvalues = raw[0..1];
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(T, owner, q, &work, &wide));
    q.ritz_correction.?.eigenvalues = &lambdas;
    try std.testing.expectEqualDeep(z(T, 77, -33), output[0]);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runBorrowed(T, owner, q, &work, &wide)).status);
    try near(T, z(T, 0.5, -0.5), output[0]);
    try std.testing.expectEqualDeep(z(T, 78, -34), output[1]);
    try std.testing.expectEqualDeep(z(T, 66, -22), work[13]);
    try std.testing.expectEqual(@as(f128, 44), wide[2]);
    weights[0] = z(T, 0, 0);
    basis[0] = z(T, std.math.nan(f64), 0);
    try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runBorrowed(T, owner, q, &work, &wide)).status);
    try std.testing.expectEqualDeep(z(T, 0, 0), output[0]);
}
test "Residual contour filter widens opposite extreme shift and Ritz values before division" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer done(owner) catch unreachable;
        var backend = Backend(T){ .order = 1 };
        const input = [_]T{z(T, 0, 0)};
        const basis = [_]T{z(T, 1, 0)};
        var output = [_]T{z(T, 77, -33)};
        const maximum = std.math.floatMax(S.Real(T));
        const lambdas = [_]S.Real(T){-maximum};
        const nodes = [_]T{z(T, maximum, 1)};
        const weights = [_]T{z(T, maximum, 0)};
        const shape = D.PanelShape{ .row_count = 1, .column_count = 1, .leading_dimension = 1, .layout = .column_major };
        const result = try F.Managed(E).runScratch(T, owner, .{ .action = backend.action(), .input = .{ .values = &input, .shape = shape }, .ritz_correction = .{ .basis = .{ .values = &basis, .shape = shape }, .eigenvalues = &lambdas }, .output = .{ .values = &output, .shape = shape }, .nodes = &nodes, .weights = &weights, .global_dimension = 1, .max_iterations = 8, .relative_tolerance = 1e-14 });
        try std.testing.expectEqual(F.Status.success, result.status);
        try near(T, z(T, 0.5, 0), output[0]);
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
