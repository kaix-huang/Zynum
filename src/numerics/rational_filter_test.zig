// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("rational_filter.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const E = @import("zynum-execution");
const C = E.context;

test "Rational filter explicit inexact iteration limit matches independent one step oracle" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = matrix(T);
        var backend = Backend(T){ .matrix = &m, .rows = 3 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const input = [_]T{ v(T, 1, 0.2), v(T, -0.3, 0.5), v(T, 2, -1) };
        var output: [5]T = @splat(v(T, 77, -33));
        const nodes = [_]T{v(T, 5, 0.8)};
        const weights = [_]T{v(T, 1, 0)};
        var q = F.Request(T){ .action = backend.action(), .input = .{ .values = &input, .shape = .{ .row_count = 3, .column_count = 1, .leading_dimension = 3, .layout = .column_major } }, .output = .{ .values = &output, .shape = .{ .row_count = 3, .column_count = 1, .leading_dimension = 3, .layout = .column_major } }, .nodes = &nodes, .weights = &weights, .global_dimension = 3, .max_iterations = 1, .relative_tolerance = 0 };
        try std.testing.expectEqual(F.Status.linear_stop, (try F.Managed(E).runScratch(T, &owner, q)).status);
        for (output) |value| try std.testing.expectEqualDeep(v(T, 77, -33), value);
        q.linear_policy = .finite_iteration_limit;
        backend.calls = 0;
        const result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.inexact_systems);
        try std.testing.expectEqual(@as(usize, 1), result.completed_systems);
        const expected = [_]T{ v(T, 0.21433970231891447, 0.12821819526690603), v(T, -0.08007010101866843, 0.04286344957654431), v(T, 0.8665990690124192, -1.1126357242499838) };
        for (expected, output[0..3]) |a, b| try expectValue(T, a, b);
        const tolerance: S.Real(T) = if (S.Real(T) == f32) 4e-6 else 3e-13;
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0.723623212200517), result.max_residual_norm, tolerance);
        for (output[3..]) |value| try std.testing.expectEqualDeep(v(T, 77, -33), value);
        const last = backend.calls;
        backend.calls = 0;
        backend.fail_at = last;
        @memset(&output, v(T, 77, -33));
        try std.testing.expectError(error.InjectedRationalAction, F.Managed(E).runScratch(T, &owner, q));
        for (output) |value| try std.testing.expectEqualDeep(v(T, 77, -33), value);
    }
}
fn v(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn done(owner: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
fn Backend(comptime T: type) type {
    return struct {
        matrix: []const T,
        rows: usize,
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        fn action(self: *@This()) O.Operator(T) {
            return .{ .context = self, .rows = self.rows, .columns = self.rows, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, op: *const C.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            if (self.close_at == self.calls) {
                _ = op.borrowedContext().requestClose();
                try std.testing.expectEqual(C.CloseResult.busy, op.borrowedContext().tryProgressClose());
            }
            for (y, 0..) |*out, row| {
                out.* = S.zero(T);
                for (x, 0..) |a, col| out.* = S.add(T, out.*, S.mul(T, self.matrix[row * self.rows + col], a));
            }
            if (self.fail_at == self.calls) return error.InjectedRationalAction;
            if (self.nan_at == self.calls and y.len != 0) y[0] = v(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
fn matrix(comptime T: type) [9]T {
    return .{ v(T, 1, 0), v(T, 0.3, 0.2), v(T, -0.2, 0.1), v(T, 0, 0), v(T, 2, 0), v(T, 0.4, -0.3), v(T, 0, 0), v(T, 0, 0), v(T, 4, 0) };
}
fn rhs(comptime T: type, row: usize, column: usize) T {
    return v(T, 1 + @as(S.Real(T), @floatFromInt(row)) * 0.25 + @as(S.Real(T), @floatFromInt(column)) * 0.5, @as(S.Real(T), @floatFromInt(row)) * 0.25 - @as(S.Real(T), @floatFromInt(column)) * 0.5);
}
fn inverseOracle(comptime T: type, node: T, column: usize) [3]T {
    const x2 = S.divv(T, rhs(T, 2, column), S.sub(T, node, v(T, 4, 0)));
    const x1 = S.divv(T, S.add(T, rhs(T, 1, column), S.mul(T, v(T, 0.4, -0.3), x2)), S.sub(T, node, v(T, 2, 0)));
    const x0 = S.divv(T, S.add(T, rhs(T, 0, column), S.add(T, S.mul(T, v(T, 0.3, 0.2), x1), S.mul(T, v(T, -0.2, 0.1), x2))), S.sub(T, node, v(T, 1, 0)));
    return .{ x0, x1, x2 };
}
fn expectValue(comptime T: type, expected: T, actual: T) !void {
    const tol: S.Real(T) = if (S.Real(T) == f32) 4e-5 else 3e-12;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tol);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tol);
}
fn position(layout: D.Layout, ld: usize, row: usize, col: usize) usize {
    return if (layout == .column_major) col * ld + row else row * ld + col;
}
test "Rational filter complex two precisions multi-RHS independent triangular oracle all layouts" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = matrix(T);
        var backend = Backend(T){ .matrix = &m, .rows = 3 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const nodes = [_]T{ v(T, 2, 0.75), v(T, 5, -0.5), v(T, 1, 0) };
        const weights = [_]T{ v(T, 0.3, 0.1), v(T, -0.2, 0.25), v(T, 0, 0) };
        for ([_]D.Layout{ .column_major, .row_major }) |il| for ([_]D.Layout{ .column_major, .row_major }) |ol| {
            const ild: usize = if (il == .column_major) 5 else 4;
            const old: usize = if (ol == .column_major) 6 else 5;
            var input: [18]T = @splat(v(T, std.math.nan(S.Real(T)), std.math.nan(S.Real(T))));
            var output: [18]T = @splat(v(T, 77, -33));
            for (0..2) |col| for (0..3) |row| {
                input[position(il, ild, row, col)] = rhs(T, row, col);
            };
            const before = input;
            backend.calls = 0;
            const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .input = .{ .values = &input, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = ild, .layout = il } }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &output, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = old, .layout = ol } }, .global_dimension = 3, .max_iterations = 32, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 });
            try std.testing.expectEqual(F.Status.success, result.status);
            try std.testing.expectEqual(@as(usize, 4), result.completed_systems);
            try std.testing.expectEqual(backend.calls, result.action_count);
            for (0..2) |col| {
                const a = inverseOracle(T, nodes[0], col);
                const b = inverseOracle(T, nodes[1], col);
                for (0..3) |row| try expectValue(T, S.add(T, S.mul(T, weights[0], a[row]), S.mul(T, weights[1], b[row])), output[position(ol, old, row, col)]);
            }
            for (output, 0..) |actual, i| {
                var logical = false;
                for (0..2) |col| for (0..3) |row| {
                    logical = logical or i == position(ol, old, row, col);
                };
                if (!logical) try std.testing.expectEqualDeep(v(T, 77, -33), actual);
            }
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&input));
        };
    }
}
test "Rational filter input output alias stages every RHS before publication" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = matrix(T);
        var backend = Backend(T){ .matrix = &m, .rows = 3 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const nodes = [_]T{v(T, 3, 1)};
        const weights = [_]T{v(T, 0.5, 0.25)};
        for ([_]D.Layout{ .column_major, .row_major }) |layout| {
            const ld: usize = if (layout == .column_major) 4 else 3;
            var data: [12]T = @splat(v(T, 77, -33));
            for (0..2) |col| for (0..3) |row| {
                data[position(layout, ld, row, col)] = rhs(T, row, col);
            };
            const shape: D.PanelShape = .{ .row_count = 3, .column_count = 2, .leading_dimension = ld, .layout = layout };
            const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .input = .{ .values = &data, .shape = shape }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &data, .shape = shape }, .global_dimension = 3, .max_iterations = 32, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 });
            try std.testing.expectEqual(F.Status.success, result.status);
            for (0..2) |col| {
                const expected = inverseOracle(T, nodes[0], col);
                for (0..3) |row| try expectValue(T, S.mul(T, weights[0], expected[row]), data[position(layout, ld, row, col)]);
            }
        }
    }
}
test "Rational filter widened sums cancel individually overflowing weighted solutions" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const zero = [_]T{v(T, 0, 0)};
        var backend = Backend(T){ .matrix = &zero, .rows = 1 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const nodes = [_]T{ v(T, 1, 0), v(T, 1, 0), v(T, 1, 0), v(T, 1, 0), v(T, 1, 0) };
        const big = std.math.floatMax(S.Real(T));
        const weights = [_]T{ v(T, big, 0), v(T, big, 0), v(T, -big, 0), v(T, -big, 0), v(T, 1, 0) };
        const b = [_]T{v(T, 2, -1)};
        var out = [_]T{ v(T, 77, -33), v(T, 78, -34) };
        const shape: D.PanelShape = .{ .row_count = 1, .column_count = 1, .leading_dimension = 1, .layout = .column_major };
        const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .input = .{ .values = &b, .shape = shape }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &out, .shape = shape }, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 });
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 5), result.completed_systems);
        try expectValue(T, b[0], out[0]);
        try std.testing.expectEqualDeep(v(T, 78, -34), out[1]);
    }
}
test "Rational filter zero coefficients empty sum and empty geometry ignore unused values and callbacks" {
    const T = S.ComplexF64;
    const m = matrix(T);
    var backend = Backend(T){ .matrix = &m, .rows = 3, .fail_at = 1 };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer done(&owner) catch unreachable;
    const b = [_]T{ v(T, std.math.nan(f64), 0), v(T, std.math.nan(f64), 0), v(T, std.math.nan(f64), 0) };
    const nodes = [_]T{v(T, 1, 0)};
    const weights = [_]T{v(T, 0, 0)};
    var out: [4]T = @splat(v(T, 77, -33));
    const shape: D.PanelShape = .{ .row_count = 3, .column_count = 1, .leading_dimension = 3, .layout = .column_major };
    var q: F.Request(T) = .{ .action = backend.action(), .input = .{ .values = &b, .shape = shape }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &out, .shape = shape }, .global_dimension = 3, .max_iterations = 0, .relative_tolerance = 0 };
    for ([_]bool{ false, true }) |empty| {
        if (empty) {
            q.nodes = &.{};
            q.weights = &.{};
        }
        const result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.completed_systems);
        for (out[0..3]) |value| try std.testing.expectEqualDeep(v(T, 0, 0), value);
        try std.testing.expectEqualDeep(v(T, 77, -33), out[3]);
    }
    q.action.rows = 0;
    q.action.columns = 0;
    q.global_dimension = 0;
    q.input = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = std.math.maxInt(usize), .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    q.output = .{ .values = &out, .shape = q.input.shape };
    q.nodes = &nodes;
    q.weights = &[_]T{v(T, 1, 0)};
    const result = try F.Managed(E).runScratch(T, &owner, q);
    try std.testing.expectEqual(F.Status.success, result.status);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
}

const P = @import("synchronization.zig");
const Red = @import("reductions.zig");
const B = @import("bicgstab.zig");
fn scalarRequest(comptime T: type, backend: *Backend(T), input: []const T, output: []T, nodes: []const T, weights: []const T) F.Request(T) {
    const shape: D.PanelShape = .{ .row_count = 1, .column_count = input.len, .leading_dimension = 1, .layout = .column_major };
    return .{ .action = backend.action(), .input = .{ .values = input, .shape = shape }, .nodes = nodes, .weights = weights, .output = .{ .values = output, .shape = shape }, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-14 };
}
fn released(owner: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
test "Rational filter controlled stops after earlier systems preserve the entire output" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = [_]T{v(T, 0, 0)};
        var backend = Backend(T){ .matrix = &m, .rows = 1 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const input = [_]T{ v(T, 2, -1), v(T, -1, 1) };
        var output: [4]T = @splat(v(T, 77, -33));
        const before = output;
        var nodes = [_]T{ v(T, 1, 0), v(T, 0, 0) };
        const weights = [_]T{ v(T, 1, 0), v(T, 1, 0) };
        var q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
        const stopped = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.linear_stop, stopped.status);
        try std.testing.expectEqual(@as(usize, 2), stopped.completed_systems);
        try std.testing.expectEqual(@as(?usize, 1), stopped.stopped_node);
        try std.testing.expectEqual(@as(?usize, 0), stopped.stopped_column);
        try std.testing.expectEqual(@as(?B.Status, .alpha_breakdown), stopped.linear_status);
        try std.testing.expect(stopped.max_residual_norm > 1);
        try std.testing.expectEqualDeep(before, output);
        try released(&owner);
        // Retry after replacing the pole; successful publication is simultaneous.
        nodes[1] = v(T, 2, 0);
        const result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 4), result.completed_systems);
        for (input, output[0..2]) |x, y| try expectValue(T, S.mul(T, v(T, 1.5, 0), x), y);
        // A zero RHS finishes before a later RHS exhausts the zero budget.
        const partly_zero = [_]T{ v(T, 0, 0), v(T, 2, -1) };
        output = before;
        q.input.values = &partly_zero;
        q.max_iterations = 0;
        const limited = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.linear_stop, limited.status);
        try std.testing.expectEqual(@as(usize, 1), limited.completed_systems);
        try std.testing.expectEqual(@as(?usize, 0), limited.stopped_node);
        try std.testing.expectEqual(@as(?usize, 1), limited.stopped_column);
        try std.testing.expectEqual(@as(?B.Status, .iteration_limit), limited.linear_status);
        try std.testing.expectEqualDeep(before, output);
    }
}
test "Rational filter late action nonfinite and final narrowing errors roll back and retry" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = [_]T{v(T, 0, 0)};
        var backend = Backend(T){ .matrix = &m, .rows = 1 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const input = [_]T{ v(T, 2, -1), v(T, -1, 1) };
        var output: [4]T = @splat(v(T, 77, -33));
        const before = output;
        const nodes = [_]T{ v(T, 1, 0), v(T, 2, 0) };
        var weights = [_]T{ v(T, 1, 0), v(T, 1, 0) };
        const q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
        backend.fail_at = 5;
        try std.testing.expectError(error.InjectedRationalAction, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        try released(&owner);
        backend.fail_at = 0;
        backend.nan_at = 5;
        backend.calls = 0;
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        try released(&owner);
        backend.nan_at = 0;
        weights[0] = v(T, std.math.floatMax(S.Real(T)), 0);
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        try released(&owner);
        weights[0] = v(T, 1, 0);
        try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runScratch(T, &owner, q)).status);
        for (input, output[0..2]) |x, y| try expectValue(T, S.mul(T, v(T, 1.5, 0), x), y);
    }
}
const Observer = struct {
    reject: ?P.Stage = null,
    index: u64 = 0,
    inner_entries: usize = 0,
    commits: usize = 0,
    fn agree(raw: *const anyopaque, operation: ?*const C.Operation, point: P.Point, local_error: ?anyerror) !void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        if (local_error) |err| return err;
        if (operation) |op| try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
        if (point.stage == .entry and point.control == @backingInt(P.Algorithm.bicgstab)) self.inner_entries += 1;
        if (point.stage == .rational_commit) self.commits += 1;
        if (self.reject == point.stage and self.index == point.index) return error.RemoteRationalFailure;
    }
    fn descriptor(self: *@This()) P.Synchronization {
        return .{ .context = self, .agree_fn = agree };
    }
};
test "Rational filter remote final commit rejection preserves all systems and shares admission" {
    const T = S.ComplexF64;
    const m = [_]T{v(T, 0, 0)};
    var backend = Backend(T){ .matrix = &m, .rows = 1 };
    var observer = Observer{ .reject = .rational_commit };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer done(&owner) catch unreachable;
    const input = [_]T{ v(T, 2, -1), v(T, -1, 1) };
    var output: [4]T = @splat(v(T, 77, -33));
    const before = output;
    const nodes = [_]T{ v(T, 1, 0), v(T, 2, 0) };
    const weights = [_]T{ v(T, 1, 0), v(T, 1, 0) };
    var q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
    q.action.synchronization = observer.descriptor();
    try std.testing.expectError(error.RemoteRationalFailure, F.Managed(E).runScratch(T, &owner, q));
    try std.testing.expectEqualDeep(before, output);
    try std.testing.expectEqual(@as(usize, 4), observer.inner_entries);
    try std.testing.expectEqual(@as(usize, 1), observer.commits);
    try released(&owner);
    observer.reject = .rational_column;
    observer.index = 1;
    try std.testing.expectError(error.RemoteRationalFailure, F.Managed(E).runScratch(T, &owner, q));
    try std.testing.expectEqualDeep(before, output);
    try released(&owner);
    observer.reject = null;
    const result = try F.Managed(E).runScratch(T, &owner, q);
    try std.testing.expectEqual(F.Status.success, result.status);
    for (input, output[0..2]) |x, y| try expectValue(T, S.mul(T, v(T, 1.5, 0), x), y);
}
test "Rational filter closing owner continues all admitted nodes and RHS" {
    const T = S.ComplexF64;
    const m = [_]T{v(T, 0, 0)};
    var backend = Backend(T){ .matrix = &m, .rows = 1, .close_at = 1 };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer done(&owner) catch unreachable;
    const input = [_]T{ v(T, 2, -1), v(T, -1, 1) };
    var output: [4]T = @splat(v(T, 77, -33));
    const nodes = [_]T{ v(T, 1, 0), v(T, 2, 0) };
    const weights = [_]T{ v(T, 1, 0), v(T, 1, 0) };
    const q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
    const result = try F.Managed(E).runScratch(T, &owner, q);
    try std.testing.expectEqual(F.Status.success, result.status);
    try std.testing.expectEqual(@as(usize, 4), result.completed_systems);
    try std.testing.expectEqual(backend.calls, result.action_count);
    try std.testing.expectError(error.Closing, owner.begin());
    try released(&owner);
}
test "Rational filter allocation failures release earlier leases before retry" {
    const T = S.ComplexF64;
    const m = [_]T{v(T, 0, 0)};
    var backend = Backend(T){ .matrix = &m, .rows = 1 };
    const input = [_]T{v(T, 2, -1)};
    var output: [2]T = @splat(v(T, 77, -33));
    const before = output;
    const nodes = [_]T{v(T, 1, 0)};
    const weights = [_]T{v(T, 1, 0)};
    const q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
    for (0..2) |budget| {
        var allocations = E.test_support.Allocations{ .budget = budget };
        var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(T, &owner, q));
        try released(&owner);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqual(@as(usize, 0), backend.calls);
        const failed = allocations.stats();
        try std.testing.expectEqual(budget, failed.accepted);
        try std.testing.expectEqual(failed.accepted, failed.freed);
        allocations.budget = 100;
        try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runScratch(T, &owner, q)).status);
        try expectValue(T, input[0], output[0]);
        try released(&owner);
        const retried = allocations.stats();
        try std.testing.expectEqual(retried.accepted, retried.freed);
        output = before;
        backend.calls = 0;
    }
}
test "Rational filter borrowed workspace capacities byte aliases options and overflow reject before callbacks" {
    const T = S.ComplexF64;
    const W = F.Wide(T);
    const m = [_]T{v(T, 0, 0)};
    var backend = Backend(T){ .matrix = &m, .rows = 1 };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer done(&owner) catch unreachable;
    var input = [_]T{v(T, 2, -1)};
    var output: [2]T = @splat(v(T, 77, -33));
    var nodes = [_]T{v(T, 1, 0)};
    var weights = [_]T{v(T, 1, 0)};
    var q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
    const length = try F.workspaceLength(T, q);
    var work: [32]T align(@alignOf(W)) = undefined;
    var wide: [8]W = undefined;
    @memset(&work, v(T, 66, -22));
    @memset(&wide, 55);
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, &owner, q, work[0 .. length.elements - 1], &wide));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, &owner, q, &work, wide[0 .. length.wide_elements - 1]));
    const alias_wide: [*]W = @ptrCast(&work);
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(T, &owner, q, &work, alias_wide[0..length.wide_elements]));
    q.nodes = work[0..1];
    work[0] = v(T, 1, 0);
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(T, &owner, q, &work, &wide));
    q.nodes = &nodes;
    q.output.values = &weights;
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runScratch(T, &owner, q));
    q.output.values = &output;
    q.relative_tolerance = -1;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runScratch(T, &owner, q));
    q.relative_tolerance = 1e-14;
    q.global_dimension = 2;
    try std.testing.expectError(error.MissingGlobalReduction, F.Managed(E).runScratch(T, &owner, q));
    q.global_dimension = 1;
    nodes[0] = v(T, std.math.nan(f64), 0);
    weights[0] = v(T, 0, 0);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runScratch(T, &owner, q));
    nodes[0] = v(T, 1, 0);
    weights[0] = v(T, 1, 0);
    q.weights = &.{};
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runScratch(T, &owner, q));
    q.weights = &weights;
    q.input.shape.leading_dimension = 0;
    try std.testing.expectError(error.InvalidLeadingDimension, F.Managed(E).runScratch(T, &owner, q));
    q.input.shape.leading_dimension = 1;
    var excessive = q;
    excessive.action.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(T, excessive));
    excessive = q;
    excessive.action.workspace_elements = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(T, excessive));
    excessive = q;
    excessive.input.shape.column_count = std.math.maxInt(usize) / 16;
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(T, excessive));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try released(&owner);
    try std.testing.expectEqual(F.Status.success, (try F.Managed(E).runBorrowed(T, &owner, q, &work, &wide)).status);
    for (work[length.elements..]) |x| try std.testing.expectEqualDeep(v(T, 66, -22), x);
    for (wide[length.wide_elements..]) |x| try std.testing.expectEqual(@as(W, 55), x);
    try expectValue(T, input[0], output[0]);
    try std.testing.expectEqualDeep(v(T, 77, -33), output[1]);
}
fn ReductionBackend(comptime T: type) type {
    return struct {
        norms: usize = 0,
        inners: usize = 0,
        fail_norm_at: usize = 0,
        nan_inner_at: usize = 0,
        empty_global: bool = false,
        fn norm(raw: *const anyopaque, operation: *const C.Operation, x: []const T) !S.Real(T) {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.norms += 1;
            try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
            if (self.fail_norm_at == self.norms) return error.RationalReductionFailure;
            if (self.empty_global) {
                try std.testing.expectEqual(@as(usize, 0), x.len);
                return 1;
            }
            return Red.localNorm(T, x);
        }
        fn inner(raw: *const anyopaque, _: *const C.Operation, x: []const T, y: []const T) !T {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.inners += 1;
            if (self.nan_inner_at == self.inners) return v(T, std.math.nan(S.Real(T)), 0);
            if (self.empty_global) {
                try std.testing.expectEqual(@as(usize, 0), x.len);
                try std.testing.expectEqual(@as(usize, 0), y.len);
                return v(T, 1, 0);
            }
            return Red.localInner(T, x, y);
        }
        fn descriptor(self: *@This()) Red.Reduction(T) {
            return .{ .context = self, .inner_fn = inner, .norm_fn = norm };
        }
    };
}
test "Rational filter shared reductions and preconditioner errors preserve output and diagnostics count real calls" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const m = [_]T{v(T, 0, 0)};
        const identity = [_]T{v(T, 1, 0)};
        var backend = Backend(T){ .matrix = &m, .rows = 1 };
        var inverse = Backend(T){ .matrix = &identity, .rows = 1, .fail_at = 2 };
        var reductions = ReductionBackend(T){};
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        const input = [_]T{ v(T, 2, -1), v(T, -1, 1) };
        var output: [4]T = @splat(v(T, 77, -33));
        const before = output;
        const nodes = [_]T{v(T, 1, 0)};
        const weights = [_]T{v(T, 1, 0)};
        var q = scalarRequest(T, &backend, &input, &output, &nodes, &weights);
        q.preconditioner = inverse.action();
        q.reduction = reductions.descriptor();
        try std.testing.expectError(error.InjectedRationalAction, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        try released(&owner);
        inverse.fail_at = 0;
        reductions.fail_norm_at = reductions.norms + 1;
        try std.testing.expectError(error.RationalReductionFailure, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        reductions.fail_norm_at = 0;
        reductions.nan_inner_at = reductions.inners + 1;
        try std.testing.expectError(error.NonFiniteReduction, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(before, output);
        reductions.nan_inner_at = 0;
        backend.calls = 0;
        inverse.calls = 0;
        const result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(backend.calls, result.action_count);
        try std.testing.expectEqual(inverse.calls, result.preconditioner_count);
        try std.testing.expect(result.preconditioner_count >= 2);
        for (input, output[0..2]) |x, y| try expectValue(T, x, y);
    }
}
test "Rational filter empty local shard participates in global solves and cannot infer convergence" {
    const T = S.ComplexF64;
    var backend = Backend(T){ .matrix = &.{}, .rows = 0 };
    var reductions = ReductionBackend(T){ .empty_global = true };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer done(&owner) catch unreachable;
    const nodes = [_]T{v(T, 1, 0)};
    const weights = [_]T{v(T, 1, 0)};
    var output = [_]T{v(T, 77, -33)};
    const shape: D.PanelShape = .{ .row_count = 0, .column_count = 2, .leading_dimension = std.math.maxInt(usize), .layout = .column_major };
    const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .input = .{ .values = &.{}, .shape = shape }, .output = .{ .values = &output, .shape = shape }, .nodes = &nodes, .weights = &weights, .global_dimension = 2, .max_iterations = 1, .relative_tolerance = 0, .reduction = reductions.descriptor() });
    try std.testing.expectEqual(F.Status.linear_stop, result.status);
    try std.testing.expectEqual(@as(?B.Status, .iteration_limit), result.linear_status);
    try std.testing.expectEqual(@as(usize, 0), result.completed_systems);
    try std.testing.expect(backend.calls >= 3 and reductions.norms > 0 and reductions.inners > 0);
    try std.testing.expectEqualDeep(v(T, 77, -33), output[0]);
}

test "Rational filter full circular contour recovers complex Hermitian spectral projector" {
    // H=[[2,i],[-i,2]] has eigenvalues 1,3 and P_1=[[1,-i],[i,1]]/2.
    // The exact projector uses its analytic eigenvectors, independently of the
    // shifted-system implementation. A full contour is essential for complex B.
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const m = [_]T{ v(T, 2, 0), v(T, 0, 1), v(T, 0, -1), v(T, 2, 0) };
        var backend = Backend(T){ .matrix = &m, .rows = 2 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer done(&owner) catch unreachable;
        var nodes: [32]T = undefined;
        var weights: [32]T = undefined;
        for (&nodes, &weights, 0..) |*node, *weight, j| {
            const angle = 2 * std.math.pi * (@as(f64, @floatFromInt(j)) + 0.5) / 32;
            const direction = v(T, @floatCast(@cos(angle)), @floatCast(@sin(angle)));
            node.* = S.add(T, v(T, 1, 0), S.mul(T, v(T, 0.5, 0), direction));
            weight.* = S.mul(T, v(T, 0.5 / 32.0, 0), direction);
        }
        const input = [_]T{ v(T, 1, 0.25), v(T, -0.5, 1), v(T, -1, -2), v(T, 0.75, -0.25) };
        var output: [6]T = @splat(v(T, 77, -33));
        const shape: D.PanelShape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 2, .layout = .column_major };
        const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .input = .{ .values = &input, .shape = shape }, .nodes = &nodes, .weights = &weights, .output = .{ .values = &output, .shape = shape }, .global_dimension = 2, .max_iterations = 12, .relative_tolerance = if (R == f32) 1e-6 else 1e-14 });
        try std.testing.expectEqual(F.Status.success, result.status);
        try std.testing.expectEqual(@as(usize, 64), result.completed_systems);
        try std.testing.expectEqual(backend.calls, result.action_count);
        for (0..2) |col| {
            const x = input[2 * col];
            const y = input[2 * col + 1];
            try expectValue(T, S.mul(T, v(T, 0.5, 0), S.add(T, x, S.mul(T, v(T, 0, -1), y))), output[2 * col]);
            try expectValue(T, S.mul(T, v(T, 0.5, 0), S.add(T, S.mul(T, v(T, 0, 1), x), y)), output[2 * col + 1]);
        }
        try std.testing.expectEqualDeep(v(T, 77, -33), output[4]);
        try std.testing.expectEqualDeep(v(T, 77, -33), output[5]);
    }
}
