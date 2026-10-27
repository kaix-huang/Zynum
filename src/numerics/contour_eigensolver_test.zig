// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("contour_eigensolver.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const P = @import("synchronization.zig");
const E = @import("zynum-execution");
const H = @import("../lapack/core/hermitian_eigen.zig");
const Red = @import("reductions.zig");
const Projection = @import("subspace_projection.zig");
fn z(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn close(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}
fn Backend(comptime T: type) type {
    return struct {
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        diagonal: bool = false,
        repeated: bool = false,
        distort_at: usize = 0,
        fn action(self: *@This()) O.Operator(T) {
            return .{ .context = self, .rows = 4, .columns = 4, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, op: *const E.context.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            if (self.close_at == self.calls) {
                _ = op.borrowedContext().requestClose();
                try std.testing.expectEqual(E.context.CloseResult.busy, fixtureProgressClose(op.borrowedContext()));
            }
            if (self.diagonal) {
                const eigenvalues = [_]S.Real(T){ 1, if (self.repeated) 1 else 2, 4, 8 };
                for (y, x, eigenvalues) |*out, in, lambda| out.* = S.mul(T, z(T, lambda, 0), in);
            } else {
                y[0] = S.add(T, S.mul(T, z(T, 2, 0), x[0]), S.mul(T, z(T, 0, 1), x[1]));
                y[1] = S.add(T, S.mul(T, z(T, 0, -1), x[0]), S.mul(T, z(T, 2, 0), x[1]));
                y[2] = S.mul(T, z(T, 5, 0), x[2]);
                y[3] = S.mul(T, z(T, 8, 0), x[3]);
            }
            if (self.fail_at == self.calls) return error.InjectedContourAction;
            if (self.nan_at == self.calls) y[0].re = std.math.nan(S.Real(T));
            if (self.distort_at != 0 and self.calls >= self.distort_at) y[0].re += 0.1;
        }
    };
}
fn position(layout: D.Layout, ld: usize, row: usize, column: usize) usize {
    return if (layout == .column_major) row + column * ld else column + row * ld;
}
fn Fixture(comptime T: type) type {
    return struct {
        input: [24]T = undefined,
        output: [24]T = undefined,
        values: [6]S.Real(T) = undefined,
        errors: [6]S.Real(T) = undefined,
        shape: D.PanelShape = undefined,
        fn init(self: *@This(), layout: D.Layout, columns: usize, identity: bool) void {
            self.shape = .{ .row_count = 4, .column_count = columns, .leading_dimension = if (layout == .column_major) 6 else 5, .layout = layout };
            self.input = @splat(z(T, std.math.nan(S.Real(T)), std.math.nan(S.Real(T))));
            self.reset();
            for (0..columns) |column| for (0..4) |row| {
                self.input[position(layout, self.shape.leading_dimension, row, column)] = if (identity) z(T, if (row == column) 1 else 0, 0) else z(T, if (row == column) 1 else 0.13 * @as(S.Real(T), @floatFromInt(row + column + 1)), 0.07 * @as(S.Real(T), @floatFromInt(row)) - 0.11 * @as(S.Real(T), @floatFromInt(column)));
            };
        }
        fn reset(self: *@This()) void {
            self.output = @splat(z(T, 77, -33));
            self.values = @splat(91);
            self.errors = @splat(92);
        }
        fn request(self: *@This(), backend: *Backend(T)) F.Request(T) {
            return .{ .action = backend.action(), .initial = .{ .values = &self.input, .shape = self.shape }, .vectors = .{ .values = &self.output, .shape = self.shape }, .eigenvalues = &self.values, .residuals = &self.errors, .global_dimension = 4, .contour = .{ .lower = 0.5, .upper = 3.5, .half_nodes = 8, .rule = .gauss_legendre }, .max_iterations = 12, .relative_tolerance = if (S.Real(T) == f32) 3e-4 else 3e-10, .linear_max_iterations = 64, .linear_relative_tolerance = if (S.Real(T) == f32) 1e-6 else 1e-13 };
        }
        fn untouched(self: *@This()) !void {
            for (self.output) |v| try std.testing.expectEqualDeep(z(T, 77, -33), v);
            for (self.values) |v| try std.testing.expectEqual(@as(S.Real(T), 91), v);
            for (self.errors) |v| try std.testing.expectEqual(@as(S.Real(T), 92), v);
        }
    };
}
fn physical(comptime T: type, f: *Fixture(T), columns: usize) !void {
    const tolerance: S.Real(T) = if (S.Real(T) == f32) 1e-3 else 2e-9;
    for (0..columns) |column| {
        try std.testing.expectApproxEqAbs(@as(S.Real(T), if (column == 0) 1 else 3), f.values[column], tolerance);
        try std.testing.expect(f.errors[column] < tolerance);
        const a = f.output[position(f.shape.layout, f.shape.leading_dimension, 0, column)];
        const b = f.output[position(f.shape.layout, f.shape.leading_dimension, 1, column)];
        const cross = S.mul(T, a, z(T, b.re, -b.im));
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0), cross.re, tolerance);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), if (column == 0) -0.5 else 0.5), cross.im, tolerance);
        for (2..4) |row| {
            const v = f.output[position(f.shape.layout, f.shape.leading_dimension, row, column)];
            try std.testing.expect(@abs(v.re) + @abs(v.im) < tolerance);
        }
    }
    for (f.output, 0..) |v, i| {
        var logical = false;
        for (0..columns) |column| for (0..4) |row| {
            logical = logical or i == position(f.shape.layout, f.shape.leading_dimension, row, column);
        };
        if (!logical) try std.testing.expectEqualDeep(z(T, 77, -33), v);
    }
    for (f.values[columns..]) |v| try std.testing.expectEqual(@as(S.Real(T), 91), v);
    for (f.errors[columns..]) |v| try std.testing.expectEqual(@as(S.Real(T), 92), v);
}
test "Contour eigensolver two precisions both panels recover complex independent projectors" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| for ([_]D.Layout{ .column_major, .row_major }) |layout| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        var backend = Backend(T){};
        var f: Fixture(T) = undefined;
        f.init(layout, 3, false);
        const before = f.input;
        const result = try F.Managed(E).runScratch(T, owner, f.request(&backend));
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.found);
        try std.testing.expectEqual(@as(usize, 2), result.converged_pairs);
        try std.testing.expectEqual(backend.calls, result.action_count);
        try physical(T, &f, 2);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&f.input));
    };
}
test "Contour eigensolver overlapping initial output publishes only selected columns" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){};
    var f: Fixture(T) = undefined;
    f.init(.row_major, 3, false);
    f.output = f.input;
    const before = f.output;
    var q = f.request(&backend);
    q.initial.values = &f.output;
    const result = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 2), result.found);
    for (0..24) |i| {
        var logical = false;
        for (0..2) |column| for (0..4) |row| {
            logical = logical or i == position(.row_major, 5, row, column);
        };
        if (!logical) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[i]), std.mem.asBytes(&f.output[i]));
    }
}
test "Contour eigensolver capacity warning and closed endpoint multiplicity preserve semantics" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 2, true);
    var q = f.request(&backend);
    q.contour.upper = 2.5;
    const full = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.capacity_limit, full.status);
    try std.testing.expectEqual(@as(usize, 2), full.found);
    try f.untouched();
    f.init(.row_major, 3, true);
    backend.repeated = true;
    q = f.request(&backend);
    q.contour.lower = 1;
    q.contour.upper = 2;
    const boundary = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, boundary.status);
    try std.testing.expectEqual(@as(usize, 2), boundary.found);
    try std.testing.expectEqual(@as(f64, 1), f.values[0]);
    try std.testing.expectEqual(@as(f64, 1), f.values[1]);
    try std.testing.expectEqual(@as(f64, 91), f.values[2]);
}
test "Contour eigensolver zero space absent interval and zero budget never certify missing spectrum" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, true);
    var q = f.request(&backend);
    q.max_iterations = 0;
    f.input = @splat(z(T, std.math.nan(f64), 0));
    try std.testing.expectEqual(F.Status.iteration_limit, (try F.Managed(E).runScratch(T, owner, q)).status);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try f.untouched();
    f.input = @splat(z(T, 0, 0));
    q.max_iterations = 2;
    try std.testing.expectEqual(F.Status.empty_subspace, (try F.Managed(E).runScratch(T, owner, q)).status);
    try f.untouched();
    f.init(.column_major, 4, true);
    q = f.request(&backend);
    q.contour.lower = 9;
    q.contour.upper = 10;
    q.max_iterations = 2;
    const absent = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.iteration_limit, absent.status);
    try std.testing.expectEqual(@as(usize, 0), absent.found);
    try std.testing.expectEqual(@as(usize, 2), absent.iterations);
    try f.untouched();
}
test "Contour eigensolver linear and reduced solver stops retain full output and diagnostics" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){};
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, false);
    var q = f.request(&backend);
    q.linear_max_iterations = 0;
    const linear = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.linear_stop, linear.status);
    try std.testing.expect(linear.linear.?.status == .linear_stop);
    try f.untouched();
    q = f.request(&backend);
    q.eigen_options.max_sweeps = 0;
    const ritz = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.ritz_stop, ritz.status);
    try std.testing.expect(ritz.ritz.?.status != .converged);
    try f.untouched();
}
test "Contour eigensolver late action failure nonfinite final action and retry are transactional" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, true);
    const q = f.request(&backend);
    const baseline = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, baseline.status);
    const last = backend.calls;
    try std.testing.expect(last > baseline.completed_systems);
    f.reset();
    backend.calls = 0;
    backend.fail_at = last;
    try std.testing.expectError(error.InjectedContourAction, F.Managed(E).runScratch(T, owner, q));
    try f.untouched();
    backend.calls = 0;
    backend.fail_at = 0;
    backend.nan_at = last;
    try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, q));
    try f.untouched();
    backend.calls = 0;
    backend.nan_at = 0;
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(T, owner, q)).status);
    // A finite perturbation only on fresh final A*x must prevent convergence.
    f.reset();
    backend.calls = 0;
    backend.distort_at = last - 2;
    var single = q;
    single.max_iterations = 1;
    try std.testing.expectEqual(F.Status.iteration_limit, (try F.Managed(E).runScratch(T, owner, single)).status);
    try f.untouched();
}
test "Contour eigensolver borrowed capacity aliases options overflow and scratch tails" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, true);
    var q = f.request(&backend);
    const w = try F.workspaceLength(T, q);
    const a = try std.testing.allocator.alloc(T, w.elements + 1);
    defer std.testing.allocator.free(a);
    @memset(a, z(T, 66, -22));
    const b = try std.testing.allocator.alloc(f64, w.real_elements + 1);
    defer std.testing.allocator.free(b);
    @memset(b, 55);
    const c = try std.testing.allocator.alloc(f128, w.wide_elements + 1);
    defer std.testing.allocator.free(c);
    @memset(c, 44);
    const d = try std.testing.allocator.alloc(H.WorkspaceScalar(T), w.complex_elements + 1);
    defer std.testing.allocator.free(d);
    @memset(d, .{ .re = 33, .im = -11 });
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, a[0 .. w.elements - 1], b, c, d));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, a, b[0 .. w.real_elements - 1], c, d));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, a, b, c[0 .. w.wide_elements - 1], d));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, a, b, c, d[0 .. w.complex_elements - 1]));
    q.rank_relative_tolerance = 1;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runBorrowed(T, owner, q, a, b, c, d));
    q.rank_relative_tolerance = null;
    q.residuals = q.eigenvalues;
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(T, owner, q, a, b, c, d));
    q.residuals = &f.errors;
    q.initial.shape.column_count = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(T, q));
    q.initial.shape.column_count = 3;
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try f.untouched();
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runBorrowed(T, owner, q, a, b, c, d)).status);
    try std.testing.expectEqualDeep(z(T, 66, -22), a[w.elements]);
    try std.testing.expectEqual(@as(f64, 55), b[w.real_elements]);
    try std.testing.expectEqual(@as(f128, 44), c[w.wide_elements]);
    try std.testing.expectEqualDeep(@as(H.WorkspaceScalar(T), .{ .re = 33, .im = -11 }), d[w.complex_elements]);
}
test "Contour eigensolver allocation rollback closing operation and synchronization rejection" {
    const T = S.ComplexF64;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, true);
    var q = f.request(&backend);
    for (0..4) |budget| {
        var allocations = E.test_support.Allocations{ .budget = budget };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(T, owner, q));
        try f.untouched();
        try std.testing.expectEqual(budget, allocations.stats().accepted);
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    }
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    const Agreement = struct {
        fn agree(_: *const anyopaque, _: ?*const E.context.Operation, point: P.Point, failure: ?anyerror) !void {
            if (failure) |err| return err;
            if (point.stage == .contour_control and point.index == 4) return error.InjectedContourControl;
        }
    };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    q.action.synchronization = .{ .context = &backend, .agree_fn = Agreement.agree };
    try std.testing.expectError(error.InjectedContourControl, F.Managed(E).runScratch(T, owner, q));
    try f.untouched();
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    q.action.synchronization = null;
    backend.close_at = 1;
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(T, owner, q)).status);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}

test "Contour eigensolver twelve dimensional Fourier oracle resolves interior zero multiplicity" {
    const T = S.ComplexF64;
    const n = 12;
    const m = 6;
    const Oracle = struct {
        const spectrum = [_]f64{ -5, -3, -1, 0, 0, 1, 2, 4, 6, 9, 12, 15 };
        fn fourier(row: usize, column: usize) T {
            const theta = 2 * std.math.pi * @as(f64, @floatFromInt(row * column)) / n;
            return z(T, @cos(theta) / @sqrt(@as(f64, n)), @sin(theta) / @sqrt(@as(f64, n)));
        }
        fn apply(_: *const anyopaque, op: *const E.context.Operation, x: []const T, y: []T, _: []T) !void {
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            var spectral: [n]T = undefined;
            for (0..n) |k| {
                var total = z(T, 0, 0);
                for (x, 0..) |value, row| {
                    const u = fourier(row, k);
                    total = S.add(T, total, S.mul(T, z(T, u.re, -u.im), value));
                }
                spectral[k] = S.mul(T, total, z(T, spectrum[k], 0));
            }
            for (y, 0..) |*out, row| {
                out.* = z(T, 0, 0);
                for (spectral, 0..) |value, k| out.* = S.add(T, out.*, S.mul(T, fourier(row, k), value));
            }
        }
    };
    var token: u8 = 0;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var input: [n * m]T = undefined;
    for (0..m) |column| for (0..n) |row| {
        const a: f64 = @floatFromInt((row + 1) * (column + 2));
        input[row + column * n] = z(T, @sin(a * 1.37) + @cos(a * 0.31), @cos(a * 0.73));
    };
    var output: [n * m + 2]T = @splat(z(T, 77, -33));
    var values: [m + 2]f64 = @splat(91);
    var errors: [m + 2]f64 = @splat(92);
    const result = try F.Managed(E).runScratch(T, owner, .{ .action = .{ .context = &token, .rows = n, .columns = n, .apply_fn = Oracle.apply }, .global_dimension = n, .initial = .{ .values = &input, .shape = .{ .row_count = n, .column_count = m, .leading_dimension = n, .layout = .column_major } }, .vectors = .{ .values = &output, .shape = .{ .row_count = n, .column_count = m, .leading_dimension = n, .layout = .column_major } }, .eigenvalues = &values, .residuals = &errors, .contour = .{ .lower = -0.5, .upper = 2.5, .half_nodes = 12, .rule = .gauss_legendre }, .max_iterations = 12, .relative_tolerance = 3e-9, .linear_max_iterations = 192, .linear_relative_tolerance = 3e-13 });
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 4), result.found);
    for ([_]f64{ 0, 0, 1, 2 }, values[0..4], errors[0..4]) |expected, value, residual_norm| {
        try std.testing.expectApproxEqAbs(expected, value, 2e-9);
        try std.testing.expect(residual_norm < 6e-9);
    }
    // Compare whole invariant projector, so the repeated zero eigenspace is
    // independent of arbitrary phases and rotations within its multiplicity.
    for (0..n) |column| for (0..n) |row| {
        var actual = z(T, 0, 0);
        var expected = z(T, 0, 0);
        for (0..4) |k| {
            const a = output[row + k * n];
            const b = output[column + k * n];
            actual = S.add(T, actual, S.mul(T, a, z(T, b.re, -b.im)));
            const c = Oracle.fourier(row, k + 3);
            const d = Oracle.fourier(column, k + 3);
            expected = S.add(T, expected, S.mul(T, c, z(T, d.re, -d.im)));
        }
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 2e-8);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 2e-8);
    };
    for (output[4 * n ..]) |value| try std.testing.expectEqualDeep(z(T, 77, -33), value);
    for (values[4..]) |value| try std.testing.expectEqual(@as(f64, 91), value);
    for (errors[4..]) |value| try std.testing.expectEqual(@as(f64, 92), value);
}

test "Contour eigensolver empty local shard replays global reductions and all backend participation" {
    const T = S.ComplexF64;
    const Record = struct {
        norms: [4096]f64 = undefined,
        inners: [4096]T = undefined,
        projections: [16][18]T = undefined,
        norm_count: usize = 0,
        inner_count: usize = 0,
        projection_count: usize = 0,
        norm_index: usize = 0,
        inner_index: usize = 0,
        projection_index: usize = 0,
        actions: usize = 0,
        replay: bool = false,
        fn norm(raw: *const anyopaque, op: *const E.context.Operation, x: []const T) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            if (self.replay) {
                try std.testing.expectEqual(@as(usize, 0), x.len);
                if (self.norm_index >= self.norm_count) return error.ExtraGlobalNorm;
                defer self.norm_index += 1;
                return self.norms[self.norm_index];
            }
            if (self.norm_count >= self.norms.len) return error.RecordTooSmall;
            const value = try Red.localNorm(T, x);
            self.norms[self.norm_count] = value;
            self.norm_count += 1;
            return value;
        }
        fn inner(raw: *const anyopaque, _: *const E.context.Operation, x: []const T, y: []const T) !T {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            if (self.replay) {
                try std.testing.expectEqual(@as(usize, 0), x.len);
                try std.testing.expectEqual(@as(usize, 0), y.len);
                if (self.inner_index >= self.inner_count) return error.ExtraGlobalInner;
                defer self.inner_index += 1;
                return self.inners[self.inner_index];
            }
            if (self.inner_count >= self.inners.len) return error.RecordTooSmall;
            const value = try Red.localInner(T, x, y);
            self.inners[self.inner_count] = value;
            self.inner_count += 1;
            return value;
        }
        fn project(raw: *const anyopaque, _: *const E.context.Operation, values: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 18), values.len);
            if (self.replay) {
                if (self.projection_index >= self.projection_count) return error.ExtraProjection;
                @memcpy(values, &self.projections[self.projection_index]);
                self.projection_index += 1;
            } else {
                if (self.projection_count >= self.projections.len) return error.RecordTooSmall;
                @memcpy(&self.projections[self.projection_count], values);
                self.projection_count += 1;
            }
        }
        fn emptyApply(raw: *const anyopaque, _: *const E.context.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 0), x.len);
            try std.testing.expectEqual(@as(usize, 0), y.len);
            self.actions += 1;
        }
    };
    var record = Record{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, true);
    var q = f.request(&backend);
    q.method = .residual_inverse;
    q.convergence = .trace;
    q.trace_tolerance = 1e-12;
    q.reduction = Red.Reduction(T){ .context = &record, .inner_fn = Record.inner, .norm_fn = Record.norm };
    q.projection_reduction = Projection.Reduction(T){ .context = &record, .reduce_fn = Record.project };
    const reference = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, reference.status);
    const values = f.values;
    const errors = f.errors;
    f.reset();
    record.replay = true;
    q.action = .{ .context = &record, .rows = 0, .columns = 0, .apply_fn = Record.emptyApply };
    q.initial = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 3, .leading_dimension = 0, .layout = .row_major } };
    q.vectors = .{ .values = &.{}, .shape = q.initial.shape };
    const empty = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqualDeep(reference, empty);
    try std.testing.expectEqualDeep(values, f.values);
    try std.testing.expectEqualDeep(errors, f.errors);
    try std.testing.expectEqual(record.norm_count, record.norm_index);
    try std.testing.expectEqual(record.inner_count, record.inner_index);
    try std.testing.expectEqual(record.projection_count, record.projection_index);
    try std.testing.expectEqual(backend.calls, record.actions);
    // A scalar reduction alone cannot replace the packed global Gram reduction.
    q.projection_reduction = null;
    try std.testing.expectError(error.MissingGlobalReduction, F.Managed(E).runScratch(T, owner, q));
    try std.testing.expectEqual(backend.calls, record.actions);
}

test "Contour eigensolver globally empty geometry and zero iterations use no scratch allocation" {
    const T = S.ComplexF64;
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){};
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, false);
    var q = f.request(&backend);
    q.max_iterations = 0;
    try std.testing.expectEqual(F.Status.iteration_limit, (try F.Managed(E).runScratch(T, owner, q)).status);
    try f.untouched();
    q.global_dimension = 0;
    q.action.rows = 0;
    q.action.columns = 0;
    q.max_iterations = 4;
    q.initial = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 0, .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    q.vectors = .{ .values = &.{}, .shape = q.initial.shape };
    const result = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 0), result.found);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().accepted);
    try f.untouched();
}

test "Contour eigensolver all Ritz candidates place selected pairs first and preserve unused tails" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 4, true);
    var q = f.request(&backend);
    q.output_selection = .all_ritz;
    q.contour.lower = 1.5;
    q.contour.upper = 4.5;
    const result = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 2), result.found);
    try std.testing.expectEqual(@as(usize, 4), result.published);
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 1, 8 }, f.values[0..4]);
    const row_for_column = [_]usize{ 1, 2, 0, 3 };
    for (row_for_column, 0..) |active_row, column| for (0..4) |row| {
        const value = f.output[row + column * 6];
        try std.testing.expectApproxEqAbs(@as(f64, if (row == active_row) 1 else 0), value.re * value.re + value.im * value.im, 1e-12);
    };
    try std.testing.expectEqual(@as(f64, 91), f.values[4]);
    try std.testing.expectEqual(@as(f64, 92), f.errors[4]);
    for (0..4) |column| for (4..6) |row| {
        try std.testing.expectEqualDeep(z(T, 77, -33), f.output[row + column * 6]);
    };
}

test "Contour eigensolver explicit finite stop publication retains error rollback and capacity diagnostics" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = true };
    var f: Fixture(T) = undefined;
    f.init(.column_major, 4, true);
    var q = f.request(&backend);
    q.output_selection = .all_ritz;
    q.publish_finite_stops = true;
    q.max_iterations = 1;
    q.contour.lower = 9;
    q.contour.upper = 10;
    const absent = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.iteration_limit, absent.status);
    try std.testing.expectEqual(@as(usize, 0), absent.found);
    try std.testing.expectEqual(@as(usize, 4), absent.published);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 4, 8 }, f.values[0..4]);
    f.init(.row_major, 2, true);
    q = f.request(&backend);
    q.output_selection = .all_ritz;
    q.publish_finite_stops = true;
    q.contour.upper = 2.5;
    backend.calls = 0;
    const full = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.capacity_limit, full.status);
    try std.testing.expectEqual(@as(usize, 2), full.found);
    try std.testing.expectEqual(@as(usize, 2), full.published);
    const last = backend.calls;
    f.reset();
    backend.calls = 0;
    backend.fail_at = last;
    try std.testing.expectError(error.InjectedContourAction, F.Managed(E).runScratch(T, owner, q));
    try f.untouched();
    backend.calls = 0;
    backend.fail_at = 0;
    backend.nan_at = last;
    try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, q));
    try f.untouched();
}

test "Contour eigensolver inexact residual scaled solves converge through finite inner limits" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(T){};
    var f: Fixture(T) = undefined;
    f.init(.column_major, 3, false);
    var q = f.request(&backend);
    q.linear_accuracy = .residual_scaled;
    q.linear_relative_tolerance = 0.1;
    q.linear_max_iterations = 2;
    q.allow_inexact_iteration_limit = true;
    q.max_iterations = 32;
    const result = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expect(result.inexact_systems > 0);
    try std.testing.expect(result.iterations > 1);
    try std.testing.expectEqual(backend.calls, result.action_count);
    try physical(T, &f, 2);
}

test "Contour eigensolver residual inverse two precisions converge with one step inner budgets" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        var backend = Backend(T){};
        var f: Fixture(T) = undefined;
        f.init(.column_major, 3, false);
        var q = f.request(&backend);
        q.method = .residual_inverse;
        q.linear_accuracy = .fixed;
        q.linear_relative_tolerance = 0.1;
        q.linear_max_iterations = 1;
        q.allow_inexact_iteration_limit = true;
        q.max_iterations = 32;
        const result = try F.Managed(E).runScratch(T, owner, q);
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expect(result.inexact_systems > 0);
        try std.testing.expect(result.iterations > 1);
        try std.testing.expectEqual(backend.calls, result.action_count);
        try physical(T, &f, 2);
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
