// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const lapack = @import("zynum-lapack");
const owner = @import("zynum-core");
const E = @import("zynum-execution");
const core = owner.internal.numerics;
const C32 = lapack.ComplexF32;
const C64 = lapack.ComplexF64;
fn Real(comptime T: type) type {
    return if (T == C32) f32 else if (T == C64) f64 else T;
}
fn value(comptime T: type, r: f64, i: f64) T {
    return if (T == C32 or T == C64) .{ .re = @floatCast(r), .im = @floatCast(i) } else @floatCast(r);
}
fn workValue(comptime T: type, r: f64) lapack.SvdWorkspaceScalar(T) {
    return if (T == C32 or T == C64) .{ .re = r, .im = -r } else r;
}
fn matrix(comptime T: type, a: []T, m: i32, n: i32, ld: i32) lapack.Matrix(T) {
    return .{ .values = a, .row_count = m, .column_count = n, .leading_dimension = ld };
}
fn equal(comptime T: type, expected: T, actual: T) !void {
    if (T == C32 or T == C64) {
        try std.testing.expectEqual(expected.re, actual.re);
        try std.testing.expectEqual(expected.im, actual.im);
    } else try std.testing.expectEqual(expected, actual);
}
test "complex SVD facade forwards existing scalar identity and widened workspace without changing real plans" {
    inline for (.{ C32, C64 }) |C| {
        try std.testing.expect(lapack.SvdWorkspaceScalar(C) == core.ComplexSvd.WorkspaceScalar(C));
        var a = [_]C{ value(C, 3, 0), value(C, 0, 4), value(C, 73, -74) };
        var s = [_]Real(C){ 0, 75 };
        var v = [_]C{value(C, 77, 78)};
        var work: [5]lapack.SvdWorkspaceScalar(C) = @splat(workValue(C, 79));
        try std.testing.expectEqual(@as(usize, 4), try lapack.svdWorkspaceLength(C, 2, 1));
        const result = try lapack.svdDecompose(C, matrix(C, &a, 2, 1, 3), &s, matrix(C, &v, 1, 1, 1), &work, .{});
        try std.testing.expectEqual(lapack.SvdStatus.converged, result.status);
        try std.testing.expectEqual(@as(Real(C), 5), s[0]);
        try std.testing.expectEqual(@as(Real(C), 75), s[1]);
        try equal(C, value(C, 1, 0), v[0]);
        const tol: Real(C) = if (C == C32) 1e-6 else 2e-14;
        try std.testing.expectApproxEqAbs(@as(Real(C), 0.6), a[0].re, tol);
        try std.testing.expectApproxEqAbs(@as(Real(C), 0.8), a[1].im, tol);
        try equal(C, value(C, 73, -74), a[2]);
        try std.testing.expectEqual(@as(@TypeOf(work[4].re), 79), work[4].re);
    }
    try std.testing.expect(lapack.SvdWorkspaceScalar(f32) == f64);
    try std.testing.expect(lapack.SvdWorkspaceScalar(f64) == f128);
}
test "rectangular SVD facade handles all four scalars and keeps trailing wide input columns untouched" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        var a = [_]T{ value(T, 3, 0), value(T, 81, 82), if (T == C32 or T == C64) value(T, 0, 4) else value(T, 4, 0), value(T, 83, 84) };
        const original = a;
        var s = [_]Real(T){ 0, 85 };
        var v: [3]T = @splat(value(T, 87, 88));
        var work: [5]lapack.SvdWorkspaceScalar(T) = @splat(workValue(T, 89));
        var trans: [4]T = @splat(value(T, 91, 92));
        try std.testing.expectEqual(@as(usize, 4), try lapack.svdRectangularWorkspaceLength(T, 1, 2));
        try std.testing.expectEqual(@as(usize, 3), try lapack.svdRectangularScratchLength(T, 1, 2));
        const result = try lapack.svdDecomposeRectangular(T, matrix(T, &a, 1, 2, 2), &s, matrix(T, &v, 2, 1, 3), &work, &trans, .{});
        try std.testing.expectEqual(lapack.SvdStatus.converged, result.status);
        try std.testing.expectEqual(@as(Real(T), 5), s[0]);
        try std.testing.expectEqual(@as(Real(T), 85), s[1]);
        try equal(T, value(T, 1, 0), a[0]);
        for (1..4) |i| try equal(T, original[i], a[i]);
        try equal(T, value(T, 87, 88), v[2]);
        try equal(T, value(T, 91, 92), trans[3]);
        const tol: Real(T) = if (T == f32 or T == C32) 1e-6 else 2e-14;
        if (T == C32 or T == C64) {
            try std.testing.expectApproxEqAbs(@as(Real(T), 0.6), v[0].re, tol);
            try std.testing.expectApproxEqAbs(@as(Real(T), -0.8), v[1].im, tol);
            try std.testing.expectEqual(@as(@TypeOf(work[4].re), 89), work[4].re);
        } else {
            try std.testing.expectApproxEqAbs(@as(T, 0.6), v[0], tol);
            try std.testing.expectApproxEqAbs(@as(T, 0.8), v[1], tol);
            try std.testing.expectEqual(@as(@TypeOf(work[4]), 89), work[4]);
        }
    }
}
const Observe = struct {
    var entries: usize = 0;
    var finishes: usize = 0;
    var watch: union(enum) { none, f32: *const f32, f64: *const f64 } = .none;
    fn callback(ctx: *E.context.Context, event: E.context.TestEvent) void {
        if (ctx != E.getContext() or &ctx.executor != E.get()) @panic("SVD facade changed its configured owner");
        if (ctx.snapshot().active_operations != 1 or ctx.snapshot().resource_pins != 0) @panic("SVD facade did not borrow exactly one allocation-free operation");
        switch (event) {
            .admitted => entries += 1,
            .finishing => {
                switch (watch) {
                    .none => {},
                    .f32 => |s| {
                        if (s.* != 5) @panic("SVD released operation before committing output");
                    },
                    .f64 => |s| {
                        if (s.* != 5) @panic("SVD released operation before committing output");
                    },
                }
                finishes += 1;
            },
        }
    }
};
test "complex SVD facade releases one existing admission on errors empty calls and successful commit" {
    Observe.entries = 0;
    Observe.finishes = 0;
    Observe.watch = .none;
    E.context.Context.testing.configure(E.getContext(), 1, Observe.callback);
    defer E.context.Context.testing.configure(E.getContext(), std.math.maxInt(usize), null);
    inline for (.{ C32, C64 }) |C| {
        var empty: [0]C = .{};
        try std.testing.expectError(error.InvalidLength, lapack.svdDecompose(C, matrix(C, &empty, -1, 0, 0), &.{}, matrix(C, &empty, 0, 0, 0), &.{}, .{}));
        try std.testing.expectEqual(lapack.SvdStatus.converged, (try lapack.svdDecomposeRectangular(C, matrix(C, &empty, 0, 0, 0), &.{}, matrix(C, &empty, 0, 0, 0), &.{}, &.{}, .{})).status);
        var a = [_]C{value(C, 3, 4)};
        var s: [1]Real(C) = @splat(0);
        var v: [1]C = undefined;
        var work: [3]lapack.SvdWorkspaceScalar(C) = undefined;
        Observe.watch = if (C == C32) .{ .f32 = &s[0] } else .{ .f64 = &s[0] };
        try std.testing.expectEqual(lapack.SvdStatus.converged, (try lapack.svdDecompose(C, matrix(C, &a, 1, 1, 1), &s, matrix(C, &v, 1, 1, 1), &work, .{})).status);
        Observe.watch = .none;
    }
    try std.testing.expectEqual(@as(usize, 6), Observe.entries);
    try std.testing.expectEqual(Observe.entries, Observe.finishes);
    try std.testing.expectEqual(@as(usize, 0), E.getContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), E.getContext().snapshot().resource_pins);
}
test "complex SVD reconstruction and rank solve borrow a closing caller operation without readmission" {
    const Probe = struct {
        context: E.context.Context,
        entries: usize = 0,
        finishes: usize = 0,
        completed: bool = false,
        fn callback(ctx: *E.context.Context, event: E.context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("context", ctx);
            if (ctx.snapshot().active_operations != 1 or ctx.snapshot().resource_pins != 0) @panic("pure complex Core changed borrowed lifetime");
            switch (event) {
                .admitted => self.entries += 1,
                .finishing => {
                    if (!self.completed) @panic("caller ended before pure complex numerical work");
                    self.finishes += 1;
                },
            }
        }
    };
    inline for (.{ C32, C64 }) |C| {
        var probe: Probe = .{ .context = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable) };
        E.context.Context.testing.configure(&probe.context, 1, Probe.callback);
        var op = try probe.context.begin();
        var ended = false;
        defer {
            if (!ended) {
                probe.completed = true;
                op.end();
            }
            _ = probe.context.requestClose();
            _ = probe.context.tryProgressClose();
        }
        try std.testing.expectEqual(E.context.RequestResult.pending, probe.context.requestClose());
        try std.testing.expectEqual(E.context.CloseResult.busy, probe.context.tryProgressClose());
        const original = [_]C{value(C, 3, 4)};
        const rhs = [_]C{value(C, 6, 8)};
        var a = original;
        var s: [1]Real(C) = undefined;
        var v: [1]C = undefined;
        var work: [16]core.ComplexSvd.WorkspaceScalar(C) = undefined;
        const result = try core.ComplexSvd.decompose(C, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &s, .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &work, .{});
        try std.testing.expectEqual(.converged, result.status);
        var x: [1]C = undefined;
        const reconstruction = try core.ComplexMinimumNorm.solve(C, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &s, .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, -1, &work);
        try std.testing.expectEqual(.success, reconstruction.status);
        const tol: Real(C) = if (C == C32) 1e-6 else 2e-14;
        try std.testing.expectApproxEqAbs(@as(Real(C), 2), x[0].re, tol);
        try std.testing.expectApproxEqAbs(@as(Real(C), 0), x[0].im, tol);
        var scratch: [3]C = undefined;
        const rank = try core.ComplexRankLeastSquares.solve(C, .{ .values = &original, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &s, -1, &scratch, &work, .{});
        try std.testing.expectEqual(.converged, rank.status);
        try std.testing.expectEqual(@as(usize, 1), rank.rank);
        try std.testing.expectEqual(@as(usize, 1), probe.entries);
        try std.testing.expectEqual(@as(usize, 0), probe.finishes);
        try std.testing.expectEqual(E.context.CloseResult.busy, probe.context.tryProgressClose());
        probe.completed = true;
        op.end();
        ended = true;
        try std.testing.expectEqual(@as(usize, 1), probe.finishes);
        try std.testing.expectEqual(E.context.CloseResult.closed, probe.context.tryProgressClose());
        try std.testing.expectError(error.Closing, probe.context.begin());
    }
    try std.testing.expectEqual(@as(usize, 0), E.getContext().snapshot().active_operations);
}
