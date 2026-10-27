// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("chebyshev.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const P = @import("synchronization.zig");
const E = @import("zynum-execution");
const C = E.context;
fn context(storage: *FixtureReferenceContext) *C.Context {
    storage.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    return storage.borrowedContext();
}
fn close(owner: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(owner));
}
fn value(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn realPart(comptime T: type, x: T) f64 {
    return if (comptime S.isComplex(T)) x.re else x;
}
fn imaginaryPart(comptime T: type, x: T) f64 {
    return if (comptime S.isComplex(T)) x.im else 0;
}
fn near(comptime T: type, expected: T, actual: T) !void {
    const tolerance: f64 = if (S.Real(T) == f32) 8e-5 else 8e-13;
    try std.testing.expectApproxEqAbs(realPart(T, expected), realPart(T, actual), tolerance * @max(1, @abs(realPart(T, expected))));
    try std.testing.expectApproxEqAbs(imaginaryPart(T, expected), imaginaryPart(T, actual), tolerance * @max(1, @abs(imaginaryPart(T, expected))));
}
fn index(shape: D.PanelShape, row: usize, column: usize) usize {
    return if (shape.layout == .column_major) row + column * shape.leading_dimension else column + row * shape.leading_dimension;
}
fn Backend(comptime T: type) type {
    return struct {
        diagonal: []const S.Real(T),
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        sync: ?P.Synchronization = null,
        fn action(self: *@This()) O.Operator(T) {
            return .{ .context = self, .rows = self.diagonal.len, .columns = self.diagonal.len, .apply_fn = apply, .synchronization = self.sync };
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, input: []const T, output: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
            if (self.close_at == self.calls) {
                _ = operation.borrowedContext().requestClose();
                try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(operation.borrowedContext()));
            }
            for (input, output, self.diagonal) |x, *y, lambda| y.* = S.mul(T, S.realScalar(T, lambda), x);
            if (self.calls == self.fail_at) return error.InjectedSeriesSetAction;
            if (self.calls == self.nan_at and output.len != 0) output[0] = value(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
// Independent cosine evaluation, without the implementation's recurrence.
fn polynomial(comptime T: type, coefficients: D.ConstPanel(T), column: usize, count: usize, lambda: f64) T {
    var real: f64 = 0;
    var imaginary: f64 = 0;
    for (0..count) |k| {
        const basis = std.math.cos(@as(f64, @floatFromInt(k)) * std.math.acos(lambda));
        const c = coefficients.column(column).at(k);
        real += basis * realPart(T, c);
        imaginary += basis * imaginaryPart(T, c);
    }
    return value(T, @floatCast(real), @floatCast(imaginary));
}
test "series set four scalars ragged coefficients and both panel layouts match independent cosine oracle" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |cl| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |yl| {
        const R = S.Real(T);
        const counts = [_]usize{ 1, 2, 6, 33 };
        const cs: D.PanelShape = .{ .row_count = 35, .column_count = 4, .leading_dimension = if (cl == .column_major) 37 else 6, .layout = cl };
        var coefficients: [215]T = @splat(value(T, std.math.nan(R), 7));
        for (counts, 0..) |count, j| for (0..count) |k| {
            const q: R = @floatFromInt(k + 1);
            coefficients[index(cs, k, j)] = value(T, 1 / q, if (k % 2 == 0) 0.03 else -0.02);
        };
        const coeff: D.ConstPanel(T) = .{ .values = &coefficients, .shape = cs };
        const before_coeff = coefficients;
        const ys: D.PanelShape = .{ .row_count = 4, .column_count = 4, .leading_dimension = 6, .layout = yl };
        var output: [27]T = @splat(value(T, 91, -11));
        const before_out = output;
        const diagonal = [_]R{ -0.85, -0.2, 0.35, 0.9 };
        const input = [_]T{ value(T, 1, 0.25), value(T, 2, -0.5), value(T, -1, 0.75), value(T, 0.5, -0.125) };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Backend(T){ .diagonal = &diagonal };
        try F.Managed(E).runSeriesSetScratch(T, owner, .{ .action = backend.action(), .input = &input, .coefficients = coeff, .coefficient_counts = &counts, .output = .{ .values = &output, .shape = ys } });
        try std.testing.expectEqual(@as(usize, 32), backend.calls);
        for (counts, 0..) |count, j| for (diagonal, input, 0..) |lambda, x, i| try near(T, S.mul(T, polynomial(T, coeff, j, count, lambda), x), output[index(ys, i, j)]);
        for (0..output.len) |p| {
            var logical = false;
            for (0..4) |j| for (0..4) |i| {
                if (index(ys, i, j) == p) logical = true;
            };
            if (!logical) try std.testing.expectEqualDeep(before_out[p], output[p]);
        }
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&before_coeff), std.mem.sliceAsBytes(&coefficients));
    };
}
test "series set packs aliased input before committing every output column" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        const diag = [_]S.Real(T){ -0.5, 0, 0.75 };
        var backend = Backend(T){ .diagonal = &diag };
        const counts = [_]usize{ 3, 6 };
        const coefficients = [_]T{ value(T, 1, 0.1), value(T, 0.2, 0), value(T, -0.3, 0.1), value(T, 99, 0), value(T, 99, 0), value(T, 99, 0), value(T, 0.5, -0.2), value(T, -0.1, 0), value(T, 0.05, 0.1), value(T, 0.03, 0), value(T, -0.02, 0.01), value(T, 0.01, 0) };
        const coeff: D.ConstPanel(T) = .{ .values = &coefficients, .shape = .{ .row_count = 6, .column_count = 2, .leading_dimension = 6, .layout = .column_major } };
        var output = [_]T{ value(T, 1, 0.2), value(T, 2, 0.3), value(T, 3, 0.4), value(T, 94, 0), value(T, 95, 0), value(T, 96, 0), value(T, 97, 0), value(T, 98, 0), value(T, 99, 0), value(T, 100, 0), value(T, 101, 0), value(T, 102, 0) };
        const before = output;
        const shape: D.PanelShape = .{ .row_count = 3, .column_count = 2, .leading_dimension = 4, .layout = .row_major };
        try F.Managed(E).runSeriesSetScratch(T, owner, .{ .action = backend.action(), .input = output[0..3], .coefficients = coeff, .coefficient_counts = &counts, .output = .{ .values = &output, .shape = shape } });
        for (counts, 0..) |count, j| for (diag, 0..) |lambda, i| try near(T, S.mul(T, polynomial(T, coeff, j, count, lambda), before[i]), output[index(shape, i, j)]);
        for ([_]usize{ 2, 3, 6, 7, 10, 11 }) |p| try std.testing.expectEqualDeep(before[p], output[p]);
    }
}
fn smallRequest(backend: *Backend(f64), input: []const f64, coefficients: []const f64, counts: []const usize, output: []f64) F.SeriesSetRequest(f64) {
    return .{ .action = backend.action(), .input = input, .coefficients = .{ .values = coefficients, .shape = .{ .row_count = 4, .column_count = 2, .leading_dimension = 4, .layout = .column_major } }, .coefficient_counts = counts, .output = .{ .values = output, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 2, .layout = .column_major } } };
}
test "series set validates counts finite prefixes geometry scratch aliases and overflow before actions" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{ 0.25, 0.5 } };
    const input = [_]f64{ 1, 2 };
    var coefficients = [_]f64{ 1, 0.2, 0.3, 0.4, 1, 0.1, 0.2, 0.3 };
    var counts = [_]usize{ 3, 4 };
    var output: [20]f64 = @splat(91);
    const before = output;
    var work: [15]f64 = undefined;
    var q = smallRequest(&backend, &input, &coefficients, &counts, output[0..4]);
    counts[1] = 0;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, &work));
    counts[1] = 5;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, &work));
    counts[1] = 4;
    coefficients[7] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, &work));
    coefficients[7] = 0.3;
    q.output.shape.column_count = 1;
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, &work));
    q.output.shape.column_count = 2;
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, work[0..9]));
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runSeriesSetBorrowed(f64, owner, q, output[0..10]));
    var huge = backend.action();
    huge.rows = std.math.maxInt(usize);
    huge.columns = huge.rows;
    try std.testing.expectError(error.SizeOverflow, F.seriesSetWorkspaceLength(f64, huge, 2));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqualDeep(before, output);
}
test "series set callback failures and nonfinite arithmetic preserve all output columns then permit retry" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{ 0.25, 0.5 } };
    const input = [_]f64{ 1, 2 };
    const coefficients = [_]f64{ 1, 0.2, 0.3, 0.4, 1, 0.1, 0.2, 0.3 };
    const counts = [_]usize{ 1, 4 };
    var output = [_]f64{ 91, 92, 93, 94 };
    const before = output;
    const q = smallRequest(&backend, &input, &coefficients, &counts, &output);
    backend.fail_at = 2;
    try std.testing.expectError(error.InjectedSeriesSetAction, F.Managed(E).runSeriesSetScratch(f64, owner, q));
    try std.testing.expectEqualDeep(before, output);
    backend.calls = 0;
    backend.fail_at = 0;
    backend.nan_at = 2;
    try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runSeriesSetScratch(f64, owner, q));
    try std.testing.expectEqualDeep(before, output);
    backend.calls = 0;
    backend.nan_at = 0;
    try F.Managed(E).runSeriesSetScratch(f64, owner, q);
    try std.testing.expectEqual(@as(usize, 3), backend.calls);
    try std.testing.expectEqualSlices(f64, &input, output[0..2]);
    var overflow = Backend(f64){ .diagonal = &.{1} };
    const x = [_]f64{std.math.floatMax(f64) * 0.75};
    const c = [_]f64{ 1, 1 };
    const length = [_]usize{2};
    var y = [_]f64{91};
    try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runSeriesSetScratch(f64, owner, .{ .action = overflow.action(), .input = &x, .coefficients = .{ .values = &c, .shape = .{ .row_count = 2, .column_count = 1, .leading_dimension = 2, .layout = .column_major } }, .coefficient_counts = &length, .output = .{ .values = &y, .shape = .{ .row_count = 1, .column_count = 1, .leading_dimension = 1, .layout = .column_major } } }));
    try std.testing.expectEqual(@as(f64, 91), y[0]);
}
const Observer = struct {
    failure_stage: ?P.Stage = null,
    failure_index: u64 = 0,
    last_arithmetic: u64 = 0,
    final_only: bool = false,
    fn descriptor(self: *@This()) P.Synchronization {
        return .{ .context = self, .agree_fn = agree };
    }
    fn agree(raw: *const anyopaque, _: ?*const C.Operation, point: P.Point, local: ?anyerror) !void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        if (local) |err| return err;
        if (point.stage == .local_arithmetic) self.last_arithmetic = point.index;
        if (self.failure_stage == point.stage and self.failure_index == point.index and (!self.final_only or self.last_arithmetic == 3)) return error.RemoteSeriesSetFailure;
    }
};
test "series set remote coefficient arithmetic and final commit failures preserve every output" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var observer = Observer{};
    var backend = Backend(f64){ .diagonal = &.{ 0.25, 0.5 }, .sync = observer.descriptor() };
    const input = [_]f64{ 1, 2 };
    const coefficients = [_]f64{ 1, 0.2, 0.3, 0.4, 1, 0.1, 0.2, 0.3 };
    const counts = [_]usize{ 1, 4 };
    var output = [_]f64{ 91, 92, 93, 94 };
    const before = output;
    for ([_]Observer{ .{ .failure_stage = .coefficients_count, .failure_index = 1 }, .{ .failure_stage = .local_arithmetic, .failure_index = 2 }, .{ .failure_stage = .commit, .final_only = true } }) |fault| {
        observer = fault;
        backend.calls = 0;
        const q = smallRequest(&backend, &input, &coefficients, &counts, &output);
        try std.testing.expectError(error.RemoteSeriesSetFailure, F.Managed(E).runSeriesSetScratch(f64, owner, q));
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}
test "series set allocation failure and closing owner release the same operation and scratch" {
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{ 0.25, 0.5 } };
    const input = [_]f64{ 1, 2 };
    const coefficients = [_]f64{ 1, 0.2, 0.3, 0.4, 1, 0.1, 0.2, 0.3 };
    const counts = [_]usize{ 1, 4 };
    var output = [_]f64{ 91, 92, 93, 94 };
    const q = smallRequest(&backend, &input, &coefficients, &counts, &output);
    try std.testing.expectError(error.OutOfMemory, F.Managed(E).runSeriesSetScratch(f64, owner, q));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    allocations.budget = 1;
    backend.close_at = 1;
    try F.Managed(E).runSeriesSetScratch(f64, owner, q);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().freed);
    try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(owner));
    try std.testing.expectError(error.Closing, F.Managed(E).runSeriesSetScratch(f64, owner, q));
}
test "series set empty output and empty participating shard retain callback and padding contracts" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{ 0.25, 0.5 } };
    const input = [_]f64{ 1, 2 };
    var output = [_]f64{91};
    try F.Managed(E).runSeriesSetScratch(f64, owner, .{ .action = backend.action(), .input = &input, .coefficients = .{ .values = &.{}, .shape = .{ .row_count = std.math.maxInt(usize), .column_count = 0, .leading_dimension = 0, .layout = .column_major } }, .coefficient_counts = &.{}, .output = .{ .values = &output, .shape = .{ .row_count = 2, .column_count = 0, .leading_dimension = 0, .layout = .column_major } } });
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    var observer = Observer{};
    var empty = Backend(f64){ .diagonal = &.{}, .sync = observer.descriptor() };
    const coefficients = [_]f64{ 1, 2, 99, 99, 99, 1, 2, 3, 4, 5 };
    try F.Managed(E).runSeriesSetScratch(f64, owner, .{ .action = empty.action(), .input = &.{}, .coefficients = .{ .values = &coefficients, .shape = .{ .row_count = 5, .column_count = 2, .leading_dimension = 5, .layout = .column_major } }, .coefficient_counts = &.{ 2, 5 }, .output = .{ .values = &output, .shape = .{ .row_count = 0, .column_count = 2, .leading_dimension = 0, .layout = .column_major } } });
    try std.testing.expectEqual(@as(usize, 4), empty.calls);
    try std.testing.expectEqual(@as(f64, 91), output[0]);
}
test "series set widened accumulation retains finite values beyond native intermediate range" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Backend(T){ .diagonal = &.{1} };
        const x = [_]T{value(T, std.math.floatMax(S.Real(T)) * 0.75, -std.math.floatMax(S.Real(T)) * 0.5)};
        const coefficients = [_]T{ value(T, -1, 0), value(T, 2, 0) };
        var output = [_]T{value(T, 91, 0)};
        try F.Managed(E).runSeriesSetScratch(T, owner, .{ .action = backend.action(), .input = &x, .coefficients = .{ .values = &coefficients, .shape = .{ .row_count = 2, .column_count = 1, .leading_dimension = 2, .layout = .column_major } }, .coefficient_counts = &.{2}, .output = .{ .values = &output, .shape = .{ .row_count = 1, .column_count = 1, .leading_dimension = 1, .layout = .column_major } } });
        try std.testing.expectEqualDeep(x, output);
    }
}
test "series set non diagonal real and complex Hermitian actions match spectral projectors" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Foreign = struct {
            calls: usize = 0,
            fn apply(raw: *const anyopaque, _: *const C.Operation, x: []const T, y: []T, _: []T) !void {
                const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                self.calls += 1;
                const off = value(T, if (S.isComplex(T)) 0 else 0.6, if (S.isComplex(T)) 0.6 else 0);
                y[0] = S.add(T, S.mul(T, value(T, 0.1, 0), x[0]), S.mul(T, off, x[1]));
                y[1] = S.add(T, S.mul(T, S.conj(T, off), x[0]), S.mul(T, value(T, 0.1, 0), x[1]));
            }
        };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Foreign{};
        const action: O.Operator(T) = .{ .context = &backend, .rows = 2, .columns = 2, .apply_fn = Foreign.apply };
        const input = [_]T{ value(T, 0.5, -0.25), value(T, -0.75, 0.4) };
        const counts = [_]usize{ 1, 3, 9 };
        var coefficients: [27]T = undefined;
        for (&coefficients, 0..) |*x, k| x.* = value(T, 1 / @as(S.Real(T), @floatFromInt(k + 1)), if (k % 2 == 0) 0.03 else -0.02);
        const coeff: D.ConstPanel(T) = .{ .values = &coefficients, .shape = .{ .row_count = 9, .column_count = 3, .leading_dimension = 9, .layout = .column_major } };
        var output: [6]T = undefined;
        try F.Managed(E).runSeriesSetScratch(T, owner, .{ .action = action, .input = &input, .coefficients = coeff, .coefficient_counts = &counts, .output = .{ .values = &output, .shape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 2, .layout = .column_major } } });
        try std.testing.expectEqual(@as(usize, 8), backend.calls);
        const k = value(T, if (S.isComplex(T)) 0 else 1, if (S.isComplex(T)) 1 else 0);
        for (counts, 0..) |count, j| {
            const low = polynomial(T, coeff, j, count, -0.5);
            const high = polynomial(T, coeff, j, count, 0.7);
            const mean = S.mul(T, value(T, 0.5, 0), S.add(T, low, high));
            const difference = S.mul(T, value(T, 0.5, 0), S.sub(T, high, low));
            try near(T, S.add(T, S.mul(T, mean, input[0]), S.mul(T, S.mul(T, difference, k), input[1])), output[j * 2]);
            try near(T, S.add(T, S.mul(T, mean, input[1]), S.mul(T, S.mul(T, difference, S.conj(T, k)), input[0])), output[j * 2 + 1]);
        }
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
