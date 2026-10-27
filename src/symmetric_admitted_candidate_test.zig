// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const E = @import("zynum-execution");
const C = E.context;
const Core = @import("zynum-core");
const S = struct {
    pub const Matrix = Core.lapack.Matrix;
    pub const ConstMatrix = Core.lapack.ConstMatrix;
    pub const Triangle = Core.lapack.Triangle;
    pub const Status = Core.lapack.SymmetricEigenStatus;
    pub const decomposeAdmitted = Core.internal.lapack_checked.symmetricEigenDecomposeAdmitted;
};
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

fn matrix(comptime T: type, values: []T, n: usize, ld: usize) S.Matrix(T) {
    return .{ .values = values, .row_count = @intCast(n), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}
fn constant(comptime T: type, values: []const T, n: usize, ld: usize) S.ConstMatrix(T) {
    return .{ .values = values, .row_count = @intCast(n), .column_count = @intCast(n), .leading_dimension = @intCast(ld), .operation = .normal };
}
fn unchanged(comptime T: type, before: []const T, after: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(after));
}
fn finish(owner: *C.Context) void {
    _ = owner.requestClose();
    equal(C.CloseResult.closed, fixtureProgressClose(owner)) catch unreachable;
}
fn prohibitDefaultAdmission() void {
    C.Context.testing.configure(E.getContext(), 0, null);
}
fn restoreDefaultAdmission() void {
    C.Context.testing.configure(E.getContext(), std.math.maxInt(usize), null);
}

pub fn checkClosingCaller() !void {
    prohibitDefaultAdmission();
    defer restoreDefaultAdmission();
    inline for (.{ f32, f64 }) |T| {
        const tolerance: f64 = if (T == f32) 3e-5 else 1e-12;
        for ([_]S.Triangle{ .lower, .upper }) |triangle| {
            var spy: E.test_support.Allocations = .{ .budget = 0 };
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(spy.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            defer finish(owner);
            C.Context.testing.configure(owner, 1, null);
            var operation = try owner.begin();
            defer operation.end();
            try equal(C.RequestResult.pending, owner.requestClose());
            try std.testing.expectError(error.Closing, owner.begin());
            const original = [_]T{ 2, 1, 0, 1, 2, 0, 0, 0, 7 };
            var a: [18]T = @splat(std.math.nan(T));
            for (0..3) |j| for (0..3) |i| {
                if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) a[i + j * 5] = original[i + j * 3];
            };
            const before_a = a;
            var work: [18]T = @splat(-918);
            var vectors: [18]T = @splat(-917);
            var values = [_]T{ -916, -916, -916, -916, -916 };
            const result = try S.decomposeAdmitted(&operation, T, constant(T, &a, 3, 5), triangle, matrix(T, &work, 3, 5), &values, matrix(T, &vectors, 3, 5), .{});
            try equal(S.Status.converged, result.status);
            for ([_]f64{ 1, 3, 7 }, 0..) |expected, j| {
                try std.testing.expectApproxEqAbs(expected, @as(f64, values[j]), tolerance);
                for (0..3) |i| {
                    var av: f64 = 0;
                    for (0..3) |k| av += @as(f64, original[i + k * 3]) * @as(f64, vectors[k + j * 5]);
                    try std.testing.expectApproxEqAbs(av, @as(f64, vectors[i + j * 5]) * @as(f64, values[j]), tolerance);
                }
                for (0..3) |k| {
                    var dot: f64 = 0;
                    for (0..3) |i| dot += @as(f64, vectors[i + j * 5]) * @as(f64, vectors[i + k * 5]);
                    try std.testing.expectApproxEqAbs(@as(f64, if (j == k) 1 else 0), dot, tolerance);
                }
            }
            try unchanged(T, &before_a, &a);
            for (0..18) |i| if (i / 5 >= 3 or i % 5 >= 3) {
                try equal(@as(T, -918), work[i]);
                try equal(@as(T, -917), vectors[i]);
            };
            try std.testing.expectEqualSlices(T, &.{ -916, -916 }, values[3..]);
            try equal(@as(usize, 1), owner.snapshot().active_operations);
            try equal(@as(usize, 0), owner.snapshot().resource_pins);
            try equal(@as(usize, 0), spy.stats().attempts);
            try equal(@as(usize, 0), E.getContext().snapshot().active_operations);
        }
    }
}

test "symmetric admitted candidate closing caller and disabled default preserve oracle and padding" {
    try checkClosingCaller();
}

pub fn checkArgumentPreservation() !void {
    prohibitDefaultAdmission();
    defer restoreDefaultAdmission();
    inline for (.{ f32, f64 }) |T| {
        var spy: E.test_support.Allocations = .{ .budget = 0 };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(spy.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer finish(owner);
        var operation = try owner.begin();
        defer operation.end();
        _ = owner.requestClose();
        const a = [_]T{ 2, 1, 1, 2 };
        var work = [_]T{ -7, -7, -7, -7 };
        var vectors = [_]T{ -8, -8, -8, -8 };
        var values = [_]T{ -9, -9 };
        const old_work = work;
        const old_vectors = vectors;
        const old_values = values;
        try std.testing.expectError(error.InvalidLeadingDimension, S.decomposeAdmitted(&operation, T, constant(T, &a, 2, 1), .upper, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{}));
        try std.testing.expectError(error.BufferTooSmall, S.decomposeAdmitted(&operation, T, constant(T, &a, 2, 2), .upper, matrix(T, &work, 2, 2), values[0..1], matrix(T, &vectors, 2, 2), .{}));
        try std.testing.expectError(error.AliasingNotAllowed, S.decomposeAdmitted(&operation, T, constant(T, &a, 2, 2), .upper, matrix(T, &work, 2, 2), &values, matrix(T, &work, 2, 2), .{}));
        const bad = [_]T{ 2, 1, std.math.nan(T), 2 };
        try std.testing.expectError(error.NonFiniteInput, S.decomposeAdmitted(&operation, T, constant(T, &bad, 2, 2), .upper, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{}));
        try unchanged(T, &old_work, &work);
        try unchanged(T, &old_vectors, &vectors);
        try unchanged(T, &old_values, &values);
        try equal(@as(usize, 1), owner.snapshot().active_operations);
        try equal(@as(usize, 0), owner.snapshot().resource_pins);
        try equal(@as(usize, 0), spy.stats().attempts);
    }
}

test "symmetric admitted candidate argument errors preserve all outputs and borrowed lifetime" {
    try checkArgumentPreservation();
}

pub fn checkControlledStops() !void {
    prohibitDefaultAdmission();
    defer restoreDefaultAdmission();
    inline for (.{ f32, f64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer finish(owner);
        var operation = try owner.begin();
        defer operation.end();
        _ = owner.requestClose();
        const a = [_]T{ 2, 1, 1, 2 };
        var work: [4]T = undefined;
        var vectors: [4]T = undefined;
        var values = [_]T{ -90, -91 };
        const old_values = values;
        const stop = try S.decomposeAdmitted(&operation, T, constant(T, &a, 2, 2), .upper, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{ .max_sweeps = 0 });
        try equal(S.Status.iteration_limit, stop.status);
        try unchanged(T, &old_values, &values);
        const huge = [_]T{ std.math.floatMax(T), std.math.floatMax(T), std.math.floatMax(T), std.math.floatMax(T) };
        const range = try S.decomposeAdmitted(&operation, T, constant(T, &huge, 2, 2), .upper, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{});
        try equal(S.Status.eigenvalue_out_of_range, range.status);
        try unchanged(T, &old_values, &values);
        const empty = try S.decomposeAdmitted(&operation, T, constant(T, &.{}, 0, 0), .upper, matrix(T, &.{}, 0, 0), &.{}, matrix(T, &.{}, 0, 0), .{});
        try equal(S.Status.converged, empty.status);
        const singleton = [_]T{3};
        const scalar = try S.decomposeAdmitted(&operation, T, constant(T, &singleton, 1, 1), .lower, matrix(T, work[0..1], 1, 1), values[0..1], matrix(T, vectors[0..1], 1, 1), .{ .max_sweeps = 0 });
        try equal(S.Status.converged, scalar.status);
        try equal(@as(T, 3), values[0]);
        try equal(@as(T, 1), vectors[0]);
        try equal(@as(usize, 1), owner.snapshot().active_operations);
        try equal(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

test "symmetric admitted candidate controlled stops zero size and scalar keep the caller admission" {
    try checkControlledStops();
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
