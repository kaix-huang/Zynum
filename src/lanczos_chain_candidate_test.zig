// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const S = N.Scalar;
const E = N.Execution;
const C = E.context;
const Chain = N.LanczosChain;
const M = Chain.Managed(E);
fn context(storage: *FixtureReferenceContext) *C.Context {
    storage.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    return storage.borrowedContext();
}
fn close(owner: *C.Context) !void {
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(owner));
}
fn value(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn tolerance(comptime T: type) S.Real(T) {
    return if (S.Real(T) == f32) 4e-5 else 4e-13;
}
fn Backend(comptime T: type) type {
    return struct {
        const Self = @This();
        diagonal: []const S.Real(T),
        off: []const S.Real(T) = &.{},
        phase: bool = false,
        calls: usize = 0,
        fail_at: usize = 0,
        nonfinite_at: usize = 0,
        close_at: usize = 0,
        expected_pins: usize = 0,
        fn action(self: *Self) N.Operator(T) {
            return .{ .context = self, .rows = self.diagonal.len, .columns = self.diagonal.len, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, input: []const T, output: []T, _: []T) !void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            const owner = operation.borrowedContext();
            try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
            try std.testing.expectEqual(self.expected_pins, owner.snapshot().resource_pins);
            if (self.close_at == self.calls) _ = owner.requestClose();
            for (input, output, self.diagonal, 0..) |x, *y, d, i| {
                y.* = S.mul(T, value(T, d, 0), x);
                if (self.off.len > 0) {
                    if (i > 0) {
                        const b = if (self.phase and S.isComplex(T)) value(T, 0, self.off[i - 1]) else value(T, self.off[i - 1], 0);
                        y.* = S.add(T, y.*, S.mul(T, b, input[i - 1]));
                    }
                    if (i + 1 < input.len) {
                        const b = if (self.phase and S.isComplex(T)) value(T, 0, -self.off[i]) else value(T, self.off[i], 0);
                        y.* = S.add(T, y.*, S.mul(T, b, input[i + 1]));
                    }
                }
            }
            if (self.fail_at == self.calls) return error.InjectedActionFailure;
            if (self.nonfinite_at == self.calls and output.len > 0) output[0] = value(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
pub fn checkExactTridiagonal() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        inline for (.{ false, true }) |phase| {
            var reference_owner: FixtureReferenceContext = undefined;
            const owner = context(&reference_owner);
            defer close(owner) catch unreachable;
            var backend = Backend(T){ .diagonal = &.{ 0.5, -1, 2 }, .off = &.{ 0.25, 0.5 }, .phase = phase };
            const input = [_]T{ if (phase and S.isComplex(T)) value(T, 0, 1) else value(T, 1, 0), value(T, 0, 0), value(T, 0, 0) };
            const saved = input;
            var alpha: [7]R = @splat(91);
            var beta: [6]R = @splat(92);
            var vectors: [11]T = @splat(value(T, 77, 13));
            var real: [11]R = @splat(78);
            const result = try M.runBorrowed(T, owner, .{ .action = backend.action(), .initial = &input, .global_dimension = 3, .max_order = 5, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta }, .{ .vectors = &vectors, .real = &real });
            try std.testing.expectEqual(Chain.Status.breakdown, result.status);
            try std.testing.expectEqual(@as(usize, 3), result.order);
            try std.testing.expectEqual(@as(usize, 3), result.action_count);
            try std.testing.expectEqual(@as(usize, 3), backend.calls);
            try std.testing.expectEqualSlices(R, &.{ 0.5, -1, 2 }, alpha[0..3]);
            try std.testing.expectEqualSlices(R, &.{ 0.25, 0.5 }, beta[0..2]);
            for (alpha[3..]) |x| try std.testing.expectEqual(@as(R, 91), x);
            for (beta[2..]) |x| try std.testing.expectEqual(@as(R, 92), x);
            for (vectors[9..]) |x| try std.testing.expectEqualDeep(value(T, 77, 13), x);
            for (real[9..]) |x| try std.testing.expectEqual(@as(R, 78), x);
            try std.testing.expectEqualDeep(saved, input);
        }
    }
}
const Z = struct {
    re: f64,
    im: f64,
    fn reciprocal(z: Z) Z {
        const d = z.re * z.re + z.im * z.im;
        return .{ .re = z.re / d, .im = -z.im / d };
    }
};
pub fn checkIndependentSpectralMeasure() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Backend(T){ .diagonal = &.{ -1, 0.5, 2 } };
        const input = [_]T{ value(T, 1, 0), if (S.isComplex(T)) value(T, 0, 2) else value(T, -2, 0), value(T, -3, 0) };
        var alpha: [3]R = undefined;
        var beta: [2]R = undefined;
        var vectors: [9]T = undefined;
        var real: [5]R = undefined;
        const result = try M.runBorrowed(T, owner, .{ .action = backend.action(), .initial = &input, .global_dimension = 3, .max_order = 3, .breakdown_tolerance = tolerance(T), .alpha = &alpha, .beta = &beta }, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(@as(usize, 3), result.order);
        for (0..21) |i| {
            const energy = -2 + @as(f64, @floatFromInt(i)) * 0.25;
            const eta: f64 = 0.2;
            var expected = Z{ .re = 0, .im = 0 };
            for ([_]f64{ -1, 0.5, 2 }, [_]f64{ 1, 4, 9 }) |e, weight| {
                const pole = Z.reciprocal(.{ .re = energy - e, .im = eta });
                expected.re += weight / 14 * pole.re;
                expected.im += weight / 14 * pole.im;
            }
            var tail = Z.reciprocal(.{ .re = energy - alpha[2], .im = eta });
            var j: usize = 2;
            while (j > 0) {
                j -= 1;
                const b2 = @as(f64, beta[j]) * beta[j];
                tail = Z.reciprocal(.{ .re = energy - alpha[j] - b2 * tail.re, .im = eta - b2 * tail.im });
            }
            const eps: f64 = if (R == f32) 8e-6 else 8e-14;
            try std.testing.expectApproxEqAbs(expected.re, tail.re, eps);
            try std.testing.expectApproxEqAbs(expected.im, tail.im, eps);
        }
    }
}
pub fn checkStrictThresholdAndDepthOne() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Backend(T){ .diagonal = &.{ 0, 0 }, .off = &.{0.5} };
        const input = [_]T{ value(T, 1, 0), value(T, 0, 0) };
        var alpha: [3]R = @splat(91);
        var beta: [2]R = @splat(92);
        var vectors: [6]T = undefined;
        var real: [5]R = undefined;
        var request = Chain.Request(T){ .action = backend.action(), .initial = &input, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0.5, .alpha = &alpha, .beta = &beta };
        var result = try M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(@as(usize, 2), result.order);
        try std.testing.expectEqual(@as(R, 0.5), beta[0]);
        alpha = @splat(91);
        beta = @splat(92);
        request.breakdown_tolerance = 0.75;
        result = try M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(@as(usize, 1), result.order);
        try std.testing.expectEqual(Chain.Status.breakdown, result.status);
        try std.testing.expectEqualSlices(R, &.{ 92, 92 }, &beta);
        request.breakdown_tolerance = 0;
        request.max_order = 1;
        result = try M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(Chain.Status.depth_limit, result.status);
        try std.testing.expectEqual(@as(R, 0.5), result.residual_norm);
        try std.testing.expectEqual(@as(R, 91), alpha[1]);
    }
}
fn sliceAt(comptime T: type, bytes: []u8, offset: usize, count: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(bytes.ptr + offset)))[0..count];
}
pub fn checkMixedAliasesAndTails() !void {
    const T = S.ComplexF64;
    for (0..5) |first| {
        for (first + 1..5) |second| {
            var reference_owner: FixtureReferenceContext = undefined;
            const owner = context(&reference_owner);
            defer close(owner) catch unreachable;
            var backend = Backend(T){ .diagonal = &.{ 1, 2 } };
            var bytes: [1024]u8 align(16) = @splat(0);
            var offsets = [_]usize{ 0, 128, 256, 384, 512 };
            offsets[second] = offsets[first];
            const initial = sliceAt(T, &bytes, offsets[0], 2);
            const vectors = sliceAt(T, &bytes, offsets[1], 6);
            const real = sliceAt(f64, &bytes, offsets[2], 3);
            const alpha = sliceAt(f64, &bytes, offsets[3], 2);
            const beta = sliceAt(f64, &bytes, offsets[4], 1);
            const saved = bytes;
            try std.testing.expectError(error.AliasingNotAllowed, M.runBorrowed(T, owner, .{ .action = backend.action(), .initial = initial, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = alpha, .beta = beta }, .{ .vectors = vectors, .real = real }));
            try std.testing.expectEqual(@as(usize, 0), backend.calls);
            try std.testing.expectEqualDeep(saved, bytes);
        }
    }
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = &.{ 1, 2 } };
    var bytes: [128]u8 align(16) = @splat(0);
    const alpha = sliceAt(f64, &bytes, 0, 4);
    const initial = sliceAt(T, &bytes, 16, 2); // Only alpha's declared tail overlaps.
    var beta: [1]f64 = @splat(92);
    var vectors: [6]T = undefined;
    var real: [3]f64 = undefined;
    try std.testing.expectError(error.AliasingNotAllowed, M.runBorrowed(T, owner, .{ .action = backend.action(), .initial = initial, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = alpha, .beta = &beta }, .{ .vectors = &vectors, .real = &real }));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    // Only active workspace spans are reserved. Adjacent inactive tails may
    // serve caller coefficients without creating a hidden numerical alias.
    var real_backend = Backend(f64){ .diagonal = &.{ 0, 0 }, .off = &.{0.5} };
    var slots: [10]f64 = @splat(91);
    var rw: [3]f64 = undefined;
    const r = try M.runBorrowed(f64, owner, .{ .action = real_backend.action(), .initial = &.{ 1, 0 }, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = slots[6..8], .beta = slots[8..10] }, .{ .vectors = &slots, .real = &rw });
    try std.testing.expectEqual(@as(usize, 2), r.order);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0.5, 91 }, slots[6..10]);
}
pub fn checkPreflight() !void {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{ 1, 2 } };
    var input = [_]f64{ 1, 0 };
    var alpha = [_]f64{ 91, 91 };
    var beta = [_]f64{92};
    var vectors: [6]f64 = undefined;
    var real: [3]f64 = undefined;
    const workspace = Chain.Workspace(f64){ .vectors = &vectors, .real = &real };
    var request = Chain.Request(f64){ .action = backend.action(), .initial = &input, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta };
    request.action.columns = 1;
    try std.testing.expectError(error.DimensionMismatch, M.runBorrowed(f64, owner, request, workspace));
    request.action = backend.action();
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |invalid| {
        request.breakdown_tolerance = invalid;
        try std.testing.expectError(error.InvalidOptions, M.runBorrowed(f64, owner, request, workspace));
        request.breakdown_tolerance = 0;
        request.hermitian_tolerance = invalid;
        try std.testing.expectError(error.InvalidOptions, M.runBorrowed(f64, owner, request, workspace));
        request.hermitian_tolerance = 0;
    }
    request.global_dimension = 0;
    try std.testing.expectError(error.InvalidOptions, M.runBorrowed(f64, owner, request, workspace));
    request.global_dimension = 3;
    try std.testing.expectError(error.MissingGlobalReduction, M.runBorrowed(f64, owner, request, workspace));
    request.global_dimension = 2;
    request.max_order = 0;
    try std.testing.expectError(error.InvalidOptions, M.runBorrowed(f64, owner, request, workspace));
    request.max_order = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, M.runBorrowed(f64, owner, request, workspace));
    request.max_order = 2;
    request.alpha = alpha[0..1];
    try std.testing.expectError(error.BufferTooSmall, M.runBorrowed(f64, owner, request, workspace));
    request.alpha = &alpha;
    request.beta = &.{};
    try std.testing.expectError(error.BufferTooSmall, M.runBorrowed(f64, owner, request, workspace));
    request.beta = &beta;
    try std.testing.expectError(error.WorkspaceTooSmall, M.runBorrowed(f64, owner, request, .{ .vectors = vectors[0..5], .real = &real }));
    try std.testing.expectError(error.WorkspaceTooSmall, M.runBorrowed(f64, owner, request, .{ .vectors = &vectors, .real = real[0..2] }));
    input[0] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, M.runBorrowed(f64, owner, request, workspace));
    input[0] = 1;
    const overflowing: [*]f64 = @ptrFromInt(std.math.maxInt(usize) - 7);
    request.alpha = overflowing[0..2];
    try std.testing.expectError(error.SizeOverflow, M.runBorrowed(f64, owner, request, workspace));
    request.alpha = &alpha;
    request.beta = @as([*]f64, @ptrFromInt(8))[0..std.math.maxInt(usize)];
    try std.testing.expectError(error.SizeOverflow, M.runBorrowed(f64, owner, request, workspace));
    request.beta = &beta;
    request.action.workspace_elements = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, M.runBorrowed(f64, owner, request, workspace));
    var huge_action = backend.action();
    huge_action.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, Chain.workspaceLength(f64, huge_action, 1));
    try std.testing.expectError(error.SizeOverflow, Chain.workspaceLength(f64, backend.action(), std.math.maxInt(usize) / 2));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqualSlices(f64, &.{ 91, 91 }, &alpha);
    try std.testing.expectEqual(@as(f64, 92), beta[0]);
    // A malformed managed request does not enter even allocator callbacks.
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var reference_managed_owner: FixtureReferenceContext = undefined;
    reference_managed_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const managed_owner = reference_managed_owner.borrowedContext();
    defer close(managed_owner) catch unreachable;
    request.action = backend.action();
    request.alpha = alpha[0..1];
    try std.testing.expectError(error.BufferTooSmall, M.runScratch(f64, managed_owner, request));
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    try std.testing.expectEqual(@as(usize, 0), managed_owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), managed_owner.snapshot().resource_pins);
}
pub fn checkLateFailureTransaction() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var backend = Backend(T){ .diagonal = &.{ 0.5, -1, 2 }, .off = &.{ 0.25, 0.5 }, .phase = true, .fail_at = 3 };
        const input = [_]T{ value(T, 1, 0), value(T, 0, 0), value(T, 0, 0) };
        var alpha: [5]R = @splat(91);
        var beta: [4]R = @splat(92);
        var vectors: [9]T = undefined;
        var real: [5]R = undefined;
        const request = Chain.Request(T){ .action = backend.action(), .initial = &input, .global_dimension = 3, .max_order = 3, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta };
        try std.testing.expectError(error.InjectedActionFailure, M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real }));
        for (alpha) |x| try std.testing.expectEqual(@as(R, 91), x);
        for (beta) |x| try std.testing.expectEqual(@as(R, 92), x);
        backend.calls = 0;
        backend.fail_at = 0;
        backend.nonfinite_at = 2;
        try std.testing.expectError(error.NonFiniteResult, M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real }));
        for (alpha) |x| try std.testing.expectEqual(@as(R, 91), x);
        for (beta) |x| try std.testing.expectEqual(@as(R, 92), x);
    }
}
pub fn checkReductionFailureAndHermitianThreshold() !void {
    const T = S.ComplexF64;
    const Metric = struct {
        inner_calls: usize = 0,
        norm_calls: usize = 0,
        fail_inner: usize = 0,
        fail_norm: usize = 0,
        imaginary: f64 = 0,
        fn inner(raw: *const anyopaque, _: *const C.Operation, x: []const T, y: []const T) !T {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.inner_calls += 1;
            if (self.inner_calls == self.fail_inner) return error.InjectedReductionFailure;
            var out = try N.Lanczos.localInner(T, x, y);
            out.im = self.imaginary;
            return out;
        }
        fn norm(raw: *const anyopaque, _: *const C.Operation, x: []const T) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.norm_calls += 1;
            if (self.norm_calls == self.fail_norm) return error.InjectedReductionFailure;
            return N.Lanczos.localNorm(T, x);
        }
    };
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(T){ .diagonal = &.{ 0.5, -1, 2 }, .off = &.{ 0.25, 0.5 } };
    const initial = [_]T{ value(T, 1, 0), value(T, 0, 0), value(T, 0, 0) };
    var alpha: [4]f64 = @splat(91);
    var beta: [3]f64 = @splat(92);
    var vectors: [9]T = undefined;
    var real: [5]f64 = undefined;
    var metric = Metric{ .fail_inner = 2 };
    var request = Chain.Request(T){ .action = backend.action(), .initial = &initial, .global_dimension = 3, .max_order = 3, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta, .reduction = .{ .context = &metric, .inner_fn = Metric.inner, .norm_fn = Metric.norm } };
    try std.testing.expectError(error.InjectedReductionFailure, M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real }));
    metric = .{ .fail_norm = 4 };
    try std.testing.expectError(error.InjectedReductionFailure, M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real }));
    metric = .{ .imaginary = 0.5 };
    try std.testing.expectError(error.NonHermitianAction, M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real }));
    for (alpha) |x| try std.testing.expectEqual(@as(f64, 91), x);
    for (beta) |x| try std.testing.expectEqual(@as(f64, 92), x);
    // The largest finite tolerance and imaginary projection remain accepted
    // by the widened comparison without requiring a finite narrowed product.
    request.hermitian_tolerance = std.math.floatMax(f64);
    request.max_order = 1;
    metric = .{ .imaginary = std.math.floatMax(f64) };
    _ = try M.runBorrowed(T, owner, request, .{ .vectors = &vectors, .real = &real });
    try std.testing.expectEqual(@as(f64, 0.5), alpha[0]);
}
pub fn checkEmptyGlobalAndDepth() !void {
    const Metric = struct {
        inner_calls: usize = 0,
        norm_calls: usize = 0,
        fn inner(raw: *const anyopaque, op: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expect(x.len == 0 and y.len == 0);
            self.inner_calls += 1;
            return @floatFromInt(self.inner_calls);
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), x.len);
            self.norm_calls += 1;
            return if (self.norm_calls == 1) 1 else 0.5;
        }
    };
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{} };
    var metric = Metric{};
    var alpha: [6]f64 = @splat(91);
    var beta: [5]f64 = @splat(92);
    var real: [7]f64 = undefined;
    const request = Chain.Request(f64){ .action = backend.action(), .initial = &.{}, .global_dimension = 1, .max_order = 4, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta, .reduction = .{ .context = &metric, .inner_fn = Metric.inner, .norm_fn = Metric.norm } };
    const result = try M.runBorrowed(f64, owner, request, .{ .vectors = &.{}, .real = &real });
    try std.testing.expectEqual(Chain.Status.depth_limit, result.status);
    try std.testing.expectEqual(@as(usize, 4), result.order);
    try std.testing.expectEqual(@as(usize, 4), backend.calls);
    try std.testing.expectEqual(@as(usize, 4), metric.inner_calls);
    try std.testing.expectEqual(@as(usize, 5), metric.norm_calls);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 91, 91 }, &alpha);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.5, 0.5, 92, 92 }, &beta);
}
pub fn checkSubnormalStartsAndContinuation() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        inline for (.{ 1, 2, 16 }) |multiple| {
            var reference_owner: FixtureReferenceContext = undefined;
            const owner = context(&reference_owner);
            defer close(owner) catch unreachable;
            var backend = Backend(T){ .diagonal = &.{ 1, 1 } };
            const tiny = std.math.floatTrueMin(R) * multiple;
            const initial = [_]T{ value(T, tiny, 0), if (S.isComplex(T)) value(T, 0, tiny) else value(T, -tiny, 0) };
            var alpha: [2]R = @splat(91);
            var beta: [1]R = @splat(92);
            var vectors: [6]T = undefined;
            var real: [3]R = undefined;
            const result = try M.runBorrowed(T, owner, .{ .action = backend.action(), .initial = &initial, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = tolerance(T), .alpha = &alpha, .beta = &beta }, .{ .vectors = &vectors, .real = &real });
            try std.testing.expectEqual(@as(usize, 1), result.order);
            try std.testing.expectApproxEqAbs(@as(R, 1), alpha[0], tolerance(T));
            try std.testing.expectEqual(@as(R, 91), alpha[1]);
            try std.testing.expectEqual(@as(R, 92), beta[0]);
        }
        const Star = struct {
            calls: usize = 0,
            fn apply(raw: *const anyopaque, _: *const C.Operation, x: []const T, y: []T, _: []T) !void {
                const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                self.calls += 1;
                try std.testing.expectApproxEqAbs(@as(R, 1), try N.Lanczos.localNorm(T, x), tolerance(T));
                const tiny = value(T, std.math.floatTrueMin(R), 0);
                y[0] = S.mul(T, tiny, S.add(T, x[1], x[2]));
                y[1] = S.mul(T, tiny, x[0]);
                y[2] = S.mul(T, tiny, x[0]);
            }
        };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var star = Star{};
        const action = N.Operator(T){ .context = &star, .rows = 3, .columns = 3, .apply_fn = Star.apply };
        const initial = [_]T{ value(T, 1, 0), value(T, 0, 0), value(T, 0, 0) };
        var alpha: [2]R = @splat(91);
        var beta: [1]R = @splat(92);
        var vectors: [9]T = undefined;
        var real: [3]R = undefined;
        const result = try M.runBorrowed(T, owner, .{ .action = action, .initial = &initial, .global_dimension = 3, .max_order = 2, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta }, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(@as(usize, 2), result.order);
        try std.testing.expectEqual(@as(usize, 2), star.calls);
        try std.testing.expectEqual(std.math.floatTrueMin(R), beta[0]);
    }
}
pub fn checkSubnormalMetricFailures() !void {
    const Metric = struct {
        calls: usize = 0,
        invalid: ?f64 = null,
        fail: bool = false,
        continuing: bool = false,
        fn inner(_: *const anyopaque, _: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            return N.Lanczos.localInner(f64, x, y);
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            const state = op.borrowedContext().snapshot();
            try std.testing.expectEqual(@as(usize, 1), state.active_operations);
            try std.testing.expectEqual(@as(usize, 2), state.resource_pins);
            if (self.continuing and self.calls == 2) return std.math.floatTrueMin(f64);
            if (self.calls == (if (self.continuing) @as(usize, 3) else 2)) {
                if (self.fail) return error.InjectedReductionFailure;
                if (self.invalid) |invalid| return invalid;
            }
            return N.Lanczos.localNorm(f64, x);
        }
    };
    inline for (.{ false, true }) |continuing| {
        for ([_]?f64{ 0, std.math.nan(f64), std.math.inf(f64), null }) |invalid| {
            var reference_owner: FixtureReferenceContext = undefined;
            const owner = context(&reference_owner);
            defer close(owner) catch unreachable;
            var backend = Backend(f64){ .diagonal = &.{ 0, 0 }, .off = &.{if (continuing) std.math.floatTrueMin(f64) else 0}, .expected_pins = 2 };
            var metric = Metric{ .invalid = invalid, .fail = invalid == null, .continuing = continuing };
            const tiny = std.math.floatTrueMin(f64);
            const input = if (continuing) [_]f64{ 1, 0 } else [_]f64{ tiny, tiny };
            var alpha: [3]f64 = @splat(91);
            var beta: [2]f64 = @splat(92);
            const request = Chain.Request(f64){ .action = backend.action(), .initial = &input, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta, .reduction = .{ .context = &metric, .inner_fn = Metric.inner, .norm_fn = Metric.norm } };
            if (invalid == null) try std.testing.expectError(error.InjectedReductionFailure, M.runScratch(f64, owner, request)) else try std.testing.expectError(error.NonFiniteReduction, M.runScratch(f64, owner, request));
            try std.testing.expectEqual(@as(usize, if (continuing) 1 else 0), backend.calls);
            try std.testing.expectEqual(@as(usize, if (continuing) 3 else 2), metric.calls);
            for (alpha) |x| try std.testing.expectEqual(@as(f64, 91), x);
            for (beta) |x| try std.testing.expectEqual(@as(f64, 92), x);
            try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        }
    }
    const EmptyMetric = struct {
        calls: usize = 0,
        invalid: ?f64 = null,
        fn inner(_: *const anyopaque, op: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expect(x.len == 0 and y.len == 0);
            return 2;
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 2), op.borrowedContext().snapshot().resource_pins);
            try std.testing.expectEqual(@as(usize, 0), x.len);
            return switch (self.calls) {
                1 => std.math.floatTrueMin(f64),
                2 => self.invalid orelse 1,
                else => 0,
            };
        }
    };
    for ([_]?f64{ null, 0, std.math.nan(f64), std.math.inf(f64) }) |invalid| {
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var metric = EmptyMetric{ .invalid = invalid };
        var backend = Backend(f64){ .diagonal = &.{}, .expected_pins = 2 };
        var alpha: [2]f64 = @splat(91);
        var beta: [1]f64 = @splat(92);
        const request = Chain.Request(f64){ .action = backend.action(), .initial = &.{}, .global_dimension = 1, .max_order = 1, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta, .reduction = .{ .context = &metric, .inner_fn = EmptyMetric.inner, .norm_fn = EmptyMetric.norm } };
        if (invalid == null) {
            const result = try M.runScratch(f64, owner, request);
            try std.testing.expectEqual(@as(usize, 1), result.order);
            try std.testing.expectEqual(Chain.Status.breakdown, result.status);
            try std.testing.expectEqual(@as(f64, 2), alpha[0]);
            try std.testing.expectEqual(@as(usize, 3), metric.calls);
            try std.testing.expectEqual(@as(usize, 1), backend.calls);
        } else {
            try std.testing.expectError(error.NonFiniteReduction, M.runScratch(f64, owner, request));
            try std.testing.expectEqual(@as(f64, 91), alpha[0]);
            try std.testing.expectEqual(@as(usize, 2), metric.calls);
            try std.testing.expectEqual(@as(usize, 0), backend.calls);
        }
        try std.testing.expectEqual(@as(f64, 91), alpha[1]);
        try std.testing.expectEqual(@as(f64, 92), beta[0]);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}
pub fn checkAllRoutesCloseAndAllocationFailures() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        for (0..5) |route| {
            var allocations = E.test_support.Allocations{ .budget = 2 };
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            var backend = Backend(T){ .diagonal = &.{ 0.5, -1, 2 }, .off = &.{ 0.25, 0.5 }, .close_at = 1, .expected_pins = if (route == 2 or route == 4) 2 else 0 };
            const initial = [_]T{ value(T, 1, 0), value(T, 0, 0), value(T, 0, 0) };
            var alpha: [3]R = @splat(91);
            var beta: [2]R = @splat(92);
            var vectors: [9]T = undefined;
            var real: [5]R = undefined;
            const workspace = Chain.Workspace(T){ .vectors = &vectors, .real = &real };
            const request = Chain.Request(T){ .action = backend.action(), .initial = &initial, .global_dimension = 3, .max_order = 3, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta };
            var result: Chain.Result(T) = undefined;
            if (route < 3) {
                var operation = try owner.begin();
                result = switch (route) {
                    0 => try Chain.runAdmitted(T, &operation, request, workspace),
                    1 => try M.runBorrowedAdmitted(T, &operation, request, workspace),
                    else => try M.runScratchAdmitted(T, &operation, request),
                };
                try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
                operation.end();
            } else result = if (route == 3) try M.runBorrowed(T, owner, request, workspace) else try M.runScratch(T, owner, request);
            try std.testing.expectEqual(@as(usize, 3), result.order);
            try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
            const stats = allocations.stats();
            try std.testing.expectEqual(@as(usize, if (route == 2 or route == 4) 2 else 0), stats.accepted);
            try std.testing.expectEqual(stats.accepted, stats.freed);
            try std.testing.expectError(error.Closing, M.runScratch(T, owner, request));
            try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(owner));
        }
        for (0..2) |budget| {
            inline for (.{ false, true }) |admitted| {
                var allocations = E.test_support.Allocations{ .budget = budget };
                var reference_owner: FixtureReferenceContext = undefined;
                reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
                const owner = reference_owner.borrowedContext();
                defer close(owner) catch unreachable;
                var backend = Backend(T){ .diagonal = &.{ 0, 0 }, .off = &.{0.5}, .expected_pins = 2 };
                const initial = [_]T{ value(T, 1, 0), value(T, 0, 0) };
                var alpha: [3]R = @splat(91);
                var beta: [2]R = @splat(92);
                const request = Chain.Request(T){ .action = backend.action(), .initial = &initial, .global_dimension = 2, .max_order = 2, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta };
                if (admitted) {
                    var operation = try owner.begin();
                    try std.testing.expectError(error.OutOfMemory, M.runScratchAdmitted(T, &operation, request));
                    try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
                    operation.end();
                } else try std.testing.expectError(error.OutOfMemory, M.runScratch(T, owner, request));
                try std.testing.expectEqual(@as(usize, 0), backend.calls);
                try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
                try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
                for (alpha) |x| try std.testing.expectEqual(@as(R, 91), x);
                for (beta) |x| try std.testing.expectEqual(@as(R, 92), x);
                var stats = allocations.stats();
                try std.testing.expectEqual(budget, stats.accepted);
                try std.testing.expectEqual(budget, stats.freed);
                try std.testing.expectEqual(@as(usize, 1), stats.rejected);
                allocations.budget = budget + 2;
                _ = try M.runScratch(T, owner, request);
                stats = allocations.stats();
                try std.testing.expectEqual(budget + 2, stats.accepted);
                try std.testing.expectEqual(stats.accepted, stats.freed);
            }
        }
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
