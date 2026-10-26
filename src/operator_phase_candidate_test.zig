// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const P = N.Synchronization;
const E = N.Execution;
const C = E.context;
const S = N.Scalar;
const Cheb = N.Chebyshev.Managed(E);

fn value(comptime T: type, real: S.Real(T)) T {
    return S.realScalar(T, real);
}
fn owner() C.Context {
    return C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
}
fn close(context: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    _ = context.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, context.tryProgressClose());
}
const Observer = struct {
    points: [256]P.Point = undefined,
    count: usize = 0,
    failure_stage: ?P.Stage = null,
    failure_index: u64 = 0,
    last_local_error: ?anyerror = null,
    null_operation_on_admission_failure: bool = false,
    close_at_commit: bool = false,
    commit_pins: usize = 0,
    fn descriptor(self: *@This()) P.Synchronization {
        return .{ .context = self, .agree_fn = agree };
    }
    fn agree(raw: *const anyopaque, operation: ?*const C.Operation, point: P.Point, local: ?anyerror) !void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        try std.testing.expectEqual(@as(u32, 0), point.reserved);
        try std.testing.expect(self.count < self.points.len);
        self.points[self.count] = point;
        self.count += 1;
        if (operation) |op| {
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
        } else {
            try std.testing.expect(point.stage == .admission and local != null);
            self.null_operation_on_admission_failure = true;
        }
        if (local) |err| {
            self.last_local_error = err;
            return err;
        }
        if (point.stage == .commit and self.close_at_commit) {
            const context = operation.?.borrowedContext();
            self.commit_pins = context.snapshot().resource_pins;
            _ = context.requestClose();
            try std.testing.expectEqual(C.CloseResult.busy, context.tryProgressClose());
        }
        if (self.failure_stage == point.stage and self.failure_index == point.index) return error.RemotePhaseFailure;
    }
};
fn Backend(comptime T: type) type {
    return struct {
        factor: S.Real(T),
        rows: usize = 1,
        calls: usize = 0,
        fail_at: usize = 0,
        fn action(self: *@This(), observer: *Observer) N.Operator(T) {
            return .{ .context = self, .rows = self.rows, .columns = self.rows, .apply_fn = apply, .synchronization = observer.descriptor() };
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
            self.calls += 1;
            for (x, y) |source, *destination| destination.* = S.mul(T, value(T, self.factor), source);
            if (self.calls == self.fail_at) return error.InjectedActionFailure;
        }
    };
}

pub fn checkChebyshevLocalAndRemoteFailures() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var context = owner();
        defer close(&context) catch unreachable;
        var observer = Observer{};
        var backend = Backend(T){ .factor = @sqrt(std.math.floatMax(R)) * @sqrt(@as(R, 1.2)) };
        const input = [_]T{value(T, 0.5)};
        const coefficients = [_]T{ value(T, 0), value(T, 0), value(T, 0), value(T, 1) };
        var output = [_]T{value(T, 91)};
        try std.testing.expectError(error.NonFiniteResult, Cheb.runScratch(T, &context, .{ .action = backend.action(&observer), .input = &input, .coefficients = &coefficients, .output = &output }));
        try std.testing.expectEqual(@as(usize, 2), backend.calls);
        try std.testing.expectEqualDeep(value(T, 91), output[0]);
        try std.testing.expectEqual(error.NonFiniteResult, observer.last_local_error.?);
        try std.testing.expectEqual(P.Stage.local_arithmetic, observer.points[observer.count - 1].stage);
        try std.testing.expectEqual(@as(u64, 2), observer.points[observer.count - 1].index);
        observer = .{ .failure_stage = .local_arithmetic, .failure_index = 2 };
        backend = .{ .factor = 0.5 };
        try std.testing.expectError(error.RemotePhaseFailure, Cheb.runScratch(T, &context, .{ .action = backend.action(&observer), .input = &input, .coefficients = &coefficients, .output = &output }));
        try std.testing.expectEqual(@as(usize, 2), backend.calls);
        try std.testing.expectEqualDeep(value(T, 91), output[0]);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    }
}

pub fn checkAllocationAdmissionAndCommit() !void {
    var allocations = E.test_support.Allocations{ .budget = 1 };
    var context = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    defer close(&context) catch unreachable;
    var observer = Observer{ .failure_stage = .vector_allocation };
    var backend = Backend(f64){ .factor = 0.5 };
    const input = [_]f64{2};
    var output = [_]f64{91};
    var request = N.Chebyshev.Request(f64){ .action = backend.action(&observer), .input = &input, .coefficients = &.{ 1, 2 }, .output = &output };
    try std.testing.expectError(error.RemotePhaseFailure, Cheb.runScratch(f64, &context, request));
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    allocations.budget = 0;
    observer = .{};
    request.action = backend.action(&observer);
    try std.testing.expectError(error.OutOfMemory, Cheb.runScratch(f64, &context, request));
    try std.testing.expectEqual(error.OutOfMemory, observer.last_local_error.?);
    allocations.budget = allocations.stats().accepted + 1;
    observer = .{ .failure_stage = .commit };
    request.action = backend.action(&observer);
    try std.testing.expectError(error.RemotePhaseFailure, Cheb.runScratch(f64, &context, request));
    try std.testing.expectEqual(@as(f64, 91), output[0]);
    observer = .{ .close_at_commit = true };
    allocations.budget = allocations.stats().accepted + 1;
    request.action = backend.action(&observer);
    try Cheb.runScratch(f64, &context, request);
    try std.testing.expectEqual(@as(f64, 4), output[0]);
    try std.testing.expectEqual(@as(usize, 1), observer.commit_pins);
    observer = .{};
    request.action = backend.action(&observer);
    try std.testing.expectError(error.Closing, Cheb.runScratch(f64, &context, request));
    try std.testing.expect(observer.null_operation_on_admission_failure);
}

pub fn checkExactCoefficientsAndEmptyPanel() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var context = owner();
        defer close(&context) catch unreachable;
        var observer = Observer{};
        var backend = Backend(T){ .factor = 0.5 };
        var coefficient = value(T, -0.0);
        if (comptime S.isComplex(T)) coefficient.im = -0.0;
        const input = [_]T{value(T, 1)};
        var output = [_]T{value(T, 91)};
        try Cheb.runScratch(T, &context, .{ .action = backend.action(&observer), .input = &input, .coefficients = &.{coefficient}, .output = &output });
        var seen_real = false;
        var seen_imaginary = false;
        for (observer.points[0..observer.count]) |point| {
            if (point.stage == .coefficient_real) {
                seen_real = true;
                try std.testing.expectEqual(P.realBits(S.Real(T), -0.0), point.control);
            }
            if (point.stage == .coefficient_imaginary) {
                seen_imaginary = true;
                try std.testing.expectEqual(P.realBits(S.Real(T), -0.0), point.control);
            }
        }
        try std.testing.expect(seen_real);
        try std.testing.expectEqual(S.isComplex(T), seen_imaginary);
        observer = .{};
        backend = .{ .factor = 0.5, .rows = 0 };
        const shape = N.Dense.PanelShape{ .row_count = 0, .column_count = 2, .leading_dimension = 0, .layout = .column_major };
        try Cheb.runPanelScratch(T, &context, .{ .action = backend.action(&observer), .input = .{ .values = &.{}, .shape = shape }, .coefficients = &.{ value(T, 1), value(T, 2), value(T, 3) }, .output = .{ .values = &.{}, .shape = shape } });
        try std.testing.expectEqual(@as(usize, 4), backend.calls);
    }
}

pub fn checkLanczosSecondAllocationAndStopping() !void {
    var allocations = E.test_support.Allocations{ .budget = 2 };
    var context = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    defer close(&context) catch unreachable;
    var observer = Observer{ .failure_stage = .real_allocation };
    var backend = Backend(f64){ .factor = 1 };
    const input = [_]f64{1};
    var request = N.Lanczos.Request(f64){ .action = backend.action(&observer), .initial = &input, .global_dimension = 1, .max_order = 2, .ritz_tolerance = 0 };
    try std.testing.expectError(error.RemotePhaseFailure, N.Lanczos.Managed(E).runScratch(f64, &context, request));
    try std.testing.expectEqual(@as(usize, 2), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 2), allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    observer = .{ .failure_stage = .stopping_decision, .failure_index = 1 };
    allocations.budget = allocations.stats().accepted + 2;
    request.action = backend.action(&observer);
    try std.testing.expectError(error.RemotePhaseFailure, N.Lanczos.Managed(E).runScratch(f64, &context, request));
    try std.testing.expectEqual(@as(usize, 1), backend.calls);
    observer = .{};
    request.action = backend.action(&observer);
    allocations.budget = allocations.stats().accepted + 2;
    const result = try N.Lanczos.Managed(E).runScratch(f64, &context, request);
    try std.testing.expectEqual(N.Lanczos.Status.exhausted_dimension, result.status);
    try std.testing.expectEqual(@as(f64, 1), result.minimum);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
}

pub fn checkChainSubnormalAndCommit() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var context = owner();
        defer close(&context) catch unreachable;
        var observer = Observer{ .failure_stage = .unit_norm };
        var backend = Backend(T){ .factor = 1 };
        const input = [_]T{value(T, 7 * std.math.floatMin(R) * std.math.floatEps(R))};
        var alpha: [5]R = @splat(91);
        var beta: [4]R = @splat(92);
        var request = N.LanczosChain.Request(T){ .action = backend.action(&observer), .initial = &input, .global_dimension = 1, .max_order = 3, .breakdown_tolerance = 0, .alpha = &alpha, .beta = &beta };
        try std.testing.expectError(error.RemotePhaseFailure, N.LanczosChain.Managed(E).runScratch(T, &context, request));
        try std.testing.expectEqual(@as(usize, 0), backend.calls);
        try std.testing.expectEqualSlices(R, &@as([5]R, @splat(91)), &alpha);
        try std.testing.expectEqualSlices(R, &@as([4]R, @splat(92)), &beta);
        observer = .{ .failure_stage = .commit, .failure_index = 1 };
        request.action = backend.action(&observer);
        try std.testing.expectError(error.RemotePhaseFailure, N.LanczosChain.Managed(E).runScratch(T, &context, request));
        try std.testing.expectEqual(@as(usize, 1), backend.calls);
        try std.testing.expectEqualSlices(R, &@as([5]R, @splat(91)), &alpha);
        try std.testing.expectEqualSlices(R, &@as([4]R, @splat(92)), &beta);
        observer = .{ .close_at_commit = true };
        request.action = backend.action(&observer);
        const result = try N.LanczosChain.Managed(E).runScratch(T, &context, request);
        try std.testing.expectEqual(N.LanczosChain.Status.breakdown, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.order);
        try std.testing.expectEqual(@as(R, 1), alpha[0]);
        try std.testing.expectEqual(@as(usize, 2), observer.commit_pins);
        for (alpha[1..]) |tail| try std.testing.expectEqual(@as(R, 91), tail);
        for (beta) |tail| try std.testing.expectEqual(@as(R, 92), tail);
    }
}
