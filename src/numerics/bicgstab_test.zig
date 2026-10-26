// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("bicgstab.zig");
const S = @import("../blas/core/shared/scalar.zig");
const E = @import("zynum-execution");
const C = E.context;
const O = @import("operator.zig");
const Red = @import("reductions.zig");
const P = @import("synchronization.zig");
fn val(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn close(owner: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
fn Backend(comptime T: type) type {
    return struct {
        const Self = @This();
        matrix: []const T,
        rows: usize,
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        fn action(self: *Self) O.Operator(T) {
            return .{ .context = self, .rows = self.rows, .columns = self.rows, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, op: *const C.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            if (self.close_at == self.calls) {
                _ = op.borrowedContext().requestClose();
                try std.testing.expectEqual(C.CloseResult.busy, op.borrowedContext().tryProgressClose());
            }
            for (y, 0..) |*out, row| {
                out.* = S.zero(T);
                for (x, 0..) |v, col| out.* = S.add(T, out.*, S.mul(T, self.matrix[row * self.rows + col], v));
            }
            if (self.fail_at == self.calls) return error.InjectedLinearAction;
            if (self.nan_at == self.calls and y.len != 0) y[y.len - 1] = val(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
fn checkValue(comptime T: type, want: T, actual: T, tol: S.Real(T)) !void {
    if (comptime S.isComplex(T)) {
        try std.testing.expectApproxEqAbs(want.re, actual.re, tol);
        try std.testing.expectApproxEqAbs(want.im, actual.im, tol);
    } else try std.testing.expectApproxEqAbs(want, actual, tol);
}
test "BiCGStab four scalar nonsymmetric solves and nonzero initial states" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const tol: R = if (R == f32) 2e-5 else 2e-12;
        const matrix = [_]T{ val(T, 4, -0.5), val(T, 1, 0.2), val(T, -0.25, 0), val(T, -2, -0.1), val(T, 5, 0.7), val(T, 1, 0), val(T, 0.5, 0), val(T, -1, 0.1), val(T, 3, -0.3) };
        const exact = [_]T{ val(T, 1, 0.5), val(T, -2, 0.25), val(T, 0.25, -0.75) };
        var backend = Backend(T){ .matrix = &matrix, .rows = 3 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var b: [3]T = undefined;
        for (&b, 0..) |*out, row| {
            out.* = S.zero(T);
            for (exact, 0..) |x, col| out.* = S.add(T, out.*, S.mul(T, matrix[row * 3 + col], x));
        }
        const bcopy = b;
        var output: [5]T = @splat(val(T, 77, -13));
        const seed = [_]T{ val(T, 0.5, 0.2), val(T, 0.1, -0.1), val(T, -0.2, 0.3) };
        for ([_]bool{ false, true }) |warm| {
            const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .rhs = &b, .initial = if (warm) &seed else null, .output = &output, .global_dimension = 3, .max_iterations = 40, .relative_tolerance = if (R == f32) 2e-6 else 1e-14 });
            try std.testing.expectEqual(F.Status.converged, result.status);
            try std.testing.expect(result.iterations > 0 and result.iterations <= 40);
            try std.testing.expect(result.action_count > 0);
            for (exact, output[0..3]) |x, y| try checkValue(T, x, y, tol);
            try std.testing.expectEqualDeep(val(T, 77, -13), output[3]);
            try std.testing.expectEqualDeep(val(T, 77, -13), output[4]);
            try std.testing.expectEqualDeep(bcopy, b);
            var residual: [3]T = undefined;
            for (&residual, 0..) |*out, row| {
                out.* = b[row];
                for (output[0..3], 0..) |x, col| out.* = S.sub(T, out.*, S.mul(T, matrix[row * 3 + col], x));
            }
            const actual = try Red.localNorm(T, &residual);
            try std.testing.expect(actual <= tol * 10);
            try std.testing.expect(result.residual_norm <= tol * 10);
        }
    }
}
test "BiCGStab tiny RHS and widened alpha avoid false underflow breakdown" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const tiny = std.math.floatTrueMin(R);
        const matrix = [_]T{val(T, tiny, 0)};
        const b = [_]T{val(T, tiny, 0)};
        var output = [_]T{val(T, 77, 0)};
        var backend = Backend(T){ .matrix = &matrix, .rows = 1 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .rhs = &b, .output = &output, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = 0, .absolute_tolerance = 0 });
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqualDeep(val(T, 1, 0), output[0]);
        try std.testing.expectEqual(@as(R, 0), result.residual_norm);
    }
}
test "BiCGStab controlled alpha and omega breakdowns return verified finite candidates" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        const b = [_]T{ val(T, 1, 0), val(T, 0, 0) };
        var output: [2]T = @splat(val(T, 77, 0));
        const zero_matrix = [_]T{ val(T, 0, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0, 0) };
        var backend = Backend(T){ .matrix = &zero_matrix, .rows = 2 };
        var result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .rhs = &b, .output = &output, .global_dimension = 2, .max_iterations = 4, .relative_tolerance = 0 });
        try std.testing.expectEqual(F.Status.alpha_breakdown, result.status);
        try std.testing.expectEqual(@as(S.Real(T), 1), result.residual_norm);
        try std.testing.expectEqualDeep([_]T{ val(T, 0, 0), val(T, 0, 0) }, output);
        const omega_matrix = [_]T{ val(T, 1, 0), val(T, 1, 0), val(T, 1, 0), val(T, 0, 0) };
        backend = .{ .matrix = &omega_matrix, .rows = 2 };
        result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .rhs = &b, .output = &output, .global_dimension = 2, .max_iterations = 4, .relative_tolerance = 0 });
        try std.testing.expectEqual(F.Status.omega_breakdown, result.status);
        try std.testing.expectEqual(@as(S.Real(T), 1), result.residual_norm);
        try std.testing.expectEqualDeep([_]T{ val(T, 1, 0), val(T, 0, 0) }, output);
    }
}

test "BiCGStab optional right preconditioning and aliased initial output" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Real = S.Real(T);
        const a = [_]T{ val(T, 2, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0, 0), val(T, 4, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0, 0), val(T, 8, 0) };
        const inverse = [_]T{ val(T, 0.5, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0.25, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0, 0), val(T, 0.125, 0) };
        var action = Backend(T){ .matrix = &a, .rows = 3 };
        var preconditioner = Backend(T){ .matrix = &inverse, .rows = 3 };
        const exact = [_]T{ val(T, 1, 0.5), val(T, -2, 0.25), val(T, 0.5, -0.5) };
        const b = [_]T{ S.mul(T, val(T, 2, 0), exact[0]), S.mul(T, val(T, 4, 0), exact[1]), S.mul(T, val(T, 8, 0), exact[2]) };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var out = [_]T{ val(T, 0.25, 0), val(T, 0.5, 0), val(T, 0.75, 0), val(T, 71, 0) };
        const result = try F.Managed(E).runScratch(T, &owner, .{ .action = action.action(), .preconditioner = preconditioner.action(), .rhs = &b, .initial = out[0..3], .output = &out, .global_dimension = 3, .max_iterations = 8, .relative_tolerance = if (Real == f32) 1e-6 else 1e-14 });
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expect(result.preconditioner_count > 0);
        try std.testing.expectEqual(preconditioner.calls, result.preconditioner_count);
        try std.testing.expectEqual(action.calls, result.action_count);
        for (exact, out[0..3]) |x, y| try checkValue(T, x, y, if (Real == f32) 1e-5 else 1e-13);
        try std.testing.expectEqualDeep(val(T, 71, 0), out[3]);
    }
}

test "BiCGStab zero budget and empty systems retain explicit stop semantics" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        const matrix = [_]T{val(T, 1, 0)};
        var backend = Backend(T){ .matrix = &matrix, .rows = 1 };
        const b = [_]T{val(T, 2, 0)};
        var out = [_]T{val(T, 77, 0)};
        var request = F.Request(T){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 1, .max_iterations = 0, .relative_tolerance = 0 };
        var result = try F.Managed(E).runScratch(T, &owner, request);
        try std.testing.expectEqual(F.Status.iteration_limit, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
        try std.testing.expectEqual(@as(usize, 0), result.action_count);
        try std.testing.expectEqualDeep(val(T, 0, 0), out[0]);
        try std.testing.expectEqual(@as(S.Real(T), 2), result.residual_norm);
        request.initial = &b;
        result = try F.Managed(E).runScratch(T, &owner, request);
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
        try std.testing.expectEqual(@as(usize, 1), result.action_count);
        var empty = Backend(T){ .matrix = &.{}, .rows = 0 };
        out[0] = val(T, 77, 0);
        result = try F.Managed(E).runScratch(T, &owner, .{ .action = empty.action(), .rhs = &.{}, .output = &out, .global_dimension = 0, .max_iterations = 0, .relative_tolerance = 0 });
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), empty.calls);
        try std.testing.expectEqualDeep(val(T, 77, 0), out[0]);
    }
}

test "BiCGStab rho breakdown returns independently verified physical residual" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const a = [_]T{ val(T, 1, 0), val(T, 1, 0), val(T, 1, 0), val(T, 1, 0), val(T, 2, 0), val(T, 0, 0), val(T, -1, 0), val(T, 0, 0), val(T, 3, 0) };
        const b = [_]T{ val(T, 1, 0), val(T, 0, 0), val(T, 0, 0) };
        var out: [3]T = undefined;
        var backend = Backend(T){ .matrix = &a, .rows = 3 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        const result = try F.Managed(E).runScratch(T, &owner, .{ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 3, .max_iterations = 8, .relative_tolerance = 0 });
        try std.testing.expectEqual(F.Status.rho_breakdown, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.iterations);
        const tol: S.Real(T) = if (S.Real(T) == f32) 1e-6 else 1e-14;
        try checkValue(T, val(T, 1, 0), out[0], tol);
        try checkValue(T, val(T, -5.0 / 13.0, 0), out[1], tol);
        try checkValue(T, val(T, 5.0 / 13.0, 0), out[2], tol);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), @floatCast(@sqrt(@as(f64, 13)) / 13)), result.residual_norm, tol);
    }
}

test "BiCGStab preflight rejects aliases geometry flags and short scratch before callbacks" {
    const identity = [_]f64{ 1, 0, 0, 1 };
    var backend = Backend(f64){ .matrix = &identity, .rows = 2 };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    var b = [_]f64{ 1, 2 };
    var out = [_]f64{ 77, 78, 79 };
    var work: [18]f64 = undefined;
    var q = F.Request(f64){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 2, .max_iterations = 2, .relative_tolerance = 1e-12 };
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(f64, &owner, q, work[0..17]));
    q.output = b[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.output = &out;
    q.relative_tolerance = -1;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.relative_tolerance = 1e-12;
    q.absolute_tolerance = std.math.nan(f64);
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.absolute_tolerance = 0;
    q.global_dimension = 3;
    try std.testing.expectError(error.MissingGlobalReduction, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.global_dimension = 2;
    q.initial = &.{1};
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.initial = null;
    b[1] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    b[1] = 2;
    var bad = backend.action();
    bad.columns = 1;
    q.preconditioner = bad;
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runBorrowed(f64, &owner, q, &work));
    q.preconditioner = null;
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqualSlices(f64, &.{ 77, 78, 79 }, &out);
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(f64, .{ .context = &backend, .rows = std.math.maxInt(usize), .columns = std.math.maxInt(usize), .apply_fn = Backend(f64).apply }, null));
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runBorrowed(f64, &owner, q, &work)).status);
}

test "BiCGStab final verification and preconditioner callback failures roll back and retry" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const matrix = [_]T{val(T, 1, 0)};
        const b = [_]T{val(T, 2, 0)};
        var out = [_]T{val(T, 77, 0)};
        var backend = Backend(T){ .matrix = &matrix, .rows = 1 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var q = F.Request(T){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = 1e-6 };
        backend.fail_at = 2;
        try std.testing.expectError(error.InjectedLinearAction, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(val(T, 77, 0), out[0]);
        backend.calls = 0;
        backend.fail_at = 0;
        backend.nan_at = 2;
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(val(T, 77, 0), out[0]);
        backend.calls = 0;
        backend.nan_at = 0;
        var preconditioner = Backend(T){ .matrix = &matrix, .rows = 1, .fail_at = 1 };
        q.preconditioner = preconditioner.action();
        try std.testing.expectError(error.InjectedLinearAction, F.Managed(E).runScratch(T, &owner, q));
        try std.testing.expectEqualDeep(val(T, 77, 0), out[0]);
        preconditioner.fail_at = 0;
        try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(T, &owner, q)).status);
        try std.testing.expectEqualDeep(b, out);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

const Observer = struct {
    reject: ?P.Stage = null,
    fn agree(raw: *const anyopaque, _: ?*const C.Operation, point: P.Point, local_error: ?anyerror) !void {
        const self: *const @This() = @ptrCast(@alignCast(raw));
        if (local_error) |err| return err;
        if (self.reject == point.stage) return error.RemoteLinearFailure;
    }
    fn sync(self: *@This()) P.Synchronization {
        return .{ .context = self, .agree_fn = agree };
    }
};
test "BiCGStab final phase rejection preserves candidate and retry releases leases" {
    const matrix = [_]f64{1};
    const b = [_]f64{2};
    var out = [_]f64{77};
    var backend = Backend(f64){ .matrix = &matrix, .rows = 1 };
    var observer = Observer{ .reject = .commit };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    var action = backend.action();
    action.synchronization = observer.sync();
    const q = F.Request(f64){ .action = action, .rhs = &b, .output = &out, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = 1e-12 };
    try std.testing.expectError(error.RemoteLinearFailure, F.Managed(E).runScratch(f64, &owner, q));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    observer.reject = null;
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(f64, &owner, q)).status);
    try std.testing.expectEqual(@as(f64, 2), out[0]);
}

test "BiCGStab closing owner completes admitted work and allocation failure leaves output" {
    const matrix = [_]f64{1};
    const b = [_]f64{2};
    var out = [_]f64{77};
    var backend = Backend(f64){ .matrix = &matrix, .rows = 1, .close_at = 1 };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    const q = F.Request(f64){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = 1e-12 };
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(f64, &owner, q)).status);
    try std.testing.expectError(error.Closing, owner.begin());
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var limited = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    defer close(&limited) catch unreachable;
    out[0] = 77;
    try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(f64, &limited, q));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}

const ReductionsMock = struct {
    fail_norm: bool = false,
    nan_inner: bool = false,
    underestimate_once: bool = false,
    used_underestimate: bool = false,
    norms: usize = 0,
    inners: usize = 0,
    fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        self.norms += 1;
        try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
        if (self.fail_norm) return error.InjectedLinearReduction;
        if (self.underestimate_once and !self.used_underestimate and x.len == 2 and x[0] > 0 and x[1] < 0 and @abs(x[0]) < 0.3 and @abs(x[1]) < 0.3) {
            self.used_underestimate = true;
            return 0;
        }
        return Red.localNorm(f64, x);
    }
    fn inner(raw: *const anyopaque, _: *const C.Operation, x: []const f64, y: []const f64) !f64 {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        self.inners += 1;
        if (self.nan_inner) return std.math.nan(f64);
        return Red.localInner(f64, x, y);
    }
    fn descriptor(self: *@This()) F.Reduction(f64) {
        return .{ .context = self, .inner_fn = inner, .norm_fn = norm };
    }
};
test "BiCGStab borrowed scalar reduction errors are transactional" {
    const matrix = [_]f64{1};
    const b = [_]f64{2};
    var out = [_]f64{77};
    var backend = Backend(f64){ .matrix = &matrix, .rows = 1 };
    var reducer = ReductionsMock{ .fail_norm = true };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    const q = F.Request(f64){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 1, .max_iterations = 4, .relative_tolerance = 1e-12, .reduction = reducer.descriptor() };
    try std.testing.expectError(error.InjectedLinearReduction, F.Managed(E).runScratch(f64, &owner, q));
    reducer.fail_norm = false;
    reducer.nan_inner = true;
    try std.testing.expectError(error.NonFiniteReduction, F.Managed(E).runScratch(f64, &owner, q));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
    reducer.nan_inner = false;
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(f64, &owner, q)).status);
}
test "BiCGStab underestimated recurrence residual forces true verification and budgeted restart" {
    const matrix = [_]f64{ 1, 0, 0, 2 };
    const b = [_]f64{ 1, 1 };
    var out = [_]f64{ 77, 78 };
    var backend = Backend(f64){ .matrix = &matrix, .rows = 2 };
    var reducer = ReductionsMock{ .underestimate_once = true };
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    const q = F.Request(f64){ .action = backend.action(), .rhs = &b, .output = &out, .global_dimension = 2, .max_iterations = 8, .relative_tolerance = 1e-12, .reduction = reducer.descriptor() };
    const result = try F.Managed(E).runScratch(f64, &owner, q);
    try std.testing.expect(reducer.used_underestimate);
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 1), result.residual_restarts);
    try std.testing.expect(result.iterations <= 8);
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[1], 1e-12);
    reducer = .{ .underestimate_once = true };
    backend.calls = 0;
    var limited = q;
    limited.max_iterations = 1;
    const stopped = try F.Managed(E).runScratch(f64, &owner, limited);
    try std.testing.expectEqual(F.Status.iteration_limit, stopped.status);
    try std.testing.expectEqual(@as(usize, 1), stopped.iterations);
    try std.testing.expect(stopped.residual_norm > 0.1);
}

const EmptyGlobal = struct {
    norms: usize = 0,
    inners: usize = 0,
    fn norm(raw: *const anyopaque, _: *const C.Operation, x: []const f64) !f64 {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        try std.testing.expectEqual(@as(usize, 0), x.len);
        self.norms += 1;
        return 1;
    }
    fn inner(raw: *const anyopaque, _: *const C.Operation, x: []const f64, y: []const f64) !f64 {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        try std.testing.expectEqual(@as(usize, 0), x.len);
        try std.testing.expectEqual(@as(usize, 0), y.len);
        self.inners += 1;
        return 1;
    }
};
test "BiCGStab empty local shard participates and cannot infer global convergence" {
    var backend = Backend(f64){ .matrix = &.{}, .rows = 0 };
    var global = EmptyGlobal{};
    var out = [_]f64{77};
    var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    const result = try F.Managed(E).runScratch(f64, &owner, .{ .action = backend.action(), .rhs = &.{}, .output = &out, .global_dimension = 2, .max_iterations = 1, .relative_tolerance = 0, .reduction = .{ .context = &global, .inner_fn = EmptyGlobal.inner, .norm_fn = EmptyGlobal.norm } });
    try std.testing.expectEqual(F.Status.iteration_limit, result.status);
    try std.testing.expect(backend.calls >= 3 and global.norms > 0 and global.inners > 0);
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}

test "BiCGStab native affine system covers complex contour shifts and zero action scale" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Real = S.Real(T);
        const matrix = [_]T{ val(T, 1, 0), val(T, 0, 0), val(T, 0, 0), val(T, 2, 0) };
        var backend = Backend(T){ .matrix = &matrix, .rows = 2 };
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        const shift = val(T, 4, 0.75);
        const scale = val(T, -1, 0);
        const exact = [_]T{ val(T, 1, 0.5), val(T, -0.5, 0.25) };
        const b = [_]T{ S.mul(T, S.add(T, shift, scale), exact[0]), S.mul(T, S.add(T, shift, S.mul(T, scale, val(T, 2, 0))), exact[1]) };
        var out = [_]T{ val(T, 77, 0), val(T, 78, 0) };
        var q = F.Request(T){ .action = backend.action(), .action_scale = scale, .diagonal_shift = shift, .rhs = &b, .output = &out, .global_dimension = 2, .max_iterations = 12, .relative_tolerance = if (Real == f32) 2e-6 else 1e-14 };
        var result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.converged, result.status);
        for (exact, out) |x, y| try checkValue(T, x, y, if (Real == f32) 2e-5 else 1e-13);
        backend.calls = 0;
        backend.fail_at = 1;
        q.action_scale = val(T, 0, 0);
        q.diagonal_shift = val(T, 2, 0);
        result = try F.Managed(E).runScratch(T, &owner, q);
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), backend.calls);
        try std.testing.expectEqual(@as(usize, 0), result.action_count);
        for (b, out) |x, y| try checkValue(T, S.mul(T, val(T, 0.5, 0), x), y, if (Real == f32) 2e-5 else 1e-13);
    }
}
