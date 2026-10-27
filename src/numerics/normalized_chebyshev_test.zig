// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("normalized_chebyshev.zig");
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
fn expectValue(comptime T: type, expected: T, actual: T) !void {
    const tol: S.Real(T) = if (S.Real(T) == f32) 5e-5 else 3e-13;
    if (comptime S.isComplex(T)) {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tol * @max(1, @abs(expected.re)));
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tol * @max(1, @abs(expected.im)));
    } else try std.testing.expectApproxEqAbs(expected, actual, tol * @max(1, @abs(expected)));
}
fn Backend(comptime T: type) type {
    return struct {
        const Self = @This();
        diagonal: []const S.Real(T),
        calls: usize = 0,
        fail_at: usize = 0,
        nan_at: usize = 0,
        close_at: usize = 0,
        sync: ?P.Synchronization = null,
        fused: bool = false,
        fn action(self: *Self) O.Operator(T) {
            return .{ .context = self, .rows = self.diagonal.len, .columns = self.diagonal.len, .apply_fn = apply, .apply_panel_fn = if (self.fused) panel else null, .synchronization = self.sync };
        }
        fn enter(self: *Self, operation: *const C.Operation) !void {
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
            if (self.calls == self.close_at) _ = operation.borrowedContext().requestClose();
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []T, _: []T) !void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            try self.enter(operation);
            for (x, y, self.diagonal) |v, *out, d| out.* = S.mul(T, S.realScalar(T, d), v);
            if (self.calls == self.fail_at) return error.InjectedFilterAction;
            if (self.calls == self.nan_at and y.len != 0) y[0] = value(T, std.math.nan(S.Real(T)), 0);
        }
        fn panel(raw: *const anyopaque, operation: *const C.Operation, alpha: T, x: D.ConstPanel(T), beta: T, y: D.Panel(T), _: []T) !void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            try self.enter(operation);
            try std.testing.expect(S.isOne(T, alpha) and S.isZero(T, beta));
            for (0..x.shape.column_count) |j| for (self.diagonal, 0..) |d, i| {
                y.column(j).at(i).* = S.mul(T, S.realScalar(T, d), x.column(j).at(i));
            };
            if (self.calls == self.fail_at) return error.InjectedFilterAction;
        }
    };
}
// Independent evaluation through cosine/cosh. No sigma recurrence is reused.
fn logCosh(x: f64) f64 {
    return x + std.math.log1p(std.math.exp(-2 * x)) - std.math.ln2;
}
fn oracle(lambda: f64, degree: usize, lower: f64, cutoff: f64, upper: f64) f64 {
    if (degree == 0) return 1;
    const center = (upper + cutoff) / 2;
    const half = (upper - cutoff) / 2;
    const x = (lambda - center) / half;
    const x0 = (lower - center) / half;
    const m: f64 = @floatFromInt(degree);
    const sign: f64 = if (degree % 2 == 0) 1 else -1;
    const logden = logCosh(m * std.math.acosh(-x0));
    if (@abs(x) <= 1) return sign * std.math.cos(m * std.math.acos(x)) * std.math.exp(-logden);
    return (if (x < 0) 1 else sign) * std.math.exp(logCosh(m * std.math.acosh(@abs(x))) - logden);
}
test "normalized Chebyshev four scalar families match independent polynomial ratios through degree 512" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const diagonal = [_]S.Real(T){ -3, -2, -1, 0, 5 };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        for ([_]usize{ 0, 1, 2, 17, 64, 512 }) |degree| {
            var backend = Backend(T){ .diagonal = &diagonal };
            var x: [5]T = undefined;
            for (&x, 0..) |*v, i| v.* = value(T, @floatFromInt(i + 1), 0.25);
            const before = x;
            var y: [5]T = @splat(value(T, 91, 3));
            var work: [27]T = @splat(value(T, 77, 2));
            try F.Managed(E).runBorrowed(T, owner, .{ .action = backend.action(), .input = &x, .output = &y, .degree = degree, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } }, &work);
            try std.testing.expectEqual(degree, backend.calls);
            try std.testing.expectEqualDeep(before, x);
            for (diagonal, x, y) |d, v, actual| try expectValue(T, S.mul(T, S.realScalar(T, @floatCast(oracle(d, degree, -3, -1, 5))), v), actual);
            for (work[25..]) |tail| try std.testing.expectEqualDeep(value(T, 77, 2), tail);
        }
    }
}
test "normalized Chebyshev in-place row-major panels preserve padding and share fused or column fallback semantics" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ false, true }) |fused| {
        const diag = [_]S.Real(T){ -3, 0, 5 };
        var backend = Backend(T){ .diagonal = &diag, .fused = fused };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        const pad = value(T, std.math.nan(S.Real(T)), 7);
        var data = [_]T{ value(T, 1, 0.2), value(T, 2, -0.4), pad, pad, value(T, 3, 0.1), value(T, 4, 0.6), pad, pad, value(T, 5, -0.1), value(T, 6, 0.7), pad, pad };
        const before = data;
        const panel: D.Panel(T) = .{ .values = &data, .shape = .{ .row_count = 3, .column_count = 2, .leading_dimension = 4, .layout = .row_major } };
        try F.Managed(E).runPanelScratch(T, owner, .{ .action = backend.action(), .input = panel.asConst(), .output = panel, .degree = 7, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } });
        try std.testing.expectEqual(@as(usize, if (fused) 7 else 14), backend.calls);
        for (diag, 0..) |d, i| for (0..2) |j| try expectValue(T, S.mul(T, S.realScalar(T, @floatCast(oracle(d, 7, -3, -1, 5))), before[i * 4 + j]), data[i * 4 + j]);
        for ([_]usize{ 2, 3, 6, 7, 10, 11 }) |i| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[i]), std.mem.asBytes(&data[i]));
    };
}
test "normalized Chebyshev rejects bad bounds geometry capacity alias and nonfinite input before callbacks" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    const diag = [_]f64{ 1, 2 };
    var backend = Backend(f64){ .diagonal = &diag };
    const x = [_]f64{ 1, 2 };
    var y = [_]f64{ 91, 92 };
    const saved = y;
    var work: [10]f64 = undefined;
    var request = F.Request(f64){ .action = backend.action(), .input = &x, .output = &y, .degree = 3, .bounds = .{ .lower = 0, .cutoff = 1, .upper = 3 } };
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(f64, owner, request, work[0..9]));
    request.bounds.cutoff = 0;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runBorrowed(f64, owner, request, &work));
    request.bounds.cutoff = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(f64, owner, request, &work));
    request.bounds.cutoff = 1;
    request.input = &.{ std.math.nan(f64), 1 };
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(f64, owner, request, &work));
    request.input = &x;
    request.output = y[0..1];
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runBorrowed(f64, owner, request, &work));
    request.output = &y;
    request.input = work[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(f64, owner, request, &work));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    try std.testing.expectEqualDeep(saved, y);
    request.input = &x;
    request.degree = 0;
    request.bounds = .{ .lower = std.math.nan(f64), .cutoff = std.math.inf(f64), .upper = 0 };
    try F.Managed(E).runBorrowed(f64, owner, request, &work);
    try std.testing.expectEqualDeep(x, y);
}
test "normalized Chebyshev callback failure nonfinite result and close retain transactional output and release shared resources" {
    inline for (.{ f64, S.ComplexF64 }) |T| for (0..3) |mode| {
        const diag = [_]S.Real(T){ -3, 1 };
        var backend = Backend(T){ .diagonal = &diag, .fail_at = if (mode == 0) 3 else 0, .nan_at = if (mode == 1) 2 else 0, .close_at = if (mode == 2) 2 else 0 };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        const x = [_]T{ value(T, 1, 0.25), value(T, 2, -0.5) };
        var y = [_]T{ value(T, 91, 3), value(T, 92, 4) };
        const saved = y;
        const req = F.Request(T){ .action = backend.action(), .input = &x, .output = &y, .degree = 4, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } };
        const result = F.Managed(E).runScratch(T, owner, req);
        if (mode < 2) {
            try std.testing.expectError(if (mode == 0) error.InjectedFilterAction else error.NonFiniteResult, result);
            try std.testing.expectEqualDeep(saved, y);
        } else {
            try result;
            try expectValue(T, x[0], y[0]);
        }
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    };
}
test "normalized Chebyshev lower-bound normalization survives subnormal and full finite-range spectral intervals" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| for ([_]S.Real(T){ std.math.floatMax(S.Real(T)), std.math.floatTrueMin(S.Real(T)) }) |magnitude| {
        const diag = [_]S.Real(T){-magnitude};
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        for ([_]usize{ 1, 3, 17 }) |degree| {
            var backend = Backend(T){ .diagonal = &diag };
            const x = [_]T{value(T, 1, 0)};
            var y = [_]T{value(T, 91, 3)};
            try F.Managed(E).runScratch(T, owner, .{ .action = backend.action(), .input = &x, .output = &y, .degree = degree, .bounds = .{ .lower = -magnitude, .cutoff = 0, .upper = magnitude } });
            try expectValue(T, x[0], y[0]);
        }
    };
}
const Observer = struct {
    failure: ?P.Stage = null,
    count: usize = 0,
    last_error: ?anyerror = null,
    fn descriptor(self: *@This()) P.Synchronization {
        return .{ .context = self, .agree_fn = agree };
    }
    fn agree(raw: *const anyopaque, operation: ?*const C.Operation, point: P.Point, local: ?anyerror) !void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        self.count += 1;
        if (local) |err| {
            self.last_error = err;
            return err;
        }
        if (operation) |op| try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
        if (self.failure == point.stage) return error.InjectedRemoteFilterFailure;
    }
};
test "normalized Chebyshev remote configuration arithmetic and commit failures converge before final publication" {
    for ([_]P.Stage{ .max_order, .local_arithmetic, .commit }) |stage| {
        var observe = Observer{ .failure = stage };
        const diag = [_]f64{ -3, 1 };
        var backend = Backend(f64){ .diagonal = &diag, .sync = observe.descriptor() };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        const x = [_]f64{ 1, 2 };
        var y = [_]f64{ 91, 92 };
        const saved = y;
        try std.testing.expectError(error.InjectedRemoteFilterFailure, F.Managed(E).runScratch(f64, owner, .{ .action = backend.action(), .input = &x, .output = &y, .degree = 3, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } }));
        try std.testing.expectEqualDeep(saved, y);
        if (stage == .max_order) try std.testing.expectEqual(@as(usize, 0), backend.calls);
    }
}
test "normalized Chebyshev empty distributed panels retain callbacks while workspace arithmetic stays native width" {
    var observe = Observer{};
    var backend = Backend(f64){ .diagonal = &.{}, .sync = observe.descriptor() };
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    const panel: D.Panel(f64) = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 2, .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    try F.Managed(E).runPanelScratch(f64, owner, .{ .action = backend.action(), .input = panel.asConst(), .output = panel, .degree = 3, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } });
    try std.testing.expectEqual(@as(usize, 6), backend.calls);
    var huge = backend.action();
    huge.rows = std.math.maxInt(usize);
    huge.columns = huge.rows;
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(f64, huge));
}
test "normalized Chebyshev shared scratch allocation failure never invokes the operator" {
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    const diag = [_]f64{ -3, 1 };
    var backend = Backend(f64){ .diagonal = &diag };
    const x = [_]f64{ 1, 2 };
    var y = [_]f64{ 91, 92 };
    const saved = y;
    try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(f64, owner, .{ .action = backend.action(), .input = &x, .output = &y, .degree = 3, .bounds = .{ .lower = -3, .cutoff = -1, .upper = 5 } }));
    try std.testing.expectEqualDeep(saved, y);
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
}

test "normalized Chebyshev acts on non-diagonal Hermitian operators against independent spectral projectors" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Matrix = struct {
            matrix: [4]T,
            fn apply(raw: *const anyopaque, _: *const C.Operation, x: []const T, y: []T, _: []T) !void {
                const self: *const @This() = @ptrCast(@alignCast(raw));
                for (0..2) |row| {
                    var sum = S.zero(T);
                    for (0..2) |column| sum = S.add(T, sum, S.mul(T, self.matrix[row + 2 * column], x[column]));
                    y[row] = sum;
                }
            }
        };
        const off = if (comptime S.isComplex(T)) value(T, 0, 1) else value(T, 1, 0);
        var matrix = Matrix{ .matrix = .{ value(T, 2, 0), S.conj(T, off), off, value(T, 2, 0) } };
        const action: O.Operator(T) = .{ .context = &matrix, .rows = 2, .columns = 2, .apply_fn = Matrix.apply };
        const x = [_]T{ value(T, 1, 0.2), value(T, -0.3, 0.5) };
        const hx = [_]T{ S.add(T, S.mul(T, value(T, 2, 0), x[0]), S.mul(T, off, x[1])), S.add(T, S.mul(T, S.conj(T, off), x[0]), S.mul(T, value(T, 2, 0), x[1])) };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        for ([_]usize{ 0, 1, 2, 9, 64 }) |degree| {
            var y = x;
            try F.Managed(E).runScratch(T, owner, .{ .action = action, .input = &y, .output = &y, .degree = degree, .bounds = .{ .lower = 0, .cutoff = 2, .upper = 4 } });
            for (x, hx, y) |v, av, actual| {
                const p1 = S.mul(T, S.realScalar(T, 0.5), S.sub(T, S.mul(T, S.realScalar(T, 3), v), av));
                const p3 = S.mul(T, S.realScalar(T, 0.5), S.sub(T, av, v));
                const expected = S.add(T, S.mul(T, S.realScalar(T, @floatCast(oracle(1, degree, 0, 2, 4))), p1), S.mul(T, S.realScalar(T, @floatCast(oracle(3, degree, 0, 2, 4))), p3));
                try expectValue(T, expected, actual);
            }
        }
    }
}

test "normalized Chebyshev reports final arithmetic overflow after a finite callback without publishing" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const tiny = std.math.floatTrueMin(S.Real(T));
        const diag = [_]S.Real(T){1};
        var backend = Backend(T){ .diagonal = &diag };
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        const x = [_]T{value(T, 1, 0)};
        var y = [_]T{value(T, 91, 3)};
        const before = y;
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, .{ .action = backend.action(), .input = &x, .output = &y, .degree = 1, .bounds = .{ .lower = -2 * tiny, .cutoff = -tiny, .upper = 0 } }));
        try std.testing.expectEqual(@as(usize, 1), backend.calls);
        try std.testing.expectEqualDeep(before, y);
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
