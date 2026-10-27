// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("subspace_projection.zig");
const D = @import("dense.zig");
const S = @import("../blas/core/shared/scalar.zig");
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
fn near(comptime T: type, expected: T, actual: T) !void {
    const tolerance: S.Real(T) = if (S.Real(T) == f32) 2e-5 else 2e-13;
    try std.testing.expectApproxEqAbs(S.realPart(T, expected), S.realPart(T, actual), tolerance * @max(1, @abs(S.realPart(T, expected))));
    if (comptime S.isComplex(T)) try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance * @max(1, @abs(expected.im)));
}
fn offset(shape: D.PanelShape, row: usize, column: usize) usize {
    return if (shape.layout == .column_major) row + column * shape.leading_dimension else column + row * shape.leading_dimension;
}
fn Fixture(comptime T: type) type {
    return struct {
        y: [14]T = undefined,
        z: [14]T = undefined,
        h: [8]T = undefined,
        s: [8]T = undefined,
        weights: [3]S.Real(T) = .{ 1, 2, 0.5 },
        ys: D.PanelShape = undefined,
        zs: D.PanelShape = undefined,
        hs: D.PanelShape = undefined,
        ss: D.PanelShape = undefined,
        fn init(self: *@This(), yl: D.Layout, zl: D.Layout, hl: D.Layout, sl: D.Layout) void {
            self.ys = .{ .row_count = 3, .column_count = 2, .leading_dimension = if (yl == .row_major) 4 else 5, .layout = yl };
            self.zs = .{ .row_count = 3, .column_count = 2, .leading_dimension = if (zl == .row_major) 4 else 5, .layout = zl };
            self.hs = .{ .row_count = 2, .column_count = 2, .leading_dimension = 4, .layout = hl };
            self.ss = .{ .row_count = 2, .column_count = 2, .leading_dimension = 4, .layout = sl };
            self.y = @splat(value(T, std.math.nan(S.Real(T)), 0));
            self.z = self.y;
            self.reset();
            const left = [_]S.Real(T){ 1, 2, -1 };
            const right = [_]S.Real(T){ 0, 1, 2 };
            const lambda = [_]S.Real(T){ -2, 3, 4 };
            for (0..3) |i| {
                self.y[offset(self.ys, i, 0)] = value(T, left[i], 0);
                self.y[offset(self.ys, i, 1)] = value(T, if (S.isComplex(T)) 0 else right[i], if (S.isComplex(T)) right[i] else 0);
                for (0..2) |j| self.z[offset(self.zs, i, j)] = S.mul(T, S.realScalar(T, lambda[i]), self.y[offset(self.ys, i, j)]);
            }
        }
        fn reset(self: *@This()) void {
            self.h = @splat(value(T, 91, -72));
            self.s = @splat(value(T, 91, -72));
        }
        fn request(self: *@This()) F.Request(T) {
            return .{
                .basis = .{ .values = &self.y, .shape = self.ys },
                .applied = .{ .values = &self.z, .shape = self.zs },
                .weights = &self.weights,
                .hamiltonian = .{ .values = &self.h, .shape = self.hs },
                .overlap = .{ .values = &self.s, .shape = self.ss },
            };
        }
        fn unchanged(self: *@This()) !void {
            for (self.h) |x| try std.testing.expectEqual(value(T, 91, -72), x);
            for (self.s) |x| try std.testing.expectEqual(value(T, 91, -72), x);
        }
        fn expected(self: *@This(), scale: S.Real(T)) !void {
            // Exact hand-calculated Y* W diag(-2,3,4) Y and Y* W Y.
            const h = [_]S.Real(T){ 24, 8, 8, 14 };
            const s = [_]S.Real(T){ 9.5, 3, 3, 4 };
            for (0..2) |j| for (0..2) |i| {
                const cross = i != j and S.isComplex(T);
                const sign: S.Real(T) = if (i > j) -1 else 1;
                try near(T, value(T, if (cross) 0 else scale * h[i + j * 2], if (cross) sign * scale * h[i + j * 2] else 0), self.h[offset(self.hs, i, j)]);
                try near(T, value(T, if (cross) 0 else scale * s[i + j * 2], if (cross) sign * scale * s[i + j * 2] else 0), self.s[offset(self.ss, i, j)]);
            };
            for (0..8) |k| {
                var h_logical = false;
                var s_logical = false;
                for (0..2) |j| for (0..2) |i| {
                    h_logical = h_logical or offset(self.hs, i, j) == k;
                    s_logical = s_logical or offset(self.ss, i, j) == k;
                };
                if (!h_logical) try std.testing.expectEqual(value(T, 91, -72), self.h[k]);
                if (!s_logical) try std.testing.expectEqual(value(T, 91, -72), self.s[k]);
            }
        }
    };
}
fn Reducer(comptime T: type) type {
    return struct {
        calls: usize = 0,
        fail: bool = false,
        nan: bool = false,
        close_owner: bool = false,
        scale: S.Real(T) = 1,
        fn borrow(self: *@This()) F.Reduction(T) {
            return .{ .context = self, .reduce_fn = apply };
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, values: []T) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            const owner = operation.borrowedContext();
            try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 8), values.len);
            if (self.close_owner) {
                _ = owner.requestClose();
                try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(owner));
            }
            for (values) |*x| x.* = S.mul(T, S.realScalar(T, self.scale), x.*);
            if (self.fail) {
                values[0] = value(T, 123, 0);
                return error.InjectedProjectionReduction;
            }
            if (self.nan) values[7] = value(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
test "subspace projection four scalars all layouts match exact weighted complex Gram oracle" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |yl| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |zl| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |hl| inline for (.{ D.Layout.row_major, D.Layout.column_major }) |sl| {
        var f: Fixture(T) = .{};
        f.init(yl, zl, hl, sl);
        const before_y = f.y;
        const before_z = f.z;
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        try F.Managed(E).runScratch(T, owner, f.request());
        try f.expected(1);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_y), std.mem.asBytes(&f.y));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_z), std.mem.asBytes(&f.z));
    };
}
test "subspace projection one packed reduction serves both matrices and fails transactionally with retry" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var f: Fixture(T) = .{};
        f.init(.row_major, .column_major, .column_major, .row_major);
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var reducer: Reducer(T) = .{ .scale = 2 };
        var r = f.request();
        r.reduction = reducer.borrow();
        try F.Managed(E).runScratch(T, owner, r);
        try f.expected(2);
        try std.testing.expectEqual(@as(usize, 1), reducer.calls);
        for (0..2) |fault| {
            f.reset();
            reducer.fail = fault == 0;
            reducer.nan = fault == 1;
            if (fault == 0) try std.testing.expectError(error.InjectedProjectionReduction, F.Managed(E).runScratch(T, owner, r)) else try std.testing.expectError(error.NonFiniteReduction, F.Managed(E).runScratch(T, owner, r));
            try f.unchanged();
            reducer.fail = false;
            reducer.nan = false;
            try F.Managed(E).runScratch(T, owner, r);
            try f.expected(2);
        }
    }
}
test "subspace projection invalid shape metric finite inputs alias and short scratch fail before reduction" {
    var f: Fixture(f64) = .{};
    f.init(.row_major, .column_major, .column_major, .row_major);
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var reducer: Reducer(f64) = .{};
    var r = f.request();
    r.reduction = reducer.borrow();
    var work: [26]f64 = undefined;
    var bad = r;
    bad.applied.shape.column_count = 1;
    try std.testing.expectError(error.DimensionMismatch, F.Managed(E).runBorrowed(f64, owner, bad, &work));
    f.weights[1] = -1;
    try std.testing.expectError(error.InvalidMetric, F.Managed(E).runBorrowed(f64, owner, r, &work));
    f.weights[1] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(f64, owner, r, &work));
    f.weights[1] = 2;
    f.y[offset(f.ys, 2, 1)] = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(f64, owner, r, &work));
    f.y[offset(f.ys, 2, 1)] = 2;
    bad = r;
    bad.overlap = r.hamiltonian;
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(f64, owner, bad, &work));
    bad = r;
    bad.hamiltonian.values = &f.y;
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(f64, owner, bad, &work));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(f64, owner, r, work[0..25]));
    bad = r;
    bad.hamiltonian.values = &work;
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(f64, owner, bad, &work));
    try std.testing.expectEqual(@as(usize, 0), reducer.calls);
    try f.unchanged();
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(f64, std.math.maxInt(usize), 2));
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(f64, 0, std.math.maxInt(usize)));
    try F.Managed(E).runScratch(f64, owner, r);
    try f.expected(1);
}
test "subspace projection arithmetic overflow preserves both outputs and releases the operation" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var f: Fixture(T) = .{};
        f.init(.row_major, .column_major, .column_major, .row_major);
        var reference_owner: FixtureReferenceContext = undefined;
        const owner = context(&reference_owner);
        defer close(owner) catch unreachable;
        var r = f.request();
        f.y[offset(f.ys, 0, 0)] = value(T, std.math.floatMax(S.Real(T)), 0);
        try std.testing.expectError(error.NonFiniteReduction, F.Managed(E).runScratch(T, owner, r));
        try f.unchanged();
        f.init(.row_major, .column_major, .column_major, .row_major);
        f.weights[1] = std.math.floatMax(S.Real(T));
        r = f.request();
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, r));
        try f.unchanged();
    }
}
test "subspace projection final and local remote rejection preserves both matrices" {
    const Observer = struct {
        fail_stage: P.Stage,
        fn agree(raw: *const anyopaque, _: ?*const C.Operation, point: P.Point, _: ?anyerror) !void {
            const self: *const @This() = @ptrCast(@alignCast(raw));
            if (point.stage == self.fail_stage) return error.InjectedRemoteProjection;
        }
    };
    var f: Fixture(f64) = .{};
    f.init(.row_major, .column_major, .column_major, .row_major);
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    for ([_]P.Stage{ .local_arithmetic, .commit }) |stage| {
        var observer: Observer = .{ .fail_stage = stage };
        var r = f.request();
        r.synchronization = .{ .context = &observer, .agree_fn = Observer.agree };
        try std.testing.expectError(error.InjectedRemoteProjection, F.Managed(E).runScratch(f64, owner, r));
        try f.unchanged();
    }
    try F.Managed(E).runScratch(f64, owner, f.request());
    try f.expected(1);
}
test "subspace projection closing callback and OOM preserve shared owner lifetime" {
    var f: Fixture(f64) = .{};
    f.init(.row_major, .column_major, .column_major, .row_major);
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var reducer: Reducer(f64) = .{ .close_owner = true };
    var r = f.request();
    r.reduction = reducer.borrow();
    try F.Managed(E).runScratch(f64, owner, r);
    try f.expected(1);
    try std.testing.expectError(error.Closing, F.Managed(E).runScratch(f64, owner, r));
    var tracker: E.test_support.Allocations = .{ .budget = 0 };
    var reference_denied: FixtureReferenceContext = undefined;
    reference_denied.initInPlace(tracker.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const denied = reference_denied.borrowedContext();
    defer close(denied) catch unreachable;
    f.reset();
    try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(f64, denied, f.request()));
    try f.unchanged();
    try std.testing.expectEqual(@as(usize, 0), tracker.stats().accepted);
}
test "subspace projection empty column set is no-op while empty shard still participates in reduction" {
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    const shape: D.PanelShape = .{ .row_count = 3, .column_count = 0, .leading_dimension = 0, .layout = .row_major };
    var h = [_]f64{91};
    var s = [_]f64{92};
    const weights = [_]f64{ std.math.nan(f64), -1, std.math.inf(f64) };
    const target: D.PanelShape = .{ .row_count = 0, .column_count = 0, .leading_dimension = 0, .layout = .column_major };
    try F.Managed(E).runScratch(f64, owner, .{ .basis = .{ .values = &.{}, .shape = shape }, .applied = .{ .values = &.{}, .shape = shape }, .weights = &weights, .hamiltonian = .{ .values = &h, .shape = target }, .overlap = .{ .values = &s, .shape = target } });
    try std.testing.expectEqual(@as(f64, 91), h[0]);
    try std.testing.expectEqual(@as(f64, 92), s[0]);
    const Remote = struct {
        calls: usize = 0,
        fn apply(raw: *const anyopaque, _: *const C.Operation, values: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            const global = [_]f64{ 24, 8, 8, 14, 9.5, 3, 3, 4 };
            @memcpy(values, &global);
        }
    };
    var remote: Remote = .{};
    var f: Fixture(f64) = .{};
    f.init(.row_major, .column_major, .column_major, .row_major);
    var r = f.request();
    r.basis.shape.row_count = 0;
    r.applied.shape.row_count = 0;
    r.basis.values = &.{};
    r.applied.values = &.{};
    r.weights = &.{};
    r.reduction = .{ .context = &remote, .reduce_fn = Remote.apply };
    try F.Managed(E).runScratch(f64, owner, r);
    try f.expected(1);
    try std.testing.expectEqual(@as(usize, 1), remote.calls);
}
test "subspace projection stable Hermitian average avoids overflowing equal finite entries" {
    const Remote = struct {
        fn apply(_: *const anyopaque, _: *const C.Operation, values: []f64) !void {
            values[0] = 1;
            values[1] = std.math.floatMax(f64);
            values[2] = std.math.floatMax(f64);
            values[3] = 2;
            values[4] = 1;
            values[5] = 0;
            values[6] = 0;
            values[7] = 1;
        }
    };
    var dummy: u8 = 0;
    var f: Fixture(f64) = .{};
    f.init(.row_major, .column_major, .column_major, .row_major);
    var reference_owner: FixtureReferenceContext = undefined;
    const owner = context(&reference_owner);
    defer close(owner) catch unreachable;
    var r = f.request();
    r.reduction = .{ .context = &dummy, .reduce_fn = Remote.apply };
    try F.Managed(E).runScratch(f64, owner, r);
    try std.testing.expectEqual(std.math.floatMax(f64), f.h[offset(f.hs, 0, 1)]);
    try std.testing.expectEqual(std.math.floatMax(f64), f.h[offset(f.hs, 1, 0)]);
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
