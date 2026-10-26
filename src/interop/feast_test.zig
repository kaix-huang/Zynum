// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const E = @import("zynum-execution");
const S = @import("../blas/core/shared/scalar.zig");
const Proto = @import("feast.zig");
const Provider = struct {
    pub const Scalar = S;
    pub const Dense = @import("../numerics/dense.zig");
    pub const Storage = @import("../numerics/storage.zig");
    pub const Execution = E;
    pub const Contour = @import("../numerics/contour_eigensolver.zig");
    pub const SparseOperator = @import("../sparse/operator.zig");
};
const A = Proto.Adapter(Provider);
const T = S.ComplexF64;
fn z(re: f64, im: f64) T {
    return .{ .re = re, .im = im };
}
fn close(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, owner.tryProgressClose());
}
const Fixture = struct {
    matrix: [6]T = undefined,
    rows: [5]i32 = undefined,
    columns: [6]i32 = undefined,
    fpm: [128]i32 = undefined,
    values: [5]f64 = undefined,
    vectors: [16]T = undefined,
    residuals: [5]f64 = undefined,
    entries: usize = 0,
    uplo: u8 = 'F',
    capacity: i32 = 3,
    fn init(self: *@This(), uplo: u8, reuse: bool) void {
        self.uplo = uplo;
        self.capacity = 3;
        self.fpm = @splat(123456789);
        Proto.Parameters.initialize(&self.fpm) catch unreachable;
        self.fpm[0] = 0;
        self.fpm[2] = 12;
        self.fpm[3] = 20;
        self.fpm[42] = 1;
        self.fpm[45] = 40;
        self.fpm[4] = @intFromBool(reuse);
        self.values = @splat(91);
        self.residuals = @splat(92);
        self.vectors = @splat(z(77, -33));
        if (reuse) {
            @memset(self.vectors[0..12], z(0, 0));
            for (0..3) |k| self.vectors[k + k * 4] = z(1, 0);
        }
        self.matrix = @splat(z(std.math.nan(f64), std.math.nan(f64)));
        self.columns = @splat(-99);
        if (uplo == 'F') {
            self.entries = 6;
            self.rows = .{ 1, 3, 5, 6, 7 };
            self.columns = .{ 1, 2, 1, 2, 3, 4 };
            self.matrix = .{ z(2, 0), z(0, 1), z(0, -1), z(2, 0), z(5, 0), z(8, 0) };
        } else if (uplo == 'U') {
            self.entries = 5;
            self.rows = .{ 1, 3, 4, 5, 6 };
            self.columns[0..5].* = .{ 1, 2, 2, 3, 4 };
            self.matrix[0..5].* = .{ z(2, 0), z(0, 1), z(2, 0), z(5, 0), z(8, 0) };
        } else {
            self.entries = 5;
            self.rows = .{ 1, 2, 4, 5, 6 };
            self.columns[0..5].* = .{ 1, 1, 2, 3, 4 };
            self.matrix[0..5].* = .{ z(2, 0), z(0, -1), z(2, 0), z(5, 0), z(8, 0) };
        }
    }
    fn args(self: *@This()) A.Arguments {
        return .{ .uplo = self.uplo, .dimension = 4, .capacity = self.capacity, .matrix = self.matrix[0..self.entries], .rows = &self.rows, .columns = self.columns[0..self.entries], .fpm = &self.fpm, .lower = 0.5, .upper = 3.5, .eigenvalues = &self.values, .vectors = &self.vectors, .residuals = &self.residuals };
    }
};
fn verify(f: *const Fixture, found: usize) !void {
    try std.testing.expectEqual(@as(usize, 2), found);
    for (0..2) |column| {
        const lambda: f64 = if (column == 0) 1 else 3;
        try std.testing.expectApproxEqAbs(lambda, f.values[column], 3e-11);
        var norm: f128 = 0;
        var error_squared: f128 = 0;
        const x = f.vectors[column * 4 ..][0..4];
        const ax = [_]T{ S.add(T, S.mul(T, z(2, 0), x[0]), S.mul(T, z(0, 1), x[1])), S.add(T, S.mul(T, z(0, -1), x[0]), S.mul(T, z(2, 0), x[1])), S.mul(T, z(5, 0), x[2]), S.mul(T, z(8, 0), x[3]) };
        for (x, ax) |a, b| {
            norm += @as(f128, a.re) * a.re + @as(f128, a.im) * a.im;
            const re = @as(f128, b.re) - @as(f128, f.values[column]) * a.re;
            const im = @as(f128, b.im) - @as(f128, f.values[column]) * a.im;
            error_squared += re * re + im * im;
        }
        try std.testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(norm)), 3e-12);
        try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@sqrt(error_squared) / 3.5)), f.residuals[column], 2e-15);
        try std.testing.expect(f.residuals[column] <= 1e-12);
    }
}
test "FEAST typed LP64 parameter initialization defaults and reviewed error boundaries" {
    var fpm: [128]i32 = @splat(123456789);
    try Proto.Parameters.initialize(&fpm);
    for (fpm[0..64]) |v| try std.testing.expectEqual(@as(i32, -111), v);
    for (fpm[64..]) |v| try std.testing.expectEqual(@as(i32, 123456789), v);
    fpm[42] = 1;
    const controls = Proto.Parameters.decode(&fpm).valid;
    try std.testing.expectEqual(@as(usize, 4), controls.nodes);
    try std.testing.expectEqual(@as(usize, 50), controls.refinements);
    try std.testing.expectEqual(@as(i32, 12), controls.exponent);
    try std.testing.expectEqual(@as(usize, 40), controls.linear_iterations);
    try std.testing.expectEqual(@as(f64, 1), controls.ratio);
    const cases = [_][3]i32{ .{ 1, 0, 102 }, .{ 2, -1, 103 }, .{ 2, 17, 103 }, .{ 3, -1, 104 }, .{ 4, 2, 105 }, .{ 15, 3, 116 }, .{ 17, -1, 118 } };
    for (cases) |row| {
        var invalid = fpm;
        invalid[@intCast(row[0])] = row[1];
        try std.testing.expectEqual(row[2], Proto.Parameters.decode(&invalid).invalid);
    }
    var unsupported = fpm;
    unsupported[15] = 2;
    try std.testing.expectEqual(Proto.Parameters.unsupported, Proto.Parameters.decode(&unsupported).invalid);
    var owner = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    var f: Fixture = undefined;
    f.init('F', false);
    var q = f.args();
    q.uplo = 'Z';
    try std.testing.expectEqual(@as(i32, -101), (try A.solve(&owner, q)).info);
    q.uplo = 'F';
    q.dimension = 0;
    try std.testing.expectEqual(@as(i32, -102), (try A.solve(&owner, q)).info);
    q = f.args();
    q.capacity = 0;
    try std.testing.expectEqual(@as(i32, 201), (try A.solve(&owner, q)).info);
    q.capacity = 1;
    try std.testing.expectEqual(@as(i32, 201), (try A.solve(&owner, q)).info);
    q.capacity = 5;
    try std.testing.expectEqual(@as(i32, 201), (try A.solve(&owner, q)).info);
    q = f.args();
    q.lower = 4;
    try std.testing.expectEqual(@as(i32, 200), (try A.solve(&owner, q)).info);
    for (f.values) |v| try std.testing.expectEqual(@as(f64, 91), v);
}
test "FEAST typed LP64 full lower upper CSR random and supplied starts recover Hermitian spectrum" {
    var owner = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    for ([_]u8{ 'F', 'L', 'U' }) |uplo| for ([_]bool{ false, true }) |reuse| {
        var f: Fixture = undefined;
        f.init(uplo, reuse);
        const before = f.matrix;
        const result = try A.solve(&owner, f.args());
        try std.testing.expectEqual(@as(i32, 0), result.info);
        try verify(&f, result.found.?);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&f.matrix));
        try std.testing.expect(result.native.?.published >= 2);
        try std.testing.expect(f.fpm[59] > 0);
        try std.testing.expectEqual(@as(i32, 4), f.fpm[1]);
        for (f.fpm[64..]) |v| try std.testing.expectEqual(@as(i32, 123456789), v);
        for (f.vectors[12..]) |v| try std.testing.expectEqualDeep(z(77, -33), v);
        for (f.values[3..]) |v| try std.testing.expectEqual(@as(f64, 91), v);
    };
}
test "FEAST typed LP64 trace stopping finite budget capacity warning and empty interval" {
    var owner = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    var f: Fixture = undefined;
    f.init('F', true);
    f.fpm[5] = 0;
    const trace = try A.solve(&owner, f.args());
    try std.testing.expectEqual(@as(i32, 0), trace.info);
    try std.testing.expect(trace.loop.? >= 1);
    try std.testing.expect(trace.epsout.? < 1e-12);
    try verify(&f, trace.found.?);
    f.init('F', false);
    f.fpm[3] = 0;
    f.fpm[45] = 1;
    const limited = try A.solve(&owner, f.args());
    try std.testing.expectEqual(@as(i32, 2), limited.info);
    try std.testing.expectEqual(@as(usize, 0), limited.loop.?);
    try std.testing.expect(limited.native.?.published > 0);
    f.init('F', false);
    f.fpm[45] = 0;
    const zero_inner = try A.solve(&owner, f.args());
    try std.testing.expectEqual(@as(i32, 0), zero_inner.info);
    try verify(&f, zero_inner.found.?);
    f.init('F', true);
    f.capacity = 2;
    const capacity = try A.solve(&owner, f.args());
    try std.testing.expectEqual(@as(i32, 3), capacity.info);
    try std.testing.expectEqual(@as(usize, 2), capacity.found.?);
    // A filled M0=N subspace is complete, including with no refinement budget.
    // Check both stopping policies and fresh independently evaluated residuals.
    for ([_]bool{ false, true }) |trace_stopping| {
        f.init('F', false);
        f.capacity = 4;
        f.fpm[5] = if (trace_stopping) 0 else 1;
        if (!trace_stopping) f.fpm[3] = 0;
        var full_args = f.args();
        full_args.lower = -10;
        full_args.upper = 10;
        const full = try A.solve(&owner, full_args);
        try std.testing.expectEqual(@as(i32, 0), full.info);
        try std.testing.expectEqual(@as(usize, 4), full.found.?);
        try std.testing.expectEqual(@as(usize, if (trace_stopping) 1 else 0), full.loop.?);
        for ([_]f64{ 1, 3, 5, 8 }, 0..) |lambda, column| {
            try std.testing.expectApproxEqAbs(lambda, f.values[column], 3e-11);
            const x = f.vectors[column * 4 ..][0..4];
            const ax = [_]T{ S.add(T, S.mul(T, z(2, 0), x[0]), S.mul(T, z(0, 1), x[1])), S.add(T, S.mul(T, z(0, -1), x[0]), S.mul(T, z(2, 0), x[1])), S.mul(T, z(5, 0), x[2]), S.mul(T, z(8, 0), x[3]) };
            var norm: f128 = 0;
            var error_squared: f128 = 0;
            for (x, ax) |a, b| {
                norm += @as(f128, a.re) * a.re + @as(f128, a.im) * a.im;
                const re = @as(f128, b.re) - @as(f128, lambda) * a.re;
                const im = @as(f128, b.im) - @as(f128, lambda) * a.im;
                error_squared += re * re + im * im;
            }
            try std.testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(norm)), 3e-12);
            try std.testing.expect(@sqrt(error_squared) / 10 <= 1e-12);
            try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@sqrt(error_squared) / 10)), f.residuals[column], 2e-15);
        }
        try std.testing.expectEqual(@as(f64, 91), f.values[4]);
        try std.testing.expectEqual(@as(f64, 92), f.residuals[4]);
    }
    f.init('F', false);
    f.fpm[3] = 2;
    var q = f.args();
    q.lower = 9;
    q.upper = 10;
    const empty = try A.solve(&owner, q);
    try std.testing.expectEqual(@as(i32, 1), empty.info);
    try std.testing.expectEqual(@as(usize, 0), empty.found.?);
}
test "FEAST typed LP64 invalid CSR finite values capacity and byte aliases are transactional" {
    var owner = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    defer close(&owner) catch unreachable;
    var f: Fixture = undefined;
    f.init('F', false);
    const before = f;
    f.rows[0] = 0;
    try std.testing.expectEqual(@as(i32, -104), (try A.solve(&owner, f.args())).info);
    f.rows = before.rows;
    f.columns[0] = 5;
    try std.testing.expectEqual(@as(i32, -105), (try A.solve(&owner, f.args())).info);
    f.columns = before.columns;
    f.matrix[0].re = std.math.nan(f64);
    try std.testing.expectEqual(@as(i32, -103), (try A.solve(&owner, f.args())).info);
    f.matrix = before.matrix;
    var q = f.args();
    q.vectors = f.vectors[0..11];
    try std.testing.expectError(error.BufferTooSmall, A.solve(&owner, q));
    q = f.args();
    q.residuals = q.eigenvalues;
    try std.testing.expectError(error.AliasingNotAllowed, A.solve(&owner, q));
    try std.testing.expectEqualDeep(before, f);
}
test "FEAST typed LP64 seven scratch allocation rollback and captured closing operation" {
    for (0..7) |budget| {
        var allocations = E.test_support.Allocations{ .budget = budget };
        var owner = E.context.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var f: Fixture = undefined;
        f.init('F', false);
        const before = f;
        try std.testing.expectError(error.OutOfMemory, A.solve(&owner, f.args()));
        try std.testing.expectEqualDeep(before, f);
        try std.testing.expectEqual(budget, allocations.stats().accepted);
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    }
    var owner = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    var op = try owner.begin();
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.busy, owner.tryProgressClose());
    var f: Fixture = undefined;
    f.init('F', false);
    const result = try A.solveAdmitted(&op, f.args());
    try std.testing.expectEqual(@as(i32, 0), result.info);
    try verify(&f, result.found.?);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try std.testing.expectEqual(E.context.CloseResult.closed, owner.tryProgressClose());
}
