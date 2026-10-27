// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const lq = Core.internal.numerics.LQ;
const views = Core.lapack;
const scalar = Core.internal.numerics.Scalar;
const types = Core.lapack;
const execution = Core.internal.numerics.Execution;
const scalars = .{ f32, f64, types.ComplexF32, types.ComplexF64 };
const W = struct { re: f64 = 0, im: f64 = 0 };
fn add(a: W, b: W) W {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn sub(a: W, b: W) W {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn mul(a: W, b: W) W {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: W) W {
    return .{ .re = a.re, .im = -a.im };
}
fn val(comptime T: type, re: f64, im: f64) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn wide(comptime T: type, a: T) W {
    return if (comptime scalar.isComplex(T)) .{ .re = a.re, .im = a.im } else .{ .re = a };
}
fn mat(comptime T: type, a: []T, m: usize, n: usize, ld: usize) views.Matrix(T) {
    return .{ .values = a, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}
fn same(comptime T: type, a: []const T, b: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn near(comptime T: type, a: W, b: T) !void {
    const actual = wide(T, b);
    const epsilon: f64 = if (T == f32 or T == types.ComplexF32) 4e-5 else 5e-12;
    const tolerance = epsilon * @max(1, @abs(a.re) + @abs(a.im));
    try std.testing.expectApproxEqAbs(a.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(a.im, actual.im, tolerance);
}
fn padding(comptime T: type, a: []const T, rows: usize, columns: usize, ld: usize) !void {
    for (0..columns) |j| for (rows..ld) |i| try near(T, wide(T, val(T, -987, 65)), a[i + j * ld]);
}

// Independent dense Householder definition, never the candidate's Q action.
// Each left multiplication builds H_i^H * previous, hence H_k^H...H_1^H.
fn explicitQ(comptime T: type, factor: []const T, tau: []const T, m: usize, n: usize, ld: usize, q: []W) void {
    for (0..n) |j| for (0..n) |i| {
        q[i + j * n] = .{ .re = if (i == j) 1 else 0 };
    };
    for (0..@min(m, n)) |step| {
        var v: [8]W = @splat(.{});
        v[step] = .{ .re = 1 };
        for (step + 1..n) |j| v[j] = conj(wide(T, factor[step + j * ld]));
        const t = conj(wide(T, tau[step]));
        for (0..n) |column| {
            var dot: W = .{};
            for (step..n) |i| dot = add(dot, mul(conj(v[i]), q[i + column * n]));
            for (step..n) |i| q[i + column * n] = sub(q[i + column * n], mul(mul(t, v[i]), dot));
        }
    }
}
fn qElement(q: []const W, n: usize, i: usize, j: usize, op: lq.QOperation) W {
    return switch (op) {
        .normal => q[i + j * n],
        .transposed => q[j + i * n],
        .conjugate_transposed => conj(q[j + i * n]),
    };
}
fn shapeCase(comptime T: type, comptime m: usize, comptime n: usize) !void {
    const ld: usize = m + 2;
    const k: usize = @min(m, n);
    const sentinel = val(T, -987, 65);
    var a: [ld * n + 3]T = @splat(sentinel);
    for (0..n) |j| for (0..m) |i| {
        a[i + j * ld] = val(T, (@as(f64, @floatFromInt((i * 7 + j * 5 + 3) % 13)) - 6) / 4, (@as(f64, @floatFromInt((i * 5 + j * 2 + 4) % 11)) - 5) / 7);
    };
    const original = a;
    var tau: [k + 2]T = @splat(sentinel);
    var factor_work: [m * n + 3]T = @splat(sentinel);
    try std.testing.expectEqual(@as(usize, 0), try lq.factorize(T, mat(T, &a, m, n, ld), &tau, &factor_work));
    try padding(T, &a, m, n, ld);
    try same(T, original[ld * n ..], a[ld * n ..]);
    try near(T, wide(T, sentinel), tau[k]);
    try near(T, wide(T, sentinel), tau[k + 1]);
    for (factor_work[m * n ..]) |z| try near(T, wide(T, sentinel), z);
    var q: [n * n]W = undefined;
    explicitQ(T, &a, &tau, m, n, ld, &q);
    for (0..n) |j| for (0..m) |i| {
        var z: W = .{};
        for (0..@min(i + 1, n)) |r| z = add(z, mul(wide(T, a[i + r * ld]), q[r + j * n]));
        try near(T, z, original[i + j * ld]);
    };
    for (0..n) |j| for (0..n) |i| {
        var z: W = .{};
        for (0..n) |r| z = add(z, mul(conj(q[r + i * n]), q[r + j * n]));
        const expected: T = val(T, if (i == j) 1 else 0, 0);
        try near(T, z, expected);
    };
    const before_a = a;
    const before_tau = tau;
    inline for (.{ lq.QSide.left, lq.QSide.right }) |side| inline for (.{ lq.QOperation.normal, lq.QOperation.transposed, lq.QOperation.conjugate_transposed }) |op| {
        const rows: usize = if (side == .left) n else 3;
        const columns: usize = if (side == .left) 3 else n;
        const bld: usize = rows + 2;
        var b: [bld * columns + 2]T = @splat(sentinel);
        for (0..columns) |j| for (0..rows) |i| {
            b[i + j * bld] = val(T, @as(f64, @floatFromInt(1 + i + 3 * j)) / 7, @as(f64, @floatFromInt(2 + i + j)) / 11);
        };
        const original_b = b;
        var work: [m * n + k + 3]T = @splat(sentinel);
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQ(T, mat(T, &a, m, n, ld).asConst(), &tau, mat(T, &b, rows, columns, bld), side, op, &work));
        for (0..columns) |j| for (0..rows) |i| {
            var expected: W = .{};
            for (0..n) |r| expected = add(expected, if (side == .left) mul(qElement(&q, n, i, r, op), wide(T, original_b[r + j * bld])) else mul(wide(T, original_b[i + r * bld]), qElement(&q, n, r, j, op)));
            try near(T, expected, b[i + j * bld]);
        };
        try padding(T, &b, rows, columns, bld);
        try same(T, original_b[bld * columns ..], b[bld * columns ..]);
        try same(T, &before_a, &a);
        try same(T, &before_tau, &tau);
        for (work[m * n + k ..]) |z| try near(T, wide(T, sentinel), z);
    };
}

test "LQ four scalars tall wide square independent factor reconstruction and left right N T H actions" {
    inline for (scalars) |T| {
        try shapeCase(T, 2, 4);
        try shapeCase(T, 4, 2);
        try shapeCase(T, 3, 3);
        try shapeCase(T, 1, 4);
    }
}

test "LQ factor shape overlap capacity finite errors preserve A tau workspace and tails" {
    inline for (scalars) |T| {
        var a = [_]T{ val(T, 1, 1), val(T, 2, 0), val(T, -987, 65), val(T, 3, -1), val(T, 4, 2), val(T, -987, 65), val(T, 5, 0), val(T, 6, 1), val(T, -987, 65) };
        var tau: [3]T = @splat(val(T, -987, 65));
        var work: [8]T = @splat(val(T, -987, 65));
        const original = a;
        const ot = tau;
        const ow = work;
        try std.testing.expectError(error.BufferTooSmall, lq.factorize(T, mat(T, &a, 2, 3, 3), &tau, work[0..5]));
        try std.testing.expectError(error.AliasingNotAllowed, lq.factorize(T, mat(T, &a, 2, 3, 3), &tau, a[0..6]));
        try std.testing.expectError(error.BufferTooSmall, lq.factorize(T, mat(T, &a, 2, 3, 3), tau[0..1], &work));
        try same(T, &original, &a);
        try same(T, &ot, &tau);
        try same(T, &ow, &work);
        a[1] = val(T, std.math.nan(f64), 0);
        const poisoned = a;
        try std.testing.expectError(error.NonFiniteInput, lq.factorize(T, mat(T, &a, 2, 3, 3), &tau, &work));
        try same(T, &poisoned, &a);
        try same(T, &ot, &tau);
        try same(T, &ow, &work);
        var empty = mat(T, &a, 0, 3, 0);
        empty.leading_dimension = -1;
        try std.testing.expectError(error.InvalidLeadingDimension, lq.factorize(T, empty, &tau, &work));
        try std.testing.expectError(error.InvalidLength, lq.factorWorkspaceLength(T, std.math.maxInt(usize), 2));
        try std.testing.expectError(error.InvalidLength, lq.factorWorkspaceLength(T, std.math.maxInt(usize) / @sizeOf(T) + 1, 1));
    }
}

test "LQ Q action ignores L and unused tau and rejects invalid reflectors before workspace or RHS mutation" {
    inline for (scalars) |T| {
        var a = [_]T{ val(T, 1, 1), val(T, 2, 0), val(T, 3, -1), val(T, 4, 2), val(T, 5, 0), val(T, 6, 1) };
        var tau: [3]T = @splat(val(T, std.math.nan(f64), 0));
        var fw: [6]T = undefined;
        try std.testing.expectEqual(@as(usize, 0), try lq.factorize(T, mat(T, &a, 2, 3, 2), &tau, &fw));
        var q: [9]W = undefined;
        explicitQ(T, &a, &tau, 2, 3, 2, &q);
        for (0..3) |j| for (j..2) |i| {
            a[i + j * 2] = val(T, std.math.nan(f64), 0);
        };
        var b = [_]T{ val(T, 1, 0), val(T, 0, 0), val(T, 0, 0) };
        var work: [9]T = @splat(val(T, -987, 65));
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQ(T, mat(T, &a, 2, 3, 2).asConst(), &tau, mat(T, &b, 3, 1, 3), .left, .normal, &work));
        for (0..3) |i| try near(T, q[i], b[i]);
        const ob = b;
        const ow = work;
        const valid_tau = tau[0];
        tau[0] = val(T, 3, 0);
        try std.testing.expectError(error.InvalidReflector, lq.applyQ(T, mat(T, &a, 2, 3, 2).asConst(), &tau, mat(T, &b, 3, 1, 3), .left, .normal, &work));
        try same(T, &ob, &b);
        try same(T, &ow, &work);
        tau[0] = valid_tau;
        a[2] = val(T, std.math.nan(f64), 0);
        try std.testing.expectError(error.NonFiniteInput, lq.applyQ(T, mat(T, &a, 2, 3, 2).asConst(), &tau, mat(T, &b, 3, 1, 3), .left, .normal, &work));
        try same(T, &ob, &b);
        try same(T, &ow, &work);
    }
}

test "LQ zero extents use no workspace and represent identity without touching unused storage" {
    inline for (scalars) |T| {
        var empty: [0]T = .{};
        var a = [_]T{val(T, -987, 65)};
        var tau = [_]T{val(T, -987, 65)};
        try std.testing.expectEqual(@as(usize, 0), try lq.factorize(T, mat(T, &a, 0, 4, 0), &tau, &empty));
        var b = [_]T{ val(T, 1, 2), val(T, 3, 4), val(T, 5, 6), val(T, 7, 8) };
        const ob = b;
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQ(T, mat(T, &a, 0, 4, 0).asConst(), &tau, mat(T, &b, 4, 1, 4), .left, .normal, &empty));
        try same(T, &ob, &b);
        try std.testing.expectEqual(@as(usize, 0), try lq.factorize(T, mat(T, &a, 3, 0, 0), &tau, &empty));
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQ(T, mat(T, &a, 3, 0, 0).asConst(), &tau, mat(T, &empty, 0, 2, 0), .left, .conjugate_transposed, &empty));
        try near(T, wide(T, val(T, -987, 65)), a[0]);
        try near(T, wide(T, val(T, -987, 65)), tau[0]);
    }
}

test "LQ admitted factor and Q action finish while closing with one operation and no resource pins" {
    inline for (scalars) |T| {
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        var op = try context.begin();
        _ = context.requestClose();
        var a = [_]T{ val(T, 1, 1), val(T, 2, 0), val(T, 3, 0) };
        var tau: [1]T = undefined;
        var fw: [3]T = undefined;
        try std.testing.expectEqual(@as(usize, 0), try lq.factorizeAdmitted(&op, T, mat(T, &a, 1, 3, 1), &tau, &fw));
        var b = [_]T{ val(T, 1, 0), val(T, 0, 0), val(T, 0, 0) };
        var aw: [4]T = undefined;
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQAdmitted(&op, T, mat(T, &a, 1, 3, 1).asConst(), &tau, mat(T, &b, 3, 1, 3), .left, .normal, &aw));
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        try std.testing.expectError(error.Closing, context.begin());
        op.end();
        try std.testing.expectEqual(execution.context.CloseResult.closed, fixtureProgressClose(context));
    }
}

test "LQ checked arithmetic range status and IEEE projection reuse the existing QR policy" {
    inline for (scalars) |T| {
        const R = scalar.Real(T);
        const huge = std.math.floatMax(R);
        var a = [_]T{ val(T, huge, 0), val(T, huge, 0) };
        var tau = [_]T{val(T, -987, 65)};
        var work: [2]T = undefined;
        try std.testing.expectEqual(@as(usize, 1), try lq.factorize(T, mat(T, &a, 1, 2, 1), &tau, &work));
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        var op = try context.begin();
        a = .{ val(T, std.math.nan(f64), 0), val(T, 1, 0) };
        try std.testing.expectEqual(@as(usize, 0), try lq.factorizeWithContractAdmitted(&op, T, mat(T, &a, 1, 2, 1), &tau, &work, .lapack_ieee_v1));
        var rhs = [_]T{ val(T, 1, 0), val(T, 0, 0) };
        var apply_work: [3]T = undefined;
        try std.testing.expectEqual(@as(usize, 0), try lq.applyQWithContractAdmitted(&op, T, mat(T, &a, 1, 2, 1).asConst(), &tau, mat(T, &rhs, 2, 1, 2), .left, .normal, &apply_work, .lapack_ieee_v1));
        op.end();
        _ = context.requestClose();
        try std.testing.expectEqual(execution.context.CloseResult.closed, fixtureProgressClose(context));
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
