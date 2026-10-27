// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const B = @import("../sparse/operator.zig");
const S = @import("../blas/core/shared/scalar.zig");
const P = @import("../sparse/pattern.zig");
const Product = @import("../sparse/core/product.zig");
const E = @import("zynum-execution");
fn z(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn close(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}
fn near(comptime T: type, expected: T, actual: T) !void {
    const tolerance: S.Real(T) = if (S.Real(T) == f32) 3e-5 else 1e-13;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}
test "Pattern operator mixed indices two bases CSR CSC half Hermitian transforms and both panels" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        var op = try owner.begin();
        defer op.end();
        for ([_]P.IndexBase{ .zero, .one }) |base| for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]Product.StoredTriangle{ .lower, .upper }) |triangle| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |transform| {
            const first: i64 = if (base == .zero) 0 else 1;
            const long_first = (axis == .row and triangle == .upper) or (axis == .column and triangle == .lower);
            const offsets = [_]i64{ first, first + @as(i64, if (long_first) 2 else 1), first + 3 };
            const indices = [_]i32{ @intCast(first), @intCast(first + @as(i64, if (long_first) 1 else 0)), @intCast(first + 1) };
            const values = [_]T{ z(T, 2, std.math.nan(S.Real(T))), z(T, 0, if (triangle == .lower) -1 else 1), z(T, 3, std.math.nan(S.Real(T))) };
            var backend = B.PatternBackend(T, i64, i32){ .operand = .{ .pattern = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &indices, .row_count = 2, .column_count = 2, .index_base = base, .major_axis = axis }, .values = &values, .operation = transform }, .semantics = .{ .hermitian = .{ .triangle = triangle, .diagonal = .stored } } };
            const action = backend.borrow();
            const x = [_]T{ z(T, 1, 2), z(T, 3, -1) };
            const expected = [_]T{ z(T, if (transform == .transposed) 1 else 3, if (transform == .transposed) 1 else 7), z(T, if (transform == .transposed) 7 else 11, if (transform == .transposed) -2 else -4) };
            const before = [_]T{ z(T, 4, -3), z(T, -1, 5) };
            var y = before;
            try action.applyScaledAdmitted(&op, z(T, 2, 0), &x, z(T, -1, 0), &y, &.{});
            for (expected, y, before) |a, b, c| try near(T, S.sub(T, S.mul(T, z(T, 2, 0), a), c), b);
            const input_panel = [_]T{ x[0], S.mul(T, z(T, 2, 0), x[0]), z(T, std.math.nan(S.Real(T)), 0), x[1], S.mul(T, z(T, 2, 0), x[1]), z(T, std.math.nan(S.Real(T)), 0) };
            var output: [6]T = @splat(z(T, 77, -33));
            try action.applyPanelAdmitted(&op, S.one(T), .{ .values = &input_panel, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .row_major } }, S.zero(T), .{ .values = &output, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .column_major } }, &.{});
            for (0..2) |column| for (0..2) |row| {
                try near(T, S.mul(T, z(T, @floatFromInt(column + 1), 0), expected[row]), output[row + column * 3]);
            };
            try std.testing.expectEqualDeep(z(T, 77, -33), output[2]);
            try std.testing.expectEqualDeep(z(T, 77, -33), output[5]);
        };
    }
}
test "Pattern operator revalidates metadata byte aliases and borrowed closing operation" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var op = try owner.begin();
    var rows = [_]i64{ 0, 1, 2, 91 };
    var columns = [_]i32{ 0, 1 };
    const values = [_]T{ z(T, 2, 0), z(T, 3, 0) };
    var backend = B.PatternBackend(T, i64, i32){ .operand = .{ .pattern = .{ .ranges = .{ .contiguous = rows[0..3] }, .ordinals = &columns, .row_count = 2, .column_count = 2 }, .values = &values } };
    const action = backend.borrow();
    const x = [_]T{ z(T, 1, 2), z(T, 3, -1) };
    var y: [2]T = @splat(z(T, 77, -33));
    columns[0] = 2;
    try std.testing.expectError(error.InvalidOrdinal, action.applyScaledAdmitted(&op, S.zero(T), &x, S.zero(T), &y, &.{}));
    columns[0] = 0;
    rows[2] = 3;
    try std.testing.expectError(error.InvalidOffsets, action.applyAdmitted(&op, &x, &y, &.{}));
    rows[2] = 2;
    for (y) |v| try std.testing.expectEqualDeep(z(T, 77, -33), v);
    const raw: [*]T = @ptrCast(&rows);
    const before = rows;
    try std.testing.expectError(error.AliasingNotAllowed, action.applyAdmitted(&op, &x, raw[0..2], &.{}));
    try std.testing.expectEqualDeep(before, rows);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.busy, fixtureProgressClose(owner));
    try action.applyAdmitted(&op, &x, &y, &.{});
    try near(T, z(T, 2, 4), y[0]);
    try near(T, z(T, 9, -3), y[1]);
    op.end();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
