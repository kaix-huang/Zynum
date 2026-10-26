// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Shared finite GELSD convergence contract; no vendor-runtime equivalence claim.
const std = @import("std");
const Core = @import("zynum-core");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const N = Core.internal.numerics;
const S = N.Scalar;
fn raw(comptime T: type, a: []T) ?[*]align(1) T {
    return a.ptr;
}
test "GELSD shared backend returns the actual two unfinished bidiagonal edges" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |C| {
        const R = S.Real(C);
        var rig: h.Rig = .{};
        rig.init();
        defer rig.done() catch @panic("contract probe lifecycle");
        // A real upper bidiagonal matrix embedded in each complex precision.
        // The two independent, exactly nonzero superdiagonal entries require
        // two unfinished edges when the actual QR iteration budget is zero.
        const original = [_]C{
            .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
            .{ .re = 1, .im = 0 }, .{ .re = 3, .im = 0 }, .{ .re = 0, .im = 0 },
            .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 },
        };
        var a = original;
        var b = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = 0 } };
        var singular = [_]R{ -7, -8, -9 };
        var rank = [_]i32{-10};
        var rw = [_]R{-11};
        var iw = [_]i32{-12};
        const p = try owner.complex_gelsd_workspace.Planner(owner.Provider).make(C, i32, 3, 3, 1);
        const work = try std.testing.allocator.alloc(C, @intCast(p.work));
        defer std.testing.allocator.free(work);
        const legacy = try owner.RawComplexGelsdColumnWork.callManaged(C, i32, &rig.context, &rig.services, .{
            .m = 3,
            .n = 3,
            .nrhs = 1,
            .lda = 3,
            .ldb = 3,
            .a = raw(C, &a),
            .b = raw(C, &b),
            .singular = raw(R, &singular),
            .rank = raw(i32, &rank),
            .rcond = -1,
            .work = raw(C, work),
            .lwork = p.work,
            .rwork = raw(R, &rw),
            .iwork = raw(i32, &iw),
        }, .{ .max_sweeps = 0 });
        const token = switch (legacy) {
            .lapacke_info => |v| v,
            else => return error.TestUnexpectedResult,
        };
        try std.testing.expectEqual(@as(i32, 2), token);
        try std.testing.expectEqualSlices(C, &original, &a);
        for (b, 0..) |z, i| try std.testing.expect(z.re == @as(R, @floatFromInt(i + 1)) and z.im == 0);
        for (singular, 0..) |s, i| try std.testing.expectEqual(-@as(R, @floatFromInt(i + 7)), s);
        try std.testing.expectEqual(@as(i32, -10), rank[0]);
        var x: [3]C = @splat(.{ .re = -13, .im = 0 });
        var sigma: [3]R = @splat(-14);
        var v: [9]C = @splat(.{ .re = -15, .im = 0 });
        const rhs = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = 0 } };
        const result = try N.SvdLeastSquares.solveManaged(&rig.context, C, .{ .values = &original, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .{ .values = &rhs, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, .{ .solution = .{ .values = &x, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, .singular = &sigma, .right_vectors = .{ .values = &v, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .residual = .{ .values = &.{}, .row_count = 0, .column_count = 1, .leading_dimension = 0 } }, .{ .relative = std.math.floatEps(R) / 2, .absolute = 0 }, .{ .max_steps = 0 });
        try std.testing.expectEqual(N.SvdLeastSquares.Status.no_convergence, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.unconverged);
        for (x) |z| try std.testing.expect(z.re == -13 and z.im == 0);
        for (sigma) |z| try std.testing.expectEqual(@as(R, -14), z);
        for (v) |z| try std.testing.expect(z.re == -15 and z.im == 0);
        std.debug.print("GELSD_INFO_PROBE scalar={s} actual_GELSD_INFO={d} retained_bidiagonal_edges={d} outputs_preserved=true\n", .{ @typeName(C), token, result.unconverged });
    }
}
