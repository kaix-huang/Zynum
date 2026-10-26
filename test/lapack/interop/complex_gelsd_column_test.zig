// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const binding = @import("numerical-owner");
const P = binding.Provider;
const E = binding.ComplexGelsdColumnWork;
const State = struct {
    fn nan(_: *anyopaque) bool {
        return true;
    }
    fn event(_: *anyopaque, _: binding.common.ErrorEvent) void {}
    fn services(self: *State, allocator: std.mem.Allocator) P.Services {
        return .{ .allocator = allocator, .context = self, .nan_enabled = nan, .error_event = event };
    }
};
fn success(result: anytype) !void {
    switch (result) {
        .lapacke_info => |i| try std.testing.expectEqual(@as(@TypeOf(i), 0), i),
        else => return error.TestUnexpectedResult,
    }
}
test "complex GELSD column finite tall minimum and recommended workspace give same solution" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| inline for (.{ i32, i64 }) |I| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const services = state.services(std.testing.allocator);
        const plan = try binding.complex_gelsd_workspace.Planner(P).make(C, I, 3, 2, 1);
        const minimum = try binding.complex_gelsd_policy.minimumWork(I, 3, 2, 1);
        inline for (.{ false, true }) |short| {
            const count: usize = @intCast(if (short) minimum else plan.work);
            const work = try std.testing.allocator.alloc(C, count + 1);
            defer std.testing.allocator.free(work);
            @memset(work, .{ .re = 57, .im = 58 });
            var real = [_]R{ -1, 59 };
            var integer = [_]I{ -2, 60 };
            var a = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 }, .{ .re = 1, .im = 0 } };
            const before = a;
            var b = [_]C{ .{ .re = 1, .im = 2 }, .{ .re = -4, .im = 2 }, .{ .re = -2, .im = 1 }, .{ .re = 61, .im = 62 } };
            var s = [_]R{ -3, -4, 63 };
            var rank: I = -5;
            const args: E.Arguments(C, I) = .{ .m = 3, .n = 2, .nrhs = 1, .lda = 3, .ldb = 3, .a = &a, .b = &b, .singular = &s, .rank = @as([*]I, @ptrCast(&rank))[0..1], .rcond = -1, .work = work[0..count], .lwork = @intCast(count), .rwork = &real, .iwork = &integer };
            try success(E.call(C, I, &services, args, .{}));
            const tol: R = if (R == f32) 2e-5 else 2e-12;
            try std.testing.expectApproxEqAbs(@as(R, 1), b[0].re, tol);
            try std.testing.expectApproxEqAbs(@as(R, 2), b[0].im, tol);
            try std.testing.expectApproxEqAbs(@as(R, -2), b[1].re, tol);
            try std.testing.expectApproxEqAbs(@as(R, 1), b[1].im, tol);
            try std.testing.expectApproxEqAbs(@as(R, 0), b[2].re * b[2].re + b[2].im * b[2].im, tol);
            try std.testing.expectEqual(@as(I, 2), rank);
            try std.testing.expectEqualDeep(before, a);
            try std.testing.expectEqualDeep(C{ .re = 61, .im = 62 }, b[3]);
            try std.testing.expectEqual(@as(R, 63), s[2]);
            try std.testing.expectEqualDeep(C{ .re = 57, .im = 58 }, work[count]);
            try std.testing.expectEqual(@as(R, 59), real[1]);
            try std.testing.expectEqual(@as(I, 60), integer[1]);
        }
    };
}
test "complex GELSD column empty dimension nonreads and all zero matrix clear different spans" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const services = state.services(std.testing.allocator);
        var work = [_]C{.{ .re = 9, .im = 9 }};
        var real = [_]R{9};
        var integer = [_]i64{9};
        var rank = [_]i64{9};
        const big: i64 = @as(i64, std.math.maxInt(i32)) + 17;
        const empty: E.Arguments(C, i64) = .{ .m = 0, .n = big, .nrhs = big, .lda = 1, .ldb = big, .a = null, .b = null, .singular = null, .rank = &rank, .rcond = -1, .work = &work, .lwork = 1, .rwork = &real, .iwork = &integer };
        try success(E.call(C, i64, &services, empty, .{}));
        try std.testing.expectEqual(@as(i64, 0), rank[0]);
        const plan = try binding.complex_gelsd_workspace.Planner(P).make(C, i64, 3, 1, 1);
        const full = try std.testing.allocator.alloc(C, @intCast(plan.work));
        defer std.testing.allocator.free(full);
        var a: [3]C = @splat(.{ .re = 0, .im = 0 });
        var b: [4]C = @splat(.{ .re = 7, .im = 8 });
        var s = [_]R{9};
        rank[0] = 9;
        const zero: E.Arguments(C, i64) = .{ .m = 3, .n = 1, .nrhs = 1, .lda = 3, .ldb = 3, .a = &a, .b = &b, .singular = &s, .rank = &rank, .rcond = -1, .work = full, .lwork = plan.work, .rwork = &real, .iwork = &integer };
        try success(E.call(C, i64, &services, zero, .{}));
        for (b[0..3]) |z| try std.testing.expectEqualDeep(C{ .re = 0, .im = 0 }, z);
        try std.testing.expectEqualDeep(C{ .re = 7, .im = 8 }, b[3]);
        try std.testing.expectEqual(@as(R, 0), s[0]);
    }
}
