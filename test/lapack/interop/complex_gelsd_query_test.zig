// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const binding = @import("numerical-owner");
const P = binding.Provider;
const Q = binding.ComplexGelsdQuery;
const Raw = binding.RawComplexGelsdQuery;
const common = binding.common;
const State = struct {
    events: usize = 0,
    last: ?common.ErrorEvent = null,
    fn nan(_: *anyopaque) bool {
        return true;
    }
    fn event(pointer: *anyopaque, value: common.ErrorEvent) void {
        const self: *State = @ptrCast(@alignCast(pointer));
        self.events += 1;
        self.last = value;
    }
    fn services(self: *State) P.Services {
        return .{ .allocator = std.testing.failing_allocator, .context = self, .nan_enabled = nan, .error_event = event };
    }
};
fn success(result: anytype) !void {
    switch (result) {
        .lapacke_info => |info| try std.testing.expectEqual(@as(@TypeOf(info), 0), info),
        else => return error.TestUnexpectedResult,
    }
}
fn fault(result: anytype, wanted: anyerror) !void {
    switch (result) {
        .admission_fault => |e| try std.testing.expectEqual(wanted, e),
        else => return error.TestUnexpectedResult,
    }
}
test "complex GELSD query writes three slots with no operand access or allocation" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| inline for (.{ i32, i64 }) |I| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const service = state.services();
        var work = [_]C{ .{ .re = -7, .im = -8 }, .{ .re = 71, .im = 72 } };
        var real = [_]R{ -9, 73 };
        var integer = [_]I{ -10, 74 };
        const args: Q.Arguments(C, I) = .{ .layout = 102, .m = 5, .n = 3, .nrhs = 2, .lda = 5, .ldb = 5, .rcond = std.math.nan(R), .work = &work, .rwork = &real, .iwork = &integer };
        try success(Q.call(C, I, &service, args));
        const plan = try binding.complex_gelsd_workspace.Planner(P).make(C, I, 5, 3, 2);
        try std.testing.expectEqual(@as(R, @floatFromInt(plan.work)), work[0].re);
        try std.testing.expectEqual(@as(R, 0), work[0].im);
        try std.testing.expectEqual(@as(R, 1), real[0]);
        try std.testing.expectEqual(@as(I, 1), integer[0]);
        try std.testing.expectEqualDeep(C{ .re = 71, .im = 72 }, work[1]);
        try std.testing.expectEqual(@as(R, 73), real[1]);
        try std.testing.expectEqual(@as(I, 74), integer[1]);
        try std.testing.expectEqual(@as(usize, 0), state.events);
        var raw_args: Raw.Arguments(C, I) = .{ .layout = 102, .m = 5, .n = 3, .nrhs = 2, .lda = 5, .ldb = 5, .rcond = std.math.nan(R), .work = &work, .rwork = &real, .iwork = &integer };
        // Raw unused operands retain NULL too; no foreign buffer provenance is inferred.
        raw_args.a = null;
        try success(Raw.call(C, I, &service, raw_args));
    };
}
test "complex GELSD query missing final slot leaves earlier metadata untouched" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const service = state.services();
        var work = [_]C{.{ .re = 31, .im = 32 }};
        var real = [_]R{33};
        const args: Q.Arguments(C, i32) = .{ .layout = 102, .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .work = &work, .rwork = &real, .iwork = null };
        try fault(Q.call(C, i32, &service, args), error.MissingBuffer);
        try std.testing.expectEqualDeep(C{ .re = 31, .im = 32 }, work[0]);
        try std.testing.expectEqual(@as(R, 33), real[0]);
    }
}
test "complex GELSD query mixed output aliases fail before any metadata write" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const service = state.services();
        var work = [_]C{.{ .re = 41, .im = 42 }};
        var integer = [_]i32{43};
        const real = @as([*]R, @ptrCast(&work))[0..1];
        const args: Q.Arguments(C, i32) = .{ .layout = 102, .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .work = &work, .rwork = real, .iwork = &integer };
        try fault(Q.call(C, i32, &service, args), error.AliasingNotAllowed);
        try std.testing.expectEqualDeep(C{ .re = 41, .im = 42 }, work[0]);
        try std.testing.expectEqual(@as(i32, 43), integer[0]);
    }
}
test "complex GELSD query preserves scalar error precedence before outputs" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| {
        var state: State = .{};
        const service = state.services();
        var args: Q.Arguments(C, i32) = .{ .layout = 101, .m = -1, .n = 3, .nrhs = 2, .lda = 2, .ldb = 0, .work = null, .rwork = null, .iwork = null };
        try std.testing.expectEqual(@as(i32, -6), Q.call(C, i32, &service, args).lapacke_info);
        try std.testing.expectEqual(common.HookKind.lapacke, state.last.?.kind);
        args.lda = 3;
        try std.testing.expectEqual(@as(i32, -8), Q.call(C, i32, &service, args).lapacke_info);
        args.ldb = 2;
        try std.testing.expectEqual(@as(i32, -2), Q.call(C, i32, &service, args).lapacke_info);
        try std.testing.expectEqual(common.HookKind.fortran_returning_test, state.last.?.kind);
        try std.testing.expectEqual(@as(i32, 1), state.last.?.argument);
        args.layout = -9;
        try std.testing.expectEqual(@as(i32, -1), Q.call(C, i32, &service, args).lapacke_info);
    }
}
test "complex GELSD query empty wide integer extents require only three slots" {
    inline for (.{ P.Scalar.ComplexF32, P.Scalar.ComplexF64 }) |C| {
        const R = P.Scalar.Real(C);
        var state: State = .{};
        const service = state.services();
        var work = [_]C{.{ .re = -1, .im = -2 }};
        var real = [_]R{-3};
        var integer = [_]i64{-4};
        const large: i64 = @as(i64, std.math.maxInt(i32)) + 17;
        const args: Q.Arguments(C, i64) = .{ .layout = 102, .m = 0, .n = large, .nrhs = large, .lda = 1, .ldb = large, .work = &work, .rwork = &real, .iwork = &integer };
        try success(Q.call(C, i64, &service, args));
        try std.testing.expectEqualDeep(C{ .re = 1, .im = 0 }, work[0]);
        try std.testing.expectEqual(@as(R, 1), real[0]);
        try std.testing.expectEqual(@as(i64, 1), integer[0]);
    }
}
