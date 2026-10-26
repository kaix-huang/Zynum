// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const policy = @import("numerical-owner").gelsd_policy;
test "GELSD policy finite cutoff matches measured reference rank boundaries" {
    inline for (.{ f32, f64 }) |T| {
        for ([_]T{ -1, 0, 1, 2 }) |rcond| {
            const cutoff = policy.effectiveRcond(T, 2, rcond);
            try std.testing.expectEqual(std.math.floatEps(T) * 0.5, cutoff);
            try std.testing.expect(std.math.floatEps(T) * 0.75 > cutoff);
            try std.testing.expectEqual(@as(T, 0), policy.effectiveRcond(T, 1, rcond));
        }
        try std.testing.expectEqual(@as(T, 0.5), policy.effectiveRcond(T, 2, 0.5));
        try std.testing.expect(std.math.isNan(policy.effectiveRcond(T, 2, std.math.nan(T))));
    }
}
test "GELSD policy scalar order distinguishes row and column layouts across widths" {
    inline for (.{ i32, i64 }) |I| {
        try std.testing.expectEqual(@as(I, -1), policy.scalarInfo(I, 0, -1, -1, -1, -1, -1));
        try std.testing.expectEqual(@as(I, -2), policy.scalarInfo(I, 102, -1, 2, 1, 0, 0));
        try std.testing.expectEqual(@as(I, -6), policy.scalarInfo(I, 101, -1, 2, 1, 0, 0));
        try std.testing.expectEqual(@as(I, -8), policy.scalarInfo(I, 101, -1, 2, 1, 2, 0));
        try std.testing.expectEqual(@as(I, -2), policy.scalarInfo(I, 101, -1, 2, 1, 2, 1));
        try std.testing.expectEqual(@as(I, -3), policy.scalarInfo(I, 102, 0, -1, -1, 0, 0));
        try std.testing.expectEqual(@as(I, -4), policy.scalarInfo(I, 102, 0, 0, -1, 0, 0));
        try std.testing.expectEqual(@as(I, -6), policy.scalarInfo(I, 102, 0, 0, 0, 0, 0));
        try std.testing.expectEqual(@as(I, -8), policy.scalarInfo(I, 102, 0, 0, 0, 1, 0));
        try std.testing.expectEqual(@as(I, 0), policy.scalarInfo(I, 101, 0, 0, 0, 0, 0));
        try std.testing.expectEqual(@as(I, 0), policy.scalarInfo(I, 102, 0, 0, 0, 1, 1));
    }
}

test "GELSD workspace packs widened storage into caller work at every alignment" {
    const owner = @import("numerical-owner");
    const P = owner.gelsd_workspace.Planner(owner.Provider);
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| {
        const p = try P.make(T, I, 3, 2, 2);
        const storage = try std.testing.allocator.alloc(T, @as(usize, @intCast(p.work)) + 4);
        defer std.testing.allocator.free(storage);
        for (0..4) |offset| {
            @memset(storage, 77);
            const work = storage[offset..][0..@intCast(p.work)];
            const buffers = try P.split(T, I, p, work);
            const W = owner.Provider.RankLeastSquares.WorkspaceScalar(T);
            try std.testing.expectEqual(@as(usize, 0), @intFromPtr(buffers.wide.ptr) % @alignOf(W));
            var singular: [2]T = undefined;
            const result = try owner.Provider.RankLeastSquares.solveWithResidual(T, .{ .values = &.{ 1, 0, 1, 0, 1, 1 }, .row_count = 3, .column_count = 2, .leading_dimension = 3 }, .{ .values = &.{ 1, 2, 4, 2, 4, 8 }, .row_count = 3, .column_count = 2, .leading_dimension = 3 }, .{ .values = buffers.solution, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &singular, .{ .values = buffers.residual, .row_count = 1, .column_count = 2, .leading_dimension = 1 }, -1, buffers.scalar, buffers.wide, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(@as(usize, 2), result.rank);
            const tol: T = if (T == f32) 2e-5 else 2e-12;
            try std.testing.expectApproxEqAbs(@as(T, 4.0 / 3.0), buffers.solution[0], tol);
            try std.testing.expectApproxEqAbs(@as(T, 1.0 / 3.0), buffers.residual[0] * buffers.residual[0], tol);
            try std.testing.expectEqual(@as(T, 77), work[0]);
            for (storage[0..offset]) |value| try std.testing.expectEqual(@as(T, 77), value);
            for (storage[offset + work.len ..]) |value| try std.testing.expectEqual(@as(T, 77), value);
            try std.testing.expectError(error.BufferTooSmall, P.split(T, I, p, work[0 .. work.len - 1]));
        }
    };
}

test "GELSD workspace query rounds upward and rejects ABI overflow" {
    const owner = @import("numerical-owner");
    const workspace = owner.gelsd_workspace;
    const P = workspace.Planner(owner.Provider);
    try std.testing.expectEqual(@as(f32, 16777218), try workspace.queryLength(f32, i32, 16777217));
    try std.testing.expectEqual(@as(f64, 9007199254740994), try workspace.queryLength(f64, i64, 9007199254740993));
    try std.testing.expectError(error.SizeOverflow, workspace.queryLength(f32, i32, std.math.maxInt(i32)));
    try std.testing.expectError(error.SizeOverflow, workspace.queryLength(f64, i64, std.math.maxInt(i64)));
    try std.testing.expectError(error.SizeOverflow, P.make(f64, i64, std.math.maxInt(usize), 1, 1));
    const empty = try P.make(f64, i64, 0, 3, 2);
    try std.testing.expectEqual(@as(i64, 1), empty.work);
    try std.testing.expectEqual(@as(i64, 1), empty.iwork);
    for ([_]usize{ 13, 14, 25, 26, 51, 52 }, [_]usize{ 0, 1, 1, 1, 1, 2 }) |k, levels| {
        const p = try P.make(f64, i64, k, k, 1);
        try std.testing.expectEqual(@as(i64, @intCast(11 * k + 3 * k * levels)), p.iwork);
    }
}

test "GELSD query writes only lengths without matrix reads scans or allocation" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        const Q = if (raw) owner.RawGelsdQuery else owner.GelsdQuery;
        for ([_]c_int{ 101, 102 }) |layout| {
            var context: support.Context = .{ .fail_index = 0 };
            const services = context.services();
            var work = [_]T{ -99, 77 };
            var iwork = [_]I{ -88, 66 };
            var rank = [_]I{55};
            const args: Q.Arguments(T, I) = .{ .layout = layout, .m = 3, .n = 2, .nrhs = 1, .lda = 3, .ldb = 3, .rcond = std.math.nan(T), .rank = if (raw) @as([*]align(1) I, @ptrCast(&rank)) else &rank, .work = if (raw) @as([*]align(1) T, @ptrCast(&work)) else &work, .iwork = if (raw) @as([*]align(1) I, @ptrCast(&iwork)) else &iwork };
            try support.expectInfo(I, 0, Q.call(T, I, &services, args));
            const plan = try owner.gelsd_workspace.Planner(owner.Provider).make(T, I, 3, 2, 1);
            try std.testing.expectEqual(@as(T, @floatFromInt(plan.work)), work[0]);
            try std.testing.expectEqual(plan.iwork, iwork[0]);
            try std.testing.expectEqual(@as(T, 77), work[1]);
            try std.testing.expectEqual(@as(I, 66), iwork[1]);
            try std.testing.expectEqual(@as(I, 55), rank[0]);
            try std.testing.expectEqual(@as(usize, 0), context.reads);
            try std.testing.expectEqual(@as(usize, 0), context.attempts);
            try std.testing.expectEqual(@as(usize, 0), context.hook_count);
        }
    };
}

test "GELSD query errors retain channel and validate both outputs before writing" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const Q = owner.GelsdQuery;
    var context: support.Context = .{};
    const services = context.services();
    var work = [_]f64{-99};
    var args: Q.Arguments(f64, i32) = .{ .layout = 102, .m = 3, .n = 2, .nrhs = 1, .lda = 3, .ldb = 3, .work = &work, .iwork = null };
    switch (Q.call(f64, i32, &services, args)) {
        .admission_fault => |err| try std.testing.expectEqual(error.MissingBuffer, err),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(f64, -99), work[0]);
    args.m = -1;
    try support.expectInfo(i32, -2, Q.call(f64, i32, &services, args));
    try std.testing.expectEqual(owner.common.HookKind.fortran_returning_test, context.hooks[0].kind);
    try std.testing.expectEqual(@as(i32, 1), context.hooks[0].argument);
    try std.testing.expectEqual(owner.common.Routine.gelsd, context.hooks[0].routine);
    args.layout = 101;
    args.lda = 0;
    try support.expectInfo(i32, -6, Q.call(f64, i32, &services, args));
    try std.testing.expectEqual(owner.common.HookKind.lapacke, context.hooks[1].kind);
    try std.testing.expectEqual(@as(i32, -6), context.hooks[1].argument);
    try std.testing.expectEqual(@as(f64, -99), work[0]);
}

test "GELSD query shares one admission and honors closing from its diagnostic" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const h = @import("managed_support.zig");
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("cleanup");
    var work: [1]f64 = undefined;
    var iwork: [1]i64 = undefined;
    var args: owner.GelsdQuery.Arguments(f64, i64) = .{ .layout = 102, .m = 3, .n = 2, .nrhs = 1, .lda = 3, .ldb = 3, .work = &work, .iwork = &iwork };
    try support.expectInfo(i64, 0, try owner.GelsdQuery.callManaged(f64, i64, &rig.context, &rig.services, args));
    rig.trigger = .hook;
    rig.action = .close_nested;
    args.m = -1;
    try support.expectInfo(i64, -2, try owner.GelsdQuery.callManaged(f64, i64, &rig.context, &rig.services, args));
    try std.testing.expectEqual(error.Closing, rig.nested_error.?);
    try std.testing.expectError(error.Closing, owner.GelsdQuery.callManaged(f64, i64, &rig.context, &rig.services, args));
    try std.testing.expectEqual(@as(usize, 1), rig.max_active);
    try std.testing.expectEqual(@as(usize, 0), rig.reads);
    try std.testing.expectEqual(@as(usize, 0), rig.x.attempts);
    var admissions: usize = 0;
    var finishes: usize = 0;
    for (rig.events[0..rig.count]) |event| {
        if (event.tag == .admitted) admissions += 1;
        if (event.tag == .finishing) {
            finishes += 1;
            try std.testing.expectEqual(@as(usize, 0), event.pins);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), admissions);
    try std.testing.expectEqual(admissions, finishes);
}

test "GELSD column work solves rank deficient tall and wide systems across scalar and integer ABIs" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const P = owner.gelsd_workspace.Planner(owner.Provider);
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        const C = if (raw) owner.RawGelsdColumnWork else owner.GelsdColumnWork;
        for (0..3) |shape| {
            const m: usize = if (shape == 1) 3 else 2;
            const n: usize = if (shape == 2) 3 else 2;
            var a = if (shape == 0) [_]T{ 1, 2, 2, 4, 88, 88 } else if (shape == 1) [_]T{ 1, 0, 1, 0, 1, 1 } else [_]T{ 1, 0, 0, 1, 1, 1 };
            const original_a = a;
            var b = if (shape == 0) [_]T{ 3, 6, 77, 77 } else if (shape == 1) [_]T{ 1, 2, 4, 77 } else [_]T{ 3, 0, 77, 77 };
            var singular = [_]T{ -1, -1 };
            var rank = [_]I{-1};
            const p = try P.make(T, I, m, n, 1);
            const work = try std.testing.allocator.alloc(T, @intCast(p.work));
            defer std.testing.allocator.free(work);
            const iwork = try std.testing.allocator.alloc(I, @intCast(p.iwork));
            defer std.testing.allocator.free(iwork);
            var context: support.Context = .{ .fail_index = 0 };
            const services = context.services();
            const args: C.Arguments(T, I) = .{ .m = @intCast(m), .n = @intCast(n), .nrhs = 1, .lda = @intCast(m), .ldb = 4, .a = if (raw) @as([*]align(1) T, @ptrCast(&a)) else &a, .b = if (raw) @as([*]align(1) T, @ptrCast(&b)) else &b, .singular = if (raw) @as([*]align(1) T, @ptrCast(&singular)) else &singular, .rank = if (raw) @as([*]align(1) I, @ptrCast(&rank)) else &rank, .rcond = -1, .work = if (raw) @as([*]align(1) T, work.ptr) else work, .lwork = p.work, .iwork = if (raw) @as([*]align(1) I, iwork.ptr) else iwork };
            const result = C.call(T, I, &services, args, .{});
            try std.testing.expect(result == .lapacke_info);
            try std.testing.expectEqual(@as(I, 0), result.lapacke_info);
            try std.testing.expectEqual(@as(I, if (shape == 0) 1 else 2), rank[0]);
            const tol: T = if (T == f32) 2e-5 else 2e-12;
            try std.testing.expectApproxEqAbs(@as(T, if (shape == 0) 0.6 else if (shape == 1) 4.0 / 3.0 else 2), b[0], tol);
            try std.testing.expectApproxEqAbs(@as(T, if (shape == 0) 1.2 else if (shape == 1) 7.0 / 3.0 else -1), b[1], tol);
            if (shape == 1) try std.testing.expectApproxEqAbs(@as(T, 1.0 / 3.0), b[2] * b[2], tol);
            if (shape == 2) try std.testing.expectApproxEqAbs(@as(T, 1), b[2], tol);
            try std.testing.expectEqual(@as(T, 77), b[3]);
            try std.testing.expectEqualSlices(T, &original_a, &a);
            try std.testing.expect(singular[0] >= singular[1] and singular[1] >= 0);
            try std.testing.expectEqual(@as(T, @floatFromInt(p.work)), work[0]);
            try std.testing.expectEqual(p.iwork, iwork[0]);
            try std.testing.expectEqual(@as(usize, 0), context.attempts);
            try std.testing.expectEqual(@as(usize, 0), context.reads);
        }
    };
}

test "GELSD column work distinguishes short workspace nonconvergence and empty quick return" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const C = owner.GelsdColumnWork;
    const p = try owner.gelsd_workspace.Planner(owner.Provider).make(f64, i32, 2, 2, 1);
    const work = try std.testing.allocator.alloc(f64, @intCast(p.work));
    defer std.testing.allocator.free(work);
    const iwork = try std.testing.allocator.alloc(i32, @intCast(p.iwork));
    defer std.testing.allocator.free(iwork);
    var a = [_]f64{ 1, 0, 1, 1 };
    var b = [_]f64{ 3, 2 };
    var singular = [_]f64{ 66, 66 };
    var rank = [_]i32{77};
    var context: support.Context = .{};
    const services = context.services();
    var args: C.Arguments(f64, i32) = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = &a, .b = &b, .singular = &singular, .rank = &rank, .rcond = -1, .work = work, .lwork = p.work - 1, .iwork = iwork };
    const short = C.call(f64, i32, &services, args, .{});
    try std.testing.expect(short == .lapacke_info);
    try std.testing.expectEqual(@as(i32, -13), short.lapacke_info);
    try std.testing.expectEqual(@as(i32, 12), context.hooks[0].argument);
    try std.testing.expectEqual(owner.common.HookKind.fortran_returning_test, context.hooks[0].kind);
    args.lwork = p.work;
    const failed = C.call(f64, i32, &services, args, .{ .max_steps = 0 });
    try std.testing.expect(failed == .lapacke_info);
    try std.testing.expectEqual(@as(i32, 1), failed.lapacke_info);
    try std.testing.expectEqualSlices(f64, &.{ 3, 2 }, &b);
    try std.testing.expectEqualSlices(f64, &.{ 66, 66 }, &singular);
    try std.testing.expectEqual(@as(i32, 77), rank[0]);
    args.m = 0;
    args.a = null;
    args.singular = null;
    args.rcond = std.math.nan(f64);
    const empty = C.call(f64, i32, &services, args, .{});
    try std.testing.expect(empty == .lapacke_info);
    try std.testing.expectEqual(@as(i32, 0), empty.lapacke_info);
    try std.testing.expectEqual(@as(i32, 0), rank[0]);
    try std.testing.expectEqualSlices(f64, &.{ 3, 2 }, &b);
    args.m = 2;
    args.a = &a;
    args.singular = &singular;
    @memset(&a, 0);
    @memset(&b, std.math.nan(f64));
    const zero = C.call(f64, i32, &services, args, .{});
    try std.testing.expect(zero == .lapacke_info);
    try std.testing.expectEqual(@as(i32, 0), zero.lapacke_info);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &b);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &singular);
}

test "GELSD high and work adapters preserve row column layouts and allocation ownership" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| inline for (.{ owner.common.Level.high, .work }) |level| {
        const A = if (raw) owner.RawGelsdAdapter else owner.GelsdAdapter;
        for ([_]c_int{ 101, 102 }) |layout| {
            var a = if (layout == 101) [_]T{ 2, 1, 88, 0, 3, 88 } else [_]T{ 2, 0, 88, 1, 3, 88 };
            const before_a = a;
            var b = if (layout == 101) [_]T{ 4, 77, 6, 77 } else [_]T{ 4, 6, 77, 77 };
            var singular: [2]T = undefined;
            var rank = [_]I{-1};
            const p = try owner.gelsd_workspace.Planner(owner.Provider).make(T, I, 2, 2, 1);
            const work = try std.testing.allocator.alloc(T, @intCast(p.work));
            defer std.testing.allocator.free(work);
            const iwork = try std.testing.allocator.alloc(I, @intCast(p.iwork));
            defer std.testing.allocator.free(iwork);
            var context: support.Context = .{};
            const services = context.services();
            const args: A.Arguments(T, I) = .{ .layout = layout, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 3, .ldb = if (layout == 101) 2 else 3, .a = if (raw) @as([*]align(1) T, @ptrCast(&a)) else &a, .b = if (raw) @as([*]align(1) T, @ptrCast(&b)) else &b, .singular = if (raw) @as([*]align(1) T, @ptrCast(&singular)) else &singular, .rank = if (raw) @as([*]align(1) I, @ptrCast(&rank)) else &rank, .rcond = -1, .work = if (level == .high) null else if (raw) @as([*]align(1) T, work.ptr) else work, .lwork = if (level == .high) 0 else p.work, .iwork = if (level == .high) null else if (raw) @as([*]align(1) I, iwork.ptr) else iwork } };
            const result = A.call(T, I, level, &services, args);
            try std.testing.expect(result == .lapacke_info);
            try std.testing.expectEqual(@as(I, 0), result.lapacke_info);
            try std.testing.expectEqual(@as(I, 2), rank[0]);
            const tol: T = if (T == f32) 2e-5 else 2e-12;
            try std.testing.expectApproxEqAbs(@as(T, 1), b[0], tol);
            try std.testing.expectApproxEqAbs(@as(T, 2), b[if (layout == 101) 2 else 1], tol);
            try std.testing.expectEqual(@as(T, 77), b[if (layout == 101) 1 else 2]);
            try std.testing.expectEqual(@as(T, 77), b[3]);
            try std.testing.expectEqualSlices(T, &before_a, &a);
            try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), context.reads);
            const allocations: usize = (if (level == .high) @as(usize, 2) else 0) + (if (layout == 101) @as(usize, 2) else 0);
            try std.testing.expectEqual(allocations, context.attempts);
            try std.testing.expectEqual(@as(usize, 0), context.live);
        }
    };
}

test "GELSD high allocation failures unwind integer work and both transpose buffers" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    for (0..4) |failure| {
        var a = [_]f64{ 2, 1, 0, 3 };
        var b = [_]f64{ 4, 6 };
        var s = [_]f64{ -1, -1 };
        var rank = [_]i32{-1};
        var context: support.Context = .{ .fail_index = failure };
        const services = context.services();
        const result = owner.GelsdAdapter.call(f64, i32, .high, &services, .{ .layout = 101, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = &a, .b = &b, .singular = &s, .rank = &rank, .rcond = -1, .work = null, .lwork = 0, .iwork = null } });
        try std.testing.expect(result == .lapacke_info);
        try std.testing.expectEqual(@as(i32, if (failure < 2) -1010 else -1011), result.lapacke_info);
        try std.testing.expectEqual(failure + 1, context.attempts);
        try std.testing.expectEqual(@as(usize, 0), context.live);
        try std.testing.expectEqual(@as(usize, 1), context.hook_count);
        try std.testing.expectEqual(if (failure < 2) owner.common.Entry.high else .work, context.hooks[0].entry);
        try std.testing.expectEqualSlices(f64, &.{ 2, 1, 0, 3 }, &a);
        try std.testing.expectEqualSlices(f64, &.{ 4, 6 }, &b);
        try std.testing.expectEqual(@as(i32, -1), rank[0]);
    }
}

test "GELSD high managed invocation remains single while callbacks request closing" {
    const owner = @import("numerical-owner");
    const h = @import("managed_support.zig");
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free, .hook }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        defer rig.done() catch @panic("cleanup");
        rig.trigger = trigger;
        rig.action = .close_nested;
        if (trigger == .hook) rig.x.fail_at = 1;
        var a = [_]f64{ 2, 1, 0, 3 };
        var b = [_]f64{ 4, 6 };
        var s: [2]f64 = undefined;
        var rank = [_]i32{-1};
        const result = try owner.GelsdAdapter.callManaged(f64, i32, .high, &rig.context, &rig.services, .{ .layout = 101, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = &a, .b = &b, .singular = &s, .rank = &rank, .rcond = -1, .work = null, .lwork = 0, .iwork = null } });
        try std.testing.expect(result == .lapacke_info);
        try std.testing.expectEqual(@as(i32, if (trigger == .hook) -1010 else 0), result.lapacke_info);
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 0), rig.x.live);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        var finished: usize = 0;
        for (rig.events[0..rig.count]) |event| if (event.tag == .finishing) {
            finished += 1;
            try std.testing.expectEqual(@as(usize, 0), event.pins);
        };
        try std.testing.expectEqual(@as(usize, 1), finished);
    }
}

test "GELSD row work rejects original matrices overlapping foreign outputs before mutation" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const A = owner.GelsdAdapter;
    const p = try owner.gelsd_workspace.Planner(owner.Provider).make(f64, i32, 2, 2, 1);
    for (0..2) |input| for (0..4) |target| {
        var a: [64]f64 = @splat(0);
        var b: [64]f64 = @splat(0);
        @memcpy(a[0..4], &[_]f64{ 2, 1, 0, 3 });
        @memcpy(b[0..2], &[_]f64{ 4, 6 });
        const before_a = a;
        const before_b = b;
        var work: [64]f64 = undefined;
        var iwork: [32]i32 = undefined;
        var s: [2]f64 = undefined;
        var rank: [1]i32 = undefined;
        const aliased = if (input == 0) a[0..] else b[0..];
        const ints: [*]i32 = @ptrCast(aliased.ptr);
        var context: support.Context = .{};
        const services = context.services();
        const result = A.call(f64, i32, .work, &services, .{ .layout = 101, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = a[0..4], .b = b[0..2], .singular = if (target == 1) aliased[0..2] else &s, .rank = if (target == 2) ints[0..1] else &rank, .rcond = -1, .work = if (target == 0) aliased else &work, .lwork = p.work, .iwork = if (target == 3) ints[0..@intCast(p.iwork)] else &iwork } });
        try std.testing.expect(result == .admission_fault);
        try std.testing.expectEqual(error.AliasingNotAllowed, result.admission_fault);
        try std.testing.expectEqualSlices(f64, &before_a, &a);
        try std.testing.expectEqualSlices(f64, &before_b, &b);
        try std.testing.expectEqual(@as(usize, 2), context.attempts);
        try std.testing.expectEqual(@as(usize, 0), context.live);
    };
}

test "GELSD high NaN precedence and work query bypass scans and transpose allocation" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const A = owner.GelsdAdapter;
    var a = [_]f64{ std.math.nan(f64), 0, 0, 1 };
    var b = [_]f64{ std.math.nan(f64), 2 };
    var singular: [2]f64 = undefined;
    var rank = [_]i32{77};
    var work: [1]f64 = undefined;
    var iwork: [1]i32 = undefined;
    var context: support.Context = .{};
    const services = context.services();
    var args: A.Arguments(f64, i32) = .{ .layout = 101, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = &a, .b = &b, .singular = &singular, .rank = &rank, .rcond = std.math.nan(f64), .work = &work, .lwork = -1, .iwork = &iwork } };
    for ([_]i32{ -5, -7, -10 }, 0..) |expected, step| {
        if (step == 1) a[0] = 1;
        if (step == 2) b[0] = 1;
        const result = A.call(f64, i32, .high, &services, args);
        try std.testing.expect(result == .lapacke_info);
        try std.testing.expectEqual(expected, result.lapacke_info);
    }
    try std.testing.expectEqual(@as(usize, 0), context.attempts);
    try std.testing.expectEqual(@as(usize, 0), context.hook_count);
    args.column.a = null;
    args.column.b = null;
    const queried = A.call(f64, i32, .work, &services, args);
    try std.testing.expect(queried == .lapacke_info);
    try std.testing.expectEqual(@as(i32, 0), queried.lapacke_info);
    try std.testing.expect(work[0] > 1 and iwork[0] > 1);
    try std.testing.expectEqual(@as(i32, 77), rank[0]);
    try std.testing.expectEqual(@as(usize, 3), context.reads);
    try std.testing.expectEqual(@as(usize, 0), context.attempts);
}

test "GELSD shared raw high route uses existing owner admission and rejects closing" {
    const owner = @import("numerical-owner");
    const h = @import("managed_support.zig");
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("cleanup");
    var a = [_]f64{ 2, 1, 0, 3 };
    var b = [_]f64{ 4, 6 };
    var s: [2]f64 = undefined;
    var rank = [_]i64{-1};
    const result = try owner.Shared.gelsdRawManaged(false, f64, i64, .high, &rig.context, .{ .layout = 101, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr, .singular = s[0..].ptr, .rank = rank[0..].ptr, .rcond = -1, .work = null, .lwork = 0, .iwork = null } });
    try std.testing.expect(result == .lapacke_info);
    try std.testing.expectEqual(@as(i64, 0), result.lapacke_info);
    try std.testing.expectApproxEqAbs(@as(f64, 1), b[0], 2e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), b[1], 2e-12);
    try std.testing.expectEqual(@as(i64, 2), rank[0]);
    var starts: usize = 0;
    var ends: usize = 0;
    for (rig.events[0..rig.count]) |event| {
        if (event.tag == .admitted) starts += 1;
        if (event.tag == .finishing) ends += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), starts);
    try std.testing.expectEqual(@as(usize, 1), ends);
    try std.testing.expectEqual(@as(usize, 4), rig.x.attempts);
    try std.testing.expectEqual(@as(usize, 0), rig.x.live);
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.gelsdRawManaged(false, f64, i64, .high, &rig.context, undefined));
}

test "GELSD finite input overflow returns success with IEEE outputs across integer widths" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const C = owner.GelsdColumnWork;
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| {
        const p = try owner.gelsd_workspace.Planner(owner.Provider).make(T, I, 2, 2, 1);
        const work = try std.testing.allocator.alloc(T, @intCast(p.work));
        defer std.testing.allocator.free(work);
        const iwork = try std.testing.allocator.alloc(I, @intCast(p.iwork));
        defer std.testing.allocator.free(iwork);
        var context: support.Context = .{};
        const services = context.services();
        const huge = std.math.floatMax(T);
        var a = [_]T{ huge, huge, huge, huge };
        var b = [_]T{ huge, huge };
        var singular = [_]T{ 66, 66 };
        var rank = [_]I{77};
        var args: C.Arguments(T, I) = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = &a, .b = &b, .singular = &singular, .rank = &rank, .rcond = -1, .work = work, .lwork = p.work, .iwork = iwork };
        const result = C.call(T, I, &services, args, .{});
        try std.testing.expect(result == .lapacke_info);
        try std.testing.expectEqual(@as(I, 0), result.lapacke_info);
        try std.testing.expectEqual(@as(I, 1), rank[0]);
        try std.testing.expectEqual(std.math.inf(T), singular[0]);
        for (b) |value| try std.testing.expectApproxEqAbs(@as(T, 0.5), value, 8 * std.math.floatEps(T));
        args.m = 1;
        args.n = 1;
        args.lda = 1;
        args.ldb = 1;
        a[0] = std.math.floatTrueMin(T);
        b[0] = -huge;
        const overflow = C.call(T, I, &services, args, .{});
        try std.testing.expect(overflow == .lapacke_info);
        try std.testing.expectEqual(@as(I, 0), overflow.lapacke_info);
        try std.testing.expectEqual(@as(I, 1), rank[0]);
        try std.testing.expectEqual(-std.math.inf(T), b[0]);
        try std.testing.expectEqual(std.math.floatTrueMin(T), singular[0]);
    };
}

test "GELSD work propagates nonfinite RHS without contaminating finite RHS or rank" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const C = owner.GelsdColumnWork;
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| {
        const p = try owner.gelsd_workspace.Planner(owner.Provider).make(T, I, 3, 2, 2);
        const work = try std.testing.allocator.alloc(T, @intCast(p.work));
        defer std.testing.allocator.free(work);
        const iwork = try std.testing.allocator.alloc(I, @intCast(p.iwork));
        defer std.testing.allocator.free(iwork);
        var context: support.Context = .{};
        const services = context.services();
        for ([_]T{ std.math.nan(T), std.math.inf(T), -std.math.inf(T) }) |bad| {
            var a = [_]T{ 1, 0, 0, 0, 2, 0 };
            var b = [_]T{ bad, 1, 1, 3, 8, 5 };
            var singular = [_]T{ 66, 66 };
            var rank = [_]I{77};
            const args: C.Arguments(T, I) = .{ .m = 3, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &a, .b = &b, .singular = &singular, .rank = &rank, .rcond = -1, .work = work, .lwork = p.work, .iwork = iwork };
            const result = C.call(T, I, &services, args, .{});
            try std.testing.expect(result == .lapacke_info);
            try std.testing.expectEqual(@as(I, 0), result.lapacke_info);
            try std.testing.expectEqual(@as(I, 2), rank[0]);
            try std.testing.expectEqualSlices(T, &.{ 2, 1 }, &singular);
            try std.testing.expect(!std.math.isFinite(b[0]));
            try std.testing.expectEqual(@as(T, 3), b[3]);
            try std.testing.expectEqual(@as(T, 4), b[4]);
            try std.testing.expectEqual(@as(T, 25), b[5] * b[5]);
        }
    };
}

test "GELSD work IEEE coefficient path reports actual unconverged edges" {
    const owner = @import("numerical-owner");
    const support = @import("potrs_support.zig");
    const C = owner.GelsdColumnWork;
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, i64 }) |I| {
        const p = try owner.gelsd_workspace.Planner(owner.Provider).make(T, I, 2, 2, 1);
        const work = try std.testing.allocator.alloc(T, @intCast(p.work));
        defer std.testing.allocator.free(work);
        const iwork = try std.testing.allocator.alloc(I, @intCast(p.iwork));
        defer std.testing.allocator.free(iwork);
        var ctx: support.Context = .{};
        const services = ctx.services();
        var a = [_]T{ 1, 0, std.math.nan(T), 2 };
        var b = [_]T{ 1, 2 };
        var singular = [_]T{ 66, 66 };
        var rank = [_]I{77};
        var args: C.Arguments(T, I) = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = &a, .b = &b, .singular = &singular, .rank = &rank, .rcond = -1, .work = work, .lwork = p.work, .iwork = iwork };
        const failed = C.call(T, I, &services, args, .{ .max_steps = 3 });
        try std.testing.expect(failed == .lapacke_info);
        try std.testing.expectEqual(@as(I, 1), failed.lapacke_info);
        try std.testing.expectEqualSlices(T, &.{ 1, 2 }, &b);
        try std.testing.expectEqual(@as(I, 77), rank[0]);
        a = .{ 1, 0, 0, 0.5 };
        args.rcond = std.math.nan(T);
        const unordered = C.call(T, I, &services, args, .{});
        try std.testing.expect(unordered == .lapacke_info);
        try std.testing.expectEqual(@as(I, 0), unordered.lapacke_info);
        try std.testing.expectEqual(@as(I, 2), rank[0]);
        try std.testing.expectEqualSlices(T, &.{ 1, 4 }, &b);
    };
}
