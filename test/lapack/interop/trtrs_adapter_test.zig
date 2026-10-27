// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! TRTRS contracts using independent products and the configured shared owner.
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const A = owner.TrtrsAdapter;
const Raw = owner.RawTrtrsAdapter;
const Ref = struct { re: f64 = 0, im: f64 = 0 };
fn times(x: Ref, y: Ref) Ref {
    return .{ .re = x.re * y.re - x.im * y.im, .im = x.re * y.im + x.im * y.re };
}
fn read(comptime T: type, x: T) Ref {
    return .{ .re = s.re(T, x), .im = s.im(T, x) };
}
fn number(comptime T: type, x: Ref) T {
    return s.value(T, x.re, x.im);
}
fn expected(comptime T: type, i: usize, j: usize) Ref {
    return .{ .re = 0.5 * @as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(j)), .im = if (owner.Core.Scalar.isComplex(T)) 0.25 * (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(j))) else 0 };
}
fn near(comptime T: type, want: Ref, got: T) !void {
    const eps: f64 = if (T == f32 or T == s.C32) std.math.floatEps(f32) else std.math.floatEps(f64);
    try std.testing.expectApproxEqAbs(want.re, s.re(T, got), 256 * eps);
    try std.testing.expectApproxEqAbs(want.im, s.im(T, got), 256 * eps);
}
fn fault(want: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |got| try std.testing.expectEqual(want, @as(anyerror, got)),
        .lapacke_info => return error.TestUnexpectedResult,
    }
}
fn numerical(comptime T: type, comptime I: type, comptime raw: bool, managed: bool, layout: c_int, upper: bool, unit: bool, trans: u8, level: owner.common.Level) !void {
    const Adapter = if (raw) Raw else A;
    var a: [15]T = @splat(s.value(T, std.math.nan(f64), std.math.nan(f64)));
    for (0..3) |i| for (0..3) |j| {
        if ((if (upper) i <= j else i >= j) and !(unit and i == j)) a[s.pos(layout, i, j, 5)] = s.value(T, if (i == j) 2 + @as(f64, @floatFromInt(i)) else 0.125 * @as(f64, @floatFromInt(i + j + 1)), if (owner.Core.Scalar.isComplex(T)) 0.125 * (@as(f64, @floatFromInt(i + 1)) - @as(f64, @floatFromInt(j))) else 0);
    };
    const before_a = a;
    var b: [12]T = @splat(s.value(T, -777, 23));
    for (0..3) |i| for (0..2) |j| {
        var sum: Ref = .{};
        for (0..3) |k| {
            const r = if (trans == 'N') i else k;
            const c = if (trans == 'N') k else i;
            if (if (upper) r > c else r < c) continue;
            var coefficient = if (unit and r == c) Ref{ .re = 1 } else read(T, a[s.pos(layout, r, c, 5)]);
            if (trans == 'C') coefficient.im = -coefficient.im;
            const product = times(coefficient, expected(T, k, j));
            sum.re += product.re;
            sum.im += product.im;
        }
        b[s.pos(layout, i, j, 4)] = number(T, sum);
    };
    const before_b = b;
    const args: Adapter.Arguments(T, I) = .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .trans = trans, .diag = if (unit) 'U' else 'N', .n = 3, .nrhs = 2, .a = if (raw) a[0..].ptr else &a, .lda = 5, .b = if (raw) b[0..].ptr else &b, .ldb = 4 };
    if (managed) {
        var rig: h.Rig = .{};
        rig.init();
        try s.expectInfo(I, 0, try Adapter.callManaged(T, I, level, rig.context, &rig.services, args));
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), rig.reads);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 0), rig.x.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try rig.done();
    } else {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, Adapter.call(T, I, level, &services, args));
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    }
    try s.bytes(&before_a, &a);
    for (b, 0..) |v, index| {
        var active = false;
        for (0..3) |i| for (0..2) |j| if (index == s.pos(layout, i, j, 4)) {
            active = true;
            try near(T, expected(T, i, j), v);
        };
        if (!active) try s.bytes(before_b[index..][0..1], b[index..][0..1]);
    }
}

test "TRTRS 1536 real complex width access layout transpose unit and lifetime numerical configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]bool{ false, true }) |managed| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]bool{ false, true }) |unit| for ([_]u8{ 'N', 'T', 'C' }) |trans| for ([_]owner.common.Level{ .high, .work }) |level| {
            try numerical(T, I, raw, managed, layout, upper, unit, trans, level);
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 1536), count);
}

test "TRTRS scalar and NaN priorities keep exact shifted INFO and unit diagonal nonreads" {
    const base: A.Arguments(f64, i64) = .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'N', .n = 0, .nrhs = 0, .a = null, .lda = 1, .b = null, .ldb = 1 };
    for (0..7) |kind| {
        var args = base;
        const info: i64 = switch (kind) {
            0 => blk: {
                args.uplo = '?';
                break :blk -2;
            },
            1 => blk: {
                args.trans = '?';
                break :blk -3;
            },
            2 => blk: {
                args.diag = '?';
                break :blk -4;
            },
            3 => blk: {
                args.n = -1;
                break :blk -5;
            },
            4 => blk: {
                args.nrhs = -1;
                break :blk -6;
            },
            5 => blk: {
                args.lda = 0;
                break :blk -8;
            },
            else => blk: {
                args.ldb = 0;
                break :blk -10;
            },
        };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(i64, info, A.call(f64, i64, .work, &services, args));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(owner.common.Routine.trtrs, ctx.hooks[0].routine);
        try std.testing.expectEqual(owner.common.Entry.fortran_trtrs, ctx.hooks[0].entry);
        try std.testing.expectEqual(@as(i32, @intCast(-info - 1)), ctx.hooks[0].argument);
        try std.testing.expectEqual(@as(usize, 0), ctx.reads);
    }
    var a = [_]f64{std.math.nan(f64)};
    var b = [_]f64{std.math.nan(f64)};
    var args = base;
    args.n = 1;
    args.nrhs = 1;
    args.a = &a;
    args.b = &b;
    args.trans = '?';
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i64, -7, A.call(f64, i64, .high, &services, args));
    args.diag = 'U';
    try s.expectInfo(i64, -9, A.call(f64, i64, .high, &services, args));
    args.diag = '?';
    try s.expectInfo(i64, -9, A.call(f64, i64, .high, &services, args));
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
}

test "TRTRS zero RHS still checks nonunit exact singularity and unit singleton raw invalid A is not accessed" {
    const a = [_]f64{ 2, std.math.nan(f64), 0.5, 0 };
    for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        const row_a = [_]f64{ 2, 0.5, std.math.nan(f64), 0 };
        try s.expectInfo(i64, 2, A.call(f64, i64, .work, &services, .{ .layout = layout, .uplo = 'U', .trans = 'N', .diag = 'N', .n = 2, .nrhs = 0, .a = if (layout == 101) &row_a else &a, .lda = 2, .b = null, .ldb = if (layout == 101) 0 else 2 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        var b = [_]f64{17};
        try s.expectInfo(i64, 0, Raw.call(f64, i64, .high, &services, .{ .layout = layout, .uplo = 'L', .trans = 'C', .diag = 'U', .n = 1, .nrhs = 1, .a = @ptrFromInt(1), .lda = 1, .b = b[0..].ptr, .ldb = 1 }));
        try std.testing.expectEqual(@as(f64, 17), b[0]);
    }
}

test "TRTRS actual managed owner close and rebind during getter allocation and free completes copyback and reverse cleanup" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        rig.trigger = trigger;
        rig.action = .close_nested;
        rig.rebind = true;
        const a = [_]f64{ 2, std.math.nan(f64), -777, 1, 3 };
        var b = [_]f64{ 20, -900, 70 };
        rig.b = &b;
        rig.expect_copyback = true;
        try s.expectInfo(i64, 0, try A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .uplo = 'L', .trans = 'N', .diag = 'N', .n = 2, .nrhs = 1, .a = &a, .lda = 3, .b = &b, .ldb = 2 }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 2), rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, &b);
        try rig.done();
    }
}

test "TRTRS row OOM releases prior pins before diagnostic and raw storage faults stay outside INFO" {
    for (0..2) |fail_at| {
        var rig: h.Rig = .{};
        rig.init();
        rig.x.fail_at = fail_at;
        const a = [_]f64{ 2, 1, 0, 3 };
        var b = [_]f64{ 2, 7 };
        try s.expectInfo(i64, -1011, try A.callManaged(f64, i64, .work, rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .trans = '?', .diag = 'N', .n = 2, .nrhs = 1, .a = &a, .lda = 2, .b = &b, .ldb = 1 }));
        try std.testing.expectEqual(fail_at, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 1), rig.hooks);
        for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
            try std.testing.expectEqual(@as(usize, 0), event.pins);
            try std.testing.expectEqual(@as(usize, 1), event.active);
        };
        try rig.done();
    }
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(error.MisalignedPointer, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'U', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = @ptrFromInt(1), .ldb = 1 }));
}

test "TRTRS shared routes use the actual configured owner and reject closed admission before service capture" {
    comptime {
        if (A.OwnerIdentity != owner.implementation or owner.Provider.Triangular != owner.implementation.internal.numerics.Triangular) @compileError("TRTRS owner diverged");
    }
    var rig: h.Rig = .{};
    rig.init();
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.trtrsRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'U', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = null, .ldb = 1 }));
    try std.testing.expectEqual(@as(usize, 0), rig.count);
    try rig.done();
    var b = [_]f64{19};
    try s.expectInfo(i32, 0, try owner.Shared.trtrsDefault(false, f64, i32, .work, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'U', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = &b, .ldb = 1 }));
    try std.testing.expectEqual(@as(f64, 19), b[0]);
}

test "TRTRS unit clipped NaN scan precedes invalid leading dimension for both leading-part layouts" {
    const a = [_]f64{ 5, std.math.nan(f64) };
    for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        const args: A.Arguments(f64, i64) = .{ .layout = layout, .uplo = if (layout == 101) 'L' else 'U', .trans = 'N', .diag = 'U', .n = 2, .nrhs = 0, .a = &a, .lda = 1, .b = null, .ldb = if (layout == 101) 0 else 2 };
        try s.expectInfo(i64, -7, A.call(f64, i64, .high, &services, args));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try s.expectInfo(i64, -8, A.call(f64, i64, .work, &services, args));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
    }
}

test "TRTRS alias address capacity and admitted pin overflow preserve outputs with separate fault channels" {
    var a = [_]f64{ 2, 0, 1, 0 };
    const before_a = a;
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(error.AliasingNotAllowed, A.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'N', .n = 2, .nrhs = 1, .a = &a, .lda = 2, .b = a[0..2], .ldb = 2 }));
    try s.bytes(&before_a, &a);
    const last_aligned = std.math.maxInt(usize) - 7;
    try fault(error.SizeOverflow, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'U', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = @ptrFromInt(last_aligned), .ldb = 1 }));
    var b = [_]f64{ 2, 7 };
    const before_b = b;
    try fault(error.MisalignedPointer, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .trans = 'N', .diag = 'U', .n = 2, .nrhs = 1, .a = @ptrFromInt(1), .lda = 2, .b = b[0..].ptr, .ldb = 2 }));
    try fault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'U', .trans = '?', .diag = 'N', .n = 2, .nrhs = 1, .a = a[0..3], .lda = 2, .b = &b, .ldb = 1 }));
    try s.bytes(&before_b, &b);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    for ([_]usize{ 0, 1 }) |limit| {
        var rig: h.Rig = .{};
        rig.init();
        h.C.Context.testing.configure(rig.context, limit, null);
        try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .work, rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .trans = '?', .diag = 'N', .n = 2, .nrhs = 1, .a = &a, .lda = 2, .b = &b, .ldb = 1 }));
        try std.testing.expectEqual(limit, rig.x.attempts);
        try std.testing.expectEqual(limit, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        try s.bytes(&before_b, &b);
        try rig.done();
    }
}

test "TRTRS returning Fortran hook keeps one operation through callback replacement copyback and reverse frees" {
    var rig: h.Rig = .{};
    rig.init();
    rig.trigger = .hook;
    rig.action = .close_nested;
    rig.rebind = true;
    const a = [_]f64{ 2, 1, -777, std.math.nan(f64), 3 };
    var b = [_]f64{ 10, -777, 20 };
    rig.b = &b;
    rig.expect_copyback = true;
    try s.expectInfo(i64, -3, try A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .trans = '?', .diag = 'N', .n = 2, .nrhs = 1, .a = &a, .lda = 3, .b = &b, .ldb = 2 }));
    try std.testing.expectEqual(error.Closing, rig.nested_error.?);
    try std.testing.expectEqual(@as(usize, 1), rig.hooks);
    try std.testing.expectEqual(@as(usize, 1), rig.reads);
    for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
        try std.testing.expectEqual(@as(usize, 1), event.active);
        try std.testing.expectEqual(@as(usize, 2), event.pins);
    };
    try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, &b);
    try rig.done();
}

test "TRTRS concurrent close waits for real paused scratch free after logical RHS copyback" {
    const Call = struct {
        rig: *h.Rig,
        a: []const f64,
        b: []f64,
        result: ?A.Result(i64) = null,
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            self.result = A.callManaged(f64, i64, .work, self.rig.context, &self.rig.services, .{ .layout = 101, .uplo = 'L', .trans = 'N', .diag = 'N', .n = 2, .nrhs = 1, .a = self.a, .lda = 3, .b = self.b, .ldb = 2 }) catch |err| {
                self.failure = err;
                return;
            };
        }
    };
    var rig: h.Rig = .{};
    rig.init();
    rig.pause_free = true;
    rig.expect_copyback = true;
    const a = [_]f64{ 2, std.math.nan(f64), -777, 1, 3 };
    var b = [_]f64{ 20, -900, 70 };
    rig.b = &b;
    var call: Call = .{ .rig = &rig, .a = &a, .b = &b };
    {
        const worker = try std.Thread.spawn(.{}, Call.run, .{&call});
        defer {
            rig.continue_free.store(true, .release);
            worker.join();
        }
        h.execution.test_support.wait(&rig.paused);
        try std.testing.expectEqual(h.C.RequestResult.pending, rig.context.requestClose());
        try std.testing.expectEqual(h.C.CloseResult.busy, fixtureProgressClose(rig.context));
        const snapshot = rig.context.snapshot();
        try std.testing.expectEqual(@as(usize, 1), snapshot.active_operations);
        try std.testing.expectEqual(@as(usize, 2), snapshot.resource_pins);
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, &b);
        try std.testing.expectError(error.Closing, owner.Shared.trtrsRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .uplo = 'U', .trans = 'N', .diag = 'U', .n = 0, .nrhs = 0, .a = null, .lda = 1, .b = null, .ldb = 1 }));
    }
    try std.testing.expect(call.failure == null);
    try s.expectInfo(i64, 0, call.result.?);
    try rig.done();
}

fn fixtureProgressClose(ctx: *@import("managed_support.zig").Core.Execution.Context) @import("managed_support.zig").Core.Execution.CloseResult {
    var report: @import("managed_support.zig").Core.Execution.CallReport = .{};
    return ctx.tryProgressClose(&report);
}
