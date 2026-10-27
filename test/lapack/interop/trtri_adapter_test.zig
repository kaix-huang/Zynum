// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Private TRTRI adapter and real configured-owner lifecycle contracts.
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const A = owner.TrtriAdapter;
const Raw = owner.RawTrtriAdapter;
fn expectFullInverseAtFree(rig: *h.Rig) void {
    const values = rig.b.?;
    const expected = [_]f64{ 10, -900, 20, -200, 20 };
    if (values.len != expected.len) @panic("inverse free observer extent");
    for (values, expected) |actual, wanted| {
        if (!(@abs(actual - wanted) <= 2e-12)) @panic("full inverse was not published before resource free");
    }
}
const Ref = struct { re: f64 = 0, im: f64 = 0 };
fn times(a: Ref, b: Ref) Ref {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn coefficient(comptime T: type, values: []const T, layout: c_int, row: usize, col: usize, upper: bool, unit: bool) Ref {
    if (if (upper) row > col else row < col) return .{};
    if (unit and row == col) return .{ .re = 1 };
    const value = values[s.pos(layout, row, col, 5)];
    return .{ .re = s.re(T, value), .im = s.im(T, value) };
}
fn fault(expected: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |actual| try std.testing.expectEqual(expected, @as(anyerror, actual)),
        .lapacke_info => return error.TestUnexpectedResult,
    }
}
fn numerical(comptime T: type, comptime I: type, comptime raw: bool, managed: bool, layout: c_int, upper: bool, unit: bool, level: owner.common.Level) !void {
    const Adapter = if (raw) Raw else A;
    var a: [15]T = @splat(s.value(T, std.math.nan(f64), std.math.nan(f64)));
    for (0..3) |i| for (0..3) |j| {
        if ((if (upper) i <= j else i >= j) and !(unit and i == j)) a[s.pos(layout, i, j, 5)] = s.value(T, if (i == j) 2 + @as(f64, @floatFromInt(i)) else 0.125 * @as(f64, @floatFromInt(i + j + 1)), if (owner.Core.Scalar.isComplex(T)) 0.125 * (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(j + 1))) else 0);
    };
    const before = a;
    const args: Adapter.Arguments(T, I) = .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .diag = if (unit) 'U' else 'N', .n = 3, .a = if (raw) a[0..].ptr else &a, .lda = 5 };
    if (managed) {
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        try s.expectInfo(I, 0, try Adapter.callManaged(T, I, level, rig.context, &rig.services, args));
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), rig.reads);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 1), rig.x.attempts);
        try std.testing.expectEqual(rig.x.attempts, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        const slot: usize = if (layout == 101) 1 else 0;
        const W = owner.Core.TriangularInverse.WorkValue(T);
        try std.testing.expectEqual(@as(usize, 3 * @sizeOf(W)), rig.x.lens[slot]);
        try std.testing.expectEqual(std.mem.Alignment.of(W), rig.x.aligns[slot]);
        var admitted: usize = 0;
        var finishing: usize = 0;
        for (rig.events[0..rig.count]) |event| {
            if (event.tag == .admitted) admitted += 1;
            if (event.tag == .finishing) finishing += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), admitted);
        try std.testing.expectEqual(@as(usize, 1), finishing);
        try rig.done();
    } else {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, Adapter.call(T, I, level, &services, args));
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 1), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    }
    const eps: f64 = if (T == f32 or T == s.C32) std.math.floatEps(f32) else std.math.floatEps(f64);
    for (0..3) |i| for (0..3) |j| {
        var left: Ref = .{};
        var right: Ref = .{};
        for (0..3) |k| {
            const x = times(coefficient(T, &before, layout, i, k, upper, unit), coefficient(T, &a, layout, k, j, upper, unit));
            const y = times(coefficient(T, &a, layout, i, k, upper, unit), coefficient(T, &before, layout, k, j, upper, unit));
            left.re += x.re;
            left.im += x.im;
            right.re += y.re;
            right.im += y.im;
        }
        const wanted: f64 = if (i == j) 1 else 0;
        try std.testing.expectApproxEqAbs(wanted, left.re, 128 * eps);
        try std.testing.expectApproxEqAbs(@as(f64, 0), left.im, 128 * eps);
        try std.testing.expectApproxEqAbs(wanted, right.re, 128 * eps);
        try std.testing.expectApproxEqAbs(@as(f64, 0), right.im, 128 * eps);
    };
    for (0..a.len) |index| {
        const row = if (layout == 101) index / 5 else index % 5;
        const col = if (layout == 101) index % 5 else index / 5;
        if (row < 3 and col < 3 and (if (upper) row <= col else row >= col) and !(unit and row == col)) continue;
        try s.bytes(before[index..][0..1], a[index..][0..1]);
    }
}

test "TRTRI 512 scalar width access layout unit level and lifetime configurations preserve both identity products" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]bool{ false, true }) |managed| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]bool{ false, true }) |unit| for ([_]owner.common.Level{ .high, .work }) |level| {
            try numerical(T, I, raw, managed, layout, upper, unit, level);
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 512), count);
}

test "TRTRI scalar INFO priorities and NaN scans retain shifted arguments and implicit diagonal nonreads" {
    inline for (.{ i32, i64 }) |I| {
        const base: A.Arguments(f64, I) = .{ .layout = 102, .uplo = 'U', .diag = 'N', .n = 0, .a = null, .lda = 1 };
        for (0..4) |kind| {
            var args = base;
            const info: I = switch (kind) {
                0 => blk: {
                    args.uplo = '?';
                    break :blk -2;
                },
                1 => blk: {
                    args.diag = '?';
                    break :blk -3;
                },
                2 => blk: {
                    args.n = -1;
                    break :blk -4;
                },
                else => blk: {
                    args.lda = 0;
                    break :blk -6;
                },
            };
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, info, A.call(f64, I, .work, &services, args));
            try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
            try std.testing.expectEqual(owner.common.Routine.trtri, ctx.hooks[0].routine);
            try std.testing.expectEqual(owner.common.Entry.fortran_trtri, ctx.hooks[0].entry);
            try std.testing.expectEqual(@as(i32, @intCast(-info - 1)), ctx.hooks[0].argument);
            try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.reads);
        }
    }
    var a = [_]f64{std.math.nan(f64)};
    var args: A.Arguments(f64, i64) = .{ .layout = 0, .uplo = 'U', .diag = 'N', .n = 1, .a = &a, .lda = 1 };
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i64, -1, A.call(f64, i64, .high, &services, args));
    try std.testing.expectEqual(@as(usize, 0), ctx.reads);
    args.layout = 102;
    args.uplo = '?';
    try s.expectInfo(i64, -2, A.call(f64, i64, .high, &services, args));
    args.uplo = 'U';
    try s.expectInfo(i64, -5, A.call(f64, i64, .high, &services, args));
    args.diag = 'U';
    try s.expectInfo(i64, 0, A.call(f64, i64, .high, &services, args));
    try std.testing.expect(std.math.isNan(a[0]));
    try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
}

test "TRTRI raw unit empty singleton and clipped NaN scan keep their actual nonreads" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]usize{ 0, 1 }) |n| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, 0, Raw.call(T, I, .high, &services, .{ .layout = layout, .uplo = 'L', .diag = 'U', .n = @intCast(n), .a = @ptrFromInt(1), .lda = 1 }));
            try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
        };
    };
    var a = [_]f64{ 5, std.math.nan(f64) };
    for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        const args: A.Arguments(f64, i64) = .{ .layout = layout, .uplo = if (layout == 101) 'L' else 'U', .diag = 'U', .n = 2, .a = &a, .lda = 1 };
        try s.expectInfo(i64, -5, A.call(f64, i64, .high, &services, args));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try s.expectInfo(i64, -6, A.call(f64, i64, .work, &services, args));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    }
}

test "TRTRI singular preflight preserves all A bytes and bypasses intrinsic wide allocation" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| {
            var a = [_]T{ s.value(T, 2, 0), s.value(T, 1, 0.25), s.value(T, 1, -0.25), s.value(T, -0.0, 0) };
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            rig.x.fail_at = if (layout == 101) 1 else 0;
            try s.expectInfo(I, 2, try A.callManaged(T, I, .work, rig.context, &rig.services, .{ .layout = layout, .uplo = 'L', .diag = 'N', .n = 2, .a = &a, .lda = 2 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), rig.x.attempts);
            try std.testing.expectEqual(@as(usize, 0), rig.hooks);
            try rig.done();
        }
    };
}

test "TRTRI transpose and intrinsic workspace OOM have distinct INFO and release all prior pins before hooks" {
    for ([_]c_int{ 101, 102 }) |layout| {
        const failures: usize = if (layout == 101) 2 else 1;
        for (0..failures) |fail_at| {
            var a = [_]f64{ 2, 1, 0, 3 };
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            rig.x.fail_at = fail_at;
            const expected: i64 = if (layout == 101 and fail_at == 0) -1011 else -1010;
            try s.expectInfo(i64, expected, try A.callManaged(f64, i64, .work, rig.context, &rig.services, .{ .layout = layout, .uplo = 'U', .diag = 'N', .n = 2, .a = &a, .lda = 2 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(fail_at + 1, rig.x.attempts);
            try std.testing.expectEqual(fail_at, rig.x.frees);
            try std.testing.expectEqual(@as(usize, 1), rig.hooks);
            for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
                try std.testing.expectEqual(@as(usize, 0), event.pins);
                try std.testing.expectEqual(@as(usize, 1), event.active);
                try std.testing.expectEqual(@as(i32, @intCast(expected)), event.detail);
            };
            try rig.done();
        }
    }
}

test "TRTRI captured allocator and services survive getter allocation and free close and rebind" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        rig.trigger = trigger;
        rig.action = .close_nested;
        rig.rebind = true;
        var a = [_]f64{ 0.1, -900, 20, 1, 0.05 };
        rig.b = &a;
        rig.expect_copyback = true;
        rig.free_observer = expectFullInverseAtFree;
        try s.expectInfo(i64, 0, try A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .uplo = 'L', .diag = 'N', .n = 2, .a = &a, .lda = 3 }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 2), rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, a[0..3]);
        try rig.done();
    }
}

test "TRTRI address capacity and pin counter failures preserve A outside INFO" {
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(error.MisalignedPointer, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .diag = 'N', .n = 1, .a = @ptrFromInt(1), .lda = 1 }));
    try fault(error.SizeOverflow, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .diag = 'N', .n = 1, .a = @ptrFromInt(std.math.maxInt(usize) - 7), .lda = 1 }));
    var a = [_]f64{ 2, 1, 0, 3 };
    const before = a;
    try fault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'U', .diag = 'N', .n = 2, .a = a[0..3], .lda = 2 }));
    try s.bytes(&before, &a);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    for ([_]usize{ 0, 1 }) |limit| {
        var rig: h.Rig = .{};
        rig.init();
        h.C.Context.testing.configure(rig.context, limit, null);
        try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .work, rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .diag = 'N', .n = 2, .a = &a, .lda = 2 }));
        try std.testing.expectEqual(limit, rig.x.attempts);
        try std.testing.expectEqual(limit, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        try s.bytes(&before, &a);
        try rig.done();
    }
}

test "TRTRI own configured Core and shared routes admit once and reject close before callback capture" {
    comptime {
        if (A.OwnerIdentity != owner.implementation or owner.Provider.TriangularInverse != owner.implementation.internal.numerics.TriangularInverse) @compileError("TRTRI owner diverged");
    }
    var rig: h.Rig = .{};
    rig.init();
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.trtriRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .uplo = 'U', .diag = 'U', .n = 0, .a = null, .lda = 1 }));
    try std.testing.expectEqual(@as(usize, 0), rig.count);
    try rig.done();
    try s.expectInfo(i32, 0, try owner.Shared.trtriDefault(false, f64, i32, .work, .{ .layout = 102, .uplo = 'U', .diag = 'U', .n = 1, .a = null, .lda = 1 }));
    try s.expectInfo(i64, 0, try owner.Shared.trtriRawDefault(false, s.C64, i64, .high, .{ .layout = 102, .uplo = 'L', .diag = 'U', .n = 1, .a = @ptrFromInt(1), .lda = 1 }));
}

test "TRTRI returning Fortran diagnostics retain live admission and transpose pin through rebind" {
    var rig: h.Rig = .{};
    rig.init();
    rig.trigger = .hook;
    rig.action = .close_nested;
    rig.rebind = true;
    try s.expectInfo(i64, -4, try A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .diag = 'N', .n = -1, .a = null, .lda = 0 }));
    try std.testing.expectEqual(error.Closing, rig.nested_error.?);
    try std.testing.expectEqual(@as(usize, 1), rig.hooks);
    try std.testing.expectEqual(@as(usize, 1), rig.reads);
    try std.testing.expectEqual(@as(usize, 1), rig.x.frees);
    for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
        try std.testing.expectEqual(@as(usize, 1), event.active);
        try std.testing.expectEqual(@as(usize, 1), event.pins);
    };
    try rig.done();
    const D = owner.Diagnostics(i64);
    const message = try D.translate(.{ .kind = .fortran_returning_test, .entry = .fortran_trtri, .scalar = 'z', .argument = 2, .routine = .trtri });
    try std.testing.expectEqualStrings("ZTRTRI", message.name());
}

test "TRTRI concurrent close waits for paused wide free after numerical copyback" {
    const Call = struct {
        rig: *h.Rig,
        a: []f64,
        result: ?A.Result(i64) = null,
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            self.result = A.callManaged(f64, i64, .work, self.rig.context, &self.rig.services, .{ .layout = 101, .uplo = 'L', .diag = 'N', .n = 2, .a = self.a, .lda = 3 }) catch |err| {
                self.failure = err;
                return;
            };
        }
    };
    var rig: h.Rig = .{};
    rig.init();
    rig.pause_free = true;
    rig.expect_copyback = true;
    var a = [_]f64{ 0.1, -900, 20, 1, 0.05 };
    rig.b = &a;
    rig.free_observer = expectFullInverseAtFree;
    var call: Call = .{ .rig = &rig, .a = &a };
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
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, a[0..3]);
        try std.testing.expectError(error.Closing, owner.Shared.trtriRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .uplo = 'U', .diag = 'U', .n = 0, .a = null, .lda = 1 }));
    }
    try std.testing.expect(call.failure == null);
    try s.expectInfo(i64, 0, call.result.?);
    try rig.done();
}

test "TRTRI unmanaged capture and disabled scan never reread replaced service allocator or callbacks" {
    var a = [_]f64{ 2, 1, 0, 3 };
    var ctx: s.Context = .{};
    var services = ctx.services();
    ctx.change_binding = &services;
    try s.expectInfo(i64, 0, A.call(f64, i64, .high, &services, .{ .layout = 101, .uplo = 'U', .diag = 'N', .n = 2, .a = &a, .lda = 2 }));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    var nan = [_]f64{std.math.nan(f64)};
    var no_scan: s.Context = .{};
    const untouched = no_scan.services();
    try s.expectInfo(i64, 0, owner.NoScanTrtriAdapter.call(f64, i64, .high, &untouched, .{ .layout = 102, .uplo = 'U', .diag = 'N', .n = 1, .a = &nan, .lda = 1 }));
    try std.testing.expectEqual(@as(usize, 0), no_scan.reads);
    try std.testing.expect(std.math.isNan(nan[0]));
}

test "TRTRI both width and layout routes preserve scaled representable cross entries in 32 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        const R = owner.Core.Scalar.Real(T);
        const large: f64 = if (comptime R == f32) 0x1p127 else 0x1p1023;
        const small: f64 = if (comptime R == f32) 0x1p-126 else 0x1p-1022;
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| {
            var a: [9]T = @splat(s.value(T, 0, 0));
            a[0] = s.value(T, large, 0);
            a[4] = s.value(T, large, 0);
            a[8] = s.value(T, small, 0);
            a[s.pos(layout, if (upper) 0 else 1, if (upper) 1 else 0, 3)] = s.value(T, small, 0);
            a[s.pos(layout, if (upper) 1 else 2, if (upper) 2 else 1, 3)] = s.value(T, large, 0);
            var rig: h.Rig = .{};
            rig.init();
            try s.expectInfo(I, 0, try A.callManaged(T, I, .high, rig.context, &rig.services, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .diag = 'N', .n = 3, .a = &a, .lda = 3 }));
            const value = a[s.pos(layout, if (upper) 0 else 2, if (upper) 2 else 0, 3)];
            const expected: f128 = 1 / @as(f128, large);
            try std.testing.expectApproxEqAbs(expected, @as(f128, s.re(T, value)), expected * 1e-6);
            try rig.done();
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 32), count);
}

fn fixtureProgressClose(ctx: *@import("managed_support.zig").Core.Execution.Context) @import("managed_support.zig").Core.Execution.CloseResult {
    var report: @import("managed_support.zig").Core.Execution.CallReport = .{};
    return ctx.tryProgressClose(&report);
}
