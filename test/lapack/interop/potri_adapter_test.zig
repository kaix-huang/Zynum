// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Private POTRI boundary, actual configured-owner and lifetime qualification.
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const A = owner.PotriAdapter;
const Raw = owner.RawPotriAdapter;
fn expectFullInverseAtFree(rig: *h.Rig) void {
    const expected = [_]f64{ 16448, -900, 20, -2048, 256 };
    const a = rig.b.?;
    if (a.len != expected.len) @panic("POTRI free observer extent");
    for (a, expected) |actual, wanted| if (actual != wanted) @panic("POTRI full result not published before free");
}
const Ref = struct { re: f64 = 0, im: f64 = 0 };
fn times(a: Ref, b: Ref) Ref {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conjugate(a: Ref) Ref {
    return .{ .re = a.re, .im = -a.im };
}
fn coefficient(comptime T: type, values: []const T, layout: c_int, row: usize, col: usize, upper: bool) Ref {
    if (if (upper) row > col else row < col) return .{};
    const v = values[s.pos(layout, row, col, 5)];
    return .{ .re = s.re(T, v), .im = s.im(T, v) };
}
fn matrixEntry(comptime T: type, values: []const T, layout: c_int, row: usize, col: usize, upper: bool) Ref {
    var sum: Ref = .{};
    for (0..3) |k| {
        const left = if (upper) conjugate(coefficient(T, values, layout, k, row, upper)) else coefficient(T, values, layout, row, k, upper);
        const right = if (upper) coefficient(T, values, layout, k, col, upper) else conjugate(coefficient(T, values, layout, col, k, upper));
        const v = times(left, right);
        sum.re += v.re;
        sum.im += v.im;
    }
    return sum;
}
fn outputEntry(comptime T: type, values: []const T, layout: c_int, row: usize, col: usize, upper: bool) Ref {
    return if (if (upper) row <= col else row >= col) coefficient(T, values, layout, row, col, upper) else conjugate(coefficient(T, values, layout, col, row, upper));
}
fn fault(expected: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |actual| try std.testing.expectEqual(expected, @as(anyerror, actual)),
        .lapacke_info => return error.TestUnexpectedResult,
    }
}
fn numerical(comptime T: type, comptime I: type, comptime raw: bool, managed: bool, layout: c_int, upper: bool, level: owner.common.Level) !void {
    const Adapter = if (raw) Raw else A;
    var a: [15]T = @splat(s.value(T, std.math.nan(f64), std.math.nan(f64)));
    for (0..3) |i| for (0..3) |j| {
        if (if (upper) i <= j else i >= j) a[s.pos(layout, i, j, 5)] = s.value(T, if (i == j) 2 + @as(f64, @floatFromInt(i)) else 0.125 * @as(f64, @floatFromInt(i + j + 1)), if (i == j) 0 else 0.125 * (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(j + 1))));
    };
    const before = a;
    const args: Adapter.Arguments(T, I) = .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 3, .a = if (raw) a[0..].ptr else &a, .lda = 5 };
    if (managed) {
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        try s.expectInfo(I, 0, try Adapter.callManaged(T, I, level, &rig.context, &rig.services, args));
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), rig.reads);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 1), rig.x.attempts);
        try std.testing.expectEqual(rig.x.attempts, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        const W = owner.Core.CholeskyInverse.WorkValue(T);
        const slot: usize = if (layout == 101) 1 else 0;
        try std.testing.expectEqual(@as(usize, 9 * @sizeOf(W)), rig.x.lens[slot]);
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
            const x = times(matrixEntry(T, &before, layout, i, k, upper), outputEntry(T, &a, layout, k, j, upper));
            const y = times(outputEntry(T, &a, layout, i, k, upper), matrixEntry(T, &before, layout, k, j, upper));
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
        if (row < 3 and col < 3 and (if (upper) row <= col else row >= col)) continue;
        try s.bytes(before[index..][0..1], a[index..][0..1]);
    }
}
test "POTRI 256 scalar width access layout level and lifetime configurations preserve both identity products" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]bool{ false, true }) |managed| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]owner.common.Level{ .high, .work }) |level| {
            try numerical(T, I, raw, managed, layout, upper, level);
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 256), count);
}
test "POTRI shifted scalar priorities and clipped NaN scans retain the five parameter boundary" {
    inline for (.{ i32, i64 }) |I| for ([_]owner.common.Level{ .high, .work }) |level| for (0..3) |kind| {
        var args: A.Arguments(f64, I) = .{ .layout = 102, .uplo = 'U', .n = 0, .a = null, .lda = 1 };
        const info: I = switch (kind) {
            0 => blk: {
                args.uplo = '?';
                break :blk -2;
            },
            1 => blk: {
                args.n = -1;
                break :blk -3;
            },
            else => blk: {
                args.lda = 0;
                break :blk -5;
            },
        };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, info, A.call(f64, I, level, &services, args));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(owner.common.Routine.potri, ctx.hooks[0].routine);
        try std.testing.expectEqual(owner.common.Entry.fortran_potri, ctx.hooks[0].entry);
        try std.testing.expectEqual(@as(i32, @intCast(-info - 1)), ctx.hooks[0].argument);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
    };
    var a = [_]f64{ 5, std.math.nan(f64) };
    for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        const args: A.Arguments(f64, i64) = .{ .layout = layout, .uplo = if (layout == 101) 'L' else 'U', .n = 2, .a = &a, .lda = 1 };
        try s.expectInfo(i64, -4, A.call(f64, i64, .high, &services, args));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try s.expectInfo(i64, -5, A.call(f64, i64, .work, &services, args));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    }
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i64, -1, Raw.call(f64, i64, .high, &services, .{ .layout = 0, .uplo = '?', .n = -1, .a = @ptrFromInt(1), .lda = 0 }));
    try std.testing.expectEqual(@as(usize, 0), ctx.reads);
    try std.testing.expectEqual(owner.common.Entry.high, ctx.hooks[0].entry);
}
test "POTRI empty raw pointers are unread and singleton inverses borrow no wide lease" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, Raw.call(T, I, level, &services, .{ .layout = layout, .uplo = 'L', .n = 0, .a = @ptrFromInt(1), .lda = 1 }));
        try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        var a = [_]T{s.value(T, -2, 0)};
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        try s.expectInfo(I, 0, try A.callManaged(T, I, level, &rig.context, &rig.services, .{ .layout = layout, .uplo = 'U', .n = 1, .a = &a, .lda = 5 }));
        try std.testing.expectEqual(s.value(T, 0.25, 0), a[0]);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), rig.x.attempts);
        try rig.done();
    };
}
test "POTRI singular preflight preserves all A bytes and bypasses intrinsic wide allocation" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| {
            var a = [_]T{ s.value(T, 2, 0), s.value(T, 1, 0.25), s.value(T, 1, -0.25), s.value(T, -0.0, 0) };
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            rig.x.fail_at = if (layout == 101) 1 else 0;
            try s.expectInfo(I, 2, try A.callManaged(T, I, .work, &rig.context, &rig.services, .{ .layout = layout, .uplo = 'L', .n = 2, .a = &a, .lda = 2 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), rig.x.attempts);
            try std.testing.expectEqual(@as(usize, 0), rig.hooks);
            try rig.done();
        }
    };
}

test "POTRI transpose and intrinsic workspace OOM have distinct INFO and release all prior pins before hooks" {
    for ([_]c_int{ 101, 102 }) |layout| {
        const failures: usize = if (layout == 101) 2 else 1;
        for (0..failures) |fail_at| {
            var a = [_]f64{ 2, 1, 0, 3 };
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            rig.x.fail_at = fail_at;
            const expected: i64 = if (layout == 101 and fail_at == 0) -1011 else -1010;
            try s.expectInfo(i64, expected, try A.callManaged(f64, i64, .work, &rig.context, &rig.services, .{ .layout = layout, .uplo = 'U', .n = 2, .a = &a, .lda = 2 }));
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

test "POTRI captured allocator and services survive getter allocation and free close and rebind" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        rig.trigger = trigger;
        rig.action = .close_nested;
        rig.rebind = true;
        var a = [_]f64{ 0.125, -900, 20, 1, 0.0625 };
        rig.b = &a;
        rig.expect_copyback = false;
        rig.free_observer = expectFullInverseAtFree;
        try s.expectInfo(i64, 0, try A.callManaged(f64, i64, .high, &rig.context, &rig.services, .{ .layout = 101, .uplo = 'L', .n = 2, .a = &a, .lda = 3 }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 2), rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
        try std.testing.expectEqualSlices(f64, &.{ 16448, -900, 20 }, a[0..3]);
        try rig.done();
    }
}

test "POTRI address capacity and pin counter failures preserve A outside INFO" {
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(error.MisalignedPointer, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .n = 1, .a = @ptrFromInt(1), .lda = 1 }));
    try fault(error.SizeOverflow, Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'U', .n = 1, .a = @ptrFromInt(std.math.maxInt(usize) - 7), .lda = 1 }));
    var a = [_]f64{ 2, 1, 0, 3 };
    const before = a;
    try fault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'U', .n = 2, .a = a[0..3], .lda = 2 }));
    try s.bytes(&before, &a);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    for ([_]usize{ 0, 1 }) |limit| {
        var rig: h.Rig = .{};
        rig.init();
        h.C.Context.testing.configure(&rig.context, limit, null);
        try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .work, &rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .n = 2, .a = &a, .lda = 2 }));
        try std.testing.expectEqual(limit, rig.x.attempts);
        try std.testing.expectEqual(limit, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        try s.bytes(&before, &a);
        try rig.done();
    }
}

test "POTRI own configured Core and shared routes reject close before callbacks or invalid services are read" {
    comptime {
        if (A.OwnerIdentity != owner.implementation or owner.Provider.CholeskyInverse != owner.implementation.internal.numerics.CholeskyInverse) @compileError("POTRI owner diverged");
    }
    var rig: h.Rig = .{};
    rig.init();
    _ = rig.context.requestClose();
    const args: Raw.Arguments(f64, i64) = .{ .layout = 102, .uplo = 'U', .n = 0, .a = null, .lda = 1 };
    try std.testing.expectError(error.Closing, owner.Shared.potriRawManaged(false, f64, i64, .work, &rig.context, args));
    try std.testing.expectError(error.Closing, Raw.callManaged(f64, i64, .high, &rig.context, @ptrFromInt(@alignOf(A.Services)), args));
    try std.testing.expectEqual(@as(usize, 0), rig.count);
    try rig.done();
}

test "POTRI returning Fortran diagnostics retain live admission and transpose pin through rebind" {
    var rig: h.Rig = .{};
    rig.init();
    rig.trigger = .hook;
    rig.action = .close_nested;
    rig.rebind = true;
    try s.expectInfo(i64, -3, try A.callManaged(f64, i64, .high, &rig.context, &rig.services, .{ .layout = 101, .uplo = 'U', .n = -1, .a = null, .lda = 0 }));
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
    const message = try D.translate(.{ .kind = .fortran_returning_test, .entry = .fortran_potri, .scalar = 'z', .argument = 2, .routine = .potri });
    try std.testing.expectEqualStrings("ZPOTRI", message.name());
    inline for (.{ i32, i64 }) |I| for ("sdcz") |tag| inline for (.{ owner.common.Entry.high, .work }) |entry| {
        const translated = try owner.Diagnostics(I).translate(.{ .kind = .lapacke, .entry = entry, .scalar = tag, .argument = -1010, .routine = .potri });
        var buffer: [32]u8 = undefined;
        const expected = try std.fmt.bufPrint(&buffer, "LAPACKE_{c}potri{s}", .{ tag, if (entry == .work) "_work" else "" });
        try std.testing.expectEqualStrings(expected, translated.name());
        try std.testing.expectEqual(@as(I, -1010), translated.info);
    };
}

test "POTRI concurrent close waits for paused wide free after numerical copyback" {
    const Call = struct {
        rig: *h.Rig,
        a: []f64,
        result: ?A.Result(i64) = null,
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            self.result = A.callManaged(f64, i64, .work, &self.rig.context, &self.rig.services, .{ .layout = 101, .uplo = 'L', .n = 2, .a = self.a, .lda = 3 }) catch |err| {
                self.failure = err;
                return;
            };
        }
    };
    var rig: h.Rig = .{};
    rig.init();
    rig.pause_free = true;
    rig.expect_copyback = false;
    var a = [_]f64{ 0.125, -900, 20, 1, 0.0625 };
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
        try std.testing.expectEqual(h.C.CloseResult.busy, rig.context.tryProgressClose());
        const snapshot = rig.context.snapshot();
        try std.testing.expectEqual(@as(usize, 1), snapshot.active_operations);
        try std.testing.expectEqual(@as(usize, 2), snapshot.resource_pins);
        try std.testing.expectEqualSlices(f64, &.{ 16448, -900, 20 }, a[0..3]);
        try std.testing.expectError(error.Closing, owner.Shared.potriRawManaged(false, f64, i64, .work, &rig.context, .{ .layout = 102, .uplo = 'U', .n = 0, .a = null, .lda = 1 }));
    }
    try std.testing.expect(call.failure == null);
    try s.expectInfo(i64, 0, call.result.?);
    try rig.done();
}

test "POTRI unmanaged capture and disabled scan never reread replaced service allocator or callbacks" {
    var a = [_]f64{ 2, 1, 0, 3 };
    var ctx: s.Context = .{};
    var services = ctx.services();
    ctx.change_binding = &services;
    try s.expectInfo(i64, 0, A.call(f64, i64, .high, &services, .{ .layout = 101, .uplo = 'U', .n = 2, .a = &a, .lda = 2 }));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    var nan = [_]f64{std.math.nan(f64)};
    var no_scan: s.Context = .{};
    const untouched = no_scan.services();
    try s.expectInfo(i64, 0, owner.NoScanPotriAdapter.call(f64, i64, .high, &untouched, .{ .layout = 102, .uplo = 'U', .n = 1, .a = &nan, .lda = 1 }));
    try std.testing.expectEqual(@as(usize, 0), no_scan.reads);
    try std.testing.expect(std.math.isNan(nan[0]));
}

test "POTRI all five shared routes use the original owner and agree on the numerical result" {
    inline for (0..5) |route| {
        var a = [_]f64{ 2, 1, 91, 4 };
        var rig: h.Rig = .{};
        rig.init();
        const typed: A.Arguments(f64, i32) = .{ .layout = 102, .uplo = 'L', .n = 2, .a = &a, .lda = 2 };
        const raw: Raw.Arguments(f64, i32) = .{ .layout = 102, .uplo = 'L', .n = 2, .a = a[0..].ptr, .lda = 2 };
        switch (route) {
            0 => try s.expectInfo(i32, 0, try owner.Shared.potriManaged(false, f64, i32, .high, &rig.context, typed)),
            1 => try s.expectInfo(i32, 0, try owner.Shared.potriRawManaged(false, f64, i32, .work, &rig.context, raw)),
            2 => try s.expectInfo(i32, 0, try owner.Shared.potriRawManagedWithDefaults(false, f64, i32, .high, &rig.context, raw, .{})),
            3 => try s.expectInfo(i32, 0, try owner.Shared.potriDefault(false, f64, i32, .high, typed)),
            4 => try s.expectInfo(i32, 0, try owner.Shared.potriRawDefault(false, f64, i32, .work, raw)),
            else => unreachable,
        }
        try std.testing.expectEqualSlices(f64, &.{ 0.265625, -0.03125, 91, 0.0625 }, &a);
        if (route < 3) {
            try std.testing.expectEqual(@as(usize, 1), rig.max_active);
            try std.testing.expectEqual(@as(usize, 1), rig.x.attempts);
            try std.testing.expectEqual(@as(usize, 1), rig.x.frees);
        }
        try rig.done();
    }
}
test "POTRI admitted typed and raw routes never create a child admission or use service allocator bytes" {
    inline for (.{ false, true }) |raw| {
        const Adapter = if (raw) Raw else A;
        var a = [_]f64{ 2, 1, 91, 4 };
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        {
            var operation = try rig.context.begin();
            defer operation.end();
            try s.expectInfo(i64, 0, try Adapter.callAdmitted(f64, i64, .high, &operation, &rig.services, .{ .layout = 102, .uplo = 'L', .n = 2, .a = if (raw) a[0..].ptr else &a, .lda = 2 }));
        }
        var admissions: usize = 0;
        for (rig.events[0..rig.count]) |event| {
            if (event.tag == .admitted) admissions += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), admissions);
        try std.testing.expectEqual(@as(usize, 1), rig.reads);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try rig.done();
    }
}
test "POTRI complex diagonal imaginary NaN is scanned while runtime and compiled disabled routes retain IEEE output" {
    inline for (.{ s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var a = [_]T{s.value(T, 2, std.math.nan(f64))};
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, -4, A.call(T, I, .high, &services, .{ .layout = 102, .uplo = 'U', .n = 1, .a = &a, .lda = 1 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        ctx.enabled = false;
        try s.expectInfo(I, 0, A.call(T, I, .high, &services, .{ .layout = 102, .uplo = 'U', .n = 1, .a = &a, .lda = 1 }));
        try std.testing.expect(std.math.isNan(a[0].re));
        try std.testing.expectEqual(@as(owner.Core.Scalar.Real(T), 0), a[0].im);
        a[0] = s.value(T, 2, std.math.nan(f64));
        var no_scan: s.Context = .{};
        const untouched = no_scan.services();
        try s.expectInfo(I, 0, owner.NoScanPotriAdapter.call(T, I, .high, &untouched, .{ .layout = 102, .uplo = 'L', .n = 1, .a = &a, .lda = 1 }));
        try std.testing.expectEqual(@as(usize, 0), no_scan.reads);
        try std.testing.expect(std.math.isNan(a[0].re));
    };
}
test "POTRI managed widths layouts and triangles retain underflowed-factor Hermitian couplings in 64 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        const R = owner.Core.Scalar.Real(T);
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for (0..2) |subnormal| {
            const small: f64 = if (comptime R == f32) (if (subnormal == 0) 0x1p-126 else 0x1p-127) else (if (subnormal == 0) 0x1p-1022 else 0x1p-1023);
            const large: f64 = if (comptime R == f32) (if (subnormal == 0) 0x1p126 else 0x1p127) else (if (subnormal == 0) 0x1p1022 else 0x1p1023);
            var a: [20]T = @splat(s.value(T, -777, 333));
            for (0..4) |i| for (0..4) |j| {
                if (if (upper) i <= j else i >= j) a[s.pos(layout, i, j, 5)] = s.value(T, if (i == j) 1 else 0, 0);
            };
            const links = [_]struct { i: usize, j: usize, v: T }{ .{ .i = 0, .j = 1, .v = s.value(T, small, if (owner.Core.Scalar.isComplex(T)) small else 0) }, .{ .i = 1, .j = 3, .v = s.value(T, small, if (owner.Core.Scalar.isComplex(T)) -small else 0) }, .{ .i = 2, .j = 3, .v = s.value(T, if (owner.Core.Scalar.isComplex(T)) large / 2 else large, if (owner.Core.Scalar.isComplex(T)) large / 2 else 0) } };
            for (links) |link| a[s.pos(layout, if (upper) link.i else link.j, if (upper) link.j else link.i, 5)] = if (upper) link.v else owner.Core.Scalar.conj(T, link.v);
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            try s.expectInfo(I, 0, try A.callManaged(T, I, .high, &rig.context, &rig.services, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 4, .a = &a, .lda = 5 }));
            const value = a[s.pos(layout, if (upper) 0 else 2, if (upper) 2 else 0, 5)];
            try std.testing.expectEqual(s.value(T, -small, if (upper) small else -small), value);
            for (0..a.len) |index| {
                const row = if (layout == 101) index / 5 else index % 5;
                const col = if (layout == 101) index % 5 else index / 5;
                if (row < 4 and col < 4 and (if (upper) row <= col else row >= col)) continue;
                try s.bytes(before[index..][0..1], a[index..][0..1]);
            }
            try rig.done();
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 64), count);
}

test "POTRI row-major malformed scalar empty and OOM priorities preserve nonreads" {
    inline for (.{ i32, i64 }) |I| for ([_]owner.common.Level{ .high, .work }) |level| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, -5, Raw.call(f64, I, level, &services, .{ .layout = 101, .uplo = '?', .n = -1, .a = @ptrFromInt(1), .lda = -2 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try std.testing.expectEqual(owner.common.HookKind.lapacke, ctx.hooks[0].kind);
        var bad: s.Context = .{};
        const captured = bad.services();
        try s.expectInfo(I, -2, Raw.call(f64, I, level, &captured, .{ .layout = 101, .uplo = '?', .n = 2, .a = @ptrFromInt(1), .lda = 2 }));
        try std.testing.expectEqual(@as(usize, 1), bad.attempts);
        try std.testing.expectEqual(@as(usize, 0), bad.live);
        try std.testing.expectEqual(owner.common.Entry.fortran_potri, bad.hooks[0].entry);
        var oom: s.Context = .{ .fail_index = 0 };
        const failed = oom.services();
        try s.expectInfo(I, -1011, Raw.call(f64, I, level, &failed, .{ .layout = 101, .uplo = '?', .n = 2, .a = @ptrFromInt(1), .lda = 2 }));
        try std.testing.expectEqual(@as(usize, 1), oom.attempts);
        try std.testing.expectEqual(owner.common.Entry.work, oom.hooks[0].entry);
        var empty: s.Context = .{};
        const es = empty.services();
        try s.expectInfo(I, 0, Raw.call(f64, I, level, &es, .{ .layout = 101, .uplo = 'L', .n = 0, .a = @ptrFromInt(1), .lda = 0 }));
        try std.testing.expectEqual(@as(usize, 1), empty.attempts);
        try std.testing.expectEqual(@as(usize, 0), empty.live);
        var empty_oom: s.Context = .{ .fail_index = 0 };
        const eos = empty_oom.services();
        try s.expectInfo(I, -1011, Raw.call(f64, I, level, &eos, .{ .layout = 101, .uplo = 'L', .n = 0, .a = @ptrFromInt(1), .lda = 0 }));
        var col: s.Context = .{};
        const cs = col.services();
        try s.expectInfo(I, -5, Raw.call(f64, I, level, &cs, .{ .layout = 102, .uplo = 'L', .n = 0, .a = @ptrFromInt(1), .lda = 0 }));
        try std.testing.expectEqual(@as(usize, 0), col.attempts);
    };
}
test "POTRI scalar width raw spans typed capacity and metadata overflow remain admission faults" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            try fault(error.MisalignedPointer, Raw.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .a = @ptrFromInt(1), .lda = 1 }));
            try fault(error.SizeOverflow, Raw.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .a = @ptrFromInt(std.math.maxInt(usize) - (@sizeOf(T) - 1)), .lda = 1 }));
            try fault(error.MissingBuffer, Raw.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .a = null, .lda = 1 }));
            var values = [_]T{s.value(T, 2, 0)};
            try fault(error.BufferTooSmall, A.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'U', .n = 1, .a = values[0..0], .lda = 1 }));
            try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        }
        var rig: h.Rig = .{};
        rig.init();
        try fault(error.SizeOverflow, try Raw.callManaged(T, i64, .work, &rig.context, &rig.services, .{ .layout = 101, .uplo = 'L', .n = std.math.maxInt(i64), .a = @ptrFromInt(1), .lda = std.math.maxInt(i64) }));
        try std.testing.expectEqual(@as(usize, 0), rig.x.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        try rig.done();
    }
}
test "POTRI all scalar width and level OOM positions free prior leases and retain original unmanaged hook capture" {
    var failures: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]owner.common.Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| {
        const positions: usize = if (layout == 101) 2 else 1;
        for (0..positions) |position| {
            var a = [_]T{ s.value(T, 2, 0), s.value(T, 1, 0.25), s.value(T, 1, -0.25), s.value(T, 4, 0) };
            const before = a;
            var rig: h.Rig = .{};
            rig.init();
            rig.services.allocator = undefined;
            rig.x.fail_at = position;
            const info: I = if (layout == 101 and position == 0) -1011 else -1010;
            try s.expectInfo(I, info, try A.callManaged(T, I, level, &rig.context, &rig.services, .{ .layout = layout, .uplo = 'L', .n = 2, .a = &a, .lda = 2 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(position + 1, rig.x.attempts);
            try std.testing.expectEqual(position, rig.x.frees);
            try std.testing.expectEqual(@as(usize, 1), rig.hooks);
            try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), rig.reads);
            for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
                try std.testing.expectEqual(@as(usize, 0), event.pins);
                try std.testing.expectEqual(@as(usize, 1), event.active);
            };
            try rig.done();
            failures += 1;
        }
    };
    try std.testing.expectEqual(@as(usize, 48), failures);
    var a = [_]f64{ 2, 1, 0, 4 };
    const before = a;
    var ctx: s.Context = .{ .fail_index = 0 };
    var services = ctx.services();
    ctx.change_binding = &services;
    try s.expectInfo(i64, -1011, A.call(f64, i64, .high, &services, .{ .layout = 101, .uplo = 'L', .n = 2, .a = &a, .lda = 2 }));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 1), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    try s.bytes(&before, &a);
}
