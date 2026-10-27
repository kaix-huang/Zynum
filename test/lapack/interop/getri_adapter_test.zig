// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! GETRI private adapter qualification over the actual configured owner.
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const A = owner.GetriAdapter;
const Raw = owner.RawGetriAdapter;
fn fault(expected: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |actual| try std.testing.expectEqual(expected, @as(anyerror, actual)),
        .lapacke_info => return error.TestUnexpectedResult,
    }
}
const Ref = struct { re: f64 = 0, im: f64 = 0 };
fn times(a: Ref, b: Ref) Ref {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn ref(comptime T: type, v: T) Ref {
    return .{ .re = s.re(T, v), .im = s.im(T, v) };
}
fn numerical(comptime T: type, comptime I: type, comptime raw: bool, managed: bool, layout: c_int, level: owner.common.Level, n: usize) !void {
    const Adapter = if (raw) Raw else A;
    const ld = n + 2;
    var original: [25]T = undefined;
    var factors: [25]T = undefined;
    for (0..n) |i| for (0..n) |j| {
        // Rotating rows of a diagonally dominant matrix yields nontrivial,
        // noncommuting pivot products without a candidate-inverse recurrence.
        const r = if (n > 1) (i + n - 1) % n else i;
        const re: f64 = if (r == j) 3 + @as(f64, @floatFromInt(r)) else @as(f64, @floatFromInt(r + 2 * j + 1)) / 32;
        original[i + j * n] = s.value(T, re, if (owner.Core.Scalar.isComplex(T)) @as(f64, @floatFromInt(i + j + 1)) / 128 else 0);
        factors[i + j * n] = original[i + j * n];
    };
    var pivots: [5]I = @splat(-999);
    try std.testing.expectEqual(@as(usize, 0), owner.Core.factorGetrf(T, I, .{ .values = factors[0 .. n * n], .row_count = n, .column_count = n, .leading_dimension = @max(@as(usize, 1), n) }, pivots[0..n]));
    if (n != 0) pivots[n - 1] = -999;
    const pivot_before = pivots;
    var a: [40]T = @splat(s.value(T, std.math.nan(f64), std.math.nan(f64)));
    for (0..n) |i| for (0..n) |j| {
        a[2 + s.pos(layout, i, j, ld)] = factors[i + j * n];
    };
    const before = a;
    var work: [7]T = @splat(s.value(T, -71, 22));
    const work_before = work;
    const args: Adapter.Arguments(T, I) = .{ .layout = layout, .n = @intCast(n), .a = if (raw) (if (n == 0) @ptrFromInt(1) else a[2..].ptr) else (if (n == 0) null else a[2..]), .lda = @intCast(ld), .pivots = if (raw) (if (n <= 1) @ptrFromInt(1) else pivots[0..].ptr) else (if (n <= 1) null else pivots[0 .. n - 1]), .work = if (raw) work[0..].ptr else &work, .lwork = @intCast(@max(@as(usize, 1), n)) };
    const expected_allocations = @as(usize, if (level == .high) 1 else 0) + @as(usize, if (layout == 101) 1 else 0) + @as(usize, if (n > 1) 1 else 0);
    if (managed) {
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        try s.expectInfo(I, 0, try Adapter.callManaged(T, I, level, rig.context, &rig.services, args));
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), rig.reads);
        try std.testing.expectEqual(expected_allocations, rig.x.attempts);
        try std.testing.expectEqual(rig.x.attempts, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        var admissions: usize = 0;
        var finishes: usize = 0;
        for (rig.events[0..rig.count]) |event| {
            if (event.tag == .admitted) admissions += 1;
            if (event.tag == .finishing) finishes += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), admissions);
        try std.testing.expectEqual(@as(usize, 1), finishes);
        try rig.done();
    } else {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, Adapter.call(T, I, level, &services, args));
        try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
        try std.testing.expectEqual(expected_allocations, ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    }
    const eps: f64 = if (T == f32 or T == s.C32) 3e-5 else 2e-12;
    for (0..n) |i| for (0..n) |j| {
        var left: Ref = .{};
        var right: Ref = .{};
        for (0..n) |k| {
            const x = times(ref(T, original[i + k * n]), ref(T, a[2 + s.pos(layout, k, j, ld)]));
            const y = times(ref(T, a[2 + s.pos(layout, i, k, ld)]), ref(T, original[k + j * n]));
            left.re += x.re;
            left.im += x.im;
            right.re += y.re;
            right.im += y.im;
        }
        const target: f64 = if (i == j) 1 else 0;
        try std.testing.expectApproxEqAbs(target, left.re, eps);
        try std.testing.expectApproxEqAbs(@as(f64, 0), left.im, eps);
        try std.testing.expectApproxEqAbs(target, right.re, eps);
        try std.testing.expectApproxEqAbs(@as(f64, 0), right.im, eps);
    };
    for (0..a.len) |index| {
        const row = if (index < 2) n else if (layout == 101) (index - 2) / ld else (index - 2) % ld;
        const col = if (index < 2) n else if (layout == 101) (index - 2) % ld else (index - 2) / ld;
        if (row < n and col < n) continue;
        try s.bytes(before[index..][0..1], a[index..][0..1]);
    }
    try s.bytes(&pivot_before, &pivots);
    const first: usize = if (level == .high) 0 else 1;
    try s.bytes(work_before[first..], work[first..]);
    if (level == .work) {
        try std.testing.expectEqual(@as(f64, @floatFromInt(@max(@as(usize, 1), n))), s.re(T, work[0]));
        try std.testing.expectEqual(@as(f64, 0), s.im(T, work[0]));
    }
}

test "GETRI 512 scalar width access lifetime layout level and size configurations preserve two-sided identities" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]bool{ false, true }) |managed| for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| for ([_]usize{ 0, 1, 3, 5 }) |n| {
            try numerical(T, I, raw, managed, layout, level, n);
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 512), count);
}

test "GETRI raw query has one output and no factor pivot transpose or allocator access including upward f32 encoding" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]I{ 0, 1, 16777217, 16777219 }) |n| {
            var work = [_]T{ s.value(T, -7, 11), s.value(T, 99, 21) };
            const before = work;
            var rig: h.Rig = .{};
            rig.init();
            rig.services.allocator = undefined;
            const args: Raw.Arguments(T, I) = .{ .layout = layout, .n = n, .a = @ptrFromInt(1), .lda = @max(@as(I, 1), n), .pivots = @ptrFromInt(1), .work = work[0..].ptr, .lwork = -1 };
            try s.expectInfo(I, 0, try Raw.callManaged(T, I, .work, rig.context, &rig.services, args));
            const expected: f64 = if (T == f32 or T == s.C32) (if (n == 16777217) 16777218 else if (n == 16777219) 16777220 else @floatFromInt(@max(@as(I, 1), n))) else @floatFromInt(@max(@as(I, 1), n));
            try std.testing.expectEqual(expected, s.re(T, work[0]));
            try std.testing.expectEqual(@as(f64, 0), s.im(T, work[0]));
            try s.bytes(before[1..], work[1..]);
            try std.testing.expectEqual(@as(usize, 0), rig.x.attempts);
            try std.testing.expectEqual(@as(usize, 0), rig.reads);
            try rig.done();
        };
    };
}

test "GETRI argument and row query priorities write WORK before backend errors" {
    inline for (.{ i32, i64 }) |I| {
        for (0..3) |which| {
            var work = [_]f64{-7};
            var args: A.Arguments(f64, I) = .{ .layout = 102, .n = 0, .a = null, .lda = 1, .pivots = null, .work = &work, .lwork = 1 };
            const expected: I = switch (which) {
                0 => blk: {
                    args.n = -1;
                    break :blk -2;
                },
                1 => blk: {
                    args.lda = 0;
                    break :blk -4;
                },
                else => blk: {
                    args.lwork = 0;
                    break :blk -7;
                },
            };
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, expected, A.call(f64, I, .work, &services, args));
            try std.testing.expectEqual(@as(f64, 1), work[0]);
            try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
            try std.testing.expectEqual(owner.common.Routine.getri, ctx.hooks[0].routine);
            try std.testing.expectEqual(owner.common.Entry.fortran_getri, ctx.hooks[0].entry);
            try std.testing.expectEqual(@as(i32, @intCast(-expected - 1)), ctx.hooks[0].argument);
        }
        var work = [_]f64{-7};
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, -4, A.call(f64, I, .work, &services, .{ .layout = 101, .n = 3, .a = null, .lda = 2, .pivots = null, .work = &work, .lwork = -1 }));
        try std.testing.expectEqual(@as(f64, -7), work[0]);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try std.testing.expectEqual(owner.common.HookKind.lapacke, ctx.hooks[0].kind);
    }
}

test "GETRI high NaN layout and disabled scan stages retain their exact callback counts" {
    var nan = [_]f64{std.math.nan(f64)};
    var work = [_]f64{-7};
    var ctx: s.Context = .{};
    var services = ctx.services();
    var args: A.Arguments(f64, i64) = .{ .layout = 0, .n = 1, .a = &nan, .lda = 1, .pivots = null, .work = &work, .lwork = 1 };
    try s.expectInfo(i64, -1, A.call(f64, i64, .high, &services, args));
    try std.testing.expectEqual(@as(usize, 0), ctx.reads);
    args.layout = 102;
    try s.expectInfo(i64, -3, A.call(f64, i64, .high, &services, args));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    try s.expectInfo(i64, 0, A.call(f64, i64, .work, &services, args));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expect(std.math.isNan(nan[0]));
    var no_scan: s.Context = .{};
    services = no_scan.services();
    try s.expectInfo(i64, 0, owner.NoScanGetriAdapter.call(f64, i64, .high, &services, .{ .layout = 102, .n = 1, .a = &nan, .lda = 1, .pivots = null }));
    try std.testing.expectEqual(@as(usize, 0), no_scan.reads);
    try std.testing.expectEqual(@as(usize, 1), no_scan.attempts);
    try std.testing.expectEqual(@as(usize, 0), no_scan.live);
}

test "GETRI singular factor returns first positive INFO without pivot reads or intrinsic allocation" {
    inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| {
        var a = [_]f64{ 2, 0.25, 0.5, -900, 0.125, -0.0, 1, -900, 0.25, 0.5, 0, -900 };
        const before = a;
        var work = [_]f64{ -7, -8, -9 };
        var rig: h.Rig = .{};
        rig.init();
        rig.services.allocator = undefined;
        try s.expectInfo(I, 2, try A.callManaged(f64, I, level, rig.context, &rig.services, .{ .layout = layout, .n = 3, .a = &a, .lda = 4, .pivots = null, .work = &work, .lwork = 3 }));
        try s.bytes(&before, &a);
        const expected = @as(usize, if (level == .high) 1 else 0) + @as(usize, if (layout == 101) 1 else 0);
        try std.testing.expectEqual(expected, rig.x.attempts);
        try std.testing.expectEqual(expected, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        if (level == .work) try std.testing.expectEqual(@as(f64, 3), work[0]);
        try rig.done();
    };
}

test "GETRI invalid used pivot prefix and output aliases are admission faults without factor stores" {
    var a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const before = a;
    var work = [_]f64{ -7, -8, -9 };
    inline for (.{ i32, i64 }) |I| {
        for ([_][2]I{ .{ 0, 2 }, .{ -1, 2 }, .{ 4, 2 }, .{ 1, 1 } }) |pivots| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            try fault(error.InvalidPivot, A.call(f64, I, .work, &services, .{ .layout = 102, .n = 3, .a = &a, .lda = 3, .pivots = &pivots, .work = &work, .lwork = 3 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        }
    }
    var ctx: s.Context = .{};
    const services = ctx.services();
    const pivots = [_]i64{ 1, 2, -999 };
    try fault(error.AliasingNotAllowed, A.call(f64, i64, .work, &services, .{ .layout = 102, .n = 3, .a = &a, .lda = 3, .pivots = &pivots, .work = a[0..3], .lwork = 3 }));
    try s.bytes(&before, &a);
    const aliased_pivots: [*]const i64 = @ptrCast(&a);
    try fault(error.AliasingNotAllowed, A.call(f64, i64, .work, &services, .{ .layout = 102, .n = 3, .a = &a, .lda = 3, .pivots = aliased_pivots[0..2], .work = &work, .lwork = 3 }));
    try s.bytes(&before, &a);
}

test "GETRI query encoding overflow raw addresses and typed capacities remain outside scalar INFO" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var query = [_]T{s.value(T, -7, 11)};
        const before = query;
        var ctx: s.Context = .{};
        const services = ctx.services();
        const result = Raw.call(T, I, .work, &services, .{ .layout = 102, .n = std.math.maxInt(I), .a = @ptrFromInt(1), .lda = std.math.maxInt(I), .pivots = @ptrFromInt(1), .work = query[0..].ptr, .lwork = -1 });
        if (comptime I == i32 and (T == f64 or T == s.C64)) {
            try s.expectInfo(I, 0, result);
            try std.testing.expectEqual(@as(f64, @floatFromInt(std.math.maxInt(I))), s.re(T, query[0]));
        } else {
            try fault(error.SizeOverflow, result);
            try s.bytes(&before, &query);
        }
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    };
    var ctx: s.Context = .{};
    const services = ctx.services();
    const query_args: Raw.Arguments(f64, i64) = .{ .layout = 102, .n = 1, .a = @ptrFromInt(1), .lda = 1, .pivots = @ptrFromInt(1), .work = @ptrFromInt(1), .lwork = -1 };
    try fault(error.MisalignedPointer, Raw.call(f64, i64, .work, &services, query_args));
    var overflow = query_args;
    overflow.work = @ptrFromInt(std.math.maxInt(usize) - 7);
    try fault(error.SizeOverflow, Raw.call(f64, i64, .work, &services, overflow));
    var a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const before = a;
    var short = [_]f64{ -7, -8 };
    const pivots = [_]i64{ 1, 2, -999 };
    try fault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = 102, .n = 3, .a = &a, .lda = 3, .pivots = &pivots, .work = &short, .lwork = 3 }));
    try s.bytes(&before, &a);
    try std.testing.expectEqual(@as(f64, -7), short[0]);
}

test "GETRI every public transpose and wide allocation failure releases captured leases with distinct diagnostics" {
    for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| {
        const high = level == .high;
        const allocations = @as(usize, if (high) 1 else 0) + @as(usize, if (layout == 101) 1 else 0) + 1;
        for (0..allocations) |fail_at| {
            var a = [_]f64{ 2, 0, 0, 4 };
            const before = a;
            const pivots = [_]i64{ 1, -999 };
            var work = [_]f64{ -7, -8 };
            var rig: h.Rig = .{};
            rig.init();
            rig.services.allocator = undefined;
            rig.x.fail_at = fail_at;
            const public_failure = high and fail_at == 0;
            const transpose_failure = layout == 101 and fail_at == @as(usize, if (high) 1 else 0);
            const expected: i64 = if (transpose_failure) -1011 else -1010;
            try s.expectInfo(i64, expected, try A.callManaged(f64, i64, level, rig.context, &rig.services, .{ .layout = layout, .n = 2, .a = &a, .lda = 2, .pivots = &pivots, .work = &work, .lwork = 2 }));
            try s.bytes(&before, &a);
            try std.testing.expectEqual(fail_at + 1, rig.x.attempts);
            try std.testing.expectEqual(fail_at, rig.x.frees);
            const expected_hooks: usize = if (high and !public_failure and !transpose_failure) 2 else 1;
            try std.testing.expectEqual(expected_hooks, rig.hooks);
            var hook_index: usize = 0;
            for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
                const pins: usize = if (high and !public_failure and hook_index == 0) 1 else 0;
                try std.testing.expectEqual(pins, event.pins);
                try std.testing.expectEqual(@as(usize, 1), event.active);
                try std.testing.expectEqual(@as(i32, @intCast(expected)), event.detail);
                hook_index += 1;
            };
            try rig.done();
        }
    };
}

fn expectCompleteAtFree(rig: *h.Rig) void {
    const values = rig.b.?;
    const expected = [_]f64{ 0.5, -0.125, -900, 0, 0.25, -800 };
    for (values, expected) |actual, wanted| if (@abs(actual - wanted) > 1e-14) @panic("GETRI full matrix not published before free");
}
test "GETRI high query and execution retain one allocator and admission through close rebind and every free" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        rig.trigger = trigger;
        rig.action = .close_nested;
        rig.rebind = true;
        rig.services.allocator = undefined;
        var a = [_]f64{ 2, 1, -900, 0, 4, -800 };
        const pivots = [_]i64{ 1, -999 };
        rig.b = &a;
        rig.free_observer = expectCompleteAtFree;
        try s.expectInfo(i64, 0, try A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .n = 2, .a = &a, .lda = 3, .pivots = &pivots }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 3), rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
        var frees: usize = 0;
        for (rig.events[0..rig.count]) |event| if (event.tag == .free) {
            try std.testing.expectEqual(@as(usize, 1), event.active);
            try std.testing.expectEqual(3 - frees, event.pins);
            frees += 1;
        };
        try std.testing.expectEqual(@as(usize, 3), frees);
        try rig.done();
    }
}

test "GETRI pin counter exhaustion is propagated outside INFO at each high row lease" {
    for (0..3) |limit| {
        var rig: h.Rig = .{};
        rig.init();
        h.C.Context.testing.configure(rig.context, limit, null);
        rig.services.allocator = undefined;
        var a = [_]f64{ 2, 1, 0, 4 };
        const before = a;
        const pivots = [_]i64{ 1, -999 };
        try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .high, rig.context, &rig.services, .{ .layout = 101, .n = 2, .a = &a, .lda = 2, .pivots = &pivots }));
        try s.bytes(&before, &a);
        try std.testing.expectEqual(limit, rig.x.attempts);
        try std.testing.expectEqual(limit, rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.hooks);
        try rig.done();
    }
}

fn expectOriginalAtFree(rig: *h.Rig) void {
    const expected = [_]f64{ 2, 1, -900, 0, 4, -900 };
    for (rig.b.?, expected) |actual, wanted| if (actual != wanted) @panic("GETRI returning-error copyback missing at free");
}
test "GETRI returning backend diagnostic keeps transpose pinned and restores logical factor after the hook" {
    var rig: h.Rig = .{};
    rig.init();
    rig.trigger = .hook;
    rig.action = .close_nested;
    rig.services.allocator = undefined;
    var a = [_]f64{ 2, 1, -900, 0, 4, -900 };
    rig.b = &a;
    rig.free_observer = expectOriginalAtFree;
    var work = [_]f64{ -7, 99 };
    try s.expectInfo(i64, -7, try A.callManaged(f64, i64, .work, rig.context, &rig.services, .{ .layout = 101, .n = 2, .a = &a, .lda = 3, .pivots = null, .work = &work, .lwork = 1 }));
    try std.testing.expectEqual(error.Closing, rig.nested_error.?);
    try std.testing.expectEqualSlices(f64, &.{ 2, 99 }, &work);
    try std.testing.expectEqual(@as(usize, 1), rig.x.frees);
    for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
        try std.testing.expectEqual(@as(usize, 1), event.active);
        try std.testing.expectEqual(@as(usize, 1), event.pins);
        try std.testing.expectEqual(@as(i32, 6), event.detail);
    };
    try rig.done();
    const message = try owner.Diagnostics(i64).translate(.{ .kind = .fortran_returning_test, .entry = .fortran_getri, .scalar = 'z', .argument = 6, .routine = .getri });
    try std.testing.expectEqualStrings("ZGETRI", message.name());
}

test "GETRI concurrent close waits for full output and all three paused high row leases" {
    const Call = struct {
        rig: *h.Rig,
        a: []f64,
        pivots: []const i64,
        result: ?A.Result(i64) = null,
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            self.result = A.callManaged(f64, i64, .high, self.rig.context, &self.rig.services, .{ .layout = 101, .n = 2, .a = self.a, .lda = 3, .pivots = self.pivots }) catch |err| {
                self.failure = err;
                return;
            };
        }
    };
    var rig: h.Rig = .{};
    rig.init();
    rig.pause_free = true;
    rig.services.allocator = undefined;
    var a = [_]f64{ 2, 1, -900, 0, 4, -800 };
    const pivots = [_]i64{ 1, -999 };
    rig.b = &a;
    rig.free_observer = expectCompleteAtFree;
    var call: Call = .{ .rig = &rig, .a = &a, .pivots = &pivots };
    {
        const worker = try std.Thread.spawn(.{}, Call.run, .{&call});
        defer {
            rig.continue_free.store(true, .release);
            worker.join();
        }
        h.execution.test_support.wait(&rig.paused);
        try std.testing.expectEqual(h.C.RequestResult.pending, rig.context.requestClose());
        try std.testing.expectEqual(h.C.CloseResult.busy, fixtureProgressClose(rig.context));
        const state = rig.context.snapshot();
        try std.testing.expectEqual(@as(usize, 1), state.active_operations);
        try std.testing.expectEqual(@as(usize, 3), state.resource_pins);
        expectCompleteAtFree(&rig);
        try std.testing.expectError(error.Closing, owner.Shared.getriRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .n = 0, .a = @ptrFromInt(1), .lda = 1, .pivots = @ptrFromInt(1), .work = @ptrFromInt(1), .lwork = 1 }));
    }
    try std.testing.expect(call.failure == null);
    try s.expectInfo(i64, 0, call.result.?);
    try rig.done();
}

test "GETRI configured owner and all shared route families keep default and local contexts distinct" {
    comptime {
        if (A.OwnerIdentity != owner.implementation or owner.Provider.LuInverse != owner.implementation.internal.numerics.LuInverse) @compileError("GETRI owner split");
    }
    var rig: h.Rig = .{};
    rig.init();
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.getriRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .n = 0, .a = null, .lda = 1, .pivots = null }));
    try std.testing.expectEqual(@as(usize, 0), rig.count);
    try rig.done();
    var work = [_]f64{-7};
    try s.expectInfo(i32, 0, try owner.Shared.getriDefault(false, f64, i32, .work, .{ .layout = 102, .n = 0, .a = null, .lda = 1, .pivots = null, .work = &work, .lwork = 1 }));
    try std.testing.expectEqual(@as(f64, 1), work[0]);
    try s.expectInfo(i64, 0, try owner.Shared.getriRawDefault(false, s.C64, i64, .high, .{ .layout = 102, .n = 0, .a = @ptrFromInt(1), .lda = 1, .pivots = @ptrFromInt(1) }));
}

test "GETRI managed access and public workspace routes preserve extreme cross entries in 128 configurations" {
    var count: usize = 0;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        const Adapter = if (raw) Raw else A;
        const R = owner.Core.Scalar.Real(T);
        for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| for (0..2) |edge| {
            const small: f64 = if (comptime R == f32) (if (edge == 0) 0x1p-126 else 0x1p-127) else (if (edge == 0) 0x1p-1022 else 0x1p-1023);
            const large: f64 = if (comptime R == f32) (if (edge == 0) 0x1p126 else 0x1p127) else (if (edge == 0) 0x1p1022 else 0x1p1023);
            var a: [20]T = @splat(s.value(T, std.math.nan(f64), std.math.nan(f64)));
            for (0..4) |i| for (0..4) |j| {
                a[s.pos(layout, i, j, 5)] = s.value(T, if (i == j) 1 else 0, 0);
            };
            a[s.pos(layout, 1, 0, 5)] = s.value(T, small, if (owner.Core.Scalar.isComplex(T)) small else 0);
            a[s.pos(layout, 2, 1, 5)] = s.value(T, small, if (owner.Core.Scalar.isComplex(T)) -small else 0);
            a[s.pos(layout, 3, 2, 5)] = s.value(T, if (owner.Core.Scalar.isComplex(T)) large / 2 else large, if (owner.Core.Scalar.isComplex(T)) large / 2 else 0);
            const before = a;
            const pivots = [_]I{ 1, 2, 3, -999 };
            var work: [4]T = @splat(s.value(T, -7, -8));
            var rig: h.Rig = .{};
            rig.init();
            rig.services.allocator = undefined;
            try s.expectInfo(I, 0, try Adapter.callManaged(T, I, level, rig.context, &rig.services, .{ .layout = layout, .n = 4, .a = if (raw) a[0..].ptr else &a, .lda = 5, .pivots = if (raw) pivots[0..].ptr else &pivots, .work = if (raw) work[0..].ptr else &work, .lwork = 4 }));
            const value = a[s.pos(layout, 3, 0, 5)];
            try std.testing.expectEqual(-small, s.re(T, value));
            try std.testing.expectEqual(if (owner.Core.Scalar.isComplex(T)) -small else @as(f64, 0), s.im(T, value));
            for (0..4) |j| try s.bytes(before[4 + j * 5 ..][0..1], a[4 + j * 5 ..][0..1]);
            try rig.done();
            count += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 128), count);
}

test "GETRI empty row stride and negative size distinguish high query from work transpose stages" {
    inline for (.{ i32, i64 }) |I| {
        for ([_]owner.common.Level{ .high, .work }) |level| {
            var rig: h.Rig = .{};
            rig.init();
            rig.services.allocator = undefined;
            var work = [_]f64{-7};
            try s.expectInfo(I, 0, try Raw.callManaged(f64, I, level, rig.context, &rig.services, .{ .layout = 101, .n = 0, .a = @ptrFromInt(1), .lda = 0, .pivots = @ptrFromInt(1), .work = work[0..].ptr, .lwork = 1 }));
            const allocations: usize = if (level == .high) 2 else 1;
            try std.testing.expectEqual(allocations, rig.x.attempts);
            try std.testing.expectEqual(allocations, rig.x.frees);
            try rig.done();
        }
        for ([_]bool{ false, true }) |fail| for ([_]owner.common.Level{ .high, .work }) |level| {
            var rig: h.Rig = .{};
            rig.init();
            if (fail) rig.x.fail_at = 0;
            var work = [_]f64{-7};
            const expected: I = if (fail and level == .work) -1011 else -2;
            try s.expectInfo(I, expected, try A.callManaged(f64, I, level, rig.context, &rig.services, .{ .layout = 101, .n = -1, .a = null, .lda = 0, .pivots = null, .work = &work, .lwork = 0 }));
            try std.testing.expectEqual(@as(usize, if (level == .high) 0 else 1), rig.x.attempts);
            try rig.done();
        };
        var work = [_]f64{ -7, 99 };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, -7, A.call(f64, I, .work, &services, .{ .layout = 102, .n = 3, .a = null, .lda = 3, .pivots = null, .work = &work, .lwork = 0 }));
        try std.testing.expectEqualSlices(f64, &.{ 3, 99 }, &work);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        work[0] = -7;
        try s.expectInfo(I, -4, A.call(f64, I, .work, &services, .{ .layout = 101, .n = -1, .a = null, .lda = -2, .pivots = null, .work = &work, .lwork = -1 }));
        try std.testing.expectEqual(@as(f64, -7), work[0]);
    }
}

test "GETRI unmanaged high call captures services before a getter replaces their allocator and callback" {
    var ctx: s.Context = .{};
    var services = ctx.services();
    ctx.change_binding = &services;
    var a = [_]f64{ 2, 1, 0, 4 };
    const pivots = [_]i64{ 1, -999 };
    try s.expectInfo(i64, 0, A.call(f64, i64, .high, &services, .{ .layout = 101, .n = 2, .a = &a, .lda = 2, .pivots = &pivots }));
    try std.testing.expectEqualSlices(f64, &.{ 0.5, -0.125, 0, 0.25 }, &a);
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 3), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
}

test "GETRI shared profile defaults capture once across both wide OOM callbacks and preserve raw and typed managed routes" {
    const cfg = owner.shared_services;
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(1);
    const Recorder = struct {
        rig: *h.Rig,
        calls: usize = 0,
        fn event(p: *anyopaque, e: owner.common.ErrorEvent) void {
            const self: *@This() = @ptrCast(@alignCast(p));
            if (e.routine != .getri or e.argument != -1010 or e.kind != .lapacke) @panic("wrong GETRI profile callback");
            if (e.entry != (if (self.calls == 0) owner.common.Entry.work else owner.common.Entry.high)) @panic("GETRI callback entry order");
            self.rig.record(.hook, e.argument);
            self.calls += 1;
            _ = cfg.setCallbacks(.{ .lapacke = .{ .userdata = self, .function = unexpected } });
        }
        fn unexpected(_: *anyopaque, _: owner.common.ErrorEvent) void {
            @panic("GETRI profile reread a replacement callback");
        }
    };
    var rig: h.Rig = .{};
    rig.init();
    rig.x.fail_at = 2;
    var recorder: Recorder = .{ .rig = &rig };
    var a = [_]f64{ 2, 1, 0, 4 };
    const before = a;
    const pivots = [_]i64{ 1, -999 };
    try s.expectInfo(i64, -1010, try owner.Shared.getriRawManagedWithDefaults(true, f64, i64, .high, rig.context, .{ .layout = 101, .n = 2, .a = a[0..].ptr, .lda = 2, .pivots = pivots[0..].ptr }, .{ .lapacke = .{ .userdata = &recorder, .function = Recorder.event } }));
    try s.bytes(&before, &a);
    try std.testing.expectEqual(@as(usize, 2), recorder.calls);
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().getter_calls);
    var hooks: usize = 0;
    for (rig.events[0..rig.count]) |event| if (event.tag == .hook) {
        try std.testing.expectEqual(@as(usize, 1), event.active);
        try std.testing.expectEqual(@as(usize, if (hooks == 0) 1 else 0), event.pins);
        hooks += 1;
    };
    _ = cfg.setCallbacks(.{});
    var work = [_]f64{-7};
    try s.expectInfo(i64, 0, try owner.Shared.getriRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .n = 0, .a = @ptrFromInt(1), .lda = 1, .pivots = @ptrFromInt(1), .work = work[0..].ptr, .lwork = 1 }));
    try s.expectInfo(i64, 0, try owner.Shared.getriManaged(false, f64, i64, .work, rig.context, .{ .layout = 102, .n = 0, .a = null, .lda = 1, .pivots = null, .work = &work, .lwork = 1 }));
    try std.testing.expectEqual(@as(usize, 3), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().getter_calls);
    var admissions: usize = 0;
    var finishes: usize = 0;
    for (rig.events[0..rig.count]) |event| {
        if (event.tag == .admitted) admissions += 1;
        if (event.tag == .finishing) finishes += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), admissions);
    try std.testing.expectEqual(@as(usize, 3), finishes);
    try rig.done();
}

fn fixtureProgressClose(ctx: *@import("managed_support.zig").Core.Execution.Context) @import("managed_support.zig").Core.Execution.CloseResult {
    var report: @import("managed_support.zig").Core.Execution.CallReport = .{};
    return ctx.tryProgressClose(&report);
}
