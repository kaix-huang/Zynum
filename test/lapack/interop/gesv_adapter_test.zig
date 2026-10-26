// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");

test "GESV shared raw route admits once and rejects a closing Context before operands" {
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("GESV shared fixture cleanup");
    var a = [_]f64{ 0, 2, 1, 3 };
    var b = [_]f64{ 6, 10 };
    var pivots: [2]i32 = undefined;
    try s.expectInfo(i32, 0, try owner.Shared.gesvRawManaged(false, f64, i32, .work, &rig.context, .{ .layout = 101, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr, .pivots = pivots[0..].ptr }));
    try std.testing.expectEqualSlices(f64, &.{ 1, 3 }, &b);
    var starts: usize = 0;
    var ends: usize = 0;
    for (rig.events[0..rig.count]) |event| {
        if (event.tag == .admitted) starts += 1;
        if (event.tag == .finishing) ends += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), starts);
    try std.testing.expectEqual(@as(usize, 1), ends);
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.gesvRawManaged(false, f64, i32, .work, &rig.context, undefined));
}

test "GESV parameter and NaN diagnostics preserve caller outputs" {
    inline for (.{ false, true }) |raw| {
        const A = if (raw) owner.RawGesvAdapter else owner.GesvAdapter;
        for (0..4) |which| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            var a = [_]f64{ 0, 2, 1, 3 };
            var b = [_]f64{ 6, 10 };
            var pivots = [_]i32{ -9, -9 };
            if (which == 3) b[0] = std.math.nan(f64);
            const before_b = b;
            const result = A.call(f64, i32, if (which == 3) .high else .work, &services, .{ .layout = if (which == 0) -1 else if (which == 2) 102 else 101, .n = if (which == 2) -1 else 2, .nrhs = 1, .lda = 2, .ldb = if (which == 1) 0 else if (which == 3) 1 else 2, .a = if (raw) a[0..].ptr else &a, .b = if (raw) b[0..].ptr else &b, .pivots = if (raw) pivots[0..].ptr else &pivots });
            try s.expectInfo(i32, ([_]i32{ -1, -8, -2, -7 })[which], result);
            try std.testing.expectEqualSlices(f64, &.{ 0, 2, 1, 3 }, &a);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_b), std.mem.asBytes(&b));
            try std.testing.expectEqualSlices(i32, &.{ -9, -9 }, &pivots);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
        }
    }
}

test "GESV complex coefficients and solutions satisfy independent matrix products" {
    inline for (.{ s.C32, s.C64 }) |T| {
        for ([_]c_int{ 101, 102 }) |layout| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            const original = [_]T{ s.value(T, 0, 1), s.value(T, 2, -1), s.value(T, 3, 2), s.value(T, 1, 1) };
            const expected = [_]T{ s.value(T, 1, -2), s.value(T, -3, 1) };
            var a: [4]T = undefined;
            var b: [2]T = undefined;
            for (0..2) |i| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (0..2) |j| {
                    const v = original[i * 2 + j];
                    const x = expected[j];
                    a[s.pos(layout, i, j, 2)] = v;
                    re += s.re(T, v) * s.re(T, x) - s.im(T, v) * s.im(T, x);
                    im += s.re(T, v) * s.im(T, x) + s.im(T, v) * s.re(T, x);
                }
                b[i] = s.value(T, re, im);
            }
            var pivots: [2]i64 = undefined;
            try s.expectInfo(i64, 0, owner.GesvAdapter.call(T, i64, .high, &services, .{ .layout = layout, .n = 2, .nrhs = 1, .lda = 2, .ldb = if (layout == 101) 1 else 2, .a = &a, .b = &b, .pivots = &pivots }));
            for (b, expected) |actual, wanted| {
                try std.testing.expectApproxEqAbs(s.re(T, wanted), s.re(T, actual), 0.00001);
                try std.testing.expectApproxEqAbs(s.im(T, wanted), s.im(T, actual), 0.00001);
            }
        }
    }
}

test "GESV managed closing during callbacks retains one operation through all frees" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .hook, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        defer rig.done() catch @panic("GESV fixture cleanup");
        rig.trigger = trigger;
        rig.action = .close_nested;
        var a = [_]f64{ 0, 2, 1, 3 };
        var b = [_]f64{ 6, 10 };
        var pivots: [2]i64 = undefined;
        const result = try owner.GesvAdapter.callManaged(f64, i64, .high, &rig.context, &rig.services, .{ .layout = 101, .n = if (trigger == .hook) -1 else 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = &a, .b = &b, .pivots = &pivots });
        try s.expectInfo(i64, if (trigger == .hook) -2 else 0, result);
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        var starts: usize = 0;
        var ends: usize = 0;
        var frees: usize = 0;
        for (rig.events[0..rig.count]) |event| {
            if (event.tag == .admitted) starts += 1;
            if (event.tag == .free) {
                frees += 1;
                try std.testing.expectEqual(@as(usize, 1), event.active);
            }
            if (event.tag == .finishing) {
                ends += 1;
                try std.testing.expectEqual(@as(usize, 2), frees);
                try std.testing.expectEqual(@as(usize, 0), event.pins);
            }
        }
        try std.testing.expectEqual(@as(usize, 1), starts);
        try std.testing.expectEqual(@as(usize, 1), ends);
        if (trigger != .hook) try std.testing.expectEqualSlices(f64, &.{ 1, 3 }, &b);
    }
}

test "GESV diagnostic names use routine length and preserve argument signs" {
    inline for (.{ i32, i64 }) |I| {
        const D = owner.Diagnostics(I);
        const high = try D.translate(.{ .routine = .gesv, .scalar = 'd', .kind = .lapacke, .entry = .high, .argument = -1 });
        const work = try D.translate(.{ .routine = .gesv, .scalar = 'd', .kind = .lapacke, .entry = .work, .argument = -8 });
        const underlying = try D.translate(.{ .routine = .gesv, .scalar = 'd', .kind = .fortran_returning_test, .entry = .fortran_gesv, .argument = 7 });
        try std.testing.expectEqualStrings("LAPACKE_dgesv", high.name());
        try std.testing.expectEqualStrings("LAPACKE_dgesv_work", work.name());
        try std.testing.expectEqualStrings("DGESV", underlying.name());
        try std.testing.expectEqual(@as(I, -8), work.info);
        try std.testing.expectEqual(@as(I, 7), underlying.info);
    }
}

test "GESV typed and raw solve pivoted multiple RHS across scalars layouts and widths" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            inline for (.{ false, true }) |raw| {
                const A = if (raw) owner.RawGesvAdapter else owner.GesvAdapter;
                for ([_]c_int{ 101, 102 }) |layout| {
                    var ctx: s.Context = .{};
                    const services = ctx.services();
                    var a: [6]T = @splat(s.value(T, 91, 0));
                    var b: [6]T = @splat(s.value(T, 92, 0));
                    const av = [_]f64{ 0, 2, 1, 3 };
                    const bv = [_]f64{ 6, 8, 10, 14 };
                    for (0..2) |i| for (0..2) |j| {
                        a[s.pos(layout, i, j, 3)] = s.value(T, av[i * 2 + j], 0);
                        b[s.pos(layout, i, j, 3)] = s.value(T, bv[i * 2 + j], 0);
                    };
                    var pivots: [2]I = undefined;
                    const result = A.call(T, I, .high, &services, .{ .layout = layout, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = if (raw) a[0..].ptr else &a, .b = if (raw) b[0..].ptr else &b, .pivots = if (raw) pivots[0..].ptr else &pivots });
                    try s.expectInfo(I, 0, result);
                    for (0..2) |i| for (0..2) |j| {
                        const value = b[s.pos(layout, i, j, 3)];
                        try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 2 + j + 1)), s.re(T, value), 0.00001);
                        try std.testing.expectEqual(@as(f64, 0), s.im(T, value));
                    };
                    try std.testing.expectEqual(@as(I, 2), pivots[0]);
                    try std.testing.expectEqual(@as(f64, 91), s.re(T, a[2]));
                    try std.testing.expectEqual(@as(f64, 92), s.re(T, b[2]));
                    try std.testing.expectEqual(@as(usize, 0), ctx.live);
                }
            }
        }
    }
}

test "GESV singular factorization leaves RHS unchanged and zero RHS still factors" {
    for ([_]c_int{ 101, 102 }) |layout| {
        for ([_]i32{ 0, 1 }) |rhs| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            var a = [_]f64{ 1, 2, 2, 4 };
            var b = [_]f64{ 7, 11 };
            var pivots = [_]i32{ -9, -9 };
            const result = owner.GesvAdapter.call(f64, i32, .work, &services, .{ .layout = layout, .n = 2, .nrhs = rhs, .lda = 2, .ldb = if (layout == 101) rhs else 2, .a = &a, .b = if (rhs == 0) null else &b, .pivots = &pivots });
            try s.expectInfo(i32, 2, result);
            try std.testing.expectEqualSlices(f64, &.{ 7, 11 }, &b);
            try std.testing.expectEqual(@as(i32, 2), pivots[0]);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
        }
    }
}

test "GESV second row workspace failure frees first allocation before diagnostic" {
    var ctx: s.Context = .{ .fail_index = 1 };
    const services = ctx.services();
    var a = [_]f64{ 0, 2, 1, 3 };
    var b = [_]f64{ 6, 10 };
    var pivots = [_]i32{ -9, -9 };
    const result = owner.GesvAdapter.call(f64, i32, .work, &services, .{ .layout = 101, .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = &a, .b = &b, .pivots = &pivots });
    try s.expectInfo(i32, -1011, result);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 1, 3 }, &a);
    try std.testing.expectEqualSlices(f64, &.{ 6, 10 }, &b);
    try std.testing.expectEqualSlices(i32, &.{ -9, -9 }, &pivots);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    try std.testing.expectEqual(owner.common.Routine.gesv, ctx.hooks[0].routine);
}
