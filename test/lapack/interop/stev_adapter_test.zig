// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
test "STEV typed raw layouts widths and precisions preserve eigenpair residuals" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            inline for (.{ false, true }) |raw| {
                const A = if (raw) owner.RawStevAdapter else owner.StevAdapter;
                for ([_]c_int{ 101, 102 }) |layout| {
                    var ctx: s.Context = .{};
                    const services = ctx.services();
                    var d = [_]T{ 2, 2, 2 };
                    var e = [_]T{ -1, -1 };
                    var z: [11]T = @splat(99);
                    var work: [4]T = undefined;
                    try s.expectInfo(I, 0, A.call(T, I, .high, &services, .{ .layout = layout, .jobz = 'V', .n = 3, .d = if (raw) d[0..].ptr else &d, .e = if (raw) e[0..].ptr else &e, .z = if (raw) z[0..].ptr else &z, .ldz = 4, .work = if (raw) work[0..].ptr else &work }));
                    for (0..3) |j| {
                        for (0..3) |i| {
                            const v = z[s.pos(layout, i, j, 4)];
                            var av = 2 * v;
                            if (i > 0) av -= z[s.pos(layout, i - 1, j, 4)];
                            if (i < 2) av -= z[s.pos(layout, i + 1, j, 4)];
                            try std.testing.expectApproxEqAbs(d[j] * v, av, 0.00001);
                        }
                    }
                    try std.testing.expectEqual(@as(T, 99), z[3]);
                    try std.testing.expectEqual(@as(T, 99), z[7]);
                    try std.testing.expectEqual(@as(usize, 0), ctx.live);
                }
            }
        }
    }
}
test "STEV values only ignores Z and WORK and does not allocate" {
    var ctx: s.Context = .{};
    const services = ctx.services();
    var d = [_]f64{ 2, 2 };
    var e = [_]f64{-1};
    try s.expectInfo(i32, 0, owner.StevAdapter.call(f64, i32, .high, &services, .{ .layout = 102, .jobz = 'N', .n = 2, .d = &d, .e = &e, .z = undefined, .ldz = 1, .work = undefined }));
    try std.testing.expectApproxEqAbs(@as(f64, 1), d[0], 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 3), d[1], 1e-14);
    try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
}
test "STEV allocation failures release earlier workspace" {
    for (0..2) |fail| {
        var ctx: s.Context = .{ .fail_index = fail };
        const services = ctx.services();
        var d = [_]f64{ 2, 2 };
        var e = [_]f64{-1};
        var z: [4]f64 = undefined;
        try s.expectInfo(i64, if (fail == 0) -1010 else -1011, owner.StevAdapter.call(f64, i64, .high, &services, .{ .layout = 101, .jobz = 'V', .n = 2, .d = &d, .e = &e, .z = &z, .ldz = 2, .work = null }));
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqualSlices(f64, &.{ 2, 2 }, &d);
    }
}
test "STEV managed lifetime ends after all scratch frees" {
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("cleanup");
    var d = [_]f64{ 2, 2 };
    var e = [_]f64{-1};
    var z: [4]f64 = undefined;
    try s.expectInfo(i32, 0, try owner.StevAdapter.callManaged(f64, i32, .high, rig.context, &rig.services, .{ .layout = 101, .jobz = 'V', .n = 2, .d = &d, .e = &e, .z = &z, .ldz = 2, .work = null }));
    var starts: usize = 0;
    var ends: usize = 0;
    var frees: usize = 0;
    for (rig.events[0..rig.count]) |ev| {
        if (ev.tag == .admitted) starts += 1;
        if (ev.tag == .free) {
            frees += 1;
            try std.testing.expectEqual(@as(usize, 1), ev.active);
        }
        if (ev.tag == .finishing) {
            ends += 1;
            try std.testing.expectEqual(@as(usize, 2), frees);
        }
    }
    try std.testing.expectEqual(@as(usize, 1), starts);
    try std.testing.expectEqual(@as(usize, 1), ends);
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.StevAdapter.callManaged(f64, i32, .high, rig.context, &rig.services, undefined));
}
test "STEV scalar validation order and diagnostic translation" {
    inline for (.{ i32, i64 }) |I| {
        for (0..5) |which| {
            var ctx: s.Context = .{};
            const services = ctx.services();
            const result = owner.StevAdapter.call(f64, I, .work, &services, .{ .layout = if (which == 0) 0 else if (which == 1) 101 else 102, .jobz = if (which < 3) '?' else 'N', .n = if (which == 3) -1 else 2, .d = null, .e = null, .z = null, .ldz = if (which == 1 or which == 4) 0 else 2, .work = null });
            try s.expectInfo(I, ([_]I{ -1, -7, -2, -3, -7 })[which], result);
            try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
            const msg = try owner.Diagnostics(I).translate(ctx.hooks[0]);
            try std.testing.expectEqualStrings(if (which < 2) "LAPACKE_dstev_work" else "DSTEV", msg.name());
            try std.testing.expectEqual(([_]I{ -1, -7, 1, 2, 6 })[which], msg.info);
        }
    }
}
test "STEV high NaN scan precedes job validation and work bypasses scan" {
    for (0..2) |which| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        var d = [_]f64{ 2, 2 };
        var e = [_]f64{1};
        if (which == 0) d[0] = std.math.nan(f64) else e[0] = std.math.nan(f64);
        try s.expectInfo(i32, if (which == 0) -4 else -5, owner.StevAdapter.call(f64, i32, .high, &services, .{ .layout = 102, .jobz = '?', .n = 2, .d = &d, .e = &e, .z = null, .ldz = 2, .work = null }));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try s.expectInfo(i32, -2, owner.StevAdapter.call(f64, i32, .work, &services, .{ .layout = 102, .jobz = '?', .n = 2, .d = &d, .e = &e, .z = null, .ldz = 2, .work = null }));
        try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    }
}
test "STEV zero and singleton honor inactive operands and padding" {
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i32, 0, owner.StevAdapter.call(f64, i32, .work, &services, .{ .layout = 102, .jobz = 'V', .n = 0, .d = null, .e = null, .z = null, .ldz = 1, .work = null }));
    var d = [_]f64{ 7, 98 };
    var z = [_]f64{ 99, 99 };
    try s.expectInfo(i32, 0, owner.StevAdapter.call(f64, i32, .work, &services, .{ .layout = 102, .jobz = 'V', .n = 1, .d = &d, .e = null, .z = &z, .ldz = 2, .work = undefined }));
    try std.testing.expectEqualSlices(f64, &.{ 7, 98 }, &d);
    try std.testing.expectEqualSlices(f64, &.{ 1, 99 }, &z);
}
test "STEV shared raw route uses single existing owner admission" {
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("cleanup");
    var d = [_]f64{ 2, 2 };
    var e = [_]f64{-1};
    var z: [4]f64 = undefined;
    var work: [2]f64 = undefined;
    try s.expectInfo(i64, 0, try owner.Shared.stevRawManaged(false, f64, i64, .work, rig.context, .{ .layout = 101, .jobz = 'V', .n = 2, .d = d[0..].ptr, .e = e[0..].ptr, .z = z[0..].ptr, .ldz = 2, .work = work[0..].ptr }));
    var starts: usize = 0;
    var ends: usize = 0;
    for (rig.events[0..rig.count]) |ev| {
        if (ev.tag == .admitted) starts += 1;
        if (ev.tag == .finishing) ends += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), starts);
    try std.testing.expectEqual(@as(usize, 1), ends);
    _ = rig.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.stevRawManaged(false, f64, i64, .work, rig.context, undefined));
}
test "STEV closing from callbacks retains scratch until operation completes" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .hook, .free }) |trigger| {
        var rig: h.Rig = .{};
        rig.init();
        defer rig.done() catch @panic("cleanup");
        rig.trigger = trigger;
        rig.action = .close_nested;
        var d = [_]f64{ 2, 2 };
        var e = [_]f64{-1};
        var z: [4]f64 = undefined;
        try s.expectInfo(i32, if (trigger == .hook) -3 else 0, try owner.StevAdapter.callManaged(f64, i32, .high, rig.context, &rig.services, .{ .layout = 101, .jobz = 'V', .n = if (trigger == .hook) -1 else 2, .d = &d, .e = &e, .z = &z, .ldz = 2, .work = null }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        var frees: usize = 0;
        var ends: usize = 0;
        for (rig.events[0..rig.count]) |event| {
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
        try std.testing.expectEqual(@as(usize, 1), ends);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
    }
}
test "STEV no scan preserves bounded IEEE nonconvergence behavior" {
    inline for (.{ false, true }) |raw| {
        const A = if (raw) owner.RawNoScanStevAdapter else owner.NoScanStevAdapter;
        var ctx: s.Context = .{};
        const services = ctx.services();
        var d = [_]f64{ 1, 2 };
        var e = [_]f64{std.math.nan(f64)};
        try s.expectInfo(i64, 1, A.call(f64, i64, .high, &services, .{ .layout = 102, .jobz = 'N', .n = 2, .d = if (raw) d[0..].ptr else &d, .e = if (raw) e[0..].ptr else &e, .z = undefined, .ldz = 1, .work = undefined }));
        try std.testing.expectEqual(@as(usize, 0), ctx.reads);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    }
}
