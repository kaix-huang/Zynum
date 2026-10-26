// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! POSV orchestration tests. Independent matrix products, no vendor output oracle.
const std = @import("std");
const owner = @import("numerical-owner");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const P = owner.PosvAdapter;
const Raw = owner.RawPosvAdapter;
const Z = struct { re: f64 = 0, im: f64 = 0 };
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn read(comptime T: type, a: T) Z {
    return .{ .re = s.re(T, a), .im = s.im(T, a) };
}
fn check(comptime T: type, expected: Z, actual: T) !void {
    const eps: f64 = if (T == f32 or T == s.C32) std.math.floatEps(f32) else std.math.floatEps(f64);
    const tol = 512 * eps * @max(1, @abs(expected.re) + @abs(expected.im));
    try std.testing.expectApproxEqAbs(expected.re, s.re(T, actual), tol);
    try std.testing.expectApproxEqAbs(expected.im, s.im(T, actual), tol);
}
fn same(comptime T: type, a: []const T, b: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn numericalCase(comptime T: type, comptime I: type, comptime raw: bool, layout: c_int, upper: bool, level: owner.common.Level) !void {
    const A = if (raw) Raw else P;
    const imag: f64 = if (owner.Core.Scalar.isComplex(T)) 1 else 0;
    // A = L L^H with strictly positive diagonal, assembled independently.
    const lower = [_]Z{ .{ .re = 2 }, .{}, .{}, .{ .re = 1, .im = imag }, .{ .re = 3 }, .{}, .{ .re = -1, .im = 0.5 * imag }, .{ .re = 2, .im = -imag }, .{ .re = 4 } };
    var original: [9]Z = @splat(.{});
    for (0..3) |i| for (0..3) |j| {
        for (0..3) |k| original[i * 3 + j] = add(original[i * 3 + j], mul(lower[i * 3 + k], conj(lower[j * 3 + k])));
    };
    const sentinel = s.value(T, -777, 37 * imag);
    var a: [15]T = @splat(sentinel);
    for (0..3) |i| for (0..3) |j| {
        if (if (upper) i <= j else i >= j) a[s.pos(layout, i, j, 5)] = s.value(T, original[i * 3 + j].re, original[i * 3 + j].im);
    };
    const before_a = a;
    var b: [12]T = @splat(sentinel);
    var expected: [6]Z = undefined;
    for (0..3) |i| for (0..2) |j| {
        expected[i * 2 + j] = .{ .re = @as(f64, @floatFromInt(1 + i + 2 * j)), .im = imag * (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(j))) };
    };
    for (0..3) |i| for (0..2) |j| {
        var v: Z = .{};
        for (0..3) |k| v = add(v, mul(original[i * 3 + k], expected[k * 2 + j]));
        b[s.pos(layout, i, j, 4)] = s.value(T, v.re, v.im);
    };
    const before_b = b;
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(I, 0, A.call(T, I, level, &services, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 3, .nrhs = 2, .lda = 5, .ldb = 4, .a = if (raw) a[0..].ptr else &a, .b = if (raw) b[0..].ptr else &b }));
    try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    for (0..3) |i| for (0..2) |j| try check(T, expected[i * 2 + j], b[s.pos(layout, i, j, 4)]);
    for (0..3) |i| for (0..3) |j| {
        var v: Z = .{};
        for (0..@min(i, j) + 1) |k| {
            const left = read(T, a[s.pos(layout, if (upper) k else i, if (upper) i else k, 5)]);
            const right = read(T, a[s.pos(layout, if (upper) k else j, if (upper) j else k, 5)]);
            v = add(v, if (upper) mul(conj(left), right) else mul(left, conj(right)));
        }
        try check(T, original[i * 3 + j], s.value(T, v.re, v.im));
    };
    for (a, 0..) |_, index| {
        var selected = false;
        for (0..3) |i| for (0..3) |j| {
            if ((if (upper) i <= j else i >= j) and s.pos(layout, i, j, 5) == index) selected = true;
        };
        if (!selected) try same(T, a[index..][0..1], before_a[index..][0..1]);
    }
    for (b, 0..) |_, index| {
        var selected = false;
        for (0..3) |i| for (0..2) |j| {
            if (s.pos(layout, i, j, 4) == index) selected = true;
        };
        if (!selected) try same(T, b[index..][0..1], before_b[index..][0..1]);
    }
    const factor = a;
    // Reuse the caller-owned factor after POSV return for two independent RHS.
    for (0..2) |batch| {
        var rhs: [12]T = @splat(sentinel);
        for (0..3) |i| {
            var v: Z = .{};
            for (0..3) |k| v = add(v, mul(original[i * 3 + k], .{ .re = @as(f64, @floatFromInt(k + batch + 2)), .im = imag * @as(f64, @floatFromInt(batch)) }));
            rhs[s.pos(layout, i, 0, 4)] = s.value(T, v.re, v.im);
        }
        var reuse: s.Context = .{};
        const srv = reuse.services();
        try s.expectInfo(I, 0, owner.SolveAdapter.call(T, I, .work, &srv, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 3, .nrhs = 1, .lda = 5, .ldb = 4, .a = &a, .b = &rhs }));
        for (0..3) |i| try check(T, .{ .re = @as(f64, @floatFromInt(i + batch + 2)), .im = imag * @as(f64, @floatFromInt(batch)) }, rhs[s.pos(layout, i, 0, 4)]);
        try same(T, &factor, &a);
        try std.testing.expectEqual(@as(usize, 0), reuse.live);
    }
}

test "POSV four scalars two integer widths both layouts and triangles with independent products and reusable factors" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]owner.common.Level{ .high, .work }) |level| try numericalCase(T, I, raw, layout, upper, level);
    };
}

test "POSV failure leading minors preserve RHS and zero RHS still factors A" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        const A = if (raw) Raw else P;
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for (0..3) |failure| for ([_]I{ 0, 2 }) |nrhs| {
            var a: [12]T = @splat(s.value(T, 31, 0));
            for (0..3) |i| for (0..3) |j| {
                if (if (upper) i <= j else i >= j) a[s.pos(layout, i, j, 4)] = s.value(T, if (i != j) 0 else if (i == failure) -1 else 4, 0);
            };
            var b: [12]T = @splat(s.value(T, 13, -2));
            const before = b;
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, @intCast(failure + 1), A.call(T, I, .work, &services, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 3, .nrhs = nrhs, .a = if (raw) a[0..].ptr else &a, .lda = 4, .b = if (raw) b[0..].ptr else &b, .ldb = 4 }));
            try same(T, &before, &b);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
            for (0..failure) |i| try check(T, .{ .re = 2 }, a[s.pos(layout, i, i, 4)]);
        };
    };
}

test "POSV high scan ordering and ordinary complex diagonal semantics" {
    inline for (.{ false, true }) |raw| inline for (.{ i32, i64 }) |I| {
        const A = if (raw) Raw else P;
        for (0..6) |which| {
            var a = [_]f64{if (which == 0) std.math.nan(f64) else if (which == 1 or which == 2) -1 else 4};
            var b = [_]f64{if (which <= 2) std.math.nan(f64) else 8};
            var ctx: s.Context = .{ .enabled = which != 4 };
            const services = ctx.services();
            const before_a = a;
            const before_b = b;
            try s.expectInfo(I, ([_]I{ -5, -7, -7, 0, 0, -2 })[which], A.call(f64, I, if (which == 3) .work else .high, &services, .{ .layout = 102, .uplo = if (which == 2 or which == 5) '?' else 'L', .n = 1, .nrhs = 1, .a = if (raw) a[0..].ptr else &a, .lda = 1, .b = if (raw) b[0..].ptr else &b, .ldb = 1 }));
            if (which <= 2 or which == 5) {
                try same(f64, &before_a, &a);
                try same(f64, &before_b, &b);
            } else {
                try check(f64, .{ .re = 2 }, b[0]);
            }
            try std.testing.expectEqual(@as(usize, if (which == 5) 1 else 0), ctx.hook_count);
        }
        inline for (.{ s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| {
            var a = [_]T{ s.value(T, 4, 7), s.value(T, 0, 0), s.value(T, 0, 0), s.value(T, 9, -8) };
            const unused = s.pos(layout, if (upper) 1 else 0, if (upper) 0 else 1, 2);
            a[unused] = s.value(T, std.math.nan(f64), std.math.nan(f64));
            const before = a[unused];
            var b = [_]T{ s.value(T, 8, 4), s.value(T, 18, -9) };
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, 0, A.call(T, I, .high, &services, .{ .layout = layout, .uplo = if (upper) 'U' else 'L', .n = 2, .nrhs = 1, .a = if (raw) a[0..].ptr else &a, .lda = 2, .b = if (raw) b[0..].ptr else &b, .ldb = if (layout == 101) 1 else 2 }));
            try check(T, .{ .re = 2, .im = 1 }, b[0]);
            try check(T, .{ .re = 2, .im = -1 }, b[1]);
            try same(T, @as(*const [1]T, &before), a[unused..][0..1]);
        };
    };
}

test "POSV parameter priority and POSV diagnostic identity" {
    const Case = struct { layout: c_int = 102, uplo: u8 = '?', n: i64 = -1, nrhs: i64 = -1, lda: i64 = 0, ldb: i64 = 0, info: i64, arg: i32, kind: owner.common.HookKind = .fortran_returning_test };
    const cases = [_]Case{
        .{ .layout = 0, .info = -1, .arg = -1, .kind = .lapacke },
        .{ .info = -2, .arg = 1 },
        .{ .uplo = 'L', .info = -3, .arg = 2 },
        .{ .uplo = 'L', .n = 1, .info = -4, .arg = 3 },
        .{ .uplo = 'L', .n = 1, .nrhs = 0, .info = -6, .arg = 5 },
        .{ .uplo = 'L', .n = 1, .nrhs = 0, .lda = 1, .info = -8, .arg = 7 },
        .{ .layout = 101, .n = 2, .nrhs = 3, .lda = 1, .info = -6, .arg = -6, .kind = .lapacke },
        .{ .layout = 101, .n = 2, .nrhs = 3, .lda = 2, .ldb = 2, .info = -8, .arg = -8, .kind = .lapacke },
        .{ .layout = 101, .n = 0, .nrhs = 0, .info = -2, .arg = 1 },
    };
    inline for (.{ false, true }) |raw| inline for (.{ i32, i64 }) |I| for (cases) |c| {
        const A = if (raw) Raw else P;
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, @intCast(c.info), A.call(f64, I, .work, &services, .{ .layout = c.layout, .uplo = c.uplo, .n = @intCast(c.n), .nrhs = @intCast(c.nrhs), .lda = @intCast(c.lda), .ldb = @intCast(c.ldb), .a = null, .b = null }));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        const event = ctx.hooks[0];
        try std.testing.expectEqual(owner.common.Routine.posv, event.routine);
        try std.testing.expectEqual(c.arg, event.argument);
        try std.testing.expectEqual(c.kind, event.kind);
        const msg = try owner.Diagnostics(I).translate(event);
        try std.testing.expectEqualStrings(if (c.kind == .lapacke) "LAPACKE_dposv_work" else "DPOSV", msg.name());
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    };
    // ILP64 scalar errors and quick return retain values beyond signed 32 bits.
    const large: i64 = @as(i64, std.math.maxInt(i32)) + 17;
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i64, -3, P.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = -large, .nrhs = 0, .lda = 1, .ldb = 1, .a = null, .b = null }));
    try s.expectInfo(i64, 0, P.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = large, .lda = 1, .ldb = large, .a = null, .b = null }));
}

test "POSV raw and typed admission faults precede caller mutation" {
    var ctx: s.Context = .{};
    const services = ctx.services();
    var a = [_]f64{ 4, 0, 0, 9 };
    const before = a;
    var b = [_]f64{ 8, 18 };
    const before_b = b;
    try s.expectFault(error.BufferTooSmall, P.call(f64, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = a[0..3], .b = &b }));
    try s.expectFault(error.AliasingNotAllowed, P.call(f64, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = &a, .b = a[0..2] }));
    try s.expectInfo(i32, 0, P.call(f64, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = 0, .lda = 1, .ldb = 1, .a = null, .b = null }));
    switch (Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = null, .b = b[0..].ptr })) {
        .admission_fault => |e| try std.testing.expectEqual(error.MissingBuffer, e),
        else => return error.TestUnexpectedResult,
    }
    const bad: [*]align(1) f64 = @ptrFromInt(1);
    switch (Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = bad, .b = b[0..].ptr })) {
        .admission_fault => |e| try std.testing.expectEqual(error.MisalignedPointer, e),
        else => return error.TestUnexpectedResult,
    }
    switch (Raw.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = std.math.maxInt(i64), .nrhs = 0, .lda = std.math.maxInt(i64), .ldb = std.math.maxInt(i64), .a = a[0..].ptr, .b = null })) {
        .admission_fault => |e| try std.testing.expectEqual(error.SizeOverflow, e),
        else => return error.TestUnexpectedResult,
    }
    try same(f64, &before, &a);
    try same(f64, &before_b, &b);
}

test "POSV row staging allocation failures preserve bytes and release before error hook" {
    inline for (.{ false, true }) |raw| for (0..2) |failure| {
        const A = if (raw) Raw else P;
        var ctx: s.Context = .{ .fail_index = failure };
        const services = ctx.services();
        var a = [_]f64{ 4, 0, 0, 9 };
        const before = a;
        var b = [_]f64{ 8, 18 };
        const before_b = b;
        try s.expectInfo(i32, -1011, A.call(f64, i32, .high, &services, .{ .layout = 101, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = if (raw) a[0..].ptr else &a, .b = if (raw) b[0..].ptr else &b }));
        try same(f64, &before, &a);
        try same(f64, &before_b, &b);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(failure + 1, ctx.attempts);
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(@as(i32, 4), ctx.events[ctx.count - 1].tag);
        if (failure == 1) try std.testing.expectEqual(@as(i32, 3), ctx.events[ctx.count - 2].tag);
    };
    for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(i64, 0, P.call(f64, i64, .work, &services, .{ .layout = layout, .uplo = 'L', .n = 0, .nrhs = 0, .lda = if (layout == 101) 0 else 1, .ldb = if (layout == 101) 0 else 1, .a = null, .b = null }));
        try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    }
}

test "POSV one admitted operation spans callbacks staged storage and reverse cleanup" {
    inline for (.{ false, true }) |raw| for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .free, .hook }) |trigger| {
        const A = if (raw) Raw else P;
        var rig: h.Rig = .{ .trigger = trigger, .action = .close_nested, .rebind = true };
        rig.init();
        defer rig.done() catch @panic("POSV cleanup");
        var a = [_]f64{ 4, 0, 0, 9 };
        var b = [_]f64{ 8, 18 };
        try s.expectInfo(i64, if (trigger == .hook) -2 else 0, try A.callManaged(f64, i64, .high, &rig.context, &rig.services, .{ .layout = 101, .uplo = if (trigger == .hook) '?' else 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = if (raw) a[0..].ptr else &a, .b = if (raw) b[0..].ptr else &b }));
        try std.testing.expectEqual(error.Closing, rig.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), rig.max_active);
        try std.testing.expectEqual(@as(usize, 2), rig.x.attempts);
        try std.testing.expectEqual(@as(usize, 2), rig.x.frees);
        try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
        var starts: usize = 0;
        var ends: usize = 0;
        for (rig.events[0..rig.count]) |event| {
            if (event.tag == .admitted) starts += 1;
            if (event.tag == .finishing) ends += 1;
            if (event.tag == .free) {
                try std.testing.expectEqual(@as(usize, 1), event.active);
                try std.testing.expect(event.pins > 0);
            }
        }
        try std.testing.expectEqual(@as(usize, 1), starts);
        try std.testing.expectEqual(@as(usize, 1), ends);
        if (trigger != .hook) try std.testing.expectEqualSlices(f64, &.{ 2, 2 }, &b);
        try std.testing.expectError(error.Closing, A.callManaged(f64, i64, .work, &rig.context, undefined, undefined));
    };
    var rig: h.Rig = .{};
    rig.init();
    defer rig.done() catch @panic("POSV shared cleanup");
    var a = [_]f64{ 4, 0, 0, 9 };
    var b = [_]f64{ 8, 18 };
    try s.expectInfo(i32, 0, try owner.Shared.posvRawManaged(false, f64, i32, .work, &rig.context, .{ .layout = 101, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr }));
    try std.testing.expectEqualSlices(f64, &.{ 2, 2 }, &b);
    try std.testing.expectEqual(@as(usize, 1), rig.max_active);
}
