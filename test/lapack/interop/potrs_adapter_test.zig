const std = @import("std");
const s = @import("potrs_support.zig");
const Core = s.Core;
const A = s.Adapter;
const owner = s.owner;

test "POTRF and POTRS share owner types services resources and numerical core" {
    try std.testing.expect(A.Services == owner.Adapter.Services and A.OwnerIdentity == owner.Adapter.OwnerIdentity);
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| {
        try std.testing.expect(A.Dense.Matrix(T) == owner.Adapter.Dense.Matrix(T));
        try std.testing.expect(A.ScratchLease(T) == Core.Resources.ScratchLease(T));
    }
    try std.testing.expect(s.C32 == owner.implementation.ComplexF32 and s.C64 == owner.implementation.lapack.ComplexF64);
    var ctx = s.Context{};
    const services = ctx.services();
    var factor = [_]f64{4};
    var b = [_]f64{8};
    try s.expectInfo(i32, 0, owner.Adapter.call(f64, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &factor }));
    try s.expectInfo(i32, 0, A.call(f64, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &factor, .lda = 1, .b = &b, .ldb = 1 }));
    try std.testing.expectEqual(@as(f64, 2), b[0]);
    try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
}

test "all scalar integer layout triangle and entry variants solve independent integer Gram fixtures" {
    @setEvalBranchQuota(10000);
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| inline for (.{ owner.protocol.Level.high, owner.protocol.Level.work }) |level| {
            for (0..if (Core.Scalar.isComplex(T)) @as(usize, 2) else 1) |variant| {
                const complex = Core.Scalar.isComplex(T);
                const lre = if (variant == 1) [2][2]f64{ .{ 2, 0 }, .{ 1, 3 } } else [2][2]f64{ .{ 2, 0 }, .{ 1, 3 } };
                const lim = if (variant == 1) [2][2]f64{ .{ 1, 0 }, .{ -2, -1 } } else if (complex) [2][2]f64{ .{ 0, 0 }, .{ 1, 0 } } else [2][2]f64{ .{ 0, 0 }, .{ 0, 0 } };
                const bre = if (variant == 1) [2][2]f64{ .{ 10, 0 }, .{ 35, -15 } } else if (complex) [2][2]f64{ .{ 6, 10 }, .{ 22, -7 } } else [2][2]f64{ .{ 8, 6 }, .{ 22, -6 } };
                const bim = if (variant == 1) [2][2]f64{ .{ 15, -5 }, .{ -20, 20 } } else if (complex) [2][2]f64{ .{ -2, 6 }, .{ -7, 26 } } else [2][2]f64{ .{ 0, 0 }, .{ 0, 0 } };
                const xre = [2][2]f64{ .{ 1, 2 }, .{ 2, -1 } };
                const xim = if (complex) [2][2]f64{ .{ 1, 0 }, .{ -1, 2 } } else [2][2]f64{ .{ 0, 0 }, .{ 0, 0 } };
                var a: [10]T = @splat(s.value(T, std.math.nan(f64), -73));
                var b: [12]T = @splat(s.value(T, -91, 63));
                for (0..2) |r| for (0..2) |c| {
                    if (if (upper) r <= c else r >= c) a[s.pos(layout, r, c, 4)] = s.value(T, lre[if (upper) c else r][if (upper) r else c], (if (upper) @as(f64, -1) else 1) * lim[if (upper) c else r][if (upper) r else c]);
                    b[s.pos(layout, r, c, 5)] = s.value(T, bre[r][c], bim[r][c]);
                };
                const before_a = a;
                const before_b = b;
                var ctx = s.Context{};
                const services = ctx.services();
                try s.expectInfo(I, 0, A.call(T, I, level, &services, .{ .layout = layout, .uplo = if (upper) 'u' else 'l', .n = 2, .nrhs = 2, .a = &a, .lda = 4, .b = &b, .ldb = 5 }));
                try s.bytes(before_a[0..], a[0..]);
                try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 0), ctx.attempts);
                try std.testing.expectEqual(@as(usize, 0), ctx.live);
                for (0..2) |r| for (0..2) |c| {
                    const z = b[s.pos(layout, r, c, 5)];
                    try std.testing.expectApproxEqAbs(xre[r][c], s.re(T, z), 0.000003);
                    try std.testing.expectApproxEqAbs(xim[r][c], s.im(T, z), 0.000003);
                    // Dense Gram*X residual, independent of substitutions.
                    var rr: f64 = 0;
                    var ii: f64 = 0;
                    for (0..2) |j| {
                        var gr: f64 = 0;
                        var gi: f64 = 0;
                        for (0..2) |k| {
                            gr += lre[r][k] * lre[j][k] + lim[r][k] * lim[j][k];
                            gi += lim[r][k] * lre[j][k] - lre[r][k] * lim[j][k];
                        }
                        const x = b[s.pos(layout, j, c, 5)];
                        rr += gr * s.re(T, x) - gi * s.im(T, x);
                        ii += gr * s.im(T, x) + gi * s.re(T, x);
                    }
                    try std.testing.expectApproxEqAbs(bre[r][c], rr, 0.00004);
                    try std.testing.expectApproxEqAbs(bim[r][c], ii, 0.00004);
                };
                for (0..b.len) |p| {
                    const r = if (layout == 101) p / 5 else p % 5;
                    const c = if (layout == 101) p % 5 else p / 5;
                    if (r >= 2 or c >= 2) try s.bytes(before_b[p..][0..1], b[p..][0..1]);
                }
            }
        };
    };
}

test "full complex diagonal scalar oracles and scaled huge tiny division" {
    inline for (.{ s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| {
        const R = Core.Scalar.Real(T);
        const power: i32 = if (R == f32) 80 else 600;
        for ([_]i32{ 0, 1, 2, power, -power }) |which| {
            var a = [_]T{if (which == 0) s.value(T, 2, 1) else if (which == 1) s.value(T, 0, 2) else if (which == 2) s.value(T, -2, 0) else s.value(T, std.math.ldexp(@as(f64, 1), which), std.math.ldexp(@as(f64, 1), which))};
            var b = [_]T{if (which == 0) s.value(T, 10, 0) else if (which == 1 or which == 2) s.value(T, 8, 0) else s.value(T, std.math.ldexp(@as(f64, 1), which), 0)};
            const before = a;
            var ctx = s.Context{};
            const services = ctx.services();
            try s.expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = layout, .uplo = uplo, .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
            try std.testing.expectApproxEqRel(if (which <= 2 and which >= 0) @as(f64, 2) else std.math.ldexp(@as(f64, 1), -which - 1), s.re(T, b[0]), 0.000001);
            try std.testing.expectEqual(@as(f64, 0), s.im(T, b[0]));
            try s.bytes(before[0..], a[0..]);
        }
    };
}

test "ordered scalar priorities and distinct C versus returning Fortran error events" {
    const Case = struct { layout: c_int, uplo: u8, n: i64, nrhs: i64, lda: i64, ldb: i64, info: i64, argument: i32 };
    const cases = [_]Case{
        .{ .layout = 0, .uplo = '?', .n = -1, .nrhs = -1, .lda = 0, .ldb = 0, .info = -1, .argument = -1 },
        .{ .layout = 102, .uplo = '?', .n = -1, .nrhs = -1, .lda = 0, .ldb = 0, .info = -2, .argument = 1 },
        .{ .layout = 102, .uplo = 'L', .n = -1, .nrhs = -1, .lda = 0, .ldb = 0, .info = -3, .argument = 2 },
        .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = -1, .lda = 0, .ldb = 0, .info = -4, .argument = 3 },
        .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = 0, .lda = 0, .ldb = 0, .info = -6, .argument = 5 },
        .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 0, .lda = 2, .ldb = 0, .info = -8, .argument = 7 },
        .{ .layout = 101, .uplo = '?', .n = 2, .nrhs = 2, .lda = 1, .ldb = 0, .info = -6, .argument = -6 },
        .{ .layout = 101, .uplo = 'L', .n = -1, .nrhs = 2, .lda = 0, .ldb = 1, .info = -8, .argument = -8 },
        .{ .layout = 101, .uplo = 'L', .n = -1, .nrhs = 2, .lda = 0, .ldb = 2, .info = -3, .argument = 2 },
        .{ .layout = 101, .uplo = 'L', .n = 2, .nrhs = -1, .lda = 2, .ldb = 0, .info = -4, .argument = 3 },
    };
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for (cases) |c| {
        var a = [_]T{ s.value(T, 2, 0), s.value(T, 0, 0), s.value(T, 0, 0), s.value(T, 2, 0) };
        var ctx = s.Context{ .enabled = false };
        const services = ctx.services();
        try s.expectInfo(I, @intCast(c.info), A.call(T, I, .high, &services, .{ .layout = c.layout, .uplo = c.uplo, .n = @intCast(c.n), .nrhs = @intCast(c.nrhs), .a = &a, .lda = @intCast(c.lda), .b = null, .ldb = @intCast(c.ldb) }));
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(c.argument, ctx.hooks[0].argument);
        try std.testing.expectEqual(.potrs, ctx.hooks[0].routine);
        try std.testing.expectEqual(if (c.argument < 0) owner.protocol.HookKind.lapacke else .fortran_returning_test, ctx.hooks[0].kind);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    };
}

test "NaN A then B order clipped scans and ignored poison are distinct from finite checks" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var a = [_]T{ s.value(T, std.math.nan(f64), 0), s.value(T, 0, 0), s.value(T, 0, 0), s.value(T, 2, 0) };
        var b = [_]T{ s.value(T, std.math.nan(f64), 0), s.value(T, 1, 0), s.value(T, 9, 0), s.value(T, 9, 0) };
        var ctx = s.Context{};
        const services = ctx.services();
        try s.expectInfo(i64, -5, A.call(T, i64, .high, &services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 1, .a = &a, .lda = 2, .b = &b, .ldb = 2 }));
        try s.expectInfo(i64, -7, A.call(T, i64, .high, &services, .{ .layout = layout, .uplo = '?', .n = 2, .nrhs = 1, .a = &a, .lda = 2, .b = &b, .ldb = 2 }));
        try s.expectInfo(i64, -5, A.call(T, i64, .high, &services, .{ .layout = layout, .uplo = 'U', .n = 3, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 2 }));
        a[0] = s.value(T, 2, 0);
        try s.expectInfo(i64, -7, A.call(T, i64, .high, &services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 2, .a = &a, .lda = 2, .b = &b, .ldb = 1 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 4), ctx.reads);
        if (comptime Core.Scalar.isComplex(T)) {
            a[0] = s.value(T, 2, std.math.nan(f64));
            try s.expectInfo(i64, -5, A.call(T, i64, .high, &services, .{ .layout = layout, .uplo = 'L', .n = 1, .nrhs = 0, .a = &a, .lda = 2, .b = null, .ldb = 2 }));
        }
    };
}

test "work disabled scan and mutable NaN setting share one captured Services" {
    var ctx = s.Context{};
    var services = ctx.services();
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| {
        var a = [_]T{s.value(T, std.math.nan(f64), 0)};
        var b = [_]T{s.value(T, 1, 0)};
        try s.expectInfo(i32, -5, A.call(T, i32, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
        ctx.enabled = false;
        try s.expectInfo(i32, 0, A.call(T, i32, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
        ctx.enabled = true;
        const reads = ctx.reads;
        try s.expectInfo(i32, 0, A.call(T, i32, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
        try s.expectInfo(i32, 0, owner.NoScanSolveAdapter.call(T, i32, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
        try std.testing.expectEqual(reads, ctx.reads);
    }
    ctx = s.Context{};
    services = ctx.services();
    ctx.change_binding = &services;
    var b = [_]f64{3};
    try s.expectInfo(i32, -2, A.call(f64, i32, .high, &services, .{ .layout = 101, .uplo = '?', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = &b, .ldb = 1 }));
    try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
}

test "row empty acquisitions and selected A reads survive Fortran quick return" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var ctx = s.Context{};
        const services = ctx.services();
        try s.expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 0, .nrhs = 0, .a = null, .lda = 0, .b = null, .ldb = 0 }));
        const size: @TypeOf(@as(i64, 0)) = @sizeOf(T);
        try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = size }, .{ .tag = 2, .argument = 2, .value = size }, .{ .tag = 3, .argument = 2, .value = size }, .{ .tag = 3, .argument = 1, .value = size } });
        ctx = s.Context{};
        try s.expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = 3, .a = null, .lda = std.math.maxInt(I), .b = null, .ldb = std.math.maxInt(I) }));
        try std.testing.expectEqual(@as(usize, 0), ctx.count);
        try s.expectFault(error.MissingBuffer, A.call(T, I, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 2, .nrhs = 0, .a = null, .lda = 2, .b = null, .ldb = 0 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    };
}

test "two allocator failures have ordered release before C hook and no caller mutation" {
    for ([_]usize{ 0, 1 }) |fail| {
        var ctx = s.Context{ .fail_index = fail };
        const services = ctx.services();
        var b = [_]f64{4};
        const before = b;
        try s.expectInfo(i32, -1011, A.call(f64, i32, .work, &services, .{ .layout = 101, .uplo = '?', .n = 1, .nrhs = 1, .a = null, .lda = 1, .b = &b, .ldb = 1 }));
        if (fail == 0) try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 8 }, .{ .tag = 4, .argument = -1011, .value = 2 } }) else try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 8 }, .{ .tag = 2, .argument = 2, .value = 8 }, .{ .tag = 3, .argument = 1, .value = 8 }, .{ .tag = 4, .argument = -1011, .value = 2 } });
        try s.bytes(before[0..], b[0..]);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    }
}

test "row returning uplo hook precedes B copyback and reverse releases" {
    var b = [_]f64{ 3, 7, 11, 13, 17, 19 };
    const before = b;
    var ctx = s.Context{ .mutate_f64 = &b };
    const services = ctx.services();
    try s.expectInfo(i64, -2, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = '?', .n = 2, .nrhs = 1, .a = null, .lda = 2, .b = &b, .ldb = 3 }));
    try std.testing.expectEqual(before[0], b[0]);
    try std.testing.expectEqual(before[3], b[3]);
    try std.testing.expectEqual(@as(f64, -900), b[1]);
    try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 32 }, .{ .tag = 2, .argument = 2, .value = 16 }, .{ .tag = 5, .argument = 1, .value = 2 }, .{ .tag = 3, .argument = 2, .value = 16 }, .{ .tag = 3, .argument = 1, .value = 32 } });
}

test "admission capacity alias and second extent overflow remain separate from INFO" {
    for ([_]c_int{ 101, 102 }) |layout| {
        var data = [_]f64{ 2, 0, 0, 2, 8, 8, 8, 8 };
        const before = data;
        var ctx = s.Context{};
        const services = ctx.services();
        try s.expectFault(error.AliasingNotAllowed, A.call(f64, i64, .work, &services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 1, .a = data[0..4], .lda = 2, .b = data[0..4], .ldb = 2 }));
        try s.bytes(before[0..], data[0..]);
        ctx = s.Context{};
        try s.expectFault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 1, .a = data[0..1], .lda = 2, .b = data[4..], .ldb = 2 }));
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try s.bytes(before[0..], data[0..]);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    }
    var ctx = s.Context{};
    const services = ctx.services();
    try s.expectFault(if (@bitSizeOf(usize) < 64) error.IntegerNotRepresentable else error.SizeOverflow, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 1, .nrhs = std.math.maxInt(i64), .a = null, .lda = 1, .b = null, .ldb = std.math.maxInt(i64) }));
    try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 8 }, .{ .tag = 3, .argument = 1, .value = 8 } });
}

test "native width actual one element strides and empty no conversion" {
    const wide: i64 = @as(i64, std.math.maxInt(i32)) + 1;
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var a = [_]T{s.value(T, 2, 0)};
        var b = [_]T{s.value(T, 8, 0)};
        var ctx = s.Context{};
        const services = ctx.services();
        try s.expectInfo(i64, 0, A.call(T, i64, .work, &services, .{ .layout = layout, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = wide, .b = &b, .ldb = wide }));
        try std.testing.expectEqual(@as(f64, 2), s.re(T, b[0]));
        try s.expectInfo(i64, 0, A.call(T, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 0, .nrhs = 0, .a = null, .lda = std.math.maxInt(i64), .b = null, .ldb = std.math.maxInt(i64) }));
        if (@bitSizeOf(usize) < 64) {
            const too_wide: i64 = @as(i64, std.math.maxInt(u32)) + 1;
            try s.expectFault(error.IntegerNotRepresentable, A.call(T, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .a = &a, .lda = too_wide, .b = &b, .ldb = too_wide }));
        }
    };
}

test "IEEE exceptional solves return no positive INFO and continue all RHS" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| {
        const R = Core.Scalar.Real(T);
        var a = [_]T{s.value(T, 0.5, 0)};
        var b: [12]T = @splat(s.value(T, 79, -31));
        b[s.pos(layout, 0, 0, 4)] = s.value(T, 1, if (Core.Scalar.isComplex(T)) 1 else 0);
        b[s.pos(layout, 0, 1, 4)] = s.value(T, 0.375 * @as(f64, std.math.floatMax(R)), 0);
        b[s.pos(layout, 0, 2, 4)] = s.value(T, 2, if (Core.Scalar.isComplex(T)) 1 else 0);
        const before = b;
        var ctx = s.Context{};
        const services = ctx.services();
        try s.expectInfo(i64, 0, A.call(T, i64, .work, &services, .{ .layout = layout, .uplo = uplo, .n = 1, .nrhs = 3, .a = &a, .lda = 1, .b = &b, .ldb = 4 }));
        try std.testing.expectEqual(@as(f64, 4), s.re(T, b[0]));
        try std.testing.expect(std.math.isInf(s.re(T, b[s.pos(layout, 0, 1, 4)])));
        try std.testing.expectEqual(@as(f64, 8), s.re(T, b[s.pos(layout, 0, 2, 4)]));
        for (0..b.len) |p| {
            if (p != s.pos(layout, 0, 0, 4) and p != s.pos(layout, 0, 1, 4) and p != s.pos(layout, 0, 2, 4)) try s.bytes(before[p..][0..1], b[p..][0..1]);
        }
        if (comptime Core.Scalar.isComplex(T)) {
            a[0] = s.value(T, 0, 0);
            b[0] = s.value(T, 1, 1);
            try s.expectInfo(i64, 0, A.call(T, i64, .work, &services, .{ .layout = layout, .uplo = uplo, .n = 1, .nrhs = 1, .a = &a, .lda = 1, .b = &b, .ldb = 1 }));
            if (uplo == 'L') {
                try std.testing.expect(std.math.isNan(b[0].re) and std.math.isNan(b[0].im));
            } else {
                try std.testing.expect(std.math.isPositiveInf(b[0].re) and std.math.isPositiveInf(b[0].im));
            }
        }
    };
}
