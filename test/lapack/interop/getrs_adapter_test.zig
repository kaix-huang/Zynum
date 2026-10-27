const std = @import("std");
const owner = @import("numerical-owner");
const actual_core = @import("zynum-core");
const actual_execution = @import("zynum-execution");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const Core = owner.Provider;
const A = owner.GetrsAdapter;
const R = owner.RawGetrsAdapter;
const C = Core.Execution.context;
const common = owner.common;
const cfg = owner.shared_services;

fn fault(expected: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |err| try std.testing.expectEqual(expected, @as(anyerror, err)),
        .lapacke_info => return error.ExpectedSeparateAccessFault,
    }
}
fn Adapter(comptime raw: bool) type {
    return if (raw) R else A;
}
fn invoke(comptime raw: bool, comptime T: type, comptime I: type, level: common.Level, services: *const Core.Services, layout: c_int, trans: u8, n: I, nrhs: I, a: []const T, lda: I, pivots: []const I, b: []T, ldb: I) Adapter(raw).Result(I) {
    return Adapter(raw).call(T, I, level, services, .{ .layout = layout, .trans = trans, .n = n, .nrhs = nrhs, .a = if (raw) a.ptr else a, .lda = lda, .pivots = if (raw) pivots.ptr else pivots, .b = if (raw) b.ptr else b, .ldb = ldb });
}
fn args(a: ?[]const f64, pivots: ?[]const i64, b: ?[]f64) A.Arguments(f64, i64) {
    return .{ .layout = 101, .trans = 'N', .n = 2, .nrhs = 1, .a = a, .lda = 2, .pivots = pivots, .b = b, .ldb = 1 };
}
fn close(r: *h.Rig) void {
    r.done() catch @panic("GETRS fixture cleanup");
}
const Z = [2]f64;
fn mul(a: Z, b: Z) Z {
    return .{ a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0] };
}
fn add(a: Z, b: Z) Z {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn coord(comptime T: type, a: T) Z {
    return .{ s.re(T, a), s.im(T, a) };
}
fn checkValue(comptime T: type, expected: Z, actual: T) !void {
    try std.testing.expectApproxEqAbs(expected[0], s.re(T, actual), 0.00003);
    try std.testing.expectApproxEqAbs(expected[1], s.im(T, actual), 0.00003);
}

// Independent supplied L/U and noncommuting swaps (1<->3, then 2<->3).
// Build A by a dense product and the inverse known row cycle, not GETRF or any
// solve implementation. The oracle forms op(A)*X directly in f64 coordinates.
fn knownSolve(comptime T: type, comptime I: type, comptime raw: bool, layout: c_int, trans: u8, level: common.Level) !void {
    const im: f64 = if (Core.Scalar.isComplex(T)) 1 else 0;
    const l = [3][3]Z{ .{ .{ 1, 0 }, .{ 0, 0 }, .{ 0, 0 } }, .{ .{ 0.5, 0.25 * im }, .{ 1, 0 }, .{ 0, 0 } }, .{ .{ -0.25, 0.5 * im }, .{ 0.75, -0.25 * im }, .{ 1, 0 } } };
    const u = [3][3]Z{ .{ .{ 2, 0.5 * im }, .{ -1, 0.25 * im }, .{ 1.5, -0.5 * im } }, .{ .{ 0, 0 }, .{ 3, -0.5 * im }, .{ 0.5, 0.75 * im } }, .{ .{ 0, 0 }, .{ 0, 0 }, .{ 4, 0.25 * im } } };
    const x = [3][2]Z{ .{ .{ 1, 0.2 * im }, .{ -2, 0.3 * im } }, .{ .{ 3, -0.4 * im }, .{ 0.5, 0.5 * im } }, .{ .{ -1, 0.6 * im }, .{ 2, -0.7 * im } } };
    var dense: [3][3]Z = undefined;
    for (0..3) |i| for (0..3) |j| {
        var z: Z = .{ 0, 0 };
        for (0..3) |k| z = add(z, mul(l[(i + 1) % 3][k], u[k][j]));
        dense[i][j] = z;
    };
    var factor: [18]T = @splat(s.value(T, std.math.nan(f64), -79));
    var b: [18]T = @splat(s.value(T, std.math.nan(f64), -81));
    for (0..3) |i| for (0..3) |j| {
        const z = if (i > j) l[i][j] else u[i][j];
        factor[s.pos(layout, i, j, 5)] = s.value(T, z[0], z[1]);
    };
    for (0..3) |i| for (0..2) |j| {
        var z: Z = .{ 0, 0 };
        for (0..3) |k| {
            var a = if (trans == 'N' or trans == 'n') dense[i][k] else dense[k][i];
            if (trans == 'C' or trans == 'c') a[1] = -a[1];
            z = add(z, mul(a, x[k][j]));
        }
        b[s.pos(layout, i, j, 5)] = s.value(T, z[0], z[1]);
    };
    const before_a = factor;
    const before_b = b;
    const pivots = [_]I{ 3, 3, 3, -777 };
    const before_p = pivots;
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(I, 0, invoke(raw, T, I, level, &services, layout, trans, 3, 2, &factor, 5, &pivots, &b, 5));
    for (0..3) |i| for (0..2) |j| try checkValue(T, x[i][j], b[s.pos(layout, i, j, 5)]);
    // Residual from the independently supplied full A and computed X.
    for (0..3) |i| for (0..2) |j| {
        var z: Z = .{ 0, 0 };
        for (0..3) |k| {
            var a = if (trans == 'N' or trans == 'n') dense[i][k] else dense[k][i];
            if (trans == 'C' or trans == 'c') a[1] = -a[1];
            z = add(z, mul(a, coord(T, b[s.pos(layout, k, j, 5)])));
        }
        try checkValue(T, z, before_b[s.pos(layout, i, j, 5)]);
    };
    for (0..b.len) |p| {
        const row = if (layout == 101) p / 5 else p % 5;
        const column = if (layout == 101) p % 5 else p / 5;
        if (row >= 3 or column >= 2) try s.bytes(before_b[p..][0..1], b[p..][0..1]);
    }
    try s.bytes(&before_a, &factor);
    try s.bytes(&before_p, &pivots);
    try std.testing.expectEqual(@as(usize, if (layout == 101) 2 else 0), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
}

test "GETRS reuses actual Core LU readonly Pivot and one shared service default" {
    comptime {
        if (A.OwnerIdentity != actual_core or R.OwnerIdentity != actual_core or Core.Execution != actual_execution or Core.LU != actual_core.internal.numerics.LU or Core.Pivot != actual_core.internal.numerics.Pivot)
            @compileError("GETRS must use the actual common owner");
        if (A.Services != owner.GetrfAdapter.Services or A.Dense != owner.SolveAdapter.Dense or A.ScratchLease(f64) != Core.Resources.ScratchLease(f64))
            @compileError("GETRS must not instantiate resources or service state");
        if (@FieldType(A.Arguments(f64, i64), "pivots") != ?[]const i64 or @FieldType(R.Arguments(f32, i32), "pivots") != ?[*]align(1) const i32 or @FieldType(A.Arguments(s.C64, i32), "a") != ?[]const s.C64)
            @compileError("factor and pivots must remain borrowed readonly operands");
        if (@FieldType(A.Result(i32), "admission_fault") != (common.access.TypedFault || error{InvalidPivot}))
            @compileError("InvalidPivot is local admission, never a fabricated INFO");
    }
    try std.testing.expect(actual_execution.getContext() == Core.Execution.getContext());
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(0);
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        const a = [_]T{s.value(T, 2, 0)};
        const p = [_]I{1};
        var b = [_]T{s.value(T, 8, 0)};
        try s.expectInfo(I, 0, try owner.Shared.getrsDefault(true, T, I, .high, .{ .layout = 102, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .pivots = &p, .b = &b }));
        try checkValue(T, .{ 4, 0 }, b[0]);
        b[0] = s.value(T, 8, 0);
        try s.expectInfo(I, 0, try owner.Shared.getrsRawDefault(false, T, I, .high, .{ .layout = 102, .trans = 'C', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .pivots = p[0..].ptr, .b = b[0..].ptr }));
        try checkValue(T, .{ 4, 0 }, b[0]);
    };
    try std.testing.expectEqual(@as(usize, 16), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 8), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
}

test "GETRS independent noncommuting LU known X and residuals cover NTC cases all scalars widths layouts and access" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ("NTCntc") |trans| for ([_]common.Level{ .high, .work }) |level| {
            try knownSolve(T, I, raw, layout, trans, level);
        };
    };
}

test "GETRS exact scalar INFO shift channels and row allocation precedence preserve output" {
    const Case = struct { layout: c_int = 102, trans: u8 = 'N', n: i64 = 0, nrhs: i64 = 0, lda: i64 = 1, ldb: i64 = 1, info: i64, argument: i32, allocations: usize = 0 };
    const cases = [_]Case{
        .{ .layout = 0, .trans = 'H', .n = -1, .info = -1, .argument = -1 },
        .{ .trans = 'H', .n = -1, .info = -2, .argument = 1 },
        .{ .trans = 'h', .info = -2, .argument = 1 },
        .{ .n = -1, .nrhs = -1, .info = -3, .argument = 2 },
        .{ .nrhs = -1, .info = -4, .argument = 3 },
        .{ .lda = 0, .ldb = 0, .info = -6, .argument = 5 },
        .{ .ldb = 0, .info = -9, .argument = 8 },
        .{ .layout = 101, .trans = 'H', .n = 2, .nrhs = 1, .lda = 1, .ldb = 0, .info = -6, .argument = -6 },
        .{ .layout = 101, .trans = 'H', .n = 2, .nrhs = 1, .lda = 2, .ldb = 0, .info = -9, .argument = -9 },
        .{ .layout = 101, .trans = 'H', .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .info = -2, .argument = 1, .allocations = 2 },
        .{ .layout = 101, .n = -1, .nrhs = 0, .lda = 0, .ldb = 0, .info = -3, .argument = 2, .allocations = 2 },
        .{ .layout = 101, .n = 2, .nrhs = -1, .lda = 2, .ldb = 0, .info = -4, .argument = 3, .allocations = 2 },
    };
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for (cases) |c| {
        const a = [_]T{ s.value(T, 2, 0), s.value(T, 1, 0), s.value(T, 0.5, 0), s.value(T, 3, 0) };
        var b = [_]T{ s.value(T, 4, 0), s.value(T, 8, 0) };
        const before = b;
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, @intCast(c.info), R.call(T, I, .work, &services, .{ .layout = c.layout, .trans = c.trans, .n = @intCast(c.n), .nrhs = @intCast(c.nrhs), .lda = @intCast(c.lda), .ldb = @intCast(c.ldb), .a = a[0..].ptr, .b = b[0..].ptr, .pivots = @ptrFromInt(1) }));
        try s.bytes(&before, &b);
        try std.testing.expectEqual(c.allocations, ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(@as(usize, 0), ctx.reads);
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(c.argument, ctx.hooks[0].argument);
        try std.testing.expectEqual(common.Routine.getrs, ctx.hooks[0].routine);
        try std.testing.expectEqual(if (c.argument > 0) common.Entry.fortran_getrs else .work, ctx.hooks[0].entry);
        try std.testing.expectEqual(if (c.argument > 0) common.HookKind.fortran_returning_test else .lapacke, ctx.hooks[0].kind);
    };
}

test "GETRS high general A then B NaN scans precede trans scalars and pivot reads" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var a = [_]T{ s.value(T, 2, 0), s.value(T, std.math.nan(f64), 0) };
        var b = [_]T{ s.value(T, std.math.nan(f64), 0), s.value(T, 8, 0) };
        const before = b;
        var ctx: s.Context = .{};
        const services = ctx.services();
        // lda=1 clips the full-general scan to two valid elements despite n=2.
        try s.expectInfo(i64, -5, R.call(T, i64, .high, &services, .{ .layout = layout, .trans = 'H', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr, .pivots = @ptrFromInt(1) }));
        a[1] = s.value(T, 3, 0);
        try s.expectInfo(i64, -8, R.call(T, i64, .high, &services, .{ .layout = layout, .trans = 'H', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr, .pivots = @ptrFromInt(1) }));
        try s.bytes(&before, &b);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts + ctx.hook_count);
        if (comptime Core.Scalar.isComplex(T)) {
            a[0] = s.value(T, 2, std.math.nan(f64));
            try s.expectInfo(i64, -5, A.call(T, i64, .high, &services, .{ .layout = layout, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b, .pivots = null }));
        }
    };
    var ctx: s.Context = .{};
    const services = ctx.services();
    // Layout refusal must not call the getter or resolve any invalid address.
    try s.expectInfo(i32, -1, R.call(f64, i32, .high, &services, .{ .layout = 0, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try std.testing.expectEqual(@as(usize, 0), ctx.reads);
    try std.testing.expectEqual(common.Entry.high, ctx.hooks[0].entry);
    var b = [_]f64{std.math.nan(f64)};
    try s.expectInfo(i32, -8, R.call(f64, i32, .high, &services, .{ .layout = 102, .trans = 'H', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = b[0..].ptr, .pivots = null }));
}

test "GETRS IEEE zero diagonal has no positive INFO and N zero RHS differs from transposed evaluation" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| {
        const p = [_]I{1};
        const a = [_]T{s.value(T, 0, 0)};
        for ("NTC") |trans| {
            var b = [_]T{s.value(T, 0, 0)};
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, 0, invoke(false, T, I, .work, &services, layout, trans, 1, 1, &a, 1, &p, &b, 1));
            if (trans == 'N') try checkValue(T, .{ 0, 0 }, b[0]) else try std.testing.expect(std.math.isNan(s.re(T, b[0])));
            try std.testing.expectEqual(@as(usize, 0), ctx.reads + ctx.hook_count);
        }
        const nan_a = [_]T{s.value(T, std.math.nan(f64), 0)};
        var b = [_]T{s.value(T, 1, 0)};
        var ctx: s.Context = .{ .enabled = false };
        const services = ctx.services();
        try s.expectInfo(I, 0, invoke(true, T, I, .high, &services, layout, 'N', 1, 1, &nan_a, 1, &p, &b, 1));
        try std.testing.expect(std.math.isNan(s.re(T, b[0])));
        ctx = .{};
        b[0] = s.value(T, 1, 0);
        try s.expectInfo(I, 0, owner.NoScanGetrsAdapter.call(T, I, .high, &services, .{ .layout = layout, .trans = 'C', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &nan_a, .pivots = &p, .b = &b }));
        try std.testing.expectEqual(@as(usize, 0), ctx.reads + ctx.hook_count);
        ctx = .{};
        const inf_a = [_]T{s.value(T, std.math.inf(f64), 0)};
        b[0] = s.value(T, 2, 0);
        try s.expectInfo(I, 0, invoke(true, T, I, .high, &services, layout, 'N', 1, 1, &inf_a, 1, &p, &b, 1));
        try checkValue(T, .{ 0, 0 }, b[0]);
    };
}

test "GETRS readonly one based pivots are fully revalidated every call before any B mutation" {
    inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const a = [_]f64{ 1, 0, 0, 1 };
        var p = [_]I{ 1, 2, -88 };
        var b = [_]f64{ 4, 8, 99 };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, invoke(raw, f64, I, .work, &services, layout, 'N', 2, 1, &a, 2, &p, &b, if (layout == 101) 1 else 2));
        for ([_]I{ -1, 0, 1, 3, std.math.maxInt(I) }) |bad| {
            ctx = .{};
            p[1] = bad;
            const before_b = b;
            const before_p = p;
            try fault(error.InvalidPivot, invoke(raw, f64, I, .work, &services, layout, 'C', 2, 1, &a, 2, &p, &b, if (layout == 101) 1 else 2));
            try s.bytes(&before_b, &b);
            try s.bytes(&before_p, &p);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count + ctx.live);
        }
    };
}

test "GETRS typed capacities and all three pairwise aliases reject before output changes" {
    for ([_]c_int{ 101, 102 }) |layout| for (0..3) |slot| {
        const a = [_]f64{ 1, 0, 0, 1 };
        const p = [_]i64{ 1, 2 };
        var b = [_]f64{ 4, 8 };
        const before = b;
        var ctx: s.Context = .{};
        const services = ctx.services();
        try fault(error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = layout, .trans = 'N', .n = 2, .nrhs = 1, .lda = 2, .ldb = if (layout == 101) 1 else 2, .a = if (slot == 0) a[0..3] else &a, .b = if (slot == 1) b[0..1] else &b, .pivots = if (slot == 2) p[0..1] else &p }));
        try s.bytes(&before, &b);
        try std.testing.expectEqual(@as(usize, 0), ctx.live + ctx.hook_count);
    };
    for ([_]c_int{ 101, 102 }) |layout| for (0..3) |pair| {
        // Both interpretations borrow initialized, aligned actual storage. The
        // overlap check fires before invalid integer encodings are inspected.
        var storage: [12]u64 = @splat(0);
        const numbers: [*]f64 = @ptrCast(&storage);
        const integers: [*]i32 = @ptrCast(&storage);
        numbers[0] = 1;
        numbers[3] = 1;
        numbers[6] = 4;
        numbers[7] = 8;
        const separate_p = [_]i32{ 1, 2 };
        const before = storage;
        var ctx: s.Context = .{};
        const services = ctx.services();
        try fault(error.AliasingNotAllowed, A.call(f64, i32, .work, &services, .{ .layout = layout, .trans = 'N', .n = 2, .nrhs = 1, .lda = 2, .ldb = if (layout == 101) 1 else 2, .a = numbers[0..4], .b = if (pair == 0) numbers[1..3] else numbers[6..8], .pivots = if (pair == 0) &separate_p else if (pair == 1) integers[1..3] else integers[13..15] }));
        try s.bytes(&before, &storage);
        try std.testing.expectEqual(@as(usize, 0), ctx.live + ctx.hook_count);
    };
}

test "GETRS raw faults are stage local and reject pointer alignment endpoints and sizes without reads" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| for (0..9) |which| {
        const a = [_]T{s.value(T, 2, 0)};
        var b = [_]T{s.value(T, 8, 0)};
        const p = [_]I{1};
        const before = b;
        const slot = which / 3;
        const kind = which % 3;
        const bad_a: ?[*]align(1) const T = if (kind == 0) null else if (kind == 1) @ptrFromInt(1) else @ptrFromInt(std.math.maxInt(usize) - (@alignOf(T) - 1));
        const bad_b: ?[*]align(1) T = if (kind == 0) null else if (kind == 1) @ptrFromInt(1) else @ptrFromInt(std.math.maxInt(usize) - (@alignOf(T) - 1));
        const bad_p: ?[*]align(1) const I = if (kind == 0) null else if (kind == 1) @ptrFromInt(1) else @ptrFromInt(std.math.maxInt(usize) - (@alignOf(I) - 1));
        var ctx: s.Context = .{};
        const services = ctx.services();
        try fault(if (kind == 0) error.MissingBuffer else if (kind == 1) error.MisalignedPointer else error.SizeOverflow, R.call(T, I, .work, &services, .{ .layout = layout, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = if (slot == 0) bad_a else a[0..].ptr, .b = if (slot == 1) bad_b else b[0..].ptr, .pivots = if (slot == 2) bad_p else p[0..].ptr }));
        try s.bytes(&before, &b);
        try std.testing.expectEqual(@as(usize, 0), ctx.live + ctx.hook_count);
    };
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(error.MisalignedPointer, R.call(f64, i64, .high, &services, .{ .layout = 102, .trans = 'H', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = null, .pivots = null }));
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    try fault(if (@bitSizeOf(usize) < 64) error.IntegerNotRepresentable else error.SizeOverflow, R.call(f64, i64, .work, &services, .{ .layout = 102, .trans = 'N', .n = 2, .nrhs = 1, .lda = std.math.maxInt(i64), .ldb = 2, .a = null, .b = null, .pivots = @ptrFromInt(1) }));
}

test "GETRS empty column nonreads row allocations and wide strides retain stage semantics" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        for ([_][2]I{ .{ 0, 0 }, .{ 0, 2 }, .{ 2, 0 } }) |shape| {
            try s.expectInfo(I, 0, R.call(T, I, .work, &services, .{ .layout = 102, .trans = 'C', .n = shape[0], .nrhs = shape[1], .lda = std.math.maxInt(I), .ldb = std.math.maxInt(I), .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        }
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts + ctx.reads);
        for ([_]I{ 0, 2 }) |rhs| {
            ctx = .{};
            try s.expectInfo(I, 0, R.call(T, I, .high, &services, .{ .layout = 101, .trans = 'N', .n = 0, .nrhs = rhs, .lda = 0, .ldb = rhs, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
            try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
        }
        ctx = .{};
        const a = [_]T{s.value(T, 2, 0)};
        try s.expectInfo(I, 0, R.call(T, I, .work, &services, .{ .layout = 101, .trans = 'N', .n = 1, .nrhs = 0, .lda = 1, .ldb = 0, .a = a[0..].ptr, .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
        ctx = .{};
        try fault(error.MisalignedPointer, R.call(T, I, .high, &services, .{ .layout = 102, .trans = 'N', .n = 1, .nrhs = 0, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = null, .pivots = null }));
        ctx = .{};
        try fault(error.MissingBuffer, R.call(T, I, .work, &services, .{ .layout = 101, .trans = 'N', .n = 1, .nrhs = 0, .lda = 1, .ldb = 0, .a = null, .b = null, .pivots = null }));
        try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
    };
    if (comptime @bitSizeOf(usize) >= 64) for ([_]c_int{ 101, 102 }) |layout| {
        const a = [_]f64{2};
        const p = [_]i64{1};
        var b = [_]f64{8};
        var ctx: s.Context = .{};
        const services = ctx.services();
        const wide: i64 = @as(i64, std.math.maxInt(i32)) + 1;
        try s.expectInfo(i64, 0, invoke(false, f64, i64, .work, &services, layout, 'N', 1, 1, &a, wide, &p, &b, wide));
        try std.testing.expectEqual(@as(f64, 4), b[0]);
    };
}

test "GETRS first second row OOM and post first allocation extent failure precede invalid trans and pointers" {
    for (0..2) |fail_index| {
        var ctx: s.Context = .{ .fail_index = fail_index };
        const services = ctx.services();
        try s.expectInfo(i64, -1011, R.call(f64, i64, .work, &services, .{ .layout = 101, .trans = 'H', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(fail_index + 1, ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(common.Entry.work, ctx.hooks[0].entry);
        try std.testing.expectEqual(@as(i32, -1011), ctx.hooks[0].argument);
        if (fail_index == 1) try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 8 }, .{ .tag = 2, .argument = 2, .value = 8 }, .{ .tag = 3, .argument = 1, .value = 8 }, .{ .tag = 4, .argument = -1011, .value = 2 } });
    }
    var ctx: s.Context = .{};
    const services = ctx.services();
    try fault(if (@bitSizeOf(usize) < 64) error.IntegerNotRepresentable else error.SizeOverflow, R.call(f64, i64, .work, &services, .{ .layout = 101, .trans = 'H', .n = 1, .nrhs = std.math.maxInt(i64), .lda = 1, .ldb = std.math.maxInt(i64), .a = null, .b = null, .pivots = null }));
    try std.testing.expectEqual(@as(usize, 1), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.live + ctx.hook_count);
    try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 8 }, .{ .tag = 3, .argument = 1, .value = 8 } });
}

test "GETRS row returning argument hook retains Reference copyback before reverse frees" {
    inline for (.{ false, true }) |raw| {
        const a = [_]f64{ 1, 0, 0, 1 };
        var b = [_]f64{ 10, 77, 20 };
        var ctx: s.Context = .{ .mutate_f64 = &b };
        const services = ctx.services();
        try s.expectInfo(i64, -2, Adapter(raw).call(f64, i64, .work, &services, .{ .layout = 101, .trans = 'H', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = if (raw) a[0..].ptr else &a, .pivots = null, .b = if (raw) b[0..].ptr else &b }));
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, &b);
        try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 32 }, .{ .tag = 2, .argument = 2, .value = 16 }, .{ .tag = 5, .argument = 1, .value = 2 }, .{ .tag = 3, .argument = 2, .value = 16 }, .{ .tag = 3, .argument = 1, .value = 32 } });
    }
}

test "GETRS unmanaged Services capture is stable across getter replacement" {
    const a = [_]f64{2};
    const p = [_]i64{1};
    var b = [_]f64{8};
    var ctx: s.Context = .{};
    var services = ctx.services();
    ctx.change_binding = &services;
    try s.expectInfo(i64, -2, A.call(f64, i64, .high, &services, .{ .layout = 101, .trans = 'H', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .pivots = &p, .b = &b }));
    try std.testing.expectEqual(@as(f64, 8), b[0]);
    try std.testing.expectEqual(@as(usize, 2), ctx.attempts);
    try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
    try std.testing.expectEqual(@as(usize, 0), ctx.live);
}

test "GETRS managed admission and resource overflow remain distinct and precede borrowed reads" {
    for ([_]bool{ false, true }) |overflow| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        var held: ?C.Operation = null;
        if (overflow) {
            C.Context.testing.configure(r.context, 1, null);
            held = try r.context.begin();
        } else _ = r.context.requestClose();
        defer if (held) |*op| op.end();
        var services: Core.Services = undefined;
        const expected = if (overflow) error.CounterOverflow else error.Closing;
        try std.testing.expectError(expected, A.callManaged(f64, i64, .high, r.context, &services, args(null, null, null)));
        try std.testing.expectError(expected, R.callManaged(f64, i64, .high, r.context, &services, .{ .layout = 102, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 0), r.count);
    }
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    C.Context.testing.configure(r.context, 1, null);
    const a = [_]f64{ 2, 1, 0.5, 3 };
    const p = [_]i64{ 1, 2 };
    var b = [_]f64{ 4, 8 };
    try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .work, r.context, &r.services, args(&a, &p, &b)));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts); // A acquired; B pin refuses before allocator.
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
    try std.testing.expectEqualSlices(f64, &.{ 4, 8 }, &b);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
}

test "GETRS managed captures Context allocator and callbacks and both OOM hooks follow pin cleanup" {
    for (0..2) |fail_index| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        r.rebind = true;
        r.services.allocator = undefined;
        r.x.fail_at = fail_index;
        const a = [_]f64{ 2, 1, 0.5, 3 };
        const p = [_]i64{ 1, 2 };
        var b = [_]f64{ 4, 8 };
        try s.expectInfo(i64, -1011, try A.callManaged(f64, i64, .high, r.context, &r.services, args(&a, &p, &b)));
        try std.testing.expectEqual(fail_index + 1, r.x.attempts);
        try std.testing.expectEqual(fail_index, r.x.frees);
        try std.testing.expectEqual(@as(usize, 0), r.y.attempts + r.z.attempts);
        try std.testing.expectEqual(@as(usize, 1), r.hooks);
        for (r.events[0..r.count]) |e| if (e.tag == .hook) {
            try std.testing.expectEqual(@as(usize, 1), e.active);
            try std.testing.expectEqual(@as(usize, 0), e.pins);
            try std.testing.expectEqual(@as(i32, -1011), e.detail);
        };
        try std.testing.expectEqualSlices(f64, &.{ 4, 8 }, &b);
    }
}

test "GETRS close from getter allocation hook and free retains one operation through real solve" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .hook, .free }) |trigger| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        r.trigger = trigger;
        r.action = .close_nested;
        const a = [_]f64{ 2, 1, 0.5, 3 };
        const p = [_]i64{ 1, 2 };
        var b = [_]f64{ 4, 8 };
        var f = args(&a, &p, &b);
        if (trigger == .hook) f.trans = 'H';
        try s.expectInfo(i64, if (trigger == .hook) -2 else 0, try A.callManaged(f64, i64, .high, r.context, &r.services, f));
        try std.testing.expectEqual(error.Closing, r.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        var starts: usize = 0;
        var ends: usize = 0;
        var frees: usize = 0;
        for (r.events[0..r.count]) |e| {
            if (e.tag == .admitted) starts += 1;
            if (e.tag == .free) {
                try std.testing.expectEqual(@as(usize, 1), e.active);
                try std.testing.expectEqual(@as(usize, 2) - frees, e.pins);
                frees += 1;
            }
            if (e.tag == .finishing) {
                ends += 1;
                try std.testing.expectEqual(@as(usize, 2), frees);
                try std.testing.expectEqual(@as(usize, 0), e.pins);
            }
        }
        try std.testing.expectEqual(@as(usize, 1), starts);
        try std.testing.expectEqual(@as(usize, 1), ends);
        try std.testing.expectEqualSlices(f64, if (trigger == .hook) &.{ 4, 8 } else &.{ 1, 2 }, &b);
    }
}

const FreeAudit = struct {
    reference_context: FixtureReferenceContext = undefined,
    context: *C.Context = undefined,
    a: [4]f64 = .{ 2, 1, 0.5, 3 },
    pivots: [2]i64 = .{ 1, 2 },
    b: [3]f64 = .{ 4, 77, 8 },
    pointers: [2]?[*]u8 = .{ null, null },
    attempts: usize = 0,
    frees: usize = 0,
    fn allocator(self: *FreeAudit) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FreeAudit = @ptrCast(@alignCast(raw));
        const p = std.testing.allocator.rawAlloc(len, alignment, ra) orelse return null;
        self.pointers[self.attempts] = p;
        self.attempts += 1;
        return p;
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, mem: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FreeAudit = @ptrCast(@alignCast(raw));
        if (self.attempts != 2 or self.frees >= 2 or self.pointers[1 - self.frees] != mem.ptr) @panic("GETRS reverse captured allocator order");
        if (!std.mem.eql(f64, &self.b, &.{ 1, 77, 2 }) or !std.mem.eql(f64, &self.a, &.{ 2, 1, 0.5, 3 }) or !std.mem.eql(i64, &self.pivots, &.{ 1, 2 })) @panic("GETRS copyback/readonly inputs before free");
        const state = self.context.snapshot();
        if (state.active_operations != 1 or state.resource_pins != 2 - self.frees) @panic("GETRS free operation/pin ownership");
        if (self.context.requestClose() != .pending or fixtureProgressClose(self.context) != .busy) @panic("GETRS close during actual free");
        std.testing.allocator.rawFree(mem, alignment, ra);
        const after = self.context.snapshot();
        if (after.active_operations != 1 or after.resource_pins != state.resource_pins) @panic("GETRS unpinned before allocator return");
        self.frees += 1;
    }
};

test "GETRS typed and raw actual copyback precedes reverse captured free unpin and final operation end" {
    inline for (.{ false, true }) |raw| {
        var audit: FreeAudit = .{};
        audit.reference_context.initInPlace(audit.allocator(), actual_execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        audit.context = audit.reference_context.borrowedContext();
        var ctx: s.Context = .{};
        var services = ctx.services();
        services.allocator = undefined;
        try s.expectInfo(i64, 0, try Adapter(raw).callManaged(f64, i64, .work, audit.context, &services, .{ .layout = 101, .trans = 'N', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = if (raw) audit.a[0..].ptr else &audit.a, .pivots = if (raw) audit.pivots[0..].ptr else &audit.pivots, .b = if (raw) audit.b[0..].ptr else &audit.b }));
        try std.testing.expectEqual(@as(usize, 2), audit.frees);
        try std.testing.expectEqual(@as(usize, 0), audit.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), audit.context.snapshot().resource_pins);
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(audit.context));
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    }
}

test "GETRS shared typed and raw routes capture once with one admission and refuse before capture" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(0);
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    const a = [_]f64{ 2, 1, 0.5, 3 };
    const p = [_]i64{ 1, 2 };
    var b = [_]f64{ 4, 8 };
    try s.expectInfo(i64, 0, try owner.Shared.getrsManaged(true, f64, i64, .high, r.context, args(&a, &p, &b)));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, &b);
    b = .{ 4, 8 };
    try s.expectInfo(i64, 0, try owner.Shared.getrsRawManaged(true, f64, i64, .high, r.context, .{ .layout = 101, .trans = 'N', .n = 2, .nrhs = 1, .lda = 2, .ldb = 1, .a = a[0..].ptr, .pivots = p[0..].ptr, .b = b[0..].ptr }));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, &b);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
    var starts: usize = 0;
    for (r.events[0..r.count]) |e| if (e.tag == .admitted) {
        starts += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), starts);
    _ = r.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.getrsRawManaged(true, f64, i64, .high, r.context, .{ .layout = 0, .trans = 'N', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().getter_calls);
}

const PausedSolve = struct {
    rig: *h.Rig,
    b: [2]f64 = .{ 4, 8 },
    info: ?i64 = null,
    err: ?anyerror = null,
    finished: std.atomic.Value(bool) = .init(false),
    fn run(self: *PausedSolve) void {
        defer self.finished.store(true, .release);
        const a = [_]f64{ 2, 1, 0.5, 3 };
        const p = [_]i64{ 1, 2 };
        const result = A.callManaged(f64, i64, .work, self.rig.context, &self.rig.services, args(&a, &p, &self.b)) catch |err| {
            self.err = err;
            return;
        };
        switch (result) {
            .lapacke_info => |info| self.info = info,
            .admission_fault => |err| self.err = err,
        }
    }
};

test "GETRS another thread cannot close while the actual B scratch allocator free is paused" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    r.pause_free = true;
    var work: PausedSolve = .{ .rig = &r };
    const thread = try std.Thread.spawn(.{}, PausedSolve.run, .{&work});
    var joined = false;
    defer if (!joined) {
        r.continue_free.store(true, .release);
        thread.join();
    };
    while (!r.paused.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(r.paused.load(.acquire));
    const state = r.context.snapshot();
    try std.testing.expectEqual(@as(usize, 1), state.active_operations);
    try std.testing.expectEqual(@as(usize, 2), state.resource_pins);
    try std.testing.expectEqual(C.RequestResult.pending, r.context.requestClose());
    try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(r.context));
    // Release publication precedes reading worker results; the real body has
    // completed and the first allocator return is the controlled boundary.
    r.continue_free.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expectEqual(@as(?anyerror, null), work.err);
    try std.testing.expectEqual(@as(?i64, 0), work.info);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, &work.b);
    try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(r.context));
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
