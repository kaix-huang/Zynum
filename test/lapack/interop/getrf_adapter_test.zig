const std = @import("std");
const owner = @import("numerical-owner");
const actual_core = @import("zynum-core");
const actual_execution = @import("zynum-execution");
const s = @import("potrs_support.zig");
const h = @import("managed_support.zig");
const Core = owner.Provider;
const A = owner.GetrfAdapter;
const R = owner.RawGetrfAdapter;
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
fn invoke(comptime raw: bool, comptime T: type, comptime I: type, level: common.Level, services: *const Core.Services, layout: c_int, m: I, n: I, lda: I, values: []T, pivots: []I) Adapter(raw).Result(I) {
    return Adapter(raw).call(T, I, level, services, .{ .layout = layout, .m = m, .n = n, .lda = lda, .values = if (raw) values.ptr else values, .pivots = if (raw) pivots.ptr else pivots });
}
fn args(values: ?[]f64, pivots: ?[]i64) A.Arguments(f64, i64) {
    return .{ .layout = 101, .m = 2, .n = 2, .lda = 2, .values = values, .pivots = pivots };
}
fn close(r: *h.Rig) void {
    r.done() catch @panic("GETRF fixture cleanup");
}

// Independent scalar arithmetic for S*A=L*U, never a second factorization or
// an invocation of the checked wrapper/core as an oracle. Work in f64 complex
// coordinates, including for the f32 input/output representations.
fn reconstruction(comptime T: type, comptime I: type, layout: c_int, m: usize, n: usize, ld: usize, before: []const T, after: []const T, pivots: []const I) !void {
    var permuted: [4][4][2]f64 = undefined;
    for (0..m) |i| for (0..n) |j| {
        const z = before[s.pos(layout, i, j, ld)];
        permuted[i][j] = .{ s.re(T, z), s.im(T, z) };
    };
    const steps = @min(m, n);
    for (0..steps) |k| {
        try std.testing.expect(pivots[k] >= @as(I, @intCast(k + 1)) and pivots[k] <= @as(I, @intCast(m)));
        const row: usize = @intCast(pivots[k] - 1);
        for (0..n) |j| std.mem.swap([2]f64, &permuted[k][j], &permuted[row][j]);
    }
    for (0..m) |i| for (0..n) |j| {
        var rr: f64 = 0;
        var ii: f64 = 0;
        for (0..steps) |k| {
            const l = if (i == k) s.value(T, 1, 0) else if (i > k) after[s.pos(layout, i, k, ld)] else s.value(T, 0, 0);
            const u = if (k <= j) after[s.pos(layout, k, j, ld)] else s.value(T, 0, 0);
            rr += s.re(T, l) * s.re(T, u) - s.im(T, l) * s.im(T, u);
            ii += s.re(T, l) * s.im(T, u) + s.im(T, l) * s.re(T, u);
        }
        try std.testing.expectApproxEqAbs(permuted[i][j][0], rr, 0.00002);
        try std.testing.expectApproxEqAbs(permuted[i][j][1], ii, 0.00002);
    };
    for (0..after.len) |p| {
        const row = if (layout == 101) p / ld else p % ld;
        const col = if (layout == 101) p % ld else p / ld;
        if (row >= m or col >= n) try s.bytes(before[p..][0..1], after[p..][0..1]);
    }
}

test "GETRF uses actual Core LU Pivot Execution Resources and existing shared configuration" {
    comptime {
        if (A.OwnerIdentity != actual_core or R.OwnerIdentity != actual_core or Core.LU != actual_core.internal.numerics.LU or Core.Pivot != actual_core.internal.numerics.Pivot or Core.Execution != actual_execution)
            @compileError("GETRF must borrow the one actual owner");
        if (A.Services != owner.Adapter.Services or A.Services != owner.SolveAdapter.Services or A.Dense != Core.Dense or A.ScratchLease(f64) != Core.Resources.ScratchLease(f64))
            @compileError("GETRF cannot create per-routine services or storage identities");
        if (@FieldType(A.Arguments(f64, i64), "pivots") != ?[]i64 or @FieldType(R.Arguments(f32, i32), "pivots") != ?[*]align(1) i32)
            @compileError("pivot representation must track access kind and integer width");
        if (@FieldType(A.Result(i32), "admission_fault") != common.access.TypedFault or @FieldType(R.Result(i64), "admission_fault") != common.access.RawFault)
            @compileError("GETRF output must not add an InvalidPivot input fault");
    }
    try std.testing.expect(Core.Execution.getContext() == actual_execution.getContext());
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(0);
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var a = [_]T{s.value(T, 4, 0)};
        var p = [_]I{-999};
        try s.expectInfo(I, 0, try owner.Shared.getrfDefault(true, T, I, .high, .{ .layout = 102, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        try std.testing.expectEqual(@as(I, 1), p[0]);
        try s.expectInfo(I, 0, try owner.Shared.factorDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &a }));
        var b = [_]T{s.value(T, 8, 0)};
        try s.expectInfo(I, 0, try owner.Shared.solveDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }));
        try std.testing.expectEqual(@as(f64, 2), s.re(T, b[0]));
        a[0] = s.value(T, 3, 0);
        try s.expectInfo(I, 0, try owner.Shared.getrfRawDefault(false, T, I, .high, .{ .layout = 102, .m = 1, .n = 1, .lda = 1, .values = a[0..].ptr, .pivots = p[0..].ptr }));
    };
    try std.testing.expectEqual(@as(usize, 32), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 24), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
}

test "GETRF all scalar widths layouts access kinds and entry levels reconstruct rectangular factors with padding" {
    @setEvalBranchQuota(20000);
    const values = [4][4]f64{ .{ 0, 2, 1, 4 }, .{ 0, 0, 3, 2 }, .{ 4, 1, 2, 1 }, .{ 1, 5, 0, 3 } };
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]common.Level{ .high, .work }) |level| for ([_][2]usize{ .{ 3, 3 }, .{ 4, 3 }, .{ 3, 4 } }) |shape| {
            const m = shape[0];
            const n = shape[1];
            var a: [28]T = @splat(s.value(T, std.math.nan(f64), -71));
            for (0..m) |i| for (0..n) |j| {
                a[s.pos(layout, i, j, 6)] = s.value(T, values[i][j], if (Core.Scalar.isComplex(T)) values[i][j] * @as(f64, @floatFromInt(j + 1)) / 4 else 0);
            };
            const before = a;
            var p: [6]I = @splat(-77);
            var ctx: s.Context = .{};
            const services = ctx.services();
            try s.expectInfo(I, 0, invoke(raw, T, I, level, &services, layout, @intCast(m), @intCast(n), 6, &a, &p));
            try reconstruction(T, I, layout, m, n, 6, &before, &a, &p);
            try std.testing.expectEqual(@as(I, 3), p[0]);
            if (m == 3) try std.testing.expectEqual(@as(I, 3), p[1]); // Noncommuting swaps.
            for (p[@min(m, n)..]) |v| try std.testing.expectEqual(@as(I, -77), v);
            try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), ctx.attempts);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        };
    };
}

test "GETRF singularity records first zero while completing later pivots and row copyback" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        var a: [16]T = @splat(s.value(T, 79, -31));
        const values = [3][3]f64{ .{ 0, 2, 0 }, .{ 0, 0, 1 }, .{ 0, 4, 3 } };
        for (0..3) |i| for (0..3) |j| {
            a[s.pos(layout, i, j, 4)] = s.value(T, values[i][j], 0);
        };
        const before = a;
        var p = [_]I{ -10, -20, -30, -40 };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 1, invoke(raw, T, I, .work, &services, layout, 3, 3, 4, &a, &p));
        try std.testing.expectEqualSlices(I, &.{ 1, 3, 3, -40 }, &p);
        try reconstruction(T, I, layout, 3, 3, 4, &before, &a, &p);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    };
}

test "GETRF scalar validation has exact INFO shifts and event channels before raw pointer access" {
    const Case = struct { layout: c_int, m: i64, n: i64, ld: i64, info: i64, argument: i32, allocations: usize };
    const cases = [_]Case{
        .{ .layout = 0, .m = -1, .n = -1, .ld = 0, .info = -1, .argument = -1, .allocations = 0 },
        .{ .layout = 102, .m = -1, .n = -1, .ld = 0, .info = -2, .argument = 1, .allocations = 0 },
        .{ .layout = 102, .m = 0, .n = -1, .ld = 0, .info = -3, .argument = 2, .allocations = 0 },
        .{ .layout = 102, .m = 0, .n = 0, .ld = 0, .info = -5, .argument = 4, .allocations = 0 },
        .{ .layout = 101, .m = -1, .n = 2, .ld = 1, .info = -5, .argument = -5, .allocations = 0 },
        .{ .layout = 101, .m = -1, .n = -2, .ld = 0, .info = -2, .argument = 1, .allocations = 1 },
        .{ .layout = 101, .m = 2, .n = -1, .ld = 0, .info = -3, .argument = 2, .allocations = 1 },
    };
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for (cases) |c| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        // Both non-NULL addresses are unaligned and intentionally non-readable;
        // work scalar failures never resolve them. No bogus typed slice exists.
        try s.expectInfo(I, @intCast(c.info), R.call(T, I, .work, &services, .{ .layout = c.layout, .m = @intCast(c.m), .n = @intCast(c.n), .lda = @intCast(c.ld), .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 0), ctx.reads);
        try std.testing.expectEqual(c.allocations, ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
        try std.testing.expectEqual(c.argument, ctx.hooks[0].argument);
        try std.testing.expectEqual(common.Routine.getrf, ctx.hooks[0].routine);
        try std.testing.expectEqual(if (c.argument > 0) common.Entry.fortran_getrf else .work, ctx.hooks[0].entry);
        try std.testing.expectEqual(if (c.argument > 0) common.HookKind.fortran_returning_test else .lapacke, ctx.hooks[0].kind);
    };
    var ctx: s.Context = .{};
    const services = ctx.services();
    try s.expectInfo(i32, -1, R.call(f64, i32, .high, &services, .{ .layout = 0, .m = 1, .n = 1, .lda = 1, .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try std.testing.expectEqual(common.Entry.high, ctx.hooks[0].entry);
    try std.testing.expectEqual(@as(usize, 0), ctx.reads);
}

test "GETRF high scans general clipped coordinates before invalid lda and never inspects output pivots" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var a = [_]T{ s.value(T, 2, 0), s.value(T, std.math.nan(f64), 0) };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(i64, -4, R.call(T, i64, .high, &services, .{ .layout = layout, .m = 2, .n = 2, .lda = 1, .values = a[0..].ptr, .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
        a[1] = s.value(T, 3, 0);
        try s.expectInfo(i64, -5, R.call(T, i64, .high, &services, .{ .layout = layout, .m = 2, .n = 2, .lda = 1, .values = a[0..].ptr, .pivots = @ptrFromInt(1) }));
        if (comptime Core.Scalar.isComplex(T)) {
            a[0] = s.value(T, 2, std.math.nan(f64));
            try s.expectInfo(i64, -4, R.call(T, i64, .high, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = a[0..].ptr, .pivots = null }));
        }
        // NULL scan is a Reference helper nonread; column scalar error follows.
        try s.expectInfo(i64, -5, R.call(T, i64, .high, &services, .{ .layout = 102, .m = 2, .n = 2, .lda = 1, .values = null, .pivots = null }));
    };
}

test "GETRF NaN control work and compile-disabled paths preserve IEEE singular-only semantics" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        var a = [_]T{s.value(T, std.math.nan(f64), 0)};
        var p = [_]i32{-77};
        const before = a;
        try s.expectInfo(i32, -4, A.call(T, i32, .high, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        try s.bytes(&before, &a);
        try std.testing.expectEqual(@as(i32, -77), p[0]);
        ctx.enabled = false;
        try s.expectInfo(i32, 0, A.call(T, i32, .high, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        ctx.enabled = true;
        const reads = ctx.reads;
        try s.expectInfo(i32, 0, A.call(T, i32, .work, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        try s.expectInfo(i32, 0, owner.NoScanGetrfAdapter.call(T, i32, .high, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        try std.testing.expectEqual(reads, ctx.reads);
        // Reset allocator trace slots before a further row allocation.
        ctx = .{};
        a[0] = s.value(T, std.math.inf(f64), 0);
        try s.expectInfo(i32, 0, A.call(T, i32, .high, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = 1, .values = &a, .pivots = &p }));
        try std.testing.expectEqual(@as(i32, 1), p[0]);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    };
}

test "GETRF overflow and NaN complete factor storage and every pivot instead of checked early return" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| {
        const Real = Core.Scalar.Real(T);
        var a: [12]T = @splat(s.value(T, 91, -3));
        const big: f64 = @floatCast(std.math.floatMax(Real));
        a[s.pos(layout, 0, 0, 4)] = s.value(T, 1, 0);
        a[s.pos(layout, 1, 0, 4)] = s.value(T, -1, 0);
        a[s.pos(layout, 0, 1, 4)] = s.value(T, big, 0);
        a[s.pos(layout, 1, 1, 4)] = s.value(T, big, 0);
        const before = a;
        var p = [_]I{ -77, -77, -77 };
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = layout, .m = 2, .n = 2, .lda = 4, .values = &a, .pivots = &p }));
        try std.testing.expectEqualSlices(I, &.{ 1, 2, -77 }, &p);
        try std.testing.expect(std.math.isInf(s.re(T, a[s.pos(layout, 1, 1, 4)])));
        for (0..a.len) |pos| {
            const row = if (layout == 101) pos / 4 else pos % 4;
            const col = if (layout == 101) pos % 4 else pos / 4;
            if (row >= 2 or col >= 2) try s.bytes(before[pos..][0..1], a[pos..][0..1]);
        }
        a[s.pos(layout, 0, 0, 4)] = s.value(T, std.math.nan(f64), 0);
        @memset(&p, -99);
        ctx = .{};
        try s.expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = layout, .m = 2, .n = 2, .lda = 4, .values = &a, .pivots = &p }));
        try std.testing.expectEqualSlices(I, &.{ 1, 2, -99 }, &p);
        try std.testing.expect(std.math.isNan(s.re(T, a[s.pos(layout, 1, 1, 4)])));
    };
}

test "GETRF typed A then pivot capacities precede stores and row faults clean scratch" {
    for ([_]c_int{ 101, 102 }) |layout| for (0..4) |which| {
        var a = [_]f64{ 2, 1, 4, 3, 83 };
        var p = [_]i64{ -70, -80, -90 };
        const before_a = a;
        const before_p = p;
        var ctx: s.Context = .{};
        const services = ctx.services();
        const values: ?[]f64 = switch (which) {
            0 => null,
            1 => a[0..3],
            else => &a,
        };
        const pivots: ?[]i64 = if (which == 3) p[0..1] else null;
        try fault(if (which == 0 or which == 2) error.MissingBuffer else error.BufferTooSmall, A.call(f64, i64, .work, &services, .{ .layout = layout, .m = 2, .n = 2, .lda = 2, .values = values, .pivots = pivots }));
        try s.bytes(&before_a, &a);
        try s.bytes(&before_p, &p);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    };
}

test "GETRF mixed-width partial aliases reject used spans while adjacent unvisited tails are accepted" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        // A is 3x1; ld=1 for row layout and 3 for column layout. Both have the
        // same contiguous three-element used span. Backing storage is real,
        // aligned and large enough for every constructed typed view.
        var backing: [128]u8 align(16) = @splat(0x5a);
        const ap: [*]T = @ptrCast(&backing);
        for ([_]usize{ 0, @alignOf(I) }) |offset| {
            ap[0] = s.value(T, 4, 0);
            ap[1] = s.value(T, 2, 0);
            ap[2] = s.value(T, 1, 0);
            const pp: [*]I = @ptrCast(@alignCast(backing[offset..].ptr));
            const before = backing;
            var ctx: s.Context = .{};
            const services = ctx.services();
            try fault(error.AliasingNotAllowed, invoke(raw, T, I, .work, &services, layout, 3, 1, if (layout == 101) 1 else 3, ap[0..3], pp[0..1]));
            try s.bytes(&before, &backing);
            try std.testing.expectEqual(@as(usize, 0), ctx.live);
        }
        const offset = std.mem.alignForward(usize, 3 * @sizeOf(T), @alignOf(I));
        const pp: [*]I = @ptrCast(@alignCast(backing[offset..].ptr));
        pp[0] = -123;
        var ctx: s.Context = .{};
        const services = ctx.services();
        // Deliberately pass longer slice tails. Only A's used extent and the
        // first pivot participate in alias checks, not the claimed full tails.
        try s.expectInfo(I, 0, invoke(raw, T, I, .work, &services, layout, 3, 1, if (layout == 101) 1 else 3, ap[0..4], pp[0..2]));
        try std.testing.expectEqual(@as(I, 1), pp[0]);
        try std.testing.expectEqual(@as(f64, 0.5), s.re(T, ap[1]));
        try std.testing.expectEqual(@as(f64, 0.25), s.re(T, ap[2]));
    };
}

test "GETRF raw NULL alignment size and endpoint faults occur only at reached access stages" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_]c_int{ 101, 102 }) |layout| for (0..6) |which| {
        var a = [_]T{s.value(T, 4, 0)};
        var p = [_]I{-77};
        const before = a;
        const bad_a: [*]align(1) T = @ptrFromInt(std.math.maxInt(usize) & ~@as(usize, @alignOf(T) - 1));
        const bad_p: [*]align(1) I = @ptrFromInt(std.math.maxInt(usize) & ~@as(usize, @alignOf(I) - 1));
        var ctx: s.Context = .{};
        const services = ctx.services();
        // The endpoint case for row A is rejected before its layout copy. For
        // pivot faults row A has already been copied, but no caller store occurs.
        try fault(switch (which) {
            0, 3 => error.MissingBuffer,
            1, 4 => error.MisalignedPointer,
            else => error.SizeOverflow,
        }, R.call(T, I, .work, &services, .{
            .layout = layout,
            .m = 1,
            .n = 1,
            .lda = 1,
            .values = switch (which) {
                0 => null,
                1 => @ptrFromInt(1),
                2 => bad_a,
                else => a[0..].ptr,
            },
            .pivots = switch (which) {
                3 => null,
                4 => @ptrFromInt(1),
                5 => bad_p,
                else => p[0..].ptr,
            },
        }));
        try s.bytes(&before, &a);
        try std.testing.expectEqual(@as(I, -77), p[0]);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
    };
    var ctx: s.Context = .{};
    const services = ctx.services();
    // A reached high scan precedes invalid lda; the raw alignment check itself
    // rejects this address, so it is never read or aligned into a typed slice.
    try fault(error.MisalignedPointer, R.call(f64, i64, .high, &services, .{ .layout = 102, .m = 2, .n = 2, .lda = 1, .values = @ptrFromInt(1), .pivots = null }));
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    // Byte arithmetic rejects before attempting to resolve NULL or an address.
    try fault(if (@bitSizeOf(usize) < 64) error.IntegerNotRepresentable else error.SizeOverflow, R.call(f64, i64, .work, &services, .{ .layout = 102, .m = 1, .n = 2, .lda = std.math.maxInt(i64), .values = null, .pivots = @ptrFromInt(1) }));
    try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
}

test "GETRF empty row allocation and column nonreads preserve large native integer geometry" {
    inline for (.{ f32, f64, s.C32, s.C64 }) |T| inline for (.{ i32, i64 }) |I| for ([_][2]I{ .{ 0, 0 }, .{ 0, 2 }, .{ 2, 0 } }) |shape| {
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(I, 0, R.call(T, I, .high, &services, .{ .layout = 101, .m = shape[0], .n = shape[1], .lda = shape[1], .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 1), ctx.attempts);
        try std.testing.expectEqual(@as(usize, 0), ctx.live);
        try std.testing.expectEqual(@as(i64, @intCast(@max(shape[0], 1) * @max(shape[1], 1) * @as(I, @sizeOf(T)))), ctx.events[1].value);
        ctx = .{};
        try s.expectInfo(I, 0, R.call(T, I, .high, &services, .{ .layout = 102, .m = shape[0], .n = shape[1], .lda = std.math.maxInt(I), .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 0), ctx.attempts);
    };
    if (comptime @bitSizeOf(usize) >= 64) inline for (.{ f32, f64, s.C32, s.C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var a = [_]T{s.value(T, 4, 0)};
        var p = [_]i64{-77};
        var ctx: s.Context = .{};
        const services = ctx.services();
        try s.expectInfo(i64, 0, A.call(T, i64, .work, &services, .{ .layout = layout, .m = 1, .n = 1, .lda = @as(i64, std.math.maxInt(i32)) + 1, .values = &a, .pivots = &p }));
        try std.testing.expectEqual(@as(i64, 1), p[0]);
    };
}

test "GETRF row allocation failure and size errors precede later scalar and pointer admission" {
    var ctx: s.Context = .{ .fail_index = 0 };
    const services = ctx.services();
    try s.expectInfo(i64, -1011, R.call(f64, i64, .work, &services, .{ .layout = 101, .m = -1, .n = 2, .lda = 2, .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 16 }, .{ .tag = 4, .argument = -1011, .value = 2 } });
    try std.testing.expectEqual(common.Routine.getrf, ctx.hooks[0].routine);
    ctx = .{};
    try s.expectInfo(i64, -2, R.call(f64, i64, .work, &services, .{ .layout = 101, .m = -1, .n = 2, .lda = 2, .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try s.trace(&ctx, &.{ .{ .tag = 2, .argument = 1, .value = 16 }, .{ .tag = 5, .argument = 1, .value = 1 }, .{ .tag = 3, .argument = 1, .value = 16 } });
    ctx = .{};
    try fault(if (@bitSizeOf(usize) < 64) error.IntegerNotRepresentable else error.SizeOverflow, R.call(f64, i64, .work, &services, .{ .layout = 101, .m = std.math.maxInt(i64), .n = 2, .lda = 2, .values = null, .pivots = null }));
    try std.testing.expectEqual(@as(usize, 0), ctx.count);
}

test "GETRF managed refusal precedes borrowed Services and raw reads and pin overflow stays separate" {
    for ([_]bool{ false, true }) |overflow| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        var held: ?C.Operation = null;
        if (overflow) {
            C.Context.testing.configure(&r.context, 1, null);
            held = try r.context.begin();
        } else _ = r.context.requestClose();
        defer if (held) |*op| op.end();
        var services: Core.Services = undefined;
        const expected = if (overflow) error.CounterOverflow else error.Closing;
        try std.testing.expectError(expected, A.callManaged(f64, i64, .high, &r.context, &services, args(null, null)));
        try std.testing.expectError(expected, R.callManaged(f64, i64, .high, &r.context, &services, .{ .layout = 102, .m = 1, .n = 1, .lda = 1, .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
        try std.testing.expectEqual(@as(usize, 0), r.count);
    }
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    C.Context.testing.configure(&r.context, 1, null);
    var op = try r.context.begin();
    var held = try C.ContextScratch(u8).acquire(&op, 1);
    op.end();
    var a = [_]f64{ 2, 1, 4, 3 };
    var p = [_]i64{ -77, -77 };
    try std.testing.expectError(error.CounterOverflow, A.callManaged(f64, i64, .work, &r.context, &r.services, args(&a, &p)));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts); // Only the held lease.
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
    try std.testing.expectEqualSlices(f64, &.{ 2, 1, 4, 3 }, &a);
    try std.testing.expectEqualSlices(i64, &.{ -77, -77 }, &p);
    held.deinit();
}

test "GETRF managed captures allocator and callbacks once and real OOM rolls back pin before hook" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    r.rebind = true;
    var f = args(null, null);
    f.m = -1;
    try s.expectInfo(i64, -2, try A.callManaged(f64, i64, .high, &r.context, &r.services, f));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts + r.z.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.hooks);
    r.services.allocator = undefined;
    r.services.error_event = h.Rig.hookForTest;
    r.x.fail_at = r.x.attempts;
    var a = [_]f64{ 2, 1, 4, 3 };
    var p = [_]i64{ -77, -77 };
    const first = r.count;
    try s.expectInfo(i64, -1011, try A.callManaged(f64, i64, .work, &r.context, &r.services, args(&a, &p)));
    var found = false;
    for (r.events[first..r.count]) |e| if (e.tag == .hook) {
        found = true;
        try std.testing.expectEqual(@as(usize, 1), e.active);
        try std.testing.expectEqual(@as(usize, 0), e.pins);
        try std.testing.expectEqual(@as(i32, -1011), e.detail);
    };
    try std.testing.expect(found);
    try std.testing.expectEqualSlices(f64, &.{ 2, 1, 4, 3 }, &a);
    try std.testing.expectEqualSlices(i64, &.{ -77, -77 }, &p);
}

test "GETRF managed close from getter allocator hook and free retains one operation through real body" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .alloc, .hook, .free }) |trigger| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        r.trigger = trigger;
        r.action = .close_nested;
        var a = [_]f64{ 2, 1, 4, 3 };
        var p = [_]i64{ -77, -77 };
        var f = args(&a, &p);
        if (trigger == .hook) f.m = -1;
        try s.expectInfo(i64, if (trigger == .hook) -2 else 0, try A.callManaged(f64, i64, .high, &r.context, &r.services, f));
        try std.testing.expectEqual(error.Closing, r.nested_error.?);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        var starts: usize = 0;
        var ends: usize = 0;
        var freed = false;
        for (r.events[0..r.count]) |e| {
            if (e.tag == .admitted) starts += 1;
            if (e.tag == .free) {
                freed = true;
                try std.testing.expectEqual(@as(usize, 1), e.active);
                try std.testing.expectEqual(@as(usize, 1), e.pins);
            }
            if (e.tag == .finishing) {
                ends += 1;
                try std.testing.expect(freed);
                try std.testing.expectEqual(@as(usize, 0), e.pins);
            }
        }
        try std.testing.expectEqual(@as(usize, 1), starts);
        try std.testing.expectEqual(@as(usize, 1), ends);
        if (trigger != .hook) {
            try std.testing.expectEqualSlices(f64, &.{ 4, 3, 0.5, -0.5 }, &a);
            try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, &p);
        } else {
            try std.testing.expectEqualSlices(f64, &.{ 2, 1, 4, 3 }, &a);
            try std.testing.expectEqualSlices(i64, &.{ -77, -77 }, &p);
        }
    }
}

// This allocator inspects actual factor/pivot copyback at the real free call,
// including positive INFO; it never supplies a fake numerical body.
const FreeAudit = struct {
    context: C.Context = undefined,
    a: [9]f64 = .{ 0, 2, 0, 0, 0, 1, 0, 4, 3 },
    p: [3]i64 = .{ -77, -77, -77 },
    allocated: bool = false,
    freed: bool = false,
    fn allocator(self: *FreeAudit) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FreeAudit = @ptrCast(@alignCast(raw));
        const p = std.testing.allocator.rawAlloc(len, alignment, ra) orelse return null;
        self.allocated = true;
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
        if (!self.allocated or self.freed) @panic("allocator provenance");
        if (!std.mem.eql(f64, &self.a, &.{ 0, 2, 0, 0, 4, 3, 0, 0, 1 }) or !std.mem.eql(i64, &self.p, &.{ 1, 3, 3 })) @panic("free preceded completed copyback");
        const state = self.context.snapshot();
        if (state.active_operations != 1 or state.resource_pins != 1) @panic("free lost operation/pin");
        if (self.context.requestClose() != .pending or self.context.tryProgressClose() != .busy) @panic("close during free");
        std.testing.allocator.rawFree(mem, alignment, ra);
        const after = self.context.snapshot();
        if (after.active_operations != 1 or after.resource_pins != 1) @panic("unpin preceded allocator return");
        self.freed = true;
    }
};

test "GETRF singular row copyback and immediate pivots precede captured allocator free then unpin" {
    var audit: FreeAudit = .{};
    audit.context = C.Context.init(audit.allocator(), actual_execution.test_support.policy, .internal_closable);
    var services_ctx: s.Context = .{};
    var services = services_ctx.services();
    services.allocator = undefined;
    try s.expectInfo(i64, 1, try A.callManaged(f64, i64, .work, &audit.context, &services, .{ .layout = 101, .m = 3, .n = 3, .lda = 3, .values = &audit.a, .pivots = &audit.p }));
    try std.testing.expect(audit.freed);
    try std.testing.expectEqual(@as(usize, 0), audit.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), audit.context.snapshot().resource_pins);
    try std.testing.expectEqual(C.CloseResult.closed, audit.context.tryProgressClose());
    try std.testing.expectEqual(@as(usize, 0), services_ctx.attempts);
}

test "GETRF shared typed and raw routes capture once and preserve admission-before-capture on refusal" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(0);
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    var a = [_]f64{ 2, 1, 4, 3 };
    var p = [_]i64{ -77, -77 };
    try s.expectInfo(i64, 0, try owner.Shared.getrfManaged(true, f64, i64, .high, &r.context, args(&a, &p)));
    a = .{ 2, 1, 4, 3 };
    try s.expectInfo(i64, 0, try owner.Shared.getrfRawManaged(true, f64, i64, .high, &r.context, .{ .layout = 101, .m = 2, .n = 2, .lda = 2, .values = a[0..].ptr, .pivots = p[0..].ptr }));
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
    var starts: usize = 0;
    for (r.events[0..r.count]) |e| if (e.tag == .admitted) {
        starts += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), starts);
    _ = r.context.requestClose();
    try std.testing.expectError(error.Closing, owner.Shared.getrfRawManaged(true, f64, i64, .high, &r.context, .{ .layout = 0, .m = 1, .n = 1, .lda = 1, .values = @ptrFromInt(1), .pivots = @ptrFromInt(1) }));
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
}
