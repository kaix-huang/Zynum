// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const Core = owner.Provider;
const Shared = owner.Shared;
const cfg = owner.shared_services;
const policy = owner.complex_gelsd_policy;
const h = @import("managed_support.zig");
const Ctx = Core.Execution.context;
const C32 = Core.Scalar.ComplexF32;
const C64 = Core.Scalar.ComplexF64;
const Level = owner.common.Level;
fn Raw(comptime scan: bool) type {
    return if (scan) owner.RawComplexGelsdAdapter else owner.RawNoScanComplexGelsdAdapter;
}
fn raw(comptime T: type, values: []T) ?[*]align(1) T {
    return values.ptr;
}
fn z(comptime C: type, re: f64, im: f64) C {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}
fn pos(layout: c_int, row: usize, col: usize, ld: usize) usize {
    return if (layout == 101) row * ld + col else row + col * ld;
}
fn bytes(a: anytype, b: anytype) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn project(result: anytype) !i32 {
    return switch (result) {
        .lapacke_info => |i| @intCast(i),
        .numerical_fault => |e| policy.standardNumericalInfo(e) orelse {
            std.debug.print("unprojected numerical {s}\n", .{@errorName(e)});
            return error.TestUnexpectedResult;
        },
        .admission_fault => |e| {
            std.debug.print("unexpected admission {s}\n", .{@errorName(e)});
            return error.TestUnexpectedResult;
        },
    };
}
fn numerical(result: anytype, expected: anyerror) !void {
    switch (result) {
        .numerical_fault => |e| try std.testing.expectEqual(expected, e),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(i32, 1), try project(result));
}
fn eventCount(r: *const h.Rig, tag: @FieldType(h.Event, "tag")) usize {
    var count: usize = 0;
    for (r.events[0..r.count]) |e| if (e.tag == tag) {
        count += 1;
    };
    return count;
}
fn done(r: *h.Rig, expected_allocations: ?usize) !void {
    try std.testing.expectEqual(@as(usize, 1), eventCount(r, .admitted));
    try std.testing.expectEqual(@as(usize, 1), eventCount(r, .finishing));
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
    if (expected_allocations) |n| try std.testing.expectEqual(n, r.x.attempts);
    try std.testing.expectEqual(r.x.attempts, r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.x.live);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
}
fn Fixture(comptime C: type, comptime I: type) type {
    const R = Core.Scalar.Real(C);
    return struct {
        a: [8]C = @splat(.{ .re = 71, .im = -17 }),
        b: [8]C = @splat(.{ .re = 72, .im = -18 }),
        s: [3]R = @splat(73),
        rank: [2]I = .{ -74, 75 },
        rw: [2]R = .{ -76, 77 },
        iw: [2]I = .{ -78, 79 },
        layout: c_int,
        ldb: usize,
        fn init(layout: c_int) @This() {
            var f: @This() = .{ .layout = layout, .ldb = if (layout == 101) 2 else 3 };
            f.a[pos(layout, 0, 0, 3)] = z(C, 1, 0);
            f.a[pos(layout, 1, 0, 3)] = z(C, 0, 0);
            f.a[pos(layout, 0, 1, 3)] = z(C, 1, 1);
            f.a[pos(layout, 1, 1, 3)] = z(C, 2, 0);
            f.b[pos(layout, 0, 0, f.ldb)] = z(C, 4, 2);
            f.b[pos(layout, 1, 0, f.ldb)] = z(C, 4, -2);
            return f;
        }
        fn args(f: *@This(), comptime scan: bool, work: []C, lwork: I) Raw(scan).Arguments(C, I) {
            return .{ .layout = f.layout, .column = .{ .m = 2, .n = 2, .nrhs = 1, .lda = 3, .ldb = @intCast(f.ldb), .a = raw(C, &f.a), .b = raw(C, &f.b), .singular = raw(R, &f.s), .rank = raw(I, &f.rank), .rcond = -1, .work = raw(C, work), .lwork = lwork, .rwork = raw(R, &f.rw), .iwork = raw(I, &f.iw) } };
        }
        fn verify(f: *@This(), before: *const @This()) !void {
            const tol: R = if (R == f32) 2e-5 else 2e-12;
            const x0 = f.b[pos(f.layout, 0, 0, f.ldb)];
            const x1 = f.b[pos(f.layout, 1, 0, f.ldb)];
            try std.testing.expectApproxEqAbs(@as(R, 1), x0.re, tol);
            try std.testing.expectApproxEqAbs(@as(R, 1), x0.im, tol);
            try std.testing.expectApproxEqAbs(@as(R, 2), x1.re, tol);
            try std.testing.expectApproxEqAbs(@as(R, -1), x1.im, tol);
            try std.testing.expectEqual(@as(I, 2), f.rank[0]);
            try std.testing.expectEqual(before.rank[1], f.rank[1]);
            try bytes(&before.a, &f.a);
            for (f.b, before.b, 0..) |x, y, i| if (i != pos(f.layout, 0, 0, f.ldb) and i != pos(f.layout, 1, 0, f.ldb)) try std.testing.expectEqualDeep(y, x);
            try std.testing.expectEqual(before.s[2], f.s[2]);
            try std.testing.expectEqual(before.rw[1], f.rw[1]);
            try std.testing.expectEqual(before.iw[1], f.iw[1]);
        }
    };
}
fn workFor(comptime C: type) ![]C {
    const p = try owner.complex_gelsd_workspace.Planner(Core).make(C, i32, 2, 2, 1);
    const w = try std.testing.allocator.alloc(C, @intCast(p.work));
    @memset(w, z(C, 81, -21));
    return w;
}
fn snapshotSingle(expected_getters: usize) !void {
    const state = cfg.testing.snapshot();
    try std.testing.expectEqual(@as(usize, 1), state.captures);
    try std.testing.expectEqual(expected_getters, state.getter_calls);
    try std.testing.expectEqual(@as(usize, 0), state.environment_reads);
}

test "complex GELSD shared raw managed high work query capture one original operation and cleanly close" {
    comptime {
        const actual = @import("zynum-core");
        const execution = @import("zynum-execution");
        if (Core.OwnerIdentity != actual or Core.Execution != execution or Core.Resources != actual.internal.numerics.Resources) @compileError("shared complex GELSD must use the actual configured owner");
    }
    var configs: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ true, false }) |scan| for ([_]Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |query| {
        if (query and level == .high) continue;
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("shared cleanup");
        var f = Fixture(C, i32).init(layout);
        const before = f;
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        var args = f.args(scan, w, @intCast(w.len));
        if (query) {
            args.column.lwork = -1;
            args.column.a = null;
            args.column.b = null;
            args.column.singular = null;
            args.column.rank = null;
            args.column.rcond = std.math.nan(Core.Scalar.Real(C));
        }
        try std.testing.expectEqual(@as(i32, 0), try project(try Shared.complexGelsdRawManagedWithDefaults(scan, C, i32, level, &r.context, args, .{})));
        if (query) {
            try bytes(&before.a, &f.a);
            try bytes(&before.b, &f.b);
            try bytes(&before.s, &f.s);
            try bytes(&before.rank, &f.rank);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), 0), w[0].im);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), 1), f.rw[0]);
            try std.testing.expectEqual(@as(i32, 1), f.iw[0]);
            try std.testing.expectEqual(before.rw[1], f.rw[1]);
            try std.testing.expectEqual(before.iw[1], f.iw[1]);
        } else try f.verify(&before);
        const n: usize = if (query) 0 else if (level == .high) (if (layout == 101) 5 else 3) else (if (layout == 101) 2 else 0);
        try done(&r, n);
        try snapshotSingle(if (scan and level == .high) 1 else 0);
        configs += 1;
        allocations += r.x.attempts;
    };
    std.debug.print("SHARED configs={d} admissions=finishes=closes={d} captures={d} allocs=frees={d}\n", .{ configs, configs, configs, allocations });
}
const Path = enum { work, high_disabled, high_noscan };
test "complex GELSD unchecked nonfinite A B and NaN cutoff return projectable failures without B rank commits" {
    var configs: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ Path.work, Path.high_disabled, Path.high_noscan }) |path| for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 0, 1, 2 }) |location| for ([_]bool{ false, true }) |infinity| {
        if (location == 2 and infinity) continue;
        const R = Core.Scalar.Real(C);
        const scan = path != .high_noscan;
        const level: Level = if (path == .work) .work else .high;
        const invalid: R = if (infinity) std.math.inf(R) else std.math.nan(R);
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(if (path == .high_disabled) 0 else 1);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("nonfinite cleanup");
        var f = Fixture(C, i32).init(layout);
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        var args = f.args(scan, w, @intCast(w.len));
        switch (location) {
            0 => f.a[pos(layout, 0, 0, 3)].im = invalid,
            1 => f.b[pos(layout, 1, 0, f.ldb)].im = invalid,
            2 => args.column.rcond = invalid,
            else => unreachable,
        }
        const before = f;
        const result = try Shared.complexGelsdRawManagedWithDefaults(scan, C, i32, level, &r.context, args, .{});
        try numerical(result, if (location == 2) error.InvalidTolerance else error.NonFiniteInput);
        try bytes(&before.b, &f.b);
        try bytes(&before.rank, &f.rank);
        try bytes(&before.a, &f.a);
        try done(&r, null);
        try snapshotSingle(if (path == .high_disabled) 1 else 0);
        configs += 1;
        allocations += r.x.attempts;
    };
    std.debug.print("NONFINITE configs={d} admissions=finishes=closes={d} captures={d} projected_info=1 allocs=frees={d}\n", .{ configs, configs, configs, allocations });
}
test "complex GELSD shared enabled high NaN scanning retains minus5 minus7 minus10" {
    var configs: usize = 0;
    inline for (.{ C32, C64 }) |C| for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 0, 1, 2 }) |location| {
        const R = Core.Scalar.Real(C);
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("scan cleanup");
        var f = Fixture(C, i32).init(layout);
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        var args = f.args(true, w, @intCast(w.len));
        switch (location) {
            0 => f.a[pos(layout, 0, 0, 3)].re = std.math.nan(R),
            1 => f.b[pos(layout, 0, 0, f.ldb)].re = std.math.nan(R),
            2 => args.column.rcond = std.math.nan(R),
            else => unreachable,
        }
        const before = f;
        try std.testing.expectEqual(@as(i32, if (location == 0) -5 else if (location == 1) -7 else -10), try project(try Shared.complexGelsdRawManagedWithDefaults(true, C, i32, .high, &r.context, args, .{})));
        try bytes(&before.b, &f.b);
        try bytes(&before.rank, &f.rank);
        try done(&r, 0);
        try snapshotSingle(1);
        configs += 1;
    };
    std.debug.print("SCAN configs={d} admissions=finishes=closes={d} captures={d} allocs=frees=0\n", .{ configs, configs, configs });
}
test "complex GELSD infinite cutoffs retain reference comparison normalization and successful commit" {
    var configs: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ true, false }) |scan| for ([_]Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |negative| {
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("inf cutoff cleanup");
        var f = Fixture(C, i32).init(layout);
        const before = f;
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        var args = f.args(scan, w, @intCast(w.len));
        args.column.rcond = if (negative) -std.math.inf(Core.Scalar.Real(C)) else std.math.inf(Core.Scalar.Real(C));
        try std.testing.expectEqual(@as(i32, 0), try project(try Shared.complexGelsdRawManagedWithDefaults(scan, C, i32, level, &r.context, args, .{})));
        try f.verify(&before);
        try done(&r, null);
        try snapshotSingle(if (scan and level == .high) 1 else 0);
        configs += 1;
        allocations += r.x.attempts;
    };
    std.debug.print("INFCUTOFF configs={d} admissions=finishes=closes={d} captures={d} successful_commits={d} allocs=frees={d}\n", .{ configs, configs, configs, configs, allocations });
}
const Recorder = struct {
    calls: usize = 0,
    last: ?owner.common.ErrorEvent = null,
    fn call(p: *anyopaque, e: owner.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(p));
        self.calls += 1;
        self.last = e;
    }
    fn callback(self: *@This()) cfg.Callback {
        return .{ .userdata = self, .function = call };
    }
};
const CaptureAllocator = struct {
    original: std.mem.Allocator,
    replacement: cfg.Callbacks,
    switched: bool = false,
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(p: *anyopaque) *@This() {
        return @ptrCast(@alignCast(p));
    }
    fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(p);
        if (!self.switched) {
            self.switched = true;
            _ = cfg.setCallbacks(self.replacement);
        }
        return self.original.rawAlloc(n, a, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(p: *anyopaque, b: []u8, a: std.mem.Alignment, ret: usize) void {
        cast(p).original.rawFree(b, a, ret);
    }
};
test "complex GELSD shared defaults are call local captured before allocation and configured channels win later" {
    var configs: usize = 0;
    inline for (.{ C32, C64 }) |C| {
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(0);
        var first: Recorder = .{};
        var next: Recorder = .{};
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("capture cleanup");
        var changing: CaptureAllocator = .{ .original = r.x.allocator(), .replacement = .{ .lapacke = next.callback(), .fortran_returning_test = next.callback() } };
        const test_state = r.context.test_state;
        r.context = Ctx.Context.init(changing.allocator(), Core.Execution.test_support.policy, .internal_closable);
        r.context.test_state = test_state;
        r.x.fail_at = 0;
        var f = Fixture(C, i32).init(102);
        const before = f;
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        var args = f.args(true, w, @intCast(w.len));
        const defaults: cfg.Callbacks = .{ .lapacke = first.callback(), .fortran_returning_test = first.callback() };
        try std.testing.expectEqual(@as(i32, -1010), try project(try Shared.complexGelsdRawManagedWithDefaults(true, C, i32, .high, &r.context, args, defaults)));
        try std.testing.expect(changing.switched);
        try std.testing.expectEqual(@as(usize, 1), first.calls);
        try std.testing.expectEqual(@as(usize, 0), next.calls);
        try std.testing.expectEqual(@as(i32, -1010), first.last.?.argument);
        try bytes(&before.b, &f.b);
        try bytes(&before.rank, &f.rank);
        try snapshotSingle(1);
        try std.testing.expectEqual(@as(usize, 1), eventCount(&r, .admitted));
        try std.testing.expectEqual(@as(usize, 1), eventCount(&r, .finishing));
        args.layout = 777;
        try std.testing.expectEqual(@as(i32, -1), try project(try Shared.complexGelsdRawManagedWithDefaults(true, C, i32, .work, &r.context, args, defaults)));
        try std.testing.expectEqual(@as(usize, 1), first.calls);
        try std.testing.expectEqual(@as(usize, 1), next.calls);
        try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
        try std.testing.expectEqual(@as(usize, 2), eventCount(&r, .admitted));
        try std.testing.expectEqual(@as(usize, 2), eventCount(&r, .finishing));
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        try std.testing.expectEqual(@as(usize, 0), r.x.live);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
        configs += 2;
    }
    // Distinct configured/fallback channels are selected independently and per
    // call defaults are never published to the shared configuration.
    cfg.testing.reset();
    defer cfg.testing.reset();
    var selected: Recorder = .{};
    var fallback: Recorder = .{};
    _ = cfg.setCallbacks(.{ .lapacke = selected.callback() });
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("channel cleanup");
    var f = Fixture(C64, i32).init(102);
    const w = try workFor(C64);
    defer std.testing.allocator.free(w);
    var args = f.args(true, w, @intCast(w.len));
    args.column.m = -1;
    try std.testing.expectEqual(@as(i32, -2), try project(try Shared.complexGelsdRawManagedWithDefaults(true, C64, i32, .work, &r.context, args, .{ .lapacke = fallback.callback(), .fortran_returning_test = fallback.callback() })));
    try std.testing.expectEqual(@as(usize, 0), selected.calls);
    try std.testing.expectEqual(@as(usize, 1), fallback.calls);
    try std.testing.expectEqual(owner.common.HookKind.fortran_returning_test, fallback.last.?.kind);
    try done(&r, 0);
    try snapshotSingle(0);
    const live = cfg.captureCallbacks();
    try std.testing.expect(live.fortran_returning_test == null);
    configs += 1;
    std.debug.print("CAPTURE calls={d} admissions=finishes={d} contexts_closed=3 allocation_attempts=2 forced_failures=2 frees=0\n", .{ configs, configs });
}
var default_admissions: usize = 0;
var default_finishes: usize = 0;
var default_max_active: usize = 0;
fn defaultObserver(context: *Ctx.Context, event: Ctx.TestEvent) void {
    if (context != Core.Execution.getContext()) @panic("default wrapper admitted a different Context");
    switch (event) {
        .admitted => default_admissions += 1,
        .finishing => default_finishes += 1,
    }
    default_max_active = @max(default_max_active, context.snapshot().active_operations);
}
test "complex GELSD default wrapper preserves the one immortal actual Context after high work and query" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    cfg.setNancheck(0);
    const context = Core.Execution.getContext();
    const before = context.snapshot();
    const saved_test_state = context.test_state;
    defer context.test_state = saved_test_state;
    default_admissions = 0;
    default_finishes = 0;
    default_max_active = 0;
    Ctx.Context.testing.configure(context, std.math.maxInt(usize), defaultObserver);
    var calls: usize = 0;
    inline for (.{ C32, C64 }) |C| for ([_]Level{ .high, .work }) |level| {
        var f = Fixture(C, i32).init(102);
        const original = f;
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        const args = f.args(true, w, @intCast(w.len));
        try std.testing.expectEqual(@as(i32, 0), try project(try Shared.complexGelsdRawDefault(true, C, i32, level, args)));
        try f.verify(&original);
        try std.testing.expectEqual(before, context.snapshot());
        calls += 1;
    };
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        var f = Fixture(C, I).init(102);
        var work = [_]C{ z(C, 91, -22), z(C, 92, -23) };
        var args = f.args(true, &work, -1);
        args.column.a = null;
        args.column.b = null;
        args.column.rank = null;
        args.column.singular = null;
        args.column.rcond = std.math.nan(Core.Scalar.Real(C));
        try std.testing.expectEqual(@as(i32, 0), try project(try Shared.complexGelsdRawDefault(true, C, I, .work, args)));
        try std.testing.expectEqual(before, context.snapshot());
        try std.testing.expectEqualDeep(z(C, 92, -23), work[1]);
        calls += 1;
    };
    try std.testing.expect(context == @import("zynum-execution").getContext());
    try std.testing.expectEqual(Ctx.Lifetime.immortal_default, context.lifetime);
    try std.testing.expectEqual(calls, default_admissions);
    try std.testing.expectEqual(calls, default_finishes);
    try std.testing.expectEqual(@as(usize, 1), default_max_active);
    try std.testing.expectEqual(calls, cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
    std.debug.print("DEFAULT calls={d} captures={d} admissions=finishes={d} immutable_context_snapshots=true max_active=1 default_context_not_closed=true\n", .{ calls, calls, calls });
}
test "complex GELSD real Column exhaustion returns own one token and stable numerical faults project safely" {
    var configs: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("column exhaustion cleanup");
        var f = Fixture(C, I).init(102);
        const before = f;
        const w = try workFor(C);
        defer std.testing.allocator.free(w);
        const args = f.args(true, w, @intCast(w.len));
        const result = try owner.RawComplexGelsdColumnWork.callManaged(C, I, &r.context, &r.services, args.column, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(@as(i32, 1), try project(result));
        switch (result) {
            .lapacke_info => |i| try std.testing.expectEqual(@as(I, 1), i),
            else => return error.TestUnexpectedResult,
        }
        try bytes(&before.b, &f.b);
        try bytes(&before.rank, &f.rank);
        try done(&r, 0);
        configs += 1;
    };
    // This is the same established real Core range case as the complex SVD
    // regression: a native-rounded single U column cannot meet tolerance1e-30.
    inline for (.{ C32, C64 }) |C| for ([_]u8{ 0, 1, 2 }) |scenario| {
        const R = Core.Scalar.Real(C);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("column projection cleanup");
        var a = [_]C{ z(C, 1, 1), z(C, 2, 3) };
        if (scenario == 2) @memset(&a, .{ .re = std.math.floatMax(R), .im = 0 });
        var b = [_]C{ z(C, 4, 1), z(C, 5, -2), z(C, 96, -27) };
        const before = b;
        var singular = [_]R{ -7, 8 };
        var rank = [_]i32{ -9, 10 };
        const original_rank = rank;
        var rw = [_]R{ -11, 12 };
        var iw = [_]i32{ -13, 14 };
        const p = try owner.complex_gelsd_workspace.Planner(Core).make(C, i32, 2, 1, 1);
        const w = try std.testing.allocator.alloc(C, @intCast(p.work));
        defer std.testing.allocator.free(w);
        const args: owner.RawComplexGelsdColumnWork.Arguments(C, i32) = .{ .m = 2, .n = 1, .nrhs = 1, .lda = 2, .ldb = 2, .a = raw(C, &a), .b = raw(C, &b), .singular = raw(R, &singular), .rank = raw(i32, &rank), .rcond = -1, .work = raw(C, w), .lwork = p.work, .rwork = raw(R, &rw), .iwork = raw(i32, &iw) };
        const options: Core.ComplexRankLeastSquares.Options = if (scenario == 0) .{ .relative_tolerance = 1e-30 } else if (scenario == 1) .{ .relative_tolerance = 0 } else .{};
        try numerical(try owner.RawComplexGelsdColumnWork.callManaged(C, i32, &r.context, &r.services, args, options), if (scenario == 0) error.RangeLimit else if (scenario == 1) error.InvalidTolerance else error.NonFiniteResult);
        try bytes(&before, &b);
        try bytes(&original_rank, &rank);
        try done(&r, 0);
        configs += 1;
    };
    inline for (.{ error.NonFiniteInput, error.InvalidTolerance, error.NonFiniteResult, error.RangeLimit }) |e| try std.testing.expectEqual(@as(?i32, 1), policy.standardNumericalInfo(e));
    inline for (.{ error.AliasingNotAllowed, error.BufferTooSmall, error.InvalidLength, error.SizeOverflow }) |e| try std.testing.expectEqual(@as(?i32, null), policy.standardNumericalInfo(e));
    std.debug.print("COLUMN configs={d} admissions=finishes=closes={d} allocs=frees=0 exhaustion_token=1 stable_range_limit=true\n", .{ configs, configs });
}
test "complex GELSD minimum WORK honors reference f32 ILAENV6 conversion at large threshold without allocations" {
    inline for (.{ i32, i64 }) |I| {
        try std.testing.expectEqual(@as(I, 9437194), try policy.minimumWork(I, 4194308, 2621443, 0));
        try std.testing.expectEqual(@as(I, 7864329), try policy.minimumWork(I, 4194309, 2621443, 0));
    }
}
