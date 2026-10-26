// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const Core = owner.Provider;
const Ctx = Core.Execution.context;
const C32 = Core.Scalar.ComplexF32;
const C64 = Core.Scalar.ComplexF64;
const Level = owner.common.Level;
fn adapter(comptime raw: bool, comptime scan: bool) type {
    return if (raw) (if (scan) owner.RawComplexGelsdAdapter else owner.RawNoScanComplexGelsdAdapter) else (if (scan) owner.ComplexGelsdAdapter else owner.NoScanComplexGelsdAdapter);
}
fn operand(comptime T: type, comptime raw: bool, values: []T) if (raw) ?[*]align(1) T else ?[]T {
    return if (raw) @as([*]align(1) T, values.ptr) else values;
}
fn info(result: anytype, expected: anytype) !void {
    switch (result) {
        .lapacke_info => |returned| try std.testing.expectEqual(@as(@TypeOf(returned), expected), returned),
        .admission_fault, .numerical_fault => |err| {
            std.debug.print("unexpected GELSD fault {s}\n", .{@errorName(err)});
            return error.TestUnexpectedResult;
        },
    }
}
fn bytes(a: anytype, b: anytype) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn pos(layout: c_int, row: usize, col: usize, ld: usize) usize {
    return if (layout == 101) row * ld + col else row + col * ld;
}
fn value(comptime C: type, re: f64, im: f64) C {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}
fn counts(r: *const h.Rig, tag: @FieldType(h.Event, "tag")) usize {
    var count: usize = 0;
    for (r.events[0..r.count]) |e| if (e.tag == tag) {
        count += 1;
    };
    return count;
}
fn lifecycle(r: *h.Rig, expected_allocations: usize) !void {
    try std.testing.expectEqual(@as(usize, 1), counts(r, .admitted));
    try std.testing.expectEqual(@as(usize, 1), counts(r, .finishing));
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
    try std.testing.expectEqual(expected_allocations, r.x.attempts);
    try std.testing.expectEqual(expected_allocations, r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), r.x.live);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
}
fn Fixture(comptime C: type, comptime I: type) type {
    const R = Core.Scalar.Real(C);
    return struct {
        a: [24]C = @splat(.{ .re = 71, .im = -17 }),
        b: [24]C = @splat(.{ .re = 72, .im = -18 }),
        s: [4]R = @splat(73),
        rank: [2]I = .{ -74, 75 },
        rwork: [2]R = .{ -76, 77 },
        iwork: [2]I = .{ -78, 79 },
        m: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        layout: c_int,
        fn init(layout: c_int, wide: bool) @This() {
            const m: usize = if (wide) 2 else 3;
            const n: usize = if (wide) 3 else 2;
            var self: @This() = .{ .m = m, .n = n, .lda = (if (layout == 101) n else m) + 2, .ldb = if (layout == 101) 3 else 5, .layout = layout };
            for (0..m) |row| for (0..n) |col| {
                self.a[pos(layout, row, col, self.lda)] = value(C, 0, 0);
            };
            if (wide) {
                self.a[pos(layout, 0, 0, self.lda)] = value(C, 1, 0);
                self.a[pos(layout, 0, 1, self.lda)] = value(C, 0, 1);
                self.a[pos(layout, 1, 2, self.lda)] = value(C, 1, 0);
                self.b[pos(layout, 0, 0, self.ldb)] = value(C, 1, 2);
                self.b[pos(layout, 1, 0, self.ldb)] = value(C, -2, 1);
                self.b[pos(layout, 2, 0, self.ldb)] = value(C, 13, -9);
            } else {
                self.a[pos(layout, 0, 0, self.lda)] = value(C, 1, 0);
                self.a[pos(layout, 1, 0, self.lda)] = value(C, 0, 1);
                self.a[pos(layout, 1, 1, self.lda)] = value(C, 1, 0);
                self.a[pos(layout, 2, 1, self.lda)] = value(C, 1, 0);
                self.b[pos(layout, 0, 0, self.ldb)] = value(C, 2, 3);
                self.b[pos(layout, 1, 0, self.ldb)] = value(C, -3, 1);
                self.b[pos(layout, 2, 0, self.ldb)] = value(C, -3, 2);
            }
            return self;
        }
        fn args(self: *@This(), comptime raw: bool, comptime scan: bool, work: []C, lwork: I) adapter(raw, scan).Arguments(C, I) {
            return .{ .layout = self.layout, .column = .{ .m = @intCast(self.m), .n = @intCast(self.n), .nrhs = 1, .lda = @intCast(self.lda), .ldb = @intCast(self.ldb), .a = operand(C, raw, &self.a), .b = operand(C, raw, &self.b), .singular = operand(R, raw, &self.s), .rank = operand(I, raw, &self.rank), .rcond = -1, .work = operand(C, raw, work), .lwork = lwork, .rwork = operand(R, raw, &self.rwork), .iwork = operand(I, raw, &self.iwork) } };
        }
        fn verify(self: *@This(), before: *const @This()) !void {
            const tol: R = if (R == f32) 3e-5 else 3e-12;
            const expected = if (self.m < self.n) [_]C{ value(C, 0.5, 1), value(C, 1, -0.5), value(C, -2, 1) } else [_]C{ value(C, 1, 2), value(C, -2, 1), value(C, 0, 0) };
            for (0..self.n) |row| {
                const x = self.b[pos(self.layout, row, 0, self.ldb)];
                try std.testing.expectApproxEqAbs(expected[row].re, x.re, tol);
                try std.testing.expectApproxEqAbs(expected[row].im, x.im, tol);
            }
            if (self.m > self.n) {
                const residual = self.b[pos(self.layout, self.n, 0, self.ldb)];
                try std.testing.expectApproxEqAbs(@as(R, 6), residual.re * residual.re + residual.im * residual.im, tol * 10);
            } else {
                // Minimum norm: the first two components lie in row(A)^H.
                const x0 = self.b[pos(self.layout, 0, 0, self.ldb)];
                const x1 = self.b[pos(self.layout, 1, 0, self.ldb)];
                try std.testing.expectApproxEqAbs(x0.im, x1.re, tol);
                try std.testing.expectApproxEqAbs(-x0.re, x1.im, tol);
            }
            try bytes(&before.a, &self.a);
            for (self.b, before.b, 0..) |x, y, i| {
                var numerical = false;
                for (0..@max(self.m, self.n)) |row| numerical = numerical or i == pos(self.layout, row, 0, self.ldb);
                if (!numerical) try std.testing.expectEqualDeep(y, x);
            }
            try std.testing.expectEqual(@as(I, 2), self.rank[0]);
            try std.testing.expectEqual(before.rank[1], self.rank[1]);
            try std.testing.expect(self.s[0] >= self.s[1] and self.s[1] > 0);
            for (self.s[2..], before.s[2..]) |x, y| try std.testing.expectEqual(y, x);
        }
    };
}
fn finite(comptime C: type, comptime I: type, comptime raw: bool, comptime scan: bool, level: Level, layout: c_int, wide: bool, minimum: bool) !usize {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("complex GELSD context cleanup");
    r.services.allocator = undefined;
    var f = Fixture(C, I).init(layout, wide);
    const before = f;
    const plan = try owner.complex_gelsd_workspace.Planner(Core).make(C, I, f.m, f.n, 1);
    const min = try owner.complex_gelsd_policy.minimumWork(I, f.m, f.n, 1);
    try std.testing.expect(min < plan.work);
    const count: usize = @intCast(if (minimum) min else plan.work);
    const work = try std.testing.allocator.alloc(C, count + 1);
    defer std.testing.allocator.free(work);
    @memset(work, value(C, 81, -21));
    try info(try adapter(raw, scan).callManaged(C, I, level, &r.context, &r.services, f.args(raw, scan, work[0..count], @intCast(count))), 0);
    try f.verify(&before);
    try std.testing.expectEqualDeep(value(C, 81, -21), work[count]);
    const expected_allocations: usize = if (level == .high) (if (layout == 101) 5 else 3) else @as(usize, if (layout == 101) 2 else 0) + @intFromBool(minimum);
    try lifecycle(&r, expected_allocations);
    try std.testing.expectEqual(@as(usize, if (level == .high and scan) 1 else 0), r.reads);
    if (level == .work) {
        try std.testing.expectEqual(@as(Core.Scalar.Real(C), @floatFromInt(plan.work)), work[0].re);
        try std.testing.expectEqual(@as(Core.Scalar.Real(C), 0), work[0].im);
        try std.testing.expectEqual(@as(Core.Scalar.Real(C), 1), f.rwork[0]);
        try std.testing.expectEqual(@as(I, 1), f.iwork[0]);
    } else {
        for (work) |x| try std.testing.expectEqualDeep(value(C, 81, -21), x);
        try bytes(&before.rwork, &f.rwork);
        try bytes(&before.iwork, &f.iwork);
    }
    try std.testing.expectEqual(before.rwork[1], f.rwork[1]);
    try std.testing.expectEqual(before.iwork[1], f.iwork[1]);
    return r.x.attempts;
}
test "complex GELSD finite managed typed raw high work row column solves use the actual owner" {
    comptime {
        const real_core = @import("zynum-core");
        const real_execution = @import("zynum-execution");
        if (Core.OwnerIdentity != real_core or Core.Execution != real_execution or Core.Resources != real_core.internal.numerics.Resources or Core.ComplexRankLeastSquares != real_core.internal.numerics.ComplexRankLeastSquares) @compileError("complex GELSD tests must use actual owner identities");
    }
    var configs: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |wide| {
        allocations += try finite(C, I, raw, true, level, layout, wide, false);
        configs += 1;
    };
    std.debug.print("FINITE configs={d} admissions={d} finishes={d} allocs=frees={d} closes={d}\n", .{ configs, configs, configs, allocations, configs });
}
test "complex GELSD legal vendor minimum WORK shorter than own recommendation uses original Context fallback" {
    var configs: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |wide| {
        allocations += try finite(C, I, raw, true, .work, layout, wide, true);
        configs += 1;
    };
    std.debug.print("MINIMUM configs={d} admissions={d} finishes={d} allocs=frees={d} closes={d}\n", .{ configs, configs, configs, allocations, configs });
}
test "complex GELSD no scan high profiles share finite numerical and managed owner behavior" {
    var cases: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        allocations += try finite(C, i32, raw, false, .high, layout, true, false);
        cases += 1;
    };
    std.debug.print("NOSCAN configs={d} admissions={d} finishes={d} allocs=frees={d} closes={d}\n", .{ cases, cases, cases, allocations, cases });
}
test "complex GELSD adapter work query writes three metadata slots without numerical reads or allocation" {
    var cases: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const R = Core.Scalar.Real(C);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("query cleanup");
        var f = Fixture(C, I).init(layout, true);
        const before = f;
        var work = [_]C{ value(C, 91, -22), value(C, 92, -23) };
        var args = f.args(raw, true, &work, -1);
        args.column.a = null;
        args.column.b = null;
        args.column.singular = null;
        args.column.rank = null;
        args.column.rcond = std.math.nan(R);
        try info(try adapter(raw, true).callManaged(C, I, .work, &r.context, &r.services, args), 0);
        const plan = try owner.complex_gelsd_workspace.Planner(Core).make(C, I, f.m, f.n, 1);
        try std.testing.expectEqual(@as(R, @floatFromInt(plan.work)), work[0].re);
        try std.testing.expectEqual(@as(R, 0), work[0].im);
        try std.testing.expectEqualDeep(value(C, 92, -23), work[1]);
        try std.testing.expectEqual(@as(R, 1), f.rwork[0]);
        try std.testing.expectEqual(@as(I, 1), f.iwork[0]);
        try bytes(&before.a, &f.a);
        try bytes(&before.b, &f.b);
        try bytes(&before.s, &f.s);
        try bytes(&before.rank, &f.rank);
        try std.testing.expectEqual(before.rwork[1], f.rwork[1]);
        try std.testing.expectEqual(before.iwork[1], f.iwork[1]);
        try std.testing.expectEqual(@as(usize, 0), r.reads);
        try lifecycle(&r, 0);
        cases += 1;
    };
    std.debug.print("QUERY configs={d} admissions={d} finishes={d} allocs=frees=0 closes={d}\n", .{ cases, cases, cases, cases });
}
test "complex GELSD high NaN A then extended B tail then rcond order follows layout validation" {
    var cases: usize = 0;
    var reads: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const R = Core.Scalar.Real(C);
        const nan = std.math.nan(R);
        for ([_]u8{ 0, 1, 2, 3 }) |scenario| {
            var r: h.Rig = .{};
            r.init();
            defer r.done() catch @panic("nan cleanup");
            var f = Fixture(C, i32).init(layout, true);
            var work = [_]C{value(C, 93, -24)};
            var args = f.args(raw, true, &work, 1);
            args.column.rcond = nan;
            f.b[pos(layout, 2, 0, f.ldb)].im = nan;
            if (scenario == 0 or scenario == 3) f.a[pos(layout, 0, 0, f.lda)].im = nan;
            if (scenario == 2) f.b[pos(layout, 2, 0, f.ldb)].im = 0;
            if (scenario == 3) args.layout = 777;
            const before = f;
            try info(try adapter(raw, true).callManaged(C, i32, .high, &r.context, &r.services, args), @as(i32, if (scenario == 3) -1 else if (scenario == 0) -5 else if (scenario == 1) -7 else -10));
            try bytes(&before.a, &f.a);
            try bytes(&before.b, &f.b);
            try bytes(&before.s, &f.s);
            try bytes(&before.rank, &f.rank);
            try bytes(&before.rwork, &f.rwork);
            try bytes(&before.iwork, &f.iwork);
            try std.testing.expectEqual(@as(usize, if (scenario == 3) 0 else 1), r.reads);
            try lifecycle(&r, 0);
            cases += 1;
            reads += r.reads;
        }
    };
    std.debug.print("NAN configs={d} admissions={d} finishes={d} allocs=frees=0 getter_reads={d} closes={d}\n", .{ cases, cases, cases, reads, cases });
}
test "complex GELSD row preflight rejects mixed outputs aliased into original A B before scratch and commits" {
    var cases: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |minimum| for ([_]u8{ 0, 1, 2, 3 }) |scenario| {
        const R = Core.Scalar.Real(C);
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("alias cleanup");
        var f = Fixture(C, i32).init(101, false);
        const plan = try owner.complex_gelsd_workspace.Planner(Core).make(C, i32, 3, 2, 1);
        const count = if (minimum) try owner.complex_gelsd_policy.minimumWork(i32, 3, 2, 1) else plan.work;
        const work = try std.testing.allocator.alloc(C, @intCast(count));
        defer std.testing.allocator.free(work);
        @memset(work, value(C, 94, -25));
        var args = f.args(raw, true, work, count);
        const ap: [*]R = @ptrCast(&f.a);
        const bp: [*]R = @ptrCast(&f.b);
        const ai: [*]i32 = @ptrCast(&f.a);
        const bi: [*]i32 = @ptrCast(&f.b);
        switch (scenario) {
            0 => args.column.singular = operand(R, raw, ap[0..2]),
            1 => args.column.rank = operand(i32, raw, ai[0..1]),
            2 => args.column.rwork = operand(R, raw, bp[0..1]),
            3 => args.column.iwork = operand(i32, raw, bi[0..1]),
            else => unreachable,
        }
        const before = f;
        const result = try adapter(raw, true).callManaged(C, i32, .work, &r.context, &r.services, args);
        switch (result) {
            .admission_fault => |err| try std.testing.expectEqual(error.AliasingNotAllowed, err),
            else => return error.TestUnexpectedResult,
        }
        try bytes(&before.a, &f.a);
        try bytes(&before.b, &f.b);
        try bytes(&before.s, &f.s);
        try bytes(&before.rank, &f.rank);
        try bytes(&before.rwork, &f.rwork);
        try bytes(&before.iwork, &f.iwork);
        for (work) |x| try std.testing.expectEqualDeep(value(C, 94, -25), x);
        try lifecycle(&r, 0);
        cases += 1;
    };
    std.debug.print("ALIAS configs={d} admissions={d} finishes={d} allocs=frees=0 closes={d}\n", .{ cases, cases, cases, cases });
}
test "complex GELSD OOM at every high staging and minimum fallback allocation releases original Context leases" {
    var cases: usize = 0;
    var attempts: usize = 0;
    var frees: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| for ([_]Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| {
        const stages: usize = if (level == .high) (if (layout == 101) 5 else 3) else (if (layout == 101) 3 else 1);
        for (0..stages) |failed| {
            var r: h.Rig = .{};
            r.init();
            defer r.done() catch @panic("oom cleanup");
            r.x.fail_at = failed;
            r.services.allocator = undefined;
            var f = Fixture(C, I).init(layout, false);
            const before = f;
            const minimum = try owner.complex_gelsd_policy.minimumWork(I, 3, 2, 1);
            const work = try std.testing.allocator.alloc(C, @intCast(minimum));
            defer std.testing.allocator.free(work);
            @memset(work, value(C, 95, -26));
            const result = try adapter(false, true).callManaged(C, I, level, &r.context, &r.services, f.args(false, true, work, minimum));
            const expected: i32 = if (level == .high) (if (failed < 3) -1010 else -1011) else (if (layout == 101 and failed < 2) -1011 else -1010);
            try info(result, expected);
            try bytes(&before.a, &f.a);
            try bytes(&before.b, &f.b);
            try bytes(&before.s, &f.s);
            try bytes(&before.rank, &f.rank);
            try std.testing.expectEqual(failed + 1, r.x.attempts);
            try std.testing.expectEqual(failed, r.x.frees);
            try std.testing.expectEqual(@as(usize, 0), r.x.live);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
            try std.testing.expectEqual(@as(usize, 1), counts(&r, .admitted));
            try std.testing.expectEqual(@as(usize, 1), counts(&r, .finishing));
            try std.testing.expectEqual(@as(usize, 1), r.max_active);
            try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
            cases += 1;
            attempts += r.x.attempts;
            frees += r.x.frees;
        }
    };
    std.debug.print("OOM configs={d} admissions={d} finishes={d} attempts={d} frees={d} closes={d}\n", .{ cases, cases, cases, attempts, frees, cases });
}
test "complex GELSD already admitted call continues during close and releases final lease before Context close" {
    var cases: usize = 0;
    var allocations: usize = 0;
    var max_pins: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ false, true }) |raw| for ([_]Level{ .high, .work }) |level| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("admitted cleanup");
        r.services.allocator = undefined;
        var operation = try r.context.begin();
        defer if (operation.context != null) operation.end();
        var lease = try Ctx.ContextScratch(u8).acquire(&operation, 16);
        var leased = true;
        defer if (leased) lease.deinit();
        try std.testing.expectEqual(Ctx.RequestResult.pending, r.context.requestClose());
        try std.testing.expectEqual(Ctx.CloseResult.busy, r.context.tryProgressClose());
        var f = Fixture(C, i32).init(101, false);
        const before = f;
        const minimum = try owner.complex_gelsd_policy.minimumWork(i32, 3, 2, 1);
        const work = try std.testing.allocator.alloc(C, @intCast(minimum));
        defer std.testing.allocator.free(work);
        const args = f.args(raw, true, work, minimum);
        const reads = r.reads;
        var unusable: Core.Services = undefined;
        try std.testing.expectError(error.Closing, adapter(raw, true).callManaged(C, i32, level, &r.context, &unusable, args));
        try std.testing.expectEqual(reads, r.reads);
        try info(try adapter(raw, true).callAdmitted(C, i32, level, &operation, &r.services, args), 0);
        try f.verify(&before);
        try std.testing.expectEqual(@as(usize, 1), counts(&r, .admitted));
        try std.testing.expectEqual(@as(usize, 0), counts(&r, .finishing));
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        try std.testing.expectEqual(@as(usize, 1), r.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 1), r.context.snapshot().resource_pins);
        try std.testing.expectEqual(@as(usize, 1), r.x.live);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.waiting_resources, r.context.tryProgressClose());
        lease.deinit();
        leased = false;
        try std.testing.expectEqual(Ctx.CloseResult.closed, r.context.tryProgressClose());
        try std.testing.expectEqual(@as(usize, 1), counts(&r, .finishing));
        try std.testing.expectEqual(r.x.attempts, r.x.frees);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
        try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
        cases += 1;
        allocations += r.x.attempts;
        for (r.events[0..r.count]) |e| max_pins = @max(max_pins, e.pins);
    };
    std.debug.print("ADMITTED configs={d} admissions={d} finishes={d} allocs=frees={d} rejected_new_admissions={d} max_pins={d} closes={d}\n", .{ cases, cases, cases, allocations, cases, max_pins, cases });
}
fn lalsdHook(comptime C: type) *const fn (*anyopaque, owner.common.ErrorEvent) void {
    return struct {
        fn hook(raw: *anyopaque, e: owner.common.ErrorEvent) void {
            if (e.kind != .fortran_returning_test or e.entry != .fortran_lalsd or e.routine != .lalsd or e.argument != 4 or e.scalar != (if (C == C32) @as(u8, 'c') else @as(u8, 'z')))
                @panic("zero RHS must retain the selected nested CLALSD/ZLALSD diagnostic");
            h.Rig.hookForTest(raw, e);
        }
    }.hook;
}
test "complex GELSD selected zero RHS behavior distinguishes zero matrix from finite nonzero factor without rank commit" {
    var cases: usize = 0;
    var allocations: usize = 0;
    inline for (.{ C32, C64 }) |C| inline for (.{ false, true }) |raw| for ([_]Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |zero| {
        var r: h.Rig = .{};
        r.init();
        r.services.error_event = lalsdHook(C);
        defer r.done() catch @panic("zero rhs cleanup");
        var f = Fixture(C, i32).init(layout, false);
        if (zero) for (0..f.m) |row| for (0..f.n) |col| {
            f.a[pos(layout, row, col, f.lda)] = value(C, 0, 0);
        };
        const before = f;
        const plan = try owner.complex_gelsd_workspace.Planner(Core).make(C, i32, 3, 2, 0);
        const work = try std.testing.allocator.alloc(C, @intCast(plan.work));
        defer std.testing.allocator.free(work);
        var args = f.args(raw, true, work, plan.work);
        args.column.nrhs = 0;
        try info(try adapter(raw, true).callManaged(C, i32, level, &r.context, &r.services, args), @as(i32, if (zero) 0 else -5));
        try bytes(&before.a, &f.a);
        try bytes(&before.b, &f.b);
        for (f.s[2..], before.s[2..]) |x, y| try std.testing.expectEqual(y, x);
        try std.testing.expectEqual(before.rwork[1], f.rwork[1]);
        try std.testing.expectEqual(before.iwork[1], f.iwork[1]);
        if (zero) {
            try std.testing.expectEqual(@as(i32, 0), f.rank[0]);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), 0), f.s[0]);
        } else {
            try bytes(&before.rank, &f.rank);
            var nested = false;
            for (r.events[0..r.count]) |e| if (e.tag == .hook and e.detail == 4) {
                nested = true;
            };
            try std.testing.expect(nested);
        }
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
        try std.testing.expectEqual(r.x.attempts, r.x.frees);
        try std.testing.expectEqual(@as(usize, 1), counts(&r, .admitted));
        try std.testing.expectEqual(@as(usize, 1), counts(&r, .finishing));
        try std.testing.expectEqual(@as(usize, 0), r.x.live);
        try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
        cases += 1;
        allocations += r.x.attempts;
    };
    std.debug.print("ZERO_RHS configs={d} admissions={d} finishes={d} allocs=frees={d} closes={d}\n", .{ cases, cases, cases, allocations, cases });
}
