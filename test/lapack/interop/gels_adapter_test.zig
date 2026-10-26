// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const B = @import("numerical-owner");
const A = B.GelsAdapter;
const Raw = B.RawGelsAdapter;
const N = Core.internal.numerics;
const S = N.Scalar;
const Ctx = N.Execution.context;
const F = N.Execution.test_support;
const E = std.testing;
const types = .{ f32, f64, S.ComplexF32, S.ComplexF64 };
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
fn value(comptime T: type, re: f64, im: f64) T {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn read(comptime T: type, x: T) Z {
    return if (comptime S.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x };
}
fn near(comptime T: type, expected: Z, actual: T) !void {
    const got = read(T, actual);
    const tol = std.math.floatEps(S.Real(T)) * 2048 * @max(1, @abs(expected.re) + @abs(expected.im));
    try E.expectApproxEqAbs(expected.re, got.re, tol);
    try E.expectApproxEqAbs(expected.im, got.im, tol);
}
fn same(comptime T: type, a: []const T, b: []const T) !void {
    try E.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn at(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
fn operand(comptime raw: bool, comptime T: type, values: []T) B.common.access.Operand(T, true, if (raw) .raw else .typed) {
    return if (raw) @ptrCast(values.ptr) else values;
}
fn finish(owner: *Ctx.Context, alloc: *F.Allocations) !void {
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    _ = owner.requestClose();
    try E.expectEqual(Ctx.CloseResult.closed, owner.tryProgressClose());
    try E.expectEqual(alloc.stats().accepted, alloc.stats().freed);
}
const Recorder = struct {
    owner: *Ctx.Context,
    enabled: bool = true,
    scans: usize = 0,
    count: usize = 0,
    events: [20]B.common.ErrorEvent = undefined,
    fn nan(ptr: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.scans += 1;
        return self.enabled;
    }
    fn event(ptr: *anyopaque, e: B.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.owner.snapshot().active_operations != 1) @panic("GELS lost its one admitted operation");
        self.events[self.count] = e;
        self.count += 1;
    }
    fn services(self: *@This()) B.Provider.Services {
        return .{ .allocator = undefined, .context = self, .nan_enabled = nan, .error_event = event };
    }
};

test "GELS boundary query rounds upward with NULL matrices on one closing owner without allocations" {
    comptime {
        if (A.OwnerIdentity != Core or Raw.OwnerIdentity != Core) @compileError("GELS owner mismatch");
    }
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        _ = owner.requestClose();
        for ([_][4]i32{ .{ 2, 3, 2, 4 }, .{ 1, 1, 16777216, 16777217 }, .{ 0, 4, 3, 3 }, .{ 2147483647, 2147483647, 2147483647, 0 } }) |dims| {
            var work = [_]T{value(T, 77, 33)};
            const expected: usize = if (dims[3] == 0) 4294967294 else @intCast(dims[3]);
            const args: Engine.Arguments(T) = .{ .layout = layout, .trans = 'N', .m = dims[0], .n = dims[1], .nrhs = dims[2], .lda = if (layout == 101) dims[1] else @max(1, dims[0]), .ldb = if (layout == 101) dims[2] else @max(1, @max(dims[0], dims[1])), .work = operand(raw, T, &work), .lwork = -1 };
            try E.expectEqual(@as(i32, 0), try Engine.workAdmitted(T, &op, args));
            const got = read(T, work[0]);
            try E.expect(std.math.isFinite(got.re));
            try E.expect(@as(f128, got.re) >= @as(f128, @floatFromInt(expected)));
            try E.expectEqual(@as(f64, 0), got.im);
            if (S.Real(T) == f64) try E.expectEqual(@as(f64, @floatFromInt(expected)), got.re);
        }
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        op.end();
        try finish(&owner, &alloc);
    };
}

fn tall(comptime T: type, i: usize, j: usize) Z {
    const re = [2][3]f64{ .{ 2, 0, 1 }, .{ 0, 3, 0 } };
    const im = [2][3]f64{ .{ 1, 0, 0 }, .{ 0, -2, 0.5 } };
    return .{ .re = re[j][i], .im = if (S.isComplex(T)) im[j][i] else 0 };
}
fn numeric(comptime T: type, comptime raw: bool, high: bool, layout: c_int, is_wide: bool, adjoint: bool) !void {
    const Engine = if (raw) Raw else A;
    const m: usize = if (is_wide) 2 else 3;
    const n: usize = if (is_wide) 3 else 2;
    const p: usize = if (adjoint) n else m;
    const q: usize = if (adjoint) m else n;
    const lda: usize = 5;
    const ldb: usize = 5;
    var am: [6]Z = undefined;
    for (0..n) |j| for (0..m) |i| {
        am[i + j * m] = if (is_wide) conj(tall(T, j, i)) else tall(T, i, j);
    };
    var d: [6]Z = undefined;
    for (0..q) |j| for (0..p) |i| {
        d[i + j * p] = if (adjoint) conj(am[j + i * m]) else am[i + j * m];
    };
    var x: [6]Z = @splat(.{});
    var rhs: [6]Z = @splat(.{});
    for (0..2) |j| {
        if (p >= q) for (0..q) |i| {
            x[i + j * 3] = .{ .re = if (i == 0) 1 else -1, .im = if (S.isComplex(T)) @as(f64, @floatFromInt(i + j + 1)) / 4 else 0 };
        } else for (0..q) |i| for (0..p) |r| {
            x[i + j * 3] = add(x[i + j * 3], mul(conj(d[r + i * p]), .{ .re = @as(f64, @floatFromInt(r + j + 1)) / 4 }));
        };
        for (0..p) |i| for (0..q) |r| {
            rhs[i + j * 3] = add(rhs[i + j * 3], mul(d[i + r * p], x[r + j * 3]));
        };
    }
    var a: [24]T = @splat(value(T, -997, 0));
    var b: [24]T = @splat(value(T, -993, 0));
    var work: [16]T = @splat(value(T, 77, 0));
    for (0..n) |j| for (0..m) |i| {
        const z = am[i + j * m];
        a[at(layout, i, j, lda)] = value(T, z.re, z.im);
    };
    for (0..2) |j| for (0..3) |i| {
        const z = rhs[i + j * 3];
        b[at(layout, i, j, ldb)] = if (i < p) value(T, z.re, z.im) else value(T, 9, 0);
    };
    const oa = a;
    const ob = b;
    var alloc: F.Allocations = .{ .budget = 64 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    _ = owner.requestClose();
    var rec: Recorder = .{ .owner = &owner };
    const args: Engine.Arguments(T) = .{ .layout = layout, .trans = if (!adjoint) 'N' else if (S.isComplex(T)) 'c' else 't', .m = @intCast(m), .n = @intCast(n), .nrhs = 2, .a = operand(raw, T, &a), .lda = @intCast(lda), .b = operand(raw, T, &b), .ldb = @intCast(ldb), .work = operand(raw, T, &work), .lwork = 4 };
    try E.expectEqual(@as(i32, 0), if (high) try Engine.highAdmitted(true, T, &op, &rec.services(), args) else try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
    for (0..2) |j| for (0..q) |i| try near(T, x[i + j * 3], b[at(layout, i, j, ldb)]);
    for (a, 0..) |_, index| {
        var used = false;
        for (0..n) |j| for (0..m) |i| {
            used = used or index == at(layout, i, j, lda);
        };
        if (!used) try same(T, oa[index..][0..1], a[index..][0..1]);
    }
    for (b, 0..) |_, index| {
        var used = false;
        for (0..2) |j| for (0..3) |i| {
            used = used or index == at(layout, i, j, ldb);
        };
        if (!used) try same(T, ob[index..][0..1], b[index..][0..1]);
    }
    for (work[4..]) |v| try near(T, .{ .re = 77 }, v);
    try E.expectEqual(@as(usize, 0), rec.count);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try finish(&owner, &alloc);
}
test "GELS boundary typed raw high work row column N H match independent full rank and minimum norm solutions" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |is_wide| for ([_]bool{ false, true }) |adjoint| try numeric(T, raw, high, layout, is_wide, adjoint);
}

test "GELS boundary scalar work errors preserve driver positions and diagnostic names" {
    inline for (types) |T| {
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var work = [_]T{value(T, 77, 0)};
        var args: A.Arguments(T) = .{ .layout = 99, .trans = 'X', .m = -1, .n = -1, .nrhs = -1, .lda = 0, .ldb = 0, .work = &work };
        const expected = [_]i32{ -1, -2, -3, -4, -5, -7, -9, -11 };
        for (expected, 0..) |info, step| {
            switch (step) {
                0 => {},
                1 => args.layout = 102,
                2 => args.trans = 'N',
                3 => args.m = 0,
                4 => args.n = 0,
                5 => args.nrhs = 0,
                6 => args.lda = 1,
                7 => args.ldb = 1,
                else => unreachable,
            }
            try E.expectEqual(info, try A.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
            const e = rec.events[step];
            try E.expectEqual(B.common.Routine.gels, e.routine);
            try E.expectEqual(if (step == 0) B.common.HookKind.lapacke else .fortran_returning_test, e.kind);
            try E.expectEqual(if (step == 0) info else -info - 1, e.argument);
            const translated = try B.Diagnostics(i32).translate(e);
            var name: [32]u8 = undefined;
            const scalar_tag: u8 = if (T == f32) 's' else if (T == f64) 'd' else if (T == S.ComplexF32) 'c' else 'z';
            const expected_name = if (step == 0) try std.fmt.bufPrint(&name, "LAPACKE_{c}gels_work", .{scalar_tag}) else try std.fmt.bufPrint(&name, "{c}GELS", .{std.ascii.toUpper(scalar_tag)});
            try E.expectEqualStrings(expected_name, translated.name());
        }
        try near(T, .{ .re = 1 }, work[0]);
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        op.end();
        try finish(&owner, &alloc);
    }
}

test "GELS boundary row leading dimensions precede driver errors and negative empty copies remain safe" {
    var alloc: F.Allocations = .{ .budget = 8 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var args: A.Arguments(f64) = .{ .layout = 101, .trans = 'X', .m = -2, .n = 2, .nrhs = 1, .lda = 0, .ldb = 0, .lwork = -1 };
    try E.expectEqual(@as(i32, -7), try A.workAdmitted(f64, &op, args));
    args.lda = 2;
    try E.expectEqual(@as(i32, -9), try A.workAdmitted(f64, &op, args));
    args.ldb = 1;
    try E.expectEqual(@as(i32, -2), try A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    args.trans = 'N';
    args.n = -3;
    args.nrhs = -1;
    args.lda = -2;
    args.ldb = -1;
    args.lwork = 0;
    try E.expectEqual(@as(i32, -3), try A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(usize, 2), alloc.stats().accepted);
    op.end();
    try finish(&owner, &alloc);
}

test "GELS boundary high NaN scans A then full max rows of B with runtime compile controls" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 32 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var a = [_]T{ value(T, std.math.nan(f64), 0), value(T, 0, 0) };
        var b = [_]T{ value(T, 2, 0), value(T, std.math.nan(f64), 0) };
        var args: Engine.Arguments(T) = .{ .layout = 102, .trans = 'X', .m = 1, .n = 2, .nrhs = 1, .a = operand(raw, T, &a), .lda = 1, .b = operand(raw, T, &b), .ldb = 2 };
        try E.expectEqual(@as(i32, -6), try Engine.highAdmitted(true, T, &op, &rec.services(), args));
        a[0] = value(T, 1, 0);
        try E.expectEqual(@as(i32, -8), try Engine.highAdmitted(true, T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 0), rec.count);
        const scans = rec.scans;
        try E.expectEqual(@as(i32, -2), try Engine.highAdmitted(false, T, &op, &rec.services(), args));
        try E.expectEqual(scans, rec.scans);
        args.trans = 'N';
        rec.enabled = false;
        try E.expectEqual(@as(i32, 0), try Engine.highAdmitted(true, T, &op, &rec.services(), args));
        try near(T, .{ .re = 2 }, b[0]);
        try near(T, .{}, b[1]);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GELS boundary all allocation budgets clean row high and native leases and emit one diagnostic" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        var failures: usize = 0;
        var successes: usize = 0;
        for (0..7) |budget| {
            var alloc: F.Allocations = .{ .budget = budget };
            var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
            var op = try owner.begin();
            var rec: Recorder = .{ .owner = &owner };
            var a = [_]T{ value(T, 2, 0), value(T, 0, 0), value(T, 0, 0), value(T, 3, 0) };
            var b = [_]T{ value(T, 4, 0), value(T, 9, 0) };
            const oa = a;
            const ob = b;
            var work: [4]T = undefined;
            const args: Engine.Arguments(T) = .{ .layout = layout, .trans = 'N', .m = 2, .n = 2, .nrhs = 1, .a = operand(raw, T, &a), .lda = 2, .b = operand(raw, T, &b), .ldb = if (layout == 102) 2 else 1, .work = operand(raw, T, &work), .lwork = 4 };
            const info = if (high) try Engine.highAdmitted(true, T, &op, &rec.services(), args) else try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args);
            if (info == 0) {
                successes += 1;
                try E.expectEqual(@as(usize, 0), rec.count);
                try near(T, .{ .re = 2 }, b[0]);
                try near(T, .{ .re = 3 }, b[1]);
            } else {
                failures += 1;
                try E.expect(info == -1010 or info == -1011);
                try E.expectEqual(@as(usize, 1), rec.count);
                try E.expectEqual(info, rec.events[0].argument);
                try E.expectEqual(if (high and info == -1010) B.common.Entry.high else .work, rec.events[0].entry);
                try same(T, &oa, &a);
                try same(T, &ob, &b);
            }
            op.end();
            try finish(&owner, &alloc);
        }
        try E.expect(failures > 0 and successes > 0);
    };
}

test "GELS boundary query head access is staged and raw alignment faults allocate nothing" {
    var alloc: F.Allocations = .{ .budget = 0 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var args: Raw.Arguments(f64) = .{ .layout = 102, .trans = 'X', .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .lwork = -1 };
    try E.expectEqual(@as(i32, -2), try Raw.workAdmitted(f64, &op, args));
    args.trans = 'N';
    try E.expectError(error.MissingBuffer, Raw.workAdmitted(f64, &op, args));
    var bytes: [32]u8 align(8) = @splat(77);
    args.work = @ptrCast(&bytes[1]);
    try E.expectError(error.MisalignedPointer, Raw.workAdmitted(f64, &op, args));
    for (bytes) |z| try E.expectEqual(@as(u8, 77), z);
    var work = [_]f64{77};
    args.work = @ptrCast(&work);
    args.a = @ptrCast(&bytes[1]);
    args.b = @ptrCast(&bytes[1]);
    try E.expectEqual(@as(i32, 0), try Raw.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(f64, 4), work[0]);
    var typed: A.Arguments(f64) = .{ .layout = 102, .trans = 'N', .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .work = work[0..0], .lwork = -1 };
    try E.expectError(error.BufferTooSmall, A.workAdmitted(f64, &op, typed));
    typed.work = &work;
    typed.a = &work;
    typed.b = &work;
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(f64, &op, typed));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner, &alloc);
}

test "GELS boundary column empty calls read no A and zero all max rows with padding kept" {
    inline for (types) |T| inline for (.{ false, true }) |raw| inline for (.{ true, false }) |zero_m| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var b: [10]T = @splat(value(T, -993, 0));
        for (0..2) |j| for (0..3) |i| {
            b[i + j * 4] = value(T, std.math.nan(f64), 0);
        };
        const ob = b;
        var work = [_]T{value(T, 77, 0)};
        var args: Engine.Arguments(T) = .{ .layout = 102, .trans = 'N', .m = if (zero_m) 0 else 3, .n = if (zero_m) 3 else 0, .nrhs = 2, .lda = if (zero_m) 1 else 3, .b = operand(raw, T, &b), .ldb = 4, .work = operand(raw, T, &work), .lwork = 2 };
        try E.expectEqual(@as(i32, 0), try Engine.workAdmitted(T, &op, args));
        for (0..2) |j| for (0..3) |i| try near(T, .{}, b[i + j * 4]);
        try same(T, ob[3..4], b[3..4]);
        try same(T, ob[7..], b[7..]);
        args.m = 3;
        args.n = 2;
        args.nrhs = 0;
        args.lda = 3;
        args.ldb = 3;
        args.b = null;
        args.lwork = 4;
        try E.expectEqual(@as(i32, 0), try Engine.workAdmitted(T, &op, args));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GELS boundary singular INFO copies back factored A and transformed RHS without error callback" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 16 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var a: [6]T = @splat(value(T, 0, 0));
        a[at(layout, 1, 1, if (layout == 102) 3 else 2)] = value(T, 1, 0);
        a[at(layout, 2, 1, if (layout == 102) 3 else 2)] = value(T, 1, 0);
        var b = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0) };
        var work: [4]T = undefined;
        const args: Engine.Arguments(T) = .{ .layout = layout, .trans = 'N', .m = 3, .n = 2, .nrhs = 1, .a = operand(raw, T, &a), .lda = if (layout == 102) 3 else 2, .b = operand(raw, T, &b), .ldb = if (layout == 102) 3 else 1, .work = operand(raw, T, &work), .lwork = 4 };
        try E.expectEqual(@as(i32, 1), if (high) try Engine.highAdmitted(true, T, &op, &rec.services(), args) else try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try near(T, .{ .re = 1 }, b[0]);
        try near(T, .{ .re = -5 / @sqrt(@as(f64, 2)) }, b[1]);
        try near(T, .{ .re = 1 / @sqrt(@as(f64, 2)) }, b[2]);
        try E.expectEqual(@as(usize, 0), rec.count);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GELS boundary shared managed route captures existing callbacks and rejects a closed Context" {
    inline for (types) |T| {
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var rec: Recorder = .{ .owner = &owner };
        const callback: B.shared_services.Callback = .{ .userdata = &rec, .function = Recorder.event };
        const args: Raw.Arguments(T) = .{ .layout = 99, .trans = 'N', .m = 0, .n = 0, .nrhs = 0, .lda = 1, .ldb = 1 };
        try E.expectEqual(@as(i32, -1), try B.Shared.gelsRawManagedWithDefaults(true, T, .high, &owner, args, .{ .lapacke = callback, .fortran_returning_test = callback }));
        try E.expectEqual(@as(usize, 1), rec.count);
        try E.expectEqual(B.common.Entry.high, rec.events[0].entry);
        try E.expectEqual(B.common.Routine.gels, rec.events[0].routine);
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try finish(&owner, &alloc);
        try E.expectError(error.Closing, B.Shared.gelsRawManagedWithDefaults(true, T, .high, &owner, args, .{}));
    }
}

test "GELS boundary high oversized upward query fails allocation cleanly before integer conversion" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        const args: Engine.Arguments(T) = .{ .layout = layout, .trans = 'N', .m = 1, .n = 1, .nrhs = 2147483647, .lda = 1, .ldb = if (layout == 101) 2147483647 else 1 };
        try E.expectEqual(@as(i32, -1010), try Engine.highAdmitted(false, T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 1), rec.count);
        try E.expectEqual(B.common.Entry.high, rec.events[0].entry);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GELS boundary scalar operation restrictions alias faults and raw data alignment retain their stages" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var work: [2]T = @splat(value(T, 77, 0));
        var a = [_]T{ value(T, 1, 0), value(T, 9, 0) };
        var b = [_]T{value(T, 3, 0)};
        var args: Engine.Arguments(T) = .{ .layout = 102, .trans = if (S.isComplex(T)) 'T' else 'C', .m = 1, .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .lwork = -1 };
        try E.expectEqual(@as(i32, -2), try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        args.trans = 'N';
        args.lwork = 1;
        args.work = operand(raw, T, &work);
        try E.expectEqual(@as(i32, -11), try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try near(T, .{ .re = 2 }, work[0]);
        const events = rec.count;
        args.lwork = 2;
        args.a = operand(raw, T, &a);
        args.b = operand(raw, T, a[0..1]);
        try E.expectError(error.AliasingNotAllowed, Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try near(T, .{ .re = 1 }, a[0]);
        args.b = operand(raw, T, &b);
        args.work = operand(raw, T, &a);
        try E.expectError(error.AliasingNotAllowed, Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        // The query head is published before a nonquery overlap is admitted.
        try near(T, .{ .re = 2 }, a[0]);
        if (comptime raw) {
            var bytes: [80]u8 align(16) = @splat(77);
            args.work = operand(raw, T, &work);
            args.a = @ptrCast(&bytes[1]);
            try E.expectError(error.MisalignedPointer, Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
            args.a = operand(raw, T, &a);
            args.b = @ptrCast(&bytes[1]);
            try E.expectError(error.MisalignedPointer, Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
            for (bytes) |byte| try E.expectEqual(@as(u8, 77), byte);
        }
        try E.expectEqual(events, rec.count);
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GELS boundary short work follows row copies but precedes all column matrix access" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        const Engine = if (raw) Raw else A;
        var alloc: F.Allocations = .{ .budget = 8 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var work = [_]T{value(T, 77, 0)};
        var b = [_]T{ value(T, 4, 0), value(T, 9, 0) };
        var args: Engine.Arguments(T) = .{ .layout = 102, .trans = 'N', .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .work = operand(raw, T, &work), .lwork = 3 };
        try E.expectEqual(@as(i32, -11), try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try near(T, .{ .re = 4 }, work[0]);
        work[0] = value(T, 77, 0);
        args.layout = 101;
        args.ldb = 1;
        args.b = operand(raw, T, &b);
        const events = rec.count;
        try E.expectError(error.MissingBuffer, Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try E.expectEqual(events, rec.count);
        try near(T, .{ .re = 77 }, work[0]);
        try E.expectEqual(@as(usize, 2), alloc.stats().accepted);
        var a = [_]T{ value(T, 2, 0), value(T, 0, 0), value(T, 0, 0), value(T, 3, 0) };
        const before_a = a;
        const before_b = b;
        args.a = operand(raw, T, &a);
        try E.expectEqual(@as(i32, -11), try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 4), alloc.stats().accepted);
        try E.expectEqual(@as(i32, 10), rec.events[rec.count - 1].argument);
        try near(T, .{ .re = 4 }, work[0]);
        try same(T, &before_a, &a);
        try same(T, &before_b, &b);
        op.end();
        try finish(&owner, &alloc);
    };
}
