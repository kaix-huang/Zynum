// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const Owner = @import("numerical-owner");
const A = Owner.GeneralizedSymmetricAdapter;
const Raw = Owner.RawGeneralizedSymmetricAdapter;
const Ex = Core.internal.numerics.Execution;
const context = Ex.context;
const fixture = Ex.test_support;
const E = std.testing;

fn finish(owner: *context.Context, allocations: *fixture.Allocations) !void {
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.closed, owner.tryProgressClose());
}
fn same(comptime T: type, a: []const T, b: []const T) !void {
    try E.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn near(expected: f64, actual: f64, tolerance: f64) !void {
    try E.expect(std.math.isFinite(actual));
    try E.expectApproxEqAbs(expected, actual, tolerance);
}
fn index(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
const Recorder = struct {
    owner: *context.Context,
    enabled: bool = true,
    nan_calls: usize = 0,
    events: [8]Owner.common.ErrorEvent = undefined,
    count: usize = 0,
    fn nan(ptr: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.nan_calls += 1;
        return self.enabled;
    }
    fn event(ptr: *anyopaque, value: Owner.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.owner.snapshot().active_operations != 1) @panic("SYGV lost admitted owner");
        self.events[self.count] = value;
        self.count += 1;
    }
    fn services(self: *@This()) Owner.Provider.Services {
        return .{ .allocator = undefined, .context = self, .nan_enabled = nan, .error_event = event };
    }
};

test "SYGV queries preserve scalar and row dimension precedence without allocation after close" {
    comptime {
        if (A.OwnerIdentity != Core or Raw.OwnerIdentity != Core) @compileError("SYGV owner differs");
    }
    inline for (.{ f32, f64 }) |T| {
        var allocations: fixture.Allocations = .{ .budget = 0 };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        _ = owner.requestClose();
        var work = [_]T{999};
        var args: A.Arguments(T) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 4, .lda = 4, .ldb = 4, .work = &work, .lwork = -1 };
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &operation, args));
        try E.expect(work[0] >= 11);
        args.itype = 0;
        args.layout = 101;
        args.lda = 3;
        work[0] = 999;
        try E.expectEqual(@as(i32, -7), try A.workAdmitted(T, &operation, args));
        args.lda = 4;
        args.ldb = 3;
        try E.expectEqual(@as(i32, -9), try A.workAdmitted(T, &operation, args));
        args.ldb = 4;
        try E.expectEqual(@as(i32, -2), try A.workAdmitted(T, &operation, args));
        try E.expectEqual(@as(T, 999), work[0]);
        args.itype = 1;
        args.layout = 102;
        args.lwork = 10;
        try E.expectEqual(@as(i32, -12), try A.workAdmitted(T, &operation, args));
        try E.expect(work[0] >= 11);
        args.n = 0;
        args.lda = 1;
        args.ldb = 1;
        args.lwork = 1;
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &operation, args));
        try E.expectEqual(@as(usize, 0), allocations.stats().attempts);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        operation.end();
        try finish(&owner, &allocations);
    }
}

test "SYGV all three equations satisfy analytic eigenvalues residuals metric normalization and B factors" {
    const original_a = [2][2]f64{ .{ 4, 1 }, .{ 1, 3 } };
    const original_b = [2][2]f64{ .{ 3, 0.5 }, .{ 0.5, 2 } };
    const inverse_b = [2][2]f64{ .{ 2.0 / 5.75, -0.5 / 5.75 }, .{ -0.5 / 5.75, 3.0 / 5.75 } };
    inline for (.{ f32, f64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ("UL") |uplo| for ([_]i32{ 1, 2, 3 }) |itype| for ("VN") |jobz| for ([_]bool{ false, true }) |high| {
        var allocations: fixture.Allocations = .{ .budget = 20 };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        _ = owner.requestClose();
        var recorder: Recorder = .{ .owner = &owner };
        var a: [8]T = @splat(std.math.nan(T));
        var b: [8]T = @splat(std.math.nan(T));
        var values: [4]T = @splat(std.math.nan(T));
        var work: [8]T = @splat(std.math.nan(T));
        for (0..2) |i| for (0..2) |j| {
            if (if (uplo == 'U') i <= j else i >= j) {
                a[index(layout, i, j, 3)] = @floatCast(original_a[i][j]);
                b[index(layout, i, j, 3)] = @floatCast(original_b[i][j]);
            }
        };
        const before_a = a;
        const before_b = b;
        const args: A.Arguments(T) = .{ .layout = layout, .itype = itype, .jobz = jobz, .uplo = uplo, .n = 2, .a = &a, .lda = 3, .b = &b, .ldb = 3, .w = &values, .work = &work, .lwork = 5 };
        const services = recorder.services();
        const info = if (high) try A.highAdmitted(true, T, &operation, &services, args) else try A.workWithDiagnosticsAdmitted(T, &operation, &services, args);
        try E.expectEqual(@as(i32, 0), info);
        try E.expectEqual(@as(usize, 0), recorder.count);
        const tol = 128 * @as(f64, std.math.floatEps(T));
        const expected = if (itype == 1) [2]f64{ (16 - @sqrt(@as(f64, 3))) / 11.5, (16 + @sqrt(@as(f64, 3))) / 11.5 } else [2]f64{ 9.5 - 3 * @sqrt(@as(f64, 3)), 9.5 + 3 * @sqrt(@as(f64, 3)) };
        for (expected, 0..) |value, j| try near(value, values[j], tol * 16);
        if (jobz == 'V') {
            for (0..2) |j| for (0..2) |i| {
                var ax: f64 = 0;
                var bx: f64 = 0;
                var abx: f64 = 0;
                var bax: f64 = 0;
                var metric: f64 = 0;
                for (0..2) |k| {
                    const x: f64 = a[index(layout, k, j, 3)];
                    ax += original_a[i][k] * x;
                    bx += original_b[i][k] * x;
                    for (0..2) |l| {
                        abx += original_a[i][k] * original_b[k][l] * @as(f64, a[index(layout, l, j, 3)]);
                        bax += original_b[i][k] * original_a[k][l] * @as(f64, a[index(layout, l, j, 3)]);
                        metric += @as(f64, a[index(layout, k, i, 3)]) * (if (itype == 3) inverse_b[k][l] else original_b[k][l]) * @as(f64, a[index(layout, l, j, 3)]);
                    }
                }
                const residual = switch (itype) {
                    1 => ax - @as(f64, values[j]) * bx,
                    2 => abx - @as(f64, values[j]) * @as(f64, a[index(layout, i, j, 3)]),
                    3 => bax - @as(f64, values[j]) * @as(f64, a[index(layout, i, j, 3)]),
                    else => unreachable,
                };
                try near(0, residual, tol * 32);
                try near(if (i == j) 1 else 0, metric, tol * 8);
            };
        } else try same(T, &before_a, &a);
        try near(@sqrt(@as(f64, 3)), b[index(layout, 0, 0, 3)], tol);
        try near(@sqrt(@as(f64, 23.0 / 12.0)), b[index(layout, 1, 1, 3)], tol);
        try near(0.5 / @sqrt(@as(f64, 3)), b[index(layout, if (uplo == 'U') 0 else 1, if (uplo == 'U') 1 else 0, 3)], tol);
        const ignored = index(layout, if (uplo == 'U') 1 else 0, if (uplo == 'U') 0 else 1, 3);
        try same(T, before_b[ignored .. ignored + 1], b[ignored .. ignored + 1]);
        for ([_]usize{ 2, 5, 6, 7 }) |at| {
            try same(T, before_a[at .. at + 1], a[at .. at + 1]);
            try same(T, before_b[at .. at + 1], b[at .. at + 1]);
        }
        try E.expect(std.math.isNan(values[2]) and std.math.isNan(values[3]));
        if (!high) for (work[5..]) |value| try E.expect(std.math.isNan(value));
        operation.end();
        try finish(&owner, &allocations);
    };
}

test "SYGV B failure reports n plus the exact minor without touching absent column A or eigenvalues" {
    inline for (.{ f32, f64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]usize{ 0, 1 }) |bad| {
        var allocations: fixture.Allocations = .{ .budget = 20 };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        var a = [_]T{ 4, 1, 1, 3 };
        const before = a;
        var b = [_]T{ 1, 0, 0, 1 };
        b[bad * 3] = -1;
        var work: [5]T = undefined;
        const args: A.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .a = if (layout == 101) &a else null, .lda = 2, .b = &b, .ldb = 2, .work = &work, .lwork = 5 };
        try E.expectEqual(@as(i32, @intCast(3 + bad)), try A.workAdmitted(T, &operation, args));
        try same(T, &before, &a);
        operation.end();
        try finish(&owner, &allocations);
    };
}

test "SYGV row full-matrix conversion occurs before scalar validation and releases both transpose leases" {
    for ([_]usize{ 0, 1, 2 }) |budget| {
        var allocations: fixture.Allocations = .{ .budget = budget };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        var a = [_]f64{ 4, std.math.nan(f64), 1, 3 };
        var b = [_]f64{ 3, std.math.nan(f64), 0.5, 2 };
        const before_a = a;
        const before_b = b;
        var args: A.Arguments(f64) = .{ .layout = 101, .itype = 0, .jobz = 'X', .uplo = 'X', .n = 2, .a = &a, .lda = 2, .b = &b, .ldb = 2 };
        try E.expectEqual(@as(i32, if (budget < 2) -1011 else -2), try A.workAdmitted(f64, &operation, args));
        try same(f64, &before_a, &a);
        try same(f64, &before_b, &b);
        if (budget == 2) {
            allocations.budget += 2;
            args.a = null;
            try E.expectError(error.MissingBuffer, A.workAdmitted(f64, &operation, args));
            allocations.budget += 2;
            args.a = &.{};
            try E.expectError(error.BufferTooSmall, A.workAdmitted(f64, &operation, args));
        }
        operation.end();
        try finish(&owner, &allocations);
    }
}

test "SYGV raw query skips poison matrices while real access checks alignment and aliases" {
    var allocations: fixture.Allocations = .{ .budget = 20 };
    var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
    var operation = try owner.begin();
    var work: [5]f64 = undefined;
    var args: Raw.Arguments(f64) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = @ptrFromInt(1), .lda = 2, .b = @ptrFromInt(1), .ldb = 2, .w = @ptrFromInt(1), .work = @ptrCast(&work), .lwork = -1 };
    try E.expectEqual(@as(i32, 0), try Raw.workAdmitted(f64, &operation, args));
    try E.expectEqual(@as(usize, 0), allocations.stats().attempts);
    args.lwork = 5;
    try E.expectError(error.MisalignedPointer, Raw.workAdmitted(f64, &operation, args));
    var a = [_]f64{ 4, 1, 1, 3 };
    var b = [_]f64{ 3, 0.5, 0.5, 2 };
    args.a = @ptrCast(&a);
    args.b = @ptrCast(&b);
    args.w = @ptrCast(&a);
    try E.expectError(error.AliasingNotAllowed, Raw.workAdmitted(f64, &operation, args));
    operation.end();
    try finish(&owner, &allocations);
}

test "SYGV high scan precedes invalid itype and diagnostics use exact SYGV channels and signs" {
    var allocations: fixture.Allocations = .{ .budget = 20 };
    var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
    var operation = try owner.begin();
    var recorder: Recorder = .{ .owner = &owner };
    const services = recorder.services();
    var a = [_]f64{ std.math.nan(f64), 0, 0, 1 };
    var b = [_]f64{ std.math.nan(f64), 0, 0, 1 };
    var args: A.Arguments(f64) = .{ .layout = 102, .itype = 0, .jobz = 'V', .uplo = 'L', .n = 2, .a = &a, .lda = 2, .b = &b, .ldb = 2 };
    try E.expectEqual(@as(i32, -6), try A.highAdmitted(true, f64, &operation, &services, args));
    a[0] = 1;
    try E.expectEqual(@as(i32, -8), try A.highAdmitted(true, f64, &operation, &services, args));
    try E.expectEqual(@as(usize, 0), recorder.count);
    recorder.enabled = false;
    try E.expectEqual(@as(i32, -2), try A.highAdmitted(true, f64, &operation, &services, args));
    try E.expectEqual(Owner.common.Routine.sygv, recorder.events[0].routine);
    try E.expectEqual(Owner.common.Entry.fortran_sygv, recorder.events[0].entry);
    try E.expectEqual(@as(i32, 1), recorder.events[0].argument);
    const message = try Owner.Diagnostics(i32).translate(recorder.events[0]);
    try E.expectEqualStrings("DSYGV", message.name());
    args.layout = 101;
    args.lda = 1;
    try E.expectEqual(@as(i32, -7), try A.highAdmitted(false, f64, &operation, &services, args));
    try E.expectEqual(Owner.common.HookKind.lapacke, recorder.events[1].kind);
    try E.expectEqual(Owner.common.Entry.work, recorder.events[1].entry);
    try E.expectEqual(@as(i32, -7), recorder.events[1].argument);
    operation.end();
    try finish(&owner, &allocations);
}

test "SYGV allocation failures retain one admission release every lease and emit one memory diagnostic" {
    for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |high| for (0..7) |budget| {
        var allocations: fixture.Allocations = .{ .budget = budget };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        _ = owner.requestClose();
        var recorder: Recorder = .{ .owner = &owner };
        const services = recorder.services();
        var a = [_]f64{ 4, 1, 1, 3 };
        var b = [_]f64{ 3, 0.5, 0.5, 2 };
        var w: [2]f64 = undefined;
        var work: [5]f64 = undefined;
        const args: A.Arguments(f64) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .a = &a, .lda = 2, .b = &b, .ldb = 2, .w = &w, .work = &work, .lwork = 5 };
        const info = if (high) try A.highAdmitted(true, f64, &operation, &services, args) else try A.workWithDiagnosticsAdmitted(f64, &operation, &services, args);
        try E.expect(info == 0 or info == -1010 or info == -1011);
        try E.expectEqual(@as(usize, if (info == 0) 0 else 1), recorder.count);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        operation.end();
        try finish(&owner, &allocations);
    };
}

test "SYGV IEEE driver reports two actual unconverged tridiagonal edges" {
    inline for (.{ f32, f64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var allocations: fixture.Allocations = .{ .budget = 20 };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var operation = try owner.begin();
        var a = [_]T{ 4, 1, 0, 1, 3, 1, 0, 1, 2 };
        var b = [_]T{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        var w: [3]T = undefined;
        var work: [8]T = undefined;
        const args: A.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 3, .a = &a, .lda = 3, .b = &b, .ldb = 3, .w = &w, .work = &work, .lwork = 8, .native_options = .{ .max_sweeps = 0 } };
        try E.expectEqual(@as(i32, 2), try A.workAdmitted(T, &operation, args));
        operation.end();
        try finish(&owner, &allocations);
    };
}

const Probe = struct {
    backing: fixture.Allocations = .{ .budget = 10 },
    owner: *context.Context = undefined,
    close_on_alloc: bool = false,
    alloc_snapshot: ?context.Snapshot = null,
    free_snapshot: ?context.Snapshot = null,
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(raw: *anyopaque) *@This() {
        return @ptrCast(@alignCast(raw));
    }
    fn alloc(raw: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(raw);
        self.alloc_snapshot = self.owner.snapshot();
        if (self.close_on_alloc) {
            _ = self.owner.requestClose();
            if (self.owner.tryProgressClose() != .busy) @panic("SYGV allocation is not admitted");
        }
        return self.backing.allocator().rawAlloc(n, a, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, data: []u8, a: std.mem.Alignment, ret: usize) void {
        const self = cast(raw);
        self.free_snapshot = self.owner.snapshot();
        if (self.owner.tryProgressClose() == .closed) @panic("SYGV close passed managed free");
        self.backing.allocator().rawFree(data, a, ret);
    }
};
test "SYGV configured Shared entry holds exactly one operation through closing allocations and frees" {
    var probe: Probe = .{ .close_on_alloc = true };
    var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
    probe.owner = &owner;
    var other = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var a = [_]f64{ 4, 1, 1, 3 };
    var b = [_]f64{ 3, 0.5, 0.5, 2 };
    var w: [2]f64 = undefined;
    const args: Raw.Arguments(f64) = .{ .layout = 101, .jobz = 'V', .uplo = 'L', .n = 2, .a = @ptrCast(&a), .lda = 2, .b = @ptrCast(&b), .ldb = 2, .w = @ptrCast(&w) };
    try E.expectEqual(@as(i32, 0), try Owner.Shared.generalizedSymmetricRawManagedWithDefaults(true, f64, .high, &owner, args, .{}));
    try near((16 - @sqrt(@as(f64, 3))) / 11.5, w[0], 1e-12);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), other.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), other.snapshot().resource_pins);
    try E.expectError(error.Closing, owner.begin());
    try finish(&owner, &probe.backing);
    _ = other.requestClose();
    try E.expectEqual(context.CloseResult.closed, other.tryProgressClose());
}
