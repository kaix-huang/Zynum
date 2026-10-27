// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const B = @import("numerical-owner");
const A = B.SymmetricAdapter;
const Raw = B.RawSymmetricAdapter;
const E = B.Provider.Execution;
const C = E.context;
const equal = std.testing.expectEqual;
const expect = std.testing.expect;
const Log = struct {
    owner: *C.Context,
    events: [16]B.common.ErrorEvent = undefined,
    pins: [16]usize = undefined,
    count: usize = 0,
    nan_calls: usize = 0,
    nan_enabled: bool = true,
    close_on_event: bool = false,
    fn nan(raw: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.nan_calls += 1;
        return self.nan_enabled;
    }
    fn record(raw: *anyopaque, event: B.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.events[self.count] = event;
        self.pins[self.count] = self.owner.snapshot().resource_pins;
        self.count += 1;
        if (self.close_on_event) _ = self.owner.requestClose();
    }
    fn services(self: *@This()) B.Provider.Services {
        return .{ .allocator = std.testing.failing_allocator, .context = self, .nan_enabled = nan, .error_event = record };
    }
};
fn finish(owner: *C.Context) void {
    _ = owner.requestClose();
    equal(C.CloseResult.closed, fixtureProgressClose(owner)) catch unreachable;
}
fn prohibitDefault() void {
    C.Context.testing.configure(E.getContext(), 0, null);
}
fn restoreDefault() void {
    C.Context.testing.configure(E.getContext(), std.math.maxInt(usize), null);
}
fn idx(layout: c_int, ld: usize, i: usize, j: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
fn same(comptime T: type, before: []const T, after: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(after));
}

pub fn checkFiniteOracle() !void {
    prohibitDefault();
    defer restoreDefault();
    inline for (.{ f32, f64 }) |T| {
        const tol: f64 = if (T == f32) 5e-5 else 2e-12;
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'U', 'L' }) |uplo| for ([_]u8{ 'N', 'V' }) |job| for ([_]bool{ false, true }) |high| {
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            defer finish(owner);
            var operation = try owner.begin();
            defer operation.end();
            _ = owner.requestClose();
            var log: Log = .{ .owner = owner };
            const services = log.services();
            const original = [_]T{ 2, 1, 0, 0, 1, 2, 0, 0, 0, 0, 5, 2, 0, 0, 2, 5 };
            var a: [28]T = @splat(std.math.nan(T));
            for (0..4) |j| for (0..4) |i| {
                if (if (uplo == 'U') i <= j else i >= j) a[idx(layout, 6, i, j)] = original[i + j * 4];
            };
            const before = a;
            var values: [6]T = @splat(-81);
            var work: [16]T = @splat(-82);
            const args: A.Arguments(T) = .{ .layout = layout, .jobz = job, .uplo = uplo, .n = 4, .a = &a, .lda = 6, .w = &values, .work = &work, .lwork = 11 };
            const info = if (high) try A.highAdmitted(true, T, &operation, &services, args) else try A.workWithDiagnosticsAdmitted(T, &operation, &services, args);
            try equal(@as(i32, 0), info);
            for ([_]f64{ 1, 3, 3, 7 }, 0..) |expected, j| try std.testing.expectApproxEqAbs(expected, @as(f64, values[j]), tol);
            if (job == 'V') for (0..4) |j| {
                for (0..4) |i| {
                    var av: f64 = 0;
                    for (0..4) |k| av += @as(f64, original[i + k * 4]) * @as(f64, a[idx(layout, 6, k, j)]);
                    try std.testing.expectApproxEqAbs(av, @as(f64, a[idx(layout, 6, i, j)]) * @as(f64, values[j]), tol);
                }
                for (0..4) |k| {
                    var dot: f64 = 0;
                    for (0..4) |i| dot += @as(f64, a[idx(layout, 6, i, j)]) * @as(f64, a[idx(layout, 6, i, k)]);
                    try std.testing.expectApproxEqAbs(@as(f64, if (j == k) 1 else 0), dot, tol);
                }
            };
            for (0..28) |k| {
                const i = if (layout == 102) k % 6 else k / 6;
                const j = if (layout == 102) k / 6 else k % 6;
                if (i >= 4 or j >= 4 or (job == 'N' and (if (uplo == 'U') i > j else i < j))) try same(T, before[k..][0..1], a[k..][0..1]);
            }
            try std.testing.expectEqualSlices(T, &.{ -81, -81 }, values[4..]);
            try std.testing.expectEqualSlices(T, &.{ -82, -82, -82, -82, -82 }, work[11..]);
            try equal(@as(usize, if (high) 1 else 0), log.nan_calls);
            try equal(@as(usize, 0), log.count);
            try equal(@as(usize, 1), owner.snapshot().active_operations);
            try equal(@as(usize, 0), owner.snapshot().resource_pins);
        };
    }
}

test "syev adapter candidate both precisions layouts triangles jobs and levels satisfy independent eigen oracle" {
    try checkFiniteOracle();
}

pub fn checkQueryAndScalar() !void {
    prohibitDefault();
    defer restoreDefault();
    inline for (.{ f32, f64 }) |T| {
        var spy: E.test_support.Allocations = .{ .budget = 0 };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(spy.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer finish(owner);
        var operation = try owner.begin();
        defer operation.end();
        _ = owner.requestClose();
        var log: Log = .{ .owner = owner };
        const services = log.services();
        var query = [_]T{-7};
        for ([_]c_int{ 101, 102 }) |layout| {
            const info = try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = layout, .jobz = 'v', .uplo = 'l', .n = 4, .lda = 4, .work = &query, .lwork = -1 });
            try equal(@as(i32, 0), info);
            try expect(query[0] >= 11);
        }
        try equal(@as(usize, 0), spy.stats().attempts);
        query[0] = -7;
        try equal(@as(i32, -6), try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = 101, .jobz = '?', .uplo = '?', .n = 4, .lda = 1, .work = &query, .lwork = -1 }));
        try equal(@as(T, -7), query[0]);
        try equal(B.common.HookKind.lapacke, log.events[0].kind);
        try equal(B.common.Entry.work, log.events[0].entry);
        try equal(@as(i32, -2), try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = 102, .jobz = '?', .uplo = '?', .n = -1, .lda = 0 }));
        try equal(B.common.Entry.fortran_syev, log.events[1].entry);
        try equal(@as(i32, 1), log.events[1].argument);
        try equal(@as(i32, -9), try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = 102, .jobz = 'N', .uplo = 'U', .n = 4, .lda = 4, .work = &query, .lwork = 0 }));
        try expect(query[0] >= 11);
        try equal(@as(i32, 8), log.events[2].argument);
        try equal(@as(i32, 0), try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = 102, .jobz = 'N', .uplo = 'U', .n = 0, .lda = 1, .work = &query, .lwork = 1 }));
        try equal(@as(T, 1), query[0]);
        var a = [_]T{std.math.nan(T)};
        var value: [1]T = undefined;
        try equal(@as(i32, 0), try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = 102, .jobz = 'V', .uplo = 'U', .n = 1, .a = &a, .lda = 1, .w = &value, .work = &query, .lwork = 2 }));
        try expect(std.math.isNan(value[0]));
        try equal(@as(T, 1), a[0]);
        try equal(@as(T, 2), query[0]);
        try equal(@as(usize, 0), log.nan_calls);
        try equal(@as(usize, 0), spy.stats().attempts);
        try equal(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

test "syev adapter candidate queries scalar quick returns and scalar error precedence avoid unrelated reads" {
    try checkQueryAndScalar();
}

pub fn checkNanAndDiagnosticOrder() !void {
    prohibitDefault();
    defer restoreDefault();
    var spy: E.test_support.Allocations = .{ .budget = 0 };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(spy.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer finish(owner);
    var operation = try owner.begin();
    defer operation.end();
    var log: Log = .{ .owner = owner, .close_on_event = true };
    const services = log.services();
    try equal(@as(i32, -1), try A.highAdmitted(true, f64, &operation, &services, .{ .layout = 7, .jobz = 'V', .uplo = 'U', .n = 2, .lda = 2 }));
    try equal(@as(usize, 0), log.nan_calls);
    try equal(B.common.Entry.high, log.events[0].entry);
    var bad = [_]f64{ std.math.nan(f64), 1, 1, 2 };
    try equal(@as(i32, -5), try A.highAdmitted(true, f64, &operation, &services, .{ .layout = 102, .jobz = '?', .uplo = 'U', .n = 2, .a = &bad, .lda = 1 }));
    try equal(@as(usize, 1), log.count);
    try equal(@as(usize, 1), log.nan_calls);
    try equal(@as(i32, -2), try A.highAdmitted(false, f64, &operation, &services, .{ .layout = 102, .jobz = '?', .uplo = 'U', .n = 2, .lda = 2 }));
    try equal(@as(usize, 1), log.nan_calls);
    try equal(B.common.Entry.fortran_syev, log.events[1].entry);
    try equal(@as(i32, -1010), try A.highAdmitted(false, f64, &operation, &services, .{ .layout = 102, .jobz = 'N', .uplo = 'U', .n = 2, .lda = 2 }));
    try equal(B.common.Entry.high, log.events[2].entry);
    try equal(@as(i32, -1010), log.events[2].argument);
    try equal(@as(usize, 0), log.pins[2]);
    try equal(@as(i32, -1011), try A.workWithDiagnosticsAdmitted(f64, &operation, &services, .{ .layout = 101, .jobz = '?', .uplo = 'U', .n = 2, .lda = 2 }));
    try equal(B.common.Entry.work, log.events[3].entry);
    try equal(@as(usize, 0), log.pins[3]);
    try equal(@as(usize, 1), owner.snapshot().active_operations);
    try equal(@as(usize, 0), owner.snapshot().resource_pins);
    try equal(@as(usize, 0), spy.stats().accepted);
}

test "syev adapter candidate high NaN order compile scan and allocation diagnostics use original stages" {
    try checkNanAndDiagnosticOrder();
}

pub fn checkAllocationFailures() !void {
    prohibitDefault();
    defer restoreDefault();
    for ([_]c_int{ 101, 102 }) |layout| {
        const required: usize = if (layout == 101) 3 else 2;
        for (0..required + 1) |budget| {
            var spy: E.test_support.Allocations = .{ .budget = budget };
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(spy.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            defer finish(owner);
            var operation = try owner.begin();
            defer operation.end();
            var log: Log = .{ .owner = owner, .close_on_event = true };
            const services = log.services();
            var a = [_]f64{ 2, 1, 1, 2 };
            const before = a;
            var values = [_]f64{ -8, -9 };
            var work: [5]f64 = @splat(-7);
            const info = try A.workWithDiagnosticsAdmitted(f64, &operation, &services, .{ .layout = layout, .jobz = 'V', .uplo = 'U', .n = 2, .a = &a, .lda = 2, .w = &values, .work = &work, .lwork = 5 });
            if (budget < required) {
                try equal(@as(i32, if (layout == 101 and budget == 0) -1011 else -1010), info);
                try same(f64, &before, &a);
                try std.testing.expectEqualSlices(f64, &.{ -8, -9 }, &values);
                try equal(@as(usize, 1), log.count);
                try equal(@as(usize, 0), log.pins[0]);
            } else {
                try equal(@as(i32, 0), info);
                try std.testing.expectApproxEqAbs(@as(f64, 1), values[0], 1e-12);
                try std.testing.expectApproxEqAbs(@as(f64, 3), values[1], 1e-12);
                try equal(@as(usize, 0), log.count);
            }
            try equal(spy.stats().accepted, spy.stats().freed);
            try equal(@as(usize, 1), owner.snapshot().active_operations);
            try equal(@as(usize, 0), owner.snapshot().resource_pins);
        }
    }
}

test "syev adapter candidate every scratch allocation failure releases before diagnostics" {
    try checkAllocationFailures();
}

pub fn checkPositiveInfo() !void {
    prohibitDefault();
    defer restoreDefault();
    inline for (.{ f32, f64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer finish(owner);
        var operation = try owner.begin();
        defer operation.end();
        _ = owner.requestClose();
        var log: Log = .{ .owner = owner };
        const services = log.services();
        var a = [_]T{ 2, 1, 0, 1, 3, 1, 0, 1, 4 };
        var values: [3]T = @splat(-9);
        var work: [8]T = @splat(-7);
        const info = try A.workWithDiagnosticsAdmitted(T, &operation, &services, .{ .layout = layout, .jobz = 'V', .uplo = 'U', .n = 3, .a = &a, .lda = 3, .w = &values, .work = &work, .lwork = 8, .native_options = .{ .max_sweeps = 0 } });
        try equal(@as(i32, 2), info);
        try equal(@as(usize, 0), log.count);
        try equal(@as(usize, 0), owner.snapshot().resource_pins);
    };
}

test "syev adapter candidate positive INFO counts actual native tridiagonal edges" {
    try checkPositiveInfo();
}

pub fn checkRawAndClosingOwner() !void {
    prohibitDefault();
    defer restoreDefault();
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer finish(owner);
    {
        var operation = try owner.begin();
        defer operation.end();
        var log: Log = .{ .owner = owner };
        const services = log.services();
        var query = [_]f64{-7};
        const probe: Raw.Arguments(f64) = .{ .layout = 101, .jobz = 'V', .uplo = 'U', .n = 4, .a = @ptrFromInt(1), .lda = 4, .w = @ptrFromInt(1), .work = @ptrCast(&query), .lwork = -1 };
        try equal(@as(i32, 0), try Raw.workWithDiagnosticsAdmitted(f64, &operation, &services, probe));
        try expect(query[0] >= 11);
        var invalid = probe;
        invalid.layout = 102;
        invalid.jobz = '?';
        invalid.work = @ptrFromInt(1);
        try equal(@as(i32, -2), try Raw.workWithDiagnosticsAdmitted(f64, &operation, &services, invalid));
        invalid.jobz = 'V';
        try std.testing.expectError(error.MisalignedPointer, Raw.workWithDiagnosticsAdmitted(f64, &operation, &services, invalid));
    }
    C.Context.testing.configure(owner, 1, null);
    B.shared_services.testing.reset();
    defer B.shared_services.testing.reset();
    var log: Log = .{ .owner = owner, .close_on_event = true };
    const info = try B.Shared.symmetricRawManagedWithDefaults(false, f64, .work, owner, .{ .layout = 102, .jobz = '?', .uplo = 'U', .n = 4, .lda = 4 }, .{ .fortran_returning_test = .{ .userdata = &log, .function = Log.record } });
    try equal(@as(i32, -2), info);
    try equal(@as(usize, 1), log.count);
    try equal(B.common.Routine.syev, log.events[0].routine);
    try equal(@as(usize, 0), owner.snapshot().active_operations);
    try equal(@as(usize, 0), owner.snapshot().resource_pins);
    try equal(C.Phase.closing, owner.snapshot().phase);
}

test "syev adapter candidate raw query nonreads and managed closing callback preserve the one owner" {
    try checkRawAndClosingOwner();
}

pub fn checkExtremeScales() !void {
    prohibitDefault();
    defer restoreDefault();
    inline for (.{ f32, f64 }) |T| {
        const exponents = if (T == f32) [_]i32{ 0, 50, -50, -145 } else [_]i32{ 0, 500, -500, -1070 };
        const tol: f64 = if (T == f32) 3e-6 else 2e-12;
        // A=Q*diag(1,2,4,8)*Q^T, Q=the normalized 4-by-4 Hadamard matrix.
        // Every off-diagonal is nonzero; this is independent of the driver's
        // tridiagonal reduction, iterations, and eigenvector sign choices.
        const original = [_]T{ 3.75, -1.25, -2.25, 0.75, -1.25, 3.75, 0.75, -2.25, -2.25, 0.75, 3.75, -1.25, 0.75, -2.25, -1.25, 3.75 };
        for (exponents) |exponent| for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'U', 'L' }) |uplo| for ([_]bool{ false, true }) |high| {
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            defer finish(owner);
            var operation = try owner.begin();
            defer operation.end();
            _ = owner.requestClose();
            var log: Log = .{ .owner = owner };
            const services = log.services();
            const scale = std.math.ldexp(@as(T, 1), exponent);
            try expect(scale > 0 and std.math.isFinite(scale));
            var a: [24]T = @splat(std.math.nan(T));
            for (0..4) |j| for (0..4) |i| {
                if (if (uplo == 'U') i <= j else i >= j) a[idx(layout, 5, i, j)] = original[i + j * 4] * scale;
            };
            const before = a;
            var values: [4]T = undefined;
            var work: [11]T = undefined;
            const args: A.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = uplo, .n = 4, .a = &a, .lda = 5, .w = &values, .work = &work, .lwork = 11 };
            const info = if (high) try A.highAdmitted(true, T, &operation, &services, args) else try A.workWithDiagnosticsAdmitted(T, &operation, &services, args);
            try equal(@as(i32, 0), info);
            for ([_]f64{ 1, 2, 4, 8 }, 0..) |expected, j| {
                const scaled_value = @as(f64, values[j]) / @as(f64, scale);
                try std.testing.expectApproxEqAbs(expected, scaled_value, tol);
                for (0..4) |i| {
                    var av: f64 = 0;
                    for (0..4) |k| av += @as(f64, original[i + k * 4]) * @as(f64, a[idx(layout, 5, k, j)]);
                    try std.testing.expectApproxEqAbs(av, @as(f64, a[idx(layout, 5, i, j)]) * scaled_value, tol);
                }
                for (0..4) |k| {
                    var dot: f64 = 0;
                    for (0..4) |i| dot += @as(f64, a[idx(layout, 5, i, j)]) * @as(f64, a[idx(layout, 5, i, k)]);
                    try std.testing.expectApproxEqAbs(@as(f64, if (j == k) 1 else 0), dot, tol);
                }
            }
            for (0..24) |k| {
                const i = if (layout == 102) k % 5 else k / 5;
                const j = if (layout == 102) k / 5 else k % 5;
                if (i >= 4 or j >= 4) try same(T, before[k..][0..1], a[k..][0..1]);
            }
            try equal(@as(usize, 0), log.count);
            try equal(@as(usize, 1), owner.snapshot().active_operations);
            try equal(@as(usize, 0), owner.snapshot().resource_pins);
        };
    }
}

test "syev adapter candidate dense known spectrum survives huge tiny and common subnormal scales" {
    try checkExtremeScales();
}

pub fn checkDiagnosticNames() !void {
    inline for (.{ i32, i64 }) |I| {
        const D = B.Diagnostics(I);
        for ("sd") |scalar| {
            const high = try D.translate(.{ .kind = .lapacke, .entry = .high, .scalar = scalar, .argument = -1, .routine = .syev });
            try std.testing.expectEqualStrings(if (scalar == 's') "LAPACKE_ssyev" else "LAPACKE_dsyev", high.name());
            try equal(@as(I, -1), high.info);
            try equal(D.Channel.lapacke, high.channel);
            const work = try D.translate(.{ .kind = .lapacke, .entry = .work, .scalar = scalar, .argument = -1011, .routine = .syev });
            try std.testing.expectEqualStrings(if (scalar == 's') "LAPACKE_ssyev_work" else "LAPACKE_dsyev_work", work.name());
            try equal(@as(I, -1011), work.info);
            const underlying = try D.translate(.{ .kind = .fortran_returning_test, .entry = .fortran_syev, .scalar = scalar, .argument = 8, .routine = .syev });
            try std.testing.expectEqualStrings(if (scalar == 's') "SSYEV" else "DSYEV", underlying.name());
            try equal(@as(I, 8), underlying.info);
            try equal(D.Channel.fortran, underlying.channel);
            try std.testing.expectError(error.InvalidEvent, D.translate(.{ .kind = .fortran_returning_test, .entry = .fortran_heev, .scalar = scalar, .argument = 8, .routine = .syev }));
        }
    }
}

test "syev component diagnostics preserve LAPACKE and Fortran names signs and argument positions" {
    try checkDiagnosticNames();
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
