// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const core = @import("zynum-core");
const execution = @import("zynum-execution");
const wire = @import("abi_types.zig");
const api = @import("cholesky_exports.zig");
const C = execution.context;

fn factor(comptime T: type, a: wire.dense, uplo: i32) wire.result {
    return if (T == core.lapack.ComplexF32) api.zynum_checked_cholesky_factor_complex_f32(a, uplo) else api.zynum_checked_cholesky_factor_complex_f64(a, uplo);
}
fn solve(comptime T: type, a: wire.const_dense, b: wire.dense, uplo: i32) wire.result {
    return if (T == core.lapack.ComplexF32) api.zynum_checked_cholesky_solve_complex_f32(a, b, uplo) else api.zynum_checked_cholesky_solve_complex_f64(a, b, uplo);
}

test "generated checked Cholesky enters one default admission and ends after writes on success and error" {
    const Probe = struct {
        var admitted: usize = 0;
        var finished: usize = 0;
        var actual: []const u8 = &.{};
        var expected: []const u8 = &.{};
        fn observe(ctx: *C.Context, event: C.TestEvent) void {
            if (ctx != execution.getContext() or ctx.snapshot().active_operations != 1 or ctx.snapshot().resource_pins != 0)
                @panic("checked Cholesky split admission or acquired unexpected resources");
            switch (event) {
                .admitted => admitted += 1,
                .finishing => {
                    if (!std.mem.eql(u8, actual, expected)) @panic("checked Cholesky ended before output/error contract");
                    finished += 1;
                },
            }
        }
    };
    C.Context.testing.configure(execution.getContext(), 1, Probe.observe);
    defer C.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);
    inline for (.{ core.lapack.ComplexF32, core.lapack.ComplexF64 }) |T| {
        var a = [_]T{.{ .re = 4, .im = 0 }};
        const fa = [_]T{.{ .re = 2, .im = 0 }};
        Probe.actual = std.mem.asBytes(&a);
        Probe.expected = std.mem.asBytes(&fa);
        try std.testing.expectEqual(wire.OK, factor(T, .{ .storage = .{ .data = &a, .length = 1 }, .rows = 1, .cols = 1, .ld = 1 }, wire.LOWER).status);
        var b = [_]T{.{ .re = 12, .im = 4 }};
        const xb = [_]T{.{ .re = 3, .im = 1 }};
        Probe.actual = std.mem.asBytes(&b);
        Probe.expected = std.mem.asBytes(&xb);
        try std.testing.expectEqual(wire.OK, solve(T, .{ .storage = .{ .data = &a, .length = 1 }, .rows = 1, .cols = 1, .ld = 1 }, .{ .storage = .{ .data = &b, .length = 1 }, .rows = 1, .cols = 1, .ld = 1 }, wire.LOWER).status);
        a[0].im = 1;
        const saved = a;
        Probe.actual = std.mem.asBytes(&a);
        Probe.expected = std.mem.asBytes(&saved);
        const rejected = factor(T, .{ .storage = .{ .data = &a, .length = 1 }, .rows = 1, .cols = 1, .ld = 1 }, wire.UPPER);
        try std.testing.expectEqual(@as(i32, 15), rejected.status);
        try std.testing.expectEqual(@as(i64, 0), rejected.info);
        try std.testing.expectEqual(@as(usize, 0), execution.getContext().snapshot().active_operations);
    }
    try std.testing.expectEqual(@as(usize, 6), Probe.admitted);
    try std.testing.expectEqual(Probe.admitted, Probe.finished);
}

test "checked factor storage survives its admitted Context close and is reused by another Context without allocations" {
    inline for (.{ core.lapack.ComplexF32, core.lapack.ComplexF64 }) |T| {
        var allocations: execution.test_support.Allocations = .{ .budget = 0 };
        var reference_first: FixtureReferenceContext = undefined;
        reference_first.initInPlace(allocations.allocator(), execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const first = reference_first.borrowedContext();
        var a = [_]T{.{ .re = 4, .im = 0 }};
        var admission = try first.begin();
        try std.testing.expectEqual(C.RequestResult.pending, first.requestClose());
        try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(first));
        try std.testing.expectEqual(@as(usize, 0), try core.internal.lapack_checked.choleskyFactorizeAdmitted(&admission, T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .lower));
        admission.end();
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(first));
        const saved = a;
        var reference_second: FixtureReferenceContext = undefined;
        reference_second.initInPlace(allocations.allocator(), execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const second = reference_second.borrowedContext();
        for (0..3) |_| {
            var b = [_]T{.{ .re = 12, .im = 4 }};
            var op = try second.begin();
            try std.testing.expectEqual(@as(usize, 0), try core.internal.lapack_checked.choleskySolveAdmitted(&op, T, .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .upper));
            op.end();
            try std.testing.expectEqual(T{ .re = 3, .im = 1 }, b[0]);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&a));
        }
        try std.testing.expectEqual(C.RequestResult.pending, second.requestClose());
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(second));
        try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
