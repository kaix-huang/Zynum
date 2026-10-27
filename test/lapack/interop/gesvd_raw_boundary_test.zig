// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pointer/capacity boundaries, independent of the analytic SVD test matrix.
const std = @import("std");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const Core = owner.Provider;
const C32 = Core.Scalar.ComplexF32;
const C64 = Core.Scalar.ComplexF64;
const Field = enum { a, singular, u, vt, work, rwork, superb };
const BadAddress = enum { alignment, end };
fn adapter(comptime raw: bool) type {
    return if (raw) owner.RawGesvdAdapter else owner.GesvdAdapter;
}
fn z(comptime T: type, re: f64, im: f64) T {
    return if (comptime Core.Scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn operand(comptime raw: bool, x: anytype) if (raw) ?[*]align(1) @TypeOf(x[0]) else ?[]@TypeOf(x[0]) {
    const slice: []@TypeOf(x[0]) = x;
    return if (raw) @as([*]align(1) @TypeOf(x[0]), slice.ptr) else slice;
}
fn bad(comptime T: type, which: BadAddress) [*]align(1) T {
    const address: usize = if (which == .alignment) 1 else std.math.maxInt(usize) - (@alignOf(T) - 1);
    return @ptrFromInt(address);
}
fn fault(r: anytype, expected: anyerror) !void {
    try std.testing.expect(r == .admission_fault);
    try std.testing.expectEqual(expected, r.admission_fault);
}
fn success(r: anytype) !void {
    try std.testing.expect(r == .lapacke_info);
    try std.testing.expectEqual(@as(i32, 0), r.lapacke_info);
}
fn bytes(a: anytype, b: anytype) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn idle(r: *h.Rig, no_allocations: bool) !void {
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
    var admissions: usize = 0;
    for (r.events[0..r.count]) |event| if (event.tag == .admitted) {
        admissions += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), admissions);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), r.x.live);
    try std.testing.expectEqual(r.x.attempts, r.x.frees);
    if (no_allocations) try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
    try r.done();
}
fn Fixture(comptime T: type) type {
    const R = Core.Scalar.Real(T);
    return struct {
        a: [7]T = @splat(z(T, 71, -17)),
        u: [7]T = @splat(z(T, 72, -18)),
        vt: [7]T = @splat(z(T, 73, -19)),
        s: [3]R = @splat(74),
        superb: [2]R = @splat(75),
        work: [13]T = @splat(z(T, 76, -21)),
        rwork: [11]R = @splat(77),
        fn init() @This() {
            var f: @This() = .{};
            f.a[0] = z(T, 3, 0);
            f.a[1] = z(T, 0, 0);
            f.a[3] = z(T, 0, 0);
            f.a[4] = z(T, 1, 0);
            return f;
        }
        fn args(f: *@This(), comptime raw: bool, layout: c_int) adapter(raw).Arguments(T) {
            return .{ .layout = layout, .jobu = 'S', .jobvt = 'S', .m = 2, .n = 2, .lda = 3, .ldu = 3, .ldvt = 3, .a = operand(raw, f.a[0..]), .singular = operand(raw, f.s[0..]), .u = operand(raw, f.u[0..]), .vt = operand(raw, f.vt[0..]), .work = operand(raw, f.work[0..]), .lwork = 12, .rwork = operand(raw, f.rwork[0..]), .superb = operand(raw, f.superb[0..]) };
        }
        fn unchanged(f: *const @This(), before: *const @This()) !void {
            try bytes(&before.a, &f.a);
            try bytes(&before.u, &f.u);
            try bytes(&before.vt, &f.vt);
            try bytes(&before.s, &f.s);
            try bytes(&before.superb, &f.superb);
            try bytes(&before.work, &f.work);
            try bytes(&before.rwork, &f.rwork);
        }
    };
}
fn badField(comptime T: type, a: *owner.RawGesvdAdapter.Arguments(T), comptime field: Field, address: BadAddress) void {
    const R = Core.Scalar.Real(T);
    switch (field) {
        .a => a.a = bad(T, address),
        .u => a.u = bad(T, address),
        .vt => a.vt = bad(T, address),
        .work => a.work = bad(T, address),
        .singular => a.singular = bad(R, address),
        .rwork => a.rwork = bad(R, address),
        .superb => a.superb = bad(R, address),
    }
}
test "GESVD raw active pointer alignment and checked endpoints fail before reads allocations or public mutation" {
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for ([_]Field{ .a, .singular, .u, .vt, .work, .rwork, .superb }) |field| {
        if (comptime field == .rwork and !Core.Scalar.isComplex(T)) continue;
        for ([_]c_int{ 101, 102 }) |layout| for ([_]BadAddress{ .alignment, .end }) |address| {
            const entries: usize = if (field == .a or field == .singular or field == .u or field == .vt) 2 else 1;
            for (0..entries) |entry| {
                const level: owner.common.Level = if (field == .superb or entry == 1) .high else .work;
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init();
                const before = f;
                var a = f.args(true, layout);
                badField(T, &a, field, address);
                try fault(try owner.RawGesvdAdapter.callManaged(true, T, level, r.context, &r.services, a, .{}), if (address == .alignment) error.MisalignedPointer else error.SizeOverflow);
                try f.unchanged(&before);
                try idle(&r, true);
                cases += 1;
            }
        };
    };
    try std.testing.expectEqual(@as(usize, 168), cases);
    std.debug.print("GESVD raw pointer reject controls={d}\n", .{cases});
}
test "GESVD query ignores all nonwork addresses checks WORK and unused N outputs and real RWORK stay nonread" {
    var queries: usize = 0;
    var ignored: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| {
        const R = Core.Scalar.Real(T);
        for ([_]c_int{ 101, 102 }) |layout| for ([_]BadAddress{ .alignment, .end }) |address| {
            var r: h.Rig = .{};
            r.init();
            var f = Fixture(T).init();
            const before = f;
            var a = f.args(true, layout);
            a.lwork = -1;
            inline for ([_]Field{ .a, .singular, .u, .vt, .rwork, .superb }) |field| badField(T, &a, field, address);
            try success(try owner.RawGesvdAdapter.callManaged(true, T, .work, r.context, &r.services, a, .{}));
            try bytes(&before.a, &f.a);
            try bytes(&before.u, &f.u);
            try bytes(&before.vt, &f.vt);
            try bytes(&before.s, &f.s);
            try bytes(&before.superb, &f.superb);
            try bytes(&before.rwork, &f.rwork);
            try bytes(before.work[1..], f.work[1..]);
            const got: R = if (comptime Core.Scalar.isComplex(T)) f.work[0].re else f.work[0];
            try std.testing.expect(got >= 1 and std.math.isFinite(got));
            if (comptime Core.Scalar.isComplex(T)) try std.testing.expectEqual(@as(R, 0), f.work[0].im);
            try std.testing.expectEqual(@as(usize, 0), r.reads);
            try idle(&r, true);
            queries += 1;
            var bad_r: h.Rig = .{};
            bad_r.init();
            var bad_f = Fixture(T).init();
            const saved = bad_f;
            var qa = bad_f.args(true, layout);
            qa.lwork = -1;
            badField(T, &qa, .work, address);
            try fault(try owner.RawGesvdAdapter.callManaged(true, T, .work, bad_r.context, &bad_r.services, qa, .{}), if (address == .alignment) error.MisalignedPointer else error.SizeOverflow);
            try bad_f.unchanged(&saved);
            try idle(&bad_r, true);
            queries += 1;
            for ([_]owner.common.Level{ .high, .work }) |level| {
                var unused_r: h.Rig = .{};
                unused_r.init();
                var unused = Fixture(T).init();
                const saved_unused = unused;
                var na = unused.args(true, layout);
                na.jobu = 'N';
                na.jobvt = 'N';
                na.ldu = 1;
                na.ldvt = 1;
                na.u = bad(T, address);
                na.vt = bad(T, address);
                if (comptime !Core.Scalar.isComplex(T)) na.rwork = bad(R, address);
                try success(try owner.RawGesvdAdapter.callManaged(true, T, level, unused_r.context, &unused_r.services, na, .{}));
                try std.testing.expectApproxEqAbs(@as(R, 3), unused.s[0], 32 * std.math.floatEps(R));
                try std.testing.expectApproxEqAbs(@as(R, 1), unused.s[1], 32 * std.math.floatEps(R));
                try bytes(&saved_unused.a, &unused.a);
                try bytes(&saved_unused.u, &unused.u);
                try bytes(&saved_unused.vt, &unused.vt);
                try bytes(&saved_unused.rwork, &unused.rwork);
                try bytes(saved_unused.s[2..], unused.s[2..]);
                try bytes(saved_unused.work[1..], unused.work[1..]);
                try bytes(saved_unused.superb[1..], unused.superb[1..]);
                try idle(&unused_r, false);
                ignored += 1;
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 32), queries);
    try std.testing.expectEqual(@as(usize, 32), ignored);
    std.debug.print("GESVD query controls={d}, unused nonreads={d}\n", .{ queries, ignored });
}
test "GESVD typed active capacities and high real SUPERB mixed aliases reject transactionally before allocation" {
    var capacities: usize = 0;
    var aliases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| {
        const R = Core.Scalar.Real(T);
        inline for ([_]Field{ .a, .singular, .u, .vt, .work, .rwork, .superb }) |field| {
            if (comptime field == .rwork and !Core.Scalar.isComplex(T)) continue;
            for ([_]c_int{ 101, 102 }) |layout| {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init();
                const before = f;
                var a = f.args(false, layout);
                switch (field) {
                    .a => a.a = f.a[0..4],
                    .singular => a.singular = f.s[0..1],
                    .u => a.u = f.u[0..4],
                    .vt => a.vt = f.vt[0..4],
                    .work => a.work = f.work[0..11],
                    .rwork => a.rwork = f.rwork[0..9],
                    .superb => a.superb = f.superb[0..0],
                }
                try fault(try owner.GesvdAdapter.callManaged(true, T, if (field == .superb) .high else .work, r.context, &r.services, a, .{}), error.BufferTooSmall);
                try f.unchanged(&before);
                try idle(&r, true);
                capacities += 1;
            }
        }
        inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| for (0..6) |target| {
            var r: h.Rig = .{};
            r.init();
            var f = Fixture(T).init();
            const before = f;
            var a = f.args(raw, layout);
            const complex_offset: usize = if (Core.Scalar.isComplex(T)) @sizeOf(R) else 0;
            const pointer: [*]R = @ptrFromInt(switch (target) {
                0 => @intFromPtr(&f.a[0]) + complex_offset,
                1 => @intFromPtr(&f.s[1]),
                2 => @intFromPtr(&f.u[0]) + complex_offset,
                3 => @intFromPtr(&f.vt[0]) + complex_offset,
                4 => @intFromPtr(&f.u[2]) + complex_offset,
                else => @intFromPtr(&f.vt[2]) + complex_offset,
            });
            a.superb = operand(raw, pointer[0..1]);
            try fault(try adapter(raw).callManaged(true, T, .high, r.context, &r.services, a, .{}), error.AliasingNotAllowed);
            try f.unchanged(&before);
            try idle(&r, true);
            aliases += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 52), capacities);
    try std.testing.expectEqual(@as(usize, 96), aliases);
    std.debug.print("GESVD typed capacity rejects={d}, high mixed SUPERB aliases={d}\n", .{ capacities, aliases });
}
