// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const Ctx = owner.Provider.Execution.context;
const Scalar = owner.Provider.Scalar;
const Math = owner.Provider.SvdLeastSquares;
fn raw(comptime T: type, values: []T) ?[*]align(1) T {
    return values.ptr;
}
fn pos(layout: c_int, i: usize, ld: usize) usize {
    return if (layout == 101) i * ld else i;
}
test "GELSD RHS scaling retains reference residual units with original solution and one operation" {
    var configurations: usize = 0;
    var failures: usize = 0;
    inline for (.{ Scalar.ComplexF32, Scalar.ComplexF64 }) |C| inline for (.{ i32, i64 }) |I| {
        const R = Scalar.Real(C);
        const W = Math.Wide(C);
        const tiny: R = if (R == f32) 1e-35 else 1e-300;
        const huge: R = if (R == f32) 1e35 else 1e300;
        const least: R = std.math.floatMin(R) * std.math.floatEps(R);
        for ([_]R{ 1, tiny, huge, std.math.floatMin(R), least, -huge }) |units| {
            for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 0, 1, 2 }) |mode| {
                var rig: h.Rig = .{};
                rig.init();
                defer rig.done() catch @panic("RHS scale lifecycle");
                const lda: usize = if (layout == 101) 3 else 4;
                const ldb: usize = if (layout == 101) 3 else 4;
                var a: [8]C = @splat(.{ .re = -71, .im = 7 });
                var b: [8]C = @splat(.{ .re = -72, .im = 8 });
                a[0] = .{ .re = 1, .im = 0 };
                a[pos(layout, 1, lda)] = .{ .re = 0, .im = 0 };
                b[0] = .{ .re = 2 * units, .im = -units };
                b[pos(layout, 1, ldb)] = .{ .re = 3 * units, .im = 4 * units };
                var singular = [_]R{ -81, -82 };
                var rank = [_]I{ -83, -84 };
                var rw = [_]R{ -85, -86 };
                var iw = [_]I{ -87, -88 };
                const plan = try owner.complex_gelsd_workspace.Planner(owner.Provider).make(C, I, 2, 1, 1);
                const minimum = try owner.complex_gelsd_policy.minimumWork(I, 2, 1, 1);
                const count: usize = @intCast(if (mode == 1) minimum else plan.work);
                const work = try std.testing.allocator.alloc(C, count + 2);
                defer std.testing.allocator.free(work);
                @memset(work, .{ .re = -89, .im = 9 });
                const result = try owner.RawComplexGelsdAdapter.callManaged(C, I, if (mode == 0) .high else .work, rig.context, &rig.services, .{
                    .layout = layout,
                    .column = .{ .m = 2, .n = 1, .nrhs = 1, .a = raw(C, &a), .lda = @intCast(lda), .b = raw(C, &b), .ldb = @intCast(ldb), .singular = raw(R, &singular), .rcond = -1, .rank = raw(I, &rank), .work = raw(C, work), .lwork = @intCast(count), .rwork = raw(R, &rw), .iwork = raw(I, &iw) },
                });
                switch (result) {
                    .lapacke_info => |info| try std.testing.expectEqual(@as(I, 0), info),
                    else => return error.TestUnexpectedResult,
                }
                try std.testing.expectEqual(@as(I, 1), rank[0]);
                try std.testing.expectEqual(@as(R, 1), singular[0]);
                const tolerance: W = 512 * @as(W, std.math.floatEps(R));
                try std.testing.expect(@abs(@as(W, b[0].re) / @as(W, units) - 2) <= tolerance);
                try std.testing.expect(@abs(@as(W, b[0].im) / @as(W, units) + 1) <= tolerance);
                const small = @as(W, std.math.floatMin(R)) / @as(W, std.math.floatEps(R));
                const large = 1 / small;
                const maximum = 5 * @abs(@as(W, units));
                const expected = if (maximum < small) small else if (maximum > large) large else maximum;
                const tail = b[pos(layout, 1, ldb)];
                const residual = std.math.hypot(@as(W, tail.re), @as(W, tail.im));
                const matches = @abs(residual / expected - 1) <= tolerance;
                if (!matches) {
                    failures += 1;
                    std.debug.print("RHS_SCALE_MISMATCH scalar={s} index={s} layout={d} mode={d} units={e} residual={e} expected={e}\n", .{ @typeName(C), @typeName(I), layout, mode, units, residual, expected });
                }
                for (0..8) |i| {
                    if (i != 0 and i != pos(layout, 1, lda)) try std.testing.expect(a[i].re == -71 and a[i].im == 7);
                    if (i != 0 and i != pos(layout, 1, ldb)) try std.testing.expect(b[i].re == -72 and b[i].im == 8);
                }
                try std.testing.expectEqual(@as(R, -82), singular[1]);
                try std.testing.expectEqual(@as(I, -84), rank[1]);
                try std.testing.expectEqual(@as(R, -86), rw[1]);
                try std.testing.expectEqual(@as(I, -88), iw[1]);
                for (work[count..]) |v| try std.testing.expect(v.re == -89 and v.im == 9);
                try std.testing.expectEqual(@as(usize, 0), rig.context.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), rig.context.snapshot().resource_pins);
                try std.testing.expectEqual(rig.x.attempts, rig.x.frees);
                try std.testing.expectEqual(@as(usize, 0), rig.x.live);
                try std.testing.expectEqual(@as(usize, 0), rig.y.attempts);
                try std.testing.expectEqual(@as(usize, 0), rig.z.attempts);
                var admitted: usize = 0;
                var finished: usize = 0;
                for (rig.events[0..rig.count]) |event| {
                    if (event.tag == .admitted) admitted += 1;
                    if (event.tag == .finishing) finished += 1;
                }
                try std.testing.expectEqual(@as(usize, 1), admitted);
                try std.testing.expectEqual(@as(usize, 1), finished);
                configurations += 1;
            };
        }
    };
    std.debug.print("RHS_SCALE configurations={d} residual_mismatches={d} one_admission=true padding_and_suffixes_preserved=true\n", .{ configurations, failures });
    try std.testing.expectEqual(@as(usize, 144), configurations);
    try std.testing.expectEqual(@as(usize, 0), failures);
}
