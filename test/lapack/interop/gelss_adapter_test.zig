// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const B = @import("numerical-owner");
const A = B.GelssAdapter;
const Raw = B.RawGelssAdapter;
const N = Core.internal.numerics;
const S = N.Scalar;
const Ctx = N.Execution.context;
const F = N.Execution.test_support;
const types = .{ f32, f64, S.ComplexF32, S.ComplexF64 };
const Z = struct { re: f128 = 0, im: f128 = 0 };
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn scale(a: Z, factor: f128) Z {
    return .{ .re = a.re * factor, .im = a.im * factor };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn value(comptime T: type, z: Z) T {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(z.re), .im = @floatCast(z.im) } else @floatCast(z.re);
}
fn read(comptime T: type, x: T) Z {
    return if (comptime S.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x };
}
fn near(comptime T: type, expected: Z, actual: T, denominator: f128) !void {
    const z = read(T, actual);
    const tolerance = 8192 * @as(f128, std.math.floatEps(S.Real(T)));
    try std.testing.expect(@abs(z.re - expected.re) / denominator <= tolerance);
    try std.testing.expect(@abs(z.im - expected.im) / denominator <= tolerance);
}
fn at(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
fn basis(comptime T: type, dimension: usize, row: usize, column: usize, right: bool) Z {
    const plane = [3][3]f128{ .{ 0.6, -0.8, 0 }, .{ 0.8, 0.6, 0 }, .{ 0, 0, 1 } };
    const amplitude: f128 = if (dimension == 3) plane[row][column] else if (row == column) 1 else 0;
    if (comptime !S.isComplex(T)) return .{ .re = amplitude };
    const phases = [3]Z{ .{ .re = 1 }, .{ .im = 1 }, .{ .re = -1 } };
    return scale(phases[(row + @intFromBool(right)) % 3], amplitude);
}
const Probe = struct {
    nan: bool = true,
    events: [8]B.common.ErrorEvent = undefined,
    count: usize = 0,
    fn enabled(raw: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(raw));
        return self.nan;
    }
    fn event(raw: *anyopaque, event_value: B.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (self.count >= self.events.len) @panic("GELSS event overflow");
        self.events[self.count] = event_value;
        self.count += 1;
    }
    const Services = struct { context: *anyopaque, nan_enabled: *const fn (*anyopaque) bool, error_event: *const fn (*anyopaque, B.common.ErrorEvent) void };
    fn services(self: *@This()) Services {
        return .{ .context = self, .nan_enabled = enabled, .error_event = event };
    }
};
fn rawArgs(comptime T: type, args: A.Arguments(T)) Raw.Arguments(T) {
    return .{ .layout = args.layout, .m = args.m, .n = args.n, .nrhs = args.nrhs, .lda = args.lda, .ldb = args.ldb, .rcond = args.rcond, .lwork = args.lwork, .a = if (args.a) |x| @ptrCast(x.ptr) else null, .b = if (args.b) |x| @ptrCast(x.ptr) else null, .singular = if (args.singular) |x| @ptrCast(x.ptr) else null, .rank = if (args.rank) |x| @ptrCast(x.ptr) else null, .work = if (args.work) |x| @ptrCast(x.ptr) else null, .rwork = if (args.rwork) |x| @ptrCast(x.ptr) else null };
}
fn call(comptime T: type, comptime raw: bool, high: bool, operation: *const Ctx.Operation, services: *const Probe.Services, args: A.Arguments(T)) !i32 {
    if (comptime raw) return if (high) Raw.highAdmitted(true, T, operation, services, rawArgs(T, args)) else Raw.workWithDiagnosticsAdmitted(T, operation, services, rawArgs(T, args));
    return if (high) A.highAdmitted(true, T, operation, services, args) else A.workWithDiagnosticsAdmitted(T, operation, services, args);
}
fn padding(comptime T: type, values: []const T, m: usize, n: usize, ld: usize, layout: c_int, marker: T) !void {
    for (values, 0..) |x, index| {
        const row = if (layout == 102) index % ld else index / ld;
        const col = if (layout == 102) index / ld else index % ld;
        if (row < m and col < n) continue;
        try std.testing.expectEqual(marker, x);
    }
}
fn numeric(comptime T: type, comptime raw: bool, high: bool, layout: c_int, m: usize, n: usize, deficient: bool, rcond: f128, ascale: f128, bscale: f128) !void {
    const marker = value(T, .{ .re = 73, .im = -19 });
    var av: [48]T = @splat(marker);
    var bv: [40]T = @splat(marker);
    var singular: [3]S.Real(T) = @splat(37);
    var rank = [1]i32{-71};
    var work: [128]T = @splat(marker);
    var rwork: [32]S.Real(T) = @splat(59);
    const rows = @max(m, n);
    const lda = (if (layout == 102) m else n) + 2;
    const ldb = (if (layout == 102) rows else @as(usize, 2)) + 2;
    const sigmas = [2]f128{ 4, if (deficient) @as(f128, 0) else 1 };
    const coordinates = [2][2]f128{ .{ 2, -1 }, .{ -1, 3 } };
    const effective = if (rcond < 0) @as(f128, std.math.floatEps(S.Real(T))) else rcond;
    var expected_rank: usize = 0;
    for (sigmas) |sigma| if (sigma > effective * 4) {
        expected_rank += 1;
    };
    for (0..n) |j| for (0..m) |i| {
        var z: Z = .{};
        for (0..2) |q| z = add(z, scale(mul(basis(T, m, i, q, false), conj(basis(T, n, j, q, true))), sigmas[q] * ascale));
        av[at(layout, i, j, lda)] = value(T, z);
    };
    for (0..2) |j| for (0..rows) |i| {
        var z: Z = .{ .re = 61 };
        if (i < m) {
            z = .{};
            for (0..2) |q| z = add(z, scale(basis(T, m, i, q, false), sigmas[q] * coordinates[q][j] * bscale));
            if (m == 3) z = add(z, scale(basis(T, 3, i, 2, false), @as(f128, @floatFromInt(j + 1)) * bscale));
        }
        bv[at(layout, i, j, ldb)] = value(T, z);
    };
    const old_a = av;
    var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
    var operation = try context.begin();
    var probe: Probe = .{};
    const services = probe.services();
    const minimum = try A.minimumWork(T, @intCast(m), @intCast(n), 2);
    const args: A.Arguments(T) = .{ .layout = layout, .m = @intCast(m), .n = @intCast(n), .nrhs = 2, .a = &av, .lda = @intCast(lda), .b = &bv, .ldb = @intCast(ldb), .singular = &singular, .rcond = @floatCast(rcond), .rank = &rank, .work = &work, .lwork = @intCast(minimum), .rwork = &rwork };
    try std.testing.expectEqual(@as(i32, 0), try call(T, raw, high, &operation, &services, args));
    try std.testing.expectEqual(@as(i32, @intCast(expected_rank)), rank[0]);
    try std.testing.expectEqual(@as(usize, 0), probe.count);
    for (0..2) |j| for (0..n) |i| {
        var expected: Z = .{};
        for (0..2) |q| if (sigmas[q] > effective * 4) {
            expected = add(expected, scale(basis(T, n, i, q, true), coordinates[q][j] * bscale / ascale));
        };
        try near(T, expected, bv[at(layout, i, j, ldb)], @abs(bscale / ascale));
    };
    for (sigmas, singular[0..2]) |wanted, observed| try std.testing.expect(@abs(@as(f128, observed) / @abs(ascale) - wanted) <= 8192 * @as(f128, std.math.floatEps(S.Real(T))));
    // A stores V^H rowwise. Check orthogonality and A_original^H*A_original*V=S^2*V.
    for (0..2) |p| for (0..2) |q| {
        var dot: Z = .{};
        for (0..n) |j| dot = add(dot, mul(read(T, av[at(layout, p, j, lda)]), conj(read(T, av[at(layout, q, j, lda)]))));
        try std.testing.expect(@abs(dot.re - @as(f128, if (p == q) 1 else 0)) <= 8192 * @as(f128, std.math.floatEps(S.Real(T))));
        try std.testing.expect(@abs(dot.im) <= 8192 * @as(f128, std.math.floatEps(S.Real(T))));
    };
    for (0..2) |q| for (0..n) |j| {
        var gram: Z = .{};
        for (0..m) |i| {
            var projection: Z = .{};
            for (0..n) |r| projection = add(projection, mul(scale(read(T, old_a[at(layout, i, r, lda)]), 1 / ascale), conj(read(T, av[at(layout, q, r, lda)]))));
            gram = add(gram, mul(conj(scale(read(T, old_a[at(layout, i, j, lda)]), 1 / ascale)), projection));
        }
        const sigma: f128 = @as(f128, singular[q]) / @abs(ascale);
        const wanted = scale(conj(read(T, av[at(layout, q, j, lda)])), sigma * sigma);
        try std.testing.expect(@abs(gram.re - wanted.re) <= 65536 * @as(f128, std.math.floatEps(S.Real(T))));
        try std.testing.expect(@abs(gram.im - wanted.im) <= 65536 * @as(f128, std.math.floatEps(S.Real(T))));
    };
    if (m > n) {
        for (n..m) |i| for (0..n) |j| try std.testing.expectEqual(old_a[at(layout, i, j, lda)], av[at(layout, i, j, lda)]);
        if (expected_rank == n) {
            const small = @as(f128, std.math.floatMin(S.Real(T))) / @as(f128, std.math.floatEps(S.Real(T)));
            var bnorm: f128 = 0;
            for (0..2) |j| for (0..m) |i| {
                var z: Z = .{};
                for (0..2) |q| z = add(z, scale(basis(T, m, i, q, false), sigmas[q] * coordinates[q][j] * bscale));
                z = add(z, scale(basis(T, 3, i, 2, false), @as(f128, @floatFromInt(j + 1)) * bscale));
                bnorm = @max(bnorm, @sqrt(z.re * z.re + z.im * z.im));
            };
            const factor = if (bnorm > 0 and bnorm < small) small / bnorm else if (bnorm > 1 / small) (1 / small) / bnorm else 1;
            for (0..2) |j| {
                const z = read(T, bv[at(layout, n, j, ldb)]);
                const norm = std.math.hypot(z.re / (bscale * factor), z.im / (bscale * factor));
                try std.testing.expect(@abs(norm - @as(f128, @floatFromInt(j + 1))) <= 32768 * @as(f128, std.math.floatEps(S.Real(T))));
            }
        }
    }
    try padding(T, &av, m, n, lda, layout, marker);
    try padding(T, &bv, rows, 2, ldb, layout, marker);
    try std.testing.expectEqual(@as(S.Real(T), 37), singular[2]);
    try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    operation.end();
    _ = context.requestClose();
    try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
}
test "GELSS boundary four scalar typed raw high work layouts ranks scales and right vector overwrite" {
    var configurations: usize = 0;
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        const tiny: f128 = if (S.Real(T) == f32) 1e-35 else 1e-300;
        const huge: f128 = if (S.Real(T) == f32) 1e35 else 1e300;
        const scales = [8][2]f128{ .{ 1, 1 }, .{ -1, 1 }, .{ tiny, 1 }, .{ huge, 1 }, .{ 1, tiny }, .{ 1, huge }, .{ tiny, tiny }, .{ huge, huge } };
        for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| for ([_][2]usize{ .{ 3, 2 }, .{ 2, 3 }, .{ 2, 2 } }) |shape| for ([_]bool{ false, true }) |deficient| for ([_]f128{ -1, 0.4, 1, 2 }) |rcond| for (scales) |pair| {
            try numeric(T, raw, high, layout, shape[0], shape[1], deficient, rcond, pair[0], pair[1]);
            configurations += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 6144), configurations);
}
test "GELSS boundary queries touch only WORK head with safe rounding and no allocation" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        var context = Ctx.Context.init(failing.allocator(), F.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        var head = [2]T{ value(T, .{ .re = 71, .im = -19 }), value(T, .{ .re = 73, .im = -23 }) };
        var rank = [1]i32{-31};
        var singular = [1]S.Real(T){37};
        var probe: Probe = .{};
        const services = probe.services();
        var args: A.Arguments(T) = .{ .layout = 102, .m = 2, .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .singular = &singular, .rcond = -1, .rank = &rank, .work = &head, .lwork = -1 };
        for ([_]c_int{ 101, 102 }) |layout| {
            args.layout = layout;
            args.lda = 2;
            args.ldb = if (layout == 101) 1 else 2;
            try std.testing.expectEqual(@as(i32, 0), try call(T, raw, false, &operation, &services, args));
            const minimum = try A.minimumWork(T, 2, 2, 1);
            try near(T, .{ .re = @floatFromInt(minimum) }, head[0], 1);
            try std.testing.expectEqual(value(T, .{ .re = 73, .im = -23 }), head[1]);
            try std.testing.expectEqual(@as(i32, -31), rank[0]);
            try std.testing.expectEqual(@as(S.Real(T), 37), singular[0]);
        }
        args.layout = 102;
        args.m = if (S.isComplex(T)) 16777213 else 16777211;
        args.n = 2;
        args.lda = args.m;
        args.ldb = args.m;
        try std.testing.expectEqual(@as(i32, 0), try call(T, raw, false, &operation, &services, args));
        const exact = try A.minimumWork(T, args.m, args.n, args.nrhs);
        try std.testing.expect(read(T, head[0]).re >= @as(f128, @floatFromInt(exact)));
        if (S.Real(T) == f32) try std.testing.expectEqual(@as(f128, 16777218), read(T, head[0]).re);
        try std.testing.expectEqual(@as(usize, 0), failing.allocated_bytes);
        try std.testing.expectEqual(@as(usize, 0), probe.count);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary scalar and row control precedence plus documented short WORK preservation" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var operation = try context.begin();
        var probe: Probe = .{};
        const services = probe.services();
        const marker = value(T, .{ .re = 71, .im = -19 });
        var work: [32]T = @splat(marker);
        var av: [4]T = @splat(S.zero(T));
        var bv: [2]T = @splat(marker);
        var singular: [2]S.Real(T) = @splat(37);
        var rank = [1]i32{-31};
        var real: [10]S.Real(T) = undefined;
        const base: A.Arguments(T) = .{ .layout = 102, .m = 2, .n = 2, .nrhs = 1, .a = &av, .lda = 2, .b = &bv, .ldb = 2, .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = -1, .rwork = &real };
        for (0..5) |which| {
            var args = base;
            const expected: i32 = switch (which) {
                0 => blk: {
                    args.m = -1;
                    break :blk -2;
                },
                1 => blk: {
                    args.n = -1;
                    break :blk -3;
                },
                2 => blk: {
                    args.nrhs = -1;
                    break :blk -4;
                },
                3 => blk: {
                    args.lda = 1;
                    break :blk -6;
                },
                else => blk: {
                    args.ldb = 1;
                    break :blk -8;
                },
            };
            probe.count = 0;
            try std.testing.expectEqual(expected, try call(T, raw, false, &operation, &services, args));
            try std.testing.expectEqual(@as(usize, 1), probe.count);
            try std.testing.expectEqual(B.common.HookKind.fortran_returning_test, probe.events[0].kind);
            try std.testing.expectEqual(B.common.Entry.fortran_gelss, probe.events[0].entry);
            try std.testing.expectEqual(-expected - 1, probe.events[0].argument);
            try std.testing.expectEqual(marker, work[0]);
        }
        var row = base;
        row.layout = 101;
        row.m = -1;
        row.lda = 1;
        row.ldb = 0;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -6), try call(T, raw, false, &operation, &services, row));
        try std.testing.expectEqual(B.common.HookKind.lapacke, probe.events[0].kind);
        row.lda = 2;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -8), try call(T, raw, false, &operation, &services, row));
        for ([_]c_int{ 101, 102 }) |layout| {
            var args = base;
            args.layout = layout;
            args.ldb = if (layout == 101) 1 else 2;
            args.lwork = @intCast((try A.minimumWork(T, 2, 2, 1)) - 1);
            probe.count = 0;
            try std.testing.expectEqual(@as(i32, -13), try call(T, raw, false, &operation, &services, args));
            try std.testing.expectEqual(marker, work[0]);
            try std.testing.expectEqual(@as(i32, -31), rank[0]);
            for (singular) |x| try std.testing.expectEqual(@as(S.Real(T), 37), x);
            try std.testing.expectEqual(@as(usize, 1), probe.count);
            try std.testing.expectEqual(@as(i32, 12), probe.events[0].argument);
        }
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        _ = context.requestClose();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary high NaN scans A then full B tail then real RCOND with compile and runtime controls" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var operation = try context.begin();
        var probe: Probe = .{};
        const services = probe.services();
        var av = [2]T{ value(T, .{ .re = std.math.nan(f128) }), S.zero(T) };
        var bv = [2]T{ value(T, .{ .re = 2 }), value(T, .{ .re = std.math.nan(f128) }) };
        var singular: [1]S.Real(T) = undefined;
        var rank = [1]i32{-31};
        var args: A.Arguments(T) = .{ .layout = layout, .m = 1, .n = 2, .nrhs = 1, .a = &av, .lda = if (layout == 101) 2 else 1, .b = &bv, .ldb = if (layout == 101) 1 else 2, .singular = &singular, .rcond = std.math.nan(S.Real(T)), .rank = &rank };
        try std.testing.expectEqual(@as(i32, -5), try call(T, raw, true, &operation, &services, args));
        av[0] = value(T, .{ .re = 1 });
        try std.testing.expectEqual(@as(i32, -7), try call(T, raw, true, &operation, &services, args));
        bv[1] = S.zero(T);
        try std.testing.expectEqual(@as(i32, -10), try call(T, raw, true, &operation, &services, args));
        try std.testing.expectEqual(@as(usize, 0), probe.count);
        try std.testing.expectEqual(@as(i32, -31), rank[0]);
        probe.nan = false;
        try std.testing.expectEqual(@as(i32, 0), try call(T, raw, true, &operation, &services, args));
        try std.testing.expectEqual(@as(i32, 0), rank[0]);
        probe.nan = true;
        args.rcond = std.math.nan(S.Real(T));
        if (comptime raw) try std.testing.expectEqual(@as(i32, 0), try Raw.highAdmitted(false, T, &operation, &services, rawArgs(T, args))) else try std.testing.expectEqual(@as(i32, 0), try A.highAdmitted(false, T, &operation, &services, args));
        try std.testing.expectEqual(@as(usize, 0), probe.count);
        operation.end();
        _ = context.requestClose();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary every allocation failure has exact memory INFO one callback and one closing operation" {
    var configurations: usize = 0;
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| for (0..9) |budget| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = budget });
        var context = Ctx.Context.init(failing.allocator(), F.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        var av = [4]T{ value(T, .{ .re = 2 }), S.zero(T), S.zero(T), value(T, .{ .re = 4 }) };
        var bv = [2]T{ value(T, .{ .re = 6 }), value(T, .{ .re = 8 }) };
        const old_a = av;
        const old_b = bv;
        var work: [32]T = undefined;
        var real: [10]S.Real(T) = undefined;
        var singular: [2]S.Real(T) = @splat(37);
        var rank = [1]i32{-31};
        var probe: Probe = .{};
        const services = probe.services();
        const minimum = try A.minimumWork(T, 2, 2, 1);
        const args: A.Arguments(T) = .{ .layout = layout, .m = 2, .n = 2, .nrhs = 1, .a = &av, .lda = 2, .b = &bv, .ldb = if (layout == 101) 1 else 2, .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = @intCast(minimum), .rwork = &real };
        const high_allocations: usize = if (!high) 0 else if (S.isComplex(T)) 2 else 1;
        const row_allocations: usize = if (layout == 101) 2 else 0;
        const total = high_allocations + row_allocations + 4;
        const expected: i32 = if (budget >= total) 0 else if (layout == 101 and budget >= high_allocations and budget < high_allocations + 2) -1011 else -1010;
        const info = try call(T, raw, high, &operation, &services, args);
        try std.testing.expectEqual(expected, info);
        try std.testing.expectEqual(@as(usize, if (info == 0) 0 else 1), probe.count);
        if (info != 0) {
            try std.testing.expectEqual(B.common.HookKind.lapacke, probe.events[0].kind);
            try std.testing.expectEqual(if (info == -1010 and high) B.common.Entry.high else B.common.Entry.work, probe.events[0].entry);
            try std.testing.expectEqual(info, probe.events[0].argument);
            try std.testing.expectEqualSlices(T, &old_a, &av);
            try std.testing.expectEqualSlices(T, &old_b, &bv);
            try std.testing.expectEqual(@as(i32, -31), rank[0]);
            for (singular) |x| try std.testing.expectEqual(@as(S.Real(T), 37), x);
        } else {
            try near(T, .{ .re = 3 }, bv[0], 1);
            try near(T, .{ .re = 2 }, bv[1], 1);
        }
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
        configurations += 1;
    };
    try std.testing.expectEqual(@as(usize, 288), configurations);
}
fn emptyCase(comptime T: type, comptime raw: bool, high: bool, layout: c_int, m: usize, n: usize, rhs: usize, all_zero: bool) !void {
    const rows = @max(m, n);
    const k = @min(m, n);
    const lda: usize = if (layout == 102) @max(1, m) else @max(1, n);
    const ldb: usize = if (layout == 102) @max(1, rows) else rhs;
    var av: [16]T = @splat(S.zero(T));
    if (!all_zero) for (0..k) |i| {
        av[at(layout, i, i, lda)] = value(T, .{ .re = if (i == 0) 4 else 1 });
    };
    var bv: [16]T = @splat(value(T, .{ .re = 19, .im = -23 }));
    const old_b = bv;
    var singular: [4]S.Real(T) = @splat(37);
    var rank = [1]i32{-31};
    var work: [32]T = undefined;
    var real: [20]S.Real(T) = undefined;
    var probe: Probe = .{};
    const services = probe.services();
    var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
    var operation = try context.begin();
    const minimum = try A.minimumWork(T, @intCast(m), @intCast(n), @intCast(rhs));
    try std.testing.expectEqual(@as(i32, 0), try call(T, raw, high, &operation, &services, .{ .layout = layout, .m = @intCast(m), .n = @intCast(n), .nrhs = @intCast(rhs), .a = &av, .lda = @intCast(lda), .b = &bv, .ldb = @intCast(ldb), .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = @intCast(minimum), .rwork = &real }));
    try std.testing.expectEqual(@as(i32, @intCast(if (all_zero) 0 else k)), rank[0]);
    if (k == 0 or rhs == 0) try std.testing.expectEqualSlices(T, &old_b, &bv);
    if (all_zero and k != 0) {
        for (0..rows) |i| for (0..rhs) |j| try std.testing.expectEqual(S.zero(T), bv[at(layout, i, j, ldb)]);
        for (singular[0..k]) |x| try std.testing.expectEqual(@as(S.Real(T), 0), x);
    } else if (k != 0) {
        try std.testing.expectEqual(@as(S.Real(T), 4), singular[0]);
        try std.testing.expectEqual(@as(S.Real(T), 1), singular[1]);
    }
    for (singular[k..]) |x| try std.testing.expectEqual(@as(S.Real(T), 37), x);
    try std.testing.expectEqual(@as(usize, 0), probe.count);
    operation.end();
    _ = context.requestClose();
    try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
}
test "GELSS boundary empty B preservation zero A clearing and zero RHS nonempty factorization" {
    var configurations: usize = 0;
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| {
        for ([_][2]usize{ .{ 0, 2 }, .{ 2, 0 }, .{ 0, 0 } }) |shape| {
            try emptyCase(T, raw, high, layout, shape[0], shape[1], 1, false);
            configurations += 1;
        }
        try emptyCase(T, raw, high, layout, 3, 2, 1, true);
        configurations += 1;
        try emptyCase(T, raw, high, layout, 3, 2, 0, false);
        configurations += 1;
        try emptyCase(T, raw, high, layout, 2, 3, 0, false);
        configurations += 1;
    };
    try std.testing.expectEqual(@as(usize, 192), configurations);
}
test "GELSS boundary typed storage and raw pointer faults remain separate from numerical INFO" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const marker = value(T, .{ .re = 71, .im = -19 });
        var av: [16]T = @splat(value(T, .{ .re = 2 }));
        var bv = [1]T{value(T, .{ .re = 6 })};
        var singular = [1]S.Real(T){37};
        var rank = [1]i32{-31};
        var work: [32]T = @splat(marker);
        var real: [5]S.Real(T) = undefined;
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var operation = try context.begin();
        const minimum = try A.minimumWork(T, 1, 1, 1);
        const base: A.Arguments(T) = .{ .layout = layout, .m = 1, .n = 1, .nrhs = 1, .a = &av, .lda = 1, .b = &bv, .ldb = 1, .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = @intCast(minimum), .rwork = &real };
        var short = base;
        short.work = work[0 .. minimum - 1];
        try std.testing.expectError(error.BufferTooSmall, A.workAdmitted(T, &operation, short));
        var alias = base;
        alias.work = &av;
        try std.testing.expectError(error.AliasingNotAllowed, A.workAdmitted(T, &operation, alias));
        const rank_alias: [*]i32 = @ptrCast(&av);
        alias = base;
        alias.rank = rank_alias[0..1];
        try std.testing.expectError(error.AliasingNotAllowed, A.workAdmitted(T, &operation, alias));
        var raw_args = rawArgs(T, base);
        raw_args.rank = null;
        try std.testing.expectError(error.MissingBuffer, Raw.workAdmitted(T, &operation, raw_args));
        var bytes: [16]u8 align(16) = undefined;
        raw_args = rawArgs(T, base);
        raw_args.rank = @ptrFromInt(@intFromPtr(&bytes) + 1);
        try std.testing.expectError(error.MisalignedPointer, Raw.workAdmitted(T, &operation, raw_args));
        if (comptime S.isComplex(T)) {
            raw_args = rawArgs(T, base);
            raw_args.rwork = null;
            try std.testing.expectError(error.MissingBuffer, Raw.workAdmitted(T, &operation, raw_args));
            var short_real = base;
            short_real.rwork = real[0..4];
            try std.testing.expectError(error.BufferTooSmall, A.workAdmitted(T, &operation, short_real));
            const real_alias: [*]S.Real(T) = @ptrCast(&av);
            var alias_real = base;
            alias_real.rwork = real_alias[0..5];
            try std.testing.expectError(error.AliasingNotAllowed, A.workAdmitted(T, &operation, alias_real));
        }
        try std.testing.expectEqual(marker, work[0]);
        try std.testing.expectEqual(@as(i32, -31), rank[0]);
        try std.testing.expectEqual(@as(S.Real(T), 37), singular[0]);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        _ = context.requestClose();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary positive INFO equals retained bidiagonal edge count and preserves uncommitted outputs" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        var av: [9]T = undefined;
        var column: [9]T = undefined;
        for (0..3) |j| for (0..3) |i| {
            const z = value(T, .{ .re = @floatFromInt(1 + (i + 2 * j) % 5), .im = @floatFromInt(i + j + 1) });
            av[at(layout, i, j, 3)] = z;
            column[i + j * 3] = z;
        };
        const old_a = av;
        var bv: [3]T = @splat(value(T, .{ .re = 1 }));
        const old_b = bv;
        var singular: [3]S.Real(T) = @splat(37);
        var rank = [1]i32{-31};
        var work: [64]T = undefined;
        var real: [15]S.Real(T) = undefined;
        const W = N.SvdLeastSquares.Wide(T);
        const CW = N.SvdLeastSquares.WorkspaceScalar(T);
        var sv: [3]W = undefined;
        var right: [9]T = undefined;
        var computational: [128]CW = undefined;
        var real_work: [128]W = undefined;
        const a_matrix: N.Dense.Matrix(T) = .{ .values = &column, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const v_matrix: N.Dense.Matrix(T) = .{ .values = &right, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const result = if (comptime S.isComplex(T)) try N.ComplexBidiagonalSvd.decomposeWithSingularType(T, W, a_matrix, &sv, v_matrix, &computational, &real_work, .{ .max_steps = 0, .numerical_policy = .ieee }) else try N.BidiagonalSvd.decomposeWithSingularType(T, W, a_matrix, &sv, v_matrix, &computational, .{ .max_steps = 0, .numerical_policy = .ieee });
        try std.testing.expectEqual(.no_convergence, result.status);
        const state = if (comptime S.isComplex(T)) try N.ComplexBidiagonalSvd.factors(T, 3, 3, &computational) else try N.BidiagonalSvd.factors(T, 3, 3, &computational);
        var edges: usize = 0;
        for (0..2) |i| {
            const z = state.b_tall.values[i + (i + 1) * 3];
            const edge = if (comptime S.isComplex(T)) z.re else z;
            if (edge != 0) edges += 1;
        }
        try std.testing.expect(edges > 0 and edges == result.unconverged);
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        const args: A.Arguments(T) = .{ .layout = layout, .m = 3, .n = 3, .nrhs = 1, .a = &av, .lda = 3, .b = &bv, .ldb = if (layout == 101) 1 else 3, .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = @intCast(try A.minimumWork(T, 3, 3, 1)), .rwork = &real };
        const info = if (comptime raw) try Raw.workWithOptionsAdmitted(T, &operation, rawArgs(T, args), .{ .max_steps = 0, .numerical_policy = .ieee }) else try A.workWithOptionsAdmitted(T, &operation, args, .{ .max_steps = 0, .numerical_policy = .ieee });
        try std.testing.expectEqual(@as(i32, @intCast(edges)), info);
        try std.testing.expectEqualSlices(T, &old_a, &av);
        try std.testing.expectEqualSlices(T, &old_b, &bv);
        try std.testing.expectEqual(@as(i32, -31), rank[0]);
        for (singular) |x| try std.testing.expectEqual(@as(S.Real(T), 37), x);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary Shared captures existing configuration callbacks and original operation owner" {
    const previous = B.shared_services.getNancheck();
    defer B.shared_services.setNancheck(previous);
    B.shared_services.setNancheck(1);
    inline for (types) |T| for ([_]bool{ false, true }) |high| {
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var probe: Probe = .{};
        const defaults: B.shared_services.Callbacks = .{ .lapacke = .{ .userdata = &probe, .function = Probe.event }, .fortran_returning_test = .{ .userdata = &probe, .function = Probe.event } };
        const args: A.Arguments(T) = .{ .layout = 102, .m = -1, .n = 0, .nrhs = 0, .lda = 1, .ldb = 1, .rcond = -1 };
        const info = try B.Shared.gelssRawManagedWithDefaults(true, T, if (high) .high else .work, &context, rawArgs(T, args), defaults);
        try std.testing.expectEqual(@as(i32, -2), info);
        try std.testing.expectEqual(@as(usize, 1), probe.count);
        try std.testing.expectEqual(B.common.Routine.gelss, probe.events[0].routine);
        try std.testing.expectEqual(B.common.Entry.fortran_gelss, probe.events[0].entry);
        try std.testing.expectEqual(@as(i32, 1), probe.events[0].argument);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        _ = context.requestClose();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
        try std.testing.expectError(error.Closing, B.Shared.gelssRawManagedWithDefaults(true, T, .high, &context, rawArgs(T, args), defaults));
    };
    try std.testing.expect(A.OwnerIdentity == Core and Raw.OwnerIdentity == Core and B.Provider.OwnerIdentity == Core);
}
test "GELSS boundary retains FFT and mixed index Sparse resources through one closing Context" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| {
        const C = if (S.Real(T) == f32) S.ComplexF32 else S.ComplexF64;
        const Descriptor = N.FFT.Persistent(N.Execution, C);
        const Control = N.SparseObject.ManagedMatrixControl(Ctx, T, i64, i32);
        const offsets = [_]i64{ 0, 1, 2 };
        const ordinals = [_]i32{ 0, 1 };
        const numbers = [_]T{ value(T, .{ .re = 2 }), value(T, .{ .re = 4 }) };
        const pattern: N.SparsePattern.Pattern(i64, i32) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &ordinals, .row_count = 2, .column_count = 2 };
        var context = Ctx.Context.init(std.testing.allocator, F.policy, .internal_closable);
        var operation = try context.begin();
        var descriptor = try Descriptor.createAdmitted(&operation, .{ .complex_1d = 2 });
        try descriptor.commitAdmitted(&operation);
        const sparse = try Control.borrow(&operation, pattern, &numbers);
        _ = context.requestClose();
        const view = try sparse.exportView(&operation, .row);
        var bv = [2]T{ view.values[0], view.values[1] };
        var av = [4]T{ value(T, .{ .re = 2 }), S.zero(T), S.zero(T), value(T, .{ .re = 4 }) };
        var singular: [2]S.Real(T) = undefined;
        var rank = [1]i32{-31};
        var work: [32]T = undefined;
        var real: [10]S.Real(T) = undefined;
        var probe: Probe = .{};
        const services = probe.services();
        try std.testing.expectEqual(@as(i32, 0), try call(T, raw, high, &operation, &services, .{ .layout = layout, .m = 2, .n = 2, .nrhs = 1, .a = &av, .lda = 2, .b = &bv, .ldb = if (layout == 101) 1 else 2, .singular = &singular, .rcond = -1, .rank = &rank, .work = &work, .lwork = @intCast(try A.minimumWork(T, 2, 2, 1)), .rwork = &real }));
        for (bv) |x| try near(T, .{ .re = 1 }, x, 1);
        try std.testing.expectEqual(@as(usize, 2), context.snapshot().resource_pins);
        const input = [2]C{ .{ .re = 1, .im = 0 }, .{ .re = 1, .im = 0 } };
        var output: [2]C = undefined;
        try descriptor.runScratchAdmitted(.complex_forward, &operation, &input, &output);
        try near(C, .{ .re = 2 }, output[0], 1);
        try near(C, .{}, output[1], 1);
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 2), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.waiting_resources, context.tryProgressClose());
        sparse.release();
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().resource_pins);
        descriptor.deinit();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
test "GELSS boundary auxiliary and row allocations retain parameter and memory error precedence" {
    inline for (types) |T| inline for (.{ false, true }) |raw| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        var context = Ctx.Context.init(failing.allocator(), F.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        var probe: Probe = .{};
        const services = probe.services();
        var work = [1]T{value(T, .{ .re = 71, .im = -19 })};
        var args: A.Arguments(T) = .{ .layout = 102, .m = -1, .n = 2, .nrhs = 1, .lda = 1, .ldb = 2, .rcond = -1, .work = &work, .lwork = 0 };
        const expected_high: i32 = if (S.isComplex(T)) -1010 else -2;
        try std.testing.expectEqual(expected_high, try call(T, raw, true, &operation, &services, args));
        try std.testing.expectEqual(@as(usize, 1), probe.count);
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -2), try call(T, raw, false, &operation, &services, args));
        args.layout = 101;
        args.lda = 2;
        args.ldb = 1;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -1011), try call(T, raw, false, &operation, &services, args));
        args.lda = 1;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -6), try call(T, raw, false, &operation, &services, args));
        args.m = 2;
        args.lda = 2;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -1011), try call(T, raw, false, &operation, &services, args));
        args.layout = 102;
        args.ldb = 2;
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -13), try call(T, raw, false, &operation, &services, args));
        args.rcond = std.math.nan(S.Real(T));
        probe.count = 0;
        try std.testing.expectEqual(@as(i32, -10), try call(T, raw, true, &operation, &services, args));
        try std.testing.expectEqual(@as(usize, 0), probe.count);
        try std.testing.expectEqual(value(T, .{ .re = 71, .im = -19 }), work[0]);
        try std.testing.expectEqual(@as(usize, 0), failing.allocated_bytes);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(Ctx.CloseResult.closed, context.tryProgressClose());
    };
}
