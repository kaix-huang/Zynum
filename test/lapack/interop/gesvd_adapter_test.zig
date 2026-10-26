// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const Core = owner.Provider;
const C32 = Core.Scalar.ComplexF32;
const C64 = Core.Scalar.ComplexF64;
fn adapter(comptime raw: bool) type {
    return if (raw) owner.RawGesvdAdapter else owner.GesvdAdapter;
}

test "GESVD shared raw high work query captures one original owner and rejects closed before operands" {
    const cfg = owner.shared_services;
    comptime {
        const actual = @import("zynum-core");
        if (Core.OwnerIdentity != actual or Core.Execution != @import("zynum-execution") or Core.Resources != actual.internal.numerics.Resources) @compileError("GESVD must borrow the configured owner");
    }
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |tall| for (0..3) |entry| {
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        var f = Fixture(T).init(layout, tall, 'A', 'A', 0);
        const before = f;
        var args = f.args(true);
        const level: owner.common.Level = if (entry == 0) .high else .work;
        if (entry == 2) {
            args.lwork = -1;
            args.a = null;
            args.singular = null;
            args.u = null;
            args.vt = null;
            args.rwork = null;
        }
        try info(try owner.Shared.gesvdRawManaged(true, T, level, &r.context, args), 0);
        if (entry != 2) try f.verify(&before, level) else {
            try bytes(&before.a, &f.a);
            try bytes(&before.s, &f.s);
            try bytes(&before.u, &f.u);
            try bytes(&before.vt, &f.vt);
            try bytes(&before.rwork, &f.rwork);
        }
        const state = cfg.testing.snapshot();
        try std.testing.expectEqual(@as(usize, 1), state.captures);
        try std.testing.expectEqual(@as(usize, if (entry == 0) 1 else 0), state.getter_calls);
        try std.testing.expectEqual(@as(usize, 0), state.environment_reads);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        try std.testing.expectEqual(r.x.attempts, r.x.frees);
        try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
        try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
        try r.done();
        try std.testing.expectError(error.Closing, owner.Shared.gesvdRawManaged(true, T, .high, &r.context, undefined));
        try std.testing.expectEqual(state.captures, cfg.testing.snapshot().captures);
        cases += 1;
    };
    try std.testing.expectEqual(@as(usize, 48), cases);
    std.debug.print("GESVD shared original-owner high/work/query configurations={d}\n", .{cases});
}

test "GESVD shared profile defaults preserve captured diagnostic channel and configured override during close" {
    const cfg = owner.shared_services;
    const Recorder = struct {
        context: *Core.Execution.context.Context,
        calls: usize = 0,
        event: ?owner.common.ErrorEvent = null,
        close: bool = false,
        fn callback(p: *anyopaque, e: owner.common.ErrorEvent) void {
            const self: *@This() = @ptrCast(@alignCast(p));
            self.calls += 1;
            self.event = e;
            if (self.close) {
                _ = cfg.setCallbacks(.{});
                std.debug.assert(self.context.requestClose() == .pending);
                std.debug.assert(self.context.tryProgressClose() == .busy);
            }
        }
    };
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| for ([_]bool{ false, true }) |configured| for (0..3) |which| {
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        var fallback: Recorder = .{ .context = &r.context, .close = true };
        var replacement: Recorder = .{ .context = &r.context, .close = true };
        const defaults: cfg.Callbacks = .{ .lapacke = .{ .userdata = &fallback, .function = Recorder.callback }, .fortran_returning_test = .{ .userdata = &fallback, .function = Recorder.callback } };
        if (configured) _ = cfg.setCallbacks(.{ .lapacke = .{ .userdata = &replacement, .function = Recorder.callback }, .fortran_returning_test = .{ .userdata = &replacement, .function = Recorder.callback } });
        var f = Fixture(T).init(if (which == 1) 101 else 102, true, 'S', 'S', 0);
        const before = f;
        var args = f.args(true);
        if (which == 0) args.layout = 0 else if (which == 1) args.ldu = 1 else args.jobu = 'X';
        const level: owner.common.Level = if (which == 0) .high else .work;
        try info(try owner.Shared.gesvdRawManagedWithDefaults(true, T, level, &r.context, args, defaults), if (which == 0) -1 else if (which == 1) -10 else -2);
        try std.testing.expectEqual(@as(usize, if (configured) 0 else 1), fallback.calls);
        try std.testing.expectEqual(@as(usize, if (configured) 1 else 0), replacement.calls);
        const selected = if (configured) &replacement else &fallback;
        const message = try owner.Diagnostics(i32).translate(selected.event.?);
        const scalar: u8 = if (T == f32) 's' else if (T == f64) 'd' else if (T == C32) 'c' else 'z';
        var name: [24]u8 = undefined;
        const expected = if (which == 2) try std.fmt.bufPrint(&name, "{c}GESVD", .{std.ascii.toUpper(scalar)}) else try std.fmt.bufPrint(&name, "LAPACKE_{c}gesvd{s}", .{ scalar, if (which == 1) "_work" else "" });
        try std.testing.expectEqualStrings(expected, message.name());
        try std.testing.expectEqual(@as(i32, if (which == 0) -1 else if (which == 1) -10 else 1), message.info);
        try bytes(&before.a, &f.a);
        try bytes(&before.s, &f.s);
        try bytes(&before.u, &f.u);
        try bytes(&before.vt, &f.vt);
        try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
        try r.done();
        cases += 1;
    };
    try std.testing.expectEqual(@as(usize, 24), cases);
    std.debug.print("GESVD captured default/configured diagnostics and close controls={d}\n", .{cases});
}
fn operand(comptime T: type, comptime raw: bool, x: []T) if (raw) ?[*]align(1) T else ?[]T {
    return if (raw) @as([*]align(1) T, x.ptr) else x;
}

test "GESVD allocation replacement and close still dispatch captured defaults after OOM" {
    const cfg = owner.shared_services;
    const Counter = struct {
        calls: usize = 0,
        info: i32 = 0,
        fn callback(p: *anyopaque, e: owner.common.ErrorEvent) void {
            const self: *@This() = @ptrCast(@alignCast(p));
            self.calls += 1;
            self.info = e.argument;
        }
    };
    const ReplacingAllocator = struct {
        rig: *h.Rig,
        next: *Counter,
        fn allocator(self: *@This()) std.mem.Allocator {
            return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
        }
        fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
            const self: *@This() = @ptrCast(@alignCast(p));
            _ = cfg.setCallbacks(.{ .lapacke = .{ .userdata = self.next, .function = Counter.callback } });
            std.debug.assert(self.rig.context.requestClose() == .pending);
            std.debug.assert(self.rig.context.tryProgressClose() == .busy);
            return self.rig.x.allocator().rawAlloc(n, a, ret);
        }
        fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
            return false;
        }
        fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
            return null;
        }
        fn free(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {
            @panic("OOM allocator must never return storage");
        }
        fn observe(ctx: *Core.Execution.context.Context, e: Core.Execution.context.TestEvent) void {
            const r: *h.Rig = @fieldParentPtr("context", ctx);
            r.record(if (e == .admitted) .admitted else .finishing, 0);
        }
    };
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]owner.common.Level{ .high, .work }) |level| {
        cfg.testing.reset();
        defer cfg.testing.reset();
        cfg.setNancheck(1);
        var r: h.Rig = .{};
        r.init();
        r.x.fail_at = 0;
        var original: Counter = .{};
        var next: Counter = .{};
        var wrapped: ReplacingAllocator = .{ .rig = &r, .next = &next };
        r.context = Core.Execution.context.Context.init(wrapped.allocator(), Core.Execution.test_support.policy, .internal_closable);
        Core.Execution.context.Context.testing.configure(&r.context, std.math.maxInt(usize), ReplacingAllocator.observe);
        var f = Fixture(T).init(layout, true, 'A', 'A', 0);
        const before = f;
        const expected: i32 = if (level == .work and layout == 101) -1011 else -1010;
        try info(try owner.Shared.gesvdRawManagedWithDefaults(true, T, level, &r.context, f.args(true), .{ .lapacke = .{ .userdata = &original, .function = Counter.callback } }), expected);
        try std.testing.expectEqual(@as(usize, 1), original.calls);
        try std.testing.expectEqual(expected, original.info);
        try std.testing.expectEqual(@as(usize, 0), next.calls);
        try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 0), r.x.frees);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        try bytes(&before.a, &f.a);
        try bytes(&before.s, &f.s);
        try bytes(&before.u, &f.u);
        try bytes(&before.vt, &f.vt);
        try bytes(&before.superb, &f.superb);
        try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
        try r.done();
        try std.testing.expectError(error.Closing, owner.Shared.gesvdRawManaged(true, T, level, &r.context, undefined));
        try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
        cases += 1;
    };
    try std.testing.expectEqual(@as(usize, 16), cases);
    std.debug.print("GESVD captured callback survives allocation replacement close and OOM={d}\n", .{cases});
}
fn z(comptime T: type, re: f64, im: f64) T {
    return if (comptime Core.Scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
const Wide = struct { re: f64, im: f64 };
fn wide(comptime T: type, x: T) Wide {
    return if (comptime Core.Scalar.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x, .im = 0 };
}
fn mul(a: Wide, b: Wide) Wide {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Wide) Wide {
    return .{ .re = a.re, .im = -a.im };
}
fn add(a: Wide, b: Wide) Wide {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn pos(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 101) i * ld + j else i + j * ld;
}
fn bytes(a: anytype, b: anytype) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn info(r: anytype, expected: i32) !void {
    switch (r) {
        .lapacke_info => |got| try std.testing.expectEqual(expected, got),
        .admission_fault, .numerical_fault => |err| {
            std.debug.print("GESVD unexpected {s}\n", .{@errorName(err)});
            return error.TestUnexpectedResult;
        },
    }
}
fn Fixture(comptime T: type) type {
    const R = Core.Scalar.Real(T);
    return struct {
        a: [40]T = @splat(z(T, 71, -17)),
        u: [40]T = @splat(z(T, 72, -18)),
        vt: [40]T = @splat(z(T, 73, -19)),
        s: [4]R = @splat(74),
        superb: [3]R = @splat(75),
        rwork: [20]R = @splat(76),
        work: [100]T = @splat(z(T, 77, -21)),
        m: usize,
        n: usize,
        lda: usize,
        ldu: usize,
        ldvt: usize,
        layout: c_int,
        jobu: u8,
        jobvt: u8,
        variant: usize,
        fn init(layout: c_int, tall: bool, jobu: u8, jobvt: u8, variant: usize) @This() {
            const m: usize = if (tall) 4 else 2;
            const n: usize = if (tall) 2 else 4;
            var f: @This() = .{ .m = m, .n = n, .layout = layout, .jobu = jobu, .jobvt = jobvt, .variant = variant, .lda = (if (layout == 101) n else m) + 2, .ldu = m + 2, .ldvt = n + 2 };
            const sig = [_]f64{ if (variant == 1) 2 else if (variant == 3) 0 else 3, if (variant == 1) 2 else if (variant >= 2) 0 else 1 };
            for (0..n) |j| for (0..m) |i| {
                var sum: Wide = .{ .re = 0, .im = 0 };
                for (0..2) |k| {
                    const uv = basis(m, i, k);
                    const vv = basis(n, j, k);
                    const term = mul(uv, conj(vv));
                    sum = add(sum, .{ .re = term.re * sig[k], .im = term.im * sig[k] });
                }
                f.a[pos(layout, i, j, f.lda)] = z(T, sum.re, sum.im);
            };
            return f;
        }
        fn basis(d: usize, i: usize, k: usize) Wide {
            const base: f64 = if (d == 4) (if (k == 1 and i % 2 == 1) -0.5 else 0.5) else if (i == k) 0.6 else if (k == 1) -0.8 else 0.8;
            return if (Core.Scalar.isComplex(T) and i % 2 == 1) .{ .re = 0, .im = base } else .{ .re = base, .im = 0 };
        }
        fn args(f: *@This(), comptime raw: bool) adapter(raw).Arguments(T) {
            return .{ .layout = f.layout, .jobu = f.jobu, .jobvt = f.jobvt, .m = @intCast(f.m), .n = @intCast(f.n), .lda = @intCast(f.lda), .ldu = @intCast(f.ldu), .ldvt = @intCast(f.ldvt), .a = operand(T, raw, &f.a), .singular = operand(R, raw, &f.s), .u = operand(T, raw, &f.u), .vt = operand(T, raw, &f.vt), .work = operand(T, raw, &f.work), .lwork = 100, .rwork = operand(R, raw, &f.rwork), .superb = operand(R, raw, &f.superb) };
        }
        fn U(f: *const @This(), i: usize, j: usize) Wide {
            return wide(T, if (f.jobu == 'O') f.a[pos(f.layout, i, j, f.lda)] else f.u[pos(f.layout, i, j, f.ldu)]);
        }
        fn Vt(f: *const @This(), i: usize, j: usize) Wide {
            return wide(T, if (f.jobvt == 'O') f.a[pos(f.layout, i, j, f.lda)] else f.vt[pos(f.layout, i, j, f.ldvt)]);
        }
        fn verifyPartial(f: *const @This(), original: *const @This(), level: owner.common.Level) !void {
            const tolerance: f64 = if (T == f32 or T == C32) 4e-5 else 4e-13;
            const edge: f64 = if (level == .high) f.superb[0] else if (comptime Core.Scalar.isComplex(T)) f.rwork[0] else f.work[1];
            try std.testing.expect(std.math.isFinite(edge) and edge != 0);
            var energy: f64 = edge * edge;
            for (f.s[0..2]) |s| energy += @as(f64, s) * s;
            try std.testing.expectApproxEqAbs(@as(f64, 10), energy, tolerance * 10);
            for (0..f.n) |j| for (0..f.m) |i| {
                var reconstructed: Wide = .{ .re = 0, .im = 0 };
                for (0..2) |p| {
                    const term = mul(f.U(i, p), f.Vt(p, j));
                    reconstructed = add(reconstructed, .{ .re = term.re * f.s[p], .im = term.im * f.s[p] });
                }
                const term = mul(f.U(i, 0), f.Vt(1, j));
                reconstructed = add(reconstructed, .{ .re = term.re * edge, .im = term.im * edge });
                const expected = wide(T, original.a[pos(f.layout, i, j, f.lda)]);
                try std.testing.expectApproxEqAbs(expected.re, reconstructed.re, tolerance);
                try std.testing.expectApproxEqAbs(expected.im, reconstructed.im, tolerance);
            };
            inline for (.{ true, false }) |left| {
                const dimension = if (left) f.m else f.n;
                for (0..dimension) |p| for (0..dimension) |q| {
                    var dot: Wide = .{ .re = 0, .im = 0 };
                    for (0..dimension) |i| dot = add(dot, if (left) mul(conj(f.U(i, p)), f.U(i, q)) else mul(f.Vt(p, i), conj(f.Vt(q, i))));
                    try std.testing.expectApproxEqAbs(@as(f64, if (p == q) 1 else 0), dot.re, tolerance);
                    try std.testing.expectApproxEqAbs(@as(f64, 0), dot.im, tolerance);
                };
            }
            try bytes(&original.a, &f.a);
            try bytes(original.s[2..], f.s[2..]);
            for (0..f.u.len) |i| if (i / f.ldu >= f.m or i % f.ldu >= f.m) try bytes(original.u[i..][0..1], f.u[i..][0..1]);
            for (0..f.vt.len) |i| if (i / f.ldvt >= f.n or i % f.ldvt >= f.n) try bytes(original.vt[i..][0..1], f.vt[i..][0..1]);
            const work_prefix: usize = if (level == .high) 0 else if (comptime Core.Scalar.isComplex(T)) 1 else 2;
            const real_prefix: usize = if (level == .work and Core.Scalar.isComplex(T)) 1 else 0;
            const superb_prefix: usize = if (level == .high) 1 else 0;
            try bytes(original.work[work_prefix..], f.work[work_prefix..]);
            try bytes(original.rwork[real_prefix..], f.rwork[real_prefix..]);
            try bytes(original.superb[superb_prefix..], f.superb[superb_prefix..]);
        }
        fn verify(f: *const @This(), original: *const @This(), level: owner.common.Level) !void {
            const tol: f64 = if (T == f32 or T == C32) 4e-5 else 4e-13;
            const expect_s = [_]f64{ if (f.variant == 1) 2 else if (f.variant == 3) 0 else 3, if (f.variant == 1) 2 else if (f.variant >= 2) 0 else 1 };
            for (0..2) |i| try std.testing.expectApproxEqAbs(expect_s[i], @as(f64, f.s[i]), tol);
            try bytes(original.s[2..], f.s[2..]);
            if (level == .high) {
                try std.testing.expectEqual(@as(R, 0), f.superb[0]);
                try bytes(original.superb[1..], f.superb[1..]);
            } else try bytes(&original.superb, &f.superb);
            const ucols: usize = if (f.jobu == 'A') f.m else if (f.jobu == 'N') 0 else 2;
            const vrows: usize = if (f.jobvt == 'A') f.n else if (f.jobvt == 'N') 0 else 2;
            for (0..ucols) |p| for (0..ucols) |q| {
                var dot: Wide = .{ .re = 0, .im = 0 };
                for (0..f.m) |i| dot = add(dot, mul(conj(f.U(i, p)), f.U(i, q)));
                try std.testing.expectApproxEqAbs(@as(f64, if (p == q) 1 else 0), dot.re, tol);
                try std.testing.expectApproxEqAbs(@as(f64, 0), dot.im, tol);
            };
            for (0..vrows) |p| for (0..vrows) |q| {
                var dot: Wide = .{ .re = 0, .im = 0 };
                for (0..f.n) |j| dot = add(dot, mul(f.Vt(p, j), conj(f.Vt(q, j))));
                try std.testing.expectApproxEqAbs(@as(f64, if (p == q) 1 else 0), dot.re, tol);
                try std.testing.expectApproxEqAbs(@as(f64, 0), dot.im, tol);
            };
            if (f.jobu != 'N' and f.jobvt != 'N') for (0..f.n) |j| {
                for (0..f.m) |i| {
                    var sum: Wide = .{ .re = 0, .im = 0 };
                    for (0..2) |k| {
                        const t = mul(f.U(i, k), f.Vt(k, j));
                        sum = add(sum, .{ .re = t.re * @as(f64, f.s[k]), .im = t.im * @as(f64, f.s[k]) });
                    }
                    const a0 = wide(T, original.a[pos(f.layout, i, j, f.lda)]);
                    try std.testing.expectApproxEqAbs(a0.re, sum.re, tol);
                    try std.testing.expectApproxEqAbs(a0.im, sum.im, tol);
                }
            };
            for (0..40) |p| {
                var aw = false;
                var uw = false;
                var vw = false;
                for (0..f.n) |j| for (0..f.m) |i| {
                    if ((f.jobu == 'O' and j < 2) or (f.jobvt == 'O' and i < 2)) aw = aw or p == pos(f.layout, i, j, f.lda);
                };
                if (f.jobu == 'A' or f.jobu == 'S') for (0..ucols) |j| {
                    for (0..f.m) |i| uw = uw or p == pos(f.layout, i, j, f.ldu);
                };
                if (f.jobvt == 'A' or f.jobvt == 'S') for (0..f.n) |j| {
                    for (0..vrows) |i| vw = vw or p == pos(f.layout, i, j, f.ldvt);
                };
                if (!aw) try bytes(original.a[p..][0..1], f.a[p..][0..1]);
                if (!uw) try bytes(original.u[p..][0..1], f.u[p..][0..1]);
                if (!vw) try bytes(original.vt[p..][0..1], f.vt[p..][0..1]);
            }
            try bytes(original.work[1..], f.work[1..]);
            try bytes(&original.rwork, &f.rwork);
        }
    };
}
test "GESVD four scalar full thin overwrite and values-only jobs preserve geometry and reconstruct all rectangular shapes" {
    @setEvalBranchQuota(20000);
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |tall| for ("ASON") |ju| for ("ASON") |jv| {
            if (ju == 'O' and jv == 'O') continue;
            for (0..4) |variant| for ([_]owner.common.Level{ .high, .work }) |level| {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init(layout, tall, ju, jv, variant);
                const saved = f;
                try info(try adapter(raw).callManaged(true, T, level, &r.context, &r.services, f.args(raw), .{}), 0);
                try f.verify(&saved, level);
                try std.testing.expectEqual(@as(usize, 1), r.max_active);
                try std.testing.expectEqual(r.x.attempts, r.x.frees);
                try r.done();
                cases += 1;
            };
        };
    };
    std.debug.print("GESVD numerical jobs/layout/shape/rank/scalar/access/entry configs={d}\n", .{cases});
}
test "GESVD workspace query compressed paths single rounding and scalar errors have no operand or allocation access" {
    @setEvalBranchQuota(20000);
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |tall| for ([_]bool{ false, true }) |compressed| {
            var r: h.Rig = .{};
            r.init();
            var w = [_]T{ z(T, 81, 82), z(T, 83, 84) };
            var a: adapter(raw).Arguments(T) = .{ .layout = layout, .jobu = if (tall and compressed) 'N' else 'A', .jobvt = if (!tall and compressed) 'N' else 'A', .m = if (tall) 14 else 3, .n = if (tall) 3 else 14, .lda = 16, .ldu = 16, .ldvt = 16, .work = operand(T, raw, &w), .lwork = -1 };
            try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), 0);
            const expected: f64 = if (comptime Core.Scalar.isComplex(T)) (if (compressed) 9 else 20) else (if (compressed) 15 else 23);
            try std.testing.expectEqual(expected, wide(T, w[0]).re);
            try bytes(&[_]T{z(T, 83, 84)}, w[1..]);
            a.lwork = @intFromFloat(expected - 1);
            try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), -14);
            a.lwork = -1;
            a.jobu = 'X';
            try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), -2);
            a.jobu = 'O';
            a.jobvt = 'O';
            try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), -3);
            try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
            try r.done();
            cases += 1;
        };
    };
    inline for (.{ f32, f64, C32, C64 }) |T| {
        var r: h.Rig = .{};
        r.init();
        var w: [1]T = undefined;
        const args: owner.GesvdAdapter.Arguments(T) = .{ .layout = 102, .jobu = 'A', .jobvt = 'N', .m = 16777216, .n = 1, .lda = 16777216, .ldu = 16777216, .ldvt = 1, .work = &w, .lwork = -1 };
        try info(try owner.GesvdAdapter.callManaged(true, T, .work, &r.context, &r.services, args, .{}), 0);
        const expected: f64 = if (T == f32) 16777220 else (if (T == C32 or T == C64) 16777218 else 16777219);
        try std.testing.expectEqual(expected, wide(T, w[0]).re);
        try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
        try r.done();
    }
    std.debug.print("GESVD query scalar compressed controls={d} and four large single-rounding cases\n", .{cases});
}
test "GESVD every allocation failure is transactional and shared Context releases all original scratch" {
    @setEvalBranchQuota(20000);
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |tall| for ([_]owner.common.Level{ .high, .work }) |level| {
            var control: h.Rig = .{};
            control.init();
            var valid = Fixture(T).init(layout, tall, 'A', 'A', 0);
            try info(try adapter(raw).callManaged(true, T, level, &control.context, &control.services, valid.args(raw), .{}), 0);
            const count_a = control.x.attempts;
            try control.done();
            for (0..count_a) |fail| {
                var r: h.Rig = .{};
                r.init();
                r.x.fail_at = fail;
                var f = Fixture(T).init(layout, tall, 'A', 'A', 0);
                const saved = f;
                const result = try adapter(raw).callManaged(true, T, level, &r.context, &r.services, f.args(raw), .{});
                try std.testing.expect(result == .lapacke_info);
                try std.testing.expect(result.lapacke_info == -1010 or result.lapacke_info == -1011);
                try bytes(&saved.a, &f.a);
                try bytes(&saved.s, &f.s);
                try bytes(&saved.u, &f.u);
                try bytes(&saved.vt, &f.vt);
                try bytes(&saved.superb, &f.superb);
                try std.testing.expectEqual(fail, r.x.frees);
                try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
                try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
                try r.done();
                cases += 1;
            }
        };
    };
    std.debug.print("GESVD injected allocation failure configs={d}\n", .{cases});
}
test "GESVD admitted work finishes during close and partial exhaustion preserves factorization while numerical faults preserve outputs" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        for ([_]owner.common.Level{ .high, .work }) |level| {
            var r: h.Rig = .{};
            r.init();
            r.trigger = .alloc;
            r.action = .close;
            var f = Fixture(T).init(101, true, 'A', 'A', 0);
            const saved = f;
            try info(try adapter(raw).callManaged(true, T, level, &r.context, &r.services, f.args(raw), .{}), 0);
            try f.verify(&saved, level);
            try r.done();
            try std.testing.expectError(error.Closing, adapter(raw).callManaged(true, T, level, &r.context, &r.services, f.args(raw), .{}));
            var other: h.Rig = .{};
            other.init();
            var bad = Fixture(T).init(102, false, 'A', 'A', 0);
            const b0 = bad;
            const result = try adapter(raw).callManaged(true, T, level, &other.context, &other.services, bad.args(raw), .{ .max_sweeps = 0 });
            try info(result, 1);
            try bad.verifyPartial(&b0, level);
            try std.testing.expectEqual(@as(usize, 1), other.max_active);
            try std.testing.expectEqual(other.x.attempts, other.x.frees);
            try other.done();
            var fault_rig: h.Rig = .{};
            fault_rig.init();
            var broken = Fixture(T).init(102, false, 'A', 'A', 0);
            broken.a[0] = z(T, std.math.inf(f64), 0);
            const fault_before = broken;
            const fault = try adapter(raw).callManaged(false, T, level, &fault_rig.context, &fault_rig.services, broken.args(raw), .{});
            try std.testing.expect(fault == .numerical_fault and fault.numerical_fault == error.NonFiniteInput);
            try bytes(&fault_before.a, &broken.a);
            try bytes(&fault_before.s, &broken.s);
            try bytes(&fault_before.u, &broken.u);
            try bytes(&fault_before.vt, &broken.vt);
            try bytes(&fault_before.superb, &broken.superb);
            try bytes(&fault_before.rwork, &broken.rwork);
            try std.testing.expectEqual(@as(usize, 1), fault_rig.max_active);
            try std.testing.expectEqual(fault_rig.x.attempts, fault_rig.x.frees);
            try fault_rig.done();
        }
    };
    // Two nonzero upper bidiagonal edges remain when the QR budget is zero.
    var configurations: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        const R = Core.Scalar.Real(T);
        for ([_]owner.common.Level{ .high, .work }) |level| for ([_]c_int{ 101, 102 }) |layout| {
            var r: h.Rig = .{};
            r.init();
            var a: [20]T = @splat(z(T, 71, -17));
            for (0..3) |j| for (0..3) |i| {
                a[pos(layout, i, j, 5)] = z(T, 0, 0);
            };
            a[pos(layout, 0, 0, 5)] = z(T, 4, 0);
            a[pos(layout, 1, 1, 5)] = z(T, 3, 0);
            a[pos(layout, 2, 2, 5)] = z(T, 2, 0);
            a[pos(layout, 0, 1, 5)] = z(T, 1, 0);
            a[pos(layout, 1, 2, 5)] = z(T, 1, 0);
            var singular: [5]R = @splat(74);
            var work: [34]T = @splat(z(T, 77, -21));
            var real_work: [17]R = @splat(76);
            var superb: [4]R = @splat(75);
            const aa = a;
            const ss = singular;
            const ww = work;
            const rr = real_work;
            const ee = superb;
            try info(try adapter(raw).callManaged(true, T, level, &r.context, &r.services, .{ .layout = layout, .jobu = 'N', .jobvt = 'N', .m = 3, .n = 3, .lda = 5, .ldu = 1, .ldvt = 1, .a = operand(T, raw, &a), .singular = operand(R, raw, &singular), .work = operand(T, raw, &work), .lwork = 32, .rwork = operand(R, raw, &real_work), .superb = operand(R, raw, &superb) }, .{ .max_sweeps = 0 }), 2);
            try bytes(&aa, &a);
            var returned_energy: f64 = 0;
            for (singular[0..3]) |s| returned_energy += @as(f64, s) * s;
            for (0..2) |i| {
                const edge: f64 = if (level == .high) superb[i] else if (comptime Core.Scalar.isComplex(T)) real_work[i] else work[i + 1];
                try std.testing.expect(std.math.isFinite(edge) and edge != 0);
                returned_energy += edge * edge;
            }
            try std.testing.expectApproxEqAbs(@as(f64, 31), returned_energy, if (T == f32 or T == C32) 1e-4 else 1e-12);
            try bytes(ss[3..], singular[3..]);
            const real_untouched: usize = if (level == .work and Core.Scalar.isComplex(T)) 2 else 0;
            try bytes(rr[real_untouched..], real_work[real_untouched..]);
            const superb_untouched: usize = if (level == .high) 2 else 0;
            try bytes(ee[superb_untouched..], superb[superb_untouched..]);
            const untouched: usize = if (level == .high) 0 else if (comptime Core.Scalar.isComplex(T)) 1 else 3;
            try bytes(ww[untouched..], work[untouched..]);
            try std.testing.expectEqual(@as(usize, 1), r.max_active);
            try std.testing.expectEqual(r.x.attempts, r.x.frees);
            try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
            try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
            try r.done();
            configurations += 1;
        };
    };
    try std.testing.expectEqual(@as(usize, 32), configurations);
    std.debug.print("GESVD actual retained-edge INFO configurations={d} partial_energy_preserved=true lifecycle_clean=true\n", .{configurations});
}

test "GESVD empty extents NaN priority mixed byte aliases and unused outputs retain the original admission policy" {
    @setEvalBranchQuota(20000);
    var cases: usize = 0;
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |raw| {
        const R = Core.Scalar.Real(T);
        for ([_]c_int{ 101, 102 }) |layout| {
            for ([_]bool{ false, true }) |zero_rows| {
                var r: h.Rig = .{};
                r.init();
                var work = [_]T{ z(T, 91, 92), z(T, 93, 94) };
                const m: i32 = if (zero_rows) 0 else std.math.maxInt(i32);
                const n: i32 = if (zero_rows) std.math.maxInt(i32) else 0;
                const a: adapter(raw).Arguments(T) = .{
                    .layout = layout,
                    .jobu = 'A',
                    .jobvt = 'A',
                    .m = m,
                    .n = n,
                    .lda = if (layout == 101) n else @max(1, m),
                    .ldu = if (layout == 101) m else @max(1, m),
                    .ldvt = if (layout == 101) n else @max(1, n),
                    .work = operand(T, raw, &work),
                    .lwork = 1,
                };
                try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), 0);
                try std.testing.expectEqual(@as(f64, 1), wide(T, work[0]).re);
                try bytes(&[_]T{z(T, 93, 94)}, work[1..]);
                try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
                try r.done();
                cases += 1;
            }
            {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init(layout, true, 'N', 'N', 0);
                f.a[pos(layout, 0, 0, f.lda)] = z(T, std.math.nan(f64), 0);
                const saved = f;
                var a = f.args(raw);
                a.jobu = 'X';
                try info(try adapter(raw).callManaged(true, T, .high, &r.context, &r.services, a, .{}), -6);
                a.lwork = -1;
                try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), -2);
                try bytes(&saved.a, &f.a);
                try bytes(&saved.s, &f.s);
                try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
                try r.done();
                cases += 1;
            }
            {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init(layout, true, 'A', 'A', 0);
                const saved = f;
                var a = f.args(raw);
                const singular: [*]R = @ptrCast(&f.a);
                a.singular = operand(R, raw, singular[0..2]);
                const result = try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{});
                try std.testing.expect(result == .admission_fault);
                try std.testing.expectEqual(error.AliasingNotAllowed, result.admission_fault);
                try bytes(&saved.a, &f.a);
                try bytes(&saved.s, &f.s);
                try bytes(&saved.u, &f.u);
                try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
                try r.done();
                cases += 1;
            }
            {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init(layout, false, 'N', 'N', 0);
                var a = f.args(raw);
                a.u = null;
                a.vt = null;
                a.ldu = 1;
                a.ldvt = 1;
                try info(try adapter(raw).callManaged(true, T, .work, &r.context, &r.services, a, .{}), 0);
                try r.done();
                cases += 1;
            }
            {
                var r: h.Rig = .{};
                r.init();
                var f = Fixture(T).init(layout, false, 'S', 'S', 0);
                const saved = f;
                var a = f.args(raw);
                a.jobu = 's';
                a.jobvt = 's';
                try info(try adapter(raw).callManaged(true, T, .high, &r.context, &r.services, a, .{}), 0);
                try f.verify(&saved, .high);
                try r.done();
                cases += 1;
            }
        }
    };
    std.debug.print("GESVD empty/preflight/alias/unused/case-fold controls={d}\n", .{cases});
}
