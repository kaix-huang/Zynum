const std = @import("std");
const h = @import("managed_support.zig");
const s = h.s;
const Core = h.Core;
const F = h.F;
const S = h.S;
const C = h.C;
const common = @import("numerical-owner").common;

test "managed invocation uses the same owner service resource and matrix identities" {
    const actual_core = @import("zynum-core");
    const actual_execution = @import("zynum-execution");
    comptime {
        if (Core.OwnerIdentity != actual_core or F.OwnerIdentity != actual_core or S.OwnerIdentity != actual_core)
            @compileError("typed adapters must alias the actual implementation owner");
        if (Core.Dense != actual_core.internal.numerics.Dense or
            Core.Storage != actual_core.internal.numerics.Storage or
            Core.Scalar != actual_core.internal.numerics.Scalar or
            Core.Resources != actual_core.internal.numerics.Resources or
            Core.Execution != actual_core.internal.numerics.Execution or
            Core.Execution != actual_execution)
            @compileError("typed adapters must borrow the actual owner's providers");
    }
    // Address identity is within this artifact; do not close the default Context.
    try std.testing.expect(Core.Execution.getContext() == actual_execution.getContext());
    try std.testing.expect(Core.Execution.getContext() == actual_core.internal.numerics.Execution.getContext());
    try std.testing.expectEqual(@import("zynum-execution").reference.LegacyLifetime.immortal_default, Core.Execution.getContext().binding.legacy.lifetime);
    try std.testing.expect(F.Services == S.Services and F.Services == Core.Services);
    try std.testing.expect(F.Dense == S.Dense and F.Dense.Matrix(f64) == Core.Dense.Matrix(f64));
    try std.testing.expect(F.OwnerIdentity == S.OwnerIdentity);
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var invocation = try common.ManagedInvocation(Core).begin(r.context, &r.services);
    defer invocation.operation.end();
    try std.testing.expect(@TypeOf(invocation.services) == Core.Services);
    try std.testing.expect(invocation.services.allocator.ptr == r.x.allocator().ptr);
    var lease = try C.ContextScratch(Core.Scalar.ComplexF64).acquire(&invocation.operation, 1);
    defer lease.deinit();
    try std.testing.expect(@TypeOf(lease.lease) == Core.Resources.ScratchLease(Core.Scalar.ComplexF64));
    try std.testing.expect(common.Scratch(Core, f64, .context_managed).Source == *const C.Operation);
}

test "managed versus unmanaged normal domain has identical full output and original events" {
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| for ([_]s.owner.protocol.Level{ .high, .work }) |level| {
            var a: [6]T = @splat(s.value(T, 71, -3));
            a[s.pos(layout, 0, 0, 3)] = s.value(T, 4, 0);
            a[s.pos(layout, 1, 1, 3)] = s.value(T, 10, 0);
            a[s.pos(layout, if (uplo == 'L') 1 else 0, if (uplo == 'L') 0 else 1, 3)] = s.value(T, 2, if (Core.Scalar.isComplex(T)) 2 else 0);
            var b = a;
            var old: s.Context = .{};
            var new: s.Context = .{};
            var os = old.services();
            var ns = new.services();
            var reference_context: FixtureReferenceContext = undefined;
            reference_context.initInPlace(ns.allocator, h.execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const context = reference_context.borrowedContext();
            const old_result = F.call(T, I, level, &os, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &a });
            const new_result = try F.callManaged(T, I, level, context, &ns, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &b });
            try std.testing.expectEqual(old_result.lapacke_info, new_result.lapacke_info);
            try s.bytes(&a, &b);
            try s.trace(&old, new.events[0..new.count]);
            // Both solve the exact same factor and two RHS, including padding.
            var rhs: [6]T = @splat(s.value(T, 83, -7));
            rhs[s.pos(layout, 0, 0, 3)] = s.value(T, 10, 2);
            rhs[s.pos(layout, 1, 0, 3)] = s.value(T, 20, -1);
            rhs[s.pos(layout, 0, 1, 3)] = s.value(T, -6, 1);
            rhs[s.pos(layout, 1, 1, 3)] = s.value(T, 12, 3);
            var rhs2 = rhs;
            old = .{};
            new = .{};
            os = old.services();
            ns = new.services();
            const so = S.call(T, I, level, &os, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &a, .b = &rhs });
            const sn = try S.callManaged(T, I, level, context, &ns, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &b, .b = &rhs2 });
            try std.testing.expectEqual(so.lapacke_info, sn.lapacke_info);
            try s.bytes(&rhs, &rhs2);
            try s.bytes(&a, &b);
            try s.trace(&old, new.events[0..new.count]);
            try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
            _ = context.requestClose();
            try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(context));
        };
    };
}

test "managed capture ignores allocator Y and Z and retains the original callback" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    r.rebind = true;
    var a = [_]f64{4};
    var args = h.fargs(&a);
    args.uplo = '?';
    try h.info(i64, try F.callManaged(f64, i64, .high, r.context, &r.services, args), -2);
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.hooks);
    // An entirely undefined allocator field is legal for managed capture.
    r.services.allocator = undefined;
    r.services.error_event = h.Rig.hookForTest;
    try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, h.fargs(&a)), 0);
}

test "unmanaged compatibility still allocates using its captured input allocator" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var a = [_]f64{4};
    try h.info(i64, F.call(f64, i64, .work, &r.services, h.fargs(&a)), 0);
    try std.testing.expectEqual(@as(usize, 1), r.y.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.count);
}

test "Closing and operation overflow precede all service reads hooks and arrays" {
    for ([_]bool{ false, true }) |overflow| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        var held: ?C.Operation = null;
        if (overflow) {
            C.Context.testing.configure(r.context, 1, null);
            held = try r.context.begin();
        } else _ = r.context.requestClose();
        defer if (held) |*operation| operation.end();
        const err = if (overflow) error.CounterOverflow else error.Closing;
        var services: Core.Services = undefined;
        var a = [_]f64{9};
        var args = h.fargs(&a);
        args.layout = 777;
        try std.testing.expectError(err, F.callManaged(f64, i64, .high, r.context, &services, args));
        try std.testing.expectError(err, S.callManaged(f64, i64, .high, r.context, &services, h.sargs(null, null)));
        try std.testing.expectEqual(@as(f64, 9), a[0]);
        try std.testing.expectEqual(@as(usize, 0), r.count);
    }
}

test "managed getter and scan priority with work and disabled bypass" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    const nan = std.math.nan(f64);
    var a = [_]f64{nan};
    var b = [_]f64{nan};
    try h.info(i64, try S.callManaged(f64, i64, .high, r.context, &r.services, h.sargs(&a, &b)), -5);
    a[0] = 2;
    try h.info(i64, try S.callManaged(f64, i64, .high, r.context, &r.services, h.sargs(&a, &b)), -7);
    try std.testing.expectEqual(@as(usize, 2), r.reads);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
    try h.info(i64, try S.callManaged(f64, i64, .work, r.context, &r.services, h.sargs(&a, &b)), 0);
    try h.info(i64, try s.owner.NoScanSolveAdapter.callManaged(f64, i64, .high, r.context, &r.services, h.sargs(&a, &b)), 0);
    try std.testing.expectEqual(@as(usize, 2), r.reads);
}

test "getter and returning hook reentry admit separately with real nested numerical work" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .hook }) |trigger| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        r.trigger = trigger;
        r.action = .nested;
        var a = [_]f64{4};
        var args = h.fargs(&a);
        if (trigger == .hook) args.uplo = '?';
        try h.info(i64, try F.callManaged(f64, i64, .high, r.context, &r.services, args), if (trigger == .hook) -2 else 0);
        try std.testing.expectEqual(@as(f64, 2), r.nested_value);
        try std.testing.expectEqual(@as(usize, 2), r.max_active);
        try std.testing.expect(r.nested_error == null);
    }
}

test "getter and hook close deny nested admission while admitted outer work completes" {
    for ([_]@FieldType(h.Rig, "trigger"){ .getter, .hook }) |trigger| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        r.trigger = trigger;
        r.action = .close_nested;
        var a = [_]f64{2};
        var b = [_]f64{8};
        var args = h.sargs(&a, &b);
        if (trigger == .hook) args.uplo = '?';
        try h.info(i64, try S.callManaged(f64, i64, .high, r.context, &r.services, args), if (trigger == .hook) -2 else 0);
        try std.testing.expectEqual(error.Closing, r.nested_error.?);
        try std.testing.expectEqual(@as(f64, 4), r.nested_value);
        try std.testing.expectEqual(@as(usize, 1), r.reads);
        try std.testing.expectEqual(@as(usize, 2), r.x.attempts);
        try std.testing.expectEqual(@as(f64, if (trigger == .hook) 8 else 2), b[0]);
    }
}

test "first scratch pin overflow uses no allocator and no standard error hook" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    C.Context.testing.configure(r.context, 1, null);
    var holding = try r.context.begin();
    var held = try C.ContextScratch(u8).acquire(&holding, 1);
    holding.end();
    var a = [_]f64{4};
    try std.testing.expectError(error.CounterOverflow, F.callManaged(f64, i64, .work, r.context, &r.services, h.fargs(&a)));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
    try std.testing.expectEqual(@as(f64, 4), a[0]);
    held.deinit();
}

test "POTRS second pin overflow frees A after no B allocation and no hook" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    C.Context.testing.configure(r.context, 1, null);
    var a = [_]f64{2};
    var b = [_]f64{8};
    try std.testing.expectError(error.CounterOverflow, S.callManaged(f64, i64, .work, r.context, &r.services, h.sargs(&a, &b)));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
    try std.testing.expectEqual(@as(f64, 8), b[0]);
}

test "only real first OOM maps to minus1011 after its pin rolls back" {
    for ([_]bool{ false, true }) |solve| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        r.x.fail_at = 0;
        var a = [_]f64{2};
        var b = [_]f64{8};
        if (solve) try h.info(i64, try S.callManaged(f64, i64, .work, r.context, &r.services, h.sargs(&a, &b)), -1011) else try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, h.fargs(&a)), -1011);
        for (r.events[0..r.count]) |e| if (e.tag == .hook) {
            try std.testing.expectEqual(@as(usize, 0), e.pins);
            try std.testing.expectEqual(@as(usize, 1), e.active);
        };
        try std.testing.expectEqual(@as(usize, 1), r.hooks);
        try std.testing.expectEqual(@as(usize, 0), r.x.frees);
    }
}

test "POTRS second OOM releases A before standard hook with no pins remaining" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    r.x.fail_at = 1;
    var a = [_]f64{2};
    var b = [_]f64{8};
    try h.info(i64, try S.callManaged(f64, i64, .work, r.context, &r.services, h.sargs(&a, &b)), -1011);
    var freed = false;
    for (r.events[0..r.count]) |e| {
        if (e.tag == .free) {
            freed = true;
            try std.testing.expectEqual(@as(usize, 1), e.pins);
        }
        if (e.tag == .hook) {
            try std.testing.expect(freed);
            try std.testing.expectEqual(@as(usize, 0), e.pins);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 1), r.hooks);
}

test "POTRS B extent fault stays after A allocation with typed cleanup" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var args = h.sargs(null, null);
    args.nrhs = std.math.maxInt(i64);
    args.ldb = args.nrhs;
    const result = try S.callManaged(f64, i64, .work, r.context, &r.services, args);
    try std.testing.expect(result == .admission_fault);
    try std.testing.expect(result.admission_fault == error.SizeOverflow or result.admission_fault == error.IntegerNotRepresentable);
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
}

test "row empty calls retain scratch acquisitions while column empty holds only operation" {
    for ([_]c_int{ 101, 102 }) |layout| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        var f = h.fargs(null);
        f.layout = layout;
        f.n = 0;
        try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, f), 0);
        var a = h.sargs(null, null);
        a.layout = layout;
        a.n = 0;
        a.nrhs = 0;
        try h.info(i64, try S.callManaged(f64, i64, .work, r.context, &r.services, a), 0);
        try std.testing.expectEqual(@as(usize, if (layout == 101) 3 else 0), r.x.attempts);
        var starts: usize = 0;
        var ends: usize = 0;
        for (r.events[0..r.count]) |e| {
            if (e.tag == .admitted) starts += 1;
            if (e.tag == .finishing) {
                ends += 1;
                try std.testing.expectEqual(@as(usize, 0), e.pins);
                try std.testing.expectEqual(@as(usize, 1), e.active);
            }
        }
        try std.testing.expectEqual(@as(usize, 2), starts);
        try std.testing.expectEqual(@as(usize, 2), ends);
    }
}

const Worker = struct {
    rig: *h.Rig,
    a: []const f64,
    b: []f64,
    result: ?S.Result(i64) = null,
    err: ?anyerror = null,
    fn run(self: *Worker) void {
        self.result = S.callManaged(f64, i64, .work, self.rig.context, &self.rig.services, .{ .layout = 101, .uplo = '?', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = self.a, .b = self.b }) catch |err| {
            self.err = err;
            return;
        };
    }
};
test "returning hook close still copies active B back then frees B and A before end" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var a = [_]f64{ 2, 0, 0, 3 };
    var b = [_]f64{ 10, 77, 20, 88 };
    r.b = &b;
    r.expect_copyback = true;
    r.trigger = .hook;
    r.action = .close;
    var worker = Worker{ .rig = &r, .a = &a, .b = &b };
    worker.run();
    try std.testing.expect(worker.err == null);
    try h.info(i64, worker.result.?, -2);
    try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20, -900 }, &b);
    var frees: usize = 0;
    for (r.events[0..r.count]) |e| {
        if (e.tag == .free) {
            try std.testing.expectEqual(2 - frees, e.pins);
            try std.testing.expectEqual(@as(usize, 1), e.active);
            frees += 1;
        }
        if (e.tag == .finishing) {
            try std.testing.expectEqual(@as(usize, 0), e.pins);
            try std.testing.expectEqual(@as(usize, 1), e.active);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), frees);
}

test "real paused returning hook or free blocks close until copyback cleanup and operation end" {
    for ([_]bool{ false, true }) |pause_free| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        var a = [_]f64{ 2, 0, 0, 3 };
        var b = [_]f64{ 10, 77, 20, 88 };
        r.b = &b;
        r.expect_copyback = true;
        r.pause_free = pause_free;
        r.pause_hook = !pause_free;
        var worker = Worker{ .rig = &r, .a = &a, .b = &b };
        const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
        h.execution.test_support.wait(&r.paused);
        const state = r.context.snapshot();
        try std.testing.expectEqual(@as(usize, 1), state.active_operations);
        try std.testing.expectEqual(@as(usize, 2), state.resource_pins);
        _ = r.context.requestClose();
        try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(r.context));
        r.continue_free.store(true, .release);
        thread.join();
        try std.testing.expect(worker.err == null);
        try h.info(i64, worker.result.?, -2);
        try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20, -900 }, &b);
    }
}

test "allocator callbacks may reenter and request close outside Context gate" {
    for ([_]@FieldType(h.Rig, "trigger"){ .alloc, .free }) |trigger| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        r.trigger = trigger;
        r.action = .nested;
        var a = [_]f64{4};
        try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, h.fargs(&a)), 0);
        try std.testing.expectEqual(@as(f64, 2), r.nested_value);
        try std.testing.expectEqual(@as(usize, 2), r.max_active);
        r.action = .close;
        r.nested_value = 4;
        a[0] = 4;
        try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, h.fargs(&a)), 0);
        try std.testing.expectEqual(@as(f64, 2), a[0]);
    }
}

test "managed admission faults and scalar errors preserve unmanaged result and output" {
    for (0..7) |which| {
        var old: s.Context = .{};
        var new: s.Context = .{};
        var os = old.services();
        var ns = new.services();
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(ns.allocator, h.execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        var a = [_]f64{2};
        var b = [_]f64{ 8, 71 };
        var b2 = b;
        var args = h.sargs(&a, &b);
        switch (which) {
            0 => args.layout = 99,
            1 => args.lda = 0,
            2 => args.ldb = 0,
            3 => args.uplo = '?',
            4 => args.b = null,
            5 => args.b = b[0..0],
            6 => {
                args.layout = 102;
                args.n = -1;
            },
            else => unreachable,
        }
        var args2 = args;
        if (args.b != null) args2.b = if (which == 5) b2[0..0] else &b2;
        const x = S.call(f64, i64, .work, &os, args);
        const y = try S.callManaged(f64, i64, .work, ctx, &ns, args2);
        try std.testing.expectEqual(std.meta.activeTag(x), std.meta.activeTag(y));
        switch (x) {
            .lapacke_info => |v| try std.testing.expectEqual(v, y.lapacke_info),
            .admission_fault => |v| try std.testing.expectEqual(v, y.admission_fault),
        }
        try s.bytes(&b, &b2);
        try s.trace(&old, new.events[0..new.count]);
        _ = ctx.requestClose();
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(ctx));
    }
}

test "managed native width one element accepts strides above i32 on 64 bit targets" {
    if (@sizeOf(usize) < 8) return;
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var a = [_]f64{4};
    var b = [_]f64{8};
    const ld = @as(i64, std.math.maxInt(i32)) + 19;
    try h.info(i64, try F.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = ld, .values = &a }), 0);
    try h.info(i64, try S.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .lda = ld, .ldb = ld, .a = &a, .b = &b }), 0);
    try std.testing.expectEqual(@as(f64, 2), b[0]);
}

test "unmanaged provider needs no Execution field and reuses owner identities" {
    const UnmanagedCore = struct {
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Dense = Core.Dense;
        pub const Storage = Core.Storage;
        pub const Scalar = Core.Scalar;
        pub const ScratchLease = Core.ScratchLease;
        pub const Services = Core.Services;
        pub const factorPotrf = Core.factorPotrf;
        pub const solvePotrs = Core.solvePotrs;
    };
    const LegacyF = s.owner.protocol.ReferencePotrfAdapter(UnmanagedCore, true);
    const LegacyS = s.owner.solve_protocol.ReferencePotrsAdapter(UnmanagedCore, true);
    try std.testing.expect(LegacyF.Services == Core.Services and LegacyS.Dense == Core.Dense);
    var old: s.Context = .{};
    var services = old.services();
    var a = [_]f64{4};
    var b = [_]f64{8};
    try h.info(i64, LegacyF.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 1, .lda = 1, .values = &a }), 0);
    try h.info(i64, LegacyS.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }), 0);
    try std.testing.expectEqual(@as(f64, 2), b[0]);
    try std.testing.expectEqual(@as(usize, 0), old.live);
}

const FactorWorker = struct {
    rig: *h.Rig,
    a: [1]f64 = .{4},
    err: ?anyerror = null,
    fn run(self: *@This()) void {
        const result = F.callManaged(f64, i64, .work, self.rig.context, &self.rig.services, h.fargs(&self.a)) catch |err| {
            self.err = err;
            return;
        };
        if (result != .lapacke_info or result.lapacke_info != 0) @panic("factor worker result");
    }
};
test "POTRF free POTRS A free and final operation boundary retain exact live counts" {
    for (0..4) |stage| {
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        r.pause_free = stage < 2;
        r.pause_free_index = if (stage == 1) 1 else 0;
        r.pause_finish = stage >= 2;
        var a = [_]f64{ 2, 0, 0, 3 };
        var b = [_]f64{ 10, 77, 20, 88 };
        var sw = Worker{ .rig = &r, .a = &a, .b = &b };
        var fw = FactorWorker{ .rig = &r };
        const factor = stage == 0 or stage == 2;
        const thread = if (factor) try std.Thread.spawn(.{}, FactorWorker.run, .{&fw}) else try std.Thread.spawn(.{}, Worker.run, .{&sw});
        h.execution.test_support.wait(&r.paused);
        const state = r.context.snapshot();
        try std.testing.expectEqual(@as(usize, 1), state.active_operations);
        try std.testing.expectEqual(@as(usize, if (stage < 2) 1 else 0), state.resource_pins);
        _ = r.context.requestClose();
        try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(r.context));
        r.continue_free.store(true, .release);
        thread.join();
        try std.testing.expect(fw.err == null and sw.err == null);
        if (factor) try std.testing.expectEqual(@as(f64, 2), fw.a[0]) else try h.info(i64, sw.result.?, -2);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(r.context));
        std.debug.print("pause stage={d} observed active={d} pins={d}; after join active=0 pins=0 closed\n", .{ stage, state.active_operations, state.resource_pins });
    }
}

test "managed numerical failure and IEEE policies match unmanaged full buffer and events" {
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| {
            var old: s.Context = .{};
            var new: s.Context = .{};
            var os = old.services();
            var ns = new.services();
            var reference_ctx: FixtureReferenceContext = undefined;
            reference_ctx.initInPlace(ns.allocator, h.execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const ctx = reference_ctx.borrowedContext();
            defer {
                _ = ctx.requestClose();
                std.debug.assert(fixtureProgressClose(ctx) == .closed);
            }
            var a: [6]T = @splat(s.value(T, 77, -5));
            a[s.pos(layout, 0, 0, 3)] = s.value(T, 1, 0);
            a[s.pos(layout, 1, 1, 3)] = s.value(T, 1, 0);
            a[s.pos(layout, if (uplo == 'L') 1 else 0, if (uplo == 'L') 0 else 1, 3)] = s.value(T, 2, 0);
            var a2 = a;
            const x = F.call(T, i64, .work, &os, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &a });
            const y = try F.callManaged(T, i64, .work, ctx, &ns, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &a2 });
            try h.info(i64, x, 2);
            try h.info(i64, y, 2);
            try s.bytes(&a, &a2);
            try s.trace(&old, new.events[0..new.count]);
            for (0..7) |scenario| {
                old = .{};
                new = .{};
                os = old.services();
                ns = new.services();
                const diag: f64 = switch (scenario) {
                    0 => 2,
                    1 => 0,
                    2 => std.math.inf(f64),
                    3 => std.math.nan(f64),
                    4 => if (T == f32 or T == Core.Scalar.ComplexF32) 1e20 else 1e200,
                    5 => if (T == f32 or T == Core.Scalar.ComplexF32) 1e-20 else 1e-200,
                    6 => -0.0,
                    else => unreachable,
                };
                var f = [_]T{s.value(T, diag, if (scenario == 0) 1 else 0)};
                var b = [_]T{ s.value(T, 10, 0), s.value(T, 89, -0.0) };
                var b2 = b;
                const sx = S.call(T, i64, .work, &os, .{ .layout = layout, .uplo = uplo, .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &f, .b = &b });
                const sy = try S.callManaged(T, i64, .work, ctx, &ns, .{ .layout = layout, .uplo = uplo, .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &f, .b = &b2 });
                try h.info(i64, sx, 0);
                try h.info(i64, sy, 0);
                try s.bytes(&b, &b2);
                try s.trace(&old, new.events[0..new.count]);
                if (scenario == 0 and Core.Scalar.isComplex(T)) {
                    try std.testing.expectEqual(@as(f64, 2), s.re(T, b[0]));
                    try std.testing.expectEqual(@as(f64, 0), s.im(T, b[0]));
                }
            }
        };
    }
}

test "managed Cholesky retains instance admission scratch and complex diagonal policy without default" {
    const execution = @import("zynum-execution");
    const binding = @import("numerical-owner");
    const cfg = binding.shared_services;
    const ManagedProbe = struct {
        var actual: []const u8 = &.{};
        var expected: []const u8 = &.{};
        fn watch(a: anytype, b: anytype) void {
            actual = std.mem.sliceAsBytes(a);
            expected = std.mem.sliceAsBytes(b);
        }
        fn observe(ctx: *C.Context, event: C.TestEvent) void {
            const rig: *h.Rig = @fieldParentPtr("reference_context", @as(*FixtureReferenceContext, @fieldParentPtr("context", ctx)));
            if (ctx == execution.getContext() or ctx.snapshot().active_operations != 1) @panic("managed Cholesky admission split");
            rig.record(if (event == .admitted) .admitted else .finishing, 0);
            if (event == .finishing and (ctx.snapshot().resource_pins != 0 or rig.x.live != 0)) @panic("managed scratch outlived operation");
            if (event == .finishing and !std.mem.eql(u8, actual, expected)) @panic("managed copyback incomplete before end");
        }
    };
    cfg.testing.reset();
    defer cfg.testing.reset();
    // A stray call through the new public checked frontend must fail immediately.
    C.Context.testing.configure(execution.getContext(), 0, null);
    defer C.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| {
            var r: h.Rig = .{};
            r.init();
            defer r.done() catch @panic("managed Cholesky cleanup");
            // The same limit also bounds pins: row-major solve needs two.
            // Observer active==1 supplies the strict single-operation assertion.
            C.Context.testing.configure(r.context, 2, ManagedProbe.observe);
            var a = [_]T{ s.value(T, 4, 7), s.value(T, 71, -3) };
            var b = [_]T{ s.value(T, 12, 4), s.value(T, 83, -7) };
            const expected_a = [_]T{ s.value(T, 2, 0), a[1] };
            const expected_b = [_]T{ s.value(T, 3, 1), b[1] };
            // Complex imaginary diagonal is ignored by lapack_potrf. Routing
            // through the checked admitted frontend would reject it instead.
            ManagedProbe.watch(&a, &expected_a);
            try h.info(I, try binding.Shared.factorManaged(true, T, I, .work, r.context, .{ .layout = layout, .uplo = 'L', .n = 1, .lda = 2, .values = &a }), 0);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
            ManagedProbe.watch(&b, &expected_b);
            try h.info(I, try binding.Shared.solveManaged(true, T, I, .work, r.context, .{ .layout = layout, .uplo = 'U', .n = 1, .nrhs = 1, .lda = 2, .ldb = 2, .a = &a, .b = &b }), 0);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
            try s.bytes(&expected_a, &a);
            try s.bytes(&expected_b, &b);
            var admits: usize = 0;
            var finishes: usize = 0;
            for (r.events[0..r.count]) |event| {
                try std.testing.expectEqual(@as(usize, 1), event.active);
                if (event.tag == .admitted) admits += 1;
                if (event.tag == .finishing) {
                    finishes += 1;
                    try std.testing.expectEqual(@as(usize, 0), event.pins);
                }
                if (event.tag == .free) try std.testing.expect(event.pins > 0);
            }
            try std.testing.expectEqual(@as(usize, 2), admits);
            try std.testing.expectEqual(@as(usize, 2), finishes);
            try std.testing.expectEqual(@as(usize, if (layout == 101) 3 else 0), r.x.attempts);
            try std.testing.expectEqual(r.x.attempts, r.x.frees);
            try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
            try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
            try std.testing.expectEqual(@as(usize, 0), execution.getContext().snapshot().active_operations);
        }
    };
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
