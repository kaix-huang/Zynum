const std = @import("std");
const owner = @import("numerical-owner");
const actual_core = @import("zynum-core");
const actual_execution = @import("zynum-execution");
const h = @import("managed_support.zig");
const s = h.s;
const cfg = owner.shared_services;
const Shared = owner.Shared;
const Core = owner.Provider;
const C = Core.Execution.context;

fn check(ok: bool) void {
    if (!ok) @panic("shared service fixture contract violation");
}
fn reset() void {
    cfg.testing.reset();
}
fn bindRig(r: *h.Rig) void {
    const callback: cfg.Callback = .{ .userdata = r, .function = h.Rig.hookForTest };
    _ = cfg.setCallbacks(.{ .lapacke = callback, .fortran_returning_test = callback });
}

test "shared config default routes retain actual owner identities across all scalar and integer specializations" {
    reset();
    defer reset();
    comptime {
        if (Core.OwnerIdentity != actual_core or Core.Execution != actual_execution or
            Core.Resources != actual_core.internal.numerics.Resources or
            Core.Execution != actual_core.internal.numerics.Execution)
            @compileError("shared services require the actual injected owner");
    }
    try std.testing.expect(Core.Execution.getContext() == actual_execution.getContext());
    try std.testing.expect(Core.Execution.get() == fixtureExecutor(actual_execution.getContext()));
    const before = actual_execution.getContext().snapshot();
    const identity = cfg.testing.identity();
    cfg.setNancheck(0);
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| {
            var a = [_]T{s.value(T, 4, 0)};
            var b = [_]T{s.value(T, 12, 0)};
            try h.info(I, try Shared.factorDefault(true, T, I, .high, .{ .layout = layout, .uplo = 'L', .n = 1, .lda = 1, .values = &a }), 0);
            try std.testing.expectEqual(s.value(T, 2, 0), a[0]);
            try h.info(I, try Shared.solveDefault(true, T, I, .high, .{ .layout = layout, .uplo = 'U', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }), 0);
            try std.testing.expectEqual(s.value(T, 3, 0), b[0]);
            try std.testing.expect(cfg.testing.identity() == identity);
        }
    };
    const stats = cfg.testing.snapshot();
    try std.testing.expectEqual(@as(usize, 32), stats.captures);
    try std.testing.expectEqual(@as(usize, 32), stats.getter_calls);
    try std.testing.expectEqual(@as(usize, 0), stats.environment_reads);
    const after = actual_execution.getContext().snapshot();
    try std.testing.expectEqual(before, after);
    try std.testing.expectEqual(@import("zynum-execution").reference.LegacyLifetime.immortal_default, actual_execution.getContext().binding.legacy.lifetime);
}

test "shared NaN state reads the real stable environment once and setter avoids initialization" {
    reset();
    defer reset();
    // Run this fixture in fresh processes with unset, 0 and 1. It never mutates
    // environ; unexpected ambient strings are a harness error, not a skip.
    const raw = std.c.getenv("LAPACKE_NANCHECK");
    const expected: c_int = if (raw) |v| blk: {
        const text = std.mem.span(v);
        if (std.mem.eql(u8, text, "0")) break :blk 0;
        if (std.mem.eql(u8, text, "1")) break :blk 1;
        return error.SharedServicesEnvironmentMustBeUnsetZeroOrOne;
    } else 1;
    try std.testing.expectEqual(expected, cfg.getNancheck());
    try std.testing.expectEqual(expected, cfg.getNancheck());
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().environment_reads);
    cfg.setNancheck(-7);
    try std.testing.expectEqual(@as(c_int, 1), cfg.getNancheck());
    cfg.setNancheck(0);
    try std.testing.expectEqual(@as(c_int, 0), cfg.getNancheck());
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().environment_reads);
    reset();
    cfg.setNancheck(3);
    try std.testing.expectEqual(@as(c_int, 1), cfg.getNancheck());
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
}

test "shared NaN parser uses defined ASCII decimal prefix and explicit overflow extension" {
    const Case = struct { text: []const u8, enabled: bool };
    for ([_]Case{
        .{ .text = "", .enabled = false },                              .{ .text = "0", .enabled = false },
        .{ .text = "-000tail", .enabled = false },                      .{ .text = " \t\n\r\x0b\x0c+002tail", .enabled = true },
        .{ .text = "-17", .enabled = true },                            .{ .text = "+", .enabled = false },
        .{ .text = "x1", .enabled = false },                            .{ .text = "- 2", .enabled = false },
        .{ .text = "0x10", .enabled = false },                          .{ .text = "000000000000000000000000000000", .enabled = false },
        // The next behavior is an explicit extension outside atoi's defined
        // C-int range, not a Reference or MKL conformance assertion.
        .{ .text = "429496729600000000000000000000", .enabled = true },
    }) |item| try std.testing.expectEqual(item.enabled, cfg.testing.parseEnvironment(item.text));
}

test "shared NaN setting changes actual high behavior across scalar and integer variants" {
    reset();
    defer reset();
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        const R = if (T == f32 or T == Core.Scalar.ComplexF32) f32 else f64;
        var a = [_]T{s.value(T, 4, 0)};
        if (comptime Core.Scalar.isComplex(T)) a[0].re = std.math.nan(R) else a[0] = std.math.nan(R);
        cfg.setNancheck(-1);
        try h.info(I, try Shared.factorDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &a }), -4);
        cfg.setNancheck(0);
        try h.info(I, try Shared.factorDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &a }), 1);
        a[0] = s.value(T, 2, 0);
        var b = [_]T{s.value(T, 12, 0)};
        if (comptime Core.Scalar.isComplex(T)) b[0].re = std.math.nan(R) else b[0] = std.math.nan(R);
        cfg.setNancheck(1);
        try h.info(I, try Shared.solveDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'U', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }), -7);
        cfg.setNancheck(0);
        try h.info(I, try Shared.solveDefault(true, T, I, .high, .{ .layout = 102, .uplo = 'U', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }), 0);
    };
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
}

test "shared service capture does not read NaN until the original valid high entry point" {
    defer reset();
    inline for (.{ true, false }) |scan| for ([_]owner.common.Level{ .high, .work }) |level| {
        reset();
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        var a = [_]f64{4};
        try h.info(i64, try Shared.factorManaged(scan, f64, i64, level, r.context, h.fargs(&a)), 0);
        var rhs = [_]f64{12};
        try h.info(i64, try Shared.solveManaged(scan, f64, i64, level, r.context, h.sargs(&a, &rhs)), 0);
        try std.testing.expectEqual(@as(f64, 3), rhs[0]);
        const stats = cfg.testing.snapshot();
        const reads: usize = if (scan and level == .high) 2 else 0;
        try std.testing.expectEqual(reads, stats.getter_calls);
        try std.testing.expectEqual(@as(usize, if (reads > 0) 1 else 0), stats.environment_reads);
        try std.testing.expectEqual(@as(usize, 2), stats.captures);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
    };
    reset();
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    try h.info(i64, try Shared.factorManaged(true, f64, i64, .high, r.context, .{ .layout = -1, .uplo = 'L', .n = 1, .lda = 1, .values = null }), -1);
    try h.info(i64, try Shared.solveManaged(true, f64, i64, .high, r.context, .{ .layout = -1, .uplo = 'L', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = null }), -1);
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().getter_calls);
    try std.testing.expect(!cfg.testing.snapshot().initialized);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
}

const Events = struct {
    kind: owner.common.HookKind,
    calls: usize = 0,
    last: ?owner.common.ErrorEvent = null,
    fn hook(raw: *anyopaque, event: owner.common.ErrorEvent) void {
        const self: *Events = @ptrCast(@alignCast(raw));
        check(event.kind == self.kind);
        self.calls += 1;
        self.last = event;
    }
    fn callback(self: *Events) cfg.Callback {
        return .{ .userdata = self, .function = hook };
    }
};

test "shared callback channels remain separate and replacement returns the exact previous tuple" {
    reset();
    defer reset();
    var c = Events{ .kind = .lapacke };
    var f = Events{ .kind = .fortran_returning_test };
    var newer = Events{ .kind = .lapacke };
    const empty = cfg.setCallbacks(.{ .lapacke = c.callback(), .fortran_returning_test = f.callback() });
    try std.testing.expect(empty.lapacke == null and empty.fortran_returning_test == null);
    try h.info(i32, try Shared.factorDefault(true, f64, i32, .high, .{ .layout = 0, .uplo = 'L', .n = 0, .lda = 1, .values = null }), -1);
    try h.info(i64, try Shared.solveDefault(true, f64, i64, .work, .{ .layout = 102, .uplo = '?', .n = 0, .nrhs = 0, .lda = 1, .ldb = 1, .a = null, .b = null }), -2);
    try std.testing.expectEqual(@as(usize, 1), c.calls);
    try std.testing.expectEqual(@as(usize, 1), f.calls);
    try std.testing.expectEqual(owner.common.Routine.potrf, c.last.?.routine);
    try std.testing.expectEqual(owner.common.Routine.potrs, f.last.?.routine);
    const old = cfg.setCallbacks(.{ .lapacke = newer.callback(), .fortran_returning_test = f.callback() });
    try std.testing.expect(old.lapacke.?.userdata == @as(*anyopaque, &c) and old.lapacke.?.function == Events.hook);
    try std.testing.expect(old.fortran_returning_test.?.userdata == @as(*anyopaque, &f));
    try h.info(i32, try Shared.factorDefault(false, f64, i32, .work, .{ .layout = 0, .uplo = 'L', .n = 0, .lda = 1, .values = null }), -1);
    try std.testing.expectEqual(@as(usize, 1), newer.calls);
    try std.testing.expectEqual(@as(usize, 1), c.calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().getter_calls);
}

test "admitted seams rebuild allocator authority without another admission or reading borrowed allocator" {
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    var a = [_]f64{4};
    var b = [_]f64{12};
    {
        var operation = try r.context.begin();
        defer operation.end();
        try h.info(i64, try h.F.callAdmitted(f64, i64, .work, &operation, &r.services, h.fargs(&a)), 0);
        r.services.allocator = undefined;
        try h.info(i64, try h.S.callAdmitted(f64, i64, .work, &operation, &r.services, h.sargs(&a, &b)), 0);
        try std.testing.expectEqual(@as(usize, 1), r.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
    }
    try std.testing.expectEqual(@as(f64, 2), a[0]);
    try std.testing.expectEqual(@as(f64, 3), b[0]);
    try std.testing.expectEqual(@as(usize, 3), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 3), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
    try std.testing.expectEqual(@as(usize, 0), r.z.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
}

test "shared begin refusal precedes configuration capture getter and allocation" {
    reset();
    defer reset();
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    C.Context.testing.configure(r.context, 0, null);
    try std.testing.expectError(error.CounterOverflow, Shared.factorManaged(true, f64, i64, .high, r.context, h.fargs(null)));
    C.Context.testing.configure(r.context, 1, null);
    _ = r.context.requestClose();
    try std.testing.expectError(error.Closing, Shared.solveManaged(true, f64, i64, .high, r.context, h.sargs(null, null)));
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
}

test "shared row OOM retains real allocation and reverse cleanup before the C hook" {
    defer reset();
    for ([_]bool{ false, true }) |solve| {
        reset();
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        bindRig(&r);
        r.x.fail_at = if (solve) 1 else 0;
        var a = [_]f64{2};
        var b = [_]f64{12};
        if (solve) {
            try h.info(i64, try Shared.solveManaged(true, f64, i64, .work, r.context, h.sargs(&a, &b)), -1011);
        } else {
            try h.info(i64, try Shared.factorManaged(true, f64, i64, .work, r.context, h.fargs(&a)), -1011);
        }
        try std.testing.expectEqual(@as(usize, if (solve) 2 else 1), r.x.attempts);
        try std.testing.expectEqual(@as(usize, if (solve) 1 else 0), r.x.frees);
        try std.testing.expectEqual(@as(usize, 1), r.hooks);
        try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
        try std.testing.expectEqual(@as(f64, 2), a[0]);
        try std.testing.expectEqual(@as(f64, 12), b[0]);
        var seen_free = false;
        for (r.events[0..r.count]) |event| {
            if (event.tag == .free) seen_free = true;
            if (event.tag == .hook) {
                try std.testing.expectEqual(@as(usize, 1), event.active);
                try std.testing.expectEqual(@as(usize, 0), event.pins);
                try std.testing.expectEqual(solve, seen_free);
            }
        }
    }
}

test "shared second scratch pin overflow frees the first lease without a standard error hook" {
    reset();
    defer reset();
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    bindRig(&r);
    C.Context.testing.configure(r.context, 1, null);
    var a = [_]f64{2};
    var b = [_]f64{12};
    try std.testing.expectError(error.CounterOverflow, Shared.solveManaged(true, f64, i64, .work, r.context, h.sargs(&a, &b)));
    try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
    try std.testing.expectEqual(@as(f64, 12), b[0]);
    try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
}

test "shared first scratch pin overflow does not allocate and empty row calls retain original leases" {
    reset();
    defer reset();
    var r: h.Rig = .{};
    r.init();
    defer r.done() catch @panic("cleanup");
    bindRig(&r);
    C.Context.testing.configure(r.context, 1, null);
    {
        var operation = try r.context.begin();
        var held = C.ContextScratch(u8).acquire(&operation, 1) catch |err| {
            operation.end();
            return err;
        };
        operation.end();
        defer held.deinit();
        var a = [_]f64{4};
        try std.testing.expectError(error.CounterOverflow, Shared.factorManaged(true, f64, i64, .work, r.context, h.fargs(&a)));
        try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 0), r.hooks);
        try std.testing.expectEqual(@as(usize, 1), r.context.snapshot().resource_pins);
    }
    C.Context.testing.configure(r.context, std.math.maxInt(usize), null);
    const attempts = r.x.attempts;
    for ([_]c_int{ 101, 102 }) |layout| {
        const ld: i64 = if (layout == 101) 0 else 1;
        try h.info(i64, try Shared.factorManaged(true, f64, i64, .work, r.context, .{ .layout = layout, .uplo = 'L', .n = 0, .lda = ld, .values = null }), 0);
        try h.info(i64, try Shared.solveManaged(true, f64, i64, .work, r.context, .{ .layout = layout, .uplo = 'L', .n = 0, .nrhs = 0, .lda = ld, .ldb = ld, .a = null, .b = null }), 0);
    }
    try std.testing.expectEqual(attempts + 3, r.x.attempts);
    try std.testing.expectEqual(r.x.attempts, r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().getter_calls);
}

const Reentrant = struct {
    context: *C.Context,
    close_first: bool,
    old_calls: usize = 0,
    new_calls: usize = 0,
    nested_value: f64 = 4,
    nested_error: ?anyerror = null,
    fn newer(raw: *anyopaque, event: owner.common.ErrorEvent) void {
        const self: *Reentrant = @ptrCast(@alignCast(raw));
        check(event.kind == .fortran_returning_test and event.routine == .potrs);
        self.new_calls += 1;
    }
    fn original(raw: *anyopaque, _: owner.common.ErrorEvent) void {
        const self: *Reentrant = @ptrCast(@alignCast(raw));
        self.old_calls += 1;
        _ = cfg.setCallbacks(.{ .lapacke = .{ .userdata = self, .function = newer }, .fortran_returning_test = .{ .userdata = self, .function = newer } });
        cfg.setNancheck(-1);
        check(cfg.getNancheck() == 1);
        if (self.close_first) {
            check(self.context.requestClose() == .pending);
            check(fixtureProgressClose(self.context) == .busy);
        }
        const value: *[1]f64 = @ptrCast(&self.nested_value);
        const factor = Shared.factorManaged(true, f64, i64, .high, self.context, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = value }) catch |err| {
            self.nested_error = err;
            return;
        };
        check(factor == .lapacke_info and factor.lapacke_info == 0);
        const bad = Shared.solveManaged(true, f64, i64, .work, self.context, .{ .layout = 102, .uplo = '?', .n = 0, .nrhs = 0, .lda = 1, .ldb = 1, .a = null, .b = null }) catch |err| {
            self.nested_error = err;
            return;
        };
        check(bad == .lapacke_info and bad.lapacke_info == -2);
    }
};

test "shared callback replacement and finite reentry keep outer snapshot and one operation per call" {
    defer reset();
    for ([_]bool{ false, true }) |close_first| {
        reset();
        var r: h.Rig = .{};
        r.init();
        defer r.done() catch @panic("cleanup");
        var callback = Reentrant{ .context = r.context, .close_first = close_first };
        _ = cfg.setCallbacks(.{ .fortran_returning_test = .{ .userdata = &callback, .function = Reentrant.original } });
        var a = [_]f64{4};
        var args = h.fargs(&a);
        args.uplo = '?';
        try h.info(i64, try Shared.factorManaged(true, f64, i64, .work, r.context, args), -2);
        try std.testing.expectEqual(@as(usize, 1), callback.old_calls);
        try std.testing.expectEqual(@as(usize, if (close_first) 0 else 1), callback.new_calls);
        try std.testing.expectEqual(@as(f64, if (close_first) 4 else 2), callback.nested_value);
        if (close_first) {
            try std.testing.expect(callback.nested_error != null);
            try std.testing.expectEqual(@as(anyerror, error.Closing), callback.nested_error.?);
        } else try std.testing.expect(callback.nested_error == null);
        try std.testing.expectEqual(@as(usize, if (close_first) 1 else 3), cfg.testing.snapshot().captures);
        try std.testing.expectEqual(@as(usize, if (close_first) 1 else 2), r.max_active);
        try std.testing.expectEqual(@as(usize, 1), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 1), r.x.frees);
    }
}

const PausingAllocator = struct {
    backing: h.Spy = .{},
    pause_free: bool,
    paused: std.atomic.Value(bool) = .init(false),
    proceed: std.atomic.Value(bool) = .init(false),
    fn allocator(self: *PausingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(p: *anyopaque) *PausingAllocator {
        return @ptrCast(@alignCast(p));
    }
    fn pause(self: *PausingAllocator) void {
        self.paused.store(true, .release);
        Core.Execution.test_support.wait(&self.proceed);
    }
    fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(p);
        if (!self.pause_free) self.pause();
        return self.backing.allocator().rawAlloc(n, a, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(p: *anyopaque, memory: []u8, a: std.mem.Alignment, ret: usize) void {
        const self = cast(p);
        if (self.pause_free) self.pause();
        self.backing.allocator().rawFree(memory, a, ret);
    }
};
const Worker = struct {
    context: *C.Context,
    value: [1]f64 = .{4},
    result: ?owner.Adapter.Result(i64) = null,
    err: ?anyerror = null,
    fn run(self: *Worker) void {
        var args = h.fargs(&self.value);
        args.uplo = '?';
        self.result = Shared.factorManaged(true, f64, i64, .work, self.context, args) catch |err| {
            self.err = err;
            return;
        };
    }
};

test "shared captured userdata survives replacement while real allocation or free pauses terminal close" {
    defer reset();
    for ([_]bool{ false, true }) |pause_free| {
        reset();
        var allocator = PausingAllocator{ .pause_free = pause_free };
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(allocator.allocator(), Core.Execution.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        var old = Events{ .kind = .fortran_returning_test };
        var new = Events{ .kind = .fortran_returning_test };
        _ = cfg.setCallbacks(.{ .fortran_returning_test = old.callback() });
        var worker = Worker{ .context = context };
        const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
        {
            defer {
                allocator.proceed.store(true, .release);
                thread.join();
            }
            Core.Execution.test_support.wait(&allocator.paused);
            const previous = cfg.setCallbacks(.{ .fortran_returning_test = new.callback() });
            try std.testing.expect(previous.fortran_returning_test.?.userdata == @as(*anyopaque, &old));
            const snapshot = context.snapshot();
            try std.testing.expectEqual(@as(usize, 1), snapshot.active_operations);
            try std.testing.expectEqual(@as(usize, 1), snapshot.resource_pins);
            try std.testing.expectEqual(C.RequestResult.pending, context.requestClose());
            try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(context));
            try std.testing.expectEqual(@as(usize, 1), cfg.testing.snapshot().captures);
        }
        // Old/new userdata are still alive here; replacing the tuple did not
        // authorize their reclamation while the worker was paused.
        try std.testing.expect(worker.err == null);
        try h.info(i64, worker.result.?, -2);
        try std.testing.expectEqual(@as(usize, 1), old.calls);
        try std.testing.expectEqual(@as(usize, 0), new.calls);
        try std.testing.expectEqual(@as(usize, 1), allocator.backing.attempts);
        try std.testing.expectEqual(@as(usize, 1), allocator.backing.frees);
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(context));
    }
}

const Concurrent = struct {
    const Item = struct { id: u8, calls: std.atomic.Value(usize) = .init(0) };
    a: Item = .{ .id = 1 },
    b: Item = .{ .id = 2 },
    start: std.atomic.Value(bool) = .init(false),
    const rounds = 2048;
    fn one(raw: *anyopaque, event: owner.common.ErrorEvent) void {
        const item: *Item = @ptrCast(@alignCast(raw));
        check(item.id == 1 and event.kind == .lapacke);
        _ = item.calls.fetchAdd(1, .monotonic);
    }
    fn two(raw: *anyopaque, event: owner.common.ErrorEvent) void {
        const item: *Item = @ptrCast(@alignCast(raw));
        check(item.id == 2 and event.kind == .lapacke);
        _ = item.calls.fetchAdd(1, .monotonic);
    }
    fn writer(self: *Concurrent) void {
        Core.Execution.test_support.wait(&self.start);
        for (0..rounds) |i| {
            _ = cfg.setCallbacks(.{ .lapacke = if (i % 2 == 0) .{ .userdata = &self.a, .function = one } else .{ .userdata = &self.b, .function = two } });
            cfg.setNancheck(@intCast(i % 2));
            std.Thread.yield() catch {};
        }
    }
    fn reader(self: *Concurrent) void {
        Core.Execution.test_support.wait(&self.start);
        for (0..rounds) |_| {
            const flag = cfg.getNancheck();
            check(flag == 0 or flag == 1);
            const result = Shared.factorDefault(true, f64, i32, .work, .{ .layout = 0, .uplo = 'L', .n = 0, .lda = 1, .values = null }) catch @panic("unexpected default admission failure");
            check(result == .lapacke_info and result.lapacke_info == -1);
            std.Thread.yield() catch {};
        }
    }
};

test "shared concurrent replacement never mixes handler and userdata records" {
    reset();
    defer reset();
    var state: Concurrent = .{};
    cfg.setNancheck(1);
    _ = cfg.setCallbacks(.{ .lapacke = .{ .userdata = &state.a, .function = Concurrent.one } });
    const writer = try std.Thread.spawn(.{}, Concurrent.writer, .{&state});
    const reader = std.Thread.spawn(.{}, Concurrent.reader, .{&state}) catch |err| {
        state.start.store(true, .release);
        writer.join();
        return err;
    };
    state.start.store(true, .release);
    writer.join();
    reader.join();
    try std.testing.expectEqual(@as(usize, Concurrent.rounds), state.a.calls.load(.acquire) + state.b.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, Concurrent.rounds), cfg.testing.snapshot().captures);
    try std.testing.expectEqual(@as(usize, Concurrent.rounds), cfg.testing.snapshot().getter_calls);
    try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
}

test "shared managed engines match original managed full padded real and complex buffers" {
    reset();
    defer reset();
    cfg.setNancheck(1);
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| {
            var r: h.Rig = .{};
            r.init();
            defer r.done() catch @panic("cleanup");
            var a: [6]T = @splat(s.value(T, 71, -3));
            a[s.pos(layout, 0, 0, 3)] = s.value(T, 4, 0);
            a[s.pos(layout, 1, 1, 3)] = s.value(T, 10, 0);
            a[s.pos(layout, if (uplo == 'L') 1 else 0, if (uplo == 'L') 0 else 1, 3)] = s.value(T, 2, if (Core.Scalar.isComplex(T)) 2 else 0);
            var other = a;
            const fa: h.F.Arguments(T, I) = .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &a };
            var fb = fa;
            fb.values = &other;
            try h.info(I, try h.F.callManaged(T, I, .high, r.context, &r.services, fa), 0);
            try h.info(I, try Shared.factorManaged(true, T, I, .high, r.context, fb), 0);
            try s.bytes(&a, &other);
            var rhs: [6]T = @splat(s.value(T, 83, -7));
            rhs[s.pos(layout, 0, 0, 3)] = s.value(T, 10, 2);
            rhs[s.pos(layout, 1, 0, 3)] = s.value(T, 20, -1);
            rhs[s.pos(layout, 0, 1, 3)] = s.value(T, -6, 1);
            rhs[s.pos(layout, 1, 1, 3)] = s.value(T, 12, 3);
            var rhs2 = rhs;
            try h.info(I, try h.S.callManaged(T, I, .work, r.context, &r.services, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &a, .b = &rhs }), 0);
            try h.info(I, try Shared.solveManaged(true, T, I, .work, r.context, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &other, .b = &rhs2 }), 0);
            try s.bytes(&rhs, &rhs2);
            try s.bytes(&a, &other);
            try std.testing.expectEqual(@as(usize, 0), r.y.attempts);
        };
    };
}

const ProfileDefaultsRecorder = struct {
    calls: usize = 0,
    reenter: bool = false,
    fn call(raw: *anyopaque, _: owner.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.calls += 1;
        if (self.reenter) _ = cfg.captureCallbacks();
    }
    fn callback(self: *@This()) cfg.Callback {
        return .{ .userdata = self, .function = call };
    }
};
const profile_defaults_lapacke_event: owner.common.ErrorEvent = .{ .kind = .lapacke, .entry = .high, .scalar = 'd', .argument = -1 };
const profile_defaults_fortran_event: owner.common.ErrorEvent = .{ .kind = .fortran_returning_test, .entry = .fortran_potrf, .scalar = 'd', .argument = 1 };

test "profile defaults are local, captured once and do not initialize NaN configuration" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    var profile: ProfileDefaultsRecorder = .{};
    const snapshot = cfg.captureCallbacksWithDefaults(.{ .lapacke = profile.callback() });
    const state = cfg.testing.snapshot();
    try std.testing.expectEqual(@as(usize, 1), state.captures);
    try std.testing.expect(!state.initialized);
    try std.testing.expectEqual(@as(usize, 0), state.environment_reads);
    try std.testing.expect(cfg.captureCallbacks().lapacke == null);
    snapshot.dispatch(profile_defaults_lapacke_event);
    snapshot.dispatch(profile_defaults_fortran_event);
    try std.testing.expectEqual(@as(usize, 1), profile.calls);
}

test "configured channel wins and fallback snapshots survive replacement and reentry" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    var configured: ProfileDefaultsRecorder = .{ .reenter = true };
    var fallback: ProfileDefaultsRecorder = .{};
    var replacement: ProfileDefaultsRecorder = .{};
    _ = cfg.setCallbacks(.{ .lapacke = configured.callback() });
    const captured = cfg.captureCallbacksWithDefaults(.{ .lapacke = fallback.callback(), .fortran_returning_test = fallback.callback() });
    _ = cfg.setCallbacks(.{ .lapacke = replacement.callback(), .fortran_returning_test = replacement.callback() });
    captured.dispatch(profile_defaults_lapacke_event);
    captured.dispatch(profile_defaults_fortran_event);
    try std.testing.expectEqual(@as(usize, 1), configured.calls);
    try std.testing.expectEqual(@as(usize, 1), fallback.calls);
    try std.testing.expectEqual(@as(usize, 0), replacement.calls);
    const current = cfg.captureCallbacks();
    current.dispatch(profile_defaults_lapacke_event);
    current.dispatch(profile_defaults_fortran_event);
    try std.testing.expectEqual(@as(usize, 2), replacement.calls);
}

test "two profiles with null shared configuration keep independent defaults" {
    cfg.testing.reset();
    defer cfg.testing.reset();
    var first: ProfileDefaultsRecorder = .{};
    var second: ProfileDefaultsRecorder = .{};
    const a = cfg.captureCallbacksWithDefaults(.{ .lapacke = first.callback() });
    const b = cfg.captureCallbacksWithDefaults(.{ .lapacke = second.callback() });
    b.dispatch(profile_defaults_lapacke_event);
    a.dispatch(profile_defaults_lapacke_event);
    a.dispatch(profile_defaults_lapacke_event);
    try std.testing.expectEqual(@as(usize, 2), first.calls);
    try std.testing.expectEqual(@as(usize, 1), second.calls);
    try std.testing.expect(cfg.captureCallbacks().lapacke == null);
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
fn fixtureExecutor(ctx: *@import("zynum-execution").Context) *@import("zynum-execution").reference.owner.Executor {
    const fixture_execution = @import("zynum-execution");
    if (ctx == fixture_execution.reference.getContext()) return @ptrCast(@alignCast(ctx.binding.legacy.userdata));
    const wrapper: *FixtureReferenceContext = @fieldParentPtr("context", ctx);
    return &wrapper.executor;
}
