const std = @import("std");
const Core = @import("zynum-core");
const facade = Core.internal.numerics.FFT;
const Execution = Core.internal.numerics.Execution;
const context = Execution.context;
const fixture = Execution.test_support;
const M = facade.Managed(Execution);
const C = Core.internal.numerics.Scalar.ComplexF64;
const E = std.testing;
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
            if (fixtureProgressClose(self.owner) != .busy) @panic("FFT allocation is not admitted");
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
        if (fixtureProgressClose(self.owner) == .closed) @panic("FFT close passed managed free");
        self.backing.allocator().rawFree(data, a, ret);
    }
};
fn finish(owner: *context.Context) !void {
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.closed, fixtureProgressClose(owner));
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
test "FFT scratch pins the exact owner before callbacks and frees before ending one operation" {
    var probe: Probe = .{ .close_on_alloc = true };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var reference_other: FixtureReferenceContext = undefined;
    reference_other.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other = reference_other.borrowedContext();
    const plan = try facade.core.ComplexPlan.init(3, .forward);
    const input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = 0 } };
    var output: [3]C = undefined;
    try M.runScratch(.complex, C, owner, plan, &input, &output);
    try E.expectApproxEqAbs(@as(f64, 6), output[0].re, 1e-12);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.resource_pins);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), other.snapshot().active_operations);
    try E.expectEqual(@as(usize, 1), probe.backing.stats().freed);
    try E.expectError(error.Closing, M.runScratch(.complex, C, owner, plan, &input, &output));
    try finish(owner);
    try finish(other);
}
test "FFT allocation and numerical failures unwind the resource and preserve output" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    const plan = try facade.core.ComplexPlan.init(3, .forward);
    var input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = 0 } };
    var output: [3]C = @splat(.{ .re = 123, .im = 456 });
    try E.expectError(error.OutOfMemory, M.runScratch(.complex, C, owner, plan, &input, &output));
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try finish(owner);
    var allocated: Probe = .{};
    var reference_good: FixtureReferenceContext = undefined;
    reference_good.initInPlace(allocated.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const good = reference_good.borrowedContext();
    allocated.owner = good;
    input[0].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, M.runScratch(.complex, C, good, plan, &input, &output));
    try E.expectEqual(@as(usize, 1), allocated.backing.stats().freed);
    for (output) |x| try E.expectEqual(@as(f64, 123), x.re);
    try E.expectEqual(@as(usize, 0), good.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), good.snapshot().resource_pins);
    try finish(good);
}
test "FFT admitted workspace remains usable after close request without nested admission or pin" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var op = try owner.begin();
    _ = owner.requestClose();
    const plan = try facade.core.ComplexPlan.init(2, .forward);
    const input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } };
    var output: [2]C = undefined;
    var work: [2]facade.core.WorkspaceScalar(C) = undefined;
    try M.runBorrowedAdmitted(.complex, C, &op, plan, &input, &output, &work);
    try E.expectEqual(@as(f64, 3), output[0].re);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    try E.expectError(error.Closing, M.runBorrowed(.complex, C, owner, plan, &input, &output, &work));
    op.end();
    try finish(owner);
}
test "FFT managed facade dispatches all eight transform actions through one shared context" {
    inline for (comptime std.meta.tags(facade.Action)) |action| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        const plan: facade.Plan(action) = switch (action) {
            .complex => try facade.core.ComplexPlan.init(3, .forward),
            .real_forward, .real_backward => try facade.core.RealPlan.init(3),
            .cosine => try facade.core.CosinePlan.init(3),
            .real_transform => try facade.core.RealTransformPlan.init(3, .cosine_ii),
            .tensor_real_transform => try facade.core.TensorRealTransformPlan.init(&.{ 1, 1, 3 }, &.{ .hartley, .cosine_ii, .sine_ii }),
            .tensor_forward, .tensor_backward, .tensor_real_forward, .tensor_real_backward => try facade.core.TensorPlan.init(&.{ 1, 1, 3 }),
            else => try facade.core.VolumePlan.init(.{ 1, 1, 3 }),
        };
        var input: [3]facade.Input(action, C) = undefined;
        for (&input) |*v| v.* = if (@TypeOf(v.*) == C) .{ .re = 1, .im = 0 } else 1;
        var output: [3]facade.Output(action, C) = undefined;
        try M.runScratch(action, C, owner, plan, &input, &output);
        try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try finish(owner);
    }
}
test "FFT admission and resource counter exhaustion preserve existing caller lifetimes" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    owner.test_state.limit = 0;
    const plan = try facade.core.ComplexPlan.init(2, .forward);
    const input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } };
    var output: [2]C = @splat(.{ .re = 123, .im = 456 });
    try E.expectError(error.CounterOverflow, M.runScratch(.complex, C, owner, plan, &input, &output));
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    owner.test_state.limit = 1;
    var op = try owner.begin();
    var existing = try op.pinResource();
    try E.expectError(error.CounterOverflow, M.runScratchAdmitted(.complex, C, &op, plan, &input, &output));
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    for (output) |x| try E.expectEqual(@as(f64, 123), x.re);
    existing.releaseAfterResourceFree();
    _ = owner.requestClose();
    try E.expectError(error.OutOfMemory, M.runScratchAdmitted(.complex, C, &op, plan, &input, &output));
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try finish(owner);
}

test "FFT facade borrows the actual Core scalar execution and default identities" {
    const actual = @import("zynum-execution");
    comptime {
        if (Execution != actual or context.Operation != actual.context.Operation) @compileError("FFT execution owner mismatch");
    }
    try E.expect(Execution.getContext() == actual.getContext());
    try E.expectEqual(@import("zynum-execution").reference.LegacyLifetime.immortal_default, actual.getContext().binding.legacy.lifetime);
    const p = try facade.core.ComplexPlan.init(1, .forward);
    const x = [_]C{.{ .re = 2, .im = 3 }};
    var y: [1]C = undefined;
    var work: [1]facade.core.WorkspaceScalar(C) = undefined;
    try M.runBorrowed(.complex, C, actual.getContext(), p, &x, &y, &work);
    try E.expectEqual(@as(f64, 2), y[0].re);
    try E.expectEqual(@as(usize, 0), actual.getContext().snapshot().active_operations);
}

const D = facade.Persistent(Execution, C);
test "FFT persistent metadata owns one original ContextScratch pin across operations" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var descriptor = try D.create(owner, .{ .complex_1d = 3 });
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    const input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = 0 } };
    var output: [3]C = @splat(.{ .re = 999, .im = 3 });
    try E.expectError(error.NotCommitted, descriptor.runScratch(.complex_forward, &input, &output));
    try E.expectEqual(@as(usize, 1), probe.backing.stats().attempts);
    try descriptor.commit();
    try descriptor.runScratch(.complex_forward, &input, &output);
    try E.expectApproxEqAbs(@as(f64, 6), output[0].re, 1e-12);
    var inverse: [3]C = undefined;
    try descriptor.runScratch(.complex_backward, &output, &inverse);
    for (input, inverse) |x, y| {
        try E.expectApproxEqAbs(3 * x.re, y.re, 1e-11);
        try E.expectApproxEqAbs(@as(f64, 0), y.im, 1e-11);
    }
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
    descriptor.deinit();
    try E.expectEqual(@as(usize, 3), probe.backing.stats().freed);
    try finish(owner);
}
test "FFT descriptor reconfiguration is transactional and invalidates successful commit" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(E.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var descriptor = try D.create(owner, .{ .complex_1d = 2 });
    try descriptor.commit();
    var op = try owner.begin();
    try E.expectError(error.InvalidLength, descriptor.reconfigureAdmitted(&op, .{ .real_1d = 0 }));
    try E.expectError(error.SizeOverflow, descriptor.reconfigureAdmitted(&op, .{ .complex_3d = .{ std.math.maxInt(usize), 2, 3 } }));
    try E.expect(try descriptor.committedAdmitted(&op));
    try E.expectEqual(@as(usize, 2), (try descriptor.geometryAdmitted(&op)).complex_1d);
    try descriptor.reconfigureAdmitted(&op, .{ .real_1d = 3 });
    try E.expect(!try descriptor.committedAdmitted(&op));
    var x = [_]f64{ 1, 1, 1 };
    var y: [2]C = undefined;
    var work: [32]facade.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.NotCommitted, descriptor.runBorrowedAdmitted(.real_forward, &op, &x, &y, &work));
    try descriptor.commitAdmitted(&op);
    try descriptor.commitAdmitted(&op);
    try descriptor.runBorrowedAdmitted(.real_forward, &op, &x, &y, &work);
    try E.expectApproxEqAbs(@as(f64, 3), y[0].re, 1e-12);
    var cx: [3]C = @splat(.{ .re = 1, .im = 0 });
    try E.expectError(error.WrongTransform, descriptor.runBorrowedAdmitted(.complex_forward, &op, &cx, &y, &work));
    op.end();
    descriptor.deinit();
    try finish(owner);
}
test "FFT descriptor invalid geometry allocation rejection and pin exhaustion unwind independently" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    try E.expectError(error.InvalidLength, D.create(owner, .{ .cosine_1d = 1 }));
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    try E.expectError(error.OutOfMemory, D.create(owner, .{ .complex_1d = 2 }));
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    owner.test_state.limit = 1;
    var op = try owner.begin();
    var pin = try op.pinResource();
    try E.expectError(error.CounterOverflow, D.createAdmitted(&op, .{ .complex_1d = 2 }));
    try E.expectEqual(@as(usize, 1), probe.backing.stats().attempts);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    pin.releaseAfterResourceFree();
    op.end();
    owner.test_state.limit = 0;
    try E.expectError(error.CounterOverflow, D.create(owner, .{ .complex_1d = 2 }));
    try finish(owner);
}
test "FFT descriptor callback close preserves admitted execution and frees metadata before its pin" {
    var probe: Probe = .{ .close_on_alloc = true };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var op = try owner.begin();
    var descriptor = try D.createAdmitted(&op, .{ .complex_1d = 2 });
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.resource_pins);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try descriptor.commitAdmitted(&op);
    var input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } };
    var output: [2]C = undefined;
    var work: [2]facade.core.WorkspaceScalar(C) = undefined;
    try descriptor.runBorrowedAdmitted(.complex_forward, &op, &input, &output, &work);
    try E.expectEqual(@as(f64, 3), output[0].re);
    try E.expectEqual(@as(usize, 1), probe.backing.stats().attempts);
    try E.expectError(error.Closing, descriptor.commit());
    try E.expectError(error.Closing, descriptor.runBorrowed(.complex_forward, &input, &output, &work));
    op.end();
    try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
    descriptor.deinit();
    try E.expectEqual(@as(usize, 0), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.resource_pins);
    try finish(owner);
}
test "FFT descriptor rejects another owner and retains its captured metadata allocator" {
    var first: Probe = .{};
    var second: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(first.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var reference_other: FixtureReferenceContext = undefined;
    reference_other.initInPlace(second.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other = reference_other.borrowedContext();
    first.owner = owner;
    second.owner = other;
    var descriptor = try D.create(owner, .{ .complex_1d = 2 });
    var wrong = try other.begin();
    try E.expectError(error.ContextMismatch, descriptor.commitAdmitted(&wrong));
    try E.expectError(error.ContextMismatch, descriptor.reconfigureAdmitted(&wrong, .{ .complex_1d = 3 }));
    try E.expectError(error.ContextMismatch, descriptor.geometryAdmitted(&wrong));
    var input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } };
    var output: [2]C = undefined;
    var work: [2]facade.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.ContextMismatch, descriptor.runBorrowedAdmitted(.complex_forward, &wrong, &input, &output, &work));
    wrong.end();
    try descriptor.commit();
    // Deliberate fixture mutation verifies every allocation's captured owner.
    second.owner = owner;
    fixtureReplaceAllocator(&reference_owner, second.allocator());
    try descriptor.runScratch(.complex_forward, &input, &output);
    try E.expectEqual(@as(usize, 1), second.backing.stats().freed);
    _ = owner.requestClose();
    descriptor.deinit();
    try E.expectEqual(@as(usize, 1), first.backing.stats().freed);
    try E.expectEqual(@as(usize, 1), first.free_snapshot.?.resource_pins);
    try finish(owner);
    try finish(other);
}
test "FFT descriptor dispatches all nine directions and both complex precisions with preserved tails" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Scalar| {
        inline for (comptime std.meta.tags(facade.Transform)) |transform| {
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(E.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            const geometry: facade.Geometry = switch (transform) {
                .complex_forward, .complex_backward => .{ .complex_1d = 3 },
                .real_forward, .real_backward => .{ .real_1d = 3 },
                .cosine => .{ .cosine_1d = 3 },
                .volume_forward, .volume_backward => .{ .complex_3d = .{ 1, 1, 3 } },
                .volume_real_forward, .volume_real_backward => .{ .real_3d = .{ 1, 1, 3 } },
            };
            var descriptor = try facade.Persistent(Execution, Scalar).create(owner, geometry);
            try descriptor.commit();
            const In = facade.Input(facade.actionFor(transform), Scalar);
            const Out = facade.Output(facade.actionFor(transform), Scalar);
            var input: [3]In = undefined;
            for (&input) |*v| v.* = if (In == Scalar) .{ .re = 1, .im = 0 } else 1;
            var output: [4]Out = undefined;
            for (&output) |*v| v.* = if (Out == Scalar) .{ .re = 999, .im = 333 } else 999;
            try descriptor.runScratch(transform, &input, &output);
            const R = if (Scalar == C) f64 else f32;
            const dc: R = if (transform == .cosine) 4 else 3;
            const actual: R = if (Out == Scalar) output[0].re else output[0];
            try E.expectApproxEqAbs(dc, actual, if (R == f64) 1e-11 else 1e-5);
            try E.expectEqual(@as(R, 999), if (Out == Scalar) output[3].re else output[3]);
            try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
            try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
            descriptor.deinit();
            try finish(owner);
        }
    }
}
test "FFT descriptor protects heap metadata and affine lease bookkeeping from buffer writes" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(E.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var descriptor: D align(16) = try D.create(owner, .{ .complex_1d = 1 });
    try descriptor.commit();
    var op = try owner.begin();
    const input = [_]C{.{ .re = 2, .im = 3 }};
    var output: [1]C = @splat(.{ .re = 999, .im = 0 });
    var work: [1]facade.core.WorkspaceScalar(C) = undefined;
    const metadata: [*]C = @ptrCast(descriptor.resource.lease.values.ptr);
    const shell: [*]C = @ptrCast(&descriptor);
    const shell_work: [*]facade.core.WorkspaceScalar(C) = @ptrCast(&descriptor);
    try E.expectError(error.AliasingNotAllowed, descriptor.runBorrowedAdmitted(.complex_forward, &op, &input, metadata[0..1], &work));
    try E.expectError(error.AliasingNotAllowed, descriptor.runBorrowedAdmitted(.complex_forward, &op, metadata[0..1], &output, &work));
    try E.expectError(error.AliasingNotAllowed, descriptor.runBorrowedAdmitted(.complex_forward, &op, &input, shell[0..1], &work));
    try E.expectError(error.AliasingNotAllowed, descriptor.runBorrowedAdmitted(.complex_forward, &op, &input, &output, shell_work[0..1]));
    try E.expect(try descriptor.committedAdmitted(&op));
    try descriptor.runBorrowedAdmitted(.complex_forward, &op, &input, &output, &work);
    try E.expectEqual(@as(f64, 2), output[0].re);
    op.end();
    descriptor.deinit();
    try finish(owner);
}
test "FFT descriptor failed execution preserves metadata output tails and workspace resource balance" {
    var probe: Probe = .{ .backing = .{ .budget = 2 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var descriptor = try D.create(owner, .{ .complex_1d = 2 });
    try descriptor.commit();
    var input = [_]C{ .{ .re = std.math.nan(f64), .im = 0 }, .{ .re = 2, .im = 0 } };
    var output: [3]C = @splat(.{ .re = 999, .im = 33 });
    try E.expectError(error.NonFiniteInput, descriptor.runScratch(.complex_forward, &input, &output));
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 2), probe.free_snapshot.?.resource_pins);
    input[0].re = 1;
    try E.expectError(error.OutOfMemory, descriptor.runScratch(.complex_forward, &input, &output));
    for (output) |v| try E.expectEqual(@as(f64, 999), v.re);
    var op = try owner.begin();
    try E.expect(try descriptor.committedAdmitted(&op));
    owner.test_state.limit = 1;
    try E.expectError(error.CounterOverflow, descriptor.runScratchAdmitted(.complex_forward, &op, &input, &output));
    op.end();
    descriptor.deinit();
    try finish(owner);
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}

// These serial FFT fixtures rebind only before the reference Executor owns
// worker/Io metadata. Keep the original per-allocation provenance assertions.
fn fixtureReplaceAllocator(wrapper: *FixtureReferenceContext, allocator: std.mem.Allocator) void {
    if (wrapper.executor.io_init_state.load(.acquire) != 0 or
        wrapper.executor.persistent_init_state.load(.acquire) != 0)
        @panic("FFT allocator fixture requires an unused reference Executor");
    wrapper.executor.allocator = allocator;
}
