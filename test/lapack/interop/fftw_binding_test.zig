const std = @import("std");
const Core = @import("zynum-core");
const B = @import("fftw-interface-owner");
const Execution = Core.internal.numerics.Execution;
const context = Execution.context;
const fixture = Execution.test_support;
const F = Core.internal.numerics.FFT;
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
test "FFTW exact Core plan owner and per-call workspace retain captured cleanup ordering" {
    try E.expect(B.Provider.OwnerIdentity == Core);
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [4]C = @splat(.{ .re = 1, .im = 0 });
    var output: [5]C = @splat(.{ .re = 999, .im = 999 });
    const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.resource_pins);
    try B.executeWithArrays(h, &input, &output);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 2), probe.free_snapshot.?.resource_pins);
    try E.expectApproxEqAbs(@as(f64, 4), output[0].re, 1e-12);
    try E.expectEqual(@as(f64, 999), output[4].re);
    var other: Probe = .{};
    other.owner = owner;
    fixtureReplaceAllocator(&reference_owner, other.allocator());
    try B.executeWithArrays(h, &input, &output);
    try E.expectEqual(@as(usize, 1), other.backing.stats().freed);
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
    B.destroy(h);
    try E.expectEqual(@as(usize, 0), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.resource_pins);
    try E.expectEqual(@as(usize, 2), probe.backing.stats().freed);
    try finish(owner);
}
test "FFTW callback close during plan or workspace acquisition completes its admitted work" {
    inline for (.{ false, true }) |close_during_execute| {
        var probe: Probe = .{ .close_on_alloc = !close_during_execute };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [4]C = @splat(.{ .re = 1, .im = 0 });
        var output: [4]C = @splat(.{ .re = 999, .im = 999 });
        const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
        if (close_during_execute) {
            probe.close_on_alloc = true;
            try B.executeWithArrays(h, &input, &output);
            try E.expectApproxEqAbs(@as(f64, 4), output[0].re, 1e-12);
        }
        try E.expectError(error.Closing, B.executeWithArrays(h, &input, &output));
        B.destroy(h);
        try finish(owner);
    }
}
test "FFTW failed planning execution allocation and capacity retain output and pins" {
    inline for (.{ @as(usize, 0), 1 }) |budget| {
        var probe: Probe = .{ .backing = .{ .budget = budget } };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [4]C = @splat(.{ .re = 1, .im = 0 });
        var output: [4]C = @splat(.{ .re = 999, .im = 999 });
        const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64);
        if (h) |live| {
            try E.expectError(error.OutOfMemory, B.executeWithArrays(live, &input, &output));
            try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
            B.destroy(live);
        } else try E.expectEqual(@as(usize, 0), budget);
        try E.expectEqual(@as(f64, 999), output[0].re);
        try finish(owner);
    }
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [4]C = @splat(.{ .re = 1, .im = 0 });
    var output: [4]C = @splat(.{ .re = 999, .im = 999 });
    context.Context.testing.configure(owner, 0, null);
    try E.expect(B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64) == null);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    context.Context.testing.configure(owner, std.math.maxInt(usize), null);
    try E.expect(B.createWithContext(owner, .volume_real_forward, .{ std.math.maxInt(usize), 2, 2 }, &input, &output, -1, 64) == null);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
    try E.expectError(error.AliasingNotAllowed, B.executeWithArrays(h, &input, @ptrCast(h)));
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    B.destroy(h);
    try finish(owner);
}
test "FFTW shared buffer alignment overflow failure and teardown after close preserve lifetime" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    try E.expect(B.allocateWithContext(owner, std.math.maxInt(usize)) == null);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    const buffer = B.allocateWithContext(owner, 137).?;
    try E.expectEqual(@as(usize, 0), @intFromPtr(buffer) % 64);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
    B.free(buffer);
    B.free(null);
    try E.expectEqual(@as(usize, 0), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.resource_pins);
    try finish(owner);
}
test "FFTW IEEE semantic policy shares all leaves without weakening checked callers" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        inline for (comptime std.meta.tags(F.Action)) |action| {
            const I = F.Input(action, Z);
            const O = F.Output(action, Z);
            const p = switch (action) {
                .complex => try F.core.ComplexPlan.init(4, .forward),
                .real_forward, .real_backward => try F.core.RealPlan.init(4),
                .cosine => try F.core.CosinePlan.init(4),
                .real_transform => try F.core.RealTransformPlan.init(4, .cosine_ii),
                .tensor_real_transform => try F.core.TensorRealTransformPlan.init(&.{ 2, 1, 2 }, &.{ .cosine_ii, .hartley, .sine_ii }),
                .tensor_forward, .tensor_backward, .tensor_real_forward, .tensor_real_backward => try F.core.TensorPlan.init(&.{ 2, 1, 2 }),
                else => try F.core.VolumePlan.init(.{ 2, 1, 2 }),
            };
            var input: [8]I = @splat(if (I == Z) .{ .re = 1, .im = 0 } else 1);
            var output: [8]O = @splat(if (O == Z) .{ .re = 777, .im = 777 } else 777);
            input[0] = if (I == Z) .{ .re = std.math.nan(Core.internal.numerics.Scalar.Real(Z)), .im = 0 } else std.math.nan(I);
            var work: [256]F.core.WorkspaceScalar(Z) = undefined;
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(E.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            var operation = try owner.begin();
            try E.expectError(error.NonFiniteInput, F.Managed(Execution).runBorrowedAdmitted(action, Z, &operation, p, &input, &output, &work));
            try E.expectEqual(@as(Core.internal.numerics.Scalar.Real(Z), 777), if (O == Z) output[0].re else output[0]);
            try F.Managed(Execution).runBorrowedIEEEAdmitted(action, Z, &operation, p, &input, &output, &work);
            try E.expect(!std.math.isFinite(if (O == Z) output[0].re else output[0]));
            try E.expectEqual(@as(Core.internal.numerics.Scalar.Real(Z), 777), if (O == Z) output[7].re else output[7]);
            operation.end();
            try finish(owner);
        }
    }
}
test "FFTW finite final overflow propagates through IEEE while checked output is transactional" {
    var input: [2]C = @splat(.{ .re = std.math.floatMax(f64), .im = 0 });
    var output: [3]C = @splat(.{ .re = 777, .im = 777 });
    var work: [2]F.core.WorkspaceScalar(C) = undefined;
    const p = try F.core.ComplexPlan.init(2, .forward);
    try E.expectError(error.NonFiniteResult, p.execute(C, &input, &output, &work));
    try E.expectEqual(@as(f64, 777), output[0].re);
    try p.executePolicy(C, &input, &output, &work, .ieee);
    try E.expect(std.math.isPositiveInf(output[0].re));
    try E.expectEqual(@as(f64, 777), output[2].re);
}
test "FFTW same immutable plan concurrently executes on independent arrays through one Core" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.heap.c_allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var input: [13]C = @splat(.{ .re = 1, .im = 0 });
    var output: [13]C = undefined;
    const h = B.createWithContext(owner, .complex, .{ 13, 1, 1 }, &input, &output, -1, 64).?;
    const Worker = struct {
        handle: B.Handle,
        value: f64,
        failed: bool = false,
        fn run(self: *@This()) void {
            var local_in: [13]C = @splat(.{ .re = self.value, .im = 0 });
            var local_out: [14]C = @splat(.{ .re = 777, .im = 777 });
            for (0..8) |_| {
                B.executeNewArrays(self.handle, .complex, &local_in, &local_out) catch {
                    self.failed = true;
                    return;
                };
                if (@abs(local_out[0].re - 13 * self.value) > 1e-10 or local_out[13].re != 777) self.failed = true;
                for (local_out[1..13]) |x| if (@abs(x.re) > 1e-10 or @abs(x.im) > 1e-10) {
                    self.failed = true;
                };
            }
        }
    };
    var workers: [6]Worker = undefined;
    var threads: [6]std.Thread = undefined;
    for (&workers, &threads, 0..) |*worker, *thread, i| {
        worker.* = .{ .handle = h, .value = @as(f64, @floatFromInt(i + 1)) };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    for (threads) |thread| thread.join();
    for (workers) |worker| try E.expect(!worker.failed);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    B.destroy(h);
    try finish(owner);
}

test "FFTW admitted arrays reuse one operation after close and reject a foreign owner" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [4]C = @splat(.{ .re = 1, .im = 0 });
    var output: [5]C = @splat(.{ .re = 999, .im = 999 });
    const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
    var foreign_probe: Probe = .{};
    var reference_foreign: FixtureReferenceContext = undefined;
    reference_foreign.initInPlace(foreign_probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const foreign = reference_foreign.borrowedContext();
    foreign_probe.owner = foreign;
    var wrong = try foreign.begin();
    const attempts = probe.backing.stats().attempts;
    try E.expectError(error.ContextMismatch, B.executeWithArraysAdmitted(h, &wrong, &input, &output));
    try E.expectEqual(attempts, probe.backing.stats().attempts);
    try E.expectEqual(@as(f64, 999), output[0].re);
    wrong.end();
    try finish(foreign);
    var operation = try owner.begin();
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.busy, fixtureProgressClose(owner));
    try E.expectError(error.Closing, B.executeWithArrays(h, &input, &output));
    try B.executeWithArraysAdmitted(h, &operation, &input, &output);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(f64, 4), output[0].re);
    try E.expectEqual(@as(f64, 999), output[4].re);
    B.destroy(h);
    try E.expectEqual(context.CloseResult.busy, fixtureProgressClose(owner));
    operation.end();
    try finish(owner);
}
test "FFTW admitted allocation failure preserves output operation and plan for retry" {
    var probe: Probe = .{ .backing = .{ .budget = 1 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [4]C = @splat(.{ .re = 1, .im = 0 });
    var output: [4]C = @splat(.{ .re = 999, .im = 999 });
    const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
    var operation = try owner.begin();
    try E.expectError(error.OutOfMemory, B.executeWithArraysAdmitted(h, &operation, &input, &output));
    try E.expectEqual(@as(f64, 999), output[0].re);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    probe.backing.budget = 10;
    try B.executeWithArraysAdmitted(h, &operation, &input, &output);
    try E.expectEqual(@as(f64, 4), output[0].re);
    operation.end();
    B.destroy(h);
    try finish(owner);
}

test "FFTW admitted real volume roundtrip preserves physical padding and guards" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var storage = [_]f64{ 1, 2, 3, 999, 4, 5, 6, 999, 777, 888 };
    const forward = B.createWithContext(owner, .volume_real_forward, .{ 2, 1, 3 }, &storage, &storage, -1, 64).?;
    const backward = B.createWithContext(owner, .volume_real_backward, .{ 2, 1, 3 }, &storage, &storage, 1, 64).?;
    var operation = try owner.begin();
    try B.executeWithArraysAdmitted(forward, &operation, &storage, &storage);
    for (0..2) |a| {
        for (0..2) |b| {
            var expected: C = .{ .re = 0, .im = 0 };
            for (0..2) |x| {
                for (0..3) |z| {
                    const phase = -2 * std.math.pi * (@as(f64, @floatFromInt(a * x)) / 2 + @as(f64, @floatFromInt(b * z)) / 3);
                    const value: f64 = @floatFromInt(x * 3 + z + 1);
                    expected.re += value * @cos(phase);
                    expected.im += value * @sin(phase);
                }
            }
            try E.expectApproxEqAbs(expected.re, storage[(a * 2 + b) * 2], 1e-11);
            try E.expectApproxEqAbs(expected.im, storage[(a * 2 + b) * 2 + 1], 1e-11);
        }
    }
    try B.executeWithArraysAdmitted(backward, &operation, &storage, &storage);
    for (0..2) |row| {
        for (0..3) |col| {
            const expected: f64 = @floatFromInt((row * 3 + col + 1) * 6);
            try E.expectApproxEqAbs(expected, storage[row * 4 + col], 1e-10);
        }
    }
    try E.expectEqual(@as(f64, 777), storage[8]);
    try E.expectEqual(@as(f64, 888), storage[9]);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    operation.end();
    B.destroy(forward);
    B.destroy(backward);
    try finish(owner);
}

test "Real transform all eleven kinds preserve guards and inverse normalization for both scalar families" {
    const inverse = [_]F.core.RealTransform{ .halfcomplex_backward, .halfcomplex_forward, .hartley, .cosine_i, .cosine_ii, .cosine_iii, .cosine_iv, .sine_i, .sine_ii, .sine_iii, .sine_iv };
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const eps: R = if (Z == C) 2e-10 else 2e-4;
        inline for (std.meta.tags(F.core.RealTransform), 0..) |kind, ki| {
            for ([_]usize{ 1, 2, 3, 4, 5, 8, 11 }) |n| {
                if (kind == .cosine_i and n == 1) continue;
                const plan = try F.core.RealTransformPlan.init(n, kind);
                const back = try F.core.RealTransformPlan.init(n, inverse[ki]);
                var input: [12]R = undefined;
                var result: [12]R = undefined;
                var output: [12]R = undefined;
                var work: [512]F.core.WorkspaceScalar(Z) = undefined;
                for (input[0..n], 0..) |*x, j| x.* = @floatCast(@sin(@as(f64, @floatFromInt(j)) * 0.7) + 0.2);
                result[n] = 777;
                output[n] = 888;
                work[try plan.workspaceLength(Z)] = .{ .re = 555, .im = 666 };
                try plan.execute(Z, input[0..n], &result, &work);
                try E.expectEqual(@as(R, 777), result[n]);
                try E.expectEqual(@as(@TypeOf(work[0].re), 555), work[try plan.workspaceLength(Z)].re);
                try back.execute(Z, result[0..n], &output, &work);
                const scale: R = @floatFromInt(try plan.logicalLength());
                for (output[0..n], input[0..n]) |y, x| try E.expectApproxEqAbs(x * scale, y, eps);
                try E.expectEqual(@as(R, 888), output[n]);
                try plan.execute(Z, input[0..n], input[0..n], &work);
                for (input[0..n], result[0..n]) |x, y| try E.expectApproxEqAbs(y, x, eps);
            }
        }
    }
}

test "Real transform checked boundaries and late overflow preserve destination" {
    var input = [_]f64{ 1, 2, 3, 4 };
    var output = [_]f64{ 77, 88, 99, 111 };
    const saved = output;
    var work: [4]F.core.WorkspaceScalar(C) = undefined;
    const plan = try F.core.RealTransformPlan.init(4, .cosine_ii);
    try E.expectError(error.InvalidLength, F.core.RealTransformPlan.init(0, .hartley));
    try E.expectError(error.InvalidLength, F.core.RealTransformPlan.init(1, .cosine_i));
    try E.expectError(error.SizeOverflow, F.core.RealTransformPlan.init(std.math.maxInt(usize), .sine_i));
    try E.expectError(error.BufferTooSmall, plan.execute(C, &input, &output, work[0..3]));
    try E.expectEqualDeep(saved, output);
    var overlap = [_]f64{ 1, 2, 3, 4, 5 };
    try E.expectError(error.AliasingNotAllowed, plan.execute(C, overlap[0..4], overlap[1..5], &work));
    input[2] = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, plan.execute(C, &input, &output, &work));
    try E.expectEqualDeep(saved, output);
    input = @splat(std.math.floatMax(f64));
    try E.expectError(error.NonFiniteResult, plan.execute(C, &input, &output, &work));
    try E.expectEqualDeep(saved, output);
    try plan.executePolicy(C, &input, &output, &work, .ieee);
    try E.expect(std.math.isInf(output[0]));
}

test "Real transform admitted FFTW handle preserves OOM pins close and same operation retries" {
    inline for (std.meta.tags(F.core.RealTransform), 0..) |_, kind| {
        var probe = Probe{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input = [_]f64{ 1, 2, 3, 4 };
        var output = [_]f64{ 77, 88, 99, 111 };
        const saved = output;
        const h = B.createWithContext(owner, .real_transform, .{ 4, 1, 1 }, &input, &output, @intCast(kind), 64).?;
        var operation = try owner.begin();
        probe.backing.budget = probe.backing.stats().accepted;
        try E.expectError(error.OutOfMemory, B.executeAdmitted(h, &operation));
        try E.expectEqualDeep(saved, output);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        probe.backing.budget += 1;
        _ = owner.requestClose();
        try B.executeAdmitted(h, &operation);
        var work: [128]F.core.WorkspaceScalar(C) = undefined;
        var expected: [4]f64 = undefined;
        const plan = try F.core.RealTransformPlan.init(4, @fromBackingInt(@intCast(kind)));
        try plan.execute(C, &input, &expected, &work);
        for (output, expected) |a, b| try E.expectApproxEqAbs(b, a, 1e-12);
        B.destroy(h);
        operation.end();
        try E.expectEqual(context.CloseResult.closed, fixtureProgressClose(owner));
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW typed new arrays reject layout type and alias before allocation preserve original plan" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var original: [4]C = @splat(.{ .re = 1, .im = 0 });
    var original_out: [4]C = @splat(.{ .re = 99, .im = 99 });
    var next: [5]C = @splat(.{ .re = 2, .im = 0 });
    var next_out: [5]C = @splat(.{ .re = 777, .im = 777 });
    const h = B.createWithContext(owner, .complex, .{ 4, 1, 1 }, &original, &original_out, -1, 64).?;
    const before = probe.backing.stats().accepted;
    try E.expectError(error.InvalidArgument, B.executeNewArrays(h, .real_forward, &next, &next_out));
    try E.expectError(error.InvalidArgument, B.executeNewArrays(h, .complex, &next, &next));
    try E.expectError(error.AliasingNotAllowed, B.executeNewArrays(h, .complex, &next, @ptrCast(&next[1])));
    try E.expectEqual(before, probe.backing.stats().accepted);
    try E.expectEqual(@as(c_int, 0), B.alignmentOf(&next));
    probe.backing.budget = 0;
    try E.expectError(error.OutOfMemory, B.executeNewArrays(h, .complex, &next, &next_out));
    try E.expectEqual(@as(f64, 777), next_out[0].re);
    probe.backing.budget = 10;
    try B.executeNewArrays(h, .complex, &next, &next_out);
    try E.expectApproxEqAbs(@as(f64, 8), next_out[0].re, 1e-12);
    try E.expectEqual(@as(f64, 777), next_out[4].re);
    try E.expectEqual(@as(f64, 99), original_out[0].re);
    var op = try owner.begin();
    _ = owner.requestClose();
    try E.expectError(error.Closing, B.executeNewArrays(h, .complex, &next, &next_out));
    try B.executeNewArraysAdmitted(h, .complex, &op, &next, &next_out);
    B.destroy(h);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW typed new arrays real volume and all r2r kinds match original execution and retain guards" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.heap.c_allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    inline for (.{ B.Kind.real_forward, B.Kind.volume_real_forward, B.Kind.volume_real_backward, B.Kind.real_transform }) |kind| {
        const count: usize = if (kind == .real_transform) 11 else 1;
        for (0..count) |k| {
            var input: [32]f64 = undefined;
            var other: [32]f64 = undefined;
            for (&input, &other, 0..) |*a, *b, i| {
                a.* = @as(f64, @floatFromInt(i + 1)) / 11;
                b.* = a.*;
            }
            var out: [32]f64 = @splat(777);
            var next: [32]f64 = @splat(777);
            const h = B.createWithContext(owner, kind, if (kind == .volume_real_forward or kind == .volume_real_backward) .{ 2, 2, 3 } else .{ 5, 1, 1 }, &input, &out, @intCast(k), 64).?;
            B.execute(h);
            const t: B.ArrayTransform = switch (kind) {
                .real_forward, .volume_real_forward => .real_forward,
                .volume_real_backward => .real_backward,
                else => .real_transform,
            };
            try B.executeNewArrays(h, t, &other, &next);
            for (out, next) |a, b| try E.expectApproxEqAbs(a, b, 1e-12);
            try E.expectEqual(@as(f64, 777), next[31]);
            B.destroy(h);
        }
    }
    try finish(owner);
}

test "FFTW new array in place physical padding original arrays and independent owner contracts" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.heap.c_allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var reference_other_owner: FixtureReferenceContext = undefined;
    reference_other_owner.initInPlace(std.heap.c_allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other_owner = reference_other_owner.borrowedContext();
    var foreign = try other_owner.begin();
    inline for (.{ B.Kind.complex, B.Kind.real_forward, B.Kind.volume_real_forward, B.Kind.volume_real_backward, B.Kind.real_transform }) |kind| {
        for (0..(if (kind == .real_transform) @as(usize, 11) else 1)) |k| {
            var a: [32]f64 = undefined;
            var b: [32]f64 = undefined;
            for (&a, &b, 0..) |*x, *y, i| {
                x.* = @as(f64, @floatFromInt(i + 1)) / 7;
                y.* = x.*;
            }
            const shape: [3]usize = if (kind == .volume_real_forward or kind == .volume_real_backward) .{ 2, 2, 3 } else .{ 5, 1, 1 };
            const h = B.createWithContext(owner, kind, shape, &a, &a, if (kind == .complex) -1 else @intCast(k), 64).?;
            const t: B.ArrayTransform = switch (kind) {
                .complex => .complex,
                .real_forward, .volume_real_forward => .real_forward,
                .volume_real_backward => .real_backward,
                else => .real_transform,
            };
            try E.expectError(error.ContextMismatch, B.executeNewArraysAdmitted(h, t, &foreign, &b, &b));
            try E.expectEqual(@as(f64, 1.0 / 7.0), b[0]);
            B.execute(h);
            try B.executeNewArrays(h, t, &b, &b);
            for (a, b) |x, y| try E.expectApproxEqAbs(x, y, 1e-12);
            try E.expectEqual(@as(f64, 32.0 / 7.0), b[31]);
            B.destroy(h);
        }
    }
    foreign.end();
    try finish(other_owner);
    try finish(owner);
}

fn tensorReference(comptime Z: type, shape: []const usize, input: []const Z, output: []Z, sign: f64) void {
    for (output, 0..) |*o, j| {
        var re: f64 = 0;
        var im: f64 = 0;
        for (input, 0..) |x, i| {
            var a = i;
            var b = j;
            var phase: f64 = 0;
            var axis = shape.len;
            while (axis > 0) {
                axis -= 1;
                const n = shape[axis];
                phase += @as(f64, @floatFromInt((a % n) * (b % n))) / @as(f64, @floatFromInt(n));
                a /= n;
                b /= n;
            }
            const angle = sign * 2 * std.math.pi * phase;
            re += @as(f64, x.re) * @cos(angle) - @as(f64, x.im) * @sin(angle);
            im += @as(f64, x.re) * @sin(angle) + @as(f64, x.im) * @cos(angle);
        }
        o.* = .{ .re = @floatCast(re), .im = @floatCast(im) };
    }
}
test "Tensor FFT zero through seven ranks match independent direct definitions in both scalar families" {
    const shapes = [_][]const usize{ &.{}, &.{7}, &.{ 2, 3 }, &.{ 2, 3, 2 }, &.{ 2, 1, 3, 2 }, &.{ 1, 2, 1, 3, 1, 2, 1 } };
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const tolerance: R = if (Z == C) 2e-11 else 2e-5;
        for (shapes) |shape| {
            const plan = try F.core.TensorPlan.init(shape);
            const n = try plan.count();
            const k = try plan.spectrumLength();
            const width = plan.rowWidth();
            const half = width / 2 + 1;
            var x: [32]Z = undefined;
            var rx: [32]R = undefined;
            for (x[0..n], rx[0..n], 0..) |*v, *r, i| {
                r.* = @as(R, @floatFromInt(i + 1)) / 7;
                v.* = .{ .re = r.*, .im = @as(R, @floatFromInt(i % 3)) / 11 };
            }
            var expected: [32]Z = undefined;
            var out: [33]Z = @splat(.{ .re = 777, .im = 888 });
            var back: [33]Z = @splat(.{ .re = 777, .im = 888 });
            var work: [512]F.core.WorkspaceScalar(Z) = undefined;
            tensorReference(Z, shape, x[0..n], expected[0..n], -1);
            try plan.execute(Z, x[0..n], &out, &work, .forward);
            for (out[0..n], expected[0..n]) |a, b| {
                try E.expectApproxEqAbs(b.re, a.re, tolerance);
                try E.expectApproxEqAbs(b.im, a.im, tolerance);
            }
            try plan.execute(Z, out[0..n], &back, &work, .backward);
            for (back[0..n], x[0..n]) |a, b| {
                try E.expectApproxEqAbs(b.re * @as(R, @floatFromInt(n)), a.re, tolerance);
                try E.expectApproxEqAbs(b.im * @as(R, @floatFromInt(n)), a.im, tolerance);
            }
            try E.expectEqual(@as(R, 777), out[n].re);
            try E.expectEqual(@as(R, 777), back[n].re);
            for (x[0..n], rx[0..n]) |*v, r| v.* = .{ .re = r, .im = 0 };
            tensorReference(Z, shape, x[0..n], expected[0..n], -1);
            @memset(&out, .{ .re = 777, .im = 888 });
            try plan.forwardReal(Z, rx[0..n], &out, &work);
            for (0..n / width) |row| for (0..half) |j| {
                const a = out[row * half + j];
                const b = expected[row * width + j];
                try E.expectApproxEqAbs(b.re, a.re, tolerance);
                try E.expectApproxEqAbs(b.im, a.im, tolerance);
            };
            var real_back: [33]R = @splat(777);
            try plan.backwardReal(Z, out[0..k], &real_back, &work);
            for (real_back[0..n], rx[0..n]) |a, b| try E.expectApproxEqAbs(b * @as(R, @floatFromInt(n)), a, tolerance);
            try E.expectEqual(@as(R, 777), out[k].re);
            try E.expectEqual(@as(R, 777), real_back[n]);
        }
    }
}
test "Tensor FFT checked bounds metadata alias and IEEE failures preserve transactional output" {
    try E.expectError(error.InvalidLength, F.core.TensorPlan.init(&.{ 2, 0, 3 }));
    try E.expectError(error.SizeOverflow, F.core.TensorPlan.init(&.{ std.math.maxInt(usize), 2 }));
    const p = try F.core.TensorPlan.init(&.{ 2, 3 });
    var input: [7]C = @splat(.{ .re = 1, .im = 0 });
    var output: [7]C = @splat(.{ .re = 777, .im = 888 });
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.execute(C, input[0..5], &output, &work, .forward));
    try E.expectError(error.BufferTooSmall, p.execute(C, &input, &output, work[0..1], .forward));
    try E.expectError(error.AliasingNotAllowed, p.execute(C, input[0..6], input[1..7], &work, .forward));
    input[5].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.execute(C, &input, &output, &work, .forward));
    for (output) |x| try E.expectEqual(@as(f64, 777), x.re);
    try p.executePolicy(C, &input, &output, &work, .forward, .ieee);
    try E.expect(std.math.isNan(output[0].re));
    @memset(&output, .{ .re = 777, .im = 888 });
    @memset(&input, .{ .re = std.math.floatMax(f64), .im = 0 });
    try E.expectError(error.NonFiniteResult, p.execute(C, &input, &output, &work, .forward));
    for (output) |x| try E.expectEqual(@as(f64, 777), x.re);
    var dimensions: [2]usize = .{ 2, 3 };
    const borrowed = try F.core.TensorPlan.init(&dimensions);
    const alias_output: [*]C = @ptrCast(&dimensions);
    // Shape alias is rejected before either view can be dereferenced as data.
    try E.expectError(error.AliasingNotAllowed, borrowed.execute(C, &input, alias_output[0..6], &work, .forward));
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        for ([_][]const usize{ &.{}, &.{1}, &.{2}, &.{7}, &.{8}, &.{ 1, 2 }, &.{ 2, 3 }, &.{ 2, 2, 2 }, &.{ 2, 1, 3, 2 } }) |shape| {
            const tensor = try F.core.TensorPlan.init(shape);
            const n = try tensor.count();
            const k = try tensor.spectrumLength();
            var spectrum: [32]Z = @splat(.{ .re = 0, .im = 0 });
            spectrum[0] = .{ .re = 1, .im = std.math.nan(R) };
            for (spectrum[1..k], 1..) |*x, j| {
                var index = j;
                var axis = shape.len;
                var self_conjugate = true;
                while (axis > 0) {
                    axis -= 1;
                    const dim = shape[axis];
                    const coord = index % (if (axis == shape.len - 1) dim / 2 + 1 else dim);
                    index /= if (axis == shape.len - 1) dim / 2 + 1 else dim;
                    if (coord != 0 and (dim % 2 != 0 or coord != dim / 2)) self_conjugate = false;
                }
                if (self_conjugate) x.im = std.math.inf(R);
            }
            var result: [33]R = @splat(777);
            var workspace: [512]F.core.WorkspaceScalar(Z) = undefined;
            try tensor.backwardRealPolicy(Z, spectrum[0..k], &result, &workspace, .ieee);
            for (result[0..n]) |x| try E.expectApproxEqAbs(@as(R, 1), x, 1e-5);
            try E.expectEqual(@as(R, 777), result[n]);
            if (shape.len == 3) {
                const volume = try F.core.VolumePlan.init(.{ shape[0], shape[1], shape[2] });
                try volume.backwardRealPolicy(Z, spectrum[0..k], &result, &workspace, .ieee);
                for (result[0..n]) |x| try E.expectApproxEqAbs(@as(R, 1), x, 1e-5);
            }
            try E.expectError(error.NonFiniteInput, tensor.backwardReal(Z, spectrum[0..k], &result, &workspace));
        }
    }
}
test "FFTW tensor shape resources copy dimensions roll back staged OOM and retain admitted retry after close" {
    for (0..3) |budget| {
        var probe: Probe = .{ .backing = .{ .budget = budget } };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var dims: [4]c_int = .{ 2, 1, 3, 2 };
        var input: [12]C = @splat(.{ .re = 2, .im = 0 });
        var output: [13]C = @splat(.{ .re = 777, .im = 888 });
        const h = B.createTensorWithContext(owner, .complex, 4, &dims, &input, &output, -1, 64);
        if (budget < 2) {
            try E.expect(h == null);
            try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        } else {
            try E.expect(h != null);
            try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
            @memset(&dims, 0); // The handle no longer borrows this caller storage.
            try E.expectError(error.OutOfMemory, B.executeNewArrays(h.?, .complex, &input, &output));
            for (output) |x| try E.expectEqual(@as(f64, 777), x.re);
            probe.backing.budget = 10;
            var op = try owner.begin();
            _ = owner.requestClose();
            try E.expectError(error.Closing, B.executeNewArrays(h.?, .complex, &input, &output));
            try B.executeNewArraysAdmitted(h.?, .complex, &op, &input, &output);
            try E.expectApproxEqAbs(@as(f64, 24), output[0].re, 1e-11);
            try E.expectEqual(@as(f64, 777), output[12].re);
            B.destroy(h);
            try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
            op.end();
        }
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW tensor metadata close during allocation and allocator replacement retain one captured owner" {
    inline for (.{ B.ArrayTransform.complex, B.ArrayTransform.real_forward, B.ArrayTransform.real_backward }) |transform| {
        var first: Probe = .{ .close_on_alloc = true };
        var second: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(first.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        first.owner = owner;
        second.owner = owner;
        var op = try owner.begin();
        var input: [2]C = .{ .{ .re = 2, .im = 0 }, .{ .re = 777, .im = 777 } };
        var output: [2]C = @splat(.{ .re = 777, .im = 888 });
        const h = B.createTensorWithContext(owner, transform, 0, null, &input, &output, -1, 64).?;
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        fixtureReplaceAllocator(&reference_owner, second.allocator());
        try B.executeNewArraysAdmitted(h, transform, &op, &input, &output);
        try E.expectApproxEqAbs(@as(f64, 2), output[0].re, 1e-12);
        try E.expectEqual(@as(f64, 777), output[1].re);
        try E.expectEqual(@as(usize, 1), second.backing.stats().accepted);
        try E.expectEqual(@as(usize, 1), second.backing.stats().freed);
        B.destroy(h);
        try E.expectEqual(@as(usize, 2), first.backing.stats().accepted);
        try E.expectEqual(@as(usize, 2), first.backing.stats().freed);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        op.end();
        try finish(owner);
    }
}

fn inverseRealKind(kind: F.core.RealTransform) F.core.RealTransform {
    return switch (kind) {
        .halfcomplex_forward => .halfcomplex_backward,
        .halfcomplex_backward => .halfcomplex_forward,
        .cosine_ii => .cosine_iii,
        .cosine_iii => .cosine_ii,
        .sine_ii => .sine_iii,
        .sine_iii => .sine_ii,
        else => kind,
    };
}
test "Tensor r2r all eleven mixed kinds normalize both scalar families and preserve in-place guards" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const tolerance: R = if (Z == C) 2e-10 else 2e-4;
        for (std.enums.values(F.core.RealTransform)) |first| for (std.enums.values(F.core.RealTransform)) |second| {
            const shape = [_]usize{ 2, 3 };
            const kinds = [_]F.core.RealTransform{ first, second };
            const inverse = [_]F.core.RealTransform{ inverseRealKind(first), inverseRealKind(second) };
            const p = try F.core.TensorRealTransformPlan.init(&shape, &kinds);
            const q = try F.core.TensorRealTransformPlan.init(&shape, &inverse);
            var input: [7]R = undefined;
            for (input[0..6], 0..) |*x, i| x.* = @as(R, @floatFromInt(i + 1)) / 7;
            input[6] = 777;
            var output: [7]R = @splat(777);
            var back: [7]R = @splat(777);
            var work: [256]F.core.WorkspaceScalar(Z) = undefined;
            try p.execute(Z, input[0..6], &output, &work);
            try q.execute(Z, output[0..6], &back, &work);
            const factor: R = @floatFromInt(try p.logicalLength());
            for (back[0..6], input[0..6]) |a, b| try E.expectApproxEqAbs(b * factor, a, tolerance);
            var same = input;
            try p.execute(Z, same[0..6], &same, &work);
            for (same[0..6], output[0..6]) |a, b| try E.expectApproxEqAbs(b, a, tolerance);
            try E.expectEqual(@as(R, 777), same[6]);
            try E.expectEqual(@as(R, 777), back[6]);
            try E.expectEqual(@as(R, 777), output[6]);
        };
        const identity = try F.core.TensorRealTransformPlan.init(&.{}, &.{});
        var scalar = [_]R{ 2, 777 };
        var result = [_]R{ 99, 777 };
        var work: [8]F.core.WorkspaceScalar(Z) = undefined;
        try identity.execute(Z, scalar[0..1], &result, &work);
        try E.expectEqual(@as(R, 2), result[0]);
        try E.expectEqual(@as(R, 777), result[1]);
    }
}
test "Tensor r2r checked metadata bounds and late overflow preserve complete destination" {
    try E.expectError(error.InvalidLength, F.core.TensorRealTransformPlan.init(&.{ 2, 3 }, &.{.hartley}));
    try E.expectError(error.InvalidLength, F.core.TensorRealTransformPlan.init(&.{1}, &.{.cosine_i}));
    const p = try F.core.TensorRealTransformPlan.init(&.{ 2, 3 }, &.{ .sine_i, .cosine_ii });
    var input: [7]f64 = @splat(1);
    var output: [7]f64 = @splat(777);
    var work: [256]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.execute(C, input[0..5], &output, &work));
    try E.expectError(error.BufferTooSmall, p.execute(C, &input, &output, work[0..1]));
    try E.expectError(error.AliasingNotAllowed, p.execute(C, input[0..6], input[1..7], &work));
    input[5] = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.execute(C, &input, &output, &work));
    for (output) |x| try E.expectEqual(@as(f64, 777), x);
    @memset(&input, std.math.floatMax(f64));
    try E.expectError(error.NonFiniteResult, p.execute(C, &input, &output, &work));
    for (output) |x| try E.expectEqual(@as(f64, 777), x);
    try p.executePolicy(C, &input, &output, &work, .ieee);
    try E.expect(!std.math.isFinite(output[0]));
    try E.expectEqual(@as(f64, 777), output[6]);
}
test "FFTW tensor r2r copied axis descriptors survive metadata OOM close and allocator replacement" {
    for (0..3) |budget| {
        var first: Probe = .{ .backing = .{ .budget = budget } };
        var second: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(first.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        first.owner = owner;
        second.owner = owner;
        var dims = [_]c_int{ 2, 3 };
        var modes = [_]c_uint{ 5, 9 };
        var input = [_]f64{ 1, 2, 3, 4, 5, 6 };
        var output: [7]f64 = @splat(777);
        const h = B.createTensorRealWithContext(owner, 2, &dims, &modes, &input, &output, 64);
        if (budget < 2) {
            try E.expect(h == null);
            try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        } else {
            try E.expect(h != null);
            try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
            @memset(&dims, 0);
            @memset(&modes, 99);
            try E.expectError(error.OutOfMemory, B.executeNewArrays(h.?, .real_transform, &input, &output));
            for (output) |x| try E.expectEqual(@as(f64, 777), x);
            fixtureReplaceAllocator(&reference_owner, second.allocator());
            var op = try owner.begin();
            _ = owner.requestClose();
            try E.expectError(error.Closing, B.executeNewArrays(h.?, .real_transform, &input, &output));
            try B.executeNewArraysAdmitted(h.?, .real_transform, &op, &input, &output);
            try E.expect(std.math.isFinite(output[0]));
            try E.expectEqual(@as(f64, 777), output[6]);
            B.destroy(h);
            op.end();
            try E.expectEqual(first.backing.stats().accepted, first.backing.stats().freed);
            try E.expectEqual(second.backing.stats().accepted, second.backing.stats().freed);
        }
        try finish(owner);
    }
}

test "Neutral strided complex batches preserve signed layouts holes and complete in-place gather" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const tolerance: R = if (Z == C) 2e-11 else 3e-5;
        const p = try F.core.StridedComplexPlan.init(.{ .shape = &.{ 2, 3 }, .input_strides = &.{ 9, -2 }, .output_strides = &.{ 3, 2 }, .batch_shape = &.{2}, .input_batch_strides = &.{20}, .output_batch_strides = &.{12}, .direction = .forward });
        const ib = try p.inputBounds();
        const ob = try p.outputBounds();
        try E.expectEqual(@as(isize, -4), ib.lower);
        try E.expectEqual(@as(usize, 34), try ib.length());
        try E.expectEqual(@as(usize, 20), try ob.length());
        var input: [40]Z = @splat(.{ .re = 777, .im = 888 });
        var out: [40]Z = @splat(.{ .re = 777, .im = 888 });
        var expected: [12]Z = undefined;
        var logical: [12]Z = undefined;
        for (&logical, 0..) |*x, i| x.* = .{ .re = @as(R, @floatFromInt(i + 1)) / 7, .im = @as(R, @floatFromInt(i % 3)) / 11 };
        for (logical, 0..) |x, i| {
            const pos = try p.offset(i, false) + 4;
            input[@intCast(pos)] = x;
        }
        for (0..2) |batch| tensorReference(Z, &.{ 2, 3 }, logical[batch * 6 ..][0..6], expected[batch * 6 ..][0..6], -1);
        var work: [256]F.core.WorkspaceScalar(Z) = undefined;
        try p.execute(Z, &input, 4, &out, 0, &work);
        var touched: [40]bool = @splat(false);
        for (expected, 0..) |x, i| {
            const pos: usize = @intCast(try p.offset(i, true));
            touched[pos] = true;
            try E.expectApproxEqAbs(x.re, out[pos].re, tolerance);
            try E.expectApproxEqAbs(x.im, out[pos].im, tolerance);
        }
        for (out, touched) |x, t| if (!t) {
            try E.expectEqual(@as(R, 777), x.re);
            try E.expectEqual(@as(R, 888), x.im);
        };
        var same = input;
        try p.execute(Z, &same, 4, &same, 0, &work);
        for (expected, 0..) |x, i| try E.expectApproxEqAbs(x.re, same[@intCast(try p.offset(i, true))].re, tolerance);
        const identity = try F.core.StridedComplexPlan.init(.{ .shape = &.{}, .input_strides = &.{}, .output_strides = &.{}, .batch_shape = &.{3}, .input_batch_strides = &.{-2}, .output_batch_strides = &.{3}, .direction = .backward });
        var scalar: [8]Z = @splat(.{ .re = 777, .im = 888 });
        try identity.execute(Z, &input, 4, &scalar, 0, &work);
        try E.expectEqual(input[4].re, scalar[0].re);
        try E.expectEqual(input[2].re, scalar[3].re);
        try E.expectEqual(input[0].re, scalar[6].re);
    }
}
test "Neutral strided complex rejects duplicate writes overflow short views and late nonfinite without touching holes" {
    try E.expectError(error.AliasingNotAllowed, F.core.StridedComplexPlan.init(.{ .shape = &.{ 2, 3 }, .input_strides = &.{ 3, 1 }, .output_strides = &.{ 2, 1 }, .batch_shape = &.{}, .input_batch_strides = &.{}, .output_batch_strides = &.{}, .direction = .forward }));
    try E.expectError(error.SizeOverflow, F.core.StridedComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{std.math.maxInt(isize)}, .output_strides = &.{1}, .batch_shape = &.{}, .input_batch_strides = &.{}, .output_batch_strides = &.{}, .direction = .forward }));
    const p = try F.core.StridedComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{-2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{8}, .output_batch_strides = &.{7}, .direction = .forward });
    var input: [16]C = @splat(.{ .re = 1, .im = 0 });
    var out: [16]C = @splat(.{ .re = 777, .im = 888 });
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.execute(C, &input, 3, &out, 0, &work));
    try E.expectError(error.BufferTooSmall, p.execute(C, input[0..12], 4, &out, 0, &work));
    try E.expectError(error.BufferTooSmall, p.execute(C, &input, 4, &out, 0, work[0..1]));
    input[12].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.execute(C, &input, 4, &out, 0, &work));
    for (out) |x| try E.expectEqual(@as(f64, 777), x.re);
    @memset(&input, .{ .re = std.math.floatMax(f64), .im = 0 });
    try E.expectError(error.NonFiniteResult, p.execute(C, &input, 4, &out, 0, &work));
    for (out) |x| try E.expectEqual(@as(f64, 777), x.re);
    try p.executePolicy(C, &input, 4, &out, 0, &work, .ieee);
    try E.expect(!std.math.isFinite(out[0].re));
    try E.expectEqual(@as(f64, 777), out[1].re);
    const empty = try F.core.StridedComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{1}, .output_strides = &.{1}, .batch_shape = &.{0}, .input_batch_strides = &.{0}, .output_batch_strides = &.{0}, .direction = .forward });
    try E.expectEqual(@as(usize, 0), try empty.workspaceLength(C));
    try empty.execute(C, &.{}, 0, &.{}, 0, &.{});
}
test "Neutral strided managed scratch OOM close and borrowed operation share captured Core" {
    const p = try F.core.StridedComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{1}, .output_batch_strides = &.{1}, .direction = .forward });
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [6]C = @splat(.{ .re = 1, .im = 0 });
    var out: [7]C = @splat(.{ .re = 777, .im = 888 });
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.OutOfMemory, F.Managed(Execution).runStridedScratch(C, owner, p, &input, 0, &out, 0, .checked));
    for (out) |x| try E.expectEqual(@as(f64, 777), x.re);
    var op = try owner.begin();
    _ = owner.requestClose();
    try E.expectError(error.Closing, F.Managed(Execution).runStridedScratch(C, owner, p, &input, 0, &out, 0, .checked));
    try F.Managed(Execution).runStridedBorrowedAdmitted(C, &op, p, &input, 0, &out, 0, &work, .checked);
    try E.expectApproxEqAbs(@as(f64, 3), out[0].re, 1e-12);
    try E.expectApproxEqAbs(@as(f64, 3), out[1].re, 1e-12);
    try E.expectEqual(@as(f64, 777), out[6].re);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW many and both guru widths copy layouts and retain captured resources" {
    inline for (.{ 0, 1, 2 }) |kind| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [7]C = @splat(.{ .re = 1, .im = 0 });
        var output: [7]C = @splat(.{ .re = 777, .im = 0 });
        var dims: [1]c_int = .{3};
        var gd: [1]B.IODim = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
        var gb: [1]B.IODim = .{.{ .n = 2, .input_stride = 1, .output_stride = 1 }};
        var gd64: [1]B.IODim64 = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
        var gb64: [1]B.IODim64 = .{.{ .n = 2, .input_stride = 1, .output_stride = 1 }};
        const h = (switch (kind) {
            0 => B.createManyWithContext(owner, 1, &dims, 2, &input, null, 2, 1, &output, null, 2, 1, -1, 64),
            1 => B.createGuruWithContext(B.IODim, owner, 1, &gd, 1, &gb, &input, &output, -1, 64),
            2 => B.createGuruWithContext(B.IODim64, owner, 1, &gd64, 1, &gb64, &input, &output, -1, 64),
            else => unreachable,
        }).?;
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        dims[0] = 99;
        gd[0].n = 99;
        gb[0].n = 99;
        gd64[0].n = 99;
        gb64[0].n = 99;
        try B.executeWithArrays(h, &input, &output);
        try E.expectApproxEqAbs(@as(f64, 3), output[0].re, 1e-12);
        try E.expectApproxEqAbs(@as(f64, 3), output[1].re, 1e-12);
        try E.expectEqual(@as(f64, 777), output[6].re);
        _ = owner.requestClose();
        try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}
test "FFTW strided constructors release partial allocations and reject invalid metadata" {
    inline for (.{ @as(usize, 0), 1 }) |budget| {
        var probe: Probe = .{};
        probe.backing.budget = budget;
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [6]C = @splat(.{ .re = 1, .im = 0 });
        var output: [6]C = @splat(.{ .re = 777, .im = 0 });
        try E.expect(B.createManyWithContext(owner, 1, &.{3}, 2, &input, null, 2, 1, &output, null, 2, 1, -1, 64) == null);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [6]C = @splat(.{ .re = 1, .im = 0 });
    var output: [6]C = @splat(.{ .re = 777, .im = 0 });
    try E.expect(B.createManyWithContext(owner, 1, &.{3}, 1, &input, &.{2}, 1, 3, &output, null, 1, 3, -1, 64) == null);
    try E.expect(B.createGuruWithContext(B.IODim64, owner, 1, &.{.{ .n = 3, .input_stride = std.math.maxInt(isize), .output_stride = 1 }}, 0, null, &input, &output, -1, 64) == null);
    try E.expect(B.createGuruWithContext(B.IODim, owner, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 0 }}, 0, null, &input, &output, -1, 64) == null);
    const empty = B.createManyWithContext(owner, 1, &.{3}, 0, null, null, 1, 3, null, null, 1, 3, -1, 64).?;
    B.destroy(empty);
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW strided empty and unbound plans honor inactive strides and new-array ownership" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    const empty = B.createManyWithContext(owner, 1, &.{3}, 0, null, null, 2, 6, null, null, 1, 3, -1, 64).?;
    var op = try owner.begin();
    try B.executeAdmitted(empty, &op);
    op.end();
    B.destroy(empty);
    const h = B.createManyWithContext(owner, 1, &.{3}, 1, null, null, 2, 6, null, null, 2, 3, -1, 64).?;
    var input: [5]C = .{ .{ .re = 1, .im = 0 }, .{ .re = 777, .im = 888 }, .{ .re = 1, .im = 0 }, .{ .re = 777, .im = 888 }, .{ .re = 1, .im = 0 } };
    try B.executeNewArrays(h, .complex, &input, &input);
    try E.expectApproxEqAbs(@as(f64, 3), input[0].re, 1e-12);
    try E.expectApproxEqAbs(@as(f64, 0), input[2].re, 1e-12);
    try E.expectApproxEqAbs(@as(f64, 0), input[4].re, 1e-12);
    try E.expectEqual(@as(f64, 777), input[1].re);
    try E.expectEqual(@as(f64, 888), input[3].im);
    B.destroy(h);
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "Neutral strided real batches preserve half spectra signed holes and inverse normalization" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const forward = try F.core.StridedRealPlan.init(.{ .mode = .forward, .shape = &.{ 2, 3 }, .half_shape = &.{ 2, 2 }, .input_strides = &.{ 9, -2 }, .output_strides = &.{ 5, 2 }, .batch_shape = &.{2}, .input_batch_strides = &.{20}, .output_batch_strides = &.{12} });
        const backward = try F.core.StridedRealPlan.init(.{ .mode = .backward, .shape = &.{ 2, 3 }, .half_shape = &.{ 2, 2 }, .input_strides = &.{ 5, 2 }, .output_strides = &.{ 9, -2 }, .batch_shape = &.{2}, .input_batch_strides = &.{12}, .output_batch_strides = &.{20} });
        var input: [34]R = @splat(777);
        var spectrum: [20]Z = @splat(.{ .re = 777, .im = 888 });
        var output: [34]R = @splat(777);
        var logical: [12]Z = undefined;
        var reference: [12]Z = undefined;
        var work: [256]F.core.WorkspaceScalar(Z) = undefined;
        const ig = forward.geometry(false);
        const og = forward.geometry(true);
        for (0..12) |j| {
            const x: @TypeOf(input[0]) = @floatFromInt(j + 1);
            logical[j] = .{ .re = x, .im = 0 };
            const k = try ig.offset(j, false);
            input[@intCast(4 + k)] = x;
        }
        try forward.executePolicy(Z, .forward, &input, 4, &spectrum, 0, &work, .checked);
        for (0..2) |batch| tensorReference(Z, &.{ 2, 3 }, logical[batch * 6 ..][0..6], reference[batch * 6 ..][0..6], -1);
        for (0..8) |j| {
            const k: @TypeOf(j) = (j / 4) * 6 + ((j % 4) / 2) * 3 + j % 2;
            const ix: usize = @intCast(try og.offset(j, true));
            try E.expectApproxEqAbs(reference[k].re, spectrum[ix].re, if (Z == C) 1e-9 else 2e-4);
            try E.expectApproxEqAbs(reference[k].im, spectrum[ix].im, if (Z == C) 1e-9 else 2e-4);
        }
        try backward.executePolicy(Z, .backward, &spectrum, 0, &output, 4, &work, .checked);
        var touched: [34]bool = @splat(false);
        for (0..12) |j| {
            const ix: usize = @intCast(4 + try ig.offset(j, false));
            touched[ix] = true;
            try E.expectApproxEqAbs(logical[j].re * 6, output[ix], if (Z == C) 1e-9 else 2e-4);
        }
        for (output, touched) |x, used| if (!used) try E.expectEqual(@as(R, 777), x);
    }
}
test "Neutral strided mixed r2r batches allow complete inplace gather and transactional late errors" {
    const p = try F.core.StridedRealPlan.init(.{ .mode = .transform, .shape = &.{ 2, 3 }, .kinds = &.{ .cosine_ii, .sine_iv }, .input_strides = &.{ 9, -2 }, .output_strides = &.{ 3, 2 }, .batch_shape = &.{2}, .input_batch_strides = &.{20}, .output_batch_strides = &.{12} });
    var input: [34]f64 = @splat(777);
    var output: [34]f64 = @splat(777);
    var expected: [12]f64 = undefined;
    var unit_in: [6]f64 = undefined;
    var work: [256]F.core.WorkspaceScalar(C) = undefined;
    var unit_work: [128]F.core.WorkspaceScalar(C) = undefined;
    const ig = p.geometry(false);
    const og = p.geometry(true);
    for (0..12) |j| input[@intCast(4 + try ig.offset(j, false))] = @floatFromInt(j + 1);
    for (0..2) |batch| {
        for (0..6) |j| unit_in[j] = @floatFromInt(batch * 6 + j + 1);
        try (try F.core.TensorRealTransformPlan.init(&.{ 2, 3 }, &.{ .cosine_ii, .sine_iv })).execute(C, &unit_in, expected[batch * 6 ..][0..6], &unit_work);
    }
    try p.executePolicy(C, .transform, &input, 4, &output, 0, &work, .checked);
    for (0..12) |j| try E.expectApproxEqAbs(expected[j], output[@intCast(try og.offset(j, true))], 1e-9);
    var inplace = input;
    try p.executePolicy(C, .transform, &inplace, 4, &inplace, 0, &work, .checked);
    for (0..12) |j| try E.expectApproxEqAbs(expected[j], inplace[@intCast(try og.offset(j, true))], 1e-9);
    output = @splat(777);
    input[@intCast(4 + try ig.offset(11, false))] = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.executePolicy(C, .transform, &input, 4, &output, 0, &work, .checked));
    for (output) |x| try E.expectEqual(@as(f64, 777), x);
    input = @splat(std.math.floatMax(f64));
    try E.expectError(error.NonFiniteResult, p.executePolicy(C, .transform, &input, 4, &output, 0, &work, .checked));
    for (output) |x| try E.expectEqual(@as(f64, 777), x);
}
test "Neutral strided real validates descriptors capacities zero batches and managed close allocation" {
    const p = try F.core.StridedRealPlan.init(.{ .mode = .forward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{1}, .output_batch_strides = &.{1} });
    var bad = p;
    bad.half_shape = &.{3};
    try E.expectError(error.InvalidLength, bad.init());
    bad = p;
    bad.output_strides = &.{0};
    try E.expectError(error.AliasingNotAllowed, bad.init());
    var input: [6]f64 = @splat(1);
    var output: [4]C = @splat(.{ .re = 777, .im = 888 });
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.executePolicy(C, .forward, input[0..5], 0, &output, 0, &work, .checked));
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    try E.expectError(error.OutOfMemory, F.Managed(Execution).runStridedRealScratch(C, .forward, owner, p, &input, 0, &output, 0, .checked));
    for (output) |x| try E.expectEqual(@as(f64, 777), x.re);
    var op = try owner.begin();
    _ = owner.requestClose();
    try E.expectError(error.Closing, F.Managed(Execution).runStridedRealScratch(C, .forward, owner, p, &input, 0, &output, 0, .checked));
    try F.Managed(Execution).runStridedRealBorrowedAdmitted(C, .forward, &op, p, &input, 0, &output, 0, &work, .checked);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    var empty = p;
    empty.batch_shape = &.{0};
    try E.expectEqual(@as(usize, 0), try empty.workspaceLength(C));
    try empty.executePolicy(C, .forward, &.{}, 0, &.{}, 0, &.{}, .checked);
}

test "Neutral strided real overlapping scalar complex views stage every batch and preserve ignored IEEE endpoints" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        var raw: [16]R = @splat(777);
        for (0..6) |j| raw[j] = @floatFromInt(j + 1);
        const spectra = std.mem.bytesAsSlice(Z, std.mem.sliceAsBytes(&raw));
        var work: [128]F.core.WorkspaceScalar(Z) = undefined;
        const fwd = try F.core.StridedRealPlan.init(.{ .mode = .forward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{1}, .output_batch_strides = &.{1} });
        const bwd = try F.core.StridedRealPlan.init(.{ .mode = .backward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{1}, .output_batch_strides = &.{1} });
        try fwd.executePolicy(Z, .forward, &raw, 0, spectra, 0, &work, .checked);
        try E.expectApproxEqAbs(@as(R, 9), spectra[0].re, 1e-4);
        try E.expectApproxEqAbs(@as(R, 12), spectra[1].re, 1e-4);
        spectra[0].im = std.math.nan(R);
        spectra[1].im = std.math.inf(R);
        try E.expectError(error.NonFiniteInput, bwd.executePolicy(Z, .backward, spectra, 0, &raw, 0, &work, .checked));
        try bwd.executePolicy(Z, .backward, spectra, 0, &raw, 0, &work, .ieee);
        for (0..6) |j| try E.expectApproxEqAbs(@as(R, @floatFromInt(3 * (j + 1))), raw[j], 1e-4);
        for (raw[8..]) |x| try E.expectEqual(@as(R, 777), x);
    }
}
test "Neutral strided real rank zero identities and metadata work aliases fail without committing" {
    const scalar = try F.core.StridedRealPlan.init(.{ .mode = .forward, .shape = &.{}, .half_shape = &.{}, .input_strides = &.{}, .output_strides = &.{}, .batch_shape = &.{3}, .input_batch_strides = &.{-2}, .output_batch_strides = &.{3} });
    var input: [5]f64 = .{ 3, 777, 2, 777, 1 };
    var output: [7]C = @splat(.{ .re = 777, .im = 888 });
    var work: [64]F.core.WorkspaceScalar(C) = undefined;
    try scalar.executePolicy(C, .forward, &input, 4, &output, 0, &work, .checked);
    try E.expectEqual(@as(f64, 1), output[0].re);
    try E.expectEqual(@as(f64, 2), output[3].re);
    try E.expectEqual(@as(f64, 3), output[6].re);
    try E.expectEqual(@as(f64, 777), output[1].re);
    var shape: [1]usize = .{3};
    var p = try F.core.StridedRealPlan.init(.{ .mode = .forward, .shape = &shape, .half_shape = &.{2}, .input_strides = &.{1}, .output_strides = &.{1}, .batch_shape = &.{}, .input_batch_strides = &.{}, .output_batch_strides = &.{} });
    output = @splat(.{ .re = 777, .im = 888 });
    try E.expectError(error.BufferTooSmall, p.executePolicy(C, .forward, input[0..3], 0, output[0..1], 0, &work, .checked));
    try E.expectError(error.BufferTooSmall, p.executePolicy(C, .forward, input[0..3], 0, &output, 0, work[0..1], .checked));
    const alias: [*]const usize = @ptrCast(&output);
    p.shape = alias[0..1];
    output[0].re = @bitCast(@as(u64, 3));
    const before = output;
    try E.expectError(error.AliasingNotAllowed, p.executePolicy(C, .forward, input[0..3], 0, &output, 0, &work, .checked));
    try E.expectEqualSlices(u8, std.mem.sliceAsBytes(&before), std.mem.sliceAsBytes(&output));
}

test "FFTW strided real seam captures half shape strides kinds and exact Context cleanup" {
    inline for (.{ F.core.StridedRealPlan.Mode.forward, .backward, .transform }) |mode| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var shape: [1]usize = .{3};
        var half: [1]usize = .{2};
        var strides: [1]isize = .{2};
        var batch: [1]usize = .{2};
        var distance: [1]isize = .{1};
        var kinds: [1]F.core.RealTransform = .{.hartley};
        const p: F.core.StridedRealPlan = .{ .mode = mode, .shape = &shape, .half_shape = if (mode == .transform) &.{} else &half, .kinds = if (mode == .transform) &kinds else &.{}, .input_strides = &strides, .output_strides = &strides, .batch_shape = &batch, .input_batch_strides = &distance, .output_batch_strides = &distance };
        var reals: [6]f64 = @splat(1);
        var spectra: [6]C = @splat(.{ .re = 0, .im = 0 });
        spectra[0].re = 3;
        spectra[1].re = 3;
        var out: [6]f64 = @splat(777);
        var cout: [4]C = @splat(.{ .re = 777, .im = 888 });
        const input: *anyopaque = if (mode == .backward) @ptrCast(&spectra) else @ptrCast(&reals);
        const output: *anyopaque = if (mode == .forward) @ptrCast(&cout) else @ptrCast(&out);
        const h = B.createStridedRealWithContext(owner, p, input, output, 64).?;
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        shape[0] = 99;
        half[0] = 99;
        strides[0] = 99;
        batch[0] = 99;
        distance[0] = 99;
        kinds[0] = .sine_i;
        try B.executeWithArrays(h, input, output);
        if (mode == .forward) {
            try E.expectApproxEqAbs(@as(f64, 3), cout[0].re, 1e-12);
            try E.expectApproxEqAbs(@as(f64, 3), cout[1].re, 1e-12);
        } else if (mode == .backward) {
            for (out) |x| try E.expectApproxEqAbs(@as(f64, 3), x, 1e-12);
        } else {
            try E.expectApproxEqAbs(@as(f64, 3), out[0], 1e-12);
            try E.expectApproxEqAbs(@as(f64, 3), out[1], 1e-12);
        }
        _ = owner.requestClose();
        try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW real many guru and guru64 wire constructors share copied neutral modes" {
    inline for (.{ F.core.StridedRealPlan.Mode.forward, .backward, .transform }) |mode| {
        inline for (.{ 0, 1, 2 }) |api| {
            var probe: Probe = .{};
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            probe.owner = owner;
            var n: [1]c_int = .{3};
            var kind: [1]c_uint = .{2};
            var d: [1]B.IODim = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
            var b: [1]B.IODim = .{.{ .n = 2, .input_stride = 1, .output_stride = 1 }};
            var d64: [1]B.IODim64 = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
            var b64: [1]B.IODim64 = .{.{ .n = 2, .input_stride = 1, .output_stride = 1 }};
            var reals: [6]f64 = @splat(1);
            var spectra: [4]C = @splat(.{ .re = 0, .im = 0 });
            spectra[0].re = 3;
            spectra[1].re = 3;
            var output: [6]f64 = @splat(777);
            var cout: [4]C = @splat(.{ .re = 777, .im = 888 });
            const i: *anyopaque = if (mode == .backward) @ptrCast(&spectra) else @ptrCast(&reals);
            const o: *anyopaque = if (mode == .forward) @ptrCast(&cout) else @ptrCast(&output);
            const kinds: ?[*]const c_uint = if (mode == .transform) &kind else null;
            const h = (switch (api) {
                0 => B.createManyRealWithContext(owner, mode, 1, &n, 2, i, null, 2, 1, o, null, 2, 1, kinds, 64),
                1 => B.createGuruRealWithContext(B.IODim, owner, mode, 1, &d, 1, &b, i, o, kinds, 64),
                2 => B.createGuruRealWithContext(B.IODim64, owner, mode, 1, &d64, 1, &b64, i, o, kinds, 64),
                else => unreachable,
            }).?;
            n[0] = 99;
            kind[0] = 99;
            d[0].n = 99;
            b[0].n = 99;
            d64[0].n = 99;
            b64[0].n = 99;
            try B.executeWithArrays(h, i, o);
            if (mode == .forward) {
                try E.expectApproxEqAbs(@as(f64, 3), cout[0].re, 1e-12);
                try E.expectApproxEqAbs(@as(f64, 3), cout[1].re, 1e-12);
            } else if (mode == .backward) {
                for (output) |x| try E.expectApproxEqAbs(@as(f64, 3), x, 1e-12);
            } else {
                try E.expectApproxEqAbs(@as(f64, 3), output[0], 1e-12);
                try E.expectApproxEqAbs(@as(f64, 3), output[1], 1e-12);
            }
            B.destroy(h);
            try finish(owner);
            try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
        }
    }
}
test "FFTW real many implicit inplace padding and partial metadata failure retain one admission" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var raw: [8]f64 = .{ 1, 1, 1, 777, 1, 1, 1, 777 };
    const h = B.createManyRealWithContext(owner, .forward, 2, &.{ 2, 3 }, 1, &raw, null, 1, 8, &raw, null, 1, 4, null, 64).?;
    try B.executeWithArrays(h, &raw, &raw);
    const spectrum = std.mem.bytesAsSlice(C, std.mem.sliceAsBytes(&raw));
    try E.expectApproxEqAbs(@as(f64, 6), spectrum[0].re, 1e-12);
    for (spectrum[1..]) |x| {
        try E.expectApproxEqAbs(@as(f64, 0), x.re, 1e-12);
        try E.expectApproxEqAbs(@as(f64, 0), x.im, 1e-12);
    }
    B.destroy(h);
    try finish(owner);
    inline for (.{ @as(usize, 0), 1 }) |budget| {
        var oom: Probe = .{ .backing = .{ .budget = budget } };
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(oom.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        oom.owner = ctx;
        var input: [3]f64 = @splat(1);
        var output: [2]C = @splat(.{ .re = 777, .im = 888 });
        try E.expect(B.createManyRealWithContext(ctx, .forward, 1, &.{3}, 1, &input, null, 1, 3, &output, null, 1, 2, null, 64) == null);
        try E.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        try finish(ctx);
        try E.expectEqual(oom.backing.stats().accepted, oom.backing.stats().freed);
    }
}

test "FFTW real inplace unequal scalar complex strides reuse complete staged gather" {
    inline for (.{ 0, 1, 2 }) |api| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var raw: [12]f64 = @splat(777);
        raw[0] = 1;
        raw[2] = 3;
        raw[4] = 5;
        const h = (switch (api) {
            0 => B.createManyRealWithContext(owner, .forward, 1, &.{3}, 1, &raw, null, 2, 0, &raw, null, 1, 0, null, 66),
            1 => B.createGuruRealWithContext(B.IODim, owner, .forward, 1, &.{.{ .n = 3, .input_stride = 2, .output_stride = 1 }}, 0, null, &raw, &raw, null, 66),
            2 => B.createGuruRealWithContext(B.IODim64, owner, .forward, 1, &.{.{ .n = 3, .input_stride = 2, .output_stride = 1 }}, 0, null, &raw, &raw, null, 66),
            else => unreachable,
        }).?;
        try B.executeWithArrays(h, &raw, &raw);
        const out = std.mem.bytesAsSlice(C, std.mem.sliceAsBytes(&raw));
        try E.expectApproxEqAbs(@as(f64, 9), out[0].re, 1e-12);
        try E.expectApproxEqAbs(@as(f64, -3), out[1].re, 1e-12);
        try E.expectApproxEqAbs(@sqrt(@as(f64, 3)), out[1].im, 1e-12);
        for (raw[5..]) |x| try E.expectEqual(@as(f64, 777), x);
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW real explicit unpadded embeddings accept captured unbound and empty plans" {
    inline for (.{ F.core.StridedRealPlan.Mode.forward, .backward }) |mode| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        const empty = B.createManyRealWithContext(owner, mode, 2, &.{ 2, 3 }, 0, null, &.{ 2, 3 }, 1, 12, null, &.{ 2, 3 }, 1, 6, null, 64).?;
        var op = try owner.begin();
        try B.executeAdmitted(empty, &op);
        op.end();
        B.destroy(empty);
        const h = B.createManyRealWithContext(owner, mode, 2, &.{ 2, 3 }, 1, null, &.{ 2, 3 }, 1, 12, null, &.{ 2, 3 }, 1, 6, null, 64).?;
        var raw: [12]f64 = @splat(0);
        if (mode == .forward) {
            for (raw[0..6]) |*x| x.* = 1;
        } else raw[0] = 6;
        try B.executeNewArrays(h, if (mode == .forward) .real_forward else .real_backward, &raw, &raw);
        if (mode == .forward) {
            const spectrum = std.mem.bytesAsSlice(C, std.mem.sliceAsBytes(&raw));
            try E.expectApproxEqAbs(@as(f64, 6), spectrum[0].re, 1e-12);
        } else {
            for (raw[0..6]) |x| try E.expectApproxEqAbs(@as(f64, 6), x, 1e-12);
        }
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW real wire metadata allocation close finishes only the existing admitted operation" {
    inline for (.{ F.core.StridedRealPlan.Mode.forward, .backward, .transform }) |mode| {
        inline for (.{ 0, 1, 2 }) |api| {
            var probe: Probe = .{ .close_on_alloc = true };
            var reference_owner: FixtureReferenceContext = undefined;
            reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const owner = reference_owner.borrowedContext();
            probe.owner = owner;
            var op = try owner.begin();
            var input: [3]f64 = @splat(1);
            var spectrum: [2]C = .{ .{ .re = 3, .im = 0 }, .{ .re = 0, .im = 0 } };
            var output: [3]f64 = @splat(777);
            var cout: [2]C = @splat(.{ .re = 777, .im = 888 });
            const i: *anyopaque = if (mode == .backward) @ptrCast(&spectrum) else @ptrCast(&input);
            const o: *anyopaque = if (mode == .forward) @ptrCast(&cout) else @ptrCast(&output);
            const kinds: ?[*]const c_uint = if (mode == .transform) &.{2} else null;
            const h = (switch (api) {
                0 => B.createManyRealWithContext(owner, mode, 1, &.{3}, 1, i, null, 1, 3, o, null, 1, 2, kinds, 64),
                1 => B.createGuruRealWithContext(B.IODim, owner, mode, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, i, o, kinds, 64),
                2 => B.createGuruRealWithContext(B.IODim64, owner, mode, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, i, o, kinds, 64),
                else => unreachable,
            }).?;
            try E.expectError(error.Closing, B.executeWithArrays(h, i, o));
            try B.executeAdmitted(h, &op);
            if (mode == .forward) try E.expectApproxEqAbs(@as(f64, 3), cout[0].re, 1e-12) else try E.expectApproxEqAbs(@as(f64, 3), output[0], 1e-12);
            B.destroy(h);
            op.end();
            try finish(owner);
            try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
        }
    }
}

test "Neutral split complex signed batches match direct DFT and preserve independent plane holes" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const p = try F.core.SplitComplexPlan.init(.{ .shape = &.{ 2, 3 }, .input_strides = &.{ 9, -2 }, .output_strides = &.{ 3, 2 }, .batch_shape = &.{2}, .input_batch_strides = &.{20}, .output_batch_strides = &.{12}, .direction = .forward });
        var ri: [34]R = @splat(777);
        var ii: [34]R = @splat(888);
        var ro: [20]R = @splat(777);
        var io: [20]R = @splat(888);
        var work: [256]F.core.WorkspaceScalar(Z) = undefined;
        var logical: [12]Z = undefined;
        var expected: [12]Z = undefined;
        for (0..12) |j| {
            const x: Z = .{ .re = @floatFromInt(j + 1), .im = @as(R, @floatFromInt(j)) * 0.1 };
            logical[j] = x;
            const ix: usize = @intCast(4 + try p.layout.offset(j, false));
            ri[ix] = x.re;
            ii[ix] = x.im;
        }
        for (0..2) |b| tensorReference(Z, &.{ 2, 3 }, logical[b * 6 ..][0..6], expected[b * 6 ..][0..6], -1);
        try p.executePolicy(Z, .{ .values = &ri, .origin = 4 }, .{ .values = &ii, .origin = 4 }, .{ .values = &ro }, .{ .values = &io }, &work, .checked);
        var touched: [20]bool = @splat(false);
        for (0..12) |j| {
            const ix: usize = @intCast(try p.layout.offset(j, true));
            touched[ix] = true;
            try E.expectApproxEqAbs(expected[j].re, ro[ix], if (Z == C) 1e-9 else 1e-4);
            try E.expectApproxEqAbs(expected[j].im, io[ix], if (Z == C) 1e-9 else 1e-4);
        }
        for (ro, io, touched) |r, i, used| if (!used) {
            try E.expectEqual(@as(R, 777), r);
            try E.expectEqual(@as(R, 888), i);
        };
    }
}
test "Neutral split complex interleaved inplace planes reject touched collisions and late failures atomically" {
    const p = try F.core.SplitComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{2}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{6}, .output_batch_strides = &.{6}, .direction = .forward });
    var raw: [12]f64 = undefined;
    var logical: [6]C = undefined;
    var expected: [6]C = undefined;
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    for (0..6) |j| {
        logical[j] = .{ .re = @floatFromInt(j + 1), .im = @as(f64, @floatFromInt(j)) * 0.1 };
        raw[2 * j] = logical[j].re;
        raw[2 * j + 1] = logical[j].im;
    }
    for (0..2) |b| tensorReference(C, &.{3}, logical[b * 3 ..][0..3], expected[b * 3 ..][0..3], -1);
    try p.executePolicy(C, .{ .values = &raw }, .{ .values = raw[1..] }, .{ .values = &raw }, .{ .values = raw[1..] }, &work, .checked);
    for (0..6) |j| {
        try E.expectApproxEqAbs(expected[j].re, raw[2 * j], 1e-9);
        try E.expectApproxEqAbs(expected[j].im, raw[2 * j + 1], 1e-9);
    }
    var ri: [12]f64 = @splat(1);
    var ii: [12]f64 = @splat(0);
    var out: [12]f64 = @splat(777);
    var other: [12]f64 = @splat(888);
    try E.expectError(error.AliasingNotAllowed, p.executePolicy(C, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &out }, .{ .values = &out }, &work, .checked));
    for (out) |x| try E.expectEqual(@as(f64, 777), x);
    ii[10] = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.executePolicy(C, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &out }, .{ .values = &other }, &work, .checked));
    for (out) |x| try E.expectEqual(@as(f64, 777), x);
    for (other) |x| try E.expectEqual(@as(f64, 888), x);
    ii = @splat(0);
    ri = @splat(std.math.floatMax(f64));
    try E.expectError(error.NonFiniteResult, p.executePolicy(C, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &out }, .{ .values = &other }, &work, .checked));
    for (out) |x| try E.expectEqual(@as(f64, 777), x);
    for (other) |x| try E.expectEqual(@as(f64, 888), x);
}
test "Neutral split complex shared scratch short planes OOM empty and borrowed close preserve ownership" {
    const p = try F.core.SplitComplexPlan.init(.{ .shape = &.{3}, .input_strides = &.{1}, .output_strides = &.{1}, .batch_shape = &.{}, .input_batch_strides = &.{}, .output_batch_strides = &.{}, .direction = .forward });
    var ri: [3]f64 = @splat(1);
    var ii: [3]f64 = @splat(0);
    var ro: [3]f64 = @splat(777);
    var io: [3]f64 = @splat(888);
    var work: [64]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.executePolicy(C, .{ .values = &ri }, .{ .values = ii[0..2] }, .{ .values = &ro }, .{ .values = &io }, &work, .checked));
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    try E.expectError(error.OutOfMemory, F.Managed(Execution).runSplitScratch(C, owner, p, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &ro }, .{ .values = &io }, .checked));
    var op = try owner.begin();
    _ = owner.requestClose();
    try E.expectError(error.Closing, F.Managed(Execution).runSplitScratch(C, owner, p, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &ro }, .{ .values = &io }, .checked));
    try F.Managed(Execution).runSplitBorrowedAdmitted(C, &op, p, .{ .values = &ri }, .{ .values = &ii }, .{ .values = &ro }, .{ .values = &io }, &work, .checked);
    try E.expectApproxEqAbs(@as(f64, 3), ro[0], 1e-12);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    var layout = p.layout;
    layout.batch_shape = &.{0};
    layout.input_batch_strides = &.{0};
    layout.output_batch_strides = &.{0};
    const empty = try F.core.SplitComplexPlan.init(layout);
    try empty.executePolicy(C, .{ .values = &.{} }, .{ .values = &.{} }, .{ .values = &.{} }, .{ .values = &.{} }, &.{}, .checked);
}

test "Neutral split real batches half spectra inverse normalization and IEEE endpoints share leaves" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const fwd = try F.core.SplitRealPlan.init(.{ .mode = .forward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{2}, .output_strides = &.{3}, .batch_shape = &.{2}, .input_batch_strides = &.{1}, .output_batch_strides = &.{8} });
        const bwd = try F.core.SplitRealPlan.init(.{ .mode = .backward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{3}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{8}, .output_batch_strides = &.{1} });
        var input: [6]R = @splat(1);
        var re: [12]R = @splat(777);
        var im: [12]R = @splat(888);
        var out: [6]R = @splat(777);
        var work: [128]F.core.WorkspaceScalar(Z) = undefined;
        try fwd.executePolicy(Z, .forward, .{ .values = &input }, .{ .values = &.{} }, .{ .values = &re }, .{ .values = &im }, &work, .checked);
        try E.expectApproxEqAbs(@as(R, 3), re[0], 1e-5);
        try E.expectApproxEqAbs(@as(R, 3), re[8], 1e-5);
        try E.expectEqual(@as(R, 777), re[1]);
        try E.expectEqual(@as(R, 888), im[1]);
        im[0] = std.math.nan(R);
        im[8] = std.math.inf(R);
        try E.expectError(error.NonFiniteInput, bwd.executePolicy(Z, .backward, .{ .values = &re }, .{ .values = &im }, .{ .values = &out }, .{ .values = &.{} }, &work, .checked));
        for (out) |x| try E.expectEqual(@as(R, 777), x);
        try bwd.executePolicy(Z, .backward, .{ .values = &re }, .{ .values = &im }, .{ .values = &out }, .{ .values = &.{} }, &work, .ieee);
        for (out) |x| try E.expectApproxEqAbs(@as(R, 3), x, 1e-5);
    }
}
test "Neutral split real overlapping planes preserve complete gather collision and overflow rollback" {
    const p = try F.core.SplitRealPlan.init(.{ .mode = .forward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{1}, .output_strides = &.{2}, .batch_shape = &.{2}, .input_batch_strides = &.{3}, .output_batch_strides = &.{4} });
    var raw: [12]f64 = @splat(777);
    for (raw[0..6], 0..) |*x, j| x.* = @floatFromInt(j + 1);
    var work: [128]F.core.WorkspaceScalar(C) = undefined;
    try p.executePolicy(C, .forward, .{ .values = &raw }, .{ .values = &.{} }, .{ .values = &raw }, .{ .values = raw[1..] }, &work, .checked);
    try E.expectApproxEqAbs(@as(f64, 6), raw[0], 1e-12);
    try E.expectApproxEqAbs(@as(f64, 15), raw[4], 1e-12);
    for (raw[8..]) |x| try E.expectEqual(@as(f64, 777), x);
    var input: [6]f64 = @splat(std.math.floatMax(f64));
    var ro: [8]f64 = @splat(777);
    var io: [8]f64 = @splat(888);
    try E.expectError(error.AliasingNotAllowed, p.executePolicy(C, .forward, .{ .values = &input }, .{ .values = &.{} }, .{ .values = &ro }, .{ .values = &ro }, &work, .checked));
    try E.expectError(error.NonFiniteResult, p.executePolicy(C, .forward, .{ .values = &input }, .{ .values = &.{} }, .{ .values = &ro }, .{ .values = &io }, &work, .checked));
    for (ro) |x| try E.expectEqual(@as(f64, 777), x);
    for (io) |x| try E.expectEqual(@as(f64, 888), x);
}
test "Neutral split real shared scratch zero batch short plane and captured borrowed close remain managed" {
    const p = try F.core.SplitRealPlan.init(.{ .mode = .forward, .shape = &.{3}, .half_shape = &.{2}, .input_strides = &.{1}, .output_strides = &.{1}, .batch_shape = &.{}, .input_batch_strides = &.{}, .output_batch_strides = &.{} });
    var ri: [3]f64 = @splat(1);
    var ro: [2]f64 = @splat(777);
    var io: [2]f64 = @splat(888);
    var work: [64]F.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, p.executePolicy(C, .forward, .{ .values = &ri }, .{ .values = &.{} }, .{ .values = &ro }, .{ .values = io[0..1] }, &work, .checked));
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    try E.expectError(error.OutOfMemory, F.Managed(Execution).runSplitRealScratch(C, .forward, owner, p, .{ .values = &ri }, .{ .values = &.{} }, .{ .values = &ro }, .{ .values = &io }, .checked));
    var op = try owner.begin();
    _ = owner.requestClose();
    try F.Managed(Execution).runSplitRealBorrowedAdmitted(C, .forward, &op, p, .{ .values = &ri }, .{ .values = &.{} }, .{ .values = &ro }, .{ .values = &io }, &work, .checked);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    var layout = p.layout;
    layout.batch_shape = &.{0};
    layout.input_batch_strides = &.{0};
    layout.output_batch_strides = &.{0};
    const empty = try F.core.SplitRealPlan.init(layout);
    try empty.executePolicy(C, .forward, .{ .values = &.{} }, .{ .values = &.{} }, .{ .values = &.{} }, .{ .values = &.{} }, &.{}, .checked);
}

test "Neutral split real multidimensional signed layouts match independent full complex DFT" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |Z| {
        const R = Core.internal.numerics.Scalar.Real(Z);
        const forward = try F.core.SplitRealPlan.init(.{ .mode = .forward, .shape = &.{ 2, 3 }, .half_shape = &.{ 2, 2 }, .input_strides = &.{ 9, -2 }, .output_strides = &.{ 5, 2 }, .batch_shape = &.{2}, .input_batch_strides = &.{20}, .output_batch_strides = &.{12} });
        const backward = try F.core.SplitRealPlan.init(.{ .mode = .backward, .shape = &.{ 2, 3 }, .half_shape = &.{ 2, 2 }, .input_strides = &.{ 5, 2 }, .output_strides = &.{ 9, -2 }, .batch_shape = &.{2}, .input_batch_strides = &.{12}, .output_batch_strides = &.{20} });
        var input: [34]R = @splat(777);
        var re: [20]R = @splat(777);
        var im: [20]R = @splat(888);
        var output: [34]R = @splat(777);
        var logical: [12]Z = undefined;
        var reference: [12]Z = undefined;
        var work: [256]F.core.WorkspaceScalar(Z) = undefined;
        const ig = forward.layout.geometry(false);
        const og = forward.layout.geometry(true);
        for (0..12) |j| {
            const x: R = @floatFromInt(j + 1);
            logical[j] = .{ .re = x, .im = 0 };
            input[@intCast(4 + try ig.offset(j, false))] = x;
        }
        for (0..2) |b| tensorReference(Z, &.{ 2, 3 }, logical[b * 6 ..][0..6], reference[b * 6 ..][0..6], -1);
        try forward.executePolicy(Z, .forward, .{ .values = &input, .origin = 4 }, .{ .values = &.{} }, .{ .values = &re }, .{ .values = &im }, &work, .checked);
        var touched: [20]bool = @splat(false);
        for (0..8) |j| {
            const k = (j / 4) * 6 + ((j % 4) / 2) * 3 + j % 2;
            const ix: usize = @intCast(try og.offset(j, true));
            touched[ix] = true;
            try E.expectApproxEqAbs(reference[k].re, re[ix], if (Z == C) 1e-9 else 2e-4);
            try E.expectApproxEqAbs(reference[k].im, im[ix], if (Z == C) 1e-9 else 2e-4);
        }
        for (re, im, touched) |r, i, used| if (!used) {
            try E.expectEqual(@as(R, 777), r);
            try E.expectEqual(@as(R, 888), i);
        };
        try backward.executePolicy(Z, .backward, .{ .values = &re }, .{ .values = &im }, .{ .values = &output, .origin = 4 }, .{ .values = &.{} }, &work, .checked);
        for (0..12) |j| try E.expectApproxEqAbs(logical[j].re * 6, output[@intCast(4 + try ig.offset(j, false))], if (Z == C) 1e-9 else 2e-4);
    }
}

test "FFTW split guru widths preserve original arrays and fixed plane separation in new arrays" {
    inline for (.{ B.IODim, B.IODim64 }) |D| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [8]f64 = .{ 1, 0, 1, 0, 1, 0, 777, 888 };
        var output: [8]f64 = @splat(777);
        var next_input = input;
        var next_output: [8]f64 = @splat(777);
        var dim: [1]D = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
        const h = B.createGuruSplitWithContext(D, owner, .complex, 1, &dim, 0, null, &input[0], &input[1], &output[0], &output[1], 64).?;
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        dim[0].n = 99;
        var op = try owner.begin();
        try B.executeAdmitted(h, &op);
        op.end();
        try E.expectApproxEqAbs(@as(f64, 3), output[0], 1e-12);
        try B.executeSplitNewArrays(h, .complex, &next_input[0], &next_input[1], &next_output[0], &next_output[1]);
        try E.expectApproxEqAbs(@as(f64, 3), next_output[0], 1e-12);
        try E.expectEqual(@as(f64, 777), next_output[6]);
        next_output = @splat(777);
        try E.expectError(error.InvalidArgument, B.executeSplitNewArrays(h, .complex, &next_input[0], &next_input[2], &next_output[0], &next_output[1]));
        for (next_output) |x| try E.expectEqual(@as(f64, 777), x);
        _ = owner.requestClose();
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}
test "FFTW split real handles reuse one captured owner and family exact original execution" {
    inline for (.{ B.SplitTransform.real_forward, .real_backward }) |kind| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var input: [3]f64 = @splat(1);
        var re: [2]f64 = .{ 3, 0 };
        var im: [2]f64 = @splat(0);
        var out: [3]f64 = @splat(777);
        const h = if (kind == .real_forward)
            B.createGuruSplitWithContext(B.IODim64, owner, kind, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &input, null, &re, &im, 64).?
        else
            B.createGuruSplitWithContext(B.IODim64, owner, kind, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &re, &im, &out, null, 64).?;
        var op = try owner.begin();
        try B.executeAdmitted(h, &op);
        op.end();
        if (kind == .real_forward) try E.expectApproxEqAbs(@as(f64, 3), re[0], 1e-12) else for (out) |x| try E.expectApproxEqAbs(@as(f64, 3), x, 1e-12);
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW split acquisition OOM output collision and captured callback close release exact resources" {
    inline for (.{ @as(usize, 0), 1 }) |budget| {
        var probe: Probe = .{ .backing = .{ .budget = budget } };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        var ri: [3]f64 = @splat(1);
        var ii: [3]f64 = @splat(0);
        var ro: [3]f64 = @splat(777);
        var io: [3]f64 = @splat(888);
        try E.expect(B.createGuruSplitWithContext(B.IODim64, owner, .complex, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &ri, &ii, &ro, &io, 64) == null);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
    var probe: Probe = .{ .close_on_alloc = true };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var ri: [3]f64 = @splat(1);
    var ii: [3]f64 = @splat(0);
    var ro: [3]f64 = @splat(777);
    var io: [3]f64 = @splat(888);
    var op = try owner.begin();
    const h = B.createGuruSplitWithContext(B.IODim64, owner, .complex, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &ri, &ii, &ro, &io, 64).?;
    try E.expectError(error.Closing, B.executeSplitWithArrays(h, &ri, &ii, &ro, &io));
    try B.executeAdmitted(h, &op);
    try E.expectApproxEqAbs(@as(f64, 3), ro[0], 1e-12);
    B.destroy(h);
    op.end();
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    var other: Probe = .{};
    var reference_other_owner: FixtureReferenceContext = undefined;
    reference_other_owner.initInPlace(other.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other_owner = reference_other_owner.borrowedContext();
    other.owner = other_owner;
    try E.expect(B.createGuruSplitWithContext(B.IODim64, other_owner, .complex, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &ri, &ii, &ro, &ro, 64) == null);
    try E.expectEqual(@as(usize, 0), other_owner.snapshot().resource_pins);
    try finish(other_owner);
    try E.expectEqual(other.backing.stats().accepted, other.backing.stats().freed);
}
test "FFTW split empty and unbound metadata stay admitted without original array access" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    const empty = B.createGuruSplitWithContext(B.IODim64, owner, .complex, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 1, &.{.{ .n = 0, .input_stride = 0, .output_stride = 0 }}, null, null, null, null, 64).?;
    var op = try owner.begin();
    try B.executeAdmitted(empty, &op);
    op.end();
    B.destroy(empty);
    const unbound = B.createGuruSplitWithContext(B.IODim64, owner, .complex, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, null, null, null, null, 64).?;
    op = try owner.begin();
    try E.expectError(error.InvalidArgument, B.executeAdmitted(unbound, &op));
    op.end();
    var ri: [3]f64 = @splat(1);
    var ii: [3]f64 = @splat(0);
    try B.executeSplitNewArrays(unbound, .complex, &ri, &ii, &ri, &ii);
    try E.expectApproxEqAbs(@as(f64, 3), ri[0], 1e-12);
    B.destroy(unbound);
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW complex cross format execution preserves interleaved separation and shared admission" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var input: [6]f64 = .{ 1, 0, 1, 0, 1, 0 };
    var output: [6]f64 = @splat(777);
    const ordinary = B.createWithContext(owner, .complex, .{ 3, 1, 1 }, &input, &output, -1, 64).?;
    try B.executeSplitNewArrays(ordinary, .complex, &input[0], &input[1], &output[0], &output[1]);
    try E.expectApproxEqAbs(@as(f64, 3), output[0], 1e-12);
    B.destroy(ordinary);
    output = @splat(777);
    const split = B.createGuruSplitWithContext(B.IODim, owner, .complex, 1, &.{.{ .n = 3, .input_stride = 2, .output_stride = 2 }}, 0, null, &input[0], &input[1], &output[0], &output[1], 64).?;
    var op = try owner.begin();
    try B.executeNewArraysAdmitted(split, .complex, &op, &input, &output);
    op.end();
    try E.expectApproxEqAbs(@as(f64, 3), output[0], 1e-12);
    output = @splat(777);
    try E.expectError(error.InvalidArgument, B.executeSplitNewArrays(split, .real_forward, &input[0], null, &output[0], &output[1]));
    B.destroy(split);
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW split multidimensional paired inplace real plans retain native accepted layouts" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var re: [6]f64 = @splat(1);
    var im: [4]f64 = @splat(777);
    const fwd = B.createGuruSplitWithContext(B.IODim64, owner, .real_forward, 2, &.{ .{ .n = 2, .input_stride = 3, .output_stride = 2 }, .{ .n = 3, .input_stride = 1, .output_stride = 1 } }, 0, null, &re, null, &re, &im, 66).?;
    var op = try owner.begin();
    try B.executeAdmitted(fwd, &op);
    op.end();
    try E.expectApproxEqAbs(@as(f64, 6), re[0], 1e-12);
    for (re[1..4]) |x| try E.expectApproxEqAbs(@as(f64, 0), x, 1e-12);
    for (im) |x| try E.expectApproxEqAbs(@as(f64, 0), x, 1e-12);
    B.destroy(fwd);
    const back = B.createGuruSplitWithContext(B.IODim64, owner, .real_backward, 2, &.{ .{ .n = 2, .input_stride = 2, .output_stride = 3 }, .{ .n = 3, .input_stride = 1, .output_stride = 1 } }, 0, null, &re, &im, &re, null, 66).?;
    op = try owner.begin();
    try B.executeAdmitted(back, &op);
    op.end();
    for (re) |x| try E.expectApproxEqAbs(@as(f64, 6), x, 1e-12);
    B.destroy(back);
    try finish(owner);
    try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}

test "FFTW real and byte aligned allocation retain captured cleanup zero size and overflow policy" {
    inline for (.{ @as(usize, 0), 1, 17, 137 }) |count| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        const real = B.allocRealWithContext(owner, count).?;
        try E.expectEqual(@as(usize, 0), @intFromPtr(real) % 64);
        const values = @as([*]f64, @ptrCast(@alignCast(real)))[0..count];
        for (values, 0..) |*v, j| v.* = @floatFromInt(j + 1);
        var replacement: Probe = .{};
        replacement.owner = owner;
        fixtureReplaceAllocator(&reference_owner, replacement.allocator());
        _ = owner.requestClose();
        try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
        B.free(real);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
        try E.expectEqual(@as(usize, 0), replacement.backing.stats().accepted);
    }
    var oom: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(oom.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    oom.owner = ctx;
    try E.expect(B.allocRealWithContext(ctx, std.math.maxInt(usize)) == null);
    try E.expectEqual(@as(usize, 0), oom.backing.stats().attempts);
    try E.expect(B.allocRealWithContext(ctx, 17) == null);
    try E.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try finish(ctx);
}
test "FFTW basic tensor and r2r unbound ESTIMATE plans require real new arrays and keep ownership" {
    inline for (.{ 0, 1, 2 }) |api| {
        var probe: Probe = .{};
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        probe.owner = owner;
        const h = (switch (api) {
            0 => B.createWithContext(owner, .complex, .{ 3, 1, 1 }, null, null, -1, 64),
            1 => B.createTensorWithContext(owner, .complex, 1, &.{3}, null, null, -1, 64),
            2 => B.createTensorRealWithContext(owner, 1, &.{3}, &.{2}, null, null, 64),
            else => unreachable,
        }).?;
        var op = try owner.begin();
        try E.expectError(error.InvalidArgument, B.executeAdmitted(h, &op));
        op.end();
        var values: [3]C = @splat(.{ .re = 1, .im = 0 });
        var real: [3]f64 = @splat(1);
        if (api == 2) try B.executeNewArrays(h, .real_transform, &real, &real) else try B.executeNewArrays(h, .complex, &values, &values);
        if (api == 2) try E.expectApproxEqAbs(@as(f64, 3), real[0], 1e-12) else try E.expectApproxEqAbs(@as(f64, 3), values[0].re, 1e-12);
        B.destroy(h);
        try finish(owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    }
}

test "FFTW byte real allocation and unbound metadata callback close retain their admitted work" {
    var allocation: Probe = .{ .close_on_alloc = true };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(allocation.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    allocation.owner = owner;
    const buffer = B.allocRealWithContext(owner, 17).?;
    try E.expectEqual(context.CloseResult.waiting_resources, fixtureProgressClose(owner));
    B.free(buffer);
    try finish(owner);
    try E.expectEqual(allocation.backing.stats().accepted, allocation.backing.stats().freed);
    var plan_probe: Probe = .{ .close_on_alloc = true };
    var reference_plan_owner: FixtureReferenceContext = undefined;
    reference_plan_owner.initInPlace(plan_probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const plan_owner = reference_plan_owner.borrowedContext();
    plan_probe.owner = plan_owner;
    var op = try plan_owner.begin();
    const h = B.createTensorWithContext(plan_owner, .complex, 1, &.{3}, null, null, -1, 64).?;
    var values: [3]C = @splat(.{ .re = 1, .im = 0 });
    try E.expectError(error.Closing, B.executeNewArrays(h, .complex, &values, &values));
    try B.executeNewArraysAdmitted(h, .complex, &op, &values, &values);
    try E.expectApproxEqAbs(@as(f64, 3), values[0].re, 1e-12);
    B.destroy(h);
    op.end();
    try finish(plan_owner);
    try E.expectEqual(plan_probe.backing.stats().accepted, plan_probe.backing.stats().freed);
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
