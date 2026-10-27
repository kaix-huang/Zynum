const std = @import("std");
const Core = @import("zynum-core");
const facade = Core.internal.numerics.Hermitian;
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
            if (fixtureProgressClose(self.owner) != .busy) @panic("Hermitian allocation is not admitted");
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
        if (fixtureProgressClose(self.owner) == .closed) @panic("Hermitian close passed managed free");
        self.backing.allocator().rawFree(data, a, ret);
    }
};
fn finish(owner: *context.Context) !void {
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.closed, fixtureProgressClose(owner));
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
fn request(a: []const C, values: []f64) facade.Request(C) {
    return .{ .a = .{ .values = a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .eigenvalues = values };
}
const input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -1 }, .{ .re = 0, .im = 1 }, .{ .re = 2, .im = 0 } };
test "generalized failed-vector recovery preserves the existing default and admits a reduced-basis policy" {
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |T| {
        const R = Core.internal.numerics.Scalar.Real(T);
        const eps = std.math.floatEps(R) * 2048;
        const roots = [_]R{ 2, 4, 8, 16 };
        var a: [16]T = undefined;
        var b: [16]T = @splat(.{ .re = 0, .im = 0 });
        for (0..4) |j| for (0..4) |i| {
            a[i + j * 4] = .{ .re = if (i == j) @floatFromInt(4 + i) else @as(R, 1) / @as(R, @floatFromInt(1 + i + j)), .im = if (i == j) 0 else if (i < j) 0.25 else -0.25 };
            if (i == j) b[i + j * 4].re = roots[i] * roots[i];
        };
        var alloc: fixture.Allocations = .{ .budget = 0 };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(alloc.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        var op = try owner.begin();
        _ = owner.requestClose();
        var work: [64]facade.core.WorkspaceScalar(T) = undefined;
        inline for (comptime std.meta.tags(facade.core.Problem)) |problem| {
            var reduced = a;
            for (0..4) |j| for (0..4) |i| {
                const scale = if (problem == .ax_lambda_bx) 1 / (roots[i] * roots[j]) else roots[i] * roots[j];
                reduced[i + j * 4].re *= scale;
                reduced[i + j * 4].im *= scale;
            };
            var standard_values: [4]R = undefined;
            var standard_vectors: [16]T = undefined;
            var values: [4]R = undefined;
            var vectors: [16]T = undefined;
            var default_values: [4]R = undefined;
            var default_vectors: [16]T = undefined;
            var req: facade.Request(T) = .{
                .a = .{ .values = &reduced, .row_count = 4, .column_count = 4, .leading_dimension = 4 },
                .eigenvalues = &standard_values,
                .vectors = .{ .values = &standard_vectors, .row_count = 4, .column_count = 4, .leading_dimension = 4 },
                .contract = .heev_ieee_v1,
                .options = .{ .max_sweeps = 0 },
            };
            const standard = try M.runBorrowedAdmitted(T, &op, req, &work);
            try E.expectEqual(facade.core.Status.iteration_limit, standard.status);
            try E.expect(standard.unconverged >= 2);
            req.a.values = &a;
            req.b = .{ .values = &b, .row_count = 4, .column_count = 4, .leading_dimension = 4 };
            req.problem = problem;
            req.eigenvalues = &values;
            req.vectors.?.values = &vectors;
            req.options.generalized_failure_vectors = .keep_reduced;
            const kept = try M.runBorrowedAdmitted(T, &op, req, &work);
            try E.expectEqual(standard.status, kept.status);
            try E.expectEqualDeep(standard.failed_block, kept.failed_block);
            try E.expectEqual(standard.unconverged, kept.unconverged);
            req.eigenvalues = &default_values;
            req.vectors.?.values = &default_vectors;
            req.options.generalized_failure_vectors = .backtransform_available;
            const recovered = try M.runBorrowedAdmitted(T, &op, req, &work);
            try E.expectEqual(kept.status, recovered.status);
            for (0..4) |j| {
                try E.expectApproxEqRel(standard_values[j], values[j], eps);
                try E.expectEqual(values[j], default_values[j]);
                for (0..4) |i| {
                    try E.expectApproxEqAbs(standard_vectors[i + j * 4].re, vectors[i + j * 4].re, eps);
                    try E.expectApproxEqAbs(standard_vectors[i + j * 4].im, vectors[i + j * 4].im, eps);
                    const scale: R = if (j >= kept.unconverged - 1) 1 else if (problem == .bax_lambda_x) roots[i] else 1 / roots[i];
                    try E.expectApproxEqAbs(vectors[i + j * 4].re * scale, default_vectors[i + j * 4].re, eps * @max(1, scale));
                    try E.expectApproxEqAbs(vectors[i + j * 4].im * scale, default_vectors[i + j * 4].im, eps * @max(1, scale));
                }
            }
        }
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        op.end();
        try finish(owner);
    }
}
test "Hermitian scratch uses exact shared owner through closing allocation and free" {
    var probe: Probe = .{ .close_on_alloc = true };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var reference_other: FixtureReferenceContext = undefined;
    reference_other.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other = reference_other.borrowedContext();
    var values: [2]f64 = @splat(123);
    const r = try M.runScratch(C, owner, request(&input, &values));
    try E.expectEqual(facade.core.Status.converged, r.status);
    try E.expectApproxEqAbs(@as(f64, 1), values[0], 1e-14);
    try E.expectApproxEqAbs(@as(f64, 3), values[1], 1e-14);
    for ([_]context.Snapshot{ probe.alloc_snapshot.?, probe.free_snapshot.? }) |snap| {
        try E.expectEqual(@as(usize, 1), snap.active_operations);
        try E.expectEqual(@as(usize, 1), snap.resource_pins);
    }
    try E.expectEqual(@as(usize, 1), probe.backing.stats().freed);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), other.snapshot().active_operations);
    try E.expectError(error.Closing, M.runScratch(C, owner, request(&input, &values)));
    try finish(owner);
    try finish(other);
}
test "Hermitian borrowed admitted work survives close with no nested admission or allocation" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var op = try owner.begin();
    _ = owner.requestClose();
    var values: [2]f64 = undefined;
    var work: [16]facade.core.WorkspaceScalar(C) = undefined;
    try E.expectEqual(facade.core.Status.converged, (try M.runBorrowedAdmitted(C, &op, request(&input, &values), &work)).status);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    try E.expectError(error.Closing, M.runBorrowed(C, owner, request(&input, &values), &work));
    op.end();
    try finish(owner);
}
test "Hermitian counter limits and size overflow precede allocation and preserve caller pins" {
    var probe: Probe = .{ .backing = .{ .budget = 0 } };
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var values: [2]f64 = @splat(123);
    owner.test_state.limit = 0;
    try E.expectError(error.CounterOverflow, M.runScratch(C, owner, request(&input, &values)));
    owner.test_state.limit = 1;
    var op = try owner.begin();
    var existing = try op.pinResource();
    try E.expectError(error.CounterOverflow, M.runScratchAdmitted(C, &op, request(&input, &values)));
    var huge = request(&input, &values);
    huge.a.row_count = std.math.maxInt(usize);
    try E.expectError(error.InvalidLength, M.runScratchAdmitted(C, &op, huge));
    try E.expectEqual(@as(usize, 0), probe.backing.stats().attempts);
    try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    existing.releaseAfterResourceFree();
    try E.expectError(error.OutOfMemory, M.runScratchAdmitted(C, &op, request(&input, &values)));
    for (values) |v| try E.expectEqual(@as(f64, 123), v);
    op.end();
    try finish(owner);
}
test "Hermitian numerical failure unwinds scratch and preserves outputs" {
    var probe: Probe = .{};
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(probe.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    probe.owner = owner;
    var values: [2]f64 = @splat(123);
    var a = input;
    a[1].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, M.runScratch(C, owner, request(&a, &values)));
    var req = request(&input, &values);
    req.options.max_sweeps = 0;
    try E.expectEqual(facade.core.Status.iteration_limit, (try M.runScratch(C, owner, req)).status);
    const bad = [_]C{ .{ .re = -1, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 } };
    req = request(&input, &values);
    req.b = request(&bad, &values).a;
    try E.expectEqual(facade.core.Status.not_positive_definite, (try M.runScratch(C, owner, req)).status);
    for (values) |v| try E.expectEqual(@as(f64, 123), v);
    try E.expectEqual(@as(usize, 3), probe.backing.stats().freed);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try finish(owner);
}
test "Hermitian standard and all generalized problems borrow actual Core scalar identities" {
    const actual = @import("zynum-execution");
    comptime {
        if (Execution != actual) @compileError("Hermitian execution owner mismatch");
    }
    try E.expect(Execution.getContext() == actual.getContext());
    inline for (.{ Core.internal.numerics.Scalar.ComplexF32, C }) |T| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        const real = @TypeOf(@as(T, undefined).re);
        const a = [_]T{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -1 }, .{ .re = 0, .im = 1 }, .{ .re = 2, .im = 0 } };
        const b = [_]T{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 } };
        var values: [2]real = undefined;
        var vectors: [4]T = undefined;
        var req: facade.Request(T) = .{ .a = .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .eigenvalues = &values, .vectors = .{ .values = &vectors, .row_count = 2, .column_count = 2, .leading_dimension = 2 } };
        try E.expectEqual(facade.core.Status.converged, (try M.runScratch(T, owner, req)).status);
        try E.expectApproxEqAbs(@as(real, 1), values[0], 1e-6);
        req.b = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        inline for (comptime std.meta.tags(facade.core.Problem)) |problem| {
            req.problem = problem;
            try E.expectEqual(facade.core.Status.converged, (try M.runScratch(T, owner, req)).status);
            const scale: real = if (problem == .ax_lambda_bx) 0.5 else 2;
            try E.expectApproxEqAbs(scale, values[0], 1e-6);
            try E.expectApproxEqAbs(3 * scale, values[1], 1e-6);
            var gram: real = 0;
            for (vectors[0..2]) |z| gram += z.re * z.re + z.im * z.im;
            try E.expectApproxEqAbs(@as(real, if (problem == .bax_lambda_x) 2 else 0.5), gram, 1e-6);
        }
        try finish(owner);
    }
}

test "Hermitian range failure and invalid borrowed workspace preserve output and admission" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var op = try owner.begin();
    var values: [2]f64 = @splat(123);
    var work: [16]facade.core.WorkspaceScalar(C) = undefined;
    try E.expectError(error.BufferTooSmall, M.runBorrowedAdmitted(C, &op, request(&input, &values), work[0..1]));
    const a = [_]C{ .{ .re = 1e300, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1e300, .im = 0 } };
    const b = [_]C{ .{ .re = 1e-300, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1e-300, .im = 0 } };
    var req = request(&a, &values);
    req.b = request(&b, &values).a;
    try E.expectEqual(facade.core.Status.range_limit, (try M.runScratchAdmitted(C, &op, req)).status);
    for (values) |v| try E.expectEqual(@as(f64, 123), v);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try finish(owner);
}
const Worker = struct {
    owner: *context.Context,
    passed: bool = false,
    fn run(self: *@This()) void {
        var values: [2]f64 = undefined;
        for (0..12) |_| {
            const r = M.runScratch(C, self.owner, request(&input, &values)) catch return;
            if (r.status != .converged or @abs(values[0] - 1) > 1e-14 or @abs(values[1] - 3) > 1e-14) return;
        }
        self.passed = true;
    }
};
test "Hermitian concurrent callers use isolated scratch on one shared execution context" {
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.heap.c_allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var workers: [6]Worker = undefined;
    var threads: [6]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&workers, &threads) |*worker, *thread| {
        worker.* = .{ .owner = owner };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
        started += 1;
    }
    for (threads) |thread| thread.join();
    started = 0;
    for (workers) |worker| try E.expect(worker.passed);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try finish(owner);
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
