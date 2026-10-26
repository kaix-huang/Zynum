const std = @import("std");
const Core = @import("zynum-core");
const protocol = @import("numerical-owner").hermitian_protocol;
const A = protocol.Adapter(Core);
const Ex = Core.internal.numerics.Execution;
const context = Ex.context;
const fixture = Ex.test_support;
const E = std.testing;
const C = Core.internal.numerics.Scalar.ComplexF64;
fn finish(owner: *context.Context) !void {
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.closed, owner.tryProgressClose());
}
test "Hermitian query validates scalar precedence and reads no matrices after close" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var work: [1]C = .{.{ .re = 999, .im = 33 }};
    var args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 7, .lda = 7, .ldb = 7, .work = &work, .lwork = -1 };
    _ = owner.requestClose();
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(.heev, C, &op, args));
    try E.expectEqual(@as(f64, 13), work[0].re);
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(.hegv, C, &op, args));
    args.layout = 101;
    args.lda = 6;
    args.itype = 0;
    work[0].re = 999;
    try E.expectEqual(@as(i32, -7), try A.workAdmitted(.hegv, C, &op, args));
    try E.expectEqual(@as(f64, 999), work[0].re);
    args.lda = 7;
    try E.expectEqual(@as(i32, -2), try A.workAdmitted(.hegv, C, &op, args));
    args.itype = 1;
    args.layout = 102;
    args.lwork = 1;
    try E.expectEqual(@as(i32, -12), try A.workAdmitted(.hegv, C, &op, args));
    try E.expectEqual(@as(f64, 13), work[0].re);
    args.n = 0;
    args.lda = 1;
    args.ldb = 1;
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(.hegv, C, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    op.end();
    try finish(&owner);
}
test "Hermitian scalar quick return preserves IEEE and ignores diagonal imaginary and RWORK" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var input: [1]C = .{.{ .re = std.math.inf(f64), .im = std.math.nan(f64) }};
    var values: [1]f64 = undefined;
    var work: [1]C = undefined;
    const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 1, .lda = 1, .a = &input, .w = &values, .work = &work, .lwork = 1 };
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(.heev, C, &op, args));
    try E.expect(std.math.isInf(values[0]));
    try E.expectEqual(@as(f64, 1), input[0].re);
    try E.expectEqual(@as(f64, 0), input[0].im);
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner);
}
fn at(layout: c_int, i: usize, j: usize) usize {
    return if (layout == 102) i + j * 3 else i * 3 + j;
}
fn finiteCases(comptime T: type) !void {
    const R = Core.internal.numerics.Scalar.Real(T);
    const tol: R = if (R == f32) 2e-5 else 2e-14;
    for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]u8{ 'N', 'V' }) |job| for (0..4) |problem| {
        var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
        var op = try owner.begin();
        var input: [6]T = @splat(.{ .re = 321, .im = 432 });
        var b: [6]T = input;
        input[at(layout, 0, 0)] = .{ .re = 2, .im = std.math.nan(R) };
        input[at(layout, 1, 1)] = input[at(layout, 0, 0)];
        b[at(layout, 0, 0)] = .{ .re = 2, .im = 0 };
        b[at(layout, 1, 1)] = b[at(layout, 0, 0)];
        const off = at(layout, if (upper) 0 else 1, if (upper) 1 else 0);
        const ignored = at(layout, if (upper) 1 else 0, if (upper) 0 else 1);
        input[off] = .{ .re = 0, .im = if (upper) 1 else -1 };
        b[off] = .{ .re = 0, .im = 0 };
        input[ignored] = .{ .re = std.math.nan(R), .im = std.math.nan(R) };
        b[ignored] = input[ignored];
        var work: [3]T = undefined;
        var rw: [4]R = undefined;
        var values: [2]R = undefined;
        const args: A.Arguments(T) = .{ .layout = layout, .itype = @intCast(@max(1, problem)), .jobz = job, .uplo = if (upper) 'U' else 'L', .n = 2, .a = &input, .lda = 3, .b = &b, .ldb = 3, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
        const info = if (problem == 0) try A.workAdmitted(.heev, T, &op, args) else try A.workAdmitted(.hegv, T, &op, args);
        try E.expectEqual(@as(i32, 0), info);
        const scale: R = if (problem == 0) 1 else if (problem == 1) 0.5 else 2;
        try E.expectApproxEqAbs(scale, values[0], tol);
        try E.expectApproxEqAbs(3 * scale, values[1], tol);
        if (job == 'V') for (0..2) |j| {
            const x = input[at(layout, 0, j)];
            const y = input[at(layout, 1, j)];
            const norm = x.re * x.re + x.im * x.im + y.re * y.re + y.im * y.im;
            const expected: R = if (problem == 0) 1 else if (problem == 3) 2 else 0.5;
            try E.expectApproxEqAbs(expected, norm, tol);
            const lambda = values[j] / scale;
            // A0*x = lambda*x with A0=[[2,i],[-i,2]].
            try E.expectApproxEqAbs(lambda * x.re, 2 * x.re - y.im, tol);
            try E.expectApproxEqAbs(lambda * x.im, 2 * x.im + y.re, tol);
            try E.expectApproxEqAbs(lambda * y.re, 2 * y.re + x.im, tol);
            try E.expectApproxEqAbs(lambda * y.im, 2 * y.im - x.re, tol);
        };
        try E.expectEqual(@as(R, 321), input[2].re);
        try E.expectEqual(@as(R, 321), input[5].re);
        if (problem != 0) {
            try E.expectApproxEqAbs(@sqrt(@as(R, 2)), b[0].re, tol);
            try E.expectApproxEqAbs(@sqrt(@as(R, 2)), b[4].re, tol);
            try E.expect(std.math.isNan(b[ignored].re));
            try E.expectEqual(@as(R, 321), b[2].re);
        }
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        op.end();
        try finish(&owner);
    };
}
test "Hermitian work finite residuals both precisions layouts triangles jobs and generalized types" {
    try finiteCases(Core.internal.numerics.Scalar.ComplexF32);
    try finiteCases(C);
}
test "Hermitian work SPD failure retains partial B factor and preserves A and eigenvalues" {
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    var input: [4]C = @splat(.{ .re = 2, .im = 0 });
    const before = input;
    var b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 55, .im = 7 }, .{ .re = -1, .im = 0 } };
    var values: [2]f64 = @splat(888);
    var work: [3]C = undefined;
    var rw: [4]f64 = undefined;
    const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
    try E.expectEqual(@as(i32, 4), try A.workAdmitted(.hegv, C, &op, args));
    try E.expectEqual(@as(f64, 2), b[0].re);
    try E.expectEqual(@as(f64, -1), b[3].re);
    try E.expectEqual(@as(f64, 55), b[2].re);
    try E.expectEqualSlices(C, &before, &input);
    try E.expectEqual(@as(f64, 888), values[0]);
    op.end();
    try finish(&owner);
}
test "Hermitian staging OOM releases pins and does not overwrite matrices" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var input: [4]C = @splat(.{ .re = 0, .im = 0 });
    var values: [2]f64 = @splat(99);
    var work: [3]C = undefined;
    var rw: [4]f64 = undefined;
    const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
    try E.expectError(error.OutOfMemory, A.workAdmitted(.heev, C, &op, args));
    try E.expectEqual(@as(f64, 99), values[0]);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try finish(&owner);
}

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
            if (self.owner.tryProgressClose() != .busy) @panic("Hermitian allocation is not admitted");
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
        if (self.owner.tryProgressClose() == .closed) @panic("Hermitian close passed managed free");
        self.backing.allocator().rawFree(data, a, ret);
    }
};

test "Hermitian generalized allocation stages retain B overwrite and exact shared operation through close" {
    for (0..4) |budget| {
        var probe: Probe = .{ .backing = .{ .budget = budget }, .close_on_alloc = true };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        probe.owner = &owner;
        var other = context.Context.init(E.allocator, fixture.policy, .internal_closable);
        var op = try owner.begin();
        var input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -1 }, .{ .re = 0, .im = 1 }, .{ .re = 2, .im = 0 } };
        const original = input;
        var b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 55, .im = 9 }, .{ .re = 4, .im = 0 } };
        var values: [2]f64 = @splat(99);
        var work: [3]C = undefined;
        var rw: [4]f64 = undefined;
        const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
        if (budget < 3) {
            try E.expectError(error.OutOfMemory, A.workAdmitted(.hegv, C, &op, args));
            try E.expectEqualSlices(C, &original, &input);
            try E.expectEqual(@as(f64, 99), values[0]);
        } else {
            try E.expectEqual(@as(i32, 0), try A.workAdmitted(.hegv, C, &op, args));
            try E.expectApproxEqAbs(@as(f64, 0.25), values[0], 1e-14);
        }
        try E.expectEqual(@as(f64, if (budget < 2) 4 else 2), b[0].re);
        try E.expectEqual(@as(f64, 55), b[2].re);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try E.expectEqual(@as(usize, 0), other.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), other.snapshot().resource_pins);
        try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
        if (budget > 0) try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
        try E.expectEqual(@as(usize, @min(budget, 3)), probe.backing.stats().freed);
        try E.expectError(error.Closing, owner.begin());
        op.end();
        try finish(&owner);
        try finish(&other);
    }
}
test "Hermitian typed capacity and alias faults precede allocation" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var input: [4]C = @splat(.{ .re = 0, .im = 0 });
    var values: [2]f64 = @splat(99);
    var work: [3]C = undefined;
    var rw: [4]f64 = undefined;
    var args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
    args.rwork = rw[0..3];
    try E.expectError(error.BufferTooSmall, A.workAdmitted(.heev, C, &op, args));
    args.rwork = &rw;
    args.a = input[0..3];
    try E.expectError(error.BufferTooSmall, A.workAdmitted(.heev, C, &op, args));
    args.a = &input;
    args.w = values[0..1];
    try E.expectError(error.BufferTooSmall, A.workAdmitted(.heev, C, &op, args));
    args.w = &values;
    args.work = input[0..3];
    try E.expectError(error.AliasingNotAllowed, A.workAdmitted(.heev, C, &op, args));
    args.work = &work;
    args.rwork = &values;
    try E.expectError(error.BufferTooSmall, A.workAdmitted(.heev, C, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner);
}
fn nonScalarMetric(comptime T: type) !void {
    const R = Core.internal.numerics.Scalar.Real(T);
    const tol: R = if (R == f32) 3e-5 else 4e-13;
    for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for (1..4) |problem| {
        var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
        var op = try owner.begin();
        var input: [6]T = @splat(.{ .re = 99, .im = 33 });
        var b: [6]T = input;
        input[0] = .{ .re = 2, .im = 0 };
        input[4] = .{ .re = 3, .im = 0 };
        b[0] = .{ .re = 4, .im = 0 };
        b[4] = .{ .re = 9, .im = 0 };
        const off = at(layout, if (upper) 0 else 1, if (upper) 1 else 0);
        input[off] = .{ .re = 0, .im = if (upper) 1 else -1 };
        b[off] = .{ .re = 0, .im = 0 };
        var work: [3]T = undefined;
        var rw: [4]R = undefined;
        var values: [2]R = undefined;
        const args: A.Arguments(T) = .{ .layout = layout, .itype = @intCast(problem), .jobz = 'V', .uplo = if (upper) 'U' else 'L', .n = 2, .a = &input, .lda = 3, .b = &b, .ldb = 3, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(.hegv, T, &op, args));
        const expected = if (problem == 1) [2]R{ (5 - @sqrt(@as(R, 5))) / 12, (5 + @sqrt(@as(R, 5))) / 12 } else [2]R{ (35 - @sqrt(@as(R, 505))) / 2, (35 + @sqrt(@as(R, 505))) / 2 };
        for (0..2) |j| {
            try E.expectApproxEqAbs(expected[j], values[j], tol);
            const x = input[at(layout, 0, j)];
            const y = input[at(layout, 1, j)];
            const dx: R = if (problem == 2) 4 else 1;
            const dy: R = if (problem == 2) 9 else 1;
            const lx: R = if (problem == 3) 4 else 1;
            const ly: R = if (problem == 3) 9 else 1;
            const rx: R = if (problem == 1) 4 else 1;
            const ry: R = if (problem == 1) 9 else 1;
            try E.expectApproxEqAbs(values[j] * rx * x.re, lx * (2 * dx * x.re - dy * y.im), tol);
            try E.expectApproxEqAbs(values[j] * rx * x.im, lx * (2 * dx * x.im + dy * y.re), tol);
            try E.expectApproxEqAbs(values[j] * ry * y.re, ly * (3 * dy * y.re + dx * x.im), tol);
            try E.expectApproxEqAbs(values[j] * ry * y.im, ly * (3 * dy * y.im - dx * x.re), tol);
            const metricx: R = if (problem == 3) 0.25 else 4;
            const metricy: R = if (problem == 3) 1.0 / 9.0 else 9;
            try E.expectApproxEqAbs(@as(R, 1), metricx * (x.re * x.re + x.im * x.im) + metricy * (y.re * y.re + y.im * y.im), tol);
        }
        try E.expectEqual(@as(R, 2), b[0].re);
        try E.expectEqual(@as(R, 3), b[4].re);
        op.end();
        try finish(&owner);
    };
}
test "Hermitian generalized non scalar B verifies original equations and metric normalization" {
    try nonScalarMetric(Core.internal.numerics.Scalar.ComplexF32);
    try nonScalarMetric(C);
}
test "Hermitian query single precision rounds up and LP64 workspace overflow is explicit" {
    const F = Core.internal.numerics.Scalar.ComplexF32;
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var work: [1]F = undefined;
    var args: A.Arguments(F) = .{ .layout = 102, .jobz = 'N', .uplo = 'U', .n = 8388609, .lda = 8388609, .work = &work, .lwork = -1 };
    try E.expectEqual(@as(i32, 0), try A.workAdmitted(.heev, F, &op, args));
    try E.expectEqual(@as(f32, 16777218), work[0].re);
    args.n = std.math.maxInt(i32);
    args.lda = args.n;
    try E.expectError(error.IntegerNotRepresentable, A.workAdmitted(.heev, F, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner);
}

const Owner = @import("numerical-owner");
const Raw = protocol.RawAdapter(Core);
const ServiceProbe = struct {
    enabled: bool = true,
    nan_calls: usize = 0,
    events: [8]Owner.protocol.ErrorEvent = undefined,
    count: usize = 0,
    owner: *context.Context,
    pin_snapshot: usize = 999,
    fn nan(ptr: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.nan_calls += 1;
        return self.enabled;
    }
    fn event(ptr: *anyopaque, value: Owner.protocol.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.events[self.count] = value;
        self.count += 1;
        self.pin_snapshot = self.owner.snapshot().resource_pins;
        if (self.owner.snapshot().active_operations != 1) @panic("diagnostic lost shared operation");
    }
    fn services(self: *@This()) Owner.Provider.Services {
        return .{ .allocator = undefined, .context = self, .nan_enabled = nan, .error_event = event };
    }
};
test "Hermitian raw queries do not access invalid matrix addresses and report alignment faults" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var query: [1]C = undefined;
    var args: Raw.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .a = @ptrFromInt(1), .b = @ptrFromInt(1), .w = @ptrFromInt(1), .rwork = @ptrFromInt(1), .work = @ptrCast(&query), .lwork = -1 };
    try E.expectEqual(@as(i32, 0), try Raw.workAdmitted(.hegv, C, &op, args));
    args.work = @ptrFromInt(1);
    try E.expectError(error.MisalignedPointer, Raw.workAdmitted(.heev, C, &op, args));
    args.work = null;
    try E.expectError(error.MissingBuffer, Raw.workAdmitted(.heev, C, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner);
}
test "Hermitian high NaN order includes diagonal imaginary and ignores opposite triangle" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var probe: ServiceProbe = .{ .owner = &owner };
    var services = probe.services();
    var input = [_]C{ .{ .re = 2, .im = std.math.nan(f64) }, .{ .re = 0, .im = 0 }, .{ .re = std.math.nan(f64), .im = 0 }, .{ .re = 2, .im = 0 } };
    var b = [_]C{ .{ .re = 1, .im = std.math.nan(f64) }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 } };
    var args: A.Arguments(C) = .{ .layout = 102, .jobz = '?', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2 };
    try E.expectEqual(@as(i32, -6), try A.highAdmitted(true, .hegv, C, &op, &services, args));
    try E.expectEqual(@as(usize, 0), probe.count);
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    input[0].im = 0;
    try E.expectEqual(@as(i32, -8), try A.highAdmitted(true, .hegv, C, &op, &services, args));
    b[0].im = 0;
    // Opposite triangle NaN is not scanned; real-workspace OOM precedes JOBZ validation.
    try E.expectEqual(@as(i32, -1010), try A.highAdmitted(true, .hegv, C, &op, &services, args));
    try E.expectEqual(@as(usize, 1), probe.count);
    try E.expectEqual(Owner.protocol.Entry.high, probe.events[0].entry);
    try E.expectEqual(@as(usize, 0), probe.pin_snapshot);
    args.layout = 0;
    try E.expectEqual(@as(i32, -1), try A.highAdmitted(true, .heev, C, &op, &services, args));
    try E.expectEqual(@as(usize, 3), probe.nan_calls);
    probe.enabled = false;
    args.layout = 102;
    try E.expectEqual(@as(i32, -1010), try A.highAdmitted(true, .heev, C, &op, &services, args));
    try E.expectEqual(@as(usize, 4), probe.nan_calls);
    try E.expectEqual(@as(i32, -1010), try A.highAdmitted(false, .heev, C, &op, &services, args));
    try E.expectEqual(@as(usize, 4), probe.nan_calls);
    op.end();
    try finish(&owner);
}
test "Hermitian work diagnostics preserve Fortran argument and row-major LAPACKE channel" {
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    var probe: ServiceProbe = .{ .owner = &owner };
    var services = probe.services();
    var work: [1]C = undefined;
    var args: A.Arguments(C) = .{ .layout = 102, .itype = 0, .jobz = '?', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .work = &work, .lwork = -1 };
    try E.expectEqual(@as(i32, -2), try A.workWithDiagnosticsAdmitted(.hegv, C, &op, &services, args));
    try E.expectEqual(Owner.protocol.HookKind.fortran_returning_test, probe.events[0].kind);
    try E.expectEqual(@as(i32, 1), probe.events[0].argument);
    try E.expectEqual(Owner.protocol.Entry.fortran_hegv, probe.events[0].entry);
    args.layout = 101;
    args.lda = 1;
    try E.expectEqual(@as(i32, -7), try A.workWithDiagnosticsAdmitted(.hegv, C, &op, &services, args));
    try E.expectEqual(Owner.protocol.HookKind.lapacke, probe.events[1].kind);
    try E.expectEqual(Owner.protocol.Entry.work, probe.events[1].entry);
    try E.expectEqual(@as(i32, -7), probe.events[1].argument);
    op.end();
    try finish(&owner);
}
test "Hermitian raw high finite solve owns one operation and preserves caller padding" {
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    var probe: ServiceProbe = .{ .owner = &owner };
    var services = probe.services();
    var input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 1 }, .{ .re = 77, .im = 88 }, .{ .re = 0, .im = -1 }, .{ .re = 2, .im = 0 }, .{ .re = 77, .im = 88 } };
    var b = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 77, .im = 88 }, .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 77, .im = 88 } };
    var values: [2]f64 = undefined;
    const args: Raw.Arguments(C) = .{ .layout = 101, .jobz = 'V', .uplo = 'L', .n = 2, .a = @ptrCast(&input), .lda = 3, .b = @ptrCast(&b), .ldb = 3, .w = @ptrCast(&values), .work = @ptrFromInt(1), .rwork = @ptrFromInt(1), .lwork = 0 };
    try E.expectEqual(@as(i32, 0), try Raw.highAdmitted(true, .hegv, C, &op, &services, args));
    try E.expectApproxEqAbs(@as(f64, 0.5), values[0], 1e-14);
    try E.expectApproxEqAbs(@as(f64, 1.5), values[1], 1e-14);
    try E.expectEqual(@as(f64, 77), input[2].re);
    try E.expectEqual(@as(f64, 77), b[2].re);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), probe.count);
    op.end();
    try finish(&owner);
}
test "Hermitian high OOM cleans all nested scratch before diagnostic callback" {
    for (0..6) |budget| {
        var alloc: fixture.Allocations = .{ .budget = budget };
        var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
        var op = try owner.begin();
        var probe: ServiceProbe = .{ .owner = &owner };
        var services = probe.services();
        var input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -1 }, .{ .re = 0, .im = 1 }, .{ .re = 2, .im = 0 } };
        const original = input;
        var b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 55, .im = 9 }, .{ .re = 4, .im = 0 } };
        var values: [2]f64 = @splat(99);
        const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2, .w = &values };
        const info = try A.highAdmitted(true, .hegv, C, &op, &services, args);
        try E.expectEqual(@as(i32, if (budget < 5) -1010 else 0), info);
        if (budget < 5) {
            try E.expectEqual(@as(usize, 1), probe.count);
            try E.expectEqual(@as(usize, 0), probe.pin_snapshot);
            try E.expectEqual(Owner.protocol.Entry.high, probe.events[0].entry);
            try E.expectEqualSlices(C, &original, &input);
            try E.expectEqual(@as(f64, 99), values[0]);
        }
        try E.expectEqual(@as(f64, if (budget < 4) 4 else 2), b[0].re);
        try E.expectEqual(@as(usize, @min(budget, 5)), alloc.stats().freed);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        op.end();
        try finish(&owner);
    }
}

fn allEntryFiniteCases(comptime T: type, comptime variant: u8) !void {
    const R = Core.internal.numerics.Scalar.Real(T);
    const tol: R = if (R == f32) 2e-5 else 2e-14;
    for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]u8{ 'N', 'V' }) |job| for (0..4) |problem| {
        var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
        var op = try owner.begin();
        var input: [6]T = @splat(.{ .re = 321, .im = 432 });
        var b: [6]T = input;
        input[at(layout, 0, 0)] = .{ .re = 2, .im = std.math.nan(R) };
        input[at(layout, 1, 1)] = input[at(layout, 0, 0)];
        b[at(layout, 0, 0)] = .{ .re = 2, .im = 0 };
        b[at(layout, 1, 1)] = b[at(layout, 0, 0)];
        const off = at(layout, if (upper) 0 else 1, if (upper) 1 else 0);
        const ignored = at(layout, if (upper) 1 else 0, if (upper) 0 else 1);
        input[off] = .{ .re = 0, .im = if (upper) 1 else -1 };
        b[off] = .{ .re = 0, .im = 0 };
        input[ignored] = .{ .re = std.math.nan(R), .im = std.math.nan(R) };
        b[ignored] = input[ignored];
        var work: [3]T = undefined;
        var rw: [4]R = undefined;
        var values: [2]R = undefined;
        const args: A.Arguments(T) = .{ .layout = layout, .itype = @intCast(@max(1, problem)), .jobz = job, .uplo = if (upper) 'U' else 'L', .n = 2, .a = &input, .lda = 3, .b = &b, .ldb = 3, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
        var service_probe: ServiceProbe = .{ .owner = &owner, .enabled = false };
        var services = service_probe.services();
        const raw_args: Raw.Arguments(T) = .{ .layout = layout, .itype = args.itype, .jobz = job, .uplo = args.uplo, .n = 2, .a = @ptrCast(&input), .lda = 3, .b = @ptrCast(&b), .ldb = 3, .w = @ptrCast(&values), .work = @ptrCast(&work), .lwork = 3, .rwork = @ptrCast(&rw) };
        const info = if (variant == 1)
            (if (problem == 0) try A.highAdmitted(true, .heev, T, &op, &services, args) else try A.highAdmitted(true, .hegv, T, &op, &services, args))
        else if (variant == 2)
            (if (problem == 0) try Raw.highAdmitted(true, .heev, T, &op, &services, raw_args) else try Raw.highAdmitted(true, .hegv, T, &op, &services, raw_args))
        else
            (if (problem == 0) try Raw.workWithDiagnosticsAdmitted(.heev, T, &op, &services, raw_args) else try Raw.workWithDiagnosticsAdmitted(.hegv, T, &op, &services, raw_args));
        try E.expectEqual(@as(usize, 0), service_probe.count);
        try E.expectEqual(@as(usize, if (variant == 3) 0 else 1), service_probe.nan_calls);
        try E.expectEqual(@as(i32, 0), info);
        const scale: R = if (problem == 0) 1 else if (problem == 1) 0.5 else 2;
        try E.expectApproxEqAbs(scale, values[0], tol);
        try E.expectApproxEqAbs(3 * scale, values[1], tol);
        if (job == 'V') for (0..2) |j| {
            const x = input[at(layout, 0, j)];
            const y = input[at(layout, 1, j)];
            const norm = x.re * x.re + x.im * x.im + y.re * y.re + y.im * y.im;
            const expected: R = if (problem == 0) 1 else if (problem == 3) 2 else 0.5;
            try E.expectApproxEqAbs(expected, norm, tol);
            const lambda = values[j] / scale;
            // A0*x = lambda*x with A0=[[2,i],[-i,2]].
            try E.expectApproxEqAbs(lambda * x.re, 2 * x.re - y.im, tol);
            try E.expectApproxEqAbs(lambda * x.im, 2 * x.im + y.re, tol);
            try E.expectApproxEqAbs(lambda * y.re, 2 * y.re + x.im, tol);
            try E.expectApproxEqAbs(lambda * y.im, 2 * y.im - x.re, tol);
        };
        try E.expectEqual(@as(R, 321), input[2].re);
        try E.expectEqual(@as(R, 321), input[5].re);
        if (problem != 0) {
            try E.expectApproxEqAbs(@sqrt(@as(R, 2)), b[0].re, tol);
            try E.expectApproxEqAbs(@sqrt(@as(R, 2)), b[4].re, tol);
            try E.expect(std.math.isNan(b[ignored].re));
            try E.expectEqual(@as(R, 321), b[2].re);
        }
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        op.end();
        try finish(&owner);
    };
}

test "Hermitian typed raw high and raw work both precisions full finite entry coverage" {
    inline for (1..4) |variant| {
        try allEntryFiniteCases(Core.internal.numerics.Scalar.ComplexF32, variant);
        try allEntryFiniteCases(C, variant);
    }
}
test "Hermitian new diagnostic routine names and integer signs are exact" {
    inline for (.{ i32, i64 }) |I| {
        const D = Owner.Diagnostics(I);
        const f = try D.translate(.{ .kind = .fortran_returning_test, .entry = .fortran_hegv, .scalar = 'z', .argument = 1, .routine = .hegv });
        try E.expectEqualStrings("ZHEGV", f.name());
        try E.expectEqual(@as(I, 1), f.info);
        const c = try D.translate(.{ .kind = .lapacke, .entry = .work, .scalar = 'c', .argument = -9, .routine = .heev });
        try E.expectEqualStrings("LAPACKE_cheev_work", c.name());
        try E.expectEqual(@as(I, -9), c.info);
    }
}
test "Hermitian high solve survives closing allocations and frees with exact shared owner" {
    var probe: Probe = .{ .backing = .{ .budget = 5 }, .close_on_alloc = true };
    var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
    probe.owner = &owner;
    var op = try owner.begin();
    var services_probe: ServiceProbe = .{ .owner = &owner };
    var services = services_probe.services();
    var input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -1 }, .{ .re = 0, .im = 1 }, .{ .re = 2, .im = 0 } };
    var b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 4, .im = 0 } };
    var values: [2]f64 = undefined;
    const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2, .w = &values };
    try E.expectEqual(@as(i32, 0), try A.highAdmitted(true, .hegv, C, &op, &services, args));
    try E.expectApproxEqAbs(@as(f64, 0.25), values[0], 1e-14);
    try E.expectEqual(@as(usize, 1), probe.alloc_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
    try E.expectEqual(@as(usize, 5), probe.backing.stats().freed);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectError(error.Closing, owner.begin());
    op.end();
    try finish(&owner);
}
fn overflowContract(comptime T: type) !void {
    const R = Core.internal.numerics.Scalar.Real(T);
    const Herm = Core.internal.numerics.Hermitian;
    const K = Herm.core;
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    for ([_]bool{ false, true }) |generalized| {
        var input: [4]T = @splat(.{ .re = std.math.floatMax(R), .im = 0 });
        const before = input;
        var b = [_]T{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 } };
        var values: [2]R = @splat(99);
        var work: [3]T = undefined;
        var rw: [4]R = undefined;
        const args: A.Arguments(T) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .a = &input, .lda = 2, .b = &b, .ldb = 2, .w = &values, .work = &work, .lwork = 3, .rwork = &rw };
        const info = if (generalized) try A.workAdmitted(.hegv, T, &op, args) else try A.workAdmitted(.heev, T, &op, args);
        try E.expectEqual(@as(i32, 0), info);
        try E.expect(std.math.isInf(values[1]) and values[1] > 0);
        for (input) |x| {
            try E.expect(std.math.isFinite(x.re) and std.math.isFinite(x.im));
        }
        var checked_values: [2]R = @splat(99);
        var checked_work: [16]K.WorkspaceScalar(T) = undefined;
        const checked_a: Core.internal.numerics.Dense.ConstMatrix(T) = .{ .values = &before, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const checked = try K.solve(T, checked_a, .lower, &checked_values, null, &checked_work, .{});
        try E.expectEqual(K.Status.range_limit, checked.status);
        try E.expectEqual(@as(R, 99), checked_values[0]);
    }
    op.end();
    try finish(&owner);
}
test "Hermitian LAPACK finite output overflow succeeds while checked contract remains transactional" {
    try overflowContract(Core.internal.numerics.Scalar.ComplexF32);
    try overflowContract(C);
}
test "Hermitian nonconvergence counts actual tridiagonal edges and commits partial IEEE outputs" {
    const K = Core.internal.numerics.Hermitian.core;
    var input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 1, .im = -1 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 1 }, .{ .re = 3, .im = 0 }, .{ .re = 1, .im = -1 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 1 }, .{ .re = 4, .im = 0 } };
    const matrix: Core.internal.numerics.Dense.ConstMatrix(C) = .{ .values = &input, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
    var work: [36]K.WorkspaceScalar(C) = undefined;
    var values: [3]f64 = @splat(99);
    var q: [9]C = undefined;
    const result = try K.solveWithContract(C, matrix, .lower, &values, .{ .values = &q, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, &work, .{ .max_sweeps = 0 }, .heev_ieee_v1);
    try E.expectEqual(K.Status.iteration_limit, result.status);
    var remaining: usize = 0;
    for (0..2) |j| {
        if (work[j + 1 + j * 3].re != 0) remaining += 1;
    }
    try E.expectEqual(@as(usize, 2), remaining);
    try E.expectEqual(remaining, result.unconverged);
    try E.expect(values[0] != 99);
    for (q) |x| {
        try E.expect(std.math.isFinite(x.re) and std.math.isFinite(x.im));
    }
    values = @splat(99);
    try E.expectEqual(K.Status.iteration_limit, (try K.solve(C, matrix, .lower, &values, null, &work, .{ .max_sweeps = 0 })).status);
    try E.expectEqual(@as(f64, 99), values[0]);
}
test "Hermitian work permits nonfinite selected values and bounds real tridiagonal iterations" {
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    var input = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = std.math.nan(f64), .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 3, .im = 0 } };
    var values: [3]f64 = undefined;
    var work: [5]C = undefined;
    var rw: [7]f64 = undefined;
    const args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 3, .a = &input, .lda = 3, .w = &values, .work = &work, .lwork = 5, .rwork = &rw };
    const info = try A.workAdmitted(.heev, C, &op, args);
    try E.expect(info > 0 and info <= 2);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    op.end();
    try finish(&owner);
}
test "Hermitian generalized unfinished iteration backtransforms only documented converged prefix" {
    const K = Core.internal.numerics.Hermitian.core;
    const input = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 1, .im = -1 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 1 }, .{ .re = 3, .im = 0 }, .{ .re = 1, .im = -1 }, .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 1 }, .{ .re = 4, .im = 0 } };
    const b = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 3, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 4, .im = 0 } };
    const matrix: Core.internal.numerics.Dense.ConstMatrix(C) = .{ .values = &input, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
    const metric: Core.internal.numerics.Dense.ConstMatrix(C) = .{ .values = &b, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
    for ([_]K.Problem{ .ax_lambda_bx, .abx_lambda_x, .bax_lambda_x }) |problem| {
        var work: [36]K.WorkspaceScalar(C) = undefined;
        var values: [3]f64 = @splat(99);
        var q: [9]C = undefined;
        const result = try K.solveGeneralizedWithContract(C, matrix, metric, problem, .lower, &values, .{ .values = &q, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, &work, .{ .max_sweeps = 0 }, .heev_ieee_v1);
        try E.expectEqual(K.Status.iteration_limit, result.status);
        try E.expectEqual(@as(usize, 2), result.unconverged);
        for (q) |x| {
            try E.expect(std.math.isFinite(x.re) and std.math.isFinite(x.im));
        }
    }
}
fn scalarGeneralized(comptime T: type) !void {
    const R = Core.internal.numerics.Scalar.Real(T);
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| for ([_]u8{ 'N', 'V' }) |job| for (1..4) |kind| {
        var a = [_]T{.{ .re = 8, .im = std.math.nan(R) }};
        var b = [_]T{.{ .re = 4, .im = std.math.nan(R) }};
        var values: [1]R = undefined;
        var work: [1]T = undefined;
        const args: Raw.Arguments(T) = .{ .layout = layout, .itype = @intCast(kind), .jobz = job, .uplo = uplo, .n = 1, .a = @ptrCast(&a), .lda = 1, .b = @ptrCast(&b), .ldb = 1, .w = @ptrCast(&values), .work = @ptrCast(&work), .lwork = 1, .rwork = @ptrFromInt(1) };
        try E.expectEqual(@as(i32, 0), try Raw.workAdmitted(.hegv, T, &op, args));
        try E.expectEqual(@as(R, if (kind == 1) 2 else 32), values[0]);
        try E.expectEqual(@as(R, 2), b[0].re);
        if (job == 'V') try E.expectEqual(@as(R, if (kind == 3) 2 else 0.5), a[0].re);
        try E.expectEqual(@as(R, 1), work[0].re);
    };
    op.end();
    try finish(&owner);
}
test "Hermitian generalized scalar has no RWORK reads in both precisions layouts and problem kinds" {
    try scalarGeneralized(Core.internal.numerics.Scalar.ComplexF32);
    try scalarGeneralized(C);
}
test "Hermitian column-major nonpositive B precedes all A W RWORK storage access" {
    var owner = context.Context.init(E.allocator, fixture.policy, .internal_closable);
    var op = try owner.begin();
    for ([_]u8{ 'L', 'U' }) |uplo| for ([_]usize{ 1, 2 }) |pivot| {
        var b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 4, .im = 0 } };
        b[(pivot - 1) * 3].re = -1;
        var work: [3]C = undefined;
        const raw_args: Raw.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = uplo, .n = 2, .a = @ptrFromInt(1), .lda = 2, .b = @ptrCast(&b), .ldb = 2, .w = @ptrFromInt(1), .work = @ptrCast(&work), .lwork = 3, .rwork = @ptrFromInt(1) };
        try E.expectEqual(@as(i32, @intCast(2 + pivot)), try Raw.workAdmitted(.hegv, C, &op, raw_args));
        b = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 4, .im = 0 } };
        b[(pivot - 1) * 3].re = -1;
        const typed_args: A.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = uplo, .n = 2, .lda = 2, .b = &b, .ldb = 2, .work = &work, .lwork = 3 };
        try E.expectEqual(@as(i32, @intCast(2 + pivot)), try A.workAdmitted(.hegv, C, &op, typed_args));
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    };
    op.end();
    try finish(&owner);
}
test "Hermitian row-major conversion still requires A before nonpositive B factorization" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var op = try owner.begin();
    var b = [_]C{ .{ .re = -1, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 4, .im = 0 } };
    var work: [3]C = undefined;
    const args: A.Arguments(C) = .{ .layout = 101, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .b = &b, .ldb = 2, .work = &work, .lwork = 3 };
    try E.expectError(error.MissingBuffer, A.workAdmitted(.hegv, C, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner);
}
test "Hermitian production binding borrows configured callbacks and ends its one operation after cleanup" {
    var alloc: fixture.Allocations = .{ .budget = 0 };
    var owner = context.Context.init(alloc.allocator(), fixture.policy, .internal_closable);
    var probe: ServiceProbe = .{ .owner = &owner };
    const defaults: Owner.shared_services.Callbacks = .{ .lapacke = .{ .userdata = &probe, .function = ServiceProbe.event }, .fortran_returning_test = .{ .userdata = &probe, .function = ServiceProbe.event } };
    const bad: Raw.Arguments(C) = .{ .layout = 0, .jobz = 'V', .uplo = 'L', .n = 0, .lda = 1 };
    try E.expectEqual(@as(i32, -1), try Owner.Shared.hermitianRawManagedWithDefaults(false, C, .hegv, .high, &owner, bad, defaults));
    try E.expectEqual(@as(usize, 1), probe.count);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    const empty: Raw.Arguments(C) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 0, .lda = 1 };
    try E.expectEqual(@as(i32, -1010), try Owner.Shared.hermitianRawManagedWithDefaults(false, C, .hegv, .high, &owner, empty, defaults));
    try E.expectEqual(@as(usize, 2), probe.count);
    try E.expectEqual(@as(usize, 0), probe.pin_snapshot);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    _ = owner.requestClose();
    try E.expectError(error.Closing, Owner.Shared.hermitianRawManagedWithDefaults(false, C, .hegv, .high, &owner, bad, defaults));
    try E.expectEqual(@as(usize, 2), probe.count);
    try finish(&owner);
}
