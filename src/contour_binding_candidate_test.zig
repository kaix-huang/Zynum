const std = @import("std");
const Core = @import("zynum-core");
const CW = @import("zynum-contour-wire");
const N = Core.internal.numerics;
const E = N.Execution;
const C = E.context;
const S = N.Scalar;
const B = @import("interop/operator_binding.zig").Binding(Core, @import("checked/implementation.zig"));
const W = B.Wire;

fn close(owner: *C.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
fn Fixture(comptime T: type) type {
    const R = S.Real(T);
    return struct {
        const Self = @This();
        owner: *C.Context,
        calls: usize = 0,
        fail_at: usize = 0,
        close_at: usize = 0,
        initial: [16]T,
        vectors: [24]T,
        eigenvalues: [6]R,
        residuals: [6]R,
        options: CW.Options(R),
        summary: B.ContourSummary(T, CW),
        fn init(owner: *C.Context) Self {
            var self = Self{ .owner = owner, .initial = @splat(S.zero(T)), .vectors = @splat(S.realScalar(T, 77)), .eigenvalues = @splat(88), .residuals = @splat(99), .options = std.mem.zeroes(CW.Options(R)), .summary = std.mem.zeroes(B.ContourSummary(T, CW)) };
            for (0..4) |i| self.initial[i + i * 4] = S.one(T);
            self.options.version = CW.abi_version;
            self.options.global_dimension = 4;
            self.options.half_nodes = 8;
            self.options.gauss_max_iterations = 64;
            self.options.max_iterations = 4;
            self.options.linear_max_iterations = 16;
            self.options.ritz_max_sweeps = 96;
            self.options.lower = -2;
            self.options.upper = 3;
            self.options.vertical_ratio = 1;
            self.options.relative_tolerance = if (R == f32) 1e-4 else 1e-10;
            self.options.linear_relative_tolerance = if (R == f32) 1e-5 else 1e-12;
            self.options.rank_relative_tolerance = std.math.nan(R);
            self.options.ritz_relative_tolerance = std.math.nan(f64);
            self.summary.status = 123;
            return self;
        }
        fn apply(raw: ?*anyopaque, _: *const B.Token, input: W.const_buffer, output: W.buffer, _: W.buffer) callconv(.c) i32 {
            const self: *Self = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            const snapshot = self.owner.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins < 2) return 901;
            if (self.close_at == self.calls) _ = self.owner.requestClose();
            const x: [*]const T = @ptrCast(@alignCast(input.data.?));
            const y: [*]T = @ptrCast(@alignCast(output.data.?));
            for (0..4) |i| y[i] = S.mul(T, S.realScalar(T, @as(R, @floatFromInt(i)) - 1), x[i]);
            return if (self.fail_at == self.calls) 213 else 0;
        }
        fn call(self: *Self, operation: *const C.Operation, agreement: ?B.Agreement) W.result {
            const a: B.Action = .{ .context = self, .rows = 4, .columns = 4, .workspace_elements = 0, .apply = apply };
            const x: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &self.initial, .length = 16 }, .rows = 4, .cols = 4, .ld = 4 }, .layout = 102, .reserved = 0 };
            const y: B.Panel = .{ .dense = .{ .storage = .{ .data = &self.vectors, .length = 24 }, .rows = 4, .cols = 4, .ld = 6 }, .layout = 102, .reserved = 0 };
            const values: W.buffer = .{ .data = &self.eigenvalues, .length = 6 };
            const errors: W.buffer = .{ .data = &self.residuals, .length = 6 };
            return if (agreement) |g|
                B.contourAgreedAdmitted(T, CW, operation, a, null, null, null, g, &self.options, x, y, values, errors, &self.summary)
            else
                B.contourAdmitted(T, CW, operation, a, null, null, null, &self.options, x, y, values, errors, &self.summary);
        }
        fn preserved(self: *const Self, prior: *const Self, summary_too: bool) !void {
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&prior.vectors), std.mem.asBytes(&self.vectors));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&prior.eigenvalues), std.mem.asBytes(&self.eigenvalues));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&prior.residuals), std.mem.asBytes(&self.residuals));
            if (summary_too) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&prior.summary), std.mem.asBytes(&self.summary));
        }
    };
}

pub fn checkCompleteSpace() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        var f = Fixture(T).init(&owner);
        const result = f.call(&operation, null);
        try std.testing.expectEqual(@as(i32, 0), result.status);
        try std.testing.expectEqual(@as(i32, 0), f.summary.status);
        try std.testing.expectEqual(@as(i64, 4), f.summary.found);
        try std.testing.expectEqual(@as(i64, 4), f.summary.published);
        const tolerance: R = if (R == f32) 2e-4 else 1e-9;
        for (0..4) |column| {
            const lambda = @as(R, @floatFromInt(column)) - 1;
            try std.testing.expectApproxEqAbs(lambda, f.eigenvalues[column], tolerance);
            var norm: R = 0;
            for (0..4) |row| {
                const z = f.vectors[column * 6 + row];
                norm += z.re * z.re + z.im * z.im;
                const d = @as(R, @floatFromInt(row)) - 1 - f.eigenvalues[column];
                try std.testing.expect(@abs(d * z.re) + @abs(d * z.im) <= tolerance);
            }
            try std.testing.expectApproxEqAbs(@as(R, 1), norm, tolerance);
            try std.testing.expectEqual(S.realScalar(T, 77), f.vectors[column * 6 + 4]);
            try std.testing.expectEqual(S.realScalar(T, 77), f.vectors[column * 6 + 5]);
        }
        try std.testing.expectEqualSlices(R, &.{ 88, 88 }, f.eigenvalues[4..]);
        try std.testing.expectEqualSlices(R, &.{ 99, 99 }, f.residuals[4..]);
        try std.testing.expect(f.calls > 0);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

test "candidate contour binding returns complete-space eigenpairs with one borrowed operation" {
    try checkCompleteSpace();
}

pub fn checkFormatAndControlledStop() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        var f = Fixture(T).init(&owner);
        f.options.flags = 16;
        var prior = f;
        try std.testing.expectEqual(B.INVALID_OPTIONS, f.call(&operation, null).status);
        try f.preserved(&prior, true);
        f.options.flags = 0;
        f.options.max_iterations = 0;
        prior = f;
        try std.testing.expectEqual(@as(i32, 0), f.call(&operation, null).status);
        try std.testing.expectEqual(@as(i32, 1), f.summary.status);
        try std.testing.expectEqual(@as(i64, 0), f.summary.published);
        try f.preserved(&prior, false);
        try std.testing.expectEqual(@as(usize, 0), f.calls);
    }
}

test "candidate contour binding preserves outputs on format rejection and controlled stop" {
    try checkFormatAndControlledStop();
}

pub fn checkCallbackAndClosingOwner() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        var f = Fixture(T).init(&owner);
        f.fail_at = 1;
        f.close_at = 1;
        const prior = f;
        const result = f.call(&operation, null);
        try std.testing.expectEqual(B.ACTION_FAILED, result.status);
        try std.testing.expectEqual(@as(i64, 213), result.info);
        try f.preserved(&prior, true);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

test "candidate contour binding unwinds callback failure and closing owner without publication" {
    try checkCallbackAndClosingOwner();
}

pub fn checkFinalAgreement() !void {
    const Reject = struct {
        seen: bool = false,
        fn agree(raw: ?*anyopaque, _: ?*const B.Token, point: B.Point, local: W.result) callconv(.c) W.result {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (local.status != 0) return local;
            if (point.stage == .commit and point.index == 2) {
                self.seen = true;
                return .{ .status = B.PHASE_MISMATCH, .reserved = 0, .info = 123 };
            }
            return local;
        }
    };
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        var f = Fixture(T).init(&owner);
        const prior = f;
        var reject: Reject = .{};
        const result = f.call(&operation, .{ .context = &reject, .agree = Reject.agree });
        try std.testing.expectEqual(B.PHASE_MISMATCH, result.status);
        try std.testing.expectEqual(@as(i64, 123), result.info);
        try std.testing.expect(reject.seen and f.calls > 0);
        try f.preserved(&prior, true);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

test "candidate contour binding final agreement failure preserves every public output" {
    try checkFinalAgreement();
}

pub fn checkNativeControlRejections() !void {
    const Pass = struct {
        fn agree(_: ?*anyopaque, _: ?*const B.Token, _: B.Point, local: W.result) callconv(.c) W.result {
            return local;
        }
    };
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        var owner = C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        defer close(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        for (0..2) |agreed| {
            for (0..8) |kind| {
                var f = Fixture(T).init(&owner);
                switch (kind) {
                    0 => f.options.lower = f.options.upper,
                    1 => f.options.vertical_ratio = 0,
                    2 => f.options.half_nodes = 0,
                    3 => f.options.global_dimension = -1,
                    4 => f.initial[0].re = std.math.nan(S.Real(T)),
                    5 => f.options.global_dimension = 8,
                    6 => f.options.flags = CW.rank_relative_present,
                    7 => f.options.flags = CW.ritz_relative_present,
                    else => unreachable,
                }
                const prior = f;
                const g: ?B.Agreement = if (agreed == 0) null else .{ .context = null, .agree = Pass.agree };
                const expected: i32 = if (kind == 3) W.INVALID_DIMENSION else if (kind == 4) W.NONFINITE_INPUT else B.INVALID_OPTIONS;
                try std.testing.expectEqual(expected, f.call(&operation, g).status);
                try f.preserved(&prior, true);
                try std.testing.expectEqual(@as(usize, 0), f.calls);
                try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
            }
        }
    }
}

test "candidate contour binding native-control rejection preserves public outputs" {
    try checkNativeControlRejections();
}
