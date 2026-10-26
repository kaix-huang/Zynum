const std = @import("std");
const Core = @import("zynum-core");
const Boundary = @import("fftw-interface-owner");
const Execution = Core.internal.numerics.Execution;
const context = Execution.context;
const fixture = Execution.test_support;
const E = std.testing;

const LifecycleProbe = struct {
    backing: fixture.Allocations = .{ .budget = 32 },
    owner: *context.Context = undefined,
    attempts: usize = 0,
    close_at: ?usize = null,
    replace_at: ?usize = null,
    replacement: ?std.mem.Allocator = null,
    close_snapshot: ?context.Snapshot = null,
    free_snapshot: ?context.Snapshot = null,
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(raw: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.attempts += 1;
        if (self.replace_at == self.attempts) self.owner.allocator = self.replacement.?;
        if (self.close_at == self.attempts) {
            self.close_snapshot = self.owner.snapshot();
            _ = self.owner.requestClose();
            if (self.owner.tryProgressClose() != .busy) @panic("FFTW callback lost active admission");
        }
        return self.backing.allocator().rawAlloc(n, alignment, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, data: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.free_snapshot = self.owner.snapshot();
        if (self.owner.tryProgressClose() == .closed) @panic("FFTW resource pin released before captured free");
        self.backing.allocator().rawFree(data, alignment, ret);
    }
};

fn Scalar(comptime R: type) type {
    return if (R == f32) Core.internal.numerics.Scalar.ComplexF32 else Core.internal.numerics.Scalar.ComplexF64;
}
fn tolerance(comptime R: type) f64 {
    return if (R == f32) 3e-5 else 3e-12;
}
fn finish(owner: *context.Context) !void {
    _ = owner.requestClose();
    try E.expectEqual(context.CloseResult.closed, owner.tryProgressClose());
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
fn dft(comptime C: type, input: []const C, k: usize, sign: f64) struct { re: f64, im: f64 } {
    var re: f64 = 0;
    var im: f64 = 0;
    for (input, 0..) |v, j| {
        const angle = sign * 2 * std.math.pi * @as(f64, @floatFromInt(j * k)) / @as(f64, @floatFromInt(input.len));
        re += @as(f64, v.re) * @cos(angle) - @as(f64, v.im) * @sin(angle);
        im += @as(f64, v.re) * @sin(angle) + @as(f64, v.im) * @cos(angle);
    }
    return .{ .re = re, .im = im };
}
fn realDefinition(kind: c_uint, input: []const f64, k: usize) f64 {
    const n: f64 = @floatFromInt(input.len);
    const out: f64 = @floatFromInt(k);
    var result: f64 = 0;
    switch (kind) {
        0 => for (input, 0..) |v, j| {
            const frequency = if (k <= input.len / 2) k else input.len - k;
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(j * frequency)) / n;
            result += v * (if (k <= input.len / 2) @cos(angle) else @sin(angle));
        },
        1 => {
            result = input[0];
            if (input.len % 2 == 0) result += input[input.len / 2] * (if (k % 2 == 0) @as(f64, 1) else -1);
            for (1..(input.len + 1) / 2) |j| {
                const angle = 2 * std.math.pi * @as(f64, @floatFromInt(j * k)) / n;
                result += 2 * (input[j] * @cos(angle) - input[input.len - j] * @sin(angle));
            }
        },
        2 => for (input, 0..) |v, j| {
            const angle = 2 * std.math.pi * @as(f64, @floatFromInt(j * k)) / n;
            result += v * (@cos(angle) + @sin(angle));
        },
        3 => {
            result = input[0] + input[input.len - 1] * (if (k % 2 == 0) @as(f64, 1) else -1);
            for (1..input.len - 1) |j| result += 2 * input[j] * @cos(std.math.pi * @as(f64, @floatFromInt(j)) * out / (n - 1));
        },
        4 => {
            result = input[0];
            for (1..input.len) |j| result += 2 * input[j] * @cos(std.math.pi * @as(f64, @floatFromInt(j)) * (out + 0.5) / n);
        },
        5, 6, 7, 8, 9, 10 => {
            for (input, 0..) |v, j| {
                const in: f64 = @floatFromInt(j);
                result += switch (kind) {
                    5 => 2 * v * @cos(std.math.pi * (in + 0.5) * out / n),
                    6 => 2 * v * @cos(std.math.pi * (in + 0.5) * (out + 0.5) / n),
                    7 => 2 * v * @sin(std.math.pi * (in + 1) * (out + 1) / (n + 1)),
                    8 => if (j == input.len - 1) v * (if (k % 2 == 0) @as(f64, 1) else -1) else 2 * v * @sin(std.math.pi * (in + 1) * (out + 0.5) / n),
                    9 => 2 * v * @sin(std.math.pi * (in + 0.5) * (out + 1) / n),
                    10 => 2 * v * @sin(std.math.pi * (in + 0.5) * (out + 0.5) / n),
                    else => unreachable,
                };
            }
        },
        else => unreachable,
    }
    return result;
}

test "Scalar FFTW complex new arrays use native real alignment and independent direct definitions" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        try E.expect(B.Provider.OwnerIdentity == Core);
        try E.expect(B.Provider.Execution == Execution);
        var input: [16]R align(16) = @splat(777);
        var output: [16]R align(16) = @splat(888);
        const logical: [7]C = .{ .{ .re = 1, .im = -1 }, .{ .re = 2, .im = 0.5 }, .{ .re = -3, .im = 0 }, .{ .re = 4, .im = 1 }, .{ .re = 0, .im = 2 }, .{ .re = 0.5, .im = -1 }, .{ .re = 2, .im = 3 } };
        const values: [*]C = @ptrCast(&input[1]);
        const result: [*]C = @ptrCast(&output[1]);
        @memcpy(values[0..7], &logical);
        const h = B.createWithContext(&owner, .complex, .{ 7, 1, 1 }, &input[1], &output[1], -1, 64).?;
        try E.expectEqual(@as(c_int, 0), B.alignmentOf(&input[1]));
        try B.executeNewArrays(h, .complex, &input[1], &output[1]);
        for (result[0..7], 0..) |v, k| {
            const expected = dft(C, &logical, k, -1);
            try E.expectApproxEqAbs(expected.re, @as(f64, v.re), tolerance(R));
            try E.expectApproxEqAbs(expected.im, @as(f64, v.im), tolerance(R));
        }
        try E.expectEqual(@as(R, 888), output[0]);
        try E.expectEqual(@as(R, 888), output[15]);
        B.destroy(h);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW all eleven real transforms use independent sums and preserve four byte boundaries" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        var probe: fixture.Allocations = .{ .budget = 512 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        for ([_]usize{ 2, 3, 4, 7 }) |n| for (0..11) |kind| {
            var input: [9]R align(16) = @splat(777);
            var output: [9]R align(16) = @splat(888);
            var exact: [7]f64 = undefined;
            for (input[1..][0..n], exact[0..n], 0..) |*v, *x, j| {
                v.* = @floatCast(@as(f64, @floatFromInt(j + 1)) / 4 - 0.5);
                x.* = v.*;
            }
            const mode: c_uint = @intCast(kind);
            const h = B.createTensorRealWithContext(&owner, 1, &.{@intCast(n)}, &.{mode}, &input[1], &output[1], 64).?;
            try B.executeNewArrays(h, .real_transform, &input[1], &output[1]);
            for (output[1..][0..n], 0..) |v, k| try E.expectApproxEqAbs(realDefinition(mode, exact[0..n], k), @as(f64, v), tolerance(R));
            try E.expectEqual(@as(R, 888), output[0]);
            for (output[n + 1 ..]) |v| try E.expectEqual(@as(R, 888), v);
            B.destroy(h);
        };
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW tensor real in place roundtrips retain odd row padding and guards" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var data: [10]R = .{ 1, 2, 3, 777, 4, 5, 6, 777, 888, 999 };
        const forward = B.createTensorWithContext(&owner, .real_forward, 2, &.{ 2, 3 }, &data, &data, -1, 64).?;
        const backward = B.createTensorWithContext(&owner, .real_backward, 2, &.{ 2, 3 }, &data, &data, 1, 64).?;
        try B.executeWithArrays(forward, &data, &data);
        try B.executeWithArrays(backward, &data, &data);
        for (0..2) |row| for (0..3) |column| try E.expectApproxEqAbs(@as(f64, @floatFromInt((row * 3 + column + 1) * 6)), @as(f64, data[row * 4 + column]), tolerance(R));
        try E.expectEqual(@as(R, 888), data[8]);
        try E.expectEqual(@as(R, 999), data[9]);
        B.destroy(forward);
        B.destroy(backward);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW guru widths copy signed layouts and keep only selected output cells" {
    inline for (.{ f32, f64 }) |R| inline for (.{ Boundary.Binding(R).IODim, Boundary.Binding(R).IODim64 }) |D| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [6]C = .{ .{ .re = 3, .im = 1 }, .{ .re = 2, .im = -1 }, .{ .re = 1, .im = 0 }, .{ .re = 6, .im = 2 }, .{ .re = 5, .im = 1 }, .{ .re = 4, .im = -1 } };
        var output: [13]C = @splat(.{ .re = 777, .im = 888 });
        var axes: [1]D = .{.{ .n = 3, .input_stride = -1, .output_stride = 2 }};
        var batches: [1]D = .{.{ .n = 2, .input_stride = 3, .output_stride = 6 }};
        const h = B.createGuruWithContext(D, &owner, 1, &axes, 1, &batches, &input[2], &output, -1, 64).?;
        axes[0].n = 1;
        batches[0].n = 0;
        try B.executeWithArrays(h, &input[2], &output);
        for (0..2) |batch| {
            const logical: [3]C = .{ input[batch * 3 + 2], input[batch * 3 + 1], input[batch * 3] };
            for (0..3) |k| {
                const expected = dft(C, &logical, k, -1);
                try E.expectApproxEqAbs(expected.re, @as(f64, output[batch * 6 + 2 * k].re), tolerance(R));
                try E.expectApproxEqAbs(expected.im, @as(f64, output[batch * 6 + 2 * k].im), tolerance(R));
            }
        }
        for (output, 0..) |v, j| if (j % 2 == 1 or j == 12) {
            try E.expectEqual(@as(R, 777), v.re);
            try E.expectEqual(@as(R, 888), v.im);
        };
        B.destroy(h);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    };
}

test "Scalar FFTW split and interleaved new arrays use the selected scalar plane separation" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [8]R = .{ 1, 0, 2, 0, 3, 0, 777, 888 };
        var output: [8]R = @splat(999);
        const h = B.createWithContext(&owner, .complex, .{ 3, 1, 1 }, &input, &output, -1, 64).?;
        try B.executeSplitNewArrays(h, .complex, &input[0], &input[1], &output[0], &output[1]);
        try E.expectApproxEqAbs(@as(f64, 6), @as(f64, output[0]), tolerance(R));
        try E.expectApproxEqAbs(@as(f64, -1.5), @as(f64, output[2]), tolerance(R));
        try E.expectApproxEqAbs(@sqrt(@as(f64, 3)) / 2, @as(f64, output[3]), tolerance(R));
        try E.expectEqual(@as(R, 999), output[6]);
        try E.expectEqual(@as(R, 999), output[7]);
        @memset(&output, 999);
        try E.expectError(error.InvalidArgument, B.executeSplitNewArrays(h, .complex, &input[0], &input[2], &output[0], &output[1]));
        for (output) |v| try E.expectEqual(@as(R, 999), v);
        B.destroy(h);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW failed metadata plan or workspace acquisitions retain outputs and captured ownership" {
    inline for (.{ f32, f64 }) |R| for (0..4) |budget| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: fixture.Allocations = .{ .budget = budget };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [3]C = @splat(.{ .re = 1, .im = 0 });
        var output: [3]C = @splat(.{ .re = 777, .im = 888 });
        const h = B.createTensorWithContext(&owner, .complex, 1, &.{3}, &input, &output, -1, 64);
        if (h) |live| {
            if (budget < 3) {
                try E.expectError(error.OutOfMemory, B.executeWithArrays(live, &input, &output));
                for (output) |v| try E.expectEqual(@as(R, 777), v.re);
            } else {
                try B.executeWithArrays(live, &input, &output);
                try E.expectApproxEqAbs(@as(f64, 3), @as(f64, output[0].re), tolerance(R));
            }
            B.destroy(live);
        } else {
            try E.expect(budget < 2);
            for (output) |v| try E.expectEqual(@as(R, 777), v.re);
        }
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    };
}

test "Scalar FFTW both precisions share one admitted context and allocation cleanup across close" {
    const S = Boundary.F32;
    const D = Boundary;
    var probe: fixture.Allocations = .{ .budget = 32 };
    var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
    var input_s: [3]Scalar(f32) = @splat(.{ .re = 1, .im = 0 });
    var output_s: [3]Scalar(f32) = @splat(.{ .re = 777, .im = 888 });
    var input_d: [3]Scalar(f64) = @splat(.{ .re = 2, .im = 0 });
    var output_d: [3]Scalar(f64) = @splat(.{ .re = 777, .im = 888 });
    const s = S.createWithContext(&owner, .complex, .{ 3, 1, 1 }, &input_s, &output_s, -1, 64).?;
    const d = D.createWithContext(&owner, .complex, .{ 3, 1, 1 }, &input_d, &output_d, -1, 64).?;
    const bytes_s = S.allocateWithContext(&owner, 37).?;
    const bytes_d = D.allocRealWithContext(&owner, 5).?;
    try E.expectEqual(@as(usize, 0), @intFromPtr(bytes_s) % 64);
    try E.expectEqual(@as(usize, 4), owner.snapshot().resource_pins);
    var op = try owner.begin();
    _ = owner.requestClose();
    try S.executeWithArraysAdmitted(s, &op, &input_s, &output_s);
    try D.executeWithArraysAdmitted(d, &op, &input_d, &output_d);
    try E.expectEqual(@as(f32, 3), output_s[0].re);
    try E.expectEqual(@as(f64, 6), output_d[0].re);
    try E.expectError(error.Closing, S.executeWithArrays(s, &input_s, &output_s));
    try E.expectEqual(context.CloseResult.busy, owner.tryProgressClose());
    D.free(bytes_s);
    S.free(bytes_d);
    S.destroy(s);
    D.destroy(d);
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    op.end();
    try finish(&owner);
    try E.expectEqual(probe.stats().accepted, probe.stats().freed);
}

test "Scalar FFTW many complex batches retain strided cells and direct definition results" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [6]C = .{ .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = -1 }, .{ .re = 4, .im = 1 }, .{ .re = 5, .im = 0 }, .{ .re = 6, .im = -1 } };
        var output: [13]C = @splat(.{ .re = 777, .im = 888 });
        const h = B.createManyWithContext(&owner, 1, &.{3}, 2, &input, null, 1, 3, &output, null, 2, 6, -1, 64).?;
        try B.executeNewArrays(h, .complex, &input, &output);
        for (0..2) |batch| for (0..3) |k| {
            const expected = dft(C, input[batch * 3 ..][0..3], k, -1);
            try E.expectApproxEqAbs(expected.re, @as(f64, output[batch * 6 + 2 * k].re), tolerance(R));
            try E.expectApproxEqAbs(expected.im, @as(f64, output[batch * 6 + 2 * k].im), tolerance(R));
        };
        for (output, 0..) |v, j| if (j % 2 == 1 or j == 12) {
            try E.expectEqual(@as(R, 777), v.re);
            try E.expectEqual(@as(R, 888), v.im);
        };
        B.destroy(h);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW many guru and guru64 real batches preserve physical gaps and inverse normalization" {
    inline for (.{ f32, f64 }) |R| inline for (.{ @as(u2, 0), 1, 2 }) |route| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        const D = if (route == 2) B.IODim64 else B.IODim;
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [14]R = @splat(777);
        var spectrum: [10]C = @splat(.{ .re = 888, .im = 999 });
        var output: [14]R = @splat(666);
        for (0..2) |batch| for (0..3) |j| {
            input[batch * 7 + 2 * j] = @floatFromInt(batch * 3 + j + 1);
        };
        const axes: [1]D = .{.{ .n = 3, .input_stride = 2, .output_stride = 2 }};
        const forward_batches: [1]D = .{.{ .n = 2, .input_stride = 7, .output_stride = 5 }};
        const backward_batches: [1]D = .{.{ .n = 2, .input_stride = 5, .output_stride = 7 }};
        const forward = if (route == 0)
            B.createManyRealWithContext(&owner, .forward, 1, &.{3}, 2, &input, null, 2, 7, &spectrum, null, 2, 5, null, 64).?
        else
            B.createGuruRealWithContext(D, &owner, .forward, 1, &axes, 1, &forward_batches, &input, &spectrum, null, 64).?;
        const backward = if (route == 0)
            B.createManyRealWithContext(&owner, .backward, 1, &.{3}, 2, &spectrum, null, 2, 5, &output, null, 2, 7, null, 64).?
        else
            B.createGuruRealWithContext(D, &owner, .backward, 1, &axes, 1, &backward_batches, &spectrum, &output, null, 64).?;
        try B.executeNewArrays(forward, .real_forward, &input, &spectrum);
        for (0..2) |batch| {
            var logical: [3]C = undefined;
            for (&logical, 0..) |*v, j| v.* = .{ .re = input[batch * 7 + 2 * j], .im = 0 };
            for (0..2) |k| {
                const expected = dft(C, &logical, k, -1);
                try E.expectApproxEqAbs(expected.re, @as(f64, spectrum[batch * 5 + 2 * k].re), tolerance(R));
                try E.expectApproxEqAbs(expected.im, @as(f64, spectrum[batch * 5 + 2 * k].im), tolerance(R));
            }
        }
        for (spectrum, 0..) |v, j| if (j != 0 and j != 2 and j != 5 and j != 7) {
            try E.expectEqual(@as(R, 888), v.re);
            try E.expectEqual(@as(R, 999), v.im);
        };
        try B.executeNewArrays(backward, .real_backward, &spectrum, &output);
        for (output, 0..) |v, j| {
            if ((j < 5 and j % 2 == 0) or (j >= 7 and j < 12 and j % 2 == 1)) {
                try E.expectApproxEqAbs(@as(f64, input[j]) * 3, @as(f64, v), tolerance(R));
            } else try E.expectEqual(@as(R, 666), v);
        }
        B.destroy(forward);
        B.destroy(backward);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    };
}

test "Scalar FFTW split guru widths execute all three transform families with distinct planes" {
    inline for (.{ f32, f64 }) |R| inline for (.{ Boundary.Binding(R).IODim, Boundary.Binding(R).IODim64 }) |D| inline for (.{ Boundary.Binding(R).SplitTransform.complex, .real_forward, .real_backward }) |kind| {
        const B = Boundary.Binding(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var real_input: [4]R = .{ 1, 1, 1, 777 };
        var imaginary_input: [4]R = @splat(0);
        var real_output: [4]R = @splat(888);
        var imaginary_output: [4]R = @splat(999);
        if (kind == .real_backward) real_input = .{ 3, 0, 777, 777 };
        const h = B.createGuruSplitWithContext(D, &owner, kind, 1, &.{.{ .n = 3, .input_stride = 1, .output_stride = 1 }}, 0, null, &real_input, if (kind == .real_forward) null else &imaginary_input, &real_output, if (kind == .real_backward) null else &imaginary_output, 64).?;
        try B.executeSplitNewArrays(h, kind, &real_input, if (kind == .real_forward) null else &imaginary_input, &real_output, if (kind == .real_backward) null else &imaginary_output);
        const output_count: usize = if (kind == .real_forward) 2 else 3;
        for (real_output[0..output_count], 0..) |v, j| {
            const expected: f64 = if (kind == .real_backward or j == 0) 3 else 0;
            try E.expectApproxEqAbs(expected, @as(f64, v), tolerance(R));
        }
        for (real_output[output_count..]) |v| try E.expectEqual(@as(R, 888), v);
        if (kind != .real_backward) {
            for (imaginary_output[0..output_count]) |v| try E.expectApproxEqAbs(@as(f64, 0), @as(f64, v), tolerance(R));
            for (imaginary_output[output_count..]) |v| try E.expectEqual(@as(R, 999), v);
        } else for (imaginary_output) |v| try E.expectEqual(@as(R, 999), v);
        B.destroy(h);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    };
}

test "Scalar FFTW IEEE values zero batches flags and scalar byte overflow retain the shared contract" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: fixture.Allocations = .{ .budget = 32 };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        var input: [2]C = @splat(.{ .re = std.math.floatMax(R), .im = 0 });
        var output: [3]C = @splat(.{ .re = 777, .im = 888 });
        try E.expect(B.allocRealWithContext(&owner, std.math.maxInt(usize) / @sizeOf(R) + 1) == null);
        try E.expect(B.allocateWithContext(&owner, std.math.maxInt(usize)) == null);
        try E.expect(B.createWithContext(&owner, .complex, .{ 2, 1, 1 }, &input, &output, -1, 1 << 21) == null);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        const h = B.createWithContext(&owner, .complex, .{ 2, 1, 1 }, &input, &output, -1, 64).?;
        try B.executeWithArrays(h, &input, &output);
        try E.expect(std.math.isInf(output[0].re));
        try E.expectEqual(@as(R, 777), output[2].re);
        input[0].re = std.math.nan(R);
        try B.executeWithArrays(h, &input, &output);
        try E.expect(std.math.isNan(output[0].re));
        B.destroy(h);
        const empty = B.createManyWithContext(&owner, 1, &.{3}, 0, null, null, 1, 3, null, null, 1, 3, -1, 0).?;
        var op = try owner.begin();
        try B.executeAdmitted(empty, &op);
        op.end();
        B.destroy(empty);
        const allocation = B.allocRealWithContext(&owner, 0).?;
        Boundary.free(allocation);
        try finish(&owner);
        try E.expectEqual(probe.stats().accepted, probe.stats().freed);
    }
}

test "Scalar FFTW planning and workspace close callbacks preserve prior admissions and cleanup pins" {
    inline for (.{ f32, f64 }) |R| for ([_]usize{ 1, 2 }) |close_at| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var probe: LifecycleProbe = .{ .close_at = close_at };
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        probe.owner = &owner;
        var input: [4]C = @splat(.{ .re = 1, .im = 0 });
        var output: [5]C = @splat(.{ .re = 777, .im = 888 });
        var admitted = try owner.begin();
        const handle = B.createWithContext(&owner, .complex, .{ 4, 1, 1 }, &input, &output, -1, 64).?;
        try B.executeWithArraysAdmitted(handle, &admitted, &input, &output);
        const captured = probe.close_snapshot.?;
        try E.expectEqual(@as(usize, if (close_at == 1) 2 else 1), captured.active_operations);
        try E.expectEqual(@as(usize, close_at), captured.resource_pins);
        try E.expectEqual(@as(R, 4), output[0].re);
        try E.expectEqual(@as(R, 777), output[4].re);
        try E.expectError(error.Closing, B.executeWithArrays(handle, &input, &output));
        try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
        try E.expectEqual(@as(usize, 2), probe.free_snapshot.?.resource_pins);
        B.destroy(handle);
        try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.active_operations);
        try E.expectEqual(@as(usize, 1), probe.free_snapshot.?.resource_pins);
        try E.expectEqual(context.CloseResult.busy, owner.tryProgressClose());
        admitted.end();
        try finish(&owner);
        try E.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
    };
}

test "Scalar FFTW every lease retains its acquiring allocator across callback replacement and close" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var first: LifecycleProbe = .{ .replace_at = 1 };
        var second: LifecycleProbe = .{};
        var unused: fixture.Allocations = .{ .budget = 0 };
        var owner = context.Context.init(first.allocator(), fixture.policy, .internal_closable);
        first.owner = &owner;
        second.owner = &owner;
        first.replacement = second.allocator();
        var input: [4]C = @splat(.{ .re = 1, .im = 0 });
        var output: [5]C = @splat(.{ .re = 777, .im = 888 });
        const handle = B.createTensorWithContext(&owner, .complex, 2, &.{ 2, 2 }, &input, &output, -1, 64).?;
        try E.expectEqual(@as(usize, 1), first.backing.stats().accepted);
        try E.expectEqual(@as(usize, 1), second.backing.stats().accepted);
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        try B.executeWithArrays(handle, &input, &output);
        try E.expectEqual(@as(R, 4), output[0].re);
        try E.expectEqual(@as(R, 777), output[4].re);
        try E.expectEqual(@as(usize, 2), second.backing.stats().accepted);
        try E.expectEqual(@as(usize, 1), second.backing.stats().freed);
        _ = owner.requestClose();
        try E.expectEqual(context.CloseResult.waiting_resources, owner.tryProgressClose());
        owner.allocator = unused.allocator();
        B.destroy(handle);
        try E.expectEqual(@as(usize, 2), first.free_snapshot.?.resource_pins);
        try E.expectEqual(@as(usize, 1), second.free_snapshot.?.resource_pins);
        try E.expectEqual(@as(usize, 0), unused.stats().attempts);
        try finish(&owner);
        try E.expectEqual(first.backing.stats().accepted, first.backing.stats().freed);
        try E.expectEqual(second.backing.stats().accepted, second.backing.stats().freed);
    }
}

test "Scalar FFTW admitted OOM retry after close rejects foreign owners without publishing output" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        var allocations: fixture.Allocations = .{ .budget = 2 };
        var foreign_allocations: fixture.Allocations = .{ .budget = 8 };
        var owner = context.Context.init(allocations.allocator(), fixture.policy, .internal_closable);
        var foreign = context.Context.init(foreign_allocations.allocator(), fixture.policy, .internal_closable);
        var input: [3]C = @splat(.{ .re = 1, .im = 0 });
        var output: [4]C = @splat(.{ .re = 777, .im = 888 });
        const handle = B.createTensorWithContext(&owner, .complex, 1, &.{3}, &input, &output, -1, 64).?;
        var admitted = try owner.begin();
        var wrong = try foreign.begin();
        try E.expectError(error.ContextMismatch, B.executeNewArraysAdmitted(handle, .complex, &wrong, &input, &output));
        wrong.end();
        _ = owner.requestClose();
        try E.expectError(error.Closing, B.executeWithArrays(handle, &input, &output));
        try E.expectError(error.OutOfMemory, B.executeNewArraysAdmitted(handle, .complex, &admitted, &input, &output));
        for (output) |v| try E.expectEqual(@as(R, 777), v.re);
        try E.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        allocations.budget = 8;
        try B.executeNewArraysAdmitted(handle, .complex, &admitted, &input, &output);
        try E.expectEqual(@as(R, 3), output[0].re);
        try E.expectEqual(@as(R, 777), output[3].re);
        B.destroy(handle);
        try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try E.expectEqual(context.CloseResult.busy, owner.tryProgressClose());
        admitted.end();
        try finish(&owner);
        try finish(&foreign);
        try E.expectEqual(allocations.stats().accepted, allocations.stats().freed);
        try E.expectEqual(@as(usize, 0), foreign_allocations.stats().accepted);
    }
}

const ConcurrentProbe = struct {
    owner: *context.Context = undefined,
    blocking: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    ready: std.atomic.Value(usize) = .init(0),
    finished: std.atomic.Value(usize) = .init(0),
    accepted: std.atomic.Value(usize) = .init(0),
    freed: std.atomic.Value(usize) = .init(0),
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(raw: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (self.blocking.load(.acquire)) {
            _ = self.ready.fetchAdd(1, .acq_rel);
            fixture.wait(&self.release);
        }
        const result = std.heap.c_allocator.rawAlloc(n, alignment, ret);
        if (result != null) _ = self.accepted.fetchAdd(1, .monotonic);
        return result;
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, data: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (self.owner.tryProgressClose() == .closed) @panic("concurrent FFTW free lost its resource pin");
        std.heap.c_allocator.rawFree(data, alignment, ret);
        _ = self.freed.fetchAdd(1, .monotonic);
    }
};

test "Scalar FFTW four simultaneous workspace admissions finish captured calls after close" {
    inline for (.{ f32, f64 }) |R| {
        const B = Boundary.Binding(R);
        const C = Scalar(R);
        const Worker = struct {
            handle: B.Handle,
            probe: *ConcurrentProbe,
            value: R,
            failed: bool = false,
            fn run(self: *@This()) void {
                defer _ = self.probe.finished.fetchAdd(1, .release);
                var input: [13]C = @splat(.{ .re = self.value, .im = 0 });
                var output: [14]C = @splat(.{ .re = 777, .im = 888 });
                B.executeNewArrays(self.handle, .complex, &input, &output) catch {
                    self.failed = true;
                    return;
                };
                if (@abs(@as(f64, output[0].re) - 13 * @as(f64, self.value)) > tolerance(R) * 16 or output[13].re != 777) self.failed = true;
                for (output[1..13]) |v| if (@abs(@as(f64, v.re)) > tolerance(R) * 16 or @abs(@as(f64, v.im)) > tolerance(R) * 16) {
                    self.failed = true;
                };
            }
        };
        var probe: ConcurrentProbe = .{};
        var owner = context.Context.init(probe.allocator(), fixture.policy, .internal_closable);
        probe.owner = &owner;
        var input: [13]C = @splat(.{ .re = 1, .im = 0 });
        var output: [13]C = @splat(.{ .re = 777, .im = 888 });
        const handle = B.createWithContext(&owner, .complex, .{ 13, 1, 1 }, &input, &output, -1, 64).?;
        var live = true;
        defer if (live) B.destroy(handle);
        var workers: [4]Worker = undefined;
        var threads: [4]std.Thread = undefined;
        var started: usize = 0;
        errdefer {
            probe.release.store(true, .release);
            for (threads[0..started]) |thread| thread.join();
        }
        probe.blocking.store(true, .release);
        for (&workers, &threads, 0..) |*worker, *thread, j| {
            worker.* = .{ .handle = handle, .probe = &probe, .value = @floatFromInt(j + 1) };
            thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
            started += 1;
        }
        while (probe.ready.load(.acquire) != 4 and probe.finished.load(.acquire) == 0) std.Thread.yield() catch {};
        try E.expectEqual(@as(usize, 4), probe.ready.load(.acquire));
        const paused = owner.snapshot();
        try E.expectEqual(@as(usize, 4), paused.active_operations);
        try E.expectEqual(@as(usize, 5), paused.resource_pins);
        _ = owner.requestClose();
        try E.expectEqual(context.CloseResult.busy, owner.tryProgressClose());
        try E.expectError(error.Closing, B.executeWithArrays(handle, &input, &output));
        probe.release.store(true, .release);
        for (threads) |thread| thread.join();
        started = 0;
        for (workers) |worker| try E.expect(!worker.failed);
        try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 1), owner.snapshot().resource_pins);
        try E.expectEqual(context.CloseResult.waiting_resources, owner.tryProgressClose());
        for (output) |v| try E.expectEqual(@as(R, 777), v.re);
        B.destroy(handle);
        live = false;
        try finish(&owner);
        try E.expectEqual(@as(usize, 5), probe.accepted.load(.acquire));
        try E.expectEqual(probe.accepted.load(.acquire), probe.freed.load(.acquire));
    }
}
