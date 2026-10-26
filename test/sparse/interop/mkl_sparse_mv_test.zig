// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Source-only candidate: these declarations have not been compiled or run.
const std = @import("std");
const binding = @import("sparse-interface-owner");
const Core = @import("zynum-core");
const execution = @import("zynum-execution");
const resources = @import("zynum-numerics-resources");
const fixture = execution.test_support;
const Context = execution.context.Context;
const P = Core.internal.numerics.SparsePattern;
const Typed = binding.Adapter;
const Raw = binding.Raw;
const Handles = binding.Handles;
const helpers = @import("handle_test.zig");
const equal = std.testing.expectEqual;
const expect = std.testing.expect;
const expectError = std.testing.expectError;
const general: Raw.MatrixDescriptor = .{ .matrix_type = 20, .mode = 0xfedcba98, .diag = 0x87654321 };

fn complex(comptime T: type) bool {
    return T == Core.ComplexF32 or T == Core.ComplexF64;
}
fn Real(comptime T: type) type {
    return if (T == f32 or T == Core.ComplexF32) f32 else f64;
}
fn number(comptime T: type, re: f64, im: f64) T {
    return if (comptime complex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn rawOperation(operation: P.Operation) u32 {
    return switch (operation) {
        .normal => 10,
        .transposed => 11,
        .adjoint => 12,
    };
}
fn release(handle: ?Handles.Handle) void {
    Typed.destroy(handle) catch @panic("valid MV handle destruction failed");
}

// Independent dense complex arithmetic in f64; no Core scalar/product helpers,
// sparse expansion, or CSR/CSC addressing is used by this reference.
const C = struct { re: f64, im: f64 };
fn add(a: C, b: C) C {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: C, b: C) C {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn asC(comptime T: type, value: T) C {
    return if (comptime complex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else .{ .re = @floatCast(value), .im = 0 };
}
fn denseExpected(comptime T: type, operation: P.Operation, alpha: T, input: []const T, beta: T, before: []const T, output: []C) void {
    const real = [_]f64{ 1, 0, 3, -2, 4, 0 };
    const imag = [_]f64{ 2, 0, -1, 1, 2, 0 };
    for (output, 0..) |*value, row| {
        var sum: C = .{ .re = 0, .im = 0 };
        for (input, 0..) |x, column| {
            const at = if (operation == .normal) row * 3 + column else column * 3 + row;
            const imaginary = if (complex(T)) imag[at] else 0;
            const a: C = .{ .re = real[at], .im = if (operation == .adjoint) -imaginary else imaginary };
            sum = add(sum, mul(a, asC(T, x)));
        }
        value.* = add(mul(asC(T, alpha), sum), mul(asC(T, beta), asC(T, before[row])));
    }
}
fn expectNumeric(comptime T: type, expected: C, actual: T) !void {
    const got = asC(T, actual);
    const tolerance: f64 = if (Real(T) == f32) 0.00005 else 0.000000000005;
    try std.testing.expectApproxEqAbs(expected.re, got.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, got.im, tolerance);
}

const Encoding = enum { split, null_end, shifted };
fn MatrixFixture(comptime T: type, comptime I: type) type {
    return struct {
        starts: [3]I,
        ends: [3]I,
        offsets: [4]I,
        ordinals: [8]I,
        values: [8]T,
        axis: P.MajorAxis,
        base: u32,
        encoding: Encoding,

        fn init(axis: P.MajorAxis, base: u32, encoding: Encoding) @This() {
            const b: I = @intCast(base);
            var result: @This() = .{ .starts = @splat(b), .ends = @splat(b), .offsets = @splat(b), .ordinals = @splat(-99), .values = @splat(number(T, std.math.nan(f64), std.math.nan(f64))), .axis = axis, .base = base, .encoding = encoding };
            if (axis == .row) {
                // Dense [[1+2i,0,3-i],[-2+i,4+2i,0]]. Row0 is unsorted
                // and its column2 appears twice, in the listed physical order.
                const indices = [_]I{ b + 2, b, b + 2, b + 1, b };
                const values = [_]T{ number(T, 1, 1), number(T, 1, 2), number(T, 2, -2), number(T, 4, 2), number(T, -2, 1) };
                result.offsets = .{ b, b + 3, b + 5, b + 5 };
                if (encoding == .split) {
                    result.starts = .{ b + 1, b + 5, b };
                    result.ends = .{ b + 4, b + 7, b };
                    for ([_]usize{ 1, 2, 3, 5, 6 }, 0..) |at, index| {
                        result.ordinals[at] = indices[index];
                        result.values[at] = values[index];
                    }
                } else {
                    @memcpy(result.ordinals[0..5], &indices);
                    @memcpy(result.values[0..5], &values);
                }
            } else {
                const indices = [_]I{ b + 1, b, b + 1, b, b };
                const values = [_]T{ number(T, -2, 1), number(T, 1, 2), number(T, 4, 2), number(T, 1, 1), number(T, 2, -2) };
                result.offsets = .{ b, b + 2, b + 3, b + 5 };
                if (encoding == .split) {
                    result.starts = .{ b + 1, b + 4, b + 6 };
                    result.ends = .{ b + 3, b + 5, b + 8 };
                    for ([_]usize{ 1, 2, 4, 6, 7 }, 0..) |at, index| {
                        result.ordinals[at] = indices[index];
                        result.values[at] = values[index];
                    }
                } else {
                    @memcpy(result.ordinals[0..5], &indices);
                    @memcpy(result.values[0..5], &values);
                }
            }
            return result;
        }
        fn raw(self: *@This()) Raw.Input(T, I) {
            return .{ .base = self.base, .rows = 2, .columns = 3, .axis = self.axis, .starts = if (self.encoding == .split) &self.starts else &self.offsets, .ends = switch (self.encoding) {
                .split => &self.ends,
                .null_end => null,
                .shifted => self.offsets[1..].ptr,
            }, .ordinals = &self.ordinals, .values = &self.values };
        }
    };
}

test "Sparse MV binding uses the original Product scalar Context and resource owner" {
    comptime {
        if (binding.Provider.Product != Core.internal.numerics.SparseProduct or binding.Provider.Pattern != P or
            binding.Provider.OwnerIdentity != Core or binding.Provider.Execution != execution or binding.Provider.Resources != resources or
            Handles.Context != Context or Raw.Context != Context or Raw.Handle != Typed.Handle)
            @compileError("MV introduced an implementation or resource owner");
        if (Raw.ConstPointer(Core.ComplexF64) != ?[*]align(1) const Core.ComplexF64)
            @compileError("raw input must retain its readonly capability");
    }
    try expect(binding.Provider.Execution.getContext() == execution.getContext());
}

test "Sparse MV typed and raw rectangular real complex products match independent dense arithmetic with full padding" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            for ([_]P.MajorAxis{ .row, .column }) |axis| {
                for ([_]u32{ 0, 1 }) |base| {
                    for ([_]Encoding{ .split, .null_end, .shifted }) |encoding| {
                        var matrix = MatrixFixture(T, I).init(axis, base, encoding);
                        const matrix_values = std.mem.asBytes(&matrix.values).*;
                        const matrix_ordinals = matrix.ordinals;
                        var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                        var handle: ?Handles.Handle = null;
                        try Raw.createBorrowed(T, I, &context, matrix.raw(), &handle);
                        {
                            defer release(handle);
                            for ([_]P.Operation{ .normal, .transposed, .adjoint }) |operation| {
                                const input = [_]T{ number(T, 2, 1), number(T, -1, 2), number(T, 3, -2) };
                                const nx: usize = if (operation == .normal) 3 else 2;
                                const ny: usize = if (operation == .normal) 2 else 3;
                                var typed: [5]T = @splat(number(T, -37, 29));
                                for (typed[1..][0..ny], 0..) |*y, index| y.* = number(T, 5 + @as(f64, @floatFromInt(index)), -2 - @as(f64, @floatFromInt(index)));
                                var raw = typed;
                                const before = typed;
                                var expected: [3]C = undefined;
                                const alpha = number(T, 2, -1);
                                const beta = number(T, -1, 1);
                                denseExpected(T, operation, alpha, input[0..nx], beta, before[1..][0..ny], expected[0..ny]);
                                try Typed.matrixVectorMultiply(T, I, handle, operation, alpha, input[0..nx], beta, typed[1..][0..ny]);
                                try Raw.matrixVectorMultiply(T, I, handle, rawOperation(operation), general, alpha, &input, beta, raw[1..].ptr);
                                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&typed), std.mem.asBytes(&raw));
                                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[0]), std.mem.asBytes(&typed[0]));
                                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before[1 + ny ..]), std.mem.sliceAsBytes(typed[1 + ny ..]));
                                for (expected[0..ny], typed[1..][0..ny]) |want, got| try expectNumeric(T, want, got);
                            }
                            try std.testing.expectEqualSlices(u8, &matrix_values, std.mem.asBytes(&matrix.values));
                            try std.testing.expectEqualSlices(I, &matrix_ordinals, &matrix.ordinals);
                        }
                        try helpers.close(&context);
                    }
                }
            }
        }
    }
}

test "Sparse MV duplicate cancellation follows stored order for every scalar width orientation and transform" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            const large: f64 = if (Real(T) == f32) 16777216 else 9007199254740992;
            var starts = [_]I{0};
            var ends = [_]I{3};
            var ordinals = [_]I{ 0, 0, 0 };
            var values = [_]T{ number(T, large, 0), number(T, -large, 0), number(T, 1, 0) };
            for ([_]P.MajorAxis{ .row, .column }) |axis| {
                var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                var h: ?Handles.Handle = null;
                try Typed.createBorrowed(T, I, &context, .{ .base = 0, .rows = 1, .columns = 1, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
                {
                    defer release(h);
                    for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
                        var output = [_]T{number(T, 99, 99)};
                        try Typed.matrixVectorMultiply(T, I, h, op, number(T, 1, 0), &.{number(T, 1, 0)}, number(T, 0, 0), &output);
                        try expectNumeric(T, .{ .re = 1, .im = 0 }, output[0]);
                    }
                }
                try helpers.close(&context);
            }
        }
    }
}

test "Sparse MV zero scales avoid unused NaN Inf values but still validate structure and vector capacities" {
    @setEvalBranchQuota(10000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            var starts = [_]I{ 0, 1 };
            var ends = [_]I{ 1, 2 };
            var ordinals = [_]I{ 0, 1 };
            var values = [_]T{ number(T, std.math.nan(f64), std.math.inf(f64)), number(T, std.math.inf(f64), std.math.nan(f64)) };
            const poison = [_]T{ number(T, std.math.nan(f64), std.math.nan(f64)), number(T, std.math.nan(f64), std.math.nan(f64)) };
            for ([_]P.MajorAxis{ .row, .column }) |axis| {
                var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                var h: ?Handles.Handle = null;
                try Typed.createBorrowed(T, I, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
                {
                    defer release(h);
                    for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
                        var output = [_]T{ number(T, 2, 3), number(T, -5, 7) };
                        try Raw.matrixVectorMultiply(T, I, h, rawOperation(op), general, number(T, -0.0, -0.0), &poison, number(T, 2, 0), &output);
                        try expectNumeric(T, asC(T, number(T, 4, 6)), output[0]);
                        try expectNumeric(T, asC(T, number(T, -10, 14)), output[1]);
                        output = poison;
                        try Typed.matrixVectorMultiply(T, I, h, op, number(T, 0, 0), &poison, number(T, -0.0, -0.0), &output);
                        const zeros = [_]T{ number(T, 0, 0), number(T, 0, 0) };
                        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&zeros), std.mem.asBytes(&output));
                    }
                    var canary = [_]T{ number(T, 13, 17), number(T, 19, 23) };
                    const before = std.mem.asBytes(&canary).*;
                    try expectError(error.DimensionMismatch, Typed.matrixVectorMultiply(T, I, h, .normal, number(T, 0, 0), poison[0..0], number(T, 0, 0), &canary));
                    try expectError(error.NullInput, Raw.matrixVectorMultiply(T, I, h, 10, general, number(T, 0, 0), null, number(T, 0, 0), &canary));
                    try expectError(error.NullOutput, Raw.matrixVectorMultiply(T, I, h, 10, general, number(T, 0, 0), &poison, number(T, 0, 0), null));
                    ordinals[0] = 9;
                    try expectError(error.InvalidOrdinal, Raw.matrixVectorMultiply(T, I, h, 10, general, number(T, 0, 0), null, number(T, 0, 0), &canary));
                    ordinals[0] = 0;
                    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&canary));
                    values = .{ number(T, 2, 0), number(T, 3, 0) };
                    const x = [_]T{ number(T, 5, 7), number(T, -2, 3) };
                    canary = poison;
                    try Typed.matrixVectorMultiply(T, I, h, .adjoint, number(T, 1, 0), &x, number(T, 0, 0), &canary);
                    try expectNumeric(T, asC(T, number(T, 10, 14)), canary[0]);
                    try expectNumeric(T, asC(T, number(T, -6, 9)), canary[1]);
                }
                try helpers.close(&context);
                values = .{ number(T, std.math.nan(f64), std.math.inf(f64)), number(T, std.math.inf(f64), std.math.nan(f64)) };
            }
        }
    }
}

test "Sparse MV descriptor type mismatch and CSC metadata faults leave nonzero complete outputs unchanged" {
    fixture.setThreads(1);
    var matrix = MatrixFixture(f64, i32).init(.column, 1, .split);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, matrix.raw(), &h);
    {
        defer release(h);
        var output = [_]f64{ 31, 37, 41, 43 };
        const before = std.mem.asBytes(&output).*;
        const x = [_]f64{ 2, 3, 5 };
        try expectError(error.NullHandle, Raw.matrixVectorMultiply(f64, i32, null, 99, general, 0, null, 0, &output));
        for ([_]u32{ 21, 22, 23, 24, 25, 26 }) |kind| {
            var descriptor = general;
            descriptor.matrix_type = kind;
            try expectError(if (kind <= 24) error.InvalidDescriptor else error.UnsupportedDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, descriptor, 0, null, 0, &output));
            try expectError(error.InvalidOperation, Raw.matrixVectorMultiply(f64, i32, h, 99, descriptor, 0, null, 0, &output));
        }
        var unknown = general;
        unknown.matrix_type = 0xffffffff;
        try expectError(error.InvalidDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, unknown, 0, null, 0, &output));
        try expectError(error.WrongIndexWidth, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 1, &x, 0, &output));
        var float_output = [_]f32{ 7, 11, 13, 17 };
        const float_before = std.mem.asBytes(&float_output).*;
        try expectError(error.WrongScalar, Raw.matrixVectorMultiply(f32, i32, h, 10, general, 0, null, 0, &float_output));
        try std.testing.expectEqualSlices(u8, &float_before, std.mem.asBytes(&float_output));
        matrix.ordinals[1] = 99;
        try expectError(error.InvalidOrdinal, Typed.matrixVectorMultiply(f64, i32, h, .normal, 0, &x, 0, output[0..2]));
        matrix.ordinals[1] = 2;
        matrix.ends[2] = 99;
        try expectError(error.InvalidOffsets, Raw.matrixVectorMultiply(f64, i32, h, 10, general, 0, null, 0, &output));
        matrix.ends[2] = 9;
        matrix.starts[1] = 2;
        matrix.ends[1] = 4; // Overlap only valid ordinal positions, not poison gaps.
        try expectError(error.UnsupportedDomain, Raw.matrixVectorMultiply(f64, i32, h, 10, general, 0, null, 0, &output));
        matrix.starts[1] = 5;
        matrix.ends[1] = 6;
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&output));
    }
    try helpers.close(&context);
}

test "Sparse MV rejects partial vector array shell and control aliases while allowing unused capacity and readonly sharing" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts: [2]i64 align(@alignOf(f64)) = .{ 0, 1 };
    var ends: [2]i64 align(@alignOf(f64)) = .{ 1, 2 };
    var ordinals: [2]i64 align(@alignOf(f64)) = .{ 0, 1 };
    var values = [_]f64{ 2, 3, 17, 19 };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
    {
        defer release(h);
        var shared = [_]f64{ 23, 29, 31 };
        const before = std.mem.asBytes(&shared).*;
        try expectError(error.BufferOverlap, Typed.matrixVectorMultiply(f64, i64, h, .normal, 0, shared[0..2], 0, shared[1..3]));
        try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, &shared, 0, shared[1..].ptr));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&shared));
        const x = [_]f64{ 4, 5 };
        const values_before = std.mem.asBytes(&values).*;
        try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, &x, 0, values[1..].ptr));
        try std.testing.expectEqualSlices(u8, &values_before, std.mem.asBytes(&values));
        const starts_before = starts;
        const ends_before = ends;
        const ordinals_before = ordinals;
        for ([_]Raw.Pointer(f64){ @ptrCast(&starts), @ptrCast(&ends), @ptrCast(&ordinals), @ptrCast(h.?) }) |output| {
            try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, &x, 0, output));
        }
        const control = try Handles.testing.retainControl(f64, i64, h.?);
        try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, &x, 0, @ptrCast(control)));
        try equal(@as(usize, 2), control.references);
        control.release();
        try std.testing.expectEqualSlices(i64, &starts_before, &starts);
        try std.testing.expectEqualSlices(i64, &ends_before, &ends);
        try std.testing.expectEqualSlices(i64, &ordinals_before, &ordinals);
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try Typed.matrixVectorMultiply(f64, i64, h, .normal, 1, &x, 0, values[2..4]);
        try std.testing.expectEqualSlices(f64, &.{ 2, 3, 8, 15 }, &values);
        var result = [_]f64{ 47, 53 };
        try Typed.matrixVectorMultiply(f64, i64, h, .normal, 1, values[0..2], 0, &result);
        try std.testing.expectEqualSlices(f64, &.{ 4, 9 }, &result);
    }
    try helpers.close(&context);
}

test "Sparse MV empty raw prefixes avoid alignment casts and native byte overflow precedes access" {
    fixture.setThreads(1);
    var bytes: [32]u8 align(16) = @splat(0xa5);
    const unaligned_input: Raw.ConstPointer(f64) = @ptrCast(&bytes[1]);
    const unaligned_output: Raw.Pointer(f64) = @ptrCast(&bytes[1]);
    const before = bytes;
    for ([_]P.MajorAxis{ .row, .column }) |axis| {
        var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
        var offsets = [_]i64{ 0, 0, 0, 0 };
        var h: ?Handles.Handle = null;
        try Raw.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = 0, .columns = 3, .axis = axis, .starts = if (axis == .row) null else &offsets, .ends = null, .ordinals = null, .values = null }, &h);
        {
            defer release(h);
            const x = [_]f64{ 2, 3, 5 };
            try Raw.matrixVectorMultiply(f64, i64, h, 10, general, 1, &x, 0, unaligned_output);
            var y = [_]f64{ 7, 11, 13 };
            try Raw.matrixVectorMultiply(f64, i64, h, 11, general, std.math.nan(f64), unaligned_input, 2, &y);
            try std.testing.expectEqualSlices(f64, &.{ 14, 22, 26 }, &y);
            try Raw.matrixVectorMultiply(f64, i64, h, 12, general, 1, null, 0, &y);
            try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, &y);
            try expectError(error.InvalidAlignment, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, unaligned_input, 0, null));
            try expectError(error.InvalidAlignment, Raw.matrixVectorMultiply(f64, i64, h, 11, general, 0, null, 0, unaligned_output));
        }
        try helpers.close(&context);
    }
    try std.testing.expectEqualSlices(u8, &before, &bytes);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    const too_many: i64 = @intCast(std.math.maxInt(usize) / @sizeOf(f64) + 1);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = 0, .columns = too_many, .axis = .row, .starts = null, .ends = null, .ordinals = null, .values = null }, &h);
    {
        defer release(h);
        const live_input: f64 = 7;
        // A real pointer, but byte overflow must reject before any huge slice
        // or numerical access could be formed. No giant allocation is claimed.
        try expectError(error.SizeOverflow, Raw.matrixVectorMultiply(f64, i64, h, 10, general, 0, @ptrCast(&live_input), 0, null));
    }
    try helpers.close(&context);
}

test "Sparse MV one source Context admission covers closing computation and complete output with no allocation" {
    const Observation = struct {
        context: Context,
        enabled: bool = false,
        admitted: usize = 0,
        finishing: usize = 0,
        output: [4]f64 = .{ 41, 7, 11, 43 },
        fn observe(context: *Context, event: execution.context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("context", context);
            if (!self.enabled) return;
            const snapshot = context.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins != 1) @panic("MV lost single source admission/control pin");
            switch (event) {
                .admitted => {
                    self.admitted += 1;
                    if (self.output[1] != 7 or self.output[2] != 11) @panic("MV wrote before admission");
                    _ = context.requestClose();
                },
                .finishing => {
                    self.finishing += 1;
                    if (!std.mem.eql(f64, &self.output, &.{ 41, 6, 20, 43 })) @panic("MV admission ended before full output/padding commit");
                },
            }
            if (context.tryProgressClose() != .busy) @panic("MV close escaped active operation");
        }
    };
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var observed: Observation = .{ .context = Context.init(spy.allocator(), fixture.policy, .internal_closable) };
    spy.ctx = &observed.context;
    Context.testing.configure(&observed.context, 1, Observation.observe);
    var starts = [_]i32{ 0, 1 };
    var ends = [_]i32{ 1, 2 };
    var indices = [_]i32{ 0, 1 };
    var values = [_]f64{ 2, 4 };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, &observed.context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .column, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, &h);
    try expect(try Handles.borrowedContext(h.?) == &observed.context);
    try expect(execution.getContext() != &observed.context);
    observed.enabled = true;
    const attempts = spy.attempts;
    const x = [_]f64{ 3, 5 };
    try Raw.matrixVectorMultiply(f64, i32, h, 10, general, 1, &x, 0, observed.output[1..].ptr);
    try equal(@as(usize, 1), observed.admitted);
    try equal(@as(usize, 1), observed.finishing);
    try equal(attempts, spy.attempts);
    try equal(@as(usize, 0), observed.context.snapshot().active_operations);
    try equal(execution.context.CloseResult.waiting_resources, observed.context.tryProgressClose());
    const before = std.mem.asBytes(&observed.output).*;
    try expectError(error.Closing, Raw.matrixVectorMultiply(f64, i32, h, 99, general, 0, null, 0, &observed.output));
    try expectError(error.Closing, Typed.matrixVectorMultiply(f64, i32, h, .normal, 0, &.{}, 0, &observed.output));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&observed.output));
    try Typed.destroy(h);
    try spy.balanced();
    try helpers.close(&observed.context);
}

test "Sparse MV admission overflow and foreign admitted seams preserve output without allocator fallback" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    Context.testing.configure(&context, 1, null);
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var indices = [_]i32{0};
    var values = [_]f64{2};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, &context, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, &h);
    {
        defer release(h);
        var source_op = try context.begin();
        const attempts = spy.attempts;
        var output = [_]f64{ 17, 19, 23 };
        const before = std.mem.asBytes(&output).*;
        try expectError(error.CounterOverflow, Typed.matrixVectorMultiply(f64, i32, h, .normal, 0, &.{}, 0, output[1..2]));
        try expectError(error.CounterOverflow, Raw.matrixVectorMultiply(f64, i32, h, 99, general, 0, null, 0, output[1..].ptr));
        const exported = try Typed.matrixAdmitted(f64, i32, &source_op, h.?);
        var foreign = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
        var foreign_op = try foreign.begin();
        try expectError(error.ContextMismatch, Typed.matrixAdmitted(f64, i32, &foreign_op, h.?));
        try expectError(error.ContextMismatch, Typed.productAdmitted(f64, i32, &foreign_op, h.?, exported, .normal, 1, &.{3}, 0, output[1..2]));
        foreign_op.end();
        source_op.end();
        try helpers.close(&foreign);
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&output));
        try equal(attempts, spy.attempts);
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try equal(@as(usize, 0), context.snapshot().active_operations);
    }
    try spy.balanced();
    try helpers.close(&context);
}

fn structureView(kind: u32, lower: bool, unit: bool) Typed.MatrixSemantics {
    const half: binding.Provider.Product.Half = .{ .triangle = if (lower) .lower else .upper, .diagonal = if (unit) .unit else .stored };
    return switch (kind) {
        21 => .{ .symmetric = half },
        22 => .{ .hermitian = half },
        23 => .{ .triangular = half },
        24 => .{ .diagonal = half.diagonal },
        else => unreachable,
    };
}
fn structuredExpected(comptime T: type, kind: u32, lower: bool, unit: bool, op: P.Operation, x: [2]T, old: [2]T, alpha: T, beta: T) [2]C {
    // Literal full matrices, independent of storage and of the Core's decoder.
    const imag: f64 = if (complex(T)) 1 else 0;
    const zero: C = .{ .re = 0, .im = 0 };
    const upper: C = .{ .re = 2, .im = 3 * imag };
    const lower_value: C = .{ .re = 2, .im = -3 * imag };
    const d0: C = if (unit) .{ .re = 1, .im = 0 } else .{ .re = 1, .im = if (kind == 22) 0 else 7 * imag };
    const d1: C = if (unit) .{ .re = 1, .im = 0 } else .{ .re = 4, .im = if (kind == 22) 0 else -2 * imag };
    const matrix: [4]C = switch (kind) {
        21 => .{ d0, if (lower) lower_value else upper, if (lower) lower_value else upper, d1 },
        22 => .{ d0, upper, lower_value, d1 },
        23 => .{ d0, if (lower) zero else upper, if (lower) lower_value else zero, d1 },
        24 => .{ d0, zero, zero, d1 },
        else => unreachable,
    };
    var result: [2]C = undefined;
    for (&result, 0..) |*out, row| {
        var sum = zero;
        for (x, 0..) |value, column| {
            var a = matrix[if (op == .normal) 2 * row + column else 2 * column + row];
            if (op == .adjoint) a.im = -a.im;
            sum = add(sum, mul(a, asC(T, value)));
        }
        out.* = add(mul(asC(T, alpha), sum), mul(asC(T, beta), asC(T, old[row])));
    }
    return result;
}

test "Sparse structured MV typed raw paths match independent square dense products with complete padding" {
    @setEvalBranchQuota(30000);
    fixture.setThreads(1);
    comptime {
        if (Typed.MatrixSemantics != Core.internal.numerics.SparseProduct.MatrixSemantics)
            @compileError("structured MV must retain the actual Core semantics identity");
    }
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]I{ 0, 1 }) |base| {
                var starts = [_]I{ base + 1, base + 4 };
                var ends = [_]I{ base + 3, base + 6 };
                var ordinals = [_]I{ -99, base + 1, base, -99, base + 1, base };
                const poison = number(T, std.math.nan(f64), std.math.nan(f64));
                var values = [_]T{ poison, if (axis == .row) number(T, 2, 3) else number(T, 2, -3), number(T, 1, 7), poison, number(T, 4, -2), if (axis == .row) number(T, 2, -3) else number(T, 2, 3) };
                var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                var handle: ?Handles.Handle = null;
                try Typed.createBorrowed(T, I, &context, .{ .base = @intCast(base), .rows = 2, .columns = 2, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &handle);
                {
                    defer release(handle);
                    const storage_before = std.mem.asBytes(&values).*;
                    const x = [_]T{ number(T, 1, 1), number(T, 2, -1) };
                    const alpha = number(T, 2, -1);
                    const beta = number(T, -1, 1);
                    for ([_]u32{ 21, 22, 23, 24 }) |kind| for ([_]bool{ false, true }) |lower| {
                        for ([_]bool{ false, true }) |unit| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
                            const descriptor: Raw.MatrixDescriptor = .{ .matrix_type = kind, .mode = if (kind == 24) 0xfedcba98 else if (lower) 40 else 41, .diag = if (unit) 51 else 50 };
                            var typed = [_]T{ number(T, -31, 29), number(T, 7, -3), number(T, 11, 2), number(T, 37, -41) };
                            var raw = typed;
                            const before = typed;
                            const want = structuredExpected(T, kind, lower, unit, op, x, .{ before[1], before[2] }, alpha, beta);
                            try Typed.structuredMatrixVectorMultiply(T, I, handle, op, structureView(kind, lower, unit), alpha, &x, beta, typed[1..3]);
                            try Raw.matrixVectorMultiply(T, I, handle, rawOperation(op), descriptor, alpha, &x, beta, raw[1..].ptr);
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&typed), std.mem.asBytes(&raw));
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[0]), std.mem.asBytes(&typed[0]));
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[3]), std.mem.asBytes(&typed[3]));
                            for (want, typed[1..3]) |expected, actual| try expectNumeric(T, expected, actual);
                        };
                    };
                    try std.testing.expectEqualSlices(u8, &storage_before, std.mem.asBytes(&values));
                }
                try helpers.close(&context);
            };
        }
    }
}

test "Sparse structured MV empty stored unit matrices and zero length raw prefixes preserve their distinct meanings" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts = [_]i64{ 0, 0 };
    var ends = starts;
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(Core.ComplexF64, i64, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .column, .starts = &starts, .ends = &ends, .ordinals = &.{}, .values = &.{} }, &h);
    {
        defer release(h);
        const T = Core.ComplexF64;
        const x = [_]T{ number(T, 1, 1), number(T, 2, -1) };
        for ([_]u32{ 21, 22, 23, 24 }) |kind| for ([_]u32{ 10, 11, 12 }) |op| {
            var y = [_]T{ number(T, -31, 29), number(T, std.math.nan(f64), std.math.nan(f64)), number(T, std.math.nan(f64), std.math.nan(f64)), number(T, 37, -41) };
            const before = y;
            try Raw.matrixVectorMultiply(T, i64, h, op, .{ .matrix_type = kind, .mode = 40, .diag = 51 }, number(T, 2, 0), &x, number(T, 0, 0), y[1..].ptr);
            try expectNumeric(T, .{ .re = 2, .im = 2 }, y[1]);
            try expectNumeric(T, .{ .re = 4, .im = -2 }, y[2]);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[0]), std.mem.asBytes(&y[0]));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[3]), std.mem.asBytes(&y[3]));
        };
    }
    var empty: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, .{ .base = 1, .rows = 0, .columns = 0, .axis = .row, .starts = null, .ends = null, .ordinals = null, .values = null }, &empty);
    {
        defer release(empty);
        var bytes: [17]u8 align(16) = @splat(0xa5);
        const before = bytes;
        for ([_]u32{ 21, 22, 23, 24 }) |kind| {
            try Raw.matrixVectorMultiply(f64, i32, empty, 12, .{ .matrix_type = kind, .mode = 41, .diag = 51 }, std.math.nan(f64), @ptrCast(&bytes[1]), std.math.nan(f64), @ptrCast(&bytes[1]));
            try Raw.matrixVectorMultiply(f64, i32, empty, 10, .{ .matrix_type = kind, .mode = 40, .diag = 50 }, 1, null, 0, null);
        }
        try std.testing.expectEqualSlices(u8, &before, &bytes);
    }
    try helpers.close(&context);
}

test "Sparse structured MV ignored numerical entries retain full metadata capacities and scale checks" {
    fixture.setThreads(1);
    const T = Core.ComplexF64;
    const nan = std.math.nan(f64);
    var starts = [_]i32{ 0, 2 };
    var ends = [_]i32{ 2, 4 };
    var ordinals = [_]i32{ 0, 1, 0, 1 };
    var values = [_]T{ number(T, 1, nan), number(T, 2, 3), number(T, nan, nan), number(T, 4, nan) };
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(T, i32, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
    {
        defer release(h);
        const x = [_]T{ number(T, 1, 1), number(T, 2, -1) };
        var y = [_]T{ number(T, 31, 37), number(T, 41, 43) };
        try Raw.matrixVectorMultiply(T, i32, h, 10, .{ .matrix_type = 22, .mode = 41, .diag = 50 }, number(T, 1, 0), &x, number(T, 0, 0), &y);
        try expectNumeric(T, .{ .re = 8, .im = 5 }, y[0]);
        try expectNumeric(T, .{ .re = 13, .im = -5 }, y[1]);
        @memset(&values, number(T, nan, nan));
        const before = std.mem.asBytes(&y).*;
        for ([_]u32{ 21, 22, 23, 24 }) |kind| {
            const desc: Raw.MatrixDescriptor = .{ .matrix_type = kind, .mode = 41, .diag = 51 };
            try Raw.matrixVectorMultiply(T, i32, h, 10, desc, number(T, -0.0, -0.0), &values, number(T, 1, 0), &y);
            try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&y));
            var zero_output: [2]T = @splat(number(T, nan, nan));
            try Raw.matrixVectorMultiply(T, i32, h, 12, desc, number(T, 0, -0.0), &values, number(T, -0.0, 0), &zero_output);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]T{ number(T, 0, 0), number(T, 0, 0) }), std.mem.asBytes(&zero_output));
            try expectError(error.NullInput, Raw.matrixVectorMultiply(T, i32, h, 10, desc, number(T, 0, 0), null, number(T, 0, 0), &y));
            try expectError(error.DimensionMismatch, Typed.structuredMatrixVectorMultiply(T, i32, h, .normal, structureView(kind, false, true), number(T, 0, 0), x[0..1], number(T, 0, 0), &y));
            ordinals[2] = 2; // Ignored opposite-half numeric value still has an invalid index.
            try expectError(error.InvalidOrdinal, Raw.matrixVectorMultiply(T, i32, h, 10, desc, number(T, 0, 0), null, number(T, 0, 0), &y));
            ordinals[2] = 0;
            ends[1] = 5;
            try expectError(error.InvalidOffsets, Raw.matrixVectorMultiply(T, i32, h, 10, desc, number(T, 0, 0), null, number(T, 0, 0), &y));
            ends[1] = 4;
            try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&y));
        }
    }
    try helpers.close(&context);
}

test "Sparse structured descriptor decoding distinguishes unsupported domains from invalid subordinate fields" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var matrix = MatrixFixture(f64, i32).init(.row, 0, .split);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, matrix.raw(), &h);
    {
        defer release(h);
        var y = [_]f64{ 31, 37, 41, 43 };
        const before = std.mem.asBytes(&y).*;
        for ([_]u32{ 21, 22, 23 }) |kind| {
            try expectError(error.UnsupportedDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, .{ .matrix_type = kind, .mode = 42, .diag = 99 }, 0, null, 0, &y));
            try expectError(error.InvalidDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, .{ .matrix_type = kind, .mode = 99, .diag = 50 }, 0, null, 0, &y));
            try expectError(error.InvalidDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, .{ .matrix_type = kind, .mode = 40, .diag = 99 }, 0, null, 0, &y));
        }
        for ([_]u32{ 21, 22, 23, 24 }) |kind| {
            const desc: Raw.MatrixDescriptor = .{ .matrix_type = kind, .mode = if (kind == 24) 0xffffffff else 40, .diag = 51 };
            try expectError(error.UnsupportedDomain, Raw.matrixVectorMultiply(f64, i32, h, 10, desc, 0, null, 0, &y));
            try expectError(error.UnsupportedDomain, Typed.structuredMatrixVectorMultiply(f64, i32, h, .normal, structureView(kind, true, true), 0, &.{}, 0, &y));
            try expectError(error.WrongIndexWidth, Raw.matrixVectorMultiply(f64, i64, h, 10, desc, 0, null, 0, &y));
            var float_output = [_]f32{ 17, 19, 23 };
            const float_before = std.mem.asBytes(&float_output).*;
            try expectError(error.WrongScalar, Raw.matrixVectorMultiply(f32, i32, h, 10, desc, 0, null, 0, &float_output));
            try std.testing.expectEqualSlices(u8, &float_before, std.mem.asBytes(&float_output));
            matrix.ordinals[1] = 99;
            try expectError(error.InvalidOrdinal, Raw.matrixVectorMultiply(f64, i32, h, 10, desc, 0, null, 0, &y));
            try expectError(error.InvalidOperation, Raw.matrixVectorMultiply(f64, i32, h, 99, desc, 0, null, 0, &y));
            matrix.ordinals[1] = 2;
        }
        for ([_]u32{ 25, 26 }) |kind| try expectError(error.UnsupportedDescriptor, Raw.matrixVectorMultiply(f64, i32, h, 10, .{ .matrix_type = kind, .mode = 99, .diag = 99 }, 0, null, 0, &y));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&y));
    }
    try helpers.close(&context);
}

test "Sparse structured unit output rejects partial vector and ignored data shell control aliases before mutation" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts: [2]i64 align(@alignOf(f64)) = .{ 0, 1 };
    var ends: [2]i64 align(@alignOf(f64)) = .{ 1, 2 };
    var ordinals: [2]i64 align(@alignOf(f64)) = .{ 0, 1 };
    var values = [_]f64{ std.math.nan(f64), std.math.inf(f64), 17 };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
    {
        defer release(h);
        const desc: Raw.MatrixDescriptor = .{ .matrix_type = 24, .mode = 0xffffffff, .diag = 51 };
        var shared = [_]f64{ 23, 29, 31 };
        const before = std.mem.asBytes(&shared).*;
        try expectError(error.BufferOverlap, Typed.structuredMatrixVectorMultiply(f64, i64, h, .normal, .{ .diagonal = .unit }, 0, shared[0..2], 0, shared[1..3]));
        try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, desc, 0, &shared, 0, shared[1..].ptr));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&shared));
        const arrays_before = .{ starts, ends, ordinals, std.mem.asBytes(&values).* };
        const x = [_]f64{ 4, 5 };
        const control = try Handles.testing.retainControl(f64, i64, h.?);
        for ([_]Raw.Pointer(f64){ values[1..].ptr, @ptrCast(&starts), @ptrCast(&ends), @ptrCast(&ordinals), @ptrCast(h.?), @ptrCast(control) }) |out| {
            try expectError(error.BufferOverlap, Raw.matrixVectorMultiply(f64, i64, h, 10, desc, 0, &x, 0, out));
        }
        try equal(@as(usize, 2), control.references);
        control.release();
        try std.testing.expectEqualSlices(i64, &arrays_before[0], &starts);
        try std.testing.expectEqualSlices(i64, &arrays_before[1], &ends);
        try std.testing.expectEqualSlices(i64, &arrays_before[2], &ordinals);
        try std.testing.expectEqualSlices(u8, &arrays_before[3], std.mem.asBytes(&values));
    }
    try helpers.close(&context);
}

test "Sparse structured MV source admission survives close through final output without allocating" {
    const Observation = struct {
        context: Context,
        enabled: bool = false,
        admitted: usize = 0,
        finishing: usize = 0,
        output: [4]f64 = .{ 41, 7, 11, 43 },
        fn observe(context: *Context, event: execution.context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("context", context);
            if (!self.enabled) return;
            const snapshot = context.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins != 1) @panic("structured MV lost its admitted owner");
            switch (event) {
                .admitted => {
                    self.admitted += 1;
                    if (!std.mem.eql(f64, &self.output, &.{ 41, 7, 11, 43 })) @panic("structured MV wrote before admission");
                    _ = context.requestClose();
                },
                .finishing => {
                    self.finishing += 1;
                    if (!std.mem.eql(f64, &self.output, &.{ 41, 13, 11, 43 })) @panic("structured MV ended before complete output");
                },
            }
            if (context.tryProgressClose() != .busy) @panic("structured MV prematurely closed");
        }
    };
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var observed: Observation = .{ .context = Context.init(spy.allocator(), fixture.policy, .internal_closable) };
    spy.ctx = &observed.context;
    Context.testing.configure(&observed.context, 1, Observation.observe);
    var starts = [_]i32{ 0, 1 };
    var ends = [_]i32{ 1, 1 };
    var ordinals = [_]i32{1};
    var values = [_]f64{2};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, &observed.context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
    observed.enabled = true;
    const attempts = spy.attempts;
    const x = [_]f64{ 3, 5 };
    try Raw.matrixVectorMultiply(f64, i32, h, 10, .{ .matrix_type = 21, .mode = 41, .diag = 51 }, 1, &x, 0, observed.output[1..].ptr);
    try equal(@as(usize, 1), observed.admitted);
    try equal(@as(usize, 1), observed.finishing);
    try equal(attempts, spy.attempts);
    try equal(@as(usize, 0), observed.context.snapshot().active_operations);
    const before = std.mem.asBytes(&observed.output).*;
    try expectError(error.Closing, Raw.matrixVectorMultiply(f64, i32, h, 99, .{ .matrix_type = 21, .mode = 99, .diag = 99 }, 0, null, 0, &observed.output));
    try expectError(error.Closing, Typed.structuredMatrixVectorMultiply(f64, i32, h, .normal, .{ .diagonal = .unit }, 0, &.{}, 0, &observed.output));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&observed.output));
    try Typed.destroy(h);
    try spy.balanced();
    try helpers.close(&observed.context);
}

test "Sparse structured admission overflow and foreign operations reject before output and descriptor access" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    Context.testing.configure(&context, 1, null);
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, &context, .{ .base = 0, .rows = 0, .columns = 0, .axis = .row, .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} }, &h);
    {
        defer release(h);
        var operation = try context.begin();
        const attempts = spy.attempts;
        var y = [_]f64{ 17, 19, 23 };
        const before = std.mem.asBytes(&y).*;
        try expectError(error.CounterOverflow, Typed.structuredMatrixVectorMultiply(f64, i32, h, .normal, .{ .diagonal = .unit }, 0, &.{}, 0, &y));
        try expectError(error.CounterOverflow, Raw.matrixVectorMultiply(f64, i32, h, 99, .{ .matrix_type = 21, .mode = 99, .diag = 99 }, 0, null, 0, &y));
        const exported = try Typed.matrixAdmitted(f64, i32, &operation, h.?);
        var foreign = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
        var other = try foreign.begin();
        try expectError(error.ContextMismatch, Typed.structuredProductAdmitted(f64, i32, &other, h.?, exported, .normal, .{ .diagonal = .unit }, 1, &.{}, 0, &y));
        other.end();
        operation.end();
        try helpers.close(&foreign);
        try equal(attempts, spy.attempts);
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&y));
    }
    try spy.balanced();
    try helpers.close(&context);
}

test "Sparse structured general entry delegates without changing status or complete numerical bytes" {
    @setEvalBranchQuota(15000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        var matrix = MatrixFixture(T, i64).init(.column, 1, .split);
        var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
        var h: ?Handles.Handle = null;
        try Raw.createBorrowed(T, i64, &context, matrix.raw(), &h);
        {
            defer release(h);
            const nan = number(T, std.math.nan(f64), std.math.nan(f64));
            const scales = [_]T{ number(T, 0, -0.0), number(T, 1, 0), number(T, -2, 1), number(T, std.math.inf(f64), 0), nan };
            const x = [_]T{ number(T, 2, 1), nan, number(T, -0.0, -0.0) };
            for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
                const nx: usize = if (op == .normal) 3 else 2;
                const ny: usize = if (op == .normal) 2 else 3;
                for (scales) |alpha| for (scales) |beta| {
                    var old: [5]T = @splat(nan);
                    var result = old;
                    try Typed.matrixVectorMultiply(T, i64, h, op, alpha, x[0..nx], beta, old[1..][0..ny]);
                    try Typed.structuredMatrixVectorMultiply(T, i64, h, op, .general, alpha, x[0..nx], beta, result[1..][0..ny]);
                    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&old), std.mem.asBytes(&result));
                };
            }
            var output: [3]T = @splat(number(T, 17, 19));
            const before = std.mem.asBytes(&output).*;
            const old_error = Typed.matrixVectorMultiply(T, i64, h, .normal, number(T, 0, 0), &.{}, number(T, 0, 0), &output);
            const new_error = Typed.structuredMatrixVectorMultiply(T, i64, h, .normal, .general, number(T, 0, 0), &.{}, number(T, 0, 0), &output);
            try expectError(error.DimensionMismatch, old_error);
            try expectError(error.DimensionMismatch, new_error);
            try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&output));
        }
        try helpers.close(&context);
    }
}
