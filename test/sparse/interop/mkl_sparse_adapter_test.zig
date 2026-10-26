// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const binding = @import("sparse-interface-owner");
const Core = @import("zynum-core");
const Adapter = binding.Adapter;
const Handles = binding.Handles;
const execution = @import("zynum-execution");
const Context = execution.context.Context;
const fixture = execution.test_support;
const helpers = @import("handle_test.zig");
const equal = std.testing.expectEqual;
const expect = std.testing.expect;

fn empty() Adapter.Input(f64, i32) {
    return .{ .base = 0, .rows = 0, .columns = 0, .axis = .row, .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} };
}
fn number(comptime T: type, real: anytype) T {
    if (T == f32 or T == f64) return real;
    return .{ .re = real, .im = -real };
}

test "typed Sparse all scalar widths formats and bases preserve mutable pointers gaps duplicates and numerical bits" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            const R = if (T == f32 or T == Core.ComplexF32) f32 else f64;
            for ([_]i32{ 0, 1 }) |base| {
                const b: I = @intCast(base);
                var starts = [_]I{ b + 1, b + 5, b + 7 };
                var ends = [_]I{ b + 4, b + 5, b + 8 };
                var indices = [_]I{ -99, b + 1, b, b + 1, -99, -99, -99, b, -99 };
                var values: [9]T = @splat(number(T, std.math.nan(R)));
                values[1] = number(T, @as(R, -0.0));
                values[2] = number(T, std.math.inf(R));
                values[3] = number(T, @as(R, 2));
                values[7] = number(T, @as(R, -3));
                const original_values: [@sizeOf(@TypeOf(values))]u8 = std.mem.asBytes(&values).*;
                const original_indices = indices;
                for ([_]Adapter.Axis{ .row, .column }) |axis| {
                    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                    var h: ?Handles.Handle = null;
                    try Adapter.createBorrowed(T, I, &context, .{ .base = base, .rows = if (axis == .row) 3 else 2, .columns = if (axis == .row) 2 else 3, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, &h);
                    var out: Adapter.Export(T, I) = undefined;
                    try Adapter.exportBorrowed(T, I, h, axis, &out);
                    try equal(base, out.base);
                    try equal(@as(I, if (axis == .row) 3 else 2), out.rows);
                    try equal(@as(I, if (axis == .row) 2 else 3), out.columns);
                    try expect(out.starts.ptr == starts[0..].ptr and out.ends.ptr == ends[0..].ptr);
                    try expect(out.ordinals.ptr == indices[0..].ptr and out.values.ptr == values[0..].ptr);
                    try equal(@as(usize, 8), out.values.len);
                    try equal(@as(usize, 8), out.ordinals.len);
                    // Independent physical coordinate expectations; no conversion
                    // or sorting of the duplicate (major=0, minor=1) references.
                    try std.testing.expectEqualSlices(I, &.{ b + 1, b, b + 1 }, out.ordinals[1..4]);
                    try equal(b, out.ordinals[7]);
                    try equal(@as(usize, 1), context.snapshot().resource_pins);
                    try Adapter.destroy(h);
                    try std.testing.expectEqualSlices(u8, &original_values, std.mem.asBytes(&values));
                    try std.testing.expectEqualSlices(I, &original_indices, &indices);
                    try helpers.close(&context);
                }
            }
        }
    }
}

test "typed Sparse shifted ends share caller offsets without copying or rejecting readonly input overlap" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var offsets = [_]i32{ 1, 3, 3, 4 };
    var indices = [_]i32{ 2, 1, 2 };
    var values = [_]f32{ -0.0, std.math.nan(f32), 7 };
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f32, i32, &context, .{ .base = 1, .rows = 3, .columns = 2, .axis = .row, .starts = offsets[0..3], .ends = offsets[1..4], .ordinals = &indices, .values = &values }, &h);
    var out: Adapter.Export(f32, i32) = undefined;
    try Adapter.exportBorrowed(f32, i32, h, .row, &out);
    try expect(out.starts.ptr == offsets[0..3].ptr and out.ends.ptr == offsets[1..4].ptr);
    try equal(@as(i32, 3), out.starts[1]);
    try equal(@as(i32, 3), out.ends[1]);
    try Adapter.destroy(h);
    try helpers.close(&context);
}

test "typed Sparse empty dimensions and empty rows keep explicit backing and still own a control pin" {
    fixture.setThreads(1);
    for ([_]struct { rows: i64, columns: i64, axis: Adapter.Axis }{
        .{ .rows = 0, .columns = 7, .axis = .row },
        .{ .rows = 7, .columns = 0, .axis = .column },
        .{ .rows = 0, .columns = 0, .axis = .row },
        .{ .rows = 2, .columns = 0, .axis = .row },
    }) |shape| {
        var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
        var dummy = [_]i64{ 1, 1 };
        var indices = [_]i64{-99};
        var values = [_]f64{std.math.nan(f64)};
        var h: ?Handles.Handle = null;
        try Adapter.createBorrowed(f64, i64, &context, .{ .base = 1, .rows = shape.rows, .columns = shape.columns, .axis = shape.axis, .starts = &dummy, .ends = &dummy, .ordinals = &indices, .values = &values }, &h);
        var out: Adapter.Export(f64, i64) = undefined;
        try Adapter.exportBorrowed(f64, i64, h, shape.axis, &out);
        try equal(@as(usize, 0), out.values.len);
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try expect(std.math.isNan(values[0]));
        try Adapter.destroy(h);
        try helpers.close(&context);
    }
}

test "typed Sparse create precedence keeps the prior output handle and separates domain and capacity faults" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var prior: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, empty(), &prior);
    var output = prior;
    var input = empty();
    input.base = 9;
    input.rows = -1;
    try std.testing.expectError(error.NullOutput, Adapter.createBorrowed(f64, i32, &context, input, null));
    try std.testing.expectError(error.InvalidBase, Adapter.createBorrowed(f64, i32, &context, input, &output));
    input.base = 0;
    try std.testing.expectError(error.InvalidDimension, Adapter.createBorrowed(f64, i32, &context, input, &output));
    input.rows = 2;
    input.columns = 2;
    try std.testing.expectError(error.BufferTooSmall, Adapter.createBorrowed(f64, i32, &context, input, &output));
    var starts = [_]i32{ -1, 1 };
    var ends = [_]i32{ 1, 2 };
    var indices = [_]i32{ 0, 1 };
    var values = [_]f64{ 3, 4 };
    input.starts = &starts;
    input.ends = &ends;
    input.ordinals = &indices;
    input.values = &values;
    try std.testing.expectError(error.InvalidOffsets, Adapter.createBorrowed(f64, i32, &context, input, &output));
    starts[0] = 2;
    try std.testing.expectError(error.InvalidOffsets, Adapter.createBorrowed(f64, i32, &context, input, &output));
    starts[0] = 0;
    starts[1] = 0;
    try std.testing.expectError(error.UnsupportedDomain, Adapter.createBorrowed(f64, i32, &context, input, &output));
    starts[1] = 1;
    ends[1] = 3;
    try std.testing.expectError(error.BufferTooSmall, Adapter.createBorrowed(f64, i32, &context, input, &output));
    ends[1] = 2;
    indices[0] = 2;
    try std.testing.expectError(error.InvalidOrdinal, Adapter.createBorrowed(f64, i32, &context, input, &output));
    try expect(output == prior);
    try equal(@as(usize, 2), spy.attempts);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try Adapter.destroy(prior);
    try spy.balanced();
    try helpers.close(&context);
}

test "typed Sparse export rejects type format and changed metadata before touching complete output bytes" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts = [_]i32{ 0, 1 };
    var ends = [_]i32{ 1, 2 };
    var indices = [_]i32{ 1, 0 };
    var values = [_]f64{ 2, 3 };
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, &h);
    const Out = Adapter.Export(f64, i32);
    var bytes: [@sizeOf(Out)]u8 align(@alignOf(Out)) = @splat(0xa5);
    const out: *Out = @ptrCast(&bytes);
    const before = bytes;
    try std.testing.expectError(error.NullHandle, Adapter.exportBorrowed(f64, i32, null, .row, null));
    try std.testing.expectError(error.NullOutput, Adapter.exportBorrowed(f64, i32, h, .column, null));
    // A null output precedes type inspection. Real output buffers below let each
    // scalar/width mismatch run without constructing any reinterpreted arrays.
    var float_out: Adapter.Export(f32, i32) = undefined;
    var wide_out: Adapter.Export(f64, i64) = undefined;
    @memset(std.mem.asBytes(&float_out), 0x5a);
    @memset(std.mem.asBytes(&wide_out), 0x3c);
    const float_before = std.mem.asBytes(&float_out).*;
    const wide_before = std.mem.asBytes(&wide_out).*;
    try std.testing.expectError(error.WrongScalar, Adapter.exportBorrowed(f32, i32, h, .column, &float_out));
    try std.testing.expectError(error.WrongIndexWidth, Adapter.exportBorrowed(f64, i64, h, .column, &wide_out));
    try std.testing.expectEqualSlices(u8, &float_before, std.mem.asBytes(&float_out));
    try std.testing.expectEqualSlices(u8, &wide_before, std.mem.asBytes(&wide_out));
    indices[0] = 9;
    try std.testing.expectError(error.WrongFormat, Adapter.exportBorrowed(f64, i32, h, .column, out));
    try std.testing.expectError(error.InvalidOrdinal, Adapter.exportBorrowed(f64, i32, h, .row, out));
    indices[0] = 1;
    starts[1] = 0; // Internally valid overlap, outside this interface slice.
    try std.testing.expectError(error.UnsupportedDomain, Adapter.exportBorrowed(f64, i32, h, .row, out));
    starts[1] = 1;
    ends[1] = 3; // Original captured prefix has capacity 2.
    try std.testing.expectError(error.InvalidOffsets, Adapter.exportBorrowed(f64, i32, h, .row, out));
    try std.testing.expectEqualSlices(u8, &before, &bytes);
    ends[1] = 2;
    values[0] = -7;
    try Adapter.exportBorrowed(f64, i32, h, .row, out);
    try equal(@as(f64, -7), out.values[0]);
    try expect(out.values.ptr == values[0..].ptr);
    comptime {
        if (@FieldType(Handles.Control(f64, i32).Export, "values") != []const f64 or @FieldType(Out, "values") != []f64)
            @compileError("wire provenance must not change the common readonly Export type");
    }
    try Adapter.destroy(h);
    try helpers.close(&context);
}

test "typed Sparse active output aliases reject before mutation including shell and control metadata" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    const Out = Adapter.Export(f64, i32);
    var values: [64]f64 align(@max(@alignOf(Out), @alignOf(?Handles.Handle))) = @splat(0);
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var indices = [_]i32{0};
    const input: Adapter.Input(f64, i32) = .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values };
    const handle_slot: *?Handles.Handle = @ptrCast(&values);
    try std.testing.expectError(error.BufferOverlap, Adapter.createBorrowed(f64, i32, &context, input, handle_slot));
    for (values) |v| try equal(@as(f64, 0), v);
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, input, &h);
    const out: *Out = @ptrCast(&values);
    try std.testing.expectError(error.BufferOverlap, Adapter.exportBorrowed(f64, i32, h, .row, out));
    for (values) |v| try equal(@as(f64, 0), v);
    const Storage = Core.internal.numerics.Storage;
    try std.testing.expectError(error.BufferOverlap, Handles.rejectMetadataOverlap(h.?, try Storage.byteRange(@intFromPtr(h.?), 1)));
    const control = try Handles.testing.retainControl(f64, i32, h.?);
    try std.testing.expectError(error.BufferOverlap, Handles.rejectMetadataOverlap(h.?, try Storage.byteRange(@intFromPtr(control), 1)));
    control.release();
    try Adapter.destroy(h);
    try helpers.close(&context);
}

test "typed Sparse capacity outside captured prefixes may be reused as a separate writable output" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var values: [4]f64 align(@max(@alignOf(f64), @alignOf(?Handles.Handle))) = .{ 3, 17, 0, 19 };
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var indices = [_]i32{0};
    const slot: *?Handles.Handle = @ptrCast(&values[2]);
    try Adapter.createBorrowed(f64, i32, &context, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, slot);
    const h = slot.*;
    var out: Adapter.Export(f64, i32) = undefined;
    try Adapter.exportBorrowed(f64, i32, h, .row, &out);
    try equal(@as(usize, 1), out.values.len);
    try equal(@as(f64, 3), values[0]);
    try equal(@as(f64, 17), values[1]);
    try equal(@as(f64, 19), values[3]);
    try Adapter.destroy(h);
    try helpers.close(&context);
}

test "typed Sparse empty interval endpoints still require the explicit interface backing prefix" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts = [_]i64{2};
    var ends = [_]i64{2};
    var indices = [_]i64{ -99, -99 };
    var values = [_]f64{ std.math.nan(f64), std.math.inf(f64) };
    var input: Adapter.Input(f64, i64) = .{ .base = 0, .rows = 1, .columns = 0, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = values[0..0] };
    var h: ?Handles.Handle = null;
    try std.testing.expectError(error.BufferTooSmall, Adapter.createBorrowed(f64, i64, &context, input, &h));
    input.values = &values;
    try Adapter.createBorrowed(f64, i64, &context, input, &h);
    var out: Adapter.Export(f64, i64) = undefined;
    try Adapter.exportBorrowed(f64, i64, h, .row, &out);
    try equal(@as(usize, 2), out.values.len);
    try expect(std.math.isNan(values[0]) and std.math.isInf(values[1]));
    try Adapter.destroy(h);
    try helpers.close(&context);
}

test "typed Sparse signed i64 dimensions narrow only after admission without artificial giant slices" {
    if (comptime @bitSizeOf(usize) >= 64) return;
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try std.testing.expectError(error.SizeOverflow, Adapter.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = @as(i64, std.math.maxInt(u32)) + 1, .columns = 0, .axis = .row, .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} }, &h));
    try expect(h == null);
    try equal(@as(usize, 0), context.snapshot().resource_pins);
    try helpers.close(&context);
}
