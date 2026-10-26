// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Test-strengthening overlay on the frozen source-only candidate.
// This revision has not yet been compiled or executed.
const std = @import("std");
const binding = @import("sparse-interface-owner");
const Core = @import("zynum-core");
const execution = @import("zynum-execution");
const resources = @import("zynum-numerics-resources");
const fixture = execution.test_support;
const Context = execution.context.Context;
const Handles = binding.Handles;
const Typed = binding.Adapter;
const Raw = binding.Raw;
const helpers = @import("handle_test.zig");
const equal = std.testing.expectEqual;
const expect = std.testing.expect;
const expectError = std.testing.expectError;

fn empty() Raw.Input(f64, i32) {
    return .{ .base = 0, .rows = 0, .columns = 0, .axis = .row, .starts = null, .ends = null, .ordinals = null, .values = null };
}
fn number(comptime T: type, real: anytype) T {
    if (T == f32 or T == f64) return real;
    return .{ .re = real, .im = -real };
}
fn OutputState(comptime T: type, comptime I: type) type {
    return struct {
        base: u32 = 77,
        rows: I = -77,
        columns: I = -77,
        starts: Raw.Pointer(I) = null,
        ends: Raw.Pointer(I) = null,
        ordinals: Raw.Pointer(I) = null,
        values: Raw.Pointer(T) = null,
        canary_ordinal: I = 0,
        canary_value: T = undefined,
        // Stable receiver storage gives each pointer a real live mutable target.
        // Initialize padding first, then nonzero legal fields. Do not move this
        // state after initializing it; no pointer is synthesized from integers.
        fn sentinel(self: *@This()) void {
            @memset(std.mem.asBytes(self), 0xa5);
            self.base = 0x87654321;
            self.rows = -37;
            self.columns = -41;
            self.canary_ordinal = 7;
            self.canary_value = number(T, 9);
            self.starts = @ptrCast(&self.canary_ordinal);
            self.ends = @ptrCast(&self.canary_ordinal);
            self.ordinals = @ptrCast(&self.canary_ordinal);
            self.values = @ptrCast(&self.canary_value);
        }
        fn slots(self: *@This()) Raw.Output(T, I) {
            return .{ .base = &self.base, .rows = &self.rows, .columns = &self.columns, .starts = &self.starts, .ends = &self.ends, .ordinals = &self.ordinals, .values = &self.values };
        }
    };
}
// Every cell is real writable storage, sufficiently large/aligned for each
// native output slot. The bytes are compared without reading invalid pointers.
const Cells = struct { bytes: [7][16]u8 align(16) = @splat(@splat(0xa5)) };
fn cellsOutput(cells: *Cells) Raw.Output(f64, i32) {
    return .{ .base = @ptrCast(&cells.bytes[0]), .rows = @ptrCast(&cells.bytes[1]), .columns = @ptrCast(&cells.bytes[2]), .starts = @ptrCast(&cells.bytes[3]), .ends = @ptrCast(&cells.bytes[4]), .ordinals = @ptrCast(&cells.bytes[5]), .values = @ptrCast(&cells.bytes[6]) };
}
fn replaceSlot(output: *Raw.Output(f64, i32), index: usize, pointer: *align(1) anyopaque) void {
    switch (index) {
        0 => output.base = @ptrCast(pointer),
        1 => output.rows = @ptrCast(pointer),
        2 => output.columns = @ptrCast(pointer),
        3 => output.starts = @ptrCast(pointer),
        4 => output.ends = @ptrCast(pointer),
        5 => output.ordinals = @ptrCast(pointer),
        6 => output.values = @ptrCast(pointer),
        else => unreachable,
    }
}
fn releaseHandle(handle: Handles.Handle) void {
    Raw.destroy(handle) catch @panic("raw handle destruction failed");
}

test "raw Sparse reuses the typed family actual Core and Context without another owner" {
    comptime {
        if (Raw.Handle != Handles.Handle or Raw.Context != Context or Raw.Operation != execution.context.Operation or
            Typed.Access.OwnerIdentity != Core or binding.Provider.OwnerIdentity != Core or
            binding.Provider.Execution != execution or binding.Provider.Resources != resources)
            @compileError("raw lifecycle introduced an owner or type identity");
        if (Raw.Pointer(i64) != ?[*]align(1) i64 or Handles.EmptySource(i64) != @FieldType(Handles.Provenance(f64, i64), "starts"))
            @compileError("empty provenance must preserve optional alignment-one pointer values");
    }
    try expect(Core.internal.numerics.Execution.getContext() == execution.getContext());
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, empty(), &h);
    var out: OutputState(f64, i32) = .{};
    out.sentinel();
    try Raw.exportBorrowed(f64, i32, h, .row, out.slots());
    try equal(@as(usize, 2), spy.attempts);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try equal(@as(usize, 0), context.snapshot().active_operations);
    try expect(out.starts == null and out.ends == null and out.ordinals == null and out.values == null);
    try Raw.destroy(h);
    try spy.balanced();
    try helpers.close(&context);
}

test "raw Sparse all eight scalar index pairs preserve split gaps duplicate order and numerical bits in CSR CSC" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            const R = if (T == f32 or T == Core.ComplexF32) f32 else f64;
            for ([_]u32{ 0, 1 }) |base| {
                const b: I = @intCast(base);
                var starts = [_]I{ b + 1, b + 5, b + 7 };
                var ends = [_]I{ b + 4, b + 5, b + 8 };
                var ordinals = [_]I{ -99, b + 1, b, b + 1, -99, -99, -99, b, -99 };
                var values: [9]T = @splat(number(T, std.math.nan(R)));
                values[1] = number(T, @as(R, -0.0));
                values[2] = number(T, std.math.inf(R));
                values[3] = number(T, @as(R, 2));
                values[7] = number(T, @as(R, -3));
                const value_bytes = std.mem.asBytes(&values).*;
                const ordinal_copy = ordinals;
                for ([_]Raw.Axis{ .row, .column }) |axis| {
                    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                    const input: Raw.Input(T, I) = .{ .base = base, .rows = if (axis == .row) 3 else 2, .columns = if (axis == .row) 2 else 3, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values };
                    var h: ?Handles.Handle = null;
                    try Raw.createBorrowed(T, I, &context, input, &h);
                    var out: OutputState(T, I) = .{};
                    out.sentinel();
                    try Raw.exportBorrowed(T, I, h, axis, out.slots());
                    try equal(input.base, out.base);
                    try equal(input.rows, out.rows);
                    try equal(input.columns, out.columns);
                    try expect(out.starts == input.starts and out.ends == input.ends and out.ordinals == input.ordinals and out.values == input.values);
                    var typed: Typed.Export(T, I) = undefined;
                    try Typed.exportBorrowed(T, I, h, axis, &typed);
                    try equal(@as(usize, 8), typed.values.len);
                    try std.testing.expectEqualSlices(I, &.{ b + 1, b, b + 1 }, typed.ordinals[1..4]);
                    try equal(b, typed.ordinals[7]);
                    try Raw.destroy(h);
                    try std.testing.expectEqualSlices(u8, &value_bytes, std.mem.asBytes(&values));
                    try std.testing.expectEqualSlices(I, &ordinal_copy, &ordinals);
                    try helpers.close(&context);
                }
            }
        }
    }
}

test "raw Sparse null ends use real shifted offsets while explicit shifted ends retain their pointer" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            for ([_]Raw.Axis{ .row, .column }) |axis| {
                for ([_]bool{ false, true }) |null_end| {
                    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                    var offsets = [_]I{ 1, 3, 3, 4 };
                    var ordinals = [_]I{ 2, 1, 2 };
                    var values = [_]T{ number(T, 2), number(T, 3), number(T, 4) };
                    const input: Raw.Input(T, I) = .{ .base = 1, .rows = if (axis == .row) 3 else 2, .columns = if (axis == .row) 2 else 3, .axis = axis, .starts = &offsets, .ends = if (null_end) null else offsets[1..].ptr, .ordinals = &ordinals, .values = &values };
                    var h: ?Handles.Handle = null;
                    try Raw.createBorrowed(T, I, &context, input, &h);
                    var out: OutputState(T, I) = .{};
                    out.sentinel();
                    try Raw.exportBorrowed(T, I, h, axis, out.slots());
                    try expect(out.starts == input.starts and out.ends == input.ends);
                    var typed: Typed.Export(T, I) = undefined;
                    try Typed.exportBorrowed(T, I, h, axis, &typed);
                    try expect(typed.starts.ptr == offsets[0..3].ptr and typed.ends.ptr == offsets[1..4].ptr);
                    const before = std.mem.asBytes(&out).*;
                    offsets[3] = 5;
                    try expectError(error.InvalidOffsets, Raw.exportBorrowed(T, I, h, axis, out.slots()));
                    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
                    // Destroy never revalidates the now-invalid borrowed offsets.
                    try Raw.destroy(h);
                    try equal(@as(I, 5), offsets[3]);
                    try helpers.close(&context);
                }
            }
        }
    }
}

test "raw Sparse every zero prefix preserves NULL or unaligned original pointer without access" {
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            var bytes: [32]u8 align(16) = @splat(0x6d);
            const unaligned_i: Raw.Pointer(I) = @ptrCast(&bytes[1]);
            const unaligned_t: Raw.Pointer(T) = @ptrCast(&bytes[1]);
            for (0..16) |mask| {
                var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                const input: Raw.Input(T, I) = .{ .base = 1, .rows = 0, .columns = 7, .axis = .row, .starts = if (mask & 1 == 0) null else unaligned_i, .ends = if (mask & 2 == 0) null else unaligned_i, .ordinals = if (mask & 4 == 0) null else unaligned_i, .values = if (mask & 8 == 0) null else unaligned_t };
                var h: ?Handles.Handle = null;
                try Raw.createBorrowed(T, I, &context, input, &h);
                var out: OutputState(T, I) = .{};
                out.sentinel();
                try Raw.exportBorrowed(T, I, h, .row, out.slots());
                try expect(out.starts == input.starts and out.ends == input.ends and out.ordinals == input.ordinals and out.values == input.values);
                try equal(@as(I, 0), out.rows);
                try equal(@as(I, 7), out.columns);
                try Raw.destroy(h);
                for (bytes) |byte| try equal(@as(u8, 0x6d), byte);
                try helpers.close(&context);
            }
        }
    }
}

test "raw Sparse zero extent with nonempty metadata cannot promote empty pointer tokens after mutation" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var offsets = [_]i64{ 1, 1, 1 };
    var bytes: [16]u8 align(16) = @splat(0xb3);
    const ordinals: Raw.Pointer(i64) = @ptrCast(&bytes[1]);
    const values: Raw.Pointer(f64) = @ptrCast(&bytes[3]);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i64, &context, .{ .base = 1, .rows = 2, .columns = 0, .axis = .row, .starts = &offsets, .ends = null, .ordinals = ordinals, .values = values }, &h);
    var out: OutputState(f64, i64) = .{};
    out.sentinel();
    try Raw.exportBorrowed(f64, i64, h, .row, out.slots());
    try expect(out.ordinals == ordinals and out.values == values and out.ends == null);
    const before = std.mem.asBytes(&out).*;
    offsets[2] = 2;
    try expectError(error.InvalidOffsets, Raw.exportBorrowed(f64, i64, h, .row, out.slots()));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
    for (bytes) |byte| try equal(@as(u8, 0xb3), byte);
    try Raw.destroy(h);
    try helpers.close(&context);
}

test "raw Sparse create rejects required null alignment geometry domain and extent overflow before allocation" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var h: ?Handles.Handle = null;
    var input = empty();
    input.base = 99;
    input.rows = -1;
    try expectError(error.NullOutput, Raw.createBorrowed(f64, i32, &context, input, null));
    try expectError(error.InvalidBase, Raw.createBorrowed(f64, i32, &context, input, &h));
    input.base = 0;
    try expectError(error.InvalidDimension, Raw.createBorrowed(f64, i32, &context, input, &h));
    input.rows = 2;
    input.columns = 2;
    try expectError(error.NullInput, Raw.createBorrowed(f64, i32, &context, input, &h));
    var bytes: [64]u8 align(16) = @splat(0);
    input.starts = @ptrCast(&bytes[1]);
    try expectError(error.InvalidAlignment, Raw.createBorrowed(f64, i32, &context, input, &h));
    var starts = [_]i32{ 0, 0 };
    var ends = [_]i32{ 1, 2 };
    var ordinals = [_]i32{ 0, 1 };
    var values = [_]f64{ 3, 4 };
    input.starts = &starts;
    input.ends = &ends;
    try expectError(error.UnsupportedDomain, Raw.createBorrowed(f64, i32, &context, input, &h));
    starts[1] = 1;
    try expectError(error.NullInput, Raw.createBorrowed(f64, i32, &context, input, &h));
    input.ordinals = &ordinals;
    try expectError(error.NullInput, Raw.createBorrowed(f64, i32, &context, input, &h));
    input.values = @ptrCast(&bytes[1]);
    try expectError(error.InvalidAlignment, Raw.createBorrowed(f64, i32, &context, input, &h));
    input.values = &values;
    ordinals[0] = 2;
    try expectError(error.InvalidOrdinal, Raw.createBorrowed(f64, i32, &context, input, &h));
    // Count arithmetic fails before a giant slice is formed or any offset read.
    // On 32-bit targets the same case stops earlier at dimension-to-usize cast.
    var wide = [_]i64{0};
    try expectError(error.SizeOverflow, Raw.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = std.math.maxInt(i64), .columns = 0, .axis = .row, .starts = &wide, .ends = null, .ordinals = null, .values = null }, &h));
    try expect(h == null);
    try equal(@as(usize, 0), spy.attempts);
    try equal(@as(usize, 0), context.snapshot().resource_pins);
    try helpers.close(&context);
}

test "raw Sparse all twenty one output pair overlaps fail before any output commit" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, empty(), &h);
    var pairs: usize = 0;
    for (0..7) |first| {
        for (first + 1..7) |second| {
            var cells: Cells = .{};
            var output = cellsOutput(&cells);
            replaceSlot(&output, second, @ptrCast(&cells.bytes[first]));
            const before = cells.bytes;
            try expectError(error.BufferOverlap, Raw.exportBorrowed(f64, i32, h, .row, output));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&cells.bytes));
            pairs += 1;
        }
    }
    try equal(@as(usize, 21), pairs);
    // Natural alignment still permits different-start overlap on a 64-bit
    // target: an eight-byte pointer slot at +0 intersects a four-byte scalar
    // slot at +4. These are not duplicate same-start cases above. On targets
    // whose pointer slots are only four bytes this specific geometry is absent.
    var partial_pairs: usize = 0;
    if (@sizeOf(Raw.Pointer(f64)) > @sizeOf(u32)) {
        for (0..3) |scalar_slot| {
            for (3..7) |pointer_slot| {
                var cells: Cells = .{};
                var output = cellsOutput(&cells);
                replaceSlot(&output, scalar_slot, @ptrCast(&cells.bytes[pointer_slot][@sizeOf(u32)]));
                const before = cells.bytes;
                try expectError(error.BufferOverlap, Raw.exportBorrowed(f64, i32, h, .row, output));
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&cells.bytes));
                partial_pairs += 1;
            }
        }
        try equal(@as(usize, 12), partial_pairs);
    }
    try Raw.destroy(h);
    try helpers.close(&context);
}

test "raw Sparse all seven slots require real aligned nonnull storage independently of empty pointees" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, empty(), &h);
    for (0..7) |index| {
        var cells: Cells = .{};
        var output = cellsOutput(&cells);
        const before = cells.bytes;
        switch (index) {
            0 => output.base = null,
            1 => output.rows = null,
            2 => output.columns = null,
            3 => output.starts = null,
            4 => output.ends = null,
            5 => output.ordinals = null,
            6 => output.values = null,
            else => unreachable,
        }
        try expectError(error.NullOutput, Raw.exportBorrowed(f64, i32, h, .row, output));
        output = cellsOutput(&cells);
        replaceSlot(&output, index, @ptrCast(&cells.bytes[index][1]));
        try expectError(error.InvalidAlignment, Raw.exportBorrowed(f64, i32, h, .row, output));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&cells.bytes));
    }
    try Raw.destroy(h);
    try helpers.close(&context);
}

test "raw Sparse output spans reject captured inputs gaps shell and control without writes" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var starts = [_]i32{1};
    var ends = [_]i32{2};
    var ordinals = [_]i32{ -99, 0 };
    var values: [4]f64 align(16) = .{ 9, 3, 7, 8 };
    const input: Raw.Input(f64, i32) = .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values };
    const before_values = std.mem.asBytes(&values).*;
    const handle_slot: *align(1) ?Handles.Handle = @ptrCast(&values[0]);
    try expectError(error.BufferOverlap, Raw.createBorrowed(f64, i32, &context, input, handle_slot));
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, input, &h);
    const control = try Handles.testing.retainControl(f64, i32, h.?);
    const targets = [_]*align(1) anyopaque{ @ptrCast(&starts), @ptrCast(&ends), @ptrCast(&ordinals[0]), @ptrCast(&ordinals[1]), @ptrCast(&values[0]), @ptrCast(&values[1]), @ptrCast(h.?), @ptrCast(control) };
    for (targets) |target| {
        var cells: Cells = .{};
        var output = cellsOutput(&cells);
        output.base = @ptrCast(target);
        const before = cells.bytes;
        try expectError(error.BufferOverlap, Raw.exportBorrowed(f64, i32, h, .row, output));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&cells.bytes));
    }
    try equal(@as(i32, 1), starts[0]);
    try equal(@as(i32, 2), ends[0]);
    try std.testing.expectEqualSlices(i32, &.{ -99, 0 }, &ordinals);
    try std.testing.expectEqualSlices(u8, &before_values, std.mem.asBytes(&values));
    // Unused tail is outside the captured extent: it may itself be a slot.
    var tail_out: OutputState(f64, i32) = .{};
    tail_out.sentinel();
    var slots = tail_out.slots();
    slots.rows = @ptrCast(&values[2]);
    try Raw.exportBorrowed(f64, i32, h, .row, slots);
    const stored_rows: *const i32 = @ptrCast(&values[2]);
    try equal(@as(i32, 1), stored_rows.*);
    try std.testing.expectEqualSlices(u8, before_values[0 .. 2 * @sizeOf(f64)], std.mem.asBytes(&values)[0 .. 2 * @sizeOf(f64)]);
    control.release();
    try Raw.destroy(h);
    try helpers.close(&context);
}

test "raw Sparse type format and revalidation errors preserve all outputs and typed split provenance" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var offsets = [_]i32{ 0, 1, 2 };
    var ordinals = [_]i32{ 0, 1 };
    var values = [_]f64{ 2, 3 };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, &context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = offsets[0..2], .ends = offsets[1..3], .ordinals = &ordinals, .values = &values }, &h);
    var scalar_out: OutputState(f32, i32) = .{};
    scalar_out.sentinel();
    const scalar_before = std.mem.asBytes(&scalar_out).*;
    try expectError(error.WrongScalar, Raw.exportBorrowed(f32, i32, h, .column, scalar_out.slots()));
    try std.testing.expectEqualSlices(u8, &scalar_before, std.mem.asBytes(&scalar_out));
    var index_out: OutputState(f64, i64) = .{};
    index_out.sentinel();
    const index_before = std.mem.asBytes(&index_out).*;
    try expectError(error.WrongIndexWidth, Raw.exportBorrowed(f64, i64, h, .column, index_out.slots()));
    try std.testing.expectEqualSlices(u8, &index_before, std.mem.asBytes(&index_out));
    var out: OutputState(f64, i32) = .{};
    out.sentinel();
    const before = std.mem.asBytes(&out).*;
    try expectError(error.NullHandle, Raw.exportBorrowed(f64, i32, null, .row, out.slots()));
    ordinals[0] = -1;
    try expectError(error.WrongFormat, Raw.exportBorrowed(f64, i32, h, .column, out.slots()));
    try expectError(error.InvalidOrdinal, Raw.exportBorrowed(f64, i32, h, .row, out.slots()));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
    ordinals[0] = 0;
    try Raw.exportBorrowed(f64, i32, h, .row, out.slots());
    try expect(out.starts == @as(Raw.Pointer(i32), offsets[0..2].ptr));
    try expect(out.ends == @as(Raw.Pointer(i32), offsets[1..3].ptr));
    try Raw.destroy(h);
    try helpers.close(&context);
}

test "raw Sparse provenance rejects missing empty tokens encoding conflicts and fake null-end shifting" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var operation = try context.begin();
    const data: Handles.MutableData(f64, i32) = .{ .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} };
    var source: Handles.Provenance(f64, i32) = .{ .encoding = .c_null_end };
    try expectError(error.InvalidHandle, Handles.borrowWithProvenance(f64, i32, &operation, 0, 0, .row, .zero, data, source));
    source.starts = .{ .empty_original = null };
    source.ordinals = .{ .empty_original = null };
    source.values = .{ .empty_original = null };
    source.ends = .{ .empty_original = null };
    try expectError(error.InvalidHandle, Handles.borrowWithProvenance(f64, i32, &operation, 0, 0, .row, .zero, data, source));
    source.encoding = .c_split;
    source.ends = .not_captured;
    try expectError(error.InvalidHandle, Handles.borrowWithProvenance(f64, i32, &operation, 0, 0, .row, .zero, data, source));
    source.encoding = .typed_split;
    try expectError(error.InvalidHandle, Handles.borrowWithProvenance(f64, i32, &operation, 0, 0, .row, .zero, data, source));
    var offsets = [_]i32{ 0, 0, 0 };
    const unshifted: Handles.MutableData(f64, i32) = .{ .starts = offsets[0..1], .ends = offsets[2..3], .ordinals = &.{}, .values = &.{} };
    source = .{ .encoding = .c_null_end, .ordinals = .{ .empty_original = null }, .values = .{ .empty_original = null } };
    try expectError(error.InvalidHandle, Handles.borrowWithProvenance(f64, i32, &operation, 1, 0, .row, .zero, unshifted, source));
    operation.end();
    try equal(@as(usize, 0), spy.attempts);
    try equal(@as(usize, 0), context.snapshot().resource_pins);
    try helpers.close(&context);
}

test "raw Sparse control and shell OOM rollback before unpin and admission overflow remains distinct" {
    fixture.setThreads(1);
    for (0..2) |fail_at| {
        var spy: helpers.Spy = .{};
        var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
        spy.ctx = &context;
        var prior: ?Handles.Handle = null;
        try Raw.createBorrowed(f64, i32, &context, empty(), &prior);
        const prior_attempts = spy.attempts;
        spy.fail_at = prior_attempts + fail_at;
        var output = prior; // Nonnull live canary: a buggy failure clear is visible.
        try expectError(error.OutOfMemory, Raw.createBorrowed(f64, i32, &context, empty(), &output));
        try expect(output == prior);
        try equal(prior_attempts + fail_at + 1, spy.attempts);
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try equal(@as(usize, 0), context.snapshot().active_operations);
        // The preserved old handle remains usable and independently destroyable.
        var out: OutputState(f64, i32) = .{};
        out.sentinel();
        try Raw.exportBorrowed(f64, i32, output, .row, out.slots());
        try equal(@as(u32, 0), out.base);
        try equal(@as(i32, 0), out.rows);
        try expect(out.starts == null and out.ends == null and out.ordinals == null and out.values == null);
        try Raw.destroy(prior);
        try equal(@as(usize, 0), context.snapshot().resource_pins);
        try spy.balanced();
        try helpers.close(&context);
    }
    var spy: helpers.Spy = .{ .fail_at = 0 };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    Context.testing.configure(&context, 1, null);
    var operation = try context.begin();
    var pin = try operation.pinResource();
    var h: ?Handles.Handle = null;
    try expectError(error.CounterOverflow, Raw.createBorrowed(f64, i32, &context, empty(), null));
    operation.end();
    try expectError(error.CounterOverflow, Raw.createBorrowed(f64, i32, &context, empty(), &h));
    try equal(@as(usize, 0), spy.attempts);
    pin.releaseAfterResourceFree();
    try expectError(error.OutOfMemory, Raw.createBorrowed(f64, i32, &context, empty(), &h));
    try equal(@as(usize, 1), spy.attempts);
    try spy.balanced();
    try helpers.close(&context);
}

test "raw Sparse admitted creation survives close callbacks while destruction never readmits" {
    const Callback = struct {
        fn call(spy: *helpers.Spy, _: usize) void {
            spy.callback_count += 1;
            if (spy.ctx.snapshot().active_operations != 1) @panic("raw create did not admit exactly once");
            _ = spy.ctx.requestClose();
            if (spy.ctx.tryProgressClose() != .busy) @panic("close escaped admitted raw create");
        }
    };
    fixture.setThreads(1);
    var spy: helpers.Spy = .{ .on_alloc = Callback.call };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, empty(), &h);
    try equal(@as(usize, 2), spy.callback_count);
    var out: OutputState(f64, i32) = .{};
    out.sentinel();
    const before = std.mem.asBytes(&out).*;
    try expectError(error.Closing, Raw.exportBorrowed(f64, i32, h, .row, out.slots()));
    try expectError(error.Closing, Raw.createBorrowed(f64, i32, &context, empty(), null));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
    try equal(execution.context.CloseResult.waiting_resources, context.tryProgressClose());
    try Raw.destroy(h);
    try spy.balanced();
    try helpers.close(&context);
}

test "raw Sparse shell and control free callbacks keep the single pin until each free returns" {
    fixture.setThreads(1);
    for ([_]usize{ 1, 0 }) |pause_id| {
        var spy: helpers.Spy = .{ .pause_id = pause_id };
        var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
        spy.ctx = &context;
        var h: ?Handles.Handle = null;
        try Raw.createBorrowed(f64, i32, &context, empty(), &h);
        _ = context.requestClose();
        const thread = try std.Thread.spawn(.{}, releaseHandle, .{h.?});
        var joined = false;
        defer {
            spy.resume_free.store(true, .release);
            if (!joined) thread.join();
        }
        fixture.wait(&spy.entered);
        try equal(execution.context.CloseResult.waiting_resources, context.tryProgressClose());
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try equal(@as(usize, 0), context.snapshot().active_operations);
        try expect(!spy.returned.load(.acquire));
        spy.resume_free.store(true, .release);
        thread.join();
        joined = true;
        try expect(spy.returned.load(.acquire));
        try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, spy.free_order[0..spy.freed]);
        try spy.balanced();
        try helpers.close(&context);
    }
}

test "raw Sparse free callback may operate on a different handle and close without a second owner" {
    const Callback = struct {
        fn call(spy: *helpers.Spy, id: usize) void {
            if (id != 3) return;
            spy.callback_count += 1;
            var out: OutputState(f64, i32) = .{};
            out.sentinel();
            Raw.exportBorrowed(f64, i32, spy.other, .row, out.slots()) catch @panic("other raw export failed");
            var temporary: ?Handles.Handle = null;
            Raw.createBorrowed(f64, i32, spy.ctx, empty(), &temporary) catch @panic("other raw create failed");
            Raw.destroy(temporary) catch @panic("temporary raw destroy failed");
            _ = spy.ctx.requestClose();
            if (Raw.exportBorrowed(f64, i32, spy.other, .row, out.slots())) |_| @panic("closing raw export admitted") else |err| {
                if (err != error.Closing) @panic("wrong closing error");
            }
            Raw.destroy(spy.other) catch @panic("other raw destroy while closing failed");
            spy.other = null;
            if (spy.ctx.tryProgressClose() != .waiting_resources) @panic("outer raw control pin lost during callback");
        }
    };
    fixture.setThreads(1);
    var spy: helpers.Spy = .{ .on_free = Callback.call };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var first: ?Handles.Handle = null;
    var second: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &context, empty(), &first);
    try Raw.createBorrowed(f64, i32, &context, empty(), &second);
    spy.other = first;
    try Raw.destroy(second);
    try equal(@as(usize, 1), spy.callback_count);
    try equal(@as(usize, 6), spy.accepted);
    try spy.balanced();
    try helpers.close(&context);
}

test "raw Sparse export admission covers close through all seven committed slots and finishing" {
    const Observation = struct {
        context: Context,
        export_phase: bool = false,
        admitted: usize = 0,
        finishing: usize = 0,
        result: *OutputState(f64, i32),
        expected: Raw.Input(f64, i32),

        fn observe(context: *Context, event: execution.context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("context", context);
            if (!self.export_phase) return; // Already installed during creation.
            const snapshot = context.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins != 1)
                @panic("raw export did not retain its one Operation and control pin");
            if (event == .admitted) {
                self.admitted += 1;
                if (self.finishing != 0) @panic("unexpected callback order");
                _ = context.requestClose();
                if (context.tryProgressClose() != .busy) @panic("export admitted callback closed early");
            } else {
                self.finishing += 1;
                if (self.admitted != 1) @panic("export admitted more than once or skipped admission");
                const output = self.result;
                if (output.base != self.expected.base or output.rows != self.expected.rows or output.columns != self.expected.columns or
                    output.starts != self.expected.starts or output.ends != self.expected.ends or
                    output.ordinals != self.expected.ordinals or output.values != self.expected.values)
                    @panic("raw export ended before all seven output slots were committed");
                if (context.tryProgressClose() != .busy) @panic("export finishing callback lost active Operation");
            }
        }
    };
    fixture.setThreads(1);
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ordinals = [_]i32{0};
    var values = [_]f64{3};
    var out: OutputState(f64, i32) = .{};
    out.sentinel();
    const input: Raw.Input(f64, i32) = .{ .base = 0, .rows = 1, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values };
    var observed: Observation = .{ .context = Context.init(std.testing.allocator, fixture.policy, .internal_closable), .result = &out, .expected = input };
    // Quiescent configuration precedes creation; no test-only production hook.
    Context.testing.configure(&observed.context, std.math.maxInt(usize), Observation.observe);
    var h: ?Handles.Handle = null;
    try Raw.createBorrowed(f64, i32, &observed.context, input, &h);
    try equal(@as(usize, 0), observed.admitted);
    try equal(@as(usize, 0), observed.finishing);
    observed.export_phase = true;
    try Raw.exportBorrowed(f64, i32, h, .row, out.slots());
    try equal(@as(usize, 1), observed.admitted);
    try equal(@as(usize, 1), observed.finishing);
    try equal(@as(usize, 0), observed.context.snapshot().active_operations);
    try equal(execution.context.CloseResult.waiting_resources, observed.context.tryProgressClose());
    const before = std.mem.asBytes(&out).*;
    try expectError(error.Closing, Raw.exportBorrowed(f64, i32, h, .row, out.slots()));
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
    try equal(@as(usize, 1), observed.admitted);
    try equal(@as(usize, 1), observed.finishing);
    try Raw.destroy(h);
    try helpers.close(&observed.context);
}
