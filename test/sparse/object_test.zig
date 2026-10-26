// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const providers = Core.internal.numerics;
const object = providers.SparseObject;
const pattern = providers.SparsePattern;
const scalar = providers.SparseScalar;
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

// The injected allocator checks the exact original pointer, byte size and
// alignment on every free. This is real allocation failure, not an object hook.
const Counted = struct {
    const Slot = struct { pointer: [*]u8, len: usize, alignment: std.mem.Alignment, id: usize };
    live: [16]?Slot = @splat(null),
    attempts: usize = 0,
    allocations: usize = 0,
    frees: usize = 0,
    allocated_bytes: usize = 0,
    freed_bytes: usize = 0,
    fail_at: ?usize = null,
    free_order: [16]usize = undefined,

    fn allocator(self: *Counted) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .free = free, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap } };
    }
    fn alloc(context: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Counted = @ptrCast(@alignCast(context));
        const attempt = self.attempts;
        self.attempts += 1;
        if (self.fail_at == attempt) return null;
        const p = std.testing.allocator.rawAlloc(len, alignment, ret) orelse return null;
        for (&self.live) |*slot| if (slot.* == null) {
            slot.* = .{ .pointer = p, .len = len, .alignment = alignment, .id = self.allocations };
            self.allocations += 1;
            self.allocated_bytes += len;
            return p;
        };
        @panic("counted allocator fixture capacity exceeded");
    }
    fn free(context: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *Counted = @ptrCast(@alignCast(context));
        for (&self.live) |*slot| if (slot.*) |record| {
            if (record.pointer == bytes.ptr) {
                if (record.len != bytes.len or record.alignment != alignment) @panic("allocation provenance mismatch");
                self.free_order[self.frees] = record.id;
                self.frees += 1;
                self.freed_bytes += bytes.len;
                slot.* = null;
                std.testing.allocator.rawFree(bytes, alignment, ret);
                return;
            }
        };
        @panic("wrong allocator, double free or borrowed-buffer free");
    }
    fn expectReleased(self: *const Counted) !void {
        try equal(self.allocations, self.frees);
        try equal(self.allocated_bytes, self.freed_bytes);
        for (self.live) |slot| try expect(slot == null);
    }
};

fn bits(comptime T: type) [4]T {
    if (T == f32) return .{ @bitCast(@as(u32, 0x80000000)), @bitCast(@as(u32, 0x7fc13579)), @bitCast(@as(u32, 0x7f800000)), @bitCast(@as(u32, 1)) };
    if (T == f64) return .{ @bitCast(@as(u64, 0x8000000000000000)), @bitCast(@as(u64, 0x7ff813579abcdef0)), @bitCast(@as(u64, 0x7ff0000000000000)), @bitCast(@as(u64, 1)) };
    const R = if (T == scalar.ComplexF32) f32 else f64;
    const r = bits(R);
    return .{ .{ .re = r[0], .im = r[1] }, .{ .re = r[1], .im = r[2] }, .{ .re = r[2], .im = r[3] }, .{ .re = r[3], .im = r[0] } };
}
fn expectBits(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

test "object contiguous snapshots preserve all scalar bits widths bases and CSR CSC orientation" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |O| {
            inline for (.{ i32, u32, i64, u64 }) |I| {
                for ([_]pattern.IndexBase{ .zero, .one }) |base| {
                    const b: u8 = @intFromBool(base == .one);
                    const offsets = [_]O{ b, 3 + @as(O, b), 4 + @as(O, b) };
                    const indices = [_]I{ 1 + @as(I, b), b, 1 + @as(I, b), b };
                    const values = bits(T);
                    for ([_]pattern.MajorAxis{ .row, .column }) |axis| {
                        const source: pattern.Pattern(O, I) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &indices, .row_count = 2, .column_count = 2, .major_axis = axis, .index_base = base };
                        const owned = try object.MatrixControl(T, O, I).snapshot(std.testing.allocator, source, &values);
                        defer owned.release();
                        const exported = try owned.exportView(axis);
                        try equal(object.Ownership.owned_snapshot, exported.ownership);
                        try equal(@as(?u64, 1), exported.pattern_revision);
                        try equal(@as(?u64, 1), exported.values_revision);
                        try expect(!exported.capabilities.update_values);
                        try equal(base, exported.structure.index_base);
                        try equal(axis, exported.structure.major_axis);
                        try expectBits(T, &values, exported.values);
                        try std.testing.expectEqualSlices(O, &offsets, exported.structure.ranges.contiguous);
                        try std.testing.expectEqualSlices(I, &indices, exported.structure.ordinals);
                        try expect(exported.values.ptr != &values);
                    }
                }
            }
        }
    }
}

test "object borrow exports pointer identity without allocation and revalidates every data call" {
    var counter: Counted = .{};
    var offsets = [_]i64{ 0, 1, 2 };
    var indices = [_]u32{ 0, 1 };
    var values = [_]f64{ 2, 3 };
    const source: pattern.Pattern(i64, u32) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &indices, .row_count = 2, .column_count = 2 };
    const borrowed = try object.MatrixControl(f64, i64, u32).borrow(counter.allocator(), source, &values);
    try equal(@as(usize, 1), counter.allocations);
    const exported = try borrowed.exportView(.row);
    try equal(values[0..].ptr, exported.values.ptr);
    try equal(offsets[0..].ptr, exported.structure.ranges.contiguous.ptr);
    try equal(indices[0..].ptr, exported.structure.ordinals.ptr);
    try equal(@as(?u64, null), exported.pattern_revision);
    try equal(@as(?u64, null), exported.values_revision);
    try equal(@as(usize, 1), counter.attempts);
    indices[1] = 5;
    try std.testing.expectError(error.InvalidOrdinal, borrowed.exportView(.row));
    try std.testing.expectError(error.InvalidOrdinal, borrowed.deepCopy());
    try equal(@as(usize, 1), counter.attempts);
    const retained = try borrowed.retain();
    retained.release();
    borrowed.release(); // Does not inspect now-invalid borrowed structure.
    try counter.expectReleased();
    try equal(@as(u32, 5), indices[1]);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &values);
}

test "object split snapshot reads active entries only and zero fills undefined gaps and tails" {
    const starts = [_]i64{ 2, 0, 6, 9 };
    const ends = [_]i64{ 4, 0, 7, 9 };
    var indices: [10]u32 = undefined;
    indices[2] = 2;
    indices[3] = 0;
    indices[6] = 1;
    var values: [8]f64 = undefined;
    values[2] = 4;
    values[3] = -2;
    values[6] = @bitCast(@as(u64, 0x7ff8123456789abc));
    const source: pattern.Pattern(i64, u32) = .{ .ranges = .{ .split = .{ .starts = &starts, .ends = &ends } }, .ordinals = &indices, .row_count = 4, .column_count = 3 };
    const owned = try object.MatrixControl(f64, i64, u32).snapshot(std.testing.allocator, source, &values);
    defer owned.release();
    const exported = try owned.exportView(.row);
    try equal(@as(usize, 3), exported.statistics.referenced_entries);
    try equal(@as(usize, 7), exported.statistics.storage_extent);
    try std.testing.expectEqualSlices(i64, &starts, exported.structure.ranges.split.starts);
    try std.testing.expectEqualSlices(i64, &ends, exported.structure.ranges.split.ends);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 2, 0, 0, 0, 1, 0, 0, 0 }, exported.structure.ordinals);
    const expected = [_]f64{ 0, 0, 4, -2, 0, 0, @bitCast(@as(u64, 0x7ff8123456789abc)), 0 };
    try expectBits(f64, &expected, exported.values);
}

test "object shared starts ends remain borrowed and overlapping nonmonotonic ranges preserve order" {
    var counter: Counted = .{};
    const offsets = [_]u64{ 0, 2, 4 };
    const indices = [_]i32{ 1, 0, 1, 0 };
    const values = [_]f32{ 2, 3, 5, 7 };
    const source: pattern.Pattern(u64, i32) = .{ .ranges = .{ .split = .{ .starts = offsets[0..2], .ends = offsets[1..] } }, .ordinals = &indices, .row_count = 2, .column_count = 2 };
    const borrowed = try object.MatrixControl(f32, u64, i32).borrow(counter.allocator(), source, &values);
    const owned = try borrowed.deepCopy();
    borrowed.release();
    const exported = try owned.exportView(.row);
    try std.testing.expectEqualSlices(u64, &.{ 0, 2 }, exported.structure.ranges.split.starts);
    try std.testing.expectEqualSlices(u64, &.{ 2, 4 }, exported.structure.ranges.split.ends);
    owned.release();
    try counter.expectReleased();

    const overlapping: pattern.Pattern(u64, i32) = .{ .ranges = .{ .split = .{ .starts = &.{ 2, 0, 1 }, .ends = &.{ 4, 2, 3 } } }, .ordinals = &indices, .row_count = 3, .column_count = 2 };
    const repeated = try object.MatrixControl(f32, u64, i32).snapshot(std.testing.allocator, overlapping, &values);
    defer repeated.release();
    const result = try repeated.exportView(.row);
    try equal(@as(usize, 6), result.statistics.referenced_entries);
    var dense: [6]f32 = @splat(0);
    var visited: [6]usize = undefined;
    var count: usize = 0;
    for (0..3) |row| {
        const range = try result.structure.checkedMajorRange(row);
        for (range.begin..range.end) |entry| {
            const column = try result.structure.checkedOrdinalAt(entry);
            dense[row * 2 + column] += result.values[entry];
            visited[count] = entry;
            count += 1;
        }
    }
    try std.testing.expectEqualSlices(f32, &.{ 7, 5, 3, 2, 3, 5 }, &dense);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 0, 1, 1, 2 }, &visited);
}

test "object allocation failures unwind every contiguous and split owned allocation" {
    const indices = [_]i32{ 0, 1 };
    const values = [_]f64{ 1, @bitCast(@as(u64, 0x8000000000000000)) };
    for ([_]bool{ false, true }) |split| {
        const source: pattern.Pattern(u64, i32) = .{ .ranges = if (split) .{ .split = .{ .starts = &.{ 0, 1 }, .ends = &.{ 1, 2 } } } else .{ .contiguous = &.{ 0, 1, 2 } }, .ordinals = &indices, .row_count = 2, .column_count = 2 };
        const allocation_count: usize = if (split) 5 else 4;
        for (0..allocation_count + 1) |failure| {
            var counter: Counted = .{ .fail_at = failure };
            const result = object.MatrixControl(f64, u64, i32).snapshot(counter.allocator(), source, &values);
            if (failure < allocation_count) {
                try std.testing.expectError(error.OutOfMemory, result);
                try equal(failure + 1, counter.attempts);
                try equal(failure, counter.allocations);
            } else {
                const owned = try result;
                try equal(allocation_count, counter.allocations);
                owned.release();
                for (0..allocation_count) |j| try equal(allocation_count - 1 - j, counter.free_order[j]);
            }
            try counter.expectReleased();
            try expectBits(f64, &.{ 1, @bitCast(@as(u64, 0x8000000000000000)) }, &values);
        }
    }
    var failed: Counted = .{ .fail_at = 0 };
    const source: pattern.Pattern(u64, i32) = .{ .ranges = .{ .contiguous = &.{ 0, 2 } }, .ordinals = &indices, .row_count = 1, .column_count = 2 };
    try std.testing.expectError(error.OutOfMemory, object.MatrixControl(f64, u64, i32).borrow(failed.allocator(), source, &values));
    try failed.expectReleased();
}

test "object deep copies are independent and release uses captured allocator after caller reassignment" {
    var original: Counted = .{};
    var replacement: Counted = .{};
    var allocator = original.allocator();
    var indices = [_]i32{ 0, 1 };
    var values = [_]f64{ 11, -4 };
    var offsets = [_]u32{ 0, 2 };
    const source: pattern.Pattern(u32, i32) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &indices, .row_count = 1, .column_count = 2 };
    const borrowed = try object.MatrixControl(f64, u32, i32).borrow(allocator, source, &values);
    const snapshot = try borrowed.deepCopy();
    const snapshot2 = try snapshot.deepCopy();
    allocator = replacement.allocator();
    try expect(allocator.ptr == @as(*anyopaque, @ptrCast(&replacement)));
    borrowed.release();
    snapshot.release();
    indices[0] = 1;
    values[0] = 99;
    offsets[1] = 0;
    const exported = try snapshot2.exportView(.row);
    try std.testing.expectEqualSlices(f64, &.{ 11, -4 }, exported.values);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, exported.structure.ordinals);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, exported.structure.ranges.contiguous);
    snapshot2.release();
    try original.expectReleased();
    try equal(@as(usize, 0), replacement.allocations);
    try replacement.expectReleased();
}

test "object explicit retain keeps stable address and last release alone destroys owned storage" {
    var counter: Counted = .{};
    const source: pattern.Pattern(u32, u64) = .{ .ranges = .{ .contiguous = &.{ 0, 1 } }, .ordinals = &.{0}, .row_count = 1, .column_count = 1 };
    const owned = try object.MatrixControl(f32, u32, u64).snapshot(counter.allocator(), source, &.{7});
    const retained = try owned.retain();
    try equal(owned, retained);
    const before = try owned.exportView(.row);
    owned.release();
    try equal(@as(usize, 0), counter.frees);
    const after = try retained.exportView(.row);
    try equal(before.values.ptr, after.values.ptr);
    try equal(@as(f32, 7), after.values[0]);
    retained.references = std.math.maxInt(usize); // Test the checked counter boundary.
    try std.testing.expectError(error.ReferenceCountOverflow, retained.retain());
    try equal(std.math.maxInt(usize), retained.references);
    retained.references = 1;
    retained.release();
    try counter.expectReleased();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1, 0 }, counter.free_order[0..counter.frees]);
}

test "object zero major dimensions and empty split endpoints beyond value capacity are valid" {
    for ([_]pattern.MajorAxis{ .row, .column }) |axis| {
        for ([_]pattern.IndexBase{ .zero, .one }) |base| {
            const first: i64 = @intFromBool(base == .one);
            for ([_]bool{ false, true }) |split| {
                const source: pattern.Pattern(i64, u32) = .{ .ranges = if (split) .{ .split = .{ .starts = &.{}, .ends = &.{} } } else .{ .contiguous = &.{first} }, .ordinals = &.{}, .row_count = if (axis == .row) 0 else 5, .column_count = if (axis == .column) 0 else 5, .major_axis = axis, .index_base = base };
                var counter: Counted = .{};
                const owned = try object.MatrixControl(f64, i64, u32).snapshot(counter.allocator(), source, &.{});
                const exported = try owned.exportView(axis);
                try equal(@as(usize, 0), exported.statistics.referenced_entries);
                try equal(@as(usize, 0), exported.values.len);
                try std.testing.expectError(error.WrongFormat, owned.exportView(if (axis == .row) .column else .row));
                owned.release();
                try counter.expectReleased();
            }
        }
    }
    var inactive: [3]u32 = undefined;
    const source: pattern.Pattern(u64, u32) = .{ .ranges = .{ .split = .{ .starts = &.{3}, .ends = &.{3} } }, .ordinals = &inactive, .row_count = 1, .column_count = 0 };
    const owned = try object.MatrixControl(f64, u64, u32).snapshot(std.testing.allocator, source, &.{});
    defer owned.release();
    const exported = try owned.exportView(.row);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, exported.structure.ordinals);
    try equal(@as(usize, 0), exported.values.len);
}

test "object invalid structure and overflow reject before allocator or value reads" {
    var counter: Counted = .{ .fail_at = 0 };
    const Control = object.MatrixControl(f64, i64, u32);
    const overflow: pattern.Pattern(i64, u32) = .{ .ranges = .{ .contiguous = &.{} }, .ordinals = &.{}, .row_count = std.math.maxInt(usize), .column_count = 0 };
    try std.testing.expectError(error.SizeOverflow, Control.snapshot(counter.allocator(), overflow, &.{}));
    const invalid: pattern.Pattern(i64, u32) = .{ .ranges = .{ .split = .{ .starts = &.{-1}, .ends = &.{1} } }, .ordinals = &.{2}, .row_count = 1, .column_count = 1 };
    var undefined_values: [1]f64 = undefined;
    try std.testing.expectError(error.InvalidOffsets, Control.snapshot(counter.allocator(), invalid, &undefined_values));
    const too_small: pattern.Pattern(i64, u32) = .{ .ranges = .{ .contiguous = &.{ 0, 1 } }, .ordinals = &.{0}, .row_count = 1, .column_count = 1 };
    try std.testing.expectError(error.BufferTooSmall, Control.borrow(counter.allocator(), too_small, &.{}));
    try equal(@as(usize, 0), counter.attempts);
    try counter.expectReleased();
}
