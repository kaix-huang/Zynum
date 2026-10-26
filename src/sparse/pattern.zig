// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal borrowed compressed-pattern prototype; no allocation or cache.
//! Offset and ordinal types are independent. Pattern metadata never owns or
//! reads numerical values; Operand attaches values and N/T/H separately.
//!
//! Contiguous offsets cover the entire ordinal buffer exactly. Split ranges
//! independently select half-open physical intervals: leading/interior/trailing
//! gaps are untouched and not nonzeros. Overlap and nonmonotonic intervals are
//! an INTERNAL capability, not a claim about an MKL adapter's accepted input.
//! Major coordinates are visited in logical order, entries in physical order
//! within each interval. Repeated references count separately, with their exact
//! physical entry-to-value location retained; nothing sorts or coalesces.
//!
//! Both offsets and ordinals use the explicit zero/one base. Counts/dimensions
//! are target-sized. No persistent validated state exists: check re-reads all
//! metadata on every call. Accessors also check their own inputs. A caller must
//! keep truthful slices alive and stable for each check/traversal; pointer
//! ownership and concurrent mutation cannot be proved from a borrowed slice.

const std = @import("std");
const scalar = @import("scalar.zig");

pub const Error = error{ InvalidLength, InvalidOffsets, InvalidOrdinal, InvalidMajor, InvalidPhysicalEntry, BufferTooSmall, SizeOverflow };
pub const IndexBase = enum { zero, one };
pub const MajorAxis = enum { row, column };
pub const Operation = enum { normal, transposed, adjoint };
pub const EntryRange = struct { begin: usize, end: usize };
pub const Statistics = struct {
    /// References across all major coordinates, not backing-array length.
    /// Overlapping intervals count the same physical entry more than once.
    referenced_entries: usize,
    /// Highest referenced exclusive physical end, zero when no entry is used.
    /// Includes gaps before/between active ranges, never unused trailing storage.
    storage_extent: usize,
};

pub fn expectIndexType(comptime I: type) void {
    if (I != i32 and I != u32 and I != i64 and I != u64) @compileError("Sparse index type must be i32, u32, i64, or u64");
}

pub fn Ranges(comptime O: type) type {
    comptime expectIndexType(O);
    return union(enum) {
        contiguous: []const O,
        split: struct { starts: []const O, ends: []const O },
    };
}

fn baseValue(comptime I: type, base: IndexBase) I {
    return if (base == .zero) 0 else 1;
}

fn offsetToPhysical(comptime O: type, stored: O, base: IndexBase) Error!usize {
    const first = baseValue(O, base);
    if (stored < first) return error.InvalidOffsets;
    // Subtract in O before narrowing: on a narrow target a one-based stored
    // endpoint may itself be wider than usize, while its physical offset fits.
    return std.math.cast(usize, stored - first) orelse error.SizeOverflow;
}

fn ordinalToLogical(comptime I: type, stored: I, base: IndexBase, count: usize) Error!usize {
    const first = baseValue(I, base);
    if (stored < first) return error.InvalidOrdinal;
    const logical = std.math.cast(usize, stored - first) orelse return error.InvalidOrdinal;
    if (logical >= count) return error.InvalidOrdinal;
    return logical;
}

/// Validate range geometry without reading ordinals or values. Capacity is the
/// truthful physical ordinal-buffer length. All metadata is checked before any
/// caller proceeds to ordinal reads. This pure geometry seam permits overflow
/// tests without constructing inaccessible giant slices.
pub fn checkGeometry(comptime O: type, ranges: Ranges(O), base: IndexBase, major_count: usize, capacity: usize) Error!Statistics {
    switch (ranges) {
        .contiguous => |offsets| {
            const count = std.math.add(usize, major_count, 1) catch return error.SizeOverflow;
            if (offsets.len != count) return error.InvalidLength;
            const first = baseValue(O, base);
            const physical_terminal = std.math.cast(O, capacity) orelse return error.SizeOverflow;
            const terminal = std.math.add(O, physical_terminal, first) catch return error.SizeOverflow;
            if (offsets[0] != first) return error.InvalidOffsets;
            if (offsets[major_count] != terminal) return error.InvalidOffsets;
            // Keep the legacy zero-based CSR endpoint/monotonicity proof and
            // reduction: no extra per-entry casts or changed validation order.
            var invalid: u8 = 0;
            for (offsets[0..major_count], offsets[1..]) |previous, next| invalid |= @intFromBool(previous > next);
            if (invalid != 0) return error.InvalidOffsets;
            return .{ .referenced_entries = capacity, .storage_extent = capacity };
        },
        .split => |split| {
            if (split.starts.len != major_count or split.ends.len != major_count) return error.InvalidLength;
            var result: Statistics = .{ .referenced_entries = 0, .storage_extent = 0 };
            for (split.starts, split.ends) |stored_begin, stored_end| {
                const begin = try offsetToPhysical(O, stored_begin, base);
                const end = try offsetToPhysical(O, stored_end, base);
                if (begin > end or end > capacity) return error.InvalidOffsets;
                result.referenced_entries = std.math.add(usize, result.referenced_entries, end - begin) catch return error.SizeOverflow;
                if (begin != end) result.storage_extent = @max(result.storage_extent, end);
            }
            return result;
        },
    }
}

fn checkOrdinals(comptime I: type, ordinals: []const I, base: IndexBase, minor_count: usize) Error!void {
    if (base == .zero) {
        // This is byte-for-byte arithmetic-equivalent to the checked CSR
        // column reduction, including dimensions beyond I's positive range.
        var invalid: u8 = 0;
        if (std.math.cast(I, minor_count)) |limit| {
            for (ordinals) |ordinal| invalid |= @intFromBool(ordinal < 0 or ordinal >= limit);
        } else {
            for (ordinals) |ordinal| invalid |= @intFromBool(ordinal < 0);
        }
        if (invalid != 0) return error.InvalidOrdinal;
    } else {
        for (ordinals) |ordinal| _ = try ordinalToLogical(I, ordinal, base, minor_count);
    }
}

pub fn Pattern(comptime O: type, comptime I: type) type {
    comptime expectIndexType(O);
    comptime expectIndexType(I);
    return struct {
        pub const Offset = O;
        pub const Ordinal = I;
        ranges: Ranges(O),
        ordinals: []const I,
        row_count: usize,
        column_count: usize,
        major_axis: MajorAxis = .row,
        index_base: IndexBase = .zero,

        pub fn majorCount(self: @This()) usize {
            return if (self.major_axis == .row) self.row_count else self.column_count;
        }
        pub fn minorCount(self: @This()) usize {
            return if (self.major_axis == .row) self.column_count else self.row_count;
        }
        pub fn check(self: @This()) Error!Statistics {
            const stats = try checkGeometry(O, self.ranges, self.index_base, self.majorCount(), self.ordinals.len);
            switch (self.ranges) {
                .contiguous => try checkOrdinals(I, self.ordinals, self.index_base, self.minorCount()),
                .split => |split| for (split.starts, split.ends) |first, last| {
                    const begin = try offsetToPhysical(O, first, self.index_base);
                    const end = try offsetToPhysical(O, last, self.index_base);
                    try checkOrdinals(I, self.ordinals[begin..end], self.index_base, self.minorCount());
                },
            }
            return stats;
        }
        /// Local checked access, not a cached validation certificate. Full
        /// consumers still call check() before any mutation and hold storage
        /// stable through traversal. Returned coordinates are zero-based.
        pub fn checkedMajorRange(self: @This(), major: usize) Error!EntryRange {
            if (major >= self.majorCount()) return error.InvalidMajor;
            const pair = switch (self.ranges) {
                .contiguous => |offsets| block: {
                    const count = std.math.add(usize, self.majorCount(), 1) catch return error.SizeOverflow;
                    if (offsets.len != count) return error.InvalidLength;
                    break :block .{ offsets[major], offsets[major + 1] };
                },
                .split => |split| block: {
                    if (split.starts.len != self.majorCount() or split.ends.len != self.majorCount()) return error.InvalidLength;
                    break :block .{ split.starts[major], split.ends[major] };
                },
            };
            const begin = try offsetToPhysical(O, pair[0], self.index_base);
            const end = try offsetToPhysical(O, pair[1], self.index_base);
            if (begin > end or end > self.ordinals.len) return error.InvalidOffsets;
            return .{ .begin = begin, .end = end };
        }
        /// Physical location stays explicit so gaps are never flattened into nnz.
        pub fn checkedOrdinalAt(self: @This(), physical_entry: usize) Error!usize {
            if (physical_entry >= self.ordinals.len) return error.InvalidPhysicalEntry;
            return ordinalToLogical(I, self.ordinals[physical_entry], self.index_base, self.minorCount());
        }
    };
}

/// Read-only numerical attachment. Values can have an independent backing
/// capacity; only the pattern's active physical extent must fit. Values are
/// never read during structural validation. Operation does not change storage.
/// Skip empty intervals before slicing numerical storage: an empty interval may
/// have a physical endpoint beyond values.len without referencing any value.
pub fn Operand(comptime T: type, comptime O: type, comptime I: type) type {
    if (T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("Sparse scalar type must be f32, f64, ComplexF32, or ComplexF64");
    return struct {
        pattern: Pattern(O, I),
        values: []const T,
        operation: Operation = .normal,
        pub fn check(self: @This()) Error!Statistics {
            const stats = try self.pattern.check();
            if (self.values.len < stats.storage_extent) return error.BufferTooSmall;
            return stats;
        }
        pub fn effectiveRowCount(self: @This()) usize {
            return if (self.operation == .normal) self.pattern.row_count else self.pattern.column_count;
        }
        pub fn effectiveColumnCount(self: @This()) usize {
            return if (self.operation == .normal) self.pattern.column_count else self.pattern.row_count;
        }
    };
}

fn testIndex(comptime I: type, value: usize, base: IndexBase) I {
    return @intCast(value + @intFromBool(base == .one));
}
fn testUnused(comptime I: type) I {
    return if (@typeInfo(I).int.signedness == .signed) -1 else std.math.maxInt(I);
}

test "all independent offset ordinal types bases and orientations preserve contiguous physical order" {
    inline for (.{ i32, u32, i64, u64 }) |O| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            for ([_]IndexBase{ .zero, .one }) |base| {
                const offsets = [_]O{ testIndex(O, 0, base), testIndex(O, 3, base), testIndex(O, 3, base), testIndex(O, 5, base) };
                const ordinals = [_]I{ testIndex(I, 2, base), testIndex(I, 0, base), testIndex(I, 2, base), testIndex(I, 1, base), testIndex(I, 0, base) };
                for ([_]MajorAxis{ .row, .column }) |axis| {
                    const a: Pattern(O, I) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &ordinals, .row_count = 3, .column_count = 3, .major_axis = axis, .index_base = base };
                    try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 5, .storage_extent = 5 }, try a.check());
                    try std.testing.expectEqualDeep(EntryRange{ .begin = 3, .end = 3 }, try a.checkedMajorRange(1));
                    const expected = [_]usize{ 2, 0, 2, 1, 0 };
                    for (expected, 0..) |value, at| try std.testing.expectEqual(value, try a.checkedOrdinalAt(at));
                    try std.testing.expectEqualSlices(O, &.{ testIndex(O, 0, base), testIndex(O, 3, base), testIndex(O, 3, base), testIndex(O, 5, base) }, &offsets);
                }
            }
        }
    }
}

test "rectangular column-compressed geometry keeps major and minor dimensions distinct" {
    const a: Pattern(u64, i32) = .{ .ranges = .{ .contiguous = &.{ 0, 2, 3 } }, .ordinals = &.{ 2, 0, 1 }, .row_count = 3, .column_count = 2, .major_axis = .column };
    try std.testing.expectEqual(@as(usize, 2), a.majorCount());
    try std.testing.expectEqual(@as(usize, 3), a.minorCount());
    _ = try a.check();
    try std.testing.expectEqualDeep(EntryRange{ .begin = 2, .end = 3 }, try a.checkedMajorRange(1));
    try std.testing.expectEqual(@as(usize, 2), try a.checkedOrdinalAt(0));
    var wrong = a;
    wrong.major_axis = .row;
    try std.testing.expectError(error.InvalidLength, wrong.check());
    wrong = a;
    wrong.row_count = 2;
    try std.testing.expectError(error.InvalidOrdinal, wrong.check());
}

test "split gaps are not nonzeros and physical value locations stay explicit for every type pair" {
    inline for (.{ i32, u32, i64, u64 }) |O| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            for ([_]IndexBase{ .zero, .one }) |base| {
                const starts = [_]O{ testIndex(O, 1, base), testIndex(O, 4, base), testIndex(O, 5, base) };
                const ends = [_]O{ testIndex(O, 3, base), testIndex(O, 4, base), testIndex(O, 6, base) };
                const ordinals = [_]I{ testUnused(I), testIndex(I, 2, base), testIndex(I, 0, base), testUnused(I), testUnused(I), testIndex(I, 1, base), testUnused(I), testUnused(I) };
                const values = [_]f64{ std.math.nan(f64), 2, 3, std.math.nan(f64), std.math.nan(f64), 4 };
                const matrix: Operand(f64, O, I) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &starts, .ends = &ends } }, .ordinals = &ordinals, .row_count = 3, .column_count = 3, .index_base = base }, .values = &values };
                try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 3, .storage_extent = 6 }, try matrix.check());
                // Independent expected stored entries: (0,2,2),(0,0,3),(2,1,4).
                var dense: [9]f64 = @splat(0);
                for (0..3) |row| {
                    const range = try matrix.pattern.checkedMajorRange(row);
                    for (range.begin..range.end) |entry| dense[row * 3 + try matrix.pattern.checkedOrdinalAt(entry)] += matrix.values[entry];
                }
                try std.testing.expectEqualSlices(f64, &.{ 3, 0, 2, 0, 0, 0, 0, 4, 0 }, &dense);
                try std.testing.expectError(error.InvalidOrdinal, matrix.pattern.checkedOrdinalAt(0));
            }
        }
    }
}

test "internal overlapping nonmonotonic split intervals preserve logical references and duplicates" {
    const a: Pattern(u64, i32) = .{ .ranges = .{ .split = .{ .starts = &.{ 3, 1, 2 }, .ends = &.{ 5, 3, 4 } } }, .ordinals = &.{ -1, 2, 0, 1, 1, -1 }, .row_count = 3, .column_count = 3 };
    try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 6, .storage_extent = 5 }, try a.check());
    var locations: [6]usize = undefined;
    var count: usize = 0;
    for (0..3) |major| {
        const range = try a.checkedMajorRange(major);
        for (range.begin..range.end) |entry| {
            locations[count] = entry;
            count += 1;
        }
    }
    try std.testing.expectEqualSlices(usize, &.{ 3, 4, 1, 2, 2, 3 }, &locations);
}

test "empty contiguous and gapped shapes ignore all unused numerical storage" {
    inline for (.{ i32, u32, i64, u64 }) |O| {
        for ([_]IndexBase{ .zero, .one }) |base| {
            const start: O = testIndex(O, 0, base);
            const empty: Pattern(O, u64) = .{ .ranges = .{ .contiguous = &.{start} }, .ordinals = &.{}, .row_count = 0, .column_count = std.math.maxInt(usize), .index_base = base };
            try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 0, .storage_extent = 0 }, try empty.check());
            const split: Pattern(O, u64) = .{ .ranges = .{ .split = .{ .starts = &.{ testIndex(O, 2, base), testIndex(O, 3, base) }, .ends = &.{ testIndex(O, 2, base), testIndex(O, 3, base) } } }, .ordinals = &.{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) }, .row_count = 2, .column_count = 0, .index_base = base };
            try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 0, .storage_extent = 0 }, try split.check());
            const operand: Operand(f64, O, u64) = .{ .pattern = split, .values = &.{} };
            _ = try operand.check(); // No numerical array needed for empty intervals.
        }
    }
    const zero_major: Pattern(i32, i64) = .{ .ranges = .{ .split = .{ .starts = &.{}, .ends = &.{} } }, .ordinals = &.{-1}, .row_count = 0, .column_count = 0 };
    _ = try zero_major.check();
}

test "geometry rejects negative reversed out of bounds and malformed metadata before ordinal reads" {
    try std.testing.expectError(error.SizeOverflow, checkGeometry(i32, .{ .contiguous = &.{} }, .zero, std.math.maxInt(usize), 0));
    try std.testing.expectError(error.InvalidLength, checkGeometry(i32, .{ .contiguous = &.{} }, .zero, 0, std.math.maxInt(usize)));
    try std.testing.expectError(error.SizeOverflow, checkGeometry(i32, .{ .contiguous = &.{0} }, .zero, 0, @as(usize, std.math.maxInt(i32)) + 1));
    const NativeOffset = if (@bitSizeOf(usize) == 32) u32 else u64;
    try std.testing.expectError(error.SizeOverflow, checkGeometry(NativeOffset, .{ .contiguous = &.{1} }, .one, 0, std.math.maxInt(usize)));
    try std.testing.expectError(error.InvalidOffsets, checkGeometry(i32, .{ .contiguous = &.{ 0, 2, 1, 3 } }, .zero, 3, 3));
    try std.testing.expectError(error.InvalidOffsets, checkGeometry(i32, .{ .contiguous = &.{ 1, 2 } }, .zero, 1, 2));
    try std.testing.expectError(error.InvalidOffsets, checkGeometry(i32, .{ .contiguous = &.{ 0, 1 } }, .zero, 1, 2));
    const malformed = [_][2]i32{ .{ -1, 1 }, .{ 2, 1 }, .{ 0, 4 } };
    for (malformed) |pair| try std.testing.expectError(error.InvalidOffsets, checkGeometry(i32, .{ .split = .{ .starts = &.{pair[0]}, .ends = &.{pair[1]} } }, .zero, 1, 3));
    try std.testing.expectError(error.InvalidOffsets, checkGeometry(u32, .{ .split = .{ .starts = &.{0}, .ends = &.{1} } }, .one, 1, 3));
    try std.testing.expectError(error.InvalidLength, checkGeometry(u32, .{ .split = .{ .starts = &.{0}, .ends = &.{} } }, .zero, 1, 3));
    // Geometry-only counts can exercise target overflow without giant slices.
    const last = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, checkGeometry(u64, .{ .split = .{ .starts = &.{ 0, 0 }, .ends = &.{ last, last } } }, .zero, 2, last));
    const invalid: Pattern(i32, i32) = .{ .ranges = .{ .split = .{ .starts = &.{0}, .ends = &.{2} } }, .ordinals = &.{-1}, .row_count = 1, .column_count = 1 };
    try std.testing.expectError(error.InvalidOffsets, invalid.check());
}

test "dimensions beyond narrow ordinal range and one based ordinal endpoints remain valid" {
    inline for (.{ i32, u32, i64, u64 }) |I| {
        const dimension: usize = @intCast(@min(@as(u64, std.math.maxInt(I)), @as(u64, std.math.maxInt(usize))));
        const endpoint: I = @intCast(dimension);
        const one: Pattern(i32, I) = .{ .ranges = .{ .contiguous = &.{ 1, 2 } }, .ordinals = &.{endpoint}, .row_count = 1, .column_count = dimension, .index_base = .one };
        _ = try one.check();
        try std.testing.expectEqual(dimension - 1, try one.checkedOrdinalAt(0));
        var bad = one;
        bad.ordinals = &.{0};
        try std.testing.expectError(error.InvalidOrdinal, bad.check());
        bad.ordinals = &.{testUnused(I)};
        if (@typeInfo(I).int.signedness == .signed) try std.testing.expectError(error.InvalidOrdinal, bad.check());
        if (comptime std.math.maxInt(I) > std.math.maxInt(usize)) {
            bad.ordinals = &.{std.math.maxInt(I)};
            try std.testing.expectError(error.InvalidOrdinal, bad.check());
        }
    }
    var boundary: Pattern(i32, u32) = .{ .ranges = .{ .contiguous = &.{ 0, 1 } }, .ordinals = &.{std.math.maxInt(u32)}, .row_count = 1, .column_count = std.math.maxInt(u32) };
    try std.testing.expectError(error.InvalidOrdinal, boundary.check());
    if (comptime @bitSizeOf(usize) > 32) {
        boundary.column_count += 1;
        _ = try boundary.check();
    }
}

test "borrowed metadata is revalidated on every check including after successful construction" {
    var starts = [_]i64{0};
    var ends = [_]i64{1};
    var ordinals = [_]u32{0};
    const a: Pattern(i64, u32) = .{ .ranges = .{ .split = .{ .starts = &starts, .ends = &ends } }, .ordinals = &ordinals, .row_count = 1, .column_count = 1 };
    _ = try a.check();
    ends[0] = 2;
    try std.testing.expectError(error.InvalidOffsets, a.check());
    ends[0] = 1;
    ordinals[0] = 1;
    try std.testing.expectError(error.InvalidOrdinal, a.check());
    ordinals[0] = 0;
    starts[0] = -1;
    try std.testing.expectError(error.InvalidOffsets, a.check());
    starts[0] = 0;
    _ = try a.check();
}

test "local accessors bound metadata and retain physical coordinates" {
    var a: Pattern(i32, u32) = .{ .ranges = .{ .contiguous = &.{ 0, 1 } }, .ordinals = &.{0}, .row_count = 1, .column_count = 1 };
    try std.testing.expectError(error.InvalidMajor, a.checkedMajorRange(1));
    try std.testing.expectError(error.InvalidPhysicalEntry, a.checkedOrdinalAt(1));
    a.ranges = .{ .contiguous = &.{0} };
    try std.testing.expectError(error.InvalidLength, a.checkedMajorRange(0));
    a.ranges = .{ .split = .{ .starts = &.{0}, .ends = &.{2} } };
    try std.testing.expectError(error.InvalidOffsets, a.checkedMajorRange(0));
}

test "real and complex operands separate values operation and active physical capacity" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const poison: T = if (T == f32 or T == f64) std.math.nan(T) else .{ .re = std.math.nan(@FieldType(T, "re")), .im = std.math.nan(@FieldType(T, "im")) };
        const values: [6]T = @splat(poison); // All numerical bits remain unread by checks.
        var operand: Operand(T, u64, i32) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 1, 5 }, .ends = &.{ 2, 6 } } }, .ordinals = &.{ -1, 2, -1, -1, -1, 1, -1 }, .row_count = 2, .column_count = 3 }, .values = values[0..5] };
        try std.testing.expectError(error.BufferTooSmall, operand.check());
        operand.values = &values;
        try std.testing.expectEqualDeep(Statistics{ .referenced_entries = 2, .storage_extent = 6 }, try operand.check());
        try std.testing.expectEqual(@as(usize, 2), operand.effectiveRowCount());
        try std.testing.expectEqual(@as(usize, 3), operand.effectiveColumnCount());
        for ([_]Operation{ .transposed, .adjoint }) |op| {
            operand.operation = op;
            _ = try operand.check();
            try std.testing.expectEqual(@as(usize, 3), operand.effectiveRowCount());
            try std.testing.expectEqual(@as(usize, 2), operand.effectiveColumnCount());
        }
    }
}
