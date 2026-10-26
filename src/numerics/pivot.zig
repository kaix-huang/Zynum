// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed integer pivot codecs, separate from numeric LU and API validation.
//! Frontends retain geometry/capacity/alias/error order. Each successful check
//! applies only while caller-owned storage stays stable through the call.
const std = @import("std");
pub const Base = enum { zero, one };

fn expectInteger(comptime P: type) void {
    if (P != i32 and P != u32 and P != i64 and P != u64 and P != usize)
        @compileError("pivot encoding requires i32/u32/i64/u64 or native usize");
}

/// Whether every legal row can be represented by a writer at a nonempty step.
/// Does not validate capacity, addresses, overlap, or any numerical operand.
/// Empty factorizations make no pivot writes, even with huge row dimensions.
pub fn canWriteRows(comptime P: type, comptime base: Base, rows: usize, steps: usize) bool {
    comptime expectInteger(P);
    if (steps == 0) return true;
    if (steps > rows) return false;
    const last = std.math.cast(P, rows - 1) orelse return false;
    if (base == .one) _ = std.math.add(P, last, 1) catch return false;
    return true;
}

/// Representation-only normalization. The caller still checks step<=row<rows.
/// Subtract in P before the native-width cast; never normalize through i32.
pub fn decode(comptime P: type, comptime base: Base, value: P) ?usize {
    comptime expectInteger(P);
    if (value < @as(P, if (base == .one) 1 else 0)) return null;
    const zero_based = value - @as(P, if (base == .one) 1 else 0);
    return std.math.cast(usize, zero_based);
}

/// No element is read until the frontend's original pivot-validation position.
pub fn validStep(comptime P: type, comptime base: Base, value: P, step: usize, rows: usize) bool {
    const row = decode(P, base, value) orelse return false;
    return row >= step and row < rows;
}

pub fn Writer(comptime P: type, comptime base: Base) type {
    comptime expectInteger(P);
    return struct {
        values: []P,

        /// Frontend has proven capacity and canWriteRows for the shape; the
        /// numerical loop supplies a selected row in [step, rows). This exact
        /// store belongs where the original pivots[step] assignment occurred.
        pub fn writeUnchecked(self: @This(), step: usize, row: usize) void {
            const converted: P = @intCast(row);
            self.values[step] = converted + @as(P, if (base == .one) 1 else 0);
        }
    };
}

pub fn Reader(comptime P: type, comptime base: Base) type {
    comptime expectInteger(P);
    return struct {
        values: []const P,

        /// Complete sequence validation preceded mutation; no persistent proof
        /// or ownership is carried by this descriptor. Caller prevents mutation.
        pub fn readUnchecked(self: @This(), step: usize) usize {
            return @intCast(self.values[step] - @as(P, if (base == .one) 1 else 0));
        }
    };
}

test "all encodings share logical swaps and write the caller buffer immediately without a copy" {
    inline for (.{ i32, u32, i64, u64, usize }) |P| inline for (.{ Base.zero, Base.one }) |base| {
        var encoded: [4]P = @splat(77);
        const writer: Writer(P, base) = .{ .values = &encoded };
        const reader: Reader(P, base) = .{ .values = &encoded };
        try std.testing.expectEqual(@intFromPtr(&encoded), @intFromPtr(writer.values.ptr));
        try std.testing.expect(canWriteRows(P, base, 5, 3));
        writer.writeUnchecked(0, 4);
        try std.testing.expectEqual(@as(P, if (base == .one) 5 else 4), encoded[0]);
        try std.testing.expectEqualSlices(P, &.{ 77, 77, 77 }, encoded[1..]);
        writer.writeUnchecked(1, 1);
        try std.testing.expectEqual(@as(usize, 4), reader.readUnchecked(0));
        try std.testing.expectEqual(@as(usize, 1), reader.readUnchecked(1));
        // Model an arithmetic abort after step 1: its pivot was already stored,
        // even when an earlier singular step keeps the returned numeric info=1.
        try std.testing.expectEqualSlices(P, &.{ 77, 77 }, encoded[2..]);
        try std.testing.expect(validStep(P, base, encoded[0], 0, 5));
        try std.testing.expect(validStep(P, base, encoded[1], 1, 5));
    };
}

test "signedness base and complete-sequence semantic checks remain distinct" {
    inline for (.{ i32, i64 }) |P| {
        try std.testing.expectEqual(@as(?usize, null), decode(P, .zero, -1));
        try std.testing.expectEqual(@as(?usize, null), decode(P, .one, -1));
    }
    inline for (.{ i32, u32, i64, u64, usize }) |P| {
        try std.testing.expectEqual(@as(?usize, 0), decode(P, .zero, 0));
        try std.testing.expectEqual(@as(?usize, null), decode(P, .one, 0));
        try std.testing.expectEqual(@as(?usize, 0), decode(P, .one, 1));
        try std.testing.expect(!validStep(P, .zero, 0, 1, 2));
        try std.testing.expect(!validStep(P, .zero, 2, 0, 2));
        try std.testing.expect(!validStep(P, .one, 1, 1, 2));
        try std.testing.expect(!validStep(P, .one, 3, 0, 2));
        try std.testing.expect(validStep(P, .one, 2, 1, 2));
    }
}

test "row encoding is not restricted by i32 geometry or numerical info representation" {
    const wide_rows: usize = @as(usize, std.math.maxInt(i32)) + 1;
    try std.testing.expect(canWriteRows(i32, .zero, wide_rows, 1));
    try std.testing.expect(!canWriteRows(i32, .one, wide_rows, 1));
    try std.testing.expect(canWriteRows(i64, .one, wide_rows, 1));
    inline for (.{ Base.zero, Base.one }) |base| {
        var values: [1]i64 = undefined;
        const writer: Writer(i64, base) = .{ .values = &values };
        const reader: Reader(i64, base) = .{ .values = &values };
        const row: usize = @as(usize, std.math.maxInt(i32)) + 1;
        try std.testing.expect(canWriteRows(i64, base, row + 1, 1));
        writer.writeUnchecked(0, row);
        try std.testing.expectEqual(row, reader.readUnchecked(0));
    }
    try std.testing.expect(canWriteRows(i32, .one, std.math.maxInt(usize), 0));
    try std.testing.expect(!canWriteRows(i32, .zero, 0, 1));
    try std.testing.expect(canWriteRows(usize, .one, std.math.maxInt(usize), 1));
}

test "mutating borrowed pivots invalidates an earlier check and must be revalidated" {
    var pivots = [_]i32{ 0, 1 };
    try std.testing.expect(validStep(i32, .zero, pivots[1], 1, 2));
    pivots[1] = 0;
    try std.testing.expect(!validStep(i32, .zero, pivots[1], 1, 2));
    // No Reader is invoked after failed validation; it is not an owning plan.
}

test "wide pivot encodings normalize before target-width conversion and preserve maximum native rows" {
    if (comptime @bitSizeOf(usize) < 64) {
        const beyond = @as(u64, std.math.maxInt(usize)) + 1;
        inline for (.{ i64, u64 }) |P| {
            const encoded: P = @intCast(beyond);
            try std.testing.expectEqual(@as(?usize, null), decode(P, .zero, encoded));
            // One-based subtraction takes place in P; narrowing first would
            // wrongly reject this representable normalized native integer.
            try std.testing.expectEqual(@as(?usize, std.math.maxInt(usize)), decode(P, .one, encoded));
            try std.testing.expectEqual(@as(?usize, null), decode(P, .one, encoded + 1));
            // A representable integer is still not a row in any realizable
            // usize-sized row count when it equals maxInt(usize).
            try std.testing.expect(!validStep(P, .one, encoded, 0, std.math.maxInt(usize)));
        }
    }
    if (comptime @bitSizeOf(usize) > 32) {
        const rows: usize = @as(usize, std.math.maxInt(u32)) + 1;
        try std.testing.expect(canWriteRows(u32, .zero, rows, 1));
        try std.testing.expect(!canWriteRows(u32, .one, rows, 1));
        try std.testing.expect(canWriteRows(u64, .one, rows, 1));
    }
    var value: [1]usize = undefined;
    const writer: Writer(usize, .one) = .{ .values = &value };
    const reader: Reader(usize, .one) = .{ .values = &value };
    try std.testing.expect(canWriteRows(usize, .one, std.math.maxInt(usize), 1));
    writer.writeUnchecked(0, std.math.maxInt(usize) - 1);
    try std.testing.expectEqual(std.math.maxInt(usize), value[0]);
    try std.testing.expectEqual(std.math.maxInt(usize) - 1, reader.readUnchecked(0));
}
