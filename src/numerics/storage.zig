// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal storage arithmetic, independent of scalar types and API policies.
//! Callers validate signed dimensions, leading-dimension rules and capacities,
//! and translate SizeOverflow to their existing public error vocabulary.
//! These functions never dereference storage or decide numerical validity.

const std = @import("std");

pub const Error = error{SizeOverflow};
pub const ByteRange = struct { start: usize, end: usize };

/// Physical column-major element extent, including inter-column padding but
/// no padding after the last column. Empty shapes need no elements. The caller
/// must enforce its own leading-dimension policy before using this extent.
pub fn columnMajorExtent(rows: usize, columns: usize, leading_dimension: usize) Error!usize {
    if (rows == 0 or columns == 0) return 0;
    const preceding = std.math.mul(usize, columns - 1, leading_dimension) catch return error.SizeOverflow;
    return std.math.add(usize, preceding, rows) catch return error.SizeOverflow;
}

/// Checked conversion from an element count to a byte count.
pub fn byteLength(element_count: usize, element_size: usize) Error!usize {
    return std.math.mul(usize, element_count, element_size) catch return error.SizeOverflow;
}

/// Half-open address range. Length arithmetic remains separate so each caller
/// can preserve the order in which multiple buffers' lengths and ends validate.
pub fn byteRange(start: usize, byte_length: usize) Error!ByteRange {
    return .{ .start = start, .end = std.math.add(usize, start, byte_length) catch return error.SizeOverflow };
}

/// Both ranges must come from checked byteRange calls. Empty and adjacent
/// ranges do not overlap, including an empty range inside a nonempty one.
pub fn overlaps(first: ByteRange, second: ByteRange) bool {
    return first.start != first.end and second.start != second.end and
        first.start < second.end and second.start < first.end;
}
