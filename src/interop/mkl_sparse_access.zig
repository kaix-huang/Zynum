// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared interface-only geometry and byte ranges. No admission, allocation,
//! state, numerical reads, raw-pointer capacity inference or permanent proof.
const std = @import("std");

pub fn Access(comptime Provider: type, comptime Handles: type) type {
    const P = Provider.Pattern;
    const Storage = Provider.Storage;
    const Execution = Provider.Execution.context;
    return struct {
        pub const OwnerIdentity = Provider.OwnerIdentity;
        pub const Error = Handles.Error || Execution.AdmissionError || error{ NullOutput, InvalidBase, InvalidDimension, UnsupportedDomain };
        pub const Shape = struct { base: P.IndexBase, rows: usize, columns: usize, major: usize };

        pub fn range(comptime T: type, data: []const T) Error!Storage.ByteRange {
            return Storage.byteRange(@intFromPtr(data.ptr), try Storage.byteLength(data.len, @sizeOf(T)));
        }
        pub fn outputRange(comptime T: type, output: *T) Error!Storage.ByteRange {
            return Storage.byteRange(@intFromPtr(output), @sizeOf(T));
        }
        pub fn rejectDataOverlap(comptime T: type, comptime I: type, output: Storage.ByteRange, data: Handles.MutableData(T, I)) Error!void {
            const inputs = [_]Storage.ByteRange{ try range(I, data.starts), try range(I, data.ends), try range(I, data.ordinals), try range(T, data.values) };
            for (inputs) |input| if (Storage.overlaps(output, input)) return error.BufferOverlap;
        }
        fn offset(comptime I: type, stored: I, base: P.IndexBase) Error!usize {
            const b: I = if (base == .zero) 0 else 1;
            if (stored < b) return error.InvalidOffsets;
            return std.math.cast(usize, stored - b) orelse error.SizeOverflow;
        }
        /// Includes empty endpoints in the captured physical backing prefix.
        /// Nonoverlap is this interface's narrow domain, not a common Pattern rule.
        pub fn backingExtent(comptime I: type, starts: []const I, ends: []const I, base: P.IndexBase) Error!usize {
            var previous_end: usize = 0;
            for (starts, ends, 0..) |first, last, major| {
                const begin = try offset(I, first, base);
                const end = try offset(I, last, base);
                if (begin > end) return error.InvalidOffsets;
                if (major != 0 and begin < previous_end) return error.UnsupportedDomain;
                previous_end = end;
            }
            return previous_end;
        }
        pub fn shape(comptime I: type, raw_base: i32, raw_rows: I, raw_columns: I, axis: P.MajorAxis) Error!Shape {
            const base: P.IndexBase = switch (raw_base) {
                0 => .zero,
                1 => .one,
                else => return error.InvalidBase,
            };
            if (raw_rows < 0) return error.InvalidDimension;
            const rows = std.math.cast(usize, raw_rows) orelse return error.SizeOverflow;
            if (raw_columns < 0) return error.InvalidDimension;
            const columns = std.math.cast(usize, raw_columns) orelse return error.SizeOverflow;
            const major = if (axis == .row) rows else columns;
            return .{ .base = base, .rows = rows, .columns = columns, .major = major };
        }
        pub fn CapturedRanges(comptime I: type) type {
            return struct { starts: []I, ends: []I, extent: usize };
        }
        pub fn captureRanges(comptime I: type, geometry: Shape, input_starts: []I, input_ends: []I) Error!CapturedRanges(I) {
            if (input_starts.len < geometry.major or input_ends.len < geometry.major) return error.BufferTooSmall;
            const starts = input_starts[0..geometry.major];
            const ends = input_ends[0..geometry.major];
            _ = try range(I, starts);
            _ = try range(I, ends);
            const extent = try backingExtent(I, starts, ends, geometry.base);
            return .{ .starts = starts, .ends = ends, .extent = extent };
        }
        pub fn captureData(comptime T: type, comptime I: type, ranges: CapturedRanges(I), input_ordinals: []I, input_values: []T) Error!Handles.MutableData(T, I) {
            if (input_ordinals.len < ranges.extent or input_values.len < ranges.extent) return error.BufferTooSmall;
            return .{ .starts = ranges.starts, .ends = ranges.ends, .ordinals = input_ordinals[0..ranges.extent], .values = input_values[0..ranges.extent] };
        }
    };
}
