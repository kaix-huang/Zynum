// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Checked borrowed CSC/COO views and conversion into caller-owned CSR output.
//! Views and conversions neither allocate nor own buffers.
//!
//! All indices are zero-based. Unsorted indices, duplicates, and explicit zero
//! values are accepted. Conversion groups by row and orders columns ascending;
//! equal (row, column) entries retain their source storage order. Duplicates
//! remain separate additive entries, never silently coalesced. "Sorted CSR"
//! here does NOT mean the duplicate-free canonical format of some libraries.
//! Reordering distinct columns can change subsequent floating-point summation.
//!
//! Input, output and workspace storage must stay alive for each call. The CSR
//! result borrows only caller output buffers; input and workspace can be reused
//! after return. No concurrent mutation is permitted. Output active prefixes
//! and workspace must be mutually disjoint and disjoint from every input.
//! Read-only inputs can overlap one another. Only active prefixes are accessed
//! through output/workspace operands; unused capacity is excluded from alias
//! checks. An unused tail may still change through another active operand.
//!
//! Every conversion fully revalidates borrowed source data, lengths, integer
//! range, capacities and byte-range aliasing BEFORE mutating output/workspace,
//! in all optimization modes. Actual pointer validity remains a caller duty.

const std = @import("std");
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const views = @import("views.zig");

pub const Error = views.Error;

fn expectTypes(comptime T: type, comptime I: type) void {
    views.expectScalarType(T);
    if (I != i32 and I != u32 and I != i64 and I != u64) @compileError("Sparse index type must be i32, u32, i64, or u64");
}

/// Immutable, zero-based compressed sparse column view. Column offsets have
/// exactly columns+1 entries, start at zero, are nondecreasing, and end at nnz.
/// Row indices and values have equal lengths. Empty columns, unsorted rows,
/// duplicates and explicit zero values are accepted; buffers remain caller-owned.
pub fn ConstCscMatrix(comptime T: type, comptime I: type) type {
    comptime expectTypes(T, I);
    return struct {
        pub const Scalar = T;
        pub const Index = I;

        values: []const T,
        row_indices: []const I,
        column_offsets: []const I,
        row_count: usize,
        column_count: usize,

        /// O(columns + nnz) structural validation; numerical values are unread.
        pub fn check(self: @This()) Error!void {
            const offset_count = std.math.add(usize, self.column_count, 1) catch return error.SizeOverflow;
            if (self.column_offsets.len != offset_count or self.row_indices.len != self.values.len) return error.InvalidLength;
            _ = std.math.cast(I, self.values.len) orelse return error.SizeOverflow;
            if (self.column_offsets[0] != 0) return error.InvalidColumnOffsets;
            var previous: usize = 0;
            for (self.column_offsets) |stored| {
                const offset = std.math.cast(usize, stored) orelse return error.InvalidColumnOffsets;
                if (offset < previous or offset > self.values.len) return error.InvalidColumnOffsets;
                previous = offset;
            }
            if (previous != self.values.len) return error.InvalidColumnOffsets;
            for (self.row_indices) |stored| {
                const row = std.math.cast(usize, stored) orelse return error.InvalidRowIndex;
                if (row >= self.row_count) return error.InvalidRowIndex;
            }
        }
    };
}

/// Check and borrow CSC buffers in O(columns + nnz). `options` must contain
/// row_count and column_count. No sorting, copying or allocation occurs.
pub fn constCscMatrix(comptime T: type, comptime I: type, column_offsets: []const I, row_indices: []const I, values: []const T, options: anytype) Error!ConstCscMatrix(T, I) {
    const result: ConstCscMatrix(T, I) = .{
        .column_offsets = column_offsets,
        .row_indices = row_indices,
        .values = values,
        .row_count = options.row_count,
        .column_count = options.column_count,
    };
    try result.check();
    return result;
}

/// Immutable, zero-based coordinate view with equal-length row-index,
/// column-index and value slices. Arbitrary coordinate order, duplicates and
/// explicit zero values are accepted; buffers remain caller-owned.
pub fn ConstCooMatrix(comptime T: type, comptime I: type) type {
    comptime expectTypes(T, I);
    return struct {
        pub const Scalar = T;
        pub const Index = I;

        values: []const T,
        row_indices: []const I,
        column_indices: []const I,
        row_count: usize,
        column_count: usize,

        /// O(nnz) structural validation; numerical values are unread. The COO
        /// view itself needs no offset representation of nnz. Conversion checks
        /// separately that the destination index type can represent terminal nnz.
        pub fn check(self: @This()) Error!void {
            if (self.row_indices.len != self.values.len or self.column_indices.len != self.values.len) return error.InvalidLength;
            for (self.row_indices, self.column_indices) |stored_row, stored_column| {
                const row = std.math.cast(usize, stored_row) orelse return error.InvalidRowIndex;
                const column = std.math.cast(usize, stored_column) orelse return error.InvalidColumnIndex;
                if (row >= self.row_count) return error.InvalidRowIndex;
                if (column >= self.column_count) return error.InvalidColumnIndex;
            }
        }
    };
}

/// Check and borrow COO buffers in O(nnz). `options` must contain row_count and
/// column_count. No sorting, copying or allocation occurs.
pub fn constCooMatrix(comptime T: type, comptime I: type, row_indices: []const I, column_indices: []const I, values: []const T, options: anytype) Error!ConstCooMatrix(T, I) {
    const result: ConstCooMatrix(T, I) = .{
        .row_indices = row_indices,
        .column_indices = column_indices,
        .values = values,
        .row_count = options.row_count,
        .column_count = options.column_count,
    };
    try result.check();
    return result;
}

/// Exact minimum ELEMENT counts. CSR row offsets/column indices use the source
/// index type; values use its scalar type; workspace consists of usize elements.
pub const ConversionRequirements = struct {
    row_offsets: usize,
    column_indices: usize,
    values: usize,
    workspace: usize,
};

/// Arithmetic-only requirements for CSC conversion. Zero nnz needs no workspace;
/// output still needs rows+1 zero offsets. Fail on rows+1 or terminal-nnz overflow.
pub fn cscToCsrRequirements(comptime I: type, row_count: usize, nnz: usize) Error!ConversionRequirements {
    comptime expectTypes(f32, I);
    _ = std.math.cast(I, nnz) orelse return error.SizeOverflow;
    return .{
        .row_offsets = std.math.add(usize, row_count, 1) catch return error.SizeOverflow,
        .column_indices = nnz,
        .values = nnz,
        .workspace = if (nnz == 0) 0 else row_count,
    };
}

/// COO conversion requires rows + 2*nnz + 256 usize scratch elements for
/// nonempty input: row cursors, two permutation buffers and radix counters.
/// Empty input needs no workspace. All arithmetic is checked before mutation.
pub fn cooToCsrRequirements(comptime I: type, row_count: usize, nnz: usize) Error!ConversionRequirements {
    var result = try cscToCsrRequirements(I, row_count, nnz);
    if (nnz != 0) {
        const permutations = std.math.mul(usize, nnz, 2) catch return error.SizeOverflow;
        const with_rows = std.math.add(usize, row_count, permutations) catch return error.SizeOverflow;
        result.workspace = std.math.add(usize, with_rows, 256) catch return error.SizeOverflow;
    }
    return result;
}

const ByteRange = storage.ByteRange;

fn byteRange(comptime T: type, slice: []const T) Error!ByteRange {
    const start = @intFromPtr(slice.ptr);
    const size = storage.byteLength(slice.len, @sizeOf(T)) catch return error.SizeOverflow;
    return storage.byteRange(start, size);
}

fn ensureDisjoint(first: ByteRange, second: ByteRange) Error!void {
    if (storage.overlaps(first, second)) return error.AliasingNotAllowed;
}

fn preflight(arguments: anytype, requirements: ConversionRequirements) Error!void {
    const input = arguments.matrix;
    const T = @TypeOf(input).Scalar;
    const I = @TypeOf(input).Index;
    const row_offsets: []I = arguments.row_offsets;
    const column_indices: []I = arguments.column_indices;
    const values: []T = arguments.values;
    const workspace: []usize = arguments.workspace;
    if (row_offsets.len < requirements.row_offsets or column_indices.len < requirements.column_indices or values.len < requirements.values) return error.BufferTooSmall;
    if (workspace.len < requirements.workspace) return error.WorkspaceTooSmall;
    const outputs = [_]ByteRange{
        try byteRange(I, row_offsets[0..requirements.row_offsets]),
        try byteRange(I, column_indices[0..requirements.column_indices]),
        try byteRange(T, values[0..requirements.values]),
        try byteRange(usize, workspace[0..requirements.workspace]),
    };
    const inputs = [_]ByteRange{
        try byteRange(T, input.values),
        try byteRange(I, input.row_indices),
        if (@hasField(@TypeOf(input), "column_offsets")) try byteRange(I, input.column_offsets) else try byteRange(I, input.column_indices),
    };
    for (outputs, 0..) |output, index| {
        for (outputs[index + 1 ..]) |other_output| try ensureDisjoint(output, other_output);
        for (inputs) |input_range| try ensureDisjoint(output, input_range);
    }
}

fn ResultType(comptime Arguments: type) type {
    const Input = @FieldType(Arguments, "matrix");
    return views.ConstCsrMatrix(Input.Scalar, Input.Index);
}

fn resultView(arguments: anytype, requirements: ConversionRequirements) ResultType(@TypeOf(arguments)) {
    const row_offsets: []@TypeOf(arguments.matrix).Index = arguments.row_offsets;
    const column_indices: []@TypeOf(arguments.matrix).Index = arguments.column_indices;
    const values: []@TypeOf(arguments.matrix).Scalar = arguments.values;
    return .{
        .row_offsets = row_offsets[0..requirements.row_offsets],
        .column_indices = column_indices[0..requirements.column_indices],
        .values = values[0..requirements.values],
        .row_count = arguments.matrix.row_count,
        .column_count = arguments.matrix.column_count,
    };
}

fn buildRowOffsets(comptime I: type, rows: []const I, offsets: []I, cursors: []usize) void {
    @memset(offsets, 0);
    for (rows) |row| offsets[@as(usize, @intCast(row)) + 1] += 1;
    for (cursors, 0..) |*cursor, row| {
        offsets[row + 1] += offsets[row];
        cursor.* = @intCast(offsets[row]);
    }
}

/// Convert CSC to sorted CSR in O(columns + rows + nnz) time. `arguments` has
/// matrix, row_offsets, column_indices, values, workspace; see requirements.
/// Rows inside source columns may be unsorted and repeated. Equal coordinates
/// preserve their source values' order. Numerical values are copied verbatim.
/// Output indices retain the source index type. Returned slices cover only the
/// exact required prefixes and borrow caller output; workspace is reusable on
/// return. All structure, capacities and active-range aliases are checked before
/// any output or workspace mutation in every optimization mode.
pub fn cscToCsr(arguments: anytype) Error!ResultType(@TypeOf(arguments)) {
    var operation = execution.beginDefault();
    defer operation.end();
    const input = arguments.matrix;
    const I = @TypeOf(input).Index;
    const T = @TypeOf(input).Scalar;
    try input.check();
    const requirements = try cscToCsrRequirements(I, input.row_count, input.values.len);
    // Implicit source columns must also be representable as explicit CSR
    // indices. Check only nonempty columns: larger unused dimensions are valid.
    for (0..input.column_count) |column| {
        if (input.column_offsets[column] != input.column_offsets[column + 1]) {
            _ = std.math.cast(I, column) orelse return error.SizeOverflow;
        }
    }
    try preflight(arguments, requirements);
    const offsets: []I = arguments.row_offsets;
    const columns: []I = arguments.column_indices;
    const values: []T = arguments.values;
    const workspace: []usize = arguments.workspace;
    if (input.values.len == 0) {
        @memset(offsets[0..requirements.row_offsets], 0);
        return resultView(arguments, requirements);
    }
    const cursors = workspace[0..input.row_count];
    buildRowOffsets(I, input.row_indices, offsets[0..requirements.row_offsets], cursors);
    for (0..input.column_count) |column| {
        const begin: usize = @intCast(input.column_offsets[column]);
        const end: usize = @intCast(input.column_offsets[column + 1]);
        for (begin..end) |entry| {
            const row: usize = @intCast(input.row_indices[entry]);
            const target = cursors[row];
            columns[target] = @intCast(column);
            values[target] = input.values[entry];
            cursors[row] += 1;
        }
    }
    return resultView(arguments, requirements);
}

fn columnDigit(column: anytype, shift: u6) usize {
    return @intCast((@as(u64, @intCast(column)) >> shift) & 255);
}

/// Convert arbitrary-order COO to sorted CSR with stable radix ordering.
/// Complexity O(rows + sizeof(Index)*(nnz + 256)); no dimension-sized column
/// table, hidden allocation, floating-point arithmetic, or duplicate reduction.
/// Interface and returned ownership match cscToCsr.
pub fn cooToCsr(arguments: anytype) Error!ResultType(@TypeOf(arguments)) {
    var operation = execution.beginDefault();
    defer operation.end();
    const input = arguments.matrix;
    const I = @TypeOf(input).Index;
    const T = @TypeOf(input).Scalar;
    try input.check();
    const requirements = try cooToCsrRequirements(I, input.row_count, input.values.len);
    try preflight(arguments, requirements);
    const offsets: []I = arguments.row_offsets;
    const columns: []I = arguments.column_indices;
    const values: []T = arguments.values;
    const workspace: []usize = arguments.workspace;
    const nnz = input.values.len;
    if (nnz == 0) {
        @memset(offsets[0..requirements.row_offsets], 0);
        return resultView(arguments, requirements);
    }
    const cursors = workspace[0..input.row_count];
    var order: []usize = workspace[input.row_count..][0..nnz];
    var temporary: []usize = workspace[input.row_count + nnz ..][0..nnz];
    const buckets = workspace[input.row_count + 2 * nnz ..][0..256];
    for (order, 0..) |*entry, index| entry.* = index;
    // Stable sort the entry permutation by column. Row scattering below is
    // stable too, so each row is sorted and duplicate value order is retained.
    for (0..@sizeOf(I)) |pass| {
        const shift: u6 = @intCast(pass * 8);
        @memset(buckets, 0);
        for (order) |entry| buckets[columnDigit(input.column_indices[entry], shift)] += 1;
        var cursor: usize = 0;
        for (buckets) |*bucket| {
            const count = bucket.*;
            bucket.* = cursor;
            cursor += count;
        }
        for (order) |entry| {
            const digit = columnDigit(input.column_indices[entry], shift);
            temporary[buckets[digit]] = entry;
            buckets[digit] += 1;
        }
        std.mem.swap([]usize, &order, &temporary);
    }
    buildRowOffsets(I, input.row_indices, offsets[0..requirements.row_offsets], cursors);
    for (order) |entry| {
        const row: usize = @intCast(input.row_indices[entry]);
        const target = cursors[row];
        columns[target] = input.column_indices[entry];
        values[target] = input.values[entry];
        cursors[row] += 1;
    }
    return resultView(arguments, requirements);
}
