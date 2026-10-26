// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal column-major operands with native-width unsigned dimensions.
//! These are borrowed computational descriptors, not public BLAS/API views.
//! Frontends own signed arguments, transforms, finite/alias checks, and their
//! leading-dimension rules for empty operands. No adapter or index ABI is used.

const std = @import("std");
const storage = @import("storage.zig");
pub const Error = storage.Error || error{ InvalidLeadingDimension, BufferTooSmall };

/// Minimum physical extent for computation. Empty operands need no storage;
/// stricter API leading-dimension rules must be checked by that frontend.
pub fn extent(rows: usize, columns: usize, leading_dimension: usize) Error!usize {
    if (rows != 0 and columns != 0 and leading_dimension < rows) return error.InvalidLeadingDimension;
    return storage.columnMajorExtent(rows, columns, leading_dimension);
}

pub fn ConstMatrix(comptime T: type) type {
    return struct {
        values: []const T,
        row_count: usize,
        column_count: usize,
        leading_dimension: usize,

        /// Geometry/capacity only; does not read elements or validate addresses,
        /// scalar types, aliasing, numerical values, or algorithm-specific shape.
        pub fn check(self: @This()) Error!usize {
            const used = try extent(self.row_count, self.column_count, self.leading_dimension);
            if (self.values.len < used) return error.BufferTooSmall;
            return used;
        }
    };
}

pub fn Matrix(comptime T: type) type {
    return struct {
        values: []T,
        row_count: usize,
        column_count: usize,
        leading_dimension: usize,

        pub fn asConst(self: @This()) ConstMatrix(T) {
            return .{ .values = self.values, .row_count = self.row_count, .column_count = self.column_count, .leading_dimension = self.leading_dimension };
        }

        /// Same geometry/capacity check as the const descriptor.
        pub fn check(self: @This()) Error!usize {
            return self.asConst().check();
        }
    };
}

test "native width descriptors validate dimensions beyond i32 without allocation" {
    const wide = @as(usize, std.math.maxInt(i32)) + 1;
    const empty: ConstMatrix(f64) = .{ .values = &.{}, .row_count = 0, .column_count = wide, .leading_dimension = std.math.maxInt(usize) };
    try std.testing.expectEqual(@as(usize, 0), try empty.check());
    const tall_empty: ConstMatrix(f64) = .{ .values = &.{}, .row_count = wide, .column_count = 0, .leading_dimension = 0 };
    try std.testing.expectEqual(@as(usize, 0), try tall_empty.check());
    const tall: ConstMatrix(f64) = .{ .values = &.{}, .row_count = wide, .column_count = 1, .leading_dimension = wide };
    try std.testing.expectEqual(wide, try extent(wide, 1, wide));
    try std.testing.expectError(error.BufferTooSmall, tall.check());
    const many_columns: ConstMatrix(f64) = .{ .values = &.{}, .row_count = 1, .column_count = wide, .leading_dimension = 1 };
    try std.testing.expectEqual(wide, try extent(1, wide, 1));
    try std.testing.expectError(error.BufferTooSmall, many_columns.check());
}

test "descriptor checks distinguish stride capacity overflow and physical padding" {
    try std.testing.expectError(error.InvalidLeadingDimension, extent(3, 1, 2));
    try std.testing.expectError(error.SizeOverflow, extent(1, 3, std.math.maxInt(usize)));
    try std.testing.expectError(error.SizeOverflow, extent(2, 2, std.math.maxInt(usize)));
    var values = [_]u8{ 1, 2, 99, 3, 4, 77 };
    const matrix: Matrix(u8) = .{ .values = &values, .row_count = 2, .column_count = 2, .leading_dimension = 3 };
    try std.testing.expectEqual(@as(usize, 5), try matrix.check());
    const readonly = matrix.asConst();
    try std.testing.expectEqual(@intFromPtr(matrix.values.ptr), @intFromPtr(readonly.values.ptr));
    try std.testing.expectEqual(@as(usize, 5), try readonly.check());
    var short = matrix;
    short.values = values[0..4];
    try std.testing.expectError(error.BufferTooSmall, short.check());
    short.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, short.check());
}

/// Regular dense panel geometry. This is an internal borrowed operand, not a
/// public layout enum. Only the two conventional nonoverlapping layouts exist.
pub const Layout = enum { row_major, column_major };
const checkedColumnMajorExtent = extent;
pub const PanelShape = struct {
    row_count: usize,
    column_count: usize,
    leading_dimension: usize,
    layout: Layout,

    /// Minimal bounding prefix, including internal padding but not the tail.
    /// Empty panels need no storage or stride arithmetic. Frontends may impose
    /// stronger empty leading-dimension rules before constructing this operand.
    pub fn extent(self: @This()) Error!usize {
        return switch (self.layout) {
            .column_major => checkedColumnMajorExtent(self.row_count, self.column_count, self.leading_dimension),
            .row_major => checkedColumnMajorExtent(self.column_count, self.row_count, self.leading_dimension),
        };
    }
};

/// A borrowed column derived from a checked regular panel, or a contiguous
/// vector. No address is derived for an empty column. These accessors do not
/// replace per-call geometry, capacity, lifetime, numerical or alias checks.
pub fn ConstColumn(comptime T: type) type {
    return struct {
        values: []const T,
        length: usize,
        stride: usize,

        pub fn at(self: @This(), row: usize) T {
            return self.values[row * self.stride];
        }
    };
}
pub fn Column(comptime T: type) type {
    return struct {
        values: []T,
        length: usize,
        stride: usize,

        pub fn at(self: @This(), row: usize) *T {
            return &self.values[row * self.stride];
        }
    };
}

pub fn ConstPanel(comptime T: type) type {
    return struct {
        values: []const T,
        shape: PanelShape,

        /// Geometry/capacity only, with the same boundary responsibilities as
        /// ConstMatrix.check. Does not read numerical elements.
        pub fn check(self: @This()) Error!usize {
            const used = try self.shape.extent();
            if (self.values.len < used) return error.BufferTooSmall;
            return used;
        }

        /// Requires a checked panel and column < shape.column_count. The empty
        /// branch precedes any column offset even for arbitrarily large strides.
        pub fn column(self: @This(), index: usize) ConstColumn(T) {
            if (self.shape.row_count == 0) return .{ .values = self.values[0..0], .length = 0, .stride = 1 };
            const first = if (self.shape.layout == .column_major) index * self.shape.leading_dimension else index;
            return .{ .values = self.values[first..], .length = self.shape.row_count, .stride = if (self.shape.layout == .column_major) 1 else self.shape.leading_dimension };
        }
    };
}
pub fn Panel(comptime T: type) type {
    return struct {
        values: []T,
        shape: PanelShape,

        pub fn asConst(self: @This()) ConstPanel(T) {
            return .{ .values = self.values, .shape = self.shape };
        }
        pub fn check(self: @This()) Error!usize {
            return self.asConst().check();
        }
        pub fn column(self: @This(), index: usize) Column(T) {
            if (self.shape.row_count == 0) return .{ .values = self.values[0..0], .length = 0, .stride = 1 };
            const first = if (self.shape.layout == .column_major) index * self.shape.leading_dimension else index;
            return .{ .values = self.values[first..], .length = self.shape.row_count, .stride = if (self.shape.layout == .column_major) 1 else self.shape.leading_dimension };
        }
    };
}
