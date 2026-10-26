// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed CSR and column-major dense views; validation is enabled in all builds.

const std = @import("std");
const storage = @import("../numerics/storage.zig");
const scalar = @import("scalar.zig");
const pattern = @import("pattern.zig");

pub const Error = error{
    DimensionMismatch,
    InvalidLength,
    InvalidRowOffsets,
    InvalidColumnOffsets,
    InvalidRowIndex,
    InvalidColumnIndex,
    InvalidLeadingDimension,
    BufferTooSmall,
    WorkspaceTooSmall,
    SizeOverflow,
    AliasingNotAllowed,
    NonFiniteInput,
    InvalidTolerance,
    UnsupportedTransform,
};

pub const MatrixTransform = enum { normal, transposed, adjoint };

pub fn expectScalarType(comptime T: type) void {
    if (T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("Sparse scalar type must be f32, f64, ComplexF32, or ComplexF64");
}

fn expectIndexType(comptime I: type) void {
    if (I != i32 and I != u32 and I != i64 and I != u64) {
        @compileError("Sparse index type must be i32, u32, i64, or u64");
    }
}

pub fn optionField(options: anytype, comptime name: []const u8, fallback: anytype) @TypeOf(fallback) {
    return if (@hasField(@TypeOf(options), name)) @field(options, name) else fallback;
}

/// Immutable, zero-based compressed sparse row view. Explicit zero values,
/// empty rows, unsorted columns, and duplicates are permitted. Duplicate entries
/// contribute separately in stored order; the view never sorts or coalesces.
///
/// `row_offsets` has exactly `row_count + 1` entries, starts at zero, is
/// nondecreasing, and ends at `values.len`. `column_indices.len == values.len`.
/// The scalar and index buffers remain owned by the caller.
pub fn ConstCsrMatrix(comptime T: type, comptime I: type) type {
    return ConstCsrMatrixWithIndexTypes(T, I, I);
}

/// Borrow CSR with independent offset and column-index types, without conversion.
pub fn ConstCsrMatrixWithIndexTypes(comptime T: type, comptime O: type, comptime I: type) type {
    comptime expectIndexType(O);
    comptime expectScalarType(T);
    comptime expectIndexType(I);
    return struct {
        pub const Scalar = T;
        pub const Index = I;
        pub const Offset = O;

        values: []const T,
        column_indices: []const I,
        row_offsets: []const O,
        row_count: usize,
        column_count: usize,
        operation: MatrixTransform = .normal,

        /// Validate all structural entries in O(rows + nnz), without reading
        /// numerical values. Also called by every checked multiplication.
        pub fn check(self: @This()) Error!void {
            // This checked-view adapter owns legacy policy and error order;
            // the common pattern layer has no public CSR error vocabulary.
            const offset_count = std.math.add(usize, self.row_count, 1) catch return error.SizeOverflow;
            if (self.row_offsets.len != offset_count or self.column_indices.len != self.values.len) return error.InvalidLength;
            const structure: pattern.Pattern(O, I) = .{
                .ranges = .{ .contiguous = self.row_offsets },
                .ordinals = self.column_indices,
                .row_count = self.row_count,
                .column_count = self.column_count,
            };
            _ = structure.check() catch |err| return switch (err) {
                error.SizeOverflow => error.SizeOverflow,
                error.InvalidLength => error.InvalidLength,
                error.InvalidOffsets => error.InvalidRowOffsets,
                error.InvalidOrdinal => error.InvalidColumnIndex,
                else => unreachable,
            };
        }

        /// View the underlying CSR matrix as transposed, without moving data.
        pub fn transposed(self: @This()) @This() {
            var result = self;
            result.operation = .transposed;
            return result;
        }

        /// View the underlying CSR matrix as conjugate-transposed. Like
        /// transposed(), this sets a storage-relative operation, not a composition.
        /// For real scalars adjoint and transposed have identical arithmetic.
        pub fn adjoint(self: @This()) @This() {
            var result = self;
            result.operation = .adjoint;
            return result;
        }

        pub fn effectiveRowCount(self: @This()) usize {
            return if (self.operation == .normal) self.row_count else self.column_count;
        }

        pub fn effectiveColumnCount(self: @This()) usize {
            return if (self.operation == .normal) self.column_count else self.row_count;
        }
    };
}

/// Check and borrow CSR buffers. `options` must contain `row_count` and
/// `column_count`; `operation` defaults to `.normal`. This O(rows + nnz) setup
/// allocates nothing. Subsequent operations revalidate the borrowed buffers.
pub fn constCsrMatrix(comptime T: type, comptime I: type, row_offsets: []const I, column_indices: []const I, values: []const T, options: anytype) Error!ConstCsrMatrix(T, I) {
    return constCsrMatrixWithIndexTypes(T, I, I, row_offsets, column_indices, values, options);
}

/// Check and borrow mixed-width CSR buffers using the same structural contract.
pub fn constCsrMatrixWithIndexTypes(comptime T: type, comptime O: type, comptime I: type, row_offsets: []const O, column_indices: []const I, values: []const T, options: anytype) Error!ConstCsrMatrixWithIndexTypes(T, O, I) {
    const result: ConstCsrMatrixWithIndexTypes(T, O, I) = .{
        .values = values,
        .column_indices = column_indices,
        .row_offsets = row_offsets,
        .row_count = options.row_count,
        .column_count = options.column_count,
        .operation = optionField(options, "operation", MatrixTransform.normal),
    };
    try result.check();
    return result;
}

/// Storage includes all inter-column padding, but no trailing padding is
/// required after the last column. `leading_dimension >= row_count`, including
/// empty views. Zero rows or zero columns require no backing elements.
pub fn requiredMatrixStorageLength(row_count: usize, column_count: usize, leading_dimension: usize) Error!usize {
    if (leading_dimension < row_count) return error.InvalidLeadingDimension;
    return storage.columnMajorExtent(row_count, column_count, leading_dimension);
}

pub fn ConstMatrix(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        pub const Scalar = T;

        values: []const T,
        row_count: usize,
        column_count: usize,
        leading_dimension: usize,

        pub fn check(self: @This()) Error!void {
            const required = try requiredMatrixStorageLength(self.row_count, self.column_count, self.leading_dimension);
            if (self.values.len < required) return error.BufferTooSmall;
        }
    };
}

pub fn Matrix(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        pub const Scalar = T;

        values: []T,
        row_count: usize,
        column_count: usize,
        leading_dimension: usize,

        pub fn asConst(self: @This()) ConstMatrix(T) {
            return .{
                .values = self.values,
                .row_count = self.row_count,
                .column_count = self.column_count,
                .leading_dimension = self.leading_dimension,
            };
        }

        pub fn check(self: @This()) Error!void {
            try self.asConst().check();
        }
    };
}

/// Borrow a column-major dense input matrix. `options` must contain `row_count`
/// and `column_count`; `leading_dimension` defaults to `row_count`. O(1) check.
pub fn constMatrix(comptime T: type, values: []const T, options: anytype) Error!ConstMatrix(T) {
    const result: ConstMatrix(T) = .{
        .values = values,
        .row_count = options.row_count,
        .column_count = options.column_count,
        .leading_dimension = optionField(options, "leading_dimension", @as(usize, options.row_count)),
    };
    try result.check();
    return result;
}

/// Borrow a column-major dense output matrix. Options and validation match
/// `constMatrix`. Multiplication only writes logical elements, leaving padding
/// and any unused tail of the backing slice untouched.
pub fn matrix(comptime T: type, values: []T, options: anytype) Error!Matrix(T) {
    const result: Matrix(T) = .{
        .values = values,
        .row_count = options.row_count,
        .column_count = options.column_count,
        .leading_dimension = optionField(options, "leading_dimension", @as(usize, options.row_count)),
    };
    try result.check();
    return result;
}
