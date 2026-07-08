// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Checked public vector and matrix views for the Zig BLAS API.
//!
//! Views describe caller-owned buffers; they never allocate or copy data. In
//! all builds they validate cheap structural invariants such as lengths, strides,
//! and leading dimensions. Debug, ReleaseSafe, and ReleaseSmall builds also check
//! backing slice capacity before operations cross into the unchecked core layer.
//! Matrices use BLAS-style column-major storage with an explicit leading dimension.

const builtin = @import("builtin");
const std = @import("std");
const core = @import("../core.zig");
const types = @import("../types.zig");

pub const BlasInt = types.BlasInt;
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

/// Errors returned by checked public Zig views and operations.
pub const Error = error{
    DimensionMismatch,
    InvalidLength,
    InvalidStride,
    InvalidLeadingDimension,
    BufferTooSmall,
    WorkspaceTooSmall,
    AliasingNotAllowed,
};

/// Backwards-compatible alias for the public BLAS API error set.
pub const BlasError = Error;
/// True for Debug, ReleaseSafe, and ReleaseSmall builds; false only in ReleaseFast.
pub const runtime_checks_enabled = builtin.mode != .ReleaseFast;

/// Logical matrix operation applied by multiplication APIs without changing storage.
pub const MatrixTransform = enum {
    normal,
    transposed,
    adjoint,
};

/// Compatibility alias for older call sites that used `MatrixOperation`.
pub const MatrixOperation = MatrixTransform;

/// Compile-time guard for scalar types supported by the public BLAS API.
pub fn expectScalarType(comptime T: type) void {
    if (T != f32 and T != f64 and T != ComplexF32 and T != ComplexF64) {
        @compileError("BLAS scalar type must be f32, f64, ComplexF32, or ComplexF64");
    }
}

/// Translate a public matrix transform to the internal BLAS transpose mode.
pub fn toCoreTranspose(transform: MatrixTransform) core.TransposeMode {
    return switch (transform) {
        .normal => .no_trans,
        .transposed => .trans,
        .adjoint => .conj_trans,
    };
}

fn absoluteStride(stride: BlasInt) Error!usize {
    if (stride == 0) return error.InvalidStride;
    if (stride == std.math.minInt(BlasInt)) return error.InvalidStride;
    return @intCast(if (stride < 0) -stride else stride);
}

/// Return the minimum element count required to store a strided vector view.
pub fn requiredVectorStorageLength(length: BlasInt, stride: BlasInt) Error!usize {
    if (length < 0) return error.InvalidLength;
    if (length == 0) return 0;
    const step = try absoluteStride(stride);
    const len: usize = @intCast(length);
    return 1 + (len - 1) * step;
}

/// Return the minimum element count required to store a column-major matrix view.
pub fn requiredMatrixStorageLength(row_count: BlasInt, column_count: BlasInt, leading_dimension: BlasInt) Error!usize {
    if (row_count < 0 or column_count < 0) return error.InvalidLength;
    if (row_count == 0 or column_count == 0) return 0;
    if (leading_dimension < @max(@as(BlasInt, 1), row_count)) return error.InvalidLeadingDimension;
    return @as(usize, @intCast(row_count)) + (@as(usize, @intCast(column_count - 1)) * @as(usize, @intCast(leading_dimension)));
}

/// Validate a vector view's structural fields in every build, and its backing
/// slice capacity when runtime checks are enabled.
pub fn validateVectorStorage(data_len: usize, length: BlasInt, stride: BlasInt) Error!void {
    const required_len = try requiredVectorStorageLength(length, stride);
    if (runtime_checks_enabled and data_len < required_len) return error.BufferTooSmall;
}

/// Validate a matrix view's structural fields in every build, and its backing
/// slice capacity when runtime checks are enabled.
pub fn validateMatrixStorage(data_len: usize, row_count: BlasInt, column_count: BlasInt, leading_dimension: BlasInt) Error!void {
    const required_len = try requiredMatrixStorageLength(row_count, column_count, leading_dimension);
    if (runtime_checks_enabled and data_len < required_len) return error.BufferTooSmall;
}

/// Read an optional field from an anonymous options struct with a typed fallback.
pub fn optionField(options: anytype, comptime name: []const u8, fallback: anytype) @TypeOf(fallback) {
    return if (@hasField(@TypeOf(options), name)) @field(options, name) else fallback;
}

fn defaultVectorLength(data_len: usize) Error!BlasInt {
    if (data_len > @as(usize, @intCast(std.math.maxInt(BlasInt)))) return error.InvalidLength;
    return @intCast(data_len);
}

/// Immutable strided vector view over caller-owned values.
pub fn ConstVector(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        /// Scalar element type carried by this view.
        pub const Scalar = T;

        values: []const T,
        length: BlasInt,
        stride: BlasInt = 1,

        /// Validate this view's length, stride, and backing slice capacity.
        pub fn check(self: @This()) Error!void {
            try validateVectorStorage(self.values.len, self.length, self.stride);
        }
    };
}

/// Mutable strided vector view over caller-owned values.
pub fn Vector(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        /// Scalar element type carried by this view.
        pub const Scalar = T;

        values: []T,
        length: BlasInt,
        stride: BlasInt = 1,

        /// Return an immutable view over the same vector storage.
        pub fn asConst(self: @This()) ConstVector(T) {
            return .{ .values = self.values, .length = self.length, .stride = self.stride };
        }

        /// Validate this view's length, stride, and backing slice capacity.
        pub fn check(self: @This()) Error!void {
            try validateVectorStorage(self.values.len, self.length, self.stride);
        }
    };
}

/// Immutable column-major matrix view over caller-owned values.
pub fn ConstMatrix(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        /// Scalar element type carried by this view.
        pub const Scalar = T;

        values: []const T,
        row_count: BlasInt,
        column_count: BlasInt,
        leading_dimension: BlasInt,
        operation: MatrixTransform = .normal,

        /// Return the same matrix viewed as transposed for multiplication APIs.
        pub fn transposed(self: @This()) @This() {
            var result = self;
            result.operation = .transposed;
            return result;
        }

        /// Return the same matrix viewed as conjugate-transposed for multiplication APIs.
        pub fn adjoint(self: @This()) @This() {
            var result = self;
            result.operation = .adjoint;
            return result;
        }

        /// Row count after applying the current logical matrix operation.
        pub fn effectiveRowCount(self: @This()) BlasInt {
            return if (self.operation == .normal) self.row_count else self.column_count;
        }

        /// Column count after applying the current logical matrix operation.
        pub fn effectiveColumnCount(self: @This()) BlasInt {
            return if (self.operation == .normal) self.column_count else self.row_count;
        }

        /// Validate this view's dimensions, leading dimension, and backing slice
        /// capacity.
        pub fn check(self: @This()) Error!void {
            try validateMatrixStorage(self.values.len, self.row_count, self.column_count, self.leading_dimension);
        }
    };
}

/// Mutable column-major matrix view over caller-owned result values.
pub fn Matrix(comptime T: type) type {
    comptime expectScalarType(T);
    return struct {
        /// Scalar element type carried by this view.
        pub const Scalar = T;

        values: []T,
        row_count: BlasInt,
        column_count: BlasInt,
        leading_dimension: BlasInt,

        /// Return an immutable view over the same matrix storage.
        pub fn asConst(self: @This()) ConstMatrix(T) {
            return .{
                .values = self.values,
                .row_count = self.row_count,
                .column_count = self.column_count,
                .leading_dimension = self.leading_dimension,
            };
        }

        /// Validate this view's dimensions, leading dimension, and backing slice
        /// capacity.
        pub fn check(self: @This()) Error!void {
            try validateMatrixStorage(self.values.len, self.row_count, self.column_count, self.leading_dimension);
        }
    };
}

/// Create a checked immutable vector view; `options` may set `length` and
/// `stride`.
pub fn constVector(comptime T: type, values: []const T, options: anytype) Error!ConstVector(T) {
    comptime expectScalarType(T);
    const length: BlasInt = if (@hasField(@TypeOf(options), "length")) options.length else try defaultVectorLength(values.len);
    const stride: BlasInt = optionField(options, "stride", @as(BlasInt, 1));
    try validateVectorStorage(values.len, length, stride);
    return .{ .values = values, .length = length, .stride = stride };
}

/// Create a checked mutable vector view; `options` may set `length` and
/// `stride`.
pub fn vector(comptime T: type, values: []T, options: anytype) Error!Vector(T) {
    comptime expectScalarType(T);
    const length: BlasInt = if (@hasField(@TypeOf(options), "length")) options.length else try defaultVectorLength(values.len);
    const stride: BlasInt = optionField(options, "stride", @as(BlasInt, 1));
    try validateVectorStorage(values.len, length, stride);
    return .{ .values = values, .length = length, .stride = stride };
}

/// Create a checked immutable column-major matrix view.
///
/// `options` must provide `row_count` and `column_count`; `leading_dimension`
/// defaults to `row_count`.
pub fn constMatrix(comptime T: type, values: []const T, options: anytype) Error!ConstMatrix(T) {
    comptime expectScalarType(T);
    const row_count: BlasInt = options.row_count;
    const column_count: BlasInt = options.column_count;
    const leading_dimension: BlasInt = optionField(options, "leading_dimension", row_count);
    try validateMatrixStorage(values.len, row_count, column_count, leading_dimension);
    return .{
        .values = values,
        .row_count = row_count,
        .column_count = column_count,
        .leading_dimension = leading_dimension,
    };
}

/// Create a checked mutable column-major matrix view.
///
/// `options` must provide `row_count` and `column_count`; `leading_dimension`
/// defaults to `row_count`.
pub fn matrix(comptime T: type, values: []T, options: anytype) Error!Matrix(T) {
    comptime expectScalarType(T);
    const row_count: BlasInt = options.row_count;
    const column_count: BlasInt = options.column_count;
    const leading_dimension: BlasInt = optionField(options, "leading_dimension", row_count);
    try validateMatrixStorage(values.len, row_count, column_count, leading_dimension);
    return .{
        .values = values,
        .row_count = row_count,
        .column_count = column_count,
        .leading_dimension = leading_dimension,
    };
}
