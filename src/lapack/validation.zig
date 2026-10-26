// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const storage = @import("../numerics/storage.zig");
const views = @import("../blas/api/views.zig");

pub const Triangle = enum { upper, lower };
pub const Error = error{
    InvalidLength,
    InvalidLeadingDimension,
    BufferTooSmall,
    DimensionMismatch,
    InvalidMatrixOperation,
    AliasingNotAllowed,
    InvalidPivot,
    InvalidReflector,
    InvalidTolerance,
    InvalidHermitianDiagonal,
    NonFiniteInput,
};

pub fn expectReal(comptime T: type) void {
    if (T != f32 and T != f64) @compileError("LAPACK scalar type must be f32 or f64");
}

/// Validate the complete physical span without relying on BLAS fast-mode checks.
/// Empty shapes require no storage and accept any nonnegative leading dimension,
/// including zero for an m-by-0 matrix, matching BLAS empty-storage conventions.
/// Negative leading dimensions are rejected even for empty shapes.
pub fn matrixSpan(comptime T: type, value: anytype) Error!usize {
    comptime expectReal(T);
    if (value.row_count < 0 or value.column_count < 0) return error.InvalidLength;
    if (value.leading_dimension < 0) return error.InvalidLeadingDimension;
    if (value.row_count == 0 or value.column_count == 0) return 0;
    if (value.leading_dimension < value.row_count) return error.InvalidLeadingDimension;
    const rows: usize = @intCast(value.row_count);
    const columns: usize = @intCast(value.column_count);
    const stride: usize = @intCast(value.leading_dimension);
    const span = storage.columnMajorExtent(rows, columns, stride) catch return error.InvalidLength;
    if (value.values.len < span) return error.BufferTooSmall;
    return span;
}

pub fn normalFactor(comptime T: type, value: views.ConstMatrix(T)) Error!usize {
    const span = try matrixSpan(T, value);
    if (value.operation != .normal) return error.InvalidMatrixOperation;
    if (value.row_count != value.column_count) return error.DimensionMismatch;
    return span;
}

const Range = storage.ByteRange;

fn range(comptime T: type, values: []const T) Error!Range {
    const bytes = storage.byteLength(values.len, @sizeOf(T)) catch return error.InvalidLength;
    const start = @intFromPtr(values.ptr);
    return storage.byteRange(start, bytes) catch return error.InvalidLength;
}

/// Includes intervening matrix padding; only the used pivot prefix is checked.
pub fn noOverlap(comptime A: type, first: []const A, comptime B: type, second: []const B) Error!void {
    const a = try range(A, first);
    const b = try range(B, second);
    if (storage.overlaps(a, b)) {
        return error.AliasingNotAllowed;
    }
}

pub fn finiteMatrix(comptime T: type, value: anytype) Error!void {
    const rows: usize = @intCast(value.row_count);
    const columns: usize = @intCast(value.column_count);
    if (rows == 0 or columns == 0) return;
    const ld: usize = @intCast(value.leading_dimension);
    for (0..columns) |column| {
        for (0..rows) |row| {
            if (!std.math.isFinite(@as(T, value.values[row + column * ld]))) return error.NonFiniteInput;
        }
    }
}

/// Access the virtual lower factor: upper storage represents its transpose.
pub fn triangleIndex(row: usize, column: usize, ld: usize, triangle: Triangle) usize {
    return if (triangle == .lower) row + column * ld else column + row * ld;
}

pub fn finiteTriangle(comptime T: type, value: anytype, triangle: Triangle) Error!void {
    const n: usize = @intCast(value.row_count);
    if (n == 0) return;
    const ld: usize = @intCast(value.leading_dimension);
    // Validation order does not affect the error or mutation contract. Walk
    // the stored triangle contiguously instead of virtually transposing upper.
    switch (triangle) {
        .lower => for (0..n) |column| {
            for (column..n) |row| {
                if (!std.math.isFinite(@as(T, value.values[row + column * ld]))) return error.NonFiniteInput;
            }
        },
        .upper => for (0..n) |column| {
            for (0..column + 1) |row| {
                if (!std.math.isFinite(@as(T, value.values[row + column * ld]))) return error.NonFiniteInput;
            }
        },
    }
}
