// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Compact block packers for structured Level 3 algorithms.
//!
//! These routines materialize only one active block. They never expand a full
//! symmetric, Hermitian, or triangular operand.

const std = @import("std");

const types = @import("../../../types.zig");
const gemm_task = @import("task.zig");

pub const BlasInt = gemm_task.BlasInt;

pub const Triangle = enum {
    upper,
    lower,
};

pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans,
};

pub const Diagonal = enum {
    unit,
    non_unit,
};

fn zero(comptime T: type) T {
    return if (T == types.ComplexF32)
        .{ .re = 0, .im = 0 }
    else if (T == types.ComplexF64)
        .{ .re = 0, .im = 0 }
    else
        0;
}

fn one(comptime T: type) T {
    return if (T == types.ComplexF32)
        .{ .re = 1, .im = 0 }
    else if (T == types.ComplexF64)
        .{ .re = 1, .im = 0 }
    else
        1;
}

fn conjugate(comptime T: type, value: T) T {
    return if (T == types.ComplexF32 or T == types.ComplexF64)
        .{ .re = value.re, .im = -value.im }
    else
        value;
}

fn realDiagonal(comptime T: type, value: T) T {
    return if (T == types.ComplexF32 or T == types.ComplexF64)
        .{ .re = value.re, .im = 0 }
    else
        value;
}

inline fn sourceIndex(ld: BlasInt, row: usize, col: usize) usize {
    return gemm_task.matIndex(ld, row, col);
}

/// Packs a logical block of a symmetric or Hermitian matrix into column-major
/// dense storage with leading dimension `rows`.
pub fn packSymmetricBlock(comptime T: type, triangle: Triangle, hermitian: bool, source: [*]const T, ld: BlasInt, row0: usize, col0: usize, rows: usize, cols: usize, buffer: []T) void {
    std.debug.assert(buffer.len >= rows * cols);
    for (0..cols) |j| {
        const global_col = col0 + j;
        for (0..rows) |i| {
            const global_row = row0 + i;
            const direct = if (triangle == .upper) global_row <= global_col else global_row >= global_col;
            var value = if (direct)
                source[sourceIndex(ld, global_row, global_col)]
            else
                source[sourceIndex(ld, global_col, global_row)];
            if (hermitian and !direct) value = conjugate(T, value);
            if (hermitian and global_row == global_col) value = realDiagonal(T, value);
            buffer[i + j * rows] = value;
        }
    }
}

/// Packs a block of op(A) for triangular A. Values outside the effective
/// triangle are zero and unit diagonal entries are synthesized as one.
pub fn packTriangularOpBlock(comptime T: type, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, source: [*]const T, ld: BlasInt, row0: usize, col0: usize, rows: usize, cols: usize, buffer: []T) void {
    std.debug.assert(buffer.len >= rows * cols);
    for (0..cols) |j| {
        const op_col = col0 + j;
        for (0..rows) |i| {
            const op_row = row0 + i;
            const source_row = if (transpose == .no_trans) op_row else op_col;
            const source_col = if (transpose == .no_trans) op_col else op_row;
            const stored = if (triangle == .upper) source_row <= source_col else source_row >= source_col;
            var value = if (stored) source[sourceIndex(ld, source_row, source_col)] else zero(T);
            if (stored and source_row == source_col and diagonal == .unit) value = one(T);
            if (stored and transpose == .conj_trans) value = conjugate(T, value);
            buffer[i + j * rows] = value;
        }
    }
}

pub fn effectiveTriangle(triangle: Triangle, transpose: Transpose) Triangle {
    if (transpose == .no_trans) return triangle;
    return if (triangle == .upper) .lower else .upper;
}

test "structured block packers ignore poisoned unstored values" {
    const C = types.ComplexF32;
    const source = [_]C{
        .{ .re = 1, .im = 9 },   .{ .re = 2, .im = 3 },   .{ .re = 5, .im = 6 },
        .{ .re = 99, .im = 99 }, .{ .re = 4, .im = 8 },   .{ .re = 7, .im = 8 },
        .{ .re = 99, .im = 99 }, .{ .re = 99, .im = 99 }, .{ .re = 9, .im = 7 },
    };
    var buffer: [9]C = undefined;
    packSymmetricBlock(C, .lower, true, &source, 3, 0, 0, 3, 3, &buffer);
    try std.testing.expectEqual(C{ .re = 1, .im = 0 }, buffer[0]);
    try std.testing.expectEqual(C{ .re = 2, .im = -3 }, buffer[3]);
    try std.testing.expectEqual(C{ .re = 5, .im = -6 }, buffer[6]);
    try std.testing.expectEqual(C{ .re = 4, .im = 0 }, buffer[4]);

    packTriangularOpBlock(C, .lower, .conj_trans, .unit, &source, 3, 0, 0, 3, 3, &buffer);
    try std.testing.expectEqual(C{ .re = 1, .im = 0 }, buffer[0]);
    try std.testing.expectEqual(C{ .re = 2, .im = -3 }, buffer[3]);
    try std.testing.expectEqual(C{ .re = 0, .im = 0 }, buffer[1]);
}
