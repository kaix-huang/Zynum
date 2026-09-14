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

/// Packs one logical general op(A) rectangle. This works for real and
/// complex scalars; conjugation is performed exactly once after indexing.
pub fn packGeneralOpBlock(comptime T: type, transpose: Transpose, source: [*]const T, ld: BlasInt, row0: usize, col0: usize, rows: usize, cols: usize, buffer: []T) void {
    std.debug.assert(buffer.len >= rows * cols);
    if (rows == 0 or cols == 0) return;
    if (transpose == .no_trans) {
        // All current callers allocate operand panels separately from input.
        // Contiguous columns need no per-element indexing or conjugation.
        for (0..cols) |j| {
            const source_start = sourceIndex(ld, row0, col0 + j);
            @memcpy(buffer[j * rows .. (j + 1) * rows], source[source_start .. source_start + rows]);
        }
        return;
    }
    for (0..cols) |j| {
        for (0..rows) |i| {
            const row = row0 + i;
            const col = col0 + j;
            const item = source[sourceIndex(ld, col, row)];
            buffer[i + j * rows] = if (transpose == .conj_trans) conjugate(T, item) else item;
        }
    }
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
            // Unit diagonals are synthesized before touching source storage.
            var value = if (!stored)
                zero(T)
            else if (source_row == source_col and diagonal == .unit)
                one(T)
            else
                source[sourceIndex(ld, source_row, source_col)];
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
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const ld: usize = 72;
        var input: [ld * ld]T = undefined;
        for (&input, 0..) |*entry, index| {
            if (comptime T == types.ComplexF32 or T == types.ComplexF64) {
                entry.* = .{ .re = @floatFromInt(index), .im = -@as(if (T == types.ComplexF32) f32 else f64, @floatFromInt(index + 1)) };
            } else entry.* = @floatFromInt(index);
        }
        const sentinel = one(T);
        var output: [65 * 65 + 2]T = undefined;
        const shapes = [_][2]usize{ .{ 0, 65 }, .{ 65, 0 }, .{ 1, 1 }, .{ 15, 17 }, .{ 17, 15 }, .{ 63, 65 }, .{ 65, 63 }, .{ 64, 64 }, .{ 65, 65 } };
        for ([_]Transpose{ .no_trans, .trans, .conj_trans }) |transpose| {
            for (shapes) |shape| {
                @memset(&output, sentinel);
                const rows = shape[0];
                const cols = shape[1];
                const count = rows * cols;
                packGeneralOpBlock(T, transpose, &input, ld, 2, 3, rows, cols, output[1 .. 1 + count]);
                for (0..cols) |j| {
                    for (0..rows) |i| {
                        const row = 2 + i;
                        const col = 3 + j;
                        const item = input[if (transpose == .no_trans) row + col * ld else col + row * ld];
                        try std.testing.expectEqual(if (transpose == .conj_trans) conjugate(T, item) else item, output[1 + i + j * rows]);
                    }
                }
                try std.testing.expectEqual(sentinel, output[0]);
                for (output[1 + count ..]) |item| try std.testing.expectEqual(sentinel, item);
            }
        }
    }
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
