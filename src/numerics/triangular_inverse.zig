// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Triangular inverse over the existing native-width borrowed dense views.
//! Frontends own square shape, stride, extent, address and alias validation.
//! The workspace is borrowed, uses the existing arithmetic's wide precision,
//! and is never retained. No Context, allocator, thread or handle is owned.
//! Only the selected triangle is used; unit diagonals are not read or written.
//! A whole inverse column is evaluated before narrowing to the storage scalar,
//! keeping earlier rounded output entries out of subsequent computations.
const std = @import("std");
const dense = @import("dense.zig");
const arithmetic = @import("complex.zig");
const triangular = @import("triangular_ieee.zig");
const scalar = arithmetic.scalar;
pub const Triangle = triangular.Triangle;
pub const Diagonal = triangular.Diagonal;
pub const Error = error{WorkspaceTooSmall};
pub const WideError = Error || error{SizeOverflow};

pub fn WorkValue(comptime T: type) type {
    if (comptime scalar.isComplex(T)) return arithmetic.WideComplex(T);
    if (T == f32) return f64;
    if (T == f64) return f128;
    @compileError("triangular inverse requires a supported real or complex scalar");
}

pub fn workspaceLength(n: usize) usize {
    return if (n <= 1) 0 else n;
}

fn widen(comptime T: type, value: T) WorkValue(T) {
    return if (comptime scalar.isComplex(T)) arithmetic.widen(T, value) else value;
}

fn narrow(comptime T: type, value: WorkValue(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else @floatCast(value);
}

fn quotient(comptime T: type, value: WorkValue(T), denominator: T) WorkValue(T) {
    if (comptime scalar.isComplex(T)) {
        if (arithmetic.finite(T, denominator)) return arithmetic.divide(T, value, denominator);
        // Exceptional denominators use the same closed stored-complex IEEE
        // quotient convention as the other ordinary numerical kernels.
        return arithmetic.widen(T, arithmetic.divideIeee(T, narrow(T, value), denominator));
    }
    return value / @as(WorkValue(T), denominator);
}

fn subtractProduct(comptime T: type, value: *WorkValue(T), a: T, b: WorkValue(T)) void {
    if (comptime scalar.isComplex(T)) {
        const aw = arithmetic.widen(T, a);
        value.re -= aw.re * b.re - aw.im * b.im;
        value.im -= aw.re * b.im + aw.im * b.re;
    } else {
        value.* -= @as(WorkValue(T), a) * b;
    }
}

fn inverseColumn(comptime T: type, matrix: dense.ConstMatrix(T), triangle: Triangle, comptime diagonal: Diagonal, column: usize, work: []WorkValue(T)) void {
    const a = matrix.values;
    const ld = matrix.leading_dimension;
    const base = column * ld;
    work[column] = if (diagonal == .unit) widen(T, scalar.one(T)) else quotient(T, widen(T, scalar.one(T)), a[column + base]);
    if (triangle == .upper) {
        var rows = column;
        while (rows != 0) {
            rows -= 1;
            const row = rows;
            var value = widen(T, scalar.zero(T));
            subtractProduct(T, &value, a[row + base], work[column]);
            for (row + 1..column) |k| subtractProduct(T, &value, a[row + k * ld], work[k]);
            work[row] = if (diagonal == .unit) value else quotient(T, value, a[row + row * ld]);
        }
    } else {
        for (column + 1..matrix.row_count) |row| {
            var value = widen(T, scalar.zero(T));
            subtractProduct(T, &value, a[row + base], work[column]);
            for (column + 1..row) |k| subtractProduct(T, &value, a[row + k * ld], work[k]);
            work[row] = if (diagonal == .unit) value else quotient(T, value, a[row + row * ld]);
        }
    }
}

fn upper(comptime T: type, matrix: dense.Matrix(T), comptime diagonal: Diagonal, work: []WorkValue(T)) void {
    const a = matrix.values;
    var columns = matrix.row_count;
    while (columns != 0) {
        columns -= 1;
        const column = columns;
        const base = column * matrix.leading_dimension;
        inverseColumn(T, matrix.asConst(), .upper, diagonal, column, work);
        // Columns to the left retain their original coefficients. This
        // column is narrowed only after its full-precision solve is complete.
        const rows_to_store = if (diagonal == .unit) column else column + 1;
        for (0..rows_to_store) |row| a[row + base] = narrow(T, work[row]);
    }
}

fn lower(comptime T: type, matrix: dense.Matrix(T), comptime diagonal: Diagonal, work: []WorkValue(T)) void {
    const a = matrix.values;
    const ld = matrix.leading_dimension;
    const n = matrix.row_count;
    for (0..n) |column| {
        const base = column * ld;
        inverseColumn(T, matrix.asConst(), .lower, diagonal, column, work);
        // Columns to the right are still the original triangle; no narrowed
        // earlier inverse column participates in this independent solve.
        const first = if (diagonal == .unit) column + 1 else column;
        for (first..n) |row| a[row + base] = narrow(T, work[row]);
    }
}

/// Full column-major wide result, for consumers that must multiply inverse
/// factors before narrowing. This uses the same column solve as in-place
/// inversion. It never mutates the borrowed factor and stores explicit zeros
/// outside its selected triangle. Geometry, aliasing and singularity preflight
/// remain caller contracts. Size/short-work errors precede all stores/reads.
pub fn wideWorkspaceLength(n: usize) WideError!usize {
    return std.math.mul(usize, n, n) catch error.SizeOverflow;
}

pub fn invertToWideNonsingular(comptime T: type, matrix: dense.ConstMatrix(T), triangle: Triangle, diagonal: Diagonal, workspace: []WorkValue(T)) WideError!void {
    const n = matrix.row_count;
    const required = try wideWorkspaceLength(n);
    if (workspace.len < required) return error.WorkspaceTooSmall;
    if (n == 0) return;
    const output = workspace[0..required];
    @memset(output, widen(T, scalar.zero(T)));
    for (0..n) |column| {
        const work = output[column * n ..][0..n];
        switch (diagonal) {
            .unit => inverseColumn(T, matrix, triangle, .unit, column, work),
            .non_unit => inverseColumn(T, matrix, triangle, .non_unit, column, work),
        }
    }
}

/// Borrowed-workspace numerical leg, after non-unit exact-zero preflight.
/// A short workspace rejects before any matrix or workspace store. Singleton
/// inversion needs no workspace; a unit singleton needs no operand storage.
pub fn invertNonsingular(comptime T: type, matrix: dense.Matrix(T), triangle: Triangle, diagonal: Diagonal, workspace: []WorkValue(T)) Error!void {
    const n = matrix.row_count;
    if (n == 0 or (n == 1 and diagonal == .unit)) return;
    if (n == 1) {
        matrix.values[0] = narrow(T, quotient(T, widen(T, scalar.one(T)), matrix.values[0]));
        return;
    }
    if (workspace.len < workspaceLength(n)) return error.WorkspaceTooSmall;
    switch (triangle) {
        .upper => switch (diagonal) {
            .unit => upper(T, matrix, .unit, workspace),
            .non_unit => upper(T, matrix, .non_unit, workspace),
        },
        .lower => switch (diagonal) {
            .unit => lower(T, matrix, .unit, workspace),
            .non_unit => lower(T, matrix, .non_unit, workspace),
        },
    }
}

/// Return the first one-based exact-zero diagonal before any matrix store.
/// Unit diagonals bypass that scan. The input and borrowed workspace must be
/// disjoint; geometry and addresses remain an admitted frontend contract.
pub fn invert(comptime T: type, matrix: dense.Matrix(T), triangle: Triangle, diagonal: Diagonal, workspace: []WorkValue(T)) Error!usize {
    const info = triangular.firstZeroDiagonal(T, matrix.asConst(), diagonal);
    if (info != 0) return info;
    try invertNonsingular(T, matrix, triangle, diagonal, workspace);
    return 0;
}
