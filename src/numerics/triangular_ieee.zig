// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Native-width triangular substitution over the existing borrowed dense views.
//! Frontends validate geometry, capacities, addresses, aliasing and scalar
//! arguments. This module owns no Context, allocator, thread, scratch or handle.
//! Stored-precision arithmetic and zero skips retain the existing LU IEEE
//! traversal. Unit diagonals are never read. No vendor bit-equivalence is claimed.
const dense = @import("dense.zig");
const arithmetic = @import("complex.zig");
const scalar = arithmetic.scalar;
pub const Triangle = scalar.Triangle;
pub const Diagonal = scalar.Diagonal;
pub const Operation = enum { normal, transposed, adjoint };

fn quotient(comptime T: type, numerator: T, denominator: T) T {
    if (comptime scalar.isComplex(T)) return arithmetic.divideIeee(T, numerator, denominator);
    return numerator / denominator;
}

fn transformed(comptime T: type, value: T, operation: Operation) T {
    return if (operation == .adjoint) scalar.conj(T, value) else value;
}

fn lower(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), operation: Operation, comptime diagonal: Diagonal) void {
    const n = factor.row_count;
    const lda = factor.leading_dimension;
    for (0..rhs.column_count) |column| {
        const x = rhs.values[column * rhs.leading_dimension ..][0..n];
        if (operation == .normal) {
            for (0..n) |i| {
                if (scalar.isZero(T, x[i])) continue;
                if (diagonal == .non_unit) x[i] = quotient(T, x[i], factor.values[i + i * lda]);
                for (i + 1..n) |k| x[k] = scalar.sub(T, x[k], scalar.mul(T, x[i], factor.values[k + i * lda]));
            }
        } else {
            var remaining = n;
            while (remaining != 0) {
                remaining -= 1;
                const i = remaining;
                var value = scalar.mul(T, scalar.one(T), x[i]);
                for (i + 1..n) |k| value = scalar.sub(T, value, scalar.mul(T, transformed(T, factor.values[k + i * lda], operation), x[k]));
                x[i] = if (diagonal == .unit) value else quotient(T, value, transformed(T, factor.values[i + i * lda], operation));
            }
        }
    }
}

fn upper(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), operation: Operation, comptime diagonal: Diagonal) void {
    const n = factor.row_count;
    const lda = factor.leading_dimension;
    for (0..rhs.column_count) |column| {
        const x = rhs.values[column * rhs.leading_dimension ..][0..n];
        if (operation == .normal) {
            var remaining = n;
            while (remaining != 0) {
                remaining -= 1;
                const i = remaining;
                if (scalar.isZero(T, x[i])) continue;
                if (diagonal == .non_unit) x[i] = quotient(T, x[i], factor.values[i + i * lda]);
                for (0..i) |k| x[k] = scalar.sub(T, x[k], scalar.mul(T, x[i], factor.values[k + i * lda]));
            }
        } else {
            for (0..n) |i| {
                var value = scalar.mul(T, scalar.one(T), x[i]);
                for (0..i) |k| value = scalar.sub(T, value, scalar.mul(T, transformed(T, factor.values[k + i * lda], operation), x[k]));
                x[i] = if (diagonal == .unit) value else quotient(T, value, transformed(T, factor.values[i + i * lda], operation));
            }
        }
    }
}

/// Unchecked numerical solve, including explicit IEEE exceptional arithmetic.
/// No singularity/finite scan is performed. LU uses unit L and non-unit U over
/// its one combined factor; those legs must not read the other triangle.
pub fn solve(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), triangle: Triangle, operation: Operation, diagonal: Diagonal) void {
    if (comptime T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("triangular solve requires a supported real or complex scalar");
    if (factor.row_count == 0 or rhs.column_count == 0) return;
    switch (triangle) {
        .lower => switch (diagonal) {
            .unit => lower(T, factor, rhs, operation, .unit),
            .non_unit => lower(T, factor, rhs, operation, .non_unit),
        },
        .upper => switch (diagonal) {
            .unit => upper(T, factor, rhs, operation, .unit),
            .non_unit => upper(T, factor, rhs, operation, .non_unit),
        },
    }
}

/// Exact-zero preflight for a triangular driver. This reads only a non-unit
/// diagonal, including when nrhs is zero; unit or n=0 does not read storage.
/// Positive result is the first one-based zero diagonal. NaN/Inf are not zero.
pub fn firstZeroDiagonal(comptime T: type, factor: dense.ConstMatrix(T), diagonal: Diagonal) usize {
    if (diagonal == .unit) return 0;
    for (0..factor.row_count) |i| {
        if (scalar.isZero(T, factor.values[i + i * factor.leading_dimension])) return i + 1;
    }
    return 0;
}

/// Frontends own argument checks; exact singularity rejects before any RHS
/// store. No positive INFO is introduced into the unchecked LU solve above.
pub fn solveNonsingular(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), triangle: Triangle, operation: Operation, diagonal: Diagonal) usize {
    const info = firstZeroDiagonal(T, factor, diagonal);
    if (info != 0) return info;
    solve(T, factor, rhs, triangle, operation, diagonal);
    return 0;
}
