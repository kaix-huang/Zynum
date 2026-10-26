// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Inverse from an existing square combined unit-L/non-unit-U factor.
//! Operands use the existing native-width borrowed Dense descriptor. Geometry,
//! addresses, aliasing and pivot-prefix validation belong to the frontend.
//! A reader returns validated zero-based rows without allocation or failure.
//! Only its first n-1 entries are consumed, once each in descending order.
//! The borrowed output workspace retains all wide columns until publication;
//! the original LU factor is never overwritten while any solve still needs it.
//! There is no Context, allocator, thread, pool, external backend or handle here.
const std = @import("std");
const dense = @import("dense.zig");
const arithmetic = @import("complex.zig");
const triangular = @import("triangular_ieee.zig");
const triangular_inverse = @import("triangular_inverse.zig");
const scalar = arithmetic.scalar;
pub const Error = error{ WorkspaceTooSmall, SizeOverflow };
pub const WorkValue = triangular_inverse.WorkValue;

/// Internal element count, distinct from any public GETRI WORK/lwork recipe.
/// The eventual resource adapter must also validate its typed byte extent.
pub fn workspaceLength(n: usize) Error!usize {
    if (n <= 1) return 0;
    return std.math.mul(usize, n, n) catch error.SizeOverflow;
}

fn widen(comptime T: type, value: T) WorkValue(T) {
    return if (comptime scalar.isComplex(T)) arithmetic.widen(T, value) else value;
}

fn narrow(comptime T: type, value: WorkValue(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else @floatCast(value);
}

fn subtractProduct(comptime T: type, value: *WorkValue(T), factor: T, coefficient: WorkValue(T)) void {
    if (comptime scalar.isComplex(T)) {
        const a = arithmetic.widen(T, factor);
        value.re -= a.re * coefficient.re - a.im * coefficient.im;
        value.im -= a.re * coefficient.im + a.im * coefficient.re;
    } else {
        value.* -= @as(WorkValue(T), factor) * coefficient;
    }
}

fn quotient(comptime T: type, numerator: WorkValue(T), denominator: T) WorkValue(T) {
    if (comptime scalar.isComplex(T)) {
        if (arithmetic.finite(T, denominator)) return arithmetic.divide(T, numerator, denominator);
        // Preserve the existing closed stored-complex exceptional convention.
        // Finite denominators keep the full wide intermediate, without a cast.
        return arithmetic.widen(T, arithmetic.divideIeee(T, narrow(T, numerator), denominator));
    }
    return numerator / @as(WorkValue(T), denominator);
}

/// Borrowed numerical leg after an exact-zero preflight. A short/overflowing
/// workspace rejects before any matrix/workspace store or pivot read.
/// Empty and singleton inverses use no workspace and consume no pivot entry.
pub fn invertNonsingular(comptime T: type, matrix: dense.Matrix(T), reader: anytype, workspace: []WorkValue(T)) Error!void {
    const n = matrix.row_count;
    if (n == 0) return;
    if (n == 1) {
        matrix.values[0] = narrow(T, quotient(T, widen(T, scalar.one(T)), matrix.values[0]));
        return;
    }
    const required = try workspaceLength(n);
    if (workspace.len < required) return error.WorkspaceTooSmall;
    const output = workspace[0..required];
    const a = matrix.values;
    const lda = matrix.leading_dimension;
    for (0..n) |column| {
        const x = output[column * n ..][0..n];
        // Compute L^-1 e_j and then U^-1 L^-1 e_j entirely in the shared wide
        // work type. The unit L diagonal is implicit; U owns the stored one.
        for (0..n) |row| {
            var value = widen(T, if (row == column) scalar.one(T) else scalar.zero(T));
            for (0..row) |k| subtractProduct(T, &value, a[row + k * lda], x[k]);
            x[row] = value;
        }
        var remaining = n;
        while (remaining != 0) {
            remaining -= 1;
            const row = remaining;
            var value = x[row];
            for (row + 1..n) |k| subtractProduct(T, &value, a[row + k * lda], x[k]);
            x[row] = quotient(T, value, a[row + row * lda]);
        }
    }
    // LU = P A, so A^-1 = U^-1 L^-1 P. Right multiplication applies the
    // original row transpositions as column swaps in reverse order.
    var remaining = n - 1;
    while (remaining != 0) {
        remaining -= 1;
        const pivot: usize = reader.readUnchecked(remaining);
        if (pivot != remaining) for (0..n) |row| {
            std.mem.swap(WorkValue(T), &output[row + remaining * n], &output[row + pivot * n]);
        };
    }
    for (0..n) |column| for (0..n) |row| {
        a[row + column * lda] = narrow(T, output[row + column * n]);
    };
}

/// First one-based exact-zero U diagonal wins before matrix/workspace stores
/// and before any pivot read. NaN/Inf do not introduce a singular INFO.
pub fn invert(comptime T: type, matrix: dense.Matrix(T), reader: anytype, workspace: []WorkValue(T)) Error!usize {
    const info = triangular.firstZeroDiagonal(T, matrix.asConst(), .non_unit);
    if (info != 0) return info;
    try invertNonsingular(T, matrix, reader, workspace);
    return 0;
}
