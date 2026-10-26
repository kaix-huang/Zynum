// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const storage = @import("../numerics/storage.zig");
pub const views = @import("../blas/api/views.zig");
pub const scalar = @import("../blas/core/shared/scalar.zig");
const real_validation = @import("validation.zig");
pub const Error = real_validation.Error;
pub const Triangle = real_validation.Triangle;
pub const noOverlap = real_validation.noOverlap;
pub const triangleIndex = real_validation.triangleIndex;

const arithmetic = @import("../numerics/complex.zig");
pub const expectComplex = arithmetic.expectComplex;
pub const Wide = arithmetic.Wide;
pub const WideComplex = arithmetic.WideComplex;
pub const widen = arithmetic.widen;
pub const narrow = arithmetic.narrow;
pub const finite = arithmetic.finite;
pub const abs1 = arithmetic.abs1;
pub const subtractProduct = arithmetic.subtractProduct;
pub const divide = arithmetic.divide;

pub fn matrixSpan(comptime C: type, a: anytype) Error!usize {
    comptime expectComplex(C);
    if (a.row_count < 0 or a.column_count < 0) return error.InvalidLength;
    if (a.leading_dimension < 0) return error.InvalidLeadingDimension;
    if (a.row_count == 0 or a.column_count == 0) return 0;
    if (a.leading_dimension < a.row_count) return error.InvalidLeadingDimension;
    const span = storage.columnMajorExtent(@intCast(a.row_count), @intCast(a.column_count), @intCast(a.leading_dimension)) catch return error.InvalidLength;
    if (a.values.len < span) return error.BufferTooSmall;
    return span;
}
pub fn squareFactor(comptime C: type, a: views.ConstMatrix(C)) Error!usize {
    const span = try matrixSpan(C, a);
    if (a.operation != .normal) return error.InvalidMatrixOperation;
    if (a.row_count != a.column_count) return error.DimensionMismatch;
    return span;
}
pub fn finiteMatrix(comptime C: type, a: anytype) Error!void {
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    if (m == 0 or n == 0) return;
    const ld: usize = @intCast(a.leading_dimension);
    for (0..n) |j| for (0..m) |i| if (!finite(C, a.values[i + j * ld])) {
        return error.NonFiniteInput;
    };
}
pub fn hermitianTriangle(comptime C: type, a: anytype, triangle: Triangle) Error!void {
    const n: usize = @intCast(a.row_count);
    if (n == 0) return;
    const ld: usize = @intCast(a.leading_dimension);
    for (0..n) |j| {
        const start: usize = if (triangle == .lower) j else 0;
        const end: usize = if (triangle == .lower) n else j + 1;
        for (start..end) |i| if (!finite(C, a.values[i + j * ld])) {
            return error.NonFiniteInput;
        };
    }
    // A deliberate checked-API policy: an input Hermitian/factor diagonal must
    // be explicitly real. Signed imaginary zero is accepted, nonzero is not.
    for (0..n) |j| if (a.values[j + j * ld].im != 0) return error.InvalidHermitianDiagonal;
}
pub fn lower(comptime C: type, a: anytype, i: usize, j: usize, ld: usize, triangle: Triangle) C {
    const z = a.values[triangleIndex(i, j, ld, triangle)];
    return if (triangle == .upper) scalar.conj(C, z) else z;
}
pub fn storeLower(comptime C: type, a: views.Matrix(C), i: usize, j: usize, ld: usize, triangle: Triangle, z: C) void {
    a.values[triangleIndex(i, j, ld, triangle)] = if (triangle == .upper) scalar.conj(C, z) else z;
}
