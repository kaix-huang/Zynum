// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Canonical internal contracts, separate from the public API test modules.
comptime {
    // The direct numerical test root owns these sources in this module.
    _ = @import("blas/runtime.zig");
    _ = @import("numerics/residual_filter_test.zig");
    _ = @import("numerics/pattern_operator_test.zig");
    _ = @import("interop/feast_test.zig");
    _ = @import("numerics/contour_eigensolver_test.zig");
    _ = @import("numerics/rank_reduced_ritz_test.zig");
    _ = @import("numerics/contour_quadrature_test.zig");
    _ = @import("numerics/rational_filter_test.zig");
    _ = @import("numerics/bicgstab_test.zig");
    _ = @import("numerics/subspace_projection_test.zig");
    _ = @import("numerics/series_set_test.zig");
    _ = @import("numerics/normalized_chebyshev_test.zig");
    _ = @import("fft/core_test.zig");
    _ = @import("numerics/dense.zig");
    _ = @import("numerics/triangular_ieee_test.zig");
    _ = @import("numerics/triangular_inverse_test.zig");
    _ = @import("numerics/triangular_inverse_range_test.zig");
    _ = @import("numerics/lu_inverse_test.zig");
    _ = @import("numerics/lu_inverse_range_test.zig");
    _ = @import("numerics/cholesky_inverse_test.zig");
    _ = @import("numerics/cholesky_inverse_range_test.zig");
    _ = @import("lapack/core/cholesky.zig");
    _ = @import("sparse/pattern.zig");
}

// Core and checked-boundary regressions share the internal contract test root.
const std = @import("std");
const dense = @import("numerics/dense.zig");
const arithmetic = @import("numerics/complex.zig");
const complex_core = @import("lapack/core/complex_cholesky.zig");
const complex_frontend = @import("lapack/complex_cholesky.zig");
const complex_support = @import("lapack/complex_support.zig");

fn complexOperand(comptime C: type, values: []C, rows: usize, columns: usize, stride: usize) dense.Matrix(C) {
    return .{ .values = values, .row_count = rows, .column_count = columns, .leading_dimension = stride };
}
fn checkedComplexOperand(comptime C: type, values: []C, rows: i32, columns: i32, stride: i32) complex_support.views.Matrix(C) {
    return .{ .values = values, .row_count = rows, .column_count = columns, .leading_dimension = stride };
}
fn expectComplexBytes(comptime C: type, expected: []const C, actual: []const C) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}
fn storedPosition(row: usize, column: usize, stride: usize, triangle: complex_core.Triangle) usize {
    return if (triangle == .lower) row + column * stride else column + row * stride;
}
fn asStored(comptime C: type, z: C, triangle: complex_core.Triangle) C {
    return if (triangle == .upper) arithmetic.scalar.conj(C, z) else z;
}

test "complex Cholesky core computes actual one-element factors and solves with native-width strides" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        inline for (.{ complex_core.Triangle.lower, complex_core.Triangle.upper }) |triangle| {
            for ([_]usize{ @as(usize, std.math.maxInt(i32)) + 1, std.math.maxInt(usize) }) |stride| {
                var a = [_]C{.{ .re = 4, .im = -0.0 }};
                var b = [_]C{.{ .re = 12, .im = -8 }};
                const factor = complexOperand(C, &a, 1, 1, stride);
                const rhs = complexOperand(C, &b, 1, 1, stride);
                try std.testing.expectEqual(@as(usize, 1), try factor.check());
                try std.testing.expectEqual(@as(usize, 1), try rhs.check());
                try std.testing.expectEqual(@as(usize, 0), complex_core.factorize(C, factor, triangle));
                try expectComplexBytes(C, &.{.{ .re = 2, .im = 0 }}, &a);
                try std.testing.expectEqual(@as(usize, 0), complex_core.solve(C, factor.asConst(), rhs, triangle));
                try expectComplexBytes(C, &.{.{ .re = 3, .im = -2 }}, &b);
            }
            const empty = complexOperand(C, &.{}, 0, 0, 0);
            const many_rhs = complexOperand(C, &.{}, 0, @as(usize, std.math.maxInt(i32)) + 1, std.math.maxInt(usize));
            try std.testing.expectEqual(@as(usize, 0), try many_rhs.check());
            try std.testing.expectEqual(@as(usize, 0), complex_core.factorize(C, empty, triangle));
            try std.testing.expectEqual(@as(usize, 0), complex_core.solve(C, empty.asConst(), many_rhs, triangle));
        }
    }
}

test "complex checked factor preserves geometry finite and real-diagonal preflight precedence" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        var a = [_]C{ .{ .re = 4, .im = 1 }, .{ .re = std.math.nan(R), .im = 0 }, .{ .re = 7, .im = 8 }, .{ .re = 2, .im = 0 } };
        const before = a;
        var operand = checkedComplexOperand(C, &a, -1, 2, -1);
        try std.testing.expectError(error.InvalidLength, complex_frontend.factorize(C, operand, .lower));
        operand.row_count = 2;
        try std.testing.expectError(error.InvalidLeadingDimension, complex_frontend.factorize(C, operand, .lower));
        operand.leading_dimension = 2;
        operand.values = a[0..3];
        try std.testing.expectError(error.BufferTooSmall, complex_frontend.factorize(C, operand, .lower));
        operand.values = &a;
        operand.column_count = 1;
        try std.testing.expectError(error.DimensionMismatch, complex_frontend.factorize(C, operand, .lower));
        operand.column_count = 2;
        // The selected NaN is rejected before the earlier non-real diagonal.
        try std.testing.expectError(error.NonFiniteInput, complex_frontend.factorize(C, operand, .lower));
        try expectComplexBytes(C, &before, &a);
        a[1] = .{ .re = 1, .im = 0 };
        const finite_before = a;
        try std.testing.expectError(error.InvalidHermitianDiagonal, complex_frontend.factorize(C, operand, .lower));
        try expectComplexBytes(C, &finite_before, &a);
    }
}

test "complex checked solve preserves adapter preflight and zero-RHS positivity policy" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        var a = [_]C{ .{ .re = 2, .im = 1 }, .{ .re = 0, .im = 0 }, .{ .re = 7, .im = 8 }, .{ .re = 0, .im = 0 } };
        var b = [_]C{ .{ .re = std.math.nan(R), .im = 0 }, .{ .re = 3, .im = 0 } };
        const a_before = a;
        const b_before = b;
        var factor = checkedComplexOperand(C, &a, 2, 2, 2).asConst();
        var rhs = checkedComplexOperand(C, &b, 2, 1, 2);
        factor.operation = .adjoint;
        rhs.values = b[0..0];
        try std.testing.expectError(error.InvalidMatrixOperation, complex_frontend.solve(C, factor, rhs, .lower));
        factor.operation = .normal;
        rhs.row_count = 1;
        try std.testing.expectError(error.BufferTooSmall, complex_frontend.solve(C, factor, rhs, .lower));
        rhs.values = &b;
        try std.testing.expectError(error.DimensionMismatch, complex_frontend.solve(C, factor, rhs, .lower));
        rhs.row_count = 2;
        rhs.values = a[0..2];
        try std.testing.expectError(error.AliasingNotAllowed, complex_frontend.solve(C, factor, rhs, .lower));
        rhs.values = &b;
        // Factor diagonal form precedes RHS finiteness, which precedes positivity.
        try std.testing.expectError(error.InvalidHermitianDiagonal, complex_frontend.solve(C, factor, rhs, .lower));
        try expectComplexBytes(C, &a_before, &a);
        try expectComplexBytes(C, &b_before, &b);
        a[0].im = 0;
        try std.testing.expectError(error.NonFiniteInput, complex_frontend.solve(C, factor, rhs, .lower));
        try expectComplexBytes(C, &b_before, &b);
        b[0].re = 5;
        const finite_before = b;
        try std.testing.expectEqual(@as(usize, 2), try complex_frontend.solve(C, factor, rhs, .lower));
        try expectComplexBytes(C, &finite_before, &b);
        const no_rhs = checkedComplexOperand(C, &.{}, 2, 0, 0);
        try std.testing.expectEqual(@as(usize, 0), try complex_frontend.solve(C, factor, no_rhs, .lower));
        a[0].im = 1;
        try std.testing.expectError(error.InvalidHermitianDiagonal, complex_frontend.solve(C, factor, no_rhs, .lower));
    }
}

test "complex Cholesky core and checked factor retain exactly the successful prefix before narrowing failure" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, complex_core.Triangle.upper }) |triangle| {
            const checked_triangle: complex_support.Triangle = if (triangle == .lower) .lower else .upper;
            var a: [12]C = @splat(.{ .re = 71, .im = 19 });
            a[0] = .{ .re = 0.25, .im = -0.0 };
            a[5] = .{ .re = 4, .im = 0 };
            a[10] = .{ .re = 4, .im = 0 };
            a[storedPosition(1, 0, 4, triangle)] = asStored(C, .{ .re = 1, .im = 1 }, triangle);
            a[storedPosition(2, 0, 4, triangle)] = asStored(C, .{ .re = std.math.floatMax(R), .im = 0 }, triangle);
            var checked = a;
            var expected = a;
            expected[0] = .{ .re = 0.5, .im = 0 };
            expected[storedPosition(1, 0, 4, triangle)] = asStored(C, .{ .re = 2, .im = 2 }, triangle);
            try std.testing.expectEqual(@as(usize, 1), complex_core.factorize(C, complexOperand(C, &a, 3, 3, 4), triangle));
            try std.testing.expectEqual(@as(usize, 1), try complex_frontend.factorize(C, checkedComplexOperand(C, &checked, 3, 3, 4), checked_triangle));
            try expectComplexBytes(C, &expected, &a);
            try expectComplexBytes(C, &expected, &checked);
        }
    }
}

test "complex Cholesky core substitution retains earlier columns and failing-assignment prefix" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, complex_core.Triangle.upper }) |triangle| {
            const factor = [_]C{.{ .re = 0.5, .im = 0 }};
            const f: dense.ConstMatrix(C) = .{ .values = &factor, .row_count = 1, .column_count = 1, .leading_dimension = std.math.maxInt(usize) };
            const large = std.math.floatMax(R) * 0.375;
            var b = [_]C{ .{ .re = 1, .im = 0.5 }, .{ .re = 71, .im = 19 }, .{ .re = large, .im = 0 }, .{ .re = 73, .im = 23 }, .{ .re = 3, .im = -2 }, .{ .re = 79, .im = 29 } };
            var expected = b;
            expected[0] = .{ .re = 4, .im = 2 };
            // The second column's forward assignment succeeds; backward fails.
            expected[2].re = large * 2;
            try std.testing.expectEqual(@as(usize, 1), complex_core.solve(C, f, complexOperand(C, &b, 1, 3, 2), triangle));
            try expectComplexBytes(C, &expected, &b);
            try expectComplexBytes(C, &.{.{ .re = 0.5, .im = 0 }}, &factor);
        }
    }
}

test "complex core numerical failure is distinct from checked full-input rejection" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        var a = [_]C{ .{ .re = 4, .im = 0 }, .{ .re = 2, .im = 2 }, .{ .re = 71, .im = 19 }, .{ .re = std.math.nan(R), .im = 0 } };
        var checked = a;
        const before = a;
        try std.testing.expectError(error.NonFiniteInput, complex_frontend.factorize(C, checkedComplexOperand(C, &checked, 2, 2, 2), .lower));
        try expectComplexBytes(C, &before, &checked);
        // Frontends with different finite-input policy can retain the core's
        // numerical failure status; no strict input scan is hidden in the core.
        try std.testing.expectEqual(@as(usize, 2), complex_core.factorize(C, complexOperand(C, &a, 2, 2, 2), .lower));
        var expected = before;
        expected[0] = .{ .re = 2, .im = 0 };
        expected[1] = .{ .re = 1, .im = 1 };
        try expectComplexBytes(C, &expected, &a);
    }
}

const lu_core = @import("lapack/core/lu.zig");
const pivot_codec = @import("numerics/pivot.zig");

comptime {
    _ = @import("lapack/core/lu_policy_test.zig");
}

fn luValue(comptime T: type, re: arithmetic.scalar.Real(T), im: arithmetic.scalar.Real(T)) T {
    return if (comptime arithmetic.scalar.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn luOperand(comptime T: type, values: []T, rows: usize, columns: usize, stride: usize) dense.Matrix(T) {
    return .{ .values = values, .row_count = rows, .column_count = columns, .leading_dimension = stride };
}

test "LU core computes native-width single-element systems and empty shapes with every pivot encoding" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64, usize }) |P| inline for (.{ pivot_codec.Base.zero, pivot_codec.Base.one }) |base| {
            for ([_]usize{ @as(usize, std.math.maxInt(i32)) + 1, std.math.maxInt(usize) }) |stride| {
                var a = [_]T{luValue(T, 4, 0)};
                var b = [_]T{luValue(T, 12, -8)};
                var pivots = [_]P{ 77, 79 };
                const factor = luOperand(T, &a, 1, 1, stride);
                const rhs = luOperand(T, &b, 1, 1, stride);
                try std.testing.expectEqual(@as(usize, 1), try factor.check());
                try std.testing.expectEqual(@as(usize, 1), try rhs.check());
                try std.testing.expect(pivot_codec.canWriteRows(P, base, 1, 1));
                try std.testing.expectEqual(@as(usize, 0), lu_core.factorize(T, factor, pivot_codec.Writer(P, base){ .values = &pivots }));
                try std.testing.expectEqual(@as(P, if (base == .one) 1 else 0), pivots[0]);
                try std.testing.expectEqual(@as(P, 79), pivots[1]);
                try std.testing.expectEqual(@as(usize, 0), lu_core.solve(T, factor.asConst(), pivot_codec.Reader(P, base){ .values = &pivots }, rhs));
                try expectComplexBytes(T, &.{luValue(T, 3, -2)}, &b);
                try expectComplexBytes(T, &.{luValue(T, 4, 0)}, &a);
            }
            const empty = luOperand(T, &.{}, 0, 0, 0);
            const tall_empty = luOperand(T, &.{}, @as(usize, std.math.maxInt(i32)) + 1, 0, 0);
            const many_rhs = luOperand(T, &.{}, 0, @as(usize, std.math.maxInt(i32)) + 1, std.math.maxInt(usize));
            try std.testing.expectEqual(@as(usize, 0), try tall_empty.check());
            try std.testing.expectEqual(@as(usize, 0), try many_rhs.check());
            try std.testing.expectEqual(@as(usize, 0), lu_core.factorize(T, tall_empty, pivot_codec.Writer(P, base){ .values = &.{} }));
            try std.testing.expectEqual(@as(usize, 0), lu_core.solve(T, empty.asConst(), pivot_codec.Reader(P, base){ .values = &.{} }, many_rhs));
        };
    }
}

test "LU core zero and one based encodings solve the same independent pivoted real and complex systems" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64, usize }) |P| inline for (.{ pivot_codec.Base.zero, pivot_codec.Base.one }) |base| {
            const complex = arithmetic.scalar.isComplex(T);
            var a = [_]T{ luValue(T, 0, 0), luValue(T, 1, -1), luValue(T, 71, 19), luValue(T, 2, 1), luValue(T, 3, 0), luValue(T, 73, 23) };
            var b = [_]T{ luValue(T, if (complex) -4 else -2, 3), luValue(T, if (complex) 0 else -1, 5), luValue(T, 79, 29) };
            var pivots = [_]P{ 77, 77, 77 };
            const factor = luOperand(T, &a, 2, 2, 3);
            try std.testing.expectEqual(@as(usize, 0), lu_core.factorize(T, factor, pivot_codec.Writer(P, base){ .values = &pivots }));
            const first: P = if (base == .one) 2 else 1;
            try std.testing.expectEqualSlices(P, &.{ first, first, 77 }, &pivots);
            try std.testing.expectEqual(@as(usize, 0), lu_core.solve(T, factor.asConst(), pivot_codec.Reader(P, base){ .values = &pivots }, luOperand(T, &b, 2, 1, 3)));
            try expectComplexBytes(T, &.{ luValue(T, 2, 1), luValue(T, -1, 2), luValue(T, 79, 29) }, &b);
            try expectComplexBytes(T, &.{luValue(T, 71, 19)}, a[2..3]);
            try expectComplexBytes(T, &.{luValue(T, 73, 23)}, a[5..6]);
        };
    }
}

test "LU numeric info retains an earlier singularity while both completed pivots are immediately encoded" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        const huge = std.math.floatMax(arithmetic.scalar.Real(T));
        inline for (.{ i32, u32, i64, u64, usize }) |P| inline for (.{ pivot_codec.Base.zero, pivot_codec.Base.one }) |base| {
            var a = [_]T{ luValue(T, 0, 0), luValue(T, 0, 0), luValue(T, 0, 0), luValue(T, 0, 0), luValue(T, 1, 0), luValue(T, 1, 0), luValue(T, 0, 0), luValue(T, huge, huge), luValue(T, -huge, -huge) };
            const expected = a;
            var pivots = [_]P{ 77, 77, 77, 79 };
            try std.testing.expectEqual(@as(usize, 1), lu_core.factorize(T, luOperand(T, &a, 3, 3, 3), pivot_codec.Writer(P, base){ .values = &pivots }));
            const offset: P = if (base == .one) 1 else 0;
            try std.testing.expectEqualSlices(P, &.{ offset, offset + 1, 77, 79 }, &pivots);
            try expectComplexBytes(T, &expected, &a);
        };
    }
}

test "LU core diagonal preflight and arithmetic failure distinguish untouched RHS from completed permutation" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        const R = arithmetic.scalar.Real(T);
        var factor = [_]T{ luValue(T, 1, 0), luValue(T, 0, 0), luValue(T, 0, 0), luValue(T, 0, 0) };
        const pivots = [_]i64{ 2, 2 };
        var b = [_]T{ luValue(T, 2, 0), luValue(T, std.math.floatMax(R), 0), luValue(T, 71, 19), luValue(T, 3, 0), luValue(T, 4, 0), luValue(T, 73, 23) };
        const before = b;
        const f = luOperand(T, &factor, 2, 2, 2);
        const rhs = luOperand(T, &b, 2, 2, 3);
        try std.testing.expectEqual(@as(usize, 2), lu_core.solve(T, f.asConst(), pivot_codec.Reader(i64, .one){ .values = &pivots }, rhs));
        try expectComplexBytes(T, &before, &b);
        factor[3] = luValue(T, std.math.floatTrueMin(R), std.math.floatTrueMin(R));
        const fb = factor;
        try std.testing.expectEqual(@as(usize, 2), lu_core.solve(T, f.asConst(), pivot_codec.Reader(i64, .one){ .values = &pivots }, rhs));
        var expected = before;
        expected[0] = before[1];
        expected[1] = before[0];
        try expectComplexBytes(T, &expected, &b);
        try expectComplexBytes(T, &fb, &factor);
        try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, &pivots);
    }
}

test "LU checked wrappers preserve pivot alias finite and empty RHS preflight ordering" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        const frontend = if (comptime arithmetic.scalar.isComplex(T)) @import("lapack/complex_lu.zig") else @import("lapack/lu.zig");
        const R = arithmetic.scalar.Real(T);
        var a = [_]T{ luValue(T, 1, 0), luValue(T, 0, 0), luValue(T, 0, 0), luValue(T, std.math.nan(R), 0) };
        var b = [_]T{ luValue(T, 2, 0), luValue(T, 3, 0) };
        const ab = a;
        const bb = b;
        var pivots = [_]i32{ 0, 0 };
        var av = checkedComplexOperand(T, &a, 2, 2, 1);
        try std.testing.expectError(error.InvalidLeadingDimension, frontend.factorize(T, av, pivots[0..0]));
        av.leading_dimension = 2;
        try std.testing.expectError(error.BufferTooSmall, frontend.factorize(T, av, pivots[0..1]));
        const embedded: [*]i32 = @ptrCast(&a);
        try std.testing.expectError(error.AliasingNotAllowed, frontend.factorize(T, av, embedded[0..2]));
        try std.testing.expectError(error.NonFiniteInput, frontend.factorize(T, av, &pivots));
        try std.testing.expectEqualSlices(i32, &.{ 0, 0 }, &pivots);
        var cf = av.asConst();
        var bv = checkedComplexOperand(T, &b, 2, 1, -1);
        cf.operation = .adjoint;
        try std.testing.expectError(error.InvalidMatrixOperation, frontend.solve(T, cf, &pivots, bv));
        cf.operation = .normal;
        bv.leading_dimension = 2;
        try std.testing.expectError(error.BufferTooSmall, frontend.solve(T, cf, pivots[0..1], checkedComplexOperand(T, a[0..2], 2, 1, 2)));
        try std.testing.expectError(error.AliasingNotAllowed, frontend.solve(T, cf, &pivots, checkedComplexOperand(T, a[0..2], 2, 1, 2)));
        try std.testing.expectError(error.InvalidPivot, frontend.solve(T, cf, &pivots, bv));
        const empty_rhs = checkedComplexOperand(T, &.{}, 2, 0, 0);
        try std.testing.expectError(error.InvalidPivot, frontend.solve(T, cf, &pivots, empty_rhs));
        pivots[1] = 1;
        try std.testing.expectError(error.NonFiniteInput, frontend.solve(T, cf, &pivots, empty_rhs));
        try expectComplexBytes(T, &ab, &a);
        try expectComplexBytes(T, &bb, &b);
    }
}

const sparse_pattern = @import("sparse/pattern.zig");
const sparse_product = @import("sparse/core/product.zig");
const sparse_scalar = @import("sparse/scalar.zig");
const sparse_checked = @import("sparse.zig");

const ProductReference = struct { re: f64, im: f64 };
fn productReferenceAdd(a: ProductReference, b: ProductReference) ProductReference {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn productReferenceMultiply(a: ProductReference, b: ProductReference) ProductReference {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn productNumber(comptime T: type, re: f64, im: f64) T {
    if (comptime sparse_scalar.isComplex(T)) return .{ .re = @floatCast(re), .im = @floatCast(im) };
    return @floatCast(re);
}
fn productReference(comptime T: type, value: T) ProductReference {
    if (comptime sparse_scalar.isComplex(T)) return .{ .re = value.re, .im = value.im };
    return .{ .re = value, .im = 0 };
}
fn productIndex(comptime I: type, physical: usize, base: sparse_pattern.IndexBase) I {
    return @intCast(physical + @intFromBool(base == .one));
}
fn productBytes(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn checkProductRepresentations(comptime T: type, comptime O: type, comptime I: type) !void {
    // Reference coordinates/physical positions are test-owned constants, never
    // recovered through the implementation's pattern accessors or decoding.
    for ([_]bool{ false, true }) |split| {
        const begins: [3]usize = if (split) .{ 4, 10, 1 } else .{ 0, 3, 3 };
        const ends: [3]usize = if (split) .{ 7, 10, 6 } else .{ 3, 3, 5 };
        const logical: [10]usize = if (split) .{ 99, 2, 0, 2, 1, 3, 1, 99, 99, 99 } else .{ 2, 0, 2, 1, 0, 99, 99, 99, 99, 99 };
        const capacity: usize = if (split) 10 else 5;
        const value_capacity: usize = if (split) 7 else 5;
        var values: [10]T = undefined;
        for (&values, 0..) |*value, at| value.* = productNumber(T, @as(f64, @floatFromInt(at + 1)), @as(f64, @floatFromInt(at % 3)) - 1);
        if (split) values[0] = productNumber(T, std.math.nan(f64), std.math.inf(f64));
        for ([_]sparse_pattern.IndexBase{ .zero, .one }) |base| {
            var starts: [3]O = undefined;
            var finishes: [3]O = undefined;
            var offsets: [4]O = undefined;
            var ordinals: [10]I = undefined;
            for (0..3) |k| {
                starts[k] = productIndex(O, begins[k], base);
                finishes[k] = productIndex(O, ends[k], base);
                offsets[k] = productIndex(O, begins[k], base);
            }
            offsets[3] = productIndex(O, ends[2], base);
            for (&ordinals, logical) |*ordinal, value| ordinal.* = productIndex(I, value, base);
            for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
                const rows: usize = if (axis == .row) 3 else 4;
                const columns: usize = if (axis == .row) 4 else 3;
                var dense_reference: [16]ProductReference = @splat(.{ .re = 0, .im = 0 });
                for (0..3) |major| {
                    for (begins[major]..ends[major]) |physical| {
                        const row = if (axis == .row) major else logical[physical];
                        const column = if (axis == .row) logical[physical] else major;
                        const at = row * 4 + column;
                        dense_reference[at] = productReferenceAdd(dense_reference[at], productReference(T, values[physical]));
                    }
                }
                for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |operation| {
                    const a: sparse_pattern.Operand(T, O, I) = .{
                        .pattern = .{
                            .ranges = if (split) .{ .split = .{ .starts = &starts, .ends = &finishes } } else .{ .contiguous = &offsets },
                            .ordinals = ordinals[0..capacity],
                            .row_count = rows,
                            .column_count = columns,
                            .index_base = base,
                            .major_axis = axis,
                        },
                        .values = values[0..value_capacity],
                        .operation = operation,
                    };
                    const stats = try a.check();
                    try std.testing.expectEqual(@as(usize, if (split) 8 else 5), stats.referenced_entries);
                    try std.testing.expectEqual(value_capacity, stats.storage_extent);
                    const input_length = if (operation == .normal) columns else rows;
                    const output_length = if (operation == .normal) rows else columns;
                    var input: [4]T = undefined;
                    for (&input, 0..) |*v, k| v.* = productNumber(T, @as(f64, @floatFromInt(k + 1)) / 2, -@as(f64, @floatFromInt(k)) / 4);
                    const alpha = productNumber(T, 0.5, -0.25);
                    const beta = productNumber(T, -0.5, 0.25);
                    var output: [6]T = @splat(productNumber(T, 71, -19));
                    const old = output;
                    sparse_product.executeValidated(T, O, I, a, input[0..input_length], output[1..][0..output_length], alpha, beta);
                    for (0..output_length) |row| {
                        var sum: ProductReference = .{ .re = 0, .im = 0 };
                        for (0..input_length) |column| {
                            var entry = if (operation == .normal) dense_reference[row * 4 + column] else dense_reference[column * 4 + row];
                            if (operation == .adjoint) entry.im = -entry.im;
                            sum = productReferenceAdd(sum, productReferenceMultiply(entry, productReference(T, input[column])));
                        }
                        const expected = productReferenceAdd(productReferenceMultiply(productReference(T, alpha), sum), productReferenceMultiply(productReference(T, beta), productReference(T, old[row + 1])));
                        const actual = productReference(T, output[row + 1]);
                        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-5);
                        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-5);
                    }
                    try productBytes(T, old[0..1], output[0..1]);
                    try productBytes(T, old[output_length + 1 ..], output[output_length + 1 ..]);
                }
            }
        }
    }
}

test "compressed product independent dense reference covers mixed widths bases axes transforms and physical gaps" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |O| {
            inline for (.{ i32, u32, i64, u64 }) |I| try checkProductRepresentations(T, O, I);
        }
    }
}

test "compressed product empty ranges beyond values and zero scales do not use numerical inputs" {
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |operation| {
                var a: sparse_pattern.Operand(T, u64, i32) = .{
                    .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 3, 3 }, .ends = &.{ 3, 3 } } }, .ordinals = &.{ -1, -1, -1 }, .row_count = 2, .column_count = 2, .major_axis = axis },
                    .values = &.{},
                    .operation = operation,
                };
                _ = try a.check();
                const poison = productNumber(T, std.math.nan(f64), std.math.nan(f64));
                var output: [2]T = @splat(poison);
                sparse_product.executeValidated(T, u64, i32, a, &.{ poison, poison }, &output, poison, productNumber(T, 0, 0));
                try productBytes(T, &.{ productNumber(T, 0, 0), productNumber(T, 0, 0) }, &output);
                a.pattern.ranges = .{ .split = .{ .starts = &.{ 0, 3 }, .ends = &.{ 1, 3 } } };
                a.pattern.ordinals = &.{ 1, -1, -1 };
                a.values = &.{poison};
                _ = try a.check();
                output = @splat(poison);
                sparse_product.executeValidated(T, u64, i32, a, &.{ poison, poison }, &output, productNumber(T, -0.0, -0.0), productNumber(T, 0, 0));
                try productBytes(T, &.{ productNumber(T, 0, 0), productNumber(T, 0, 0) }, &output);
            }
        }
    }
}

test "compressed product preserves duplicate accumulation order and gathered empty-major arithmetic" {
    inline for (.{ f32, f64 }) |T| {
        const large: T = if (T == f32) 0x1p24 else 0x1p53;
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |operation| {
                var a: sparse_pattern.Operand(T, i64, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 1, 4, 4 } }, .ordinals = &.{ 1, 1, 1 }, .row_count = 2, .column_count = 2, .major_axis = axis, .index_base = .one }, .values = &.{ large, 1, -large }, .operation = operation };
                _ = try a.check();
                var output: [2]T = @splat(std.math.nan(T));
                sparse_product.executeValidated(T, i64, u32, a, &.{ 1, std.math.nan(T) }, &output, 1, 0);
                try productBytes(T, &.{ 0, 0 }, &output);
                a.values = &.{ -0.0, -0.0, -0.0 };
                sparse_product.executeValidated(T, i64, u32, a, &.{ 1, std.math.nan(T) }, &output, -1, -0.0);
                try productBytes(T, &.{ 0, 0 }, &output);
                sparse_product.executeValidated(T, i64, u32, a, &.{ 1, std.math.nan(T) }, &output, std.math.inf(T), 0);
                const gather = (axis == .row) == (operation == .normal);
                try std.testing.expect(std.math.isNan(output[0]));
                if (gather) try std.testing.expect(std.math.isNan(output[1])) else try std.testing.expectEqual(@as(T, 0), output[1]);
            }
        }
    }
}

test "compressed product conjugation and signed zero obey scalar expression order" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |operation| {
                const z: T = .{ .re = -0.0, .im = -0.0 };
                const x: T = .{ .re = -1, .im = 0 };
                const alpha: T = .{ .re = -1, .im = -0.0 };
                const a: sparse_pattern.Operand(T, u32, i64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 1 } }, .ordinals = &.{0}, .row_count = 1, .column_count = 1, .major_axis = axis }, .values = &.{z}, .operation = operation };
                _ = try a.check();
                const zi = if (operation == .adjoint) -z.im else z.im;
                const mul: T = .{ .re = z.re * x.re - zi * x.im, .im = z.re * x.im + zi * x.re };
                const gather = (axis == .row) == (operation == .normal);
                const sum: T = if (gather) .{ .re = @as(R, 0) + mul.re, .im = @as(R, 0) + mul.im } else mul;
                const scaled: T = .{ .re = alpha.re * sum.re - alpha.im * sum.im, .im = alpha.re * sum.im + alpha.im * sum.re };
                const expected: T = .{ .re = @as(R, 0) + scaled.re, .im = @as(R, 0) + scaled.im };
                var output = [_]T{.{ .re = std.math.nan(R), .im = std.math.nan(R) }};
                sparse_product.executeValidated(T, u32, i64, a, &.{x}, &output, alpha, .{ .re = 0, .im = -0.0 });
                try productBytes(T, &.{expected}, &output);
            }
        }
    }
}

test "compressed product borrowed metadata is revalidated before each execution" {
    var ordinals = [_]i32{0};
    var starts = [_]i64{0};
    const a: sparse_pattern.Operand(f64, i64, i32) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &starts, .ends = &.{1} } }, .ordinals = &ordinals, .row_count = 1, .column_count = 1 }, .values = &.{3} };
    var result = [_]f64{9};
    _ = try a.check();
    sparse_product.executeValidated(f64, i64, i32, a, &.{2}, &result, 1, 0);
    try std.testing.expectEqual(@as(f64, 6), result[0]);
    ordinals[0] = -1;
    try std.testing.expectError(error.InvalidOrdinal, a.check());
    starts[0] = -1;
    try std.testing.expectError(error.InvalidOffsets, a.check());
    try std.testing.expectEqual(@as(f64, 6), result[0]);
}

test "checked SpMV preserves structural dimensions and alias precedence even with zero alpha" {
    var output = [_]f64{ 17, 19 };
    var offsets = [_]i32{ 0, 1, 1 };
    var indices = [_]i32{0};
    var a: sparse_checked.ConstCsrMatrix(f64, i32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &.{2}, .row_count = 2, .column_count = 2 };
    offsets[0] = 1;
    try std.testing.expectError(error.InvalidRowOffsets, sparse_checked.matrixVectorMultiply(.{ .matrix = a, .input_vector = output[0..1], .result_vector = &output, .product_scale = 0 }));
    offsets[0] = 0;
    try std.testing.expectError(error.DimensionMismatch, sparse_checked.matrixVectorMultiply(.{ .matrix = a, .input_vector = output[0..1], .result_vector = &output, .product_scale = 0 }));
    try std.testing.expectError(error.AliasingNotAllowed, sparse_checked.matrixVectorMultiply(.{ .matrix = a, .input_vector = &output, .result_vector = &output, .product_scale = 0 }));
    a.values = output[0..1];
    try std.testing.expectError(error.AliasingNotAllowed, sparse_checked.matrixVectorMultiply(.{ .matrix = a, .input_vector = &.{ 1.0, 2.0 }, .result_vector = &output, .product_scale = 0 }));
    try productBytes(f64, &.{ 17, 19 }, &output);
}

const real_policy_core = @import("lapack/core/cholesky.zig");
const real_policy_frontend = @import("lapack/cholesky.zig");
const real_policy_views = @import("blas/api/views.zig");

fn policyMatrix(comptime T: type, values: []T, n: usize, stride: usize) dense.Matrix(T) {
    return .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = stride };
}
fn policyIndex(row: usize, column: usize, stride: usize, comptime triangle: real_policy_core.Triangle) usize {
    return if (triangle == .lower) row + column * stride else column + row * stride;
}
fn expectPolicyBytes(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

test "real factor contracts preserve exact Gram factors opposite triangle and padded storage" {
    const gram = [3][3]f64{ .{ 4, 2, -4 }, .{ 2, 10, 1 }, .{ -4, 1, 9 } };
    const lower = [3][3]f64{ .{ 2, 0, 0 }, .{ 1, 3, 0 }, .{ -2, 1, 2 } };
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        for ([_]usize{ 3, 5, 17 }) |stride| inline for (.{ real_policy_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
            var a: [55]T = @splat(std.math.nan(T));
            for (0..3) |j| for (j..3) |i| {
                a[policyIndex(i, j, stride, triangle)] = @floatCast(gram[i][j]);
            };
            var expected = a;
            for (0..3) |j| for (j..3) |i| {
                expected[policyIndex(i, j, stride, triangle)] = @floatCast(lower[i][j]);
            };
            try std.testing.expectEqual(@as(usize, 0), real_policy_core.factorizeWithContract(T, policyMatrix(T, &a, 3, stride), triangle, contract));
            try expectPolicyBytes(T, &expected, &a);
        };
    };
}

fn policyOracleLower(i: usize, j: usize) f64 {
    if (i < j) return 0;
    if (i == j) return @floatFromInt(2 + i % 3);
    const integer: i32 = @as(i32, @intCast((i * 5 + j * 3) % 7)) - 3;
    return @as(f64, @floatFromInt(integer)) / 8;
}

test "real factor policies satisfy independent Gram residuals across scalar and vector update sizes" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        for ([_]usize{ 1, 2, 5, 9, 13 }) |n| inline for (.{ real_policy_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
            const stride = n + 3;
            var a: [224]T = @splat(71);
            for (0..n) |j| for (j..n) |i| {
                var gram: f64 = 0;
                for (0..n) |k| gram += policyOracleLower(i, k) * policyOracleLower(j, k);
                a[policyIndex(i, j, stride, triangle)] = @floatCast(gram);
            };
            const before = a;
            try std.testing.expectEqual(@as(usize, 0), real_policy_core.factorizeWithContract(T, policyMatrix(T, &a, n, stride), triangle, contract));
            const epsilon: f64 = @floatCast(std.math.floatEps(T));
            for (0..n) |j| for (j..n) |i| {
                var product: f64 = 0;
                for (0..j + 1) |k| product += @as(f64, a[policyIndex(i, k, stride, triangle)]) * @as(f64, a[policyIndex(j, k, stride, triangle)]);
                const original: f64 = before[policyIndex(i, j, stride, triangle)];
                try std.testing.expectApproxEqAbs(original, product, 64 * epsilon * @max(1, @abs(original)));
                try std.testing.expectApproxEqAbs(policyOracleLower(i, j), @as(f64, a[policyIndex(i, j, stride, triangle)]), 64 * epsilon);
            };
            // Restore only selected entries; every other byte must match.
            var restored = a;
            for (0..n) |j| for (j..n) |i| {
                restored[policyIndex(i, j, stride, triangle)] = before[policyIndex(i, j, stride, triangle)];
            };
            try expectPolicyBytes(T, &before, &restored);
        };
    };
}

test "real factor policies distinguish positive infinity and reject NaN signed zero and negative pivots" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        for ([_]T{ 0, -0.0, -1, std.math.nan(T), -std.math.inf(T), std.math.inf(T) }) |value| {
            var checked = [_]T{value};
            var standard = checked;
            const before = checked;
            try std.testing.expectEqual(@as(usize, 1), real_policy_core.factorize(T, policyMatrix(T, &checked, 1, std.math.maxInt(usize)), triangle));
            const standard_info: usize = if (value == std.math.inf(T)) 0 else 1;
            try std.testing.expectEqual(standard_info, real_policy_core.factorizeWithContract(T, policyMatrix(T, &standard, 1, std.math.maxInt(usize)), triangle, .lapack_potrf));
            try expectPolicyBytes(T, &before, &checked);
            try expectPolicyBytes(T, &before, &standard);
        }
        inline for (.{ real_policy_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
            var empty: [2]T = .{ std.math.nan(T), -0.0 };
            const before = empty;
            try std.testing.expectEqual(@as(usize, 0), real_policy_core.factorizeWithContract(T, policyMatrix(T, &empty, 0, 0), triangle, contract));
            try expectPolicyBytes(T, &before, &empty);
        }
    };
}

test "real POTRF policy reports negative Schur pivot without a POTF2 residue store" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        inline for (.{ real_policy_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
            var a: [8]T = @splat(73);
            a[0] = 4;
            a[4] = 5;
            a[policyIndex(1, 0, 3, triangle)] = 6;
            var expected = a;
            expected[0] = 2;
            expected[policyIndex(1, 0, 3, triangle)] = 3;
            // The independently computed Schur complement is 5-3*3=-4;
            // this POTRF contract leaves the original diagonal, not -4.
            try std.testing.expectEqual(@as(usize, 2), real_policy_core.factorizeWithContract(T, policyMatrix(T, &a, 2, 3), triangle, contract));
            try expectPolicyBytes(T, &expected, &a);
        }
    };
}

test "real POTRF policy propagates nonfinite offdiagonal values to the later failing pivot" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        for ([_]T{ std.math.nan(T), std.math.inf(T), -std.math.inf(T) }) |value| {
            var checked: [8]T = @splat(79);
            checked[0] = 4;
            checked[4] = 5;
            const position = policyIndex(1, 0, 3, triangle);
            checked[position] = value;
            var standard = checked;
            var expected = checked;
            expected[0] = 2;
            try std.testing.expectEqual(@as(usize, 1), real_policy_core.factorize(T, policyMatrix(T, &checked, 2, 3), triangle));
            try std.testing.expectEqual(@as(usize, 2), real_policy_core.factorizeWithContract(T, policyMatrix(T, &standard, 2, 3), triangle, .lapack_potrf));
            try expectPolicyBytes(T, &expected, &checked);
            if (std.math.isNan(value)) {
                // NaN classification is required; the arithmetic's payload is not.
                try std.testing.expect(std.math.isNan(standard[position]));
                expected[position] = standard[position];
            }
            try expectPolicyBytes(T, &expected, &standard);
        }
    };
}

test "real POTRF policy stores finite-input division overflow while checked retains the failing assignment" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        var checked: [8]T = @splat(83);
        checked[0] = 0.25;
        checked[4] = 5;
        const position = policyIndex(1, 0, 3, triangle);
        checked[position] = std.math.floatMax(T);
        var standard = checked;
        var expected = checked;
        expected[0] = 0.5;
        try std.testing.expectEqual(@as(usize, 1), real_policy_core.factorize(T, policyMatrix(T, &checked, 2, 3), triangle));
        try expectPolicyBytes(T, &expected, &checked);
        expected[position] = std.math.inf(T);
        try std.testing.expectEqual(@as(usize, 2), real_policy_core.factorizeWithContract(T, policyMatrix(T, &standard, 2, 3), triangle, .lapack_potrf));
        try expectPolicyBytes(T, &expected, &standard);
    };
}

test "real checked default retains lower update mutation before an overflowing second column division" {
    inline for (.{ f32, f64 }) |T| inline for (.{ real_policy_core.Triangle.lower, .upper }) |triangle| {
        var a: [18]T = @splat(89);
        a[0] = 1;
        a[5] = 1.25;
        a[10] = std.math.floatMax(T);
        a[policyIndex(1, 0, 4, triangle)] = 1;
        a[policyIndex(2, 0, 4, triangle)] = -std.math.floatMax(T) * 0.25;
        a[policyIndex(2, 1, 4, triangle)] = std.math.floatMax(T) * 0.375;
        var explicit = a;
        var expected = a;
        expected[5] = 0.5;
        if (triangle == .lower) expected[policyIndex(2, 1, 4, triangle)] =
            a[policyIndex(2, 1, 4, triangle)] - a[policyIndex(2, 0, 4, triangle)];
        try std.testing.expectEqual(@as(usize, 2), real_policy_core.factorize(T, policyMatrix(T, &a, 3, 4), triangle));
        try std.testing.expectEqual(@as(usize, 2), real_policy_core.factorizeWithContract(T, policyMatrix(T, &explicit, 3, 4), triangle, .checked_v1));
        try expectPolicyBytes(T, &expected, &a);
        try expectPolicyBytes(T, &expected, &explicit);
    };
}

test "real checked frontend retains nonfinite preflight and geometry error precedence under the new core policy" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 4, std.math.nan(T), 97, 5 };
        const before = a;
        var view: real_policy_views.Matrix(T) = .{ .values = &a, .row_count = -1, .column_count = 2, .leading_dimension = -1 };
        try std.testing.expectError(error.InvalidLength, real_policy_frontend.factorize(T, view, .lower));
        view.row_count = 2;
        try std.testing.expectError(error.InvalidLeadingDimension, real_policy_frontend.factorize(T, view, .lower));
        view.leading_dimension = 2;
        view.values = a[0..3];
        try std.testing.expectError(error.BufferTooSmall, real_policy_frontend.factorize(T, view, .lower));
        view.values = &a;
        view.column_count = 1;
        try std.testing.expectError(error.DimensionMismatch, real_policy_frontend.factorize(T, view, .lower));
        view.column_count = 2;
        try std.testing.expectError(error.NonFiniteInput, real_policy_frontend.factorize(T, view, .lower));
        try expectPolicyBytes(T, &before, &a);
        a[1] = std.math.inf(T);
        const infinite_before = a;
        try std.testing.expectError(error.NonFiniteInput, real_policy_frontend.factorize(T, view, .lower));
        try expectPolicyBytes(T, &infinite_before, &a);
    }
}

fn complexPolicyLower(i: usize, j: usize) struct { re: f64, im: f64 } {
    if (i < j) return .{ .re = 0, .im = 0 };
    if (i == j) return .{ .re = @floatFromInt(2 + i % 3), .im = 0 };
    const re: i32 = @as(i32, @intCast((i * 7 + j * 3) % 9)) - 4;
    const im: i32 = @as(i32, @intCast((i * 3 + j * 5) % 7)) - 3;
    return .{ .re = @as(f64, @floatFromInt(re)) / 8, .im = @as(f64, @floatFromInt(im)) / 8 };
}

test "complex factor contracts reconstruct independent Hermitian products and preserve inactive bytes" {
    try std.testing.expect(complex_core.FactorContract == real_policy_core.FactorContract);
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            for ([_]usize{ 1, 2, 3, 9 }) |n| for ([_]usize{ 0, 3 }) |pad| {
                const ld = n + pad;
                inline for (.{ complex_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
                    var a: [128]C = @splat(.{ .re = std.math.nan(R), .im = -0.0 });
                    for (0..n) |j| for (j..n) |i| {
                        var re: f64 = 0;
                        var im: f64 = 0;
                        for (0..n) |k| {
                            const x = complexPolicyLower(i, k);
                            const y = complexPolicyLower(j, k);
                            re += x.re * y.re + x.im * y.im;
                            im += x.im * y.re - x.re * y.im;
                        }
                        a[storedPosition(i, j, ld, triangle)] = asStored(C, .{ .re = @floatCast(re), .im = @floatCast(im) }, triangle);
                    };
                    const before = a;
                    try std.testing.expectEqual(@as(usize, 0), complex_core.factorizeWithContract(C, complexOperand(C, &a, n, n, ld), triangle, contract));
                    const epsilon: f64 = @floatCast(std.math.floatEps(R));
                    for (0..n) |j| for (j..n) |i| {
                        var re: f64 = 0;
                        var im: f64 = 0;
                        for (0..j + 1) |k| {
                            const x = asStored(C, a[storedPosition(i, k, ld, triangle)], triangle);
                            const y = asStored(C, a[storedPosition(j, k, ld, triangle)], triangle);
                            re += @as(f64, x.re) * y.re + @as(f64, x.im) * y.im;
                            im += @as(f64, x.im) * y.re - @as(f64, x.re) * y.im;
                        }
                        const original = asStored(C, before[storedPosition(i, j, ld, triangle)], triangle);
                        const expected = complexPolicyLower(i, j);
                        const got = asStored(C, a[storedPosition(i, j, ld, triangle)], triangle);
                        try std.testing.expectApproxEqAbs(@as(f64, original.re), re, 64 * epsilon * @max(1, @abs(@as(f64, original.re))));
                        try std.testing.expectApproxEqAbs(@as(f64, original.im), im, 64 * epsilon * @max(1, @abs(@as(f64, original.im))));
                        try std.testing.expectApproxEqAbs(expected.re, @as(f64, got.re), 64 * epsilon);
                        try std.testing.expectApproxEqAbs(expected.im, @as(f64, got.im), 64 * epsilon);
                    };
                    var restored = a;
                    for (0..n) |j| for (j..n) |i| {
                        restored[storedPosition(i, j, ld, triangle)] = before[storedPosition(i, j, ld, triangle)];
                    };
                    try expectComplexBytes(C, &before, &restored);
                }
            };
        }
    }
}

test "complex factor pivot policies normalize imaginary bytes only after acceptance and allow native width empties" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            for ([_]R{ 4, std.math.inf(R), 0, -0.0, -1, std.math.nan(R), -std.math.inf(R) }) |re| {
                for ([_]R{ 0, -0.0, 7, std.math.nan(R), std.math.inf(R) }) |im| {
                    inline for (.{ complex_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
                        var a = [_]C{ .{ .re = re, .im = im }, .{ .re = 17, .im = -23 } };
                        var expected = a;
                        const accepted = re > 0 and (contract == .lapack_potrf or std.math.isFinite(re));
                        if (accepted) expected[0] = .{ .re = @sqrt(re), .im = 0 };
                        try std.testing.expectEqual(@as(usize, if (accepted) 0 else 1), complex_core.factorizeWithContract(C, complexOperand(C, &a, 1, 1, std.math.maxInt(usize)), triangle, contract));
                        try expectComplexBytes(C, &expected, &a);
                    }
                }
            }
            inline for (.{ complex_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
                var a = [_]C{.{ .re = std.math.nan(R), .im = -0.0 }};
                const before = a;
                try std.testing.expectEqual(@as(usize, 0), complex_core.factorizeWithContract(C, complexOperand(C, &a, 0, 0, 0), triangle, contract));
                try expectComplexBytes(C, &before, &a);
            }
        }
    }
}

test "complex factor contracts retain failed Schur pivot and later storage without POTF2 residue assignment" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            for ([_]R{ 5, std.math.nan(R) }) |second| inline for (.{ complex_core.FactorContract.checked_v1, .lapack_potrf }) |contract| {
                var a: [9]C = @splat(.{ .re = 73, .im = -19 });
                a[0] = .{ .re = 4, .im = std.math.nan(R) };
                a[4] = .{ .re = second, .im = -7 };
                const position = storedPosition(1, 0, 3, triangle);
                a[position] = asStored(C, .{ .re = 6, .im = 2 }, triangle);
                var expected = a;
                expected[0] = .{ .re = 2, .im = 0 };
                expected[position] = asStored(C, .{ .re = 3, .im = 1 }, triangle);
                // For second=5 the independent Schur residue is 5-(9+1)=-5.
                // Both the original real and imaginary failed-pivot bytes survive.
                try std.testing.expectEqual(@as(usize, 2), complex_core.factorizeWithContract(C, complexOperand(C, &a, 2, 2, 3), triangle, contract));
                try expectComplexBytes(C, &expected, &a);
            };
        }
    }
}

test "complex POTRF policy stores offdiagonal IEEE results and permits positive infinite pivots" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            const position = storedPosition(1, 0, 3, triangle);
            for ([_]C{ .{ .re = std.math.nan(R), .im = 2 }, .{ .re = std.math.inf(R), .im = 2 }, .{ .re = 2, .im = -std.math.inf(R) }, .{ .re = 2, .im = std.math.nan(R) } }) |value| {
                var checked: [9]C = @splat(.{ .re = 79, .im = -29 });
                checked[0] = .{ .re = 4, .im = 0 };
                checked[4] = .{ .re = 5, .im = -7 };
                checked[position] = asStored(C, value, triangle);
                var standard = checked;
                var expected = checked;
                expected[0] = .{ .re = 2, .im = 0 };
                try std.testing.expectEqual(@as(usize, 1), complex_core.factorize(C, complexOperand(C, &checked, 2, 2, 3), triangle));
                try expectComplexBytes(C, &expected, &checked);
                try std.testing.expectEqual(@as(usize, 2), complex_core.factorizeWithContract(C, complexOperand(C, &standard, 2, 2, 3), triangle, .lapack_potrf));
                expected[position] = asStored(C, .{ .re = value.re / 2, .im = value.im / 2 }, triangle);
                if (std.math.isNan(value.re)) {
                    try std.testing.expect(std.math.isNan(standard[position].re));
                    expected[position].re = standard[position].re;
                }
                if (std.math.isNan(value.im)) {
                    try std.testing.expect(std.math.isNan(standard[position].im));
                    expected[position].im = standard[position].im;
                }
                try expectComplexBytes(C, &expected, &standard);
            }
            var infinite: [9]C = @splat(.{ .re = 83, .im = -31 });
            infinite[0] = .{ .re = std.math.inf(R), .im = std.math.nan(R) };
            infinite[position] = asStored(C, .{ .re = 2, .im = -4 }, triangle);
            infinite[4] = .{ .re = 4, .im = std.math.inf(R) };
            var expected = infinite;
            expected[0].im = 0;
            expected[position] = asStored(C, .{ .re = 0, .im = -0.0 }, triangle);
            expected[4] = .{ .re = 2, .im = 0 };
            try std.testing.expectEqual(@as(usize, 0), complex_core.factorizeWithContract(C, complexOperand(C, &infinite, 2, 2, 3), triangle, .lapack_potrf));
            try expectComplexBytes(C, &expected, &infinite);
        }
    }
}

test "complex finite narrowing overflow retains checked completed prefix while POTRF stores and continues" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            for ([_]bool{ false, true }) |imaginary_overflow| {
                var checked: [16]C = @splat(.{ .re = 89, .im = -37 });
                checked[0] = .{ .re = 0.25, .im = -0.0 };
                checked[5] = .{ .re = 4, .im = 0 };
                checked[10] = .{ .re = 9, .im = 0 };
                const first = storedPosition(1, 0, 4, triangle);
                const failing = storedPosition(2, 0, 4, triangle);
                checked[first] = asStored(C, .{ .re = 0.5, .im = 0.25 }, triangle);
                checked[failing] = asStored(C, if (imaginary_overflow) .{ .re = 1, .im = std.math.floatMax(R) } else .{ .re = std.math.floatMax(R), .im = 1 }, triangle);
                checked[storedPosition(2, 1, 4, triangle)] = .{ .re = 0, .im = 0 };
                var explicit = checked;
                var standard = checked;
                var expected = checked;
                expected[0] = .{ .re = 0.5, .im = 0 };
                expected[first] = asStored(C, .{ .re = 1, .im = 0.5 }, triangle);
                try std.testing.expectEqual(@as(usize, 1), complex_core.factorize(C, complexOperand(C, &checked, 3, 3, 4), triangle));
                try std.testing.expectEqual(@as(usize, 1), complex_core.factorizeWithContract(C, complexOperand(C, &explicit, 3, 3, 4), triangle, .checked_v1));
                try expectComplexBytes(C, &expected, &checked);
                try expectComplexBytes(C, &expected, &explicit);
                try std.testing.expectEqual(@as(usize, 3), complex_core.factorizeWithContract(C, complexOperand(C, &standard, 3, 3, 4), triangle, .lapack_potrf));
                try std.testing.expect(std.math.isInf(if (imaginary_overflow) standard[failing].im else standard[failing].re));
                try expectComplexBytes(C, expected[0..1], standard[0..1]);
                try expectComplexBytes(C, checked[10..11], standard[10..11]);
                var restored = standard;
                for (0..3) |j| for (j..3) |i| {
                    restored[storedPosition(i, j, 4, triangle)] = expected[storedPosition(i, j, 4, triangle)];
                };
                try expectComplexBytes(C, &expected, &restored);
            }
        }
    }
}

test "complex checked frontend keeps geometry finite and diagonal errors before mutation with explicit core policies" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ complex_core.Triangle.lower, .upper }) |triangle| {
            const front_triangle: complex_support.Triangle = if (triangle == .lower) .lower else .upper;
            const position = storedPosition(1, 0, 3, triangle);
            var a: [9]C = @splat(.{ .re = 97, .im = -41 });
            a[0] = .{ .re = 4, .im = 7 };
            a[4] = .{ .re = 5, .im = 0 };
            a[position] = .{ .re = std.math.nan(R), .im = 0 };
            const before = a;
            var view = checkedComplexOperand(C, &a, -1, 2, -1);
            try std.testing.expectError(error.InvalidLength, complex_frontend.factorize(C, view, front_triangle));
            view.row_count = 2;
            try std.testing.expectError(error.InvalidLeadingDimension, complex_frontend.factorize(C, view, front_triangle));
            view.leading_dimension = 3;
            view.values = a[0..4];
            try std.testing.expectError(error.BufferTooSmall, complex_frontend.factorize(C, view, front_triangle));
            view.values = &a;
            view.column_count = 1;
            try std.testing.expectError(error.DimensionMismatch, complex_frontend.factorize(C, view, front_triangle));
            view.column_count = 2;
            try std.testing.expectError(error.NonFiniteInput, complex_frontend.factorize(C, view, front_triangle));
            try expectComplexBytes(C, &before, &a);
            a[position] = .{ .re = 1, .im = 0 };
            const finite = a;
            try std.testing.expectError(error.InvalidHermitianDiagonal, complex_frontend.factorize(C, view, front_triangle));
            try expectComplexBytes(C, &finite, &a);
            a[0] = .{ .re = std.math.inf(R), .im = 0 };
            const infinite = a;
            try std.testing.expectError(error.NonFiniteInput, complex_frontend.factorize(C, view, front_triangle));
            try expectComplexBytes(C, &infinite, &a);
        }
    }
}

// SolveContract regressions: finite entry calls below have finite selected
// numerical operands and nonzero diagonals. Padding/opposite poison is inactive.
const solve_contracts = @import("lapack/core/cholesky_contract.zig");
fn finiteSolve(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), comptime is_lower: bool) void {
    if (comptime arithmetic.scalar.isComplex(T)) {
        complex_core.solveWithContract(T, factor, rhs, if (is_lower) .lower else .upper, .potrs_finite_v1);
    } else {
        real_policy_core.solveWithContract(T, factor, rhs, if (is_lower) .lower else .upper, .potrs_finite_v1);
    }
}
const SolveRef = struct { re: f128, im: f128 };
fn solveRef(comptime T: type, value: T) SolveRef {
    return if (comptime arithmetic.scalar.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn solveRefMul(a: SolveRef, b: SolveRef) SolveRef {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn solveRefAdd(a: SolveRef, b: SolveRef) SolveRef {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn solveRefConj(a: SolveRef) SolveRef {
    return .{ .re = a.re, .im = -a.im };
}
fn solveStored(comptime T: type, a: SolveRef) T {
    return if (comptime arithmetic.scalar.isComplex(T)) .{ .re = @floatCast(a.re), .im = @floatCast(a.im) } else @floatCast(a.re);
}
fn solveNear(comptime T: type, expected: SolveRef, actual: T, tolerance: f128) !void {
    const value = solveRef(T, actual);
    try std.testing.expect(@abs(expected.re - value.re) <= tolerance);
    try std.testing.expect(@abs(expected.im - value.im) <= tolerance);
}

test "solve contracts share scalar identity and distinguish status from finite void result" {
    try std.testing.expect(solve_contracts.SolveResult(.checked_v1) == usize);
    try std.testing.expect(solve_contracts.SolveResult(.potrs_finite_v1) == void);
    try std.testing.expect(real_policy_core.SolveContract == complex_core.SolveContract);
    try std.testing.expect(arithmetic.scalar.ComplexF32 == complex_support.scalar.ComplexF32);
    try std.testing.expect(arithmetic.scalar.ComplexF64 == complex_support.scalar.ComplexF64);
    // The family factor policy and its identity remain unchanged.
    try std.testing.expect(real_policy_core.FactorContract == complex_core.FactorContract);
}

test "finite full complex diagonals satisfy exact independent one element oracles" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        inline for (.{ true, false }) |is_lower| {
            const cases = [_]struct { d: C, b: C, x: C }{
                .{ .d = .{ .re = 2, .im = 1 }, .b = .{ .re = 10, .im = 0 }, .x = .{ .re = 2, .im = 0 } },
                .{ .d = .{ .re = 0, .im = 1 }, .b = .{ .re = 10, .im = 0 }, .x = .{ .re = 10, .im = 0 } },
                .{ .d = .{ .re = -2, .im = 0 }, .b = .{ .re = 8, .im = 0 }, .x = .{ .re = 2, .im = 0 } },
            };
            for (cases) |case| {
                var a = [_]C{case.d};
                var b = [_]C{case.b};
                finiteSolve(C, complexOperand(C, &a, 1, 1, 1).asConst(), complexOperand(C, &b, 1, 1, 1), is_lower);
                try solveNear(C, solveRef(C, case.x), b[0], 0);
                try expectComplexBytes(C, &.{case.d}, &a);
                b[0] = case.b;
                const triangle: complex_support.Triangle = if (is_lower) .lower else .upper;
                if (case.d.im != 0) {
                    try std.testing.expectError(error.InvalidHermitianDiagonal, complex_frontend.solve(C, checkedComplexOperand(C, &a, 1, 1, 1).asConst(), checkedComplexOperand(C, &b, 1, 1, 1), triangle));
                } else {
                    try std.testing.expectEqual(@as(usize, 1), try complex_frontend.solve(C, checkedComplexOperand(C, &a, 1, 1, 1).asConst(), checkedComplexOperand(C, &b, 1, 1, 1), triangle));
                }
                try expectComplexBytes(C, &.{case.b}, &b);
            }
        }
    }
}

test "finite solve scaled complex diagonal avoids squared denominator overflow and underflow" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        const magnitude: i32 = if (R == f32) 80 else 600;
        inline for (.{ true, false }) |is_lower| {
            for ([_]i32{ magnitude, -magnitude }) |exponent| {
                const scale = std.math.ldexp(@as(R, 1), exponent);
                var a = [_]C{.{ .re = scale, .im = scale }};
                var b = [_]C{.{ .re = scale, .im = 0 }};
                finiteSolve(C, complexOperand(C, &a, 1, 1, 1).asConst(), complexOperand(C, &b, 1, 1, 1), is_lower);
                // Exact rational oracle: |d|^2=2*s^2; B=s; X=2^(-e-1).
                try solveNear(C, .{ .re = std.math.ldexp(@as(R, 1), -exponent - 1), .im = 0 }, b[0], 0);
                try std.testing.expect(std.math.isFinite(b[0].re));
            }
        }
    }
}

test "finite complex solve matches frozen rational two RHS factors with padded poison" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        inline for (.{ true, false }) |is_lower| {
            const poison: C = .{ .re = std.math.nan(R), .im = std.math.inf(R) };
            var a = [_]C{ .{ .re = 2, .im = 1 }, poison, poison, poison, .{ .re = 3, .im = -1 }, poison };
            a[if (is_lower) 1 else 3] = if (is_lower) .{ .re = 1, .im = -2 } else .{ .re = -1, .im = 2 };
            const before = a;
            var b = [_]C{ .{ .re = 10, .im = 25 }, .{ .re = 55, .im = -20 }, poison, .{ .re = -10, .im = 10 }, .{ .re = 20, .im = 10 }, poison };
            finiteSolve(C, complexOperand(C, &a, 2, 2, 3).asConst(), complexOperand(C, &b, 2, 2, 3), is_lower);
            try solveNear(C, .{ .re = 1, .im = 2 }, b[0], 0);
            try solveNear(C, .{ .re = 3, .im = -1 }, b[1], 0);
            try solveNear(C, .{ .re = -2, .im = 1 }, b[3], 0);
            try solveNear(C, .{ .re = 1, .im = 0 }, b[4], 0);
            try expectComplexBytes(C, &.{poison}, b[2..3]);
            try expectComplexBytes(C, &.{poison}, b[5..6]);
            try expectComplexBytes(C, &before, &a);
        }
    }
}

fn finiteSolveResidual(comptime T: type, n: usize, comptime is_lower: bool) !void {
    const R = arithmetic.scalar.Real(T);
    const stride = n + 2;
    const poison = luValue(T, std.math.nan(R), std.math.inf(R));
    var a: [340]T = @splat(poison);
    var factor: [17][17]SolveRef = @splat(@splat(.{ .re = 0, .im = 0 }));
    for (0..n) |i| for (0..n) |j| {
        if ((is_lower and i >= j) or (!is_lower and i <= j)) {
            const re: R = if (i == j) @as(R, @floatFromInt(4 + i)) else @as(R, @floatFromInt(@as(i32, @intCast((i * 3 + j * 7) % 11)) - 5)) * 0.125;
            const im: R = @as(R, @floatFromInt(@as(i32, @intCast((i + j * 3) % 5)) - 2)) * 0.25;
            a[i + j * stride] = luValue(T, re, im);
            factor[i][j] = solveRef(T, a[i + j * stride]);
        }
    };
    var system: [17][17]SolveRef = @splat(@splat(.{ .re = 0, .im = 0 }));
    for (0..n) |i| for (0..n) |j| {
        for (0..n) |k| system[i][j] = solveRefAdd(system[i][j], if (is_lower) solveRefMul(factor[i][k], solveRefConj(factor[j][k])) else solveRefMul(solveRefConj(factor[k][i]), factor[k][j]));
    };
    var known: [3][17]SolveRef = undefined;
    var b: [60]T = @splat(poison);
    for (0..3) |column| {
        for (0..n) |i| known[column][i] = solveRef(T, luValue(T, @as(R, @floatFromInt(@as(i32, @intCast((i + column * 2) % 7)) - 3)) * 0.5, @as(R, @floatFromInt(i + column)) * 0.125));
        for (0..n) |i| {
            var value: SolveRef = .{ .re = 0, .im = 0 };
            for (0..n) |k| value = solveRefAdd(value, solveRefMul(system[i][k], known[column][k]));
            b[i + column * stride] = solveStored(T, value);
        }
    }
    const a_before = a;
    const b_before = b;
    finiteSolve(T, complexOperand(T, &a, n, n, stride).asConst(), complexOperand(T, &b, n, 3, stride), is_lower);
    const tolerance: f128 = if (R == f32) 0.00003 else 0.0000000000001;
    for (0..3) |column| for (0..n) |i| {
        try solveNear(T, known[column][i], b[i + column * stride], tolerance);
        var residual: SolveRef = .{ .re = 0, .im = 0 };
        for (0..n) |k| residual = solveRefAdd(residual, solveRefMul(system[i][k], solveRef(T, b[k + column * stride])));
        const expected = solveRef(T, b_before[i + column * stride]);
        const bound = tolerance * (1 + @abs(expected.re) + @abs(expected.im));
        try std.testing.expect(@abs(residual.re - expected.re) <= bound);
        try std.testing.expect(@abs(residual.im - expected.im) <= bound);
    };
    try expectPolicyBytes(T, &a_before, &a);
    for (0..b.len) |i| if (i / stride >= 3 or i % stride >= n) try expectPolicyBytes(T, b_before[i .. i + 1], b[i .. i + 1]);
}

test "finite solve independent dense products and residuals cover real complex tails and multiple RHS" {
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        inline for (.{ true, false }) |is_lower| {
            for ([_]usize{ 1, 2, 7, 8, 9, 17 }) |n| try finiteSolveResidual(T, n, is_lower);
        }
    }
}

test "finite solve computes native-width strides and empty shapes without reading numerical inputs" {
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        inline for (.{ true, false }) |is_lower| {
            var a = [_]T{luValue(T, 2, 0)};
            var b = [_]T{luValue(T, 8, 0)};
            const wide = std.math.maxInt(usize);
            const f = complexOperand(T, &a, 1, 1, wide);
            const r = complexOperand(T, &b, 1, 1, wide);
            try std.testing.expectEqual(@as(usize, 1), try f.check());
            try std.testing.expectEqual(@as(usize, 1), try r.check());
            finiteSolve(T, f.asConst(), r, is_lower);
            try solveNear(T, .{ .re = 2, .im = 0 }, b[0], 0);
            // No selected numerical entries exist when n or nrhs is zero.
            const empty = complexOperand(T, &.{}, 0, 0, 0);
            const many = complexOperand(T, &.{}, 0, @as(usize, std.math.maxInt(i32)) + 1, wide);
            finiteSolve(T, empty.asConst(), many, is_lower);
            const no_rhs = complexOperand(T, &.{}, 1, 0, 0);
            finiteSolve(T, f.asConst(), no_rhs, is_lower);
        }
    }
}

test "finite normal substitution skips signed zero before division instead of after it" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ true, false }) |is_lower| {
            var a = [_]T{-2};
            var b = [_]T{-0.0};
            finiteSolve(T, complexOperand(T, &a, 1, 1, 1).asConst(), complexOperand(T, &b, 1, 1, 1), is_lower);
            // One normal leg skips zero; the transposed leg divides once.
            // Dividing in both legs instead would preserve a negative zero.
            try expectPolicyBytes(T, &.{0.0}, &b);
        }
    }
}

// IEEE classifications below are Zynum's explicit numerical choice. They do
// not assert that LAPACK documentation specifies every nonfinite output bit.
const IeeeInput = enum { zero, negative_zero, one, two, negative_two, three, negative_three, four, positive_inf, negative_inf, nan, snan, maximum, negative_maximum, minimum_subnormal, large_scale, small_scale };
const IeeeClass = enum { positive_zero, negative_zero, positive_inf, negative_inf, quiet_nan, one, two, negative_one };
const IeeeDivisionCase = struct { id: []const u8, a: [2]IeeeInput, b: [2]IeeeInput, expected: [2]IeeeClass };
const IeeeFlowCase = struct { id: []const u8, diagonal: [2]IeeeInput, rhs: [2]IeeeInput, lower: [2]IeeeClass, upper: [2]IeeeClass };
fn ieeeInput(comptime R: type, input: IeeeInput) R {
    return switch (input) {
        .zero => 0.0,
        .negative_zero => -0.0,
        .one => 1,
        .two => 2,
        .negative_two => -2,
        .three => 3,
        .negative_three => -3,
        .four => 4,
        .positive_inf => std.math.inf(R),
        .negative_inf => -std.math.inf(R),
        .nan => std.math.nan(R),
        .snan => if (R == f32) @bitCast(@as(u32, 0x7f800001)) else @bitCast(@as(u64, 0x7ff0000000000001)),
        .maximum => std.math.floatMax(R),
        .negative_maximum => -std.math.floatMax(R),
        .minimum_subnormal => std.math.floatTrueMin(R),
        .large_scale => std.math.ldexp(@as(R, 1), if (R == f32) 80 else 600),
        .small_scale => std.math.ldexp(@as(R, 1), if (R == f32) -80 else -600),
    };
}
fn ieeeComplexInput(comptime C: type, input: [2]IeeeInput) C {
    return .{ .re = ieeeInput(arithmetic.scalar.Real(C), input[0]), .im = ieeeInput(arithmetic.scalar.Real(C), input[1]) };
}
fn expectIeeeClass(comptime R: type, expected: IeeeClass, actual: R) !void {
    switch (expected) {
        .positive_zero, .negative_zero => {
            try std.testing.expect(actual == 0);
            try std.testing.expectEqual(expected == .negative_zero, std.math.signbit(actual));
        },
        .positive_inf, .negative_inf => {
            try std.testing.expect(std.math.isInf(actual));
            try std.testing.expectEqual(expected == .negative_inf, std.math.signbit(actual));
        },
        .quiet_nan => {
            try std.testing.expect(std.math.isNan(actual));
            const U = if (R == f32) u32 else u64;
            const bits: U = @bitCast(actual);
            try std.testing.expect(bits & (@as(U, 1) << (if (R == f32) 22 else 51)) != 0);
        },
        .one => try std.testing.expectEqual(@as(R, 1), actual),
        .two => try std.testing.expectEqual(@as(R, 2), actual),
        .negative_one => try std.testing.expectEqual(@as(R, -1), actual),
    }
}
fn expectIeeeComplex(comptime C: type, expected: [2]IeeeClass, actual: C) !void {
    try expectIeeeClass(arithmetic.scalar.Real(C), expected[0], actual.re);
    try expectIeeeClass(arithmetic.scalar.Real(C), expected[1], actual.im);
}
fn ieeeSolve(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), comptime is_lower: bool) void {
    if (comptime arithmetic.scalar.isComplex(T)) {
        complex_core.solveWithContract(T, factor, rhs, if (is_lower) .lower else .upper, .potrs_ieee_v1);
    } else {
        real_policy_core.solveWithContract(T, factor, rhs, if (is_lower) .lower else .upper, .potrs_ieee_v1);
    }
}
const ieeeDivisionCases = [_]IeeeDivisionCase{
    .{ .id = "nan_numerator_real", .a = .{ .nan, .one }, .b = .{ .two, .three }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "snan_numerator_imag", .a = .{ .one, .snan }, .b = .{ .two, .three }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "nan_denominator_real", .a = .{ .one, .two }, .b = .{ .nan, .three }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "nan_denominator_imag_with_inf", .a = .{ .positive_inf, .one }, .b = .{ .two, .nan }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "nan_precedes_both_infinite", .a = .{ .positive_inf, .nan }, .b = .{ .positive_inf, .one }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "both_infinite_axes", .a = .{ .positive_inf, .zero }, .b = .{ .zero, .negative_inf }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "both_infinite_all", .a = .{ .negative_inf, .positive_inf }, .b = .{ .positive_inf, .negative_inf }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "zero_denominator_positive_real_sign", .a = .{ .two, .negative_three }, .b = .{ .zero, .zero }, .expected = .{ .positive_inf, .negative_inf } },
    .{ .id = "zero_denominator_negative_real_sign", .a = .{ .two, .negative_three }, .b = .{ .negative_zero, .zero }, .expected = .{ .negative_inf, .positive_inf } },
    .{ .id = "zero_over_zero", .a = .{ .zero, .negative_zero }, .b = .{ .zero, .negative_zero }, .expected = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "infinite_over_negative_zero", .a = .{ .positive_inf, .negative_inf }, .b = .{ .negative_zero, .negative_zero }, .expected = .{ .negative_inf, .positive_inf } },
    .{ .id = "infinite_and_zero_over_zero", .a = .{ .positive_inf, .zero }, .b = .{ .zero, .zero }, .expected = .{ .positive_inf, .quiet_nan } },
    .{ .id = "zero_and_nonzero_over_zero", .a = .{ .zero, .negative_three }, .b = .{ .zero, .zero }, .expected = .{ .quiet_nan, .negative_inf } },
    .{ .id = "negative_nonzero_over_negative_zero", .a = .{ .negative_two, .zero }, .b = .{ .negative_zero, .negative_zero }, .expected = .{ .positive_inf, .quiet_nan } },
    .{ .id = "finite_over_real_infinity", .a = .{ .two, .three }, .b = .{ .positive_inf, .zero }, .expected = .{ .positive_zero, .positive_zero } },
    .{ .id = "finite_over_negative_real_infinity", .a = .{ .two, .negative_three }, .b = .{ .negative_inf, .zero }, .expected = .{ .negative_zero, .positive_zero } },
    .{ .id = "finite_over_imaginary_infinity", .a = .{ .two, .three }, .b = .{ .zero, .positive_inf }, .expected = .{ .positive_zero, .negative_zero } },
    .{ .id = "finite_over_negative_imaginary_infinity", .a = .{ .two, .three }, .b = .{ .zero, .negative_inf }, .expected = .{ .negative_zero, .positive_zero } },
    .{ .id = "finite_over_two_infinite_components", .a = .{ .two, .three }, .b = .{ .positive_inf, .positive_inf }, .expected = .{ .positive_zero, .positive_zero } },
    .{ .id = "finite_over_opposite_infinite_components", .a = .{ .two, .three }, .b = .{ .positive_inf, .negative_inf }, .expected = .{ .negative_zero, .positive_zero } },
    .{ .id = "wide_denominator_form_avoids_overflow", .a = .{ .maximum, .maximum }, .b = .{ .positive_inf, .positive_inf }, .expected = .{ .positive_zero, .positive_zero } },
    .{ .id = "wide_denominator_form_negative", .a = .{ .maximum, .negative_maximum }, .b = .{ .positive_inf, .positive_inf }, .expected = .{ .positive_zero, .negative_zero } },
    .{ .id = "negative_zero_forms", .a = .{ .negative_zero, .negative_zero }, .b = .{ .positive_inf, .positive_inf }, .expected = .{ .negative_zero, .positive_zero } },
    .{ .id = "real_infinite_numerator", .a = .{ .positive_inf, .zero }, .b = .{ .two, .three }, .expected = .{ .positive_inf, .negative_inf } },
    .{ .id = "negative_real_infinite_numerator", .a = .{ .negative_inf, .zero }, .b = .{ .two, .three }, .expected = .{ .negative_inf, .positive_inf } },
    .{ .id = "imaginary_infinite_numerator", .a = .{ .zero, .positive_inf }, .b = .{ .two, .three }, .expected = .{ .positive_inf, .positive_inf } },
    .{ .id = "negative_imaginary_infinite_numerator", .a = .{ .zero, .negative_inf }, .b = .{ .two, .three }, .expected = .{ .negative_inf, .negative_inf } },
    .{ .id = "infinite_numerator_zero_imaginary_form", .a = .{ .positive_inf, .positive_inf }, .b = .{ .two, .two }, .expected = .{ .positive_inf, .quiet_nan } },
    .{ .id = "infinite_numerator_zero_real_form", .a = .{ .positive_inf, .negative_inf }, .b = .{ .two, .two }, .expected = .{ .quiet_nan, .negative_inf } },
    .{ .id = "wide_numerator_form_avoids_overflow", .a = .{ .positive_inf, .positive_inf }, .b = .{ .maximum, .maximum }, .expected = .{ .positive_inf, .quiet_nan } },
    .{ .id = "tiny_nonzero_divisor", .a = .{ .positive_inf, .zero }, .b = .{ .minimum_subnormal, .zero }, .expected = .{ .positive_inf, .quiet_nan } },
    .{ .id = "imaginary_divisor", .a = .{ .positive_inf, .zero }, .b = .{ .zero, .two }, .expected = .{ .quiet_nan, .negative_inf } },
    .{ .id = "finite_exact_full_diagonal", .a = .{ .four, .two }, .b = .{ .two, .one }, .expected = .{ .two, .positive_zero } },
    .{ .id = "finite_exact_imaginary_diagonal", .a = .{ .two, .four }, .b = .{ .zero, .two }, .expected = .{ .two, .negative_one } },
    .{ .id = "finite_huge_denominator", .a = .{ .large_scale, .large_scale }, .b = .{ .large_scale, .large_scale }, .expected = .{ .one, .positive_zero } },
    .{ .id = "finite_tiny_denominator", .a = .{ .small_scale, .small_scale }, .b = .{ .small_scale, .small_scale }, .expected = .{ .one, .positive_zero } },
};
const ieeeFlowCases = [_]IeeeFlowCase{
    .{ .id = "zero_diagonal_two_nonzero_rhs_components", .diagonal = .{ .zero, .zero }, .rhs = .{ .one, .one }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .positive_inf, .positive_inf } },
    .{ .id = "zero_diagonal_zero_imaginary_rhs", .diagonal = .{ .zero, .zero }, .rhs = .{ .one, .zero }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "zero_diagonal_zero_rhs", .diagonal = .{ .zero, .zero }, .rhs = .{ .zero, .zero }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "real_infinite_diagonal", .diagonal = .{ .positive_inf, .zero }, .rhs = .{ .two, .three }, .lower = .{ .positive_zero, .positive_zero }, .upper = .{ .positive_zero, .positive_zero } },
    .{ .id = "imaginary_infinite_diagonal_signed_zero", .diagonal = .{ .zero, .positive_inf }, .rhs = .{ .two, .three }, .lower = .{ .positive_zero, .positive_zero }, .upper = .{ .negative_zero, .positive_zero } },
    .{ .id = "nan_diagonal", .diagonal = .{ .nan, .zero }, .rhs = .{ .one, .one }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "real_infinite_rhs_one_times_seed", .diagonal = .{ .two, .zero }, .rhs = .{ .positive_inf, .zero }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "both_infinite_rhs_one_times_seed", .diagonal = .{ .two, .one }, .rhs = .{ .positive_inf, .positive_inf }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
    .{ .id = "both_factor_and_rhs_infinite", .diagonal = .{ .positive_inf, .positive_inf }, .rhs = .{ .positive_inf, .one }, .lower = .{ .quiet_nan, .quiet_nan }, .upper = .{ .quiet_nan, .quiet_nan } },
};

test "IEEE solve contract remains void with the same scalar and factor policy identities" {
    try std.testing.expect(solve_contracts.SolveResult(.potrs_ieee_v1) == void);
    try std.testing.expect(solve_contracts.SolveResult(.potrs_finite_v1) == void);
    try std.testing.expect(solve_contracts.SolveResult(.checked_v1) == usize);
    try std.testing.expect(real_policy_core.SolveContract == complex_core.SolveContract);
    try std.testing.expect(arithmetic.scalar.ComplexF32 == complex_support.scalar.ComplexF32);
    try std.testing.expect(arithmetic.scalar.ComplexF64 == complex_support.scalar.ComplexF64);
    try std.testing.expect(real_policy_core.FactorContract == complex_core.FactorContract);
}

test "IEEE complex quotient follows the independent explicit classification table" {
    @setEvalBranchQuota(10000);
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        for (ieeeDivisionCases) |case| {
            const a = ieeeComplexInput(C, case.a);
            const b = ieeeComplexInput(C, case.b);
            const result = arithmetic.divideIeee(C, a, b);
            expectIeeeComplex(C, case.expected, result) catch |err| {
                std.debug.print("division case {s}, scalar {s}, result {any}\n", .{ case.id, @typeName(C), result });
                return err;
            };
        }
    }
}

test "IEEE complex quotient retains the unchanged finite scaled leaf bytes" {
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        const R = arithmetic.scalar.Real(C);
        const scales = [_]R{ 1, 0.25, std.math.ldexp(@as(R, 1), if (R == f32) 80 else 600), std.math.ldexp(@as(R, 1), if (R == f32) -80 else -600) };
        for (scales) |scale| {
            for ([_]C{ .{ .re = scale, .im = scale }, .{ .re = scale, .im = -scale }, .{ .re = -0.0, .im = 0 }, .{ .re = 0, .im = -0.0 } }) |a| {
                for ([_]C{ .{ .re = scale, .im = scale }, .{ .re = -scale, .im = scale }, .{ .re = 0, .im = scale } }) |b| {
                    const wide = arithmetic.divide(C, arithmetic.widen(C, a), b);
                    const expected: C = .{ .re = @floatCast(wide.re), .im = @floatCast(wide.im) };
                    const actual = arithmetic.divideIeee(C, a, b);
                    try expectComplexBytes(C, &.{expected}, &.{actual});
                }
            }
        }
    }
}

test "IEEE complex complete solves retain full diagonal conjugation one times seed and zero gates" {
    @setEvalBranchQuota(10000);
    inline for (.{ arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |C| {
        inline for (.{ true, false }) |is_lower| {
            for (ieeeFlowCases) |case| {
                var a = [_]C{ieeeComplexInput(C, case.diagonal)};
                var b = [_]C{ieeeComplexInput(C, case.rhs)};
                const before = a;
                ieeeSolve(C, complexOperand(C, &a, 1, 1, 1).asConst(), complexOperand(C, &b, 1, 1, 1), is_lower);
                expectIeeeComplex(C, if (is_lower) case.lower else case.upper, b[0]) catch |err| {
                    std.debug.print("solve case {s}, scalar {s}, lower {any}, result {any}\n", .{ case.id, @typeName(C), is_lower, b[0] });
                    return err;
                };
                try expectComplexBytes(C, &before, &a);
            }
        }
    }
}

test "IEEE real complete solves continue division and scatter even when a quotient becomes zero" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ true, false }) |is_lower| {
            const cases = [_]struct { diagonal: T, rhs: T, expected: IeeeClass }{
                .{ .diagonal = 0, .rhs = 1, .expected = .positive_inf },
                .{ .diagonal = -0.0, .rhs = 1, .expected = .positive_inf },
                .{ .diagonal = 0, .rhs = 0, .expected = .quiet_nan },
                .{ .diagonal = std.math.inf(T), .rhs = 1, .expected = .positive_zero },
                .{ .diagonal = std.math.nan(T), .rhs = 1, .expected = .quiet_nan },
                .{ .diagonal = -2, .rhs = -0.0, .expected = .positive_zero },
            };
            for (cases) |case| {
                var a = [_]T{case.diagonal};
                var b = [_]T{case.rhs};
                const before = a;
                ieeeSolve(T, complexOperand(T, &a, 1, 1, 1).asConst(), complexOperand(T, &b, 1, 1, 1), is_lower);
                try expectIeeeClass(T, case.expected, b[0]);
                try expectPolicyBytes(T, &before, &a);
            }
            const poison = std.math.nan(T);
            var factor = [_]T{ std.math.inf(T), poison, poison, poison, 1, poison };
            factor[if (is_lower) 1 else 3] = std.math.inf(T);
            const factor_before = factor;
            var rhs = [_]T{ 1, 0, poison };
            ieeeSolve(T, complexOperand(T, &factor, 2, 2, 3).asConst(), complexOperand(T, &rhs, 2, 1, 3), is_lower);
            // In lower normal scatter, nonzero 1/Inf becomes zero but must still
            // update the second entry by 0*Inf. Its NaN survives the second leg.
            try expectIeeeClass(T, .quiet_nan, rhs[0]);
            try expectIeeeClass(T, .quiet_nan, rhs[1]);
            try expectPolicyBytes(T, &.{poison}, rhs[2..3]);
            try expectPolicyBytes(T, &factor_before, &factor);
        }
    }
}

test "IEEE solves continue across range failure and later RHS with exact inactive preservation" {
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        const R = arithmetic.scalar.Real(T);
        inline for (.{ true, false }) |is_lower| {
            const padding = luValue(T, std.math.nan(R), std.math.inf(R));
            var factor = [_]T{luValue(T, 0.5, 0)};
            const factor_before = factor;
            var rhs = [_]T{ luValue(T, 1, 1), padding, luValue(T, std.math.floatMax(R) * 0.375, 0), padding, luValue(T, 2, 1), padding };
            var checked = rhs;
            ieeeSolve(T, complexOperand(T, &factor, 1, 1, 1).asConst(), complexOperand(T, &rhs, 1, 3, 2), is_lower);
            try solveNear(T, solveRef(T, luValue(T, 4, 4)), rhs[0], 0);
            if (comptime arithmetic.scalar.isComplex(T)) {
                try expectIeeeClass(R, .positive_inf, rhs[2].re);
                try expectIeeeClass(R, .positive_zero, rhs[2].im);
            } else try expectIeeeClass(R, .positive_inf, rhs[2]);
            try solveNear(T, solveRef(T, luValue(T, 8, 4)), rhs[4], 0);
            for ([_]usize{ 1, 3, 5 }) |i| try expectPolicyBytes(T, &.{padding}, rhs[i .. i + 1]);
            try expectPolicyBytes(T, &factor_before, &factor);
            const info = if (comptime arithmetic.scalar.isComplex(T))
                complex_core.solve(T, complexOperand(T, &factor, 1, 1, 1).asConst(), complexOperand(T, &checked, 1, 3, 2), if (is_lower) .lower else .upper)
            else
                real_policy_core.solve(T, complexOperand(T, &factor, 1, 1, 1).asConst(), complexOperand(T, &checked, 1, 3, 2), if (is_lower) .lower else .upper);
            try std.testing.expectEqual(@as(usize, 1), info);
            try solveNear(T, solveRef(T, luValue(T, 2, 1)), checked[4], 0);
        }
    }
}

test "IEEE empty and native-width solves keep borrowed geometry separate from numerical classification" {
    inline for (.{ f32, f64, arithmetic.scalar.ComplexF32, arithmetic.scalar.ComplexF64 }) |T| {
        const R = arithmetic.scalar.Real(T);
        inline for (.{ true, false }) |is_lower| {
            var factor = [_]T{luValue(T, 2, 0)};
            var rhs = [_]T{luValue(T, 8, 0)};
            const wide = std.math.maxInt(usize);
            const f = complexOperand(T, &factor, 1, 1, wide);
            const b = complexOperand(T, &rhs, 1, 1, wide);
            try std.testing.expectEqual(@as(usize, 1), try f.check());
            try std.testing.expectEqual(@as(usize, 1), try b.check());
            ieeeSolve(T, f.asConst(), b, is_lower);
            try solveNear(T, solveRef(T, luValue(T, 2, 0)), rhs[0], 0);
            factor[0] = luValue(T, std.math.nan(R), std.math.inf(R));
            const before = factor;
            ieeeSolve(T, f.asConst(), complexOperand(T, &.{}, 1, 0, 0), is_lower);
            try expectPolicyBytes(T, &before, &factor);
            ieeeSolve(T, complexOperand(T, &.{}, 0, 0, 0).asConst(), complexOperand(T, &.{}, 0, @as(usize, std.math.maxInt(i32)) + 1, wide), is_lower);
        }
    }
}

// Structured MV expectations are built from literal dense matrices, not from
// the Core's structure decoder, compressed traversal or scalar helpers.
const StructuredKind = enum { symmetric, hermitian, triangular, diagonal };
fn structuredSemantics(kind: StructuredKind, lower: bool, unit: bool) sparse_product.MatrixSemantics {
    const half: sparse_product.Half = .{ .triangle = if (lower) .lower else .upper, .diagonal = if (unit) .unit else .stored };
    return switch (kind) {
        .symmetric => .{ .symmetric = half },
        .hermitian => .{ .hermitian = half },
        .triangular => .{ .triangular = half },
        .diagonal => .{ .diagonal = half.diagonal },
    };
}
fn structuredDense(comptime T: type, kind: StructuredKind, lower: bool, unit: bool) [4]ProductReference {
    const imaginary: f64 = if (sparse_scalar.isComplex(T)) 1 else 0;
    const upper: ProductReference = .{ .re = 2, .im = 3 * imaginary };
    const lower_value: ProductReference = .{ .re = 2, .im = -3 * imaginary };
    const zero: ProductReference = .{ .re = 0, .im = 0 };
    const d0: ProductReference = if (unit) .{ .re = 1, .im = 0 } else .{ .re = 1, .im = if (kind == .hermitian) 0 else 7 * imaginary };
    const d1: ProductReference = if (unit) .{ .re = 1, .im = 0 } else .{ .re = 4, .im = if (kind == .hermitian) 0 else -2 * imaginary };
    return switch (kind) {
        .symmetric => .{ d0, if (lower) lower_value else upper, if (lower) lower_value else upper, d1 },
        .hermitian => .{ d0, upper, lower_value, d1 },
        .triangular => .{ d0, if (lower) zero else upper, if (lower) lower_value else zero, d1 },
        .diagonal => .{ d0, zero, zero, d1 },
    };
}
fn structuredDenseResult(comptime T: type, matrix: [4]ProductReference, operation: sparse_pattern.Operation, x: [2]T, before: [2]T, alpha: T, beta: T) [2]ProductReference {
    var result: [2]ProductReference = undefined;
    for (&result, 0..) |*out, row| {
        var sum: ProductReference = .{ .re = 0, .im = 0 };
        for (x, 0..) |value, column| {
            var a = matrix[if (operation == .normal) 2 * row + column else 2 * column + row];
            if (operation == .adjoint) a.im = -a.im;
            sum = productReferenceAdd(sum, productReferenceMultiply(a, productReference(T, value)));
        }
        out.* = productReferenceAdd(productReferenceMultiply(productReference(T, alpha), sum), productReferenceMultiply(productReference(T, beta), productReference(T, before[row])));
    }
    return result;
}
fn expectStructuredNumber(comptime T: type, want: ProductReference, got: T) !void {
    const actual = productReference(T, got);
    const tolerance: f64 = if (T == f32 or T == sparse_checked.ComplexF32) 0.0001 else 0.00000000001;
    try std.testing.expectApproxEqAbs(want.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(want.im, actual.im, tolerance);
}
fn checkStructuredRepresentations(comptime T: type, comptime O: type, comptime I: type) !void {
    const kinds = [_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal };
    for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
        const first = if (axis == .row) productNumber(T, 1, 1) else productNumber(T, 3, -1);
        const second = if (axis == .row) productNumber(T, 1, 2) else productNumber(T, -1, -2);
        const third = if (axis == .row) productNumber(T, 3, -1) else productNumber(T, 1, 1);
        const fourth = if (axis == .row) productNumber(T, -1, -2) else productNumber(T, 1, 2);
        const source_values = [_]T{ first, productNumber(T, 1, 7), second, productNumber(T, 4, -2), third, fourth };
        for ([_]sparse_pattern.IndexBase{ .zero, .one }) |base| {
            for ([_]bool{ false, true }) |split| {
                const physical = if (split) [_]usize{ 1, 2, 3, 5, 6, 7 } else [_]usize{ 0, 1, 2, 3, 4, 5 };
                const capacity: usize = if (split) 8 else 6;
                var values: [8]T = @splat(productNumber(T, std.math.nan(f64), std.math.inf(f64)));
                var ordinals: [8]I = @splat(std.math.maxInt(I));
                for (physical, [_]usize{ 1, 0, 1, 1, 0, 0 }, source_values) |at, column, value| {
                    ordinals[at] = productIndex(I, column, base);
                    values[at] = value;
                }
                const offsets = [_]O{ productIndex(O, 0, base), productIndex(O, 3, base), productIndex(O, 6, base) };
                const starts = [_]O{ productIndex(O, 1, base), productIndex(O, 5, base) };
                const ends = [_]O{ productIndex(O, 4, base), productIndex(O, 8, base) };
                var a: sparse_pattern.Operand(T, O, I) = .{ .pattern = .{ .ranges = if (split) .{ .split = .{ .starts = &starts, .ends = &ends } } else .{ .contiguous = &offsets }, .ordinals = ordinals[0..capacity], .row_count = 2, .column_count = 2, .major_axis = axis, .index_base = base }, .values = values[0..capacity] };
                _ = try a.check();
                const values_before = std.mem.asBytes(&values).*;
                const x = [_]T{ productNumber(T, 1, 1), productNumber(T, 2, -1) };
                const alpha = productNumber(T, 2, -1);
                const beta = productNumber(T, -1, 1);
                for (kinds) |kind| for ([_]bool{ false, true }) |lower| {
                    for ([_]bool{ false, true }) |unit| for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                        a.operation = op;
                        var output = [_]T{ productNumber(T, -31, 29), productNumber(T, 7, -3), productNumber(T, 11, 2), productNumber(T, 37, -41) };
                        const before = output;
                        const want = structuredDenseResult(T, structuredDense(T, kind, lower, unit), op, x, .{ before[1], before[2] }, alpha, beta);
                        sparse_product.executeStructuredValidated(T, O, I, a, structuredSemantics(kind, lower, unit), &x, output[1..3], alpha, beta);
                        for (want, output[1..3]) |expected, actual| try expectStructuredNumber(T, expected, actual);
                        try productBytes(T, before[0..1], output[0..1]);
                        try productBytes(T, before[3..4], output[3..4]);
                    };
                };
                try std.testing.expectEqualSlices(u8, &values_before, std.mem.asBytes(&values));
            }
        }
    }
}

test "structured product independent dense oracle covers all scalars mixed index widths bases axes halves and operations" {
    @setEvalBranchQuota(50000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |O| {
            inline for (.{ i32, u32, i64, u64 }) |I| try checkStructuredRepresentations(T, O, I);
        }
    }
}

test "structured complex products distinguish symmetric Hermitian triangular and diagonal literal oracles" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const x = [_]T{ productNumber(T, 1, 1), productNumber(T, 2, -1) };
        const values = [_]T{ productNumber(T, 1, 0), productNumber(T, 2, 3), productNumber(T, 4, 0) };
        var a: sparse_pattern.Operand(T, i32, i32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 2, 3 } }, .ordinals = &.{ 0, 1, 1 }, .row_count = 2, .column_count = 2 }, .values = &values };
        _ = try a.check();
        const expected = [3][3][2]ProductReference{
            .{ .{ .{ .re = 8, .im = 5 }, .{ .re = 7, .im = 1 } }, .{ .{ .re = 8, .im = 5 }, .{ .re = 7, .im = 1 } }, .{ .{ .re = 2, .im = -7 }, .{ .re = 13, .im = -5 } } },
            .{ .{ .{ .re = 8, .im = 5 }, .{ .re = 13, .im = -5 } }, .{ .{ .re = 2, .im = -7 }, .{ .re = 7, .im = 1 } }, .{ .{ .re = 8, .im = 5 }, .{ .re = 13, .im = -5 } } },
            .{ .{ .{ .re = 8, .im = 5 }, .{ .re = 8, .im = -4 } }, .{ .{ .re = 1, .im = 1 }, .{ .re = 7, .im = 1 } }, .{ .{ .re = 1, .im = 1 }, .{ .re = 13, .im = -5 } } },
        };
        for ([_]StructuredKind{ .symmetric, .hermitian, .triangular }, 0..) |kind, k| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }, 0..) |op, o| {
                a.operation = op;
                var y: [2]T = @splat(productNumber(T, std.math.nan(f64), std.math.nan(f64)));
                sparse_product.executeStructuredValidated(T, i32, i32, a, structuredSemantics(kind, false, false), &x, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
                for (expected[k][o], y) |want, got| try expectStructuredNumber(T, want, got);
            }
        }
        a.values = &.{ productNumber(T, 1, 1), productNumber(T, std.math.nan(f64), std.math.nan(f64)), productNumber(T, 4, -2) };
        const diagonal = [3][2]ProductReference{ .{ .{ .re = 0, .im = 2 }, .{ .re = 6, .im = -8 } }, .{ .{ .re = 0, .im = 2 }, .{ .re = 6, .im = -8 } }, .{ .{ .re = 2, .im = 0 }, .{ .re = 10, .im = 0 } } };
        for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }, 0..) |op, o| {
            a.operation = op;
            var y: [2]T = undefined;
            sparse_product.executeStructuredValidated(T, i32, i32, a, .{ .diagonal = .stored }, &x, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
            for (diagonal[o], y) |want, got| try expectStructuredNumber(T, want, got);
        }
    }
}

test "structured unit identity survives empty physical storage and zero scales avoid numerical reads" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            var a: sparse_pattern.Operand(T, u64, i32) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 3, 3 }, .ends = &.{ 3, 3 } } }, .ordinals = &.{ -1, -1, -1 }, .row_count = 2, .column_count = 2, .major_axis = axis }, .values = &.{} };
            _ = try a.check();
            const poison = productNumber(T, std.math.nan(f64), std.math.nan(f64));
            const x = [_]T{ productNumber(T, 1, 1), productNumber(T, 2, -1) };
            for ([_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal }) |kind| {
                for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                    a.operation = op;
                    var y: [2]T = @splat(poison);
                    sparse_product.executeStructuredValidated(T, u64, i32, a, structuredSemantics(kind, false, true), &x, &y, productNumber(T, 2, 0), productNumber(T, -0.0, -0.0));
                    try expectStructuredNumber(T, productReference(T, productNumber(T, 2, 2)), y[0]);
                    try expectStructuredNumber(T, productReference(T, productNumber(T, 4, -2)), y[1]);
                    sparse_product.executeStructuredValidated(T, u64, i32, a, structuredSemantics(kind, true, false), &.{ poison, poison }, &y, poison, productNumber(T, 0, 0));
                    try productBytes(T, &.{ productNumber(T, 0, 0), productNumber(T, 0, 0) }, &y);
                    sparse_product.executeStructuredValidated(T, u64, i32, a, structuredSemantics(kind, true, true), &.{ poison, poison }, &y, productNumber(T, -0.0, -0.0), poison);
                    // beta is actually used here; its NaN is not an unused input.
                    try std.testing.expect(std.math.isNan(productReference(T, y[0]).re));
                }
            }
        }
    }
}

test "structured Hermitian imaginary diagonal and ignored stored unit values have no numerical dependency" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const nan = std.math.nan(f64);
        const x = [_]T{ productNumber(T, 1, 1), productNumber(T, 2, -1) };
        var values = [_]T{ productNumber(T, 1, nan), productNumber(T, 2, 3), productNumber(T, nan, nan), productNumber(T, 4, std.math.inf(f64)) };
        var a: sparse_pattern.Operand(T, i64, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 2, 4 } }, .ordinals = &.{ 0, 1, 0, 1 }, .row_count = 2, .column_count = 2 }, .values = &values };
        _ = try a.check();
        for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
            a.operation = op;
            var y: [2]T = undefined;
            sparse_product.executeStructuredValidated(T, i64, u32, a, structuredSemantics(.hermitian, false, false), &x, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
            const want = structuredDenseResult(T, structuredDense(T, .hermitian, false, false), op, x, .{ productNumber(T, 0, 0), productNumber(T, 0, 0) }, productNumber(T, 1, 0), productNumber(T, 0, 0));
            for (want, y) |expected, actual| try expectStructuredNumber(T, expected, actual);
        }
        values[0].re = @floatCast(nan);
        values[3].re = @floatCast(nan);
        for ([_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal }) |kind| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                a.operation = op;
                var y: [2]T = undefined;
                sparse_product.executeStructuredValidated(T, i64, u32, a, structuredSemantics(kind, false, true), &x, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
                const want = structuredDenseResult(T, structuredDense(T, kind, false, true), op, x, .{ productNumber(T, 0, 0), productNumber(T, 0, 0) }, productNumber(T, 1, 0), productNumber(T, 0, 0));
                for (want, y) |expected, actual| try expectStructuredNumber(T, expected, actual);
            }
        }
    }
}

test "structured overlapping nonmonotone ranges preserve per-reference coordinates and unit-first duplicate order" {
    const a: sparse_pattern.Operand(f64, u64, i32) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 2, 0, 1 }, .ends = &.{ 4, 2, 3 } } }, .ordinals = &.{ 1, 0, 0, 2 }, .row_count = 3, .column_count = 3 }, .values = &.{ std.math.nan(f64), 7, 11, 13 } };
    _ = try a.check();
    var result: [3]f64 = undefined;
    sparse_product.executeStructuredValidated(f64, u64, i32, a, structuredSemantics(.symmetric, true, true), &.{ 1, 2, 3 }, &result, 1, 0);
    try std.testing.expectEqualSlices(f64, &.{ 69, 9, 21 }, &result);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const large: f64 = if (T == f32 or T == sparse_checked.ComplexF32) 16777216 else 9007199254740992;
        const b: sparse_pattern.Operand(T, i32, u64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 2, 2 } }, .ordinals = &.{ 1, 1 }, .row_count = 2, .column_count = 2 }, .values = &.{ productNumber(T, large, 0), productNumber(T, -large, 0) } };
        _ = try b.check();
        var y: [2]T = undefined;
        sparse_product.executeStructuredValidated(T, i32, u64, b, structuredSemantics(.symmetric, false, true), &.{ productNumber(T, 1, 0), productNumber(T, 1, 0) }, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
        try expectStructuredNumber(T, .{ .re = 0, .im = 0 }, y[0]);
        try expectStructuredNumber(T, .{ .re = 0, .im = 0 }, y[1]);
    }
}

test "structured general delegation retains complete output bytes and native empty shape contracts" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const nan = productNumber(T, std.math.nan(f64), std.math.nan(f64));
        const scales = [_]T{ productNumber(T, 0, -0.0), productNumber(T, 1, 0), productNumber(T, -2, 1), productNumber(T, std.math.inf(f64), 0), nan };
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            var a: sparse_pattern.Operand(T, u64, i64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 3, 3 } }, .ordinals = &.{ 1, 0, 1 }, .row_count = 2, .column_count = 2, .major_axis = axis }, .values = &.{ productNumber(T, -0.0, -0.0), productNumber(T, 3, 2), nan } };
            _ = try a.check();
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                a.operation = op;
                for (scales) |alpha| for (scales) |beta| {
                    var old = [_]T{ productNumber(T, 31, -29), nan, productNumber(T, -0.0, -0.0), productNumber(T, 37, 41) };
                    var delegated = old;
                    sparse_product.executeValidated(T, u64, i64, a, &.{ productNumber(T, -1, 0), nan }, old[1..3], alpha, beta);
                    sparse_product.executeStructuredValidated(T, u64, i64, a, .general, &.{ productNumber(T, -1, 0), nan }, delegated[1..3], alpha, beta);
                    try productBytes(T, &old, &delegated);
                };
            }
        }
        const empty: sparse_pattern.Operand(T, i32, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{0} }, .ordinals = &.{}, .row_count = 0, .column_count = 0 }, .values = &.{} };
        _ = try empty.check();
        for ([_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal }) |kind| sparse_product.executeStructuredValidated(T, i32, u32, empty, structuredSemantics(kind, true, true), &.{}, &.{}, nan, nan);
    }
    try std.testing.expect(sparse_product.acceptsShape(.general, std.math.maxInt(usize), 0));
    try std.testing.expect(sparse_product.acceptsShape(.{ .diagonal = .unit }, std.math.maxInt(usize), std.math.maxInt(usize)));
    try std.testing.expect(!sparse_product.acceptsShape(.{ .diagonal = .unit }, std.math.maxInt(usize), 0));
}

test "structured duplicate diagonals preserve stored order and unit ignores every stored diagonal reference" {
    @setEvalBranchQuota(15000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const large: f64 = if (T == f32 or T == sparse_checked.ComplexF32) 16777216 else 9007199254740992;
        var values = [_]T{ productNumber(T, large, 0), productNumber(T, 1, 0), productNumber(T, -large, 0) };
        var a: sparse_pattern.Operand(T, i64, u64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 1, 4 } }, .ordinals = &.{ 1, 1, 1 }, .row_count = 1, .column_count = 1, .index_base = .one }, .values = &values };
        _ = try a.check();
        for ([_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal }) |kind| {
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                a.operation = op;
                values = .{ productNumber(T, large, 0), productNumber(T, 1, 0), productNumber(T, -large, 0) };
                var y = [_]T{productNumber(T, 99, 99)};
                sparse_product.executeStructuredValidated(T, i64, u64, a, structuredSemantics(kind, true, false), &.{productNumber(T, 1, 0)}, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
                try expectStructuredNumber(T, .{ .re = 0, .im = 0 }, y[0]);
                values = .{ productNumber(T, large, 0), productNumber(T, -large, 0), productNumber(T, 1, 0) };
                sparse_product.executeStructuredValidated(T, i64, u64, a, structuredSemantics(kind, true, false), &.{productNumber(T, 1, 0)}, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
                try expectStructuredNumber(T, .{ .re = 1, .im = 0 }, y[0]);
                @memset(&values, productNumber(T, std.math.nan(f64), std.math.inf(f64)));
                sparse_product.executeStructuredValidated(T, i64, u64, a, structuredSemantics(kind, true, true), &.{productNumber(T, 2, 1)}, &y, productNumber(T, 1, 0), productNumber(T, 0, 0));
                try expectStructuredNumber(T, productReference(T, productNumber(T, 2, 1)), y[0]);
            }
        }
    }
}

test "structured implicit complex identity avoids synthetic one times input and specifies signed zero addition" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        var a: sparse_pattern.Operand(T, u32, i64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 0 } }, .ordinals = &.{}, .row_count = 1, .column_count = 1 }, .values = &.{} };
        _ = try a.check();
        for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
            a.operation = op;
            var y = [_]T{productNumber(T, 99, 99)};
            sparse_product.executeStructuredValidated(T, u32, i64, a, .{ .diagonal = .unit }, &.{productNumber(T, std.math.inf(f64), 1)}, &y, productNumber(T, 0, 1), productNumber(T, 0, 0));
            try std.testing.expect(std.math.isNan(y[0].re));
            try std.testing.expect(std.math.isPositiveInf(y[0].im));
            y[0] = productNumber(T, -0.0, -0.0);
            sparse_product.executeStructuredValidated(T, u32, i64, a, .{ .diagonal = .unit }, &.{productNumber(T, -0.0, -0.0)}, &y, productNumber(T, 1, -0.0), productNumber(T, 1, 0));
            try productBytes(T, &.{productNumber(T, -0.0, 0)}, &y);
        }
    }
}

// Regular-panel expectations use logical coordinates and explicit dense values;
// none calls another product entry to compute an expected numerical result.
fn panelPosition(layout: dense.Layout, row: usize, column: usize, ld: usize) usize {
    return if (layout == .row_major) row * ld + column else column * ld + row;
}
fn panelShape(layout: dense.Layout, rows: usize, columns: usize) dense.PanelShape {
    return .{ .row_count = rows, .column_count = columns, .leading_dimension = (if (layout == .row_major) columns else rows) + 2, .layout = layout };
}

test "regular dense panels retain Matrix policy and check layout capacity empty offsets and overflow" {
    for ([_]dense.Layout{ .row_major, .column_major }) |layout| {
        var values = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        const shape: dense.PanelShape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 4, .layout = layout };
        const panel: dense.Panel(f64) = .{ .values = &values, .shape = shape };
        const used: usize = if (layout == .row_major) 7 else 10;
        try std.testing.expectEqual(used, try panel.check());
        const column = panel.column(2);
        try std.testing.expectEqual(values[panelPosition(layout, 1, 2, 4)], column.at(1).*);
        try std.testing.expectEqual(column.at(1).*, panel.asConst().column(2).at(1));
        var short = panel;
        short.values = values[0 .. used - 1];
        try std.testing.expectError(error.BufferTooSmall, short.check());
        short.shape.leading_dimension = 1;
        try std.testing.expectError(error.InvalidLeadingDimension, short.check());
        const huge: dense.PanelShape = .{ .row_count = 2, .column_count = 3, .leading_dimension = std.math.maxInt(usize), .layout = layout };
        try std.testing.expectError(error.SizeOverflow, huge.extent());
        const empty: dense.Panel(f64) = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = std.math.maxInt(usize), .leading_dimension = std.math.maxInt(usize), .layout = layout } };
        try std.testing.expectEqual(@as(usize, 0), try empty.check());
        try std.testing.expectEqual(@as(usize, 0), empty.column(std.math.maxInt(usize) - 1).length);
        try std.testing.expectEqual(@as(usize, 0), empty.asConst().column(std.math.maxInt(usize) - 1).length);
        const no_columns: dense.ConstPanel(f64) = .{ .values = &.{}, .shape = .{ .row_count = std.math.maxInt(usize), .column_count = 0, .leading_dimension = 0, .layout = layout } };
        try std.testing.expectEqual(@as(usize, 0), try no_columns.check());
        var one = [_]f64{13};
        const wide: dense.Panel(f64) = .{ .values = &one, .shape = .{ .row_count = 1, .column_count = 1, .leading_dimension = std.math.maxInt(usize), .layout = layout } };
        try std.testing.expectEqual(@as(usize, 1), try wide.check());
        try std.testing.expectEqual(@as(f64, 13), wide.column(0).at(0).*);
    }
    try std.testing.expectEqual(@as(usize, 0), try dense.extent(7, 0, 0));
    try std.testing.expectError(error.InvalidLeadingDimension, dense.extent(7, 1, 0));
}

fn checkRectangularPanels(comptime T: type, comptime O: type, comptime I: type) !void {
    const logical = [_]f64{ 2, 0, -1, 3, 4, 0 };
    for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
        for ([_]sparse_pattern.IndexBase{ .zero, .one }) |base| {
            const major_count: usize = if (axis == .row) 2 else 3;
            // Unsorted duplicate entries and inactive physical gaps.
            const lengths = if (axis == .row) [_]usize{ 3, 2, 0 } else [_]usize{ 3, 1, 1 };
            const minors = if (axis == .row) [_]usize{ 2, 0, 0, 1, 0 } else [_]usize{ 1, 0, 0, 1, 0 };
            const numbers = if (axis == .row) [_]f64{ -1, 3, -1, 4, 3 } else [_]f64{ 3, 3, -1, 4, -1 };
            var starts: [3]O = undefined;
            var ends: [3]O = undefined;
            var ordinals: [8]I = @splat(std.math.maxInt(I));
            var values: [8]T = @splat(productNumber(T, std.math.nan(f64), std.math.nan(f64)));
            var next: usize = 0;
            var source: usize = 0;
            for (0..major_count) |major| {
                next += 1;
                starts[major] = productIndex(O, next, base);
                for (0..lengths[major]) |_| {
                    ordinals[next] = productIndex(I, minors[source], base);
                    values[next] = productNumber(T, numbers[source], 0);
                    next += 1;
                    source += 1;
                }
                ends[major] = productIndex(O, next, base);
            }
            var a: sparse_pattern.Operand(T, O, I) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = starts[0..major_count], .ends = ends[0..major_count] } }, .ordinals = &ordinals, .row_count = 2, .column_count = 3, .major_axis = axis, .index_base = base }, .values = &values };
            _ = try a.check();
            const source_before = values;
            for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                a.operation = op;
                const m: usize = if (op == .normal) 2 else 3;
                const k: usize = if (op == .normal) 3 else 2;
                for ([_]usize{ 1, 2, 3, 9 }) |rhs| for ([_]dense.Layout{ .row_major, .column_major }) |bl| for ([_]dense.Layout{ .row_major, .column_major }) |cl| {
                    const bs = panelShape(bl, k, rhs);
                    const cs = panelShape(cl, m, rhs);
                    var b: [96]T = @splat(productNumber(T, -71, 73));
                    var c: [96]T = @splat(productNumber(T, 79, -83));
                    var expected = c;
                    for (0..k) |row| for (0..rhs) |column| {
                        b[panelPosition(bl, row, column, bs.leading_dimension)] = productNumber(T, @floatFromInt(3 * row + column + 1), @floatFromInt(column));
                    };
                    for (0..m) |row| for (0..rhs) |column| {
                        const at = panelPosition(cl, row, column, cs.leading_dimension);
                        c[at] = productNumber(T, @floatFromInt(row + column), 0);
                        var sum: ProductReference = .{ .re = 0, .im = 0 };
                        for (0..k) |inner| {
                            const entry = logical[if (op == .normal) row * 3 + inner else inner * 3 + row];
                            sum = productReferenceAdd(sum, productReferenceMultiply(.{ .re = entry, .im = 0 }, productReference(T, b[panelPosition(bl, inner, column, bs.leading_dimension)])));
                        }
                        const want = productReferenceAdd(productReferenceMultiply(.{ .re = 2, .im = 0 }, sum), productReferenceMultiply(.{ .re = -1, .im = 0 }, productReference(T, c[at])));
                        expected[at] = productNumber(T, want.re, want.im);
                    };
                    const b_before = b;
                    const input: dense.ConstPanel(T) = .{ .values = &b, .shape = bs };
                    const output: dense.Panel(T) = .{ .values = &c, .shape = cs };
                    _ = try input.check();
                    _ = try output.check();
                    sparse_product.executePanelValidated(T, O, I, a, .general, input, output, productNumber(T, 2, 0), productNumber(T, -1, 0));
                    try productBytes(T, &expected, &c);
                    try productBytes(T, &b_before, &b);
                };
            }
            try productBytes(T, &source_before, &values);
        }
    }
}

test "panel general products match independent rectangular dense oracle across layouts lanes axes bases and mixed widths" {
    @setEvalBranchQuota(50000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        try checkRectangularPanels(T, i32, u64);
        try checkRectangularPanels(T, u32, i64);
        try checkRectangularPanels(T, i64, u32);
        try checkRectangularPanels(T, u64, i32);
    }
}

test "panel structured products use independent dense expectations for both halves unit modes and transforms" {
    @setEvalBranchQuota(50000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        for ([_]sparse_pattern.MajorAxis{ .row, .column }) |axis| {
            const upper = productNumber(T, 2, 3);
            const lower = productNumber(T, 2, -3);
            const values = [_]T{ if (axis == .row) upper else lower, productNumber(T, 1, 7), productNumber(T, 4, -2), if (axis == .row) lower else upper };
            var a: sparse_pattern.Operand(T, i64, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 1, 3, 5 } }, .ordinals = &.{ 2, 1, 2, 1 }, .row_count = 2, .column_count = 2, .major_axis = axis, .index_base = .one }, .values = &values };
            _ = try a.check();
            for ([_]StructuredKind{ .symmetric, .hermitian, .triangular, .diagonal }) |kind| for ([_]bool{ false, true }) |lower_half| for ([_]bool{ false, true }) |unit| for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
                a.operation = op;
                for ([_]usize{ 1, 2, 3, 9 }) |rhs| for ([_]dense.Layout{ .row_major, .column_major }) |layout| {
                    const bs = panelShape(layout, 2, rhs);
                    const cs = panelShape(if (layout == .row_major) .column_major else .row_major, 2, rhs);
                    var b: [96]T = @splat(productNumber(T, 101, -103));
                    var c: [96]T = @splat(productNumber(T, -107, 109));
                    const padding_before = c;
                    var expected: [18]ProductReference = undefined;
                    for (0..rhs) |column| {
                        const x = [_]T{ productNumber(T, @floatFromInt(column + 1), 1), productNumber(T, 2, -1) };
                        const old = [_]T{ productNumber(T, 7, -3), productNumber(T, 11, 2) };
                        const want = structuredDenseResult(T, structuredDense(T, kind, lower_half, unit), op, x, old, productNumber(T, 2, -1), productNumber(T, -1, 1));
                        for (0..2) |row| {
                            b[panelPosition(bs.layout, row, column, bs.leading_dimension)] = x[row];
                            c[panelPosition(cs.layout, row, column, cs.leading_dimension)] = old[row];
                            expected[row * rhs + column] = want[row];
                        }
                    }
                    const input: dense.ConstPanel(T) = .{ .values = &b, .shape = bs };
                    const output: dense.Panel(T) = .{ .values = &c, .shape = cs };
                    _ = try input.check();
                    _ = try output.check();
                    sparse_product.executePanelValidated(T, i64, u32, a, structuredSemantics(kind, lower_half, unit), input, output, productNumber(T, 2, -1), productNumber(T, -1, 1));
                    var touched: [96]bool = @splat(false);
                    for (0..2) |row| for (0..rhs) |column| {
                        const at = panelPosition(cs.layout, row, column, cs.leading_dimension);
                        touched[at] = true;
                        try expectStructuredNumber(T, expected[row * rhs + column], c[at]);
                    };
                    for (touched, 0..) |changed, at| if (!changed) {
                        try productBytes(T, padding_before[at .. at + 1], c[at .. at + 1]);
                    };
                };
            };
        }
    }
}

test "panel implicit unit and zero scale paths preserve nonreads and unused padding" {
    @setEvalBranchQuota(10000);
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const poison = productNumber(T, std.math.nan(f64), std.math.inf(f64));
        var a: sparse_pattern.Operand(T, u64, i32) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 3, 3 }, .ends = &.{ 3, 3 } } }, .ordinals = &.{ -1, -1, -1 }, .row_count = 2, .column_count = 2 }, .values = &.{} };
        _ = try a.check(); // Empty intervals beyond values are legal.
        for ([_]dense.Layout{ .row_major, .column_major }) |layout| {
            const shape = panelShape(layout, 2, 3);
            var x: [16]T = @splat(poison);
            var y: [16]T = @splat(poison);
            for (0..2) |row| for (0..3) |column| {
                x[panelPosition(layout, row, column, shape.leading_dimension)] = productNumber(T, @floatFromInt(row + column + 1), 0);
            };
            const input: dense.ConstPanel(T) = .{ .values = &x, .shape = shape };
            const output: dense.Panel(T) = .{ .values = &y, .shape = shape };
            _ = try input.check();
            _ = try output.check();
            const before = y;
            sparse_product.executePanelValidated(T, u64, i32, a, .{ .diagonal = .unit }, input, output, productNumber(T, 2, 0), productNumber(T, 0, 0));
            var expected = before;
            for (0..2) |row| for (0..3) |column| {
                expected[panelPosition(layout, row, column, shape.leading_dimension)] = productNumber(T, @floatFromInt(2 * (row + column + 1)), 0);
            };
            try productBytes(T, &expected, &y);
            @memset(&x, poison);
            y = before;
            sparse_product.executePanelValidated(T, u64, i32, a, .{ .diagonal = .unit }, input, output, productNumber(T, -0.0, -0.0), productNumber(T, -0.0, -0.0));
            for (0..2) |row| for (0..3) |column| {
                expected[panelPosition(layout, row, column, shape.leading_dimension)] = productNumber(T, 0, 0);
            };
            try productBytes(T, &expected, &y);
        }
        a.pattern.ranges = .{ .split = .{ .starts = &.{ 3, 3 }, .ends = &.{ 4, 3 } } };
        try std.testing.expectError(error.InvalidOffsets, a.check());
    }
}

test "panel empty K and empty output never derive unused huge column offsets" {
    const empty_k: sparse_pattern.Operand(f64, i64, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 0, 0 } }, .ordinals = &.{}, .row_count = 2, .column_count = 0 }, .values = &.{} };
    _ = try empty_k.check();
    var y = [_]f64{ 1, 2, 91, 3, 4, 92, 5, 6, 93 };
    const input: dense.ConstPanel(f64) = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    const output: dense.Panel(f64) = .{ .values = &y, .shape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 3, .layout = .column_major } };
    _ = try input.check();
    _ = try output.check();
    sparse_product.executePanelValidated(f64, i64, u32, empty_k, .general, input, output, std.math.inf(f64), -2);
    try std.testing.expectEqualSlices(f64, &.{ -2, -4, 91, -6, -8, 92, -10, -12, 93 }, &y);
    const empty: sparse_pattern.Operand(f64, i64, u32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{0} }, .ordinals = &.{}, .row_count = 0, .column_count = 0 }, .values = &.{} };
    const huge_shape: dense.PanelShape = .{ .row_count = 0, .column_count = std.math.maxInt(usize), .leading_dimension = std.math.maxInt(usize), .layout = .column_major };
    sparse_product.executePanelValidated(f64, i64, u32, empty, .general, .{ .values = &.{}, .shape = huge_shape }, .{ .values = &.{}, .shape = huge_shape }, 1, 0);
}

test "panel gathered empty majors and split repeated references preserve stored arithmetic order" {
    inline for (.{ f32, f64, sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const large: f64 = if (T == f32 or T == sparse_checked.ComplexF32) 16777216 else 9007199254740992;
        const a: sparse_pattern.Operand(T, i32, u64) = .{ .pattern = .{ .ranges = .{ .split = .{ .starts = &.{ 0, 3, 0 }, .ends = &.{ 3, 3, 3 } } }, .ordinals = &.{ 0, 0, 0 }, .row_count = 3, .column_count = 1 }, .values = &.{ productNumber(T, large, 0), productNumber(T, 1, 0), productNumber(T, -large, 0) } };
        _ = try a.check();
        var y: [12]T = @splat(productNumber(T, 97, -101));
        const input: dense.ConstPanel(T) = .{ .values = &.{ productNumber(T, 1, 0), productNumber(T, 1, 0), productNumber(T, 1, 0) }, .shape = .{ .row_count = 1, .column_count = 3, .leading_dimension = 3, .layout = .row_major } };
        const output: dense.Panel(T) = .{ .values = &y, .shape = .{ .row_count = 3, .column_count = 3, .leading_dimension = 4, .layout = .row_major } };
        const before = y;
        sparse_product.executePanelValidated(T, i32, u64, a, .general, input, output, productNumber(T, 1, 0), productNumber(T, 0, 0));
        var expected = before;
        for (0..3) |row| for (0..3) |column| {
            expected[row * 4 + column] = productNumber(T, 0, 0);
        };
        try productBytes(T, &expected, &y);
    }
    const a: sparse_pattern.Operand(f64, u64, i32) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 1, 1 } }, .ordinals = &.{0}, .row_count = 2, .column_count = 1 }, .values = &.{2} };
    var y = [_]f64{ 8, 9, 81, 10, 11, 82, 12, 13, 83 };
    sparse_product.executePanelValidated(f64, u64, i32, a, .general, .{ .values = &.{ 1, 1, 1 }, .shape = .{ .row_count = 1, .column_count = 3, .leading_dimension = 1, .layout = .column_major } }, .{ .values = &y, .shape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 3, .layout = .column_major } }, std.math.inf(f64), 0);
    for ([_]usize{ 0, 3, 6 }) |at| {
        try std.testing.expect(std.math.isPositiveInf(y[at]));
        try std.testing.expect(std.math.isNan(y[at + 1])); // alpha * zero in empty gathered row.
        try std.testing.expectEqual(@as(f64, @floatFromInt(81 + at / 3)), y[at + 2]);
    }
}

test "panel Hermitian literal complex oracle ignores imaginary diagonal and opposite triangle poison" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const poison = productNumber(T, std.math.nan(f64), std.math.nan(f64));
        var a: sparse_pattern.Operand(T, i32, u64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 2, 4 } }, .ordinals = &.{ 1, 0, 1, 0 }, .row_count = 2, .column_count = 2 }, .values = &.{ poison, productNumber(T, 2, std.math.nan(f64)), productNumber(T, 3, std.math.inf(f64)), productNumber(T, 1, 2) } };
        _ = try a.check();
        const x = [_]T{ productNumber(T, 1, 0), productNumber(T, 0, 1), productNumber(T, 1, 0), productNumber(T, 2, 0), productNumber(T, -1, 0), productNumber(T, 0, 0) };
        const normal = [_]ProductReference{ .{ .re = 4, .im = -4 }, .{ .re = -1, .im = 4 }, .{ .re = 2, .im = 0 }, .{ .re = 7, .im = 2 }, .{ .re = -5, .im = 1 }, .{ .re = 1, .im = 2 } };
        const transpose = [_]ProductReference{ .{ .re = 4, .im = 4 }, .{ .re = -1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 7, .im = -2 }, .{ .re = -1, .im = 1 }, .{ .re = 1, .im = -2 } };
        for ([_]sparse_pattern.Operation{ .normal, .transposed, .adjoint }) |op| {
            a.operation = op;
            var y: [8]T = @splat(poison);
            sparse_product.executePanelValidated(T, i32, u64, a, .{ .hermitian = .{ .triangle = .lower, .diagonal = .stored } }, .{ .values = &x, .shape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 3, .layout = .row_major } }, .{ .values = &y, .shape = .{ .row_count = 2, .column_count = 3, .leading_dimension = 4, .layout = .row_major } }, productNumber(T, 1, 0), productNumber(T, 0, 0));
            const want = if (op == .transposed) transpose else normal;
            for (0..2) |row| for (0..3) |column| {
                try expectStructuredNumber(T, want[row * 3 + column], y[row * 4 + column]);
            };
            try productBytes(T, &.{poison}, y[3..4]);
            try productBytes(T, &.{poison}, y[7..8]);
        }
    }
}

test "panel implicit complex identity retains infinity and signed zero expression choices in two lanes and tail" {
    inline for (.{ sparse_checked.ComplexF32, sparse_checked.ComplexF64 }) |T| {
        const a: sparse_pattern.Operand(T, u32, i64) = .{ .pattern = .{ .ranges = .{ .contiguous = &.{ 0, 0 } }, .ordinals = &.{}, .row_count = 1, .column_count = 1 }, .values = &.{} };
        const x = [_]T{ productNumber(T, std.math.inf(f64), 1), productNumber(T, std.math.inf(f64), 1), productNumber(T, std.math.inf(f64), 1) };
        var y = [_]T{ productNumber(T, 3, 4), productNumber(T, 3, 4), productNumber(T, 3, 4) };
        const shape: dense.PanelShape = .{ .row_count = 1, .column_count = 3, .leading_dimension = 3, .layout = .row_major };
        sparse_product.executePanelValidated(T, u32, i64, a, .{ .diagonal = .unit }, .{ .values = &x, .shape = shape }, .{ .values = &y, .shape = shape }, productNumber(T, 0, 1), productNumber(T, 0, 0));
        for (y) |value| {
            try std.testing.expect(std.math.isNan(value.re)); // 0*Inf - 1*1
            try std.testing.expect(std.math.isPositiveInf(value.im)); // No synthetic (1+0i)*x.
        }
        const z = [_]T{ productNumber(T, -0.0, -0.0), productNumber(T, -0.0, -0.0), productNumber(T, -0.0, -0.0) };
        @memset(&y, productNumber(T, -0.0, -0.0));
        sparse_product.executePanelValidated(T, u32, i64, a, .{ .diagonal = .unit }, .{ .values = &z, .shape = shape }, .{ .values = &y, .shape = shape }, productNumber(T, 1, -0.0), productNumber(T, 1, 0));
        for (y) |value| try productBytes(T, &.{productNumber(T, -0.0, 0.0)}, &.{value});
    }
}
