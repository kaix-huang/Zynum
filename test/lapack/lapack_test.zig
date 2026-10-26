// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const lapack = @import("zynum-lapack");

comptime {
    _ = @import("bidiagonal_partial_test.zig");
    _ = @import("general_operator_test.zig");
    _ = @import("lanczos_chain_test.zig");
    _ = @import("operator_chain_boundary_test.zig");
    _ = @import("interop/fft_managed_test.zig");
    _ = @import("interop/fftw_binding_test.zig");
    _ = @import("interop/fftw_scalar_binding_test.zig");
    _ = @import("complex_test.zig");
    _ = @import("least_squares_test.zig");
    _ = @import("symmetric_eigen_test.zig");
    _ = @import("interop/symmetric_generalized_adapter_test.zig");
    _ = @import("interop/evd_policy_test.zig");
    _ = @import("interop/evd_adapter_test.zig");
    _ = @import("svd_test.zig");
    _ = @import("complex_svd_test.zig");
    _ = @import("complex_rank_least_squares_test.zig");
    _ = @import("complex_svd_facade_test.zig");
    _ = @import("orthonormal_completion_test.zig");
    _ = @import("interop/gesvd_adapter_test.zig");
    _ = @import("interop/gesvd_raw_boundary_test.zig");
    _ = @import("bidiagonal_test.zig");
    _ = @import("qr_test.zig");
    _ = @import("optimized_paths_test.zig");
    _ = @import("interop/potrf_adapter_test.zig");
    _ = @import("interop/potrs_adapter_test.zig");
    _ = @import("interop/managed_adapter_test.zig");
    _ = @import("interop/shared_services_test.zig");
    _ = @import("interop/raw_access_test.zig");
    _ = @import("interop/getrf_adapter_test.zig");
    _ = @import("interop/getrs_adapter_test.zig");
    _ = @import("interop/trtrs_adapter_test.zig");
    _ = @import("interop/trtri_adapter_test.zig");
    _ = @import("interop/getri_adapter_test.zig");
    _ = @import("interop/potri_adapter_test.zig");
    _ = @import("interop/gesv_adapter_test.zig");
    _ = @import("interop/stev_adapter_test.zig");
    _ = @import("interop/gelsd_policy_test.zig");
    _ = @import("interop/complex_gelsd_workspace_test.zig");
    _ = @import("interop/complex_gelsd_query_test.zig");
    _ = @import("interop/complex_gelsd_column_test.zig");
    _ = @import("interop/complex_gelsd_adapter_test.zig");
    _ = @import("interop/complex_gelsd_shared_projection_test.zig");
    _ = @import("tridiagonal_eigen_test.zig");
}

fn tolerance(comptime T: type) T {
    return if (T == f32) 2e-5 else 2e-13;
}

fn near(comptime T: type, expected: T, actual: T) !void {
    try std.testing.expectApproxEqAbs(expected, actual, tolerance(T) * @max(@as(T, 1), @abs(expected)));
}

fn mutable(comptime T: type, values: []T, rows: usize, columns: usize, ld: usize) lapack.Matrix(T) {
    return .{
        .values = values,
        .row_count = @intCast(rows),
        .column_count = @intCast(columns),
        .leading_dimension = @intCast(ld),
    };
}

fn makeRhs(comptime T: type, a: []const T, x: []const T, b: []T, n: usize, nrhs: usize, ld: usize) void {
    for (0..nrhs) |column| {
        for (0..n) |row| {
            var sum: T = 0;
            for (0..n) |k| sum += a[row + k * n] * x[k + column * n];
            b[row + column * ld] = sum;
        }
    }
}

fn residual(comptime T: type, a: []const T, b: []const T, x: []const T, n: usize, nrhs: usize, ld: usize) !void {
    for (0..nrhs) |column| {
        for (0..n) |row| {
            var product: T = 0;
            var scale = @abs(b[row + column * ld]);
            for (0..n) |k| {
                product += a[row + k * n] * x[k + column * ld];
                scale += @abs(a[row + k * n]) * @abs(x[k + column * ld]);
            }
            try std.testing.expect(@abs(product - b[row + column * ld]) <= tolerance(T) * @max(@as(T, 1), scale));
        }
    }
}

fn choleskyCase(comptime T: type, triangle: lapack.Triangle) !void {
    const a = [_]T{ 4, 2, -2, 2, 10, 5, -2, 5, 21 };
    const expected = [_]T{ 1, -2, 3, -1, 0.5, 2 };
    const padding: T = -987;
    var data: [15]T = @splat(padding);
    for (0..3) |column| {
        for (0..3) |row| {
            if ((triangle == .lower and row >= column) or (triangle == .upper and row <= column)) {
                data[row + column * 5] = a[row + column * 3];
            }
        }
    }
    const view = mutable(T, &data, 3, 3, 5);
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(T, view, triangle));
    for (0..3) |column| {
        for (0..3) |row| {
            var reconstruction: T = 0;
            for (0..@min(row, column) + 1) |k| {
                const lrow = if (triangle == .lower) data[row + k * 5] else data[k + row * 5];
                const lcolumn = if (triangle == .lower) data[column + k * 5] else data[k + column * 5];
                reconstruction += lrow * lcolumn;
            }
            try near(T, a[row + column * 3], reconstruction);
            if ((triangle == .lower and row < column) or (triangle == .upper and row > column)) {
                try std.testing.expectEqual(padding, data[row + column * 5]);
            }
        }
        try std.testing.expectEqual(padding, data[3 + column * 5]);
        try std.testing.expectEqual(padding, data[4 + column * 5]);
    }
    var b: [10]T = @splat(padding);
    makeRhs(T, &a, &expected, &b, 3, 2, 5);
    const original_b = b;
    const original_factor = data;
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(T, view.asConst(), mutable(T, &b, 3, 2, 5), triangle));
    try residual(T, &a, &original_b, &b, 3, 2, 5);
    for (0..2) |column| {
        for (0..3) |row| try near(T, expected[row + column * 3], b[row + column * 5]);
        try std.testing.expectEqual(padding, b[3 + column * 5]);
        try std.testing.expectEqual(padding, b[4 + column * 5]);
    }
    try std.testing.expectEqualSlices(T, &original_factor, &data);
}

test "Cholesky reconstructs and solves padded upper and lower real matrices" {
    inline for (.{ f32, f64 }) |T| {
        try choleskyCase(T, .lower);
        try choleskyCase(T, .upper);
    }
}

test "Cholesky reports a one-based nonpositive pivot and preserves other storage" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
            var a = [_]T{ 4, 2, 777, 2, 1, 777 };
            const untouched = if (triangle == .lower) a[3] else a[1];
            try std.testing.expectEqual(@as(usize, 2), try lapack.choleskyFactorize(T, mutable(T, &a, 2, 2, 3), triangle));
            try std.testing.expectEqual(@as(T, 2), a[0]);
            try std.testing.expectEqual(untouched, if (triangle == .lower) a[3] else a[1]);
            try std.testing.expectEqual(@as(T, 777), a[2]);
            try std.testing.expectEqual(@as(T, 777), a[5]);
            var negative = [_]T{-1};
            try std.testing.expectEqual(@as(usize, 1), try lapack.choleskyFactorize(T, mutable(T, &negative, 1, 1, 1), triangle));
            try std.testing.expectEqual(@as(T, -1), negative[0]);
        }
    }
}

test "Cholesky solve checks every diagonal before changing any RHS" {
    var factor = [_]f64{ 2, 1, 777, 0 };
    var rhs = [_]f64{ 5, 7, 11, 13 };
    const before = rhs;
    try std.testing.expectEqual(@as(usize, 2), try lapack.choleskySolve(f64, mutable(f64, &factor, 2, 2, 2).asConst(), mutable(f64, &rhs, 2, 2, 2), .lower));
    try std.testing.expectEqualSlices(f64, &before, &rhs);
}

fn luReconstruction(comptime T: type, comptime rows: usize, comptime columns: usize) !void {
    const ld = rows + 2;
    const steps: usize = @min(rows, columns);
    var original: [rows * columns]T = undefined;
    for (0..columns) |column| {
        for (0..rows) |row| {
            const perturbation: i32 = @as(i32, @intCast((row * 3 + column * 7) % 11)) - 5;
            original[row + column * rows] = if (row == column) 5 else @as(T, @floatFromInt(perturbation)) * 0.125;
        }
        if (rows > 1) std.mem.swap(T, &original[column * rows], &original[rows - 1 + column * rows]);
    }
    var data: [ld * columns]T = @splat(-987);
    for (0..columns) |column| {
        for (0..rows) |row| data[row + column * ld] = original[row + column * rows];
    }
    var pivots: [steps + 2]lapack.PivotInt = @splat(-77);
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(T, mutable(T, &data, rows, columns, ld), &pivots));
    var permuted = original;
    for (0..steps) |k| {
        const pivot: usize = @intCast(pivots[k]);
        try std.testing.expect(pivot >= k and pivot < rows);
        if (pivot != k) {
            for (0..columns) |column| std.mem.swap(T, &permuted[k + column * rows], &permuted[pivot + column * rows]);
        }
    }
    try std.testing.expectEqual(@as(lapack.PivotInt, @intCast(rows - 1)), pivots[0]);
    for (0..columns) |column| {
        for (0..rows) |row| {
            var product: T = 0;
            for (0..steps) |k| {
                const l: T = if (row == k) 1 else if (row > k) data[row + k * ld] else 0;
                const u: T = if (k <= column) data[k + column * ld] else 0;
                product += l * u;
            }
            try near(T, permuted[row + column * rows], product);
        }
        try std.testing.expectEqual(@as(T, -987), data[rows + column * ld]);
        try std.testing.expectEqual(@as(T, -987), data[rows + 1 + column * ld]);
    }
    try std.testing.expectEqual(@as(lapack.PivotInt, -77), pivots[steps]);
    try std.testing.expectEqual(@as(lapack.PivotInt, -77), pivots[steps + 1]);
}

test "LU reconstructs pivoted square tall and wide matrices with padding" {
    inline for (.{ f32, f64 }) |T| {
        try luReconstruction(T, 4, 4);
        try luReconstruction(T, 5, 3);
        try luReconstruction(T, 3, 5);
        try luReconstruction(T, 1, 3);
        try luReconstruction(T, 3, 1);
    }
}

test "LU multiple RHS solve uses the ordered pivot sequence and preserves inputs" {
    inline for (.{ f32, f64 }) |T| {
        const original = [_]T{ 0, 1, 2, 2, 0, 1, 1, 3, 1 };
        const expected = [_]T{ 1, -2, 3, -1, 0.5, 2 };
        var factor: [15]T = @splat(-987);
        for (0..3) |column| {
            for (0..3) |row| factor[row + column * 5] = original[row + column * 3];
        }
        var pivots: [3]lapack.PivotInt = undefined;
        const a = mutable(T, &factor, 3, 3, 5);
        try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(T, a, &pivots));
        try std.testing.expectEqual(@as(lapack.PivotInt, 2), pivots[0]);
        var b: [10]T = @splat(-987);
        makeRhs(T, &original, &expected, &b, 3, 2, 5);
        const original_b = b;
        const saved_factor = factor;
        const saved_pivots = pivots;
        try std.testing.expectEqual(@as(usize, 0), try lapack.luSolve(T, a.asConst(), &pivots, mutable(T, &b, 3, 2, 5)));
        try residual(T, &original, &original_b, &b, 3, 2, 5);
        for (0..2) |column| {
            for (0..3) |row| try near(T, expected[row + column * 3], b[row + column * 5]);
            try std.testing.expectEqual(@as(T, -987), b[3 + column * 5]);
            try std.testing.expectEqual(@as(T, -987), b[4 + column * 5]);
        }
        try std.testing.expectEqualSlices(T, &saved_factor, &factor);
        try std.testing.expectEqualSlices(lapack.PivotInt, &saved_pivots, &pivots);
    }
}

test "LU singular detection is one-based and solve leaves RHS unchanged" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 1, 2, 2, 4 };
        var pivots = [_]lapack.PivotInt{ -77, -77, -77 };
        const view = mutable(T, &a, 2, 2, 2);
        try std.testing.expectEqual(@as(usize, 2), try lapack.luFactorize(T, view, &pivots));
        try std.testing.expectEqual(@as(lapack.PivotInt, 1), pivots[0]);
        try std.testing.expectEqual(@as(lapack.PivotInt, -77), pivots[2]);
        var b = [_]T{ 1, 2, 3, 4 };
        const before = b;
        try std.testing.expectEqual(@as(usize, 2), try lapack.luSolve(T, view.asConst(), pivots[0..2], mutable(T, &b, 2, 2, 2)));
        try std.testing.expectEqualSlices(T, &before, &b);
        var zero = [_]T{ 0, 0, 0, 0 };
        try std.testing.expectEqual(@as(usize, 1), try lapack.luFactorize(T, mutable(T, &zero, 2, 2, 2), &pivots));
    }
}

test "Singular rectangular LU completes later pivots and reconstructs the permuted matrix" {
    inline for (.{ f32, f64 }) |T| {
        // The first column is zero; later steps must still pivot and eliminate.
        // Exercise both a tall and a wide matrix, each with three factor steps.
        inline for (.{ .{ 4, 3 }, .{ 3, 4 } }) |shape| {
            const rows = shape[0];
            const columns = shape[1];
            const ld = rows + 1;
            var original: [rows * columns]T = undefined;
            for (0..columns) |column| {
                for (0..rows) |row| {
                    original[row + column * rows] = if (column == 0) 0 else @as(T, @floatFromInt((row + 1) * (column + 1))) +
                        (if (row == column) @as(T, 1) else 0);
                }
            }
            var data: [ld * columns]T = @splat(-987);
            for (0..columns) |column| {
                for (0..rows) |row| data[row + column * ld] = original[row + column * rows];
            }
            var pivots: [4]lapack.PivotInt = @splat(-77);
            try std.testing.expectEqual(@as(usize, 1), try lapack.luFactorize(T, mutable(T, &data, rows, columns, ld), &pivots));
            try std.testing.expectEqual(@as(lapack.PivotInt, 0), pivots[0]);
            try std.testing.expectEqual(@as(lapack.PivotInt, rows - 1), pivots[1]);
            try std.testing.expect(pivots[2] >= 2 and pivots[2] < rows);
            try std.testing.expectEqual(@as(lapack.PivotInt, -77), pivots[3]);
            var permuted = original;
            for (0..3) |k| {
                const pivot: usize = @intCast(pivots[k]);
                if (pivot != k) {
                    for (0..columns) |column| std.mem.swap(T, &permuted[k + column * rows], &permuted[pivot + column * rows]);
                }
            }
            for (0..columns) |column| {
                for (0..rows) |row| {
                    var product: T = 0;
                    for (0..3) |k| {
                        const l: T = if (row == k) 1 else if (row > k) data[row + k * ld] else 0;
                        const u: T = if (k <= column) data[k + column * ld] else 0;
                        product += l * u;
                    }
                    try near(T, permuted[row + column * rows], product);
                }
                try std.testing.expectEqual(@as(T, -987), data[rows + column * ld]);
            }
        }
    }
}

test "Empty factorizations and zero RHS solves preserve caller storage" {
    var storage = [_]f64{ 2, 3, 5, 7 };
    const before = storage;
    var pivots = [_]lapack.PivotInt{ 0, 1, -77 };
    const saved_pivots = pivots;
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(f64, mutable(f64, storage[0..0], 0, 0, 0), .lower));
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, mutable(f64, storage[0..0], 0, 3, 0), &pivots));
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, mutable(f64, storage[0..0], 3, 0, 0), &pivots));
    const a = mutable(f64, &storage, 2, 2, 2).asConst();
    const b = mutable(f64, storage[0..0], 2, 0, 0);
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(f64, a, b, .lower));
    try std.testing.expectEqual(@as(usize, 0), try lapack.luSolve(f64, a, &pivots, b));
    const empty = mutable(f64, storage[0..0], 0, 0, 0);
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(f64, empty.asConst(), empty, .upper));
    try std.testing.expectEqual(@as(usize, 0), try lapack.luSolve(f64, empty.asConst(), &.{}, empty));
    try std.testing.expectEqualSlices(f64, &before, &storage);
    try std.testing.expectEqualSlices(lapack.PivotInt, &saved_pivots, &pivots);
}

test "Structural and capacity errors fail before either factorization mutates" {
    var storage = [_]f64{ 4, 2, 2, 3 };
    const before = storage;
    var pivots = [_]lapack.PivotInt{ -77, -77 };
    const invalid = [_]lapack.Matrix(f64){
        .{ .values = &storage, .row_count = -1, .column_count = 2, .leading_dimension = 2 },
        .{ .values = &storage, .row_count = 2, .column_count = 2, .leading_dimension = 1 },
        .{ .values = storage[0..3], .row_count = 2, .column_count = 2, .leading_dimension = 2 },
        .{ .values = &storage, .row_count = 0, .column_count = 0, .leading_dimension = -1 },
    };
    const errors = [_]lapack.Error{ error.InvalidLength, error.InvalidLeadingDimension, error.BufferTooSmall, error.InvalidLeadingDimension };
    for (invalid, errors) |a, expected| {
        try std.testing.expectError(expected, lapack.choleskyFactorize(f64, a, .lower));
        try std.testing.expectError(expected, lapack.luFactorize(f64, a, &pivots));
    }
    try std.testing.expectError(error.DimensionMismatch, lapack.choleskyFactorize(f64, mutable(f64, &storage, 2, 1, 2), .upper));
    try std.testing.expectError(error.BufferTooSmall, lapack.luFactorize(f64, mutable(f64, &storage, 2, 2, 2), pivots[0..1]));
    try std.testing.expectEqualSlices(f64, &before, &storage);
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ -77, -77 }, &pivots);
}

test "Solves reject malformed views and transposed factors without mutation" {
    var factor = [_]f64{ 2, 0, 0, 3 };
    var rhs = [_]f64{ 1, 2, 3, 4 };
    const before = rhs;
    const a = mutable(f64, &factor, 2, 2, 2).asConst();
    const b = mutable(f64, &rhs, 2, 2, 2);
    const pivots = [_]lapack.PivotInt{ 0, 1 };
    try std.testing.expectError(error.InvalidMatrixOperation, lapack.choleskySolve(f64, a.transposed(), b, .lower));
    try std.testing.expectError(error.InvalidMatrixOperation, lapack.luSolve(f64, a.adjoint(), &pivots, b));
    var short = a;
    short.values = factor[0..3];
    try std.testing.expectError(error.BufferTooSmall, lapack.choleskySolve(f64, short, b, .lower));
    try std.testing.expectError(error.BufferTooSmall, lapack.luSolve(f64, short, &pivots, b));
    try std.testing.expectError(error.BufferTooSmall, lapack.choleskySolve(f64, a, mutable(f64, rhs[0..3], 2, 2, 2), .upper));
    try std.testing.expectError(error.BufferTooSmall, lapack.luSolve(f64, a, &pivots, mutable(f64, rhs[0..3], 2, 2, 2)));
    try std.testing.expectError(error.DimensionMismatch, lapack.choleskySolve(f64, a, mutable(f64, &rhs, 1, 2, 2), .lower));
    try std.testing.expectError(error.DimensionMismatch, lapack.luSolve(f64, a, &pivots, mutable(f64, &rhs, 1, 2, 2)));
    try std.testing.expectEqualSlices(f64, &before, &rhs);
}

test "LU validates the complete pivot sequence before modifying RHS" {
    var factor = [_]f64{ 2, 0, 0, 3 };
    var rhs = [_]f64{ 1, 2, 3, 4 };
    const before = rhs;
    const a = mutable(f64, &factor, 2, 2, 2).asConst();
    const b = mutable(f64, &rhs, 2, 2, 2);
    for ([_][2]lapack.PivotInt{ .{ 1, -1 }, .{ 1, 2 }, .{ 1, 0 } }) |pivots| {
        try std.testing.expectError(error.InvalidPivot, lapack.luSolve(f64, a, &pivots, b));
    }
    try std.testing.expectError(error.BufferTooSmall, lapack.luSolve(f64, a, &.{0}, b));
    try std.testing.expectEqualSlices(f64, &before, &rhs);
}

test "Factor and RHS overlap is rejected including partial overlap" {
    var storage = [_]f64{ 2, 1, 0, 3, 5, 7, 11, 13 };
    const before = storage;
    const a = mutable(f64, storage[0..4], 2, 2, 2).asConst();
    const pivots = [_]lapack.PivotInt{ 0, 1 };
    for ([_]usize{ 0, 1, 3 }) |offset| {
        const b = mutable(f64, storage[offset..][0..4], 2, 2, 2);
        try std.testing.expectError(error.AliasingNotAllowed, lapack.choleskySolve(f64, a, b, .lower));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(f64, a, &pivots, b));
    }
    try std.testing.expectEqualSlices(f64, &before, &storage);
}

test "LU pivot storage cannot alias factor or RHS storage" {
    var factor = [_]f64{ 2, 0, 0, 3 };
    var rhs = [_]f64{ 1, 2 };
    const original_factor = factor;
    const original_rhs = rhs;
    const factor_pivots: [*]lapack.PivotInt = @ptrCast(&factor);
    const rhs_pivots: [*]lapack.PivotInt = @ptrCast(&rhs);
    const a = mutable(f64, &factor, 2, 2, 2);
    const b = mutable(f64, &rhs, 2, 1, 2);
    try std.testing.expectError(error.AliasingNotAllowed, lapack.luFactorize(f64, a, factor_pivots[0..2]));
    try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(f64, a.asConst(), factor_pivots[0..2], b));
    try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(f64, a.asConst(), rhs_pivots[0..2], b));
    try std.testing.expectEqualSlices(f64, &original_factor, &factor);
    try std.testing.expectEqualSlices(f64, &original_rhs, &rhs);
}

test "Nonfinite logical input fails before mutation while unused triangle is ignored" {
    var invalid = [_]f64{ 4, 2, 2, std.math.inf(f64) };
    const before = invalid;
    var pivots = [_]lapack.PivotInt{ -77, -77 };
    try std.testing.expectError(error.NonFiniteInput, lapack.choleskyFactorize(f64, mutable(f64, &invalid, 2, 2, 2), .lower));
    try std.testing.expectError(error.NonFiniteInput, lapack.luFactorize(f64, mutable(f64, &invalid, 2, 2, 2), &pivots));
    try std.testing.expectEqualSlices(f64, &before, &invalid);
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ -77, -77 }, &pivots);
    var selected = [_]f64{ 4, 2, std.math.nan(f64), 3 };
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(f64, mutable(f64, &selected, 2, 2, 2), .lower));
    try std.testing.expect(std.math.isNan(selected[2]));
    var rhs = [_]f64{ 1, std.math.inf(f64) };
    const rhs_before = rhs;
    try std.testing.expectError(error.NonFiniteInput, lapack.choleskySolve(f64, mutable(f64, &selected, 2, 2, 2).asConst(), mutable(f64, &rhs, 2, 1, 2), .lower));
    try std.testing.expectEqualSlices(f64, &rhs_before, &rhs);
}

test "LAPACK storage geometry retains empty leading dimension and error policies" {
    var empty: [0]f64 = .{};
    const tall_empty = mutable(f64, &empty, 2, 0, 0);
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, tall_empty, &.{}));
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrFactorize(f64, tall_empty, &.{}));
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrLeastSquaresSolve(f64, tall_empty.asConst(), &.{}, tall_empty));
    const svd = try lapack.svdDecompose(f64, tall_empty, &.{}, mutable(f64, &empty, 0, 0, 0), &.{}, .{});
    try std.testing.expectEqual(lapack.SvdStatus.converged, svd.status);
    try std.testing.expectError(error.DimensionMismatch, lapack.choleskyFactorize(f64, tall_empty, .lower));
    var complex_empty: [0]lapack.ComplexF64 = .{};
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(lapack.ComplexF64, mutable(lapack.ComplexF64, &complex_empty, 2, 0, 0), &.{}));
    var invalid = tall_empty;
    invalid.leading_dimension = -1;
    try std.testing.expectError(error.InvalidLeadingDimension, lapack.qrFactorize(f64, invalid, &.{}));
    invalid.row_count = -1;
    try std.testing.expectError(error.InvalidLength, lapack.qrFactorize(f64, invalid, &.{}));
}

test "LAPACK used matrix span includes padding but excludes trailing suffix" {
    var factor = [_]f64{ 1, 0, -99, 0, 1, -88 };
    const before = factor;
    const a = mutable(f64, &factor, 2, 2, 3);
    const in_padding: [*]lapack.PivotInt = @ptrCast(&factor[2]);
    try std.testing.expectError(error.AliasingNotAllowed, lapack.luFactorize(f64, a, in_padding[0..2]));
    try std.testing.expectEqualSlices(f64, &before, &factor);
    // The last f64 lies beyond the matrix's five-element active span. Reuse
    // those eight bytes for the two i32 pivots, adjacent to but not overlapping it.
    const in_suffix: [*]lapack.PivotInt = @ptrCast(&factor[5]);
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, a, in_suffix[0..2]));
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ 0, 1 }, in_suffix[0..2]);
    try std.testing.expectEqual(@as(f64, -99), factor[2]);

    var qr_data = [_]f64{ 2, 0, 123, 456, 0, 3 };
    var tau: [2]f64 = undefined;
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrFactorize(f64, mutable(f64, &qr_data, 2, 2, 4), &tau));
    try std.testing.expectEqual(@as(f64, 123), qr_data[2]);
    try std.testing.expectEqual(@as(f64, 456), qr_data[3]);
    const qr_before = qr_data;
    try std.testing.expectError(error.BufferTooSmall, lapack.qrFactorize(f64, mutable(f64, qr_data[0..5], 2, 2, 4), &tau));
    try std.testing.expectEqualSlices(f64, &qr_before, &qr_data);
}

test "LAPACK rejects byte endpoint overflow before touching reflector storage" {
    var a = [_]f64{2};
    // Intentionally non-dereferenceable: checked byte geometry must reject the
    // wrapping endpoint before reading or writing either numerical buffer.
    const invalid_tau: [*]f64 = @ptrFromInt(std.math.maxInt(usize) - (@sizeOf(f64) - 1));
    try std.testing.expectError(error.InvalidLength, lapack.qrFactorize(f64, mutable(f64, &a, 1, 1, 1), invalid_tau[0..1]));
    try std.testing.expectEqual(@as(f64, 2), a[0]);
}

test "LAPACK multiple invalid arguments retain preflight error precedence" {
    var factor = [_]f64{1};
    var rhs = [_]f64{7};
    var transformed = mutable(f64, &factor, 1, 1, 1).asConst();
    transformed.operation = .transposed;
    var bad_rhs = mutable(f64, &rhs, 1, 1, 1);
    bad_rhs.leading_dimension = -1;
    try std.testing.expectError(error.InvalidMatrixOperation, lapack.luSolve(f64, transformed, &.{0}, bad_rhs));
    factor[0] = std.math.nan(f64);
    try std.testing.expectError(error.AliasingNotAllowed, lapack.qrLeastSquaresSolve(f64, mutable(f64, &factor, 1, 1, 1).asConst(), &factor, mutable(f64, &rhs, 1, 1, 1)));
    var eigenvalues = [_]f64{9};
    var vectors = [_]f64{11};
    try std.testing.expectError(error.BufferTooSmall, lapack.symmetricEigenDecompose(f64, transformed, .lower, mutable(f64, &.{}, 1, 1, 1), &eigenvalues, mutable(f64, &vectors, 1, 1, 1), .{}));
    var work: [3]f128 = undefined;
    const a = mutable(f64, &factor, 1, 1, 1);
    try std.testing.expectError(error.BufferTooSmall, lapack.svdDecompose(f64, a, &factor, mutable(f64, &vectors, 1, 1, 1), work[0..2], .{ .relative_tolerance = -1 }));
    try std.testing.expectError(error.InvalidTolerance, lapack.svdDecompose(f64, a, &factor, mutable(f64, &vectors, 1, 1, 1), &work, .{ .relative_tolerance = -1 }));
    try std.testing.expectEqual(@as(f64, 7), rhs[0]);
    try std.testing.expectEqual(@as(f64, 9), eigenvalues[0]);
    try std.testing.expectEqual(@as(f64, 11), vectors[0]);
    try std.testing.expect(std.math.isNan(factor[0]));
}

test "Cholesky checked frontend retains complete finite preflight before core mutation" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
            var factor = [_]T{ 4, 0, 0, std.math.nan(T) };
            const before = factor;
            try std.testing.expectError(error.NonFiniteInput, lapack.choleskyFactorize(T, mutable(T, &factor, 2, 2, 2), triangle));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&before), std.mem.sliceAsBytes(&factor));
            factor = .{ 2, 0, 0, 3 };
            var rhs = [_]T{ 7, 11, 13, std.math.nan(T) };
            const rhs_before = rhs;
            try std.testing.expectError(error.NonFiniteInput, lapack.choleskySolve(T, mutable(T, &factor, 2, 2, 2).asConst(), mutable(T, &rhs, 2, 2, 2), triangle));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rhs_before), std.mem.sliceAsBytes(&rhs));
            try std.testing.expectEqualSlices(T, &.{ 2, 0, 0, 3 }, &factor);
        }
    }
}

test "Cholesky checked frontend retains argument order and empty solve policy" {
    var factor = [_]f64{std.math.nan(f64)};
    const a = mutable(f64, &factor, 1, 1, 1);
    try std.testing.expectError(error.AliasingNotAllowed, lapack.choleskySolve(f64, a.asConst(), a, .lower));
    var transformed = a.asConst();
    transformed.operation = .transposed;
    var rhs_values = [_]f64{7};
    var rhs = mutable(f64, &rhs_values, 1, 1, 1);
    rhs.leading_dimension = -1;
    try std.testing.expectError(error.InvalidMatrixOperation, lapack.choleskySolve(f64, transformed, rhs, .lower));
    factor[0] = -2;
    const no_rhs = mutable(f64, &.{}, 1, 0, 0);
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(f64, a.asConst(), no_rhs, .upper));
    rhs.leading_dimension = 1;
    try std.testing.expectEqual(@as(usize, 1), try lapack.choleskySolve(f64, a.asConst(), rhs, .upper));
    try std.testing.expectEqual(@as(f64, 7), rhs_values[0]);
    try std.testing.expectEqual(@as(f64, -2), factor[0]);
}

// These observers cover the real frontend/body, not a substitute numerical loop.
const admission_core = @import("zynum-core");
const admission_execution = @import("zynum-execution");
const admission_context = admission_execution.context;
const admission_scalar = admission_core.internal.numerics.Scalar;
const CholeskyAdmission = struct {
    var admitted: usize = 0;
    var finishing: usize = 0;
    var wanted: usize = 0;
    var actual: []const u8 = &.{};
    var expected: []const u8 = &.{};
    fn observe(ctx: *admission_context.Context, event: admission_context.TestEvent) void {
        if (ctx != admission_execution.getContext() or &ctx.executor != admission_execution.get()) @panic("Cholesky default owner split");
        if (ctx.snapshot().active_operations != 1) @panic("Cholesky nested admission");
        switch (event) {
            .admitted => admitted += 1,
            .finishing => {
                if (!std.mem.eql(u8, actual, expected)) @panic("Cholesky storage not final before end");
                finishing += 1;
            },
        }
    }
    fn start() void {
        admitted = 0;
        finishing = 0;
        wanted = 0;
        actual = &.{};
        expected = &.{};
        admission_context.Context.testing.configure(admission_execution.getContext(), 1, observe);
    }
    fn stop() void {
        admission_context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
        actual = &.{};
        expected = &.{};
    }
    fn watch(a: anytype, b: anytype) void {
        actual = std.mem.sliceAsBytes(a);
        expected = std.mem.sliceAsBytes(b);
        wanted += 1;
    }
    fn pair() !void {
        try std.testing.expectEqual(wanted, admitted);
        try std.testing.expectEqual(wanted, finishing);
        try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().active_operations);
    }
};
fn admissionValue(comptime T: type, re: f64, im: f64) T {
    return if (comptime admission_scalar.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn admissionParts(comptime T: type, value: T) [2]f64 {
    return if (comptime admission_scalar.isComplex(T)) .{ @floatCast(value.re), @floatCast(value.im) } else .{ @floatCast(value), 0 };
}
fn admissionNear(comptime T: type, expected: [2]f64, actual: [2]f64) !void {
    const eps: f64 = if (T == f32 or T == lapack.ComplexF32) 2e-5 else 2e-13;
    for (expected, actual) |e, a| try std.testing.expectApproxEqAbs(e, a, eps * @max(@as(f64, 1), @abs(e)));
}
// Independent dense multiply / Gram oracle, using ordinary component arithmetic.
fn admissionResidual(comptime T: type, a: []const T, factor: []const T, x: []const T, b: []const T, triangle: lapack.Triangle) !void {
    for (0..2) |row| {
        var product: [2]f64 = .{ 0, 0 };
        for (0..2) |k| {
            const v = admissionParts(T, a[row + 2 * k]);
            const w = admissionParts(T, x[k]);
            product[0] += v[0] * w[0] - v[1] * w[1];
            product[1] += v[0] * w[1] + v[1] * w[0];
        }
        try admissionNear(T, admissionParts(T, b[row]), product);
        for (0..2) |column| {
            var gram: [2]f64 = .{ 0, 0 };
            for (0..@min(row, column) + 1) |k| {
                var l = admissionParts(T, factor[if (triangle == .lower) row + 3 * k else k + 3 * row]);
                var r = admissionParts(T, factor[if (triangle == .lower) column + 3 * k else k + 3 * column]);
                if (triangle == .upper) {
                    l[1] = -l[1];
                    r[1] = -r[1];
                }
                gram[0] += l[0] * r[0] + l[1] * r[1];
                gram[1] += l[1] * r[0] - l[0] * r[1];
            }
            try admissionNear(T, admissionParts(T, a[row + 2 * column]), gram);
        }
    }
}

test "direct Cholesky admits once across real complex triangles final storage and residuals" {
    comptime {
        if (admission_core.internal.numerics.Execution != admission_execution or
            admission_core.lapack.choleskyFactorize != lapack.choleskyFactorize or
            admission_core.lapack.choleskySolve != lapack.choleskySolve)
            @compileError("Cholesky facade or Execution identity split");
    }
    CholeskyAdmission.start();
    defer CholeskyAdmission.stop();
    inline for (.{ f32, f64, lapack.ComplexF32, lapack.ComplexF64 }) |T| {
        const z: f64 = if (admission_scalar.isComplex(T)) 1 else 0;
        inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
            const pad = admissionValue(T, 97, -13);
            const a = [_]T{ admissionValue(T, 4, 0), admissionValue(T, 2, 2 * z), admissionValue(T, 2, -2 * z), admissionValue(T, 10 + z, 0) };
            var factor: [6]T = @splat(pad);
            factor[0] = a[0];
            factor[4] = a[3];
            factor[if (triangle == .lower) 1 else 3] = a[if (triangle == .lower) 1 else 2];
            var expected = factor;
            expected[0] = admissionValue(T, 2, 0);
            expected[4] = admissionValue(T, 3, 0);
            expected[if (triangle == .lower) 1 else 3] = admissionValue(T, 1, if (triangle == .lower) z else -z);
            CholeskyAdmission.watch(&factor, &expected);
            const view = mutable(T, &factor, 2, 2, 3);
            try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(T, view, triangle));
            try CholeskyAdmission.pair();
            const factor_before = factor;
            var rhs = [_]T{ admissionValue(T, 8 - 2 * z, -2 * z), admissionValue(T, 22, -7 * z), pad };
            const original_rhs = rhs;
            const solution = [_]T{ admissionValue(T, 1, z), admissionValue(T, 2, -z), pad };
            CholeskyAdmission.watch(&rhs, &solution);
            try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(T, view.asConst(), mutable(T, &rhs, 2, 1, 3), triangle));
            try CholeskyAdmission.pair();
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&factor_before), std.mem.sliceAsBytes(&factor));
            try admissionResidual(T, &a, &factor, &rhs, &original_rhs, triangle);
        }
    }
}

test "direct Cholesky admission preserves errors empty paths and partial factor storage" {
    CholeskyAdmission.start();
    defer CholeskyAdmission.stop();
    inline for (.{ f32, f64, lapack.ComplexF32, lapack.ComplexF64 }) |T| {
        inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
            const pad = admissionValue(T, 97, -13);
            var factor = [_]T{ admissionValue(T, 4, 0), admissionValue(T, 2, 0), pad, admissionValue(T, 2, 0), admissionValue(T, 0, 0), pad };
            var partial = factor;
            partial[0] = admissionValue(T, 2, 0);
            partial[if (triangle == .lower) 1 else 3] = admissionValue(T, 1, 0);
            CholeskyAdmission.watch(&factor, &partial);
            try std.testing.expectEqual(@as(usize, 2), try lapack.choleskyFactorize(T, mutable(T, &factor, 2, 2, 3), triangle));
            try CholeskyAdmission.pair();
            var rhs = [_]T{ admissionValue(T, 7, 1), admissionValue(T, 11, -1), pad };
            const original_rhs = rhs;
            CholeskyAdmission.watch(&rhs, &original_rhs);
            try std.testing.expectEqual(@as(usize, 2), try lapack.choleskySolve(T, mutable(T, &factor, 2, 2, 3).asConst(), mutable(T, &rhs, 2, 1, 3), triangle));
            try CholeskyAdmission.pair();
            factor[0] = admissionValue(T, std.math.nan(f64), 0);
            const invalid = factor;
            CholeskyAdmission.watch(&factor, &invalid);
            try std.testing.expectError(error.NonFiniteInput, lapack.choleskyFactorize(T, mutable(T, &factor, 2, 2, 3), triangle));
            try CholeskyAdmission.pair();
            const av = mutable(T, &factor, 2, 2, 3);
            CholeskyAdmission.watch(&factor, &invalid);
            try std.testing.expectError(error.AliasingNotAllowed, lapack.choleskySolve(T, av.asConst(), av, triangle));
            try CholeskyAdmission.pair();
            CholeskyAdmission.watch(&factor, &invalid);
            try std.testing.expectError(error.BufferTooSmall, lapack.choleskyFactorize(T, mutable(T, factor[0..1], 2, 2, 3), triangle));
            try CholeskyAdmission.pair();
            CholeskyAdmission.watch(&factor, &invalid);
            try std.testing.expectError(error.DimensionMismatch, lapack.choleskyFactorize(T, mutable(T, &factor, 2, 1, 3), triangle));
            try CholeskyAdmission.pair();
            if (comptime admission_scalar.isComplex(T)) {
                factor[0] = admissionValue(T, 4, 1);
                const nonreal = factor;
                CholeskyAdmission.watch(&factor, &nonreal);
                try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskyFactorize(T, av, triangle));
                try CholeskyAdmission.pair();
            }
            factor[0] = admissionValue(T, -2, 0);
            const no_rhs_before = factor;
            CholeskyAdmission.watch(&factor, &no_rhs_before);
            try std.testing.expectEqual(@as(usize, 0), try lapack.choleskySolve(T, mutable(T, factor[0..1], 1, 1, 1).asConst(), mutable(T, &.{}, 1, 0, 0), triangle));
            try CholeskyAdmission.pair();
            CholeskyAdmission.watch(&factor, &no_rhs_before);
            try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(T, mutable(T, &.{}, 0, 0, 0), triangle));
            try CholeskyAdmission.pair();
        }
    }
}

test "borrowed Cholesky uses one real closable operation through close and never default" {
    const Context = admission_context.Context;
    const checked = admission_core.internal.lapack_checked;
    const default = admission_execution.getContext();
    Context.testing.configure(default, 0, null);
    defer Context.testing.configure(default, std.math.maxInt(usize), null);
    const Probe = struct {
        context: Context,
        admitted: usize = 0,
        finishing: usize = 0,
        fn observe(ctx: *Context, event: admission_context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("context", ctx);
            if (ctx == admission_execution.getContext() or ctx.snapshot().active_operations != 1) @panic("borrowed operation identity");
            switch (event) {
                .admitted => self.admitted += 1,
                .finishing => self.finishing += 1,
            }
        }
    };
    inline for (.{ f32, f64, lapack.ComplexF32, lapack.ComplexF64 }) |T| {
        var probe: Probe = .{ .context = Context.init(std.testing.allocator, admission_execution.test_support.policy, .internal_closable) };
        Context.testing.configure(&probe.context, 1, Probe.observe);
        var operation = try probe.context.begin();
        var ended = false;
        defer {
            if (!ended) operation.end();
            _ = probe.context.requestClose();
            _ = probe.context.tryProgressClose();
        }
        try std.testing.expectEqual(admission_context.RequestResult.pending, probe.context.requestClose());
        try std.testing.expectEqual(admission_context.CloseResult.busy, probe.context.tryProgressClose());
        var a = [_]T{ admissionValue(T, 4, 0), admissionValue(T, 97, -13) };
        var b = [_]T{ admissionValue(T, 12, 4), admissionValue(T, 89, -7) };
        const expected_a = [_]T{ admissionValue(T, 2, 0), a[1] };
        const expected_b = [_]T{ admissionValue(T, 3, 1), b[1] };
        const av = mutable(T, &a, 1, 1, 1);
        try std.testing.expectEqual(@as(usize, 0), try checked.choleskyFactorizeAdmitted(&operation, T, av, .lower));
        try std.testing.expectEqual(@as(usize, 0), try checked.choleskySolveAdmitted(&operation, T, av.asConst(), mutable(T, &b, 1, 1, 1), .lower));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_a), std.mem.sliceAsBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_b), std.mem.sliceAsBytes(&b));
        try std.testing.expectEqual(@as(usize, 1), probe.admitted);
        try std.testing.expectEqual(@as(usize, 0), probe.finishing);
        try std.testing.expectEqual(admission_context.CloseResult.busy, probe.context.tryProgressClose());
        operation.end();
        ended = true;
        try std.testing.expectEqual(@as(usize, 1), probe.finishing);
        try std.testing.expectEqual(admission_context.CloseResult.closed, probe.context.tryProgressClose());
        try std.testing.expectError(error.Closing, probe.context.begin());
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_a), std.mem.sliceAsBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_b), std.mem.sliceAsBytes(&b));
        try std.testing.expectEqual(@as(usize, 0), default.snapshot().active_operations);
    }
}

const DenseDriverExecution = @import("zynum-execution");
const DenseDriverAdmissionObservation = struct {
    var entries: usize = 0;
    var finishes: usize = 0;
    var values: []const f64 = &.{};
    var expected: []const f64 = &.{};

    fn reset() void {
        entries = 0;
        finishes = 0;
        values = &.{};
        expected = &.{};
    }
    fn watch(actual: []const f64, wanted: []const f64) void {
        values = actual;
        expected = wanted;
    }
    fn call(ctx: *DenseDriverExecution.context.Context, event: DenseDriverExecution.context.TestEvent) void {
        if (ctx != DenseDriverExecution.getContext() or &ctx.executor != DenseDriverExecution.get())
            @panic("dense driver duplicated the default owner");
        if (ctx.snapshot().active_operations != 1 or ctx.snapshot().resource_pins != 0)
            @panic("dense driver did not retain one allocation-free operation");
        switch (event) {
            .admitted => entries += 1,
            .finishing => {
                if (values.len != expected.len) @panic("invalid observation extent");
                for (values, expected) |actual, wanted| {
                    if (!std.math.isFinite(actual) or @abs(actual - wanted) > 2e-12 * @max(@as(f64, 1), @abs(wanted)))
                        @panic("dense driver ended before its numerical result or changed storage");
                }
                finishes += 1;
            },
        }
    }
    fn check(count: usize) !void {
        try std.testing.expectEqual(count, entries);
        try std.testing.expectEqual(count, finishes);
        try std.testing.expectEqual(@as(usize, 0), DenseDriverExecution.getContext().snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), DenseDriverExecution.getContext().snapshot().resource_pins);
    }
};

test "seven direct dense drivers retain one operation through numerical completion" {
    const Observe = DenseDriverAdmissionObservation;
    Observe.reset();
    DenseDriverExecution.context.Context.testing.configure(DenseDriverExecution.getContext(), 1, Observe.call);
    defer DenseDriverExecution.context.Context.testing.configure(DenseDriverExecution.getContext(), std.math.maxInt(usize), null);

    var lu = [_]f64{ 4, 2, 71, 1, 3, 73 };
    var pivots: [2]lapack.PivotInt = undefined;
    Observe.watch(&lu, &.{ 4, 0.5, 71, 1, 2.5, 73 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, mutable(f64, &lu, 2, 2, 3), &pivots));
    try Observe.check(1);
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ 0, 1 }, &pivots);
    var rhs = [_]f64{ 11, 13, 79 };
    Observe.watch(&rhs, &.{ 2, 3, 79 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.luSolve(f64, mutable(f64, &lu, 2, 2, 3).asConst(), &pivots, mutable(f64, &rhs, 2, 1, 3)));
    try Observe.check(2);
    try std.testing.expectEqualSlices(f64, &.{ 4, 0.5, 71, 1, 2.5, 73 }, &lu);
    try near(f64, 11, 4 * rhs[0] + rhs[1]);
    try near(f64, 13, 2 * rhs[0] + 3 * rhs[1]);

    var qr = [_]f64{ 3, 4, 81 };
    var tau = [_]f64{ 0, 82 };
    Observe.watch(&qr, &.{ -5, 0.5, 81 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrFactorize(f64, mutable(f64, &qr, 2, 1, 3), &tau));
    try Observe.check(3);
    try near(f64, 1.6, tau[0]);
    try std.testing.expectEqual(@as(f64, 82), tau[1]);
    var qb = [_]f64{ 7, 1, 83 };
    Observe.watch(&qb, &.{ -5, -5, 83 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrApplyQ(f64, mutable(f64, &qr, 2, 1, 3).asConst(), &tau, mutable(f64, &qb, 2, 1, 3), .transposed));
    try Observe.check(4);
    var least = [_]f64{ 6, 8, 85 };
    Observe.watch(&least, &.{ 2, 0, 85 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrLeastSquaresSolve(f64, mutable(f64, &qr, 2, 1, 3).asConst(), &tau, mutable(f64, &least, 2, 1, 3)));
    try Observe.check(5);
    try near(f64, 6, 3 * least[0]);
    try near(f64, 8, 4 * least[0]);
    try std.testing.expectEqualSlices(f64, &.{ -5, 0.5, 81 }, &qr);

    var symmetric = [_]f64{ 2, 1, 91, 99, 2, 93 };
    const original = symmetric;
    var work: [6]f64 = @splat(95);
    var vectors: [6]f64 = @splat(97);
    var eigenvalues = [_]f64{ 0, 0, 87 };
    Observe.watch(&eigenvalues, &.{ 1, 3, 87 });
    const eigen = try lapack.symmetricEigenDecompose(f64, mutable(f64, &symmetric, 2, 2, 3).asConst(), .lower, mutable(f64, &work, 2, 2, 3), &eigenvalues, mutable(f64, &vectors, 2, 2, 3), .{});
    try std.testing.expectEqual(lapack.SymmetricEigenStatus.converged, eigen.status);
    try Observe.check(6);
    for (0..2) |j| {
        const x = vectors[j * 3];
        const y = vectors[j * 3 + 1];
        try near(f64, eigenvalues[j] * x, 2 * x + y);
        try near(f64, eigenvalues[j] * y, x + 2 * y);
        try near(f64, 1, x * x + y * y);
    }
    try near(f64, 0, vectors[0] * vectors[3] + vectors[1] * vectors[4]);
    try std.testing.expectEqualSlices(f64, &original, &symmetric);
    try std.testing.expectEqual(@as(f64, 95), work[2]);
    try std.testing.expectEqual(@as(f64, 95), work[5]);
    try std.testing.expectEqual(@as(f64, 97), vectors[2]);
    try std.testing.expectEqual(@as(f64, 97), vectors[5]);

    var svd_a = [_]f64{ 3, 4, 101 };
    var singular = [_]f64{ 0, 103 };
    var right = [_]f64{ 0, 105 };
    var svd_work: [5]f128 = @splat(107);
    try std.testing.expectEqual(@as(usize, 4), try lapack.svdWorkspaceLength(f64, 2, 1));
    try Observe.check(6);
    Observe.watch(&svd_a, &.{ 0.6, 0.8, 101 });
    const svd = try lapack.svdDecompose(f64, mutable(f64, &svd_a, 2, 1, 3), &singular, mutable(f64, &right, 1, 1, 1), &svd_work, .{});
    try std.testing.expectEqual(lapack.SvdStatus.converged, svd.status);
    try Observe.check(7);
    try std.testing.expectEqualSlices(f64, &.{ 5, 103 }, &singular);
    try std.testing.expectEqualSlices(f64, &.{ 1, 105 }, &right);
    try std.testing.expectEqual(@as(f128, 107), svd_work[4]);
    try near(f64, 3, svd_a[0] * singular[0] * right[0]);
    try near(f64, 4, svd_a[1] * singular[0] * right[0]);
}

test "seven direct dense drivers release one operation on preflight errors and empty calls" {
    const Observe = DenseDriverAdmissionObservation;
    Observe.reset();
    DenseDriverExecution.context.Context.testing.configure(DenseDriverExecution.getContext(), 1, Observe.call);
    defer DenseDriverExecution.context.Context.testing.configure(DenseDriverExecution.getContext(), std.math.maxInt(usize), null);
    var storage = [_]f64{19};
    const bad = mutable(f64, &storage, 2, 2, 2);
    var pivots = [_]lapack.PivotInt{ 7, 9 };
    var tau = [_]f64{ 1, 1 };
    var auxiliary = [_]f64{23};
    var workspace: [10]f128 = @splat(29);
    Observe.watch(&storage, &.{19});
    try std.testing.expectError(error.BufferTooSmall, lapack.luFactorize(f64, bad, &pivots));
    try Observe.check(1);
    try std.testing.expectError(error.BufferTooSmall, lapack.luSolve(f64, bad.asConst(), &pivots, bad));
    try Observe.check(2);
    try std.testing.expectError(error.BufferTooSmall, lapack.qrFactorize(f64, bad, &tau));
    try Observe.check(3);
    try std.testing.expectError(error.BufferTooSmall, lapack.qrApplyQ(f64, bad.asConst(), &tau, bad, .normal));
    try Observe.check(4);
    try std.testing.expectError(error.BufferTooSmall, lapack.qrLeastSquaresSolve(f64, bad.asConst(), &tau, bad));
    try Observe.check(5);
    try std.testing.expectError(error.BufferTooSmall, lapack.symmetricEigenDecompose(f64, bad.asConst(), .upper, bad, &auxiliary, bad, .{}));
    try Observe.check(6);
    try std.testing.expectError(error.BufferTooSmall, lapack.svdDecompose(f64, bad, &auxiliary, bad, &workspace, .{}));
    try Observe.check(7);
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ 7, 9 }, &pivots);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, &tau);
    try std.testing.expectEqualSlices(f64, &.{23}, &auxiliary);
    for (workspace) |entry| try std.testing.expectEqual(@as(f128, 29), entry);

    const empty = mutable(f64, &.{}, 0, 0, 0);
    Observe.watch(&.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), try lapack.luFactorize(f64, empty, &.{}));
    try Observe.check(8);
    try std.testing.expectEqual(@as(usize, 0), try lapack.luSolve(f64, empty.asConst(), &.{}, empty));
    try Observe.check(9);
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrFactorize(f64, empty, &.{}));
    try Observe.check(10);
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrApplyQ(f64, empty.asConst(), &.{}, empty, .normal));
    try Observe.check(11);
    try std.testing.expectEqual(@as(usize, 0), try lapack.qrLeastSquaresSolve(f64, empty.asConst(), &.{}, empty));
    try Observe.check(12);
    try std.testing.expectEqual(lapack.SymmetricEigenStatus.converged, (try lapack.symmetricEigenDecompose(f64, empty.asConst(), .lower, empty, &.{}, empty, .{})).status);
    try Observe.check(13);
    try std.testing.expectEqual(lapack.SvdStatus.converged, (try lapack.svdDecompose(f64, empty, &.{}, empty, &.{}, .{})).status);
    try Observe.check(14);
    try std.testing.expectEqual(@as(usize, 0), try lapack.svdWorkspaceLength(f64, 0, 0));
    try Observe.check(14);
}

comptime {
    _ = @import("interop/hermitian_managed_test.zig");
    _ = @import("interop/gvd_adapter_test.zig");
}

comptime {
    _ = @import("interop/hermitian_adapter_test.zig");
}

comptime {
    _ = @import("lq_test.zig");
    _ = @import("full_rank_least_squares_test.zig");
    _ = @import("interop/gels_adapter_test.zig");
}

comptime {
    _ = @import("complex_bidiagonal_test.zig");
    _ = @import("complex_bidiagonal_svd_test.zig");
    _ = @import("svd_least_squares_test.zig");
    _ = @import("interop/gelss_adapter_test.zig");
}

comptime {
    _ = @import("svd_least_squares_borrowed_test.zig");
    _ = @import("interop/complex_gelsd_bidiagonal_info_test.zig");
}

comptime {
    _ = @import("interop/complex_gelsd_rhs_scale_test.zig");
}
