// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const lapack = @import("zynum-lapack");

fn matrix(comptime T: type, data: []T, rows: usize, columns: usize, ld: usize) lapack.Matrix(T) {
    return .{ .values = data, .row_count = @intCast(rows), .column_count = @intCast(columns), .leading_dimension = @intCast(ld) };
}

fn sameBytes(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn tolerance(comptime T: type) f64 {
    return if (T == f32) 2e-5 else 3e-13;
}

fn near(comptime T: type, expected: f64, actual: f64) !void {
    try std.testing.expect(std.math.isFinite(actual));
    try std.testing.expectApproxEqAbs(expected, actual, tolerance(T) * @max(@as(f64, 1), @abs(expected)));
}

fn choleskyBoundary(comptime T: type, comptime n: usize, scale: T, triangle: lapack.Triangle) !void {
    const ld = n + 3;
    var data: [ld * n]T = @splat(std.math.nan(T));
    const off_diagonal = scale / @as(T, @floatFromInt(4 * n));
    for (0..n) |j| for (0..n) |i| {
        if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) data[i + j * ld] = if (i == j) scale else off_diagonal;
    };
    const before = data;
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(T, matrix(T, &data, n, n, ld), triangle));
    // Reconstruct in independently scaled f64 arithmetic so huge and
    // subnormal source magnitudes cannot overflow/underflow the oracle.
    const normalized_root = @sqrt(@as(f64, @floatCast(scale)));
    for (0..n) |j| {
        for (0..n) |i| {
            var product: f64 = 0;
            for (0..@min(i, j) + 1) |k| {
                const first: f64 = @floatCast(if (triangle == .lower) data[i + k * ld] else data[k + i * ld]);
                const second: f64 = @floatCast(if (triangle == .lower) data[j + k * ld] else data[k + j * ld]);
                product += (first / normalized_root) * (second / normalized_root);
            }
            const expected: f64 = if (i == j) 1 else @as(f64, @floatCast(off_diagonal)) / @as(f64, @floatCast(scale));
            try near(T, expected, product);
            if ((triangle == .lower and i < j) or (triangle == .upper and i > j)) try sameBytes(T, before[i + j * ld ..][0..1], data[i + j * ld ..][0..1]);
        }
        try sameBytes(T, before[n + j * ld ..][0..3], data[n + j * ld ..][0..3]);
    }
}

test "Cholesky specialized paths reconstruct at vector boundaries and extreme finite scales" {
    inline for (.{ f32, f64 }) |T| {
        for ([_]lapack.Triangle{ .lower, .upper }) |triangle| {
            inline for (.{ 1, 7, 8, 9, 17, 33 }) |n| try choleskyBoundary(T, n, 4, triangle);
            try choleskyBoundary(T, 9, std.math.floatMin(T) * 4, triangle);
            const smallest: T = if (T == f32) 0x1p-149 else 0x1p-1074;
            try choleskyBoundary(T, 9, smallest * 4, triangle);
            try choleskyBoundary(T, 9, std.math.floatMax(T) / 4, triangle);
        }
    }
}

fn luBoundary(comptime T: type, comptime rows: usize, comptime columns: usize, singular: bool) !void {
    const ld = rows + 3;
    const steps: usize = @min(rows, columns);
    var original: [rows * columns]T = undefined;
    var data: [ld * columns]T = @splat(-987);
    for (0..columns) |j| for (0..rows) |i| {
        const value: i32 = @as(i32, @intCast((i * 17 + j * 13) % 23)) - 11;
        original[i + j * rows] = if (singular and j == 0) 0 else @as(T, @floatFromInt(value)) / 64 + if (rows - 1 - i == j) @as(T, 4) else 0;
        data[i + j * ld] = original[i + j * rows];
    };
    var pivots: [steps + 2]lapack.PivotInt = @splat(-77);
    try std.testing.expectEqual(@as(usize, if (singular) 1 else 0), try lapack.luFactorize(T, matrix(T, &data, rows, columns, ld), &pivots));
    try std.testing.expectEqual(@as(lapack.PivotInt, if (singular) 0 else rows - 1), pivots[0]);
    for (0..steps) |k| {
        try std.testing.expect(pivots[k] >= 0);
        const pivot: usize = @intCast(pivots[k]);
        try std.testing.expect(pivot >= k and pivot < rows);
        for (0..columns) |j| std.mem.swap(T, &original[k + j * rows], &original[pivot + j * rows]);
    }
    for (0..columns) |j| {
        for (0..rows) |i| {
            var product: f64 = 0;
            for (0..steps) |k| {
                const l: f64 = if (i == k) 1 else if (i > k) @floatCast(data[i + k * ld]) else 0;
                const u: f64 = if (k <= j) @floatCast(data[k + j * ld]) else 0;
                product += l * u;
            }
            try near(T, @floatCast(original[i + j * rows]), product);
        }
        for (rows..ld) |i| try std.testing.expectEqual(@as(T, -987), data[i + j * ld]);
    }
    try std.testing.expectEqualSlices(lapack.PivotInt, &.{ -77, -77 }, pivots[steps..]);
}

test "LU vector and tail updates reconstruct rectangular and singular pivoted matrices" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ .{ 7, 9 }, .{ 8, 8 }, .{ 9, 7 }, .{ 17, 21 }, .{ 33, 17 } }) |shape| {
            try luBoundary(T, shape[0], shape[1], false);
            try luBoundary(T, shape[0], shape[1], true);
        }
    }
}

test "LU nonfinite vector updates retain the finite prefix and earliest failure index" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ false, true }) |earlier_zero_pivot| {
            const rows = 12;
            const columns = 3;
            const ld = rows + 3;
            const k: usize = if (earlier_zero_pivot) 1 else 0;
            inline for (.{ 2, 5, 10, 11 }) |failing_row| {
                var data: [ld * columns]T = @splat(-987);
                for (0..columns) |j| for (0..rows) |i| {
                    data[i + j * ld] = 0;
                };
                for (k..rows) |i| data[i + k * ld] = 1;
                data[k + (k + 1) * ld] = -std.math.floatMax(T);
                data[failing_row + (k + 1) * ld] = std.math.floatMax(T);
                var expected = data;
                // Multipliers are exactly one. Earlier destination zeros
                // become +MAX; MAX - (-MAX) overflows at failing_row.
                for (k + 1..failing_row) |i| expected[i + (k + 1) * ld] = std.math.floatMax(T);
                var pivots: [5]lapack.PivotInt = @splat(-77);
                try std.testing.expectEqual(@as(usize, 1), try lapack.luFactorize(T, matrix(T, &data, rows, columns, ld), &pivots));
                try sameBytes(T, &expected, &data);
                for (0..k + 1) |i| try std.testing.expectEqual(@as(lapack.PivotInt, @intCast(i)), pivots[i]);
                for (pivots[k + 1 ..]) |pivot| try std.testing.expectEqual(@as(lapack.PivotInt, -77), pivot);
            }
        }
    }
}

const SolveKind = enum { lu, lower, upper };

fn solveBoundary(comptime T: type, comptime n: usize, comptime rhs_count: usize, kind: SolveKind) !void {
    const lda = n + 2;
    const ldb = n + 3;
    var original: [n * n]T = undefined;
    var factor: [lda * n]T = @splat(-987);
    for (0..n) |j| for (0..n) |i| {
        const row = if (kind == .lu) n - 1 - i else i;
        const coefficient: i32 = @as(i32, @intCast((@min(row, j) * 7 + @max(row, j) * 11) % 13)) - 6;
        original[i + j * n] = if (row == j) 4 else @as(T, @floatFromInt(coefficient)) / @as(T, @floatFromInt(64 * n));
        factor[i + j * lda] = original[i + j * n];
    };
    var pivots: [n]lapack.PivotInt = undefined;
    const a = matrix(T, &factor, n, n, lda);
    const triangle: lapack.Triangle = if (kind == .upper) .upper else .lower;
    const info = if (kind == .lu) try lapack.luFactorize(T, a, &pivots) else try lapack.choleskyFactorize(T, a, triangle);
    try std.testing.expectEqual(@as(usize, 0), info);
    var expected: [n * rhs_count]T = undefined;
    for (0..rhs_count) |j| for (0..n) |i| {
        expected[i + j * n] = @as(T, @floatFromInt((i * 5 + j * 3) % 17)) / 8;
    };
    var rhs: [ldb * rhs_count]T = @splat(-654);
    for (0..rhs_count) |j| for (0..n) |i| {
        var product: f64 = 0;
        for (0..n) |k| product += @as(f64, @floatCast(original[i + k * n])) * @as(f64, @floatCast(expected[k + j * n]));
        rhs[i + j * ldb] = @floatCast(product);
    };
    const saved_rhs = rhs;
    const saved_factor = factor;
    const saved_pivots = if (kind == .lu) pivots else @as([n]lapack.PivotInt, @splat(0));
    const solved = if (kind == .lu) try lapack.luSolve(T, a.asConst(), &pivots, matrix(T, &rhs, n, rhs_count, ldb)) else try lapack.choleskySolve(T, a.asConst(), matrix(T, &rhs, n, rhs_count, ldb), triangle);
    try std.testing.expectEqual(@as(usize, 0), solved);
    for (0..rhs_count) |j| {
        for (0..n) |i| {
            try near(T, @floatCast(expected[i + j * n]), @floatCast(rhs[i + j * ldb]));
            var product: f64 = 0;
            for (0..n) |k| product += @as(f64, @floatCast(original[i + k * n])) * @as(f64, @floatCast(rhs[k + j * ldb]));
            try near(T, @floatCast(saved_rhs[i + j * ldb]), product);
        }
        try sameBytes(T, saved_rhs[n + j * ldb ..][0..3], rhs[n + j * ldb ..][0..3]);
    }
    try sameBytes(T, &saved_factor, &factor);
    if (kind == .lu) try sameBytes(lapack.PivotInt, &saved_pivots, &pivots);
}

test "Column forward solves satisfy independent residuals for multiple RHS and boundary sizes" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ 1, 7, 8, 9, 17, 33 }) |n| {
            inline for (.{ 1, 8 }) |rhs_count| {
                for ([_]SolveKind{ .lu, .lower, .upper }) |kind| try solveBoundary(T, n, rhs_count, kind);
            }
        }
    }
}

test "Forward overflow identifies its row while preserving padding factors and later RHS" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ false, true }) |use_lu| {
            const n = 12;
            const ld = n + 3;
            var factor: [ld * n]T = @splat(-987);
            for (0..n) |j| for (0..n) |i| {
                factor[i + j * ld] = if (i == j) 1 else 0;
            };
            factor[5] = std.math.floatMax(T);
            const saved_factor = factor;
            var pivots: [n]lapack.PivotInt = undefined;
            for (0..n) |i| pivots[i] = @intCast(i);
            const saved_pivots = pivots;
            var rhs: [ld * 2]T = @splat(2);
            const saved_rhs = rhs;
            const a = matrix(T, &factor, n, n, ld).asConst();
            const info = if (use_lu) try lapack.luSolve(T, a, &pivots, matrix(T, &rhs, n, 2, ld)) else try lapack.choleskySolve(T, a, matrix(T, &rhs, n, 2, ld), .lower);
            try std.testing.expectEqual(@as(usize, 6), info);
            try sameBytes(T, saved_rhs[n..], rhs[n..]);
            try sameBytes(T, &saved_factor, &factor);
            try sameBytes(lapack.PivotInt, &saved_pivots, &pivots);
        }
    }
}

test "Backward substitution retains ascending contribution order and checks overflow" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ false, true }) |overflow| {
            const large = std.math.floatMax(T) / (if (overflow) @as(T, 1) else 4);
            var factor = [_]T{ 1, 0, 0, 0, 1, 1, 0, 0, large, 0, 1, 0, -large, 0, 0, 1 };
            const pivots = [_]lapack.PivotInt{ 0, 1, 2, 3 };
            var rhs = [_]T{ 3, 1, 2, 2 };
            const info = try lapack.luSolve(T, matrix(T, &factor, 4, 4, 4).asConst(), &pivots, matrix(T, &rhs, 4, 1, 4));
            try std.testing.expectEqual(@as(usize, if (overflow) 1 else 0), info);
            // Finite ascending order loses the small residual before the
            // large cancellation, yielding zero; reverse order yields -1.
            try std.testing.expectEqual(@as(T, if (overflow) 3 else 0), rhs[0]);
            try std.testing.expectEqualSlices(T, &.{ 1, 2, 2 }, rhs[1..]);
        }
    }
}

test "Optimized operand sizes retain alias rejection and full selected-triangle preflight" {
    inline for (.{ f32, f64 }) |T| {
        const n = 9;
        const ld = n + 3;
        var storage: [ld * n + n]T = @splat(1);
        var pivots: [n]lapack.PivotInt = undefined;
        for (0..n) |i| pivots[i] = @intCast(i);
        const before = storage;
        const a = matrix(T, storage[0 .. ld * n], n, n, ld).asConst();
        const overlap = matrix(T, storage[n..][0..n], n, 1, n);
        for ([_]lapack.Triangle{ .lower, .upper }) |triangle| try std.testing.expectError(error.AliasingNotAllowed, lapack.choleskySolve(T, a, overlap, triangle));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(T, a, &pivots, overlap));
        try sameBytes(T, &before, &storage);
        storage[n - 1 + (n - 1) * ld] = std.math.inf(T);
        const invalid = storage;
        const saved_pivots = pivots;
        for ([_]lapack.Triangle{ .lower, .upper }) |triangle| try std.testing.expectError(error.NonFiniteInput, lapack.choleskyFactorize(T, matrix(T, storage[0 .. ld * n], n, n, ld), triangle));
        try std.testing.expectError(error.NonFiniteInput, lapack.luFactorize(T, matrix(T, storage[0 .. ld * n], n, n, ld), &pivots));
        try sameBytes(T, &invalid, &storage);
        try sameBytes(lapack.PivotInt, &saved_pivots, &pivots);
    }
}
