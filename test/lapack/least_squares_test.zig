// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const lapack = @import("zynum-lapack");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const expectError = std.testing.expectError;

fn matrix(comptime T: type, values: []T, m: usize, n: usize, ld: usize) lapack.Matrix(T) {
    return .{ .values = values, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}

fn same(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn tolerance(comptime T: type) f64 {
    return if (T == f32) 3e-5 else 5e-13;
}

fn near(comptime T: type, expected: f64, actual: f64) !void {
    try expect(std.math.isFinite(actual));
    try std.testing.expectApproxEqAbs(expected, actual, tolerance(T) * @max(@as(f64, 1), @abs(expected)));
}

fn checkSolution(comptime T: type, comptime m: usize, comptime n: usize, comptime nrhs: usize, original_a: []const T, original_b: []const T, factor: []const T, tau: []const T, result: []const T, lda: usize, ldb: usize) !void {
    var scale_a: f64 = 0;
    var scale_b: f64 = 0;
    for (original_a) |value| scale_a = @max(scale_a, @abs(@as(f64, @floatCast(value))));
    for (0..nrhs) |j| for (0..m) |i| {
        scale_b = @max(scale_b, @abs(@as(f64, @floatCast(original_b[i + j * ldb]))));
    };
    if (scale_a == 0) scale_a = 1;
    if (scale_b == 0) scale_b = 1;
    for (0..nrhs) |j| {
        var residual: [m]f64 = undefined;
        var expected_transformed: [m]f64 = undefined;
        for (0..m) |i| {
            var product: f64 = 0;
            for (0..n) |k| product += (@as(f64, @floatCast(original_a[i + k * m])) / scale_b) * @as(f64, @floatCast(result[k + j * ldb]));
            residual[i] = @as(f64, @floatCast(original_b[i + j * ldb])) / scale_b - product;
            expected_transformed[i] = @as(f64, @floatCast(original_b[i + j * ldb])) / scale_b;
        }
        // Normal equations are an oracle only: the solver never forms A^T A.
        for (0..n) |k| {
            var orthogonality: f64 = 0;
            for (0..m) |i| orthogonality += (@as(f64, @floatCast(original_a[i + k * m])) / scale_a) * residual[i];
            try near(T, 0, orthogonality);
        }
        // Independent compact-reflector product in scaled f64 checks the
        // definition of each tail coordinate, not only its combined norm.
        for (0..n) |step| {
            var dot = expected_transformed[step];
            for (step + 1..m) |i| dot += @as(f64, @floatCast(factor[i + step * lda])) * expected_transformed[i];
            const weight = @as(f64, @floatCast(tau[step])) * dot;
            expected_transformed[step] -= weight;
            for (step + 1..m) |i| expected_transformed[i] -= @as(f64, @floatCast(factor[i + step * lda])) * weight;
        }
        var residual_squared: f64 = 0;
        for (residual) |value| residual_squared += value * value;
        var tail_squared: f64 = 0;
        for (n..m) |i| {
            const value = @as(f64, @floatCast(result[i + j * ldb])) / scale_b;
            try near(T, expected_transformed[i], value);
            tail_squared += value * value;
        }
        try near(T, residual_squared, tail_squared);
    }
}

fn systemCase(comptime T: type, comptime m: usize, comptime n: usize, comptime nrhs: usize, scale: T, noisy: bool) !void {
    const lda = m + 2;
    const ldb = m + 3;
    var original_a: [m * n]T = undefined;
    var data: [lda * n]T = @splat(std.math.nan(T));
    for (0..n) |j| for (0..m) |i| {
        const integer: i32 = @as(i32, @intCast((i * 7 + j * 11) % 13)) - 6;
        const coefficient: T = if (i == j) 3 else @as(T, @floatFromInt(integer)) / 32;
        original_a[i + j * m] = scale * coefficient;
        data[i + j * lda] = original_a[i + j * m];
    };
    var expected: [n * nrhs]T = undefined;
    for (0..nrhs) |j| for (0..n) |i| {
        expected[i + j * n] = @as(T, @floatFromInt(1 + (i * 3 + j) % 7)) / 8;
    };
    var rhs: [ldb * nrhs]T = @splat(std.math.nan(T));
    for (0..nrhs) |j| for (0..m) |i| {
        var product: f64 = 0;
        for (0..n) |k| product += @as(f64, @floatCast(original_a[i + k * m])) * @as(f64, @floatCast(expected[k + j * n]));
        if (noisy) product += @as(f64, @floatCast(scale)) * (@as(f64, @floatFromInt((i + j) % 3)) - 1) / 16;
        rhs[i + j * ldb] = @floatCast(product);
    };
    const original_rhs = rhs;
    var tau: [n + 2]T = @splat(std.math.nan(T));
    const a = matrix(T, &data, m, n, lda);
    try equal(@as(usize, 0), try lapack.qrFactorize(T, a, &tau));
    const saved_factor = data;
    const saved_tau = tau;
    try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, a.asConst(), &tau, matrix(T, &rhs, m, nrhs, ldb)));
    try checkSolution(T, m, n, nrhs, &original_a, &original_rhs, &data, &tau, &rhs, lda, ldb);
    for (0..nrhs) |j| {
        if (!noisy) for (0..n) |i| try near(T, @floatCast(expected[i + j * n]), @floatCast(rhs[i + j * ldb]));
        try same(T, original_rhs[m + j * ldb ..][0..3], rhs[m + j * ldb ..][0..3]);
    }
    try same(T, &saved_factor, &data);
    try same(T, &saved_tau, &tau);
}

test "Prefactored least squares solves exact square and tall systems with multiple RHS" {
    inline for (.{ f32, f64 }) |T| {
        try systemCase(T, 1, 1, 1, 1, false);
        try systemCase(T, 4, 4, 3, 1, false);
        try systemCase(T, 9, 9, 8, 1, false);
        try systemCase(T, 7, 3, 3, 1, false);
        try systemCase(T, 17, 8, 8, 1, false);
    }
}

test "Tall noisy residuals are orthogonal to A and match each defined tail coordinate" {
    inline for (.{ f32, f64 }) |T| {
        try systemCase(T, 7, 3, 3, 1, true);
        try systemCase(T, 17, 8, 8, 1, true);
    }
}

test "Known orthogonal noise leaves the independently known least-squares coefficients" {
    inline for (.{ f32, f64 }) |T| {
        const m = 5;
        const n = 2;
        const original_a = [_]T{ 1, 1, 1, 1, 1, -3, -1, 0, 1, 3 };
        const noise = [_]T{ 1, -2, 2, -2, 1 };
        var factor = original_a;
        var tau: [n]T = undefined;
        var rhs: [m * 3]T = undefined;
        for (0..3) |j| for (0..m) |i| {
            const intercept = @as(T, @floatFromInt(j + 1));
            const slope = -@as(T, @floatFromInt(j + 1)) / 4;
            rhs[i + j * m] = intercept + original_a[i + m] * slope + noise[i] * @as(T, @floatFromInt(j + 1)) / 8;
        };
        const original_b = rhs;
        const a = matrix(T, &factor, m, n, m);
        try equal(@as(usize, 0), try lapack.qrFactorize(T, a, &tau));
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, a.asConst(), &tau, matrix(T, &rhs, m, 3, m)));
        for (0..3) |j| {
            try near(T, @floatFromInt(j + 1), @floatCast(rhs[j * m]));
            try near(T, -@as(f64, @floatFromInt(j + 1)) / 4, @floatCast(rhs[j * m + 1]));
        }
        try checkSolution(T, m, n, 3, &original_a, &original_b, &factor, &tau, &rhs, m, m);
    }
}

test "Huge and tiny common scales preserve finite full-rank solutions and residuals" {
    inline for (.{ f32, f64 }) |T| {
        const huge: T = if (T == f32) 1e30 else 1e300;
        const tiny: T = if (T == f32) 1e-30 else 1e-300;
        try systemCase(T, 5, 3, 2, huge, false);
        try systemCase(T, 5, 3, 2, tiny, true);
        try systemCase(T, 3, 3, 2, std.math.floatMin(T) * 128, false);
    }
}

test "Exactly zero diagonal rejects before mutation while nonzero subnormal pivots are accepted" {
    inline for (.{ f32, f64 }) |T| {
        var factor = [_]T{ 1, 0, 0, 2, 0, 0 };
        const tau = [_]T{ 0, 0 };
        var rhs = [_]T{ 3, 4, 5, 6, 7, 8 };
        const before = rhs;
        const a = matrix(T, &factor, 3, 2, 3).asConst();
        try equal(@as(usize, 2), try lapack.qrLeastSquaresSolve(T, a, &tau, matrix(T, &rhs, 3, 2, 3)));
        try same(T, &before, &rhs);
        const tiny = std.math.floatMin(T) / 32;
        factor[3] = 0;
        factor[4] = tiny;
        rhs = .{ 2, tiny * 3, 7, -1, -tiny * 2, 9 };
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, a, &tau, matrix(T, &rhs, 3, 2, 3)));
        try std.testing.expectEqualSlices(T, &.{ 2, 3, 7, -1, -2, 9 }, &rhs);
    }
}

test "Invalid storage shapes operations and wide systems fail before RHS mutation" {
    var factor = [_]f64{ 1, 0, 0, 1, 0, 0 };
    const tau = [_]f64{ 0, 0, 0 };
    var rhs = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const before = rhs;
    const a = matrix(f64, &factor, 3, 2, 3).asConst();
    const b = matrix(f64, &rhs, 3, 2, 3);
    try expectError(error.DimensionMismatch, lapack.qrLeastSquaresSolve(f64, matrix(f64, &factor, 2, 3, 2).asConst(), &tau, matrix(f64, &rhs, 2, 2, 2)));
    try expectError(error.DimensionMismatch, lapack.qrLeastSquaresSolve(f64, a, &tau, matrix(f64, &rhs, 2, 2, 2)));
    try expectError(error.InvalidMatrixOperation, lapack.qrLeastSquaresSolve(f64, a.transposed(), &tau, b));
    try expectError(error.InvalidMatrixOperation, lapack.qrLeastSquaresSolve(f64, a.adjoint(), &tau, b));
    try expectError(error.BufferTooSmall, lapack.qrLeastSquaresSolve(f64, a, tau[0..1], b));
    var bad = a;
    bad.values = factor[0..5];
    try expectError(error.BufferTooSmall, lapack.qrLeastSquaresSolve(f64, bad, &tau, b));
    bad = a;
    bad.leading_dimension = 2;
    try expectError(error.InvalidLeadingDimension, lapack.qrLeastSquaresSolve(f64, bad, &tau, b));
    bad = a;
    bad.row_count = -1;
    try expectError(error.InvalidLength, lapack.qrLeastSquaresSolve(f64, bad, &tau, b));
    try expectError(error.BufferTooSmall, lapack.qrLeastSquaresSolve(f64, a, &tau, matrix(f64, rhs[0..5], 3, 2, 3)));
    try same(f64, &before, &rhs);
}

test "Every finite and reflector check precedes zero-diagonal status and mutation" {
    var factor = [_]f64{ 1, 0, 0, 2, 0, 0 };
    var tau = [_]f64{ 0, 0 };
    var rhs = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const a = matrix(f64, &factor, 3, 2, 3).asConst();
    const b = matrix(f64, &rhs, 3, 2, 3);
    const before = rhs;
    tau[1] = 3;
    try expectError(error.InvalidReflector, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    tau[1] = std.math.nan(f64);
    try expectError(error.NonFiniteInput, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    tau[1] = 0;
    factor[5] = 2;
    try expectError(error.InvalidReflector, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    factor[5] = std.math.inf(f64);
    try expectError(error.NonFiniteInput, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    factor[5] = 0;
    factor[3] = std.math.nan(f64); // R is used by the solve, unlike qrApplyQ.
    try expectError(error.NonFiniteInput, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    try same(f64, &before, &rhs);
    factor[3] = 2;
    rhs[5] = std.math.inf(f64);
    const nonfinite_rhs = rhs;
    try expectError(error.NonFiniteInput, lapack.qrLeastSquaresSolve(f64, a, &tau, b));
    try same(f64, &nonfinite_rhs, &rhs);
}

test "Factor tau and RHS used regions reject pairwise aliasing including padding" {
    inline for (.{ f32, f64 }) |T| {
        var storage: [24]T = @splat(0);
        storage[0] = 1;
        storage[5] = 1;
        var tau = [_]T{ 0, 0 };
        var rhs = [_]T{ 1, 2, 3, 4, 5, 6 };
        const before = storage;
        const before_rhs = rhs;
        const a = matrix(T, storage[0..7], 3, 2, 4).asConst();
        try expectError(error.AliasingNotAllowed, lapack.qrLeastSquaresSolve(T, a, &tau, matrix(T, storage[3..9], 3, 2, 3)));
        try expectError(error.AliasingNotAllowed, lapack.qrLeastSquaresSolve(T, a, storage[3..5], matrix(T, &rhs, 3, 2, 3)));
        try expectError(error.AliasingNotAllowed, lapack.qrLeastSquaresSolve(T, a, rhs[4..6], matrix(T, &rhs, 3, 2, 3)));
        try same(T, &before, &storage);
        try same(T, &before_rhs, &rhs);
    }
}

test "Zero shapes preserve the residual tail and zero RHS still validates reflectors" {
    inline for (.{ f32, f64 }) |T| {
        var empty: [0]T = .{};
        const unused_tau = [_]T{std.math.nan(T)};
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, matrix(T, &empty, 0, 0, 0).asConst(), &unused_tau, matrix(T, &empty, 0, 3, 0)));
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, matrix(T, &empty, 0, 0, 0).asConst(), &unused_tau, matrix(T, &empty, 0, std.math.maxInt(lapack.PivotInt), 0)));
        var rhs = [_]T{ 1, 2, 3, 4, 5, 6 };
        const before = rhs;
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, matrix(T, &empty, 3, 0, 0).asConst(), &unused_tau, matrix(T, &rhs, 3, 2, 3)));
        try same(T, &before, &rhs);
        var factor = [_]T{ 1, 0, 0, 0 };
        var tau = [_]T{ 0, 0 };
        const a = matrix(T, &factor, 2, 2, 2).asConst();
        try equal(@as(usize, 0), try lapack.qrLeastSquaresSolve(T, a, &tau, matrix(T, &empty, 2, 0, 0)));
        tau[1] = 3;
        try expectError(error.InvalidReflector, lapack.qrLeastSquaresSolve(T, a, &tau, matrix(T, &empty, 2, 0, 0)));
        var bad = matrix(T, &empty, 0, 0, 0).asConst();
        bad.leading_dimension = -1;
        try expectError(error.InvalidLeadingDimension, lapack.qrLeastSquaresSolve(T, bad, &empty, matrix(T, &empty, 0, 0, 0)));
    }
}

test "Numerical failures identify QR reflector or triangular row without changing factors" {
    inline for (.{ f32, f64 }) |T| {
        var factor = [_]T{ 1, 0, 0, std.math.floatMin(T) };
        const saved_factor = factor;
        const tau = [_]T{ 0, 0 };
        var rhs = [_]T{ 1, std.math.floatMax(T), -987 };
        try equal(@as(usize, 2), try lapack.qrLeastSquaresSolve(T, matrix(T, &factor, 2, 2, 2).asConst(), &tau, matrix(T, &rhs, 2, 1, 3)));
        try equal(@as(T, -987), rhs[2]);
        try same(T, &saved_factor, &factor);
    }
    var factor = [_]f64{ 1, 0, 0, 0, 1, -0.4142135623730951 };
    const before = factor;
    const tau = [_]f64{ 0, 1.7071067811865475 };
    var rhs = [_]f64{ 0, std.math.floatMax(f64), -std.math.floatMax(f64), -987 };
    try equal(@as(usize, 2), try lapack.qrLeastSquaresSolve(f64, matrix(f64, &factor, 3, 2, 3).asConst(), &tau, matrix(f64, &rhs, 3, 1, 4)));
    try equal(@as(f64, -987), rhs[3]);
    try same(f64, &before, &factor);
}
