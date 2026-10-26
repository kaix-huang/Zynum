// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const lapack = @import("zynum-lapack");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const Z = struct { re: f64, im: f64 };
fn real(comptime C: type) type {
    return if (C == lapack.ComplexF32) f32 else f64;
}
fn z(comptime C: type, re: f64, im: f64) C {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn scaled(a: anytype, scale: f64) Z {
    return .{ .re = @as(f64, a.re) / scale, .im = @as(f64, a.im) / scale };
}
fn nan(comptime C: type) C {
    return .{ .re = std.math.nan(real(C)), .im = std.math.nan(real(C)) };
}
fn matrix(comptime C: type, data: []C, m: usize, n: usize, ld: usize) lapack.Matrix(C) {
    return .{ .values = data, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}
fn constant(comptime C: type, data: []const C, m: usize, n: usize, ld: usize) lapack.ConstMatrix(C) {
    return .{ .values = data, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld), .operation = .normal };
}
fn same(comptime T: type, a: []const T, b: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn near(a: Z, b: Z, tol: f64) !void {
    try expect(std.math.isFinite(a.re) and std.math.isFinite(a.im) and std.math.isFinite(b.re) and std.math.isFinite(b.im));
    try std.testing.expectApproxEqAbs(a.re, b.re, tol);
    try std.testing.expectApproxEqAbs(a.im, b.im, tol);
}
fn tolerance(comptime C: type, n: usize) f64 {
    return 128 * @as(f64, std.math.floatEps(real(C))) * @as(f64, @floatFromInt(@max(n, 1)));
}
fn padding(comptime C: type, data: []const C, m: usize, n: usize, ld: usize) !void {
    for (0..n) |j| for (m..ld) |i| {
        try expect(std.math.isNan(data[i + j * ld].re) and std.math.isNan(data[i + j * ld].im));
    };
    for (data[ld * n ..]) |v| try expect(std.math.isNan(v.re) and std.math.isNan(v.im));
}
fn knownX(i: usize, j: usize, n: usize) Z {
    return .{ .re = @as(f64, @floatFromInt(i + 1)) / (10 * @as(f64, @floatFromInt(@max(n, 1)))), .im = -@as(f64, @floatFromInt(j + 1)) / 16 };
}
fn makeRhs(comptime C: type, comptime n: usize, comptime nrhs: usize, a: []const C, lda: usize, scale: f64, b: []C, ldb: usize) void {
    for (0..nrhs) |j| for (0..n) |i| {
        var value: Z = .{ .re = 0, .im = 0 };
        for (0..n) |k| value = add(value, mul(scaled(a[i + k * lda], scale), knownX(k, j, n)));
        b[i + j * ldb] = z(C, value.re * scale, value.im * scale);
    };
}
fn checkSolution(comptime C: type, comptime n: usize, comptime nrhs: usize, a: []const C, lda: usize, b: []const C, x: []const C, ldb: usize, scale: f64) !void {
    const tol = tolerance(C, n);
    for (0..nrhs) |j| for (0..n) |i| {
        try near(knownX(i, j, n), scaled(x[i + j * ldb], 1), tol);
        var value: Z = .{ .re = 0, .im = 0 };
        for (0..n) |k| value = add(value, mul(scaled(a[i + k * lda], scale), scaled(x[k + j * ldb], 1)));
        try near(scaled(b[i + j * ldb], scale), value, tol * 4);
    };
}
fn lowerEntry(i: usize, j: usize, n: usize) Z {
    if (i < j) return .{ .re = 0, .im = 0 };
    if (i == j) return .{ .re = 2 + @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)), .im = 0 };
    return .{ .re = (@as(f64, @floatFromInt((i + 2 * j) % 7)) - 3) / 16, .im = (@as(f64, @floatFromInt((3 * i + j) % 5)) - 2) / 12 };
}
fn choleskyCase(comptime C: type, comptime n: usize, comptime nrhs: usize, triangle: lapack.Triangle, scale: f64) !void {
    const ld = n + 3;
    const ldb = n + 2;
    var original: [ld * n + 2]C = @splat(nan(C));
    var a: [ld * n + 2]C = @splat(nan(C));
    for (0..n) |j| for (0..n) |i| {
        var value: Z = .{ .re = 0, .im = 0 };
        for (0..n) |k| value = add(value, mul(lowerEntry(i, k, n), conj(lowerEntry(j, k, n))));
        if (i == j) value.im = 0;
        original[i + j * ld] = z(C, value.re * scale, value.im * scale);
        if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) a[i + j * ld] = original[i + j * ld];
    };
    const before = a;
    try equal(@as(usize, 0), try lapack.choleskyFactorize(C, matrix(C, &a, n, n, ld), triangle));
    const factor = a;
    const root_scale = @sqrt(scale);
    for (0..n) |j| for (0..n) |i| {
        var reconstructed: Z = .{ .re = 0, .im = 0 };
        for (0..@min(i, j) + 1) |k| {
            const lik = if (triangle == .lower) scaled(a[i + k * ld], root_scale) else conj(scaled(a[k + i * ld], root_scale));
            const ljk = if (triangle == .lower) scaled(a[j + k * ld], root_scale) else conj(scaled(a[k + j * ld], root_scale));
            reconstructed = add(reconstructed, mul(lik, conj(ljk)));
        }
        try near(scaled(original[i + j * ld], scale), reconstructed, tolerance(C, n) * 4);
        if ((triangle == .lower and i < j) or (triangle == .upper and i > j)) try same(C, before[i + j * ld ..][0..1], a[i + j * ld ..][0..1]);
    };
    for (0..n) |i| {
        try expect(a[i + i * ld].re > 0);
        try equal(@as(real(C), 0), a[i + i * ld].im);
    }
    var rhs: [ldb * nrhs + 2]C = @splat(nan(C));
    makeRhs(C, n, nrhs, &original, ld, scale, &rhs, ldb);
    const b = rhs;
    try equal(@as(usize, 0), try lapack.choleskySolve(C, constant(C, &a, n, n, ld), matrix(C, &rhs, n, nrhs, ldb), triangle));
    try checkSolution(C, n, nrhs, &original, ld, &b, &rhs, ldb, scale);
    try same(C, &factor, &a);
    try padding(C, &a, n, n, ld);
    try padding(C, &rhs, n, nrhs, ldb);
}

test "complex Cholesky Hermitian reconstruction and conjugate multiple-RHS solve for both triangles" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        try choleskyCase(C, 1, 1, triangle, 1);
        try choleskyCase(C, 3, 3, triangle, 1);
        try choleskyCase(C, 9, 8, triangle, 1);
    };
}
test "complex Cholesky supports extreme finite common scales" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        try choleskyCase(C, 5, 3, triangle, if (C == lapack.ComplexF32) 1e30 else 1e300);
        try choleskyCase(C, 5, 3, triangle, if (C == lapack.ComplexF32) 1e-30 else 1e-300);
    };
}

fn checkLU(comptime C: type, comptime m: usize, comptime n: usize, original: []const C, a: []const C, ld: usize, pivots: []const lapack.PivotInt, scale: f64) !void {
    const steps: usize = @min(m, n);
    var perm: [m]usize = undefined;
    for (0..m) |i| perm[i] = i;
    for (0..steps) |k| {
        try expect(pivots[k] >= k and pivots[k] < m);
        std.mem.swap(usize, &perm[k], &perm[@intCast(pivots[k])]);
    }
    for (0..n) |j| for (0..m) |i| {
        var reconstructed: Z = .{ .re = 0, .im = 0 };
        for (0..steps) |k| {
            const l: Z = if (i == k) .{ .re = 1, .im = 0 } else if (i > k) scaled(a[i + k * ld], 1) else .{ .re = 0, .im = 0 };
            const u: Z = if (k <= j) scaled(a[k + j * ld], scale) else .{ .re = 0, .im = 0 };
            reconstructed = add(reconstructed, mul(l, u));
        }
        try near(scaled(original[perm[i] + j * ld], scale), reconstructed, tolerance(C, @max(m, n)) * 4);
    };
}
fn luCase(comptime C: type, comptime m: usize, comptime n: usize, scale: f64) !void {
    const ld = m + 3;
    const steps: usize = @min(m, n);
    var a: [ld * n + 2]C = @splat(nan(C));
    for (0..n) |j| for (0..m) |i| {
        const r = @as(f64, @floatFromInt((i * 7 + j * 3 + 1) % 11)) - 5;
        const im = @as(f64, @floatFromInt((i * 3 + j * 7 + 4) % 13)) - 6;
        a[i + j * ld] = z(C, scale * (r / 8 + @as(f64, if (i == j) 4 else 0)), scale * im / 8);
    };
    if (m > 1 and n > 0) {
        a[0] = z(C, 0, 0);
        a[m - 1] = z(C, 7 * scale, 2 * scale);
    }
    const original = a;
    var pivots: [steps + 2]lapack.PivotInt = @splat(-71);
    try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &a, m, n, ld), &pivots));
    if (m > 1 and n > 0) try equal(@as(lapack.PivotInt, @intCast(m - 1)), pivots[0]);
    try checkLU(C, m, n, &original, &a, ld, &pivots, scale);
    try padding(C, &a, m, n, ld);
    try equal(@as(lapack.PivotInt, -71), pivots[steps]);
    try equal(@as(lapack.PivotInt, -71), pivots[steps + 1]);
    if (m == n) {
        const nrhs = 3;
        const ldb = m + 2;
        var rhs: [ldb * nrhs + 2]C = @splat(nan(C));
        makeRhs(C, n, nrhs, &original, ld, scale, &rhs, ldb);
        const b = rhs;
        const factor = a;
        const pivots_before = pivots;
        try equal(@as(usize, 0), try lapack.luSolve(C, constant(C, &a, m, n, ld), &pivots, matrix(C, &rhs, m, nrhs, ldb)));
        try checkSolution(C, n, nrhs, &original, ld, &b, &rhs, ldb, scale);
        try same(C, &factor, &a);
        try same(lapack.PivotInt, &pivots_before, &pivots);
        try padding(C, &rhs, m, nrhs, ldb);
    }
}

test "complex LU rectangular reconstruction, pivot-required square multiple-RHS solves and padding" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        try luCase(C, 1, 1, 1);
        try luCase(C, 3, 3, 1);
        try luCase(C, 8, 8, 1);
        try luCase(C, 7, 3, 1);
        try luCase(C, 3, 7, 1);
    }
}
test "complex LU avoids large and tiny complex division intermediates" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        try luCase(C, 5, 5, if (C == lapack.ComplexF32) 1e30 else 1e300);
        try luCase(C, 5, 5, if (C == lapack.ComplexF32) 1e-30 else 1e-300);
        for ([_]real(C){ std.math.floatMax(real(C)) / 2, std.math.floatTrueMin(real(C)) }) |scale| {
            var factor = [_]C{.{ .re = scale, .im = scale }};
            var pivots: [1]lapack.PivotInt = undefined;
            try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &factor, 1, 1, 1), &pivots));
            var rhs = [_]C{.{ .re = scale, .im = -scale }};
            try equal(@as(usize, 0), try lapack.luSolve(C, constant(C, &factor, 1, 1, 1), &pivots, matrix(C, &rhs, 1, 1, 1)));
            try near(.{ .re = 0, .im = -1 }, scaled(rhs[0], 1), tolerance(C, 1));
        }
    }
}
test "complex LU pivot norm uses widened abs-real plus abs-imag and first-index ties" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var a = [_]C{ z(C, 0.875, 0), z(C, 0.5, 0.5), z(C, 0.75, 0.25) };
        var pivot: [1]lapack.PivotInt = undefined;
        try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &a, 3, 1, 3), &pivot));
        try equal(@as(lapack.PivotInt, 1), pivot[0]);
        const huge = std.math.floatMax(real(C));
        var big = [_]C{ .{ .re = huge / 2, .im = huge / 2 }, .{ .re = huge, .im = huge } };
        try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &big, 2, 1, 2), &pivot));
        try equal(@as(lapack.PivotInt, 1), pivot[0]);
        try near(.{ .re = 0.5, .im = 0 }, scaled(big[1], 1), tolerance(C, 1));
    }
}
test "complex LU singular rectangular matrices finish pivots and preserve reconstruction" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        const m = 4;
        const n = 3;
        const ld = 6;
        var a: [ld * n + 2]C = @splat(nan(C));
        for (0..m) |i| {
            a[i] = z(C, 0, 0);
            a[i + ld] = z(C, @floatFromInt(i + 1), @floatFromInt(i));
            a[i + 2 * ld] = z(C, @floatFromInt(2 * i + 3), -@as(f64, @floatFromInt(i)));
        }
        const original = a;
        var pivots = [_]lapack.PivotInt{ -71, -71, -71, -71 };
        try equal(@as(usize, 1), try lapack.luFactorize(C, matrix(C, &a, m, n, ld), &pivots));
        for (0..3) |k| try expect(pivots[k] >= k and pivots[k] < m);
        try equal(@as(lapack.PivotInt, 3), pivots[1]);
        try equal(@as(lapack.PivotInt, -71), pivots[3]);
        try checkLU(C, m, n, &original, &a, ld, &pivots, 1);
        try padding(C, &a, m, n, ld);
    }
}

test "complex Cholesky rejects nonreal diagonal before mutation and detects nonpositive pivots" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        var a = [_]C{ z(C, 1, 0), z(C, 0, 2), z(C, 0, -2), z(C, 1, 0.25) };
        const before = a;
        try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskyFactorize(C, matrix(C, &a, 2, 2, 2), triangle));
        try same(C, &before, &a);
        a[3].im = 0;
        try equal(@as(usize, 2), try lapack.choleskyFactorize(C, matrix(C, &a, 2, 2, 2), triangle));
        var factor = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 0, 0), z(C, -1, 0) };
        var rhs = [_]C{ z(C, 3, 1), z(C, 4, 2) };
        const rb = rhs;
        try equal(@as(usize, 2), try lapack.choleskySolve(C, constant(C, &factor, 2, 2, 2), matrix(C, &rhs, 2, 1, 2), triangle));
        try same(C, &rb, &rhs);
        factor[3] = z(C, 1, 0.5);
        try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskySolve(C, constant(C, &factor, 2, 2, 2), matrix(C, &rhs, 2, 1, 2), triangle));
        try same(C, &rb, &rhs);
    };
}

test "complex zero LU diagonal and all malformed pivot sequences preserve RHS" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var factor = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 2, 1), z(C, 0, 0) };
        var rhs = [_]C{ z(C, 3, 1), z(C, 4, 2) };
        const rb = rhs;
        var pivots = [_]lapack.PivotInt{ 0, 1 };
        try equal(@as(usize, 2), try lapack.luSolve(C, constant(C, &factor, 2, 2, 2), &pivots, matrix(C, &rhs, 2, 1, 2)));
        try same(C, &rb, &rhs);
        for ([_][2]lapack.PivotInt{ .{ -1, 1 }, .{ 2, 1 }, .{ 0, 0 } }) |bad| {
            try std.testing.expectError(error.InvalidPivot, lapack.luSolve(C, constant(C, &factor, 2, 2, 2), &bad, matrix(C, &rhs, 2, 1, 2)));
            try same(C, &rb, &rhs);
        }
    }
}

test "complex finite input checks cover both components and precede every mutation" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| inline for (.{ false, true }) |imaginary| {
        var a = [_]C{ z(C, 2, 0), z(C, 1, 1), z(C, 1, -1), z(C, 3, 0) };
        if (imaginary) a[3].im = std.math.nan(real(C)) else a[3].re = std.math.inf(real(C));
        const before = a;
        var pivots = [_]lapack.PivotInt{ -77, -77 };
        try std.testing.expectError(error.NonFiniteInput, lapack.luFactorize(C, matrix(C, &a, 2, 2, 2), &pivots));
        try same(C, &before, &a);
        try std.testing.expectEqualSlices(lapack.PivotInt, &.{ -77, -77 }, &pivots);
        try std.testing.expectError(error.NonFiniteInput, lapack.choleskyFactorize(C, matrix(C, &a, 2, 2, 2), .lower));
        try same(C, &before, &a);
        var rhs = [_]C{ z(C, 1, 0), z(C, 2, 0) };
        const rb = rhs;
        pivots = .{ 0, 1 };
        try std.testing.expectError(error.NonFiniteInput, lapack.luSolve(C, constant(C, &a, 2, 2, 2), &pivots, matrix(C, &rhs, 2, 1, 2)));
        try std.testing.expectError(error.NonFiniteInput, lapack.choleskySolve(C, constant(C, &a, 2, 2, 2), matrix(C, &rhs, 2, 1, 2), .upper));
        try same(C, &rb, &rhs);
        a[3] = z(C, 3, 0);
        if (imaginary) rhs[1].im = std.math.inf(real(C)) else rhs[1].re = std.math.nan(real(C));
        const invalid_rhs = rhs;
        try std.testing.expectError(error.NonFiniteInput, lapack.luSolve(C, constant(C, &a, 2, 2, 2), &pivots, matrix(C, &rhs, 2, 1, 2)));
        try std.testing.expectError(error.NonFiniteInput, lapack.choleskySolve(C, constant(C, &a, 2, 2, 2), matrix(C, &rhs, 2, 1, 2), .lower));
        try same(C, &invalid_rhs, &rhs);
    };
}

test "complex widened substitution avoids intermediate overflow with finite exact cancellation" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        const big = std.math.floatMax(real(C)) / 2;
        var factor = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 0, 0), .{ .re = big, .im = big }, z(C, 1, 0), z(C, 0, 0), .{ .re = -big, .im = -big }, z(C, 0, 0), z(C, 1, 0) };
        const pivots = [_]lapack.PivotInt{ 0, 1, 2 };
        var rhs = [_]C{ z(C, 0, 0), z(C, 4, 0), z(C, 4, 0) };
        try equal(@as(usize, 0), try lapack.luSolve(C, constant(C, &factor, 3, 3, 3), &pivots, matrix(C, &rhs, 3, 1, 3)));
        try near(.{ .re = 0, .im = 0 }, scaled(rhs[0], 1), 0);
        try near(.{ .re = 4, .im = 0 }, scaled(rhs[1], 1), 0);
        try near(.{ .re = 4, .im = 0 }, scaled(rhs[2], 1), 0);
    }
}

test "complex factorizations and solves reject malformed shapes, capacities and operations before mutation" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var a = [_]C{ z(C, 2, 0), z(C, 1, 1), z(C, 1, -1), z(C, 3, 0) };
        var rhs = [_]C{ z(C, 3, 1), z(C, 4, 2) };
        var pivots = [_]lapack.PivotInt{ 0, 1 };
        const before = a;
        const rb = rhs;
        const av = matrix(C, &a, 2, 2, 2);
        var invalid = av;
        invalid.values = a[0..3];
        try std.testing.expectError(error.BufferTooSmall, lapack.choleskyFactorize(C, invalid, .lower));
        try std.testing.expectError(error.BufferTooSmall, lapack.luFactorize(C, invalid, &pivots));
        invalid = av;
        invalid.leading_dimension = 1;
        try std.testing.expectError(error.InvalidLeadingDimension, lapack.choleskyFactorize(C, invalid, .lower));
        try std.testing.expectError(error.InvalidLeadingDimension, lapack.luFactorize(C, invalid, &pivots));
        invalid = av;
        invalid.row_count = -1;
        try std.testing.expectError(error.InvalidLength, lapack.luFactorize(C, invalid, &pivots));
        invalid = av;
        invalid.column_count = 1;
        try std.testing.expectError(error.DimensionMismatch, lapack.choleskyFactorize(C, invalid, .lower));
        try std.testing.expectError(error.BufferTooSmall, lapack.luFactorize(C, av, pivots[0..1]));
        var factor = constant(C, &a, 2, 2, 2);
        const b = matrix(C, &rhs, 2, 1, 2);
        factor.operation = .adjoint;
        try std.testing.expectError(error.InvalidMatrixOperation, lapack.choleskySolve(C, factor, b, .upper));
        try std.testing.expectError(error.InvalidMatrixOperation, lapack.luSolve(C, factor, &pivots, b));
        factor.operation = .normal;
        var wrong_rhs = b;
        wrong_rhs.row_count = 1;
        try std.testing.expectError(error.DimensionMismatch, lapack.choleskySolve(C, factor, wrong_rhs, .lower));
        try std.testing.expectError(error.DimensionMismatch, lapack.luSolve(C, factor, &pivots, wrong_rhs));
        wrong_rhs = b;
        wrong_rhs.values = rhs[0..1];
        try std.testing.expectError(error.BufferTooSmall, lapack.luSolve(C, factor, &pivots, wrong_rhs));
        try std.testing.expectError(error.BufferTooSmall, lapack.choleskySolve(C, factor, wrong_rhs, .lower));
        try same(C, &before, &a);
        try same(C, &rb, &rhs);
        try std.testing.expectEqualSlices(lapack.PivotInt, &.{ 0, 1 }, &pivots);
    }
}

test "complex factor, RHS and mixed pivot aliases include intervening padding" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var a: [24]C = @splat(z(C, 2, 0));
        var rhs: [6]C = @splat(z(C, 3, 1));
        var pivots = [_]lapack.PivotInt{ 0, 1 };
        const before = a;
        const rb = rhs;
        const factor = constant(C, a[0..8], 2, 2, 4);
        const overlapping_rhs = matrix(C, a[2..8], 2, 1, 2);
        try std.testing.expectError(error.AliasingNotAllowed, lapack.choleskySolve(C, factor, overlapping_rhs, .lower));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(C, factor, &pivots, overlapping_rhs));
        const pointer: [*]lapack.PivotInt = @ptrCast(&a);
        const padding_offset = 2 * @sizeOf(C) / @sizeOf(lapack.PivotInt);
        const embedded_pivots = pointer[padding_offset..][0..2];
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luFactorize(C, matrix(C, a[0..8], 2, 2, 4), embedded_pivots));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(C, factor, embedded_pivots, matrix(C, &rhs, 2, 1, 4)));
        const rhs_pointer: [*]lapack.PivotInt = @ptrCast(&rhs);
        try std.testing.expectError(error.AliasingNotAllowed, lapack.luSolve(C, factor, rhs_pointer[0..2], matrix(C, &rhs, 2, 1, 4)));
        try same(C, &before, &a);
        try same(C, &rb, &rhs);
    }
}

test "complex empty and zero-RHS operations remain validated no-ops" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var empty: [0]C = .{};
        var pivots: [0]lapack.PivotInt = .{};
        const zero = matrix(C, &empty, 0, 0, 0);
        try equal(@as(usize, 0), try lapack.choleskyFactorize(C, zero, .upper));
        try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &empty, 7, 0, 0), &pivots));
        try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &empty, 0, 7, 0), &pivots));
        const cf = constant(C, &empty, 0, 0, 0);
        const many = matrix(C, &empty, 0, std.math.maxInt(lapack.PivotInt), 0);
        try equal(@as(usize, 0), try lapack.choleskySolve(C, cf, many, .lower));
        try equal(@as(usize, 0), try lapack.luSolve(C, cf, &pivots, many));
        var bad = zero;
        bad.leading_dimension = -1;
        try std.testing.expectError(error.InvalidLeadingDimension, lapack.choleskyFactorize(C, bad, .lower));
        try std.testing.expectError(error.InvalidLeadingDimension, lapack.luFactorize(C, bad, &pivots));
        var singular = [_]C{z(C, 0, 0)};
        const p = [_]lapack.PivotInt{0};
        const no_rhs = matrix(C, &empty, 1, 0, 0);
        try equal(@as(usize, 0), try lapack.choleskySolve(C, constant(C, &singular, 1, 1, 1), no_rhs, .lower));
        try equal(@as(usize, 0), try lapack.luSolve(C, constant(C, &singular, 1, 1, 1), &p, no_rhs));
        singular[0].im = 1;
        try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskySolve(C, constant(C, &singular, 1, 1, 1), no_rhs, .lower));
        singular[0].im = std.math.inf(real(C));
        try std.testing.expectError(error.NonFiniteInput, lapack.luSolve(C, constant(C, &singular, 1, 1, 1), &p, no_rhs));
    }
}

test "complex factor arithmetic failure keeps failing stores, padding and incomplete pivot suffix explicit" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        const huge = std.math.floatMax(real(C));
        var a = [_]C{ z(C, 1, 0), z(C, 1, 0), nan(C), .{ .re = huge, .im = huge }, .{ .re = -huge, .im = -huge }, nan(C) };
        var pivots = [_]lapack.PivotInt{ -71, -71, -71 };
        const failing = a[4];
        try equal(@as(usize, 1), try lapack.luFactorize(C, matrix(C, &a, 2, 2, 3), &pivots));
        try same(C, &.{failing}, a[4..5]);
        try equal(@as(lapack.PivotInt, 0), pivots[0]);
        try equal(@as(lapack.PivotInt, -71), pivots[1]);
        try equal(@as(lapack.PivotInt, -71), pivots[2]);
        try padding(C, &a, 2, 2, 3);
        var ch = [_]C{ .{ .re = std.math.floatTrueMin(real(C)), .im = 0 }, .{ .re = huge, .im = huge }, nan(C), nan(C), .{ .re = huge, .im = 0 }, nan(C) };
        const off_diagonal = ch[1];
        try equal(@as(usize, 1), try lapack.choleskyFactorize(C, matrix(C, &ch, 2, 2, 3), .lower));
        try same(C, &.{off_diagonal}, ch[1..2]);
        try expect(ch[0].re > 0 and std.math.isFinite(ch[0].re));
        try padding(C, &ch, 2, 2, 3);
    }
}

test "complex substitution range failures preserve factors, later RHS columns and padding" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        var factor = [_]C{.{ .re = std.math.floatTrueMin(real(C)), .im = std.math.floatTrueMin(real(C)) }};
        var pivots: [1]lapack.PivotInt = undefined;
        try equal(@as(usize, 0), try lapack.luFactorize(C, matrix(C, &factor, 1, 1, 1), &pivots));
        var rhs = [_]C{ .{ .re = std.math.floatMax(real(C)), .im = 0 }, nan(C), z(C, 2, 3), nan(C) };
        const before = rhs;
        const fb = factor;
        try equal(@as(usize, 1), try lapack.luSolve(C, constant(C, &factor, 1, 1, 1), &pivots, matrix(C, &rhs, 1, 2, 2)));
        try same(C, &before, &rhs);
        try same(C, &fb, &factor);
        factor[0] = .{ .re = std.math.floatTrueMin(real(C)), .im = 0 };
        try equal(@as(usize, 0), try lapack.choleskyFactorize(C, matrix(C, &factor, 1, 1, 1), .lower));
        const ch = factor;
        try equal(@as(usize, 1), try lapack.choleskySolve(C, constant(C, &factor, 1, 1, 1), matrix(C, &rhs, 1, 2, 2), .lower));
        try same(C, &before, &rhs);
        try same(C, &ch, &factor);
    }
}

test "complex Cholesky accepts signed imaginary zero and rejects every nonzero diagonal imaginary part" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        var a = [_]C{.{ .re = 4, .im = -0.0 }};
        try equal(@as(usize, 0), try lapack.choleskyFactorize(C, matrix(C, &a, 1, 1, 1), triangle));
        const Bits = @Int(.unsigned, @bitSizeOf(real(C)));
        try equal(@as(Bits, 0), @as(Bits, @bitCast(a[0].im)));
        a[0].im = -0.0;
        const before = a;
        var rhs = [_]C{z(C, 8, -4)};
        try equal(@as(usize, 0), try lapack.choleskySolve(C, constant(C, &a, 1, 1, 1), matrix(C, &rhs, 1, 1, 1), triangle));
        try near(.{ .re = 2, .im = -1 }, scaled(rhs[0], 1), 0);
        try same(C, &before, &a);
        a[0].im = std.math.floatTrueMin(real(C));
        const invalid = a;
        try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskyFactorize(C, matrix(C, &a, 1, 1, 1), triangle));
        try same(C, &invalid, &a);
    };
}

test "complex LU retains the first singular index if later arithmetic aborts factorization" {
    inline for (.{ lapack.ComplexF32, lapack.ComplexF64 }) |C| {
        const huge = std.math.floatMax(real(C));
        var a = [_]C{ z(C, 0, 0), z(C, 0, 0), z(C, 0, 0), z(C, 0, 0), z(C, 1, 0), z(C, 1, 0), z(C, 0, 0), .{ .re = huge, .im = huge }, .{ .re = -huge, .im = -huge } };
        var pivots = [_]lapack.PivotInt{ -71, -71, -71 };
        const failed_store = a[8];
        try equal(@as(usize, 1), try lapack.luFactorize(C, matrix(C, &a, 3, 3, 3), &pivots));
        try std.testing.expectEqualSlices(lapack.PivotInt, &.{ 0, 1, -71 }, &pivots);
        try same(C, &.{failed_store}, a[8..9]);
    }
}
