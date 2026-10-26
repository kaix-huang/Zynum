// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const lapack = @import("zynum-lapack");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

fn matrix(comptime T: type, data: []T, n: usize, ld: usize) lapack.Matrix(T) {
    return .{ .values = data, .row_count = @intCast(n), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}
fn constant(comptime T: type, data: []const T, n: usize, ld: usize) lapack.ConstMatrix(T) {
    return .{ .values = data, .row_count = @intCast(n), .column_count = @intCast(n), .leading_dimension = @intCast(ld), .operation = .normal };
}
fn same(comptime T: type, before: []const T, after: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(after));
}
fn near(x: f64, y: f64, tolerance: f64) !void {
    try expect(std.math.isFinite(x) and std.math.isFinite(y));
    try std.testing.expectApproxEqAbs(x, y, tolerance);
}
fn padding(comptime T: type, data: []const T, n: usize, ld: usize) !void {
    for (0..n) |j| for (n..ld) |i| try expect(std.math.isNan(data[i + j * ld]));
    for (ld * n..data.len) |i| try expect(std.math.isNan(data[i]));
}
fn entry(comptime T: type, a: []const T, n: usize, i: usize, j: usize, triangle: lapack.Triangle) f64 {
    const ld = n + 3;
    return if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) a[i + j * ld] else a[j + i * ld];
}

// These checks multiply the original matrix directly; no rotations or diagonal
// updates from the implementation are reused in the numerical oracle.
fn checkEigen(comptime T: type, comptime n: usize, a: []const T, triangle: lapack.Triangle, values: []const T, vectors: []const T, ldv: usize) !void {
    var scale: f64 = 0;
    for (0..n) |j| for (0..n) |i| {
        scale = @max(scale, @abs(entry(T, a, n, i, j, triangle)));
    };
    if (scale == 0) scale = 1;
    var norm2: f64 = 0;
    var residual2: f64 = 0;
    var reconstruction2: f64 = 0;
    var orthogonality2: f64 = 0;
    for (0..n) |j| {
        if (j != 0) try expect(values[j - 1] <= values[j]);
        try expect(std.math.isFinite(values[j]));
        for (0..n) |i| {
            const original = entry(T, a, n, i, j, triangle) / scale;
            norm2 += original * original;
            var av: f64 = 0;
            var reconstructed: f64 = 0;
            var dot: f64 = 0;
            for (0..n) |k| {
                av += (entry(T, a, n, i, k, triangle) / scale) * @as(f64, vectors[k + j * ldv]);
                reconstructed += @as(f64, vectors[i + k * ldv]) * (@as(f64, values[k]) / scale) * @as(f64, vectors[j + k * ldv]);
                dot += @as(f64, vectors[k + i * ldv]) * @as(f64, vectors[k + j * ldv]);
            }
            const r = av - @as(f64, vectors[i + j * ldv]) * (@as(f64, values[j]) / scale);
            const reconstruction = reconstructed - original;
            const orthogonality = dot - @as(f64, if (i == j) 1 else 0);
            residual2 += r * r;
            reconstruction2 += reconstruction * reconstruction;
            orthogonality2 += orthogonality * orthogonality;
        }
    }
    const tol = 64 * @as(f64, std.math.floatEps(T)) * @as(f64, @floatFromInt(@max(n, 1)));
    try near(0, @sqrt(residual2 / @max(norm2, 1)), tol);
    try near(0, @sqrt(reconstruction2 / @max(norm2, 1)), tol);
    try near(0, @sqrt(orthogonality2), tol);
}

fn fixture(comptime T: type, comptime n: usize, triangle: lapack.Triangle, scale: T) !void {
    const lda = n + 3;
    const ldw = n + 1;
    const ldv = n + 2;
    var a: [lda * n + 2]T = @splat(std.math.nan(T));
    var work: [ldw * n + 2]T = @splat(std.math.nan(T));
    var vectors: [ldv * n + 2]T = @splat(std.math.nan(T));
    var values: [n + 2]T = @splat(std.math.nan(T));
    for (0..n) |j| for (0..n) |i| {
        if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) {
            const row = @max(i, j);
            const column = @min(i, j);
            const raw: i32 = @as(i32, @intCast((row * 17 + column * 13 + 5) % 15)) - 7;
            const value: T = @as(T, @floatFromInt(raw)) / 8;
            a[i + j * lda] = value * scale;
        }
    };
    const before = a;
    const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, n, lda), triangle, matrix(T, &work, n, ldw), &values, matrix(T, &vectors, n, ldv), .{});
    try equal(lapack.SymmetricEigenStatus.converged, result.status);
    try expect(result.sweeps <= 64);
    try checkEigen(T, n, &a, triangle, &values, &vectors, ldv);
    try same(T, &before, &a);
    try padding(T, &work, n, ldw);
    try padding(T, &vectors, n, ldv);
    try expect(std.math.isNan(values[n]) and std.math.isNan(values[n + 1]));
    // The complete workspace remains symmetric, including after sorting.
    for (0..n) |j| for (0..n) |i| try equal(work[i + j * ldw], work[j + i * ldw]);
}

test "Symmetric eigen reconstruction, AV-VLambda and orthogonality for both triangles and precisions" {
    inline for (.{ f32, f64 }) |T| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        try fixture(T, 1, triangle, 1);
        try fixture(T, 2, triangle, 1);
        try fixture(T, 5, triangle, 1);
        try fixture(T, 9, triangle, 1);
        try fixture(T, 17, triangle, 1);
    };
}

test "Symmetric eigen power scaling supports huge, tiny and common subnormal inputs" {
    inline for (.{ f32, f64 }) |T| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        try fixture(T, 5, triangle, std.math.floatMax(T) / 16);
        try fixture(T, 5, triangle, std.math.floatMin(T) * 16);
        // Retain enough subnormal precision for a useful relative-error oracle.
        try fixture(T, 5, triangle, std.math.floatMin(T) / 16);
    };
}

fn knownSpectrum(comptime T: type, cluster: bool, triangle: lapack.Triangle) !void {
    const n = 6;
    const ld = n + 3;
    const gap: f64 = if (T == f32) 0.0001 else 1e-10;
    const expected = if (cluster) [n]f64{ -3, 1, 1 + gap, 1 + 2 * gap, 4, 9 } else [n]f64{ -3, -3, 0, 2, 2, 2 };
    var a: [ld * n + 2]T = @splat(std.math.nan(T));
    var work: [ld * n + 2]T = @splat(std.math.nan(T));
    var vectors: [ld * n + 2]T = @splat(std.math.nan(T));
    var values: [n]T = undefined;
    var u: [n]f64 = undefined;
    var u_norm_squared: f64 = 0;
    for (0..n) |i| {
        u[i] = @floatFromInt(i + 1);
        u_norm_squared += u[i] * u[i];
    }
    // A = H*D*H for the independently known orthogonal H = I - 2*u*u^T/(u^T*u).
    for (0..n) |j| for (0..n) |i| {
        if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) {
            var value: f64 = 0;
            for (0..n) |k| {
                const hik = @as(f64, if (i == k) 1 else 0) - 2 * u[i] * u[k] / u_norm_squared;
                const hkj = @as(f64, if (k == j) 1 else 0) - 2 * u[k] * u[j] / u_norm_squared;
                value += hik * expected[k] * hkj;
            }
            a[i + j * ld] = @floatCast(value);
        }
    };
    const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, n, ld), triangle, matrix(T, &work, n, ld), &values, matrix(T, &vectors, n, ld), .{});
    try equal(lapack.SymmetricEigenStatus.converged, result.status);
    for (0..n) |i| try near(expected[i], values[i], 256 * @as(f64, std.math.floatEps(T)));
    try checkEigen(T, n, &a, triangle, &values, &vectors, ld);
}

test "Symmetric eigen repeated and clustered eigenspaces with independent known spectra" {
    inline for (.{ f32, f64 }) |T| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        try knownSpectrum(T, false, triangle);
        try knownSpectrum(T, true, triangle);
    };
}

test "Symmetric eigen negative off-diagonal equal-diagonal rotation and sorted vector pairing" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 0, -2, 99, 0 };
        var work: [4]T = undefined;
        var vectors: [4]T = undefined;
        var values: [2]T = undefined;
        const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, 2, 2), .lower, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{});
        try equal(lapack.SymmetricEigenStatus.converged, result.status);
        try near(-2, values[0], 8 * @as(f64, std.math.floatEps(T)));
        try near(2, values[1], 8 * @as(f64, std.math.floatEps(T)));
        try near(vectors[0], vectors[1], 8 * @as(f64, std.math.floatEps(T)));
        try near(vectors[2], -@as(f64, vectors[3]), 8 * @as(f64, std.math.floatEps(T)));
    }
}

test "Symmetric eigen zero, diagonal and minimum-subnormal matrices converge without sweeps" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ @as(T, 0), @as(T, 1), std.math.floatTrueMin(T) }) |scale| {
            var a = [_]T{ 3 * scale, 0, 0, std.math.nan(T), -scale, 0, std.math.nan(T), std.math.nan(T), 2 * scale };
            var work: [9]T = undefined;
            var vectors: [9]T = undefined;
            var values: [3]T = undefined;
            const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, 3, 3), .lower, matrix(T, &work, 3, 3), &values, matrix(T, &vectors, 3, 3), .{ .max_sweeps = 0 });
            try equal(lapack.SymmetricEigenStatus.converged, result.status);
            try equal(@as(usize, 0), result.sweeps);
            try equal(-scale, values[0]);
            try equal(2 * scale, values[1]);
            try equal(3 * scale, values[2]);
        }
    }
}

test "Symmetric eigen finite input with unrepresentable eigenvalue reports range failure without publishing values" {
    inline for (.{ f32, f64 }) |T| {
        const huge = std.math.floatMax(T);
        var a = [_]T{ huge, huge, std.math.nan(T), std.math.nan(T), huge, std.math.nan(T) };
        var work: [6]T = @splat(std.math.nan(T));
        var vectors: [6]T = @splat(std.math.nan(T));
        var values = [_]T{ 7, 11, 13 };
        const before = values;
        const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, 2, 3), .lower, matrix(T, &work, 2, 3), &values, matrix(T, &vectors, 2, 3), .{});
        try equal(lapack.SymmetricEigenStatus.eigenvalue_out_of_range, result.status);
        try equal(@as(usize, 2), result.failure_index);
        try same(T, &before, &values);
        try padding(T, &work, 2, 3);
        try padding(T, &vectors, 2, 3);
    }
}

test "Symmetric eigen iteration limit preserves eigenvalue outputs and exposes unfinished iterate" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 2, 1, -3, 0, 7, 4, 0, 0, -2 };
        inline for (.{ @as(usize, 0), @as(usize, 1) }) |limit| {
            var work: [12]T = @splat(std.math.nan(T));
            var vectors: [12]T = @splat(std.math.nan(T));
            var values = [_]T{ 13, 17, 19, 23 };
            const before = values;
            const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, 3, 3), .lower, matrix(T, &work, 3, 4), &values, matrix(T, &vectors, 3, 4), .{ .max_sweeps = limit });
            try equal(lapack.SymmetricEigenStatus.iteration_limit, result.status);
            try equal(limit, result.sweeps);
            try expect(result.scaled_off_diagonal_max > 0);
            try same(T, &before, &values);
            try padding(T, &work, 3, 4);
            try padding(T, &vectors, 3, 4);
            if (limit == 0) {
                for (0..3) |j| for (0..3) |i| try equal(@as(T, if (i == j) 1 else 0), vectors[i + j * 4]);
            }
        }
    }
}

test "Symmetric eigen selected triangle validation preserves every output on invalid finite input" {
    inline for (.{ f32, f64 }) |T| inline for (.{ lapack.Triangle.lower, lapack.Triangle.upper }) |triangle| {
        var a = [_]T{ 2, 1, 97, 1, 3, 97 };
        var work: [6]T = @splat(31);
        var vectors: [6]T = @splat(37);
        var values: [3]T = @splat(41);
        const wb = work;
        const vb = vectors;
        const eb = values;
        a[if (triangle == .lower) 1 else 3] = std.math.inf(T);
        try std.testing.expectError(error.NonFiniteInput, lapack.symmetricEigenDecompose(T, constant(T, &a, 2, 3), triangle, matrix(T, &work, 2, 3), &values, matrix(T, &vectors, 2, 3), .{}));
        try same(T, &wb, &work);
        try same(T, &vb, &vectors);
        try same(T, &eb, &values);
    };
}

test "Symmetric eigen shape, operation, capacities and leading dimensions are checked in all modes" {
    var a = [_]f64{ 2, 1, 1, 3 };
    var work: [4]f64 = @splat(31);
    var vectors: [4]f64 = @splat(37);
    var values: [2]f64 = @splat(41);
    const wb = work;
    const vb = vectors;
    const eb = values;
    const input = constant(f64, &a, 2, 2);
    const w = matrix(f64, &work, 2, 2);
    const v = matrix(f64, &vectors, 2, 2);
    var invalid = input;
    invalid.operation = .transposed;
    try std.testing.expectError(error.InvalidMatrixOperation, lapack.symmetricEigenDecompose(f64, invalid, .lower, w, &values, v, .{}));
    invalid = input;
    invalid.row_count = -1;
    try std.testing.expectError(error.InvalidLength, lapack.symmetricEigenDecompose(f64, invalid, .lower, w, &values, v, .{}));
    invalid = input;
    invalid.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, lapack.symmetricEigenDecompose(f64, invalid, .lower, w, &values, v, .{}));
    invalid = input;
    invalid.column_count = 1;
    try std.testing.expectError(error.DimensionMismatch, lapack.symmetricEigenDecompose(f64, invalid, .lower, w, &values, v, .{}));
    invalid = input;
    invalid.values = a[0..3];
    try std.testing.expectError(error.BufferTooSmall, lapack.symmetricEigenDecompose(f64, invalid, .lower, w, &values, v, .{}));
    var invalid_w = w;
    invalid_w.values = work[0..3];
    try std.testing.expectError(error.BufferTooSmall, lapack.symmetricEigenDecompose(f64, input, .lower, invalid_w, &values, v, .{}));
    var invalid_v = v;
    invalid_v.column_count = 1;
    try std.testing.expectError(error.DimensionMismatch, lapack.symmetricEigenDecompose(f64, input, .lower, w, &values, invalid_v, .{}));
    try std.testing.expectError(error.BufferTooSmall, lapack.symmetricEigenDecompose(f64, input, .lower, w, values[0..1], v, .{}));
    try same(f64, &wb, &work);
    try same(f64, &vb, &vectors);
    try same(f64, &eb, &values);
}

test "Symmetric eigen rejects all six pairs of used spans including padding before mutation" {
    inline for (.{ f32, f64 }) |T| {
        var storage: [80]T = @splat(2);
        const before = storage;
        const a = constant(T, storage[0..8], 2, 4);
        const w = matrix(T, storage[16..24], 2, 4);
        const v = matrix(T, storage[32..40], 2, 4);
        const e = storage[48..50];
        // The reused spans begin at an intervening padding entry.
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, matrix(T, storage[2..10], 2, 4), e, v, .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, w, e, matrix(T, storage[2..10], 2, 4), .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, w, storage[2..4], v, .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, w, e, matrix(T, storage[18..26], 2, 4), .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, w, storage[18..20], v, .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.symmetricEigenDecompose(T, a, .lower, w, storage[34..36], v, .{}));
        try same(T, &before, &storage);
    }
}

test "Symmetric eigen empty views are no-ops with zero storage and nonnegative leading dimensions" {
    inline for (.{ f32, f64 }) |T| {
        var storage = [_]T{ 7, 11, 13 };
        const before = storage;
        const empty = constant(T, storage[0..0], 0, 0);
        const output = matrix(T, storage[0..0], 0, 0);
        const result = try lapack.symmetricEigenDecompose(T, empty, .upper, output, &storage, output, .{ .max_sweeps = 0 });
        try equal(lapack.SymmetricEigenStatus.converged, result.status);
        try same(T, &before, &storage);
        var bad = empty;
        bad.leading_dimension = -1;
        try std.testing.expectError(error.InvalidLeadingDimension, lapack.symmetricEigenDecompose(T, bad, .upper, output, &storage, output, .{}));
        bad = empty;
        bad.operation = .adjoint;
        try std.testing.expectError(error.InvalidMatrixOperation, lapack.symmetricEigenDecompose(T, bad, .upper, output, &storage, output, .{}));
    }
}

test "Symmetric eigen global scaling dynamic-range limitation is normwise and explicit" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ std.math.floatMax(T) / 2, 0, std.math.nan(T), std.math.floatTrueMin(T) };
        var work: [4]T = undefined;
        var vectors: [4]T = undefined;
        var values: [2]T = undefined;
        const result = try lapack.symmetricEigenDecompose(T, constant(T, &a, 2, 2), .lower, matrix(T, &work, 2, 2), &values, matrix(T, &vectors, 2, 2), .{});
        try equal(lapack.SymmetricEigenStatus.converged, result.status);
        try equal(@as(T, 0), values[0]);
        try equal(a[0], values[1]);
    }
}
