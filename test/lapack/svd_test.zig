// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const lapack = @import("zynum-lapack");
const expect = std.testing.expect;

fn verify(comptime T: type, original: []const T, a: lapack.Matrix(T), singular: []const T, v: lapack.Matrix(T)) !void {
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    const lda: usize = @intCast(a.leading_dimension);
    const ldv: usize = @intCast(v.leading_dimension);
    var scale: f64 = 0;
    for (original) |x| scale = @max(scale, @abs(@as(f64, @floatCast(x))));
    if (scale == 0) scale = 1;
    const tolerance: f64 = if (T == f32) 4e-5 else 2e-12;
    for (0..n) |j| {
        try expect(std.math.isFinite(singular[j]) and singular[j] >= 0);
        if (j != 0) try expect(singular[j - 1] >= singular[j]);
        for (0..n) |k| {
            var utu: f64 = 0;
            var vtv: f64 = 0;
            for (0..m) |i| utu += @as(f64, @floatCast(a.values[i + j * lda])) * @as(f64, @floatCast(a.values[i + k * lda]));
            for (0..n) |i| vtv += @as(f64, @floatCast(v.values[i + j * ldv])) * @as(f64, @floatCast(v.values[i + k * ldv]));
            try std.testing.expectApproxEqAbs(@as(f64, if (j == k) 1 else 0), utu, tolerance);
            try std.testing.expectApproxEqAbs(@as(f64, if (j == k) 1 else 0), vtv, tolerance);
        }
        for (0..m) |i| {
            var reconstructed: f64 = 0;
            var av: f64 = 0;
            for (0..n) |k| {
                reconstructed += @as(f64, @floatCast(a.values[i + k * lda])) * (@as(f64, @floatCast(singular[k])) / scale) * @as(f64, @floatCast(v.values[j + k * ldv]));
                av += (@as(f64, @floatCast(original[i + k * m])) / scale) * @as(f64, @floatCast(v.values[k + j * ldv]));
            }
            try std.testing.expectApproxEqAbs(@as(f64, @floatCast(original[i + j * m])) / scale, reconstructed, tolerance);
            const us = @as(f64, @floatCast(a.values[i + j * lda])) * (@as(f64, @floatCast(singular[j])) / scale);
            try std.testing.expectApproxEqAbs(us, av, tolerance);
        }
    }
}
fn checkCase(comptime T: type, input: []const T, m: usize, n: usize) !void {
    const allocator = std.testing.allocator;
    const lda = m + 2;
    const ldv = n + 3;
    const data = try allocator.alloc(T, lda * n + 2);
    defer allocator.free(data);
    const vectors = try allocator.alloc(T, ldv * n + 2);
    defer allocator.free(vectors);
    const required = try lapack.svdWorkspaceLength(T, @intCast(m), @intCast(n));
    const workspace = try allocator.alloc(lapack.SvdWorkspaceScalar(T), required + 2);
    defer allocator.free(workspace);
    @memset(workspace, 94);
    const singular = try allocator.alloc(T, n + 2);
    defer allocator.free(singular);
    @memset(data, 91);
    @memset(vectors, 92);
    @memset(singular, 93);
    for (0..n) |j| @memcpy(data[j * lda ..][0..m], input[j * m ..][0..m]);
    const a: lapack.Matrix(T) = .{ .values = data, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(lda) };
    const v: lapack.Matrix(T) = .{ .values = vectors, .row_count = @intCast(n), .column_count = @intCast(n), .leading_dimension = @intCast(ldv) };
    const result = try lapack.svdDecompose(T, a, singular, v, workspace, .{});
    if (result.status != .converged) std.debug.print("SVD failure {s} m={d} n={d} sweeps={d} correlation={e}\n", .{ @tagName(result.status), m, n, result.sweeps, result.max_correlation });
    try expect(result.status == .converged);
    try verify(T, input, a, singular, v);
    for (0..n) |j| {
        for (data[j * lda + m .. (j + 1) * lda]) |x| try std.testing.expectEqual(@as(T, 91), x);
        for (vectors[j * ldv + n .. (j + 1) * ldv]) |x| try std.testing.expectEqual(@as(T, 92), x);
    }
    for (data[lda * n ..]) |x| try std.testing.expectEqual(@as(T, 91), x);
    for (vectors[ldv * n ..]) |x| try std.testing.expectEqual(@as(T, 92), x);
    for (singular[n..]) |x| try std.testing.expectEqual(@as(T, 93), x);
    for (workspace[required..]) |x| try std.testing.expectEqual(@as(lapack.SvdWorkspaceScalar(T), 94), x);
}

test "thin SVD reconstructs deterministic square and tall matrices and both vector spaces" {
    inline for (.{ f32, f64 }) |T| {
        var state: u64 = 773;
        var input: [128]T = undefined;
        for (1..9) |n| for (0..4) |extra| {
            const m = n + extra;
            for (input[0 .. m * n]) |*x| {
                state = state *% 6364136223846793005 +% 1;
                const value: i32 = @intCast((state >> 32) % 31);
                x.* = @as(T, @floatFromInt(value - 15)) / 7;
            }
            try checkCase(T, input[0 .. m * n], m, n);
        };
    }
}
test "thin SVD handles repeated singular values rank deficiency and zero columns" {
    inline for (.{ f32, f64 }) |T| {
        try checkCase(T, &.{ 0, 3, 0, 0, -3, 0, 0, 0, 0, 0, 1, 0 }, 4, 3);
        try checkCase(T, &.{ 1, 2, 3, 2, 4, 6, 3, 6, 9 }, 3, 3);
        try checkCase(T, &.{ 1, 2, 3, 0, 0, 0 }, 3, 2);
        try checkCase(T, &.{ 0, 0, 0, 0, 0, 0 }, 3, 2);
    }
}
test "thin SVD keeps scaled inputs finite without unscaled Gram matrix" {
    inline for (.{ f32, f64 }) |T| {
        const base = [_]T{ 1, 2, -1, 3, 4, 2, 0, 1 };
        var input: [8]T = undefined;
        const small: T = if (T == f32) 0x1p-120 else 0x1p-1000;
        const large: T = if (T == f32) 0x1p120 else 0x1p1000;
        for ([_]T{ small, large }) |scale| {
            for (&input, base) |*x, b| x.* = b * scale;
            try checkCase(T, &input, 4, 2);
        }
        const minimum = std.math.floatTrueMin(T);
        try checkCase(T, &.{ minimum, 0, 0, minimum }, 2, 2);
    }
}
test "thin SVD reports iteration limit and zero sweep orthogonal success" {
    var workspace: [32]lapack.SvdWorkspaceScalar(f64) = @splat(94);
    var a = [_]f64{ 1, 2, 3, 4 };
    var s = [_]f64{ 91, 91 };
    var v: [4]f64 = @splat(92);
    const av = try lapack.matrix(f64, &a, .{ .row_count = 2, .column_count = 2 });
    const vv = try lapack.matrix(f64, &v, .{ .row_count = 2, .column_count = 2 });
    const initial = a;
    const result = try lapack.svdDecompose(f64, av, &s, vv, &workspace, .{ .max_sweeps = 0 });
    try expect(result.status == .no_convergence and result.sweeps == 0);
    try std.testing.expectEqualSlices(f64, &initial, &a);
    try std.testing.expectEqualSlices(f64, &.{ 91, 91 }, &s);
    try std.testing.expectEqualSlices(f64, &.{ 92, 92, 92, 92 }, &v);
    a = .{ 2, 0, 0, -4 };
    const original = a;
    try expect((try lapack.svdDecompose(f64, av, &s, vv, &workspace, .{ .max_sweeps = 0 })).status == .converged);
    try std.testing.expectEqualSlices(f64, &.{ 4, 2 }, &s);
    try verify(f64, &original, av, &s, vv);
}
test "thin SVD rejects bad arguments before any mutation" {
    var workspace: [32]lapack.SvdWorkspaceScalar(f64) = @splat(94);
    var a = [_]f64{ 1, 2, 3, 4 };
    var s = [_]f64{ 91, 91 };
    var v = [_]f64{ 92, 92, 92, 92 };
    const av = try lapack.matrix(f64, &a, .{ .row_count = 2, .column_count = 2 });
    const vv = try lapack.matrix(f64, &v, .{ .row_count = 2, .column_count = 2 });
    for ([_]f64{ 0, -1, 1, std.math.inf(f64), std.math.nan(f64) }) |tolerance| try std.testing.expectError(error.InvalidTolerance, lapack.svdDecompose(f64, av, &s, vv, &workspace, .{ .relative_tolerance = tolerance }));
    try std.testing.expectError(error.BufferTooSmall, lapack.svdDecompose(f64, av, s[0..1], vv, &workspace, .{}));
    try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(f64, av, a[0..2], vv, &workspace, .{}));
    try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(f64, av, &s, av, &workspace, .{}));
    try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(f64, av, v[1..3], vv, &workspace, .{}));
    var bad = av;
    bad.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, lapack.svdDecompose(f64, bad, &s, vv, &workspace, .{}));
    bad = av;
    bad.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, lapack.svdDecompose(f64, bad, &s, vv, &workspace, .{}));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4 }, &a);
    try std.testing.expectEqualSlices(f64, &.{ 91, 91 }, &s);
    try std.testing.expectEqualSlices(f64, &.{ 92, 92, 92, 92 }, &v);
    a[3] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, lapack.svdDecompose(f64, av, &s, vv, &workspace, .{}));
    try expect(std.math.isNan(a[3]));
    try std.testing.expectEqualSlices(f64, &.{ 91, 91 }, &s);
    try std.testing.expectEqualSlices(f64, &.{ 92, 92, 92, 92 }, &v);
}
test "thin SVD empty shapes are checked no-ops" {
    var workspace: [32]lapack.SvdWorkspaceScalar(f64) = @splat(94);
    var empty: [0]f64 = .{};
    const a: lapack.Matrix(f64) = .{ .values = &empty, .row_count = 3, .column_count = 0, .leading_dimension = 0 };
    const v: lapack.Matrix(f64) = .{ .values = &empty, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
    try expect((try lapack.svdDecompose(f64, a, &empty, v, &workspace, .{})).status == .converged);
    var bad = a;
    bad.leading_dimension = -1;
    try std.testing.expectError(error.InvalidLeadingDimension, lapack.svdDecompose(f64, bad, &empty, v, &workspace, .{}));
}
test "thin SVD reports unrepresentable values and preserves extreme relative scales" {
    var workspace: [32]lapack.SvdWorkspaceScalar(f64) = @splat(94);
    var a = [_]f64{ std.math.floatMax(f64), std.math.floatMax(f64) };
    var s: [1]f64 = @splat(91);
    var v: [1]f64 = @splat(92);
    const original = a;
    const av = try lapack.matrix(f64, &a, .{ .row_count = 2, .column_count = 1 });
    const vv = try lapack.matrix(f64, &v, .{ .row_count = 1, .column_count = 1 });
    try expect((try lapack.svdDecompose(f64, av, &s, vv, &workspace, .{})).status == .non_finite_result);
    try std.testing.expectEqualSlices(f64, &original, &a);
    try std.testing.expectEqualSlices(f64, &.{91}, &s);
    try std.testing.expectEqualSlices(f64, &.{92}, &v);
    var extreme = [_]f64{ 0x1p1000, 0, std.math.floatTrueMin(f64), std.math.floatTrueMin(f64) };
    var s2: [2]f64 = undefined;
    var v2: [4]f64 = undefined;
    const e = try lapack.matrix(f64, &extreme, .{ .row_count = 2, .column_count = 2 });
    const w = try lapack.matrix(f64, &v2, .{ .row_count = 2, .column_count = 2 });
    const extreme_original = extreme;
    try expect((try lapack.svdDecompose(f64, e, &s2, w, &workspace, .{})).status == .converged);
    try verify(f64, &extreme_original, e, &s2, w);
    try std.testing.expectEqual(std.math.floatTrueMin(f64), s2[1]);
}

test "thin SVD retains a small singular value after two-sided mixing" {
    inline for (.{ f32, f64 }) |T| {
        var workspace: [32]lapack.SvdWorkspaceScalar(T) = undefined;
        const small: T = if (T == f32) 1e-3 else 1e-10;
        var a = [_]T{ 0.36 + 0.64 * small, 0.48 * (1 - small), 0, 0.48 * (1 - small), 0.64 + 0.36 * small, 0 };
        const original = a;
        var singular: [2]T = undefined;
        var vectors: [4]T = undefined;
        const av = try lapack.matrix(T, &a, .{ .row_count = 3, .column_count = 2 });
        const vv = try lapack.matrix(T, &vectors, .{ .row_count = 2, .column_count = 2 });
        try expect((try lapack.svdDecompose(T, av, &singular, vv, &workspace, .{})).status == .converged);
        try verify(T, &original, av, &singular, vv);
        try std.testing.expectApproxEqRel(@as(T, 1), singular[0], if (T == f32) 1e-6 else 1e-14);
        try std.testing.expectApproxEqRel(small, singular[1], if (T == f32) 1e-4 else 1e-5);
    }
}

comptime {
    _ = @import("zynum-lapack");
}

test "thin SVD widened storage resolves the eight-column rank-one subnormal stagnation regression" {
    inline for (.{ f32, f64 }) |T| {
        var input: [64]T = undefined;
        for (0..8) |j| for (0..8) |i| {
            input[i + j * 8] = @floatFromInt((i + 1) * (j + 1));
        };
        try checkCase(T, &input, 8, 8);
    }
}

test "thin SVD exact workspace query and short workspace fail before any mutation" {
    inline for (.{ f32, f64 }) |T| {
        const W = lapack.SvdWorkspaceScalar(T);
        try std.testing.expectEqual(@as(usize, 12), try lapack.svdWorkspaceLength(T, 3, 2));
        try std.testing.expectEqual(@as(usize, 0), try lapack.svdWorkspaceLength(T, std.math.maxInt(lapack.PivotInt), 0));
        try std.testing.expectError(error.InvalidLength, lapack.svdWorkspaceLength(T, -1, 0));
        try std.testing.expectError(error.DimensionMismatch, lapack.svdWorkspaceLength(T, 2, 3));
        try std.testing.expectError(error.InvalidLength, lapack.svdWorkspaceLength(T, std.math.maxInt(lapack.PivotInt), std.math.maxInt(lapack.PivotInt)));
        var a = [_]T{ 1, 2, 3, 4 };
        var s = [_]T{ 91, 91 };
        var v: [4]T = @splat(92);
        var workspace: [9]W = @splat(94);
        const av = try lapack.matrix(T, &a, .{ .row_count = 2, .column_count = 2 });
        const vv = try lapack.matrix(T, &v, .{ .row_count = 2, .column_count = 2 });
        try std.testing.expectError(error.BufferTooSmall, lapack.svdDecompose(T, av, &s, vv, &workspace, .{}));
        try std.testing.expectEqualSlices(T, &.{ 1, 2, 3, 4 }, &a);
        try std.testing.expectEqualSlices(T, &.{ 91, 91 }, &s);
        for (v) |x| try std.testing.expectEqual(@as(T, 92), x);
        for (workspace) |x| try std.testing.expectEqual(@as(W, 94), x);
    }
}

test "thin SVD mixed-type workspace overlap with input, singular values and right vectors is checked by bytes" {
    inline for (.{ f32, f64 }) |T| {
        const W = lapack.SvdWorkspaceScalar(T);
        var backing: [32]W = @splat(77);
        const before = backing;
        const pointer: [*]T = @ptrCast(&backing);
        const scalar = pointer[0 .. 32 * @sizeOf(W) / @sizeOf(T)];
        var a = [_]T{ 1, 2, 3, 4 };
        var s: [2]T = @splat(91);
        var v: [4]T = @splat(92);
        const av = try lapack.matrix(T, &a, .{ .row_count = 2, .column_count = 2 });
        const vv = try lapack.matrix(T, &v, .{ .row_count = 2, .column_count = 2 });
        const overlap_a: lapack.Matrix(T) = .{ .values = scalar[0..8], .row_count = 2, .column_count = 2, .leading_dimension = 4 };
        // The wide workspace begins inside the first matrix column's padding.
        try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(T, overlap_a, &s, vv, backing[1..11], .{}));
        try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(T, av, scalar[0..2], vv, backing[0..10], .{}));
        const overlap_v = try lapack.matrix(T, scalar[0..4], .{ .row_count = 2, .column_count = 2 });
        try std.testing.expectError(error.AliasingNotAllowed, lapack.svdDecompose(T, av, &s, overlap_v, backing[0..10], .{}));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&before), std.mem.sliceAsBytes(&backing));
        try std.testing.expectEqualSlices(T, &.{ 1, 2, 3, 4 }, &a);
        try std.testing.expectEqualSlices(T, &.{ 91, 91 }, &s);
        for (v) |x| try std.testing.expectEqual(@as(T, 92), x);
    }
}

test "thin SVD output-precision orthogonality failure preserves all public results" {
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 1, 1, 1, -1 };
        const original = a;
        var s: [2]T = @splat(91);
        var v: [4]T = @splat(92);
        var workspace: [10]lapack.SvdWorkspaceScalar(T) = undefined;
        const av = try lapack.matrix(T, &a, .{ .row_count = 2, .column_count = 2 });
        const vv = try lapack.matrix(T, &v, .{ .row_count = 2, .column_count = 2 });
        const result = try lapack.svdDecompose(T, av, &s, vv, &workspace, .{ .relative_tolerance = 1e-30 });
        try std.testing.expectEqual(lapack.SvdStatus.range_limit, result.status);
        try expect(result.output_orthogonality > 1e-30);
        try std.testing.expectEqualSlices(T, &original, &a);
        try std.testing.expectEqualSlices(T, &.{ 91, 91 }, &s);
        for (v) |x| try std.testing.expectEqual(@as(T, 92), x);
    }
}

test "thin SVD coupled minimum-subnormal inputs retain both representable singular values" {
    inline for (.{ f32, f64 }) |T| {
        const minimum = std.math.floatTrueMin(T);
        var a = [_]T{ minimum, 0, minimum, minimum };
        var s: [2]T = undefined;
        var v: [4]T = undefined;
        var workspace: [10]lapack.SvdWorkspaceScalar(T) = undefined;
        const av = try lapack.matrix(T, &a, .{ .row_count = 2, .column_count = 2 });
        const vv = try lapack.matrix(T, &v, .{ .row_count = 2, .column_count = 2 });
        const result = try lapack.svdDecompose(T, av, &s, vv, &workspace, .{});
        try std.testing.expectEqual(lapack.SvdStatus.converged, result.status);
        // Analytic singular values are phi*minimum and minimum/phi.
        try std.testing.expectEqual(2 * minimum, s[0]);
        try std.testing.expectEqual(minimum, s[1]);
        try expect(result.output_orthogonality <= 32 * @as(f64, std.math.floatEps(T)));
        for (0..2) |j| for (0..2) |i| {
            var reconstructed: f64 = 0;
            for (0..2) |k| reconstructed += @as(f64, a[i + 2 * k]) * (@as(f64, s[k]) / @as(f64, minimum)) * @as(f64, v[j + 2 * k]);
            const expected: f64 = if (i == 0 or j == 1) 1 else 0;
            // Quantization of these subnormal singular values is unavoidable.
            try std.testing.expectApproxEqAbs(expected, reconstructed, 0.5);
        };
    }
}

test "shared SVD core accepts native width empty geometry and rejects workspace overflow" {
    const core = @import("zynum-core").internal.numerics.Svd;
    const wide = @as(usize, std.math.maxInt(i32)) + 1;
    inline for (.{ f32, f64 }) |T| {
        try std.testing.expectEqual(@as(usize, 0), try core.workspaceLength(T, wide, 0));
        const result = try core.decompose(T, .{ .values = &.{}, .row_count = wide, .column_count = 0, .leading_dimension = 0 }, &.{}, .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 }, &.{}, .{});
        try std.testing.expectEqual(core.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.sweeps);
        try std.testing.expectError(error.InvalidLength, core.workspaceLength(T, std.math.maxInt(usize), 2));
        try std.testing.expectError(error.DimensionMismatch, core.workspaceLength(T, 1, wide));
        // A nonempty wide operand is rejected by its actual capacity, not an ABI-width cast.
        try std.testing.expectError(error.BufferTooSmall, core.decompose(T, .{ .values = &.{}, .row_count = wide, .column_count = 1, .leading_dimension = wide }, &.{}, .{ .values = &.{}, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &.{}, .{}));
    }
}

test "minimum norm reconstruction resolves ABPLaS rank one history and wide nullspace" {
    const core = @import("zynum-core").internal.numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const tolerance: T = if (T == f32) 2e-6 else 2e-14;
        const q: T = 1 / @sqrt(@as(T, 5));
        const factors = [_]T{ q, 2 * q, -2 * q, q };
        const s = [_]T{ 5, 0 };
        const b = [_]T{ 3, 6, -6, -12 };
        var x = [_]T{ 99, 99, 77, 99, 99, 77 };
        var scratch: [4]core.WorkspaceScalar(T) = undefined;
        const result = try core.solve(T, .{ .values = &factors, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &factors, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .{ .values = &x, .row_count = 2, .column_count = 2, .leading_dimension = 3 }, -1, &scratch);
        try std.testing.expectEqual(@as(usize, 1), result.rank);
        try std.testing.expectEqual(.success, result.status);
        for ([_]T{ 0.6, 1.2, 77, -1.2, -2.4, 77 }, x) |want, actual| try std.testing.expectApproxEqAbs(want, actual, tolerance);
        // A=[1,2]: the unique minimum-norm solution is orthogonal to [-2,1].
        var wide_x: [2]T = undefined;
        const wide = try core.solve(T, .{ .values = &.{1}, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &.{@sqrt(@as(T, 5))}, .{ .values = factors[0..2], .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &.{3}, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = &wide_x, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, -1, scratch[0..2]);
        try std.testing.expectEqual(@as(usize, 1), wide.rank);
        try std.testing.expectApproxEqAbs(@as(T, 3), wide_x[0] + 2 * wide_x[1], tolerance);
        try std.testing.expectApproxEqAbs(@as(T, 0), -2 * wide_x[0] + wide_x[1], tolerance);
    }
}

test "minimum norm cutoff equality and overflow preserve transactional outputs" {
    const core = @import("zynum-core").internal.numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const identity = [_]T{ 1, 0, 0, 1 };
        const factor: @import("zynum-core").internal.numerics.Dense.ConstMatrix(T) = .{ .values = &identity, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        var x = [_]T{ 77, 88 };
        var work: [2]core.WorkspaceScalar(T) = undefined;
        const rhs: @import("zynum-core").internal.numerics.Dense.ConstMatrix(T) = .{ .values = &[_]T{ 2, 1 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const output: @import("zynum-core").internal.numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const cut = try core.solve(T, factor, &.{ 2, 1 }, factor, rhs, output, 0.5, &work);
        try std.testing.expectEqual(@as(usize, 1), cut.rank);
        try std.testing.expectEqualSlices(T, &.{ 1, 0 }, &x);
        x = .{ 77, 88 };
        const failed = try core.solve(T, factor, &.{ 1, std.math.floatTrueMin(T) }, factor, rhs, output, 0, &work);
        try std.testing.expectEqual(.non_finite_result, failed.status);
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
        try std.testing.expectError(error.InvalidSingularValues, core.solve(T, factor, &.{ 1, 2 }, factor, rhs, output, -1, &work));
        try std.testing.expectError(error.InvalidTolerance, core.solve(T, factor, &.{ 2, 1 }, factor, rhs, output, std.math.nan(T), &work));
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
    }
}

test "rectangular SVD connects wide matrices to minimum norm reconstruction" {
    const numerics = @import("zynum-core").internal.numerics;
    const svd = numerics.Svd;
    inline for (.{ f32, f64 }) |T| {
        const tolerance: T = if (T == f32) 5e-5 else 2e-12;
        const original = [_]T{ 1, 0, 77, 0, 1, 77, 1, 1, 77 };
        var a = original;
        var v: [8]T = @splat(88);
        var singular: [2]T = @splat(99);
        var scratch: [12]svd.WorkspaceScalar(T) = undefined;
        var transpose: [11]T = @splat(66);
        const empty = try svd.decomposeRectangular(T, .{ .values = &.{}, .row_count = 0, .column_count = std.math.maxInt(usize), .leading_dimension = 0 }, &.{}, .{ .values = &.{}, .row_count = std.math.maxInt(usize), .column_count = 0, .leading_dimension = 0 }, &.{}, &.{}, .{});
        try std.testing.expectEqual(svd.Status.converged, empty.status);
        try std.testing.expectEqual(@as(usize, 10), try svd.rectangularScratchLength(T, 2, 3));
        const matrix: numerics.Dense.Matrix(T) = .{ .values = &a, .row_count = 2, .column_count = 3, .leading_dimension = 3 };
        const right: numerics.Dense.Matrix(T) = .{ .values = &v, .row_count = 3, .column_count = 2, .leading_dimension = 4 };
        const failed = try svd.decomposeRectangular(T, matrix, &singular, right, &scratch, &transpose, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(svd.Status.no_convergence, failed.status);
        try std.testing.expectEqualSlices(T, &original, &a);
        try std.testing.expectEqualSlices(T, &.{ 99, 99 }, &singular);
        for (v) |entry| try std.testing.expectEqual(@as(T, 88), entry);
        const result = try svd.decomposeRectangular(T, matrix, &singular, right, &scratch, &transpose, .{});
        try std.testing.expectEqual(svd.Status.converged, result.status);
        try std.testing.expectApproxEqAbs(@sqrt(@as(T, 3)), singular[0], tolerance);
        try std.testing.expectApproxEqAbs(@as(T, 1), singular[1], tolerance);
        for (0..2) |i| for (0..3) |j| {
            var reconstructed: T = 0;
            for (0..2) |k| reconstructed += a[i + k * 3] * singular[k] * v[j + k * 4];
            try std.testing.expectApproxEqAbs(original[i + j * 3], reconstructed, tolerance);
        };
        try std.testing.expectEqual(@as(T, 77), a[2]);
        try std.testing.expectEqualSlices(T, original[5..], a[5..]);
        try std.testing.expectEqual(@as(T, 88), v[3]);
        try std.testing.expectEqual(@as(T, 88), v[7]);
        try std.testing.expectEqual(@as(T, 66), transpose[10]);
        var x: [3]T = undefined;
        var solve_scratch: [3]svd.WorkspaceScalar(T) = undefined;
        const solved = try numerics.MinimumNorm.solve(T, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 3 }, &singular, right.asConst(), .{ .values = &.{ 3, 0 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &x, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, -1, &solve_scratch);
        try std.testing.expectEqual(@as(usize, 2), solved.rank);
        for ([_]T{ 2, -1, 1 }, x) |want, value| try std.testing.expectApproxEqAbs(want, value, tolerance);
        try std.testing.expectApproxEqAbs(@as(T, 0), -x[0] - x[1] + x[2], tolerance);
    }
}

test "rank deficient ABPLaS Gram matrix factors and solves across scales" {
    const numerics = @import("zynum-core").internal.numerics;
    inline for (.{ f32, f64 }) |T| {
        const tolerance: T = if (T == f32) 1e-5 else 2e-12;
        for ([_]T{ 1, 0.125, 1024 }) |scale| {
            var a = [_]T{ scale, 2 * scale, 2 * scale, 4 * scale };
            var v: [4]T = undefined;
            var s: [2]T = undefined;
            var work: [10]numerics.Svd.WorkspaceScalar(T) = undefined;
            const factor = try numerics.Svd.decomposeRectangular(T, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &work, &.{}, .{});
            try std.testing.expectEqual(numerics.Svd.Status.converged, factor.status);
            try std.testing.expectApproxEqAbs(@as(T, 5), s[0] / scale, tolerance);
            var x: [2]T = undefined;
            var solve_work: [2]numerics.Svd.WorkspaceScalar(T) = undefined;
            const b = [_]T{ 3 * scale, 6 * scale };
            const solved = try numerics.MinimumNorm.solve(T, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .{ .values = &b, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, -1, &solve_work);
            try std.testing.expectEqual(@as(usize, 1), solved.rank);
            try std.testing.expectEqual(.success, solved.status);
            try std.testing.expectApproxEqAbs(@as(T, 0.6), x[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(T, 1.2), x[1], tolerance);
        }
    }
}

test "rank least squares driver covers tall square wide and exact scratch capacities" {
    const core = @import("zynum-core").internal.numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        const Case = struct { m: usize, n: usize, a: []const T, b: []const T, want: []const T, rank: usize };
        const cases = [_]Case{
            .{ .m = 2, .n = 2, .a = &.{ 1, 2, 2, 4 }, .b = &.{ 3, 6 }, .want = &.{ 0.6, 1.2 }, .rank = 1 },
            .{ .m = 2, .n = 3, .a = &.{ 1, 0, 0, 1, 1, 1 }, .b = &.{ 3, 0 }, .want = &.{ 2, -1, 1 }, .rank = 2 },
            .{ .m = 3, .n = 2, .a = &.{ 1, 0, 1, 0, 1, 1 }, .b = &.{ 1, 2, 4 }, .want = &.{ 4.0 / 3.0, 7.0 / 3.0 }, .rank = 2 },
        };
        for (cases) |case| {
            const p = try core.plan(T, case.m, case.n, 1);
            const scalar = try std.testing.allocator.alloc(T, p.scalar_elements + 1);
            defer std.testing.allocator.free(scalar);
            const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), p.wide_elements + 1);
            defer std.testing.allocator.free(wide);
            @memset(scalar, 99);
            @memset(wide, 88);
            var x: [3]T = @splat(77);
            var s: [2]T = @splat(66);
            const a: @import("zynum-core").internal.numerics.Dense.ConstMatrix(T) = .{ .values = case.a, .row_count = case.m, .column_count = case.n, .leading_dimension = case.m };
            const b: @import("zynum-core").internal.numerics.Dense.ConstMatrix(T) = .{ .values = case.b, .row_count = case.m, .column_count = 1, .leading_dimension = case.m };
            const out: @import("zynum-core").internal.numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = case.n, .column_count = 1, .leading_dimension = case.n };
            try std.testing.expectError(error.BufferTooSmall, core.solve(T, a, b, out, &s, -1, scalar[0 .. p.scalar_elements - 1], wide, .{}));
            for (x) |v| try std.testing.expectEqual(@as(T, 77), v);
            for (s) |v| try std.testing.expectEqual(@as(T, 66), v);
            const solved = try core.solve(T, a, b, out, &s, -1, scalar, wide, .{});
            try std.testing.expectEqual(.converged, solved.status);
            try std.testing.expectEqual(case.rank, solved.rank);
            for (case.want, x[0..case.n]) |want, got| try std.testing.expectApproxEqAbs(want, got, if (T == f32) 5e-5 else 2e-12);
            try std.testing.expectEqual(@as(T, 99), scalar[p.scalar_elements]);
            try std.testing.expectEqual(@as(core.WorkspaceScalar(T), 88), wide[p.wide_elements]);
        }
    }
}

test "rank least squares zero shapes zero rank and multiple right hand sides" {
    const core = @import("zynum-core").internal.numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        const Shape = struct { m: usize, n: usize, rhs: usize };
        for ([_]Shape{ .{ .m = 0, .n = 3, .rhs = 2 }, .{ .m = 3, .n = 0, .rhs = 2 }, .{ .m = 3, .n = 2, .rhs = 0 }, .{ .m = 2, .n = 3, .rhs = 2 } }) |shape| {
            const allocator = std.testing.allocator;
            const a = try allocator.alloc(T, shape.m * shape.n);
            defer allocator.free(a);
            @memset(a, 0);
            const b = try allocator.alloc(T, shape.m * shape.rhs);
            defer allocator.free(b);
            @memset(b, 1);
            const x = try allocator.alloc(T, shape.n * shape.rhs + 1);
            defer allocator.free(x);
            @memset(x, 77);
            const s = try allocator.alloc(T, @min(shape.m, shape.n) + 1);
            defer allocator.free(s);
            @memset(s, 66);
            const p = try core.plan(T, shape.m, shape.n, shape.rhs);
            const scalar = try allocator.alloc(T, p.scalar_elements);
            defer allocator.free(scalar);
            const wide = try allocator.alloc(core.WorkspaceScalar(T), p.wide_elements);
            defer allocator.free(wide);
            const result = try core.solve(T, .{ .values = a, .row_count = shape.m, .column_count = shape.n, .leading_dimension = shape.m }, .{ .values = b, .row_count = shape.m, .column_count = shape.rhs, .leading_dimension = shape.m }, .{ .values = x, .row_count = shape.n, .column_count = shape.rhs, .leading_dimension = shape.n }, s, -1, scalar, wide, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(@as(usize, 0), result.rank);
            for (x[0 .. x.len - 1]) |value| try std.testing.expectEqual(@as(T, 0), value);
            for (s[0 .. s.len - 1]) |value| try std.testing.expectEqual(@as(T, 0), value);
            try std.testing.expectEqual(@as(T, 77), x[x.len - 1]);
            try std.testing.expectEqual(@as(T, 66), s[s.len - 1]);
            for (a) |value| try std.testing.expectEqual(@as(T, 0), value);
            for (b) |value| try std.testing.expectEqual(@as(T, 1), value);
        }
    }
}

test "rank least squares rejects overlap and preserves output on decomposition failure" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{ 1, 0, 1, 1 };
        const b = [_]T{ 3, 2, 6, 4 };
        var x: [4]T = @splat(77);
        var s: [2]T = @splat(66);
        const p = try core.plan(T, 2, 2, 2);
        const scalar = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(scalar);
        const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), p.wide_elements);
        defer std.testing.allocator.free(wide);
        const matrix: numerics.Dense.ConstMatrix(T) = .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const output: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        try std.testing.expectError(error.AliasingNotAllowed, core.solve(T, matrix, rhs, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, -1, scalar, wide, .{}));
        try std.testing.expectEqualSlices(T, &.{ 1, 0, 1, 1 }, &a);
        const failed = try core.solve(T, matrix, rhs, output, &s, -1, scalar, wide, .{ .max_steps = 0 });
        try std.testing.expectEqual(.no_convergence, failed.status);
        try std.testing.expectEqual(@as(usize, 1), failed.unconverged);
        try std.testing.expectEqual(@as(usize, 0), failed.steps);
        for (x) |value| try std.testing.expectEqual(@as(T, 77), value);
        for (s) |value| try std.testing.expectEqual(@as(T, 66), value);
        const solved = try core.solve(T, matrix, rhs, output, &s, -1, scalar, wide, .{});
        try std.testing.expectEqual(.converged, solved.status);
        try std.testing.expectEqual(@as(usize, 2), solved.rank);
        for ([_]T{ 1, 2, 2, 4 }, x) |want, got| try std.testing.expectApproxEqAbs(want, got, if (T == f32) 1e-5 else 2e-12);
        try std.testing.expectEqualSlices(T, &.{ 1, 0, 1, 1 }, &a);
    }
}

test "rank least squares residual coordinates preserve squared error and padding" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        const a = [_]T{ 1, 0, 1, 0, 1, 1 };
        const b = [_]T{ 1, 2, 4, 2, 4, 8 };
        var x: [6]T = @splat(77);
        var s: [2]T = undefined;
        var residual: [4]T = @splat(99);
        const p = try core.planWithResidual(T, 3, 2, 2, true);
        const scalar = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(scalar);
        const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), p.wide_elements);
        defer std.testing.allocator.free(wide);
        const matrix: numerics.Dense.ConstMatrix(T) = .{ .values = &a, .row_count = 3, .column_count = 2, .leading_dimension = 3 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &b, .row_count = 3, .column_count = 2, .leading_dimension = 3 };
        const out: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 2, .leading_dimension = 3 };
        const tail: numerics.Dense.Matrix(T) = .{ .values = &residual, .row_count = 1, .column_count = 2, .leading_dimension = 2 };
        const result = try core.solveWithResidual(T, matrix, rhs, out, &s, tail, -1, scalar, wide, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.rank);
        const tol: T = if (T == f32) 2e-5 else 2e-12;
        for (0..2) |j| {
            var squared: T = 0;
            for (0..3) |i| {
                var error_value = b[i + j * 3];
                for (0..2) |k| error_value -= a[i + k * 3] * x[k + j * 3];
                squared += error_value * error_value;
            }
            try std.testing.expectApproxEqAbs(squared, residual[j * 2] * residual[j * 2], tol);
            try std.testing.expectApproxEqAbs(@as(T, if (j == 0) 1.0 / 3.0 else 4.0 / 3.0), squared, tol);
            try std.testing.expectEqual(@as(T, 99), residual[j * 2 + 1]);
            try std.testing.expectEqual(@as(T, 77), x[j * 3 + 2]);
        }
        @memset(&residual, 99);
        @memset(&x, 77);
        // Truncation removes a direction: the standard full-rank tail promise no longer applies.
        const truncated = try core.solveWithResidual(T, matrix, rhs, out, &s, tail, 0.9, scalar, wide, .{});
        try std.testing.expectEqual(@as(usize, 1), truncated.rank);
        for (residual) |value| try std.testing.expectEqual(@as(T, 99), value);
        // Writable outputs must not overlap, including the residual destination.
        try std.testing.expectError(error.AliasingNotAllowed, core.solveWithResidual(T, matrix, rhs, out, &s, .{ .values = &x, .row_count = 1, .column_count = 2, .leading_dimension = 2 }, -1, scalar, wide, .{}));
    }
}

test "rank least squares residual overflow preserves every output and empty columns copy RHS" {
    const core = @import("zynum-core").internal.numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        var x = [_]T{77};
        var s = [_]T{66};
        var tail = [_]T{99};
        var scalar: [5]T = undefined;
        var wide: [8]core.WorkspaceScalar(T) = undefined;
        const result = try core.solveWithResidual(T, .{ .values = &.{ 1, 1 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &.{ std.math.floatMax(T), -std.math.floatMax(T) }, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &s, .{ .values = &tail, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, -1, &scalar, &wide, .{});
        try std.testing.expectEqual(.non_finite_result, result.status);
        try std.testing.expectEqual(@as(T, 77), x[0]);
        try std.testing.expectEqual(@as(T, 66), s[0]);
        try std.testing.expectEqual(@as(T, 99), tail[0]);
        var empty_tail: [2]T = undefined;
        var empty_scratch: [2]T = undefined;
        const empty = try core.solveWithResidual(T, .{ .values = &.{}, .row_count = 2, .column_count = 0, .leading_dimension = 0 }, .{ .values = &.{ 3, -4 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &.{}, .row_count = 0, .column_count = 1, .leading_dimension = 0 }, &.{}, .{ .values = &empty_tail, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, -1, &empty_scratch, &.{}, .{});
        try std.testing.expectEqual(.converged, empty.status);
        try std.testing.expectEqualSlices(T, &.{ 3, -4 }, &empty_tail);
    }
}

test "minimum norm widened singular values preserve finite solutions and explicit IEEE overflow" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const W = core.WorkspaceScalar(T);
        const unit: numerics.Dense.ConstMatrix(T) = .{ .values = &.{1}, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        var x = [_]T{ 77, 88, 99 };
        const output: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 1, .column_count = 2, .leading_dimension = 2 };
        var work = [_]W{ 0, 0, 66 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ std.math.floatMax(T), -std.math.floatMax(T) }, .row_count = 1, .column_count = 2, .leading_dimension = 1 };
        // S cannot be represented in T, but B/S is exactly +/-0.5.
        const huge = [_]W{2 * @as(W, std.math.floatMax(T))};
        const finite = try core.solveWithSingularType(T, W, unit, &huge, unit, rhs, output, 0, &work, .finite, .finite);
        try std.testing.expectEqual(.success, finite.status);
        try std.testing.expectEqual(@as(usize, 1), finite.rank);
        try std.testing.expectEqualSlices(T, &.{ 0.5, 88, -0.5 }, &x);
        const tiny = [_]W{std.math.floatTrueMin(T)};
        x = .{ 77, 88, 99 };
        const rejected = try core.solveWithSingularType(T, W, unit, &tiny, unit, rhs, output, 0, &work, .finite, .finite);
        try std.testing.expectEqual(.non_finite_result, rejected.status);
        try std.testing.expectEqualSlices(T, &.{ 77, 88, 99 }, &x);
        const ieee = try core.solveWithSingularType(T, W, unit, &tiny, unit, rhs, output, 0, &work, .ieee, .finite);
        try std.testing.expectEqual(.success, ieee.status);
        try std.testing.expectEqualSlices(T, &.{ std.math.inf(T), 88, -std.math.inf(T) }, &x);
        try std.testing.expectEqual(@as(W, 66), work[2]);
        try std.testing.expectError(error.InvalidSingularValues, core.solveWithSingularType(T, W, unit, &.{std.math.inf(W)}, unit, rhs, output, 0, &work, .ieee, .finite));
    }
}

test "rank least squares IEEE output policy preserves widened singular reconstruction" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        const huge = std.math.floatMax(T);
        const a: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ huge, huge, huge, huge }, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const b: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ huge, huge }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        var x = [_]T{ 77, 88 };
        var s = [_]T{ 99, 99 };
        const out: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const p = try core.plan(T, 2, 2, 1);
        const scalar = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(scalar);
        const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), p.wide_elements);
        defer std.testing.allocator.free(wide);
        const checked = try core.solve(T, a, b, out, &s, -1, scalar, wide, .{});
        try std.testing.expectEqual(.non_finite_result, checked.status);
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
        try std.testing.expectEqualSlices(T, &.{ 99, 99 }, &s);
        const ieee = try core.solve(T, a, b, out, &s, -1, scalar, wide, .{ .output_policy = .ieee });
        try std.testing.expectEqual(.converged, ieee.status);
        try std.testing.expectEqual(@as(usize, 1), ieee.rank);
        try std.testing.expectEqual(std.math.inf(T), s[0]);
        for (x) |value| try std.testing.expectApproxEqAbs(@as(T, 0.5), value, 8 * std.math.floatEps(T));
        const one_a: numerics.Dense.ConstMatrix(T) = .{ .values = &.{std.math.floatTrueMin(T)}, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const one_b: numerics.Dense.ConstMatrix(T) = .{ .values = &.{-huge}, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const one_x: numerics.Dense.Matrix(T) = .{ .values = x[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const overflow = try core.solve(T, one_a, one_b, one_x, s[0..1], -1, scalar, wide, .{ .output_policy = .ieee });
        try std.testing.expectEqual(.converged, overflow.status);
        try std.testing.expectEqual(@as(usize, 1), overflow.rank);
        try std.testing.expectEqual(-std.math.inf(T), x[0]);
        try std.testing.expectEqual(std.math.floatTrueMin(T), s[0]);
    }
    // Hadamard columns have known singular values and exact coefficients.
    // Repeated singular values previously exhausted QR at huge finite scales.
    inline for (.{ f32, f64 }) |T| {
        const powers = if (T == f32) [_]i32{ -135, -110, 0, 110 } else [_]i32{ -1065, -1000, 0, 500, 1000 };
        for (powers) |exponent| for ([_]usize{ 8, 16 }) |n| {
            const scale = std.math.scalbn(@as(T, 1), exponent);
            var a: [256]T = @splat(99);
            var rhs: [32]T = @splat(88);
            var x: [32]T = @splat(77);
            var singular: [16]T = @splat(66);
            for (0..n) |j| for (0..16) |i| {
                a[i + j * 16] = (if (@popCount(i & j) % 2 == 0) scale else -scale) / 4;
            };
            for (0..2) |j| for (0..16) |i| {
                var sum: T = 0;
                for (0..n) |k| sum += (if (@popCount(i & k) % 2 == 0) @as(T, 1) else -1) * @as(T, @floatFromInt(k + 1)) / 4;
                rhs[i + j * 16] = scale * sum / @as(T, @floatFromInt(j + 1));
            };
            const original = a;
            const original_rhs = rhs;
            const plan = try core.plan(T, 16, n, 2);
            const scalar = try std.testing.allocator.alloc(T, plan.scalar_elements + 1);
            defer std.testing.allocator.free(scalar);
            const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), plan.wide_elements + 1);
            defer std.testing.allocator.free(wide);
            @memset(scalar, 55);
            @memset(wide, 44);
            const result = try core.solve(T, .{ .values = &a, .row_count = 16, .column_count = n, .leading_dimension = 16 }, .{ .values = &rhs, .row_count = 16, .column_count = 2, .leading_dimension = 16 }, .{ .values = &x, .row_count = n, .column_count = 2, .leading_dimension = n }, &singular, 1e-4, scalar, wide, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(n, result.rank);
            for (0..n) |k| try std.testing.expectApproxEqAbs(@as(T, 1), singular[k] / scale, 64 * std.math.floatEps(T));
            for (0..2) |j| for (0..n) |k| {
                const expected = @as(T, @floatFromInt(k + 1)) / @as(T, @floatFromInt(j + 1));
                try std.testing.expectApproxEqAbs(expected, x[k + j * n], 256 * std.math.floatEps(T) * expected);
            };
            try std.testing.expectEqualSlices(T, &original, &a);
            try std.testing.expectEqualSlices(T, &original_rhs, &rhs);
            for (x[2 * n ..]) |value| try std.testing.expectEqual(@as(T, 77), value);
            for (singular[n..]) |value| try std.testing.expectEqual(@as(T, 66), value);
            try std.testing.expectEqual(@as(T, 55), scalar[plan.scalar_elements]);
            try std.testing.expectEqual(@as(core.WorkspaceScalar(T), 44), wide[plan.wide_elements]);
        };
    }
}

test "minimum norm RHS policy keeps checked rejection transactional" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const unit: numerics.Dense.ConstMatrix(T) = .{ .values = &.{1}, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &.{std.math.nan(T)}, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        var x = [_]T{77};
        var scratch: [1]core.WorkspaceScalar(T) = undefined;
        const out: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectError(error.NonFiniteInput, core.solve(T, unit, &.{1}, unit, rhs, out, -1, &scratch));
        try std.testing.expectError(error.NonFiniteInput, core.solveWithSingularType(T, T, unit, &.{1}, unit, rhs, out, -1, &scratch, .ieee, .finite));
        try std.testing.expectEqual(@as(T, 77), x[0]);
        const rejected = try core.solveWithSingularType(T, T, unit, &.{1}, unit, rhs, out, -1, &scratch, .finite, .ieee);
        try std.testing.expectEqual(.non_finite_result, rejected.status);
        try std.testing.expectEqual(@as(T, 77), x[0]);
        const propagated = try core.solveWithSingularType(T, T, unit, &.{1}, unit, rhs, out, -1, &scratch, .ieee, .ieee);
        try std.testing.expectEqual(.success, propagated.status);
        try std.testing.expect(std.math.isNan(x[0]));
    }
}

test "minimum norm unordered cutoff is explicit and keeps zero directions" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const factor: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, 0, 0, 1 }, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, 1 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        var x = [_]T{ 77, 88 };
        var scratch: [2]core.WorkspaceScalar(T) = undefined;
        const out: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const nan = std.math.nan(T);
        try std.testing.expectEqual(@as(usize, 2), core.rankWithPolicy(T, T, &.{ 1, 0 }, nan, .unordered));
        try std.testing.expectEqual(@as(usize, 1), core.rankWithPolicy(T, T, &.{ 1, 0 }, 0, .unordered));
        try std.testing.expectError(error.InvalidTolerance, core.solveWithPolicies(T, T, factor, &.{ 1, 0.5 }, factor, rhs, out, nan, &scratch, .{}));
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
        const result = try core.solveWithPolicies(T, T, factor, &.{ 1, 0.5 }, factor, rhs, out, nan, &scratch, .{ .cutoff = .unordered });
        try std.testing.expectEqual(.success, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.rank);
        try std.testing.expectEqualSlices(T, &.{ 1, 2 }, &x);
        const zero = try core.solveWithPolicies(T, T, factor, &.{ 1, 0 }, factor, rhs, out, nan, &scratch, .{ .cutoff = .unordered, .output = .ieee });
        try std.testing.expectEqual(.success, zero.status);
        try std.testing.expectEqual(@as(usize, 2), zero.rank);
        try std.testing.expect(!std.math.isFinite(x[0]));
        try std.testing.expect(!std.math.isFinite(x[1]));
    }
}

test "bidiagonal SVD explicit IEEE mode propagates singleton and bounds coupled NaN" {
    const numerics = @import("zynum-core").internal.numerics;
    const svd = numerics.BidiagonalSvd;
    inline for (.{ f32, f64 }) |T| {
        var a = [_]T{std.math.nan(T)};
        var s = [_]T{77};
        var v = [_]T{88};
        var work: [12]svd.WorkspaceScalar(T) = undefined;
        const am: numerics.Dense.Matrix(T) = .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const vm: numerics.Dense.Matrix(T) = .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectError(error.NonFiniteInput, svd.decompose(T, am, &s, vm, &work, .{}));
        try std.testing.expectEqual(@as(T, 77), s[0]);
        const singleton = try svd.decompose(T, am, &s, vm, &work, .{ .numerical_policy = .ieee });
        try std.testing.expectEqual(.converged, singleton.status);
        try std.testing.expect(std.math.isNan(s[0]));
        var coupled = [_]T{ 1, 0, std.math.nan(T), 2 };
        var right = [_]T{ 88, 88, 88, 88 };
        var singular = [_]T{ 77, 77 };
        const result = try svd.decompose(T, .{ .values = &coupled, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &singular, .{ .values = &right, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &work, .{ .max_steps = 3, .numerical_policy = .ieee });
        try std.testing.expectEqual(.no_convergence, result.status);
        try std.testing.expectEqual(@as(usize, 3), result.steps);
        try std.testing.expectEqual(@as(usize, 1), result.unconverged);
        try std.testing.expectEqualSlices(T, &.{ 77, 77 }, &singular);
        try std.testing.expectEqualSlices(T, &.{ 88, 88, 88, 88 }, &right);
        try std.testing.expect(std.math.isNan(coupled[2]));
        try std.testing.expectEqual(@as(T, 1), coupled[0]);
    }
}

test "minimum norm IEEE factors select directions individually and preserve checked default" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.MinimumNorm;
    inline for (.{ f32, f64 }) |T| {
        const unit: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, 0, 0, 1 }, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const rhs: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, 1 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        var x = [_]T{ 77, 88 };
        var scratch: [2]core.WorkspaceScalar(T) = undefined;
        const out: numerics.Dense.Matrix(T) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const s = [_]T{ 0, std.math.nan(T) };
        try std.testing.expectError(error.InvalidSingularValues, core.solve(T, unit, &s, unit, rhs, out, 0, &scratch));
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
        const result = try core.solveWithPolicies(T, T, unit, &s, unit, rhs, out, 0, &scratch, .{ .factors = .ieee, .cutoff = .unordered, .output = .ieee });
        try std.testing.expectEqual(.success, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.rank);
        // Only the second (NaN) direction survives, not the first rank entries.
        try std.testing.expect(std.math.isNan(x[0]));
        try std.testing.expect(std.math.isNan(x[1]));
    }
}

test "rank least squares explicit IEEE driver carries residual and real nonconvergence" {
    const numerics = @import("zynum-core").internal.numerics;
    const core = numerics.RankLeastSquares;
    inline for (.{ f32, f64 }) |T| {
        const p = try core.planWithResidual(T, 2, 2, 1, true);
        const scalar = try std.testing.allocator.alloc(T, p.scalar_elements + 4);
        defer std.testing.allocator.free(scalar);
        const wide = try std.testing.allocator.alloc(core.WorkspaceScalar(T), p.wide_elements);
        defer std.testing.allocator.free(wide);
        var x = [_]T{ 77, 88 };
        var s = [_]T{ 66, 66 };
        var residual = [_]T{99};
        const opts: core.Options = .{ .numerical_policy = .ieee, .output_policy = .ieee, .cutoff_policy = .unordered, .max_steps = 3 };
        const a: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, std.math.nan(T) }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const b: numerics.Dense.ConstMatrix(T) = .{ .values = &.{ 1, 2 }, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const out: numerics.Dense.Matrix(T) = .{ .values = x[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const tail: numerics.Dense.Matrix(T) = .{ .values = &residual, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectError(error.NonFiniteInput, core.solveWithResidual(T, a, b, out, &s, tail, 0, scalar, wide, .{}));
        const result = try core.solveWithResidual(T, a, b, out, &s, tail, 0, scalar, wide, opts);
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.rank);
        try std.testing.expect(std.math.isNan(x[0]));
        try std.testing.expect(std.math.isNan(s[0]));
        try std.testing.expect(std.math.isNan(residual[0]));
        x = .{ 77, 88 };
        s = .{ 66, 66 };
        const coupled = try core.solve(T, .{ .values = &.{ 1, 0, std.math.nan(T), 2 }, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, b, .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, &s, 0, scalar, wide, opts);
        try std.testing.expectEqual(.no_convergence, coupled.status);
        try std.testing.expectEqual(@as(usize, 1), coupled.unconverged);
        try std.testing.expectEqual(@as(usize, 3), coupled.steps);
        try std.testing.expectEqualSlices(T, &.{ 77, 88 }, &x);
        try std.testing.expectEqualSlices(T, &.{ 66, 66 }, &s);
    }
}
