// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core").internal.numerics;
const cx = Core.ComplexMinimumNorm;
const least = Core.ComplexRankLeastSquares;
const C32 = @import("zynum-core").lapack.ComplexF32;
const C64 = @import("zynum-core").lapack.ComplexF64;
fn z(comptime C: type, r: f64, i: f64) C {
    return .{ .re = @floatCast(r), .im = @floatCast(i) };
}
fn near(comptime C: type, expected: C, actual: C) !void {
    const tolerance: cx.Real(C) = if (C == C32) 8e-5 else 3e-12;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}
// Descriptor literals are inferred at each call; keep tests independent of new API aliases.

test "complex minimum norm uses Hermitian projection with padded multiple RHS and untouched suffixes" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        const Z = cx.WorkspaceScalar(C);
        const u = [_]C{ z(C, 0, 1), z(C, 0, 0), z(C, 91, 92), z(C, 0, 0), z(C, 1, 0), z(C, 93, 94) };
        const v = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 0, 0), z(C, 95, 96), z(C, 0, 0), z(C, 1, 0), z(C, 0, 0), z(C, 97, 98) };
        const b = [_]C{ z(C, 0, 8), z(C, 6, 0), z(C, 99, 100), z(C, 4, 0), z(C, 0, -2), z(C, 101, 102) };
        const s = [_]R{ 4, 2, std.math.nan(R) };
        var x: [8]C = @splat(z(C, 77, 78));
        var work: [7]Z = @splat(.{ .re = 83, .im = 84 });
        const result = try cx.solve(C, .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 3 }, &s, .{ .values = &v, .row_count = 3, .column_count = 2, .leading_dimension = 4 }, .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 3 }, .{ .values = &x, .row_count = 3, .column_count = 2, .leading_dimension = 4 }, -1, &work);
        try std.testing.expectEqual(.success, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.rank);
        inline for (.{ z(C, 2, 0), z(C, 3, 0), z(C, 0, 0), z(C, 77, 78), z(C, 0, -1), z(C, 0, -1), z(C, 0, 0), z(C, 77, 78) }, 0..) |expected, i| try near(C, expected, x[i]);
        try std.testing.expectEqual(@as(@TypeOf(work[6].re), 83), work[6].re);
        try std.testing.expectEqual(@as(@TypeOf(work[6].im), 84), work[6].im);
    }
}
test "complex minimum norm cutoff is strict at equality and negative tolerance uses real epsilon" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        const u = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 0, 0), z(C, 1, 0) };
        const b = [_]C{ z(C, 8, 4), z(C, 6, -2) };
        const s = [_]R{ 4, 2 };
        var x: [2]C = undefined;
        var work: [2]cx.WorkspaceScalar(C) = undefined;
        const factor: cx.ConstMatrix(C) = .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &b, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const output: cx.Matrix(C) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        try std.testing.expectEqual(@as(usize, 1), (try cx.solve(C, factor, &s, factor, rhs, output, 0.5, &work)).rank);
        try near(C, z(C, 2, 1), x[0]);
        try near(C, z(C, 0, 0), x[1]);
        try std.testing.expectEqual(@as(usize, 2), (try cx.solve(C, factor, &s, factor, rhs, output, -1, &work)).rank);
        try near(C, z(C, 3, -1), x[1]);
    }
}
test "complex minimum norm retains widened singular values and rejects unrepresentable output transactionally" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        const W = @TypeOf(@as(cx.WorkspaceScalar(C), undefined).re);
        const unit = [_]C{z(C, 1, 0)};
        const b = [_]C{.{ .re = std.math.floatMax(R), .im = 0 }};
        var x = [_]C{z(C, 17, 19)};
        var work: [1]cx.WorkspaceScalar(C) = undefined;
        const f: cx.ConstMatrix(C) = .{ .values = &unit, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const output: cx.Matrix(C) = .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const large = [_]W{2 * @as(W, std.math.floatMax(R))};
        try std.testing.expectEqual(.success, (try cx.solveWithSingularType(C, W, f, &large, f, rhs, output, -1, &work)).status);
        try near(C, z(C, 0.5, 0), x[0]);
        x[0] = z(C, 17, 19);
        const tiny = [_]R{std.math.floatTrueMin(R)};
        try std.testing.expectEqual(.non_finite_result, (try cx.solve(C, f, &tiny, f, rhs, output, -1, &work)).status);
        try near(C, z(C, 17, 19), x[0]);
    }
}
test "complex minimum norm validates singular order finite data capacity and mixed workspace aliases" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        const W = @TypeOf(@as(cx.WorkspaceScalar(C), undefined).re);
        var u = [_]C{z(C, 1, 0)};
        const b = [_]C{z(C, 1, 1)};
        var s = [_]R{1};
        var x = [_]C{z(C, 13, 15)};
        var work = [_]cx.WorkspaceScalar(C){.{ .re = 1, .im = 0 }};
        const f: cx.ConstMatrix(C) = .{ .values = &u, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const output: cx.Matrix(C) = .{ .values = &x, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectError(error.BufferTooSmall, cx.solve(C, f, &s, f, rhs, output, -1, &.{}));
        s[0] = -1;
        try std.testing.expectError(error.InvalidSingularValues, cx.solve(C, f, &s, f, rhs, output, -1, &work));
        s[0] = 1;
        try std.testing.expectError(error.InvalidTolerance, cx.solve(C, f, &s, f, rhs, output, std.math.nan(R), &work));
        u[0].im = std.math.inf(R);
        try std.testing.expectError(error.NonFiniteInput, cx.solve(C, f, &s, f, rhs, output, -1, &work));
        u[0] = z(C, 1, 0);
        const alias = @as([*]W, @ptrCast(&work))[0..1];
        try std.testing.expectError(error.AliasingNotAllowed, cx.solveWithSingularType(C, W, f, alias, f, rhs, output, -1, &work));
        try near(C, z(C, 13, 15), x[0]);
    }
}
test "complex minimum norm empty native-width RHS requires no storage and plans overflow safely" {
    inline for (.{ C32, C64 }) |C| {
        const wide = @as(usize, std.math.maxInt(i32)) + 17;
        const f: cx.ConstMatrix(C) = .{ .values = &[_]C{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &[_]C{}, .row_count = 0, .column_count = wide, .leading_dimension = std.math.maxInt(usize) };
        var none: [0]C = .{};
        const output: cx.Matrix(C) = .{ .values = &none, .row_count = 0, .column_count = wide, .leading_dimension = 0 };
        const result = try cx.solve(C, f, &.{}, f, rhs, output, -1, &.{});
        try std.testing.expectEqual(.success, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.rank);
        try std.testing.expectError(error.SizeOverflow, cx.workspaceLength(C, std.math.maxInt(usize), 2));
    }
}

test "complex rank least squares tall full rank returns solution singular values and residual energy" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        const a = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 1, 0), z(C, 31, 32), z(C, 0, 0), z(C, 0, 1), z(C, 1, 0), z(C, 33, 34) };
        const b = [_]C{ z(C, 1, 2), z(C, 3, -1), z(C, 2, 1), z(C, 35, 36), z(C, 2, -1), z(C, -3, 1), z(C, 3, 2), z(C, 37, 38) };
        var x: [6]C = @splat(z(C, 41, 42));
        var s = [_]R{ 43, 43, 43 };
        var residual: [4]C = @splat(z(C, 45, 46));
        var scalar: [80]C = @splat(z(C, 47, 48));
        var work: [128]least.WorkspaceScalar(C) = @splat(.{ .re = 49, .im = 50 });
        const p = try least.planWithResidual(C, 3, 2, 2, true);
        const result = try least.solveWithResidual(C, .{ .values = &a, .row_count = 3, .column_count = 2, .leading_dimension = 4 }, .{ .values = &b, .row_count = 3, .column_count = 2, .leading_dimension = 4 }, .{ .values = &x, .row_count = 2, .column_count = 2, .leading_dimension = 3 }, &s, .{ .values = &residual, .row_count = 1, .column_count = 2, .leading_dimension = 2 }, -1, &scalar, &work, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.rank);
        try near(C, z(C, 5.0 / 3.0, 8.0 / 3.0), x[0]);
        try near(C, z(C, -1.0 / 3.0, -7.0 / 3.0), x[1]);
        try near(C, z(C, 2, -1), x[3]);
        try near(C, z(C, 1, 3), x[4]);
        try near(C, z(C, 41, 42), x[2]);
        try near(C, z(C, 41, 42), x[5]);
        const tol: R = if (C == C32) 3e-4 else 2e-11;
        try std.testing.expectApproxEqAbs(@sqrt(@as(R, 3)), s[0], tol);
        try std.testing.expectApproxEqAbs(@as(R, 1), s[1], tol);
        try std.testing.expectEqual(@as(R, 43), s[2]);
        for (0..2) |j| {
            var energy: R = 0;
            for (0..3) |i| {
                var sum = C{ .re = 0, .im = 0 };
                for (0..2) |q| {
                    const av = a[i + q * 4];
                    const xv = x[q + j * 3];
                    sum.re += av.re * xv.re - av.im * xv.im;
                    sum.im += av.re * xv.im + av.im * xv.re;
                }
                const dr = b[i + j * 4].re - sum.re;
                const di = b[i + j * 4].im - sum.im;
                energy += dr * dr + di * di;
            }
            const r = residual[j * 2];
            try std.testing.expectApproxEqAbs(energy, r.re * r.re + r.im * r.im, tol);
        }
        try near(C, z(C, 45, 46), residual[1]);
        try near(C, z(C, 45, 46), residual[3]);
        try near(C, z(C, 47, 48), scalar[p.scalar_elements]);
        try std.testing.expectEqual(@as(@TypeOf(work[0].re), 49), work[p.wide_elements].re);
    }
}
test "complex rank least squares wide system is minimum norm and cutoff equality discards all directions" {
    inline for (.{ C32, C64 }) |C| {
        const a = [_]C{ z(C, 1, 0), z(C, 0, 1) };
        const b = [_]C{z(C, 2, 4)};
        var x: [2]C = undefined;
        var s: [1]cx.Real(C) = undefined;
        var scalar: [32]C = undefined;
        var work: [64]least.WorkspaceScalar(C) = undefined;
        const input: cx.ConstMatrix(C) = .{ .values = &a, .row_count = 1, .column_count = 2, .leading_dimension = 1 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const output: cx.Matrix(C) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const result = try least.solve(C, input, rhs, output, &s, -1, &scalar, &work, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.rank);
        try near(C, z(C, 1, 2), x[0]);
        try near(C, z(C, 2, -1), x[1]);
        try near(C, z(C, 0, 0), .{ .re = -x[0].im + x[1].re, .im = x[0].re + x[1].im });
        const cut = try least.solve(C, input, rhs, output, &s, 1, &scalar, &work, .{});
        try std.testing.expectEqual(.converged, cut.status);
        try std.testing.expectEqual(@as(usize, 0), cut.rank);
        try near(C, z(C, 0, 0), x[0]);
        try near(C, z(C, 0, 0), x[1]);
    }
}
test "complex rank least squares rank deficiency leaves residual coordinates untouched" {
    inline for (.{ C32, C64 }) |C| {
        const a = [_]C{ z(C, 1, 0), z(C, 0, 1), z(C, 0, 0), z(C, 0, 2), z(C, -2, 0), z(C, 0, 0) };
        const b = [_]C{ z(C, 1, 0), z(C, 0, 1), z(C, 1, 0) };
        var x: [2]C = undefined;
        var s: [2]cx.Real(C) = undefined;
        var residual = [_]C{z(C, 61, 62)};
        var scalar: [64]C = undefined;
        var work: [128]least.WorkspaceScalar(C) = undefined;
        const result = try least.solveWithResidual(C, .{ .values = &a, .row_count = 3, .column_count = 2, .leading_dimension = 3 }, .{ .values = &b, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, &s, .{ .values = &residual, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, -1, &scalar, &work, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.rank);
        try near(C, z(C, 0.2, 0), x[0]);
        try near(C, z(C, 0, -0.4), x[1]);
        try near(C, z(C, 61, 62), residual[0]);
    }
}
test "complex rank least squares no convergence invalid input and mixed output aliases are transactional" {
    inline for (.{ C32, C64 }) |C| {
        const R = cx.Real(C);
        var a = [_]C{ z(C, 1, 0), z(C, 0, 0), z(C, 1, 0), z(C, 0, 0), z(C, 0, 1), z(C, 1, 0) };
        const b = [_]C{ z(C, 1, 2), z(C, 3, -1), z(C, 2, 1) };
        var x: [2]C = @splat(z(C, 71, 72));
        var s: [2]R = @splat(73);
        var residual = [_]C{z(C, 75, 76)};
        var scalar: [64]C = undefined;
        var work: [128]least.WorkspaceScalar(C) = undefined;
        const input: cx.ConstMatrix(C) = .{ .values = &a, .row_count = 3, .column_count = 2, .leading_dimension = 3 };
        const rhs: cx.ConstMatrix(C) = .{ .values = &b, .row_count = 3, .column_count = 1, .leading_dimension = 3 };
        const output: cx.Matrix(C) = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const res: cx.Matrix(C) = .{ .values = &residual, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const result = try least.solveWithResidual(C, input, rhs, output, &s, res, -1, &scalar, &work, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(.no_convergence, result.status);
        try near(C, z(C, 71, 72), x[0]);
        try std.testing.expectEqual(@as(R, 73), s[0]);
        try near(C, z(C, 75, 76), residual[0]);
        try std.testing.expectError(error.BufferTooSmall, least.solve(C, input, rhs, output, &s, -1, &.{}, &work, .{}));
        a[0].re = std.math.inf(R);
        try std.testing.expectError(error.NonFiniteInput, least.solve(C, input, rhs, output, &s, -1, &scalar, &work, .{}));
        a[0] = z(C, 1, 0);
        const aliased = @as([*]R, @ptrCast(&x))[0..2];
        try std.testing.expectError(error.AliasingNotAllowed, least.solve(C, input, rhs, output, aliased, -1, &scalar, &work, .{}));
        try near(C, z(C, 71, 72), x[0]);
    }
}
test "complex rank least squares empty native width shapes and zero columns preserve finite residual identity" {
    inline for (.{ C32, C64 }) |C| {
        const wide = @as(usize, std.math.maxInt(i32)) + 17;
        var empty: [0]C = .{};
        const a: cx.ConstMatrix(C) = .{ .values = &empty, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
        const b: cx.ConstMatrix(C) = .{ .values = &empty, .row_count = 0, .column_count = wide, .leading_dimension = 0 };
        const p = try least.planWithResidual(C, 0, 0, wide, true);
        try std.testing.expectEqual(@as(usize, 0), p.scalar_elements);
        try std.testing.expectEqual(@as(usize, 0), p.wide_elements);
        const result = try least.solve(C, a, b, .{ .values = &empty, .row_count = 0, .column_count = wide, .leading_dimension = 0 }, &.{}, -1, &.{}, &.{}, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.rank);
        const rhs = [_]C{ z(C, 2, -3), z(C, 4, 5) };
        var out = [_]C{ z(C, 81, 82), z(C, 83, 84) };
        var scratch: [2]C = undefined;
        const zero = try least.solveWithResidual(C, .{ .values = &empty, .row_count = 2, .column_count = 0, .leading_dimension = 0 }, .{ .values = &rhs, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .values = &empty, .row_count = 0, .column_count = 1, .leading_dimension = 0 }, &.{}, .{ .values = &out, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, -1, &scratch, &.{}, .{});
        try std.testing.expectEqual(.converged, zero.status);
        try near(C, rhs[0], out[0]);
        try near(C, rhs[1], out[1]);
        try std.testing.expectError(error.SizeOverflow, least.plan(C, std.math.maxInt(usize), 2, 1));
    }
}
