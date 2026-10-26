// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Internal numerical policy tests. No ABI, adapter, Context or vendor oracle.
const std = @import("std");
const lu = @import("lu.zig");
const dense = @import("../../numerics/dense.zig");
const pivots = @import("../../numerics/pivot.zig");
const types = @import("../../numerics/complex.zig").scalar;

const C = struct { re: f64, im: f64 };
fn plus(a: C, b: C) C {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn times(a: C, b: C) C {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn number(comptime T: type, re: f64, im: f64) T {
    return if (comptime types.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn reference(comptime T: type, value: T) C {
    return if (comptime types.isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn near(comptime T: type, expected: C, actual: T) !void {
    const got = reference(T, actual);
    const tolerance: f64 = if (types.Real(T) == f32) 0.0002 else 0.000000000002;
    try std.testing.expectApproxEqAbs(expected.re, got.re, tolerance * (1 + @abs(expected.re)));
    try std.testing.expectApproxEqAbs(expected.im, got.im, tolerance * (1 + @abs(expected.im)));
}
fn bytes(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}
fn matrix(comptime T: type, values: []T, rows: usize, columns: usize, stride: usize) dense.Matrix(T) {
    return .{ .values = values, .row_count = rows, .column_count = columns, .leading_dimension = stride };
}
fn solve(comptime T: type, comptime I: type, factor: dense.ConstMatrix(T), ipiv: []const I, rhs: dense.Matrix(T), operation: lu.Operation) void {
    lu.solveWithContract(T, factor, pivots.Reader(I, .one){ .values = ipiv }, rhs, operation, .getrs_ieee_v1);
}

test "LU GETRF policy reconstructs independent rectangular real complex matrices and preserves all padding" {
    @setEvalBranchQuota(20000);
    const Shape = struct { rows: usize, columns: usize };
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            for ([_]Shape{ .{ .rows = 5, .columns = 3 }, .{ .rows = 3, .columns = 5 }, .{ .rows = 3, .columns = 3 } }) |shape| {
                const m = shape.rows;
                const n = shape.columns;
                const ld: usize = 7;
                const steps: usize = @min(m, n);
                var values: [40]T = @splat(number(T, 97, -31));
                var original: [25]C = undefined;
                for (0..m) |r| for (0..n) |c| {
                    const re: f64 = if (r == 0 and c == 0) 0 else @as(f64, @floatFromInt((r * 7 + c * 3 + 1) % 11)) - 5;
                    const im: f64 = @as(f64, @floatFromInt((r * 2 + c * 5) % 7)) / 4 - 0.5;
                    values[r + c * ld] = number(T, re, im);
                    original[r * n + c] = reference(T, values[r + c * ld]);
                };
                const before = values;
                var encoded: [7]I = @splat(79);
                const info = lu.factorizeWithContract(T, matrix(T, &values, m, n, ld), pivots.Writer(I, .one){ .values = encoded[0..steps] }, .getrf_ieee_v1);
                if (info != 0) {
                    try std.testing.expect(info <= steps);
                    const zero = reference(T, values[info - 1 + (info - 1) * ld]);
                    try std.testing.expect(zero.re == 0 and zero.im == 0);
                }
                // Independently apply forward swaps to A. No codec reader or
                // factorization is used to construct the expected product.
                for (0..steps) |k| {
                    try std.testing.expect(encoded[k] >= 1);
                    const selected: usize = @intCast(encoded[k] - 1);
                    try std.testing.expect(selected >= k and selected < m);
                    for (0..n) |c| std.mem.swap(C, &original[k * n + c], &original[selected * n + c]);
                }
                for (0..m) |r| for (0..n) |c| {
                    var result: C = .{ .re = 0, .im = 0 };
                    for (0..steps) |k| {
                        const lower: C = if (r == k) .{ .re = 1, .im = 0 } else if (r > k) reference(T, values[r + k * ld]) else .{ .re = 0, .im = 0 };
                        const upper: C = if (k <= c) reference(T, values[k + c * ld]) else .{ .re = 0, .im = 0 };
                        result = plus(result, times(lower, upper));
                    }
                    try near(T, original[r * n + c], number(T, result.re, result.im));
                };
                for (0..values.len) |at| if (at / ld >= n or at % ld >= m) try bytes(T, before[at..][0..1], values[at..][0..1]);
                for (encoded[steps..]) |value| try std.testing.expectEqual(@as(I, 79), value);
            }
        }
    }
}

fn knownFactor(comptime T: type, r: usize, c: usize) C {
    // Independently selected non-unit U and unit-L multipliers, not GETRF output.
    const rows = [_]C{
        .{ .re = 2, .im = 0.5 },     .{ .re = 1, .im = -0.5 },     .{ .re = -1, .im = 1 },
        .{ .re = 0.5, .im = 0.25 },  .{ .re = 3, .im = -1 },       .{ .re = 2, .im = 0.5 },
        .{ .re = -0.25, .im = 0.5 }, .{ .re = 0.75, .im = -0.25 }, .{ .re = 4, .im = 2 },
    };
    const v = rows[r * 3 + c];
    return .{ .re = v.re, .im = if (types.isComplex(T)) v.im else 0 };
}
fn knownX(comptime T: type, row: usize, column: usize) C {
    const re = [_]f64{ 1, -2, 3, -1, 4, 2 };
    const im = [_]f64{ 2, 1, -1, 3, -2, 1 };
    return .{ .re = re[column * 3 + row], .im = if (types.isComplex(T)) im[column * 3 + row] else 0 };
}

test "LU GETRS policies solve known factors with noncommuting swaps and independent dense N T H right hand sides" {
    @setEvalBranchQuota(20000);
    comptime {
        if (lu.SolveResult(.getrs_ieee_v1) != void or lu.SolveResult(.checked_v1) != usize)
            @compileError("LU numeric policies must not discard checked information");
    }
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        inline for (.{ i32, i64 }) |I| {
            var factor: [18]T = @splat(number(T, 89, -23));
            var a: [9]C = undefined;
            for (0..3) |r| for (0..3) |c| {
                const f = knownFactor(T, r, c);
                factor[r + c * 5] = number(T, f.re, f.im);
                var value: C = .{ .re = 0, .im = 0 };
                for (0..3) |k| {
                    const lower: C = if (r == k) .{ .re = 1, .im = 0 } else if (r > k) knownFactor(T, r, k) else .{ .re = 0, .im = 0 };
                    const upper: C = if (k <= c) knownFactor(T, k, c) else .{ .re = 0, .im = 0 };
                    value = plus(value, times(lower, upper));
                }
                a[r * 3 + c] = value;
            };
            // Reverse (0,2),(1,2),(2,2) to construct original A=S^-1*L*U.
            for (0..3) |step| {
                const k = 2 - step;
                for (0..3) |c| std.mem.swap(C, &a[k * 3 + c], &a[2 * 3 + c]);
            }
            const encoded = [_]I{ 3, 3, 3, 79 };
            const factor_before = factor;
            for ([_]lu.Operation{ .normal, .transposed, .adjoint }) |operation| {
                var rhs: [12]T = @splat(number(T, 83, -19));
                for (0..2) |column| for (0..3) |r| {
                    var b: C = .{ .re = 0, .im = 0 };
                    for (0..3) |k| {
                        var coefficient = a[if (operation == .normal) r * 3 + k else k * 3 + r];
                        if (operation == .adjoint) coefficient.im = -coefficient.im;
                        b = plus(b, times(coefficient, knownX(T, k, column)));
                    }
                    rhs[r + column * 5] = number(T, b.re, b.im);
                };
                const rhs_before = rhs;
                solve(T, I, matrix(T, &factor, 3, 3, 5).asConst(), encoded[0..3], matrix(T, &rhs, 3, 2, 5), operation);
                for (0..2) |column| for (0..3) |r| try near(T, knownX(T, r, column), rhs[r + column * 5]);
                for (0..rhs.len) |at| if (at / 5 >= 2 or at % 5 >= 3) try bytes(T, rhs_before[at..][0..1], rhs[at..][0..1]);
                try bytes(T, &factor_before, &factor);
                try std.testing.expectEqualSlices(I, &.{ 3, 3, 3, 79 }, &encoded);
            }
        }
    }
}

test "LU GETRF IEEE continues every trailing column after overflow while checked retains its original partial state" {
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const huge = std.math.floatMax(types.Real(T));
        const initial = [_]T{ number(T, 1, 0), number(T, 1, 0), number(T, 71, 17), number(T, huge, 0), number(T, -huge, 0), number(T, 73, 19), number(T, 2, 0), number(T, 5, 0), number(T, 79, 23), number(T, 4, 0), number(T, 9, 0), number(T, 83, 29) };
        var checked = initial;
        var checked_pivots = [_]i64{ 89, 97, 101 };
        try std.testing.expectEqual(@as(usize, 1), lu.factorize(T, matrix(T, &checked, 2, 4, 3), pivots.Writer(i64, .one){ .values = checked_pivots[0..2] }));
        try bytes(T, &initial, &checked);
        try std.testing.expectEqualSlices(i64, &.{ 1, 97, 101 }, &checked_pivots);
        var continued = initial;
        var continued_pivots = [_]i64{ 89, 97, 101 };
        try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &continued, 2, 4, 3), pivots.Writer(i64, .one){ .values = continued_pivots[0..2] }, .getrf_ieee_v1));
        const exceptional = reference(T, continued[4]);
        try std.testing.expect(std.math.isInf(exceptional.re) and exceptional.re < 0);
        try near(T, .{ .re = 3, .im = 0 }, continued[7]);
        try near(T, .{ .re = 5, .im = 0 }, continued[10]);
        for ([_]usize{ 2, 5, 8, 11 }) |at| try bytes(T, initial[at..][0..1], continued[at..][0..1]);
        try std.testing.expectEqualSlices(i64, &.{ 1, 2, 101 }, &continued_pivots);
    }
}

test "LU GETRF IEEE completes vector update tails rather than ignoring a finite-update failure" {
    inline for (.{ f32, f64 }) |T| {
        const huge = std.math.floatMax(T);
        var values: [24]T = @splat(73);
        for (0..10) |r| {
            values[r] = 1;
            values[12 + r] = if (r == 0) huge else if (r == 1) -huge else 1;
        }
        const before = values;
        var encoded = [_]i32{ 79, 83, 89 };
        try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &values, 10, 2, 12), pivots.Writer(i32, .one){ .values = encoded[0..2] }, .getrf_ieee_v1));
        try std.testing.expect(std.math.isInf(values[13]) and values[13] < 0);
        // Correct full updates produce (-huge)/(-Inf)=+0. Leaving the lanes
        // after the first overflow untouched would instead produce 1/(-Inf)=-0.
        const positive_zero = [_]T{0};
        for (14..22) |at| try bytes(T, &positive_zero, values[at..][0..1]);
        for ([_]usize{ 10, 11, 22, 23 }) |at| try bytes(T, before[at..][0..1], values[at..][0..1]);
        try std.testing.expectEqualSlices(i32, &.{ 1, 2, 89 }, &encoded);
    }
}

test "LU GETRF IEEE preserves first singular info but completes pivots after later overflow and nonfinite selected pivots" {
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const R = types.Real(T);
        const huge = std.math.floatMax(R);
        var values = [_]T{ number(T, 0, 0), number(T, 0, 0), number(T, 0, 0), number(T, 0, 0), number(T, 1, 0), number(T, 1, 0), number(T, 0, 0), number(T, huge, 0), number(T, -huge, 0) };
        var encoded = [_]i64{ 79, 83, 89, 97 };
        try std.testing.expectEqual(@as(usize, 1), lu.factorizeWithContract(T, matrix(T, &values, 3, 3, 3), pivots.Writer(i64, .one){ .values = encoded[0..3] }, .getrf_ieee_v1));
        try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 97 }, &encoded);
        try std.testing.expect(std.math.isInf(reference(T, values[8]).re));
        for ([_]T{ number(T, std.math.inf(R), 0), number(T, std.math.nan(R), 0) }) |value| {
            var one = [_]T{value};
            const before = one;
            var p = [_]i32{79};
            try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &one, 1, 1, 1), pivots.Writer(i32, .one){ .values = &p }, .getrf_ieee_v1));
            try std.testing.expectEqual(@as(i32, 1), p[0]);
            try bytes(T, &before, &one);
        }
        var tied = [_]T{ number(T, 2, 1), number(T, -2, -1) };
        var first = [_]i32{79};
        try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &tied, 2, 1, 2), pivots.Writer(i32, .one){ .values = &first }, .getrf_ieee_v1));
        try std.testing.expectEqual(@as(i32, 1), first[0]);
        try near(T, .{ .re = -1, .im = 0 }, tied[1]);
    }
}

test "LU GETRS IEEE normal zero skips and transpose dot evaluation have distinct nonfinite semantics" {
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const R = types.Real(T);
        var factor = [_]T{ number(T, 0, 0), number(T, std.math.nan(R), std.math.nan(R)), number(T, std.math.nan(R), std.math.nan(R)), number(T, 0, 0) };
        const before = factor;
        const encoded = [_]i32{ 1, 2 };
        const zeros = [_]T{ number(T, 0, 0), number(T, -0.0, -0.0) };
        var rhs = zeros;
        solve(T, i32, matrix(T, &factor, 2, 2, 2).asConst(), &encoded, matrix(T, &rhs, 2, 1, 2), .normal);
        try bytes(T, &zeros, &rhs);
        for ([_]lu.Operation{ .transposed, .adjoint }) |operation| {
            rhs = zeros;
            solve(T, i32, matrix(T, &factor, 2, 2, 2).asConst(), &encoded, matrix(T, &rhs, 2, 1, 2), operation);
            for (rhs) |value| try std.testing.expect(std.math.isNan(reference(T, value).re));
        }
        try bytes(T, &before, &factor);
        var zero = [_]T{number(T, 0, 0)};
        for ([_]lu.Operation{ .normal, .transposed, .adjoint }) |operation| {
            var nonzero = [_]T{number(T, 2, 3)};
            solve(T, i32, matrix(T, &zero, 1, 1, 1).asConst(), &.{1}, matrix(T, &nonzero, 1, 1, 1), operation);
            const got = reference(T, nonzero[0]);
            if (comptime types.isComplex(T)) {
                if (operation == .normal) {
                    try std.testing.expect(std.math.isInf(got.re) and std.math.isInf(got.im));
                } else {
                    // The subsequent unit-L transpose leg still evaluates
                    // (1+0i)*(Inf+Inf*i); both components include 0*Inf.
                    try std.testing.expect(std.math.isNan(got.re) and std.math.isNan(got.im));
                }
            } else try std.testing.expect(std.math.isInf(got.re));
        }
    }
}

test "LU GETRS IEEE stores overflow and continues later RHS instead of returning checked failure information" {
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const R = types.Real(T);
        const tiny = std.math.floatTrueMin(R);
        var factor = [_]T{number(T, tiny, 0)};
        const encoded = [_]i64{1};
        for ([_]lu.Operation{ .normal, .transposed, .adjoint }) |operation| {
            var rhs = [_]T{ number(T, std.math.floatMax(R), 0), number(T, 71, 17), number(T, @as(f64, tiny) * 2, 0), number(T, 73, 19) };
            const before = rhs;
            solve(T, i64, matrix(T, &factor, 1, 1, 1).asConst(), &encoded, matrix(T, &rhs, 1, 2, 2), operation);
            try std.testing.expect(std.math.isInf(reference(T, rhs[0]).re));
            try near(T, .{ .re = 2, .im = 0 }, rhs[2]);
            try bytes(T, before[1..2], rhs[1..2]);
            try bytes(T, before[3..4], rhs[3..4]);
        }
    }
}

test "LU IEEE empty shapes and single element native strides avoid unused factors and pivot access" {
    inline for (.{ f32, f64, types.ComplexF32, types.ComplexF64 }) |T| {
        const R = types.Real(T);
        var absent: [0]T = .{};
        var encoded: [0]i64 = .{};
        try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &absent, std.math.maxInt(usize), 0, 0), pivots.Writer(i64, .one){ .values = &encoded }, .getrf_ieee_v1));
        try std.testing.expectEqual(@as(usize, 0), lu.factorizeWithContract(T, matrix(T, &absent, 0, std.math.maxInt(usize), 0), pivots.Writer(i64, .one){ .values = &encoded }, .getrf_ieee_v1));
        var poison = [_]T{number(T, std.math.nan(R), std.math.inf(R))};
        const before = poison;
        for ([_]lu.Operation{ .normal, .transposed, .adjoint }) |operation| {
            solve(T, i64, matrix(T, &poison, 1, 1, 1).asConst(), &encoded, matrix(T, &absent, 1, 0, 0), operation);
            solve(T, i64, matrix(T, &absent, 0, 0, 0).asConst(), &encoded, matrix(T, &absent, 0, std.math.maxInt(usize), std.math.maxInt(usize)), operation);
            var factor = [_]T{number(T, 2, 0)};
            var rhs = [_]T{number(T, 6, -4)};
            solve(T, i64, matrix(T, &factor, 1, 1, std.math.maxInt(usize)).asConst(), &.{1}, matrix(T, &rhs, 1, 1, std.math.maxInt(usize)), operation);
            try near(T, reference(T, number(T, 3, -2)), rhs[0]);
        }
        try bytes(T, &before, &poison);
    }
}
