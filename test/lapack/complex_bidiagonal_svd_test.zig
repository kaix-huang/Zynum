// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const kernel = Core.internal.numerics.ComplexBidiagonalSvd;
const dense = Core.internal.numerics.Dense;
const scalar = Core.internal.numerics.Scalar;
fn multiply(comptime Z: type, a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn tolerance(comptime C: type) kernel.Wide(C) {
    return 256 * @as(kernel.Wide(C), std.math.floatEps(kernel.Real(C)));
}
fn orthogonal(comptime C: type, q: dense.Matrix(C)) !void {
    const W = kernel.Wide(C);
    for (0..q.column_count) |p| for (0..q.column_count) |j| {
        var re: W = 0;
        var im: W = 0;
        for (0..q.row_count) |i| {
            const x = q.values[i + p * q.leading_dimension];
            const y = q.values[i + j * q.leading_dimension];
            re += @as(W, x.re) * y.re + @as(W, x.im) * y.im;
            im += @as(W, x.re) * y.im - @as(W, x.im) * y.re;
        }
        try std.testing.expect(std.math.hypot(re - @as(W, if (p == j) 1 else 0), im) <= tolerance(C));
    };
}
fn reconstruction(comptime C: type, comptime S: type, original: dense.Matrix(C), u: dense.Matrix(C), singular: []const S, v: dense.Matrix(C)) !void {
    const W = kernel.Wide(C);
    const Z = kernel.WorkspaceScalar(C);
    var scale: W = 0;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        const z = original.values[i + j * original.leading_dimension];
        scale = @max(scale, @max(@abs(@as(W, z.re)), @abs(@as(W, z.im))));
    };
    if (scale == 0) scale = 1;
    const quantum: W = if (S == kernel.Real(C)) @as(W, @floatFromInt(singular.len)) * std.math.floatTrueMin(kernel.Real(C)) / scale else 0;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        var sum: Z = .{ .re = 0, .im = 0 };
        for (singular, 0..) |sigma, q| {
            const left = u.values[i + q * u.leading_dimension];
            const right = v.values[j + q * v.leading_dimension];
            const term = multiply(Z, .{ .re = left.re, .im = left.im }, .{ .re = @as(W, right.re) * sigma, .im = -@as(W, right.im) * sigma });
            sum.re += term.re;
            sum.im += term.im;
        }
        const expected = original.values[i + j * original.leading_dimension];
        try std.testing.expect(@abs(sum.re - @as(W, expected.re)) / scale <= tolerance(C) + quantum);
        try std.testing.expect(@abs(sum.im - @as(W, expected.im)) / scale <= tolerance(C) + quantum);
    };
    try orthogonal(C, u);
    try orthogonal(C, v);
}
fn partialReconstruction(comptime C: type, original: dense.Matrix(C), state: kernel.Factors(C)) !void {
    const W = kernel.Wide(C);
    const Z = kernel.WorkspaceScalar(C);
    var scale: W = 0;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        const z = original.values[i + j * original.leading_dimension];
        scale = @max(scale, @max(@abs(@as(W, z.re)), @abs(@as(W, z.im))));
    };
    if (scale == 0) scale = 1;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        var sum: Z = .{ .re = 0, .im = 0 };
        for (0..state.left.column_count) |p| for (0..state.right.column_count) |q| {
            const b = if (state.adjoint_reduction) state.b_tall.values[q + p * state.b_tall.leading_dimension] else state.b_tall.values[p + q * state.b_tall.leading_dimension];
            const z: Z = .{ .re = b.re, .im = if (state.adjoint_reduction) -b.im else b.im };
            const right = state.right.values[j + q * state.right.leading_dimension];
            const term = multiply(Z, multiply(Z, state.left.values[i + p * state.left.leading_dimension], z), .{ .re = right.re, .im = -right.im });
            sum.re += term.re;
            sum.im += term.im;
        };
        const expected = original.values[i + j * original.leading_dimension];
        try std.testing.expect(@abs(sum.re - @as(W, expected.re)) / scale <= tolerance(C));
        try std.testing.expect(@abs(sum.im - @as(W, expected.im)) / scale <= tolerance(C));
    };
}
fn unchangedOutside(comptime C: type, values: []const C, original: []const C, rows: usize, columns: usize, ld: usize) !void {
    for (values, original, 0..) |value, old, index| {
        if (index / ld < columns and index % ld < rows) continue;
        try std.testing.expectEqual(old, value);
    }
}
fn fixture(comptime C: type, comptime S: type, m: usize, n: usize, scale: kernel.Wide(C), options: kernel.Options) !void {
    const W = kernel.Wide(C);
    const Z = kernel.WorkspaceScalar(C);
    const marker: C = .{ .re = 123, .im = -45 };
    const work_marker: Z = .{ .re = 71, .im = -29 };
    var av: [128]C = @splat(marker);
    var vv: [128]C = @splat(marker);
    var singular: [8]S = @splat(17);
    var complex_work: [512]Z = @splat(work_marker);
    var real_work: [512]W = @splat(91);
    const k = @min(m, n);
    const a: dense.Matrix(C) = .{ .values = &av, .row_count = m, .column_count = n, .leading_dimension = m + 2 };
    const v: dense.Matrix(C) = .{ .values = &vv, .row_count = n, .column_count = k, .leading_dimension = n + 2 };
    for (0..n) |j| for (0..m) |i| {
        const imaginary: i32 = @as(i32, @intCast(i)) - @as(i32, @intCast(j));
        a.values[i + j * a.leading_dimension] = .{ .re = @floatCast(scale * (@as(W, @floatFromInt((i + 2 * j + 1) % 7)) - 2)), .im = @floatCast(scale * @as(W, @floatFromInt(imaginary))) };
    };
    var saved = av;
    const saved_v = vv;
    const needed = try kernel.plan(C, m, n);
    const result = try kernel.decomposeWithSingularType(C, S, a, &singular, v, &complex_work, &real_work, options);
    try std.testing.expectEqual(.converged, result.status);
    try std.testing.expectEqual(@as(usize, 0), result.unconverged);
    for (singular[0..k], 0..) |value, i| {
        try std.testing.expect(value >= 0);
        if (i + 1 < k) try std.testing.expect(value >= singular[i + 1]);
    }
    const u: dense.Matrix(C) = .{ .values = &av, .row_count = m, .column_count = k, .leading_dimension = m + 2 };
    try reconstruction(C, S, .{ .values = &saved, .row_count = m, .column_count = n, .leading_dimension = m + 2 }, u, singular[0..k], v);
    try unchangedOutside(C, &av, &saved, m, k, m + 2);
    try unchangedOutside(C, &vv, &saved_v, n, k, n + 2);
    for (singular[k..]) |value| try std.testing.expectEqual(@as(S, 17), value);
    for (complex_work[needed.complex_elements..]) |value| try std.testing.expectEqual(work_marker, value);
    for (real_work[needed.real_elements..]) |value| try std.testing.expectEqual(@as(W, 91), value);
}
test "Complex rectangular bidiagonal SVD both precisions singular widths shapes and scalar ranges" {
    var configurations: usize = 0;
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = kernel.Real(C);
        const W = kernel.Wide(C);
        const tiny: W = if (C == scalar.ComplexF32) 1e-30 else 1e-280;
        const huge: W = if (C == scalar.ComplexF32) 1e30 else 1e280;
        inline for (.{ R, W }) |S| for ([_]W{ 0, 1, -1, tiny, huge, std.math.floatTrueMin(R), @as(W, std.math.floatMax(R)) / 128 }) |scale| {
            for ([_][2]usize{ .{ 4, 3 }, .{ 3, 5 }, .{ 3, 3 }, .{ 1, 4 }, .{ 4, 1 } }) |shape| {
                inline for (.{ false, true }) |ieee| {
                    try fixture(C, S, shape[0], shape[1], scale, .{ .numerical_policy = if (ieee) .ieee else .finite });
                    configurations += 1;
                }
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 280), configurations);
}
test "Complex rectangular bidiagonal SVD known nontrivial rank deficient tall and wide spectra" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        const expected = [3]W{ 5, 2, 0 };
        const signs = [4][3]W{ .{ 1, 1, 1 }, .{ 1, -1, 1 }, .{ 1, 1, -1 }, .{ 1, -1, -1 } };
        const phases = [4]Z{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 }, .{ .re = -1, .im = 0 }, .{ .re = 0, .im = -1 } };
        const r = @sqrt(@as(W, 0.5));
        const right = [3][3]Z{
            .{ .{ .re = r, .im = 0 }, .{ .re = 0, .im = r }, .{ .re = 0, .im = 0 } },
            .{ .{ .re = 0, .im = r }, .{ .re = r, .im = 0 }, .{ .re = 0, .im = 0 } },
            .{ .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 1 } },
        };
        var tall: [12]C = undefined;
        for (0..3) |j| for (0..4) |i| {
            var value: Z = .{ .re = 0, .im = 0 };
            for (0..3) |q| {
                const left: Z = .{ .re = phases[i].re * signs[i][q] * 0.5, .im = phases[i].im * signs[i][q] * 0.5 };
                const z = right[j][q];
                const term = multiply(Z, left, .{ .re = z.re * expected[q], .im = -z.im * expected[q] });
                value.re += term.re;
                value.im += term.im;
            }
            tall[i + j * 4] = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
        };
        for ([_]bool{ false, true }) |wide| {
            const m: usize = if (wide) 3 else 4;
            const n: usize = if (wide) 4 else 3;
            var av: [12]C = undefined;
            for (0..3) |j| for (0..4) |i| {
                const z = tall[i + j * 4];
                av[if (wide) j + i * 3 else i + j * 4] = .{ .re = z.re, .im = if (wide) -z.im else z.im };
            };
            var saved = av;
            var vv: [12]C = undefined;
            var singular: [3]kernel.Real(C) = undefined;
            var complex_work: [128]Z = undefined;
            var real_work: [128]W = undefined;
            const a: dense.Matrix(C) = .{ .values = &av, .row_count = m, .column_count = n, .leading_dimension = m };
            const v: dense.Matrix(C) = .{ .values = &vv, .row_count = n, .column_count = 3, .leading_dimension = n };
            const result = try kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{});
            try std.testing.expectEqual(.converged, result.status);
            for (expected, singular) |wanted, actual| try std.testing.expect(@abs(@as(W, actual) - wanted) <= 5 * tolerance(C));
            try reconstruction(C, kernel.Real(C), .{ .values = &saved, .row_count = m, .column_count = n, .leading_dimension = m }, .{ .values = &av, .row_count = m, .column_count = 3, .leading_dimension = m }, &singular, v);
        }
    }
}
test "Complex rectangular bidiagonal SVD exhausted QR retains outputs factors original scale and edge count" {
    var configurations: usize = 0;
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        const small: W = if (C == scalar.ComplexF32) 1e-20 else 1e-200;
        const large: W = if (C == scalar.ComplexF32) 1e20 else 1e200;
        for ([_][2]usize{ .{ 4, 3 }, .{ 3, 4 } }) |shape| for ([_]W{ 1, small, large }) |scale| for ([_]usize{ 0, 1 }) |budget| {
            const m = shape[0];
            const n = shape[1];
            var av: [12]C = undefined;
            for (0..n) |j| for (0..m) |i| {
                av[i + j * m] = .{ .re = @floatCast(scale * @as(W, @floatFromInt(1 + (i + 2 * j) % 5))), .im = @floatCast(scale * @as(W, @floatFromInt(i + j + 1))) };
            };
            var saved = av;
            var vv: [12]C = @splat(.{ .re = 79, .im = -31 });
            const old_v = vv;
            var singular: [3]kernel.Real(C) = @splat(19);
            var complex_work: [128]Z = undefined;
            var real_work: [128]W = undefined;
            const a: dense.Matrix(C) = .{ .values = &av, .row_count = m, .column_count = n, .leading_dimension = m };
            const result = try kernel.decompose(C, a, &singular, .{ .values = &vv, .row_count = n, .column_count = 3, .leading_dimension = n }, &complex_work, &real_work, .{ .max_steps = budget });
            try std.testing.expectEqual(.no_convergence, result.status);
            try std.testing.expectEqual(budget, result.steps);
            try std.testing.expectEqualSlices(C, &saved, &av);
            try std.testing.expectEqualSlices(C, &old_v, &vv);
            for (singular) |value| try std.testing.expectEqual(@as(kernel.Real(C), 19), value);
            const state = try kernel.factors(C, m, n, &complex_work);
            var edges: usize = 0;
            for (0..2) |i| if (state.b_tall.values[i + (i + 1) * state.b_tall.leading_dimension].re != 0) {
                edges += 1;
            };
            try std.testing.expect(edges > 0 and edges == result.unconverged);
            try partialReconstruction(C, .{ .values = &saved, .row_count = m, .column_count = n, .leading_dimension = m }, state);
            configurations += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 24), configurations);
}
test "Complex rectangular bidiagonal SVD representability and rounded unitary failures commit no output" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = kernel.Real(C);
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        var av = [2]C{ .{ .re = std.math.floatMax(R), .im = 0 }, .{ .re = std.math.floatMax(R), .im = 0 } };
        const saved = av;
        var vv = [1]C{.{ .re = 71, .im = -31 }};
        const saved_v = vv;
        var singular = [1]R{17};
        var wide_singular: [1]W = undefined;
        var complex_work: [16]Z = undefined;
        var real_work: [16]W = undefined;
        const a: dense.Matrix(C) = .{ .values = &av, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const v: dense.Matrix(C) = .{ .values = &vv, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const result = try kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{});
        try std.testing.expectEqual(.non_finite_result, result.status);
        try std.testing.expectEqualSlices(C, &saved, &av);
        try std.testing.expectEqualSlices(C, &saved_v, &vv);
        try std.testing.expectEqual(@as(R, 17), singular[0]);
        const widened = try kernel.decomposeWithSingularType(C, W, a, &wide_singular, v, &complex_work, &real_work, .{});
        try std.testing.expectEqual(.converged, widened.status);
        try std.testing.expect(@abs(wide_singular[0] / @as(W, std.math.floatMax(R)) - @sqrt(@as(W, 2))) <= tolerance(C));
        av = .{ .{ .re = 1, .im = 0 }, .{ .re = 1, .im = 0 } };
        const rounded_input = av;
        vv = saved_v;
        const too_precise = try kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{ .relative_tolerance = 1e-40 });
        try std.testing.expectEqual(.range_limit, too_precise.status);
        try std.testing.expectEqualSlices(C, &rounded_input, &av);
        try std.testing.expectEqualSlices(C, &saved_v, &vv);
        try std.testing.expectEqual(@as(R, 17), singular[0]);
    }
}
test "Complex rectangular bidiagonal SVD preserves simultaneous huge and subnormal singular directions" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = kernel.Real(C);
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        const high: R = std.math.floatMax(R) / 16;
        const low: R = std.math.floatTrueMin(R) * 8;
        var av = [4]C{ .{ .re = high, .im = 0 }, .{ .re = 0, .im = high }, .{ .re = low, .im = 0 }, .{ .re = 0, .im = -low } };
        var saved = av;
        var vv: [4]C = undefined;
        var singular: [2]W = undefined;
        var complex_work: [32]Z = undefined;
        var real_work: [32]W = undefined;
        const a: dense.Matrix(C) = .{ .values = &av, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const v: dense.Matrix(C) = .{ .values = &vv, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const result = try kernel.decomposeWithSingularType(C, W, a, &singular, v, &complex_work, &real_work, .{});
        try std.testing.expectEqual(.converged, result.status);
        try std.testing.expect(singular[0] > 0 and singular[1] > 0);
        const precision: W = if (C == scalar.ComplexF32) 2e-12 else 2e-28;
        try std.testing.expect(@abs(singular[0] / @as(W, high) - @sqrt(@as(W, 2))) <= precision);
        try std.testing.expect(@abs(singular[1] / @as(W, low) - @sqrt(@as(W, 2))) <= precision);
        try reconstruction(C, W, .{ .values = &saved, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, a, &singular, v);
    }
}
test "Complex rectangular bidiagonal SVD rejects geometry aliases short storage tolerance and nonfinite values before mutation" {
    const C = scalar.ComplexF64;
    const Z = kernel.WorkspaceScalar(C);
    var av: [128]C = @splat(.{ .re = 1, .im = 2 });
    const saved = av;
    var vv: [128]C = @splat(.{ .re = 3, .im = -4 });
    const old_v = vv;
    var singular: [3]f64 = @splat(17);
    var complex_work: [128]Z = @splat(.{ .re = 41, .im = -43 });
    var real_work: [128]f128 = @splat(59);
    const a: dense.Matrix(C) = .{ .values = &av, .row_count = 4, .column_count = 3, .leading_dimension = 4 };
    const v: dense.Matrix(C) = .{ .values = &vv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
    const needed = try kernel.plan(C, 4, 3);
    try std.testing.expectError(error.BufferTooSmall, kernel.decompose(C, a, &singular, v, complex_work[0 .. needed.complex_elements - 1], &real_work, .{}));
    try std.testing.expectError(error.BufferTooSmall, kernel.decompose(C, a, &singular, v, &complex_work, real_work[0 .. needed.real_elements - 1], .{}));
    try std.testing.expectError(error.BufferTooSmall, kernel.decompose(C, a, singular[0..2], v, &complex_work, &real_work, .{}));
    var alias_v = v;
    alias_v.values = &av;
    try std.testing.expectError(error.AliasingNotAllowed, kernel.decompose(C, a, &singular, alias_v, &complex_work, &real_work, .{}));
    const alias_real: [*]f128 = @ptrCast(&complex_work);
    try std.testing.expectError(error.AliasingNotAllowed, kernel.decompose(C, a, &singular, v, &complex_work, alias_real[0..needed.real_elements], .{}));
    const alias_singular: [*]f64 = @ptrCast(&av);
    try std.testing.expectError(error.AliasingNotAllowed, kernel.decompose(C, a, alias_singular[0..3], v, &complex_work, &real_work, .{}));
    var wrong = v;
    wrong.column_count = 2;
    try std.testing.expectError(error.DimensionMismatch, kernel.decompose(C, a, &singular, wrong, &complex_work, &real_work, .{}));
    for ([_]f64{ 0, 1, std.math.nan(f64), std.math.inf(f64) }) |bad| try std.testing.expectError(error.InvalidTolerance, kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{ .relative_tolerance = bad }));
    try std.testing.expectEqualSlices(C, &saved, &av);
    av[0].im = std.math.nan(f64);
    const nonfinite = av;
    try std.testing.expectError(error.NonFiniteInput, kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{}));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&nonfinite), std.mem.asBytes(&av));
    try std.testing.expectEqualSlices(C, &old_v, &vv);
    for (singular) |value| try std.testing.expectEqual(@as(f64, 17), value);
    for (complex_work) |value| try std.testing.expectEqual(Z{ .re = 41, .im = -43 }, value);
    for (real_work) |value| try std.testing.expectEqual(@as(f128, 59), value);
    try std.testing.expectError(error.SizeOverflow, kernel.plan(C, std.math.maxInt(usize), 1));
}
test "Complex rectangular bidiagonal SVD empty and huge empty shapes require no workspace or publication" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        for ([_][2]usize{ .{ 0, 0 }, .{ 0, 3 }, .{ 3, 0 }, .{ std.math.maxInt(usize), 0 } }) |shape| {
            var av = [1]C{.{ .re = 17, .im = -19 }};
            const saved = av;
            var singular = [1]kernel.Real(C){29};
            const p = try kernel.plan(C, shape[0], shape[1]);
            try std.testing.expect(p.complex_elements == 0 and p.real_elements == 0);
            const result = try kernel.decompose(C, .{ .values = &av, .row_count = shape[0], .column_count = shape[1], .leading_dimension = 0 }, &singular, .{ .values = &av, .row_count = shape[1], .column_count = 0, .leading_dimension = 0 }, &.{}, &.{}, .{});
            try std.testing.expect(result.status == .converged and result.unconverged == 0 and result.steps == 0);
            try std.testing.expectEqualSlices(C, &saved, &av);
            try std.testing.expectEqual(@as(kernel.Real(C), 29), singular[0]);
        }
    }
}
test "Complex rectangular bidiagonal SVD IEEE nonfinite inputs remain bounded with actual edge counts" {
    var configurations: usize = 0;
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = kernel.Real(C);
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        for ([_]usize{ 1, 2 }) |n| for ([_]bool{ false, true }) |infinite| for ([_]usize{ 0, 2 }) |budget| {
            const invalid = if (infinite) std.math.inf(R) else std.math.nan(R);
            var av = [4]C{ .{ .re = invalid, .im = 1 }, .{ .re = 2, .im = 3 }, .{ .re = -4, .im = 2 }, .{ .re = 1, .im = invalid } };
            const original = av;
            var vv: [4]C = @splat(.{ .re = 71, .im = -31 });
            const old_v = vv;
            var singular: [2]R = @splat(17);
            var complex_work: [32]Z = undefined;
            var real_work: [32]W = undefined;
            const a: dense.Matrix(C) = .{ .values = &av, .row_count = n, .column_count = n, .leading_dimension = n };
            const v: dense.Matrix(C) = .{ .values = &vv, .row_count = n, .column_count = n, .leading_dimension = n };
            const result = try kernel.decompose(C, a, &singular, v, &complex_work, &real_work, .{ .max_steps = budget, .numerical_policy = .ieee });
            try std.testing.expect(result.steps <= budget);
            try std.testing.expect(result.status == .converged or result.status == .no_convergence);
            const state = try kernel.factors(C, n, n, &complex_work);
            var edges: usize = 0;
            if (n > 1 and state.b_tall.values[n].re != 0) edges += 1;
            try std.testing.expectEqual(edges, result.unconverged);
            if (result.status == .no_convergence) {
                try std.testing.expect(edges > 0);
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&original), std.mem.asBytes(&av));
                try std.testing.expectEqualSlices(C, &old_v, &vv);
                for (singular) |value| try std.testing.expectEqual(@as(R, 17), value);
            } else {
                try std.testing.expectEqual(@as(usize, 0), edges);
                var nonfinite = false;
                for (singular[0..n]) |value| nonfinite = nonfinite or !std.math.isFinite(value);
                try std.testing.expect(nonfinite);
            }
            configurations += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 16), configurations);
}
