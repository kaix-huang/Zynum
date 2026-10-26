// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const kernel = Core.internal.numerics.ComplexBidiagonal;
const dense = Core.internal.numerics.Dense;
const scalar = Core.internal.numerics.Scalar;
fn product(comptime Z: type, a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn tolerance(comptime C: type) kernel.Wide(C) {
    return if (C == scalar.ComplexF32) 2e-12 else 2e-28;
}
fn orthogonal(comptime C: type, q: dense.Matrix(kernel.WorkspaceScalar(C))) !void {
    const W = kernel.Wide(C);
    const Z = kernel.WorkspaceScalar(C);
    for (0..q.column_count) |p| for (0..q.column_count) |r| {
        var dot: Z = .{ .re = 0, .im = 0 };
        for (0..q.row_count) |i| {
            const x = q.values[i + p * q.leading_dimension];
            const y = q.values[i + r * q.leading_dimension];
            const term = product(Z, .{ .re = x.re, .im = -x.im }, y);
            dot.re += term.re;
            dot.im += term.im;
        }
        try std.testing.expect(@abs(dot.re - @as(W, if (p == r) 1 else 0)) <= tolerance(C));
        try std.testing.expect(@abs(dot.im) <= tolerance(C));
    };
}
fn reconstruct(comptime C: type, original: dense.Matrix(kernel.WorkspaceScalar(C)), b: dense.Matrix(kernel.WorkspaceScalar(C)), u: dense.Matrix(kernel.WorkspaceScalar(C)), v: dense.Matrix(kernel.WorkspaceScalar(C))) !void {
    const Z = kernel.WorkspaceScalar(C);
    var scale: kernel.Wide(C) = 0;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        const z = original.values[i + j * original.leading_dimension];
        scale = @max(scale, @max(@abs(z.re), @abs(z.im)));
    };
    if (scale == 0) scale = 1;
    for (0..original.column_count) |j| for (0..original.row_count) |i| {
        var sum: Z = .{ .re = 0, .im = 0 };
        for (0..b.row_count) |p| for (0..b.column_count) |q| {
            const vr = v.values[j + q * v.leading_dimension];
            const term = product(Z, product(Z, u.values[i + p * u.leading_dimension], b.values[p + q * b.leading_dimension]), .{ .re = vr.re, .im = -vr.im });
            sum.re += term.re;
            sum.im += term.im;
        };
        const expected = original.values[i + j * original.leading_dimension];
        try std.testing.expect(@abs(sum.re - expected.re) / scale <= tolerance(C));
        try std.testing.expect(@abs(sum.im - expected.im) / scale <= tolerance(C));
    };
    try orthogonal(C, u);
    try orthogonal(C, v);
}
fn padding(comptime Z: type, values: []const Z, rows: usize, cols: usize, ld: usize, marker: Z) !void {
    for (values, 0..) |value, index| {
        if (index / ld < cols and index % ld < rows) continue;
        try std.testing.expectEqual(marker, value);
    }
}
fn fixture(comptime C: type, m: usize, n: usize, scale: kernel.Wide(C), budget: usize) !void {
    const W = kernel.Wide(C);
    const Z = kernel.WorkspaceScalar(C);
    const marker: Z = .{ .re = 123, .im = -45 };
    var bv: [64]Z = @splat(marker);
    var uv: [96]Z = @splat(marker);
    var vv: [64]Z = @splat(marker);
    var row: [16]Z = @splat(marker);
    var workspace: [128]W = @splat(319);
    const b: dense.Matrix(Z) = .{ .values = &bv, .row_count = m, .column_count = n, .leading_dimension = m + 2 };
    const u: dense.Matrix(Z) = .{ .values = &uv, .row_count = m, .column_count = m, .leading_dimension = m + 2 };
    const v: dense.Matrix(Z) = .{ .values = &vv, .row_count = n, .column_count = n, .leading_dimension = n + 2 };
    for (0..n) |j| for (0..m) |i| {
        const signed: i32 = @as(i32, @intCast(i)) - @as(i32, @intCast(j));
        b.values[i + j * b.leading_dimension] = .{ .re = scale * (@as(W, @floatFromInt((i + 2 * j + 1) % 7)) - 2), .im = scale * @as(W, @floatFromInt(signed)) };
    };
    var saved = bv;
    const original: dense.Matrix(Z) = .{ .values = &saved, .row_count = m, .column_count = n, .leading_dimension = m + 2 };
    try kernel.reduce(C, b, u, v);
    try reconstruct(C, original, b, u, v);
    for (0..n) |j| for (0..m) |i| {
        const z = b.values[i + j * b.leading_dimension];
        try std.testing.expectEqual(@as(W, 0), z.im);
        if (i != j and j != i + 1) try std.testing.expectEqual(@as(W, 0), z.re);
    };
    const result = try kernel.diagonalize(C, b, u, v, &workspace, &row, .{ .max_steps = budget });
    try std.testing.expect(result.steps <= budget);
    var edges: usize = 0;
    if (n > 1) for (0..n - 1) |i| {
        if (b.values[i + (i + 1) * b.leading_dimension].re != 0) edges += 1;
    };
    try std.testing.expectEqual(edges, result.unconverged);
    try std.testing.expectEqual(edges == 0, result.converged);
    if (budget == 0 and scale != 0 and n > 1) try std.testing.expect(edges > 0);
    if (budget == 4096) try std.testing.expect(result.converged);
    if (result.converged) for (0..n) |i| {
        const diagonal = b.values[i + i * b.leading_dimension].re;
        try std.testing.expect(diagonal >= 0);
        if (i + 1 < n) try std.testing.expect(diagonal >= b.values[i + 1 + (i + 1) * b.leading_dimension].re);
    };
    try reconstruct(C, original, b, u, v);
    try padding(Z, &bv, m, n, m + 2, marker);
    try padding(Z, &uv, m, m, m + 2, marker);
    try padding(Z, &vv, n, n, n + 2, marker);
    for (workspace[try kernel.realWorkspaceLength(m, n)..]) |value| try std.testing.expectEqual(@as(W, 319), value);
    for (row[m..]) |value| try std.testing.expectEqual(marker, value);
}
test "Complex bidiagonal reconstruction scaled tall square singleton and zero" {
    var configurations: usize = 0;
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const W = kernel.Wide(C);
        const tiny: W = if (C == scalar.ComplexF32) 1e-30 else 1e-280;
        const huge: W = if (C == scalar.ComplexF32) 1e30 else 1e280;
        for ([_]W{ 0, 1, -1, tiny, huge }) |scale| {
            for ([_][2]usize{ .{ 4, 3 }, .{ 3, 3 }, .{ 5, 1 } }) |shape| {
                try fixture(C, shape[0], shape[1], scale, 4096);
                configurations += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 30), configurations);
}
test "Complex bidiagonal partial QR retains factors and actual edge count" {
    var configurations: usize = 0;
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        for ([_]usize{ 0, 1, 2 }) |budget| {
            try fixture(C, 4, 3, 1, budget);
            configurations += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 6), configurations);
}
test "Complex bidiagonal dense known singular spectrum with independent unitary factors" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        const signs = [4][4]W{ .{ 1, 1, 1, 1 }, .{ 1, -1, 1, -1 }, .{ 1, 1, -1, -1 }, .{ 1, -1, -1, 1 } };
        const phases = [4]Z{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 }, .{ .re = -1, .im = 0 }, .{ .re = 0, .im = -1 } };
        const r = @sqrt(@as(W, 0.5));
        const known_right = [3][3]Z{
            .{ .{ .re = r, .im = 0 }, .{ .re = 0, .im = r }, .{ .re = 0, .im = 0 } },
            .{ .{ .re = 0, .im = r }, .{ .re = r, .im = 0 }, .{ .re = 0, .im = 0 } },
            .{ .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 1 } },
        };
        const expected = [3]W{ 5, 2, 0 };
        var bv: [12]Z = @splat(.{ .re = 0, .im = 0 });
        for (0..3) |j| for (0..4) |i| {
            for (0..3) |q| {
                const left: Z = .{ .re = phases[i].re * signs[i][q] * 0.5, .im = phases[i].im * signs[i][q] * 0.5 };
                const right = known_right[j][q];
                const term = product(Z, left, .{ .re = right.re * expected[q], .im = -right.im * expected[q] });
                bv[i + j * 4].re += term.re;
                bv[i + j * 4].im += term.im;
            }
        };
        var original = bv;
        var uv: [16]Z = undefined;
        var vv: [9]Z = undefined;
        var real_work: [37]W = undefined;
        var row: [4]Z = undefined;
        const b: dense.Matrix(Z) = .{ .values = &bv, .row_count = 4, .column_count = 3, .leading_dimension = 4 };
        const u: dense.Matrix(Z) = .{ .values = &uv, .row_count = 4, .column_count = 4, .leading_dimension = 4 };
        const v: dense.Matrix(Z) = .{ .values = &vv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const input: dense.Matrix(Z) = .{ .values = &original, .row_count = 4, .column_count = 3, .leading_dimension = 4 };
        try kernel.reduce(C, b, u, v);
        try reconstruct(C, input, b, u, v);
        const result = try kernel.diagonalize(C, b, u, v, &real_work, &row, .{});
        try std.testing.expect(result.converged and result.unconverged == 0);
        for (expected, 0..) |singular, i| try std.testing.expect(@abs(bv[i + i * 4].re - singular) <= tolerance(C));
        try reconstruct(C, input, b, u, v);
    }
}
test "Complex bidiagonal exact complex diagonal singular values and sorted phases" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const W = kernel.Wide(C);
        const Z = kernel.WorkspaceScalar(C);
        var bv: [9]Z = @splat(.{ .re = 0, .im = 0 });
        bv[0] = .{ .re = 0, .im = -2 };
        bv[4] = .{ .re = -5, .im = 0 };
        var uv: [9]Z = undefined;
        var vv: [9]Z = undefined;
        var workspace: [27]W = undefined;
        var row: [3]Z = undefined;
        var original = bv;
        const b: dense.Matrix(Z) = .{ .values = &bv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const u: dense.Matrix(Z) = .{ .values = &uv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const v: dense.Matrix(Z) = .{ .values = &vv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        try kernel.reduce(C, b, u, v);
        const result = try kernel.diagonalize(C, b, u, v, &workspace, &row, .{ .max_steps = 0 });
        try std.testing.expect(result.converged);
        try std.testing.expectEqual(@as(W, 5), bv[0].re);
        try std.testing.expectEqual(@as(W, 2), bv[4].re);
        try std.testing.expectEqual(@as(W, 0), bv[8].re);
        try reconstruct(C, .{ .values = &original, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, b, u, v);
    }
}
test "Complex bidiagonal rejects invalid geometry alias work and nonfinite input before mutation" {
    const C = scalar.ComplexF64;
    const Z = kernel.WorkspaceScalar(C);
    var bv: [12]Z = @splat(.{ .re = 1, .im = 1 });
    var uv: [16]Z = @splat(.{ .re = 2, .im = -1 });
    var vv: [9]Z = @splat(.{ .re = 3, .im = -2 });
    var workspace: [37]f128 = @splat(42);
    var row: [4]Z = @splat(.{ .re = 17, .im = -18 });
    const b: dense.Matrix(Z) = .{ .values = &bv, .row_count = 4, .column_count = 3, .leading_dimension = 4 };
    const u: dense.Matrix(Z) = .{ .values = &uv, .row_count = 4, .column_count = 4, .leading_dimension = 4 };
    const v: dense.Matrix(Z) = .{ .values = &vv, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
    const saved = bv;
    const saved_u = uv;
    var wrong = b;
    wrong.row_count = 2;
    try std.testing.expectError(error.DimensionMismatch, kernel.reduce(C, wrong, u, v));
    wrong = b;
    wrong.values = uv[0..12];
    try std.testing.expectError(error.AliasingNotAllowed, kernel.reduce(C, wrong, u, v));
    try std.testing.expectError(error.BufferTooSmall, kernel.diagonalize(C, b, u, v, workspace[0..36], &row, .{}));
    try std.testing.expectError(error.BufferTooSmall, kernel.diagonalize(C, b, u, v, &workspace, row[0..3], .{}));
    try std.testing.expectError(error.AliasingNotAllowed, kernel.diagonalize(C, b, u, v, &workspace, uv[0..4], .{}));
    var mixed_arena: [40]Z = @splat(.{ .re = 1, .im = 0 });
    var mixed_u = u;
    mixed_u.values = mixed_arena[0..16];
    const real_alias: [*]f128 = @ptrCast(&mixed_arena);
    try std.testing.expectError(error.AliasingNotAllowed, kernel.diagonalize(C, b, mixed_u, v, real_alias[0..37], &row, .{}));
    try std.testing.expectError(error.NotBidiagonal, kernel.diagonalize(C, b, u, v, &workspace, &row, .{}));
    try std.testing.expectEqualSlices(Z, &saved, &bv);
    try std.testing.expectEqualSlices(Z, &saved_u, &uv);
    bv[0].re = std.math.nan(f128);
    try std.testing.expectError(error.NonFiniteInput, kernel.reduce(C, b, u, v));
    try std.testing.expectEqualSlices(Z, &saved_u, &uv);
    bv[0].re = saved[0].re;
    uv[0].im = std.math.nan(f128);
    try std.testing.expectError(error.NonFiniteInput, kernel.diagonalize(C, b, u, v, &workspace, &row, .{}));
    for (workspace) |value| try std.testing.expectEqual(@as(f128, 42), value);
    for (row) |value| try std.testing.expectEqual(Z{ .re = 17, .im = -18 }, value);
    try std.testing.expectError(error.SizeOverflow, kernel.realWorkspaceLength(std.math.maxInt(usize), 1));
}
test "Complex bidiagonal empty matrices use existing borrowed descriptors" {
    const C = scalar.ComplexF64;
    const Z = kernel.WorkspaceScalar(C);
    const empty: dense.Matrix(Z) = .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
    try kernel.reduce(C, empty, empty, empty);
    const result = try kernel.diagonalize(C, empty, empty, empty, &.{}, &.{}, .{ .max_steps = 0 });
    try std.testing.expect(result.converged and result.unconverged == 0 and result.steps == 0);
}
