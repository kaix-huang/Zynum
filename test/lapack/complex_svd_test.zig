// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("zynum-core");
const svd = owner.internal.numerics.ComplexSvd;
const dense = owner.internal.numerics.Dense;
const C32 = owner.ComplexF32;
const C64 = owner.ComplexF64;
fn z(comptime C: type, re: anytype, im: anytype) C {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}
fn same(comptime C: type, a: C, b: C) !void {
    try std.testing.expectEqual(a.re, b.re);
    try std.testing.expectEqual(a.im, b.im);
}
fn verify(comptime C: type, comptime S: type, original: []const C, a: dense.Matrix(C), singular: []const S, v: dense.Matrix(C)) !void {
    const W = svd.Wide(C);
    const m = a.row_count;
    const n = a.column_count;
    const k = @min(m, n);
    const tol: W = 256 * std.math.floatEps(svd.Real(C));
    var scale: W = 0;
    for (original) |x| scale = @max(scale, @max(@abs(@as(W, x.re)), @abs(@as(W, x.im))));
    if (scale == 0) scale = 1;
    for (0..k) |p| {
        try std.testing.expect(std.math.isFinite(singular[p]) and singular[p] >= 0);
        if (p != 0) try std.testing.expect(singular[p - 1] >= singular[p]);
        for (0..k) |q| {
            var ur: W = 0;
            var ui: W = 0;
            var vr: W = 0;
            var vi: W = 0;
            for (0..m) |i| {
                const x = a.values[i + p * a.leading_dimension];
                const y = a.values[i + q * a.leading_dimension];
                ur += @as(W, x.re) * y.re + @as(W, x.im) * y.im;
                ui += @as(W, x.re) * y.im - @as(W, x.im) * y.re;
            }
            for (0..n) |i| {
                const x = v.values[i + p * v.leading_dimension];
                const y = v.values[i + q * v.leading_dimension];
                vr += @as(W, x.re) * y.re + @as(W, x.im) * y.im;
                vi += @as(W, x.re) * y.im - @as(W, x.im) * y.re;
            }
            const wanted: W = if (p == q) 1 else 0;
            try std.testing.expectApproxEqAbs(wanted, ur, tol);
            try std.testing.expectApproxEqAbs(@as(W, 0), ui, tol);
            try std.testing.expectApproxEqAbs(wanted, vr, tol);
            try std.testing.expectApproxEqAbs(@as(W, 0), vi, tol);
        }
    }
    for (0..n) |j| for (0..m) |i| {
        var re: W = 0;
        var im: W = 0;
        for (0..k) |p| {
            const u = a.values[i + p * a.leading_dimension];
            const vv = v.values[j + p * v.leading_dimension];
            const s: W = @as(W, singular[p]) / scale;
            re += (@as(W, u.re) * vv.re + @as(W, u.im) * vv.im) * s;
            im += (@as(W, u.im) * vv.re - @as(W, u.re) * vv.im) * s;
        }
        try std.testing.expectApproxEqAbs(@as(W, original[i + j * m].re) / scale, re, tol);
        try std.testing.expectApproxEqAbs(@as(W, original[i + j * m].im) / scale, im, tol);
    };
}
fn checkCase(comptime C: type, comptime S: type, input: []const C, m: usize, n: usize) !void {
    const alloc = std.testing.allocator;
    const k = @min(m, n);
    const lda = m + 2;
    const ldv = n + 3;
    const data = try alloc.alloc(C, lda * n + 2);
    defer alloc.free(data);
    const right = try alloc.alloc(C, ldv * k + 2);
    defer alloc.free(right);
    const singular = try alloc.alloc(S, k + 2);
    defer alloc.free(singular);
    const used = try svd.workspaceLength(C, @max(m, n), k);
    const work = try alloc.alloc(svd.WorkspaceScalar(C), used + 2);
    defer alloc.free(work);
    const scratch_used = try svd.rectangularScratchLength(C, m, n);
    const scratch = try alloc.alloc(C, scratch_used + 2);
    defer alloc.free(scratch);
    const poison = z(C, 91.0, -17.0);
    @memset(data, poison);
    @memset(right, poison);
    @memset(singular, 93);
    @memset(work, .{ .re = 94, .im = -19 });
    @memset(scratch, poison);
    for (0..n) |j| for (0..m) |i| {
        data[i + j * lda] = input[i + j * m];
    };
    const a: dense.Matrix(C) = .{ .values = data, .row_count = m, .column_count = n, .leading_dimension = lda };
    const v: dense.Matrix(C) = .{ .values = right, .row_count = n, .column_count = k, .leading_dimension = ldv };
    const result = try svd.decomposeRectangularWithSingularType(C, S, a, singular, v, work, scratch, .{});
    try std.testing.expectEqual(svd.Status.converged, result.status);
    try std.testing.expect(result.output_orthogonality <= 32 * @as(f64, std.math.floatEps(svd.Real(C))));
    try verify(C, S, input, a, singular, v);
    for (0..n) |j| {
        for (m..lda) |i| try same(C, poison, data[i + j * lda]);
        if (j >= k) for (0..m) |i| try same(C, input[i + j * m], data[i + j * lda]);
    }
    for (0..k) |j| for (n..ldv) |i| try same(C, poison, right[i + j * ldv]);
    for (data[lda * n ..]) |x| try same(C, poison, x);
    for (right[ldv * k ..]) |x| try same(C, poison, x);
    for (singular[k..]) |x| try std.testing.expectEqual(@as(S, 93), x);
    for (work[used..]) |x| {
        try std.testing.expectEqual(@as(svd.Wide(C), 94), x.re);
        try std.testing.expectEqual(@as(svd.Wide(C), -19), x.im);
    }
    for (scratch[scratch_used..]) |x| try same(C, poison, x);
}
fn tall(comptime C: type) !void {
    const a = [_]C{ z(C, 1.0, 2.0), z(C, 3.0, -1.0), z(C, -2.0, 1.0), z(C, 0.5, 0.25), z(C, 2.0, -1.0), z(C, -1.0, 2.0), z(C, 4.0, 0.0), z(C, -2.0, -0.5), z(C, -1.0, 1.0), z(C, 0.5, -2.0), z(C, 1.0, 0.0), z(C, 3.0, 1.0) };
    try checkCase(C, svd.Real(C), &a, 4, 3);
}
fn wide(comptime C: type) !void {
    const a = [_]C{ z(C, 1.0, 2.0), z(C, 3.0, -1.0), z(C, -2.0, 1.0), z(C, 0.5, 0.25), z(C, 2.0, -1.0), z(C, -1.0, 2.0), z(C, 4.0, 0.0), z(C, -2.0, -0.5) };
    try checkCase(C, svd.Real(C), &a, 2, 4);
}
test "complex thin SVD C32 tall reconstructs Hermitian factors and preserves padding" {
    try tall(C32);
}
test "complex thin SVD C64 tall reconstructs Hermitian factors and preserves padding" {
    try tall(C64);
}
test "complex thin SVD C32 wide swaps conjugate-transpose factors transactionally" {
    try wide(C32);
}
test "complex thin SVD C64 wide swaps conjugate-transpose factors transactionally" {
    try wide(C64);
}
test "complex thin SVD imaginary phases require conjugate reconstruction" {
    inline for (.{ C32, C64 }) |C| {
        const a = [_]C{ z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 1.0), z(C, 1.0, 0.0) };
        try checkCase(C, svd.Real(C), &a, 2, 2);
    }
}
test "complex thin SVD zero columns and exact rank deficiency complete an orthonormal basis" {
    inline for (.{ C32, C64 }) |C| {
        const a = [_]C{ z(C, 1.0, 1.0), z(C, 2.0, -1.0), z(C, -1.0, 2.0), z(C, -1.0, 1.0), z(C, 1.0, 2.0), z(C, -2.0, -1.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0) };
        try checkCase(C, svd.Real(C), &a, 3, 3);
    }
}
test "complex thin SVD all zero tall and wide factors receive orthonormal completion" {
    inline for (.{ C32, C64 }) |C| {
        const a: [12]C = @splat(.{ .re = 0, .im = 0 });
        try checkCase(C, svd.Real(C), &a, 4, 3);
        try checkCase(C, svd.Real(C), a[0..8], 2, 4);
    }
}
test "complex thin SVD repeated singular values retain arbitrary unitary phases" {
    inline for (.{ C32, C64 }) |C| {
        const a = [_]C{ z(C, 0.0, 3.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 3.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, -3.0) };
        try checkCase(C, svd.Real(C), &a, 3, 3);
    }
}
test "complex thin SVD exhausted sweep budget leaves all outputs untouched" {
    inline for (.{ C32, C64 }) |C| {
        var a = [_]C{ z(C, 1.0, 1.0), z(C, 2.0, -1.0), z(C, 3.0, 2.0), z(C, 4.0, -1.0) };
        const before = a;
        var s: [2]svd.Real(C) = @splat(91);
        var v: [4]C = @splat(.{ .re = 92, .im = -7 });
        var w: [12]svd.WorkspaceScalar(C) = @splat(.{ .re = 93, .im = 9 });
        const r = try svd.decompose(C, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &w, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(svd.Status.no_convergence, r.status);
        try std.testing.expectEqual(@as(usize, 0), r.sweeps);
        for (a, before) |x, y| try same(C, x, y);
        for (s) |x| try std.testing.expectEqual(@as(svd.Real(C), 91), x);
        for (v) |x| try same(C, z(C, 92.0, -7.0), x);
        try std.testing.expectEqual(@as(svd.Wide(C), 93), w[10].re);
    }
}
test "complex thin SVD preflight rejects geometry capacity tolerance and mixed singular aliases" {
    inline for (.{ C32, C64 }) |C| {
        var a: [4]C = @splat(.{ .re = 1, .im = 0 });
        var s: [2]svd.Real(C) = @splat(91);
        var v: [4]C = @splat(.{ .re = 92, .im = 0 });
        var w: [10]svd.WorkspaceScalar(C) = @splat(.{ .re = 93, .im = 0 });
        const av: dense.Matrix(C) = .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const vv: dense.Matrix(C) = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        for ([_]f64{ 0, -1, 1, std.math.nan(f64), std.math.inf(f64) }) |tol| try std.testing.expectError(error.InvalidTolerance, svd.decompose(C, av, &s, vv, &w, .{ .relative_tolerance = tol }));
        try std.testing.expectError(error.BufferTooSmall, svd.decompose(C, av, s[0..1], vv, &w, .{}));
        try std.testing.expectError(error.BufferTooSmall, svd.decompose(C, av, &s, vv, w[0..9], .{}));
        var bad = av;
        bad.leading_dimension = 1;
        try std.testing.expectError(error.InvalidLeadingDimension, svd.decompose(C, bad, &s, vv, &w, .{}));
        bad = av;
        bad.row_count = 1;
        try std.testing.expectError(error.DimensionMismatch, svd.decompose(C, bad, &s, vv, &w, .{}));
        const aliased: [*]svd.Real(C) = @ptrCast(&a);
        try std.testing.expectError(error.AliasingNotAllowed, svd.decompose(C, av, aliased[0..2], vv, &w, .{}));
        try std.testing.expectError(error.AliasingNotAllowed, svd.decompose(C, av, &s, av, &w, .{}));
        for (a) |x| try same(C, z(C, 1.0, 0.0), x);
        for (s) |x| try std.testing.expectEqual(@as(svd.Real(C), 91), x);
        for (v) |x| try same(C, z(C, 92.0, 0.0), x);
        for (w) |x| try std.testing.expectEqual(@as(svd.Wide(C), 93), x.re);
    }
}
test "complex thin SVD widened workspace aliases use full complex byte widths" {
    inline for (.{ C32, C64 }) |C| {
        var arena: [64]C align(16) = @splat(.{ .re = 1, .im = 0 });
        var s: [2]svd.Wide(C) = @splat(91);
        var v: [4]C = @splat(.{ .re = 92, .im = 0 });
        const ws: [*]svd.WorkspaceScalar(C) = @ptrCast(@alignCast(&arena));
        try std.testing.expectError(error.AliasingNotAllowed, svd.decomposeWithSingularType(C, svd.Wide(C), .{ .values = arena[0..4], .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, ws[0..10], .{}));
        for (arena) |x| try same(C, z(C, 1.0, 0.0), x);
    }
}
test "complex thin SVD nonfinite real or imaginary input leaves public outputs unchanged" {
    inline for (.{ C32, C64 }) |C| for ([_]svd.Real(C){ std.math.nan(svd.Real(C)), std.math.inf(svd.Real(C)) }) |bad| for ([_]bool{ false, true }) |imaginary| {
        var a = [_]C{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = if (imaginary) 2 else bad, .im = if (imaginary) bad else 0 } };
        const bytes = std.mem.asBytes(&a).*;
        var s: [2]svd.Real(C) = @splat(91);
        var v: [4]C = @splat(.{ .re = 92, .im = 0 });
        var w: [10]svd.WorkspaceScalar(C) = undefined;
        try std.testing.expectError(error.NonFiniteInput, svd.decompose(C, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &w, .{}));
        try std.testing.expectEqualSlices(u8, &bytes, std.mem.asBytes(&a));
        for (s) |x| try std.testing.expectEqual(@as(svd.Real(C), 91), x);
        for (v) |x| try same(C, z(C, 92.0, 0.0), x);
    };
}
test "complex thin SVD widened singular output preserves values outside native real range" {
    inline for (.{ C32, C64 }) |C| {
        const maximum = std.math.floatMax(svd.Real(C));
        var a = [_]C{ .{ .re = maximum, .im = 0 }, .{ .re = maximum, .im = 0 } };
        const before = a;
        var s = [_]svd.Real(C){91};
        var v = [_]C{.{ .re = 92, .im = 0 }};
        var w: [4]svd.WorkspaceScalar(C) = undefined;
        const av: dense.Matrix(C) = .{ .values = &a, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const vv: dense.Matrix(C) = .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        const r = try svd.decompose(C, av, &s, vv, &w, .{});
        try std.testing.expectEqual(svd.Status.non_finite_result, r.status);
        for (a, before) |x, y| try same(C, x, y);
        try std.testing.expectEqual(@as(svd.Real(C), 91), s[0]);
        try same(C, z(C, 92.0, 0.0), v[0]);
        var sw = [_]svd.Wide(C){0};
        const rr = try svd.decomposeWithSingularType(C, svd.Wide(C), av, &sw, vv, &w, .{});
        try std.testing.expectEqual(svd.Status.converged, rr.status);
        try std.testing.expect(sw[0] > maximum);
        try verify(C, svd.Wide(C), &before, av, &sw, vv);
    }
}
test "complex thin SVD extreme dynamic range retains small representable singular directions" {
    inline for (.{ C32, C64 }) |C| {
        const R = svd.Real(C);
        const a = [_]C{ .{ .re = std.math.floatMax(R) / 4, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = std.math.floatTrueMin(R) } };
        try checkCase(C, R, &a, 2, 2);
        var data = a;
        var s: [2]R = undefined;
        var v: [4]C = undefined;
        var w: [10]svd.WorkspaceScalar(C) = undefined;
        const r = try svd.decompose(C, .{ .values = &data, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &w, .{});
        try std.testing.expectEqual(svd.Status.converged, r.status);
        try std.testing.expectEqual(std.math.floatTrueMin(R), s[1]);
    }
}
test "complex thin SVD subnormal coupled columns preserve collectively representable direction" {
    inline for (.{ C32, C64 }) |C| {
        const R = svd.Real(C);
        var a: [64]C = @splat(.{ .re = 0, .im = 0 });
        for (0..8) |j| a[j * 8] = if (j % 2 == 0) .{ .re = std.math.floatTrueMin(R), .im = 0 } else .{ .re = 0, .im = std.math.floatTrueMin(R) };
        var s: [8]R = undefined;
        var v: [64]C = undefined;
        var w: [136]svd.WorkspaceScalar(C) = undefined;
        const r = try svd.decompose(C, .{ .values = &a, .row_count = 8, .column_count = 8, .leading_dimension = 8 }, &s, .{ .values = &v, .row_count = 8, .column_count = 8, .leading_dimension = 8 }, &w, .{});
        try std.testing.expectEqual(svd.Status.converged, r.status);
        try std.testing.expect(s[0] >= 2 * std.math.floatTrueMin(R));
        try std.testing.expect(r.output_orthogonality <= 32 * @as(f64, std.math.floatEps(R)));
    }
}
test "complex thin SVD empty shapes and native width size overflow have deterministic contracts" {
    inline for (.{ C32, C64 }) |C| {
        try std.testing.expectEqual(@as(usize, 0), try svd.workspaceLength(C, 0, 0));
        try std.testing.expectEqual(@as(usize, 0), try svd.rectangularScratchLength(C, 0, std.math.maxInt(usize)));
        try std.testing.expectError(error.DimensionMismatch, svd.workspaceLength(C, 1, 2));
        try std.testing.expectError(error.InvalidLength, svd.workspaceLength(C, std.math.maxInt(usize), 2));
        const r = try svd.decomposeRectangular(C, .{ .values = &.{}, .row_count = 0, .column_count = 5, .leading_dimension = 0 }, &.{}, .{ .values = &.{}, .row_count = 5, .column_count = 0, .leading_dimension = 0 }, &.{}, &.{}, .{});
        try std.testing.expectEqual(svd.Status.converged, r.status);
    }
}
test "complex thin SVD wide invalid scratch and nonconvergence preserve trailing input columns" {
    inline for (.{ C32, C64 }) |C| {
        var a = [_]C{ z(C, 1.0, 1.0), z(C, 2.0, -1.0), z(C, 3.0, 2.0), z(C, 4.0, -1.0), z(C, 2.0, 1.0), z(C, -1.0, 3.0) };
        const before = a;
        var s: [2]svd.Real(C) = @splat(91);
        var v: [6]C = @splat(.{ .re = 92, .im = 0 });
        var w: [12]svd.WorkspaceScalar(C) = undefined;
        var scratch: [10]C = undefined;
        const av: dense.Matrix(C) = .{ .values = &a, .row_count = 2, .column_count = 3, .leading_dimension = 2 };
        const vv: dense.Matrix(C) = .{ .values = &v, .row_count = 3, .column_count = 2, .leading_dimension = 3 };
        try std.testing.expectError(error.BufferTooSmall, svd.decomposeRectangular(C, av, &s, vv, &w, scratch[0..9], .{}));
        const r = try svd.decomposeRectangular(C, av, &s, vv, &w, &scratch, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(svd.Status.no_convergence, r.status);
        for (a, before) |x, y| try same(C, x, y);
        for (s) |x| try std.testing.expectEqual(@as(svd.Real(C), 91), x);
        for (v) |x| try same(C, z(C, 92.0, 0.0), x);
    }
}

test "complex thin SVD unattainable rounded orthogonality tolerance is transactional" {
    inline for (.{ C32, C64 }) |C| {
        var a = [_]C{ z(C, 1.0, 1.0), z(C, 2.0, 3.0) };
        const before = a;
        var s = [_]svd.Real(C){91};
        var v = [_]C{.{ .re = 92, .im = -7 }};
        var w: [4]svd.WorkspaceScalar(C) = undefined;
        const r = try svd.decompose(C, .{ .values = &a, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, &s, .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, &w, .{ .relative_tolerance = 1e-30 });
        try std.testing.expectEqual(svd.Status.range_limit, r.status);
        try std.testing.expect(r.output_orthogonality > 1e-30);
        for (a, before) |x, y| try same(C, x, y);
        try std.testing.expectEqual(@as(svd.Real(C), 91), s[0]);
        try same(C, z(C, 92.0, -7.0), v[0]);
    }
}
test "complex thin SVD wide transpose scratch and widened singular work aliases are rejected" {
    inline for (.{ C32, C64 }) |C| {
        var arena: [12]C = @splat(.{ .re = 1, .im = 0 });
        var s: [2]svd.Wide(C) = @splat(91);
        var v: [6]C = @splat(.{ .re = 92, .im = 0 });
        var w: [12]svd.WorkspaceScalar(C) = undefined;
        const av: dense.Matrix(C) = .{ .values = arena[0..6], .row_count = 2, .column_count = 3, .leading_dimension = 2 };
        const vv: dense.Matrix(C) = .{ .values = &v, .row_count = 3, .column_count = 2, .leading_dimension = 3 };
        try std.testing.expectError(error.AliasingNotAllowed, svd.decomposeRectangularWithSingularType(C, svd.Wide(C), av, &s, vv, &w, arena[0..10], .{}));
        var scratch: [10]C = undefined;
        const shared: [*]svd.Wide(C) = @ptrCast(&w);
        try std.testing.expectError(error.AliasingNotAllowed, svd.decomposeRectangularWithSingularType(C, svd.Wide(C), av, shared[0..2], vv, &w, &scratch, .{}));
        for (arena) |x| try same(C, z(C, 1.0, 0.0), x);
        for (s) |x| try std.testing.expectEqual(@as(svd.Wide(C), 91), x);
        for (v) |x| try same(C, z(C, 92.0, 0.0), x);
    }
}
test "complex thin SVD active finite checks ignore nonfinite matrix padding and output old contents" {
    inline for (.{ C32, C64 }) |C| {
        const nan = std.math.nan(svd.Real(C));
        var a = [_]C{ .{ .re = 0, .im = 2 }, .{ .re = 0, .im = 0 }, .{ .re = nan, .im = nan } };
        var s = [_]svd.Real(C){ nan, nan };
        var v = [_]C{ .{ .re = nan, .im = nan }, .{ .re = nan, .im = nan } };
        var w: [5]svd.WorkspaceScalar(C) = @splat(.{ .re = std.math.nan(svd.Wide(C)), .im = std.math.nan(svd.Wide(C)) });
        const r = try svd.decompose(C, .{ .values = &a, .row_count = 2, .column_count = 1, .leading_dimension = 3 }, s[0..1], .{ .values = &v, .row_count = 1, .column_count = 1, .leading_dimension = 2 }, &w, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(svd.Status.converged, r.status);
        try std.testing.expectEqual(@as(svd.Real(C), 2), s[0]);
        try std.testing.expect(std.math.isNan(a[2].re) and std.math.isNan(a[2].im));
        try std.testing.expect(std.math.isNan(s[1]) and std.math.isNan(v[1].re));
        try std.testing.expect(std.math.isNan(w[4].re));
        try same(C, z(C, 0.0, 1.0), a[0]);
    }
}

fn wideSpectrum(comptime C: type, comptime m: usize, comptime n: usize, input: [m * n]C, expected_rank: usize) ![n]svd.Wide(C) {
    var a = input;
    var s: [n]svd.Wide(C) = undefined;
    var v: [n * n]C = undefined;
    var work: [m * n + n * n + n]svd.WorkspaceScalar(C) = undefined;
    const av: dense.Matrix(C) = .{ .values = &a, .row_count = m, .column_count = n, .leading_dimension = m };
    const vv: dense.Matrix(C) = .{ .values = &v, .row_count = n, .column_count = n, .leading_dimension = n };
    const result = try svd.decomposeWithSingularType(C, svd.Wide(C), av, &s, vv, &work, .{});
    try std.testing.expectEqual(svd.Status.converged, result.status);
    for (s, 0..) |value, i| if (i < expected_rank) {
        try std.testing.expect(value > 0);
    } else try std.testing.expectEqual(@as(svd.Wide(C), 0), value);
    try verify(C, svd.Wide(C), &input, av, &s, vv);
    return s;
}
test "complex widened singular SVD proves native phase proportional nullspaces" {
    inline for (.{ C32, C64 }) |C| {
        _ = try wideSpectrum(C, 3, 2, .{ z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 0.0), z(C, 0.0, 2.0), z(C, -2.0, 0.0), z(C, 0.0, 0.0) }, 1);
        _ = try wideSpectrum(C, 3, 2, .{ z(C, 1.0, 1.0), z(C, 2.0, -1.0), z(C, -1.0, 3.0), z(C, -1.0, 3.0), z(C, 4.0, 3.0), z(C, -7.0, 1.0) }, 1);
    }
}
test "complex widened singular SVD proves general three column linear dependency" {
    inline for (.{ C32, C64 }) |C| {
        const input = [_]C{ z(C, 1.0, 1.0), z(C, 2.0, 0.0), z(C, 0.0, 1.0), z(C, 1.0, -1.0), z(C, 2.0, -1.0), z(C, -1.0, 2.0), z(C, 1.0, 0.0), z(C, 0.0, 2.0), z(C, 5.0, -1.0), z(C, 0.0, 4.0), z(C, 2.0, 1.0), z(C, 1.0, 3.0) };
        _ = try wideSpectrum(C, 4, 3, input, 2);
        // Conjugate transpose exercises the rectangular path and exact source
        // dependency among the original matrix rows.
        var transpose: [12]C = undefined;
        for (0..4) |j| for (0..3) |i| {
            const value = input[j + i * 4];
            transpose[i + j * 3] = .{ .re = value.re, .im = -value.im };
        };
        try checkCase(C, svd.Wide(C), &transpose, 3, 4);
    }
}
test "complex widened singular SVD keeps tiny independent direction beside proven nullspace" {
    inline for (.{ C32, C64 }) |C| {
        const tiny = std.math.floatTrueMin(svd.Real(C));
        const s = try wideSpectrum(C, 3, 3, .{ z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 0.0), z(C, 0.0, 2.0), z(C, -2.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), .{ .re = tiny, .im = tiny } }, 2);
        try std.testing.expectApproxEqAbs(@sqrt(@as(svd.Wide(C), 2)) * tiny, s[1], 8 * std.math.floatEps(svd.Wide(C)) * tiny);
    }
}
test "complex widened singular SVD distinguishes exact zero from mixed exponent source relation" {
    inline for (.{ C32, C64 }) |C| {
        const tiny = std.math.floatTrueMin(svd.Real(C));
        const input = [_]C{ z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), .{ .re = tiny, .im = 0 }, z(C, 0.0, 0.0), z(C, 1.0, 0.0), .{ .re = tiny, .im = 0 }, z(C, 0.0, 0.0) };
        _ = try wideSpectrum(C, 3, 3, input, 2);
    }
}
test "complex widened singular SVD proves homogeneous scaled rational dependencies" {
    inline for (.{ C32, C64 }) |C| {
        _ = try wideSpectrum(C, 3, 2, .{ z(C, 3.0, 0.0), z(C, 0.0, 3.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 0.0) }, 1);
        _ = try wideSpectrum(C, 3, 3, .{ z(C, 3.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 3.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0) }, 2);
    }
}
test "complex widened singular SVD does not certify phase relation with subnormal independent residual" {
    inline for (.{ C32, C64 }) |C| {
        const tiny = std.math.floatTrueMin(svd.Real(C));
        const s = try wideSpectrum(C, 3, 2, .{ z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 0.0), z(C, 0.0, 2.0), z(C, -2.0, 0.0), .{ .re = tiny, .im = 0 } }, 2);
        try std.testing.expectApproxEqAbs(@as(svd.Wide(C), tiny) / @sqrt(@as(svd.Wide(C), 5)), s[1], 64 * std.math.floatEps(svd.Wide(C)) * tiny);
    }
}
test "complex widened singular SVD source certificates respect zero sweep budget" {
    inline for (.{ C32, C64 }) |C| {
        var a = [_]C{ z(C, 1.0, 0.0), z(C, 0.0, 1.0), z(C, 0.0, 0.0), z(C, 0.0, 2.0), z(C, -2.0, 0.0), z(C, 0.0, 0.0) };
        const before = a;
        var s: [2]svd.Wide(C) = @splat(91);
        var v: [4]C = @splat(.{ .re = 92, .im = -7 });
        var work: [12]svd.WorkspaceScalar(C) = undefined;
        const result = try svd.decomposeWithSingularType(C, svd.Wide(C), .{ .values = &a, .row_count = 3, .column_count = 2, .leading_dimension = 3 }, &s, .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, &work, .{ .max_sweeps = 0 });
        try std.testing.expectEqual(svd.Status.no_convergence, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.sweeps);
        for (a, before) |x, y| try same(C, x, y);
        for (s) |value| try std.testing.expectEqual(@as(svd.Wide(C), 91), value);
        for (v) |value| try same(C, z(C, 92.0, -7.0), value);
    }
}
test "complex widened singular SVD certifies rational complex phase through homogeneous source scaling" {
    inline for (.{ C32, C64 }) |C| {
        // Ratio (1+i)/(3+2i)=(5+i)/13 is not a finite binary scalar.
        _ = try wideSpectrum(C, 3, 2, .{ z(C, 3.0, 2.0), z(C, 6.0, 4.0), z(C, 0.0, 0.0), z(C, 1.0, 1.0), z(C, 2.0, 2.0), z(C, 0.0, 0.0) }, 1);
    }
}
test "complex widened singular SVD retains subnormal independent residual in general three column relation" {
    inline for (.{ C32, C64 }) |C| {
        const tiny = std.math.floatTrueMin(svd.Real(C));
        const s = try wideSpectrum(C, 3, 3, .{ z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 1.0, 0.0), .{ .re = tiny, .im = 0 } }, 3);
        try std.testing.expectApproxEqAbs(@as(svd.Wide(C), tiny) / @sqrt(@as(svd.Wide(C), 3)), s[2], 64 * std.math.floatEps(svd.Wide(C)) * tiny);
    }
}
test "complex widened singular SVD certifies rational multi basis dependency without singular threshold" {
    inline for (.{ C32, C64 }) |C| {
        // First two columns have determinant 3. Coefficients 2/3,-1/3
        // cannot prove a binary relation with target coefficient -1.
        _ = try wideSpectrum(C, 3, 3, .{ z(C, 2.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 2.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 0.0, 0.0) }, 2);
        // A minimum residual rejects that same exact source relation, keeping
        // its genuinely independent third direction nonzero in Wide.
        const tiny = std.math.floatTrueMin(svd.Real(C));
        const s = try wideSpectrum(C, 3, 3, .{ z(C, 2.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 2.0, 0.0), z(C, 0.0, 0.0), z(C, 1.0, 0.0), z(C, 0.0, 0.0), .{ .re = tiny, .im = 0 } }, 3);
        // Allow accumulated Wide rotation roundoff, while requiring relative
        // accuracy orders tighter than the native input precision.
        try std.testing.expectApproxEqAbs(@as(svd.Wide(C), tiny) * 3 / @sqrt(@as(svd.Wide(C), 14)), s[2], 1024 * std.math.floatEps(svd.Wide(C)) * tiny);
    }
}
