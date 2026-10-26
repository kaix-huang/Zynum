// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("zynum-core");
const leaf = owner.internal.numerics.OrthonormalCompletion;
const dense = owner.internal.numerics.Dense;
const real_svd = owner.internal.numerics.Svd;
const complex_svd = owner.internal.numerics.ComplexSvd;
const C32 = owner.ComplexF32;
const C64 = owner.ComplexF64;
fn complex(comptime T: type) bool {
    return T == C32 or T == C64;
}
fn Real(comptime T: type) type {
    return if (T == f32 or T == C32) f32 else f64;
}
fn z(comptime T: type, re: f64, im: f64) T {
    return if (comptime complex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn w(comptime T: type, re: f64, im: f64) leaf.WorkspaceScalar(T) {
    return if (comptime complex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn same(a: anytype, b: @TypeOf(a)) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
// An independent direct f128 Gram calculation on stored output; it neither
// calls the completion metric nor compares individual arbitrary complement axes.
fn verify(comptime T: type, basis: dense.Matrix(T)) !void {
    const tol: f128 = 32 * @as(f128, std.math.floatEps(Real(T)));
    for (0..basis.column_count) |p| for (0..basis.column_count) |q| {
        var re: f128 = 0;
        var im: f128 = 0;
        for (0..basis.row_count) |i| {
            const a = basis.values[i + p * basis.leading_dimension];
            const b = basis.values[i + q * basis.leading_dimension];
            if (comptime complex(T)) {
                re += @as(f128, a.re) * b.re + @as(f128, a.im) * b.im;
                im += @as(f128, a.re) * b.im - @as(f128, a.im) * b.re;
            } else re += @as(f128, a) * b;
        }
        try std.testing.expect(@abs(re - @as(f128, if (p == q) 1 else 0)) <= tol);
        try std.testing.expect(@abs(im) <= tol);
    };
}
fn finishCase(comptime T: type, known: usize, phase: bool) !void {
    var data: [27]T = @splat(z(T, std.math.nan(Real(T)), -17));
    const d = 4;
    const ld = 6;
    for (0..known) |j| for (0..d) |i| {
        data[i + j * ld] = if (known == 1) z(T, if (phase and complex(T)) 0 else 0.5, if (phase and complex(T)) 0.5 else 0) else z(T, if (i == j) (if (phase and complex(T)) 0 else 1) else -0.0, if (phase and complex(T) and i == j) 1 else -0.0);
    };
    const saved = data;
    var work: [22]leaf.WorkspaceScalar(T) = @splat(w(T, 73, -23));
    const saved_work = work;
    const basis: dense.Matrix(T) = .{ .values = &data, .row_count = d, .column_count = d, .leading_dimension = ld };
    const result = try leaf.complete(T, basis, known, &work, .{});
    try std.testing.expectEqual(real_svd.Status.converged, result.status);
    try std.testing.expect(result.output_orthogonality <= 32 * @as(f64, std.math.floatEps(Real(T))));
    try verify(T, basis);
    for (0..known) |j| try same(saved[j * ld ..][0..d], data[j * ld ..][0..d]);
    for (0..d) |j| try same(saved[j * ld + d ..][0 .. ld - d], data[j * ld + d ..][0 .. ld - d]);
    try same(saved[24..], data[24..]);
    try same(saved_work[20..], work[20..]);
}
test "orthonormal completion four scalars empty unknown full and phased known columns preserve bits padding and suffixes" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        inline for (.{ 0, 1, 4 }) |k| inline for (.{ false, true }) |phase| try finishCase(T, k, phase);
        var empty_data = [_]T{z(T, std.math.nan(Real(T)), 13)};
        var empty_work = [_]leaf.WorkspaceScalar(T){w(T, 77, -11)};
        const saved = empty_data;
        const saved_w = empty_work;
        const result = try leaf.complete(T, .{ .values = &empty_data, .row_count = 0, .column_count = 0, .leading_dimension = std.math.maxInt(usize) }, 0, &empty_work, .{});
        try std.testing.expectEqual(real_svd.Status.converged, result.status);
        try std.testing.expectEqual(@as(f64, 0), result.output_orthogonality);
        try same(saved[0..], empty_data[0..]);
        try same(saved_w[0..], empty_work[0..]);
        try std.testing.expectEqual(@as(usize, 0), try leaf.workspaceLength(T, 0));
        try std.testing.expectEqual(@as(usize, 20), try leaf.workspaceLength(T, 4));
    }
}
fn thinCase(comptime T: type, wide: bool, kind: usize) !void {
    var source: [8]T = undefined;
    const m: usize = if (wide) 2 else 4;
    const n: usize = if (wide) 4 else 2;
    const ordinary = [_]f64{ 1, 2, 0, 1, 0, 1, 2, -1 };
    var tall: [8]T = undefined;
    for (0..2) |j| for (0..4) |i| {
        const base = ordinary[i + j * 4];
        tall[i + j * 4] = switch (kind) {
            0 => z(T, base, if ((i + j) % 2 == 0) base * 0.5 else 0),
            1 => z(T, if (i == j) 1 else 0, 0),
            2 => if (j == 0) z(T, ordinary[i], 0) else z(T, if (comptime complex(T)) 0 else 2 * ordinary[i], 2 * ordinary[i]),
            else => z(T, 0, 0),
        };
    };
    for (0..n) |j| for (0..m) |i| {
        const a = tall[if (wide) j + i * 4 else i + j * 4];
        source[i + j * m] = if (comptime complex(T)) (if (wide) z(T, a.re, -a.im) else a) else a;
    };
    var right: [8]T = undefined;
    var singular: [2]Real(T) = undefined;
    var svd_work: [14]leaf.WorkspaceScalar(T) = undefined;
    var transpose: [12]T = undefined;
    const a: dense.Matrix(T) = .{ .values = &source, .row_count = m, .column_count = n, .leading_dimension = m };
    const v: dense.Matrix(T) = .{ .values = &right, .row_count = n, .column_count = 2, .leading_dimension = n };
    const result = if (comptime complex(T)) try complex_svd.decomposeRectangular(T, a, &singular, v, &svd_work, &transpose, .{}) else try real_svd.decomposeRectangular(T, a, &singular, v, &svd_work, &transpose, .{});
    try std.testing.expectEqual(real_svd.Status.converged, result.status);
    var data: [25]T = @splat(z(T, std.math.nan(Real(T)), 71));
    for (0..2) |j| {
        for (0..4) |i| data[i + j * 6] = if (wide) right[i + j * 4] else source[i + j * 4];
    }
    const saved = data;
    var work: [21]leaf.WorkspaceScalar(T) = @splat(w(T, 99, -99));
    const saved_w = work;
    const basis: dense.Matrix(T) = .{ .values = &data, .row_count = 4, .column_count = 4, .leading_dimension = 6 };
    const completed = try leaf.complete(T, basis, 2, &work, .{});
    try std.testing.expectEqual(real_svd.Status.converged, completed.status);
    try verify(T, basis);
    for (0..2) |j| try same(saved[j * 6 ..][0..4], data[j * 6 ..][0..4]);
    for (0..4) |j| try same(saved[j * 6 + 4 ..][0..2], data[j * 6 + 4 ..][0..2]);
    try same(saved[24..], data[24..]);
    try same(saved_w[20..], work[20..]);
}
test "orthonormal completion uses actual thin SVD tall and wide full rank repeated zero and phase rank deficient subspaces" {
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ false, true }) |wide| {
        for (0..4) |kind| try thinCase(T, wide, kind);
    };
}
test "orthonormal completion rejects geometry capacity nonfinite known and invalid tolerance with transactional numerical failure" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        var data: [16]T = @splat(z(T, 77, 19));
        var work: [21]leaf.WorkspaceScalar(T) = @splat(w(T, 91, 13));
        for (0..4) |i| data[i] = z(T, 0.5, 0);
        const saved = data;
        const saved_w = work;
        var basis: dense.Matrix(T) = .{ .values = &data, .row_count = 4, .column_count = 4, .leading_dimension = 4 };
        const reject = try leaf.complete(T, basis, 1, &work, .{ .relative_tolerance = 1e-40 });
        try std.testing.expectEqual(real_svd.Status.range_limit, reject.status);
        try same(saved[0..], data[0..]);
        try same(saved_w[20..], work[20..]);
        inline for (.{ 0.0, -1.0, 1.0, std.math.inf(f64), std.math.nan(f64) }) |tol| {
            try std.testing.expectError(error.InvalidTolerance, leaf.complete(T, basis, 1, &work, .{ .relative_tolerance = tol }));
            try same(saved[0..], data[0..]);
        }
        try std.testing.expectError(error.DimensionMismatch, leaf.complete(T, basis, 5, &work, .{}));
        basis.column_count = 3;
        try std.testing.expectError(error.DimensionMismatch, leaf.complete(T, basis, 0, &work, .{}));
        basis.column_count = 4;
        basis.leading_dimension = 3;
        try std.testing.expectError(error.InvalidLeadingDimension, leaf.complete(T, basis, 0, &work, .{}));
        basis.leading_dimension = 4;
        basis.values = data[0..15];
        try std.testing.expectError(error.BufferTooSmall, leaf.complete(T, basis, 0, &work, .{}));
        basis.values = &data;
        try std.testing.expectError(error.BufferTooSmall, leaf.complete(T, basis, 0, work[0..19], .{}));
        try std.testing.expectError(error.InvalidLength, leaf.workspaceLength(T, std.math.maxInt(usize)));
        const byte_overflow_dimension = @as(usize, 1) << (@bitSizeOf(usize) / 2 - 1);
        try std.testing.expectError(error.InvalidLength, leaf.workspaceLength(T, byte_overflow_dimension));
        basis.leading_dimension = std.math.maxInt(usize);
        try std.testing.expectError(error.InvalidLength, leaf.complete(T, basis, 0, &work, .{}));
        basis.leading_dimension = 4;
        try same(saved[0..], data[0..]);
        inline for (.{ std.math.nan(Real(T)), std.math.inf(Real(T)) }) |bad| {
            data[0] = z(T, bad, 0);
            const invalid = data;
            try std.testing.expectError(error.NonFiniteInput, leaf.complete(T, basis, 1, &work, .{}));
            try same(invalid[0..], data[0..]);
        }
        if (comptime complex(T)) {
            data[0] = z(T, 0, std.math.nan(Real(T)));
            const invalid = data;
            try std.testing.expectError(error.NonFiniteInput, leaf.complete(T, basis, 1, &work, .{}));
            try same(invalid[0..], data[0..]);
        }
        inline for (.{ 0.0, 2.0, std.math.floatMax(Real(T)) }) |bad| {
            data[0] = z(T, bad, 0);
            const invalid = data;
            const failure = try leaf.complete(T, basis, 1, &work, .{});
            try std.testing.expectEqual(real_svd.Status.range_limit, failure.status);
            try same(invalid[0..], data[0..]);
        }
        for (0..4) |i| {
            data[i] = z(T, if (i == 0) 1 else 0, 0);
            data[4 + i] = data[i];
        }
        const repeated = data;
        const failure = try leaf.complete(T, basis, 2, &work, .{});
        try std.testing.expectEqual(real_svd.Status.range_limit, failure.status);
        try same(repeated[0..], data[0..]);
    }
}
test "orthonormal completion checks mixed byte active spans while allowing unused workspace tail to alias another active basis" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        const W = leaf.WorkspaceScalar(T);
        var arena: [10]W = @splat(w(T, 71, -17));
        const raw: [*]T = @ptrCast(@alignCast(&arena[0]));
        const overlap: dense.Matrix(T) = .{ .values = raw[0..4], .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const saved = arena;
        try std.testing.expectError(error.AliasingNotAllowed, leaf.complete(T, overlap, 0, arena[0..6], .{}));
        try same(saved[0..], arena[0..]);
        const tail: [*]T = @ptrCast(@alignCast(&arena[6]));
        const separate: dense.Matrix(T) = .{ .values = tail[0..4], .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const done = try leaf.complete(T, separate, 0, &arena, .{});
        try std.testing.expectEqual(real_svd.Status.converged, done.status);
        try verify(T, separate);
        // Tail storage is also the explicit basis operand; the workspace view
        // does not use it. This overlap of unused capacity must stay legal.
        const after = arena;
        _ = try leaf.complete(T, separate, 2, &arena, .{});
        try same(after[6..], arena[6..]);
        const fake_basis_pointer: [*]T = @ptrFromInt(std.math.maxInt(usize) - (@alignOf(T) - 1));
        const fake_basis: dense.Matrix(T) = .{ .values = fake_basis_pointer[0..4], .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        try std.testing.expectError(error.InvalidLength, leaf.complete(T, fake_basis, 0, arena[0..6], .{}));
        const fake_work_pointer: [*]W = @ptrFromInt(std.math.maxInt(usize) - (@alignOf(W) - 1));
        try std.testing.expectError(error.InvalidLength, leaf.complete(T, separate, 0, fake_work_pointer[0..6], .{}));
    }
}
