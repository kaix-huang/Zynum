// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const S = N.Scalar;
const G = N.SvdLeastSquares;
fn value(comptime T: type, re: f64, im: f64) T {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn wide(comptime T: type, re: f64) G.WorkspaceScalar(T) {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = 0 } else re;
}
fn real(comptime T: type, v: T) f128 {
    return if (comptime S.isComplex(T)) v.re else v;
}
fn imag(comptime T: type, v: T) f128 {
    return if (comptime S.isComplex(T)) v.im else 0;
}
fn close(comptime T: type, got: T, re: f64, im: f64) !void {
    const tol: f128 = 256 * @as(f128, std.math.floatEps(S.Real(T)));
    try std.testing.expect(@abs(real(T, got) - re) <= tol);
    try std.testing.expect(@abs(imag(T, got) - (if (S.isComplex(T)) im else 0)) <= tol);
}
test "Borrowed SVD math independently known solutions spectra rank residuals immutable inputs and scratch suffixes" {
    var configs: usize = 0;
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        inline for (.{ .{ 3, 2 }, .{ 2, 3 }, .{ 2, 2 } }) |shape| {
            const m: usize = shape[0];
            const n: usize = shape[1];
            const k = 2;
            const lda = m + 1;
            const ldx = n + 1;
            for ([_]bool{ false, true }) |deficient| for ([_]f64{ 0, 0.75, 1, 1.5 }) |cut| {
                var a: [24]T = @splat(value(T, -72, 0));
                var b: [12]T = @splat(value(T, -73, 0));
                for (0..n) |j| for (0..m) |i| {
                    a[i + j * lda] = value(T, 0, 0);
                };
                a[0] = value(T, 4, 0);
                a[1 + lda] = value(T, if (deficient) 0 else 2, 0);
                b[0] = value(T, 8, 4);
                b[1] = value(T, 6, -2);
                if (m == 3) b[2] = value(T, 5, 3);
                const before_a = a;
                const before_b = b;
                var x: [8]T = @splat(value(T, -81, 0));
                var sigma: [4]S.Real(T) = @splat(-82);
                var v: [16]T = @splat(value(T, -83, 0));
                var residual: [4]T = @splat(value(T, -84, 0));
                const p = try G.plan(T, m, n, 1);
                const scalar_work = try std.testing.allocator.alloc(T, p.scalar_elements + 2);
                defer std.testing.allocator.free(scalar_work);
                const computational = try std.testing.allocator.alloc(G.WorkspaceScalar(T), p.computational_elements + 2);
                defer std.testing.allocator.free(computational);
                const real_work = try std.testing.allocator.alloc(G.Wide(T), p.real_elements + 2);
                defer std.testing.allocator.free(real_work);
                @memset(scalar_work, value(T, -91, 0));
                @memset(computational, wide(T, -92));
                @memset(real_work, -93);
                const result = try G.solveBorrowed(T, .{ .values = &a, .row_count = m, .column_count = n, .leading_dimension = lda }, .{ .values = &b, .row_count = m, .column_count = 1, .leading_dimension = m }, .{ .solution = .{ .values = &x, .row_count = n, .column_count = 1, .leading_dimension = ldx }, .singular = &sigma, .right_vectors = .{ .values = &v, .row_count = n, .column_count = k, .leading_dimension = ldx }, .residual = .{ .values = &residual, .row_count = m -| n, .column_count = 1, .leading_dimension = m -| n } }, .{ .relative = cut, .absolute = 0 }, .{}, .{ .scalar = scalar_work, .computational = computational, .real = real_work });
                const rank: usize = if (cut >= 1) 0 else if (cut > 0.5 or deficient) 1 else 2;
                try std.testing.expectEqual(G.Status.converged, result.status);
                try std.testing.expectEqual(rank, result.rank);
                try close(T, x[0], if (rank > 0) 2 else 0, if (rank > 0) 1 else 0);
                try close(T, x[1], if (rank > 1) 3 else 0, if (rank > 1) -1 else 0);
                if (n == 3) try close(T, x[2], 0, 0);
                for (x[n..]) |z| try close(T, z, -81, 0);
                try std.testing.expectEqual(@as(S.Real(T), 4), sigma[0]);
                try std.testing.expectEqual(@as(S.Real(T), if (deficient) 0 else 2), sigma[1]);
                try std.testing.expectEqual(@as(S.Real(T), -82), sigma[2]);
                try std.testing.expectEqual(@as(S.Real(T), -82), sigma[3]);
                if (m > n and rank == n) {
                    const norm = std.math.hypot(real(T, residual[0]), imag(T, residual[0]));
                    const expected = std.math.sqrt(@as(f128, if (S.isComplex(T)) 34 else 25));
                    try std.testing.expect(@abs(norm - expected) <= 256 * @as(f128, std.math.floatEps(S.Real(T))));
                } else for (residual) |z| try close(T, z, -84, 0);
                for (0..k) |j| for (0..k) |c| {
                    var re: f128 = 0;
                    var im: f128 = 0;
                    for (0..n) |i| {
                        const l = v[i + j * ldx];
                        const r = v[i + c * ldx];
                        re += real(T, l) * real(T, r) + imag(T, l) * imag(T, r);
                        im += real(T, l) * imag(T, r) - imag(T, l) * real(T, r);
                    }
                    try std.testing.expect(@abs(re - @as(f128, if (j == c) 1 else 0)) <= 256 * @as(f128, std.math.floatEps(S.Real(T))));
                    try std.testing.expect(@abs(im) <= 256 * @as(f128, std.math.floatEps(S.Real(T))));
                };
                for (0..v.len) |i| {
                    const active = i / ldx < k and i % ldx < n;
                    if (!active) try close(T, v[i], -83, 0);
                }
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_a), std.mem.asBytes(&a));
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before_b), std.mem.asBytes(&b));
                for (scalar_work[p.scalar_elements..]) |z| try close(T, z, -91, 0);
                for (computational[p.computational_elements..]) |z| try std.testing.expect(std.meta.eql(z, wide(T, -92)));
                for (real_work[p.real_elements..]) |z| try std.testing.expectEqual(@as(G.Wide(T), -93), z);
                configs += 1;
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 96), configs);
    std.debug.print("BORROWED_SVD_ANALYTIC configurations={d} input_and_padding_and_scratch_suffix_checks=true\n", .{configs});
}
test "Borrowed SVD scratch admission faults preserve data before arithmetic" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const p = try G.plan(T, 2, 2, 1);
        const scalar_work = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(scalar_work);
        const computational = try std.testing.allocator.alloc(G.WorkspaceScalar(T), p.computational_elements);
        defer std.testing.allocator.free(computational);
        const real_work = try std.testing.allocator.alloc(G.Wide(T), p.real_elements);
        defer std.testing.allocator.free(real_work);
        var a = [_]T{ value(T, 4, 0), value(T, 0, 0), value(T, 0, 0), value(T, 2, 0) };
        var b = [_]T{ value(T, 8, 4), value(T, 6, -2) };
        var x: [2]T = @splat(value(T, -81, 0));
        var sigma: [2]S.Real(T) = @splat(-82);
        var v: [4]T = @splat(value(T, -83, 0));
        const av: N.Dense.ConstMatrix(T) = .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        const bv: N.Dense.ConstMatrix(T) = .{ .values = &b, .row_count = 2, .column_count = 1, .leading_dimension = 2 };
        const out: G.Outputs(T) = .{ .solution = .{ .values = &x, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .singular = &sigma, .right_vectors = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .residual = .{ .values = &.{}, .row_count = 0, .column_count = 1, .leading_dimension = 0 } };
        for (0..3) |short| {
            @memset(scalar_work, value(T, -91, 0));
            @memset(computational, wide(T, -92));
            @memset(real_work, -93);
            try std.testing.expectError(error.BufferTooSmall, G.solveBorrowed(T, av, bv, out, .{}, .{}, .{
                .scalar = scalar_work[0 .. p.scalar_elements - @as(usize, if (short == 0) 1 else 0)],
                .computational = computational[0 .. p.computational_elements - @as(usize, if (short == 1) 1 else 0)],
                .real = real_work[0 .. p.real_elements - @as(usize, if (short == 2) 1 else 0)],
            }));
            for (scalar_work) |z| try close(T, z, -91, 0);
            for (computational) |z| try std.testing.expect(std.meta.eql(z, wide(T, -92)));
            for (real_work) |z| try std.testing.expectEqual(@as(G.Wide(T), -93), z);
        }
        const alias = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(alias);
        @memset(alias, value(T, 0, 0));
        alias[0] = value(T, 4, 0);
        alias[3] = value(T, 2, 0);
        const copy = try std.testing.allocator.dupe(T, alias);
        defer std.testing.allocator.free(copy);
        try std.testing.expectError(error.AliasingNotAllowed, G.solveBorrowed(T, .{ .values = alias[0..4], .row_count = 2, .column_count = 2, .leading_dimension = 2 }, bv, out, .{}, .{}, .{ .scalar = alias, .computational = computational, .real = real_work }));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(copy), std.mem.sliceAsBytes(alias));
        try std.testing.expectError(error.AliasingNotAllowed, G.solveBorrowed(T, av, bv, .{
            .solution = .{ .values = alias[0..2], .row_count = 2, .column_count = 1, .leading_dimension = 2 },
            .singular = &sigma,
            .right_vectors = out.right_vectors,
            .residual = out.residual,
        }, .{}, .{}, .{ .scalar = alias, .computational = computational, .real = real_work }));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(copy), std.mem.sliceAsBytes(alias));
        try std.testing.expect(p.real_elements * @sizeOf(G.Wide(T)) <= p.computational_elements * @sizeOf(G.WorkspaceScalar(T)));
        const overlapping_real = @as([*]G.Wide(T), @ptrCast(computational.ptr))[0..p.real_elements];
        try std.testing.expectError(error.AliasingNotAllowed, G.solveBorrowed(T, av, bv, out, .{}, .{}, .{ .scalar = scalar_work, .computational = computational, .real = overlapping_real }));
        for (x) |z| try close(T, z, -81, 0);
        for (sigma) |z| try std.testing.expectEqual(@as(S.Real(T), -82), z);
        for (v) |z| try close(T, z, -83, 0);
    }
}
test "Borrowed SVD failed factorization retains actual two edges and commits no outputs" {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        var a = [_]T{ value(T, 4, 0), value(T, 0, 0), value(T, 0, 0), value(T, 1, 0), value(T, 3, 0), value(T, 0, 0), value(T, 0, 0), value(T, 1, 0), value(T, 2, 0) };
        var b = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0) };
        const aa = a;
        const bb = b;
        var x: [3]T = @splat(value(T, -81, 0));
        var sigma: [3]S.Real(T) = @splat(-82);
        var v: [9]T = @splat(value(T, -83, 0));
        const p = try G.plan(T, 3, 3, 1);
        const sw = try std.testing.allocator.alloc(T, p.scalar_elements);
        defer std.testing.allocator.free(sw);
        const cw = try std.testing.allocator.alloc(G.WorkspaceScalar(T), p.computational_elements);
        defer std.testing.allocator.free(cw);
        const rw = try std.testing.allocator.alloc(G.Wide(T), p.real_elements);
        defer std.testing.allocator.free(rw);
        const result = try G.solveBorrowed(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .{ .values = &b, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, .{ .solution = .{ .values = &x, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, .singular = &sigma, .right_vectors = .{ .values = &v, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .residual = .{ .values = &.{}, .row_count = 0, .column_count = 1, .leading_dimension = 0 } }, .{}, .{ .max_steps = 0 }, .{ .scalar = sw, .computational = cw, .real = rw });
        try std.testing.expectEqual(G.Status.no_convergence, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.unconverged);
        try std.testing.expectEqual(G.Stage.factorization, result.stage);
        for (x) |z| try close(T, z, -81, 0);
        for (sigma) |z| try std.testing.expectEqual(@as(S.Real(T), -82), z);
        for (v) |z| try close(T, z, -83, 0);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&aa), std.mem.asBytes(&a));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&bb), std.mem.asBytes(&b));
    }
}
