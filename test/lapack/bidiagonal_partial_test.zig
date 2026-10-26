// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const S = N.Scalar;
const Z = struct { re: f128, im: f128 };
fn impl(comptime T: type) type {
    return if (comptime S.isComplex(T)) N.ComplexBidiagonalSvd else N.BidiagonalSvd;
}
fn v(comptime T: type, re: f128, im: f128) N.SvdLeastSquares.WorkspaceScalar(T) {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn z(comptime T: type, x: N.SvdLeastSquares.WorkspaceScalar(T)) Z {
    return if (comptime S.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x, .im = 0 };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn cj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn norm(a: Z) f128 {
    return std.math.hypot(a.re, a.im);
}
fn entry(comptime T: type, state: impl(T).Factors(T), i: usize, j: usize) Z {
    var sum: Z = .{ .re = 0, .im = 0 };
    for (0..state.left.column_count) |p| for (0..state.right.column_count) |q| {
        var b: Z = .{ .re = 0, .im = 0 };
        if (state.adjoint_reduction) {
            if (q < state.b_tall.row_count and p < state.b_tall.column_count) b = cj(z(T, state.b_tall.values[q + p * state.b_tall.leading_dimension]));
        } else if (p < state.b_tall.row_count and q < state.b_tall.column_count) b = z(T, state.b_tall.values[p + q * state.b_tall.leading_dimension]);
        sum = add(sum, mul(mul(z(T, state.left.values[i + p * state.left.leading_dimension]), b), cj(z(T, state.right.values[j + q * state.right.leading_dimension]))));
    };
    return sum;
}
test "Borrowed partial bidiagonal orientation reconstruction spectra canaries and admission" {
    var cases: usize = 0;
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const W = N.SvdLeastSquares.WorkspaceScalar(T);
        const R = N.SvdLeastSquares.Wide(T);
        for (0..5) |which| {
            const m: usize = if (which == 1) 4 else if (which == 3) 2 else 3;
            const n: usize = if (which == 2) 4 else 3;
            const k = @min(m, n);
            const p = @max(m, n);
            const ld = p + 1;
            var b: [32]W = @splat(v(T, -91, 0));
            var left: [24]W = @splat(v(T, -92, 0));
            var right: [24]W = @splat(v(T, -93, 0));
            for (0..k) |j| for (0..p) |i| {
                b[i + j * ld] = v(T, 0, 0);
            };
            for (0..m) |j| for (0..m) |i| {
                left[i + j * (m + 1)] = v(T, if (i == j) 1 else 0, 0);
            };
            for (0..n) |j| for (0..n) |i| {
                right[i + j * (n + 1)] = v(T, if (i == j) 1 else 0, 0);
            };
            if (comptime S.isComplex(T)) {
                left[1 + (m + 1)] = v(T, 0, 1);
                right[1 + (n + 1)] = v(T, 0, -1);
            }
            for (0..k) |i| b[i + i * ld] = v(T, if (which == 3) 0 else if (which == 4) (if (i == 0) -2 else if (i == 1) 4 else 0) else if (i == 0) -4 else @floatFromInt(4 - i), 0);
            if (which != 4) for (0..k - 1) |i| {
                b[i + (i + 1) * ld] = v(T, 1, 0);
            };
            var state: impl(T).Factors(T) = .{
                .b_tall = .{ .values = &b, .row_count = p, .column_count = k, .leading_dimension = ld },
                .left = .{ .values = &left, .row_count = m, .column_count = m, .leading_dimension = m + 1 },
                .right = .{ .values = &right, .row_count = n, .column_count = n, .leading_dimension = n + 1 },
                .adjoint_reduction = m < n,
            };
            var expected: [16]Z = undefined;
            var energy: f128 = 0;
            for (0..n) |j| for (0..m) |i| {
                const a = entry(T, state, i, j);
                expected[i + j * m] = a;
                energy += a.re * a.re + a.im * a.im;
            };
            const bb = b;
            const ll = left;
            const rr = right;
            const remaining = try impl(T).preparePartialUpper(T, &state);
            try std.testing.expect(!state.adjoint_reduction);
            try std.testing.expectEqual(@as(usize, if (which >= 3) 0 else 2), remaining);
            const native_tolerance = 32 * @as(f64, std.math.floatEps(S.Real(T)));
            try impl(T).checkPartialNative(T, state, m, n, native_tolerance);
            try std.testing.expectError(error.InvalidTolerance, impl(T).checkPartialNative(T, state, m, n, 0));
            try std.testing.expectError(error.DimensionMismatch, impl(T).checkPartialNative(T, state, m + 1, n, native_tolerance));
            const staged_b = b;
            const staged_left = left;
            left[0] = v(T, 2, 0);
            try std.testing.expectError(error.RangeLimit, impl(T).checkPartialNative(T, state, m, n, native_tolerance));
            left = staged_left;
            b[0] = v(T, 2 * @as(f128, std.math.floatMax(S.Real(T))), 0);
            try std.testing.expectError(error.NonFiniteResult, impl(T).checkPartialNative(T, state, m, n, native_tolerance));
            b = staged_b;
            b[ld] = v(T, @as(f128, std.math.floatTrueMin(S.Real(T))) / 2, 0);
            try std.testing.expectError(error.RangeLimit, impl(T).checkPartialNative(T, state, m, n, native_tolerance));
            b = staged_b;
            b[1] = v(T, 1, 0);
            try std.testing.expectError(error.NotBidiagonal, impl(T).checkPartialNative(T, state, m, n, native_tolerance));
            b = staged_b;
            const tolerance: f128 = 4096 * @as(f128, std.math.floatEps(R));
            for (0..n) |j| for (0..m) |i| {
                const a = entry(T, state, i, j);
                const old = expected[i + j * m];
                try std.testing.expect(norm(.{ .re = a.re - old.re, .im = a.im - old.im }) <= tolerance * 4);
            };
            var spectrum_energy: f128 = 0;
            for (0..k) |i| {
                const d = z(T, b[i + i * ld]).re;
                spectrum_energy += d * d;
            }
            for (0..k - 1) |i| {
                const e = z(T, b[i + (i + 1) * ld]).re;
                spectrum_energy += e * e;
            }
            try std.testing.expect(@abs(spectrum_energy - energy) <= tolerance * @max(1, energy));
            if (remaining == 0) for (0..k) |i| {
                try std.testing.expect(z(T, b[i + i * ld]).re >= 0);
                if (i + 1 < k) try std.testing.expect(z(T, b[i + i * ld]).re >= z(T, b[i + 1 + (i + 1) * ld]).re);
            };
            inline for (.{ state.left, state.right }) |q| for (0..q.column_count) |i| for (0..q.column_count) |j| {
                var sum: Z = .{ .re = 0, .im = 0 };
                for (0..q.row_count) |row| sum = add(sum, mul(cj(z(T, q.values[row + i * q.leading_dimension])), z(T, q.values[row + j * q.leading_dimension])));
                sum.re -= if (i == j) @as(f128, 1) else 0;
                try std.testing.expect(norm(sum) <= tolerance);
            };
            for (0..b.len) |i| if (i / ld >= k or i % ld >= p) try std.testing.expect(std.meta.eql(bb[i], b[i]));
            for (0..left.len) |i| if (i / (m + 1) >= m or i % (m + 1) >= m) try std.testing.expect(std.meta.eql(ll[i], left[i]));
            for (0..right.len) |i| if (i / (n + 1) >= n or i % (n + 1) >= n) try std.testing.expect(std.meta.eql(rr[i], right[i]));
            const after_b = b;
            const after_left = left;
            const after_right = right;
            try std.testing.expectEqual(remaining, try impl(T).preparePartialUpper(T, &state));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&after_b), std.mem.asBytes(&b));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&after_left), std.mem.asBytes(&left));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&after_right), std.mem.asBytes(&right));
            var bad = state;
            bad.left.column_count -= 1;
            try std.testing.expectError(error.DimensionMismatch, impl(T).preparePartialUpper(T, &bad));
            bad = state;
            bad.right.values = &left;
            try std.testing.expectError(error.AliasingNotAllowed, impl(T).preparePartialUpper(T, &bad));
            b[0] = v(T, std.math.inf(R), 0);
            const nonfinite = b;
            try std.testing.expectError(error.NonFiniteInput, impl(T).preparePartialUpper(T, &state));
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&nonfinite), std.mem.asBytes(&b));
            b = after_b;
            cases += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 20), cases);
    std.debug.print("PARTIAL_ORIENTATION cases={d} reconstruction_spectrum_unitarity_padding_idempotence_and_admission=true\n", .{cases});
}
