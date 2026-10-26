// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const qr = @import("zynum-lapack");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;
const sentinel = -987.25;

const query_boundaries = [_]i32{ 16_777_215, 16_777_216, 16_777_217, 16_777_218, 16_777_219, 33_554_433, 2_147_483_647 };
fn expectQueryMinimum(comptime T: type, value: T, minimum: i32) !void {
    const actual = if (T == f32 or T == f64) value else value.re;
    try expect(std.math.isFinite(actual));
    try expect(@as(f64, actual) >= @as(f64, @floatFromInt(minimum)));
    if (T == f64 or T == qr.ComplexF64) try expectEqual(@as(f64, @floatFromInt(minimum)), actual);
    if (T == f32 or T == qr.ComplexF32) {
        if (minimum == 16_777_217) try expectEqual(@as(f32, 16_777_218), actual);
        if (minimum == 33_554_433) try expectEqual(@as(f32, 33_554_436), actual);
    }
    if (T == qr.ComplexF32 or T == qr.ComplexF64) try expectEqual(@as(@TypeOf(value.im), 0), value.im);
}

fn matrix(comptime T: type, values: []T, m: usize, n: usize, ld: usize) qr.Matrix(T) {
    return .{ .values = values, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}

fn tolerance(comptime T: type) f64 {
    return if (T == f32) 4e-6 else 3e-13;
}

fn near(expected: f64, actual: f64, tol: f64) !void {
    try expect(std.math.isFinite(actual));
    try std.testing.expectApproxEqAbs(expected, actual, tol);
}

fn same(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn checkPadding(comptime T: type, data: []const T, m: usize, n: usize, ld: usize) !void {
    for (0..n) |j| for (m..ld) |i| try expectEqual(@as(T, sentinel), data[i + j * ld]);
}

// Materialize Q only in tests; independently reconstruct Q*R, test both
// Q^T*Q and application to multiple RHS against explicit matrix products.
fn checkFactor(comptime T: type, comptime m: usize, comptime n: usize, original: []const T, factor: []T, tau: []T, ld: usize, reconstruction_tolerance: f64) !void {
    const qld = m + 2;
    var q_data: [qld * m]T = @splat(sentinel);
    for (0..m) |j| for (0..m) |i| {
        q_data[i + j * qld] = if (i == j) 1 else 0;
    };
    const a = matrix(T, factor, m, n, ld).asConst();
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, a, tau, matrix(T, &q_data, m, m, qld), .normal));
    try checkPadding(T, &q_data, m, m, qld);
    for (0..m) |i| for (0..m) |j| {
        var dot: f64 = 0;
        for (0..m) |r| dot += @as(f64, @floatCast(q_data[r + i * qld])) * @as(f64, @floatCast(q_data[r + j * qld]));
        try near(if (i == j) 1 else 0, dot, tolerance(T) * @as(f64, @floatFromInt(@max(@as(usize, 1), m))));
    };
    var scale: f64 = 0;
    for (0..n) |j| for (0..m) |i| {
        scale = @max(scale, @abs(@as(f64, @floatCast(original[i + j * ld]))));
    };
    if (scale == 0) scale = 1;
    for (0..n) |j| for (0..m) |i| {
        var reconstructed: f64 = 0;
        for (0..@min(m, j + 1)) |r| reconstructed += @as(f64, @floatCast(q_data[i + r * qld])) * (@as(f64, @floatCast(factor[r + j * ld])) / scale);
        try near(@as(f64, @floatCast(original[i + j * ld])) / scale, reconstructed, reconstruction_tolerance);
    };
    const bld = m + 3;
    var rhs: [bld * 3]T = @splat(sentinel);
    for (0..3) |j| for (0..m) |i| {
        rhs[i + j * bld] = @floatCast(@as(f64, @floatFromInt(1 + i + 3 * j)) / 7);
    };
    const before = rhs;
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, a, tau, matrix(T, &rhs, m, 3, bld), .normal));
    for (0..3) |j| for (0..m) |i| {
        var expected: f64 = 0;
        for (0..m) |r| expected += @as(f64, @floatCast(q_data[i + r * qld])) * @as(f64, @floatCast(before[r + j * bld]));
        try near(expected, @floatCast(rhs[i + j * bld]), tolerance(T) * @as(f64, @floatFromInt(1 + m)));
    };
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, a, tau, matrix(T, &rhs, m, 3, bld), .transposed));
    for (0..3) |j| for (0..m) |i| try near(@floatCast(before[i + j * bld]), @floatCast(rhs[i + j * bld]), tolerance(T) * @as(f64, @floatFromInt(1 + m)));
    try checkPadding(T, &rhs, m, 3, bld);
}

fn shapeCase(comptime T: type, comptime m: usize, comptime n: usize, scale: T, tol: f64) !void {
    const ld: usize = m + 3;
    const steps: usize = @min(m, n);
    var data: [ld * n]T = @splat(sentinel);
    for (0..n) |j| for (0..m) |i| {
        const coefficient = @as(f64, @floatFromInt((i * 7 + j * 3 + 1) % 11)) / 8 - 0.5;
        data[i + j * ld] = @floatCast(@as(f64, @floatCast(scale)) * coefficient);
    };
    const original = data;
    var tau: [steps + 2]T = @splat(sentinel);
    try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &data, m, n, ld), &tau));
    try checkPadding(T, &data, m, n, ld);
    try expectEqual(@as(T, sentinel), tau[steps]);
    try expectEqual(@as(T, sentinel), tau[steps + 1]);
    const saved_factor = data;
    const saved_tau = tau;
    try checkFactor(T, m, n, &original, &data, &tau, ld, tol);
    try same(T, &saved_factor, &data);
    try same(T, &saved_tau, &tau);
}

test "QR reconstruction, orthogonality and multiple RHS application for square/tall/wide padded f32/f64" {
    inline for (.{ f32, f64 }) |T| {
        try shapeCase(T, 4, 4, 1, tolerance(T));
        try shapeCase(T, 7, 3, 1, tolerance(T));
        try shapeCase(T, 3, 7, 1, tolerance(T));
        try shapeCase(T, 5, 1, 1, tolerance(T));
        try shapeCase(T, 1, 5, 1, tolerance(T));
    }
}

test "QR scaled norm and reflectors retain huge, tiny and subnormal input" {
    inline for (.{ f32, f64 }) |T| {
        const huge: T = if (T == f32) 1e37 else 1e300;
        const tiny: T = if (T == f32) 1e-37 else 1e-300;
        const subnormal: T = std.math.floatMin(T) / 64;
        try shapeCase(T, 5, 3, huge, tolerance(T));
        try shapeCase(T, 5, 3, tiny, tolerance(T));
        try shapeCase(T, 5, 3, subnormal, if (T == f32) 0.0001 else 1e-12);
    }
}

test "QR avoids alpha-beta and tau-dot overflow with representable factors" {
    var data = [_]f64{ 9e307, 7e307, -9e307, -7e307 };
    const original = data;
    var tau: [2]f64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(f64, matrix(f64, &data, 2, 2, 2), &tau));
    try checkFactor(f64, 2, 2, &original, &data, &tau, 2, 2e-14);
}

test "QR zero and already upper trapezoidal matrices need no reflectors" {
    inline for (.{ f32, f64 }) |T| {
        var data = [_]T{ -0.0, 0, 0, -2, 3, 0, 4, -5, 6 };
        const original = data;
        var tau: [3]T = @splat(sentinel);
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &data, 3, 3, 3), &tau));
        try same(T, &original, &data);
        for (tau) |value| try expectEqual(@as(T, 0), value);
        try shapeCase(T, 3, 4, 0, tolerance(T));
    }
}

test "QR empty shapes and zero RHS remain validated no-ops" {
    inline for (.{ f32, f64 }) |T| {
        var none: [0]T = .{};
        var untouched = [_]T{sentinel};
        for ([_][2]usize{ .{ 0, 0 }, .{ 0, 4 }, .{ 4, 0 } }) |dims| {
            try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &none, dims[0], dims[1], 0), &untouched));
            try expectEqual(@as(T, sentinel), untouched[0]);
        }
        var identity_rhs = [_]T{ 1, 2, 3, 4 };
        const before = identity_rhs;
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, matrix(T, &none, 4, 0, 0).asConst(), &none, matrix(T, &identity_rhs, 4, 1, 4), .normal));
        try same(T, &before, &identity_rhs);
        var factor = [_]T{ 2, 0.25, 0, 3 };
        var tau = [_]T{ 1.0, 0.0 };
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, matrix(T, &factor, 2, 2, 2).asConst(), &tau, matrix(T, &none, 2, 0, 0), .normal));
    }
}

test "QR capacity, dimensions and overlap errors precede factor mutation" {
    var data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const before = data;
    var tau = [_]f64{ sentinel, sentinel };
    const tau_before = tau;
    try expectError(error.BufferTooSmall, qr.qrFactorize(f64, matrix(f64, data[0..3], 2, 2, 2), &tau));
    try expectError(error.BufferTooSmall, qr.qrFactorize(f64, matrix(f64, &data, 2, 2, 2), tau[0..1]));
    try expectError(error.InvalidLeadingDimension, qr.qrFactorize(f64, matrix(f64, &data, 2, 2, 1), &tau));
    var bad = matrix(f64, &data, 2, 2, 2);
    bad.row_count = -1;
    try expectError(error.InvalidLength, qr.qrFactorize(f64, bad, &tau));
    bad = matrix(f64, data[0..0], 2, 0, 0);
    bad.leading_dimension = -1;
    try expectError(error.InvalidLeadingDimension, qr.qrFactorize(f64, bad, &tau));
    try expectError(error.AliasingNotAllowed, qr.qrFactorize(f64, matrix(f64, &data, 2, 2, 3), data[2..4]));
    try same(f64, &before, &data);
    try same(f64, &tau_before, &tau);
}

test "QR all finite input checks precede mutation; input tau contents are outputs" {
    inline for (.{ f32, f64 }) |T| {
        var data = [_]T{ 1, 2, 3, std.math.nan(T) };
        const before = data;
        var tau = [_]T{ std.math.nan(T), std.math.nan(T) };
        const tau_before = tau;
        try expectError(error.NonFiniteInput, qr.qrFactorize(T, matrix(T, &data, 2, 2, 2), &tau));
        try same(T, &before, &data);
        try same(T, &tau_before, &tau);
        data[3] = 4;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &data, 2, 2, 2), &tau));
        for (tau) |value| try expect(std.math.isFinite(value));
    }
}

test "Q apply validates every reflector and RHS before mutation" {
    var factor = [_]f64{ 2, 0.5, 0, 3 };
    var rhs = [_]f64{ 1, 2, 3, 4 };
    const before = rhs;
    var tau = [_]f64{ 1.6, 0 };
    const a = matrix(f64, &factor, 2, 2, 2).asConst();
    const b = matrix(f64, &rhs, 2, 2, 2);
    var transposed = a;
    transposed.operation = .transposed;
    try expectError(error.InvalidMatrixOperation, qr.qrApplyQ(f64, transposed, &tau, b, .normal));
    try expectError(error.DimensionMismatch, qr.qrApplyQ(f64, a, &tau, matrix(f64, &rhs, 1, 2, 1), .normal));
    try expectError(error.BufferTooSmall, qr.qrApplyQ(f64, a, tau[0..1], b, .normal));
    try expectError(error.BufferTooSmall, qr.qrApplyQ(f64, a, &tau, matrix(f64, rhs[0..3], 2, 2, 2), .normal));
    tau[1] = -1;
    try expectError(error.InvalidReflector, qr.qrApplyQ(f64, a, &tau, b, .normal));
    tau[1] = std.math.inf(f64);
    try expectError(error.NonFiniteInput, qr.qrApplyQ(f64, a, &tau, b, .normal));
    tau[1] = 0;
    factor[1] = 2;
    try expectError(error.InvalidReflector, qr.qrApplyQ(f64, a, &tau, b, .normal));
    factor[1] = std.math.nan(f64);
    try expectError(error.NonFiniteInput, qr.qrApplyQ(f64, a, &tau, b, .normal));
    try same(f64, &before, &rhs);
    factor[1] = 0.5;
    rhs[3] = std.math.nan(f64);
    const with_nan = rhs;
    try expectError(error.NonFiniteInput, qr.qrApplyQ(f64, a, &tau, b, .normal));
    try same(f64, &with_nan, &rhs);
}

test "Q apply rejects pairwise factor, tau and RHS overlap including padding" {
    var data: [20]f64 = @splat(0);
    data[0] = 2;
    data[1] = 0.5;
    data[4] = 3;
    const before = data;
    var separate_tau = [_]f64{ 1.6, 0 };
    var separate_rhs = [_]f64{ 1, 2, 3, 4 };
    const rhs_before = separate_rhs;
    const a = matrix(f64, data[0..5], 2, 2, 3).asConst();
    try expectError(error.AliasingNotAllowed, qr.qrApplyQ(f64, a, &separate_tau, matrix(f64, data[2..6], 2, 2, 2), .normal));
    try expectError(error.AliasingNotAllowed, qr.qrApplyQ(f64, a, data[2..4], matrix(f64, &separate_rhs, 2, 2, 2), .normal));
    try expectError(error.AliasingNotAllowed, qr.qrApplyQ(f64, a, separate_rhs[1..3], matrix(f64, &separate_rhs, 2, 2, 2), .normal));
    try same(f64, &before, &data);
    try same(f64, &rhs_before, &separate_rhs);
}

test "Q application does not read upper R, matrix padding or unused tau tail" {
    var factor = [_]f64{ 3, 4, sentinel, 5, 6, sentinel };
    var tau: [3]f64 = @splat(sentinel);
    try expectEqual(@as(usize, 0), try qr.qrFactorize(f64, matrix(f64, &factor, 2, 2, 3), &tau));
    var rhs = [_]f64{ 1, 2 };
    var expected = rhs;
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(f64, matrix(f64, &factor, 2, 2, 3).asConst(), &tau, matrix(f64, &expected, 2, 1, 2), .normal));
    factor[0] = std.math.nan(f64);
    factor[3] = std.math.inf(f64);
    factor[4] = std.math.nan(f64);
    factor[2] = std.math.nan(f64);
    factor[5] = std.math.nan(f64);
    tau[2] = std.math.nan(f64);
    const factor_before = factor;
    const tau_before = tau;
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(f64, matrix(f64, &factor, 2, 2, 3).asConst(), &tau, matrix(f64, &rhs, 2, 1, 2), .normal));
    try same(f64, &expected, &rhs);
    try same(f64, &factor_before, &factor);
    try same(f64, &tau_before, &tau);
}

test "QR reports unrepresentable norm without modifying current reflector" {
    inline for (.{ f32, f64 }) |T| {
        const largest = std.math.floatMax(T);
        var data = [_]T{ largest, largest };
        const before = data;
        var tau = [_]T{sentinel};
        try expectEqual(@as(usize, 1), try qr.qrFactorize(T, matrix(T, &data, 2, 1, 2), &tau));
        try same(T, &before, &data);
        try expectEqual(@as(T, sentinel), tau[0]);
    }
}

test "Q apply reports one-based numerical failure for unrepresentable output" {
    var factor = [_]f64{ 0, -0.4142135623730951 };
    var tau = [_]f64{1.7071067811865475};
    var rhs = [_]f64{ std.math.floatMax(f64), -std.math.floatMax(f64) };
    try expectEqual(@as(usize, 1), try qr.qrApplyQ(f64, matrix(f64, &factor, 2, 1, 2).asConst(), &tau, matrix(f64, &rhs, 2, 1, 2), .normal));
}

test "Q application keeps small active entries beside huge inactive entries" {
    var factor = [_]f64{ 0, 0 };
    var tau = [_]f64{2};
    var rhs = [_]f64{ 1e-100, 1e300 };
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(f64, matrix(f64, &factor, 2, 1, 2).asConst(), &tau, matrix(f64, &rhs, 2, 1, 2), .normal));
    try expectEqual(@as(f64, -1e-100), rhs[0]);
    try expectEqual(@as(f64, 1e300), rhs[1]);
}

test "Q application preserves representable corrections whose intermediate products underflow" {
    var factor = [_]f64{ 0, 1e-200, 1e-200 };
    var tau = [_]f64{2};
    var rhs = [_]f64{ 1e-100, 1e300, 0 };
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(f64, matrix(f64, &factor, 3, 1, 3).asConst(), &tau, matrix(f64, &rhs, 3, 1, 3), .normal));
    try near(-2, rhs[0] / 1e100, 1e-14);
    try expectEqual(@as(f64, 1e300), rhs[1]);
    try near(-2, rhs[2] / 1e-100, 1e-14);
    const smallest: f64 = @bitCast(@as(u64, 1));
    var tiny_factor = [_]f64{ 0, 0.5 };
    var tiny_tau = [_]f64{1.6};
    var tiny_rhs = [_]f64{ 0, smallest };
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(f64, matrix(f64, &tiny_factor, 2, 1, 2).asConst(), &tiny_tau, matrix(f64, &tiny_rhs, 2, 1, 2), .normal));
    try expectEqual(-smallest, tiny_rhs[0]);
    try expectEqual(smallest, tiny_rhs[1]);
}

test "Numerical failure info identifies the reflector, including reverse Q application" {
    inline for (.{ f32, f64 }) |T| {
        const largest = std.math.floatMax(T);
        var data = [_]T{ 1, 0, 0, 0, largest, largest };
        var tau = [_]T{ sentinel, sentinel };
        try expectEqual(@as(usize, 2), try qr.qrFactorize(T, matrix(T, &data, 3, 2, 3), &tau));
        try expectEqual(@as(T, 0), tau[0]);
        try expectEqual(@as(T, sentinel), tau[1]);
    }
    var factor = [_]f64{ 0, 0, 0, 0, 0, -0.4142135623730951 };
    var tau = [_]f64{ 0, 1.7071067811865475 };
    for ([_]qr.QOperation{ .normal, .transposed }) |operation| {
        var rhs = [_]f64{ 0, std.math.floatMax(f64), -std.math.floatMax(f64) };
        try expectEqual(@as(usize, 2), try qr.qrApplyQ(f64, matrix(f64, &factor, 3, 2, 3).asConst(), &tau, matrix(f64, &rhs, 3, 1, 3), operation));
    }
}

fn cmul(comptime C: type, a: C, b: C) C {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn cadd(comptime C: type, a: C, b: C) C {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn cnear(comptime C: type, a: C, b: C, tol: f64) !void {
    try near(a.re, b.re, tol);
    try near(a.im, b.im, tol);
}
fn checkComplexQR(comptime C: type, comptime m: usize, comptime n: usize, scale: f64) !void {
    const ld = m + 2;
    const marker: C = .{ .re = -987, .im = 345 };
    var a: [ld * n]C = @splat(marker);
    for (0..n) |j| for (0..m) |i| {
        a[i + j * ld] = .{ .re = @floatCast(scale * (@as(f64, @floatFromInt(1 + i + 2 * j)) / 8)), .im = @floatCast(scale * (@as(f64, @floatFromInt(i + 3 * j)) / 11 - 0.5)) };
    };
    const original = a;
    var tau: [@min(m, n)]C = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(C, matrix(C, &a, m, n, ld), &tau));
    var q: [m * m]C = @splat(.{ .re = 0, .im = 0 });
    for (0..m) |i| q[i + i * m].re = 1;
    const factor = matrix(C, &a, m, n, ld).asConst();
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(C, factor, &tau, matrix(C, &q, m, m, m), .normal));
    const tol: f64 = if (C == qr.ComplexF32) 8e-6 else 2e-12;
    for (0..m) |i| for (0..m) |j| {
        var z: C = .{ .re = 0, .im = 0 };
        for (0..m) |r| z = cadd(C, z, cmul(C, .{ .re = q[r + i * m].re, .im = -q[r + i * m].im }, q[r + j * m]));
        try cnear(C, .{ .re = if (i == j) 1 else 0, .im = 0 }, z, tol);
    };
    for (0..n) |j| for (0..m) |i| {
        var zr: f64 = 0;
        var zi: f64 = 0;
        for (0..@min(m, j + 1)) |r| {
            const f = a[r + j * ld];
            const v = q[i + r * m];
            zr += @as(f64, v.re) * (@as(f64, f.re) / scale) - @as(f64, v.im) * (@as(f64, f.im) / scale);
            zi += @as(f64, v.re) * (@as(f64, f.im) / scale) + @as(f64, v.im) * (@as(f64, f.re) / scale);
        }
        try near(@as(f64, original[i + j * ld].re) / scale, zr, tol);
        try near(@as(f64, original[i + j * ld].im) / scale, zi, tol);
    };
    for (0..n) |j| for (m..ld) |i| try expectEqual(marker, a[i + j * ld]);
    inline for (.{ qr.QOperation.normal, qr.QOperation.transposed, qr.QOperation.conjugate_transposed }) |op| {
        var b: [m * 3]C = undefined;
        for (&b, 0..) |*z, i| z.* = .{ .re = @floatCast(@as(f64, @floatFromInt(i + 1)) / 7), .im = @floatCast(0.3 - @as(f64, @floatFromInt(i)) / 13) };
        const before = b;
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(C, factor, &tau, matrix(C, &b, m, 3, m), op));
        for (0..3) |j| for (0..m) |i| {
            var z: C = .{ .re = 0, .im = 0 };
            for (0..m) |r| {
                var v = if (op == .normal) q[i + r * m] else q[r + i * m];
                if (op == .conjugate_transposed) v.im = -v.im;
                z = cadd(C, z, cmul(C, v, before[r + j * m]));
            }
            try cnear(C, z, b[i + j * m], tol * 8);
        };
    }
}

test "complex QR reconstructs square tall wide padded matrices and applies Q transpose and adjoint" {
    inline for (.{ qr.ComplexF32, qr.ComplexF64 }) |C| {
        try checkComplexQR(C, 3, 3, 1);
        try checkComplexQR(C, 5, 3, 1);
        try checkComplexQR(C, 3, 5, 1);
    }
}

test "complex QR normalized reflectors cover huge tiny and subnormal scalar phases" {
    try checkComplexQR(qr.ComplexF32, 4, 2, 1e30);
    try checkComplexQR(qr.ComplexF32, 4, 2, 1e-30);
    try checkComplexQR(qr.ComplexF64, 4, 2, 1e300);
    try checkComplexQR(qr.ComplexF64, 4, 2, 1e-300);
    inline for (.{ qr.ComplexF32, qr.ComplexF64 }) |C| {
        const R = @TypeOf(@as(C, undefined).re);
        const tiny = std.math.floatTrueMin(R);
        var a = [_]C{.{ .re = 0, .im = tiny }};
        var tau: [1]C = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(C, matrix(C, &a, 1, 1, 1), &tau));
        try expectEqual(-tiny, a[0].re);
        try expectEqual(@as(R, 0), a[0].im);
        var q = [_]C{.{ .re = 1, .im = 0 }};
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(C, matrix(C, &a, 1, 1, 1).asConst(), &tau, matrix(C, &q, 1, 1, 1), .normal));
        try expectEqual(@as(R, -1), q[0].im);
    }
}

test "complex QR preflight rejects capacity overlap nonfinite and malformed reflectors without writes" {
    const C = qr.ComplexF64;
    var a = [_]C{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } };
    var tau = [_]C{.{ .re = 111, .im = 222 }};
    const before = a;
    try expectError(error.BufferTooSmall, qr.qrFactorize(C, matrix(C, &a, 2, 1, 2), tau[0..0]));
    try expectError(error.AliasingNotAllowed, qr.qrFactorize(C, matrix(C, &a, 2, 1, 2), a[0..1]));
    try same(C, &before, &a);
    a[1].im = std.math.nan(f64);
    const bad = a;
    try expectError(error.NonFiniteInput, qr.qrFactorize(C, matrix(C, &a, 2, 1, 2), &tau));
    try same(C, &bad, &a);
    a = before;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(C, matrix(C, &a, 2, 1, 2), &tau));
    tau[0].im = 2;
    var b = before;
    try expectError(error.InvalidReflector, qr.qrApplyQ(C, matrix(C, &a, 2, 1, 2).asConst(), &tau, matrix(C, &b, 2, 1, 2), .normal));
    try same(C, &before, &b);
}

test "complex QR rank deficient empty and unrepresentable norm behavior" {
    const C = qr.ComplexF64;
    var empty: [0]C = .{};
    try expectEqual(@as(usize, 0), try qr.qrFactorize(C, matrix(C, &empty, 5, 0, 0), &empty));
    var zero: [6]C = @splat(.{ .re = 0, .im = 0 });
    var tau: [2]C = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(C, matrix(C, &zero, 3, 2, 3), &tau));
    try expectEqual(@as(f64, 0), tau[0].re);
    var huge = [_]C{ .{ .re = std.math.floatMax(f64), .im = 0 }, .{ .re = std.math.floatMax(f64), .im = 0 } };
    const before = huge;
    tau[0] = .{ .re = 123, .im = 456 };
    try expectEqual(@as(usize, 1), try qr.qrFactorize(C, matrix(C, &huge, 2, 1, 2), tau[0..1]));
    try same(C, &before, &huge);
    try expectEqual(@as(f64, 123), tau[0].re);
}

test "complex QR uses the same default operation owner and leaves no resource or admission residue" {
    const E = @import("zynum-execution");
    const Core = @import("zynum-core");
    const Probe = struct {
        var admitted: usize = 0;
        var finishing: usize = 0;
        var wrong_owner: bool = false;
        fn observe(ctx: *E.context.Context, event: E.context.TestEvent) void {
            wrong_owner = wrong_owner or ctx != E.getContext();
            if (event == .admitted) admitted += 1 else finishing += 1;
        }
    };
    Probe.admitted = 0;
    Probe.finishing = 0;
    Probe.wrong_owner = false;
    const ctx = E.getContext();
    try expect(ctx == Core.internal.numerics.Execution.getContext());
    E.context.Context.testing.configure(ctx, std.math.maxInt(usize), Probe.observe);
    defer E.context.Context.testing.configure(ctx, std.math.maxInt(usize), null);
    var a = [_]qr.ComplexF64{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } };
    var tau: [1]qr.ComplexF64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2), &tau));
    var b = [_]qr.ComplexF64{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 } };
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2).asConst(), &tau, matrix(qr.ComplexF64, &b, 2, 1, 2), .conjugate_transposed));
    try expectError(error.BufferTooSmall, qr.qrFactorize(qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2), tau[0..0]));
    try expectEqual(@as(usize, 3), Probe.admitted);
    try expectEqual(@as(usize, 3), Probe.finishing);
    try expect(!Probe.wrong_owner);
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
}

test "complex Q application ignores upper R and unused tau but validates zero RHS and preserves range failure vector" {
    inline for (.{ qr.ComplexF32, qr.ComplexF64 }) |C| {
        const R = @TypeOf(@as(C, undefined).re);
        var a = [_]C{ .{ .re = std.math.nan(R), .im = std.math.inf(R) }, .{ .re = @floatCast(@sqrt(@as(f64, 2)) - 1), .im = 0 } };
        var tau = [_]C{ .{ .re = @floatCast(1 + @sqrt(@as(f64, 0.5))), .im = 0 }, .{ .re = std.math.nan(R), .im = std.math.inf(R) } };
        var empty: [0]C = .{};
        const factor = matrix(C, &a, 2, 1, 2).asConst();
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(C, factor, &tau, matrix(C, &empty, 2, 0, 0), .normal));
        const saved_tau = tau[0];
        tau[0].im = std.math.nan(R);
        try expectError(error.NonFiniteInput, qr.qrApplyQ(C, factor, &tau, matrix(C, &empty, 2, 0, 0), .normal));
        tau[0] = saved_tau;
        var b = [_]C{ .{ .re = std.math.floatMax(R), .im = 0 }, .{ .re = std.math.floatMax(R), .im = 0 } };
        const before = b;
        try expectEqual(@as(usize, 1), try qr.qrApplyQ(C, factor, &tau, matrix(C, &b, 2, 1, 2), .normal));
        try same(C, &before, &b);
        try expectEqual(@as(usize, 1), try qr.qrApplyQ(C, factor, &tau, matrix(C, &b, 2, 1, 2), .conjugate_transposed));
        try same(C, &before, &b);
    }
}

test "QR admitted composition completes after local close without second admission or workspace pins" {
    const Core = @import("zynum-core");
    const E = Core.internal.numerics.Execution;
    const checked = Core.internal.lapack_checked;
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        var operation = try ctx.begin();
        try expectEqual(E.context.RequestResult.pending, ctx.requestClose());
        try expectError(error.Closing, ctx.begin());
        try expectEqual(E.context.CloseResult.busy, ctx.tryProgressClose());
        var a: [2]T = if (complex) .{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -4 } } else .{ 1, 3 };
        const before = a;
        var tau: [1]T = undefined;
        try expectEqual(@as(usize, 0), try checked.qrFactorizeAdmitted(&operation, T, matrix(T, &a, 2, 1, 2), &tau));
        var b = before;
        try expectEqual(@as(usize, 0), try checked.qrApplyQAdmitted(&operation, T, matrix(T, &a, 2, 1, 2).asConst(), &tau, matrix(T, &b, 2, 1, 2), .conjugate_transposed));
        if (complex) {
            try cnear(T, a[0], b[0], if (T == qr.ComplexF32) 2e-5 else 2e-12);
            try cnear(T, .{ .re = 0, .im = 0 }, b[1], if (T == qr.ComplexF32) 2e-5 else 2e-12);
        } else {
            try near(a[0], b[0], if (T == f32) 2e-5 else 2e-12);
            try near(0, b[1], if (T == f32) 2e-5 else 2e-12);
        }
        try expectError(error.BufferTooSmall, checked.qrFactorizeAdmitted(&operation, T, matrix(T, &a, 2, 1, 2), tau[0..0]));
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        operation.end();
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

const QrProbe = struct {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var events: usize = 0;
    var close_on_scan: bool = false;
    fn nan(ptr: *anyopaque) bool {
        if (close_on_scan) {
            const ctx: *E.context.Context = @ptrCast(@alignCast(ptr));
            _ = ctx.requestClose();
        }
        return true;
    }
    fn event(_: *anyopaque, _: B.common.ErrorEvent) void {
        events += 1;
    }
    fn services(ctx: *E.context.Context) B.Provider.Services {
        return .{ .allocator = undefined, .context = ctx, .nan_enabled = nan, .error_event = event };
    }
};

test "GEQRF typed and raw query has no matrix tau access allocation or extra admission" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        var query_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        var ctx = E.context.Context.init(query_allocator.allocator(), E.test_support.policy, .internal_closable);
        var op = try ctx.begin();
        const services = QrProbe.services(&ctx);
        for ([_]c_int{ 101, 102 }) |layout| {
            var work: [1]T = undefined;
            const args: B.QrAdapter.Arguments(T) = .{ .layout = layout, .m = 3, .n = 2, .lda = if (layout == 101) 2 else 3, .work = &work, .lwork = -1 };
            try expectEqual(@as(i32, 0), try B.QrAdapter.workAdmitted(T, &op, &services, args));
            const count = if (T == f32 or T == f64) work[0] else work[0].re;
            try expectEqual(@as(@TypeOf(count), 2), count);
            try expectEqual(@as(i32, 0), try B.RawQrAdapter.workAdmitted(T, &op, &services, .{ .layout = layout, .m = 3, .n = 2, .lda = args.lda, .work = @ptrCast(&work), .lwork = -1 }));
        }
        for (query_boundaries) |n| for ([_]c_int{ 101, 102 }) |layout| {
            var values: [1]T = undefined;
            const args: B.QrAdapter.Arguments(T) = .{ .layout = layout, .m = 1, .n = n, .lda = if (layout == 101) n else 1, .work = &values, .lwork = -1 };
            try expectEqual(@as(i32, 0), try B.QrAdapter.workAdmitted(T, &op, &services, args));
            try expectQueryMinimum(T, values[0], n);
            try expectEqual(@as(i32, 0), try B.RawQrAdapter.workAdmitted(T, &op, &services, .{ .layout = layout, .m = 1, .n = n, .lda = args.lda, .work = @ptrCast(&values), .lwork = -1 }));
            try expectQueryMinimum(T, values[0], n);
        };
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        op.end();
        try expectEqual(@as(usize, 0), query_allocator.allocated_bytes);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "GEQRF high and work row column layouts share four scalar factor and tau identities" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var expected: [6]T = undefined;
        for (&expected, 0..) |*v, i| v.* = if (complex) .{ .re = @floatFromInt(i + 1), .im = @floatCast(@as(f64, @floatFromInt(i)) / 7) } else @floatFromInt(i + 1);
        var et: [2]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &expected, 3, 2, 3), &et));
        inline for (.{ B.common.Level.high, B.common.Level.work }) |level| {
            for ([_]c_int{ 101, 102 }) |layout| {
                const marker: T = if (complex) .{ .re = 777, .im = 333 } else 777;
                var a: [12]T = @splat(marker);
                for (0..3) |i| for (0..2) |j| {
                    a[if (layout == 101) i * 4 + j else i + j * 4] = if (complex) .{ .re = @floatFromInt(i + j * 3 + 1), .im = @floatCast(@as(f64, @floatFromInt(i + j * 3)) / 7) } else @floatFromInt(i + j * 3 + 1);
                };
                const original = a;
                var tau: [2]T = undefined;
                var work: [2]T = undefined;
                var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
                const services = QrProbe.services(&ctx);
                try expectEqual(@as(i32, 0), try B.QrAdapter.callManaged(true, T, level, &ctx, &services, .{ .layout = layout, .m = 3, .n = 2, .a = &a, .lda = 4, .tau = &tau, .work = &work, .lwork = 2 }));
                for (0..3) |i| for (0..2) |j| {
                    const actual = a[if (layout == 101) i * 4 + j else i + j * 4];
                    if (complex) try cnear(T, expected[i + j * 3], actual, 0) else try expectEqual(expected[i + j * 3], actual);
                };
                try same(T, &et, &tau);
                for (a, original, 0..) |actual, old, i| {
                    const used = if (layout == 101) i / 4 < 3 and i % 4 < 2 else i / 4 < 2 and i % 4 < 3;
                    if (!used) try expectEqual(old, actual);
                }
                try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
                try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
                _ = ctx.requestClose();
                try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
            }
        }
    }
}

test "GEQRF scalar error order capacity and IEEE numerical outcomes are distinct" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    var work = [_]f64{99};
    try expectEqual(@as(i32, -1), try B.QrAdapter.callManaged(true, f64, .work, &ctx, &services, .{ .layout = 999, .m = -1, .n = -1, .lda = 0 }));
    try expectEqual(@as(i32, -5), try B.QrAdapter.callManaged(true, f64, .work, &ctx, &services, .{ .layout = 101, .m = -1, .n = 2, .lda = 1, .work = &work, .lwork = -1 }));
    try expectEqual(@as(i32, -2), try B.QrAdapter.callManaged(true, f64, .work, &ctx, &services, .{ .layout = 102, .m = -1, .n = 2, .lda = 1, .work = &work, .lwork = -1 }));
    try expectEqual(@as(f64, 99), work[0]);
    try expectError(error.MissingBuffer, B.QrAdapter.callManaged(true, f64, .work, &ctx, &services, .{ .layout = 102, .m = 2, .n = 1, .lda = 2, .lwork = -1 }));
    var a = [_]f64{ std.math.nan(f64), 1 };
    var tau: [1]f64 = undefined;
    try expectEqual(@as(i32, -4), try B.QrAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 102, .m = 2, .n = 1, .a = &a, .lda = 2, .tau = &tau }));
    try expectEqual(@as(i32, 0), try B.QrAdapter.callManaged(false, f64, .work, &ctx, &services, .{ .layout = 102, .m = 2, .n = 1, .a = &a, .lda = 2, .tau = &tau, .work = &work, .lwork = 1 }));
    try expect(std.math.isNan(a[0]));
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "GEQRF row high allocator and NaN callbacks stay inside admission across close" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    QrProbe.close_on_scan = true;
    defer QrProbe.close_on_scan = false;
    var a = [_]qr.ComplexF64{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } };
    var tau: [1]qr.ComplexF64 = undefined;
    try expectEqual(@as(i32, 0), try B.QrAdapter.callManaged(true, qr.ComplexF64, .high, &ctx, &services, .{ .layout = 101, .m = 2, .n = 1, .a = &a, .lda = 1, .tau = &tau }));
    try expectError(error.Closing, B.QrAdapter.callManaged(true, qr.ComplexF64, .high, &ctx, &services, .{ .layout = 101, .m = 2, .n = 1, .lda = 1 }));
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "GEQRF workspace and row staging OOM plus resource counter failure release captured leases" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    for ([_]usize{ 0, 1 }) |fail| {
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail });
        var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        var a = [_]f64{ 1, 3 };
        const before = a;
        var tau = [_]f64{999};
        QrProbe.events = 0;
        try expectEqual(@as(i32, if (fail == 0) -1010 else -1011), try B.QrAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 101, .m = 2, .n = 1, .a = &a, .lda = 1, .tau = &tau }));
        try same(f64, &before, &a);
        try expectEqual(@as(f64, 999), tau[0]);
        try expectEqual(@as(usize, 1), QrProbe.events);
        try expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
    E.context.Context.testing.configure(&ctx, 1, null);
    const services = QrProbe.services(&ctx);
    var a = [_]f64{ 1, 3 };
    const before = a;
    var tau = [_]f64{999};
    try expectError(error.CounterOverflow, B.QrAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 101, .m = 2, .n = 1, .a = &a, .lda = 1, .tau = &tau }));
    try same(f64, &before, &a);
    try expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "GEQRF raw column high uses actual owner arrays and does not create parallel resources" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var a: [2]T = if (complex) .{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } } else .{ 1, 3 };
        var expected = a;
        var tau: [1]T = undefined;
        var et: [1]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &expected, 2, 1, 2), &et));
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        try expectEqual(@as(i32, 0), try B.RawQrAdapter.callManaged(true, T, .high, &ctx, &services, .{ .layout = 102, .m = 2, .n = 1, .a = @ptrCast(&a), .lda = 2, .tau = @ptrCast(&tau) }));
        try same(T, &expected, &a);
        try same(T, &et, &tau);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "right Q application matches explicit matrix products for four scalar families and padded rows" {
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var a: [6]T = undefined;
        for (&a, 0..) |*v, i| v.* = if (complex) .{ .re = @floatFromInt(i + 1), .im = @floatCast(@as(f64, @floatFromInt(i)) / 7) } else @floatFromInt(i + 1);
        var tau: [2]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &a, 3, 2, 3), &tau));
        const factor = matrix(T, &a, 3, 2, 3).asConst();
        const zero: T = if (complex) .{ .re = 0, .im = 0 } else 0;
        const one: T = if (complex) .{ .re = 1, .im = 0 } else 1;
        var q: [9]T = @splat(zero);
        for (0..3) |i| q[i + i * 3] = one;
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, factor, &tau, matrix(T, &q, 3, 3, 3), .normal));
        inline for (.{ qr.QOperation.normal, qr.QOperation.transposed, qr.QOperation.conjugate_transposed }) |op| {
            const marker: T = if (complex) .{ .re = 777, .im = 333 } else 777;
            var b: [12]T = @splat(marker);
            for (0..2) |i| for (0..3) |j| {
                b[i + j * 4] = if (complex) .{ .re = @floatCast(@as(f64, @floatFromInt(i + j + 1)) / 7), .im = @floatCast(@as(f64, @floatFromInt(i + 2 * j)) / 11) } else @floatCast(@as(f64, @floatFromInt(i + j + 1)) / 7);
            };
            const before = b;
            try expectEqual(@as(usize, 0), try qr.qrApplyQSide(T, factor, &tau, matrix(T, &b, 2, 3, 4), .right, op));
            const tol: f64 = if (T == f32 or T == qr.ComplexF32) 2e-5 else 2e-12;
            for (0..2) |i| for (0..3) |j| {
                var z: T = zero;
                for (0..3) |r| {
                    var v = if (op == .normal) q[r + j * 3] else q[j + r * 3];
                    if (complex) {
                        if (op == .conjugate_transposed) v.im = -v.im;
                        z = cadd(T, z, cmul(T, before[i + r * 4], v));
                    } else z += before[i + r * 4] * v;
                }
                if (complex) try cnear(T, z, b[i + j * 4], tol) else try near(z, b[i + j * 4], tol);
            };
            for (0..3) |j| for (2..4) |i| try expectEqual(marker, b[i + j * 4]);
        }
    }
}

test "right Q validates empty row targets and completes borrowed operations after close" {
    const Core = @import("zynum-core");
    const E = @import("zynum-execution");
    var a = [_]qr.ComplexF64{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } };
    var tau: [1]qr.ComplexF64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2), &tau));
    const factor = matrix(qr.ComplexF64, &a, 2, 1, 2).asConst();
    var empty: [0]qr.ComplexF64 = .{};
    try expectEqual(@as(usize, 0), try qr.qrApplyQSide(qr.ComplexF64, factor, &tau, matrix(qr.ComplexF64, &empty, 0, 2, 0), .right, .normal));
    const saved = tau[0];
    tau[0].re = std.math.nan(f64);
    try expectError(error.NonFiniteInput, qr.qrApplyQSide(qr.ComplexF64, factor, &tau, matrix(qr.ComplexF64, &empty, 0, 2, 0), .right, .normal));
    tau[0] = saved;
    try expectError(error.DimensionMismatch, qr.qrApplyQSide(qr.ComplexF64, factor, &tau, matrix(qr.ComplexF64, &empty, 0, 3, 0), .right, .normal));
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    var op = try ctx.begin();
    _ = ctx.requestClose();
    var b = [_]qr.ComplexF64{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 } };
    try expectEqual(@as(usize, 0), try Core.internal.lapack_checked.qrApplyQSideAdmitted(&op, qr.ComplexF64, factor, &tau, matrix(qr.ComplexF64, &b, 1, 2, 1), .right, .conjugate_transposed));
    try expectError(error.Closing, ctx.begin());
    try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    op.end();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORMQR UNMQR query validates scalar order without matrix tau target reads" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        var query_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        var ctx = E.context.Context.init(query_allocator.allocator(), E.test_support.policy, .internal_closable);
        var op = try ctx.begin();
        const services = QrProbe.services(&ctx);
        for ([_]c_int{ 101, 102 }) |layout| for ("LR") |side| {
            var work: [1]T = undefined;
            const args: B.QApplyAdapter.Arguments(T) = .{ .layout = layout, .side = side, .trans = 'n', .m = 3, .n = 2, .k = 1, .lda = if (layout == 101) 1 else 3, .ldc = if (layout == 101) 2 else 3, .work = &work, .lwork = -1 };
            try expectEqual(@as(i32, 0), try B.QApplyAdapter.workAdmitted(T, &op, &services, args));
            const actual = if (T == f32 or T == f64) work[0] else work[0].re;
            try expectEqual(@as(@TypeOf(actual), if (side == 'L') 2 else 3), actual);
            try expectEqual(@as(i32, 0), try B.RawQApplyAdapter.workAdmitted(T, &op, &services, .{ .layout = layout, .side = side, .trans = 'n', .m = 3, .n = 2, .k = 1, .lda = args.lda, .ldc = args.ldc, .work = @ptrCast(&work), .lwork = -1 }));
            var bad = args;
            bad.trans = if (T == f32 or T == f64) 'C' else 'T';
            try expectEqual(@as(i32, -3), try B.QApplyAdapter.workAdmitted(T, &op, &services, bad));
        };
        for (query_boundaries) |n| for ([_]c_int{ 101, 102 }) |layout| for ("LR") |side| {
            var values: [1]T = undefined;
            const m = if (side == 'L') 1 else n;
            const cols = if (side == 'L') n else 1;
            const ldc = if (layout == 101) cols else m;
            const args: B.QApplyAdapter.Arguments(T) = .{ .layout = layout, .side = side, .trans = 'N', .m = m, .n = cols, .k = 0, .lda = 1, .ldc = ldc, .work = &values, .lwork = -1 };
            try expectEqual(@as(i32, 0), try B.QApplyAdapter.workAdmitted(T, &op, &services, args));
            try expectQueryMinimum(T, values[0], n);
            try expectEqual(@as(i32, 0), try B.RawQApplyAdapter.workAdmitted(T, &op, &services, .{ .layout = layout, .side = side, .trans = 'N', .m = m, .n = cols, .k = 0, .lda = 1, .ldc = ldc, .work = @ptrCast(&values), .lwork = -1 }));
            try expectQueryMinimum(T, values[0], n);
        };
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        op.end();
        try expectEqual(@as(usize, 0), query_allocator.allocated_bytes);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "ORMQR UNMQR finite high work row column left right produce identical owner outputs" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var factor: [6]T = undefined;
        for (&factor, 0..) |*z, i| z.* = if (complex) .{ .re = @floatFromInt(i + 1), .im = @floatCast(@as(f64, @floatFromInt(i)) / 7) } else @floatFromInt(i + 1);
        var tau: [2]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &factor, 3, 2, 3), &tau));
        for ("LR") |side| inline for (.{ B.common.Level.high, B.common.Level.work }) |level| {
            for ([_]c_int{ 101, 102 }) |layout| inline for (.{ qr.QOperation.normal, qr.QOperation.conjugate_transposed }) |op| {
                const m: usize = if (side == 'L') 3 else 2;
                const n: usize = if (side == 'L') 2 else 3;
                const marker: T = if (complex) .{ .re = 777, .im = 333 } else 777;
                var a: [9]T = @splat(marker);
                var c: [12]T = @splat(marker);
                var expected: [6]T = undefined;
                for (0..3) |i| for (0..2) |j| {
                    a[if (layout == 101) i * 3 + j else i + j * 4] = factor[i + j * 3];
                };
                for (0..m) |i| for (0..n) |j| {
                    const value: T = if (complex) .{ .re = @floatCast(@as(f64, @floatFromInt(i + j + 1)) / 7), .im = @floatCast(@as(f64, @floatFromInt(i + 2 * j)) / 11) } else @floatCast(@as(f64, @floatFromInt(i + j + 1)) / 7);
                    c[if (layout == 101) i * 4 + j else i + j * 4] = value;
                    expected[i + j * m] = value;
                };
                const before = c;
                const af = a;
                const tf = tau;
                try expectEqual(@as(usize, 0), try qr.qrApplyQSide(T, matrix(T, &factor, 3, 2, 3).asConst(), &tau, matrix(T, &expected, m, n, m), if (side == 'L') .left else .right, op));
                var work: [3]T = undefined;
                var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
                const services = QrProbe.services(&ctx);
                try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(true, T, level, &ctx, &services, .{ .layout = layout, .side = side, .trans = if (op == .normal) 'N' else if (complex) 'C' else 'T', .m = @intCast(m), .n = @intCast(n), .k = 2, .a = &a, .lda = if (layout == 101) 3 else 4, .tau = &tau, .c = &c, .ldc = 4, .work = &work, .lwork = 3 }));
                for (0..m) |i| for (0..n) |j| {
                    const z = c[if (layout == 101) i * 4 + j else i + j * 4];
                    if (complex) try cnear(T, expected[i + j * m], z, 0) else try expectEqual(expected[i + j * m], z);
                };
                for (c, before, 0..) |z, v, i| {
                    const used = if (layout == 101) i / 4 < m and i % 4 < n else i / 4 < n and i % 4 < m;
                    if (!used) try expectEqual(v, z);
                }
                try same(T, &af, &a);
                try same(T, &tf, &tau);
                try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
                try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
                _ = ctx.requestClose();
                try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
            };
        };
    }
}

test "ORMQR UNMQR high scan precedence and work ignored R remain distinct" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    var a = [_]f64{ 1, 3 };
    var tau: [1]f64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(f64, matrix(f64, &a, 2, 1, 2), &tau));
    var c = [_]f64{ 1, 2 };
    var work = [_]f64{999};
    const args: B.QApplyAdapter.Arguments(f64) = .{ .layout = 102, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 2, .tau = &tau, .c = &c, .ldc = 2, .work = &work, .lwork = 1 };
    a[0] = std.math.nan(f64);
    c[0] = std.math.nan(f64);
    const ts = tau[0];
    tau[0] = std.math.nan(f64);
    try expectEqual(@as(i32, -7), try B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, args));
    a[0] = 0;
    try expectEqual(@as(i32, -10), try B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, args));
    c[0] = 1;
    try expectEqual(@as(i32, -9), try B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, args));
    tau[0] = ts;
    a[0] = std.math.nan(f64);
    try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORMQR UNMQR three allocation failures and closing scan callbacks release all leases" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var a = [_]f64{ 1, 3 };
    var tau: [1]f64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(f64, matrix(f64, &a, 2, 1, 2), &tau));
    for ([_]usize{ 0, 1, 2 }) |fail| {
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail });
        var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        var c = [_]f64{ 1, 2 };
        const before = c;
        try expectEqual(@as(i32, if (fail == 0) -1010 else -1011), try B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 101, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 1, .tau = &tau, .c = &c, .ldc = 1 }));
        try same(f64, &before, &c);
        try expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    var c = [_]f64{ 1, 2 };
    QrProbe.close_on_scan = true;
    defer QrProbe.close_on_scan = false;
    try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 101, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 1, .tau = &tau, .c = &c, .ldc = 1 }));
    try expectError(error.Closing, ctx.begin());
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORMQR UNMQR scalar errors precede query writes and quick returns preserve data nonreads" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    var work = [_]f64{ 999, 888, 777 };
    var args: B.QApplyAdapter.Arguments(f64) = .{ .layout = 102, .side = 'L', .trans = 'N', .m = 2, .n = 3, .k = 1, .lda = 2, .ldc = 2, .work = &work, .lwork = -1 };
    args.side = 'X';
    try expectEqual(@as(i32, -2), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.side = 'L';
    args.k = 3;
    try expectEqual(@as(i32, -6), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.k = 1;
    args.lda = 1;
    try expectEqual(@as(i32, -8), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.lda = 2;
    args.ldc = 1;
    try expectEqual(@as(i32, -11), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.ldc = 2;
    args.lwork = 2;
    try expectEqual(@as(i32, -13), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    try expectEqual(@as(f64, 999), work[0]);
    args.k = 0;
    args.lwork = 3;
    try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    try expectEqual(@as(f64, 1), work[0]);
    args.layout = 101;
    args.lda = 0;
    args.ldc = 3;
    try expectError(error.MissingBuffer, B.QApplyAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORMQR UNMQR raw execution uses captured owner in both layouts" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var a: [2]T = if (complex) .{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } } else .{ 1, 3 };
        var tau: [1]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &a, 2, 1, 2), &tau));
        for ([_]c_int{ 101, 102 }) |layout| {
            var c: [2]T = if (complex) .{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 1 } } else .{ 1, 2 };
            var expected = c;
            try expectEqual(@as(usize, 0), try qr.qrApplyQSide(T, matrix(T, &a, 2, 1, 2).asConst(), &tau, matrix(T, &expected, 1, 2, 1), .right, .normal));
            var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
            const services = QrProbe.services(&ctx);
            try expectEqual(@as(i32, 0), try B.RawQApplyAdapter.callManaged(true, T, .high, &ctx, &services, .{ .layout = layout, .side = 'R', .trans = 'N', .m = 1, .n = 2, .k = 1, .a = @ptrCast(&a), .lda = if (layout == 101) 1 else 2, .tau = @ptrCast(&tau), .c = @ptrCast(&c), .ldc = if (layout == 101) 2 else 1 }));
            try same(T, &expected, &c);
            try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
            try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
            _ = ctx.requestClose();
            try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
        }
    }
}

test "ORMQR UNMQR third resource pin rejection cleans earlier captured allocations" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
    E.context.Context.testing.configure(&ctx, 2, null);
    const services = QrProbe.services(&ctx);
    var a = [_]f64{ 1, 3 };
    var tau: [1]f64 = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(f64, matrix(f64, &a, 2, 1, 2), &tau));
    var c = [_]f64{ 1, 2 };
    const before = c;
    try expectError(error.CounterOverflow, B.QApplyAdapter.callManaged(true, f64, .high, &ctx, &services, .{ .layout = 101, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 1, .tau = &tau, .c = &c, .ldc = 1 }));
    try same(f64, &before, &c);
    try expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "QR ABI row staging byte overflow is not OOM and never attempts allocation or array access" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    const huge = std.math.maxInt(i32);
    try expectError(error.SizeOverflow, B.QrAdapter.callManaged(false, qr.ComplexF64, .work, &ctx, &services, .{ .layout = 101, .m = huge, .n = huge, .lda = huge, .lwork = huge }));
    try expectError(error.SizeOverflow, B.QApplyAdapter.callManaged(false, qr.ComplexF64, .work, &ctx, &services, .{ .layout = 101, .side = 'L', .trans = 'N', .m = huge, .n = huge, .k = huge, .lda = huge, .ldc = huge, .lwork = huge }));
    try expectEqual(@as(usize, 0), allocator.alloc_index);
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "GEQRF IEEE work accepts NaN Inf and finite unrepresentable norms without positive INFO" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        const R = if (complex) @TypeOf(@as(T, undefined).re) else T;
        for ([_]R{ std.math.nan(R), std.math.inf(R), std.math.floatMax(R) }) |head| {
            for ([_]c_int{ 101, 102 }) |layout| {
                var a: [2]T = if (complex) .{ .{ .re = head, .im = 0 }, .{ .re = if (std.math.isFinite(head)) head else 1, .im = 0 } } else .{ head, if (std.math.isFinite(head)) head else 1 };
                const before = a;
                var tau: [1]T = undefined;
                var work: [1]T = undefined;
                var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
                const services = QrProbe.services(&ctx);
                try expectEqual(@as(i32, 0), try B.QrAdapter.callManaged(false, T, .work, &ctx, &services, .{ .layout = layout, .m = 2, .n = 1, .a = &a, .lda = if (layout == 101) 1 else 2, .tau = &tau, .work = &work, .lwork = 1 }));
                const diag = if (complex) a[0].re else a[0];
                try expect(!std.math.isFinite(diag));
                var checked = before;
                var ct: [1]T = undefined;
                if (std.math.isFinite(head)) try expectEqual(@as(usize, 1), try qr.qrFactorize(T, matrix(T, &checked, 2, 1, 2), &ct)) else try expectError(error.NonFiniteInput, qr.qrFactorize(T, matrix(T, &checked, 2, 1, 2), &ct));
                try same(T, &before, &checked);
                try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
                try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
                _ = ctx.requestClose();
                try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
            }
        }
    }
}

test "Q application IEEE work permits unchecked coefficients and preserves zero tau nonreads" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        const R = if (complex) @TypeOf(@as(T, undefined).re) else T;
        var a: [2]T = if (complex) .{ .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 0 } } else .{ 0, 1 };
        var tau: [1]T = if (complex) .{.{ .re = 3, .im = 0 }} else .{3};
        var c: [2]T = if (complex) .{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } } else .{ 1, 2 };
        var work: [1]T = undefined;
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        const args: B.QApplyAdapter.Arguments(T) = .{ .layout = 102, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 2, .tau = &tau, .c = &c, .ldc = 2, .work = &work, .lwork = 1 };
        var checked = c;
        try expectError(error.InvalidReflector, qr.qrApplyQ(T, matrix(T, &a, 2, 1, 2).asConst(), &tau, matrix(T, &checked, 2, 1, 2), .normal));
        try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(false, T, .work, &ctx, &services, args));
        try expectEqual(@as(R, -8), if (complex) c[0].re else c[0]);
        try expectEqual(@as(R, -7), if (complex) c[1].re else c[1]);
        tau[0] = if (complex) .{ .re = 0, .im = 0 } else 0;
        c[0] = if (complex) .{ .re = std.math.nan(R), .im = std.math.inf(R) } else std.math.nan(R);
        const before = c;
        try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(false, T, .work, &ctx, &services, args));
        try same(T, &before, &c);
        tau[0] = if (complex) .{ .re = std.math.nan(R), .im = 0 } else std.math.nan(R);
        c[0] = if (complex) .{ .re = 1, .im = 0 } else 1;
        try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(false, T, .work, &ctx, &services, args));
        try expect(std.math.isNan(if (complex) c[0].re else c[0]));
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "Q IEEE finite unrepresentable outputs project to infinity while checked calls keep failure reporting" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        const R = if (complex) @TypeOf(@as(T, undefined).re) else T;
        var a: [2]T = if (complex) .{ .{ .re = 0, .im = 0 }, .{ .re = @floatCast(@sqrt(@as(f64, 2)) - 1), .im = 0 } } else .{ 0, @floatCast(@sqrt(@as(f64, 2)) - 1) };
        var tau: [1]T = if (complex) .{.{ .re = @floatCast(1 + @sqrt(@as(f64, 0.5))), .im = 0 }} else .{@floatCast(1 + @sqrt(@as(f64, 0.5)))};
        var c: [2]T = if (complex) .{ .{ .re = std.math.floatMax(R), .im = 0 }, .{ .re = std.math.floatMax(R), .im = 0 } } else .{ std.math.floatMax(R), std.math.floatMax(R) };
        var checked = c;
        try expectEqual(@as(usize, 1), try qr.qrApplyQ(T, matrix(T, &a, 2, 1, 2).asConst(), &tau, matrix(T, &checked, 2, 1, 2), .normal));
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        var work: [1]T = undefined;
        try expectEqual(@as(i32, 0), try B.QApplyAdapter.callManaged(false, T, .work, &ctx, &services, .{ .layout = 102, .side = 'L', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = &a, .lda = 2, .tau = &tau, .c = &c, .ldc = 2, .work = &work, .lwork = 1 }));
        const value = if (complex) c[0].re else c[0];
        try expect(std.math.isInf(value));
        try expect(value < 0);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "raw GEQRF IEEE admitted NaN work completes after close and high enabled scan still rejects" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        const R = if (complex) @TypeOf(@as(T, undefined).re) else T;
        var a: [2]T = if (complex) .{ .{ .re = std.math.nan(R), .im = 0 }, .{ .re = 1, .im = 0 } } else .{ std.math.nan(R), 1 };
        var tau: [1]T = undefined;
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        const services = QrProbe.services(&ctx);
        var op = try ctx.begin();
        _ = ctx.requestClose();
        const args: B.RawQrAdapter.Arguments(T) = .{ .layout = 101, .m = 2, .n = 1, .a = @ptrCast(&a), .lda = 1, .tau = @ptrCast(&tau) };
        try expectEqual(@as(i32, -4), try B.RawQrAdapter.highAdmitted(true, T, &op, &services, args));
        try expectEqual(@as(i32, 0), try B.RawQrAdapter.highAdmitted(false, T, &op, &services, args));
        try expect(std.math.isNan(if (complex) a[0].re else a[0]));
        try expectError(error.Closing, ctx.begin());
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        op.end();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "generated QR shared raw bridges capture diagnostics and cleanup before ending closing admission" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    const old = B.shared_services.setCallbacks(.{});
    defer _ = B.shared_services.setCallbacks(old);
    const Probe = struct {
        ctx: *E.context.Context,
        calls: usize = 0,
        pins: usize = 0,
        ops: usize = 0,
        fn event(ptr: *anyopaque, _: B.common.ErrorEvent) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            self.pins = self.ctx.snapshot().resource_pins;
            self.ops = self.ctx.snapshot().active_operations;
            _ = self.ctx.requestClose();
        }
    };
    for ([_]bool{ false, true }) |apply| {
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        var probe = Probe{ .ctx = &ctx };
        const callbacks: B.shared_services.Callbacks = .{ .fortran_returning_test = .{ .userdata = &probe, .function = Probe.event } };
        var a = [_]f64{0};
        var c = [_]f64{ 1, 2 };
        var work = [_]f64{ 0, 0 };
        const info = if (apply) try B.Shared.qApplyRawManagedWithDefaults(false, f64, .work, &ctx, .{ .layout = 101, .side = 'X', .trans = 'N', .m = 2, .n = 1, .k = 1, .a = @ptrCast(&a), .lda = 1, .c = @ptrCast(&c), .ldc = 1, .work = @ptrCast(&work), .lwork = 2 }, callbacks) else try B.Shared.qrRawManagedWithDefaults(false, f64, .work, &ctx, .{ .layout = 101, .m = -1, .n = 2, .lda = 2, .work = @ptrCast(&work), .lwork = 2 }, callbacks);
        try expectEqual(@as(i32, -2), info);
        try expectEqual(@as(usize, 1), probe.calls);
        try expectEqual(@as(usize, 1), probe.ops);
        try expectEqual(@as(usize, if (apply) 2 else 1), probe.pins);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

fn checkGeneratedQ(comptime T: type, comptime m: usize, comptime n: usize, k: usize) !void {
    const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
    const marker: T = if (complex) .{ .re = 777, .im = 333 } else 777;
    const zero: T = if (complex) .{ .re = 0, .im = 0 } else 0;
    const ld = m + 2;
    var a: [ld * n]T = @splat(marker);
    for (0..n) |j| for (0..m) |i| {
        a[i + j * ld] = if (complex) .{ .re = @floatFromInt(1 + i + 2 * j), .im = @floatCast(@as(f64, @floatFromInt(i + j)) / 7) } else @floatFromInt(1 + i + 2 * j);
    };
    var tau: [n + 2]T = undefined;
    try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &a, m, n, ld), &tau));
    var expected: [m * n]T = @splat(zero);
    for (0..n) |j| expected[j + j * m] = if (complex) .{ .re = 1, .im = 0 } else 1;
    try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, matrix(T, &a, m, k, ld).asConst(), &tau, matrix(T, &expected, m, n, m), .normal));
    const R = if (complex) @TypeOf(@as(T, undefined).re) else T;
    const poison: T = if (complex) .{ .re = std.math.nan(R), .im = std.math.inf(R) } else std.math.nan(R);
    for (0..n) |j| for (0..m) |i| {
        if (j >= k or i <= j) a[i + j * ld] = poison;
    };
    for (k..tau.len) |i| tau[i] = poison;
    const tau_before = tau;
    try expectEqual(@as(usize, 0), try qr.qrGenerateQ(T, matrix(T, &a, m, n, ld), &tau, k));
    const tol: f64 = if (T == f32 or T == qr.ComplexF32) 2e-5 else 3e-12;
    for (0..n) |j| for (0..m) |i| {
        if (complex) try cnear(T, expected[i + j * m], a[i + j * ld], tol) else try near(expected[i + j * m], a[i + j * ld], tol);
    };
    for (0..n) |j| for (m..ld) |i| try expectEqual(marker, a[i + j * ld]);
    try same(T, &tau_before, &tau);
    for (0..n) |i| for (0..n) |j| {
        var re: f64 = 0;
        var im: f64 = 0;
        for (0..m) |row| {
            const x = a[row + i * ld];
            const y = a[row + j * ld];
            if (complex) {
                re += @as(f64, x.re) * @as(f64, y.re) + @as(f64, x.im) * @as(f64, y.im);
                im += @as(f64, x.re) * @as(f64, y.im) - @as(f64, x.im) * @as(f64, y.re);
            } else re += @as(f64, x) * @as(f64, y);
        }
        try near(if (i == j) 1 else 0, re, tol);
        try near(0, im, tol);
    };
}

test "explicit Q generation covers full partial and zero reflector thin square padded four scalar matrices" {
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        try checkGeneratedQ(T, 4, 3, 3);
        try checkGeneratedQ(T, 4, 3, 1);
        try checkGeneratedQ(T, 4, 3, 0);
        try checkGeneratedQ(T, 3, 3, 3);
    }
}

test "explicit Q generation preflight shape capacity alias finite and reflector errors precede writes" {
    var a = [_]f64{ 1, 0.5, 99, 2, 3, 88 };
    const before = a;
    var tau = [_]f64{ 1, 1 };
    try expectError(error.DimensionMismatch, qr.qrGenerateQ(f64, matrix(f64, &a, 2, 2, 3), &tau, 3));
    try expectError(error.BufferTooSmall, qr.qrGenerateQ(f64, matrix(f64, &a, 2, 2, 3), tau[0..0], 1));
    try expectError(error.AliasingNotAllowed, qr.qrGenerateQ(f64, matrix(f64, &a, 2, 2, 3), a[0..1], 1));
    tau[0] = std.math.nan(f64);
    try expectError(error.NonFiniteInput, qr.qrGenerateQ(f64, matrix(f64, &a, 2, 2, 3), &tau, 1));
    tau[0] = 3;
    try expectError(error.InvalidReflector, qr.qrGenerateQ(f64, matrix(f64, &a, 2, 2, 3), &tau, 1));
    try same(f64, &before, &a);
    var empty: [0]f64 = .{};
    try expectEqual(@as(usize, 0), try qr.qrGenerateQ(f64, matrix(f64, &empty, 5, 0, 0), &empty, 0));
}

test "explicit Q generation completes after local close with one operation and no scratch pins" {
    const Core = @import("zynum-core");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var a: [2]T = if (complex) .{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = 4 } } else .{ 1, 3 };
        var tau: [1]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &a, 2, 1, 2), &tau));
        var expected: [2]T = if (complex) .{ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 0 } } else .{ 1, 0 };
        try expectEqual(@as(usize, 0), try qr.qrApplyQ(T, matrix(T, &a, 2, 1, 2).asConst(), &tau, matrix(T, &expected, 2, 1, 2), .normal));
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        var op = try ctx.begin();
        _ = ctx.requestClose();
        try expectEqual(@as(usize, 0), try Core.internal.lapack_checked.qrGenerateQWithContractAdmitted(&op, T, matrix(T, &a, 2, 1, 2), &tau, 1, .checked));
        const tol: f64 = if (T == f32 or T == qr.ComplexF32) 2e-5 else 3e-12;
        for (a, expected) |z, v| {
            if (complex) try cnear(T, v, z, tol) else try near(v, z, tol);
        }
        try expectError(error.Closing, ctx.begin());
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        op.end();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "explicit Q generation IEEE policy accepts unchecked NaN coefficients without checked mutation" {
    const Core = @import("zynum-core");
    const E = @import("zynum-execution");
    var a = [_]qr.ComplexF64{ .{ .re = 0, .im = 0 }, .{ .re = 0.5, .im = 0.5 } };
    const before = a;
    var tau = [_]qr.ComplexF64{.{ .re = std.math.nan(f64), .im = 0 }};
    try expectError(error.NonFiniteInput, qr.qrGenerateQ(qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2), &tau, 1));
    try same(qr.ComplexF64, &before, &a);
    var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
    var op = try ctx.begin();
    try expectEqual(@as(usize, 0), try Core.internal.lapack_checked.qrGenerateQWithContractAdmitted(&op, qr.ComplexF64, matrix(qr.ComplexF64, &a, 2, 1, 2), &tau, 1, .lapack_ieee_v1));
    try expect(std.math.isNan(a[0].re));
    try expect(std.math.isNan(a[1].re));
    op.end();
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORGQR UNGQR typed high work and raw high preserve layout padding" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        var factor: [6]T = if (complex) .{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 }, .{ .re = 2, .im = 1 }, .{ .re = 2, .im = -3 }, .{ .re = 5, .im = 1 }, .{ .re = 4, .im = 2 } } else .{ 1, 3, 2, 2, 5, 4 };
        var tau: [2]T = undefined;
        try expectEqual(@as(usize, 0), try qr.qrFactorize(T, matrix(T, &factor, 3, 2, 3), &tau));
        var expected = factor;
        try expectEqual(@as(usize, 0), try qr.qrGenerateQ(T, matrix(T, &expected, 3, 2, 3), &tau, 2));
        inline for (.{ @as(c_int, 101), @as(c_int, 102) }) |layout| {
            inline for (.{ 0, 1, 2 }) |route| {
                var a: [12]T = @splat(if (complex) .{ .re = 77, .im = -9 } else 77);
                for (0..3) |i| for (0..2) |j| {
                    a[if (layout == 101) i * 4 + j else i + j * 4] = factor[i + j * 3];
                };
                const before = a;
                var work: [2]T = undefined;
                var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
                const services = QrProbe.services(&ctx);
                const info = if (route == 2) try B.RawQGenerateAdapter.callManaged(true, T, .high, &ctx, &services, .{ .layout = layout, .m = 3, .n = 2, .k = 2, .a = @ptrCast(&a), .lda = 4, .tau = @ptrCast(&tau) }) else try B.QGenerateAdapter.callManaged(true, T, if (route == 0) .high else .work, &ctx, &services, .{ .layout = layout, .m = 3, .n = 2, .k = 2, .a = &a, .lda = 4, .tau = &tau, .work = &work, .lwork = 2 });
                try expectEqual(@as(i32, 0), info);
                for (0..3) |i| for (0..2) |j| {
                    try same(T, expected[i + j * 3 ..][0..1], a[if (layout == 101) i * 4 + j else i + j * 4..][0..1]);
                };
                for (a, before, 0..) |v, b, i| {
                    const used = if (layout == 101) i / 4 < 3 and i % 4 < 2 else i / 4 < 2 and i % 4 < 3;
                    if (!used) try expectEqual(b, v);
                }
                try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
                try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
                _ = ctx.requestClose();
                try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
            }
        }
    }
}

test "ORGQR UNGQR query scalar errors and high NaN precedence" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    var query_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var ctx = E.context.Context.init(query_allocator.allocator(), E.test_support.policy, .internal_closable);
    const services = QrProbe.services(&ctx);
    var work = [_]f64{99};
    var args: B.QGenerateAdapter.Arguments(f64) = .{ .layout = 102, .m = 3, .n = 2, .k = 1, .lda = 3, .work = &work, .lwork = -1 };
    try expectEqual(@as(i32, 0), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    try expectEqual(@as(f64, 2), work[0]);
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        for (query_boundaries) |n| for ([_]c_int{ 101, 102 }) |layout| {
            var values: [1]T = undefined;
            try expectEqual(@as(i32, 0), try B.QGenerateAdapter.callManaged(true, T, .work, &ctx, &services, .{ .layout = layout, .m = n, .n = n, .k = 0, .lda = n, .work = &values, .lwork = -1 }));
            try expectQueryMinimum(T, values[0], n);
            try expectEqual(@as(i32, 0), try B.RawQGenerateAdapter.callManaged(true, T, .work, &ctx, &services, .{ .layout = layout, .m = n, .n = n, .k = 0, .lda = n, .work = @ptrCast(&values), .lwork = -1 }));
            try expectQueryMinimum(T, values[0], n);
        };
    }
    try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    args.m = -1;
    try expectEqual(@as(i32, -2), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.m = 1;
    try expectEqual(@as(i32, -3), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.m = 3;
    args.k = 3;
    try expectEqual(@as(i32, -4), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.k = 1;
    args.lda = 2;
    try expectEqual(@as(i32, -6), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.lda = 3;
    args.lwork = 1;
    try expectEqual(@as(i32, -9), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    args.layout = 101;
    args.lda = 1;
    try expectEqual(@as(i32, -6), try B.QGenerateAdapter.callManaged(true, f64, .work, &ctx, &services, args));
    var a = [_]f64{ std.math.nan(f64), 0, 0, 0, 0, 0 };
    var tau = [_]f64{std.math.nan(f64)};
    args.layout = 102;
    args.lda = 3;
    args.a = &a;
    args.tau = &tau;
    try expectEqual(@as(i32, -5), try B.QGenerateAdapter.callManaged(true, f64, .high, &ctx, &services, args));
    a[0] = 0;
    try expectEqual(@as(i32, -7), try B.QGenerateAdapter.callManaged(true, f64, .high, &ctx, &services, args));
    try expectEqual(@as(usize, 0), query_allocator.allocated_bytes);
    _ = ctx.requestClose();
    try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
}

test "ORGQR UNGQR admitted row high survives closing and releases both scratch leases" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ f32, f64, qr.ComplexF32, qr.ComplexF64 }) |T| {
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        var op = try ctx.begin();
        const services = QrProbe.services(&ctx);
        _ = ctx.requestClose();
        var a: [4]T = @splat(if (T == qr.ComplexF32 or T == qr.ComplexF64) .{ .re = 99, .im = 1 } else 99);
        try expectEqual(@as(i32, 0), try B.RawQGenerateAdapter.highAdmitted(false, T, &op, &services, .{ .layout = 101, .m = 2, .n = 2, .k = 0, .a = @ptrCast(&a), .lda = 2 }));
        const complex = T == qr.ComplexF32 or T == qr.ComplexF64;
        const one: T = if (complex) .{ .re = 1, .im = 0 } else 1;
        const zero: T = if (complex) .{ .re = 0, .im = 0 } else 0;
        try expectEqual(one, a[0]);
        try expectEqual(zero, a[1]);
        try expectEqual(zero, a[2]);
        try expectEqual(one, a[3]);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        try expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        op.end();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "ORGQR UNGQR allocation failures and pin overflow release captured resources" {
    const B = @import("numerical-owner");
    const E = @import("zynum-execution");
    inline for (.{ 0, 1, 2 }) |failure| {
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = if (failure == 2) 99 else failure });
        var ctx = E.context.Context.init(allocator.allocator(), E.test_support.policy, .internal_closable);
        if (failure == 2) E.context.Context.testing.configure(&ctx, 1, null);
        const services = QrProbe.services(&ctx);
        var a = [_]f64{ 77, 77, 77, 77 };
        const before = a;
        const args: B.QGenerateAdapter.Arguments(f64) = .{ .layout = 101, .m = 2, .n = 2, .k = 0, .a = &a, .lda = 2 };
        if (failure == 2) try expectError(error.CounterOverflow, B.QGenerateAdapter.callManaged(false, f64, .high, &ctx, &services, args)) else try expectEqual(@as(i32, if (failure == 0) -1010 else -1011), try B.QGenerateAdapter.callManaged(false, f64, .high, &ctx, &services, args));
        try same(f64, &before, &a);
        try expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        try expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        try expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        _ = ctx.requestClose();
        try expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}
