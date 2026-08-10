// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Forced differential tests for registered Level 2 fused execution bodies.

const std = @import("std");

const types = @import("types.zig");
const scalar = @import("core/shared/scalar.zig");
const access = @import("core/matrix_vector/access.zig");
const general = @import("core/matrix_vector/general.zig");
const rank_update = @import("core/matrix_vector/rank_update.zig");
const symmetric = @import("core/matrix_vector/symmetric.zig");
const triangular = @import("core/matrix_vector/triangular.zig");
const matrix_vector_kernels = @import("kernels/dispatch/matrix_vector.zig");
const catalog = @import("kernels/shared/matrix_vector/catalog.zig");
const coverage = @import("kernels/shared/matrix_vector/coverage.zig");
const tuning = @import("kernels/shared/matrix_vector/tuning.zig");

comptime {
    _ = coverage;
}

fn value(comptime T: type, index: usize, phase: usize) T {
    const re = @as(f64, @floatFromInt((index * 17 + phase * 11) % 41)) / 19.0 - 0.9;
    if (T == f32 or T == f64) return @floatCast(re);
    const im = @as(f64, @floatFromInt((index * 13 + phase * 7) % 37)) / 23.0 - 0.7;
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    if (T == f32) return std.testing.expectApproxEqAbs(expected, actual, 3e-4);
    if (T == f64) return std.testing.expectApproxEqAbs(expected, actual, 2e-11);
    if (T == types.ComplexF32) {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 4e-4);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 4e-4);
        return;
    }
    try std.testing.expectApproxEqAbs(expected.re, actual.re, 3e-11);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, 3e-11);
}

fn differentialGemv(comptime T: type, comptime trans: scalar.Order) !void {
    const m = 7;
    const n = 5;
    const lda = 9;
    var a: [lda * n]T = undefined;
    var x: [m]T = undefined;
    var y: [m]T = undefined;
    var expected: [m]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 1);
    for (&x, 0..) |*item, i| item.* = value(T, i, 2);
    for (&y, 0..) |*item, i| item.* = value(T, i, 3);
    expected = y;
    const alpha = value(T, 3, 5);
    const output_len = if (trans == .no_trans) m else n;
    const inner_len = if (trans == .no_trans) n else m;
    for (0..output_len) |row| {
        var sum = scalar.zero(T);
        for (0..inner_len) |k| {
            var av = if (trans == .no_trans) a[row + k * lda] else a[k + row * lda];
            if (trans == .conj_trans) av = scalar.conj(T, av);
            sum = scalar.add(T, sum, scalar.mul(T, av, x[k]));
        }
        expected[row] = scalar.add(T, expected[row], scalar.mul(T, alpha, sum));
    }

    general.testing.gemvPanel(T, trans, m, n, alpha, &a, lda, &x, &y);
    for (0..output_len) |i| try expectClose(T, expected[i], y[i]);
}

test "forced fused GEMV N T and C panels match scalar references" {
    try differentialGemv(f32, .no_trans);
    try differentialGemv(f64, .trans);
    try differentialGemv(types.ComplexF32, .no_trans);
    try differentialGemv(types.ComplexF64, .trans);
    try differentialGemv(types.ComplexF32, .conj_trans);
    try differentialGemv(types.ComplexF64, .conj_trans);
}

fn differentialGer(comptime T: type, conjugate_y: bool) !void {
    const m = 7;
    const n = 5;
    const lda = 9;
    var a: [lda * n]T = undefined;
    var expected: [lda * n]T = undefined;
    var x: [m]T = undefined;
    var y: [n]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 7);
    expected = a;
    for (&x, 0..) |*item, i| item.* = value(T, i, 8);
    for (&y, 0..) |*item, i| item.* = value(T, i, 9);
    const alpha = value(T, 4, 10);

    for (0..n) |j| {
        const yj = if (conjugate_y) scalar.conj(T, y[j]) else y[j];
        const coefficient = scalar.mul(T, alpha, yj);
        for (0..m) |i| {
            const index = i + j * lda;
            expected[index] = scalar.add(T, expected[index], scalar.mul(T, x[i], coefficient));
        }
    }
    rank_update.testing.gerColumns(T, m, n, alpha, &x, &y, &a, lda, conjugate_y);
    for (0..n) |j| for (0..m) |i| {
        const index = i + j * lda;
        try expectClose(T, expected[index], a[index]);
    };
}

test "forced fused GER GERU and GERC columns match scalar references" {
    try differentialGer(f32, false);
    try differentialGer(f64, false);
    try differentialGer(types.ComplexF32, false);
    try differentialGer(types.ComplexF64, true);
}

fn differentialArchitectureComplexGer(comptime T: type, conjugate_y: bool) !void {
    const m = 128;
    const n = 128;
    const lda = 131;
    var a: [lda * n]T = undefined;
    var expected: [lda * n]T = undefined;
    var x: [m]T = undefined;
    var y: [n]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 27);
    expected = a;
    for (&x, 0..) |*item, i| item.* = value(T, i, 28);
    for (&y, 0..) |*item, i| item.* = value(T, i, 29);
    const alpha = value(T, 4, 30);

    for (0..n) |j| {
        const yj = if (conjugate_y) scalar.conj(T, y[j]) else y[j];
        const coefficient = scalar.mul(T, alpha, yj);
        for (0..m) |i| {
            const index = i + j * lda;
            expected[index] = scalar.add(T, expected[index], scalar.mul(T, x[i], coefficient));
        }
    }

    try std.testing.expect(matrix_vector_kernels.gerUnitComplex(
        T,
        m,
        n,
        alpha,
        &x,
        &y,
        &a,
        lda,
        conjugate_y,
    ));
    for (0..n) |j| for (0..m) |i| {
        const index = i + j * lda;
        try expectClose(T, expected[index], a[index]);
    };
}

test "forced fixed-SIMD GERU and GERC leaves match scalar references" {
    try differentialArchitectureComplexGer(types.ComplexF32, false);
    try differentialArchitectureComplexGer(types.ComplexF64, true);
}

fn differentialSymmetric(comptime T: type, hermitian: bool) !void {
    const n = 7;
    const lda = 9;
    var a: [lda * n]T = undefined;
    var x: [n]T = undefined;
    var actual: [n]T = undefined;
    var expected: [n]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 12);
    if (comptime scalar.isComplex(T)) {
        if (hermitian) {
            for (0..n) |i| a[i + i * lda].im = 0;
        }
    }
    for (&x, 0..) |*item, i| item.* = value(T, i, 13);
    @memset(&actual, scalar.zero(T));
    @memset(&expected, scalar.zero(T));
    const alpha = value(T, 6, 14);

    for (0..n) |i| {
        var sum = scalar.zero(T);
        for (0..n) |j| {
            const av = access.symValue(T, .upper, &a, lda, i, j, hermitian);
            sum = scalar.add(T, sum, scalar.mul(T, av, x[j]));
        }
        expected[i] = scalar.mul(T, alpha, sum);
    }
    symmetric.testing.symmetricColumns(T, .upper, n, 0, n, alpha, &a, lda, &x, &actual);
    for (expected, actual) |want, got| try expectClose(T, want, got);

    @memset(&actual, scalar.zero(T));
    try std.testing.expect(matrix_vector_kernels.symmetricColumnsUnit(
        T,
        true,
        hermitian,
        n,
        0,
        n,
        alpha,
        &a,
        lda,
        &x,
        &actual,
    ));
    for (expected, actual) |want, got| try expectClose(T, want, got);
}

test "forced fused SYMV and HEMV column bodies match scalar references" {
    try differentialSymmetric(f32, false);
    try differentialSymmetric(f64, false);
    try differentialSymmetric(types.ComplexF32, true);
    try differentialSymmetric(types.ComplexF64, true);
}

fn differentialTriangularBodies(comptime T: type) !void {
    const max_n = 67;
    var a: [max_n]T = undefined;
    var x: [max_n]T = undefined;
    var actual: [max_n]T = undefined;
    var expected: [max_n]T = undefined;
    for (&a, 0..) |*item, i| item.* = value(T, i, 16);
    for (&x, 0..) |*item, i| item.* = value(T, i, 17);
    const alpha = value(T, 2, 18);

    for ([_]usize{ 0, 7, 64, 67 }) |n| {
        actual = x;
        expected = x;
        for (0..n) |i| expected[i] = scalar.add(T, expected[i], scalar.mul(T, alpha, a[i]));
        triangular.testing.axpyBody(T, n, alpha, &a, &actual);
        for (0..n) |i| try expectClose(T, expected[i], actual[i]);

        actual = x;
        const architecture_axpy = matrix_vector_kernels.triangularAxpyUnit(T, n, alpha, &a, &actual);
        try std.testing.expectEqual(n != 0, architecture_axpy);
        for (0..n) |i| try expectClose(T, expected[i], actual[i]);

        inline for (.{ false, true }) |conjugate_a| {
            var want = scalar.zero(T);
            for (0..n) |i| {
                const av = if (conjugate_a) scalar.conj(T, a[i]) else a[i];
                want = scalar.add(T, want, scalar.mul(T, av, x[i]));
            }
            const got = triangular.testing.dotBody(T, n, &a, &x, conjugate_a);
            try expectClose(T, want, got);

            var architecture_got = scalar.zero(T);
            const architecture_dot = matrix_vector_kernels.triangularDotUnit(
                T,
                n,
                &a,
                &x,
                conjugate_a,
                &architecture_got,
            );
            try std.testing.expectEqual(n != 0, architecture_dot);
            if (n != 0) try expectClose(T, want, architecture_got);
        }
    }
}

test "forced triangular AXPY and DOT dependency bodies cover empty width and tail" {
    try differentialTriangularBodies(f32);
    try differentialTriangularBodies(f64);
    try differentialTriangularBodies(types.ComplexF32);
    try differentialTriangularBodies(types.ComplexF64);
}

test "fused registry selection remains below complete unit execution" {
    const request: tuning.Request = .{ .m = 7, .n = 5, .incx = 1, .incy = 1 };
    try std.testing.expectEqual(catalog.Implementation.core_unit, tuning.selectDefault(f64, .gemv, request).kernel.implementation);
    const fused = catalog.findImplementation(.gemv, .f64, .fused_gemv_no_trans).?;
    try std.testing.expectEqual(catalog.Implementation.core_unit, fused.fallback.?.implementation);
    try std.testing.expectEqual(catalog.CompletionScope.output_region, fused.completion);
}
