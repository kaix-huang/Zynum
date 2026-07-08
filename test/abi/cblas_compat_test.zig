// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const cblas = @import("zynum_blas_cblas_compat").abi;
const ref = @import("reference.zig");

fn complexF32(re: f32, im: f32) cblas.ComplexF32 {
    return .{ .re = re, .im = im };
}

fn complexF64(re: f64, im: f64) cblas.ComplexF64 {
    return .{ .re = re, .im = im };
}

fn expectComplexF64Approx(expected: cblas.ComplexF64, actual: cblas.ComplexF64) !void {
    try ref.expectApprox(cblas.ComplexF64, expected, actual, @as(f64, 1e-10));
}

fn expectComplexF64SliceApprox(expected: []const cblas.ComplexF64, actual: []const cblas.ComplexF64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| try expectComplexF64Approx(want, got);
}

fn referenceComplexScal(comptime T: type, n: usize, alpha: T, x: []T, incx: isize) void {
    for (0..n) |i| {
        const idx = ref.vectorIndex(n, incx, i);
        x[idx] = ref.mul(T, alpha, x[idx]);
    }
}

fn referenceComplexAxpy(comptime T: type, n: usize, alpha: T, x: []const T, incx: isize, y: []T, incy: isize) void {
    for (0..n) |i| {
        const ix = ref.vectorIndex(n, incx, i);
        const iy = ref.vectorIndex(n, incy, i);
        y[iy] = ref.add(T, ref.mul(T, alpha, x[ix]), y[iy]);
    }
}

fn referenceComplexAxpby(comptime T: type, n: usize, alpha: T, x: []const T, incx: isize, beta: T, y: []T, incy: isize) void {
    for (0..n) |i| {
        const ix = ref.vectorIndex(n, incx, i);
        const iy = ref.vectorIndex(n, incy, i);
        y[iy] = ref.add(T, ref.mul(T, alpha, x[ix]), ref.mul(T, beta, y[iy]));
    }
}

test "cblas level3 invalid enums leave outputs unchanged" {
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ 9, 10, 11, 12 };
    const original_c = c;
    cblas.cblas_dgemm(999, cblas.CblasNoTrans, cblas.CblasNoTrans, 2, 2, 2, 1, &a, 2, &b, 2, 0, &c, 2);
    try std.testing.expectEqualSlices(f64, &original_c, &c);

    var tri = [_]f64{ 1, 0, 2, 3 };
    var trmm_b = [_]f64{ 4, 5, 6, 7 };
    const original_b = trmm_b;
    cblas.cblas_dtrmm(cblas.CblasColMajor, cblas.CblasLeft, cblas.CblasUpper, 999, cblas.CblasNonUnit, 2, 2, 1, &tri, 2, &trmm_b, 2);
    try std.testing.expectEqualSlices(f64, &original_b, &trmm_b);
}

test "cblas level2 invalid enums leave outputs unchanged" {
    var matrix = [_]f64{ 1, 2, 3, 4 };
    var x = [_]f64{ 5, 6 };
    var y = [_]f64{ 7, 8 };
    const original_y = y;
    cblas.cblas_dgemv(cblas.CblasColMajor, 999, 2, 2, 1, &matrix, 2, &x, 1, 0, &y, 1);
    try std.testing.expectEqualSlices(f64, &original_y, &y);

    var sym_y = [_]f64{ 9, 10 };
    const original_sym_y = sym_y;
    cblas.cblas_dsymv(cblas.CblasColMajor, 999, 2, 1, &matrix, 2, &x, 1, 0, &sym_y, 1);
    try std.testing.expectEqualSlices(f64, &original_sym_y, &sym_y);

    var tri_x = [_]f64{ 11, 12 };
    const original_tri_x = tri_x;
    cblas.cblas_dtrmv(cblas.CblasColMajor, cblas.CblasUpper, cblas.CblasNoTrans, 999, 2, &matrix, 2, &tri_x, 1);
    try std.testing.expectEqualSlices(f64, &original_tri_x, &tri_x);

    var rank_matrix = [_]f64{ 1, 2, 3, 4 };
    const original_rank_matrix = rank_matrix;
    cblas.cblas_dger(999, 2, 2, 1, &x, 1, &y, 1, &rank_matrix, 2);
    try std.testing.expectEqualSlices(f64, &original_rank_matrix, &rank_matrix);
}

test "cblas row-major direct invalid parameters leave outputs unchanged" {
    const alpha = complexF64(1, 0);
    const beta = complexF64(0, 0);
    var matrix = [_]cblas.ComplexF64{
        complexF64(1, 1), complexF64(2, -1),
        complexF64(3, 2), complexF64(-1, 1),
    };
    var x = [_]cblas.ComplexF64{ complexF64(1, 0), complexF64(2, 0) };
    var y = [_]cblas.ComplexF64{ complexF64(7, 0), complexF64(8, 0) };
    const original_y = y;
    cblas.cblas_zgemv(cblas.CblasRowMajor, cblas.CblasConjTrans, 2, 2, &alpha, &matrix, 1, &x, 1, &beta, &y, 1);
    try expectComplexF64SliceApprox(&original_y, &y);

    var tri_x = [_]cblas.ComplexF64{ complexF64(3, 0), complexF64(4, 0) };
    const original_tri_x = tri_x;
    cblas.cblas_ztrmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, 999, 2, &matrix, 2, &tri_x, 1);
    try expectComplexF64SliceApprox(&original_tri_x, &tri_x);
}

test "cblas symmetric level3 invalid enums leave outputs unchanged" {
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ 9, 10, 11, 12 };
    const original_c = c;
    cblas.cblas_dsymm(cblas.CblasColMajor, 999, cblas.CblasUpper, 2, 2, 1, &a, 2, &b, 2, 0, &c, 2);
    try std.testing.expectEqualSlices(f64, &original_c, &c);

    var rank_c = [_]f64{ 13, 14, 15, 16 };
    const original_rank_c = rank_c;
    cblas.cblas_dsyrk(cblas.CblasColMajor, cblas.CblasUpper, 999, 2, 2, 1, &a, 2, 0, &rank_c, 2);
    try std.testing.expectEqualSlices(f64, &original_rank_c, &rank_c);

    var rank2_c = [_]f64{ 17, 18, 19, 20 };
    const original_rank2_c = rank2_c;
    cblas.cblas_dsyr2k(cblas.CblasColMajor, 999, cblas.CblasNoTrans, 2, 2, 1, &a, 2, &b, 2, 0, &rank2_c, 2);
    try std.testing.expectEqualSlices(f64, &original_rank2_c, &rank2_c);
}

fn expectCblasComplexScalCase(comptime T: type, n: usize, incx: isize, alpha: T, tol: anytype) !void {
    var rng = ref.Rng.init(0x5ca1_1234);
    const len = ref.vectorStorageLen(n, incx);
    const x = try std.testing.allocator.alloc(T, len);
    defer std.testing.allocator.free(x);
    const expected = try std.testing.allocator.alloc(T, len);
    defer std.testing.allocator.free(expected);

    ref.fillVector(T, &rng, x, n, incx);
    @memcpy(expected, x);

    if (T == cblas.ComplexF32) {
        cblas.cblas_cscal(@intCast(n), &alpha, x.ptr, @intCast(incx));
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zscal(@intCast(n), &alpha, x.ptr, @intCast(incx));
    } else {
        @compileError("complex scal test supports ComplexF32 and ComplexF64");
    }
    referenceComplexScal(T, n, alpha, expected, incx);

    for (expected, x) |want, got| try ref.expectApprox(T, want, got, tol);
}

fn expectCblasComplexAxpyCase(comptime T: type, n: usize, incx: isize, incy: isize, alpha: T, tol: anytype) !void {
    var rng = ref.Rng.init(0xca90_74b5);
    const x_len = ref.vectorStorageLen(n, incx);
    const y_len = ref.vectorStorageLen(n, incy);
    const x = try std.testing.allocator.alloc(T, x_len);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(T, y_len);
    defer std.testing.allocator.free(y);
    const expected = try std.testing.allocator.alloc(T, y_len);
    defer std.testing.allocator.free(expected);

    ref.fillVector(T, &rng, x, n, incx);
    ref.fillVector(T, &rng, y, n, incy);
    @memcpy(expected, y);

    if (T == cblas.ComplexF32) {
        cblas.cblas_caxpy(@intCast(n), &alpha, x.ptr, @intCast(incx), y.ptr, @intCast(incy));
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zaxpy(@intCast(n), &alpha, x.ptr, @intCast(incx), y.ptr, @intCast(incy));
    } else {
        @compileError("complex axpy test supports ComplexF32 and ComplexF64");
    }
    referenceComplexAxpy(T, n, alpha, x, incx, expected, incy);

    for (expected, y) |want, got| try ref.expectApprox(T, want, got, tol);
}

fn expectCblasComplexAxpbyCase(comptime T: type, n: usize, incx: isize, incy: isize, alpha: T, beta: T, tol: anytype) !void {
    var rng = ref.Rng.init(0xcab9_1234);
    const x_len = ref.vectorStorageLen(n, incx);
    const y_len = ref.vectorStorageLen(n, incy);
    const x = try std.testing.allocator.alloc(T, x_len);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(T, y_len);
    defer std.testing.allocator.free(y);
    const expected = try std.testing.allocator.alloc(T, y_len);
    defer std.testing.allocator.free(expected);

    ref.fillVector(T, &rng, x, n, incx);
    ref.fillVector(T, &rng, y, n, incy);
    @memcpy(expected, y);

    if (T == cblas.ComplexF32) {
        cblas.cblas_caxpby(@intCast(n), &alpha, x.ptr, @intCast(incx), &beta, y.ptr, @intCast(incy));
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zaxpby(@intCast(n), &alpha, x.ptr, @intCast(incx), &beta, y.ptr, @intCast(incy));
    } else {
        @compileError("complex axpby test supports ComplexF32 and ComplexF64");
    }
    referenceComplexAxpby(T, n, alpha, x, incx, beta, expected, incy);

    for (expected, y) |want, got| try ref.expectApprox(T, want, got, tol);
}

fn makeTriangularDiagSafe(a: []cblas.ComplexF64, n: usize, lda: usize) void {
    for (0..n) |i| {
        const step = @as(f64, @floatFromInt(i));
        a[ref.rowIndex(lda, i, i)] = complexF64(1.25 + step * 0.5, -0.35 + step * 0.2);
    }
}

fn fillRowMajorBand(comptime T: type, rng: *ref.Rng, a: []T, m: usize, n: usize, kl: usize, ku: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, 1000 + @as(f64, @floatFromInt(i)), -1000);
    for (0..m) |row| {
        for (0..n) |col| {
            if (ref.gbIndexRowMajor(m, n, kl, ku, lda, row, col)) |idx| a[idx] = rng.scalar(T);
        }
    }
}

fn fillColMajorSymBand(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, a: []T, n: usize, k: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, -2000 - @as(f64, @floatFromInt(i)), 2000);
    for (0..n) |col| {
        for (0..n) |row| {
            const stored = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
            if (!stored) continue;
            if (ref.symBandIndex(uplo, n, k, lda, row, col)) |idx| a[idx] = rng.scalar(T);
        }
    }
}

fn fillPackedSym(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, ap: []T, n: usize) void {
    for (0..ap.len) |i| ap[i] = ref.fromParts(T, 3000 + @as(f64, @floatFromInt(i)), -3000);
    for (0..n) |col| {
        for (0..n) |row| {
            const stored = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
            if (!stored) continue;
            ap[ref.packedIndex(uplo, n, row, col)] = rng.scalar(T);
        }
    }
}

fn fillColMajorTriBand(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, a: []T, n: usize, k: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, -4000 - @as(f64, @floatFromInt(i)), 4000);
    for (0..n) |col| {
        for (0..n) |row| {
            if (ref.triBandIndex(uplo, k, lda, row, col)) |idx| {
                a[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) a[idx] = ref.add(T, a[idx], ref.fromParts(T, 2.0 + @as(f64, @floatFromInt(row)), -0.25));
            }
        }
    }
}

fn fillRowMajorTriBand(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, a: []T, n: usize, k: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, -4100 - @as(f64, @floatFromInt(i)), 4100);
    for (0..n) |row| {
        for (0..n) |col| {
            if (ref.triBandIndexRowMajor(uplo, k, lda, row, col)) |idx| {
                a[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) a[idx] = ref.add(T, a[idx], ref.fromParts(T, 2.0 + @as(f64, @floatFromInt(row)), -0.25));
            }
        }
    }
}

fn fillTriPacked(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, ap: []T, n: usize) void {
    for (0..ap.len) |i| ap[i] = ref.fromParts(T, 5000 + @as(f64, @floatFromInt(i)), -5000);
    for (0..n) |col| {
        for (0..n) |row| {
            if (ref.triPackedIndex(uplo, n, row, col)) |idx| {
                ap[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) ap[idx] = ref.add(T, ap[idx], ref.fromParts(T, 2.5 + @as(f64, @floatFromInt(row)), 0.5));
            }
        }
    }
}

fn fillTriPackedRowMajor(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, ap: []T, n: usize) void {
    for (0..ap.len) |i| ap[i] = ref.fromParts(T, 5100 + @as(f64, @floatFromInt(i)), -5100);
    for (0..n) |row| {
        for (0..n) |col| {
            if (ref.triPackedIndexRowMajor(uplo, n, row, col)) |idx| {
                ap[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) ap[idx] = ref.add(T, ap[idx], ref.fromParts(T, 2.5 + @as(f64, @floatFromInt(row)), 0.5));
            }
        }
    }
}

fn complexAdd(a: cblas.ComplexF32, b: cblas.ComplexF32) cblas.ComplexF32 {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn complexMul(a: cblas.ComplexF32, b: cblas.ComplexF32) cblas.ComplexF32 {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}

fn referenceRowMajorCgemm(m: usize, n: usize, k: usize, alpha: cblas.ComplexF32, a: []const cblas.ComplexF32, lda: usize, b: []const cblas.ComplexF32, ldb: usize, beta: cblas.ComplexF32, c: []cblas.ComplexF32, ldc: usize) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var sum = complexF32(0, 0);
            for (0..k) |p| {
                sum = complexAdd(sum, complexMul(a[row * lda + p], b[p * ldb + col]));
            }
            const idx = row * ldc + col;
            c[idx] = complexAdd(complexMul(alpha, sum), complexMul(beta, c[idx]));
        }
    }
}

test "cblas row-major dgemm wrapper" {
    var left_matrix = [_]f64{
        1, 3, 5,
        2, 4, 6,
    };
    var right_matrix = [_]f64{
        7, 10,
        8, 11,
        9, 12,
    };
    var result_matrix = [_]f64{ 0, 0, 0, 0 };
    cblas.cblas_dgemm(cblas.CblasRowMajor, cblas.CblasNoTrans, cblas.CblasNoTrans, 2, 2, 3, 1, &left_matrix, 3, &right_matrix, 2, 0, &result_matrix, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 76), result_matrix[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 103), result_matrix[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), result_matrix[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 136), result_matrix[3], 1e-12);
}

test "cblas row-major cgemm wrapper" {
    const alpha = complexF32(0.75, -0.25);
    const beta = complexF32(-0.5, 0.25);
    var left_matrix = [_]cblas.ComplexF32{
        complexF32(1, 1),  complexF32(-3, 0.5), complexF32(0.25, -2),
        complexF32(2, -1), complexF32(4, 2),    complexF32(-1, 1),
    };
    var right_matrix = [_]cblas.ComplexF32{
        complexF32(2, 0),      complexF32(1, -2),
        complexF32(-1, 1),     complexF32(3, 1),
        complexF32(0.5, -0.5), complexF32(-2, 0.25),
    };
    var result_matrix = [_]cblas.ComplexF32{
        complexF32(1, 0.5), complexF32(0, -1),
        complexF32(-2, 1),  complexF32(3, 2),
    };
    var expected = result_matrix;
    referenceRowMajorCgemm(2, 2, 3, alpha, &left_matrix, 3, &right_matrix, 2, beta, &expected, 2);

    cblas.cblas_cgemm(cblas.CblasRowMajor, cblas.CblasNoTrans, cblas.CblasNoTrans, 2, 2, 3, &alpha, &left_matrix, 3, &right_matrix, 2, &beta, &result_matrix, 2);
    for (expected, result_matrix) |want, got| {
        try std.testing.expectApproxEqAbs(want.re, got.re, 1e-4);
        try std.testing.expectApproxEqAbs(want.im, got.im, 1e-4);
    }
}

test "cblas complex scal with complex alpha supports strides" {
    try expectCblasComplexScalCase(cblas.ComplexF32, 9, 1, complexF32(-0.75, 0.5), @as(f32, 1e-5));
    try expectCblasComplexScalCase(cblas.ComplexF32, 7, 2, complexF32(0.25, -1.25), @as(f32, 1e-5));
    try expectCblasComplexScalCase(cblas.ComplexF32, 7, -2, complexF32(1.5, 0.375), @as(f32, 1e-5));
    try expectCblasComplexScalCase(cblas.ComplexF64, 9, 1, complexF64(-0.75, 0.5), @as(f64, 1e-12));
    try expectCblasComplexScalCase(cblas.ComplexF64, 7, 2, complexF64(0.25, -1.25), @as(f64, 1e-12));
    try expectCblasComplexScalCase(cblas.ComplexF64, 7, -2, complexF64(1.5, 0.375), @as(f64, 1e-12));
}

test "cblas complex axpy and axpby with complex alpha support strides" {
    try expectCblasComplexAxpyCase(cblas.ComplexF32, 8, 1, 1, complexF32(-0.75, 0.5), @as(f32, 1e-5));
    try expectCblasComplexAxpyCase(cblas.ComplexF32, 7, 2, -2, complexF32(0.25, -1.25), @as(f32, 1e-5));
    try expectCblasComplexAxpyCase(cblas.ComplexF64, 8, 1, 1, complexF64(-0.75, 0.5), @as(f64, 1e-12));
    try expectCblasComplexAxpyCase(cblas.ComplexF64, 7, 2, -2, complexF64(0.25, -1.25), @as(f64, 1e-12));

    try expectCblasComplexAxpbyCase(cblas.ComplexF32, 8, 1, 1, complexF32(-0.75, 0.5), complexF32(0.25, -0.125), @as(f32, 1e-5));
    try expectCblasComplexAxpbyCase(cblas.ComplexF32, 7, -2, 2, complexF32(0.25, -1.25), complexF32(-0.5, 0.75), @as(f32, 1e-5));
    try expectCblasComplexAxpbyCase(cblas.ComplexF64, 8, 1, 1, complexF64(-0.75, 0.5), complexF64(0.25, -0.125), @as(f64, 1e-12));
    try expectCblasComplexAxpbyCase(cblas.ComplexF64, 7, -2, 2, complexF64(0.25, -1.25), complexF64(-0.5, 0.75), @as(f64, 1e-12));
}

test "cblas amax empty input returns cblas zero index" {
    var input_vector = [_]f64{1};
    try std.testing.expectEqual(@as(c_int, 0), cblas.cblas_idamax(0, &input_vector, 1));
    try std.testing.expectEqual(@as(c_int, 0), cblas.cblas_idamax(1, &input_vector, 0));
}

test "cblas row-major complex gemv conjugate transpose" {
    const one = cblas.ComplexF32{ .re = 1, .im = 0 };
    const zero = cblas.ComplexF32{ .re = 0, .im = 0 };
    var matrix = [_]cblas.ComplexF32{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 },
        .{ .re = 3, .im = 2 }, .{ .re = -1, .im = 1 },
    };
    var input_vector = [_]cblas.ComplexF32{
        .{ .re = 1, .im = -1 },
        .{ .re = 2, .im = 0 },
    };
    var result_vector = [_]cblas.ComplexF32{ zero, zero };
    cblas.cblas_cgemv(
        cblas.CblasRowMajor,
        cblas.CblasConjTrans,
        2,
        2,
        &one,
        &matrix,
        2,
        &input_vector,
        1,
        &zero,
        &result_vector,
        1,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 6), result_vector[0].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6), result_vector[0].im, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result_vector[1].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -3), result_vector[1].im, 1e-5);
}

test "cblas row-major level2 double wrappers" {
    var matrix = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };
    var x3 = [_]f64{ 1, 2, 3 };
    var y2 = [_]f64{ 0, 0 };
    cblas.cblas_dgemv(cblas.CblasRowMajor, cblas.CblasNoTrans, 2, 3, 1, &matrix, 3, &x3, 1, 0, &y2, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 14), y2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 32), y2[1], 1e-12);

    var sym = [_]f64{
        1,   2,
        777, 3,
    };
    var xsym = [_]f64{ 4, 5 };
    var ysym = [_]f64{ 0, 0 };
    cblas.cblas_dsymv(cblas.CblasRowMajor, cblas.CblasUpper, 2, 1, &sym, 2, &xsym, 1, 0, &ysym, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 14), ysym[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 23), ysym[1], 1e-12);

    var tri = [_]f64{
        1,   2,
        777, 3,
    };
    var tx = [_]f64{ 4, 5 };
    cblas.cblas_dtrmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasNoTrans, cblas.CblasNonUnit, 2, &tri, 2, &tx, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 14), tx[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), tx[1], 1e-12);

    var solve = [_]f64{ 14, 15 };
    cblas.cblas_dtrsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasNoTrans, cblas.CblasNonUnit, 2, &tri, 2, &solve, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 4), solve[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), solve[1], 1e-12);

    var ger_x = [_]f64{ 1, 2 };
    var ger_y = [_]f64{ 3, 4, 5 };
    var ger_a = [_]f64{
        0, 0, 0,
        0, 0, 0,
    };
    cblas.cblas_dger(cblas.CblasRowMajor, 2, 3, 2, &ger_x, 1, &ger_y, 1, &ger_a, 3);
    try std.testing.expectApproxEqAbs(@as(f64, 6), ger_a[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8), ger_a[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10), ger_a[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12), ger_a[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16), ger_a[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20), ger_a[5], 1e-12);
}

test "cblas row-major level3 triangular and rank2k wrappers" {
    var tri = [_]f64{
        1,   2,
        777, 3,
    };
    var b = [_]f64{
        4, 5,
        6, 7,
    };
    cblas.cblas_dtrmm(cblas.CblasRowMajor, cblas.CblasLeft, cblas.CblasUpper, cblas.CblasNoTrans, cblas.CblasNonUnit, 2, 2, 1, &tri, 2, &b, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 16), b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19), b[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18), b[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 21), b[3], 1e-12);

    cblas.cblas_dtrsm(cblas.CblasRowMajor, cblas.CblasLeft, cblas.CblasUpper, cblas.CblasNoTrans, cblas.CblasNonUnit, 2, 2, 1, &tri, 2, &b, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 4), b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), b[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), b[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7), b[3], 1e-12);

    var a_rank = [_]f64{
        1, 2,
        3, 4,
    };
    var b_rank = [_]f64{
        5, 6,
        7, 8,
    };
    var c_rank = [_]f64{
        0,   0,
        777, 0,
    };
    cblas.cblas_dsyr2k(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasNoTrans, 2, 2, 1, &a_rank, 2, &b_rank, 2, 0, &c_rank, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 34), c_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 62), c_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 777), c_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 106), c_rank[3], 1e-12);

    const z_alpha = complexF64(1, 0);
    var za_rank = [_]cblas.ComplexF64{
        complexF64(1, 0), complexF64(2, 0),
        complexF64(3, 0), complexF64(4, 0),
    };
    var zb_rank = [_]cblas.ComplexF64{
        complexF64(5, 0), complexF64(6, 0),
        complexF64(7, 0), complexF64(8, 0),
    };
    var zc_rank = [_]cblas.ComplexF64{
        complexF64(0, 0),   complexF64(0, 0),
        complexF64(777, 7), complexF64(0, 0),
    };
    cblas.cblas_zher2k(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasNoTrans, 2, 2, &z_alpha, &za_rank, 2, &zb_rank, 2, 0, &zc_rank, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 34), zc_rank[0].re, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), zc_rank[0].im, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 62), zc_rank[1].re, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), zc_rank[1].im, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 777), zc_rank[2].re, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7), zc_rank[2].im, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 106), zc_rank[3].re, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), zc_rank[3].im, 1e-12);
}

test "cblas row-major zgemv conjugate transpose strided reference" {
    const T = cblas.ComplexF64;
    const m = 3;
    const n = 2;
    const lda = 4;
    const incx: isize = 2;
    const incy: isize = -2;
    var rng = ref.Rng.init(0x5eed_0101);
    var a: [m * lda]T = undefined;
    var x: [ref.vectorStorageLen(m, incx)]T = undefined;
    var y: [ref.vectorStorageLen(n, incy)]T = undefined;
    ref.fillRowMajor(T, &rng, &a, m, n, lda);
    ref.fillVector(T, &rng, &x, m, incx);
    ref.fillVector(T, &rng, &y, n, incy);

    const alpha = rng.scalar(T);
    const beta = rng.scalar(T);
    var expected = y;
    ref.gemvRowMajor(T, .conj_trans, m, n, alpha, &a, lda, &x, incx, beta, &expected, incy);

    cblas.cblas_zgemv(cblas.CblasRowMajor, cblas.CblasConjTrans, m, n, &alpha, &a, lda, &x, incx, &beta, &y, incy);
    try expectComplexF64SliceApprox(&expected, &y);
}

test "cblas row-major zgbmv conjugate transpose band reference" {
    const T = cblas.ComplexF64;
    const m = 4;
    const n = 3;
    const kl = 2;
    const ku = 1;
    const lda = 5;
    const incx: isize = -2;
    const incy: isize = 2;
    var rng = ref.Rng.init(0x5eed_0111);
    var a: [m * lda]T = undefined;
    var x: [ref.vectorStorageLen(m, incx)]T = undefined;
    var y: [ref.vectorStorageLen(n, incy)]T = undefined;
    fillRowMajorBand(T, &rng, &a, m, n, kl, ku, lda);
    ref.fillVector(T, &rng, &x, m, incx);
    ref.fillVector(T, &rng, &y, n, incy);

    const alpha = complexF64(0.75, -0.5);
    const beta = complexF64(-0.25, 0.125);
    var expected = y;
    ref.gbmvRowMajor(T, .conj_trans, m, n, kl, ku, alpha, &a, lda, &x, incx, beta, &expected, incy);

    cblas.cblas_zgbmv(cblas.CblasRowMajor, cblas.CblasConjTrans, m, n, kl, ku, &alpha, &a, lda, &x, incx, &beta, &y, incy);
    try expectComplexF64SliceApprox(&expected, &y);
}

test "cblas row-major zgemm conjugate transpose padded reference" {
    const T = cblas.ComplexF64;
    const m = 2;
    const n = 3;
    const k = 4;
    const lda = 4;
    const ldb = 5;
    const ldc = 5;
    var rng = ref.Rng.init(0x5eed_0112);
    var a: [k * lda]T = undefined;
    var b: [k * ldb]T = undefined;
    var c: [m * ldc]T = undefined;
    ref.fillRowMajor(T, &rng, &a, k, m, lda);
    ref.fillRowMajor(T, &rng, &b, k, n, ldb);
    ref.fillRowMajor(T, &rng, &c, m, n, ldc);

    const alpha = complexF64(0.6, -0.4);
    const beta = complexF64(-0.2, 0.3);
    var expected = c;
    ref.gemmRowMajor(T, .conj_trans, .no_trans, m, n, k, alpha, &a, lda, &b, ldb, beta, &expected, ldc);

    cblas.cblas_zgemm(cblas.CblasRowMajor, cblas.CblasConjTrans, cblas.CblasNoTrans, m, n, k, &alpha, &a, lda, &b, ldb, &beta, &c, ldc);
    try expectComplexF64SliceApprox(&expected, &c);
}

test "cblas row-major zgeru and zgerc strided reference" {
    const T = cblas.ComplexF64;
    const m = 3;
    const n = 2;
    const lda = 4;
    var rng = ref.Rng.init(0x5eed_0202);

    var geru_a: [m * lda]T = undefined;
    var geru_x: [ref.vectorStorageLen(m, -2)]T = undefined;
    var geru_y: [ref.vectorStorageLen(n, 2)]T = undefined;
    ref.fillRowMajor(T, &rng, &geru_a, m, n, lda);
    ref.fillVector(T, &rng, &geru_x, m, -2);
    ref.fillVector(T, &rng, &geru_y, n, 2);
    const geru_alpha = rng.scalar(T);
    var geru_expected = geru_a;
    ref.gerRowMajor(T, m, n, geru_alpha, &geru_x, -2, &geru_y, 2, &geru_expected, lda, false);

    cblas.cblas_zgeru(cblas.CblasRowMajor, m, n, &geru_alpha, &geru_x, -2, &geru_y, 2, &geru_a, lda);
    try expectComplexF64SliceApprox(&geru_expected, &geru_a);

    var gerc_a: [m * lda]T = undefined;
    var gerc_x: [ref.vectorStorageLen(m, 2)]T = undefined;
    var gerc_y: [ref.vectorStorageLen(n, -1)]T = undefined;
    ref.fillRowMajor(T, &rng, &gerc_a, m, n, lda);
    ref.fillVector(T, &rng, &gerc_x, m, 2);
    ref.fillVector(T, &rng, &gerc_y, n, -1);
    const gerc_alpha = rng.scalar(T);
    var gerc_expected = gerc_a;
    ref.gerRowMajor(T, m, n, gerc_alpha, &gerc_x, 2, &gerc_y, -1, &gerc_expected, lda, true);

    cblas.cblas_zgerc(cblas.CblasRowMajor, m, n, &gerc_alpha, &gerc_x, 2, &gerc_y, -1, &gerc_a, lda);
    try expectComplexF64SliceApprox(&gerc_expected, &gerc_a);
}

test "cblas col-major complex packed banded level2 reference" {
    const T = cblas.ComplexF64;
    const n = 4;
    const k = 2;
    const lda = 4;
    var rng = ref.Rng.init(0x5eed_0222);

    var hb: [n * lda]T = undefined;
    var hp: [n * (n + 1) / 2]T = undefined;
    var hx: [ref.vectorStorageLen(n, -1)]T = undefined;
    var hy_band: [ref.vectorStorageLen(n, 2)]T = undefined;
    var hy_packed: [ref.vectorStorageLen(n, 2)]T = undefined;
    fillColMajorSymBand(T, &rng, .upper, &hb, n, k, lda);
    fillPackedSym(T, &rng, .upper, &hp, n);
    ref.fillVector(T, &rng, &hx, n, -1);
    ref.fillVector(T, &rng, &hy_band, n, 2);
    ref.fillVector(T, &rng, &hy_packed, n, 2);
    const alpha = complexF64(0.4, -0.6);
    const beta = complexF64(-0.25, 0.5);
    var expected_band = hy_band;
    var expected_packed = hy_packed;
    ref.sbmvColMajor(T, .upper, n, k, alpha, &hb, lda, &hx, -1, beta, &expected_band, 2, true);
    ref.spmvColMajor(T, .upper, n, alpha, &hp, &hx, -1, beta, &expected_packed, 2, true);
    cblas.cblas_zhbmv(cblas.CblasColMajor, cblas.CblasUpper, n, k, &alpha, &hb, lda, &hx, -1, &beta, &hy_band, 2);
    cblas.cblas_zhpmv(cblas.CblasColMajor, cblas.CblasUpper, n, &alpha, &hp, &hx, -1, &beta, &hy_packed, 2);
    try expectComplexF64SliceApprox(&expected_band, &hy_band);
    try expectComplexF64SliceApprox(&expected_packed, &hy_packed);

    var tb: [n * lda]T = undefined;
    var tp: [n * (n + 1) / 2]T = undefined;
    var tx: [ref.vectorStorageLen(n, -2)]T = undefined;
    var tpx: [ref.vectorStorageLen(n, 1)]T = undefined;
    var work: [n]T = undefined;
    fillColMajorTriBand(T, &rng, .lower, .non_unit, &tb, n, k, lda);
    fillTriPacked(T, &rng, .upper, .unit, &tp, n);
    ref.fillVector(T, &rng, &tx, n, -2);
    ref.fillVector(T, &rng, &tpx, n, 1);
    var expected_tb = tx;
    var expected_tp = tpx;
    ref.tbmvColMajor(T, .lower, .conj_trans, .non_unit, n, k, &tb, lda, &expected_tb, -2, &work);
    ref.tpmvColMajor(T, .upper, .conj_trans, .unit, n, &tp, &expected_tp, 1, &work);
    cblas.cblas_ztbmv(cblas.CblasColMajor, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasNonUnit, n, k, &tb, lda, &tx, -2);
    cblas.cblas_ztpmv(cblas.CblasColMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, n, &tp, &tpx, 1);
    try expectComplexF64SliceApprox(&expected_tb, &tx);
    try expectComplexF64SliceApprox(&expected_tp, &tpx);

    var tb_solve = expected_tb;
    var tp_solve = expected_tp;
    ref.tbsvColMajor(T, .lower, .conj_trans, .non_unit, n, k, &tb, lda, &tb_solve, -2);
    ref.tpsvColMajor(T, .upper, .conj_trans, .unit, n, &tp, &tp_solve, 1);
    cblas.cblas_ztbsv(cblas.CblasColMajor, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasNonUnit, n, k, &tb, lda, &expected_tb, -2);
    cblas.cblas_ztpsv(cblas.CblasColMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, n, &tp, &expected_tp, 1);
    try expectComplexF64SliceApprox(&tb_solve, &expected_tb);
    try expectComplexF64SliceApprox(&tp_solve, &expected_tp);
}

test "cblas row-major complex banded packed triangular conjugate transpose reference" {
    const T = cblas.ComplexF64;
    const n = 5;
    const k = 2;
    const lda = k + 1;
    var rng = ref.Rng.init(0x5eed_0302);

    var tb: [n * lda]T = undefined;
    var tp: [n * (n + 1) / 2]T = undefined;
    var tx: [ref.vectorStorageLen(n, 2)]T = undefined;
    var tpx: [ref.vectorStorageLen(n, -1)]T = undefined;
    var work: [n]T = undefined;
    fillRowMajorTriBand(T, &rng, .upper, .non_unit, &tb, n, k, lda);
    fillTriPackedRowMajor(T, &rng, .lower, .unit, &tp, n);
    ref.fillVector(T, &rng, &tx, n, 2);
    ref.fillVector(T, &rng, &tpx, n, -1);

    var expected_tb = tx;
    var expected_tp = tpx;
    ref.tbmvRowMajor(T, .upper, .conj_trans, .non_unit, n, k, &tb, lda, &expected_tb, 2, &work);
    ref.tpmvRowMajor(T, .lower, .conj_trans, .unit, n, &tp, &expected_tp, -1, &work);
    cblas.cblas_ztbmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasNonUnit, n, k, &tb, lda, &tx, 2);
    cblas.cblas_ztpmv(cblas.CblasRowMajor, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasUnit, n, &tp, &tpx, -1);
    try expectComplexF64SliceApprox(&expected_tb, &tx);
    try expectComplexF64SliceApprox(&expected_tp, &tpx);

    var tb_solve = expected_tb;
    var tp_solve = expected_tp;
    ref.tbsvRowMajor(T, .upper, .conj_trans, .non_unit, n, k, &tb, lda, &tb_solve, 2);
    ref.tpsvRowMajor(T, .lower, .conj_trans, .unit, n, &tp, &tp_solve, -1);
    cblas.cblas_ztbsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasNonUnit, n, k, &tb, lda, &expected_tb, 2);
    cblas.cblas_ztpsv(cblas.CblasRowMajor, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasUnit, n, &tp, &expected_tp, -1);
    try expectComplexF64SliceApprox(&tb_solve, &expected_tb);
    try expectComplexF64SliceApprox(&tp_solve, &expected_tp);
}

test "cblas row-major ztrmv conjugate transpose strided reference" {
    const T = cblas.ComplexF64;
    const n = 4;
    const lda = 5;
    const incx: isize = 2;
    var rng = ref.Rng.init(0x5eed_0303);
    var a: [n * lda]T = undefined;
    var x: [ref.vectorStorageLen(n, incx)]T = undefined;
    var work: [n]T = undefined;
    ref.fillRowMajor(T, &rng, &a, n, n, lda);
    ref.fillVector(T, &rng, &x, n, incx);
    var expected = x;
    ref.trmvRowMajor(T, .lower, .conj_trans, .unit, n, &a, lda, &expected, incx, &work);

    cblas.cblas_ztrmv(cblas.CblasRowMajor, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasUnit, n, &a, lda, &x, incx);
    try expectComplexF64SliceApprox(&expected, &x);
}

test "cblas row-major ztrsv conjugate transpose negative stride reference" {
    const T = cblas.ComplexF64;
    const n = 4;
    const lda = 5;
    const incx: isize = -2;
    var rng = ref.Rng.init(0x5eed_0304);
    var a: [n * lda]T = undefined;
    var x: [ref.vectorStorageLen(n, incx)]T = undefined;
    ref.fillRowMajor(T, &rng, &a, n, n, lda);
    makeTriangularDiagSafe(&a, n, lda);
    ref.fillVector(T, &rng, &x, n, incx);
    var expected = x;
    ref.trsvRowMajor(T, .upper, .conj_trans, .non_unit, n, &a, lda, &expected, incx);

    cblas.cblas_ztrsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasNonUnit, n, &a, lda, &x, incx);
    try expectComplexF64SliceApprox(&expected, &x);
}

test "cblas row-major ztrmm and ztrsm conjugate transpose reference" {
    const T = cblas.ComplexF64;
    var rng = ref.Rng.init(0x5eed_0404);

    const trmm_m = 3;
    const trmm_n = 2;
    const trmm_lda = 5;
    const trmm_ldb = 4;
    const trmm_alpha = complexF64(0.7, -0.3);
    var trmm_a: [trmm_m * trmm_lda]T = undefined;
    var trmm_b: [trmm_m * trmm_ldb]T = undefined;
    var trmm_work: [trmm_m * trmm_n]T = undefined;
    ref.fillRowMajor(T, &rng, &trmm_a, trmm_m, trmm_m, trmm_lda);
    makeTriangularDiagSafe(&trmm_a, trmm_m, trmm_lda);
    ref.fillRowMajor(T, &rng, &trmm_b, trmm_m, trmm_n, trmm_ldb);
    var trmm_expected = trmm_b;
    ref.trmmRowMajor(T, .left, .lower, .conj_trans, .non_unit, trmm_m, trmm_n, trmm_alpha, &trmm_a, trmm_lda, &trmm_expected, trmm_ldb, &trmm_work);

    cblas.cblas_ztrmm(cblas.CblasRowMajor, cblas.CblasLeft, cblas.CblasLower, cblas.CblasConjTrans, cblas.CblasNonUnit, trmm_m, trmm_n, &trmm_alpha, &trmm_a, trmm_lda, &trmm_b, trmm_ldb);
    try expectComplexF64SliceApprox(&trmm_expected, &trmm_b);

    const trsm_m = 2;
    const trsm_n = 3;
    const trsm_lda = 5;
    const trsm_ldb = 4;
    const trsm_alpha = complexF64(-0.5, 0.8);
    var trsm_a: [trsm_n * trsm_lda]T = undefined;
    var trsm_b: [trsm_m * trsm_ldb]T = undefined;
    var trsm_work: [trsm_m * trsm_n]T = undefined;
    ref.fillRowMajor(T, &rng, &trsm_a, trsm_n, trsm_n, trsm_lda);
    ref.fillRowMajor(T, &rng, &trsm_b, trsm_m, trsm_n, trsm_ldb);
    var trsm_expected = trsm_b;
    ref.trsmRowMajor(T, .right, .upper, .conj_trans, .unit, trsm_m, trsm_n, trsm_alpha, &trsm_a, trsm_lda, &trsm_expected, trsm_ldb, &trsm_work);

    cblas.cblas_ztrsm(cblas.CblasRowMajor, cblas.CblasRight, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, trsm_m, trsm_n, &trsm_alpha, &trsm_a, trsm_lda, &trsm_b, trsm_ldb);
    try expectComplexF64SliceApprox(&trsm_expected, &trsm_b);
}
