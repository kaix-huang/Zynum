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

fn fillColMajorBand(comptime T: type, rng: *ref.Rng, a: []T, m: usize, n: usize, kl: usize, ku: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, 1500 + @as(f64, @floatFromInt(i)), -1500);
    for (0..n) |col| {
        for (0..m) |row| {
            if (ref.gbIndexColMajor(m, n, kl, ku, lda, row, col)) |idx| a[idx] = rng.scalar(T);
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

fn cblasTrans(trans: ref.Trans) c_int {
    return switch (trans) {
        .no_trans => cblas.CblasNoTrans,
        .trans => cblas.CblasTrans,
        .conj_trans => cblas.CblasConjTrans,
    };
}

fn expectRowMajorRealGemmCase(comptime T: type, transa: ref.Trans, transb: ref.Trans) !void {
    const m: usize = 35;
    const n: usize = 37;
    const k: usize = 67;
    const a_rows = if (transa == .no_trans) m else k;
    const a_cols = if (transa == .no_trans) k else m;
    const b_rows = if (transb == .no_trans) k else n;
    const b_cols = if (transb == .no_trans) n else k;
    const lda = a_cols + 4;
    const ldb = b_cols + 4;
    const ldc = n + 5;

    const a = try std.testing.allocator.alloc(T, a_rows * lda);
    defer std.testing.allocator.free(a);
    const b = try std.testing.allocator.alloc(T, b_rows * ldb);
    defer std.testing.allocator.free(b);
    const c = try std.testing.allocator.alloc(T, m * ldc);
    defer std.testing.allocator.free(c);
    const expected = try std.testing.allocator.alloc(T, m * ldc);
    defer std.testing.allocator.free(expected);

    var rng = ref.Rng.init(0x74a3_119d);
    ref.fillRowMajor(T, &rng, a, a_rows, a_cols, lda);
    ref.fillRowMajor(T, &rng, b, b_rows, b_cols, ldb);
    ref.fillRowMajor(T, &rng, c, m, n, ldc);
    @memcpy(expected, c);

    const alpha: T = -0.5;
    const beta: T = 2.0;
    if (T == f32) {
        cblas.cblas_sgemm(cblas.CblasRowMajor, cblasTrans(transa), cblasTrans(transb), @intCast(m), @intCast(n), @intCast(k), alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), beta, c.ptr, @intCast(ldc));
    } else if (T == f64) {
        cblas.cblas_dgemm(cblas.CblasRowMajor, cblasTrans(transa), cblasTrans(transb), @intCast(m), @intCast(n), @intCast(k), alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), beta, c.ptr, @intCast(ldc));
    } else {
        @compileError("real GEMM test supports f32 and f64");
    }
    ref.gemmRowMajor(T, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    const tol: T = if (T == f32) 1e-3 else 1e-10;
    for (expected, c) |want, got| try std.testing.expectApproxEqAbs(want, got, tol);
}

fn expectRowMajorComplexGemmCase(comptime T: type, transa: ref.Trans, transb: ref.Trans, m: usize, n: usize, k: usize, alpha: T, beta: T, tol: anytype) !void {
    const a_rows = if (transa == .no_trans) m else k;
    const a_cols = if (transa == .no_trans) k else m;
    const b_rows = if (transb == .no_trans) k else n;
    const b_cols = if (transb == .no_trans) n else k;
    const lda = a_cols + 4;
    const ldb = b_cols + 5;
    const ldc = n + 6;

    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, a_rows * lda);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_rows * ldb);
    defer allocator.free(b);
    const c = try allocator.alloc(T, m * ldc);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c.len);
    defer allocator.free(expected);

    var rng = ref.Rng.init(0xc31a_5eed);
    ref.fillRowMajor(T, &rng, a, a_rows, a_cols, lda);
    ref.fillRowMajor(T, &rng, b, b_rows, b_cols, ldb);
    ref.fillRowMajor(T, &rng, c, m, n, ldc);
    @memcpy(expected, c);

    if (T == cblas.ComplexF32) {
        cblas.cblas_cgemm(cblas.CblasRowMajor, cblasTrans(transa), cblasTrans(transb), @intCast(m), @intCast(n), @intCast(k), &alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), &beta, c.ptr, @intCast(ldc));
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zgemm(cblas.CblasRowMajor, cblasTrans(transa), cblasTrans(transb), @intCast(m), @intCast(n), @intCast(k), &alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), &beta, c.ptr, @intCast(ldc));
    } else {
        @compileError("complex GEMM test supports ComplexF32 and ComplexF64");
    }
    ref.gemmRowMajor(T, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    for (expected, c) |want, got| try ref.expectApprox(T, want, got, tol);
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

test "cblas row-major real GEMM transpose pairs preserve operand semantics" {
    inline for (.{ f32, f64 }) |T| {
        try expectRowMajorRealGemmCase(T, .no_trans, .no_trans);
        try expectRowMajorRealGemmCase(T, .no_trans, .trans);
        try expectRowMajorRealGemmCase(T, .trans, .no_trans);
        try expectRowMajorRealGemmCase(T, .trans, .trans);
    }
}

test "cblas row-major complex GEMM non-NN alpha beta cases stay on fallback" {
    const pairs = [_]struct { transa: ref.Trans, transb: ref.Trans }{
        .{ .transa = .no_trans, .transb = .trans },
        .{ .transa = .no_trans, .transb = .conj_trans },
        .{ .transa = .trans, .transb = .no_trans },
        .{ .transa = .trans, .transb = .trans },
        .{ .transa = .trans, .transb = .conj_trans },
        .{ .transa = .conj_trans, .transb = .no_trans },
        .{ .transa = .conj_trans, .transb = .trans },
        .{ .transa = .conj_trans, .transb = .conj_trans },
    };
    inline for (.{ cblas.ComplexF32, cblas.ComplexF64 }) |T| {
        const tol = if (T == cblas.ComplexF32) @as(f32, 4e-2) else @as(f64, 1e-9);
        for (pairs) |pair| {
            try expectRowMajorComplexGemmCase(
                T,
                pair.transa,
                pair.transb,
                17,
                19,
                409,
                ref.fromParts(T, 0.75, -0.25),
                ref.fromParts(T, -0.5, 0.125),
                tol,
            );
        }
    }
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

fn expectUnitGbmvCase(comptime T: type, trans: ref.Trans) !void {
    const m: usize = 31;
    const n: usize = 27;
    const kl: usize = 9;
    const ku: usize = 7;
    const lda: usize = kl + ku + 4;
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    var rng = ref.Rng.init(@as(u64, 0xbaad_1001) + @backingInt(trans));
    const a = try std.testing.allocator.alloc(T, lda * n);
    defer std.testing.allocator.free(a);
    const x = try std.testing.allocator.alloc(T, lenx);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(T, leny);
    defer std.testing.allocator.free(y);
    const expected = try std.testing.allocator.alloc(T, leny);
    defer std.testing.allocator.free(expected);

    fillColMajorBand(T, &rng, a, m, n, kl, ku, lda);
    ref.fillVector(T, &rng, x, lenx, 1);
    ref.fillVector(T, &rng, y, leny, 1);
    @memcpy(expected, y);
    const alpha = if (comptime ref.isComplex(T)) ref.fromParts(T, 0.75, -0.375) else @as(T, 0.75);
    const beta = if (comptime ref.isComplex(T)) ref.fromParts(T, -0.25, 0.125) else @as(T, -0.25);
    ref.gbmvColMajor(T, trans, m, n, kl, ku, alpha, a, lda, x, 1, beta, expected, 1);

    if (T == f64) {
        cblas.cblas_dgbmv(cblas.CblasColMajor, cblasTrans(trans), m, n, kl, ku, alpha, a.ptr, lda, x.ptr, 1, beta, y.ptr, 1);
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zgbmv(cblas.CblasColMajor, cblasTrans(trans), m, n, kl, ku, &alpha, a.ptr, lda, x.ptr, 1, &beta, y.ptr, 1);
    } else {
        @compileError("unit GBMV test supports f64 and ComplexF64");
    }
    for (expected, y) |want, got| try ref.expectApprox(T, want, got, @as(f64, 1e-10));
}

fn expectUnitSbmvCase(comptime T: type, uplo: ref.Uplo, herm: bool) !void {
    const n: usize = 29;
    const k: usize = 7;
    const lda: usize = k + 3;
    var rng = ref.Rng.init(@as(u64, 0xbaad_2001) + @backingInt(uplo));
    const a = try std.testing.allocator.alloc(T, lda * n);
    defer std.testing.allocator.free(a);
    const x = try std.testing.allocator.alloc(T, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(T, n);
    defer std.testing.allocator.free(y);
    const expected = try std.testing.allocator.alloc(T, n);
    defer std.testing.allocator.free(expected);

    fillColMajorSymBand(T, &rng, uplo, a, n, k, lda);
    ref.fillVector(T, &rng, x, n, 1);
    ref.fillVector(T, &rng, y, n, 1);
    @memcpy(expected, y);
    const alpha = if (comptime ref.isComplex(T)) ref.fromParts(T, 0.625, -0.25) else @as(T, 0.625);
    const beta = if (comptime ref.isComplex(T)) ref.fromParts(T, -0.375, 0.125) else @as(T, -0.375);
    ref.sbmvColMajor(T, uplo, n, k, alpha, a, lda, x, 1, beta, expected, 1, herm);
    const cblas_uplo = if (uplo == .upper) cblas.CblasUpper else cblas.CblasLower;

    if (T == f64) {
        cblas.cblas_dsbmv(cblas.CblasColMajor, cblas_uplo, n, k, alpha, a.ptr, lda, x.ptr, 1, beta, y.ptr, 1);
    } else if (T == cblas.ComplexF64) {
        cblas.cblas_zhbmv(cblas.CblasColMajor, cblas_uplo, n, k, &alpha, a.ptr, lda, x.ptr, 1, &beta, y.ptr, 1);
    } else {
        @compileError("unit SBMV/HBMV test supports f64 and ComplexF64");
    }
    for (expected, y) |want, got| try ref.expectApprox(T, want, got, @as(f64, 1e-10));
}

test "cblas unit-stride banded column paths" {
    try expectUnitGbmvCase(f64, .no_trans);
    try expectUnitGbmvCase(f64, .trans);
    try expectUnitGbmvCase(cblas.ComplexF64, .no_trans);
    try expectUnitGbmvCase(cblas.ComplexF64, .trans);
    try expectUnitGbmvCase(cblas.ComplexF64, .conj_trans);
    try expectUnitSbmvCase(f64, .upper, false);
    try expectUnitSbmvCase(f64, .lower, false);
    try expectUnitSbmvCase(cblas.ComplexF64, .upper, true);
    try expectUnitSbmvCase(cblas.ComplexF64, .lower, true);
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

// Real conjugation is the identity; row-major C must use the same mapping as T.
test "cblas real row-major triangular conjugate transpose matches exact reference" {
    inline for (.{ f32, f64 }) |T| {
        for ([_]c_int{ cblas.CblasUpper, cblas.CblasLower }) |uplo| {
            for ([_]c_int{ cblas.CblasNonUnit, cblas.CblasUnit }) |diag| {
                var ap: [6]T = if (uplo == cblas.CblasUpper) .{ 2, 3, 5, 7, 11, 13 } else .{ 2, 3, 7, 5, 11, 13 };
                if (diag == cblas.CblasUnit) {
                    ap[0] = std.math.nan(T);
                    ap[if (uplo == cblas.CblasUpper) 3 else 2] = std.math.nan(T);
                    ap[5] = std.math.nan(T);
                }
                const want: [3]T = if (uplo == cblas.CblasUpper)
                    (if (diag == cblas.CblasUnit) .{ 1, 5, 30 } else .{ 2, 17, 66 })
                else
                    (if (diag == cblas.CblasUnit) .{ 22, 35, 3 } else .{ 23, 47, 39 });
                const nan = std.math.nan(T);
                const full: [9]T = if (uplo == cblas.CblasUpper)
                    .{ ap[0], ap[1], ap[2], nan, ap[3], ap[4], nan, nan, ap[5] }
                else
                    .{ ap[0], nan, nan, ap[1], ap[2], nan, ap[3], ap[4], ap[5] };
                const band: [9]T = if (uplo == cblas.CblasUpper)
                    .{ ap[0], ap[1], ap[2], ap[3], ap[4], nan, ap[5], nan, nan }
                else
                    .{ nan, nan, ap[0], nan, ap[1], ap[2], ap[3], ap[4], ap[5] };
                inline for (.{ "tpmv", "trmv", "tbmv", "tpsv", "trsv", "tbsv" }) |op| {
                    const solve = comptime op[2] == 's';
                    for ([_]c_int{ 1, -1, 2, -2 }) |inc| {
                        const stride: usize = @intCast(if (inc < 0) -inc else inc);
                        var x: [7]T = @splat(-123);
                        var expected = x;
                        for (0..3) |j| {
                            const index = 1 + (if (inc > 0) j else 2 - j) * stride;
                            x[index] = if (solve) want[j] else @floatFromInt(j + 1);
                            expected[index] = if (solve) @floatFromInt(j + 1) else want[j];
                        }
                        const function = @field(cblas, "cblas_" ++ (if (T == f32) "s" else "d") ++ op);
                        if (comptime op[1] == 'p') {
                            function(cblas.CblasRowMajor, uplo, cblas.CblasConjTrans, diag, 3, &ap, x[1..].ptr, inc);
                        } else if (comptime op[1] == 'b') {
                            function(cblas.CblasRowMajor, uplo, cblas.CblasConjTrans, diag, 3, 2, &band, 3, x[1..].ptr, inc);
                        } else {
                            function(cblas.CblasRowMajor, uplo, cblas.CblasConjTrans, diag, 3, &full, 3, x[1..].ptr, inc);
                        }
                        try std.testing.expectEqualSlices(T, &expected, &x);
                    }
                }
            }
        }
    }
}

// For real rank updates, conjugation leaves each input coefficient unchanged.
test "cblas real rank updates conjugate transpose independent reference" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ "syrk", "syr2k" }) |op| {
            for ([_]c_int{ cblas.CblasRowMajor, cblas.CblasColMajor }) |layout| {
                for ([_]c_int{ cblas.CblasUpper, cblas.CblasLower }) |uplo| {
                    for ([_]c_int{ cblas.CblasTrans, cblas.CblasConjTrans }) |trans| {
                        for ([_][2]usize{ .{ 2, 3 }, .{ 3, 2 } }) |shape| {
                            const n = shape[0];
                            const k = shape[1];
                            var a: [15]T = @splat(std.math.nan(T));
                            var b: [15]T = @splat(std.math.nan(T));
                            for (0..k) |l| {
                                for (0..n) |j| {
                                    const index = if (layout == cblas.CblasRowMajor) l * 5 + j else j * 5 + l;
                                    a[index] = @floatFromInt(1 + l * n + j);
                                    b[index] = @as(T, @floatFromInt(2 + l + 2 * j)) - 4;
                                }
                            }
                            for ([_]T{ 0, 1 }) |beta| {
                                var c: [15]T = @splat(-123);
                                for (0..n) |i| {
                                    for (0..n) |j| c[if (layout == cblas.CblasRowMajor) i * 5 + j else j * 5 + i] = 7;
                                }
                                var expected = c;
                                for (0..n) |i| {
                                    for (0..n) |j| {
                                        if ((uplo == cblas.CblasUpper and i > j) or (uplo == cblas.CblasLower and i < j)) continue;
                                        var sum: T = 0;
                                        for (0..k) |l| {
                                            const ai: T = @floatFromInt(1 + l * n + i);
                                            const aj: T = @floatFromInt(1 + l * n + j);
                                            if (comptime std.mem.eql(u8, op, "syrk")) {
                                                sum += ai * aj;
                                            } else {
                                                const bi = @as(T, @floatFromInt(2 + l + 2 * i)) - 4;
                                                const bj = @as(T, @floatFromInt(2 + l + 2 * j)) - 4;
                                                sum += ai * bj + bi * aj;
                                            }
                                        }
                                        expected[if (layout == cblas.CblasRowMajor) i * 5 + j else j * 5 + i] = 2 * sum + beta * 7;
                                    }
                                }
                                const function = @field(cblas, "cblas_" ++ (if (T == f32) "s" else "d") ++ op);
                                if (comptime std.mem.eql(u8, op, "syrk")) {
                                    function(layout, uplo, trans, @intCast(n), @intCast(k), 2, &a, 5, beta, &c, 5);
                                } else {
                                    function(layout, uplo, trans, @intCast(n), @intCast(k), 2, &a, 5, &b, 5, beta, &c, 5);
                                }
                                try std.testing.expectEqualSlices(T, &expected, &c);
                            }
                        }
                    }
                }
            }
        }
    }
}

// The build injects the existing Core's exact test Execution Module.
const admission_execution = @import("zynum-execution");

const CblasAdmissionObservation = struct {
    var entries: usize = 0;
    var finishes: usize = 0;
    var checked_calls: usize = 0;
    var actual: []const u8 = &.{};
    var expected: []const u8 = &.{};

    fn reset() void {
        entries = 0;
        finishes = 0;
        checked_calls = 0;
        actual = &.{};
        expected = &.{};
    }
    fn watch(comptime T: type, values: []const T, wanted: []const T) void {
        actual = std.mem.sliceAsBytes(values);
        expected = std.mem.sliceAsBytes(wanted);
    }
    fn call(ctx: *admission_execution.context.Context, event: admission_execution.context.TestEvent) void {
        if (ctx != admission_execution.getContext() or &ctx.executor != admission_execution.get())
            @panic("CBLAS ABI used a different default owner");
        if (ctx.snapshot().active_operations != 1) @panic("CBLAS ABI did not retain one operation");
        switch (event) {
            .admitted => entries += 1,
            .finishing => {
                if (!std.mem.eql(u8, actual, expected)) @panic("CBLAS operation ended before complete output or damaged padding");
                finishes += 1;
            },
        }
    }
    fn checkNext() !void {
        checked_calls += 1;
        try std.testing.expectEqual(checked_calls, entries);
        try std.testing.expectEqual(checked_calls, finishes);
        const state = admission_execution.getContext().snapshot();
        try std.testing.expectEqual(@as(usize, 0), state.active_operations);
        try std.testing.expectEqual(@as(usize, 0), state.resource_pins);
    }
};

test "CBLAS Fortran delegation borrows one operation through stores and returns" {
    const Observe = CblasAdmissionObservation;
    Observe.reset();
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    try std.testing.expectEqual(admission_execution.context.RequestResult.immortal, admission_execution.getContext().requestClose());

    // Same independently known matrix product in both layouts. Every padding
    // element is visible to finishing, not merely checked after the C call.
    for ([_]bool{ false, true }) |row_major| {
        const layout: c_int = if (row_major) cblas.CblasRowMajor else cblas.CblasColMajor;
        const a: [6]f64 = if (row_major) .{ 1, 2, 71, 3, 4, 72 } else .{ 1, 3, 71, 2, 4, 72 };
        const b: [6]f64 = if (row_major) .{ 5, 6, 73, 7, 8, 74 } else .{ 5, 7, 73, 6, 8, 74 };
        var c = [_]f64{ 0, 0, 81, 0, 0, 82 };
        const wanted: [6]f64 = if (row_major) .{ 19, 22, 81, 43, 50, 82 } else .{ 19, 43, 81, 22, 50, 82 };
        Observe.watch(f64, &c, &wanted);
        cblas.cblas_dgemm(layout, cblas.CblasNoTrans, cblas.CblasNoTrans, 2, 2, 2, 1, &a, 3, &b, 3, 0, &c, 3);
        try Observe.checkNext();
    }
    const C = cblas.ComplexF64;
    const x = [_]C{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
    const y = [_]C{ .{ .re = 2, .im = -1 }, .{ .re = -1, .im = 2 } };
    var dot = [_]C{ .{ .re = 0, .im = 0 }, .{ .re = 91, .im = 17 } };
    Observe.watch(C, &dot, &.{ .{ .re = 3, .im = 10 }, .{ .re = 91, .im = 17 } });
    cblas.cblas_zdotu_sub(2, &x, 1, &y, 1, &dot[0]);
    try Observe.checkNext();
    Observe.watch(C, &dot, &.{ .{ .re = -5, .im = 0 }, .{ .re = 91, .im = 17 } });
    cblas.cblas_zdotc_sub(2, &x, 1, &y, 1, &dot[0]);
    try Observe.checkNext();
    // The returned index is assigned after finishing; its conversion still
    // occurs under the outer CBLAS token. Fortran returns 2, CBLAS returns 1.
    Observe.watch(f64, &.{}, &.{});
    try std.testing.expectEqual(@as(c_int, 1), cblas.cblas_izamax(2, &x, 1));
    try Observe.checkNext();
    const real_x = [_]f64{ 1, 2 };
    const real_y = [_]f64{ 3, 4 };
    try std.testing.expectEqual(@as(f64, 11), cblas.cblas_ddot(2, &real_x, 1, &real_y, 1));
    try Observe.checkNext();
}

test "CBLAS row major complex local bodies remain inside one operation" {
    const Observe = CblasAdmissionObservation;
    Observe.reset();
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    const C = cblas.ComplexF64;
    const one = C{ .re = 1, .im = 0 };
    const zero = C{ .re = 0, .im = 0 };
    const pad = C{ .re = 91, .im = 19 };
    // A = [[1+i,2-i],[3+2i,4+i]], logical x = [1+i,2-i].
    // A^H x = [6-7i,8-3i]; input memory is reversed for incx=-1.
    const x = [_]C{ .{ .re = 2, .im = -1 }, .{ .re = 1, .im = 1 }, pad };
    const dense = [_]C{ .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 }, pad, .{ .re = 3, .im = 2 }, .{ .re = 4, .im = 1 }, pad };
    const band = [_]C{ pad, .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 }, pad, .{ .re = 3, .im = 2 }, .{ .re = 4, .im = 1 }, pad, pad };
    for ([_]bool{ false, true }) |banded| {
        var output = [_]C{ zero, pad, zero, pad };
        Observe.watch(C, &output, &.{ .{ .re = 6, .im = -7 }, pad, .{ .re = 8, .im = -3 }, pad });
        if (banded)
            cblas.cblas_zgbmv(cblas.CblasRowMajor, cblas.CblasConjTrans, 2, 2, 1, 1, &one, &band, 4, &x, -1, &zero, &output, 2)
        else
            cblas.cblas_zgemv(cblas.CblasRowMajor, cblas.CblasConjTrans, 2, 2, &one, &dense, 3, &x, -1, &zero, &output, 2);
        try Observe.checkNext();
    }

    // Unit diagonal is deliberately stored as a very different finite value.
    // T=[[1,2+i],[0,1]], T^H*[1+i,2-i]=[1+i,5]. These literals supply both
    // multiply and solve oracles, not a call through another product routine.
    const wrong_diagonal = C{ .re = 51, .im = 53 };
    const off_diagonal = C{ .re = 2, .im = 1 };
    const triangle = [_]C{ wrong_diagonal, off_diagonal, pad, pad, wrong_diagonal, pad };
    const triangle_band = [_]C{ wrong_diagonal, off_diagonal, pad, wrong_diagonal, pad, pad };
    const triangle_packed = [_]C{ wrong_diagonal, off_diagonal, wrong_diagonal, pad };
    const Path = enum { trmv, trsv, tbmv, tbsv, tpmv, tpsv };
    for ([_]Path{ .trmv, .trsv, .tbmv, .tbsv, .tpmv, .tpsv }) |path| {
        const solve = path == .trsv or path == .tbsv or path == .tpsv;
        var output: [4]C = if (solve) .{ .{ .re = 1, .im = 1 }, pad, .{ .re = 5, .im = 0 }, pad } else .{ .{ .re = 1, .im = 1 }, pad, .{ .re = 2, .im = -1 }, pad };
        const wanted: [4]C = if (solve) .{ .{ .re = 1, .im = 1 }, pad, .{ .re = 2, .im = -1 }, pad } else .{ .{ .re = 1, .im = 1 }, pad, .{ .re = 5, .im = 0 }, pad };
        Observe.watch(C, &output, &wanted);
        switch (path) {
            .trmv => cblas.cblas_ztrmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, &triangle, 3, &output, 2),
            .trsv => cblas.cblas_ztrsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, &triangle, 3, &output, 2),
            .tbmv => cblas.cblas_ztbmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, 1, &triangle_band, 3, &output, 2),
            .tbsv => cblas.cblas_ztbsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, 1, &triangle_band, 3, &output, 2),
            .tpmv => cblas.cblas_ztpmv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, &triangle_packed, &output, 2),
            .tpsv => cblas.cblas_ztpsv(cblas.CblasRowMajor, cblas.CblasUpper, cblas.CblasConjTrans, cblas.CblasUnit, 2, &triangle_packed, &output, 2),
        }
        try Observe.checkNext();
    }
    const gx = [_]C{ .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 } };
    const gy = [_]C{ .{ .re = 2, .im = 1 }, .{ .re = -1, .im = 2 } };
    var update = [_]C{ zero, zero, pad, zero, zero, pad };
    Observe.watch(C, &update, &.{ .{ .re = 3, .im = 1 }, .{ .re = 1, .im = -3 }, pad, .{ .re = 3, .im = -4 }, .{ .re = -4, .im = -3 }, pad });
    cblas.cblas_zgerc(cblas.CblasRowMajor, 2, 2, &one, &gx, 1, &gy, 1, &update, 3);
    try Observe.checkNext();
}

test "CBLAS invalid empty and local early returns retain one operation without stores" {
    const Observe = CblasAdmissionObservation;
    Observe.reset();
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    const a = [_]f64{ 1, 3, 2, 4 };
    var output = [_]f64{ 11, 13, 17, 19, 97 };
    Observe.watch(f64, &output, &.{ 11, 13, 17, 19, 97 });
    const Case = struct { layout: c_int, trans: c_int, m: c_int, lda: c_int };
    for ([_]Case{
        .{ .layout = -1, .trans = -1, .m = -1, .lda = 0 },
        .{ .layout = cblas.CblasColMajor, .trans = -1, .m = -1, .lda = 0 },
        .{ .layout = cblas.CblasColMajor, .trans = cblas.CblasNoTrans, .m = -1, .lda = 0 },
        .{ .layout = cblas.CblasColMajor, .trans = cblas.CblasNoTrans, .m = 2, .lda = 0 },
        .{ .layout = cblas.CblasColMajor, .trans = cblas.CblasNoTrans, .m = 0, .lda = 1 },
    }) |case| {
        cblas.cblas_dgemm(case.layout, case.trans, cblas.CblasNoTrans, case.m, 2, 2, 1, &a, case.lda, &a, 2, 0, &output, 2);
        try Observe.checkNext();
    }
    cblas.cblas_dscal(0, 3, &output, 1);
    try Observe.checkNext();
    const C = cblas.ComplexF64;
    const one = C{ .re = 1, .im = 0 };
    const zero = C{ .re = 0, .im = 0 };
    const safe_matrix = [_]C{ one, zero, zero, one };
    const input = [_]C{ one, one };
    var result = [_]C{ .{ .re = 7, .im = 2 }, .{ .re = 11, .im = 3 }, .{ .re = 97, .im = 19 } };
    Observe.watch(C, &result, &.{ .{ .re = 7, .im = 2 }, .{ .re = 11, .im = 3 }, .{ .re = 97, .im = 19 } });
    cblas.cblas_zgemv(cblas.CblasRowMajor, cblas.CblasConjTrans, 2, 2, &one, &safe_matrix, 2, &input, 0, &zero, &result, 1);
    try Observe.checkNext();
}

test "CBLAS scaling retains one operation through real busy executor fallback" {
    const shared = admission_execution.get();
    const policy = admission_execution.policy;
    const before = policy.maxThreadsOverride();
    defer {
        shared.drainAndReset();
        policy.setMaxThreads(before);
    }
    policy.setMaxThreads(2);
    if (policy.maxThreads() < 2) return error.SkipZigTest;
    const Work = struct {
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
        accepted: bool = false,
        fn call(raw: *const anyopaque, index: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = self.visits[index].fetchAdd(1, .monotonic);
            if (index == 0) {
                self.entered.store(true, .release);
                while (!self.release.load(.acquire)) std.Thread.yield() catch {};
            }
        }
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            self.accepted = admission_execution.get().runLowLatency(.{ .task_fn = call, .tasks = self, .count = 2 });
        }
    };
    var active: Work = .{};
    const caller = try std.Thread.spawn(.{}, Work.run, .{&active});
    var joined = false;
    defer {
        active.release.store(true, .release);
        if (!joined) caller.join();
    }
    while (!active.entered.load(.acquire) and !active.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(active.entered.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), shared.io_busy.load(.acquire));

    // Match the actual E4 fallback regression's size and original f64 scal path.
    // The occupier is a real Executor batch, not a fake numerical callback.
    const n = 2 * 1024 * 1024;
    const values = try std.testing.allocator.alloc(f64, n + 8);
    defer std.testing.allocator.free(values);
    @memset(values, 19);
    for (values[0..n], 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 17)) - 8;
    const Observe = struct {
        var entries: usize = 0;
        var finishes: usize = 0;
        var observed_values: []const f64 = &.{};
        fn call(ctx: *admission_execution.context.Context, event: admission_execution.context.TestEvent) void {
            if (ctx != admission_execution.getContext() or &ctx.executor != admission_execution.get())
                @panic("legacy fallback used a different default owner");
            if (ctx.snapshot().active_operations != 1 or ctx.executor.io_busy.load(.acquire) != 1)
                @panic("legacy fallback escaped its operation or busy batch");
            switch (event) {
                .admitted => entries += 1,
                .finishing => {
                    // Check every real result and trailing sentinel before end.
                    const logical_count = observed_values.len - 8;
                    for (observed_values[0..logical_count], 0..) |value, i|
                        if (value != (@as(f64, @floatFromInt(i % 17)) - 8) * -3)
                            @panic("fallback ended before the original scal body completed");
                    for (observed_values[logical_count..]) |value|
                        if (value != 19) @panic("fallback damaged trailing storage");
                    finishes += 1;
                },
            }
        }
    };
    Observe.entries = 0;
    Observe.finishes = 0;
    Observe.observed_values = values;
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    cblas.cblas_dscal(n, -3, values.ptr, 1);
    try std.testing.expectEqual(@as(usize, 1), Observe.entries);
    try std.testing.expectEqual(@as(usize, 1), Observe.finishes);
    try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().resource_pins);
    for (values[0..n], 0..) |value, i|
        try std.testing.expectEqual((@as(f64, @floatFromInt(i % 17)) - 8) * -3, value);
    for (values[n..]) |value| try std.testing.expectEqual(@as(f64, 19), value);
    try std.testing.expect(!active.finished.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), shared.io_busy.load(.acquire));
    try std.testing.expect(shared.persistent_task_fn == Work.call);
    try std.testing.expect(shared.persistent_tasks == @as(*const anyopaque, @ptrCast(&active)));
    active.release.store(true, .release);
    caller.join();
    joined = true;
    try std.testing.expect(active.accepted);
    for (&active.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
}

test "two actual CBLAS callers hold separate operations on the same default" {
    const Observe = struct {
        var entries: std.atomic.Value(usize) = .init(0);
        var finishes: std.atomic.Value(usize) = .init(0);
        var release: [2]std.atomic.Value(bool) = .{ .init(false), .init(false) };
        fn call(ctx: *admission_execution.context.Context, event: admission_execution.context.TestEvent) void {
            if (ctx != admission_execution.getContext() or &ctx.executor != admission_execution.get())
                @panic("concurrent CBLAS call used a different default owner");
            const active = ctx.snapshot().active_operations;
            if (active == 0 or active > 2) @panic("invalid concurrent CBLAS operation count");
            switch (event) {
                .admitted => {
                    const ticket = entries.fetchAdd(1, .acq_rel);
                    if (ticket >= release.len) @panic("nested CBLAS Fortran admission");
                    while (!release[ticket].load(.acquire)) std.Thread.yield() catch {};
                },
                .finishing => _ = finishes.fetchAdd(1, .acq_rel),
            }
        }
    };
    Observe.entries.store(0, .release);
    Observe.finishes.store(0, .release);
    for (&Observe.release) |*slot| slot.store(false, .release);
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 2, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);

    const Work = struct {
        input: [2]f64 = .{ 4, 5 },
        output: [3]f64,
        scale: bool,
        finished: std.atomic.Value(bool) = .init(false),
        fn call(self: *@This()) void {
            defer self.finished.store(true, .release);
            if (self.scale)
                cblas.cblas_dscal(2, -3, &self.output, 1)
            else
                cblas.cblas_dcopy(2, &self.input, 1, &self.output, 1);
        }
    };
    var scaled: Work = .{ .output = .{ 1, 2, 91 }, .scale = true };
    var copied: Work = .{ .output = .{ 0, 0, 92 }, .scale = false };
    const first = try std.Thread.spawn(.{}, Work.call, .{&scaled});
    var second: ?std.Thread = null;
    var joined = false;
    defer {
        for (&Observe.release) |*slot| slot.store(true, .release);
        if (!joined) {
            first.join();
            if (second) |thread| thread.join();
        }
    }
    second = try std.Thread.spawn(.{}, Work.call, .{&copied});
    while (Observe.entries.load(.acquire) < 2 and !scaled.finished.load(.acquire) and !copied.finished.load(.acquire))
        std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(usize, 2), Observe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), admission_execution.getContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), Observe.finishes.load(.acquire));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 91 }, &scaled.output);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 92 }, &copied.output);

    Observe.release[0].store(true, .release);
    while (!scaled.finished.load(.acquire) and !copied.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(usize, 1), Observe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), admission_execution.getContext().snapshot().active_operations);
    Observe.release[1].store(true, .release);
    first.join();
    second.?.join();
    joined = true;
    try std.testing.expectEqual(@as(usize, 2), Observe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().resource_pins);
    try std.testing.expectEqualSlices(f64, &.{ -3, -6, 91 }, &scaled.output);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 92 }, &copied.output);
}

// Focused layout reproducer: each format below has its own literal storage and
// independently calculated output. No production or test reference accessor is
// used to construct the oracle. These small stride-two cases do not exercise
// parallel scheduling or prove anything about resource ownership.
const HermitianLayoutFixture = enum { dense, band, packed_rows };

fn hermitianLayoutComplex(comptime T: type, re: f64, im: f64) T {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn expectHermitianRowMajorLayout(comptime T: type, comptime family: HermitianLayoutFixture, comptime upper: bool) !void {
    const storage_len: usize = switch (family) {
        .dense => 15, // n=3, row-major lda=5.
        .band => 9, // n=3, k=1, row-major lda=3.
        .packed_rows => 6, // n*(n+1)/2, packed by rows.
    };
    const canary = hermitianLayoutComplex(T, 901, -703);
    var matrix: [storage_len + 2]T = @splat(canary);
    const a = matrix[1..][0..storage_len];
    // Only the selected triangle is initialized. All ignored entries, row
    // padding, band edge holes, and outer guards retain the finite canary.
    const expected: [3]T = switch (family) {
        .dense => blk: {
            a[0] = hermitianLayoutComplex(T, 4, 0);
            a[6] = hermitianLayoutComplex(T, 5, 0);
            a[12] = hermitianLayoutComplex(T, 6, 0);
            if (upper) {
                a[1] = hermitianLayoutComplex(T, 1, 2);
                a[2] = hermitianLayoutComplex(T, -2, 1);
                a[7] = hermitianLayoutComplex(T, 3, -1);
            } else {
                a[5] = hermitianLayoutComplex(T, 1, -2);
                a[10] = hermitianLayoutComplex(T, -2, -1);
                a[11] = hermitianLayoutComplex(T, 3, 1);
            }
            // Ax = (8+2i, 12+i, 8i).
            break :blk .{ hermitianLayoutComplex(T, 6.25, 7), hermitianLayoutComplex(T, 12.5, 5.25), hermitianLayoutComplex(T, -5.5, 6.25) };
        },
        .band => blk: {
            // Upper rows: [d0,a01,pad], [d1,a12,pad], [d2,hole,pad].
            // Lower rows: [hole,d0,pad], [a10,d1,pad], [a21,d2,pad].
            if (upper) {
                a[0] = hermitianLayoutComplex(T, 2, 0);
                a[1] = hermitianLayoutComplex(T, -1, 3);
                a[3] = hermitianLayoutComplex(T, 4, 0);
                a[4] = hermitianLayoutComplex(T, 2, 1);
                a[6] = hermitianLayoutComplex(T, 7, 0);
            } else {
                a[1] = hermitianLayoutComplex(T, 2, 0);
                a[3] = hermitianLayoutComplex(T, -1, -3);
                a[4] = hermitianLayoutComplex(T, 4, 0);
                a[6] = hermitianLayoutComplex(T, 2, -1);
                a[7] = hermitianLayoutComplex(T, 7, 0);
            }
            // Ax = (3+9i, 6-5i, -4+10i); A02=A20=0.
            break :blk .{ hermitianLayoutComplex(T, -2.25, 11.5), hermitianLayoutComplex(T, 9.5, -3.75), hermitianLayoutComplex(T, -10.5, 6.25) };
        },
        .packed_rows => blk: {
            // Upper order: A00,A01,A02,A11,A12,A22.
            // Lower order: A00,A10,A11,A20,A21,A22.
            a[0] = hermitianLayoutComplex(T, 3, 0);
            a[5] = hermitianLayoutComplex(T, 8, 0);
            if (upper) {
                a[1] = hermitianLayoutComplex(T, 2, -1);
                a[2] = hermitianLayoutComplex(T, 1, 4);
                a[3] = hermitianLayoutComplex(T, 6, 0);
                a[4] = hermitianLayoutComplex(T, -2, 2);
            } else {
                a[1] = hermitianLayoutComplex(T, 2, 1);
                a[2] = hermitianLayoutComplex(T, 6, 0);
                a[3] = hermitianLayoutComplex(T, 1, -4);
                a[4] = hermitianLayoutComplex(T, -2, -2);
            }
            // Ax = (-3-3i, 11-9i, -9+11i).
            break :blk .{ hermitianLayoutComplex(T, -2.25, -3.5), hermitianLayoutComplex(T, 16.5, -5.25), hermitianLayoutComplex(T, -16, 4.75) };
        },
    };
    const x = [_]T{ canary, hermitianLayoutComplex(T, 1, 1), canary, hermitianLayoutComplex(T, 2, -1), canary, hermitianLayoutComplex(T, -1, 2), canary, canary };
    var y = [_]T{ canary, hermitianLayoutComplex(T, 2, -1), canary, hermitianLayoutComplex(T, -3, 2), canary, hermitianLayoutComplex(T, 1, 4), canary, canary };
    const alpha = hermitianLayoutComplex(T, 1, 0.5);
    const beta = hermitianLayoutComplex(T, -0.5, 0.25);
    const matrix_before = matrix;
    const x_before = x;
    const alpha_before = alpha;
    const beta_before = beta;
    const y_before = y;
    const uplo: c_int = if (upper) cblas.CblasUpper else cblas.CblasLower;
    switch (family) {
        .dense => if (T == cblas.ComplexF32)
            cblas.cblas_chemv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, 5, x[1..].ptr, 2, &beta, y[1..].ptr, 2)
        else
            cblas.cblas_zhemv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, 5, x[1..].ptr, 2, &beta, y[1..].ptr, 2),
        .band => if (T == cblas.ComplexF32)
            cblas.cblas_chbmv(cblas.CblasRowMajor, uplo, 3, 1, &alpha, a.ptr, 3, x[1..].ptr, 2, &beta, y[1..].ptr, 2)
        else
            cblas.cblas_zhbmv(cblas.CblasRowMajor, uplo, 3, 1, &alpha, a.ptr, 3, x[1..].ptr, 2, &beta, y[1..].ptr, 2),
        .packed_rows => if (T == cblas.ComplexF32)
            cblas.cblas_chpmv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, x[1..].ptr, 2, &beta, y[1..].ptr, 2)
        else
            cblas.cblas_zhpmv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, x[1..].ptr, 2, &beta, y[1..].ptr, 2),
    }
    // Perform immutable-input and output-gap checks before numerical assertions,
    // so a wrong result does not mask these checks within the same case.
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&matrix_before), std.mem.asBytes(&matrix));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&x_before), std.mem.asBytes(&x));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&alpha_before), std.mem.asBytes(&alpha));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&beta_before), std.mem.asBytes(&beta));
    for ([_]usize{ 0, 2, 4, 6, 7 }) |i| {
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&y_before[i]), std.mem.asBytes(&y[i]));
    }
    const Real = @TypeOf(alpha.re);
    const tolerance: Real = if (T == cblas.ComplexF32) 0.000001 else 0.000000000001;
    for (expected, 0..) |want, i| {
        try std.testing.expectApproxEqAbs(want.re, y[1 + 2 * i].re, tolerance);
        try std.testing.expectApproxEqAbs(want.im, y[1 + 2 * i].im, tolerance);
    }
}

test "cblas row-major chemv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .dense, true);
}

test "cblas row-major chemv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .dense, false);
}

test "cblas row-major zhemv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .dense, true);
}

test "cblas row-major zhemv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .dense, false);
}

test "cblas row-major chbmv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .band, true);
}

test "cblas row-major chbmv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .band, false);
}

test "cblas row-major zhbmv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .band, true);
}

test "cblas row-major zhbmv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .band, false);
}

test "cblas row-major chpmv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .packed_rows, true);
}

test "cblas row-major chpmv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF32, .packed_rows, false);
}

test "cblas row-major zhpmv upper independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .packed_rows, true);
}

test "cblas row-major zhpmv lower independent Hermitian layout oracle" {
    try expectHermitianRowMajorLayout(cblas.ComplexF64, .packed_rows, false);
}

// The original 43 declarations above, including all twelve failing literal
// layout cases, are unchanged. This extension varies only strides/scales/poison
// around the same independently known mathematical A*x vectors.
const HermitianLayoutEdgeScales = enum { normal, alpha_zero, beta_zero, both_zero, alpha_zero_beta_one };

fn hermitianLayoutEdgePosition(inc: c_int, logical: usize) usize {
    const distance: usize = @intCast(if (inc < 0) -inc else inc);
    return 1 + (if (inc < 0) 2 - logical else logical) * distance;
}

fn expectHermitianRowMajorEdges(comptime T: type, comptime family: HermitianLayoutFixture, upper: bool, incx: c_int, incy: c_int, scales: HermitianLayoutEdgeScales) !void {
    const Real = @TypeOf(@as(T, undefined).re);
    const nan = std.math.nan(Real);
    const poison: T = .{ .re = nan, .im = nan };
    const storage_len: usize = switch (family) {
        .dense => 15,
        .band => 9,
        .packed_rows => 6,
    };
    const canary = hermitianLayoutComplex(T, 901, -703);
    var matrix: [storage_len + 2]T = @splat(canary);
    const a = matrix[1..][0..storage_len];
    // Literal physical storage matches the original cases, with imaginary
    // diagonals replaced by NaN. No production indexing/access helper is used.
    const ax: [3]T = switch (family) {
        .dense => blk: {
            a[0] = .{ .re = 4, .im = nan };
            a[6] = .{ .re = 5, .im = nan };
            a[12] = .{ .re = 6, .im = nan };
            if (upper) {
                a[1] = hermitianLayoutComplex(T, 1, 2);
                a[2] = hermitianLayoutComplex(T, -2, 1);
                a[7] = hermitianLayoutComplex(T, 3, -1);
            } else {
                a[5] = hermitianLayoutComplex(T, 1, -2);
                a[10] = hermitianLayoutComplex(T, -2, -1);
                a[11] = hermitianLayoutComplex(T, 3, 1);
            }
            break :blk .{ hermitianLayoutComplex(T, 8, 2), hermitianLayoutComplex(T, 12, 1), hermitianLayoutComplex(T, 0, 8) };
        },
        .band => blk: {
            if (upper) {
                a[0] = .{ .re = 2, .im = nan };
                a[1] = hermitianLayoutComplex(T, -1, 3);
                a[3] = .{ .re = 4, .im = nan };
                a[4] = hermitianLayoutComplex(T, 2, 1);
                a[6] = .{ .re = 7, .im = nan };
            } else {
                a[1] = .{ .re = 2, .im = nan };
                a[3] = hermitianLayoutComplex(T, -1, -3);
                a[4] = .{ .re = 4, .im = nan };
                a[6] = hermitianLayoutComplex(T, 2, -1);
                a[7] = .{ .re = 7, .im = nan };
            }
            break :blk .{ hermitianLayoutComplex(T, 3, 9), hermitianLayoutComplex(T, 6, -5), hermitianLayoutComplex(T, -4, 10) };
        },
        .packed_rows => blk: {
            a[0] = .{ .re = 3, .im = nan };
            a[5] = .{ .re = 8, .im = nan };
            if (upper) {
                a[1] = hermitianLayoutComplex(T, 2, -1);
                a[2] = hermitianLayoutComplex(T, 1, 4);
                a[3] = .{ .re = 6, .im = nan };
                a[4] = hermitianLayoutComplex(T, -2, 2);
            } else {
                a[1] = hermitianLayoutComplex(T, 2, 1);
                a[2] = .{ .re = 6, .im = nan };
                a[3] = hermitianLayoutComplex(T, 1, -4);
                a[4] = hermitianLayoutComplex(T, -2, -2);
            }
            break :blk .{ hermitianLayoutComplex(T, -3, -3), hermitianLayoutComplex(T, 11, -9), hermitianLayoutComplex(T, -9, 11) };
        },
    };
    const alpha_zero = scales == .alpha_zero or scales == .both_zero or scales == .alpha_zero_beta_one;
    const beta_zero = scales == .beta_zero or scales == .both_zero;
    const alpha = if (alpha_zero) hermitianLayoutComplex(T, 0, 0) else hermitianLayoutComplex(T, 1, 0.5);
    const beta = if (beta_zero) hermitianLayoutComplex(T, 0, 0) else if (scales == .alpha_zero_beta_one) hermitianLayoutComplex(T, 1, 0) else hermitianLayoutComplex(T, -0.5, 0.25);
    const logical_x = [_]T{ hermitianLayoutComplex(T, 1, 1), hermitianLayoutComplex(T, 2, -1), hermitianLayoutComplex(T, -1, 2) };
    const logical_y = [_]T{ hermitianLayoutComplex(T, 2, -1), hermitianLayoutComplex(T, -3, 2), hermitianLayoutComplex(T, 1, 4) };
    var x: [11]T = @splat(canary);
    var y: [11]T = @splat(canary);
    var active: [11]bool = @splat(false);
    var expected: [3]T = undefined;
    if (alpha_zero) @memset(a, poison);
    for (0..3) |i| {
        x[hermitianLayoutEdgePosition(incx, i)] = if (alpha_zero) poison else logical_x[i];
        const iy = hermitianLayoutEdgePosition(incy, i);
        y[iy] = if (beta_zero) poison else logical_y[i];
        active[iy] = true;
        // These literal A*x values are fixed independently of the actual A/X
        // buffers. The old logical Y is also literal, never read back after call.
        const ar: f64 = alpha.re;
        const ai: f64 = alpha.im;
        const br: f64 = beta.re;
        const bi: f64 = beta.im;
        const pr: f64 = ax[i].re;
        const pi: f64 = ax[i].im;
        const yr: f64 = logical_y[i].re;
        const yi: f64 = logical_y[i].im;
        const product_re = if (alpha_zero) 0 else ar * pr - ai * pi;
        const product_im = if (alpha_zero) 0 else ar * pi + ai * pr;
        const prior_re = if (beta_zero) 0 else br * yr - bi * yi;
        const prior_im = if (beta_zero) 0 else br * yi + bi * yr;
        expected[i] = hermitianLayoutComplex(T, product_re + prior_re, product_im + prior_im);
    }
    const matrix_before = matrix;
    const x_before = x;
    const alpha_before = alpha;
    const beta_before = beta;
    const y_before = y;
    const uplo: c_int = if (upper) cblas.CblasUpper else cblas.CblasLower;
    switch (family) {
        .dense => if (T == cblas.ComplexF32)
            cblas.cblas_chemv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, 5, x[1..].ptr, incx, &beta, y[1..].ptr, incy)
        else
            cblas.cblas_zhemv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, 5, x[1..].ptr, incx, &beta, y[1..].ptr, incy),
        .band => if (T == cblas.ComplexF32)
            cblas.cblas_chbmv(cblas.CblasRowMajor, uplo, 3, 1, &alpha, a.ptr, 3, x[1..].ptr, incx, &beta, y[1..].ptr, incy)
        else
            cblas.cblas_zhbmv(cblas.CblasRowMajor, uplo, 3, 1, &alpha, a.ptr, 3, x[1..].ptr, incx, &beta, y[1..].ptr, incy),
        .packed_rows => if (T == cblas.ComplexF32)
            cblas.cblas_chpmv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, x[1..].ptr, incx, &beta, y[1..].ptr, incy)
        else
            cblas.cblas_zhpmv(cblas.CblasRowMajor, uplo, 3, &alpha, a.ptr, x[1..].ptr, incx, &beta, y[1..].ptr, incy),
    }
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&matrix_before), std.mem.asBytes(&matrix));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&x_before), std.mem.asBytes(&x));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&alpha_before), std.mem.asBytes(&alpha));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&beta_before), std.mem.asBytes(&beta));
    for (active, 0..) |written, i| {
        if (!written) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&y_before[i]), std.mem.asBytes(&y[i]));
    }
    if (scales == .alpha_zero_beta_one) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&y_before), std.mem.asBytes(&y));
    const tolerance: Real = if (T == cblas.ComplexF32) 0.000001 else 0.000000000001;
    for (expected, 0..) |want, i| {
        const got = y[hermitianLayoutEdgePosition(incy, i)];
        try std.testing.expectApproxEqAbs(want.re, got.re, tolerance);
        try std.testing.expectApproxEqAbs(want.im, got.im, tolerance);
    }
}

test "cblas row-major Hermitian edges retain strides zero-scale nonreads and poisoned diagonal semantics" {
    inline for (.{ cblas.ComplexF32, cblas.ComplexF64 }) |T| {
        inline for (.{ HermitianLayoutFixture.dense, HermitianLayoutFixture.band, HermitianLayoutFixture.packed_rows }) |family| {
            for ([_]bool{ true, false }) |upper| {
                for ([_][2]c_int{ .{ 1, 1 }, .{ -1, -1 }, .{ 2, -3 }, .{ -2, 3 }, .{ 3, 2 } }) |strides| {
                    for ([_]HermitianLayoutEdgeScales{ .normal, .alpha_zero, .beta_zero, .both_zero, .alpha_zero_beta_one }) |scales| {
                        expectHermitianRowMajorEdges(T, family, upper, strides[0], strides[1], scales) catch |err| {
                            std.debug.print("Hermitian edge scalar={s} storage={s} upper={} incx={d} incy={d} scales={s}\n", .{ @typeName(T), @tagName(family), upper, strides[0], strides[1], @tagName(scales) });
                            return err;
                        };
                    }
                }
            }
        }
    }
}
