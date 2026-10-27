// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const fortran = @import("zynum_blas_fortran_compat").abi;
const ref = @import("reference.zig");

fn matIndex(lda: fortran.BlasInt, row: usize, col: usize) usize {
    return row + col * @as(usize, @intCast(lda));
}

fn sampleValue(comptime T: type, row: usize, col: usize, salt: usize) T {
    const raw = ((row + 1) * (salt + 3) + (col + 1) * (salt + 7)) % 29;
    const value = (@as(f64, @floatFromInt(raw)) - 14.0) / 9.0;
    return @floatCast(value);
}

fn complexValue(comptime T: type, re: f64, im: f64) T {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn complexAdd(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn complexMul(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}

fn referenceComplexGemmNoTrans(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = complexValue(T, 0, 0);
            for (0..k) |p| {
                sum = complexAdd(T, sum, complexMul(T, a[matIndex(lda, i, p)], b[matIndex(ldb, p, j)]));
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = complexAdd(T, complexMul(T, alpha, sum), complexMul(T, beta, c[idxc]));
        }
    }
}

fn referenceComplexGemm(comptime T: type, transa: ref.Trans, transb: ref.Trans, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = complexValue(T, 0, 0);
            for (0..k) |p| {
                const av = ref.colMatrixValue(T, transa, a, @intCast(lda), i, p);
                const bv = ref.colMatrixValue(T, transb, b, @intCast(ldb), p, j);
                sum = complexAdd(T, sum, complexMul(T, av, bv));
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = complexAdd(T, complexMul(T, alpha, sum), complexMul(T, beta, c[idxc]));
        }
    }
}

fn expectComplexApprox(comptime T: type, expected: T, actual: T, tol: anytype) !void {
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tol);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tol);
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

fn expectComplexScalCase(comptime T: type, n: usize, incx: isize, alpha: T, tol: anytype) !void {
    var rng = ref.Rng.init(0x537a_6c31);
    const len = ref.vectorStorageLen(n, incx);
    const x = try std.testing.allocator.alloc(T, len);
    defer std.testing.allocator.free(x);
    const expected = try std.testing.allocator.alloc(T, len);
    defer std.testing.allocator.free(expected);

    ref.fillVector(T, &rng, x, n, incx);
    @memcpy(expected, x);

    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var alpha_arg = alpha;
    if (T == fortran.ComplexF32) {
        fortran.cscal_(&nn, &alpha_arg, x.ptr, &ix);
    } else if (T == fortran.ComplexF64) {
        fortran.zscal_(&nn, &alpha_arg, x.ptr, &ix);
    } else {
        @compileError("complex scal test supports ComplexF32 and ComplexF64");
    }
    referenceComplexScal(T, n, alpha, expected, incx);

    for (expected, x) |want, got| try expectComplexApprox(T, want, got, tol);
}

fn expectComplexAxpyCase(comptime T: type, n: usize, incx: isize, incy: isize, alpha: T, tol: anytype) !void {
    var rng = ref.Rng.init(0xa01a_78b5);
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

    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    var alpha_arg = alpha;
    if (T == fortran.ComplexF32) {
        fortran.caxpy_(&nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy);
    } else if (T == fortran.ComplexF64) {
        fortran.zaxpy_(&nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy);
    } else {
        @compileError("complex axpy test supports ComplexF32 and ComplexF64");
    }
    referenceComplexAxpy(T, n, alpha, x, incx, expected, incy);

    for (expected, y) |want, got| try expectComplexApprox(T, want, got, tol);
}

fn expectComplexAxpbyCase(comptime T: type, n: usize, incx: isize, incy: isize, alpha: T, beta: T, tol: anytype) !void {
    var rng = ref.Rng.init(0xa9b5_1234);
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

    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    var alpha_arg = alpha;
    var beta_arg = beta;
    if (T == fortran.ComplexF32) {
        fortran.caxpby_(&nn, &alpha_arg, x.ptr, &ix, &beta_arg, y.ptr, &iy);
    } else if (T == fortran.ComplexF64) {
        fortran.zaxpby_(&nn, &alpha_arg, x.ptr, &ix, &beta_arg, y.ptr, &iy);
    } else {
        @compileError("complex axpby test supports ComplexF32 and ComplexF64");
    }
    referenceComplexAxpby(T, n, alpha, x, incx, beta, expected, incy);

    for (expected, y) |want, got| try expectComplexApprox(T, want, got, tol);
}

fn sampleComplexValue(comptime T: type, row: usize, col: usize, salt: usize) T {
    return complexValue(T, sampleValue(f64, row, col, salt), sampleValue(f64, row, col, salt + 11));
}

fn fillComplexMatrix(comptime T: type, matrix: []T, rows: usize, cols: usize, ld: fortran.BlasInt, salt: usize) void {
    for (0..cols) |j| {
        for (0..rows) |i| {
            matrix[matIndex(ld, i, j)] = sampleComplexValue(T, i, j, salt);
        }
    }
}

fn fillPaddedComplexMatrix(comptime T: type, matrix: []T, rows: usize, cols: usize, ld: fortran.BlasInt, salt: usize, sentinel: T) void {
    @memset(matrix, sentinel);
    for (0..cols) |j| {
        for (0..rows) |i| {
            matrix[matIndex(ld, i, j)] = sampleComplexValue(T, i, j, salt);
        }
    }
}

fn fillPaddedMatrix(comptime T: type, matrix: []T, rows: usize, cols: usize, ld: fortran.BlasInt, salt: usize, sentinel: T) void {
    @memset(matrix, sentinel);
    for (0..cols) |j| {
        for (0..rows) |i| {
            matrix[matIndex(ld, i, j)] = sampleValue(T, i, j, salt);
        }
    }
}

fn referenceGemmNoTrans(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..k) |p| {
                sum = @mulAdd(T, a[matIndex(lda, i, p)], b[matIndex(ldb, p, j)], sum);
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = @mulAdd(T, alpha, sum, if (beta == 0) 0 else beta * c[idxc]);
        }
    }
}

fn referenceGemmNoTransTransposedB(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..k) |p| {
                sum = @mulAdd(T, a[matIndex(lda, i, p)], b[matIndex(ldb, j, p)], sum);
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = @mulAdd(T, alpha, sum, if (beta == 0) 0 else beta * c[idxc]);
        }
    }
}

const referenceGemmTransposedA = @import("gemm_reference.zig").referenceGemmTransposedA;

fn callGemmNoTrans(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var alpha_arg = alpha;
    var beta_arg = beta;
    var lda_arg = lda;
    var ldb_arg = ldb;
    var ldc_arg = ldc;
    if (T == f32) {
        fortran.sgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else if (T == f64) {
        fortran.dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else {
        @compileError("test helper supports f32 and f64");
    }
}

fn callGemmNoTransTransposedB(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    var ta = [_]u8{'N'};
    var tb = [_]u8{'T'};
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var alpha_arg = alpha;
    var beta_arg = beta;
    var lda_arg = lda;
    var ldb_arg = ldb;
    var ldc_arg = ldc;
    if (T == f32) {
        fortran.sgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else if (T == f64) {
        fortran.dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else {
        @compileError("test helper supports f32 and f64");
    }
}

fn callGemmTransposedA(comptime T: type, transposed_b: bool, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    var ta = [_]u8{'T'};
    var tb = [_]u8{if (transposed_b) 'T' else 'N'};
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var alpha_arg = alpha;
    var beta_arg = beta;
    var lda_arg = lda;
    var ldb_arg = ldb;
    var ldc_arg = ldc;
    if (T == f32) {
        fortran.sgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else if (T == f64) {
        fortran.dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else {
        @compileError("test helper supports f32 and f64");
    }
}

fn callComplexGemmNoTrans(comptime T: type, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var alpha_arg = alpha;
    var beta_arg = beta;
    var lda_arg = lda;
    var ldb_arg = ldb;
    var ldc_arg = ldc;
    if (T == fortran.ComplexF32) {
        fortran.cgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else if (T == fortran.ComplexF64) {
        fortran.zgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else {
        @compileError("test helper supports ComplexF32 and ComplexF64");
    }
}

fn transChar(trans: ref.Trans) u8 {
    return switch (trans) {
        .no_trans => 'N',
        .trans => 'T',
        .conj_trans => 'C',
    };
}

fn callComplexGemm(comptime T: type, transa: ref.Trans, transb: ref.Trans, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    var ta = [_]u8{transChar(transa)};
    var tb = [_]u8{transChar(transb)};
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var alpha_arg = alpha;
    var beta_arg = beta;
    var lda_arg = lda;
    var ldb_arg = ldb;
    var ldc_arg = ldc;
    if (T == fortran.ComplexF32) {
        fortran.cgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else if (T == fortran.ComplexF64) {
        fortran.zgemm_(&ta, &tb, &mm, &nn, &kk, &alpha_arg, a.ptr, &lda_arg, b.ptr, &ldb_arg, &beta_arg, c.ptr, &ldc_arg);
    } else {
        @compileError("test helper supports ComplexF32 and ComplexF64");
    }
}

fn expectGemmNoTransCase(comptime T: type, allocator: std.mem.Allocator, m: usize, n: usize, k: usize, lda: fortran.BlasInt, ldb: fortran.BlasInt, ldc: fortran.BlasInt, alpha: T, beta: T) !void {
    const a_len = @as(usize, @intCast(lda)) * k;
    const b_len = @as(usize, @intCast(ldb)) * n;
    const c_len = @as(usize, @intCast(ldc)) * n;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    const sentinel: T = if (T == f32) -777.0 else -999.0;
    fillPaddedMatrix(T, a, m, k, lda, 1, sentinel);
    fillPaddedMatrix(T, b, k, n, ldb, 5, sentinel);
    fillPaddedMatrix(T, c, m, n, ldc, 9, sentinel);
    @memcpy(expected, c);

    callGemmNoTrans(T, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    referenceGemmNoTrans(T, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    const tol: T = if (T == f32) 1e-3 else 1e-10;
    for (0..n) |j| {
        for (0..@as(usize, @intCast(ldc))) |i| {
            const idxc = matIndex(ldc, i, j);
            if (i < m) {
                try std.testing.expectApproxEqAbs(expected[idxc], c[idxc], tol);
            } else {
                try std.testing.expectEqual(expected[idxc], c[idxc]);
            }
        }
    }
}

fn expectGemmNoTransTransposedBCase(comptime T: type, allocator: std.mem.Allocator, m: usize, n: usize, k: usize, lda: fortran.BlasInt, ldb: fortran.BlasInt, ldc: fortran.BlasInt, alpha: T, beta: T) !void {
    const a_len = @as(usize, @intCast(lda)) * k;
    const b_len = @as(usize, @intCast(ldb)) * k;
    const c_len = @as(usize, @intCast(ldc)) * n;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    const sentinel: T = if (T == f32) -777.0 else -999.0;
    fillPaddedMatrix(T, a, m, k, lda, 13, sentinel);
    fillPaddedMatrix(T, b, n, k, ldb, 17, sentinel);
    fillPaddedMatrix(T, c, m, n, ldc, 19, sentinel);
    @memcpy(expected, c);

    callGemmNoTransTransposedB(T, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    referenceGemmNoTransTransposedB(T, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    const tol: T = if (T == f32) 1e-3 else 1e-10;
    for (0..n) |j| {
        for (0..@as(usize, @intCast(ldc))) |i| {
            const idxc = matIndex(ldc, i, j);
            if (i < m) {
                try std.testing.expectApproxEqAbs(expected[idxc], c[idxc], tol);
            } else {
                try std.testing.expectEqual(expected[idxc], c[idxc]);
            }
        }
    }
}

fn expectGemmTransposedACase(comptime T: type, allocator: std.mem.Allocator, transposed_b: bool, m: usize, n: usize, k: usize, lda: fortran.BlasInt, ldb: fortran.BlasInt, ldc: fortran.BlasInt, alpha: T, beta: T) !void {
    const a_len = @max(@as(usize, 1), @as(usize, @intCast(lda)) * m);
    const b_cols = if (transposed_b) k else n;
    const b_len = @max(@as(usize, 1), @as(usize, @intCast(ldb)) * b_cols);
    const c_len = @as(usize, @intCast(ldc)) * n;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    const sentinel: T = if (T == f32) -777.0 else -999.0;
    fillPaddedMatrix(T, a, k, m, lda, 23, sentinel);
    if (transposed_b) {
        fillPaddedMatrix(T, b, n, k, ldb, 29, sentinel);
    } else {
        fillPaddedMatrix(T, b, k, n, ldb, 29, sentinel);
    }
    fillPaddedMatrix(T, c, m, n, ldc, 31, sentinel);
    @memcpy(expected, c);

    callGemmTransposedA(T, transposed_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    referenceGemmTransposedA(T, transposed_b, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    const tol: T = if (T == f32) 1e-3 else 1e-10;
    for (0..n) |j| {
        for (0..@as(usize, @intCast(ldc))) |i| {
            const idxc = matIndex(ldc, i, j);
            if (i < m) {
                try std.testing.expectApproxEqAbs(expected[idxc], c[idxc], tol);
            } else {
                try std.testing.expectEqual(expected[idxc], c[idxc]);
            }
        }
    }
}

fn expectComplexGemmNoTransCase(comptime T: type, allocator: std.mem.Allocator, m: usize, n: usize, k: usize, lda: fortran.BlasInt, ldb: fortran.BlasInt, ldc: fortran.BlasInt, alpha: T, beta: T, tol: anytype) !void {
    const a_len = @as(usize, @intCast(lda)) * k;
    const b_len = @as(usize, @intCast(ldb)) * n;
    const c_len = @as(usize, @intCast(ldc)) * n;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    const sentinel = complexValue(T, -77, 33);
    fillPaddedComplexMatrix(T, a, m, k, lda, 3, sentinel);
    fillPaddedComplexMatrix(T, b, k, n, ldb, 7, sentinel);
    fillPaddedComplexMatrix(T, c, m, n, ldc, 11, sentinel);
    @memcpy(expected, c);

    callComplexGemmNoTrans(T, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    referenceComplexGemmNoTrans(T, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    for (0..n) |j| {
        for (0..@as(usize, @intCast(ldc))) |i| {
            const idxc = matIndex(ldc, i, j);
            if (i < m) {
                try expectComplexApprox(T, expected[idxc], c[idxc], tol);
            } else {
                try std.testing.expectEqual(expected[idxc], c[idxc]);
            }
        }
    }
}

fn expectComplexGemmTransCase(comptime T: type, transa: ref.Trans, transb: ref.Trans, m: usize, n: usize, k: usize, alpha: T, beta: T, tol: anytype) !void {
    const a_rows = if (transa == .no_trans) m else k;
    const a_cols = if (transa == .no_trans) k else m;
    const b_rows = if (transb == .no_trans) k else n;
    const b_cols = if (transb == .no_trans) n else k;
    const lda: fortran.BlasInt = @intCast(a_rows + 4);
    const ldb: fortran.BlasInt = @intCast(b_rows + 5);
    const ldc: fortran.BlasInt = @intCast(m + 6);

    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, @as(usize, @intCast(lda)) * a_cols);
    defer allocator.free(a);
    const b = try allocator.alloc(T, @as(usize, @intCast(ldb)) * b_cols);
    defer allocator.free(b);
    const c = try allocator.alloc(T, @as(usize, @intCast(ldc)) * n);
    defer allocator.free(c);
    const expected = try allocator.alloc(T, c.len);
    defer allocator.free(expected);

    const sentinel = complexValue(T, -77, 33);
    fillPaddedComplexMatrix(T, a, a_rows, a_cols, lda, 37, sentinel);
    fillPaddedComplexMatrix(T, b, b_rows, b_cols, ldb, 41, sentinel);
    fillPaddedComplexMatrix(T, c, m, n, ldc, 43, sentinel);
    @memcpy(expected, c);

    callComplexGemm(T, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    referenceComplexGemm(T, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);

    for (0..n) |j| {
        for (0..@as(usize, @intCast(ldc))) |i| {
            const idxc = matIndex(ldc, i, j);
            if (i < m) {
                try expectComplexApprox(T, expected[idxc], c[idxc], tol);
            } else {
                try std.testing.expectEqual(expected[idxc], c[idxc]);
            }
        }
    }
}

test "dgemm column major no transpose" {
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var k: fortran.BlasInt = 3;
    var lda: fortran.BlasInt = 2;
    var ldb: fortran.BlasInt = 3;
    var ldc: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var left_matrix = [_]f64{
        1, 2,
        3, 4,
        5, 6,
    };
    var right_matrix = [_]f64{
        7,  8,  9,
        10, 11, 12,
    };
    var result_matrix = [_]f64{ 0, 0, 0, 0 };
    fortran.dgemm_(&ta, &tb, &m, &n, &k, &alpha, &left_matrix, &lda, &right_matrix, &ldb, &beta, &result_matrix, &ldc);
    try std.testing.expectApproxEqAbs(@as(f64, 76), result_matrix[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), result_matrix[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 103), result_matrix[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 136), result_matrix[3], 1e-12);
}

test "complex scal with complex alpha supports strides" {
    try expectComplexScalCase(fortran.ComplexF32, 9, 1, complexValue(fortran.ComplexF32, -0.75, 0.5), @as(f32, 1e-5));
    try expectComplexScalCase(fortran.ComplexF32, 7, 2, complexValue(fortran.ComplexF32, 0.25, -1.25), @as(f32, 1e-5));
    try expectComplexScalCase(fortran.ComplexF32, 7, -2, complexValue(fortran.ComplexF32, 1.5, 0.375), @as(f32, 1e-5));
    try expectComplexScalCase(fortran.ComplexF32, 17, 2, complexValue(fortran.ComplexF32, 0.25, -1.25), @as(f32, 1e-5));
    try expectComplexScalCase(fortran.ComplexF64, 9, 1, complexValue(fortran.ComplexF64, -0.75, 0.5), @as(f64, 1e-12));
    try expectComplexScalCase(fortran.ComplexF64, 7, 2, complexValue(fortran.ComplexF64, 0.25, -1.25), @as(f64, 1e-12));
    try expectComplexScalCase(fortran.ComplexF64, 7, -2, complexValue(fortran.ComplexF64, 1.5, 0.375), @as(f64, 1e-12));
}

test "complex axpy and axpby with complex alpha support strides" {
    try expectComplexAxpyCase(fortran.ComplexF32, 8, 1, 1, complexValue(fortran.ComplexF32, -0.75, 0.5), @as(f32, 1e-5));
    try expectComplexAxpyCase(fortran.ComplexF32, 7, 2, -2, complexValue(fortran.ComplexF32, 0.25, -1.25), @as(f32, 1e-5));
    try expectComplexAxpyCase(fortran.ComplexF32, 17, 2, 2, complexValue(fortran.ComplexF32, 0.25, -1.25), @as(f32, 1e-5));
    try expectComplexAxpyCase(fortran.ComplexF64, 8, 1, 1, complexValue(fortran.ComplexF64, -0.75, 0.5), @as(f64, 1e-12));
    try expectComplexAxpyCase(fortran.ComplexF64, 7, 2, -2, complexValue(fortran.ComplexF64, 0.25, -1.25), @as(f64, 1e-12));

    try expectComplexAxpbyCase(fortran.ComplexF32, 8, 1, 1, complexValue(fortran.ComplexF32, -0.75, 0.5), complexValue(fortran.ComplexF32, 0.25, -0.125), @as(f32, 1e-5));
    try expectComplexAxpbyCase(fortran.ComplexF32, 7, -2, 2, complexValue(fortran.ComplexF32, 0.25, -1.25), complexValue(fortran.ComplexF32, -0.5, 0.75), @as(f32, 1e-5));
    try expectComplexAxpbyCase(fortran.ComplexF32, 17, 2, 2, complexValue(fortran.ComplexF32, 0.25, -1.25), complexValue(fortran.ComplexF32, -0.5, 0.75), @as(f32, 1e-5));
    try expectComplexAxpbyCase(fortran.ComplexF64, 8, 1, 1, complexValue(fortran.ComplexF64, -0.75, 0.5), complexValue(fortran.ComplexF64, 0.25, -0.125), @as(f64, 1e-12));
    try expectComplexAxpbyCase(fortran.ComplexF64, 7, -2, 2, complexValue(fortran.ComplexF64, 0.25, -1.25), complexValue(fortran.ComplexF64, -0.5, 0.75), @as(f64, 1e-12));
}

test "stride-two vector updates preserve gap elements" {
    const n: usize = 17;
    const storage_len = 2 * n - 1;
    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 2;

    var x: [storage_len]f32 = undefined;
    var y: [storage_len]f32 = undefined;
    var expected_y: [storage_len]f32 = undefined;
    for (&x, &y, &expected_y, 0..) |*xv, *yv, *want, i| {
        xv.* = (@as(f32, @floatFromInt(i)) - 11) / 17;
        yv.* = (@as(f32, @floatFromInt(i)) + 7) / 19;
        want.* = yv.*;
    }
    var alpha: f32 = 0.25;
    for (0..n) |i| expected_y[2 * i] = @mulAdd(f32, alpha, x[2 * i], expected_y[2 * i]);
    fortran.saxpy_(&nn, &alpha, &x, &inc, &y, &inc);
    for (expected_y, y) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);

    var cx: [storage_len]fortran.ComplexF32 = undefined;
    var cy: [storage_len]fortran.ComplexF32 = undefined;
    var expected_x: [storage_len]fortran.ComplexF32 = undefined;
    var expected_cy: [storage_len]fortran.ComplexF32 = undefined;
    for (&cx, &cy, &expected_x, &expected_cy, 0..) |*xv, *yv, *want_x, *want_y, i| {
        xv.* = complexValue(fortran.ComplexF32, @floatFromInt(i), -@as(f64, @floatFromInt(i + 1)));
        yv.* = complexValue(fortran.ComplexF32, @floatFromInt(i + 3), @floatFromInt(i + 5));
        want_x.* = xv.*;
        want_y.* = yv.*;
    }
    var c: f32 = 0.8;
    var s: f32 = 0.6;
    for (0..n) |i| {
        const index = 2 * i;
        const xv = expected_x[index];
        const yv = expected_cy[index];
        expected_x[index] = .{
            .re = c * xv.re + s * yv.re,
            .im = c * xv.im + s * yv.im,
        };
        expected_cy[index] = .{
            .re = c * yv.re - s * xv.re,
            .im = c * yv.im - s * xv.im,
        };
    }
    fortran.csrot_(&nn, &cx, &inc, &cy, &inc, &c, &s);
    for (expected_x, cx) |want, got| try expectComplexApprox(fortran.ComplexF32, want, got, @as(f32, 1e-5));
    for (expected_cy, cy) |want, got| try expectComplexApprox(fortran.ComplexF32, want, got, @as(f32, 1e-5));
}

test "sgemm and dgemm no-trans padded alpha beta tile tails" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 5, 5, 3, 7, 5, 8, -0.5, 2.0);
    try expectGemmNoTransCase(f64, allocator, 7, 5, 3, 9, 5, 10, -0.5, 2.0);
}

test "sgemm and dgemm NT packed path handles padding tails alpha beta" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransTransposedBCase(f32, allocator, 35, 37, 67, 39, 41, 43, -0.5, 2.0);
    try expectGemmNoTransTransposedBCase(f64, allocator, 35, 37, 67, 39, 41, 43, -0.5, 2.0);
    try expectGemmNoTransTransposedBCase(f32, allocator, 37, 35, 71, 41, 39, 43, 1.0, 0.0);
    try expectGemmNoTransTransposedBCase(f64, allocator, 37, 35, 71, 41, 39, 43, 1.0, 0.0);
    try expectGemmNoTransTransposedBCase(f32, allocator, 128, 128, 128, 131, 137, 139, 1.0, 0.0);
    try expectGemmNoTransTransposedBCase(f64, allocator, 128, 128, 128, 131, 137, 139, 1.0, 0.0);
}

test "sgemm and dgemm NT packed path supports heap packing" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransTransposedBCase(f32, allocator, 17, 19, 2049, 23, 29, 31, 0.75, -0.25);
    try expectGemmNoTransTransposedBCase(f64, allocator, 17, 19, 2049, 23, 29, 31, 0.75, -0.25);
}

test "sgemm and dgemm NT vector edges reuse GEMV with padded storage" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransTransposedBCase(f32, allocator, 1, 257, 131, 3, 263, 5, -0.5, 2.0);
    try expectGemmNoTransTransposedBCase(f64, allocator, 1, 257, 131, 3, 263, 5, -0.5, 2.0);
    try expectGemmNoTransTransposedBCase(f32, allocator, 257, 1, 131, 263, 3, 269, -0.5, 2.0);
    try expectGemmNoTransTransposedBCase(f64, allocator, 257, 1, 131, 263, 3, 269, -0.5, 2.0);
}

test "parallel sgemm and dgemm NT packed path matches reference" {
    const allocator = std.testing.allocator;
    fortran.runtime.setMaxThreads(3);
    defer fortran.runtime.setMaxThreads(0);
    try expectGemmNoTransTransposedBCase(f32, allocator, 193, 65, 577, 197, 69, 199, 0.75, -0.25);
    try expectGemmNoTransTransposedBCase(f64, allocator, 193, 65, 577, 197, 69, 199, 0.75, -0.25);
}

test "sgemm and dgemm TN TT packed paths handle padding odd tails alpha beta" {
    const allocator = std.testing.allocator;
    inline for (.{ f32, f64 }) |T| {
        try expectGemmTransposedACase(T, allocator, false, 35, 37, 67, 71, 73, 43, -0.5, 2.0);
        try expectGemmTransposedACase(T, allocator, true, 35, 37, 67, 71, 41, 43, -0.5, 2.0);
        try expectGemmTransposedACase(T, allocator, false, 37, 35, 71, 73, 79, 43, 0.75, -0.25);
        try expectGemmTransposedACase(T, allocator, true, 37, 35, 71, 73, 39, 43, 0.75, -0.25);
    }
}

test "sgemm and dgemm medium irregular AMX interior keeps padded tails disjoint" {
    const allocator = std.testing.allocator;
    inline for (.{ f32, f64 }) |T| {
        try expectGemmNoTransCase(T, allocator, 127, 129, 65, 131, 69, 137, 1.0, 0.0);
        try expectGemmNoTransTransposedBCase(T, allocator, 127, 129, 65, 131, 133, 137, 1.0, 0.0);
        try expectGemmTransposedACase(T, allocator, false, 127, 129, 65, 69, 71, 137, 1.0, 0.0);
        try expectGemmTransposedACase(T, allocator, true, 127, 129, 65, 69, 133, 137, 1.0, 0.0);
    }
}

test "sgemm and dgemm TN TT packed paths support heap A packing" {
    const allocator = std.testing.allocator;
    inline for (.{ f32, f64 }) |T| {
        try expectGemmTransposedACase(T, allocator, false, 17, 19, 2049, 2053, 2057, 23, 0.75, -0.25);
        try expectGemmTransposedACase(T, allocator, true, 17, 19, 2049, 2053, 23, 23, 0.75, -0.25);
    }
}

test "parallel sgemm and dgemm TN TT packed paths match reference" {
    const allocator = std.testing.allocator;
    fortran.runtime.setMaxThreads(3);
    defer fortran.runtime.setMaxThreads(0);
    inline for (.{ f32, f64 }) |T| {
        try expectGemmTransposedACase(T, allocator, false, 193, 65, 577, 581, 583, 199, 0.75, -0.25);
        try expectGemmTransposedACase(T, allocator, true, 193, 65, 577, 581, 69, 199, 0.75, -0.25);
    }
}

test "sgemm and dgemm TN TT alpha zero and k zero scale C" {
    const allocator = std.testing.allocator;
    inline for (.{ f32, f64 }) |T| {
        try expectGemmTransposedACase(T, allocator, false, 5, 7, 11, 13, 13, 11, 0.0, -0.25);
        try expectGemmTransposedACase(T, allocator, true, 5, 7, 11, 13, 9, 11, 0.0, -0.25);
        try expectGemmTransposedACase(T, allocator, false, 5, 7, 0, 3, 3, 11, 1.0, 2.0);
        try expectGemmTransposedACase(T, allocator, true, 5, 7, 0, 3, 9, 11, 1.0, 2.0);
    }
}

test "small sgemm and dgemm TN TT fall back before updating C" {
    const allocator = std.testing.allocator;
    inline for (.{ f32, f64 }) |T| {
        try expectGemmTransposedACase(T, allocator, false, 5, 7, 3, 5, 5, 11, -0.5, 2.0);
        try expectGemmTransposedACase(T, allocator, true, 5, 7, 3, 5, 9, 11, -0.5, 2.0);
    }
}

test "cgemm and zgemm column major no transpose" {
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var k: fortran.BlasInt = 3;
    var lda: fortran.BlasInt = 2;
    var ldb: fortran.BlasInt = 3;
    var ldc: fortran.BlasInt = 2;

    var alpha32 = complexValue(fortran.ComplexF32, 0.75, -0.25);
    var beta32 = complexValue(fortran.ComplexF32, -0.5, 0.25);
    var a32 = [_]fortran.ComplexF32{
        complexValue(fortran.ComplexF32, 1, 1),     complexValue(fortran.ComplexF32, 2, -1),
        complexValue(fortran.ComplexF32, -3, 0.5),  complexValue(fortran.ComplexF32, 4, 2),
        complexValue(fortran.ComplexF32, 0.25, -2), complexValue(fortran.ComplexF32, -1, 1),
    };
    var b32 = [_]fortran.ComplexF32{
        complexValue(fortran.ComplexF32, 2, 0),  complexValue(fortran.ComplexF32, -1, 1), complexValue(fortran.ComplexF32, 0.5, -0.5),
        complexValue(fortran.ComplexF32, 1, -2), complexValue(fortran.ComplexF32, 3, 1),  complexValue(fortran.ComplexF32, -2, 0.25),
    };
    var c32 = [_]fortran.ComplexF32{
        complexValue(fortran.ComplexF32, 1, 0.5), complexValue(fortran.ComplexF32, -2, 1),
        complexValue(fortran.ComplexF32, 0, -1),  complexValue(fortran.ComplexF32, 3, 2),
    };
    var expected32 = c32;
    referenceComplexGemmNoTrans(fortran.ComplexF32, 2, 2, 3, alpha32, &a32, lda, &b32, ldb, beta32, &expected32, ldc);
    fortran.cgemm_(&ta, &tb, &m, &n, &k, &alpha32, &a32, &lda, &b32, &ldb, &beta32, &c32, &ldc);
    for (expected32, c32) |expected, actual| try expectComplexApprox(fortran.ComplexF32, expected, actual, @as(f32, 1e-4));

    var alpha64 = complexValue(fortran.ComplexF64, 0.75, -0.25);
    var beta64 = complexValue(fortran.ComplexF64, -0.5, 0.25);
    var a64 = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 1),     complexValue(fortran.ComplexF64, 2, -1),
        complexValue(fortran.ComplexF64, -3, 0.5),  complexValue(fortran.ComplexF64, 4, 2),
        complexValue(fortran.ComplexF64, 0.25, -2), complexValue(fortran.ComplexF64, -1, 1),
    };
    var b64 = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 2, 0),  complexValue(fortran.ComplexF64, -1, 1), complexValue(fortran.ComplexF64, 0.5, -0.5),
        complexValue(fortran.ComplexF64, 1, -2), complexValue(fortran.ComplexF64, 3, 1),  complexValue(fortran.ComplexF64, -2, 0.25),
    };
    var c64 = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 0.5), complexValue(fortran.ComplexF64, -2, 1),
        complexValue(fortran.ComplexF64, 0, -1),  complexValue(fortran.ComplexF64, 3, 2),
    };
    var expected64 = c64;
    referenceComplexGemmNoTrans(fortran.ComplexF64, 2, 2, 3, alpha64, &a64, lda, &b64, ldb, beta64, &expected64, ldc);
    fortran.zgemm_(&ta, &tb, &m, &n, &k, &alpha64, &a64, &lda, &b64, &ldb, &beta64, &c64, &ldc);
    for (expected64, c64) |expected, actual| try expectComplexApprox(fortran.ComplexF64, expected, actual, @as(f64, 1e-12));
}

test "complex gemm no-trans large fast path matches reference" {
    const allocator = std.testing.allocator;
    const m_usize: usize = 16;
    const n_usize: usize = 16;
    const k_usize: usize = 512;
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = @intCast(m_usize);
    var n: fortran.BlasInt = @intCast(n_usize);
    var k: fortran.BlasInt = @intCast(k_usize);
    var lda = m;
    var ldb = k;
    var ldc = m;

    const a32 = try allocator.alloc(fortran.ComplexF32, m_usize * k_usize);
    defer allocator.free(a32);
    const b32 = try allocator.alloc(fortran.ComplexF32, k_usize * n_usize);
    defer allocator.free(b32);
    const c32 = try allocator.alloc(fortran.ComplexF32, m_usize * n_usize);
    defer allocator.free(c32);
    const expected32 = try allocator.alloc(fortran.ComplexF32, m_usize * n_usize);
    defer allocator.free(expected32);
    fillComplexMatrix(fortran.ComplexF32, a32, m_usize, k_usize, lda, 3);
    fillComplexMatrix(fortran.ComplexF32, b32, k_usize, n_usize, ldb, 7);
    @memset(c32, complexValue(fortran.ComplexF32, 0, 0));
    @memset(expected32, complexValue(fortran.ComplexF32, 0, 0));
    var alpha32 = complexValue(fortran.ComplexF32, 1, 0);
    var beta32 = complexValue(fortran.ComplexF32, 0, 0);
    referenceComplexGemmNoTrans(fortran.ComplexF32, m_usize, n_usize, k_usize, alpha32, a32, lda, b32, ldb, beta32, expected32, ldc);
    fortran.cgemm_(&ta, &tb, &m, &n, &k, &alpha32, a32.ptr, &lda, b32.ptr, &ldb, &beta32, c32.ptr, &ldc);
    for (expected32, c32) |expected, actual| try expectComplexApprox(fortran.ComplexF32, expected, actual, @as(f32, 2e-2));

    const a64 = try allocator.alloc(fortran.ComplexF64, m_usize * k_usize);
    defer allocator.free(a64);
    const b64 = try allocator.alloc(fortran.ComplexF64, k_usize * n_usize);
    defer allocator.free(b64);
    const c64 = try allocator.alloc(fortran.ComplexF64, m_usize * n_usize);
    defer allocator.free(c64);
    const expected64 = try allocator.alloc(fortran.ComplexF64, m_usize * n_usize);
    defer allocator.free(expected64);
    fillComplexMatrix(fortran.ComplexF64, a64, m_usize, k_usize, lda, 5);
    fillComplexMatrix(fortran.ComplexF64, b64, k_usize, n_usize, ldb, 11);
    @memset(c64, complexValue(fortran.ComplexF64, 0, 0));
    @memset(expected64, complexValue(fortran.ComplexF64, 0, 0));
    var alpha64 = complexValue(fortran.ComplexF64, 1, 0);
    var beta64 = complexValue(fortran.ComplexF64, 0, 0);
    referenceComplexGemmNoTrans(fortran.ComplexF64, m_usize, n_usize, k_usize, alpha64, a64, lda, b64, ldb, beta64, expected64, ldc);
    fortran.zgemm_(&ta, &tb, &m, &n, &k, &alpha64, a64.ptr, &lda, b64.ptr, &ldb, &beta64, c64.ptr, &ldc);
    for (expected64, c64) |expected, actual| try expectComplexApprox(fortran.ComplexF64, expected, actual, @as(f64, 1e-9));
}

test "complex gemm no-trans 3m padded tails match reference" {
    const allocator = std.testing.allocator;
    try expectComplexGemmNoTransCase(
        fortran.ComplexF32,
        allocator,
        33,
        35,
        128,
        37,
        131,
        39,
        complexValue(fortran.ComplexF32, 1, 0),
        complexValue(fortran.ComplexF32, 0, 0),
        @as(f32, 4e-2),
    );
    try expectComplexGemmNoTransCase(
        fortran.ComplexF64,
        allocator,
        33,
        35,
        128,
        37,
        131,
        39,
        complexValue(fortran.ComplexF64, 1, 0),
        complexValue(fortran.ComplexF64, 0, 0),
        @as(f64, 1e-9),
    );
}

test "complex gemm non-NN 3m handles all transpose pairs with odd padded storage" {
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
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        const tol = if (T == fortran.ComplexF32) @as(f32, 4e-2) else @as(f64, 1e-9);
        for (pairs) |pair| {
            try expectComplexGemmTransCase(
                T,
                pair.transa,
                pair.transb,
                33,
                35,
                128,
                complexValue(T, 1, 0),
                complexValue(T, 0, 0),
                tol,
            );
        }
    }
}

test "complex gemm 3m padded 128-row compute preserves 127-row storage and vector tails" {
    const allocator = std.testing.allocator;
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
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        const tol = if (T == fortran.ComplexF32) @as(f32, 4e-2) else @as(f64, 1e-9);
        for ([_]usize{ 31, 32, 33, 129 }) |k| {
            try expectComplexGemmNoTransCase(
                T,
                allocator,
                127,
                129,
                k,
                131,
                @intCast(k + 5),
                137,
                complexValue(T, 1, 0),
                complexValue(T, 0, 0),
                tol,
            );
            for (pairs) |pair| {
                try expectComplexGemmTransCase(
                    T,
                    pair.transa,
                    pair.transb,
                    127,
                    129,
                    k,
                    complexValue(T, 1, 0),
                    complexValue(T, 0, 0),
                    tol,
                );
            }
        }
    }
}

test "complex gemm non-NN 3m handles row and column vector edges" {
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
    const shapes = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 1, .n = 1024, .k = 128 },
        .{ .m = 1024, .n = 1, .k = 128 },
    };
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        const tol = if (T == fortran.ComplexF32) @as(f32, 4e-2) else @as(f64, 1e-9);
        for (pairs) |pair| {
            for (shapes) |shape| {
                try expectComplexGemmTransCase(
                    T,
                    pair.transa,
                    pair.transb,
                    shape.m,
                    shape.n,
                    shape.k,
                    complexValue(T, 1, 0),
                    complexValue(T, 0, 0),
                    tol,
                );
            }
        }
    }
}

test "sgemm no-trans direct tile path with tails" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 35, 37, 19, 37, 23, 39, 1.0, 0.0);
}

test "dgemm no-trans direct tile path with tails" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f64, allocator, 35, 37, 19, 37, 23, 39, 1.0, 0.0);
}

test "sgemm and dgemm no-trans direct multi-panel batches" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 70, 97, 31, 73, 37, 75, 1.0, 0.0);
    try expectGemmNoTransCase(f64, allocator, 70, 97, 31, 73, 37, 75, 1.0, 0.0);
}

test "sgemm no-trans transpose4 B-pack keeps sign" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 64, 65, 256, 67, 260, 69, 0.001, 0.0);
    try expectGemmNoTransCase(f32, allocator, 64, 257, 256, 67, 260, 69, 0.001, 0.0);
    try expectGemmNoTransCase(f32, allocator, 256, 256, 2048, 259, 2052, 263, 0.001, 0.0);
}

test "sgemm no-trans single-row wide result matches reference" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 1, 4096, 256, 1, 256, 1, 1.0, 0.0);
}

test "dgemm no-trans large panel batch matches reference" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f64, allocator, 64, 129, 128, 67, 132, 69, 0.001, 0.0);
}

test "sgemm and dgemm no-trans medium direct batches" {
    const allocator = std.testing.allocator;
    try expectGemmNoTransCase(f32, allocator, 130, 161, 67, 133, 71, 135, 1.0, 0.0);
    try expectGemmNoTransCase(f64, allocator, 130, 161, 67, 133, 71, 135, 1.0, 0.0);
}

test "gemm beta zero and alpha zero isolate inactive NaNs" {
    const nan = std.math.nan(f64);
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var k: fortran.BlasInt = 2;
    var lda: fortran.BlasInt = 2;
    var ldb: fortran.BlasInt = 2;
    var ldc: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ nan, nan, nan, nan };

    fortran.dgemm_(&ta, &tb, &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc);
    for (c) |value| try std.testing.expect(!std.math.isNan(value));
    try std.testing.expectApproxEqAbs(@as(f64, 23), c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 31), c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 46), c[3], 1e-12);

    alpha = 0;
    beta = 2;
    a = [_]f64{ nan, nan, nan, nan };
    b = [_]f64{ nan, nan, nan, nan };
    c = [_]f64{ 1, 2, 3, 4 };
    fortran.dgemm_(&ta, &tb, &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc);
    try std.testing.expectEqual(@as(f64, 2), c[0]);
    try std.testing.expectEqual(@as(f64, 4), c[1]);
    try std.testing.expectEqual(@as(f64, 6), c[2]);
    try std.testing.expectEqual(@as(f64, 8), c[3]);
}

test "complex gemm beta zero and alpha zero isolate inactive NaNs" {
    const nan = std.math.nan(f64);
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = 1;
    var n: fortran.BlasInt = 1;
    var k: fortran.BlasInt = 1;
    var lda: fortran.BlasInt = 1;
    var ldb: fortran.BlasInt = 1;
    var ldc: fortran.BlasInt = 1;
    var alpha = complexValue(fortran.ComplexF64, 1, 0);
    var beta = complexValue(fortran.ComplexF64, 0, 0);
    var a = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, 2, 1)};
    var b = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, 3, -1)};
    var c = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, nan, nan)};

    fortran.zgemm_(&ta, &tb, &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc);
    try std.testing.expect(!std.math.isNan(c[0].re));
    try std.testing.expect(!std.math.isNan(c[0].im));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 7, 1), c[0], @as(f64, 1e-12));

    alpha = complexValue(fortran.ComplexF64, 0, 0);
    beta = complexValue(fortran.ComplexF64, 2, -1);
    a = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, nan, nan)};
    b = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, nan, nan)};
    c = [_]fortran.ComplexF64{complexValue(fortran.ComplexF64, 4, 3)};
    fortran.zgemm_(&ta, &tb, &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 11, 2), c[0], @as(f64, 1e-12));
}

test "parallel sgemm and dgemm column split matches reference" {
    const allocator = std.testing.allocator;
    fortran.runtime.setMaxThreads(3);
    defer fortran.runtime.setMaxThreads(0);
    try expectGemmNoTransCase(f32, allocator, 96, 97, 96, 96, 96, 96, 0.75, -0.25);
    try expectGemmNoTransCase(f64, allocator, 96, 97, 96, 96, 96, 96, 0.75, -0.25);
}

test "parallel sgemm and dgemm row split matches reference" {
    const allocator = std.testing.allocator;
    fortran.runtime.setMaxThreads(10);
    defer fortran.runtime.setMaxThreads(0);
    try expectGemmNoTransCase(f32, allocator, 512, 16, 1024, 512, 1024, 512, 1, 0);
    try expectGemmNoTransCase(f64, allocator, 512, 16, 1024, 512, 1024, 512, 1, 0);
}

test "blas shutdown resets shared helper pool state" {
    const allocator = std.testing.allocator;
    fortran.runtime.setMaxThreads(4);
    defer fortran.runtime.setMaxThreads(0);
    try expectGemmNoTransCase(f32, allocator, 256, 32, 1024, 256, 1024, 256, 1, 0);
    fortran.zynum_blas_shutdown();
    try expectGemmNoTransCase(f32, allocator, 256, 32, 1024, 256, 1024, 256, 1, 0);
    fortran.zynum_blas_shutdown_();
}

test "dgemm invalid parameter leaves output unchanged" {
    var ta = [_]u8{'X'};
    var tb = [_]u8{'N'};
    var m: fortran.BlasInt = 1;
    var n: fortran.BlasInt = 1;
    var k: fortran.BlasInt = 1;
    var lda: fortran.BlasInt = 1;
    var ldb: fortran.BlasInt = 1;
    var ldc: fortran.BlasInt = 1;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var left_matrix = [_]f64{2};
    var right_matrix = [_]f64{3};
    var result_matrix = [_]f64{7};
    fortran.dgemm_(&ta, &tb, &m, &n, &k, &alpha, &left_matrix, &lda, &right_matrix, &ldb, &beta, &result_matrix, &ldc);
    try std.testing.expectEqual(@as(f64, 7), result_matrix[0]);
}

test "dtrmm invalid parameter leaves output unchanged" {
    var side = [_]u8{'X'};
    var upper = [_]u8{'U'};
    var no_trans = [_]u8{'N'};
    var non_unit = [_]u8{'N'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var lda: fortran.BlasInt = 2;
    var ldb: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var tri = [_]f64{ 1, 0, 2, 3 };
    var b = [_]f64{ 4, 5, 6, 7 };
    const original = b;
    fortran.dtrmm_(&side, &upper, &no_trans, &non_unit, &m, &n, &alpha, &tri, &lda, &b, &ldb);
    try std.testing.expectEqualSlices(f64, &original, &b);
}

test "dgemv invalid parameter leaves output unchanged" {
    var trans = [_]u8{'X'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var lda: fortran.BlasInt = 2;
    var inc: fortran.BlasInt = 1;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var matrix = [_]f64{ 1, 2, 3, 4 };
    var x = [_]f64{ 5, 6 };
    var y = [_]f64{ 9, 10 };
    const original = y;
    fortran.dgemv_(&trans, &m, &n, &alpha, &matrix, &lda, &x, &inc, &beta, &y, &inc);
    try std.testing.expectEqualSlices(f64, &original, &y);
}

test "dtbmv invalid parameter leaves vector unchanged" {
    var upper = [_]u8{'U'};
    var no_trans = [_]u8{'N'};
    var diag = [_]u8{'X'};
    var n: fortran.BlasInt = 2;
    var k: fortran.BlasInt = 1;
    var lda: fortran.BlasInt = 2;
    var inc: fortran.BlasInt = 1;
    var band = [_]f64{ 0, 1, 2, 3 };
    var x = [_]f64{ 4, 5 };
    const original = x;
    fortran.dtbmv_(&upper, &no_trans, &diag, &n, &k, &band, &lda, &x, &inc);
    try std.testing.expectEqualSlices(f64, &original, &x);
}

test "dger invalid parameter leaves matrix unchanged" {
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var incx: fortran.BlasInt = 0;
    var incy: fortran.BlasInt = 1;
    var lda: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var x = [_]f64{ 5, 6 };
    var y = [_]f64{ 7, 8 };
    var matrix = [_]f64{ 1, 2, 3, 4 };
    const original = matrix;
    fortran.dger_(&m, &n, &alpha, &x, &incx, &y, &incy, &matrix, &lda);
    try std.testing.expectEqualSlices(f64, &original, &matrix);
}

test "level3 symmetric invalid parameters leave outputs unchanged" {
    var invalid = [_]u8{'X'};
    var upper = [_]u8{'U'};
    var no_trans = [_]u8{'N'};
    var left = [_]u8{'L'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var k: fortran.BlasInt = 2;
    var lda: fortran.BlasInt = 2;
    var ldb: fortran.BlasInt = 2;
    var ldc: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };

    var symm_c = [_]f64{ 9, 10, 11, 12 };
    const original_symm_c = symm_c;
    fortran.dsymm_(&invalid, &upper, &m, &n, &alpha, &a, &lda, &b, &ldb, &beta, &symm_c, &ldc);
    try std.testing.expectEqualSlices(f64, &original_symm_c, &symm_c);

    var syrk_c = [_]f64{ 13, 14, 15, 16 };
    const original_syrk_c = syrk_c;
    fortran.dsyrk_(&upper, &invalid, &n, &k, &alpha, &a, &lda, &beta, &syrk_c, &ldc);
    try std.testing.expectEqualSlices(f64, &original_syrk_c, &syrk_c);

    var syr2k_c = [_]f64{ 17, 18, 19, 20 };
    const original_syr2k_c = syr2k_c;
    ldb = 1;
    fortran.dsyr2k_(&upper, &no_trans, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &syr2k_c, &ldc);
    try std.testing.expectEqualSlices(f64, &original_syr2k_c, &syr2k_c);

    var hemm_c = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 0), complexValue(fortran.ComplexF64, 2, 0),
        complexValue(fortran.ComplexF64, 3, 0), complexValue(fortran.ComplexF64, 4, 0),
    };
    const original_hemm_c = hemm_c;
    var z_alpha = complexValue(fortran.ComplexF64, 1, 0);
    var z_beta = complexValue(fortran.ComplexF64, 0, 0);
    var za = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 0), complexValue(fortran.ComplexF64, 2, 0),
        complexValue(fortran.ComplexF64, 3, 0), complexValue(fortran.ComplexF64, 4, 0),
    };
    var zb = za;
    fortran.zhemm_(&left, &invalid, &m, &n, &z_alpha, &za, &lda, &zb, &lda, &z_beta, &hemm_c, &ldc);
    for (original_hemm_c, hemm_c) |want, got| try expectComplexApprox(fortran.ComplexF64, want, got, @as(f64, 1e-12));
}

test "dgemv and negative increment" {
    var t = [_]u8{'N'};
    var m: fortran.BlasInt = 2;
    var n: fortran.BlasInt = 2;
    var lda: fortran.BlasInt = 2;
    var inc: fortran.BlasInt = -1;
    var incy: fortran.BlasInt = 1;
    var alpha: f64 = 1;
    var beta: f64 = 0;
    var matrix = [_]f64{ 1, 2, 3, 4 };
    var input_vector = [_]f64{ 20, 10 };
    var result_vector = [_]f64{ 0, 0 };
    fortran.dgemv_(&t, &m, &n, &alpha, &matrix, &lda, &input_vector, &inc, &beta, &result_vector, &incy);
    try std.testing.expectApproxEqAbs(@as(f64, 70), result_vector[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), result_vector[1], 1e-12);
}

test "complex dotc" {
    var n: fortran.BlasInt = 2;
    var inc: fortran.BlasInt = 1;
    var left_vector = [_]fortran.ComplexF32{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
    var right_vector = [_]fortran.ComplexF32{ .{ .re = 4, .im = -2 }, .{ .re = 0, .im = 5 } };
    const result = fortran.cdotc_(&n, &left_vector, &inc, &right_vector, &inc);
    try std.testing.expectApproxEqAbs(@as(f32, -5), result.re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), result.im, 1e-5);
}

test "level1 double vector fallback smoke" {
    var n: fortran.BlasInt = 3;
    var inc2: fortran.BlasInt = 2;
    var inc1: fortran.BlasInt = 1;

    var source = [_]f64{ 1, -777, 2, -777, 3 };
    var copied = [_]f64{ 0, 99, 0, 99, 0 };
    fortran.dcopy_(&n, &source, &inc2, &copied, &inc2);
    try std.testing.expectEqual(@as(f64, 1), copied[0]);
    try std.testing.expectEqual(@as(f64, 99), copied[1]);
    try std.testing.expectEqual(@as(f64, 2), copied[2]);
    try std.testing.expectEqual(@as(f64, 99), copied[3]);
    try std.testing.expectEqual(@as(f64, 3), copied[4]);

    var self_copy = [_]f64{ 1, 2, 3, 4 };
    var copy4: fortran.BlasInt = 4;
    fortran.dcopy_(&copy4, &self_copy, &inc1, &self_copy, &inc1);
    try std.testing.expectEqualSlices(f64, &[_]f64{ 1, 2, 3, 4 }, &self_copy);

    var overlap_forward = [_]f64{ 1, 2, 3, 4, 5 };
    fortran.dcopy_(&copy4, overlap_forward[0..].ptr, &inc1, overlap_forward[1..].ptr, &inc1);
    try std.testing.expectEqualSlices(f64, &[_]f64{ 1, 1, 2, 3, 4 }, &overlap_forward);

    var overlap_backward = [_]f64{ 1, 2, 3, 4, 5 };
    fortran.dcopy_(&copy4, overlap_backward[1..].ptr, &inc1, overlap_backward[0..].ptr, &inc1);
    try std.testing.expectEqualSlices(f64, &[_]f64{ 2, 3, 4, 5, 5 }, &overlap_backward);

    var alpha: f64 = 2;
    var y = [_]f64{ 10, -5, 20, -5, 30 };
    fortran.daxpy_(&n, &alpha, &source, &inc2, &y, &inc2);
    try std.testing.expectEqual(@as(f64, 12), y[0]);
    try std.testing.expectEqual(@as(f64, -5), y[1]);
    try std.testing.expectEqual(@as(f64, 24), y[2]);
    try std.testing.expectEqual(@as(f64, -5), y[3]);
    try std.testing.expectEqual(@as(f64, 36), y[4]);

    try std.testing.expectEqual(@as(f64, 168), fortran.ddot_(&n, &source, &inc2, &y, &inc2));

    var scale = [_]f64{ 2, -1, 4, -1, 6 };
    var scale_alpha: f64 = -0.5;
    fortran.dscal_(&n, &scale_alpha, &scale, &inc2);
    try std.testing.expectEqual(@as(f64, -1), scale[0]);
    try std.testing.expectEqual(@as(f64, -1), scale[1]);
    try std.testing.expectEqual(@as(f64, -2), scale[2]);
    try std.testing.expectEqual(@as(f64, -1), scale[3]);
    try std.testing.expectEqual(@as(f64, -3), scale[4]);

    var norm_input = [_]f64{ 3, -777, 4 };
    try std.testing.expectApproxEqAbs(@as(f64, 5), fortran.dnrm2_(&inc2, &norm_input, &inc2), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), fortran.dasum_(&n, &source, &inc2), 1e-12);

    var iamax_input = [_]f64{ -1, 99, 4, 99, -5 };
    try std.testing.expectEqual(@as(fortran.BlasInt, 3), fortran.idamax_(&n, &iamax_input, &inc2));

    var icamax_n: fortran.BlasInt = 4;
    var icamax_input = [_]fortran.ComplexF32{
        .{ .re = 1, .im = 1 },
        .{ .re = -4, .im = 3 },
        .{ .re = 2, .im = -5 },
        .{ .re = 0, .im = 6 },
    };
    try std.testing.expectEqual(@as(fortran.BlasInt, 2), fortran.icamax_(&icamax_n, &icamax_input, &inc1));

    var rot_n: fortran.BlasInt = 2;
    var c: f64 = 0;
    var s: f64 = 1;
    var xrot = [_]f64{ 1, 2 };
    var yrot = [_]f64{ 3, 4 };
    fortran.drot_(&rot_n, &xrot, &inc1, &yrot, &inc1, &c, &s);
    try std.testing.expectEqual(@as(f64, 3), xrot[0]);
    try std.testing.expectEqual(@as(f64, 4), xrot[1]);
    try std.testing.expectEqual(@as(f64, -1), yrot[0]);
    try std.testing.expectEqual(@as(f64, -2), yrot[1]);

    var left32 = [_]f32{ 1, -9, 2 };
    var right32 = [_]f32{ 3, -9, 4 };
    var two: fortran.BlasInt = 2;
    try std.testing.expectApproxEqAbs(@as(f64, 11), fortran.dsdot_(&two, &left32, &inc2, &right32, &inc2), 1e-12);

    var cancel_n: fortran.BlasInt = 3;
    var cancellation_x = [_]f32{ 1.0e10, 1, -1.0e10 };
    var cancellation_y = [_]f32{ 1, 1, 1 };
    var sb: f32 = 0.25;
    try std.testing.expectApproxEqAbs(@as(f64, 1), fortran.dsdot_(&cancel_n, &cancellation_x, &inc1, &cancellation_y, &inc1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), fortran.sdsdot_(&cancel_n, &sb, &cancellation_x, &inc1, &cancellation_y, &inc1), 1e-6);

    var swap_left = [_]f64{ 1, 2 };
    var swap_right = [_]f64{ 3, 4 };
    fortran.dswap_(&rot_n, &swap_left, &inc1, &swap_right, &inc1);
    try std.testing.expectEqual(@as(f64, 3), swap_left[0]);
    try std.testing.expectEqual(@as(f64, 4), swap_left[1]);
    try std.testing.expectEqual(@as(f64, 1), swap_right[0]);
    try std.testing.expectEqual(@as(f64, 2), swap_right[1]);
}

test "fortran large unit-stride real Level 1 affine and rotations" {
    const n: usize = 1024 * 1024;
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, n);
    defer allocator.free(y);

    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
        yv.* = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
    }
    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 1;

    var scale: f32 = 0.5;
    fortran.sscal_(&nn, &scale, x.ptr, &inc);
    for (x, 0..) |xv, i| {
        const x0 = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
        try std.testing.expectApproxEqAbs(scale * x0, xv, 1e-6);
    }

    for (x, 0..) |*xv, i| xv.* = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
    var axpy_alpha: f32 = 0.125;
    fortran.saxpy_(&nn, &axpy_alpha, x.ptr, &inc, y.ptr, &inc);
    for (x, y, 0..) |xv, yv, i| {
        const y0 = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
        try std.testing.expectApproxEqAbs(axpy_alpha * xv + y0, yv, 1e-6);
    }

    for (y, 0..) |*yv, i| yv.* = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
    var alpha: f32 = 0.125;
    var beta: f32 = 0.875;
    fortran.saxpby_(&nn, &alpha, x.ptr, &inc, &beta, y.ptr, &inc);
    for (x, y, 0..) |xv, yv, i| {
        const y0 = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
        try std.testing.expectApproxEqAbs(alpha * xv + beta * y0, yv, 1e-6);
    }

    for (y, 0..) |*yv, i| yv.* = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
    var c: f32 = 0.8;
    var s: f32 = 0.6;
    fortran.srot_(&nn, x.ptr, &inc, y.ptr, &inc, &c, &s);
    for (x, y, 0..) |xv, yv, i| {
        const x0 = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
        const y0 = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
        try std.testing.expectApproxEqAbs(c * x0 + s * y0, xv, 1e-6);
        try std.testing.expectApproxEqAbs(c * y0 - s * x0, yv, 1e-6);
    }

    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
        yv.* = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
    }
    const param = [_]f32{ -1, 0.8, -0.6, 0.6, 0.8 };
    fortran.srotm_(&nn, x.ptr, &inc, y.ptr, &inc, &param);
    for (x, y, 0..) |xv, yv, i| {
        const x0 = (@as(f32, @floatFromInt(i % 251)) - 125) / 251;
        const y0 = (@as(f32, @floatFromInt(i % 239)) - 119) / 239;
        try std.testing.expectApproxEqAbs(param[1] * x0 + param[3] * y0, xv, 1e-6);
        try std.testing.expectApproxEqAbs(param[2] * x0 + param[4] * y0, yv, 1e-6);
    }
}

test "fortran SME rotation tails and rotm variants" {
    const n: usize = 64 * 1024 + 17;
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, n);
    defer allocator.free(y);
    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 1;

    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = (@as(f32, @floatFromInt(i % 31)) - 15) / 31;
        yv.* = (@as(f32, @floatFromInt(i % 29)) - 14) / 29;
    }
    var c: f32 = 0.8;
    var s: f32 = 0.6;
    fortran.srot_(&nn, x.ptr, &inc, y.ptr, &inc, &c, &s);
    for (x, y, 0..) |xv, yv, i| {
        const x0 = (@as(f32, @floatFromInt(i % 31)) - 15) / 31;
        const y0 = (@as(f32, @floatFromInt(i % 29)) - 14) / 29;
        try std.testing.expectApproxEqAbs(c * x0 + s * y0, xv, 1e-6);
        try std.testing.expectApproxEqAbs(c * y0 - s * x0, yv, 1e-6);
    }

    const nan = std.math.nan(f32);
    const params = [_][5]f32{
        .{ 0, nan, -0.25, 0.5, nan },
        .{ 1, 0.75, nan, nan, 0.625 },
    };
    for (params) |param| {
        for (x, y, 0..) |*xv, *yv, i| {
            xv.* = (@as(f32, @floatFromInt(i % 31)) - 15) / 31;
            yv.* = (@as(f32, @floatFromInt(i % 29)) - 14) / 29;
        }
        fortran.srotm_(&nn, x.ptr, &inc, y.ptr, &inc, &param);
        for (x, y, 0..) |xv, yv, i| {
            const x0 = (@as(f32, @floatFromInt(i % 31)) - 15) / 31;
            const y0 = (@as(f32, @floatFromInt(i % 29)) - 14) / 29;
            if (param[0] == 0) {
                try std.testing.expectApproxEqAbs(x0 + param[3] * y0, xv, 1e-6);
                try std.testing.expectApproxEqAbs(param[2] * x0 + y0, yv, 1e-6);
            } else {
                try std.testing.expectApproxEqAbs(param[1] * x0 + y0, xv, 1e-6);
                try std.testing.expectApproxEqAbs(-x0 + param[4] * y0, yv, 1e-6);
            }
        }
    }
}

test "fortran large unit-stride swap" {
    const n: usize = 512 * 1024;
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);

    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = @floatFromInt(i % 251);
        yv.* = -@as(f64, @floatFromInt(i % 239));
    }
    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 1;
    fortran.dswap_(&nn, x.ptr, &inc, y.ptr, &inc);
    for (x, y, 0..) |xv, yv, i| {
        try std.testing.expectEqual(-@as(f64, @floatFromInt(i % 239)), xv);
        try std.testing.expectEqual(@as(f64, @floatFromInt(i % 251)), yv);
    }
}

test "fortran exact 8 KiB unit-stride copy" {
    const n: usize = 1024;
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);

    for (x, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 251)) - 73.5;
    @memset(y, std.math.nan(f64));

    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 1;
    fortran.dcopy_(&nn, x.ptr, &inc, y.ptr, &inc);
    try std.testing.expectEqualSlices(f64, x, y);
}

test "fortran streaming unit-stride swap tails" {
    const allocator = std.testing.allocator;
    const sizes = [_]usize{
        16 * 1024 + 17, // less than one predicated vgx4 byte group remains
        16 * 1024 + 127, // one full and one partial vgx4 byte group remain
    };
    for (sizes) |n| {
        const x = try allocator.alloc(f32, n);
        defer allocator.free(x);
        const y = try allocator.alloc(f32, n);
        defer allocator.free(y);
        for (x, y, 0..) |*xv, *yv, i| {
            xv.* = @floatFromInt(i % 251);
            yv.* = -@as(f32, @floatFromInt(i % 239));
        }

        var nn: fortran.BlasInt = @intCast(n);
        var inc: fortran.BlasInt = 1;
        fortran.sswap_(&nn, x.ptr, &inc, y.ptr, &inc);
        for (x, y, 0..) |xv, yv, i| {
            try std.testing.expectEqual(-@as(f32, @floatFromInt(i % 239)), xv);
            try std.testing.expectEqual(@as(f32, @floatFromInt(i % 251)), yv);
        }
    }
}

test "fortran large complex iamax keeps first tie across tasks" {
    const n: usize = 256 * 1024;
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(fortran.ComplexF32, n);
    defer allocator.free(x);
    @memset(x, .{ .re = 0, .im = 0 });
    const first: usize = 17 * 1024 + 3;
    const second: usize = 233 * 1024 + 5;
    x[first] = .{ .re = -9, .im = 8 };
    x[second] = .{ .re = 10, .im = -7 };

    var nn: fortran.BlasInt = @intCast(n);
    var inc: fortran.BlasInt = 1;
    try std.testing.expectEqual(
        @as(fortran.BlasInt, @intCast(first + 1)),
        fortran.icamax_(&nn, x.ptr, &inc),
    );
}

test "level2 band symmetric hermitian and triangular fallbacks" {
    var no_trans = [_]u8{'N'};
    var upper = [_]u8{'U'};
    var non_unit = [_]u8{'N'};
    var n3: fortran.BlasInt = 3;
    var m3: fortran.BlasInt = 3;
    var kl: fortran.BlasInt = 1;
    var ku: fortran.BlasInt = 1;
    var lda_band: fortran.BlasInt = 3;
    var inc: fortran.BlasInt = 1;
    var alpha: f64 = 1;
    var beta: f64 = 0;

    var band = [_]f64{
        777, 1, 3,
        2,   4, 6,
        5,   7, 777,
    };
    var x3 = [_]f64{ 1, 2, 3 };
    var y3 = [_]f64{ 0, 0, 0 };
    fortran.dgbmv_(&no_trans, &m3, &n3, &kl, &ku, &alpha, &band, &lda_band, &x3, &inc, &beta, &y3, &inc);
    try std.testing.expectApproxEqAbs(@as(f64, 5), y3[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26), y3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 33), y3[2], 1e-12);

    var lda3: fortran.BlasInt = 3;
    var sym = [_]f64{
        1, 999, 999,
        2, 4,   999,
        3, 5,   6,
    };
    y3 = [_]f64{ 0, 0, 0 };
    fortran.dsymv_(&upper, &n3, &alpha, &sym, &lda3, &x3, &inc, &beta, &y3, &inc);
    try std.testing.expectApproxEqAbs(@as(f64, 14), y3[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25), y3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 31), y3[2], 1e-12);

    var n2: fortran.BlasInt = 2;
    var lda2: fortran.BlasInt = 2;
    var z_alpha = complexValue(fortran.ComplexF64, 1, 0);
    var z_beta = complexValue(fortran.ComplexF64, 0, 0);
    var herm = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 99), complexValue(fortran.ComplexF64, 777, 777),
        complexValue(fortran.ComplexF64, 2, 1),  complexValue(fortran.ComplexF64, 3, -7),
    };
    var zx = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 1, 1), complexValue(fortran.ComplexF64, 2, -1) };
    var zy = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0) };
    fortran.zhemv_(&upper, &n2, &z_alpha, &herm, &lda2, &zx, &inc, &z_beta, &zy, &inc);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 6, 1), zy[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 9, -2), zy[1], @as(f64, 1e-12));

    var tri = [_]f64{
        1, 777,
        2, 3,
    };
    var tx = [_]f64{ 4, 5 };
    fortran.dtrmv_(&upper, &no_trans, &non_unit, &n2, &tri, &lda2, &tx, &inc);
    try std.testing.expectApproxEqAbs(@as(f64, 14), tx[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), tx[1], 1e-12);

    var solve = [_]f64{ 14, 15 };
    fortran.dtrsv_(&upper, &no_trans, &non_unit, &n2, &tri, &lda2, &solve, &inc);
    try std.testing.expectApproxEqAbs(@as(f64, 4), solve[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), solve[1], 1e-12);
}

test "level2 rank update fallbacks" {
    var n2: fortran.BlasInt = 2;
    var m2: fortran.BlasInt = 2;
    var inc: fortran.BlasInt = 1;
    var lda2: fortran.BlasInt = 2;
    var upper = [_]u8{'U'};
    var alpha: f64 = 2;

    var x = [_]f64{ 1, 2 };
    var y = [_]f64{ 3, 4, 5 };
    var a = [_]f64{ 0, 0, 0, 0, 0, 0 };
    var n3: fortran.BlasInt = 3;
    fortran.dger_(&m2, &n3, &alpha, &x, &inc, &y, &inc, &a, &lda2);
    try std.testing.expectApproxEqAbs(@as(f64, 6), a[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12), a[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8), a[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16), a[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10), a[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20), a[5], 1e-12);

    var z_alpha = complexValue(fortran.ComplexF64, 1, 0);
    var zx = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 1, 1), complexValue(fortran.ComplexF64, 2, 0) };
    var zy = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 3, 1), complexValue(fortran.ComplexF64, -1, 2) };
    var za = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
    };
    fortran.zgerc_(&m2, &n2, &z_alpha, &zx, &inc, &zy, &inc, &za, &lda2);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 4, 2), za[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 6, -2), za[1], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 1, -3), za[2], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, -2, -4), za[3], @as(f64, 1e-12));

    var syr2_alpha: f64 = 1;
    var sx = [_]f64{ 1, 2 };
    var sy = [_]f64{ 3, 4 };
    var sym = [_]f64{
        10, 777,
        20, 30,
    };
    fortran.dsyr2_(&upper, &n2, &syr2_alpha, &sx, &inc, &sy, &inc, &sym, &lda2);
    try std.testing.expectApproxEqAbs(@as(f64, 16), sym[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 777), sym[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30), sym[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 46), sym[3], 1e-12);

    var hz_alpha = complexValue(fortran.ComplexF64, 1, 0);
    var hx = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 1, 1), complexValue(fortran.ComplexF64, 2, 0) };
    var hy = [_]fortran.ComplexF64{ complexValue(fortran.ComplexF64, 3, 0), complexValue(fortran.ComplexF64, -1, 1) };
    var herm = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 99), complexValue(fortran.ComplexF64, 777, 777),
        complexValue(fortran.ComplexF64, 2, 1),  complexValue(fortran.ComplexF64, 5, 88),
    };
    fortran.zher2_(&upper, &n2, &hz_alpha, &hx, &inc, &hy, &inc, &herm, &lda2);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 7, 0), herm[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 777, 777), herm[1], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 8, -1), herm[2], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 1, 0), herm[3], @as(f64, 1e-12));
}

test "level3 symmetric hermitian rank-k and triangular fallbacks" {
    var left = [_]u8{'L'};
    var upper = [_]u8{'U'};
    var no_trans = [_]u8{'N'};
    var non_unit = [_]u8{'N'};
    var n2: fortran.BlasInt = 2;
    var m2: fortran.BlasInt = 2;
    var k2: fortran.BlasInt = 2;
    var lda2: fortran.BlasInt = 2;
    var ldb2: fortran.BlasInt = 2;
    var ldc2: fortran.BlasInt = 2;
    var alpha: f64 = 1;
    var beta: f64 = 0;

    var sym_a = [_]f64{
        1, 777,
        2, 3,
    };
    var b = [_]f64{
        4, 6,
        5, 7,
    };
    var cmat = [_]f64{ 0, 0, 0, 0 };
    fortran.dsymm_(&left, &upper, &m2, &n2, &alpha, &sym_a, &lda2, &b, &ldb2, &beta, &cmat, &ldc2);
    try std.testing.expectApproxEqAbs(@as(f64, 16), cmat[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26), cmat[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19), cmat[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 31), cmat[3], 1e-12);

    var z_alpha = complexValue(fortran.ComplexF64, 1, 0);
    var z_beta = complexValue(fortran.ComplexF64, 0, 0);
    var herm_a = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 99), complexValue(fortran.ComplexF64, 777, 777),
        complexValue(fortran.ComplexF64, 2, 1),  complexValue(fortran.ComplexF64, 3, -7),
    };
    var ident = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 0), complexValue(fortran.ComplexF64, 0, 0),
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 1, 0),
    };
    var zc = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
    };
    fortran.zhemm_(&left, &upper, &m2, &n2, &z_alpha, &herm_a, &lda2, &ident, &ldb2, &z_beta, &zc, &ldc2);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 1, 0), zc[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 2, -1), zc[1], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 2, 1), zc[2], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 3, 0), zc[3], @as(f64, 1e-12));

    var rank_a = [_]f64{
        1, 3,
        2, 4,
    };
    var rank_c = [_]f64{ 0, 777, 0, 0 };
    fortran.dsyrk_(&upper, &no_trans, &n2, &k2, &alpha, &rank_a, &lda2, &beta, &rank_c, &ldc2);
    try std.testing.expectApproxEqAbs(@as(f64, 5), rank_c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 777), rank_c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11), rank_c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25), rank_c[3], 1e-12);

    var zrank_a = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 1), complexValue(fortran.ComplexF64, 3, 0),
        complexValue(fortran.ComplexF64, 2, 0), complexValue(fortran.ComplexF64, -1, 1),
    };
    var zrank_c = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 777, 777),
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
    };
    fortran.zherk_(&upper, &no_trans, &n2, &k2, &alpha, &zrank_a, &lda2, &beta, &zrank_c, &ldc2);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 6, 0), zrank_c[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 777, 777), zrank_c[1], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 1, 1), zrank_c[2], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 11, 0), zrank_c[3], @as(f64, 1e-12));

    var rank_b = [_]f64{
        5, 7,
        6, 8,
    };
    var rank2_c = [_]f64{ 0, 777, 0, 0 };
    fortran.dsyr2k_(&upper, &no_trans, &n2, &k2, &alpha, &rank_a, &lda2, &rank_b, &ldb2, &beta, &rank2_c, &ldc2);
    try std.testing.expectApproxEqAbs(@as(f64, 34), rank2_c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 777), rank2_c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 62), rank2_c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 106), rank2_c[3], 1e-12);

    var zrank2_a = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 1, 0), complexValue(fortran.ComplexF64, 3, 0),
        complexValue(fortran.ComplexF64, 2, 0), complexValue(fortran.ComplexF64, 4, 0),
    };
    var zrank_b = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 5, 0), complexValue(fortran.ComplexF64, 7, 0),
        complexValue(fortran.ComplexF64, 6, 0), complexValue(fortran.ComplexF64, 8, 0),
    };
    var zrank2_c = [_]fortran.ComplexF64{
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 777, 777),
        complexValue(fortran.ComplexF64, 0, 0), complexValue(fortran.ComplexF64, 0, 0),
    };
    fortran.zher2k_(&upper, &no_trans, &n2, &k2, &z_alpha, &zrank2_a, &lda2, &zrank_b, &ldb2, &beta, &zrank2_c, &ldc2);
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 34, 0), zrank2_c[0], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 777, 777), zrank2_c[1], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 62, 0), zrank2_c[2], @as(f64, 1e-12));
    try expectComplexApprox(fortran.ComplexF64, complexValue(fortran.ComplexF64, 106, 0), zrank2_c[3], @as(f64, 1e-12));

    var tri = [_]f64{
        1, 777,
        2, 3,
    };
    var triangular_b = [_]f64{
        4, 6,
        5, 7,
    };
    fortran.dtrmm_(&left, &upper, &no_trans, &non_unit, &m2, &n2, &alpha, &tri, &lda2, &triangular_b, &ldb2);
    try std.testing.expectApproxEqAbs(@as(f64, 16), triangular_b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18), triangular_b[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19), triangular_b[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 21), triangular_b[3], 1e-12);

    fortran.dtrsm_(&left, &upper, &no_trans, &non_unit, &m2, &n2, &alpha, &tri, &lda2, &triangular_b, &ldb2);
    try std.testing.expectApproxEqAbs(@as(f64, 4), triangular_b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), triangular_b[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), triangular_b[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7), triangular_b[3], 1e-12);
}

fn refTransChar(trans: ref.Trans) [1]u8 {
    return .{switch (trans) {
        .no_trans => 'N',
        .trans => 'T',
        .conj_trans => 'C',
    }};
}

fn refUploChar(uplo: ref.Uplo) [1]u8 {
    return .{if (uplo == .lower) 'L' else 'U'};
}

fn refDiagChar(diag: ref.Diag) [1]u8 {
    return .{if (diag == .unit) 'U' else 'N'};
}

fn expectApproxSlice(comptime T: type, expected: []const T, actual: []const T, tol: anytype) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| try ref.expectApprox(T, want, got, tol);
}

fn fillGbMatrix(comptime T: type, rng: *ref.Rng, a: []T, m: usize, n: usize, kl: usize, ku: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, -300 - @as(f64, @floatFromInt(i)), 91);
    for (0..n) |col| {
        for (0..m) |row| {
            if (ref.gbIndexColMajor(m, n, kl, ku, lda, row, col)) |idx| a[idx] = rng.scalar(T);
        }
    }
}

fn fillSymBandMatrix(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, a: []T, n: usize, k: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, 500 + @as(f64, @floatFromInt(i)), -37);
    for (0..n) |col| {
        for (0..n) |row| {
            const stored = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
            if (!stored) continue;
            if (ref.symBandIndex(uplo, n, k, lda, row, col)) |idx| a[idx] = rng.scalar(T);
        }
    }
}

fn fillPackedSymMatrix(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, ap: []T, n: usize) void {
    for (0..ap.len) |i| ap[i] = ref.fromParts(T, -80 - @as(f64, @floatFromInt(i)), 55);
    for (0..n) |col| {
        for (0..n) |row| {
            const stored = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
            if (!stored) continue;
            ap[ref.packedIndex(uplo, n, row, col)] = rng.scalar(T);
        }
    }
}

fn fillTriBandMatrix(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, a: []T, n: usize, k: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = ref.fromParts(T, 700 + @as(f64, @floatFromInt(i)), -101);
    for (0..n) |col| {
        for (0..n) |row| {
            if (ref.triBandIndex(uplo, k, lda, row, col)) |idx| {
                a[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) a[idx] = ref.add(T, a[idx], ref.fromParts(T, 3.0 + @as(f64, @floatFromInt(row)), 0.5));
            }
        }
    }
}

fn fillTriPackedMatrix(comptime T: type, rng: *ref.Rng, uplo: ref.Uplo, diag: ref.Diag, ap: []T, n: usize) void {
    for (0..ap.len) |i| ap[i] = ref.fromParts(T, -900 - @as(f64, @floatFromInt(i)), 73);
    for (0..n) |col| {
        for (0..n) |row| {
            if (ref.triPackedIndex(uplo, n, row, col)) |idx| {
                ap[idx] = rng.scalar(T);
                if (row == col and diag == .non_unit) ap[idx] = ref.add(T, ap[idx], ref.fromParts(T, 2.5 + @as(f64, @floatFromInt(row)), 0.25));
            }
        }
    }
}

fn runDgbmvReferenceCase(allocator: std.mem.Allocator, rng: *ref.Rng, trans: ref.Trans, incx: isize, incy: isize) !void {
    const T = f64;
    const m: usize = 4;
    const n: usize = 3;
    const kl: usize = 2;
    const ku: usize = 1;
    const lda: usize = 5;
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, ref.vectorStorageLen(lenx, incx));
    defer allocator.free(x);
    const y = try allocator.alloc(T, ref.vectorStorageLen(leny, incy));
    defer allocator.free(y);
    const expected = try allocator.alloc(T, y.len);
    defer allocator.free(expected);

    fillGbMatrix(T, rng, a, m, n, kl, ku, lda);
    ref.fillVector(T, rng, x, lenx, incx);
    ref.fillVector(T, rng, y, leny, incy);
    @memcpy(expected, y);

    const alpha = @as(T, 0.75);
    const beta = @as(T, -0.5);
    ref.gbmvColMajor(T, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, expected, incy);

    var t = refTransChar(trans);
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kll: fortran.BlasInt = @intCast(kl);
    var kuu: fortran.BlasInt = @intCast(ku);
    var aa = alpha;
    var bb = beta;
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    fortran.dgbmv_(&t, &mm, &nn, &kll, &kuu, &aa, a.ptr, &lda_arg, x.ptr, &ix, &bb, y.ptr, &iy);
    try expectApproxSlice(T, expected, y, @as(T, 1e-12));
}

fn runZgbmvReferenceCase(allocator: std.mem.Allocator, rng: *ref.Rng, trans: ref.Trans, incx: isize, incy: isize) !void {
    const T = fortran.ComplexF64;
    const m: usize = 4;
    const n: usize = 3;
    const kl: usize = 2;
    const ku: usize = 1;
    const lda: usize = 5;
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, ref.vectorStorageLen(lenx, incx));
    defer allocator.free(x);
    const y = try allocator.alloc(T, ref.vectorStorageLen(leny, incy));
    defer allocator.free(y);
    const expected = try allocator.alloc(T, y.len);
    defer allocator.free(expected);

    fillGbMatrix(T, rng, a, m, n, kl, ku, lda);
    ref.fillVector(T, rng, x, lenx, incx);
    ref.fillVector(T, rng, y, leny, incy);
    @memcpy(expected, y);

    const alpha = ref.fromParts(T, 0.75, -0.25);
    const beta = ref.fromParts(T, -0.5, 0.125);
    ref.gbmvColMajor(T, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, expected, incy);

    var t = refTransChar(trans);
    var mm: fortran.BlasInt = @intCast(m);
    var nn: fortran.BlasInt = @intCast(n);
    var kll: fortran.BlasInt = @intCast(kl);
    var kuu: fortran.BlasInt = @intCast(ku);
    var aa = alpha;
    var bb = beta;
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    fortran.zgbmv_(&t, &mm, &nn, &kll, &kuu, &aa, a.ptr, &lda_arg, x.ptr, &ix, &bb, y.ptr, &iy);
    try expectApproxSlice(T, expected, y, @as(f64, 1e-12));
}

test "fortran gbmv randomized transpose stride lda reference" {
    const allocator = std.testing.allocator;
    var rng = ref.Rng.init(0xabc0_0101);
    try runDgbmvReferenceCase(allocator, &rng, .no_trans, 1, 2);
    try runDgbmvReferenceCase(allocator, &rng, .trans, -2, 1);
    try runDgbmvReferenceCase(allocator, &rng, .conj_trans, 2, -1);
    try runZgbmvReferenceCase(allocator, &rng, .no_trans, -1, 2);
    try runZgbmvReferenceCase(allocator, &rng, .trans, 2, -1);
    try runZgbmvReferenceCase(allocator, &rng, .conj_trans, -2, 1);
}

fn runBandPackedMvCases(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, incx: isize, incy: isize, herm: bool) !void {
    const n: usize = 4;
    const k: usize = 2;
    const lda: usize = 4;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const ap = try allocator.alloc(T, n * (n + 1) / 2);
    defer allocator.free(ap);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const y_band = try allocator.alloc(T, ref.vectorStorageLen(n, incy));
    defer allocator.free(y_band);
    const y_packed = try allocator.alloc(T, ref.vectorStorageLen(n, incy));
    defer allocator.free(y_packed);
    const expected_band = try allocator.alloc(T, y_band.len);
    defer allocator.free(expected_band);
    const expected_packed = try allocator.alloc(T, y_packed.len);
    defer allocator.free(expected_packed);

    fillSymBandMatrix(T, rng, uplo, a, n, k, lda);
    fillPackedSymMatrix(T, rng, uplo, ap, n);
    ref.fillVector(T, rng, x, n, incx);
    ref.fillVector(T, rng, y_band, n, incy);
    ref.fillVector(T, rng, y_packed, n, incy);
    @memcpy(expected_band, y_band);
    @memcpy(expected_packed, y_packed);

    const alpha = if (comptime ref.isComplex(T)) ref.fromParts(T, 0.7, -0.2) else @as(T, 0.7);
    const beta = if (comptime ref.isComplex(T)) ref.fromParts(T, -0.3, 0.4) else @as(T, -0.3);
    ref.sbmvColMajor(T, uplo, n, k, alpha, a, lda, x, incx, beta, expected_band, incy, herm);
    ref.spmvColMajor(T, uplo, n, alpha, ap, x, incx, beta, expected_packed, incy, herm);

    var uu = refUploChar(uplo);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    var aa = alpha;
    var bb = beta;
    if (T == f64) {
        fortran.dsbmv_(&uu, &nn, &kk, &aa, a.ptr, &lda_arg, x.ptr, &ix, &bb, y_band.ptr, &iy);
        fortran.dspmv_(&uu, &nn, &aa, ap.ptr, x.ptr, &ix, &bb, y_packed.ptr, &iy);
        try expectApproxSlice(T, expected_band, y_band, @as(T, 1e-12));
        try expectApproxSlice(T, expected_packed, y_packed, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64) {
        fortran.zhbmv_(&uu, &nn, &kk, &aa, a.ptr, &lda_arg, x.ptr, &ix, &bb, y_band.ptr, &iy);
        fortran.zhpmv_(&uu, &nn, &aa, ap.ptr, x.ptr, &ix, &bb, y_packed.ptr, &iy);
        try expectApproxSlice(T, expected_band, y_band, @as(f64, 1e-12));
        try expectApproxSlice(T, expected_packed, y_packed, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

test "fortran banded and packed symmetric hermitian randomized reference" {
    const allocator = std.testing.allocator;
    var rng = ref.Rng.init(0xabc0_0202);
    try runBandPackedMvCases(f64, allocator, &rng, .upper, -2, 1, false);
    try runBandPackedMvCases(f64, allocator, &rng, .lower, 1, -2, false);
    try runBandPackedMvCases(fortran.ComplexF64, allocator, &rng, .upper, 2, -1, true);
    try runBandPackedMvCases(fortran.ComplexF64, allocator, &rng, .lower, -1, 2, true);
}

fn runTriBandMvCase(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, trans: ref.Trans, diag: ref.Diag, incx: isize) !void {
    const n: usize = 4;
    const k: usize = 2;
    const lda: usize = 4;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const expected = try allocator.alloc(T, x.len);
    defer allocator.free(expected);
    const work = try allocator.alloc(T, n);
    defer allocator.free(work);
    fillTriBandMatrix(T, rng, uplo, diag, a, n, k, lda);
    ref.fillVector(T, rng, x, n, incx);
    @memcpy(expected, x);
    ref.tbmvColMajor(T, uplo, trans, diag, n, k, a, lda, expected, incx, work);

    var uu = refUploChar(uplo);
    var tt = refTransChar(trans);
    var dd = refDiagChar(diag);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var ix: fortran.BlasInt = @intCast(incx);
    if (T == f64) {
        fortran.dtbmv_(&uu, &tt, &dd, &nn, &kk, a.ptr, &lda_arg, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64) {
        fortran.ztbmv_(&uu, &tt, &dd, &nn, &kk, a.ptr, &lda_arg, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

fn runTriPackedMvCase(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, trans: ref.Trans, diag: ref.Diag, incx: isize) !void {
    const n: usize = 4;
    const ap = try allocator.alloc(T, n * (n + 1) / 2);
    defer allocator.free(ap);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const expected = try allocator.alloc(T, x.len);
    defer allocator.free(expected);
    const work = try allocator.alloc(T, n);
    defer allocator.free(work);
    fillTriPackedMatrix(T, rng, uplo, diag, ap, n);
    ref.fillVector(T, rng, x, n, incx);
    @memcpy(expected, x);
    ref.tpmvColMajor(T, uplo, trans, diag, n, ap, expected, incx, work);

    var uu = refUploChar(uplo);
    var tt = refTransChar(trans);
    var dd = refDiagChar(diag);
    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    if (T == f64) {
        fortran.dtpmv_(&uu, &tt, &dd, &nn, ap.ptr, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64) {
        fortran.ztpmv_(&uu, &tt, &dd, &nn, ap.ptr, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

fn runTriBandSvCase(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, trans: ref.Trans, diag: ref.Diag, incx: isize) !void {
    const n: usize = 4;
    const k: usize = 2;
    const lda: usize = 4;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const expected = try allocator.alloc(T, x.len);
    defer allocator.free(expected);
    fillTriBandMatrix(T, rng, uplo, diag, a, n, k, lda);
    ref.fillVector(T, rng, x, n, incx);
    @memcpy(expected, x);
    ref.tbsvColMajor(T, uplo, trans, diag, n, k, a, lda, expected, incx);

    var uu = refUploChar(uplo);
    var tt = refTransChar(trans);
    var dd = refDiagChar(diag);
    var nn: fortran.BlasInt = @intCast(n);
    var kk: fortran.BlasInt = @intCast(k);
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var ix: fortran.BlasInt = @intCast(incx);
    if (T == f64) {
        fortran.dtbsv_(&uu, &tt, &dd, &nn, &kk, a.ptr, &lda_arg, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64) {
        fortran.ztbsv_(&uu, &tt, &dd, &nn, &kk, a.ptr, &lda_arg, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

fn runTriPackedSvCase(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, trans: ref.Trans, diag: ref.Diag, incx: isize) !void {
    const n: usize = 4;
    const ap = try allocator.alloc(T, n * (n + 1) / 2);
    defer allocator.free(ap);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const expected = try allocator.alloc(T, x.len);
    defer allocator.free(expected);
    fillTriPackedMatrix(T, rng, uplo, diag, ap, n);
    ref.fillVector(T, rng, x, n, incx);
    @memcpy(expected, x);
    ref.tpsvColMajor(T, uplo, trans, diag, n, ap, expected, incx);

    var uu = refUploChar(uplo);
    var tt = refTransChar(trans);
    var dd = refDiagChar(diag);
    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    if (T == f64) {
        fortran.dtpsv_(&uu, &tt, &dd, &nn, ap.ptr, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64) {
        fortran.ztpsv_(&uu, &tt, &dd, &nn, ap.ptr, x.ptr, &ix);
        try expectApproxSlice(T, expected, x, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

test "fortran triangular banded packed transpose conjugate reference" {
    const allocator = std.testing.allocator;
    var rng = ref.Rng.init(0xabc0_0303);
    try runTriBandMvCase(f64, allocator, &rng, .upper, .trans, .non_unit, -1);
    try runTriBandMvCase(fortran.ComplexF64, allocator, &rng, .lower, .conj_trans, .unit, 2);
    try runTriPackedMvCase(f64, allocator, &rng, .lower, .no_trans, .unit, -2);
    try runTriPackedMvCase(fortran.ComplexF64, allocator, &rng, .upper, .conj_trans, .non_unit, 1);
    try runTriBandSvCase(f64, allocator, &rng, .lower, .trans, .non_unit, 1);
    try runTriBandSvCase(fortran.ComplexF64, allocator, &rng, .upper, .conj_trans, .non_unit, -2);
    try runTriPackedSvCase(f64, allocator, &rng, .upper, .trans, .non_unit, 2);
    try runTriPackedSvCase(fortran.ComplexF64, allocator, &rng, .lower, .conj_trans, .unit, -1);
}

fn runPackedRankCase(comptime T: type, allocator: std.mem.Allocator, rng: *ref.Rng, uplo: ref.Uplo, incx: isize, incy: isize, comptime herm: bool) !void {
    const n: usize = 4;
    const ap = try allocator.alloc(T, n * (n + 1) / 2);
    defer allocator.free(ap);
    const ap2 = try allocator.alloc(T, n * (n + 1) / 2);
    defer allocator.free(ap2);
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const y = try allocator.alloc(T, ref.vectorStorageLen(n, incy));
    defer allocator.free(y);
    const expected = try allocator.alloc(T, ap.len);
    defer allocator.free(expected);
    const expected2 = try allocator.alloc(T, ap2.len);
    defer allocator.free(expected2);

    fillPackedSymMatrix(T, rng, uplo, ap, n);
    fillPackedSymMatrix(T, rng, uplo, ap2, n);
    ref.fillVector(T, rng, x, n, incx);
    ref.fillVector(T, rng, y, n, incy);
    @memcpy(expected, ap);
    @memcpy(expected2, ap2);

    var uu = refUploChar(uplo);
    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    if (T == f64 and !herm) {
        var alpha: f64 = -0.75;
        ref.sprColMajor(T, uplo, n, alpha, x, incx, expected);
        fortran.dspr_(&uu, &nn, &alpha, x.ptr, &ix, ap.ptr);
        try expectApproxSlice(T, expected, ap, @as(T, 1e-12));

        var alpha2: f64 = 0.5;
        ref.spr2ColMajor(T, uplo, n, alpha2, x, incx, y, incy, expected2);
        fortran.dspr2_(&uu, &nn, &alpha2, x.ptr, &ix, y.ptr, &iy, ap2.ptr);
        try expectApproxSlice(T, expected2, ap2, @as(T, 1e-12));
    } else if (T == fortran.ComplexF64 and herm) {
        var alpha_h: f64 = 0.625;
        ref.hprColMajor(T, uplo, n, alpha_h, x, incx, expected);
        fortran.zhpr_(&uu, &nn, &alpha_h, x.ptr, &ix, ap.ptr);
        try expectApproxSlice(T, expected, ap, @as(f64, 1e-12));

        var alpha_h2 = ref.fromParts(T, 0.5, -0.25);
        ref.hpr2ColMajor(T, uplo, n, alpha_h2, x, incx, y, incy, expected2);
        fortran.zhpr2_(&uu, &nn, &alpha_h2, x.ptr, &ix, y.ptr, &iy, ap2.ptr);
        try expectApproxSlice(T, expected2, ap2, @as(f64, 1e-12));
    } else {
        @compileError("unsupported test scalar");
    }
}

test "fortran packed rank update randomized reference" {
    const allocator = std.testing.allocator;
    var rng = ref.Rng.init(0xabc0_0404);
    try runPackedRankCase(f64, allocator, &rng, .upper, -2, 1, false);
    try runPackedRankCase(f64, allocator, &rng, .lower, 1, -2, false);
    try runPackedRankCase(fortran.ComplexF64, allocator, &rng, .upper, 2, -1, true);
    try runPackedRankCase(fortran.ComplexF64, allocator, &rng, .lower, -1, 2, true);
}

test "fortran packed rank update unit stride contiguous columns" {
    const allocator = std.testing.allocator;
    var rng = ref.Rng.init(0xabc0_1414);
    try runPackedRankCase(f64, allocator, &rng, .upper, 1, 1, false);
    try runPackedRankCase(f64, allocator, &rng, .lower, 1, 1, false);
    try runPackedRankCase(fortran.ComplexF64, allocator, &rng, .upper, 1, 1, true);
    try runPackedRankCase(fortran.ComplexF64, allocator, &rng, .lower, 1, 1, true);
}

fn runDenseSyrCase(comptime T: type, uplo: ref.Uplo, comptime rank2: bool, n: usize, incx: isize, incy: isize) !void {
    const allocator = std.testing.allocator;
    const lda = n + 6;
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const y = try allocator.alloc(T, ref.vectorStorageLen(n, incy));
    defer allocator.free(y);
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const expected = try allocator.alloc(T, a.len);
    defer allocator.free(expected);

    var rng = ref.Rng.init(0x54f2_0311);
    ref.fillVector(T, &rng, x, n, incx);
    ref.fillVector(T, &rng, y, n, incy);
    ref.fillColMajor(T, &rng, a, n, n, lda);
    @memcpy(expected, a);

    const alpha: T = -0.625;
    for (0..n) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        const xj = ref.vectorGet(T, x, n, incx, j);
        const yj = ref.vectorGet(T, y, n, incy, j);
        for (row0..row1) |i| {
            const xi = ref.vectorGet(T, x, n, incx, i);
            const yi = ref.vectorGet(T, y, n, incy, i);
            const update = if (rank2)
                ref.add(T, ref.mul(T, xi, alpha * yj), ref.mul(T, yi, alpha * xj))
            else
                ref.mul(T, xi, alpha * xj);
            const index = ref.colIndex(lda, i, j);
            expected[index] = ref.add(T, expected[index], update);
        }
    }

    var uu = refUploChar(uplo);
    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var alpha_arg = alpha;
    if (T == f32 and rank2) {
        fortran.ssyr2_(&uu, &nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy, a.ptr, &lda_arg);
    } else if (T == f32) {
        fortran.ssyr_(&uu, &nn, &alpha_arg, x.ptr, &ix, a.ptr, &lda_arg);
    } else if (T == f64 and rank2) {
        fortran.dsyr2_(&uu, &nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy, a.ptr, &lda_arg);
    } else if (T == f64) {
        fortran.dsyr_(&uu, &nn, &alpha_arg, x.ptr, &ix, a.ptr, &lda_arg);
    } else {
        @compileError("parallel SYR test supports f32 and f64");
    }

    const tol: T = if (T == f32) 2e-5 else 1e-12;
    try expectApproxSlice(T, expected, a, tol);
}

fn runDenseHerCase(comptime T: type, uplo: ref.Uplo, comptime rank2: bool, n: usize, incx: isize, incy: isize) !void {
    const allocator = std.testing.allocator;
    const lda = n + 6;
    const x = try allocator.alloc(T, ref.vectorStorageLen(n, incx));
    defer allocator.free(x);
    const y = try allocator.alloc(T, ref.vectorStorageLen(n, incy));
    defer allocator.free(y);
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const expected = try allocator.alloc(T, a.len);
    defer allocator.free(expected);

    var rng = ref.Rng.init(0xa12e_2074);
    ref.fillVector(T, &rng, x, n, incx);
    ref.fillVector(T, &rng, y, n, incy);
    ref.fillColMajor(T, &rng, a, n, n, lda);
    @memcpy(expected, a);

    const real_alpha = if (T == fortran.ComplexF32) @as(f32, -0.625) else @as(f64, -0.625);
    const alpha = ref.fromParts(T, 0.5, -0.25);
    for (0..n) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        const xj = ref.vectorGet(T, x, n, incx, j);
        const yj = ref.vectorGet(T, y, n, incy, j);
        const temp1 = if (rank2)
            ref.mul(T, alpha, ref.conj(T, yj))
        else
            ref.mul(T, ref.fromParts(T, real_alpha, 0), ref.conj(T, xj));
        const temp2 = if (rank2) ref.mul(T, ref.conj(T, alpha), ref.conj(T, xj)) else ref.zero(T);
        for (row0..row1) |i| {
            const xi = ref.vectorGet(T, x, n, incx, i);
            const yi = ref.vectorGet(T, y, n, incy, i);
            var update = ref.mul(T, xi, temp1);
            if (rank2) update = ref.add(T, update, ref.mul(T, yi, temp2));
            const index = ref.colIndex(lda, i, j);
            expected[index] = ref.add(T, expected[index], update);
            if (i == j) expected[index].im = 0;
        }
    }

    var uu = refUploChar(uplo);
    var nn: fortran.BlasInt = @intCast(n);
    var ix: fortran.BlasInt = @intCast(incx);
    var iy: fortran.BlasInt = @intCast(incy);
    var lda_arg: fortran.BlasInt = @intCast(lda);
    var real_alpha_arg = real_alpha;
    var alpha_arg = alpha;
    if (T == fortran.ComplexF32 and rank2) {
        fortran.cher2_(&uu, &nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy, a.ptr, &lda_arg);
    } else if (T == fortran.ComplexF32) {
        fortran.cher_(&uu, &nn, &real_alpha_arg, x.ptr, &ix, a.ptr, &lda_arg);
    } else if (T == fortran.ComplexF64 and rank2) {
        fortran.zher2_(&uu, &nn, &alpha_arg, x.ptr, &ix, y.ptr, &iy, a.ptr, &lda_arg);
    } else if (T == fortran.ComplexF64) {
        fortran.zher_(&uu, &nn, &real_alpha_arg, x.ptr, &ix, a.ptr, &lda_arg);
    } else {
        @compileError("parallel HER test supports ComplexF32 and ComplexF64");
    }

    const tol = if (T == fortran.ComplexF32) @as(f32, 3e-5) else @as(f64, 1e-12);
    try expectApproxSlice(T, expected, a, tol);
}

test "parallel dense rank updates match reference for upper and lower storage" {
    fortran.runtime.setMaxThreads(4);
    defer fortran.runtime.setMaxThreads(0);

    inline for (.{ f32, f64 }) |T| {
        inline for (.{ ref.Uplo.upper, ref.Uplo.lower }) |uplo| {
            try runDenseSyrCase(T, uplo, false, 513, 1, 1);
            try runDenseSyrCase(T, uplo, true, 513, 1, 1);
        }
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ ref.Uplo.upper, ref.Uplo.lower }) |uplo| {
            try runDenseHerCase(T, uplo, false, 513, 1, 1);
            try runDenseHerCase(T, uplo, true, 513, 1, 1);
        }
    }
}

test "dense rank updates preserve negative and non-unit stride fallbacks" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ ref.Uplo.upper, ref.Uplo.lower }) |uplo| {
            try runDenseSyrCase(T, uplo, false, 17, -2, 2);
            try runDenseSyrCase(T, uplo, true, 17, -2, 2);
        }
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ ref.Uplo.upper, ref.Uplo.lower }) |uplo| {
            try runDenseHerCase(T, uplo, false, 17, -2, 2);
            try runDenseHerCase(T, uplo, true, 17, -2, 2);
        }
    }
}

// The build injects the existing Core's exact test Execution Module.
const admission_execution = @import("zynum-execution");

const FortranAdmissionObservation = struct {
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
        if (ctx != admission_execution.getContext() or fixtureExecutor(ctx) != admission_execution.get())
            @panic("Fortran ABI used a different default owner");
        if (ctx.snapshot().active_operations != 1) @panic("Fortran ABI did not retain one operation");
        switch (event) {
            .admitted => entries += 1,
            .finishing => {
                if (!std.mem.eql(u8, actual, expected)) @panic("Fortran operation ended before complete output or damaged padding");
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

test "Fortran numerical exports retain one default operation through output and return" {
    const Observe = FortranAdmissionObservation;
    Observe.reset();
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    try std.testing.expect(fortran.runtime.setMaxThreads == admission_execution.policy.setMaxThreads);
    try std.testing.expect(fortran.runtime.maxThreads == admission_execution.policy.maxThreads);
    try std.testing.expectEqual(admission_execution.context.RequestResult.immortal, admission_execution.getContext().requestClose());

    const n: fortran.BlasInt = 2;
    const ld: fortran.BlasInt = 3;
    const inc: fortran.BlasInt = 1;
    const negative_inc: fortran.BlasInt = -1;
    const normal = [_]u8{'n'};
    const one: f64 = 1;
    const zero: f64 = 0;
    const a = [_]f64{ 1, 3, 71, 2, 4, 72 };
    const b = [_]f64{ 5, 7, 73, 6, 8, 74 };
    var c = [_]f64{ 0, 0, 81, 0, 0, 82 };
    // [[1,2],[3,4]] * [[5,6],[7,8]], independent literal result.
    Observe.watch(f64, &c, &.{ 19, 43, 81, 22, 50, 82 });
    fortran.dgemm_(&normal, &normal, &n, &n, &n, &one, &a, &ld, &b, &ld, &zero, &c, &ld);
    try Observe.checkNext();
    const x = [_]f64{ 5, 6, 75 };
    var y = [_]f64{ 0, 0, 83 };
    Observe.watch(f64, &y, &.{ 16, 38, 83 });
    fortran.dgemv_(&normal, &n, &n, &one, &a, &ld, &x, &negative_inc, &zero, &y, &inc);
    try Observe.checkNext();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 71, 2, 4, 72 }, &a);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 73, 6, 8, 74 }, &b);

    const C = fortran.ComplexF64;
    const cx = [_]C{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
    const cy = [_]C{ .{ .re = 2, .im = -1 }, .{ .re = -1, .im = 2 } };
    // A scalar return becomes caller-visible after finishing; check it after
    // the actual call, not by pretending its assignment precedes Operation.end.
    Observe.watch(f64, &.{}, &.{});
    try std.testing.expectEqual(C{ .re = 3, .im = 10 }, fortran.zdotu_(&n, &cx, &inc, &cy, &inc));
    try Observe.checkNext();
    try std.testing.expectEqual(C{ .re = -5, .im = 0 }, fortran.zdotc_(&n, &cx, &inc, &cy, &inc));
    try Observe.checkNext();
    const cx32 = [_]fortran.ComplexF32{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
    const cy32 = [_]fortran.ComplexF32{ .{ .re = 2, .im = -1 }, .{ .re = -1, .im = 2 } };
    try std.testing.expectEqual(fortran.ComplexF32{ .re = 3, .im = 10 }, fortran.cdotu_(&n, &cx32, &inc, &cy32, &inc));
    try Observe.checkNext();
    try std.testing.expectEqual(fortran.ComplexF32{ .re = -5, .im = 0 }, fortran.cdotc_(&n, &cx32, &inc, &cy32, &inc));
    try Observe.checkNext();
    try std.testing.expectEqual(@as(f64, 4), fortran.dcabs1_(&cx[1]));
    try Observe.checkNext();
    try std.testing.expectEqual(@as(f32, 4), fortran.scabs1_(&cx32[1]));
    try Observe.checkNext();
    try std.testing.expectEqual(@as(fortran.BlasInt, 2), fortran.izamax_(&n, &cx, &inc));
    try Observe.checkNext();
    var dot = [_]C{ .{ .re = 0, .im = 0 }, .{ .re = 91, .im = 17 } };
    Observe.watch(C, &dot, &.{ .{ .re = 3, .im = 10 }, .{ .re = 91, .im = 17 } });
    fortran.zdotu_sub_(&n, &cx, &inc, &cy, &inc, &dot[0]);
    try Observe.checkNext();
    Observe.watch(C, &dot, &.{ .{ .re = -5, .im = 0 }, .{ .re = 91, .im = 17 } });
    fortran.zdotc_sub_(&n, &cx, &inc, &cy, &inc, &dot[0]);
    try Observe.checkNext();
}

test "Fortran validation and empty paths still finish exactly one operation" {
    const Observe = FortranAdmissionObservation;
    Observe.reset();
    admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, Observe.call);
    defer admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    const normal = [_]u8{'N'};
    const n: fortran.BlasInt = 2;
    const one: f64 = 1;
    const zero: f64 = 0;
    const a = [_]f64{ 1, 3, 2, 4 };
    var output = [_]f64{ 11, 13, 17, 19, 97 };
    Observe.watch(f64, &output, &.{ 11, 13, 17, 19, 97 });
    const Case = struct { trans: u8, rows: fortran.BlasInt, lda: fortran.BlasInt };
    for ([_]Case{
        .{ .trans = 'X', .rows = -1, .lda = 0 },
        .{ .trans = 'N', .rows = -1, .lda = 0 },
        .{ .trans = 'N', .rows = 2, .lda = 0 },
        .{ .trans = 'N', .rows = 0, .lda = 1 },
    }) |case| {
        const trans = [_]u8{case.trans};
        fortran.dgemm_(&trans, &normal, &case.rows, &n, &n, &one, &a, &case.lda, &a, &n, &zero, &output, &n);
        try Observe.checkNext();
    }
    // Production XERBLA text/parameter number is a separate non-test C check;
    // this root intentionally proves only lifetime plus unchanged output.
    const ScaleCase = struct { n: fortran.BlasInt, inc: fortran.BlasInt, alpha: f64 };
    for ([_]ScaleCase{
        .{ .n = 0, .inc = 1, .alpha = 3 },
        .{ .n = -1, .inc = 1, .alpha = 3 },
        .{ .n = 2, .inc = 0, .alpha = 3 },
        .{ .n = 2, .inc = 1, .alpha = 1 },
    }) |case| {
        fortran.dscal_(&case.n, &case.alpha, &output, &case.inc);
        try Observe.checkNext();
    }
    const empty: fortran.BlasInt = 0;
    const inc: fortran.BlasInt = 1;
    const negative_zero: f32 = @bitCast(@as(u32, 0x80000000));
    const safe_input = [_]f32{7};
    const result = fortran.sdsdot_(&empty, &negative_zero, &safe_input, &inc, &safe_input, &inc);
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(result)));
    try Observe.checkNext();
}

test "Fortran scaling retains one operation through real busy executor fallback" {
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
            if (ctx != admission_execution.getContext() or fixtureExecutor(ctx) != admission_execution.get())
                @panic("legacy fallback used a different default owner");
            if (ctx.snapshot().active_operations != 1 or fixtureExecutor(ctx).io_busy.load(.acquire) != 1)
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
    const count: fortran.BlasInt = n;
    const alpha: f64 = -3;
    const inc: fortran.BlasInt = 1;
    fortran.dscal_(&count, &alpha, values.ptr, &inc);
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

// These fixtures borrow real Context Operations. They neither replace the
// default allocator nor synthesize an Executor result or its busy state.
const ResourceGemmContext = admission_execution.context.Context;
const ResourceGemmAllocations = admission_execution.test_support.Allocations;

fn ResourceGemmProbe(comptime thread_limit: usize) type {
    return struct {
        var context: ?*ResourceGemmContext = null;
        var queries: std.atomic.Value(usize) = .init(0);
        var configurations: std.atomic.Value(usize) = .init(0);
        var entries: std.atomic.Value(usize) = .init(0);
        var finishes: std.atomic.Value(usize) = .init(0);
        var actual: []const u8 = &.{};
        var expected: []const u8 = &.{};
        // The caller changes these only before admission or while the separate
        // blocking batch is stopped. Its finishing callback runs after join release.
        var watch_armed: bool = false;
        var expect_busy: bool = false;
        var allocations: ?*ResourceGemmAllocations = null;
        var expected_live_allocations: ?usize = null;

        fn maxThreads() usize {
            _ = queries.fetchAdd(1, .monotonic);
            return thread_limit;
        }
        fn helpers(limit: usize) usize {
            return @min(limit, thread_limit - 1);
        }
        fn configure(_: ?usize) void {
            _ = configurations.fetchAdd(1, .monotonic);
        }
        const policy: admission_execution.owner.Policy = .{
            .max_threads = maxThreads,
            .helper_count = helpers,
            .configure_thread = configure,
            .worker_stack_size = 2 * 1024 * 1024,
        };
        fn setup(ctx: *ResourceGemmContext, allocation_log: *ResourceGemmAllocations) void {
            context = ctx;
            allocations = allocation_log;
            expected_live_allocations = null;
            queries.store(0, .release);
            configurations.store(0, .release);
            entries.store(0, .release);
            finishes.store(0, .release);
            watch_armed = false;
            expect_busy = false;
            actual = &.{};
            expected = &.{};
            ResourceGemmContext.testing.configure(ctx, std.math.maxInt(usize), observe);
        }
        fn observe(ctx: *ResourceGemmContext, event: admission_execution.context.TestEvent) void {
            if (ctx != context.? or ctx == admission_execution.getContext())
                @panic("borrowed GEMM changed Context owner");
            const state = ctx.snapshot();
            if (state.active_operations == 0 or state.resource_pins != 0)
                @panic("borrowed GEMM lost the enclosing operation or retained a resource pin");
            switch (event) {
                .admitted => _ = entries.fetchAdd(1, .monotonic),
                .finishing => {
                    if (watch_armed) {
                        if (!std.mem.eql(u8, actual, expected))
                            @panic("GEMM operation ended before final stores or damaged padding");
                        if (expect_busy and fixtureExecutor(ctx).io_busy.load(.acquire) != 1)
                            @panic("GEMM fallback outlived the real blocking batch");
                        if (expected_live_allocations) |expected_live| {
                            const allocation_state = allocations.?.stats();
                            if (allocation_state.accepted - allocation_state.freed != expected_live)
                                @panic("GEMM operation ended before its call-local workspace was freed");
                        }
                    }
                    _ = finishes.fetchAdd(1, .monotonic);
                },
            }
        }
        fn disarm() void {
            watch_armed = false;
            expect_busy = false;
            expected_live_allocations = null;
        }
    };
}

const ResourceGemmDefaultGuard = struct {
    const State = struct {
        context: admission_execution.context.Snapshot,
        busy: u8,
        io_init: u8,
        persistent_init: u8,
        workers: u32,
        job_count: u32,
        generations: [admission_execution.owner.max_tasks]u32,
        cache_domain: ?*const admission_execution.owner.WorkerCacheDomain,
        max_threads_override: usize,
    };
    before: State,
    fn state() State {
        const executor = admission_execution.get();
        var generations: [admission_execution.owner.max_tasks]u32 = undefined;
        for (&generations, 0..) |*generation, i|
            generation.* = executor.persistent_worker_generation[i].load(.acquire);
        return .{
            .context = admission_execution.getContext().snapshot(),
            .busy = executor.io_busy.load(.acquire),
            .io_init = executor.io_init_state.load(.acquire),
            .persistent_init = executor.persistent_init_state.load(.acquire),
            .workers = executor.persistent_worker_count.load(.acquire),
            .job_count = executor.persistent_job_count.load(.acquire),
            .generations = generations,
            .cache_domain = executor.persistent_cache_domain,
            .max_threads_override = admission_execution.policy.maxThreadsOverride(),
        };
    }
    fn unexpected(_: *ResourceGemmContext, _: admission_execution.context.TestEvent) void {
        @panic("borrowed real GEMM admitted or ended an operation on the default Context");
    }
    fn init() !ResourceGemmDefaultGuard {
        const before = state();
        try std.testing.expectEqual(@as(usize, 0), before.context.active_operations);
        try std.testing.expectEqual(@as(usize, 0), before.context.resource_pins);
        try std.testing.expectEqual(@as(u8, 0), before.busy);
        ResourceGemmContext.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), unexpected);
        return .{ .before = before };
    }
    fn check(self: ResourceGemmDefaultGuard) !void {
        try std.testing.expectEqualDeep(self.before, state());
    }
    fn deinit(_: ResourceGemmDefaultGuard) void {
        ResourceGemmContext.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    }
};

const ResourceGemmShape = struct {
    m: usize,
    n: usize,
    k: usize,
    ta: u8 = 'N',
    tb: u8 = 'N',
    compact_row: bool = false,
    b_padding: usize = 5,
    alpha: i64 = 2,
    beta: i64 = -3,
};

fn resourceGemmSmall(index: usize, modulus: usize, bias: i64) i64 {
    return @as(i64, @intCast(index % modulus)) - bias;
}

fn runResourceGemmCase(comptime T: type, comptime Probe: type, ctx: *ResourceGemmContext, shape: ResourceGemmShape, closing: bool, busy: bool) !void {
    const ta = [_]u8{shape.ta};
    const tb = [_]u8{shape.tb};
    const m: fortran.BlasInt = @intCast(shape.m);
    const n: fortran.BlasInt = @intCast(shape.n);
    const k: fortran.BlasInt = @intCast(shape.k);
    const a_rows = if (shape.ta == 'N') shape.m else shape.k;
    const a_cols = if (shape.ta == 'N') shape.k else shape.m;
    const b_rows = if (shape.tb == 'N') shape.k else shape.n;
    const b_cols = if (shape.tb == 'N') shape.n else shape.k;
    const lda_u = a_rows + (if (shape.compact_row) @as(usize, 0) else 3);
    const ldb_u = b_rows + shape.b_padding;
    const ldc_u = shape.m + (if (shape.compact_row) @as(usize, 0) else 7);
    const lda: fortran.BlasInt = @intCast(lda_u);
    const ldb: fortran.BlasInt = @intCast(ldb_u);
    const ldc: fortran.BlasInt = @intCast(ldc_u);
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(T, lda_u * a_cols + 9);
    defer buffers.free(a);
    const b = try buffers.alloc(T, ldb_u * b_cols + 9);
    defer buffers.free(b);
    const c = try buffers.alloc(T, ldc_u * shape.n + 9);
    defer buffers.free(c);
    @memset(a, 203);
    @memset(b, 211);
    @memset(c, 137);
    for (0..shape.m) |i| for (0..shape.k) |p| {
        const index = if (shape.ta == 'N') i + p * lda_u else p + i * lda_u;
        a[index] = @floatFromInt(resourceGemmSmall(i, 5, 2) + resourceGemmSmall(p, 7, 3));
    };
    for (0..shape.k) |p| for (0..shape.n) |j| {
        const index = if (shape.tb == 'N') p + j * ldb_u else j + p * ldb_u;
        b[index] = @floatFromInt(resourceGemmSmall(p, 3, 1) + resourceGemmSmall(j, 11, 5));
    };
    for (0..shape.n) |j| for (0..shape.m) |i| {
        c[i + j * ldc_u] = @floatFromInt(2 * ((i + j) % 5) + 1);
    };
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const b_before = try buffers.dupe(T, b);
    defer buffers.free(b_before);
    const expected = try buffers.dupe(T, c);
    defer buffers.free(expected);
    // Independent rank-two formula for (r_i + u_p)(v_p + s_j),
    // including every original C value. All intermediates here and in the
    // numerical kernels are small exactly representable integers in f32/f64.
    // The default alpha=2/beta=-3 and odd C seeds keep final results nonzero.
    // The separate 1024-square alpha=3/beta=2 fixture is also nonzero for
    // these values. Both catch repeated beta scaling without a zero-sign oracle.
    var sum_u: i64 = 0;
    var sum_v: i64 = 0;
    var sum_uv: i64 = 0;
    for (0..shape.k) |p| {
        const u = resourceGemmSmall(p, 7, 3);
        const v = resourceGemmSmall(p, 3, 1);
        sum_u += u;
        sum_v += v;
        sum_uv += u * v;
    }
    for (0..shape.n) |j| for (0..shape.m) |i| {
        const r = resourceGemmSmall(i, 5, 2);
        const s = resourceGemmSmall(j, 11, 5);
        const product = r * sum_v + r * s * @as(i64, @intCast(shape.k)) + sum_uv + s * sum_u;
        const old: i64 = @intCast(2 * ((i + j) % 5) + 1);
        expected[i + j * ldc_u] = @floatFromInt(shape.alpha * product + shape.beta * old);
    };

    const active_before = ctx.snapshot().active_operations;
    const entries_before = Probe.entries.load(.acquire);
    const finishes_before = Probe.finishes.load(.acquire);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    errdefer Probe.disarm();
    Probe.actual = std.mem.sliceAsBytes(c);
    Probe.expected = std.mem.sliceAsBytes(expected);
    Probe.expect_busy = busy;
    if (busy) {
        const allocation_state = Probe.allocations.?.stats();
        // The blocked batch keeps its own Executor storage live. Only the
        // GEMM call-local increment must be gone before this Operation ends.
        Probe.expected_live_allocations = allocation_state.accepted - allocation_state.freed;
    }
    Probe.watch_armed = true;
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    const alpha: T = @floatFromInt(shape.alpha);
    const beta: T = @floatFromInt(shape.beta);
    if (T == f32) {
        fortran.internal_admitted.sgemm_(&operation, &ta, &tb, &m, &n, &k, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldc);
    } else {
        fortran.internal_admitted.dgemm_(&operation, &ta, &tb, &m, &n, &k, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldc);
    }
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(c));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_before), std.mem.sliceAsBytes(b));
    try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
    try std.testing.expectEqual(entries_before + 1, Probe.entries.load(.acquire));
    try std.testing.expectEqual(finishes_before, Probe.finishes.load(.acquire));
    if (closing) try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    operation.end(); // The observer sees the complete, padded output while still admitted.
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
    try std.testing.expectEqual(finishes_before + 1, Probe.finishes.load(.acquire));
}

fn finishResourceGemmContext(comptime Probe: type, ctx: *ResourceGemmContext, allocations: *ResourceGemmAllocations) !void {
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    const state = ctx.snapshot();
    try std.testing.expectEqual(@as(usize, 0), state.active_operations);
    try std.testing.expectEqual(@as(usize, 0), state.resource_pins);
    const stats = allocations.stats();
    try std.testing.expectEqual(@as(usize, 0), stats.rejected);
    try std.testing.expectEqual(stats.accepted, stats.freed);
}

test "borrowed real GEMM keeps two Context policies executors and allocators independent" {
    const Single = ResourceGemmProbe(1);
    const Pair = ResourceGemmProbe(2);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations_a: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var allocations_b: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(allocations_a.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(allocations_b.allocator(), Pair.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    Single.setup(a, &allocations_a);
    Pair.setup(b, &allocations_b);
    defer {
        Single.disarm();
        Pair.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    try std.testing.expect(&reference_a.executor != &reference_b.executor and &reference_a.executor != admission_execution.get());
    try std.testing.expect(reference_a.executor.allocator.ptr == allocations_a.allocator().ptr);
    try std.testing.expect(reference_b.executor.allocator.ptr == allocations_b.allocator().ptr);
    try std.testing.expect(reference_a.executor.policy.max_threads == Single.policy.max_threads);
    try std.testing.expect(reference_b.executor.policy.max_threads == Pair.policy.max_threads);
    inline for (.{ f32, f64 }) |T| {
        for ([_]u8{ 'N', 'T' }) |ta| for ([_]u8{ 'N', 'T' }) |tb| {
            const before_b = allocations_b.stats();
            const queries_b = Pair.queries.load(.acquire);
            try runResourceGemmCase(T, Single, a, .{ .m = 65, .n = 67, .k = 129, .ta = ta, .tb = tb }, false, false);
            try std.testing.expectEqualDeep(before_b, allocations_b.stats());
            try std.testing.expectEqual(queries_b, Pair.queries.load(.acquire));
            const before_a = allocations_a.stats();
            const queries_a = Single.queries.load(.acquire);
            try runResourceGemmCase(T, Pair, b, .{ .m = 65, .n = 67, .k = 129, .ta = ta, .tb = tb }, false, false);
            try std.testing.expectEqualDeep(before_a, allocations_a.stats());
            try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
            try default_guard.check();
        };
        // Above the real planner threshold, and transposed A avoids the NN
        // architecture fast path. Packed-A allocation is architecture-dependent.
        try runResourceGemmCase(T, Single, a, .{ .m = 192, .n = 192, .k = 192, .ta = 'T' }, false, false);
        const before_a = allocations_a.stats();
        const queries_a = Single.queries.load(.acquire);
        try runResourceGemmCase(T, Pair, b, .{ .m = 192, .n = 192, .k = 192, .ta = 'T' }, false, false);
        try std.testing.expectEqualDeep(before_a, allocations_a.stats());
        try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
    }
    try std.testing.expect(Single.queries.load(.acquire) > 0);
    try std.testing.expect(Pair.queries.load(.acquire) > 0);
    try std.testing.expectEqual(@as(usize, 0), Single.configurations.load(.acquire));
    // These shipped architecture families provide non-generic packed-A plans.
    // Other targets still run every numerical case without claiming that branch.
    if (comptime @import("builtin").target.cpu.arch == .aarch64 or @import("builtin").target.cpu.arch == .x86_64) {
        try std.testing.expect(allocations_a.stats().accepted > 0);
        try std.testing.expect(allocations_b.stats().accepted > 0);
        try std.testing.expect(Pair.configurations.load(.acquire) > 0);
        try std.testing.expect(reference_b.executor.io_init_state.load(.acquire) == 2 or reference_b.executor.persistent_init_state.load(.acquire) == 2);
    }
    try finishResourceGemmContext(Pair, b, &allocations_b);
    try finishResourceGemmContext(Single, a, &allocations_a);
    try default_guard.check();
}

test "borrowed real GEMM GEMV shortcuts and short wide copyback retain call resources" {
    const Probe = ResourceGemmProbe(2);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ f32, f64 }) |T| {
        for ([_]ResourceGemmShape{
            .{ .m = 513, .n = 1, .k = 512 },
            .{ .m = 1, .n = 513, .k = 256, .compact_row = true },
            .{ .m = 1024, .n = 3, .k = 128 },
            .{ .m = 1024, .n = 17, .k = 128 },
            .{ .m = 17, .n = 2048, .k = 128 },
            .{ .m = 34, .n = 2048, .k = 256 },
            // NT vector edges use their real strided operand leading dimensions.
            .{ .m = 129, .n = 1, .k = 128, .tb = 'T' },
            .{ .m = 1, .n = 129, .k = 128, .tb = 'T' },
        }) |shape| {
            try runResourceGemmCase(T, Probe, ctx, shape, false, false);
            try default_guard.check();
        }
    }
    // The final tail path is admitted while open, then must finish copyback and
    // all synchronous work after close is requested and before Operation.end.
    try runResourceGemmCase(f64, Probe, ctx, .{ .m = 34, .n = 2048, .k = 256 }, true, false);
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try default_guard.check();
}

const ResourceGemmBlockingBatch = struct {
    context: *ResourceGemmContext,
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
    accepted: bool = false,
    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index == 0) {
            self.entered.store(true, .release);
            while (!self.release.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    fn run(self: *@This()) void {
        defer self.finished.store(true, .release);
        var operation = self.context.begin() catch return;
        defer operation.end();
        self.accepted = fixtureExecutor(operation.borrowedContext()).runLowLatency(.{
            .task_fn = execute,
            .tasks = self,
            .count = 2,
        });
    }
};

test "borrowed real GEMM falls back once while its own Context executor is busy" {
    const Probe = ResourceGemmProbe(2);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    inline for (.{ f32, f64 }) |T| {
        var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        Probe.setup(ctx, &allocations);
        defer {
            Probe.disarm();
            _ = ctx.requestClose();
            _ = fixtureProgressClose(ctx);
        }
        var work: ResourceGemmBlockingBatch = .{ .context = ctx };
        const caller = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&work});
        var joined = false;
        defer {
            Probe.disarm();
            work.release.store(true, .release);
            if (!joined) caller.join();
        }
        while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
        try std.testing.expect(work.entered.load(.acquire));
        try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        const configurations = Probe.configurations.load(.acquire);
        const queries = Probe.queries.load(.acquire);
        const allocations_before = allocations.stats();
        // The same packed-A shape is parallel with the idle two-thread policy
        // above. No executor busy flag or run return is manually substituted.
        try runResourceGemmCase(T, Probe, ctx, .{ .m = 192, .n = 192, .k = 192, .ta = 'T' }, true, true);
        try std.testing.expect(Probe.queries.load(.acquire) > queries);
        try std.testing.expectEqual(configurations, Probe.configurations.load(.acquire));
        if (comptime @import("builtin").target.cpu.arch == .aarch64 or @import("builtin").target.cpu.arch == .x86_64) {
            try std.testing.expect(allocations.stats().accepted > allocations_before.accepted);
            try std.testing.expectEqual(allocations.stats().accepted - allocations_before.accepted, allocations.stats().freed - allocations_before.freed);
        }
        try std.testing.expect(!work.finished.load(.acquire));
        try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
        try std.testing.expectEqual(@as(usize, 2), Probe.entries.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try default_guard.check();
        work.release.store(true, .release);
        caller.join();
        joined = true;
        try std.testing.expect(work.accepted);
        for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
        try std.testing.expectEqual(@as(usize, 2), Probe.finishes.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
        try finishResourceGemmContext(Probe, ctx, &allocations);
        try default_guard.check();
    }
}

test "borrowed real GEMM AMX pack rejection applies beta once before full transpose fallback" {
    const Probe = ResourceGemmProbe(1);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    // NN m=1 forwards to full transposed GEMV with m=n=1024, lda=1027.
    // The exact small-integer oracle checks 3*product+2*C and every padding
    // element; the finishing observer sees it while the Operation is live.
    try runResourceGemmCase(f64, Probe, ctx, .{
        .m = 1,
        .n = 1024,
        .k = 1024,
        .compact_row = true,
        .b_padding = 3,
        .alpha = 3,
        .beta = 2,
    }, true, false);
    const stats = allocations.stats();
    try std.testing.expectEqual(@as(usize, 0), stats.accepted);
    try std.testing.expectEqual(@as(usize, 0), stats.freed);
    try std.testing.expectEqual(stats.attempts, stats.rejected);
    const builtin = @import("builtin");
    const options = @import("zynum-build-options");
    // Require the failure branch only when this root statically selects the
    // ASIMD implementation. Dynamic tiers can choose SME2 instead; they still
    // execute the same numerical and zero-budget lifecycle fixture.
    const static_amx = comptime !options.dynamic_dispatch and options.apple_amx and
        builtin.target.cpu.arch == .aarch64 and builtin.target.os.tag == .macos and
        builtin.target.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and
        !builtin.target.cpu.has(.aarch64, .sve) and
        !(builtin.target.cpu.has(.aarch64, .sme) and builtin.target.cpu.has(.aarch64, .sme2));
    if (comptime static_amx) {
        // dgemvTransN8 requests 1024*8*2 doubles (131072 bytes) exactly once.
        // Allocations proves the rejection, not the requested byte size.
        try std.testing.expectEqual(@as(usize, 1), stats.rejected);
    }
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try default_guard.check();
}

// Source draft: complex calls borrow the very same Context/allocator fixtures as
// the real tests above. This conventional integer complex oracle is independent
// of the implementation's expanded-real/three-product decompositions.
const ComplexResourceShape = struct {
    m: usize,
    n: usize,
    k: usize,
    ta: u8 = 'N',
    tb: u8 = 'N',
    compact_row: bool = false,
    expanded_coefficients: bool = false,
    invalid: bool = false,
    zero_alpha: bool = false,
};

fn complexResourceOperand(comptime T: type, values: []const T, ld: usize, trans: u8, row: usize, col: usize) struct { re: i64, im: i64 } {
    const value = values[if (trans == 'N') row + col * ld else col + row * ld];
    return .{ .re = @intFromFloat(value.re), .im = (if (trans == 'C') @as(i64, -1) else 1) * @as(i64, @intFromFloat(value.im)) };
}

fn runComplexResourceGemm(comptime T: type, comptime Probe: type, ctx: *ResourceGemmContext, shape: ComplexResourceShape, closing: bool, busy: bool) !void {
    const R = if (T == fortran.ComplexF32) f32 else f64;
    const a_rows = if (shape.ta == 'N') shape.m else shape.k;
    const a_cols = if (shape.ta == 'N') shape.k else shape.m;
    const b_rows = if (shape.tb == 'N') shape.k else shape.n;
    const b_cols = if (shape.tb == 'N') shape.n else shape.k;
    const lda_u = a_rows + (if (shape.compact_row) @as(usize, 0) else 3);
    const ldb_u = b_rows + 5;
    const ldc_u = shape.m + (if (shape.compact_row) @as(usize, 0) else 7);
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(T, lda_u * a_cols + 9);
    defer buffers.free(a);
    const b = try buffers.alloc(T, ldb_u * b_cols + 9);
    defer buffers.free(b);
    const c = try buffers.alloc(T, ldc_u * shape.n + 9);
    defer buffers.free(c);
    @memset(a, complexValue(T, 203, -207));
    @memset(b, complexValue(T, 211, -219));
    @memset(c, complexValue(T, 137, -149));
    for (0..a_cols) |j| for (0..a_rows) |i| {
        a[i + j * lda_u] = .{ .re = @floatFromInt(2 + (i + j) % 3), .im = @floatFromInt(3 + (2 * i + j) % 3) };
    };
    for (0..b_cols) |j| for (0..b_rows) |i| {
        b[i + j * ldb_u] = .{ .re = @floatFromInt(4 + (i + 2 * j) % 3), .im = -@as(R, @floatFromInt(1 + (i + j) % 2)) };
    };
    for (0..shape.n) |j| for (0..shape.m) |i| {
        c[i + j * ldc_u] = complexValue(T, 3, 1);
    };
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const b_before = try buffers.dupe(T, b);
    defer buffers.free(b_before);
    const expected = try buffers.dupe(T, c);
    defer buffers.free(expected);
    const ar: i64 = if (shape.zero_alpha) 0 else if (shape.expanded_coefficients) 1 else 2;
    const ai: i64 = if (shape.zero_alpha or shape.expanded_coefficients) 0 else 1;
    const br: i64 = if (shape.expanded_coefficients) 0 else -3;
    const bi: i64 = if (shape.expanded_coefficients) 0 else 2;
    if (!shape.invalid) {
        for (0..shape.n) |j| for (0..shape.m) |i| {
            var sr: i64 = 0;
            var si: i64 = 0;
            for (0..shape.k) |p| {
                const av = complexResourceOperand(T, a_before, lda_u, shape.ta, i, p);
                const bv = complexResourceOperand(T, b_before, ldb_u, shape.tb, p, j);
                sr += av.re * bv.re - av.im * bv.im;
                si += av.re * bv.im + av.im * bv.re;
            }
            expected[i + j * ldc_u] = .{
                .re = @floatFromInt(ar * sr - ai * si + br * 3 - bi),
                .im = @floatFromInt(ar * si + ai * sr + br + bi * 3),
            };
        };
    }
    const alpha: T = .{ .re = @floatFromInt(ar), .im = @floatFromInt(ai) };
    const beta: T = .{ .re = @floatFromInt(br), .im = @floatFromInt(bi) };
    const ta = [_]u8{if (shape.invalid) 'Q' else shape.ta};
    const tb = [_]u8{shape.tb};
    const m: fortran.BlasInt = @intCast(shape.m);
    const n: fortran.BlasInt = @intCast(shape.n);
    const k: fortran.BlasInt = @intCast(shape.k);
    const lda: fortran.BlasInt = @intCast(lda_u);
    const ldb: fortran.BlasInt = @intCast(ldb_u);
    const ldc: fortran.BlasInt = @intCast(ldc_u);
    const active_before = ctx.snapshot().active_operations;
    const entries_before = Probe.entries.load(.acquire);
    const finishes_before = Probe.finishes.load(.acquire);
    const allocation_before = Probe.allocations.?.stats();
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    errdefer Probe.disarm();
    Probe.actual = std.mem.sliceAsBytes(c);
    Probe.expected = std.mem.sliceAsBytes(expected);
    Probe.expect_busy = busy;
    // Serial or already-busy execution cannot create a new persistent batch.
    // In these cases every additional allocation must be freed before end.
    if (busy or Probe.policy.max_threads() == 1)
        Probe.expected_live_allocations = allocation_before.accepted - allocation_before.freed;
    Probe.watch_armed = true;
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    if (T == fortran.ComplexF32) {
        fortran.internal_admitted.cgemm_(&operation, &ta, &tb, &m, &n, &k, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldc);
    } else {
        fortran.internal_admitted.zgemm_(&operation, &ta, &tb, &m, &n, &k, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldc);
    }
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(c));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_before), std.mem.sliceAsBytes(b));
    try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
    try std.testing.expectEqual(entries_before + 1, Probe.entries.load(.acquire));
    try std.testing.expectEqual(finishes_before, Probe.finishes.load(.acquire));
    operation.end();
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
    try std.testing.expectEqual(finishes_before + 1, Probe.finishes.load(.acquire));
}

test "borrowed complex GEMM keeps decomposition allocators and two Context policies independent" {
    const Single = ResourceGemmProbe(1);
    const Triple = ResourceGemmProbe(3);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var aa: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var ab: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(aa.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(ab.allocator(), Triple.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    Single.setup(a, &aa);
    Triple.setup(b, &ab);
    defer {
        Single.disarm();
        Triple.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]u8{ 'N', 'T', 'C' }) |ta| for ([_]u8{ 'N', 'T', 'C' }) |tb| {
            const before_b = ab.stats();
            const queries_b = Triple.queries.load(.acquire);
            try runComplexResourceGemm(T, Single, a, .{ .m = 65, .n = 67, .k = 33, .ta = ta, .tb = tb }, false, false);
            try std.testing.expectEqualDeep(before_b, ab.stats());
            try std.testing.expectEqual(queries_b, Triple.queries.load(.acquire));
            const before_a = aa.stats();
            const queries_a = Single.queries.load(.acquire);
            try runComplexResourceGemm(T, Triple, b, .{ .m = 65, .n = 67, .k = 33, .ta = ta, .tb = tb }, false, false);
            try std.testing.expectEqualDeep(before_a, aa.stats());
            try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
            try guard.check();
        };
        try runComplexResourceGemm(T, Triple, b, .{ .m = 127, .n = 129, .k = 32 }, false, false);
        // Both applicable expanded-real routes are retained, including their
        // original alpha=1/beta=0 restriction; x86 may select 3M for the first.
        try runComplexResourceGemm(T, Single, a, .{ .m = 32, .n = 32, .k = 128, .ta = 'T', .expanded_coefficients = true }, false, false);
        try runComplexResourceGemm(T, Single, a, .{ .m = 64, .n = 64, .k = 64, .expanded_coefficients = true }, false, false);
    }
    try std.testing.expect(aa.stats().accepted > 0 and ab.stats().accepted > 0);
    try std.testing.expectEqual(aa.stats().accepted, aa.stats().freed);
    try finishResourceGemmContext(Triple, b, &ab);
    try finishResourceGemmContext(Single, a, &aa);
    try guard.check();
}

test "borrowed complex GEMM rejects outer workspace and inner packs without default allocation" {
    const Probe = ResourceGemmProbe(1);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]usize{ 0, 1 }) |budget| {
            var allocations: ResourceGemmAllocations = .{ .budget = budget };
            var reference_ctx: FixtureReferenceContext = undefined;
            reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const ctx = reference_ctx.borrowedContext();
            Probe.setup(ctx, &allocations);
            defer {
                Probe.disarm();
                _ = ctx.requestClose();
                _ = fixtureProgressClose(ctx);
            }
            // 3M owns the first allocation. The long-K real products retain
            // their original pack fallback, with no default allocator retry.
            try runComplexResourceGemm(T, Probe, ctx, .{ .m = 13, .n = 19, .k = 4097 }, true, false);
            const state = allocations.stats();
            try std.testing.expectEqual(budget, state.accepted);
            try std.testing.expectEqual(state.accepted, state.freed);
            try std.testing.expect(state.rejected > 0);
            try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
            try guard.check();
        }
    }
}

test "borrowed complex GEMM GEMV edges and invalid empty calls keep the enclosing operation" {
    const Probe = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]ComplexResourceShape{
            .{ .m = 256, .n = 1, .k = 256 },
            .{ .m = 512, .n = 1, .k = 512 },
            .{ .m = 1, .n = 256, .k = 256, .compact_row = true },
            .{ .m = 1, .n = 129, .k = 256 },
            .{ .m = 7, .n = 5, .k = 3, .invalid = true },
            .{ .m = 0, .n = 5, .k = 7 },
            .{ .m = 3, .n = 5, .k = 0 },
            .{ .m = 3, .n = 5, .k = 7, .zero_alpha = true },
        }) |shape| {
            try runComplexResourceGemm(T, Probe, ctx, shape, false, false);
            try guard.check();
        }
    }
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

test "borrowed complex GEMM uses same Context fallback while its executor is genuinely busy" {
    const Probe = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        Probe.setup(ctx, &allocations);
        defer {
            Probe.disarm();
            _ = ctx.requestClose();
            _ = fixtureProgressClose(ctx);
        }
        var work: ResourceGemmBlockingBatch = .{ .context = ctx };
        const caller = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&work});
        var joined = false;
        defer {
            Probe.disarm();
            work.release.store(true, .release);
            if (!joined) caller.join();
        }
        while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
        try std.testing.expect(work.entered.load(.acquire));
        const configurations = Probe.configurations.load(.acquire);
        try runComplexResourceGemm(T, Probe, ctx, .{ .m = 127, .n = 129, .k = 32 }, false, true);
        try runComplexResourceGemm(T, Probe, ctx, .{ .m = 256, .n = 1, .k = 256 }, true, true);
        try std.testing.expectEqual(configurations, Probe.configurations.load(.acquire));
        try std.testing.expect(!work.finished.load(.acquire));
        try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
        try guard.check();
        work.release.store(true, .release);
        caller.join();
        joined = true;
        try std.testing.expect(work.accepted);
        for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
        try finishResourceGemmContext(Probe, ctx, &allocations);
        try guard.check();
    }
}

const IndependentGemvShape = struct {
    m: usize = 7,
    n: usize = 5,
    trans: u8 = 'N',
    incx: fortran.BlasInt = 1,
    incy: fortran.BlasInt = 1,
    invalid: bool = false,
};

fn independentGemvValue(comptime T: type, re: i64, im: i64) T {
    if (T == f32 or T == f64) return @floatFromInt(re);
    return .{ .re = @floatFromInt(re), .im = @floatFromInt(im) };
}

fn independentGemvParts(comptime T: type, element: T) struct { re: i64, im: i64 } {
    if (T == f32 or T == f64) return .{ .re = @intFromFloat(element), .im = 0 };
    return .{ .re = @intFromFloat(element.re), .im = @intFromFloat(element.im) };
}

fn independentGemvIndex(length: usize, stride: fortran.BlasInt, i: usize) usize {
    const step: usize = @intCast(if (stride < 0) -stride else stride);
    return 1 + (if (stride < 0) length - 1 - i else i) * step;
}

fn runIndependentResourceGemv(comptime T: type, comptime Probe: type, ctx: *ResourceGemmContext, shape: IndependentGemvShape, closing: bool, busy: bool) !void {
    const buffers = std.testing.allocator;
    const lda_u = shape.m + 3;
    const lenx = if (shape.trans == 'N') shape.n else shape.m;
    const leny = if (shape.trans == 'N') shape.m else shape.n;
    const sx: usize = @intCast(if (shape.incx < 0) -shape.incx else shape.incx);
    const sy: usize = @intCast(if (shape.incy < 0) -shape.incy else shape.incy);
    const a = try buffers.alloc(T, lda_u * shape.n + 2);
    defer buffers.free(a);
    const x = try buffers.alloc(T, 2 + if (lenx == 0) @as(usize, 0) else 1 + (lenx - 1) * sx);
    defer buffers.free(x);
    const y = try buffers.alloc(T, 2 + if (leny == 0) @as(usize, 0) else 1 + (leny - 1) * sy);
    defer buffers.free(y);
    @memset(a, independentGemvValue(T, 101, 103));
    @memset(x, independentGemvValue(T, 107, 109));
    @memset(y, independentGemvValue(T, 113, 127));
    for (0..shape.n) |j| for (0..shape.m) |i| {
        a[1 + i + j * lda_u] = independentGemvValue(T, @intCast(2 + i % 3 + j % 5), @intCast(1 + (i + j) % 2));
    };
    for (0..lenx) |i| x[independentGemvIndex(lenx, shape.incx, i)] = independentGemvValue(T, @intCast(2 + i % 3), -1);
    for (0..leny) |i| y[independentGemvIndex(leny, shape.incy, i)] = independentGemvValue(T, 3, 1);
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const x_before = try buffers.dupe(T, x);
    defer buffers.free(x_before);
    const expected = try buffers.dupe(T, y);
    defer buffers.free(expected);
    const complex = T != f32 and T != f64;
    const ai: i64 = if (complex) 1 else 0;
    const bi: i64 = if (complex) 2 else 0;
    const old_im: i64 = if (complex) 1 else 0;
    // Independent integer oracle over pre-call inputs, including conjugation.
    // All fixture arithmetic fits exactly in either component precision.
    if (!shape.invalid and shape.m != 0 and shape.n != 0) {
        for (0..leny) |i| {
            var re: i64 = 0;
            var im: i64 = 0;
            for (0..lenx) |k| {
                const row = if (shape.trans == 'N') i else k;
                const col = if (shape.trans == 'N') k else i;
                const av = independentGemvParts(T, a_before[1 + row + col * lda_u]);
                const xv = independentGemvParts(T, x_before[independentGemvIndex(lenx, shape.incx, k)]);
                const imag = if (shape.trans == 'C') -av.im else av.im;
                re += av.re * xv.re - imag * xv.im;
                im += av.re * xv.im + imag * xv.re;
            }
            expected[independentGemvIndex(leny, shape.incy, i)] = independentGemvValue(T, 2 * re - ai * im - 9 - bi * old_im, 2 * im + ai * re - 3 * old_im + bi * 3);
        }
    }
    const alpha = independentGemvValue(T, 2, 1);
    const beta = independentGemvValue(T, -3, 2);
    const trans = [_]u8{if (shape.invalid) 'Q' else shape.trans};
    const m: fortran.BlasInt = @intCast(shape.m);
    const n: fortran.BlasInt = @intCast(shape.n);
    const lda: fortran.BlasInt = @intCast(lda_u);
    const active_before = ctx.snapshot().active_operations;
    const entries_before = Probe.entries.load(.acquire);
    const finishes_before = Probe.finishes.load(.acquire);
    const allocation_before = Probe.allocations.?.stats();
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    errdefer Probe.disarm();
    Probe.actual = std.mem.sliceAsBytes(y);
    Probe.expected = std.mem.sliceAsBytes(expected);
    Probe.expect_busy = busy;
    if (busy or Probe.policy.max_threads() == 1)
        Probe.expected_live_allocations = allocation_before.accepted - allocation_before.freed;
    Probe.watch_armed = true;
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    const symbol = if (T == f32) "sgemv_" else if (T == f64) "dgemv_" else if (T == fortran.ComplexF32) "cgemv_" else "zgemv_";
    @field(fortran.internal_admitted, symbol)(&operation, &trans, &m, &n, &alpha, a.ptr + 1, &lda, x.ptr + 1, &shape.incx, &beta, y.ptr + 1, &shape.incy);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(y));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x_before), std.mem.sliceAsBytes(x));
    try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
    try std.testing.expectEqual(entries_before + 1, Probe.entries.load(.acquire));
    try std.testing.expectEqual(finishes_before, Probe.finishes.load(.acquire));
    operation.end();
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
    try std.testing.expectEqual(finishes_before + 1, Probe.finishes.load(.acquire));
}

test "independent borrowed GEMV retains two Contexts for four scalars transforms strides and empty calls" {
    const Single = ResourceGemmProbe(1);
    const Pair = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var aa: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var ab: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(aa.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(ab.allocator(), Pair.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    Single.setup(a, &aa);
    Pair.setup(b, &ab);
    defer {
        Single.disarm();
        Pair.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]u8{ 'N', 'T', 'C' }) |trans| {
            for ([_]fortran.BlasInt{ 1, -2 }) |stride| {
                const shape: IndependentGemvShape = .{ .trans = trans, .incx = stride, .incy = stride };
                const before_b = ab.stats();
                const queries_b = Pair.queries.load(.acquire);
                try runIndependentResourceGemv(T, Single, a, shape, false, false);
                try std.testing.expectEqualDeep(before_b, ab.stats());
                try std.testing.expectEqual(queries_b, Pair.queries.load(.acquire));
                const before_a = aa.stats();
                const queries_a = Single.queries.load(.acquire);
                try runIndependentResourceGemv(T, Pair, b, shape, false, false);
                try std.testing.expectEqualDeep(before_a, aa.stats());
                try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
            }
        }
        for ([_]IndependentGemvShape{ .{ .m = 0 }, .{ .n = 0 }, .{ .invalid = true }, .{ .m = 513, .n = 512 } }) |shape| {
            try runIndependentResourceGemv(T, Pair, b, shape, false, false);
            try guard.check();
        }
    }
    try std.testing.expect(Pair.queries.load(.acquire) > 0);
    try finishResourceGemmContext(Pair, b, &ab);
    try finishResourceGemmContext(Single, a, &aa);
    try guard.check();
}

test "independent borrowed GEMV rejects actual scratch allocations without default retry or double beta" {
    const Single = ResourceGemmProbe(1);
    const Pair = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    // The two-thread irregular complex panel cannot use the exact tiled shapes.
    // ASIMD selects the per-call workspace; other targets still verify fallback
    // numerics and ownership, without inventing an allocation on their leaf path.
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        var allocations: ResourceGemmAllocations = .{ .budget = 0 };
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(allocations.allocator(), Pair.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        Pair.setup(ctx, &allocations);
        defer {
            Pair.disarm();
            _ = ctx.requestClose();
            _ = fixtureProgressClose(ctx);
        }
        try runIndependentResourceGemv(T, Pair, ctx, .{ .m = 513, .n = 512 }, true, false);
        const stats = allocations.stats();
        try std.testing.expectEqual(@as(usize, 0), stats.accepted);
        try std.testing.expectEqual(@as(usize, 0), stats.freed);
        try std.testing.expectEqual(stats.attempts, stats.rejected);
        if (comptime @import("builtin").target.cpu.arch == .aarch64)
            try std.testing.expect(stats.rejected > 0);
        try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
        try guard.check();
    }
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Single.setup(ctx, &allocations);
    defer {
        Single.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    try runIndependentResourceGemv(f64, Single, ctx, .{ .m = 1024, .n = 1024, .trans = 'T' }, true, false);
    const stats = allocations.stats();
    try std.testing.expectEqual(@as(usize, 0), stats.accepted);
    try std.testing.expectEqual(@as(usize, 0), stats.freed);
    const builtin = @import("builtin");
    const options = @import("zynum-build-options");
    const static_amx = comptime !options.dynamic_dispatch and options.apple_amx and
        builtin.target.cpu.arch == .aarch64 and builtin.target.os.tag == .macos and
        builtin.target.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and
        !builtin.target.cpu.has(.aarch64, .sve) and
        !(builtin.target.cpu.has(.aarch64, .sme) and builtin.target.cpu.has(.aarch64, .sme2));
    if (comptime static_amx) try std.testing.expect(stats.rejected > 0);
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try guard.check();
}

test "independent borrowed GEMV finishes on its closing Context while its executor is genuinely busy" {
    const Probe = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var work: ResourceGemmBlockingBatch = .{ .context = ctx };
    const caller = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&work});
    var joined = false;
    defer {
        Probe.disarm();
        work.release.store(true, .release);
        if (!joined) caller.join();
    }
    while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(work.entered.load(.acquire));
    const configurations = Probe.configurations.load(.acquire);
    const queries = Probe.queries.load(.acquire);
    const before = allocations.stats();
    inline for (.{ f32, f64, fortran.ComplexF32 }) |T|
        try runIndependentResourceGemv(T, Probe, ctx, .{ .m = 513, .n = 512 }, false, true);
    try runIndependentResourceGemv(fortran.ComplexF64, Probe, ctx, .{ .m = 513, .n = 512 }, true, true);
    try std.testing.expect(Probe.queries.load(.acquire) > queries);
    try std.testing.expectEqual(configurations, Probe.configurations.load(.acquire));
    const after = allocations.stats();
    try std.testing.expectEqual(after.accepted - before.accepted, after.freed - before.freed);
    if (comptime @import("builtin").target.cpu.arch == .aarch64)
        try std.testing.expect(after.accepted > before.accepted);
    try std.testing.expect(!work.finished.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    try guard.check();
    work.release.store(true, .release);
    caller.join();
    joined = true;
    try std.testing.expect(work.accepted);
    for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

// Level 1 resource composition: independent integer-valued element oracles,
// not a comparison against another BLAS implementation or a second code path.
fn level1Prefix(comptime T: type) []const u8 {
    return if (T == f32) "s" else if (T == f64) "d" else if (T == fortran.ComplexF32) "c" else "z";
}
fn level1Index(n: usize, stride: fortran.BlasInt, i: usize) usize {
    return (if (stride < 0) n - 1 - i else i) * @as(usize, @intCast(@abs(stride)));
}
const Level1Update = enum { copy, swap, scal, rscal, axpy, axpby, rot, rotm };

fn level1UpdateCase(comptime T: type, comptime kind: Level1Update, comptime Probe: type, ctx: *ResourceGemmContext, n: usize, stride: fortran.BlasInt, closing: bool, busy: bool) !void {
    const complex = T == fortran.ComplexF32 or T == fortran.ComplexF64;
    const span = 1 + (n - 1) * @as(usize, @intCast(@abs(stride)));
    const storage = try std.testing.allocator.alloc(T, 2 * (span + 2));
    defer std.testing.allocator.free(storage);
    @memset(storage, independentGemvValue(T, 97, 53));
    const x = storage[1 .. span + 1];
    const y = storage[span + 3 .. 2 * span + 3];
    for (0..n) |i| {
        const index = level1Index(n, stride, i);
        x[index] = independentGemvValue(T, @as(i64, @intCast(i % 7)) + 1, 2);
        y[index] = independentGemvValue(T, @as(i64, @intCast(i % 5)) + 3, -1);
    }
    const expected = try std.testing.allocator.dupe(T, storage);
    defer std.testing.allocator.free(expected);
    const alpha = independentGemvValue(T, 2, -1);
    const beta = independentGemvValue(T, -1, 1);
    for (0..n) |i| {
        const index = level1Index(n, stride, i);
        const a = independentGemvParts(T, x[index]);
        const b = independentGemvParts(T, y[index]);
        const ar: i64 = 2;
        const ai: i64 = if (complex) -1 else 0;
        const bi: i64 = if (complex) 1 else 0;
        const ex = &expected[1 + index];
        const ey = &expected[span + 3 + index];
        switch (kind) {
            .copy => ey.* = x[index],
            .swap => {
                ex.* = y[index];
                ey.* = x[index];
            },
            .scal => ex.* = independentGemvValue(T, ar * a.re - ai * a.im, ar * a.im + ai * a.re),
            .rscal => ex.* = independentGemvValue(T, 2 * a.re, 2 * a.im),
            .axpy => ey.* = independentGemvValue(T, b.re + ar * a.re - ai * a.im, b.im + ar * a.im + ai * a.re),
            .axpby => ey.* = independentGemvValue(T, ar * a.re - ai * a.im - b.re - bi * b.im, ar * a.im + ai * a.re - b.im + bi * b.re),
            .rot => {
                ex.* = independentGemvValue(T, 2 * a.re - b.re, 2 * a.im - b.im);
                ey.* = independentGemvValue(T, 2 * b.re + a.re, 2 * b.im + a.im);
            },
            .rotm => {
                ex.* = independentGemvValue(T, 2 * a.re + 4 * b.re, 0);
                ey.* = independentGemvValue(T, 3 * a.re + 5 * b.re, 0);
            },
        }
    }
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) {
        Probe.disarm();
        operation.end();
    };
    Probe.actual = std.mem.sliceAsBytes(storage);
    Probe.expected = std.mem.sliceAsBytes(expected);
    Probe.watch_armed = true;
    Probe.expect_busy = busy;
    const before = ctx.snapshot();
    const entries = Probe.entries.load(.acquire);
    const finishes = Probe.finishes.load(.acquire);
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    }
    const ni: fortran.BlasInt = @intCast(n);
    switch (kind) {
        .copy, .swap => @field(fortran.internal_admitted, level1Prefix(T) ++ @tagName(kind) ++ "_")(&operation, &ni, x.ptr, &stride, y.ptr, &stride),
        .scal => @field(fortran.internal_admitted, level1Prefix(T) ++ "scal_")(&operation, &ni, &alpha, x.ptr, &stride),
        .rscal => {
            const R = if (T == fortran.ComplexF32) f32 else f64;
            const real_alpha: R = 2;
            @field(fortran.internal_admitted, if (T == fortran.ComplexF32) "csscal_" else "zdscal_")(&operation, &ni, &real_alpha, x.ptr, &stride);
        },
        .axpy => @field(fortran.internal_admitted, level1Prefix(T) ++ "axpy_")(&operation, &ni, &alpha, x.ptr, &stride, y.ptr, &stride),
        .axpby => @field(fortran.internal_admitted, level1Prefix(T) ++ "axpby_")(&operation, &ni, &alpha, x.ptr, &stride, &beta, y.ptr, &stride),
        .rot => {
            const R = if (T == f32 or T == fortran.ComplexF32) f32 else f64;
            const c: R = 2;
            const s: R = -1;
            const name = comptime if (T == fortran.ComplexF32) "csrot_" else if (T == fortran.ComplexF64) "zdrot_" else level1Prefix(T) ++ "rot_";
            @field(fortran.internal_admitted, name)(&operation, &ni, x.ptr, &stride, y.ptr, &stride, &c, &s);
        },
        .rotm => {
            const param = [_]T{ -1, 2, 3, 4, 5 };
            @field(fortran.internal_admitted, level1Prefix(T) ++ "rotm_")(&operation, &ni, x.ptr, &stride, y.ptr, &stride, &param);
        },
    }
    try std.testing.expectEqual(before.active_operations, ctx.snapshot().active_operations);
    try std.testing.expectEqual(before.resource_pins, ctx.snapshot().resource_pins);
    try std.testing.expectEqual(entries, Probe.entries.load(.acquire));
    try std.testing.expectEqual(finishes, Probe.finishes.load(.acquire));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(storage));
    operation.end();
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(finishes + 1, Probe.finishes.load(.acquire));
}

fn level1ReductionCase(comptime T: type, operation: *const admission_execution.context.Operation, stride: fortran.BlasInt) !void {
    const complex = T == fortran.ComplexF32 or T == fortran.ComplexF64;
    const R = if (T == f32 or T == fortran.ComplexF32) f32 else f64;
    const ni: fortran.BlasInt = 7;
    var x: [24]T = @splat(independentGemvValue(T, 71, 31));
    var y: [24]T = @splat(independentGemvValue(T, 73, 37));
    var dot_re: i64 = 0;
    var dot_im: i64 = 0;
    var conjugated_re: i64 = 0;
    var conjugated_im: i64 = 0;
    var sum_abs: i64 = 0;
    var sum_squares: i64 = 0;
    for (0..7) |i| {
        const index = 1 + level1Index(7, stride, i);
        const a: i64 = @as(i64, @intCast(i)) + 1;
        const b: i64 = 4 - @as(i64, @intCast(i));
        const ai: i64 = if (complex) -2 else 0;
        const bi: i64 = if (complex) 1 else 0;
        x[index] = independentGemvValue(T, a, ai);
        y[index] = independentGemvValue(T, b, bi);
        dot_re += a * b - ai * bi;
        dot_im += a * bi + ai * b;
        conjugated_re += a * b + ai * bi;
        conjugated_im += a * bi - ai * b;
        sum_abs += a + @as(i64, @intCast(@abs(ai)));
        sum_squares += a * a + ai * ai;
    }
    const xb = x;
    const yb = y;
    if (comptime complex) {
        var result = independentGemvValue(T, 999, 999);
        @field(fortran.internal_admitted, level1Prefix(T) ++ "dotu_sub_")(operation, &ni, x[1..].ptr, &stride, y[1..].ptr, &stride, &result);
        try std.testing.expectEqualDeep(independentGemvValue(T, dot_re, dot_im), result);
        @field(fortran.internal_admitted, level1Prefix(T) ++ "dotc_sub_")(operation, &ni, x[1..].ptr, &stride, y[1..].ptr, &stride, &result);
        try std.testing.expectEqualDeep(independentGemvValue(T, conjugated_re, conjugated_im), result);
    } else {
        const result = @field(fortran.internal_admitted, level1Prefix(T) ++ "dot_")(operation, &ni, x[1..].ptr, &stride, y[1..].ptr, &stride);
        try std.testing.expectEqual(@as(T, @floatFromInt(dot_re)), result);
    }
    const norm_name = comptime if (T == fortran.ComplexF32) "scnrm2_" else if (T == fortran.ComplexF64) "dznrm2_" else level1Prefix(T) ++ "nrm2_";
    const actual_norm = @field(fortran.internal_admitted, norm_name)(operation, &ni, x[1..].ptr, &stride);
    // ASUM/NRM2 accept negative increments here; only IAMAX rejects inc<=0.
    // Preserve these distinct existing contracts.
    const expected_norm: R = @floatCast(@sqrt(@as(f64, @floatFromInt(sum_squares))));
    try std.testing.expectApproxEqAbs(expected_norm, actual_norm, @as(R, 0.00001));
    const asum_name = comptime if (T == fortran.ComplexF32) "scasum_" else if (T == fortran.ComplexF64) "dzasum_" else level1Prefix(T) ++ "asum_";
    try std.testing.expectEqual(@as(R, @floatFromInt(sum_abs)), @field(fortran.internal_admitted, asum_name)(operation, &ni, x[1..].ptr, &stride));
    try std.testing.expectEqual(@as(fortran.BlasInt, if (stride > 0) 7 else 0), @field(fortran.internal_admitted, "i" ++ level1Prefix(T) ++ "amax_")(operation, &ni, x[1..].ptr, &stride));
    if (T == f32) {
        const offset: f32 = 3;
        try std.testing.expectEqual(@as(f64, @floatFromInt(dot_re)), fortran.internal_admitted.dsdot_(operation, &ni, x[1..].ptr, &stride, y[1..].ptr, &stride));
        try std.testing.expectEqual(@as(f32, @floatFromInt(dot_re + 3)), fortran.internal_admitted.sdsdot_(operation, &ni, &offset, x[1..].ptr, &stride, y[1..].ptr, &stride));
    }
    try std.testing.expectEqualDeep(xb, x);
    try std.testing.expectEqualDeep(yb, y);
}

test "borrowed Level 1 covers four scalars updates reductions strides and complete storage" {
    const Probe = ResourceGemmProbe(1);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]fortran.BlasInt{ 1, 2, -2, 3 }) |stride| {
            inline for (.{ Level1Update.copy, .swap, .scal, .axpy, .axpby, .rot }) |kind|
                try level1UpdateCase(T, kind, Probe, ctx, 7, stride, false, false);
            if (comptime T == f32 or T == f64) {
                try level1UpdateCase(T, .rotm, Probe, ctx, 7, stride, false, false);
            } else {
                try level1UpdateCase(T, .rscal, Probe, ctx, 7, stride, false, false);
            }
            var operation = try ctx.begin();
            defer operation.end();
            try level1ReductionCase(T, &operation, stride);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

test "borrowed Level 1 large updates keep two Context policies and allocator lifetimes separate" {
    const Single = ResourceGemmProbe(1);
    const Parallel = ResourceGemmProbe(4);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var one_alloc: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var many_alloc: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_one: FixtureReferenceContext = undefined;
    reference_one.initInPlace(one_alloc.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const one = reference_one.borrowedContext();
    var reference_many: FixtureReferenceContext = undefined;
    reference_many.initInPlace(many_alloc.allocator(), Parallel.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const many = reference_many.borrowedContext();
    Single.setup(one, &one_alloc);
    Parallel.setup(many, &many_alloc);
    defer {
        Single.disarm();
        Parallel.disarm();
        _ = many.requestClose();
        _ = fixtureProgressClose(many);
        _ = one.requestClose();
        _ = fixtureProgressClose(one);
    }
    try level1UpdateCase(f64, .scal, Single, one, 2 * 1024 * 1024, 1, false, false);
    try std.testing.expectEqual(@as(usize, 0), one_alloc.stats().attempts);
    try level1UpdateCase(f64, .scal, Parallel, many, 2 * 1024 * 1024, 1, false, false);
    try level1UpdateCase(f64, .axpby, Parallel, many, 512 * 1024, 1, false, false);
    try level1UpdateCase(fortran.ComplexF64, .scal, Parallel, many, 512 * 1024, 2, false, false);
    try level1UpdateCase(fortran.ComplexF32, .swap, Parallel, many, 512 * 1024, 2, false, false);
    try std.testing.expect(Single.queries.load(.acquire) > 0);
    try std.testing.expect(Parallel.queries.load(.acquire) > 0);
    try std.testing.expect(Parallel.configurations.load(.acquire) > 0);
    try std.testing.expect(many_alloc.stats().accepted > 0);
    try std.testing.expectEqual(@as(usize, 0), one_alloc.stats().attempts);
    // Threaded storage may remain live after every operation. It belongs to
    // this Executor and is checked for complete release only after its drain.
    try finishResourceGemmContext(Parallel, many, &many_alloc);
    try finishResourceGemmContext(Single, one, &one_alloc);
    try guard.check();
}

test "borrowed Level 1 allocator refusal and real busy closing fallback finish exactly once" {
    const Probe = ResourceGemmProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var refused: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_denied: FixtureReferenceContext = undefined;
    reference_denied.initInPlace(refused.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const denied = reference_denied.borrowedContext();
    Probe.setup(denied, &refused);
    defer {
        Probe.disarm();
        _ = denied.requestClose();
        _ = fixtureProgressClose(denied);
    }
    try level1UpdateCase(f64, .scal, Probe, denied, 2 * 1024 * 1024, 1, false, false);
    try std.testing.expect(refused.stats().rejected > 0);
    try std.testing.expectEqual(@as(usize, 0), refused.stats().accepted);
    _ = denied.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(denied));
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var work: ResourceGemmBlockingBatch = .{ .context = ctx };
    const caller = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&work});
    var joined = false;
    defer {
        Probe.disarm();
        work.release.store(true, .release);
        if (!joined) caller.join();
    }
    while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(work.entered.load(.acquire));
    const allocations_before = allocations.stats();
    try level1UpdateCase(f64, .axpy, Probe, ctx, 512 * 1024, 1, true, true);
    try std.testing.expectEqualDeep(allocations_before, allocations.stats());
    try std.testing.expect(!work.finished.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
    try std.testing.expect(reference_ctx.executor.persistent_task_fn == ResourceGemmBlockingBatch.execute);
    work.release.store(true, .release);
    caller.join();
    joined = true;
    try std.testing.expect(work.accepted);
    for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

test "Fortran complex return DOT and Level 1 empty identities preserve outer admission" {
    const Observe = struct {
        var entries: usize = 0;
        var finishes: usize = 0;
        fn observe(ctx: *ResourceGemmContext, event: admission_execution.context.TestEvent) void {
            if (ctx != admission_execution.getContext() or ctx.snapshot().active_operations != 1) @panic("Level 1 outer admission changed");
            switch (event) {
                .admitted => entries += 1,
                .finishing => finishes += 1,
            }
        }
    };
    Observe.entries = 0;
    Observe.finishes = 0;
    ResourceGemmContext.testing.configure(admission_execution.getContext(), 1, Observe.observe);
    defer ResourceGemmContext.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
    const n: fortran.BlasInt = 2;
    const inc: fortran.BlasInt = 1;
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        const x = [_]T{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
        const y = [_]T{ .{ .re = -2, .im = 1 }, .{ .re = 4, .im = 2 } };
        const unconjugated = @field(fortran, level1Prefix(T) ++ "dotu_")(&n, &x, &inc, &y, &inc);
        const conjugated = @field(fortran, level1Prefix(T) ++ "dotc_")(&n, &x, &inc, &y, &inc);
        try std.testing.expectEqualDeep(T{ .re = 10, .im = -1 }, unconjugated);
        try std.testing.expectEqualDeep(T{ .re = 10, .im = 15 }, conjugated);
    }
    var x = [_]f64{ 5, 7, 11 };
    const before = x;
    const empty: fortran.BlasInt = 0;
    const alpha: f64 = 1;
    fortran.dscal_(&n, &alpha, &x, &inc);
    fortran.dswap_(&empty, &x, &inc, &x, &inc);
    try std.testing.expectEqualDeep(before, x);
    try std.testing.expectEqual(@as(usize, 6), Observe.entries);
    try std.testing.expectEqual(Observe.entries, Observe.finishes);
}

// Resource-family coverage uses the existing Context/Policy/Allocations fixtures.
// Its oracle constructs the mathematical matrix, never decodes the stored input
// through production accessors. Small integer values keep every f32/f64 result
// exact, including the independently accumulated complex products.
const SymmetricMvResourceFixture = struct {
    const Storage = enum { dense, band, packed_storage };
    const Wide = struct { re: f64, im: f64 = 0 };
    const Shape = struct {
        n: usize = 7,
        k: usize = 2,
        upper: bool = true,
        incx: fortran.BlasInt = 1,
        incy: fortran.BlasInt = 1,
        alpha_zero: bool = false,
        beta_zero: bool = false,
    };

    fn isComplex(comptime T: type) bool {
        return T == fortran.ComplexF32 or T == fortran.ComplexF64;
    }
    fn narrow(comptime T: type, value: Wide) T {
        if (comptime isComplex(T)) return .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
        return @floatCast(value.re);
    }
    fn poison(comptime T: type) T {
        if (comptime isComplex(T)) return .{ .re = std.math.nan(@TypeOf(@as(T, undefined).re)), .im = std.math.nan(@TypeOf(@as(T, undefined).im)) };
        return std.math.nan(T);
    }
    fn add(a: Wide, b: Wide) Wide {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }
    fn mul(a: Wide, b: Wide) Wide {
        return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
    }
    fn matrix(comptime T: type, comptime storage: Storage, shape: Shape, row: usize, col: usize) Wide {
        const lo = @min(row, col);
        const hi = @max(row, col);
        if (storage == .band and hi - lo > shape.k) return .{ .re = 0 };
        if (row == col) return .{ .re = @floatFromInt(5 + row % 3) };
        const real: f64 = @floatFromInt(@as(i64, @intCast((lo * 3 + hi * 5) % 7)) - 3);
        const imaginary: f64 = if (isComplex(T)) @floatFromInt(@as(i64, @intCast((lo * 7 + hi * 2) % 5)) - 2) else 0;
        return .{ .re = real, .im = if (row < col) imaginary else -imaginary };
    }
    fn input(comptime T: type, i: usize) Wide {
        return .{ .re = @floatFromInt(@as(i64, @intCast(i % 5)) - 2), .im = if (isComplex(T)) @floatFromInt(@as(i64, @intCast(i % 3)) - 1) else 0 };
    }
    fn originalOutput(comptime T: type, i: usize) Wide {
        return .{ .re = @floatFromInt(11 + i % 7), .im = if (isComplex(T)) @floatFromInt(13 - i % 5) else 0 };
    }
    fn step(inc: fortran.BlasInt) usize {
        return @intCast(if (inc < 0) -inc else inc);
    }
    fn position(n: usize, inc: fortran.BlasInt, i: usize) usize {
        return 1 + (if (inc < 0) n - 1 - i else i) * step(inc);
    }

    // Observe the first real allocator request made by an armed call. All
    // selected parallel dense/packed paths allocate their scratch before any
    // Executor initialization; recording this request separately prevents worker
    // storage from being misreported as scratch. There is no allocator swap.
    const ScratchLog = struct {
        backing: *ResourceGemmAllocations,
        context: ?*ResourceGemmContext = null,
        lock: std.atomic.Value(bool) = .init(false),
        watching: bool = false,
        first_seen: bool = false,
        first_bytes: usize = 0,
        first_alignment: usize = 0,
        first_ptr: ?[*]u8 = null,
        first_accepted: bool = false,
        first_freed: bool = false,
        expected_active: usize = 1,
        free_barrier: ?*FreeBarrier = null,
        const Snapshot = struct { seen: bool, bytes: usize, alignment: usize, accepted: bool, freed: bool };

        fn acquire(self: *ScratchLog) void {
            while (self.lock.cmpxchgStrong(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        fn release(self: *ScratchLog) void {
            self.lock.store(false, .release);
        }
        fn arm(self: *ScratchLog) void {
            self.armWithActive(1);
        }
        fn armWithActive(self: *ScratchLog, expected_active: usize) void {
            self.acquire();
            defer self.release();
            std.debug.assert(self.first_ptr == null);
            self.watching = true;
            self.expected_active = expected_active;
            self.first_seen = false;
            self.first_bytes = 0;
            self.first_alignment = 0;
            self.first_accepted = false;
            self.first_freed = false;
        }
        fn snapshot(self: *ScratchLog) Snapshot {
            self.acquire();
            defer self.release();
            return .{ .seen = self.first_seen, .bytes = self.first_bytes, .alignment = self.first_alignment, .accepted = self.first_accepted, .freed = self.first_freed };
        }
        fn disarm(self: *ScratchLog) void {
            self.acquire();
            defer self.release();
            std.debug.assert(self.first_ptr == null);
            self.watching = false;
        }
        fn allocator(self: *ScratchLog) std.mem.Allocator {
            return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
        }
        fn cast(raw: *anyopaque) *ScratchLog {
            return @ptrCast(@alignCast(raw));
        }
        fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
            const self = cast(raw);
            self.acquire();
            defer self.release();
            const record = self.watching and !self.first_seen;
            if (record) {
                if (self.context.?.snapshot().active_operations != self.expected_active) @panic("scratch allocation escaped its borrowed operation");
                self.first_seen = true;
                self.first_bytes = len;
                self.first_alignment = alignment.toByteUnits();
            }
            const ptr = self.backing.allocator().rawAlloc(len, alignment, ret);
            if (record) {
                self.first_ptr = ptr;
                self.first_accepted = ptr != null;
            }
            return ptr;
        }
        fn resize(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ret: usize) bool {
            const self = cast(raw);
            self.acquire();
            defer self.release();
            if (self.first_ptr == memory.ptr) @panic("fixed call scratch unexpectedly resized");
            return self.backing.allocator().rawResize(memory, alignment, len, ret);
        }
        fn remap(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ret: usize) ?[*]u8 {
            const self = cast(raw);
            self.acquire();
            defer self.release();
            if (self.first_ptr == memory.ptr) @panic("fixed call scratch unexpectedly remapped");
            return self.backing.allocator().rawRemap(memory, alignment, len, ret);
        }
        fn free(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
            const self = cast(raw);
            self.acquire();
            defer self.release();
            const record = self.first_ptr == memory.ptr;
            if (record) {
                if (self.context.?.snapshot().active_operations != self.expected_active) @panic("scratch free escaped its borrowed operation");
                std.debug.assert(memory.len == self.first_bytes and alignment.toByteUnits() == self.first_alignment);
            }
            if (record) if (self.free_barrier) |barrier| {
                // Publish only after dropping the recorder lock. The reader
                // may take both this lock and the Context gate while paused.
                self.release();
                barrier.entered.store(true, .release);
                while (!barrier.allow_free.load(.acquire)) std.Thread.yield() catch {};
                self.backing.allocator().rawFree(memory, alignment, ret);
                self.acquire();
                self.first_ptr = null;
                self.first_freed = true;
                barrier.completed.store(true, .release);
                return;
            };
            self.backing.allocator().rawFree(memory, alignment, ret);
            if (record) {
                self.first_ptr = null;
                self.first_freed = true; // Only after the real backing free returned.
            }
        }
        fn checkScratch(self: *ScratchLog, comptime T: type, elements: usize, refused: bool) !void {
            const state = self.snapshot();
            try std.testing.expect(state.seen);
            try std.testing.expectEqual(elements * @sizeOf(T), state.bytes);
            try std.testing.expectEqual(@alignOf(T), state.alignment);
            try std.testing.expectEqual(!refused, state.accepted);
            try std.testing.expectEqual(!refused, state.freed);
        }
    };

    const FreeBarrier = struct {
        entered: std.atomic.Value(bool) = .init(false),
        allow_free: std.atomic.Value(bool) = .init(false),
        completed: std.atomic.Value(bool) = .init(false),
        // Written before the allocator publishes entered; read only while its
        // freeing thread is paused. The owner joins before destroying anything.
        observation: ?*const anyopaque = null,
        inspect_fn: ?*const fn (*const anyopaque) anyerror!void = null,
    };

    fn Observation(comptime T: type) type {
        return struct {
            ctx: *ResourceGemmContext,
            shape: Shape,
            y_span: usize,
            a: []const T,
            a_before: []const T,
            x: []const T,
            x_before: []const T,
            y: []const T,
            expected: []const T,
            scratch: ?*ScratchLog,
            scratch_elements: usize,
            refused: bool,
            expected_active: usize,
            busy: bool,

            fn checkOutput(self: *const @This()) !void {
                // One unchanged numerical/padding oracle serves return,
                // finishing, and the synchronized pre-backing-free observation.
                for (self.y, 0..) |actual, index| {
                    if (index >= 1 and index <= self.y_span and (index - 1) % step(self.shape.incy) == 0) {
                        if (comptime isComplex(T)) {
                            try std.testing.expectEqual(self.expected[index].re, actual.re);
                            try std.testing.expectEqual(self.expected[index].im, actual.im);
                        } else try std.testing.expectEqual(self.expected[index], actual);
                    } else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&self.expected[index]), std.mem.asBytes(&self.y[index]));
                }
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.a_before), std.mem.sliceAsBytes(self.a));
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.x_before), std.mem.sliceAsBytes(self.x));
            }
            fn inspect(raw: *const anyopaque) !void {
                const self: *const @This() = @ptrCast(@alignCast(raw));
                try self.checkOutput();
                try std.testing.expectEqual(self.expected_active, self.ctx.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), self.ctx.snapshot().resource_pins);
            }
            fn finish(raw: *const anyopaque) !void {
                const self: *const @This() = @ptrCast(@alignCast(raw));
                try inspect(raw);
                if (self.busy) try std.testing.expectEqual(@as(u8, 1), fixtureExecutor(self.ctx).io_busy.load(.acquire));
                if (self.scratch) |log| {
                    if (self.scratch_elements != 0) try log.checkScratch(T, self.scratch_elements, self.refused) else try std.testing.expect(!log.snapshot().seen);
                    if (log.free_barrier) |barrier| try std.testing.expect(barrier.completed.load(.acquire));
                }
            }
        };
    }

    fn run(comptime T: type, comptime storage: Storage, comptime Probe: type, ctx: *ResourceGemmContext, shape: Shape, scratch: ?*ScratchLog, scratch_elements: usize, refused: bool) !void {
        return runLifecycle(T, storage, Probe, ctx, shape, scratch, scratch_elements, refused, false, false, false);
    }

    fn runLifecycle(comptime T: type, comptime storage: Storage, comptime Probe: type, ctx: *ResourceGemmContext, shape: Shape, scratch: ?*ScratchLog, scratch_elements: usize, refused: bool, comptime observe: bool, closing: bool, busy: bool) !void {
        const n: fortran.BlasInt = @intCast(shape.n);
        const k: fortran.BlasInt = @intCast(shape.k);
        const uplo = [_]u8{if (shape.upper) 'U' else 'L'};
        const leading = (if (storage == .band) shape.k + 1 else shape.n) + 3;
        const lda: fortran.BlasInt = @intCast(leading);
        const matrix_len = if (storage == .packed_storage) shape.n * (shape.n + 1) / 2 else leading * shape.n;
        const x_span = 1 + (shape.n - 1) * step(shape.incx);
        const y_span = 1 + (shape.n - 1) * step(shape.incy);
        const buffers = std.testing.allocator;
        const a = try buffers.alloc(T, matrix_len + 9);
        defer buffers.free(a);
        const x = try buffers.alloc(T, x_span + 9);
        defer buffers.free(x);
        const y = try buffers.alloc(T, y_span + 9);
        defer buffers.free(y);
        @memset(a, poison(T));
        @memset(x, narrow(T, .{ .re = 197, .im = -199 }));
        @memset(y, narrow(T, .{ .re = 211, .im = -223 }));
        var packed_index: usize = 0;
        for (0..shape.n) |col| for (0..shape.n) |row| {
            if (if (shape.upper) row > col else row < col) continue;
            if (storage == .band and @max(row, col) - @min(row, col) > shape.k) continue;
            const index = switch (storage) {
                .dense => row + col * leading,
                .band => (if (shape.upper) shape.k + row - col else row - col) + col * leading,
                .packed_storage => blk: {
                    const offset = packed_index;
                    packed_index += 1;
                    break :blk offset;
                },
            };
            var value = narrow(T, matrix(T, storage, shape, row, col));
            if (comptime isComplex(T)) {
                // BLAS Hermitian diagonal is real regardless of its stored
                // imaginary bits. NaN makes an accidental numerical read visible.
                if (row == col) value.im = std.math.nan(@TypeOf(value.im));
            }
            a[1 + index] = if (shape.alpha_zero) poison(T) else value;
        };
        for (0..shape.n) |i| {
            x[position(shape.n, shape.incx, i)] = if (shape.alpha_zero) poison(T) else narrow(T, input(T, i));
            y[position(shape.n, shape.incy, i)] = if (shape.beta_zero) poison(T) else narrow(T, originalOutput(T, i));
        }
        const a_before = try buffers.dupe(T, a);
        defer buffers.free(a_before);
        const x_before = try buffers.dupe(T, x);
        defer buffers.free(x_before);
        const expected = try buffers.dupe(T, y);
        defer buffers.free(expected);
        const alpha_w: Wide = if (shape.alpha_zero) .{ .re = 0 } else .{ .re = 2, .im = if (isComplex(T)) 1 else 0 };
        const beta_w: Wide = if (shape.beta_zero) .{ .re = 0 } else .{ .re = -3, .im = if (isComplex(T)) 2 else 0 };
        for (0..shape.n) |row| {
            var sum: Wide = .{ .re = 0 };
            if (!shape.alpha_zero) for (0..shape.n) |col| {
                sum = add(sum, mul(matrix(T, storage, shape, row, col), input(T, col)));
            };
            const prior = if (shape.beta_zero) Wide{ .re = 0 } else mul(beta_w, originalOutput(T, row));
            expected[position(shape.n, shape.incy, row)] = narrow(T, add(mul(alpha_w, sum), prior));
        }
        const alpha = narrow(T, alpha_w);
        const beta = narrow(T, beta_w);
        const active_before = ctx.snapshot().active_operations;
        if (comptime observe) try std.testing.expectEqual(@as(usize, if (busy) 1 else 0), active_before);
        const entries_before = Probe.entries.load(.acquire);
        const finishes_before = Probe.finishes.load(.acquire);
        var operation = try ctx.begin();
        var ended = false;
        defer if (!ended) operation.end();
        Probe.disarm();
        if (scratch) |log| {
            if (comptime observe) log.armWithActive(active_before + 1) else log.arm();
        }
        defer if (scratch) |log| log.disarm();
        const observation: Observation(T) = .{
            .ctx = ctx,
            .shape = shape,
            .y_span = y_span,
            .a = a,
            .a_before = a_before,
            .x = x,
            .x_before = x_before,
            .y = y,
            .expected = expected,
            .scratch = scratch,
            .scratch_elements = scratch_elements,
            .refused = refused,
            .expected_active = active_before + 1,
            .busy = busy,
        };
        if (comptime observe) Probe.watch(&observation, Observation(T).finish);
        defer if (comptime observe) Probe.unwatch();
        const free_barrier = if (scratch) |log| log.free_barrier else null;
        if (free_barrier) |barrier| {
            barrier.observation = &observation;
            barrier.inspect_fn = Observation(T).inspect;
        }
        defer if (free_barrier) |barrier| {
            barrier.observation = null;
            barrier.inspect_fn = null;
        };
        if (closing) {
            try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
            try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
            try std.testing.expectError(error.Closing, ctx.begin());
        }
        const symbol = switch (storage) {
            .dense => if (T == f32) "ssymv_" else if (T == f64) "dsymv_" else if (T == fortran.ComplexF32) "chemv_" else "zhemv_",
            .band => if (T == f32) "ssbmv_" else if (T == f64) "dsbmv_" else if (T == fortran.ComplexF32) "chbmv_" else "zhbmv_",
            .packed_storage => if (T == f32) "sspmv_" else if (T == f64) "dspmv_" else if (T == fortran.ComplexF32) "chpmv_" else "zhpmv_",
        };
        switch (storage) {
            .dense => @field(fortran.internal_admitted, symbol)(&operation, &uplo, &n, &alpha, a.ptr + 1, &lda, x.ptr + 1, &shape.incx, &beta, y.ptr + 1, &shape.incy),
            .band => @field(fortran.internal_admitted, symbol)(&operation, &uplo, &n, &k, &alpha, a.ptr + 1, &lda, x.ptr + 1, &shape.incx, &beta, y.ptr + 1, &shape.incy),
            .packed_storage => @field(fortran.internal_admitted, symbol)(&operation, &uplo, &n, &alpha, a.ptr + 1, x.ptr + 1, &shape.incx, &beta, y.ptr + 1, &shape.incy),
        }
        try observation.checkOutput();
        try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
        try std.testing.expectEqual(entries_before + 1, Probe.entries.load(.acquire));
        try std.testing.expectEqual(finishes_before, Probe.finishes.load(.acquire));
        if (scratch) |log| {
            if (scratch_elements != 0) try log.checkScratch(T, scratch_elements, refused) else try std.testing.expect(!log.snapshot().seen);
        }
        if (closing) try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        operation.end();
        ended = true;
        try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
        try std.testing.expectEqual(finishes_before + 1, Probe.finishes.load(.acquire));
    }
};

test "borrowed symmetric MV twelve entries preserve dense oracle strides padding and zero scales" {
    const F = SymmetricMvResourceFixture;
    const Single = ResourceGemmProbe(1);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Single.setup(ctx, &allocations);
    defer {
        Single.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ F.Storage.dense, F.Storage.band, F.Storage.packed_storage }) |storage| {
            for ([_]bool{ true, false }) |upper| {
                for ([_]F.Shape{
                    .{},
                    .{ .incx = -2, .incy = 3 },
                    .{ .incx = 2, .incy = -2, .alpha_zero = true },
                    .{ .beta_zero = true },
                    .{ .incx = -3, .incy = 2, .alpha_zero = true, .beta_zero = true },
                }) |config| {
                    var shape = config;
                    shape.upper = upper;
                    try F.run(T, storage, Single, ctx, shape, null, 0, false);
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 120), Single.entries.load(.acquire));
    try std.testing.expectEqual(Single.entries.load(.acquire), Single.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    try finishResourceGemmContext(Single, ctx, &allocations);
    try guard.check();
}

test "borrowed symmetric MV keeps policy one and four scratch executors and allocators independent" {
    const F = SymmetricMvResourceFixture;
    const Single = ResourceGemmProbe(1);
    const Parallel = ResourceGemmProbe(4);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var aa: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var ab: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var sa: F.ScratchLog = .{ .backing = &aa };
    var sb: F.ScratchLog = .{ .backing = &ab };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(sa.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(sb.allocator(), Parallel.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    sa.context = a;
    sb.context = b;
    Single.setup(a, &aa);
    Parallel.setup(b, &ab);
    defer {
        Single.disarm();
        Parallel.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ F.Storage.dense, F.Storage.band, F.Storage.packed_storage }) |storage| {
            for ([_]bool{ true, false }) |upper| {
                const shape: F.Shape = .{ .n = 512, .k = 8, .upper = upper };
                const before_b = ab.stats();
                const queries_b = Parallel.queries.load(.acquire);
                const single_scratch: usize = if (storage == .packed_storage and !F.isComplex(T) and @import("builtin").target.cpu.arch == .aarch64) 512 else 0;
                try F.run(T, storage, Single, a, shape, &sa, single_scratch, false);
                try std.testing.expectEqualDeep(before_b, ab.stats());
                try std.testing.expectEqual(queries_b, Parallel.queries.load(.acquire));
                try std.testing.expectEqual(@as(u32, 0), reference_a.executor.persistent_worker_count.load(.acquire));
                try std.testing.expectEqual(@as(u32, 0), reference_a.executor.persistent_job_count.load(.acquire));
                const before_a = aa.stats();
                const queries_a = Single.queries.load(.acquire);
                var generations: [3]u32 = undefined;
                for (&generations, 0..) |*generation, i| generation.* = reference_b.executor.persistent_worker_generation[i].load(.acquire);
                try F.run(T, storage, Parallel, b, shape, &sb, if (storage == .band) 0 else 4 * 512, false);
                try std.testing.expectEqualDeep(before_a, aa.stats());
                try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
                if (storage != .band) {
                    // A real synchronous four-task batch ran on this Context.
                    // This is stronger than a policy query or generic allocation.
                    try std.testing.expectEqual(@as(u32, 4), reference_b.executor.persistent_job_count.load(.acquire));
                    try std.testing.expect(reference_b.executor.persistent_worker_count.load(.acquire) >= 3);
                    try std.testing.expectEqual(@as(u32, 3), reference_b.executor.persistent_done_target.load(.acquire));
                    try std.testing.expectEqual(@as(u32, 3), reference_b.executor.persistent_done_count.load(.acquire));
                    for (generations, 0..) |generation, i| try std.testing.expectEqual(generation +% 1, reference_b.executor.persistent_worker_generation[i].load(.acquire));
                } else {
                    for (generations, 0..) |generation, i| try std.testing.expectEqual(generation, reference_b.executor.persistent_worker_generation[i].load(.acquire));
                }
                try guard.check();
            }
        }
    }
    try std.testing.expect(Single.queries.load(.acquire) > 0);
    try std.testing.expect(Parallel.queries.load(.acquire) > 0);
    try std.testing.expect(Parallel.configurations.load(.acquire) > 0);
    try std.testing.expectEqual(@as(usize, 24), Single.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 24), Parallel.entries.load(.acquire));
    try finishResourceGemmContext(Parallel, b, &ab);
    try finishResourceGemmContext(Single, a, &aa);
    try guard.check();
}

test "borrowed symmetric MV refuses real scratch requests and falls back with beta applied once" {
    const F = SymmetricMvResourceFixture;
    const Parallel = ResourceGemmProbe(4);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ F.Storage.dense, F.Storage.packed_storage }) |storage| {
            for ([_]bool{ true, false }) |upper| {
                var allocations: ResourceGemmAllocations = .{ .budget = 0 };
                var scratch: F.ScratchLog = .{ .backing = &allocations };
                var reference_ctx: FixtureReferenceContext = undefined;
                reference_ctx.initInPlace(scratch.allocator(), Parallel.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
                const ctx = reference_ctx.borrowedContext();
                scratch.context = ctx;
                Parallel.setup(ctx, &allocations);
                defer {
                    Parallel.disarm();
                    _ = ctx.requestClose();
                    _ = fixtureProgressClose(ctx);
                }
                // The same independent oracle includes nonzero old Y and beta.
                // A real failed scratch allocation occurs before task submission.
                try F.run(T, storage, Parallel, ctx, .{ .n = 512, .k = 8, .upper = upper }, &scratch, 4 * 512, true);
                const stats = allocations.stats();
                try std.testing.expectEqual(@as(usize, 1), stats.attempts);
                try std.testing.expectEqual(@as(usize, 1), stats.rejected);
                try std.testing.expectEqual(@as(usize, 0), stats.accepted);
                try std.testing.expectEqual(@as(usize, 0), stats.freed);
                try std.testing.expectEqual(@as(u32, 0), reference_ctx.executor.persistent_worker_count.load(.acquire));
                try std.testing.expectEqual(@as(u32, 0), reference_ctx.executor.persistent_job_count.load(.acquire));
                try std.testing.expectEqual(@as(usize, 1), Parallel.entries.load(.acquire));
                try std.testing.expectEqual(@as(usize, 1), Parallel.finishes.load(.acquire));
                try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
                _ = ctx.requestClose();
                try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
                try guard.check();
            }
        }
    }
}

// Distinct from the small band cases above: n=512/k=8 selects the unit
// HBMV body, while dense HEMV selects the unit body at both policies.
// Alpha stays nonzero, so these are numerical-kernel cases, not alpha-zero
// shortcuts. The shared fixture poisons old Y and the imaginary diagonal.
test "borrowed Hermitian unit dense and band zero beta ignores old NaNs on both Context policies" {
    const F = SymmetricMvResourceFixture;
    const Single = ResourceGemmProbe(1);
    const Parallel = ResourceGemmProbe(4);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var aa: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var ab: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var sa: F.ScratchLog = .{ .backing = &aa };
    var sb: F.ScratchLog = .{ .backing = &ab };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(sa.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(sb.allocator(), Parallel.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    sa.context = a;
    sb.context = b;
    Single.setup(a, &aa);
    Parallel.setup(b, &ab);
    defer {
        Single.disarm();
        Parallel.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ F.Storage.dense, F.Storage.band }) |storage| {
            for ([_]bool{ true, false }) |upper| {
                const shape: F.Shape = .{
                    .n = 512,
                    .k = 8,
                    .upper = upper,
                    .incx = 1,
                    .incy = 1,
                    .beta_zero = true,
                };
                const before_b = ab.stats();
                const queries_b = Parallel.queries.load(.acquire);
                try F.run(T, storage, Single, a, shape, &sa, 0, false);
                try std.testing.expectEqualDeep(before_b, ab.stats());
                try std.testing.expectEqual(queries_b, Parallel.queries.load(.acquire));
                try std.testing.expectEqual(@as(usize, 0), aa.stats().attempts);
                try std.testing.expectEqual(@as(u32, 0), reference_a.executor.persistent_worker_count.load(.acquire));
                try std.testing.expectEqual(@as(u32, 0), reference_a.executor.persistent_job_count.load(.acquire));

                const before_a = aa.stats();
                const queries_a = Single.queries.load(.acquire);
                var generations: [3]u32 = undefined;
                for (&generations, 0..) |*generation, i|
                    generation.* = reference_b.executor.persistent_worker_generation[i].load(.acquire);
                try F.run(T, storage, Parallel, b, shape, &sb, if (storage == .dense) 4 * 512 else 0, false);
                try std.testing.expectEqualDeep(before_a, aa.stats());
                try std.testing.expectEqual(queries_a, Single.queries.load(.acquire));
                if (storage == .dense) {
                    try std.testing.expectEqual(@as(u32, 4), reference_b.executor.persistent_job_count.load(.acquire));
                    try std.testing.expect(reference_b.executor.persistent_worker_count.load(.acquire) >= 3);
                    try std.testing.expectEqual(@as(u32, 3), reference_b.executor.persistent_done_target.load(.acquire));
                    try std.testing.expectEqual(@as(u32, 3), reference_b.executor.persistent_done_count.load(.acquire));
                    for (generations, 0..) |generation, i|
                        try std.testing.expectEqual(generation +% 1, reference_b.executor.persistent_worker_generation[i].load(.acquire));
                } else {
                    for (generations, 0..) |generation, i|
                        try std.testing.expectEqual(generation, reference_b.executor.persistent_worker_generation[i].load(.acquire));
                }
                try guard.check();
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 8), Single.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 8), Parallel.entries.load(.acquire));
    try std.testing.expectEqual(Single.entries.load(.acquire), Single.finishes.load(.acquire));
    try std.testing.expectEqual(Parallel.entries.load(.acquire), Parallel.finishes.load(.acquire));
    try finishResourceGemmContext(Parallel, b, &ab);
    try finishResourceGemmContext(Single, a, &aa);
    try guard.check();
}

// A test-only observer wrapper. The original Probe still owns the real policy,
// counters and allocator. Configure the extra observer only while quiescent;
// never replace Context test configuration after admission or after closing.
fn SymmetricMvLifecycleProbe(comptime thread_limit: usize) type {
    return struct {
        const Base = ResourceGemmProbe(thread_limit);
        const policy = Base.policy;
        const entries = &Base.entries;
        const finishes = &Base.finishes;
        const configurations = &Base.configurations;
        var observation: ?*const anyopaque = null;
        var inspect_fn: ?*const fn (*const anyopaque) anyerror!void = null;

        fn setup(ctx: *ResourceGemmContext, allocations: *ResourceGemmAllocations) void {
            Base.setup(ctx, allocations);
            unwatch();
            ResourceGemmContext.testing.configure(ctx, std.math.maxInt(usize), observe);
        }
        fn disarm() void {
            Base.disarm();
            unwatch();
        }
        fn watch(raw: *const anyopaque, inspect: *const fn (*const anyopaque) anyerror!void) void {
            observation = raw;
            inspect_fn = inspect;
        }
        fn unwatch() void {
            observation = null;
            inspect_fn = null;
        }
        fn observe(ctx: *ResourceGemmContext, event: admission_execution.context.TestEvent) void {
            // Only the caller ending the measured Operation reads this watch.
            // The separate busy Operation cannot finish until its atomic release,
            // which is published after runLifecycle has unwatched the output.
            if (event == .finishing) if (inspect_fn) |inspect|
                inspect(observation.?) catch @panic("symmetric MV finishing lost output, inputs, scratch release or admission");
            Base.observe(ctx, event);
        }
    };
}

fn runSymmetricMvBusyClosing(comptime T: type, comptime storage: SymmetricMvResourceFixture.Storage, upper: bool) !void {
    const F = SymmetricMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(2);
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var scratch: F.ScratchLog = .{ .backing = &allocations };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(scratch.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    scratch.context = ctx;
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var blocker: ResourceGemmBlockingBatch = .{ .context = ctx };
    const thread = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&blocker});
    var joined = false;
    defer {
        // Every fallible assertion below unwinds through release and join before
        // the Context, allocator, or stack-resident batch can be destroyed.
        blocker.release.store(true, .release);
        if (!joined) thread.join();
    }
    while (!blocker.entered.load(.acquire) and !blocker.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(blocker.entered.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(u32, 2), reference_ctx.executor.persistent_job_count.load(.acquire));
    // The existing visits counter is monotonic, not a publication handshake.
    // Acquire the real worker's completion instead before reading its effects.
    while (reference_ctx.executor.persistent_done_count.load(.acquire) != 1 and !blocker.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(u32, 1), reference_ctx.executor.persistent_done_count.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), blocker.visits[1].load(.acquire));
    // These fields were published before the caller's release-store to entered;
    // no following batch can acquire io_busy until this blocked batch returns.
    const task_fn = reference_ctx.executor.persistent_task_fn;
    const tasks = reference_ctx.executor.persistent_tasks;
    var generations: [admission_execution.owner.max_tasks]u32 = undefined;
    for (&generations, 0..) |*generation, i| generation.* = reference_ctx.executor.persistent_worker_generation[i].load(.acquire);
    const configurations = Probe.configurations.load(.acquire);
    const before = allocations.stats();
    try F.runLifecycle(T, storage, Probe, ctx, .{ .n = 512, .k = 8, .upper = upper }, &scratch, 2 * 512, false, true, true, true);
    const after = allocations.stats();
    try std.testing.expectEqual(@as(usize, 1), after.attempts - before.attempts);
    try std.testing.expectEqual(@as(usize, 1), after.accepted - before.accepted);
    try std.testing.expectEqual(@as(usize, 1), after.freed - before.freed);
    try std.testing.expectEqual(before.rejected, after.rejected);
    try std.testing.expectEqual(configurations, Probe.configurations.load(.acquire));
    try std.testing.expect(reference_ctx.executor.persistent_task_fn == task_fn);
    try std.testing.expect(reference_ctx.executor.persistent_tasks == tasks);
    try std.testing.expectEqual(@as(u32, 2), reference_ctx.executor.persistent_job_count.load(.acquire));
    for (generations, 0..) |generation, i| try std.testing.expectEqual(generation, reference_ctx.executor.persistent_worker_generation[i].load(.acquire));
    try std.testing.expect(!blocker.finished.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 2), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    blocker.release.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(blocker.accepted);
    for (&blocker.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Probe.finishes.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

test "borrowed symmetric MV finishes after close with real busy Executor fallback and one beta" {
    const F = SymmetricMvResourceFixture;
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    // Each case owns a fresh Context; its actual two-task blocker remains live
    // through scratch acquire/free, serial fallback and the finishing callback.
    try runSymmetricMvBusyClosing(f32, F.Storage.dense, true);
    try guard.check();
    try runSymmetricMvBusyClosing(fortran.ComplexF64, F.Storage.dense, false);
    try guard.check();
    try runSymmetricMvBusyClosing(fortran.ComplexF32, F.Storage.packed_storage, true);
    try guard.check();
    try runSymmetricMvBusyClosing(f64, F.Storage.packed_storage, false);
    try guard.check();
}

test "borrowed Hermitian band finishes on an already closing Context without scratch" {
    const F = SymmetricMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(2);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        for ([_]bool{ true, false }) |upper| {
            var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
            var scratch: F.ScratchLog = .{ .backing = &allocations };
            var reference_ctx: FixtureReferenceContext = undefined;
            reference_ctx.initInPlace(scratch.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const ctx = reference_ctx.borrowedContext();
            scratch.context = ctx;
            Probe.setup(ctx, &allocations);
            defer {
                Probe.disarm();
                _ = ctx.requestClose();
                _ = fixtureProgressClose(ctx);
            }
            // This is the n512/k8 unit band path, not an outer task submission.
            try F.runLifecycle(T, .band, Probe, ctx, .{ .n = 512, .k = 8, .upper = upper }, &scratch, 0, false, true, true, false);
            try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
            try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
            try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_init_state.load(.acquire));
            try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
            try finishResourceGemmContext(Probe, ctx, &allocations);
            try guard.check();
        }
    }
}

fn runSymmetricMvPausedFree(comptime T: type, comptime storage: SymmetricMvResourceFixture.Storage, comptime thread_limit: usize, upper: bool) !void {
    const F = SymmetricMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(thread_limit);
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var barrier: F.FreeBarrier = .{};
    var scratch: F.ScratchLog = .{ .backing = &allocations, .free_barrier = &barrier };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(scratch.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    scratch.context = ctx;
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    const Work = struct {
        ctx: *ResourceGemmContext,
        scratch: *F.ScratchLog,
        upper: bool,
        finished: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            F.runLifecycle(T, storage, Probe, self.ctx, .{ .n = 512, .k = 8, .upper = self.upper }, self.scratch, thread_limit * 512, false, true, false, false) catch |err| {
                self.failure = err;
            };
        }
    };
    var work: Work = .{ .ctx = ctx, .scratch = &scratch, .upper = upper };
    const thread = try std.Thread.spawn(.{}, Work.run, .{&work});
    var joined = false;
    defer {
        barrier.allow_free.store(true, .release);
        if (!joined) thread.join();
    }
    while (!barrier.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(barrier.entered.load(.acquire));
    try std.testing.expect(!work.finished.load(.acquire));
    try std.testing.expect(!barrier.completed.load(.acquire));
    // entered's release/acquire edge publishes all final stores and input
    // snapshots. Only successful idle paths pause here: merge precedes free.
    try barrier.inspect_fn.?(barrier.observation.?);
    const pending_free = scratch.snapshot();
    try std.testing.expect(pending_free.seen and pending_free.accepted and !pending_free.freed);
    try std.testing.expectEqual(thread_limit * 512 * @sizeOf(T), pending_free.bytes);
    try std.testing.expectEqual(@alignOf(T), pending_free.alignment);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    // The numerical caller has released io_busy before releasing its workspace.
    try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_busy.load(.acquire));
    if (comptime thread_limit > 1) {
        try std.testing.expectEqual(@as(u32, thread_limit), reference_ctx.executor.persistent_job_count.load(.acquire));
        try std.testing.expectEqual(@as(u32, thread_limit - 1), reference_ctx.executor.persistent_done_count.load(.acquire));
        try std.testing.expect(Probe.configurations.load(.acquire) > 0);
    } else {
        try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
    }
    try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    try std.testing.expectError(error.Closing, ctx.begin());
    try std.testing.expectEqual(admission_execution.context.Phase.closing, ctx.snapshot().phase);
    const allocations_at_free = allocations.stats(); // No recorder lock held by the paused thread.
    try std.testing.expect(allocations_at_free.accepted > allocations_at_free.freed);
    try std.testing.expect(!barrier.completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    barrier.allow_free.store(true, .release);
    thread.join();
    joined = true;
    if (work.failure) |err| return err;
    try std.testing.expect(work.finished.load(.acquire));
    try std.testing.expect(barrier.completed.load(.acquire));
    try scratch.checkScratch(T, thread_limit * 512, false);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    // Persistent Executor storage may outlive Operation.end; only drain must
    // return all real allocations to the same captured backing allocator.
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

test "borrowed symmetric MV close stays busy until four real scratch frees return" {
    const F = SymmetricMvResourceFixture;
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    try runSymmetricMvPausedFree(f32, F.Storage.dense, 4, true);
    try guard.check();
    try runSymmetricMvPausedFree(fortran.ComplexF64, F.Storage.dense, 4, false);
    try guard.check();
    try runSymmetricMvPausedFree(fortran.ComplexF32, F.Storage.packed_storage, 4, true);
    try guard.check();
    // Only this architecture selects the real packed fused-single allocation.
    // Other targets execute the three parallel sites without claiming the fourth.
    if (comptime @import("builtin").target.cpu.arch == .aarch64) {
        try runSymmetricMvPausedFree(f64, F.Storage.packed_storage, 1, false);
        try guard.check();
    }
}

// Append to the existing Fortran compatibility root after its current tests.
// This fragment depends only on the unchanged ResourceGemm fixtures above it.
// It exercises one real lower Hermitian column, not a complete n-by-n HEMV.

fn LowerHermitianFirstColumn(comptime T: type) type {
    return struct {
        const Self = @This();
        const F = if (T == fortran.ComplexF32) f32 else f64;
        const guard_count = 8;
        n: usize,
        column: []T,
        x: []T,
        y: []T,
        column_before: []T,
        x_before: []T,
        expected: []T,

        fn init(n: usize) !Self {
            const allocator = std.testing.allocator;
            const len = n + 2 * guard_count;
            const column = try allocator.alloc(T, len);
            errdefer allocator.free(column);
            const x = try allocator.alloc(T, len);
            errdefer allocator.free(x);
            const y = try allocator.alloc(T, len);
            errdefer allocator.free(y);
            const sentinel = T{ .re = 173, .im = -181 };
            @memset(column, sentinel);
            @memset(x, sentinel);
            @memset(y, sentinel);
            @memset(column[guard_count..][0..n], .{ .re = 1, .im = 2 });
            column[guard_count] = .{ .re = 2, .im = std.math.nan(F) };
            @memset(x[guard_count..][0..n], .{ .re = 1, .im = -1 });
            x[guard_count] = .{ .re = 1, .im = 0 };
            @memset(y[guard_count..][0..n], .{ .re = 3, .im = 5 });
            const column_before = try allocator.dupe(T, column);
            errdefer allocator.free(column_before);
            const x_before = try allocator.dupe(T, x);
            errdefer allocator.free(x_before);
            const expected = try allocator.dupe(T, y);
            // Closed form, not a second HEMV implementation: alpha=1+i,
            // conj(1+2i)*(1-i)=-1-3i, so alpha times its L-term sum is
            // 2L-4Li. The diagonal contributes 2+2i and ignores its NaN im.
            const tail: i64 = @intCast(n - 1);
            expected[guard_count] = .{
                .re = @floatFromInt(2 * tail + 5),
                .im = @floatFromInt(7 - 4 * tail),
            };
            @memset(expected[guard_count + 1 ..][0 .. n - 1], .{ .re = 2, .im = 8 });
            return .{ .n = n, .column = column, .x = x, .y = y, .column_before = column_before, .x_before = x_before, .expected = expected };
        }

        fn deinit(self: Self) void {
            const allocator = std.testing.allocator;
            allocator.free(self.expected);
            allocator.free(self.x_before);
            allocator.free(self.column_before);
            allocator.free(self.y);
            allocator.free(self.x);
            allocator.free(self.column);
        }

        fn call(self: *const Self, operation: *const admission_execution.context.Operation) void {
            fortran.testing.admitted_symmetric.lowerFirstHermitianColumn(
                T,
                operation,
                self.n,
                .{ .re = 1, .im = 1 },
                self.column[guard_count..].ptr,
                self.x[guard_count..].ptr,
                self.y[guard_count..].ptr,
            );
        }

        fn check(self: *const Self) !void {
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.column_before), std.mem.sliceAsBytes(self.column));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.x_before), std.mem.sliceAsBytes(self.x));
            for (self.y[guard_count..][0..self.n], self.expected[guard_count..][0..self.n]) |actual, expected| {
                try std.testing.expectEqual(expected.re, actual.re);
                try std.testing.expectEqual(expected.im, actual.im);
            }
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.expected[0..guard_count]), std.mem.sliceAsBytes(self.y[0..guard_count]));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.expected[guard_count + self.n ..]), std.mem.sliceAsBytes(self.y[guard_count + self.n ..]));
        }
    };
}

// These wrappers observe real configure callbacks; they do not select task
// results, write io_busy, change thresholds, or substitute a numerical body.
fn LowerHermitianNestedProbe(comptime T: type) type {
    return struct {
        const Base = ResourceGemmProbe(4);
        var caller_id: std.Thread.Id = undefined;
        var data: ?*const LowerHermitianFirstColumn(T) = null;
        var finishing_armed: bool = false;
        var caller_configs: std.atomic.Value(usize) = .init(0);
        var helper_configs: [4]std.atomic.Value(usize) = @splat(.init(0));
        var foreign_helpers: std.atomic.Value(usize) = .init(0);
        var same_thread_helpers: std.atomic.Value(usize) = .init(0);

        fn configure(index: ?usize) void {
            Base.policy.configure_thread(index);
            const state = Base.context.?.snapshot();
            if (state.active_operations != 1 or state.resource_pins != 0)
                @panic("lower Hermitian L1 configuration escaped its one borrowed operation");
            if (index) |i| {
                if (i == 0 or i >= helper_configs.len) @panic("unexpected lower Hermitian helper index");
                _ = helper_configs[i].fetchAdd(1, .monotonic);
                if (std.Thread.getCurrentId() == caller_id) {
                    _ = same_thread_helpers.fetchAdd(1, .monotonic);
                } else {
                    _ = foreign_helpers.fetchAdd(1, .monotonic);
                }
            } else {
                if (std.Thread.getCurrentId() != caller_id) @panic("lower Hermitian batch caller changed thread");
                _ = caller_configs.fetchAdd(1, .monotonic);
            }
        }

        const policy: admission_execution.owner.Policy = .{
            .max_threads = Base.policy.max_threads,
            .helper_count = Base.policy.helper_count,
            .configure_thread = configure,
            .worker_stack_size = Base.policy.worker_stack_size,
        };
        const Counts = struct { caller: usize, helpers: [4]usize, foreign: usize, same_thread: usize, total: usize };
        fn counts() Counts {
            var helpers: [4]usize = undefined;
            for (&helpers, 0..) |*value, i| value.* = helper_configs[i].load(.acquire);
            return .{ .caller = caller_configs.load(.acquire), .helpers = helpers, .foreign = foreign_helpers.load(.acquire), .same_thread = same_thread_helpers.load(.acquire), .total = Base.configurations.load(.acquire) };
        }
        fn setup(ctx: *ResourceGemmContext, allocations: *ResourceGemmAllocations, buffers: *const LowerHermitianFirstColumn(T)) void {
            Base.setup(ctx, allocations);
            caller_id = std.Thread.getCurrentId();
            data = buffers;
            finishing_armed = false;
            caller_configs.store(0, .release);
            for (&helper_configs) |*count| count.store(0, .release);
            foreign_helpers.store(0, .release);
            same_thread_helpers.store(0, .release);
            // Count limit one rejects a second admission; no new admission is
            // made by the seam, inner AXPY/DOT, or the outer real batch worker.
            ResourceGemmContext.testing.configure(ctx, 1, observe);
        }
        fn observe(ctx: *ResourceGemmContext, event: admission_execution.context.TestEvent) void {
            if (ctx.snapshot().active_operations != 1) @panic("lower Hermitian admission count changed");
            Base.observe(ctx, event);
            if (event == .finishing and finishing_armed)
                data.?.check() catch @panic("lower Hermitian operation ended before complete output/input/guard checks");
        }
        fn disarm() void {
            finishing_armed = false;
            Base.disarm();
        }
    };
}

fn lowerHermitianExpectedBatches(comptime T: type) usize {
    const arch = @import("builtin").target.cpu.arch;
    const axpy: usize = if (arch == .aarch64 or arch == .x86_64) 1 else 0;
    const dot: usize = if (arch == .x86_64 or T == fortran.ComplexF64) 1 else 0;
    return axpy + dot;
}

fn lowerHermitianGenerations(ctx: *ResourceGemmContext) [admission_execution.owner.max_tasks]u32 {
    var result: [admission_execution.owner.max_tasks]u32 = undefined;
    for (&result, 0..) |*generation, i| generation.* = fixtureExecutor(ctx).persistent_worker_generation[i].load(.acquire);
    return result;
}

fn runLowerHermitianIdle(comptime T: type, n: usize) !void {
    const Probe = LowerHermitianNestedProbe(T);
    var data = try LowerHermitianFirstColumn(T).init(n);
    defer data.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations, &data);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    {
        var operation = try ctx.begin();
        defer operation.end();
        try std.testing.expect(operation.borrowedContext() == ctx);
        data.call(&operation);
        try data.check();
        const batches = if (n - 1 < 512 * 1024) @as(usize, 0) else lowerHermitianExpectedBatches(T);
        const counts = Probe.counts();
        // A fixed-column fast path may legitimately bypass L1. Do not disable
        // it or silently count coverage: this exact coverage assertion then
        // fails and the parent must report that this probe did not reach L1.
        try std.testing.expectEqual(batches, Probe.Base.queries.load(.acquire));
        try std.testing.expectEqual(batches, counts.caller);
        try std.testing.expectEqual(@as(usize, 0), counts.same_thread);
        try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_busy.load(.acquire));
        if (batches == 0) {
            try std.testing.expectEqual(@as(usize, 0), counts.total);
            try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
            try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_init_state.load(.acquire));
            try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
        } else if (comptime @import("builtin").target.cpu.arch != .x86_64) {
            // AArch64 C32: one ordinary AXPY batch, C64: AXPY plus DOT.
            // No persistent generation claim is made for these ordinary runs.
            try std.testing.expectEqual(batches * 4, counts.total);
            for (counts.helpers[1..]) |count| try std.testing.expectEqual(batches, count);
            try std.testing.expectEqual(batches * 3, counts.foreign);
            try std.testing.expectEqual(@as(u8, 2), reference_ctx.executor.io_init_state.load(.acquire));
            try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
            try std.testing.expectEqual(@as([admission_execution.owner.max_tasks]u32, @splat(0)), lowerHermitianGenerations(ctx));
            try std.testing.expect(allocations.stats().accepted > 0);
        } else {
            // x86 uses runLowLatency. Observe the actual initialized mechanism;
            // persistent configuration happens at worker startup, not each job.
            try std.testing.expect(counts.foreign >= 3);
            try std.testing.expect(allocations.stats().accepted > 0);
            if (reference_ctx.executor.persistent_init_state.load(.acquire) == 2) {
                try std.testing.expectEqual(@as(usize, 3) + batches, counts.total);
                for (counts.helpers[1..]) |count| try std.testing.expectEqual(@as(usize, 1), count);
                try std.testing.expectEqual(@as(u32, 3), reference_ctx.executor.persistent_worker_count.load(.acquire));
            } else {
                try std.testing.expectEqual(batches * 4, counts.total);
                for (counts.helpers[1..]) |count| try std.testing.expectEqual(batches, count);
                try std.testing.expectEqual(@as(u8, 2), reference_ctx.executor.io_init_state.load(.acquire));
            }
        }
        try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        Probe.finishing_armed = true;
    }
    try std.testing.expectEqual(@as(usize, 1), Probe.Base.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.Base.finishes.load(.acquire));
    try data.check();
    // Io.Threaded allocations may legitimately survive end until this drain.
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

fn LowerHermitianWorker(comptime T: type) type {
    return struct {
        const Self = @This();
        const Probe = LowerHermitianNestedProbe(T);
        operation: *const admission_execution.context.Operation,
        data: *const LowerHermitianFirstColumn(T),
        allocations: *ResourceGemmAllocations,
        caller_ready: std.atomic.Value(bool) = .init(false),
        worker_done: std.atomic.Value(bool) = .init(false),
        visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
        worker_failure: ?anyerror = null,
        real_worker: bool = false,

        fn execute(raw: *const anyopaque, index: usize) void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            if (index >= 2) @panic("unexpected outer lower Hermitian task index");
            _ = self.visits[index].fetchAdd(1, .monotonic);
            if (index == 0) {
                self.caller_ready.store(true, .release);
                while (!self.worker_done.load(.acquire)) std.Thread.yield() catch {};
                return;
            }
            defer self.worker_done.store(true, .release);
            // Group.concurrent can fail and run synchronously. Detect that
            // before waiting for task0, so lack of a real worker cannot deadlock
            // or masquerade as worker-nested coverage.
            if (std.Thread.getCurrentId() == Probe.caller_id) {
                self.worker_failure = error.RealExecutorWorkerUnavailable;
                return;
            }
            self.real_worker = true;
            while (!self.caller_ready.load(.acquire)) std.Thread.yield() catch {};
            self.perform() catch |err| {
                self.worker_failure = err;
            };
        }

        fn perform(self: *Self) !void {
            const ctx = self.operation.borrowedContext();
            try std.testing.expect(ctx == Probe.Base.context.?);
            try std.testing.expectEqual(@as(u8, 1), fixtureExecutor(ctx).io_busy.load(.acquire));
            const configurations = Probe.counts();
            const before = self.allocations.stats();
            const generations = lowerHermitianGenerations(ctx);
            const queries = Probe.Base.queries.load(.acquire);
            const entries = Probe.Base.entries.load(.acquire);
            self.data.call(self.operation);
            try self.data.check();
            // Both inner L1 submissions see the outer batch's real busy state.
            // On AArch64 C32 only AXPY is submitted; C64 attempts AXPY and DOT.
            try std.testing.expectEqual(queries + lowerHermitianExpectedBatches(T), Probe.Base.queries.load(.acquire));
            try std.testing.expectEqualDeep(configurations, Probe.counts());
            try std.testing.expectEqualDeep(before, self.allocations.stats());
            try std.testing.expectEqual(generations, lowerHermitianGenerations(ctx));
            try std.testing.expectEqual(@as(u8, 0), fixtureExecutor(ctx).persistent_init_state.load(.acquire));
            try std.testing.expectEqual(@as(u32, 0), fixtureExecutor(ctx).persistent_job_count.load(.acquire));
            try std.testing.expectEqual(@as(u8, 1), fixtureExecutor(ctx).io_busy.load(.acquire));
            try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
            try std.testing.expectEqual(entries, Probe.Base.entries.load(.acquire));
            try std.testing.expectEqual(@as(usize, 0), Probe.Base.finishes.load(.acquire));
        }
    };
}

fn runLowerHermitianNested(comptime T: type) !void {
    const Probe = LowerHermitianNestedProbe(T);
    var data = try LowerHermitianFirstColumn(T).init(524289);
    defer data.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations, &data);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    {
        var operation = try ctx.begin();
        defer operation.end();
        var work: LowerHermitianWorker(T) = .{ .operation = &operation, .data = &data, .allocations = &allocations };
        // A genuine ordinary Executor batch supplies the worker. No persistent
        // function/payload field is read: those are undefined before its init.
        const accepted = reference_ctx.executor.run(.{ .task_fn = LowerHermitianWorker(T).execute, .tasks = &work, .count = 2 });
        try std.testing.expect(accepted);
        if (work.worker_failure) |err| return err;
        try std.testing.expect(work.real_worker);
        for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), Probe.counts().caller);
        try std.testing.expectEqual([_]usize{ 0, 1, 0, 0 }, Probe.counts().helpers);
        try std.testing.expectEqual(@as(usize, 1), Probe.counts().foreign);
        try std.testing.expectEqual(@as(usize, 0), Probe.counts().same_thread);
        try std.testing.expectEqual(@as(u8, 2), reference_ctx.executor.io_init_state.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_busy.load(.acquire));
        try std.testing.expect(allocations.stats().accepted > 0);
        try data.check();
        Probe.finishing_armed = true;
    }
    try std.testing.expectEqual(@as(usize, 1), Probe.Base.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.Base.finishes.load(.acquire));
    try data.check();
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

test "borrowed lower Hermitian first column observes architecture specific L1 threshold without default resources" {
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        try runLowerHermitianIdle(T, 524288);
        try guard.check();
        try runLowerHermitianIdle(T, 524289);
        try guard.check();
    }
}

test "borrowed lower Hermitian first column finishes inside its real Executor worker without nested submission" {
    const arch = @import("builtin").target.cpu.arch;
    // Other architectures do not have the complex AXPY parallel branch. Keep
    // their numerical/idle checks above, but do not claim this paired AXPY/DOT
    // nested-submission coverage when that branch does not exist.
    if (arch != .aarch64 and arch != .x86_64) return error.SkipZigTest;
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        try runLowerHermitianNested(T);
        try guard.check();
    }
}

// Independent triangular mathematics plus the already-tested Context fixtures.
// The oracle never reads a production-decoded matrix or calls a BLAS routine.
const TriangularMvResourceFixture = struct {
    const Kind = enum { dense_multiply, dense_solve, packed_multiply, band_solve };
    const F = SymmetricMvResourceFixture;
    const Wide = F.Wide;
    const Shape = struct {
        n: usize = 7,
        k: usize = 2,
        upper: bool = true,
        trans: u8 = 'N',
        unit: bool = false,
        incx: fortran.BlasInt = 1,
        late_overflow: bool = false,
    };
    const apple_finite = @import("builtin").target.cpu.arch == .aarch64 and @import("builtin").target.os.tag == .macos;

    fn matrix(comptime T: type, comptime kind: Kind, shape: Shape, row: usize, col: usize) Wide {
        if ((shape.upper and row > col) or (!shape.upper and row < col)) return .{ .re = 0 };
        if (kind == .band_solve and @max(row, col) - @min(row, col) > shape.k) return .{ .re = 0 };
        if (comptime !F.isComplex(T)) if (shape.late_overflow) {
            if (row != col) return .{ .re = 0 };
            return .{ .re = if (row + 1 == shape.n) (if (isSolve(kind)) @as(f64, 0.5) else @as(f64, std.math.floatMax(T))) else 2 };
        };
        if (row == col) return if (shape.unit) .{ .re = 1 } else .{ .re = 2, .im = if (F.isComplex(T)) 2 else 0 };
        return .{
            .re = @floatFromInt(@as(i64, @intCast((row * 3 + col * 5) % 3)) - 1),
            .im = if (F.isComplex(T)) @floatFromInt(@as(i64, @intCast((row * 2 + col) % 3)) - 1) else 0,
        };
    }
    fn transformed(comptime T: type, comptime kind: Kind, shape: Shape, row: usize, col: usize) Wide {
        var value = if (shape.trans == 'N') matrix(T, kind, shape, row, col) else matrix(T, kind, shape, col, row);
        if (shape.trans == 'C') value.im = -value.im;
        return value;
    }
    fn solution(comptime T: type, i: usize) Wide {
        return .{ .re = @floatFromInt(@as(i64, @intCast(i % 5)) - 2), .im = if (F.isComplex(T)) @floatFromInt(@as(i64, @intCast(i % 3)) - 1) else 0 };
    }
    fn isSolve(comptime kind: Kind) bool {
        return kind == .dense_solve or kind == .band_solve;
    }
    fn staged(comptime T: type, comptime kind: Kind, shape: Shape) bool {
        return apple_finite and T == f32 and kind == .packed_multiply and shape.n >= 64 and shape.n <= 1024 and shape.trans != 'N' and shape.incx != 1;
    }
    fn scratchElements(comptime T: type, comptime kind: Kind, shape: Shape) usize {
        if (!apple_finite or F.isComplex(T)) return 0;
        if (kind == .packed_multiply and shape.n > 128) return shape.n;
        if (kind == .band_solve and shape.n >= 128 and !(shape.k == 0 and shape.unit)) return shape.n;
        return 0;
    }

    fn Observation(comptime T: type) type {
        return struct {
            ctx: *ResourceGemmContext,
            shape: Shape,
            a: []const T,
            a_before: []const T,
            x: []const T,
            x_before: []const T,
            expected: []const T,
            scratch: ?*F.ScratchLog,
            scratch_elements: usize,
            refused: bool,
            staged_copyback: bool,
            active: usize,
            busy: bool,

            fn checkOutput(self: *const @This()) !void {
                const span = if (self.shape.n == 0) 0 else 1 + (self.shape.n - 1) * F.step(self.shape.incx);
                for (self.x, 0..) |actual, i| {
                    if (i >= 1 and i <= span and (i - 1) % F.step(self.shape.incx) == 0) {
                        if (comptime F.isComplex(T)) {
                            try std.testing.expectEqual(self.expected[i].re, actual.re);
                            try std.testing.expectEqual(self.expected[i].im, actual.im);
                        } else try std.testing.expectEqual(self.expected[i], actual);
                    } else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&self.expected[i]), std.mem.asBytes(&self.x[i]));
                }
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.a_before), std.mem.sliceAsBytes(self.a));
            }
            fn inspectFree(raw: *const anyopaque) !void {
                const self: *const @This() = @ptrCast(@alignCast(raw));
                try std.testing.expectEqual(self.active, self.ctx.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), self.ctx.snapshot().resource_pins);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.a_before), std.mem.sliceAsBytes(self.a));
                // The staged f32 TPMV frees its inner heap before the outer
                // stack-to-caller copyback; other finite paths commit before free.
                if (self.staged_copyback) {
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(self.x_before), std.mem.sliceAsBytes(self.x));
                } else try self.checkOutput();
            }
            fn finish(raw: *const anyopaque) !void {
                const self: *const @This() = @ptrCast(@alignCast(raw));
                try self.checkOutput();
                try std.testing.expectEqual(self.active, self.ctx.snapshot().active_operations);
                try std.testing.expectEqual(@as(usize, 0), self.ctx.snapshot().resource_pins);
                if (self.busy) try std.testing.expectEqual(@as(u8, 1), fixtureExecutor(self.ctx).io_busy.load(.acquire));
                if (self.scratch) |log| {
                    if (self.scratch_elements != 0) try log.checkScratch(T, self.scratch_elements, self.refused) else try std.testing.expect(!log.snapshot().seen);
                    if (log.free_barrier) |barrier| try std.testing.expect(barrier.completed.load(.acquire));
                }
            }
        };
    }

    fn run(comptime T: type, comptime kind: Kind, comptime Probe: type, ctx: *ResourceGemmContext, shape: Shape, scratch: ?*F.ScratchLog, refused: bool, closing: bool, busy: bool) !void {
        const n: fortran.BlasInt = @intCast(shape.n);
        const k: fortran.BlasInt = @intCast(shape.k);
        const lda_u = if (kind == .band_solve) shape.k + 4 else shape.n + 3;
        const lda: fortran.BlasInt = @intCast(lda_u);
        const count = if (kind == .packed_multiply) shape.n * (shape.n + 1) / 2 else lda_u * shape.n;
        const span = if (shape.n == 0) 0 else 1 + (shape.n - 1) * F.step(shape.incx);
        const buffers = std.testing.allocator;
        const a = try buffers.alloc(T, count + 5);
        defer buffers.free(a);
        const x = try buffers.alloc(T, span + 5);
        defer buffers.free(x);
        @memset(a, F.poison(T));
        @memset(x, F.narrow(T, .{ .re = 101, .im = 103 }));
        var packed_index: usize = 1;
        for (0..shape.n) |col| for (0..shape.n) |row| {
            if ((shape.upper and row > col) or (!shape.upper and row < col)) continue;
            if (kind == .band_solve and @max(row, col) - @min(row, col) > shape.k) continue;
            const index = switch (kind) {
                .dense_multiply, .dense_solve => 1 + row + col * lda_u,
                .packed_multiply => packed_index,
                .band_solve => 1 + (if (shape.upper) shape.k + row - col else row - col) + col * lda_u,
            };
            if (kind == .packed_multiply) packed_index += 1;
            a[index] = if (shape.unit and row == col) F.poison(T) else F.narrow(T, matrix(T, kind, shape, row, col));
        };
        for (0..shape.n) |row| {
            if (comptime !F.isComplex(T)) if (shape.late_overflow) {
                // Analytic diagonal case: overflow only the last visited row,
                // after earlier temporary outputs have already been computed.
                x[F.position(shape.n, shape.incx, row)] = if (isSolve(kind) and row + 1 == shape.n) std.math.floatMax(T) else 2;
                continue;
            };
            var product: Wide = .{ .re = 0 };
            for (0..shape.n) |col| product = F.add(product, F.mul(transformed(T, kind, shape, row, col), solution(T, col)));
            x[F.position(shape.n, shape.incx, row)] = F.narrow(T, if (isSolve(kind)) product else solution(T, row));
        }
        const a_before = try buffers.dupe(T, a);
        defer buffers.free(a_before);
        const x_before = try buffers.dupe(T, x);
        defer buffers.free(x_before);
        const expected = try buffers.dupe(T, x);
        defer buffers.free(expected);
        for (0..shape.n) |row| {
            if (comptime !F.isComplex(T)) if (shape.late_overflow) {
                expected[F.position(shape.n, shape.incx, row)] = if (row + 1 == shape.n) std.math.inf(T) else if (isSolve(kind)) 1 else 4;
                continue;
            };
            var product: Wide = .{ .re = 0 };
            for (0..shape.n) |col| product = F.add(product, F.mul(transformed(T, kind, shape, row, col), solution(T, col)));
            expected[F.position(shape.n, shape.incx, row)] = F.narrow(T, if (isSolve(kind)) solution(T, row) else product);
        }
        const active_before = ctx.snapshot().active_operations;
        try std.testing.expectEqual(@as(usize, if (busy) 1 else 0), active_before);
        const entries = Probe.entries.load(.acquire);
        const finishes = Probe.finishes.load(.acquire);
        var operation = try ctx.begin();
        var ended = false;
        defer if (!ended) operation.end();
        Probe.disarm();
        if (scratch) |log| log.armWithActive(active_before + 1);
        defer if (scratch) |log| log.disarm();
        const observation: Observation(T) = .{
            .ctx = ctx,
            .shape = shape,
            .a = a,
            .a_before = a_before,
            .x = x,
            .x_before = x_before,
            .expected = expected,
            .scratch = scratch,
            .scratch_elements = scratchElements(T, kind, shape),
            .refused = refused,
            .staged_copyback = staged(T, kind, shape),
            .active = active_before + 1,
            .busy = busy,
        };
        Probe.watch(&observation, Observation(T).finish);
        defer Probe.unwatch();
        const barrier = if (scratch) |log| log.free_barrier else null;
        if (barrier) |b| {
            b.observation = &observation;
            b.inspect_fn = Observation(T).inspectFree;
        }
        defer if (barrier) |b| {
            b.observation = null;
            b.inspect_fn = null;
        };
        if (closing) {
            try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
            try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
            try std.testing.expectError(error.Closing, ctx.begin());
        }
        const u = [_]u8{if (shape.upper) 'U' else 'L'};
        const t = [_]u8{shape.trans};
        const d = [_]u8{if (shape.unit) 'U' else 'N'};
        const symbol = comptime switch (kind) {
            .dense_multiply => if (T == f32) "strmv_" else if (T == f64) "dtrmv_" else if (T == fortran.ComplexF32) "ctrmv_" else "ztrmv_",
            .dense_solve => if (T == f32) "strsv_" else if (T == f64) "dtrsv_" else if (T == fortran.ComplexF32) "ctrsv_" else "ztrsv_",
            .packed_multiply => if (T == f32) "stpmv_" else if (T == f64) "dtpmv_" else if (T == fortran.ComplexF32) "ctpmv_" else "ztpmv_",
            .band_solve => if (T == f32) "stbsv_" else if (T == f64) "dtbsv_" else if (T == fortran.ComplexF32) "ctbsv_" else "ztbsv_",
        };
        switch (kind) {
            .dense_multiply, .dense_solve => @field(fortran.internal_admitted, symbol)(&operation, &u, &t, &d, &n, a.ptr + 1, &lda, x.ptr + 1, &shape.incx),
            .packed_multiply => @field(fortran.internal_admitted, symbol)(&operation, &u, &t, &d, &n, a.ptr + 1, x.ptr + 1, &shape.incx),
            .band_solve => @field(fortran.internal_admitted, symbol)(&operation, &u, &t, &d, &n, &k, a.ptr + 1, &lda, x.ptr + 1, &shape.incx),
        }
        observation.checkOutput() catch |err| {
            std.debug.print("triangular resource case {s}/{s} n={d} k={d} upper={} trans={c} unit={} incx={d} late_overflow={}\n", .{ @typeName(T), @tagName(kind), shape.n, shape.k, shape.upper, shape.trans, shape.unit, shape.incx, shape.late_overflow });
            return err;
        };
        try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
        try std.testing.expectEqual(entries + 1, Probe.entries.load(.acquire));
        try std.testing.expectEqual(finishes, Probe.finishes.load(.acquire));
        if (scratch) |log| {
            const elements = scratchElements(T, kind, shape);
            if (elements != 0) try log.checkScratch(T, elements, refused) else try std.testing.expect(!log.snapshot().seen);
        }
        operation.end();
        ended = true;
        try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
        try std.testing.expectEqual(finishes + 1, Probe.finishes.load(.acquire));
    }
};

test "borrowed triangular MV sixteen entries retain independent dense results and full storage" {
    @setEvalBranchQuota(20000);
    const F = TriangularMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(1);
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ f32, f64, fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ F.Kind.dense_multiply, F.Kind.dense_solve, F.Kind.packed_multiply, F.Kind.band_solve }) |kind| {
            for ([_]bool{ true, false }) |upper| for ([_]u8{ 'N', 'T', 'C' }) |trans| for ([_]bool{ false, true }) |unit| for ([_]fortran.BlasInt{ 1, 2, -2 }) |incx| {
                try F.run(T, kind, Probe, ctx, .{ .upper = upper, .trans = trans, .unit = unit, .incx = incx }, null, false, false, false);
            };
            try F.run(T, kind, Probe, ctx, .{ .n = 0, .k = 0, .incx = -2 }, null, false, false, false);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

fn runTriangularScratchCases(comptime limit: usize, refused: bool) !void {
    const F = TriangularMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(limit);
    var allocations: ResourceGemmAllocations = .{ .budget = if (refused) 0 else std.math.maxInt(usize) };
    var scratch: SymmetricMvResourceFixture.ScratchLog = .{ .backing = &allocations };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(scratch.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    scratch.context = ctx;
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var expected_attempts: usize = 0;
    inline for (.{ f32, f64 }) |T| {
        for ([_]bool{ true, false }) |upper| {
            inline for (.{ F.Kind.packed_multiply, F.Kind.band_solve }) |kind| {
                const shape: F.Shape = if (kind == .packed_multiply)
                    .{ .n = 129, .upper = upper, .trans = 'T', .incx = -2 }
                else
                    .{ .n = 128, .k = 1, .upper = upper, .trans = 'N', .incx = -2 };
                const before = allocations.stats();
                try F.run(T, kind, Probe, ctx, shape, &scratch, refused, false, false);
                const increment: usize = if (F.scratchElements(T, kind, shape) == 0) 0 else 1;
                expected_attempts += increment;
                const after = allocations.stats();
                try std.testing.expectEqual(before.attempts + increment, after.attempts);
                try std.testing.expectEqual(before.rejected + (if (refused) increment else 0), after.rejected);
                try std.testing.expectEqual(before.accepted + (if (refused) 0 else increment), after.accepted);
                try std.testing.expectEqual(before.freed + (if (refused) 0 else increment), after.freed);
            }
            // Preserve the real stack-output and no-offdiagonal unit shortcuts.
            try F.run(T, .packed_multiply, Probe, ctx, .{ .n = 128, .upper = upper, .unit = true }, &scratch, false, false, false);
            try F.run(T, .band_solve, Probe, ctx, .{ .n = 128, .k = 0, .upper = upper, .unit = true }, &scratch, false, false, false);
        }
        inline for (.{ F.Kind.packed_multiply, F.Kind.band_solve }) |kind| {
            // The effective upper multiply / lower solve overflows last. Any
            // early caller write before refusal would change the fallback result.
            const shape: F.Shape = .{ .n = if (kind == .packed_multiply) 129 else 128, .k = 1, .upper = kind == .packed_multiply, .late_overflow = true };
            try F.run(T, kind, Probe, ctx, shape, &scratch, refused, false, false);
            expected_attempts += if (F.scratchElements(T, kind, shape) == 0) @as(usize, 0) else 1;
        }
    }
    try std.testing.expectEqual(expected_attempts, allocations.stats().attempts);
    try std.testing.expectEqual(if (refused) expected_attempts else @as(usize, 0), allocations.stats().rejected);
    try std.testing.expectEqual(if (refused) @as(usize, 0) else expected_attempts, allocations.stats().accepted);
    try std.testing.expectEqual(if (refused) @as(usize, 0) else expected_attempts, allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

test "borrowed triangular packed and band scratch use each captured allocator and preserve refusal fallback" {
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ 1, 2 }) |limit| {
        try runTriangularScratchCases(limit, false);
        try runTriangularScratchCases(limit, true);
        try guard.check();
    }
}

fn runTriangularPausedFree(comptime T: type, comptime kind: TriangularMvResourceFixture.Kind, shape: TriangularMvResourceFixture.Shape) !void {
    const F = TriangularMvResourceFixture;
    const S = SymmetricMvResourceFixture;
    const Probe = SymmetricMvLifecycleProbe(1);
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var barrier: S.FreeBarrier = .{};
    var scratch: S.ScratchLog = .{ .backing = &allocations, .free_barrier = if (F.apple_finite) &barrier else null };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(scratch.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    scratch.context = ctx;
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    if (comptime !F.apple_finite) {
        // Execute numerical/closing coverage, without pretending a heap free ran.
        try F.run(T, kind, Probe, ctx, shape, &scratch, false, true, false);
        try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
        try finishResourceGemmContext(Probe, ctx, &allocations);
        return;
    }
    const Work = struct {
        ctx: *ResourceGemmContext,
        scratch: *S.ScratchLog,
        shape: F.Shape,
        finished: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            F.run(T, kind, Probe, self.ctx, self.shape, self.scratch, false, false, false) catch |err| {
                self.failure = err;
            };
        }
    };
    var work: Work = .{ .ctx = ctx, .scratch = &scratch, .shape = shape };
    const thread = try std.Thread.spawn(.{}, Work.run, .{&work});
    var joined = false;
    defer {
        barrier.allow_free.store(true, .release);
        if (!joined) thread.join();
    }
    while (!barrier.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(barrier.entered.load(.acquire));
    try std.testing.expect(!work.finished.load(.acquire));
    try std.testing.expect(!barrier.completed.load(.acquire));
    try barrier.inspect_fn.?(barrier.observation.?);
    const pending = scratch.snapshot();
    try std.testing.expect(pending.seen and pending.accepted and !pending.freed);
    try std.testing.expectEqual(shape.n * @sizeOf(T), pending.bytes);
    try std.testing.expectEqual(@alignOf(T), pending.alignment);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    try std.testing.expectError(error.Closing, ctx.begin());
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().accepted - allocations.stats().freed);
    barrier.allow_free.store(true, .release);
    thread.join();
    joined = true;
    if (work.failure) |err| return err;
    try std.testing.expect(barrier.completed.load(.acquire));
    try scratch.checkScratch(T, shape.n, false);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

test "borrowed triangular scratch free and staged copyback remain inside the closing operation" {
    const F = TriangularMvResourceFixture;
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    try runTriangularPausedFree(f32, F.Kind.packed_multiply, .{ .n = 129, .upper = false, .trans = 'T', .incx = -2 });
    try runTriangularPausedFree(f64, F.Kind.band_solve, .{ .n = 128, .k = 1, .upper = false, .incx = 2 });
    try guard.check();
}

fn TriangularAxpyProbe(comptime limit: usize) type {
    return struct {
        const Base = ResourceGemmProbe(limit);
        var caller: std.Thread.Id = undefined;
        var foreign_helpers: std.atomic.Value(usize) = .init(0);
        var caller_helpers: std.atomic.Value(usize) = .init(0);
        fn configure(index: ?usize) void {
            Base.policy.configure_thread(index);
            if (Base.context.?.snapshot().active_operations == 0) @panic("triangular nested task lost its operation");
            if (index != null) {
                if (std.Thread.getCurrentId() == caller) {
                    _ = caller_helpers.fetchAdd(1, .monotonic);
                } else _ = foreign_helpers.fetchAdd(1, .monotonic);
            }
        }
        const policy: admission_execution.owner.Policy = .{
            .max_threads = Base.policy.max_threads,
            .helper_count = Base.policy.helper_count,
            .configure_thread = configure,
            .worker_stack_size = Base.policy.worker_stack_size,
        };
        fn setup(ctx: *ResourceGemmContext, allocations: *ResourceGemmAllocations, active_limit: usize) void {
            Base.setup(ctx, allocations);
            caller = std.Thread.getCurrentId();
            foreign_helpers.store(0, .release);
            caller_helpers.store(0, .release);
            ResourceGemmContext.testing.configure(ctx, active_limit, Base.observe);
        }
        fn disarm() void {
            Base.disarm();
        }
    };
}

fn runTriangularAxpy(comptime Probe: type, ctx: *ResourceGemmContext, n: usize, closing: bool, busy: bool) !void {
    const T = fortran.ComplexF64;
    const Base = Probe.Base;
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(T, n + 6);
    defer buffers.free(a);
    const y = try buffers.alloc(T, n + 6);
    defer buffers.free(y);
    @memset(a, .{ .re = 67, .im = 71 });
    @memset(y, .{ .re = 73, .im = 79 });
    for (0..n) |i| {
        a[i + 1] = .{ .re = @floatFromInt(1 + i % 5), .im = @floatFromInt(2 + i % 3) };
        y[i + 1] = .{ .re = @floatFromInt(3 + i % 7), .im = @floatFromInt(4 + i % 11) };
    }
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const expected = try buffers.dupe(T, y);
    defer buffers.free(expected);
    for (0..n) |i| expected[i + 1] = .{ .re = 3 * a_before[i + 1].re + expected[i + 1].re, .im = 3 * a_before[i + 1].im + expected[i + 1].im };
    const active_before = ctx.snapshot().active_operations;
    try std.testing.expectEqual(@as(usize, if (busy) 1 else 0), active_before);
    const entries = Base.entries.load(.acquire);
    const finishes = Base.finishes.load(.acquire);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    Base.disarm();
    defer Base.disarm();
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    fortran.testing.admitted_triangular.axpyBody(T, &operation, n, .{ .re = 3, .im = 0 }, a.ptr + 1, y.ptr + 1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(y));
    try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(entries + 1, Base.entries.load(.acquire));
    try std.testing.expectEqual(finishes, Base.finishes.load(.acquire));
    Base.actual = std.mem.sliceAsBytes(y);
    Base.expected = std.mem.sliceAsBytes(expected);
    Base.expect_busy = busy;
    Base.watch_armed = true;
    operation.end();
    ended = true;
    try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
    try std.testing.expectEqual(finishes + 1, Base.finishes.load(.acquire));
}

fn runTriangularAxpyIdle(comptime limit: usize, n: usize) !void {
    const Probe = TriangularAxpyProbe(limit);
    const Base = Probe.Base;
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations, 1);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    try runTriangularAxpy(Probe, ctx, n, false, false);
    if (n < 262144) {
        try std.testing.expectEqual(@as(usize, 0), Base.queries.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), Base.configurations.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    } else {
        // Current production tuning accepts this complex general helper. If a
        // future fixed leaf bypasses it, fail this coverage claim for review.
        try std.testing.expect(Base.queries.load(.acquire) > 0);
        if (comptime limit == 1) {
            try std.testing.expectEqual(@as(usize, 0), Base.configurations.load(.acquire));
            try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
        } else {
            try std.testing.expect(Base.configurations.load(.acquire) >= 2);
            try std.testing.expect(Probe.foreign_helpers.load(.acquire) >= 1);
            try std.testing.expectEqual(@as(usize, 0), Probe.caller_helpers.load(.acquire));
            try std.testing.expect(allocations.stats().accepted > 0);
            try std.testing.expect(reference_ctx.executor.io_init_state.load(.acquire) == 2 or reference_ctx.executor.persistent_init_state.load(.acquire) == 2);
            if (reference_ctx.executor.persistent_init_state.load(.acquire) == 2) {
                try std.testing.expectEqual(@as(u32, 2), reference_ctx.executor.persistent_job_count.load(.acquire));
                try std.testing.expectEqual(@as(u32, 1), reference_ctx.executor.persistent_done_count.load(.acquire));
            }
        }
    }
    try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_busy.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Base.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Base.finishes.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

test "borrowed triangular C64 nested AXPY keeps its actual threshold and two Context policies" {
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    inline for (.{ 1, 2 }) |limit| {
        try runTriangularAxpyIdle(limit, 262143);
        try runTriangularAxpyIdle(limit, 262144);
        try guard.check();
    }
}

test "borrowed triangular C64 nested AXPY finishes on its closing Context with a real busy executor" {
    const guard = try ResourceGemmDefaultGuard.init();
    defer guard.deinit();
    const Probe = TriangularAxpyProbe(2);
    const Base = Probe.Base;
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations, 2);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var blocker: ResourceGemmBlockingBatch = .{ .context = ctx };
    const thread = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&blocker});
    var joined = false;
    defer {
        Probe.disarm();
        blocker.release.store(true, .release);
        if (!joined) thread.join();
    }
    while (!blocker.entered.load(.acquire) and !blocker.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(blocker.entered.load(.acquire));
    while (reference_ctx.executor.persistent_done_count.load(.acquire) != 1 and !blocker.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(u32, 1), reference_ctx.executor.persistent_done_count.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
    const task_fn = reference_ctx.executor.persistent_task_fn;
    const tasks = reference_ctx.executor.persistent_tasks;
    const generations = lowerHermitianGenerations(ctx);
    const before = allocations.stats();
    const configurations = Base.configurations.load(.acquire);
    const queries = Base.queries.load(.acquire);
    try runTriangularAxpy(Probe, ctx, 262144, true, true);
    try std.testing.expect(Base.queries.load(.acquire) > queries);
    try std.testing.expectEqual(configurations, Base.configurations.load(.acquire));
    try std.testing.expectEqualDeep(before, allocations.stats());
    try std.testing.expectEqual(generations, lowerHermitianGenerations(ctx));
    try std.testing.expect(reference_ctx.executor.persistent_task_fn == task_fn);
    try std.testing.expect(reference_ctx.executor.persistent_tasks == tasks);
    try std.testing.expectEqual(@as(u32, 2), reference_ctx.executor.persistent_job_count.load(.acquire));
    try std.testing.expect(!blocker.finished.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    try guard.check();
    blocker.release.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(blocker.accepted);
    for (&blocker.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Base.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Base.finishes.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try guard.check();
}

// Rank-update resource tests: independent integer formulas keep arithmetic exact
// while the complete allocation also observes input bytes, holes and padding.
const RankResourceKind = enum { ger, geru, gerc, syr, spr, syr2, spr2, her, hpr, her2, hpr2 };
const RankResourceInteger = struct { re: i64, im: i64 };
const RankResourceShape = struct {
    m: usize = 5,
    n: usize = 7,
    upper: bool = true,
    incx: fortran.BlasInt = 1,
    incy: fortran.BlasInt = 1,
    zero_alpha: bool = false,
    real_coefficient: bool = false,
    closing: bool = false,
    busy: bool = false,
};

fn rankResourceIsGer(comptime kind: RankResourceKind) bool {
    return kind == .ger or kind == .geru or kind == .gerc;
}
fn rankResourceIsPacked(comptime kind: RankResourceKind) bool {
    return kind == .spr or kind == .spr2 or kind == .hpr or kind == .hpr2;
}
fn rankResourceIsHermitian(comptime kind: RankResourceKind) bool {
    return kind == .her or kind == .hpr or kind == .her2 or kind == .hpr2;
}
fn rankResourceValue(comptime T: type, p: RankResourceInteger) T {
    if (T == f32 or T == f64) return @floatFromInt(p.re);
    return .{ .re = @floatFromInt(p.re), .im = @floatFromInt(p.im) };
}
fn rankResourceParts(comptime T: type, value: T) RankResourceInteger {
    if (T == f32 or T == f64) return .{ .re = @intFromFloat(value), .im = 0 };
    return .{ .re = @intFromFloat(value.re), .im = @intFromFloat(value.im) };
}
fn rankResourceAdd(a: RankResourceInteger, b: RankResourceInteger) RankResourceInteger {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn rankResourceMul(a: RankResourceInteger, b: RankResourceInteger) RankResourceInteger {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn rankResourceConj(a: RankResourceInteger) RankResourceInteger {
    return .{ .re = a.re, .im = -a.im };
}
fn rankResourceSpan(n: usize, inc: fortran.BlasInt) usize {
    return if (n == 0) 0 else 1 + (n - 1) * @as(usize, @intCast(@abs(inc)));
}
fn rankResourceIndex(n: usize, inc: fortran.BlasInt, i: usize) usize {
    const step: usize = @intCast(@abs(inc));
    return (if (inc < 0) n - 1 - i else i) * step;
}
fn rankResourceMatrixIndex(comptime kind: RankResourceKind, shape: RankResourceShape, lda: usize, i: usize, j: usize) usize {
    if (!rankResourceIsPacked(kind)) return j * lda + i;
    return if (shape.upper) j * (j + 1) / 2 + i else j * (2 * shape.n - j + 1) / 2 + i - j;
}
fn rankResourceCall(comptime T: type, comptime kind: RankResourceKind, operation: *const admission_execution.context.Operation, shape: RankResourceShape, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: fortran.BlasInt) void {
    const prefix = comptime if (T == f32) "s" else if (T == f64) "d" else if (T == fortran.ComplexF32) "c" else "z";
    const name = comptime prefix ++ @tagName(kind) ++ "_";
    const m: fortran.BlasInt = @intCast(shape.m);
    const n: fortran.BlasInt = @intCast(shape.n);
    const uplo = [_]u8{if (shape.upper) 'U' else 'L'};
    switch (kind) {
        .ger, .geru, .gerc => @field(fortran.internal_admitted, name)(operation, &m, &n, &alpha, x, &shape.incx, y, &shape.incy, a, &lda),
        .syr => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, a, &lda),
        .spr => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, a),
        .syr2 => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, y, &shape.incy, a, &lda),
        .spr2 => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, y, &shape.incy, a),
        .her => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha.re, x, &shape.incx, a, &lda),
        .hpr => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha.re, x, &shape.incx, a),
        .her2 => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, y, &shape.incy, a, &lda),
        .hpr2 => @field(fortran.internal_admitted, name)(operation, &uplo, &n, &alpha, x, &shape.incx, y, &shape.incy, a),
    }
}

fn rankResourceCase(comptime T: type, comptime kind: RankResourceKind, comptime Probe: type, ctx: *ResourceGemmContext, shape: RankResourceShape) !void {
    const real = T == f32 or T == f64;
    const rows = if (rankResourceIsGer(kind)) shape.m else shape.n;
    const lda = rows + 3;
    const x_span = rankResourceSpan(rows, shape.incx);
    const y_span = rankResourceSpan(shape.n, shape.incy);
    const a_span = if (rankResourceIsPacked(kind)) shape.n * (shape.n + 1) / 2 else lda * shape.n;
    const x_offset: usize = 1;
    const y_offset = x_offset + x_span + 2;
    const a_offset = y_offset + y_span + 2;
    const storage = try std.testing.allocator.alloc(T, a_offset + a_span + 2);
    defer std.testing.allocator.free(storage);
    @memset(storage, rankResourceValue(T, .{ .re = 97, .im = 53 }));
    const x = storage[x_offset..][0..x_span];
    const y = storage[y_offset..][0..y_span];
    const a = storage[a_offset..][0..a_span];
    for (0..rows) |i| x[rankResourceIndex(rows, shape.incx, i)] = rankResourceValue(T, .{ .re = 1 + @as(i64, @intCast(i % 5)), .im = @as(i64, @intCast(i % 5)) - 2 });
    for (0..shape.n) |j| y[rankResourceIndex(shape.n, shape.incy, j)] = rankResourceValue(T, .{ .re = 1 + @as(i64, @intCast(j % 4)), .im = if (shape.real_coefficient) 0 else @as(i64, @intCast(j % 3)) - 1 });
    for (0..shape.n) |j| {
        for (0..rows) |i| {
            if (rankResourceIsPacked(kind) and (if (shape.upper) i > j else i < j)) continue;
            a[rankResourceMatrixIndex(kind, shape, lda, i, j)] = rankResourceValue(T, .{ .re = 1000 + @as(i64, @intCast((i + j) % 5)), .im = 2000 + @as(i64, @intCast((i + j) % 7)) });
        }
    }
    const expected = try std.testing.allocator.dupe(T, storage);
    defer std.testing.allocator.free(expected);
    const alpha_parts: RankResourceInteger = .{
        .re = if (shape.zero_alpha) 0 else 2,
        .im = if (real or shape.zero_alpha or shape.real_coefficient or kind == .her or kind == .hpr) 0 else 1,
    };
    const alpha = rankResourceValue(T, alpha_parts);
    if (!shape.zero_alpha) {
        for (0..shape.n) |j| {
            for (0..rows) |i| {
                if (!rankResourceIsGer(kind) and (if (shape.upper) i > j else i < j)) continue;
                const xi = rankResourceParts(T, x[rankResourceIndex(rows, shape.incx, i)]);
                const yj = rankResourceParts(T, y[rankResourceIndex(shape.n, shape.incy, j)]);
                const delta = if (rankResourceIsGer(kind)) rankResourceMul(alpha_parts, rankResourceMul(xi, if (kind == .gerc) rankResourceConj(yj) else yj)) else blk: {
                    const xj = rankResourceParts(T, x[rankResourceIndex(rows, shape.incx, j)]);
                    if (kind == .syr or kind == .spr) break :blk rankResourceMul(alpha_parts, rankResourceMul(xi, xj));
                    if (kind == .her or kind == .hpr) break :blk rankResourceMul(alpha_parts, rankResourceMul(xi, rankResourceConj(xj)));
                    const yi = rankResourceParts(T, y[rankResourceIndex(shape.n, shape.incy, i)]);
                    if (kind == .syr2 or kind == .spr2) break :blk rankResourceMul(alpha_parts, rankResourceAdd(rankResourceMul(xi, yj), rankResourceMul(yi, xj)));
                    break :blk rankResourceAdd(rankResourceMul(alpha_parts, rankResourceMul(xi, rankResourceConj(yj))), rankResourceMul(rankResourceConj(alpha_parts), rankResourceMul(yi, rankResourceConj(xj))));
                };
                const index = a_offset + rankResourceMatrixIndex(kind, shape, lda, i, j);
                var result = rankResourceAdd(rankResourceParts(T, expected[index]), delta);
                if (rankResourceIsHermitian(kind) and i == j) result.im = 0;
                expected[index] = rankResourceValue(T, result);
            }
        }
    }
    const active_before = ctx.snapshot().active_operations;
    try std.testing.expectEqual(@as(usize, if (shape.busy) 1 else 0), active_before);
    const entries_before = Probe.entries.load(.acquire);
    const finishes_before = Probe.finishes.load(.acquire);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) {
        Probe.disarm();
        operation.end();
    };
    Probe.actual = std.mem.sliceAsBytes(storage);
    Probe.expected = std.mem.sliceAsBytes(expected);
    Probe.expect_busy = shape.busy;
    Probe.expected_live_allocations = if (shape.busy) blk: {
        const state = Probe.allocations.?.stats();
        break :blk state.accepted - state.freed;
    } else null;
    Probe.watch_armed = true;
    if (shape.closing) {
        _ = ctx.requestClose();
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    rankResourceCall(T, kind, &operation, shape, alpha, x.ptr, y.ptr, a.ptr, @intCast(lda));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(storage));
    try std.testing.expectEqual(active_before + 1, ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(entries_before + 1, Probe.entries.load(.acquire));
    try std.testing.expectEqual(finishes_before, Probe.finishes.load(.acquire));
    if (shape.closing) try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    operation.end();
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(active_before, ctx.snapshot().active_operations);
    try std.testing.expectEqual(finishes_before + 1, Probe.finishes.load(.acquire));
}

fn rankResourceScheduled(comptime Probe: type, ctx: *ResourceGemmContext, allocations: *ResourceGemmAllocations) !void {
    try std.testing.expect(Probe.queries.load(.acquire) > 0);
    try std.testing.expect(Probe.configurations.load(.acquire) > 0);
    try std.testing.expect(allocations.stats().accepted > 0);
    try std.testing.expect(fixtureExecutor(ctx).io_init_state.load(.acquire) == 2 or fixtureExecutor(ctx).persistent_init_state.load(.acquire) == 2);
    try std.testing.expectEqual(@as(u8, 0), fixtureExecutor(ctx).io_busy.load(.acquire));
}

fn rankResourceOuterPair(comptime T: type, comptime kind: RankResourceKind, shape: RankResourceShape) !void {
    const Single = ResourceGemmProbe(1);
    const Many = ResourceGemmProbe(4);
    var allocation_a: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var allocation_b: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_a: FixtureReferenceContext = undefined;
    reference_a.initInPlace(allocation_a.allocator(), Single.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const a = reference_a.borrowedContext();
    var reference_b: FixtureReferenceContext = undefined;
    reference_b.initInPlace(allocation_b.allocator(), Many.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const b = reference_b.borrowedContext();
    Single.setup(a, &allocation_a);
    Many.setup(b, &allocation_b);
    defer {
        Single.disarm();
        Many.disarm();
        _ = b.requestClose();
        _ = fixtureProgressClose(b);
        _ = a.requestClose();
        _ = fixtureProgressClose(a);
    }
    try std.testing.expect(&reference_a.executor != &reference_b.executor and &reference_a.executor != admission_execution.get());
    try std.testing.expect(reference_a.executor.allocator.ptr == allocation_a.allocator().ptr);
    try std.testing.expect(reference_b.executor.allocator.ptr == allocation_b.allocator().ptr);
    try std.testing.expect(reference_a.executor.policy.max_threads == Single.policy.max_threads);
    try std.testing.expect(reference_b.executor.policy.max_threads == Many.policy.max_threads);
    try rankResourceCase(T, kind, Single, a, shape);
    try std.testing.expectEqual(@as(usize, 0), allocation_a.stats().attempts);
    try std.testing.expectEqual(@as(usize, 0), Single.configurations.load(.acquire));
    const a_state = a.snapshot();
    const a_allocations = allocation_a.stats();
    const a_queries = Single.queries.load(.acquire);
    var closing_shape = shape;
    closing_shape.closing = true;
    try rankResourceCase(T, kind, Many, b, closing_shape);
    try rankResourceScheduled(Many, b, &allocation_b);
    try std.testing.expectEqualDeep(a_state, a.snapshot());
    try std.testing.expectEqualDeep(a_allocations, allocation_a.stats());
    try std.testing.expectEqual(a_queries, Single.queries.load(.acquire));
    try finishResourceGemmContext(Many, b, &allocation_b);
    try finishResourceGemmContext(Single, a, &allocation_a);
}

fn rankResourceNestedCase(comptime T: type, m: usize, real_coefficient: bool, expect_schedule: bool) !void {
    const Probe = ResourceGemmProbe(4);
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    // One column leaves outer GER serial. Only its real nested AXPY can submit.
    try rankResourceCase(T, .geru, Probe, ctx, .{ .m = m, .n = 1, .real_coefficient = real_coefficient });
    if (expect_schedule) {
        try rankResourceScheduled(Probe, ctx, &allocations);
    } else {
        try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
        try std.testing.expectEqual(@as(usize, 0), Probe.configurations.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_init_state.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.persistent_init_state.load(.acquire));
    }
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

fn rankResourceBusyClosing(comptime T: type, comptime kind: RankResourceKind, original_shape: RankResourceShape) !void {
    const Probe = ResourceGemmProbe(4);
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    var work: ResourceGemmBlockingBatch = .{ .context = ctx };
    const caller = try std.Thread.spawn(.{}, ResourceGemmBlockingBatch.run, .{&work});
    var joined = false;
    defer {
        Probe.disarm();
        work.release.store(true, .release);
        if (!joined) caller.join();
    }
    while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(work.entered.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(u8, 1), reference_ctx.executor.io_busy.load(.acquire));
    const allocation_state = allocations.stats();
    const configurations = Probe.configurations.load(.acquire);
    const queries = Probe.queries.load(.acquire);
    var shape = original_shape;
    shape.busy = true;
    shape.closing = true;
    try rankResourceCase(T, kind, Probe, ctx, shape);
    try std.testing.expect(Probe.queries.load(.acquire) > queries);
    try std.testing.expectEqualDeep(allocation_state, allocations.stats());
    try std.testing.expectEqual(configurations, Probe.configurations.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expect(!work.finished.load(.acquire));
    try std.testing.expect(reference_ctx.executor.persistent_task_fn == ResourceGemmBlockingBatch.execute);
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    // Release/join is mandatory even if an earlier assertion fails. No plain
    // worker-written field is read before this acquire/join boundary.
    Probe.disarm();
    work.release.store(true, .release);
    caller.join();
    joined = true;
    try std.testing.expect(work.accepted);
    for (&work.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Probe.finishes.load(.acquire));
    try finishResourceGemmContext(Probe, ctx, &allocations);
}

fn rankResourceOom(comptime T: type, comptime kind: RankResourceKind, shape: RankResourceShape) !void {
    const Probe = ResourceGemmProbe(4);
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    try rankResourceCase(T, kind, Probe, ctx, shape);
    const state = allocations.stats();
    try std.testing.expect(state.attempts > 0 and state.rejected > 0);
    try std.testing.expectEqual(@as(usize, 0), state.accepted);
    try std.testing.expectEqual(state.attempts, state.rejected);
    // Failed concurrent submissions run their exact jobs on the caller. The
    // ordinary group still configures that caller once; it must not be confused
    // with a successfully created helper or an algorithm-level false return.
    try std.testing.expectEqual(@as(usize, 1), Probe.configurations.load(.acquire));
    try std.testing.expectEqual(@as(u8, 2), reference_ctx.executor.io_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), reference_ctx.executor.io_busy.load(.acquire));
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

// There are 22 concrete names: s/d GER,SYR,SPR,SYR2,SPR2, and c/z
// GERU,GERC,HER,HPR,HER2,HPR2. These tests call the borrowed adapters directly.
test "borrowed rank updates cover all twenty two mappings with independent full storage oracles" {
    const Probe = ResourceGemmProbe(1);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    inline for (.{ f32, f64 }) |T| {
        for ([_]fortran.BlasInt{ 1, -2 }) |inc| try rankResourceCase(T, .ger, Probe, ctx, .{ .incx = inc, .incy = if (inc == 1) 1 else 2 });
        try rankResourceCase(T, .ger, Probe, ctx, .{ .zero_alpha = true });
        try rankResourceCase(T, .ger, Probe, ctx, .{ .m = 0 });
        try rankResourceCase(T, .ger, Probe, ctx, .{ .n = 0 });
        inline for (.{ RankResourceKind.syr, .spr, .syr2, .spr2 }) |kind| {
            for ([_]bool{ true, false }) |upper| {
                for ([_]fortran.BlasInt{ 1, -2 }) |inc| try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .incx = inc, .incy = if (inc == 1) 1 else 2 });
                try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .zero_alpha = true });
                try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .n = 0 });
            }
        }
    }
    inline for (.{ fortran.ComplexF32, fortran.ComplexF64 }) |T| {
        inline for (.{ RankResourceKind.geru, .gerc }) |kind| {
            for ([_]fortran.BlasInt{ 1, -2 }) |inc| try rankResourceCase(T, kind, Probe, ctx, .{ .incx = inc, .incy = if (inc == 1) 1 else 2 });
            try rankResourceCase(T, kind, Probe, ctx, .{ .zero_alpha = true });
            try rankResourceCase(T, kind, Probe, ctx, .{ .m = 0 });
            try rankResourceCase(T, kind, Probe, ctx, .{ .n = 0 });
        }
        inline for (.{ RankResourceKind.her, .hpr, .her2, .hpr2 }) |kind| {
            for ([_]bool{ true, false }) |upper| {
                for ([_]fortran.BlasInt{ 1, -2 }) |inc| try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .incx = inc, .incy = if (inc == 1) 1 else 2 });
                try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .zero_alpha = true });
                try rankResourceCase(T, kind, Probe, ctx, .{ .upper = upper, .n = 0 });
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    try finishResourceGemmContext(Probe, ctx, &allocations);
    try default_guard.check();
}

test "borrowed rank updates use four actual outer scheduler families with two distinct Context owners" {
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    try rankResourceOuterPair(f64, .ger, .{ .m = 129, .n = 512 });
    try default_guard.check();
    try rankResourceOuterPair(fortran.ComplexF64, .gerc, .{ .m = 128, .n = 128 });
    try default_guard.check();
    try rankResourceOuterPair(fortran.ComplexF64, .her2, .{ .n = 513, .upper = false });
    try default_guard.check();
    // The public packed gate is 2048, not the private worker helper's 512.
    try rankResourceOuterPair(fortran.ComplexF32, .hpr2, .{ .n = 2048 });
    try default_guard.check();
}

test "borrowed single column complex GER reaches real nested AXPY thresholds without square storage" {
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    try rankResourceNestedCase(fortran.ComplexF64, 262143, true, false);
    try default_guard.check();
    try rankResourceNestedCase(fortran.ComplexF64, 262144, true, true);
    try default_guard.check();
    try rankResourceNestedCase(fortran.ComplexF32, 1048576, true, true);
    try default_guard.check();
    const complex_parallel = @import("builtin").target.cpu.arch == .aarch64 or @import("builtin").target.cpu.arch == .x86_64;
    try rankResourceNestedCase(fortran.ComplexF32, 524288, false, complex_parallel);
    try default_guard.check();
}

test "borrowed rank updates retain the closing owner through real busy outer and nested fallback" {
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    try rankResourceBusyClosing(fortran.ComplexF64, .her2, .{ .n = 513, .upper = false });
    try default_guard.check();
    try rankResourceBusyClosing(fortran.ComplexF32, .geru, .{ .m = 524288, .n = 1 });
    try default_guard.check();
}

test "borrowed rank updates reject captured Executor allocations and drain after numerical fallback" {
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    try rankResourceOom(f64, .ger, .{ .m = 129, .n = 512 });
    try default_guard.check();
    try rankResourceOom(fortran.ComplexF64, .her2, .{ .n = 513 });
    try default_guard.check();
}

fn level3BorrowedSyrkFailure(closing: bool) !void {
    const Probe = ResourceGemmProbe(4);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = if (closing) std.math.maxInt(usize) else 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    const n: fortran.BlasInt = 129;
    const k: fortran.BlasInt = 8;
    const lda: fortran.BlasInt = 131;
    const ldc: fortran.BlasInt = 132;
    const a = try std.testing.allocator.alloc(f64, lda * k);
    defer std.testing.allocator.free(a);
    const c = try std.testing.allocator.alloc(f64, ldc * n);
    defer std.testing.allocator.free(c);
    const expected = try std.testing.allocator.alloc(f64, c.len);
    defer std.testing.allocator.free(expected);
    @memset(a, 701);
    @memset(c, 907);
    @memset(expected, 907);
    for (0..k) |column| for (0..n) |row| {
        a[column * lda + row] = @as(f64, @floatFromInt((row + 3 * column) % 11)) - 5;
    };
    for (0..n) |column| for (0..column + 1) |row| {
        var sum: f64 = 0;
        for (0..k) |inner| sum += a[inner * lda + row] * a[inner * lda + column];
        expected[column * ldc + row] = 2 * sum;
        c[column * ldc + row] = std.math.nan(f64);
    };
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    const upper = [1]u8{'U'};
    const normal = [1]u8{'N'};
    const alpha: f64 = 2;
    const beta: f64 = 0;
    fortran.internal_admitted.dsyrk_(&operation, &upper, &normal, &n, &k, &alpha, a.ptr, &lda, &beta, c.ptr, &ldc);
    try std.testing.expectEqualSlices(f64, expected, c);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    if (!closing) try std.testing.expect(allocations.stats().rejected > 0);
    try default_guard.check();
    operation.end();
    ended = true;
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

test "borrowed Level3 SYRK allocator refusal preserves stored triangle and default owner" {
    try level3BorrowedSyrkFailure(false);
}

test "borrowed Level3 SYRK finishes inside its closing operation" {
    try level3BorrowedSyrkFailure(true);
}

fn level3TriangularClosing(right: bool, solve: bool) !void {
    const Probe = ResourceGemmProbe(4);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    const m: fortran.BlasInt = if (right) 129 else 257;
    const n: fortran.BlasInt = if (right) 257 else 129;
    const order: usize = 257;
    const lda: fortran.BlasInt = 259;
    const ldb: fortran.BlasInt = m + 3;
    const a = try std.testing.allocator.alloc(f64, lda * order);
    defer std.testing.allocator.free(a);
    const b = try std.testing.allocator.alloc(f64, @as(usize, @intCast(ldb)) * @as(usize, @intCast(n)));
    defer std.testing.allocator.free(b);
    const expected = try std.testing.allocator.alloc(f64, b.len);
    defer std.testing.allocator.free(expected);
    @memset(a, std.math.nan(f64));
    for (0..order) |column| for (column..order) |row| {
        a[column * lda + row] = if (row == column or row == column + 1) 1 else 0;
    };
    @memset(b, -707);
    @memset(expected, -707);
    for (0..@intCast(n)) |column| for (0..@intCast(m)) |row| {
        const x = @as(f64, @floatFromInt((row + 2 * column) % 7)) - 3;
        const neighbor = if (right)
            (if (column + 1 < order) @as(f64, @floatFromInt((row + 2 * (column + 1)) % 7)) - 3 else 0)
        else
            (if (row > 0) @as(f64, @floatFromInt((row - 1 + 2 * column) % 7)) - 3 else 0);
        const index = column * @as(usize, @intCast(ldb)) + row;
        b[index] = if (solve) x + neighbor else x;
        expected[index] = if (solve) x else x + neighbor;
    };
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    const side = [1]u8{if (right) 'R' else 'L'};
    const lower = [1]u8{'L'};
    const normal = [1]u8{'N'};
    const diagonal = [1]u8{'N'};
    const alpha: f64 = 1;
    if (solve) {
        fortran.internal_admitted.dtrsm_(&operation, &side, &lower, &normal, &diagonal, &m, &n, &alpha, a.ptr, &lda, b.ptr, &ldb);
    } else {
        fortran.internal_admitted.dtrmm_(&operation, &side, &lower, &normal, &diagonal, &m, &n, &alpha, a.ptr, &lda, b.ptr, &ldb);
    }
    try std.testing.expectEqualSlices(f64, expected, b);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    try default_guard.check();
    operation.end();
    ended = true;
    Probe.disarm();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

test "borrowed Level3 triangular operations finish on closing owner for both sides" {
    for ([_]bool{ false, true }) |right| {
        try level3TriangularClosing(right, false);
        try level3TriangularClosing(right, true);
    }
}

fn level3BorrowedSyr2kFailure(closing: bool) !void {
    const Probe = ResourceGemmProbe(4);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = if (closing) std.math.maxInt(usize) else 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    const n: fortran.BlasInt = 129;
    const k: fortran.BlasInt = 8;
    const lda: fortran.BlasInt = 131;
    const ldc: fortran.BlasInt = 132;
    const a = try std.testing.allocator.alloc(f64, lda * k);
    defer std.testing.allocator.free(a);
    const b = try std.testing.allocator.alloc(f64, lda * k);
    defer std.testing.allocator.free(b);
    @memset(b, 809);
    const c = try std.testing.allocator.alloc(f64, ldc * n);
    defer std.testing.allocator.free(c);
    const expected = try std.testing.allocator.alloc(f64, c.len);
    defer std.testing.allocator.free(expected);
    @memset(a, 701);
    @memset(c, 907);
    @memset(expected, 907);
    for (0..k) |column| for (0..n) |row| {
        a[column * lda + row] = @as(f64, @floatFromInt((row + 3 * column) % 11)) - 5;
        b[column * lda + row] = @as(f64, @floatFromInt((2 * row + column) % 13)) - 6;
    };
    for (0..n) |column| for (0..column + 1) |row| {
        var sum: f64 = 0;
        for (0..k) |inner| sum += a[inner * lda + row] * b[inner * lda + column] + b[inner * lda + row] * a[inner * lda + column];
        expected[column * ldc + row] = 2 * sum;
        c[column * ldc + row] = std.math.nan(f64);
    };
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
        try std.testing.expectError(error.Closing, ctx.begin());
    }
    const upper = [1]u8{'U'};
    const normal = [1]u8{'N'};
    const alpha: f64 = 2;
    const beta: f64 = 0;
    fortran.internal_admitted.dsyr2k_(&operation, &upper, &normal, &n, &k, &alpha, a.ptr, &lda, b.ptr, &lda, &beta, c.ptr, &ldc);
    try std.testing.expectEqualSlices(f64, expected, c);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    if (!closing) try std.testing.expect(allocations.stats().rejected > 0);
    try default_guard.check();
    operation.end();
    ended = true;
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

test "borrowed Level3 SYR2K allocator refusal preserves stored triangle and default owner" {
    try level3BorrowedSyr2kFailure(false);
}

test "borrowed Level3 SYR2K finishes inside its closing operation" {
    try level3BorrowedSyr2kFailure(true);
}

fn level3ComplexSymmetricResources(right: bool, herm: bool, closing: bool) !void {
    const T = fortran.ComplexF64;
    const Probe = ResourceGemmProbe(4);
    const default_guard = try ResourceGemmDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: ResourceGemmAllocations = .{ .budget = 0 };
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), Probe.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Probe.setup(ctx, &allocations);
    defer {
        Probe.disarm();
        _ = ctx.requestClose();
        _ = fixtureProgressClose(ctx);
    }
    const m: fortran.BlasInt = if (right) 129 else 257;
    const n: fortran.BlasInt = if (right) 257 else 129;
    const order: usize = 257;
    const lda: fortran.BlasInt = 259;
    const ldb: fortran.BlasInt = m + 3;
    const a = try std.testing.allocator.alloc(T, lda * order);
    defer std.testing.allocator.free(a);
    const b = try std.testing.allocator.alloc(T, @as(usize, @intCast(ldb)) * @as(usize, @intCast(n)));
    defer std.testing.allocator.free(b);
    const c = try std.testing.allocator.alloc(T, b.len);
    defer std.testing.allocator.free(c);
    const expected = try std.testing.allocator.alloc(T, b.len);
    defer std.testing.allocator.free(expected);
    const poison = complexValue(T, std.math.nan(f64), std.math.nan(f64));
    const sentinel = complexValue(T, -707, 809);
    @memset(a, poison);
    for (0..order) |column| for (column..order) |row| {
        a[column * lda + row] = if (row == column) complexValue(T, 3, 0) else if (row == column + 1) complexValue(T, 1, 2) else complexValue(T, 0, 0);
    };
    @memset(b, sentinel);
    @memset(c, sentinel);
    @memset(expected, sentinel);
    for (0..@intCast(n)) |column| for (0..@intCast(m)) |row| {
        b[column * @as(usize, @intCast(ldb)) + row] = complexValue(T, @as(f64, @floatFromInt((row + 2 * column) % 7)) - 3, @as(f64, @floatFromInt((2 * row + column) % 5)) - 2);
    };
    // Independent tridiagonal oracle: only the diagonal and its two neighbours contribute.
    for (0..@intCast(n)) |column| for (0..@intCast(m)) |row| {
        const stride: usize = @intCast(ldb);
        const index = column * stride + row;
        const axis = if (right) column else row;
        const delta: usize = if (right) stride else 1;
        const lower = complexValue(T, 1, 2);
        const upper = complexValue(T, 1, if (herm) -2 else 2);
        var value = complexMul(T, complexValue(T, 3, 0), b[index]);
        if (axis > 0) value = complexAdd(T, value, complexMul(T, if (right) upper else lower, b[index - delta]));
        if (axis + 1 < order) value = complexAdd(T, value, complexMul(T, if (right) lower else upper, b[index + delta]));
        expected[index] = value;
        c[index] = poison;
    };
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    if (closing) {
        try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(admission_execution.context.CloseResult.busy, fixtureProgressClose(ctx));
    }
    const side = [1]u8{if (right) 'R' else 'L'};
    const lower = [1]u8{'L'};
    const alpha = complexValue(T, 1, 0);
    const beta = complexValue(T, 0, 0);
    if (herm) {
        fortran.internal_admitted.zhemm_(&operation, &side, &lower, &m, &n, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldb);
    } else {
        fortran.internal_admitted.zsymm_(&operation, &side, &lower, &m, &n, &alpha, a.ptr, &lda, b.ptr, &ldb, &beta, c.ptr, &ldb);
    }
    try std.testing.expectEqualSlices(T, expected, c);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), Probe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), Probe.finishes.load(.acquire));
    if (!closing) try std.testing.expect(allocations.stats().rejected > 0);
    try default_guard.check();
    operation.end();
    ended = true;
    try std.testing.expectEqual(@as(usize, 1), Probe.finishes.load(.acquire));
    Probe.disarm();
    _ = ctx.requestClose();
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}

test "borrowed complex SYMM and HEMM preserve both sides under allocator refusal and closing" {
    for ([_]bool{ false, true }) |right| {
        for ([_]bool{ false, true }) |herm| {
            try level3ComplexSymmetricResources(right, herm, false);
            try level3ComplexSymmetricResources(right, herm, true);
        }
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
fn fixtureExecutor(ctx: *@import("zynum-execution").Context) *@import("zynum-execution").reference.owner.Executor {
    const fixture_execution = @import("zynum-execution");
    if (ctx == fixture_execution.reference.getContext()) return @ptrCast(@alignCast(ctx.binding.legacy.userdata));
    const wrapper: *FixtureReferenceContext = @fieldParentPtr("context", ctx);
    return &wrapper.executor;
}
