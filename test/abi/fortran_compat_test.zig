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

fn referenceGemmTransposedA(comptime T: type, transposed_b: bool, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: fortran.BlasInt, b: []const T, ldb: fortran.BlasInt, beta: T, c: []T, ldc: fortran.BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..k) |p| {
                const bv = if (transposed_b) b[matIndex(ldb, j, p)] else b[matIndex(ldb, p, j)];
                sum = @mulAdd(T, a[matIndex(lda, p, i)], bv, sum);
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = @mulAdd(T, alpha, sum, if (beta == 0) 0 else beta * c[idxc]);
        }
    }
}

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
