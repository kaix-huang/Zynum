// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const std = @import("std");
const core = @import("zynum-core").internal.blas_unchecked;
const execution = @import("zynum-core").internal.numerics.Execution;
pub const runtime = @import("zynum-core").blas.runtime;

pub const BlasInt = core.BlasInt;
pub const ComplexF32 = core.ComplexF32;
pub const ComplexF64 = core.ComplexF64;
pub const testing = if (builtin.is_test) core.matrix_vector.testing else struct {};

const default_xerbla_name_len = 6;
const max_xerbla_array_name_len = 64;

pub export fn lsame_(ca: [*]const u8, cb: [*]const u8) callconv(.c) c_int {
    return if (std.ascii.toUpper(ca[0]) == std.ascii.toUpper(cb[0])) 1 else 0;
}

pub export fn xerbla_(srname: [*]const u8, info: *const BlasInt) callconv(.c) void {
    reportXerbla(trimFortranName(srname, default_xerbla_name_len), info.*);
}

pub export fn xerbla_array_(srname_array: [*]const u8, srname_len: *const BlasInt, info: *const BlasInt) callconv(.c) void {
    const name_len: usize = if (srname_len.* <= 0) 0 else @min(@as(usize, @intCast(srname_len.*)), max_xerbla_array_name_len);
    reportXerbla(trimFortranName(srname_array, name_len), info.*);
}

fn trimFortranName(name: [*]const u8, max_len: usize) []const u8 {
    var len: usize = 0;
    while (len < max_len and name[len] != 0) : (len += 1) {}
    return std.mem.trim(u8, name[0..len], " ");
}

fn reportXerbla(name: []const u8, info: BlasInt) void {
    if (builtin.is_test) return;
    const routine_name = if (name.len == 0) "UNKNOWN" else name;
    std.debug.print(" ** On entry to {s} parameter number {d} had an illegal value\n", .{ routine_name, info });
}

pub export fn zynum_blas_shutdown() callconv(.c) void {
    core.shutdown();
}

pub export fn zynum_blas_shutdown_() callconv(.c) void {
    core.shutdown();
}

test "xerbla routine names are trimmed" {
    const padded = [_]u8{ 'D', 'G', 'E', 'M', 'M', ' ' };
    const nul_terminated = [_]u8{ 'D', 'G', 'E', 'M', 'M', 0, 'X' };
    try std.testing.expectEqualStrings("DGEMM", trimFortranName(&padded, padded.len));
    try std.testing.expectEqualStrings("DGEMM", trimFortranName(&nul_terminated, nul_terminated.len));
}

fn validTrans(p: [*]const u8) bool {
    return switch (core.fromChar(p)) {
        'N', 'T', 'C' => true,
        else => false,
    };
}

fn validUplo(p: [*]const u8) bool {
    return switch (core.fromChar(p)) {
        'U', 'L' => true,
        else => false,
    };
}

fn validDiag(p: [*]const u8) bool {
    return switch (core.fromChar(p)) {
        'N', 'U' => true,
        else => false,
    };
}

fn validSide(p: [*]const u8) bool {
    return switch (core.fromChar(p)) {
        'L', 'R' => true,
        else => false,
    };
}

fn max1(x: BlasInt) BlasInt {
    return @max(@as(BlasInt, 1), x);
}

fn gemmError(ta: [*]const u8, tb: [*]const u8, m: BlasInt, n: BlasInt, k: BlasInt, lda: BlasInt, ldb: BlasInt, ldc: BlasInt) BlasInt {
    if (!validTrans(ta)) return 1;
    if (!validTrans(tb)) return 2;
    if (m < 0) return 3;
    if (n < 0) return 4;
    if (k < 0) return 5;

    const nota = core.fromChar(ta) == 'N';
    const notb = core.fromChar(tb) == 'N';
    const nrowa = if (nota) m else k;
    const nrowb = if (notb) k else n;
    if (lda < max1(nrowa)) return 8;
    if (ldb < max1(nrowb)) return 10;
    if (ldc < max1(m)) return 13;
    return 0;
}

fn triangularMatrixError(side: [*]const u8, uplo: [*]const u8, trans: [*]const u8, diag: [*]const u8, m: BlasInt, n: BlasInt, lda: BlasInt, ldb: BlasInt) BlasInt {
    if (!validSide(side)) return 1;
    if (!validUplo(uplo)) return 2;
    if (!validTrans(trans)) return 3;
    if (!validDiag(diag)) return 4;
    if (m < 0) return 5;
    if (n < 0) return 6;

    const nrowa = if (core.fromChar(side) == 'L') m else n;
    if (lda < max1(nrowa)) return 9;
    if (ldb < max1(m)) return 11;
    return 0;
}

fn matrixVectorError(trans: [*]const u8, m: BlasInt, n: BlasInt, lda: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validTrans(trans)) return 1;
    if (m < 0) return 2;
    if (n < 0) return 3;
    if (lda < max1(m)) return 6;
    if (incx == 0) return 8;
    if (incy == 0) return 11;
    return 0;
}

fn bandMatrixVectorError(trans: [*]const u8, m: BlasInt, n: BlasInt, kl: BlasInt, ku: BlasInt, lda: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validTrans(trans)) return 1;
    if (m < 0) return 2;
    if (n < 0) return 3;
    if (kl < 0) return 4;
    if (ku < 0) return 5;
    if (lda < kl + ku + 1) return 8;
    if (incx == 0) return 10;
    if (incy == 0) return 13;
    return 0;
}

fn symmetricVectorError(uplo: [*]const u8, n: BlasInt, lda: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (lda < max1(n)) return 5;
    if (incx == 0) return 7;
    if (incy == 0) return 10;
    return 0;
}

fn bandSymmetricVectorError(uplo: [*]const u8, n: BlasInt, k: BlasInt, lda: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (k < 0) return 3;
    if (lda < k + 1) return 6;
    if (incx == 0) return 8;
    if (incy == 0) return 11;
    return 0;
}

fn packedSymmetricVectorError(uplo: [*]const u8, n: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 6;
    if (incy == 0) return 9;
    return 0;
}

fn triangularVectorError(uplo: [*]const u8, trans: [*]const u8, diag: [*]const u8, n: BlasInt, lda: BlasInt, incx: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (!validTrans(trans)) return 2;
    if (!validDiag(diag)) return 3;
    if (n < 0) return 4;
    if (lda < max1(n)) return 6;
    if (incx == 0) return 8;
    return 0;
}

fn bandTriangularVectorError(uplo: [*]const u8, trans: [*]const u8, diag: [*]const u8, n: BlasInt, k: BlasInt, lda: BlasInt, incx: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (!validTrans(trans)) return 2;
    if (!validDiag(diag)) return 3;
    if (n < 0) return 4;
    if (k < 0) return 5;
    if (lda < k + 1) return 7;
    if (incx == 0) return 9;
    return 0;
}

fn packedTriangularVectorError(uplo: [*]const u8, trans: [*]const u8, diag: [*]const u8, n: BlasInt, incx: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (!validTrans(trans)) return 2;
    if (!validDiag(diag)) return 3;
    if (n < 0) return 4;
    if (incx == 0) return 7;
    return 0;
}

fn rankUpdateError(m: BlasInt, n: BlasInt, incx: BlasInt, incy: BlasInt, lda: BlasInt) BlasInt {
    if (m < 0) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 5;
    if (incy == 0) return 7;
    if (lda < max1(m)) return 9;
    return 0;
}

fn symmetricRankError(uplo: [*]const u8, n: BlasInt, incx: BlasInt, lda: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 5;
    if (lda < max1(n)) return 7;
    return 0;
}

fn packedRankError(uplo: [*]const u8, n: BlasInt, incx: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 5;
    return 0;
}

fn symmetricRank2Error(uplo: [*]const u8, n: BlasInt, incx: BlasInt, incy: BlasInt, lda: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 5;
    if (incy == 0) return 7;
    if (lda < max1(n)) return 9;
    return 0;
}

fn packedRank2Error(uplo: [*]const u8, n: BlasInt, incx: BlasInt, incy: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (n < 0) return 2;
    if (incx == 0) return 5;
    if (incy == 0) return 7;
    return 0;
}

fn symmetricMatrixError(side: [*]const u8, uplo: [*]const u8, m: BlasInt, n: BlasInt, lda: BlasInt, ldb: BlasInt, ldc: BlasInt) BlasInt {
    if (!validSide(side)) return 1;
    if (!validUplo(uplo)) return 2;
    if (m < 0) return 3;
    if (n < 0) return 4;

    const nrowa = if (core.fromChar(side) == 'L') m else n;
    if (lda < max1(nrowa)) return 7;
    if (ldb < max1(m)) return 9;
    if (ldc < max1(m)) return 12;
    return 0;
}

fn rankKMatrixError(uplo: [*]const u8, trans: [*]const u8, n: BlasInt, k: BlasInt, lda: BlasInt, ldc: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (!validTrans(trans)) return 2;
    if (n < 0) return 3;
    if (k < 0) return 4;

    const nrowa = if (core.fromChar(trans) == 'N') n else k;
    if (lda < max1(nrowa)) return 7;
    if (ldc < max1(n)) return 10;
    return 0;
}

fn rank2KMatrixError(uplo: [*]const u8, trans: [*]const u8, n: BlasInt, k: BlasInt, lda: BlasInt, ldb: BlasInt, ldc: BlasInt) BlasInt {
    if (!validUplo(uplo)) return 1;
    if (!validTrans(trans)) return 2;
    if (n < 0) return 3;
    if (k < 0) return 4;

    const nrow = if (core.fromChar(trans) == 'N') n else k;
    if (lda < max1(nrow)) return 7;
    if (ldb < max1(nrow)) return 9;
    if (ldc < max1(n)) return 12;
    return 0;
}

fn reportError(comptime name: []const u8, info: BlasInt) bool {
    if (info == 0) return false;
    var info_copy = info;
    xerbla_(name.ptr, &info_copy);
    return true;
}

pub export fn scabs1_(z: *const ComplexF32) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return @abs(z.*.re) + @abs(z.*.im);
}

pub export fn dcabs1_(z: *const ComplexF64) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return @abs(z.*.re) + @abs(z.*.im);
}

// Level 1 exports.
pub export fn sswap_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sswap_(&operation, n, x, incx, y, incy);
}
pub export fn dswap_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dswap_(&operation, n, x, incx, y, incy);
}
pub export fn cswap_(n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cswap_(&operation, n, x, incx, y, incy);
}
pub export fn zswap_(n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zswap_(&operation, n, x, incx, y, incy);
}

pub export fn scopy_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.scopy_(&operation, n, x, incx, y, incy);
}
pub export fn dcopy_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dcopy_(&operation, n, x, incx, y, incy);
}
pub export fn ccopy_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ccopy_(&operation, n, x, incx, y, incy);
}
pub export fn zcopy_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zcopy_(&operation, n, x, incx, y, incy);
}

pub export fn saxpy_(n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.saxpy_(&operation, n, alpha, x, incx, y, incy);
}
pub export fn daxpy_(n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.daxpy_(&operation, n, alpha, x, incx, y, incy);
}
pub export fn caxpy_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.caxpy_(&operation, n, alpha, x, incx, y, incy);
}
pub export fn zaxpy_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zaxpy_(&operation, n, alpha, x, incx, y, incy);
}
pub export fn saxpby_(n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.saxpby_(&operation, n, alpha, x, incx, beta, y, incy);
}
pub export fn daxpby_(n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.daxpby_(&operation, n, alpha, x, incx, beta, y, incy);
}
pub export fn caxpby_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.caxpby_(&operation, n, alpha, x, incx, beta, y, incy);
}
pub export fn zaxpby_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zaxpby_(&operation, n, alpha, x, incx, beta, y, incy);
}

pub export fn sdot_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sdot_(&operation, n, x, incx, y, incy);
}
pub export fn ddot_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ddot_(&operation, n, x, incx, y, incy);
}
pub export fn cdotu_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt) callconv(.c) ComplexF32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return core.vector.dotWithResources(ComplexF32, .{ .admitted = &operation }, n.*, x, incx.*, y, incy.*, false);
}
pub export fn zdotu_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt) callconv(.c) ComplexF64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return core.vector.dotWithResources(ComplexF64, .{ .admitted = &operation }, n.*, x, incx.*, y, incy.*, false);
}
pub export fn cdotc_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt) callconv(.c) ComplexF32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return core.vector.dotWithResources(ComplexF32, .{ .admitted = &operation }, n.*, x, incx.*, y, incy.*, true);
}
pub export fn zdotc_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt) callconv(.c) ComplexF64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return core.vector.dotWithResources(ComplexF64, .{ .admitted = &operation }, n.*, x, incx.*, y, incy.*, true);
}
pub export fn cdotu_sub_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cdotu_sub_(&operation, n, x, incx, y, incy, out);
}
pub export fn zdotu_sub_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zdotu_sub_(&operation, n, x, incx, y, incy, out);
}
pub export fn cdotc_sub_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cdotc_sub_(&operation, n, x, incx, y, incy, out);
}
pub export fn zdotc_sub_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zdotc_sub_(&operation, n, x, incx, y, incy, out);
}

pub export fn sdsdot_(n: *const BlasInt, sb: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sdsdot_(&operation, n, sb, x, incx, y, incy);
}
pub export fn dsdot_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsdot_(&operation, n, x, incx, y, incy);
}

pub export fn snrm2_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.snrm2_(&operation, n, x, incx);
}
pub export fn dnrm2_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dnrm2_(&operation, n, x, incx);
}
pub export fn scnrm2_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.scnrm2_(&operation, n, x, incx);
}
pub export fn dznrm2_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dznrm2_(&operation, n, x, incx);
}

pub export fn sasum_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sasum_(&operation, n, x, incx);
}
pub export fn dasum_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dasum_(&operation, n, x, incx);
}
pub export fn scasum_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.scasum_(&operation, n, x, incx);
}
pub export fn dzasum_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dzasum_(&operation, n, x, incx);
}

pub export fn isamax_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) BlasInt {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.isamax_(&operation, n, x, incx);
}
pub export fn idamax_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) BlasInt {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.idamax_(&operation, n, x, incx);
}
pub export fn icamax_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) BlasInt {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.icamax_(&operation, n, x, incx);
}
pub export fn izamax_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) BlasInt {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.izamax_(&operation, n, x, incx);
}

pub export fn srotg_(a: *f32, b: *f32, c: *f32, s: *f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.srotg_(&operation, a, b, c, s);
}
pub export fn drotg_(a: *f64, b: *f64, c: *f64, s: *f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.drotg_(&operation, a, b, c, s);
}
pub export fn crotg_(a: *ComplexF32, b: *ComplexF32, c: *f32, s: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.crotg_(&operation, a, b, c, s);
}
pub export fn zrotg_(a: *ComplexF64, b: *ComplexF64, c: *f64, s: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zrotg_(&operation, a, b, c, s);
}
pub export fn srot_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, c: *const f32, s: *const f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.srot_(&operation, n, x, incx, y, incy, c, s);
}
pub export fn drot_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, c: *const f64, s: *const f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.drot_(&operation, n, x, incx, y, incy, c, s);
}
pub export fn csrot_(n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt, c: *const f32, s: *const f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.csrot_(&operation, n, x, incx, y, incy, c, s);
}
pub export fn zdrot_(n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt, c: *const f64, s: *const f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zdrot_(&operation, n, x, incx, y, incy, c, s);
}
pub export fn srotm_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, param: [*]const f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.srotm_(&operation, n, x, incx, y, incy, param);
}
pub export fn drotm_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, param: [*]const f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.drotm_(&operation, n, x, incx, y, incy, param);
}
pub export fn srotmg_(d1: *f32, d2: *f32, x1: *f32, y1: *const f32, param: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.srotmg_(&operation, d1, d2, x1, y1, param);
}
pub export fn drotmg_(d1: *f64, d2: *f64, x1: *f64, y1: *const f64, param: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.drotmg_(&operation, d1, d2, x1, y1, param);
}

pub export fn sscal_(n: *const BlasInt, alpha: *const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sscal_(&operation, n, alpha, x, incx);
}
pub export fn dscal_(n: *const BlasInt, alpha: *const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dscal_(&operation, n, alpha, x, incx);
}
pub export fn cscal_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cscal_(&operation, n, alpha, x, incx);
}
pub export fn zscal_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zscal_(&operation, n, alpha, x, incx);
}
pub export fn csscal_(n: *const BlasInt, alpha: *const f32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.csscal_(&operation, n, alpha, x, incx);
}
pub export fn zdscal_(n: *const BlasInt, alpha: *const f64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zdscal_(&operation, n, alpha, x, incx);
}

// Level 2 exports.
pub export fn sgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sgemv_(&operation, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn dgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dgemv_(&operation, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn cgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cgemv_(&operation, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn zgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zgemv_(&operation, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

pub export fn sgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sgbmv_(&operation, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn dgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dgbmv_(&operation, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn cgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cgbmv_(&operation, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn zgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zgbmv_(&operation, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

pub export fn ssymv_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssymv_(&operation, u, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn dsymv_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsymv_(&operation, u, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn chemv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chemv_(&operation, u, n, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn zhemv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhemv_(&operation, u, n, alpha, a, lda, x, incx, beta, y, incy);
}

pub export fn ssbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssbmv_(&operation, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn dsbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsbmv_(&operation, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn chbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chbmv_(&operation, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
pub export fn zhbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhbmv_(&operation, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

pub export fn sspmv_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, ap: [*]const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sspmv_(&operation, u, n, alpha, ap, x, incx, beta, y, incy);
}
pub export fn dspmv_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, ap: [*]const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dspmv_(&operation, u, n, alpha, ap, x, incx, beta, y, incy);
}
pub export fn chpmv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, ap: [*]const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chpmv_(&operation, u, n, alpha, ap, x, incx, beta, y, incy);
}
pub export fn zhpmv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, ap: [*]const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhpmv_(&operation, u, n, alpha, ap, x, incx, beta, y, incy);
}

pub export fn strmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.strmv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn dtrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtrmv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn ctrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctrmv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn ztrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztrmv_(&operation, u, t, d, n, a, lda, x, incx);
}

pub export fn stbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.stbmv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn dtbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtbmv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn ctbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctbmv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn ztbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztbmv_(&operation, u, t, d, n, k, a, lda, x, incx);
}

pub export fn stpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.stpmv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn dtpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtpmv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn ctpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctpmv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn ztpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztpmv_(&operation, u, t, d, n, ap, x, incx);
}

pub export fn strsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.strsv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn dtrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtrsv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn ctrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctrsv_(&operation, u, t, d, n, a, lda, x, incx);
}
pub export fn ztrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztrsv_(&operation, u, t, d, n, a, lda, x, incx);
}

pub export fn stbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.stbsv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn dtbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtbsv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn ctbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctbsv_(&operation, u, t, d, n, k, a, lda, x, incx);
}
pub export fn ztbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztbsv_(&operation, u, t, d, n, k, a, lda, x, incx);
}

pub export fn stpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.stpsv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn dtpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtpsv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn ctpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctpsv_(&operation, u, t, d, n, ap, x, incx);
}
pub export fn ztpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztpsv_(&operation, u, t, d, n, ap, x, incx);
}

pub export fn sger_(m: *const BlasInt, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sger_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn dger_(m: *const BlasInt, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dger_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn cgeru_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cgeru_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn zgeru_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zgeru_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn cgerc_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cgerc_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn zgerc_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zgerc_(&operation, m, n, alpha, x, incx, y, incy, a, lda);
}

pub export fn ssyr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssyr_(&operation, u, n, alpha, x, incx, a, lda);
}
pub export fn dsyr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsyr_(&operation, u, n, alpha, x, incx, a, lda);
}
pub export fn cher_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cher_(&operation, u, n, alpha, x, incx, a, lda);
}
pub export fn zher_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zher_(&operation, u, n, alpha, x, incx, a, lda);
}

pub export fn sspr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, ap: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sspr_(&operation, u, n, alpha, x, incx, ap);
}
pub export fn dspr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, ap: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dspr_(&operation, u, n, alpha, x, incx, ap);
}
pub export fn chpr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, ap: [*]ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chpr_(&operation, u, n, alpha, x, incx, ap);
}
pub export fn zhpr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, ap: [*]ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhpr_(&operation, u, n, alpha, x, incx, ap);
}

pub export fn ssyr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssyr2_(&operation, u, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn dsyr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsyr2_(&operation, u, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn cher2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cher2_(&operation, u, n, alpha, x, incx, y, incy, a, lda);
}
pub export fn zher2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zher2_(&operation, u, n, alpha, x, incx, y, incy, a, lda);
}

pub export fn sspr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, ap: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sspr2_(&operation, u, n, alpha, x, incx, y, incy, ap);
}
pub export fn dspr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, ap: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dspr2_(&operation, u, n, alpha, x, incx, y, incy, ap);
}
pub export fn chpr2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, ap: [*]ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chpr2_(&operation, u, n, alpha, x, incx, y, incy, ap);
}
pub export fn zhpr2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, ap: [*]ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhpr2_(&operation, u, n, alpha, x, incx, y, incy, ap);
}

// Level 3 exports.
pub export fn sgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.sgemm_(&operation, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn dgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dgemm_(&operation, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn cgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cgemm_(&operation, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn zgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zgemm_(&operation, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub export fn ssymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssymm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn dsymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsymm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn csymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.csymm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn zsymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zsymm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn chemm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.chemm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn zhemm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zhemm_(&operation, side, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub export fn ssyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssyrk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
pub export fn dsyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsyrk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
pub export fn csyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.csyrk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
pub export fn zsyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zsyrk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
pub export fn cherk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cherk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
pub export fn zherk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zherk_(&operation, u, t, n, k, alpha, a, lda, beta, c, ldc);
}

pub export fn ssyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ssyr2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn dsyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dsyr2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn csyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.csyr2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn zsyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zsyr2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn cher2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.cher2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
pub export fn zher2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.zher2k_(&operation, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub export fn strmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.strmm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn dtrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtrmm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn ctrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctrmm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn ztrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztrmm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}

pub export fn strsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.strsm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn dtrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.dtrsm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn ctrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ctrsm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}
pub export fn ztrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    return internal_admitted.ztrsm_(&operation, side, u, t, d, m, n, alpha, a, lda, b, ldb);
}

test "level1 core.dot and core.axpy" {
    var x = [_]f64{ 1, 2, 3 };
    var y = [_]f64{ 4, 5, 6 };
    var n: BlasInt = 3;
    var inc: BlasInt = 1;
    try std.testing.expectEqual(@as(f64, 32), ddot_(&n, &x, &inc, &y, &inc));
    var a: f64 = 2;
    daxpy_(&n, &a, &x, &inc, &y, &inc);
    try std.testing.expectEqual(@as(f64, 6), y[0]);
    try std.testing.expectEqual(@as(f64, 9), y[1]);
    try std.testing.expectEqual(@as(f64, 12), y[2]);
}

test "sdsdot preserves sb bits when n is nonpositive" {
    const sb: f32 = @bitCast(@as(u32, 0x80000000));
    var n: BlasInt = 0;
    const inc: BlasInt = 1;
    var x = [_]f32{1};
    var y = [_]f32{1};

    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(sdsdot_(&n, &sb, &x, &inc, &y, &inc))));
    n = -3;
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(sdsdot_(&n, &sb, &x, &inc, &y, &inc))));
}

test "complex dot sub and iamax Fortran exports" {
    var n: BlasInt = 3;
    var inc: BlasInt = 2;
    const x32 = [_]ComplexF32{
        .{ .re = 1, .im = 2 },
        .{ .re = 99, .im = 99 },
        .{ .re = -3, .im = 1 },
        .{ .re = 99, .im = 99 },
        .{ .re = 2, .im = -1 },
    };
    const y32 = [_]ComplexF32{
        .{ .re = 4, .im = -1 },
        .{ .re = 99, .im = 99 },
        .{ .re = 0.5, .im = 2 },
        .{ .re = 99, .im = 99 },
        .{ .re = -2, .im = 3 },
    };
    const x64 = [_]ComplexF64{
        .{ .re = 1, .im = 2 },
        .{ .re = 99, .im = 99 },
        .{ .re = -3, .im = 1 },
        .{ .re = 99, .im = 99 },
        .{ .re = 2, .im = -1 },
    };
    const y64 = [_]ComplexF64{
        .{ .re = 4, .im = -1 },
        .{ .re = 99, .im = 99 },
        .{ .re = 0.5, .im = 2 },
        .{ .re = 99, .im = 99 },
        .{ .re = -2, .im = 3 },
    };
    var dotu32: ComplexF32 = undefined;
    var dotc32: ComplexF32 = undefined;
    var dotu64: ComplexF64 = undefined;
    var dotc64: ComplexF64 = undefined;

    cdotu_sub_(&n, &x32, &inc, &y32, &inc, &dotu32);
    cdotc_sub_(&n, &x32, &inc, &y32, &inc, &dotc32);
    zdotu_sub_(&n, &x64, &inc, &y64, &inc, &dotu64);
    zdotc_sub_(&n, &x64, &inc, &y64, &inc, &dotc64);

    try std.testing.expectEqual(ComplexF32{ .re = 1.5, .im = 9.5 }, dotu32);
    try std.testing.expectEqual(ComplexF32{ .re = -4.5, .im = -11.5 }, dotc32);
    try std.testing.expectEqual(ComplexF64{ .re = 1.5, .im = 9.5 }, dotu64);
    try std.testing.expectEqual(ComplexF64{ .re = -4.5, .im = -11.5 }, dotc64);
    try std.testing.expectEqual(@as(BlasInt, 2), icamax_(&n, &x32, &inc));
    try std.testing.expectEqual(@as(BlasInt, 2), izamax_(&n, &x64, &inc));
}

/// Internal adapter-only bodies borrowing the external caller's live operation.
/// Not C exports or additions to the four public facade allowlists.
pub const internal_admitted = struct {
    /// Shared ABI validation bridge for a normalized Hermitian matrix view.
    /// CBLAS supplies the physical triangle and matrix conjugation only. Keep
    /// ordinary Fortran entrypoints and their validation bodies unchanged.
    pub fn hermitianMvView(comptime T: type, comptime storage: core.HermitianStorage, operation: *const execution.context.Operation, uplo: [*]const u8, n: BlasInt, k: BlasInt, alpha: *const T, a: [*]const T, lda: BlasInt, x: [*]const T, incx: BlasInt, beta: *const T, y: [*]T, incy: BlasInt, conjugate_matrix: bool) void {
        if (T != ComplexF32 and T != ComplexF64) @compileError("Hermitian ABI bridge requires a BLAS complex scalar");
        _ = operation.borrowedContext();
        const name = switch (storage) {
            .dense => if (T == ComplexF32) "CHEMV " else "ZHEMV ",
            .band => if (T == ComplexF32) "CHBMV " else "ZHBMV ",
            .packed_storage => if (T == ComplexF32) "CHPMV " else "ZHPMV ",
        };
        const info = switch (storage) {
            .dense => symmetricVectorError(uplo, n, lda, incx, incy),
            .band => bandSymmetricVectorError(uplo, n, k, lda, incx, incy),
            .packed_storage => packedSymmetricVectorError(uplo, n, incx, incy),
        };
        if (reportError(name, info)) return;
        core.hermitianMvViewAdmitted(T, storage, operation, core.parseUplo(uplo), n, k, alpha.*, a, lda, x, incx, beta.*, y, incy, conjugate_matrix);
    }

    pub fn sswap_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.swapWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn dswap_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.swapWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn cswap_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.swapWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn zswap_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.swapWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn scopy_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.copyWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn dcopy_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.copyWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn ccopy_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.copyWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn zcopy_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.copyWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn saxpy_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpyWithResources(f32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, y, incy.*);
    }

    pub fn daxpy_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpyWithResources(f64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, y, incy.*);
    }

    pub fn caxpy_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpyWithResources(ComplexF32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, y, incy.*);
    }

    pub fn zaxpy_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpyWithResources(ComplexF64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, y, incy.*);
    }

    pub fn saxpby_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpbyWithResources(f32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn daxpby_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpbyWithResources(f64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn caxpby_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpbyWithResources(ComplexF32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn zaxpby_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.axpbyWithResources(ComplexF64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn sdot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        return core.vector.dotWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, false);
    }

    pub fn ddot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.dotWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, false);
    }

    pub fn cdotu_sub_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) void {
        _ = operation.borrowedContext();
        out.* = core.vector.dotWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, false);
    }

    pub fn zdotu_sub_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) void {
        _ = operation.borrowedContext();
        out.* = core.vector.dotWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, false);
    }

    pub fn cdotc_sub_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) void {
        _ = operation.borrowedContext();
        out.* = core.vector.dotWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, true);
    }

    pub fn zdotc_sub_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) void {
        _ = operation.borrowedContext();
        out.* = core.vector.dotWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, true);
    }

    pub fn sdsdot_(operation: *const execution.context.Operation, n: *const BlasInt, sb: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        if (n.* <= 0) return sb.*;
        return @floatCast(@as(f64, sb.*) + core.vector.dotF32AccF64WithResources(.{ .admitted = operation }, n.*, x, incx.*, y, incy.*));
    }

    pub fn dsdot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.dotF32AccF64WithResources(.{ .admitted = operation }, n.*, x, incx.*, y, incy.*);
    }

    pub fn snrm2_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        return core.vector.nrm2WithResources(f32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn dnrm2_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.nrm2WithResources(f64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn scnrm2_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        return core.vector.nrm2WithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn dznrm2_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.nrm2WithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn sasum_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        return core.vector.asumWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn dasum_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.asumWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn scasum_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) f32 {
        _ = operation.borrowedContext();
        return core.vector.asumWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn dzasum_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) f64 {
        _ = operation.borrowedContext();
        return core.vector.asumWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn isamax_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) BlasInt {
        _ = operation.borrowedContext();
        return core.vector.iamaxWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn idamax_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) BlasInt {
        _ = operation.borrowedContext();
        return core.vector.iamaxWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn icamax_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) BlasInt {
        _ = operation.borrowedContext();
        return core.vector.iamaxWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn izamax_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) BlasInt {
        _ = operation.borrowedContext();
        return core.vector.iamaxWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*);
    }

    pub fn srotg_(operation: *const execution.context.Operation, a: *f32, b: *f32, c: *f32, s: *f32) void {
        _ = operation.borrowedContext();
        core.rotgReal(f32, a, b, c, s);
    }

    pub fn drotg_(operation: *const execution.context.Operation, a: *f64, b: *f64, c: *f64, s: *f64) void {
        _ = operation.borrowedContext();
        core.rotgReal(f64, a, b, c, s);
    }

    pub fn crotg_(operation: *const execution.context.Operation, a: *ComplexF32, b: *ComplexF32, c: *f32, s: *ComplexF32) void {
        _ = operation.borrowedContext();
        core.rotgComplex(ComplexF32, a, b, c, s);
    }

    pub fn zrotg_(operation: *const execution.context.Operation, a: *ComplexF64, b: *ComplexF64, c: *f64, s: *ComplexF64) void {
        _ = operation.borrowedContext();
        core.rotgComplex(ComplexF64, a, b, c, s);
    }

    pub fn srot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, c: *const f32, s: *const f32) void {
        _ = operation.borrowedContext();
        core.vector.rotWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, c.*, s.*);
    }

    pub fn drot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, c: *const f64, s: *const f64) void {
        _ = operation.borrowedContext();
        core.vector.rotWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, c.*, s.*);
    }

    pub fn csrot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt, c: *const f32, s: *const f32) void {
        _ = operation.borrowedContext();
        core.vector.rotWithResources(ComplexF32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, c.*, core.realScalar(ComplexF32, s.*));
    }

    pub fn zdrot_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt, c: *const f64, s: *const f64) void {
        _ = operation.borrowedContext();
        core.vector.rotWithResources(ComplexF64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, c.*, core.realScalar(ComplexF64, s.*));
    }

    pub fn srotm_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, param: [*]const f32) void {
        _ = operation.borrowedContext();
        core.vector.rotmWithResources(f32, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, param);
    }

    pub fn drotm_(operation: *const execution.context.Operation, n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, param: [*]const f64) void {
        _ = operation.borrowedContext();
        core.vector.rotmWithResources(f64, .{ .admitted = operation }, n.*, x, incx.*, y, incy.*, param);
    }

    pub fn srotmg_(operation: *const execution.context.Operation, d1: *f32, d2: *f32, x1: *f32, y1: *const f32, param: [*]f32) void {
        _ = operation.borrowedContext();
        core.rotmg(f32, d1, d2, x1, y1, param);
    }

    pub fn drotmg_(operation: *const execution.context.Operation, d1: *f64, d2: *f64, x1: *f64, y1: *const f64, param: [*]f64) void {
        _ = operation.borrowedContext();
        core.rotmg(f64, d1, d2, x1, y1, param);
    }

    pub fn sscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f32, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.scalWithResources(f32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn dscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f64, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.scalWithResources(f64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn cscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.scalWithResources(ComplexF32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn zscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.scalWithResources(ComplexF64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn csscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f32, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.rscalWithResources(ComplexF32, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn zdscal_(operation: *const execution.context.Operation, n: *const BlasInt, alpha: *const f64, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        core.vector.rscalWithResources(ComplexF64, .{ .admitted = operation }, n.*, alpha.*, x, incx.*);
    }

    pub fn sgemv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SGEMV ", matrixVectorError(t, m.*, n.*, lda.*, incx.*, incy.*))) return;
        core.gemvAdmitted(f32, operation, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn dgemv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DGEMV ", matrixVectorError(t, m.*, n.*, lda.*, incx.*, incy.*))) return;
        core.gemvAdmitted(f64, operation, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn cgemv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CGEMV ", matrixVectorError(t, m.*, n.*, lda.*, incx.*, incy.*))) return;
        core.gemvAdmitted(ComplexF32, operation, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn zgemv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZGEMV ", matrixVectorError(t, m.*, n.*, lda.*, incx.*, incy.*))) return;
        core.gemvAdmitted(ComplexF64, operation, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn sgbmv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SGBMV ", bandMatrixVectorError(t, m.*, n.*, kl.*, ku.*, lda.*, incx.*, incy.*))) return;
        core.gbmv(f32, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn dgbmv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DGBMV ", bandMatrixVectorError(t, m.*, n.*, kl.*, ku.*, lda.*, incx.*, incy.*))) return;
        core.gbmv(f64, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn cgbmv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CGBMV ", bandMatrixVectorError(t, m.*, n.*, kl.*, ku.*, lda.*, incx.*, incy.*))) return;
        core.gbmv(ComplexF32, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn zgbmv_(operation: *const execution.context.Operation, t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZGBMV ", bandMatrixVectorError(t, m.*, n.*, kl.*, ku.*, lda.*, incx.*, incy.*))) return;
        core.gbmv(ComplexF64, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
    }

    pub fn ssymv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYMV ", symmetricVectorError(u, n.*, lda.*, incx.*, incy.*))) return;
        core.symvAdmitted(f32, operation, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn dsymv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYMV ", symmetricVectorError(u, n.*, lda.*, incx.*, incy.*))) return;
        core.symvAdmitted(f64, operation, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn chemv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHEMV ", symmetricVectorError(u, n.*, lda.*, incx.*, incy.*))) return;
        core.symvAdmitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn zhemv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHEMV ", symmetricVectorError(u, n.*, lda.*, incx.*, incy.*))) return;
        core.symvAdmitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn ssbmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSBMV ", bandSymmetricVectorError(u, n.*, k.*, lda.*, incx.*, incy.*))) return;
        core.sbmvAdmitted(f32, operation, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn dsbmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSBMV ", bandSymmetricVectorError(u, n.*, k.*, lda.*, incx.*, incy.*))) return;
        core.sbmvAdmitted(f64, operation, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn chbmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHBMV ", bandSymmetricVectorError(u, n.*, k.*, lda.*, incx.*, incy.*))) return;
        core.sbmvAdmitted(ComplexF32, operation, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn zhbmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHBMV ", bandSymmetricVectorError(u, n.*, k.*, lda.*, incx.*, incy.*))) return;
        core.sbmvAdmitted(ComplexF64, operation, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn sspmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, ap: [*]const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSPMV ", packedSymmetricVectorError(u, n.*, incx.*, incy.*))) return;
        core.spmvAdmitted(f32, operation, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn dspmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, ap: [*]const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSPMV ", packedSymmetricVectorError(u, n.*, incx.*, incy.*))) return;
        core.spmvAdmitted(f64, operation, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, false);
    }

    pub fn chpmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, ap: [*]const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHPMV ", packedSymmetricVectorError(u, n.*, incx.*, incy.*))) return;
        core.spmvAdmitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn zhpmv_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, ap: [*]const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHPMV ", packedSymmetricVectorError(u, n.*, incx.*, incy.*))) return;
        core.spmvAdmitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, true);
    }

    pub fn strmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STRMV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trmvAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn dtrmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTRMV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trmvAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn ctrmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTRMV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trmvAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn ztrmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTRMV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trmvAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn stbmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STBMV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.tbmv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn dtbmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTBMV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.tbmv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn ctbmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTBMV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.tbmv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn ztbmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTBMV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.tbmv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn stpmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STPMV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.matrix_vector.tpmvAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn dtpmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTPMV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.matrix_vector.tpmvAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn ctpmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTPMV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.matrix_vector.tpmvAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn ztpmv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTPMV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.matrix_vector.tpmvAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn strsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STRSV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trsvAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn dtrsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTRSV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trsvAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn ctrsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTRSV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trsvAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn ztrsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTRSV ", triangularVectorError(u, t, d, n.*, lda.*, incx.*))) return;
        core.matrix_vector.trsvAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
    }

    pub fn stbsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STBSV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.matrix_vector.tbsvAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn dtbsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTBSV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.matrix_vector.tbsvAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn ctbsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTBSV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.matrix_vector.tbsvAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn ztbsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTBSV ", bandTriangularVectorError(u, t, d, n.*, k.*, lda.*, incx.*))) return;
        core.matrix_vector.tbsvAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
    }

    pub fn stpsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STPSV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.tpsv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn dtpsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTPSV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.tpsv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn ctpsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTPSV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.tpsv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn ztpsv_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTPSV ", packedTriangularVectorError(u, t, d, n.*, incx.*))) return;
        core.tpsv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
    }

    pub fn sger_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SGER  ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(f32, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
    }

    pub fn dger_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DGER  ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(f64, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
    }

    pub fn cgeru_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CGERU ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(ComplexF32, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
    }

    pub fn zgeru_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZGERU ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(ComplexF64, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
    }

    pub fn cgerc_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CGERC ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(ComplexF32, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, true);
    }

    pub fn zgerc_(operation: *const execution.context.Operation, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZGERC ", rankUpdateError(m.*, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.gerAdmitted(ComplexF64, operation, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, true);
    }

    pub fn ssyr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, a: [*]f32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYR  ", symmetricRankError(u, n.*, incx.*, lda.*))) return;
        core.matrix_vector.syrAdmitted(f32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
    }

    pub fn dsyr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, a: [*]f64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYR  ", symmetricRankError(u, n.*, incx.*, lda.*))) return;
        core.matrix_vector.syrAdmitted(f64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
    }

    pub fn cher_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHER  ", symmetricRankError(u, n.*, incx.*, lda.*))) return;
        core.matrix_vector.herAdmitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
    }

    pub fn zher_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHER  ", symmetricRankError(u, n.*, incx.*, lda.*))) return;
        core.matrix_vector.herAdmitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
    }

    pub fn sspr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, ap: [*]f32) void {
        _ = operation.borrowedContext();
        if (reportError("SSPR  ", packedRankError(u, n.*, incx.*))) return;
        core.matrix_vector.sprAdmitted(f32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
    }

    pub fn dspr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, ap: [*]f64) void {
        _ = operation.borrowedContext();
        if (reportError("DSPR  ", packedRankError(u, n.*, incx.*))) return;
        core.matrix_vector.sprAdmitted(f64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
    }

    pub fn chpr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, ap: [*]ComplexF32) void {
        _ = operation.borrowedContext();
        if (reportError("CHPR  ", packedRankError(u, n.*, incx.*))) return;
        core.matrix_vector.hprAdmitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
    }

    pub fn zhpr_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, ap: [*]ComplexF64) void {
        _ = operation.borrowedContext();
        if (reportError("ZHPR  ", packedRankError(u, n.*, incx.*))) return;
        core.matrix_vector.hprAdmitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
    }

    pub fn ssyr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYR2 ", symmetricRank2Error(u, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.syr2Admitted(f32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
    }

    pub fn dsyr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYR2 ", symmetricRank2Error(u, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.syr2Admitted(f64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
    }

    pub fn cher2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHER2 ", symmetricRank2Error(u, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.her2Admitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
    }

    pub fn zher2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHER2 ", symmetricRank2Error(u, n.*, incx.*, incy.*, lda.*))) return;
        core.matrix_vector.her2Admitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
    }

    pub fn sspr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, ap: [*]f32) void {
        _ = operation.borrowedContext();
        if (reportError("SSPR2 ", packedRank2Error(u, n.*, incx.*, incy.*))) return;
        core.matrix_vector.spr2Admitted(f32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
    }

    pub fn dspr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, ap: [*]f64) void {
        _ = operation.borrowedContext();
        if (reportError("DSPR2 ", packedRank2Error(u, n.*, incx.*, incy.*))) return;
        core.matrix_vector.spr2Admitted(f64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
    }

    pub fn chpr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, ap: [*]ComplexF32) void {
        _ = operation.borrowedContext();
        if (reportError("CHPR2 ", packedRank2Error(u, n.*, incx.*, incy.*))) return;
        core.matrix_vector.hpr2Admitted(ComplexF32, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
    }

    pub fn zhpr2_(operation: *const execution.context.Operation, u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, ap: [*]ComplexF64) void {
        _ = operation.borrowedContext();
        if (reportError("ZHPR2 ", packedRank2Error(u, n.*, incx.*, incy.*))) return;
        core.matrix_vector.hpr2Admitted(ComplexF64, operation, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
    }

    pub fn sgemm_(operation: *const execution.context.Operation, ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.gemmRealAdmitted(f32, operation, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
    }

    pub fn dgemm_(operation: *const execution.context.Operation, ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.gemmRealAdmitted(f64, operation, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
    }

    pub fn cgemm_(operation: *const execution.context.Operation, ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.gemmComplexAdmitted(ComplexF32, operation, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
    }

    pub fn zgemm_(operation: *const execution.context.Operation, ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.gemmComplexAdmitted(ComplexF64, operation, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
    }

    pub fn ssymm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(f32, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn dsymm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(f64, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn csymm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CSYMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(ComplexF32, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn zsymm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZSYMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(ComplexF64, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn chemm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHEMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(ComplexF32, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, true);
    }

    pub fn zhemm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHEMM ", symmetricMatrixError(side, u, m.*, n.*, lda.*, ldb.*, ldc.*))) return;
        core.symmAdmitted(ComplexF64, operation, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, true);
    }

    pub fn ssyrk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYRK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
    }

    pub fn dsyrk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYRK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
    }

    pub fn csyrk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CSYRK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
    }

    pub fn zsyrk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZSYRK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
    }

    pub fn cherk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHERK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, core.realScalar(ComplexF32, alpha.*), a, lda.*, core.realScalar(ComplexF32, beta.*), c, ldc.*, true);
    }

    pub fn zherk_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHERK ", rankKMatrixError(u, t, n.*, k.*, lda.*, ldc.*))) return;
        core.syrkAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, core.realScalar(ComplexF64, alpha.*), a, lda.*, core.realScalar(ComplexF64, beta.*), c, ldc.*, true);
    }

    pub fn ssyr2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("SSYR2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(f32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn dsyr2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DSYR2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(f64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn csyr2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CSYR2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn zsyr2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZSYR2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
    }

    pub fn cher2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CHER2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(ComplexF32, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, core.realScalar(ComplexF32, beta.*), c, ldc.*, true);
    }

    pub fn zher2k_(operation: *const execution.context.Operation, u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZHER2K", rank2KMatrixError(u, t, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
        core.syr2kAdmitted(ComplexF64, operation, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, core.realScalar(ComplexF64, beta.*), c, ldc.*, true);
    }

    pub fn strmm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STRMM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trmmAdmitted(f32, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn dtrmm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTRMM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trmmAdmitted(f64, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn ctrmm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTRMM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trmmAdmitted(ComplexF32, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn ztrmm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTRMM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trmmAdmitted(ComplexF64, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn strsm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("STRSM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trsmAdmitted(f32, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn dtrsm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("DTRSM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trsmAdmitted(f64, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn ctrsm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("CTRSM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trsmAdmitted(ComplexF32, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }

    pub fn ztrsm_(operation: *const execution.context.Operation, side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) void {
        _ = operation.borrowedContext();
        if (reportError("ZTRSM ", triangularMatrixError(side, u, t, d, m.*, n.*, lda.*, ldb.*))) return;
        core.trsmAdmitted(ComplexF64, operation, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
    }
};
