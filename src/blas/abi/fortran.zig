// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const std = @import("std");
const core = @import("../core.zig");

pub const BlasInt = core.BlasInt;
pub const ComplexF32 = core.ComplexF32;
pub const ComplexF64 = core.ComplexF64;

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

inline fn copyContiguous(n: BlasInt, elem_size: usize, x: [*]const u8, y: [*]u8) void {
    core.copyBytes(@as(usize, @intCast(n)) * elem_size, x, y);
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

fn reportError(comptime name: []const u8, info: BlasInt) bool {
    if (info == 0) return false;
    var info_copy = info;
    xerbla_(name.ptr, &info_copy);
    return true;
}

pub export fn scabs1_(z: *const ComplexF32) callconv(.c) f32 {
    return @abs(z.*.re) + @abs(z.*.im);
}

pub export fn dcabs1_(z: *const ComplexF64) callconv(.c) f64 {
    return @abs(z.*.re) + @abs(z.*.im);
}

// Level 1 exports.
pub export fn sswap_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.swap(f32, n.*, x, incx.*, y, incy.*);
}
pub export fn dswap_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.swap(f64, n.*, x, incx.*, y, incy.*);
}
pub export fn cswap_(n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.swap(ComplexF32, n.*, x, incx.*, y, incy.*);
}
pub export fn zswap_(n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.swap(ComplexF64, n.*, x, incx.*, y, incy.*);
}

noinline fn scopySlow(n: BlasInt, x: [*]const f32, incx: BlasInt, y: [*]f32, incy: BlasInt) void {
    core.copy(f32, n, x, incx, y, incy);
}
pub export fn scopy_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    if (n.* <= 0) return;
    if (incx.* == 1 and incy.* == 1) {
        copyContiguous(n.*, @sizeOf(f32), @ptrCast(x), @ptrCast(y));
        return;
    }
    scopySlow(n.*, x, incx.*, y, incy.*);
}
noinline fn dcopySlow(n: BlasInt, x: [*]const f64, incx: BlasInt, y: [*]f64, incy: BlasInt) void {
    core.copy(f64, n, x, incx, y, incy);
}
pub export fn dcopy_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    if (n.* <= 0) return;
    if (incx.* == 1 and incy.* == 1) {
        copyContiguous(n.*, @sizeOf(f64), @ptrCast(x), @ptrCast(y));
        return;
    }
    dcopySlow(n.*, x, incx.*, y, incy.*);
}
noinline fn ccopySlow(n: BlasInt, x: [*]const ComplexF32, incx: BlasInt, y: [*]ComplexF32, incy: BlasInt) void {
    core.copy(ComplexF32, n, x, incx, y, incy);
}
pub export fn ccopy_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    if (n.* <= 0) return;
    if (incx.* == 1 and incy.* == 1) {
        copyContiguous(n.*, @sizeOf(ComplexF32), @ptrCast(x), @ptrCast(y));
        return;
    }
    ccopySlow(n.*, x, incx.*, y, incy.*);
}
noinline fn zcopySlow(n: BlasInt, x: [*]const ComplexF64, incx: BlasInt, y: [*]ComplexF64, incy: BlasInt) void {
    core.copy(ComplexF64, n, x, incx, y, incy);
}
pub export fn zcopy_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    if (n.* <= 0) return;
    if (incx.* == 1 and incy.* == 1) {
        copyContiguous(n.*, @sizeOf(ComplexF64), @ptrCast(x), @ptrCast(y));
        return;
    }
    zcopySlow(n.*, x, incx.*, y, incy.*);
}

pub export fn saxpy_(n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.axpy(f32, n.*, alpha.*, x, incx.*, y, incy.*);
}
pub export fn daxpy_(n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.axpy(f64, n.*, alpha.*, x, incx.*, y, incy.*);
}
pub export fn caxpy_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.axpy(ComplexF32, n.*, alpha.*, x, incx.*, y, incy.*);
}
pub export fn zaxpy_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.axpy(ComplexF64, n.*, alpha.*, x, incx.*, y, incy.*);
}
pub export fn saxpby_(n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.axpby(f32, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn daxpby_(n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.axpby(f64, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn caxpby_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.axpby(ComplexF32, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn zaxpby_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.axpby(ComplexF64, n.*, alpha.*, x, incx.*, beta.*, y, incy.*);
}

pub export fn sdot_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f32 {
    return core.dot(f32, n.*, x, incx.*, y, incy.*, false);
}
pub export fn ddot_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt) callconv(.c) f64 {
    return core.dot(f64, n.*, x, incx.*, y, incy.*, false);
}
pub export fn cdotu_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt) callconv(.c) ComplexF32 {
    return core.dot(ComplexF32, n.*, x, incx.*, y, incy.*, false);
}
pub export fn zdotu_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt) callconv(.c) ComplexF64 {
    return core.dot(ComplexF64, n.*, x, incx.*, y, incy.*, false);
}
pub export fn cdotc_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt) callconv(.c) ComplexF32 {
    return core.dot(ComplexF32, n.*, x, incx.*, y, incy.*, true);
}
pub export fn zdotc_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt) callconv(.c) ComplexF64 {
    return core.dot(ComplexF64, n.*, x, incx.*, y, incy.*, true);
}
pub export fn cdotu_sub_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) callconv(.c) void {
    out.* = core.dot(ComplexF32, n.*, x, incx.*, y, incy.*, false);
}
pub export fn zdotu_sub_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) callconv(.c) void {
    out.* = core.dot(ComplexF64, n.*, x, incx.*, y, incy.*, false);
}
pub export fn cdotc_sub_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, out: *ComplexF32) callconv(.c) void {
    out.* = core.dot(ComplexF32, n.*, x, incx.*, y, incy.*, true);
}
pub export fn zdotc_sub_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, out: *ComplexF64) callconv(.c) void {
    out.* = core.dot(ComplexF64, n.*, x, incx.*, y, incy.*, true);
}

pub export fn sdsdot_(n: *const BlasInt, sb: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f32 {
    if (n.* <= 0) return sb.*;
    const sx = core.startIndex(n.*, incx.*);
    const sy = core.startIndex(n.*, incy.*);
    var sum: f64 = sb.*;
    for (0..core.toUsize(n.*)) |i| sum += @as(f64, x[core.ix(sx, i, incx.*)]) * @as(f64, y[core.ix(sy, i, incy.*)]);
    return @floatCast(sum);
}
pub export fn dsdot_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt) callconv(.c) f64 {
    if (n.* <= 0) return 0;
    const sx = core.startIndex(n.*, incx.*);
    const sy = core.startIndex(n.*, incy.*);
    var sum: f64 = 0;
    for (0..core.toUsize(n.*)) |i| sum += @as(f64, x[core.ix(sx, i, incx.*)]) * @as(f64, y[core.ix(sy, i, incy.*)]);
    return sum;
}

pub export fn snrm2_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) f32 {
    return core.nrm2(f32, n.*, x, incx.*);
}
pub export fn dnrm2_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) f64 {
    return core.nrm2(f64, n.*, x, incx.*);
}
pub export fn scnrm2_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) f32 {
    return core.nrm2(ComplexF32, n.*, x, incx.*);
}
pub export fn dznrm2_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) f64 {
    return core.nrm2(ComplexF64, n.*, x, incx.*);
}

pub export fn sasum_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) f32 {
    return core.asum(f32, n.*, x, incx.*);
}
pub export fn dasum_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) f64 {
    return core.asum(f64, n.*, x, incx.*);
}
pub export fn scasum_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) f32 {
    return core.asum(ComplexF32, n.*, x, incx.*);
}
pub export fn dzasum_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) f64 {
    return core.asum(ComplexF64, n.*, x, incx.*);
}

pub export fn isamax_(n: *const BlasInt, x: [*]const f32, incx: *const BlasInt) callconv(.c) BlasInt {
    return core.iamax(f32, n.*, x, incx.*);
}
pub export fn idamax_(n: *const BlasInt, x: [*]const f64, incx: *const BlasInt) callconv(.c) BlasInt {
    return core.iamax(f64, n.*, x, incx.*);
}
pub export fn icamax_(n: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt) callconv(.c) BlasInt {
    return core.iamax(ComplexF32, n.*, x, incx.*);
}
pub export fn izamax_(n: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt) callconv(.c) BlasInt {
    return core.iamax(ComplexF64, n.*, x, incx.*);
}

pub export fn srotg_(a: *f32, b: *f32, c: *f32, s: *f32) callconv(.c) void {
    core.rotgReal(f32, a, b, c, s);
}
pub export fn drotg_(a: *f64, b: *f64, c: *f64, s: *f64) callconv(.c) void {
    core.rotgReal(f64, a, b, c, s);
}
pub export fn crotg_(a: *ComplexF32, b: *ComplexF32, c: *f32, s: *ComplexF32) callconv(.c) void {
    core.rotgComplex(ComplexF32, a, b, c, s);
}
pub export fn zrotg_(a: *ComplexF64, b: *ComplexF64, c: *f64, s: *ComplexF64) callconv(.c) void {
    core.rotgComplex(ComplexF64, a, b, c, s);
}
pub export fn srot_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, c: *const f32, s: *const f32) callconv(.c) void {
    core.rot(f32, n.*, x, incx.*, y, incy.*, c.*, s.*);
}
pub export fn drot_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, c: *const f64, s: *const f64) callconv(.c) void {
    core.rot(f64, n.*, x, incx.*, y, incy.*, c.*, s.*);
}
pub export fn csrot_(n: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt, y: [*]ComplexF32, incy: *const BlasInt, c: *const f32, s: *const f32) callconv(.c) void {
    core.rot(ComplexF32, n.*, x, incx.*, y, incy.*, c.*, core.realScalar(ComplexF32, s.*));
}
pub export fn zdrot_(n: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt, y: [*]ComplexF64, incy: *const BlasInt, c: *const f64, s: *const f64) callconv(.c) void {
    core.rot(ComplexF64, n.*, x, incx.*, y, incy.*, c.*, core.realScalar(ComplexF64, s.*));
}
pub export fn srotm_(n: *const BlasInt, x: [*]f32, incx: *const BlasInt, y: [*]f32, incy: *const BlasInt, param: [*]const f32) callconv(.c) void {
    core.rotm(f32, n.*, x, incx.*, y, incy.*, param);
}
pub export fn drotm_(n: *const BlasInt, x: [*]f64, incx: *const BlasInt, y: [*]f64, incy: *const BlasInt, param: [*]const f64) callconv(.c) void {
    core.rotm(f64, n.*, x, incx.*, y, incy.*, param);
}
pub export fn srotmg_(d1: *f32, d2: *f32, x1: *f32, y1: *const f32, param: [*]f32) callconv(.c) void {
    core.rotmg(f32, d1, d2, x1, y1, param);
}
pub export fn drotmg_(d1: *f64, d2: *f64, x1: *f64, y1: *const f64, param: [*]f64) callconv(.c) void {
    core.rotmg(f64, d1, d2, x1, y1, param);
}

pub export fn sscal_(n: *const BlasInt, alpha: *const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.scal(f32, n.*, alpha.*, x, incx.*);
}
pub export fn dscal_(n: *const BlasInt, alpha: *const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.scal(f64, n.*, alpha.*, x, incx.*);
}
pub export fn cscal_(n: *const BlasInt, alpha: *const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.scal(ComplexF32, n.*, alpha.*, x, incx.*);
}
pub export fn zscal_(n: *const BlasInt, alpha: *const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.scal(ComplexF64, n.*, alpha.*, x, incx.*);
}
pub export fn csscal_(n: *const BlasInt, alpha: *const f32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.rscal(ComplexF32, n.*, alpha.*, x, incx.*);
}
pub export fn zdscal_(n: *const BlasInt, alpha: *const f64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.rscal(ComplexF64, n.*, alpha.*, x, incx.*);
}

// Level 2 exports.
pub export fn sgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.gemv(f32, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn dgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.gemv(f64, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn cgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.gemv(ComplexF32, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn zgemv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.gemv(ComplexF64, core.parseTrans(t), m.*, n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}

pub export fn sgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.gbmv(f32, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn dgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.gbmv(f64, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn cgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.gbmv(ComplexF32, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}
pub export fn zgbmv_(t: [*]const u8, m: *const BlasInt, n: *const BlasInt, kl: *const BlasInt, ku: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.gbmv(ComplexF64, core.parseTrans(t), m.*, n.*, kl.*, ku.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*);
}

pub export fn ssymv_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.symv(f32, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn dsymv_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.symv(f64, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn chemv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.symv(ComplexF32, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
}
pub export fn zhemv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.symv(ComplexF64, core.parseUplo(u), n.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
}

pub export fn ssbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.sbmv(f32, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn dsbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.sbmv(f64, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn chbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.sbmv(ComplexF32, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
}
pub export fn zhbmv_(u: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.sbmv(ComplexF64, core.parseUplo(u), n.*, k.*, alpha.*, a, lda.*, x, incx.*, beta.*, y, incy.*, true);
}

pub export fn sspmv_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, ap: [*]const f32, x: [*]const f32, incx: *const BlasInt, beta: *const f32, y: [*]f32, incy: *const BlasInt) callconv(.c) void {
    core.spmv(f32, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn dspmv_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, ap: [*]const f64, x: [*]const f64, incx: *const BlasInt, beta: *const f64, y: [*]f64, incy: *const BlasInt) callconv(.c) void {
    core.spmv(f64, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, false);
}
pub export fn chpmv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, ap: [*]const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, beta: *const ComplexF32, y: [*]ComplexF32, incy: *const BlasInt) callconv(.c) void {
    core.spmv(ComplexF32, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, true);
}
pub export fn zhpmv_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, ap: [*]const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, beta: *const ComplexF64, y: [*]ComplexF64, incy: *const BlasInt) callconv(.c) void {
    core.spmv(ComplexF64, core.parseUplo(u), n.*, alpha.*, ap, x, incx.*, beta.*, y, incy.*, true);
}

pub export fn strmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.trmv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn dtrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.trmv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn ctrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.trmv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn ztrmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.trmv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}

pub export fn stbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.tbmv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn dtbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.tbmv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn ctbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.tbmv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn ztbmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.tbmv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}

pub export fn stpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.tpmv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn dtpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.tpmv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn ctpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.tpmv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn ztpmv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.tpmv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}

pub export fn strsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.trsv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn dtrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.trsv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn ctrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.trsv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}
pub export fn ztrsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.trsv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, a, lda.*, x, incx.*);
}

pub export fn stbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f32, lda: *const BlasInt, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.tbsv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn dtbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const f64, lda: *const BlasInt, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.tbsv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn ctbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF32, lda: *const BlasInt, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.tbsv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}
pub export fn ztbsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, k: *const BlasInt, a: [*]const ComplexF64, lda: *const BlasInt, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.tbsv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, k.*, a, lda.*, x, incx.*);
}

pub export fn stpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f32, x: [*]f32, incx: *const BlasInt) callconv(.c) void {
    core.tpsv(f32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn dtpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const f64, x: [*]f64, incx: *const BlasInt) callconv(.c) void {
    core.tpsv(f64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn ctpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: *const BlasInt) callconv(.c) void {
    core.tpsv(ComplexF32, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}
pub export fn ztpsv_(u: [*]const u8, t: [*]const u8, d: [*]const u8, n: *const BlasInt, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: *const BlasInt) callconv(.c) void {
    core.tpsv(ComplexF64, core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), n.*, ap, x, incx.*);
}

pub export fn sger_(m: *const BlasInt, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    core.ger(f32, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
}
pub export fn dger_(m: *const BlasInt, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    core.ger(f64, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
}
pub export fn cgeru_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    core.ger(ComplexF32, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
}
pub export fn zgeru_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    core.ger(ComplexF64, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, false);
}
pub export fn cgerc_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    core.ger(ComplexF32, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, true);
}
pub export fn zgerc_(m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    core.ger(ComplexF64, m.*, n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*, true);
}

pub export fn ssyr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    core.syr(f32, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
}
pub export fn dsyr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    core.syr(f64, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
}
pub export fn cher_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    core.her(ComplexF32, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
}
pub export fn zher_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    core.her(ComplexF64, core.parseUplo(u), n.*, alpha.*, x, incx.*, a, lda.*);
}

pub export fn sspr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, ap: [*]f32) callconv(.c) void {
    core.spr(f32, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
}
pub export fn dspr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, ap: [*]f64) callconv(.c) void {
    core.spr(f64, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
}
pub export fn chpr_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const ComplexF32, incx: *const BlasInt, ap: [*]ComplexF32) callconv(.c) void {
    core.hpr(ComplexF32, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
}
pub export fn zhpr_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const ComplexF64, incx: *const BlasInt, ap: [*]ComplexF64) callconv(.c) void {
    core.hpr(ComplexF64, core.parseUplo(u), n.*, alpha.*, x, incx.*, ap);
}

pub export fn ssyr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, a: [*]f32, lda: *const BlasInt) callconv(.c) void {
    core.syr2(f32, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
}
pub export fn dsyr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, a: [*]f64, lda: *const BlasInt) callconv(.c) void {
    core.syr2(f64, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
}
pub export fn cher2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, a: [*]ComplexF32, lda: *const BlasInt) callconv(.c) void {
    core.her2(ComplexF32, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
}
pub export fn zher2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, a: [*]ComplexF64, lda: *const BlasInt) callconv(.c) void {
    core.her2(ComplexF64, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, a, lda.*);
}

pub export fn sspr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f32, x: [*]const f32, incx: *const BlasInt, y: [*]const f32, incy: *const BlasInt, ap: [*]f32) callconv(.c) void {
    core.spr2(f32, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
}
pub export fn dspr2_(u: [*]const u8, n: *const BlasInt, alpha: *const f64, x: [*]const f64, incx: *const BlasInt, y: [*]const f64, incy: *const BlasInt, ap: [*]f64) callconv(.c) void {
    core.spr2(f64, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
}
pub export fn chpr2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: *const BlasInt, y: [*]const ComplexF32, incy: *const BlasInt, ap: [*]ComplexF32) callconv(.c) void {
    core.hpr2(ComplexF32, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
}
pub export fn zhpr2_(u: [*]const u8, n: *const BlasInt, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: *const BlasInt, y: [*]const ComplexF64, incy: *const BlasInt, ap: [*]ComplexF64) callconv(.c) void {
    core.hpr2(ComplexF64, core.parseUplo(u), n.*, alpha.*, x, incx.*, y, incy.*, ap);
}

// Level 3 exports.
pub export fn sgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    if (reportError("SGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
    core.gemm(f32, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
}
pub export fn dgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    if (reportError("DGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
    core.gemm(f64, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
}
pub export fn cgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    if (reportError("CGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
    core.gemm(ComplexF32, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
}
pub export fn zgemm_(ta: [*]const u8, tb: [*]const u8, m: *const BlasInt, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    if (reportError("ZGEMM ", gemmError(ta, tb, m.*, n.*, k.*, lda.*, ldb.*, ldc.*))) return;
    core.gemm(ComplexF64, core.parseTrans(ta), core.parseTrans(tb), m.*, n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*);
}

pub export fn ssymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    core.symm(f32, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn dsymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    core.symm(f64, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn csymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.symm(ComplexF32, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn zsymm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.symm(ComplexF64, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn chemm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.symm(ComplexF32, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, true);
}
pub export fn zhemm_(side: [*]const u8, u: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.symm(ComplexF64, core.parseSide(side), core.parseUplo(u), m.*, n.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, true);
}

pub export fn ssyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(f32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
}
pub export fn dsyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(f64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
}
pub export fn csyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(ComplexF32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
}
pub export fn zsyrk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(ComplexF64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, beta.*, c, ldc.*, false);
}
pub export fn cherk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const ComplexF32, lda: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(ComplexF32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, core.realScalar(ComplexF32, alpha.*), a, lda.*, core.realScalar(ComplexF32, beta.*), c, ldc.*, true);
}
pub export fn zherk_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const ComplexF64, lda: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.syrk(ComplexF64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, core.realScalar(ComplexF64, alpha.*), a, lda.*, core.realScalar(ComplexF64, beta.*), c, ldc.*, true);
}

pub export fn ssyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]const f32, ldb: *const BlasInt, beta: *const f32, c: [*]f32, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(f32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn dsyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]const f64, ldb: *const BlasInt, beta: *const f64, c: [*]f64, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(f64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn csyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const ComplexF32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(ComplexF32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn zsyr2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const ComplexF64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(ComplexF64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, beta.*, c, ldc.*, false);
}
pub export fn cher2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]const ComplexF32, ldb: *const BlasInt, beta: *const f32, c: [*]ComplexF32, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(ComplexF32, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, core.realScalar(ComplexF32, beta.*), c, ldc.*, true);
}
pub export fn zher2k_(u: [*]const u8, t: [*]const u8, n: *const BlasInt, k: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]const ComplexF64, ldb: *const BlasInt, beta: *const f64, c: [*]ComplexF64, ldc: *const BlasInt) callconv(.c) void {
    core.syr2k(ComplexF64, core.parseUplo(u), core.parseTrans(t), n.*, k.*, alpha.*, a, lda.*, b, ldb.*, core.realScalar(ComplexF64, beta.*), c, ldc.*, true);
}

pub export fn strmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) callconv(.c) void {
    core.trmm(f32, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn dtrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) callconv(.c) void {
    core.trmm(f64, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn ctrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) callconv(.c) void {
    core.trmm(ComplexF32, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn ztrmm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) callconv(.c) void {
    core.trmm(ComplexF64, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}

pub export fn strsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f32, a: [*]const f32, lda: *const BlasInt, b: [*]f32, ldb: *const BlasInt) callconv(.c) void {
    core.trsm(f32, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn dtrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const f64, a: [*]const f64, lda: *const BlasInt, b: [*]f64, ldb: *const BlasInt) callconv(.c) void {
    core.trsm(f64, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn ctrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: *const BlasInt, b: [*]ComplexF32, ldb: *const BlasInt) callconv(.c) void {
    core.trsm(ComplexF32, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
}
pub export fn ztrsm_(side: [*]const u8, u: [*]const u8, t: [*]const u8, d: [*]const u8, m: *const BlasInt, n: *const BlasInt, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: *const BlasInt, b: [*]ComplexF64, ldb: *const BlasInt) callconv(.c) void {
    core.trsm(ComplexF64, core.parseSide(side), core.parseUplo(u), core.parseTrans(t), core.parseDiag(d), m.*, n.*, alpha.*, a, lda.*, b, ldb.*);
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
