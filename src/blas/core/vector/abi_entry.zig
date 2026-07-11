// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("../shared/scalar.zig");
const operations = @import("operations.zig");
const stride2 = @import("stride2_dispatch.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;

pub noinline fn scal(comptime T: type, n: BlasInt, alpha: T, x: [*]T, incx: BlasInt) void {
    if (stride2.scal(T, n, alpha, x, incx)) return;
    operations.scal(T, n, alpha, x, incx);
}

pub noinline fn rscal(comptime T: type, n: BlasInt, alpha: scalar.Real(T), x: [*]T, incx: BlasInt) void {
    if (stride2.rscal(T, n, alpha, x, incx)) return;
    operations.rscal(T, n, alpha, x, incx);
}

pub noinline fn swap(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt) void {
    if (stride2.swap(T, n, x, incx, y, incy)) return;
    operations.swap(T, n, x, incx, y, incy);
}

pub noinline fn axpy(comptime T: type, n: BlasInt, alpha: T, x: [*]const T, incx: BlasInt, y: [*]T, incy: BlasInt) void {
    if (stride2.axpy(T, n, alpha, x, incx, y, incy)) return;
    operations.axpy(T, n, alpha, x, incx, y, incy);
}

pub noinline fn axpby(comptime T: type, n: BlasInt, alpha: T, x: [*]const T, incx: BlasInt, beta: T, y: [*]T, incy: BlasInt) void {
    if (stride2.axpby(T, n, alpha, x, incx, beta, y, incy)) return;
    operations.axpby(T, n, alpha, x, incx, beta, y, incy);
}

pub noinline fn dot(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt, y: [*]const T, incy: BlasInt, conjx: bool) T {
    return stride2.dot(T, n, x, incx, y, incy, conjx) orelse operations.dot(T, n, x, incx, y, incy, conjx);
}

pub noinline fn dotF32AccF64(n: BlasInt, x: [*]const f32, incx: BlasInt, y: [*]const f32, incy: BlasInt) f64 {
    return stride2.dotF32AccF64(n, x, incx, y, incy) orelse operations.dotF32AccF64(n, x, incx, y, incy);
}

pub noinline fn asum(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) scalar.Real(T) {
    return stride2.asum(T, n, x, incx) orelse operations.asum(T, n, x, incx);
}

pub noinline fn nrm2(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) scalar.Real(T) {
    return stride2.nrm2(T, n, x, incx) orelse operations.nrm2(T, n, x, incx);
}

pub noinline fn iamax(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) BlasInt {
    return stride2.iamax(T, n, x, incx) orelse operations.iamax(T, n, x, incx);
}

pub noinline fn rot(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt, c: scalar.Real(T), s: T) void {
    if (stride2.rot(T, n, x, incx, y, incy, c, s)) return;
    operations.rot(T, n, x, incx, y, incy, c, s);
}

pub noinline fn rotm(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt, param: [*]const T) void {
    if (stride2.rotm(T, n, x, incx, y, incy, param)) return;
    operations.rotm(T, n, x, incx, y, incy, param);
}
