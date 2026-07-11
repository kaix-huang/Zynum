// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const stride2_parallel = @import("stride2_parallel.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;
pub const minimum_elements: usize = 512 * 1024;

fn targetUnary(n: BlasInt, incx: BlasInt) ?usize {
    if (comptime builtin.cpu.arch != .x86_64) return null;
    if (n < @as(BlasInt, @intCast(minimum_elements)) or incx != 2) return null;
    return @intCast(n);
}

fn targetBinary(n: BlasInt, incx: BlasInt, incy: BlasInt) ?usize {
    if (incx != 2 or incy != 2) return null;
    return targetUnary(n, incx);
}

fn byteRangesOverlap(x: [*]const u8, y: [*]const u8, n: usize) bool {
    if (n == 0) return false;
    const xp = @intFromPtr(x);
    const yp = @intFromPtr(y);
    if (xp == yp) return true;
    if (xp < yp) return yp - xp < n;
    return xp - yp < n;
}

fn targetNonOverlappingBinary(
    comptime T: type,
    n: BlasInt,
    x: [*]const T,
    incx: BlasInt,
    y: [*]const T,
    incy: BlasInt,
) ?usize {
    const count = targetBinary(n, incx, incy) orelse return null;
    const span_bytes = (2 * count - 1) * @sizeOf(T);
    if (byteRangesOverlap(@ptrCast(x), @ptrCast(y), span_bytes)) return null;
    return count;
}

pub fn scal(comptime T: type, n: BlasInt, alpha: T, x: [*]T, incx: BlasInt) bool {
    const count = targetUnary(n, incx) orelse return false;
    if (scalar.isOne(T, alpha)) return false;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelScalStride2Real(T, count, alpha, x);
    } else if (comptime T == ComplexF32 or T == ComplexF64) {
        return stride2_parallel.parallelComplexScalStride2(T, count, alpha, x);
    } else {
        @compileError("stride-two scal supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn rscal(comptime T: type, n: BlasInt, alpha: scalar.Real(T), x: [*]T, incx: BlasInt) bool {
    const count = targetUnary(n, incx) orelse return false;
    if (alpha == 1) return false;
    if (comptime T != ComplexF32 and T != ComplexF64) {
        @compileError("stride-two rscal supports ComplexF32 and ComplexF64");
    }
    return stride2_parallel.parallelComplexRealScalStride2(T, count, alpha, x);
}

pub fn swap(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt) bool {
    const count = targetNonOverlappingBinary(T, n, x, incx, y, incy) orelse return false;
    return stride2_parallel.parallelSwapStride2(T, count, x, y);
}

pub fn axpy(comptime T: type, n: BlasInt, alpha: T, x: [*]const T, incx: BlasInt, y: [*]T, incy: BlasInt) bool {
    const count = targetNonOverlappingBinary(T, n, x, incx, y, incy) orelse return false;
    if (scalar.isZero(T, alpha)) return false;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelAxpyStride2Real(T, count, alpha, x, y);
    } else if (comptime T == ComplexF32) {
        return stride2_parallel.parallelComplexAxpyStride2C32(count, alpha, x, y);
    } else if (comptime T == ComplexF64) {
        return stride2_parallel.parallelComplexAxpyStride2C64(count, alpha, x, y);
    } else {
        @compileError("stride-two axpy supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn axpby(comptime T: type, n: BlasInt, alpha: T, x: [*]const T, incx: BlasInt, beta: T, y: [*]T, incy: BlasInt) bool {
    const count = targetNonOverlappingBinary(T, n, x, incx, y, incy) orelse return false;
    if (scalar.isZero(T, alpha) or scalar.isOne(T, beta)) return false;
    if (scalar.isOne(T, alpha) and scalar.isZero(T, beta)) return false;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelAxpbyStride2Real(T, count, alpha, x, beta, y);
    } else if (comptime T == ComplexF32 or T == ComplexF64) {
        return stride2_parallel.parallelComplexAxpbyStride2(T, count, alpha, x, beta, y);
    } else {
        @compileError("stride-two axpby supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn dot(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt, y: [*]const T, incy: BlasInt, conjx: bool) ?T {
    const count = targetBinary(n, incx, incy) orelse return null;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelDotStride2Real(T, count, x, y);
    } else if (comptime T == ComplexF32 or T == ComplexF64) {
        return stride2_parallel.parallelDotStride2Complex(T, count, x, y, conjx);
    } else {
        @compileError("stride-two dot supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn dotF32AccF64(n: BlasInt, x: [*]const f32, incx: BlasInt, y: [*]const f32, incy: BlasInt) ?f64 {
    const count = targetBinary(n, incx, incy) orelse return null;
    return stride2_parallel.parallelDotF32AccF64Stride2(count, x, y);
}

pub fn asum(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) ?scalar.Real(T) {
    const count = targetUnary(n, incx) orelse return null;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelAsumStride2Real(T, count, x);
    } else if (comptime T == ComplexF32 or T == ComplexF64) {
        return stride2_parallel.parallelAsumStride2Complex(T, count, x);
    } else {
        @compileError("stride-two asum supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn nrm2(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) ?scalar.Real(T) {
    const count = targetUnary(n, incx) orelse return null;
    if (comptime T == f32 or T == f64) {
        return stride2_parallel.parallelNrm2Stride2Real(T, count, x);
    } else if (comptime T == ComplexF32 or T == ComplexF64) {
        return stride2_parallel.parallelNrm2Stride2Complex(T, count, x);
    } else {
        @compileError("stride-two nrm2 supports f32, f64, ComplexF32, and ComplexF64");
    }
}

pub fn iamax(comptime T: type, n: BlasInt, x: [*]const T, incx: BlasInt) ?BlasInt {
    const count = targetUnary(n, incx) orelse return null;
    return stride2_parallel.parallelIamaxStride2(T, count, x);
}

pub fn rot(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt, c: scalar.Real(T), s: T) bool {
    const count = targetNonOverlappingBinary(T, n, x, incx, y, incy) orelse return false;
    if (comptime T == ComplexF32 or T == ComplexF64) {
        if (scalar.imagPart(T, s) != 0) return false;
    }
    return stride2_parallel.parallelRotStride2(T, count, x, y, c, s);
}

pub fn rotm(comptime T: type, n: BlasInt, x: [*]T, incx: BlasInt, y: [*]T, incy: BlasInt, param: [*]const T) bool {
    const count = targetNonOverlappingBinary(T, n, x, incx, y, incy) orelse return false;
    if (comptime T != f32 and T != f64) {
        @compileError("stride-two rotm supports f32 and f64");
    }

    const flag = param[0];
    if (flag == -2) return false;
    const h11 = if (flag < 0 or flag > 0) param[1] else 0;
    const h21 = if (flag <= 0) param[2] else 0;
    const h12 = if (flag <= 0) param[3] else 0;
    const h22 = if (flag < 0 or flag > 0) param[4] else 0;
    return stride2_parallel.parallelRotmStride2Real(T, count, x, y, flag, h11, h21, h12, h22);
}
