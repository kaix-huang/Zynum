// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Local arithmetic only. Import project scalar declarations, not BLAS compute.
const types = @import("../blas/types.zig");
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

pub fn isComplex(comptime T: type) bool {
    return T == ComplexF32 or T == ComplexF64;
}
pub fn zero(comptime T: type) T {
    return if (comptime isComplex(T)) .{ .re = 0, .im = 0 } else 0;
}
pub fn one(comptime T: type) T {
    return if (comptime isComplex(T)) .{ .re = 1, .im = 0 } else 1;
}
pub fn isZero(comptime T: type, value: T) bool {
    return if (comptime isComplex(T)) value.re == 0 and value.im == 0 else value == 0;
}
pub fn isOne(comptime T: type, value: T) bool {
    return if (comptime isComplex(T)) value.re == 1 and value.im == 0 else value == 1;
}
pub fn add(comptime T: type, a: T, b: T) T {
    return if (comptime isComplex(T)) .{ .re = a.re + b.re, .im = a.im + b.im } else a + b;
}
pub fn multiply(comptime T: type, a: T, b: T) T {
    return if (comptime isComplex(T))
        .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re }
    else
        a * b;
}
pub fn maybeConjugate(comptime T: type, value: T, conjugate: bool) T {
    if (comptime isComplex(T)) {
        return if (conjugate) .{ .re = value.re, .im = -value.im } else value;
    }
    return value;
}
