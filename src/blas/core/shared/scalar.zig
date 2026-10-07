// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const types = @import("../../types.zig");

pub const BlasInt = types.BlasInt;
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

pub const TransposeMode = enum { no_trans, trans, conj_trans };
pub const Triangle = enum { upper, lower };
pub const Diagonal = enum { unit, non_unit };
pub const OperandSide = enum { left, right };

pub const Order = TransposeMode;
pub const Uplo = Triangle;
pub const Diag = Diagonal;
pub const Side = OperandSide;

pub fn isComplex(comptime T: type) bool {
    return T == ComplexF32 or T == ComplexF64;
}

pub fn Real(comptime T: type) type {
    return if (T == ComplexF32) f32 else if (T == ComplexF64) f64 else T;
}

/// Reinterpret native-memory complex pairs while preserving real/imaginary
/// order. Zig 0.17's logical @bitCast would reverse each pair on big-endian
/// targets when changing the vector element width from u64 to f32.
pub inline fn complexF32Lanes(comptime count: comptime_int, bits: @Vector(count, u64)) @Vector(2 * count, f32) {
    const words: [count]u64 = bits;
    return @as(*align(1) const [2 * count]f32, @ptrCast(&words)).*;
}

/// Encode interleaved real/imaginary lanes as native-memory complex pairs.
pub inline fn complexF32Bits(comptime count: comptime_int, lanes: @Vector(2 * count, f32)) @Vector(count, u64) {
    const components: [2 * count]f32 = lanes;
    return @as(*align(1) const [count]u64, @ptrCast(&components)).*;
}

pub fn zero(comptime T: type) T {
    return if (T == ComplexF32)
        .{ .re = 0, .im = 0 }
    else if (T == ComplexF64)
        .{ .re = 0, .im = 0 }
    else
        0;
}

pub fn one(comptime T: type) T {
    return if (T == ComplexF32)
        .{ .re = 1, .im = 0 }
    else if (T == ComplexF64)
        .{ .re = 1, .im = 0 }
    else
        1;
}

pub fn realScalar(comptime T: type, x: Real(T)) T {
    return if (T == ComplexF32)
        .{ .re = x, .im = 0 }
    else if (T == ComplexF64)
        .{ .re = x, .im = 0 }
    else
        x;
}

pub fn add(comptime T: type, a: T, b: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re + b.re, .im = a.im + b.im }
    else
        a + b;
}

pub fn sub(comptime T: type, a: T, b: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re - b.re, .im = a.im - b.im }
    else
        a - b;
}

pub fn neg(comptime T: type, a: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = -a.re, .im = -a.im }
    else
        -a;
}

pub fn mul(comptime T: type, a: T, b: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re }
    else
        a * b;
}

pub fn divv(comptime T: type, a: T, b: T) T {
    if (T == ComplexF32 or T == ComplexF64) {
        const den = b.re * b.re + b.im * b.im;
        return .{ .re = (a.re * b.re + a.im * b.im) / den, .im = (a.im * b.re - a.re * b.im) / den };
    }
    return a / b;
}

pub fn conj(comptime T: type, a: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re, .im = -a.im }
    else
        a;
}

pub fn maybeConj(comptime T: type, a: T, do_conj: bool) T {
    return if (do_conj) conj(T, a) else a;
}

pub fn realPart(comptime T: type, a: T) Real(T) {
    return if (T == ComplexF32 or T == ComplexF64) a.re else a;
}

pub fn imagPart(comptime T: type, a: T) Real(T) {
    return if (T == ComplexF32 or T == ComplexF64) a.im else 0;
}

pub fn isZero(comptime T: type, a: T) bool {
    return if (T == ComplexF32 or T == ComplexF64) a.re == 0 and a.im == 0 else a == 0;
}

pub fn isOne(comptime T: type, a: T) bool {
    return if (T == ComplexF32 or T == ComplexF64) a.re == 1 and a.im == 0 else a == 1;
}

pub fn abs1(comptime T: type, a: T) Real(T) {
    return if (T == ComplexF32 or T == ComplexF64) @abs(a.re) + @abs(a.im) else @abs(a);
}

pub fn abs2(comptime T: type, a: T) Real(T) {
    return if (T == ComplexF32 or T == ComplexF64) @sqrt(a.re * a.re + a.im * a.im) else @abs(a);
}

pub fn fromChar(p: [*]const u8) u8 {
    return std.ascii.toUpper(p[0]);
}

pub fn parseTrans(p: [*]const u8) Order {
    return switch (fromChar(p)) {
        'T' => .trans,
        'C' => .conj_trans,
        else => .no_trans,
    };
}

pub fn parseUplo(p: [*]const u8) Uplo {
    return if (fromChar(p) == 'L') .lower else .upper;
}

pub fn parseDiag(p: [*]const u8) Diag {
    return if (fromChar(p) == 'U') .unit else .non_unit;
}

pub fn parseSide(p: [*]const u8) Side {
    return if (fromChar(p) == 'R') .right else .left;
}

pub const fromReal = realScalar;
pub const divide = divv;
pub const conjugate = conj;
pub const conjugateIf = maybeConj;
pub const absoluteSum = abs1;
pub const absoluteMagnitude = abs2;
