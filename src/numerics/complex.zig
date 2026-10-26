// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared pure complex arithmetic with the existing project scalar identity.
//! These are widened computational leaves; no storage/API validation, allocation,
//! kernel/backend dispatch, or ABI conversion is performed here.
const std = @import("std");
pub const scalar = @import("../blas/core/shared/scalar.zig");

pub fn expectComplex(comptime C: type) void {
    if (!scalar.isComplex(C)) @compileError("complex LAPACK requires ComplexF32 or ComplexF64");
}
pub fn Wide(comptime C: type) type {
    comptime expectComplex(C);
    return if (C == scalar.ComplexF32) f64 else f128;
}
pub fn WideComplex(comptime C: type) type {
    return struct { re: Wide(C), im: Wide(C) };
}
pub fn widen(comptime C: type, z: C) WideComplex(C) {
    return .{ .re = z.re, .im = z.im };
}
pub fn narrow(comptime C: type, z: WideComplex(C)) ?C {
    const result: C = .{ .re = @floatCast(z.re), .im = @floatCast(z.im) };
    return if (finite(C, result)) result else null;
}
pub fn finite(comptime C: type, z: C) bool {
    return std.math.isFinite(z.re) and std.math.isFinite(z.im);
}
pub fn abs1(comptime C: type, z: C) Wide(C) {
    return @abs(@as(Wide(C), z.re)) + @abs(@as(Wide(C), z.im));
}
pub fn subtractProduct(comptime C: type, value: *WideComplex(C), a: C, b: C) void {
    const aw = widen(C, a);
    const bw = widen(C, b);
    value.re -= aw.re * bw.re - aw.im * bw.im;
    value.im -= aw.re * bw.im + aw.im * bw.re;
}

// The divisor is a stored C value, and the numerator is a widened finite sum
// of products of C values. Normalize by the divisor's largest component before
// division; neither b.re*b.re+b.im*b.im in C nor an unscaled C reciprocal occurs.
pub fn divide(comptime C: type, a: WideComplex(C), b: C) WideComplex(C) {
    const W = Wide(C);
    const scale = @max(@abs(@as(W, b.re)), @abs(@as(W, b.im)));
    const br = @as(W, b.re) / scale;
    const bi = @as(W, b.im) / scale;
    const den = br * br + bi * bi;
    const ar = a.re / scale;
    const ai = a.im / scale;
    return .{ .re = (ar * br + ai * bi) / den, .im = (ai * br - ar * bi) / den };
}

/// Zynum's closed IEEE quotient convention for stored complex operands. This
/// defines component classification, not vendor rounding, payload or trap/flag
/// equivalence. NaN wins over all other classifications; both-infinite inputs
/// yield two quiet NaNs. Zero denominator uses its real zero's sign for Inf.
/// The existing finite widened divide above and all its callers are unchanged.
pub fn divideIeee(comptime C: type, a: C, b: C) C {
    const R = scalar.Real(C);
    const W = Wide(C);
    const quiet_nan: C = .{ .re = std.math.nan(R), .im = std.math.nan(R) };
    if (std.math.isNan(a.re) or std.math.isNan(a.im) or std.math.isNan(b.re) or std.math.isNan(b.im)) return quiet_nan;
    const numerator_infinite = std.math.isInf(a.re) or std.math.isInf(a.im);
    const denominator_infinite = std.math.isInf(b.re) or std.math.isInf(b.im);
    if (numerator_infinite and denominator_infinite) return quiet_nan;
    if (b.re == 0 and b.im == 0) {
        const infinity = std.math.copysign(std.math.inf(R), b.re);
        return .{ .re = infinity * a.re, .im = infinity * a.im };
    }
    if (denominator_infinite) {
        const br = std.math.copysign(@as(W, if (std.math.isInf(b.re)) 1 else 0), @as(W, b.re));
        const bi = std.math.copysign(@as(W, if (std.math.isInf(b.im)) 1 else 0), @as(W, b.im));
        const ar: W = a.re;
        const ai: W = a.im;
        return .{
            .re = @floatCast(@as(W, 0.0) * (ar * br + ai * bi)),
            .im = @floatCast(@as(W, 0.0) * (ai * br - ar * bi)),
        };
    }
    if (numerator_infinite) {
        const ar = std.math.copysign(@as(W, if (std.math.isInf(a.re)) 1 else 0), @as(W, a.re));
        const ai = std.math.copysign(@as(W, if (std.math.isInf(a.im)) 1 else 0), @as(W, a.im));
        const br: W = b.re;
        const bi: W = b.im;
        return .{
            .re = @floatCast(std.math.inf(W) * (ar * br + ai * bi)),
            .im = @floatCast(std.math.inf(W) * (ai * br - ar * bi)),
        };
    }
    const value = divide(C, widen(C, a), b);
    return .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
}
