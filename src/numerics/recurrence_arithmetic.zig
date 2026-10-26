// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Checked, widened arithmetic shared by polynomial recurrences. No owner.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");

pub fn Wide(comptime T: type) type {
    return if (S.Real(T) == f32) f64 else f128;
}
pub fn finite(comptime T: type, value: T) bool {
    return if (comptime S.isComplex(T)) std.math.isFinite(value.re) and std.math.isFinite(value.im) else std.math.isFinite(value);
}
pub fn produced(comptime T: type, value: T) !T {
    if (!finite(T, value)) return error.NonFiniteResult;
    return value;
}
pub fn linear(comptime T: type, a: T, x: T, b: T, y: T) !T {
    const W = Wide(T);
    return produced(T, if (comptime S.isComplex(T)) .{
        .re = @floatCast(@as(W, a.re) * x.re - @as(W, a.im) * x.im + @as(W, b.re) * y.re - @as(W, b.im) * y.im),
        .im = @floatCast(@as(W, a.re) * x.im + @as(W, a.im) * x.re + @as(W, b.re) * y.im + @as(W, b.im) * y.re),
    } else @floatCast(@as(W, a) * x + @as(W, b) * y));
}
/// scale*(product-center*current) - previous_factor*previous. Keep all three
/// terms wide until the final scalar: a representable result must not fail
/// solely because its native-width shifted product/center would overflow.
pub fn shifted(comptime T: type, product: T, current: T, previous: T, scale: Wide(T), center: Wide(T), previous_factor: Wide(T)) !T {
    const W = Wide(T);
    return produced(T, if (comptime S.isComplex(T)) .{
        .re = @floatCast(scale * (@as(W, product.re) - center * current.re) - previous_factor * previous.re),
        .im = @floatCast(scale * (@as(W, product.im) - center * current.im) - previous_factor * previous.im),
    } else @floatCast(scale * (@as(W, product) - center * current) - previous_factor * previous));
}
