// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

// Fixed-width Zig vectors are portable; the compiler lowers them to the target.
// Every lane performs one multiply and subtraction, without a horizontal sum.
const lanes = 8;

pub fn subtractScaled(comptime T: type, destination: []T, source: []const T, scale: T) void {
    const Vector = @Vector(lanes, T);
    const coefficient: Vector = @splat(scale);
    var i: usize = 0;
    while (i + lanes <= destination.len) : (i += lanes) {
        const before: Vector = destination[i..][0..lanes].*;
        const column: Vector = source[i..][0..lanes].*;
        destination[i..][0..lanes].* = before - column * coefficient;
    }
    while (i < destination.len) : (i += 1) destination[i] -= source[i] * scale;
}

// A successful chunk needs one reduced finite check. On the exceptional path,
// replay only that chunk in scalar order: preserve the valid prefix and never
// store the first nonfinite result, matching the original LU update contract.
pub fn subtractScaledFinite(comptime T: type, destination: []T, source: []const T, scale: T) bool {
    const Vector = @Vector(lanes, T);
    const coefficient: Vector = @splat(scale);
    const largest: Vector = @splat(std.math.floatMax(T));
    var i: usize = 0;
    while (i + lanes <= destination.len) : (i += lanes) {
        const before: Vector = destination[i..][0..lanes].*;
        const column: Vector = source[i..][0..lanes].*;
        const after = before - column * coefficient;
        if (!@reduce(.And, @abs(after) <= largest)) {
            for (i..i + lanes) |k| {
                const value = destination[k] - source[k] * scale;
                if (!std.math.isFinite(value)) return false;
                destination[k] = value;
            }
            continue;
        }
        destination[i..][0..lanes].* = after;
    }
    while (i < destination.len) : (i += 1) {
        const value = destination[i] - source[i] * scale;
        if (!std.math.isFinite(value)) return false;
        destination[i] = value;
    }
    return true;
}
