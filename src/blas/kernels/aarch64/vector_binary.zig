// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 two-vector specialized kernels.

const features = @import("features.zig");

const enable_asimd_dcopy = true;

fn asimdDcopyF64(n: usize, x: [*]const f64, y: [*]f64) void {
    const V = @Vector(2, f64);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        const x1: V = @as(*align(1) const V, @ptrCast(x + i + 2)).*;
        const x2: V = @as(*align(1) const V, @ptrCast(x + i + 4)).*;
        const x3: V = @as(*align(1) const V, @ptrCast(x + i + 6)).*;
        const x4: V = @as(*align(1) const V, @ptrCast(x + i + 8)).*;
        const x5: V = @as(*align(1) const V, @ptrCast(x + i + 10)).*;
        const x6: V = @as(*align(1) const V, @ptrCast(x + i + 12)).*;
        const x7: V = @as(*align(1) const V, @ptrCast(x + i + 14)).*;
        @as(*align(1) V, @ptrCast(y + i)).* = x0;
        @as(*align(1) V, @ptrCast(y + i + 2)).* = x1;
        @as(*align(1) V, @ptrCast(y + i + 4)).* = x2;
        @as(*align(1) V, @ptrCast(y + i + 6)).* = x3;
        @as(*align(1) V, @ptrCast(y + i + 8)).* = x4;
        @as(*align(1) V, @ptrCast(y + i + 10)).* = x5;
        @as(*align(1) V, @ptrCast(y + i + 12)).* = x6;
        @as(*align(1) V, @ptrCast(y + i + 14)).* = x7;
    }
    while (i + 2 <= n) : (i += 2) {
        @as(*align(1) V, @ptrCast(y + i)).* = @as(*align(1) const V, @ptrCast(x + i)).*;
    }
    while (i < n) : (i += 1) y[i] = x[i];
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    if (comptime !enable_asimd_dcopy) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (n < 16 or n >= 256 * 1024) return false;
    asimdDcopyF64(n, x, y);
    return true;
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    return false;
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    _ = n;
    _ = x;
    _ = y;
    return null;
}
