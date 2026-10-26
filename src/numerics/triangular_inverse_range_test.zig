// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const inverse = @import("triangular_inverse.zig");
const scalar = @import("complex.zig").scalar;
fn number(comptime T: type, value: scalar.Real(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = value, .im = 0 } else value;
}
fn real(comptime T: type, value: T) scalar.Real(T) {
    return if (comptime scalar.isComplex(T)) value.re else value;
}

test "triangular inverse retains representable cross-column entries despite earlier rounded underflow and intermediate overflow" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const R = scalar.Real(T);
        const large: R = if (comptime R == f32) 0x1p127 else 0x1p1023;
        const small: R = if (comptime R == f32) 0x1p-126 else 0x1p-1022;
        for ([_]inverse.Triangle{ .upper, .lower }) |triangle| {
            for ([_]bool{ false, true }) |phased| {
                if (phased and !scalar.isComplex(T)) continue;
                var a: [9]T = @splat(number(T, 0));
                a[0] = number(T, large);
                a[4] = number(T, large);
                a[8] = number(T, small);
                a[if (triangle == .upper) 3 else 1] = number(T, small);
                a[if (triangle == .upper) 7 else 5] = number(T, large);
                if (comptime scalar.isComplex(T)) {
                    if (phased) for (0..3) |i| for (0..3) |j| {
                        const value = a[i + j * 3].re;
                        a[i + j * 3] = .{ .re = if (i == 2) -value else value, .im = if (i == 1) -value else value };
                    };
                }
                const offset: usize = if (triangle == .upper) 6 else 2;
                var workspace: [3]inverse.WorkValue(T) = undefined;
                try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, triangle, .non_unit, &workspace));
                // The real matrix has exact cross-entry 1/L, despite the
                // earlier S/L^2 underflow. Row phases make A=P*T, so its
                // inverse is T^-1*P^-1, an independent analytic phase oracle.
                const reciprocal: f128 = 1 / @as(f128, large);
                const expected_re = if (!phased) reciprocal else (if (triangle == .upper) @as(f128, -0.5) else 0.5) * reciprocal;
                const expected_im = if (phased) -0.5 * reciprocal else 0;
                const actual_re: f128 = real(T, a[offset]);
                const actual_im: f128 = if (comptime scalar.isComplex(T)) a[offset].im else 0;
                try std.testing.expect(std.math.isFinite(actual_re) and std.math.isFinite(actual_im));
                try std.testing.expectApproxEqAbs(expected_re, actual_re, @abs(expected_re) * 1e-6);
                try std.testing.expectApproxEqAbs(expected_im, actual_im, @abs(expected_im) * 1e-6);
                count += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 12), count);
}
