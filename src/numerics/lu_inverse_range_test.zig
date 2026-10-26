// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const inverse = @import("lu_inverse.zig");
const scalar = @import("complex.zig").scalar;
fn number(comptime T: type, re: scalar.Real(T), im: scalar.Real(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = re, .im = im } else re;
}
const Identity = struct {
    pub fn readUnchecked(_: @This(), index: usize) usize {
        if (index >= 3) @panic("range inverse read final pivot");
        return index;
    }
};

test "LU inverse retains representable lower-chain entries after an underflowing intermediate in eight extreme-scale cases" {
    var count: usize = 0;
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const R = scalar.Real(T);
        for (0..2) |subnormal| {
            const small: R = if (comptime R == f32) (if (subnormal == 0) 0x1p-126 else 0x1p-127) else (if (subnormal == 0) 0x1p-1022 else 0x1p-1023);
            const large: R = if (comptime R == f32) (if (subnormal == 0) 0x1p126 else 0x1p127) else (if (subnormal == 0) 0x1p1022 else 0x1p1023);
            var a: [20]T = @splat(number(T, -700, 300));
            for (0..4) |j| for (0..4) |i| {
                a[i + j * 5] = number(T, if (i == j) 1 else 0, 0);
            };
            a[1] = number(T, small, if (scalar.isComplex(T)) small else 0);
            a[7] = number(T, small, if (scalar.isComplex(T)) -small else 0);
            a[13] = number(T, if (scalar.isComplex(T)) large / 2 else large, if (scalar.isComplex(T)) large / 2 else 0);
            const before = a;
            var work: [16]inverse.WorkValue(T) = undefined;
            try std.testing.expectEqual(@as(usize, 0), try inverse.invert(T, .{ .values = &a, .row_count = 4, .column_count = 4, .leading_dimension = 5 }, Identity{}, &work));
            // L has three subdiagonal links. (I-N+N^2-N^3)[3,0]
            // is -small (or -small-i*small), even though [2,0] rounds to zero.
            if (comptime scalar.isComplex(T)) {
                try std.testing.expectEqual(-small, a[3].re);
                try std.testing.expectEqual(-small, a[3].im);
                try std.testing.expectEqual(@as(R, 0), a[2].re);
                try std.testing.expectEqual(@as(R, 0), a[2].im);
                for (a) |value| try std.testing.expect(std.math.isFinite(value.re) and std.math.isFinite(value.im));
            } else {
                try std.testing.expectEqual(-small, a[3]);
                try std.testing.expectEqual(@as(R, 0), a[2]);
                for (a) |value| try std.testing.expect(std.math.isFinite(value));
            }
            for (0..4) |j| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[4 + j * 5]), std.mem.asBytes(&a[4 + j * 5]));
            count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 8), count);
}
