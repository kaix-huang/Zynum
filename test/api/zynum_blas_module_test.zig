// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const blas = @import("zynum-blas");

test "BLAS package module imports directly" {
    var values = [_]f32{ 1, 2, 3, 4 };
    const vector = try blas.constVector(f32, &values, .{});
    try std.testing.expectEqual(@as(blas.BlasInt, 4), vector.length);
}
